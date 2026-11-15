"""Метка профиля (`model_label`) и авто-подбор эталона (`check-parity`): без GPU и без весов.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_profiles.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем: `model_label` — не косметика. Её печатает `[profile]`, и по ней `run --check-parity auto`
(`parity._match_entry`) выбирает запись эталона. Пока `preview` считался «тем же base, но
черновым», прогон FLUX.2-пресета `preview` подбирал запись `base-640-8` (Qwen + FUSAL R80) и
сверял несравнимое. Тест фиксирует два правила:

1. `base` — только профили **Qwen-Image-2.1 + FUSAL R80** (`base`/`qwen`/`preview-qwen`/`full-qwen`);
   остальные пресеты (включая `preview`, который теперь FLUX.2) остаются своими именами;
2. каждая запись `tests/reference.json` воспроизводится своим пресетом и **сама себя** находит
   через `parity._match_entry` — иначе авто-сверка молча сравнивала бы чужой кадр.

Сеть и веса не нужны: только конфиги и JSON эталонов.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwenstyle.cmd import parity                     # noqa: E402 — после правки sys.path
from qwenstyle.cmd.run import model_label             # noqa: E402
from qwenstyle.config import RunConfig, available_presets   # noqa: E402

#: Жёстко зафиксированные ожидания только для тех пресетов, где ошибка уже случалась.
#: Остальные проверяются правилом (см. `_family`), чтобы новый пресет не ломал тест сам по себе.
EXPECTED = {
    "base": "base",
    "qwen": "base",
    "preview-qwen": "base",
    "full-qwen": "base",
    "turbo": "turbo",
    "flux2": "flux2",
    "preview": "preview",
    "demo": "demo",
    "full-flux2": "full-flux2",
    "h3-image": "h3-image",
}


def _family(cfg: RunConfig) -> str:
    """Семейство модели по конфигу: `flux2` | `turbo` | `qwen` (Qwen-Image-2.1 + FUSAL R80).

    Любой другой бекенд — своё семейство: `minimax_h3_image` (`h3-image`) -> имя бекенда, и
    `model_label` отдаёт для него имя пресета, а не `base`.
    """
    backend = str(cfg.generation.backend)
    if backend == "flux2":
        return "flux2"
    if backend != "qwen_image21":
        return backend                     # h3-image и прочие: метка = имя пресета
    lora = getattr(cfg.generation.lora, "filename", None) or ""
    transformer = " ".join(str(part) for part in (cfg.generation.transformer or ()))
    if "turbo" in f"{transformer} {lora}".lower():
        return "turbo"
    return "qwen"


def case_labels() -> None:
    """1. Метка пресета = семейство модели (а не «черновой/финальный»).

    Правило, а не таблица: `flux2`-профили остаются своими именами (`preview` — тоже FLUX.2),
    turbo-профиль — `turbo`, бекенды со своим семейством (`minimax_h3_image`/`h3-image`) — имя
    пресета, всё остальное (Qwen-Image-2.1 + FUSAL R80) — `base`. Новый пресет (`anime`, `*`)
    не требует правки теста — требует только правильной метки.
    """
    names = sorted(available_presets())
    for name in names:
        cfg = RunConfig.from_profile(name)
        got = model_label(cfg)
        family = _family(cfg)
        want = "base" if family == "qwen" else (family if family == "turbo" else name)
        if got != want:
            raise AssertionError(f"пресет {name} ({family}): метка {got!r}, ожидалась {want!r}")
        if name in EXPECTED and got != EXPECTED[name]:
            raise AssertionError(f"пресет {name}: метка {got!r}, зафиксировано {EXPECTED[name]!r}")
    families: dict[str, int] = {}
    for name in names:
        label = model_label(RunConfig.from_profile(name))
        families[label] = families.get(label, 0) + 1
    print(f"  ok: {len(names)} пресетов -> метки {sorted(families)}")

    # Явная защита от старого бага: `preview` — FLUX.2, а не черновик base.
    preview = RunConfig.from_profile("preview")
    if model_label(preview) == "base":
        raise AssertionError("preview снова считается base (это FLUX.2-пресет)")
    if preview.generation.backend != "flux2":
        raise AssertionError(f"preview больше не flux2: backend={preview.generation.backend!r}")
    if model_label(RunConfig.from_profile("preview-qwen")) != "base":
        raise AssertionError("preview-qwen (Qwen + FUSAL R80) обязан остаться base")
    print("  ok: preview = FLUX.2 (не base), preview-qwen = base")


def case_reference_self_match() -> None:
    """2. Каждая запись эталона воспроизводится своим пресетом и находит сама себя по метке."""
    reference = parity.load_reference(REPO_ROOT / "tests" / "reference.json")
    entries = reference.get("entries") or {}
    if not entries:
        raise AssertionError("tests/reference.json пуст")

    checked = 0
    for name, entry in entries.items():
        params = entry.get("params") or {}
        cfg = parity.config_from_params(params, out="out/tests/profile-match.png")
        label = model_label(cfg)
        if label != str(params.get("model") or "base"):
            raise AssertionError(f"{name}: метка {label!r} не совпадает с model="
                                 f"{params.get('model')!r} из эталона")
        if str(entry.get("kind") or "run") != "run":
            continue                                  # батч проверяется своим путём
        match = parity._match_entry(reference, cfg)
        if match is None:
            raise AssertionError(f"{name}: авто-подбор ничего не нашёл (метка {label!r})")
        if match[0] != name:
            raise AssertionError(f"{name}: авто-подбор вернул чужую запись {match[0]!r}")
        checked += 1
    print(f"  ok: {checked} записей kind=run находят себя через _match_entry")


def main() -> int:
    cases = (
        ("1. метки пресетов по семейству модели", case_labels),
        ("2. авто-подбор эталона по метке", case_reference_self_match),
    )
    failed = 0
    for label, case in cases:
        try:
            case()
            print(f"PASS {label}", flush=True)
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {label}: {exc}", flush=True)
        except Exception:                                   # noqa: BLE001 — тест не должен врать
            failed += 1
            print(f"FAIL {label}: исключение\n{traceback.format_exc()}", flush=True)
    if failed:
        print(f"{failed} FAIL", flush=True)
        return 1
    print("ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
