"""Публичные пресеты (17) и метка профиля (`model_label`): без GPU, весов и сети.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_profiles.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем: публичная матрица профилей — это **контракт** для `README`/`gen --preset` и для
`models download --profile` (`qwenstyle/profiles.py`). Ошибка в ней не падает: она просто
обещает пользователю не тот кадр (1080p на 8 ГБ VRAM или 4K на 16 ГБ RAM). Тест фиксирует:

1. **17 публичных пресетов резолвятся** и совпадают с манифестами профилей (`PROFILES`);
2. **числа матрицы**: `size_px`, `steps`, `offload`, LoRA 0/0.7, `target_height`
   (VRAM8 → 1080 апскейлом, VRAM12 → 960 нативно, VRAM16 → 1440 нативно), апскейл/энкодер/кеш;
3. публичные пресеты не тянут dev-опции (`seedvr2`, `flux2`) и не смотрят в легаси-кеш;
4. **метки пресетов по семейству модели** (старый баг: `preview` — FLUX.2, но считался `base`);
5. ключевые dev-пресеты живы и резолвятся (на них опираются эксперименты, README их не обещает).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                   # тест не должен оставлять __pycache__ в пакете

from qwenstyle.cmd import config as config_cmd      # noqa: E402 — после правки sys.path
from qwenstyle.cmd.run import BASE_LORA, model_label   # noqa: E402
from qwenstyle.config import (                      # noqa: E402
    EMBEDS_DIR,
    MODELS_DIR,
    RunConfig,
    available_presets,
)
from qwenstyle.profiles import PROFILE_ORDER, PROFILES   # noqa: E402

#: Публичная матрица: пресет -> (size_px, steps, lora, offload, target_height).
#: Источник — план публикации (17 пресетов), а не вывод кода: тест ловит расхождение с ним.
#: `target_height` 0 = апскейла нет: у `VRAM12-release`/`VRAM16-release` это `upscale: null`
#: (кадр отдаётся native), у черновиков — просто выключенный ESRGAN.
MATRIX: dict[str, tuple[int, int, float, str, int]] = {
    "VRAM8-preview":         (480,  8, 0.0, "group", 0),
    "VRAM8-release":         (640,  8, 0.0, "group", 1080),
    "VRAM8-preview-anime":   (480,  8, 0.7, "group", 0),
    "VRAM8-release-anime":   (640, 25, 0.7, "group", 1080),
    "VRAM12-preview":        (768,  8, 0.0, "group", 0),
    "VRAM12-release":        (960, 25, 0.0, "group", 0),
    "VRAM12-preview-anime":  (768,  8, 0.7, "group", 0),
    "VRAM12-release-anime":  (960, 25, 0.7, "group", 0),
    "VRAM16-preview":        (768,  8, 0.0, "none", 0),
    "VRAM16-preview-anime":  (768,  8, 0.7, "none", 0),
    "VRAM16-release":        (1440, 25, 0.0, "group", 0),
    "VRAM16-release-anime":  (1440, 25, 0.7, "group", 0),
    "VRAM16-4k-fast":        (1088, 25, 0.0, "group", 2160),
    "VRAM16-r3-169":         (912, 25, 0.0, "group", 2160),
    "VRAM16-r3-43":          (1056, 25, 0.0, "group", 2160),
    "VRAM16-large":          (1792, 25, 0.0, "group", 0),
    "VRAM16-4k-slide":       (2160, 25, 0.0, "group", 0),
}

#: Целевая высота кадра у профиля: именно её обещает README (1080p / 960p / 1440p).
PROFILE_TARGET_HEIGHT = {"VRAM8": 1080, "VRAM12": 960, "VRAM16": 1440}

#: Жёстко зафиксированные ожидания метки там, где ошибка уже случалась.
EXPECTED_LABELS = {
    "base": "base",
    "turbo": "turbo",
    "h3-image": "h3-image",
}

#: Dev-пресеты, на которые опираются эксперименты/README разработчика (публичными не являются).
#: FLUX.2-пресеты удалены вместе с остальными dev-надстройками; бекенд `flux2` остался в коде.
DEV_PRESETS = ("base", "turbo", "h3-image")

#: Dev-опции, которых в публичных пресетах быть не должно (план публикации, решения 4 и 12).
DEV_ONLY_BACKENDS = ("seedvr2", "flux2")


def public_presets() -> list[str]:
    """17 публичных пресетов в порядке профилей — как их печатает `gen`/`models download`."""
    names: list[str] = []
    for profile_name in PROFILE_ORDER:
        names.extend(PROFILES[profile_name].presets)
    return names


def _lora_scale(cfg: RunConfig) -> float:
    """Сила LoRA числом (`0.0`, если адаптера нет) — int/float в пресетах сравнивать нельзя."""
    lora = cfg.generation.lora
    return 0.0 if lora is None else float(lora.scale)


# --------------------------------------------------------------------------- #
# Кейсы
# --------------------------------------------------------------------------- #

def case_public_presets_resolve() -> None:
    """1. Все 17 публичных пресетов из манифестов профилей существуют и резолвятся."""
    names = public_presets()
    if len(names) != 17:
        raise AssertionError(f"публичных пресетов {len(names)}, в плане 17: {names}")
    if len(set(names)) != len(names):
        raise AssertionError(f"пресет продублирован в профилях: {names}")
    if sorted(names) != sorted(MATRIX):
        raise AssertionError(f"состав публичных пресетов разошёлся с матрицей теста: "
                            f"только в профилях {sorted(set(names) - set(MATRIX))}, "
                            f"только в тесте {sorted(set(MATRIX) - set(names))}")
    registered = available_presets()
    missing = [name for name in names if name not in registered]
    if missing:
        raise AssertionError(f"нет файлов qwenstyle/presets для {missing}")
    for name in names:
        RunConfig.from_profile(name)                       # не резолвится — исключение и FAIL
    print(f"  ok: 17 публичных пресетов резолвятся, профилей {len(PROFILE_ORDER)}")


def case_matrix() -> None:
    """2. Числа матрицы: размер, шаги, LoRA, offload, целевая высота, апскейл, энкодер, кеш."""
    for name, (size_px, steps, lora, offload, target) in MATRIX.items():
        cfg = RunConfig.from_profile(name)
        got = (cfg.size_px, cfg.steps, _lora_scale(cfg), cfg.generation.offload,
               int(cfg.upscale.target_height or 0))
        want = (size_px, steps, lora, offload, target)
        if got != want:
            raise AssertionError(f"{name}: (size_px, steps, lora, offload, target_height) = "
                                 f"{got}, в матрице {want}")
        if cfg.upscale.backend != "esrgan":
            raise AssertionError(f"{name}: апскейл {cfg.upscale.backend!r}, а публичные профили "
                                 f"обещают esrgan")
        if cfg.upscale.factor:
            raise AssertionError(f"{name}: upscale.factor={cfg.upscale.factor} — фактор обязан "
                                 f"считаться из target_height (0 в пресетах)")
        if not target and cfg.upscale.target_height:
            raise AssertionError(f"{name}: target_height={cfg.upscale.target_height} при "
                                 f"нулевом факторе")
        if cfg.encoder.name != "heretic":
            raise AssertionError(f"{name}: энкодер {cfg.encoder.name!r}, публичные — heretic")
    print(f"  ok: {len(MATRIX)} пресетов совпали с публичной матрицей (size/steps/lora/offload/target)")


def case_profiles_matrix() -> None:
    """3. Профили: 4/4/7 пресетов, целевая высота 1080/960/1440, релизы её держат."""
    expected_counts = {"VRAM8": 4, "VRAM12": 4, "VRAM16": 9}
    for profile_name in PROFILE_ORDER:
        profile = PROFILES[profile_name]
        count = len(profile.presets)
        if count != expected_counts[profile_name]:
            raise AssertionError(f"профиль {profile_name}: пресетов {count}, "
                                 f"ожидалось {expected_counts[profile_name]}")
        target = PROFILE_TARGET_HEIGHT[profile_name]
        if int(profile.target_height) != target:
            raise AssertionError(f"профиль {profile_name}: target_height={profile.target_height}, "
                                 f"ожидалось {target}")
        for preset in profile.presets:
            cfg = RunConfig.from_profile(preset)
            if "release" not in preset:
                continue                            # preview — черновик, высоты не обещает
            # Высоту профиля даёт либо native-кадр (у VRAM12/VRAM16-release — `upscale: null`),
            # либо апскейл из меньшего рендера (VRAM8-release: 640p -> 1080p).
            native = int(cfg.size_px or 0)
            upscaled = int(cfg.upscale.target_height or 0)
            if target not in (native, upscaled):
                raise AssertionError(f"{preset}: native {native} и target_height {upscaled} — "
                                     f"ни одно не равно высоте профиля {profile_name} ({target})")
        if profile.disk_gib <= 0 or not profile.repos:
            raise AssertionError(f"профиль {profile_name}: пустой манифест ({profile})")
    print("  ok: VRAM8 4/1080 (апскейл), VRAM12 4/960 (native), VRAM16 7/1440 (native)")


def case_public_stays_public() -> None:
    """4. Публичные пресеты без dev-опций и без легаси-путей кеша."""
    for name in public_presets():
        cfg = RunConfig.from_profile(name)
        transport = " ".join(str(part) for part in (cfg.generation.transformer or ()))
        haystack = f"{cfg.generation.backend} {cfg.upscale.backend} {transport}".lower()
        for token in DEV_ONLY_BACKENDS:
            if token in haystack:
                raise AssertionError(f"{name}: публичный пресет ссылается на dev-опцию {token!r}")
        precision_cache = str(cfg.generation.precision_cache or "")
        if precision_cache and not precision_cache.startswith(f"{MODELS_DIR.as_posix()}/"):
            raise AssertionError(f"{name}: precision_cache={precision_cache!r} вне "
                                 f"{MODELS_DIR.as_posix()}/ (кеш моделей отделён от эмбеддингов)")
        config_cmd.describe(cfg)                   # дамп конфига не падает на публичном пресете
    # Кеши разделены: файлы эмбеддингов живут в cache/embeds, а не в корне cache/
    if EMBEDS_DIR.parent.name != "cache" or EMBEDS_DIR.name != "embeds":
        raise AssertionError(f"EMBEDS_DIR={EMBEDS_DIR!r} — ожидался cache/embeds")
    print("  ok: 17 пресетов без seedvr2/flux2, fp8-кеш в cache/models, эмбеддинги в cache/embeds")


def case_labels() -> None:
    """5. Метка пресета: `base` только для базовой модели, иначе — имя пресета.

    `model_label` — не косметика: её печатает `[profile]`, по ней видно, каким семейством сделан
    кадр. Правило (вытекает из докстринга `cmd.run.model_label`): метка `base` ставится, только
    если прогон идёт на Qwen-Image-2.1 с адаптером FUSAL R80; другой трансформер (turbo), другой
    адаптер или другой бекенд (FLUX.2, MiniMax-H3) — это **имя пресета**. Новый пресет не требует
    правки теста — только правильной метки.
    """
    for name in sorted(available_presets()):
        cfg = RunConfig.from_profile(name)
        got = model_label(cfg)
        want = _expected_label(cfg, name)
        if got != want:
            raise AssertionError(f"пресет {name}: метка {got!r}, ожидалась {want!r} "
                                 f"(backend={cfg.generation.backend}, "
                                 f"transformer={cfg.generation.transformer})")
        if name in EXPECTED_LABELS and got != EXPECTED_LABELS[name]:
            raise AssertionError(f"пресет {name}: метка {got!r}, "
                                 f"зафиксировано {EXPECTED_LABELS[name]!r}")
    base_labels = sorted(name for name in available_presets()
                         if model_label(RunConfig.from_profile(name)) == "base")
    print(f"  ok: {len(available_presets())} пресетов -> base у {len(base_labels)} "
          f"({', '.join(base_labels)})")

    # Метка не зависит от размера/шагов/апскейла: публичный флагманский пресет — всё ещё `base`.
    if model_label(RunConfig.from_profile("VRAM16-4k-fast")) != "base":
        raise AssertionError("VRAM16-4k-fast (Qwen + FUSAL R80) обязан остаться base")
    if model_label(RunConfig.from_profile("VRAM8-preview")) == "base":
        raise AssertionError("VRAM8-preview идёт на Turbo-трансформере и не может быть base")
    print("  ok: VRAM16-4k-fast = base, VRAM8-preview = свой профиль")


def case_dev_presets() -> None:
    """6. Ключевые dev-пресеты живы (на них опираются эксперименты, README их не обещает)."""
    registered = available_presets()
    missing = [name for name in DEV_PRESETS if name not in registered]
    if missing:
        raise AssertionError(f"пропали dev-пресеты {missing}")
    for name in DEV_PRESETS:
        cfg = RunConfig.from_profile(name)
        if cfg.generation.backend not in ("qwen_image21", "flux2", "minimax_h3_image"):
            raise AssertionError(f"{name}: неизвестный бекенд {cfg.generation.backend!r}")
        if name not in public_presets() and cfg.preset != name:
            raise AssertionError(f"{name}: preset={cfg.preset!r}, ожидалось имя пресета")
    print(f"  ok: {len(DEV_PRESETS)} dev-пресетов резолвятся и не входят в публичную матрицу")


def _expected_label(cfg: RunConfig, name: str) -> str:
    """Ожидаемая метка по конфигу: `base` только для Qwen-Image-2.1 + FUSAL R80, иначе имя.

    Зеркалит правило `model_label`, но записано как спецификация (какие условия дают `base`),
    а не вызовом того же кода: тест обязан ловить регресс правила, а не повторять его.
    """
    gen = cfg.generation
    if str(gen.backend) != "qwen_image21":
        return name                        # FLUX.2 / MiniMax-H3: своё семейство
    transformer = gen.transformer or ()
    if transformer and str(transformer[0]) != str(gen.repo):
        return name                        # turbo: трансформер из другого репозитория
    lora = getattr(gen.lora, "filename", None) or ""
    if lora and lora != BASE_LORA:
        return name                        # другой адаптер = другой профиль
    return "base"


CASES = (
    ("1. 17 публичных пресетов резолвятся", case_public_presets_resolve),
    ("2. числа публичной матрицы", case_matrix),
    ("3. профили: состав и целевая высота", case_profiles_matrix),
    ("4. публичные пресеты без dev-опций", case_public_stays_public),
    ("5. метки пресетов по семейству модели", case_labels),
    ("6. dev-пресеты живы", case_dev_presets),
)


def main() -> int:
    failed = 0
    for label, case in CASES:
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
