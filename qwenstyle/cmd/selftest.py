"""`python -m qwenstyle selftest` — проверка установки: окружение + один дешёвый рендер.

Отвечает на вопрос «машина готова?» тремя секциями:

    [selftest] окружение   python / torch + CUDA / diffusers / transformers / pillow / psutil,
                           VRAM, RAM, диск, fp8-кэш трансформера;
    [selftest] профиль     (только с `--profile VRAM8|VRAM12|VRAM16`) наличие репозиториев
                           манифеста (`qwenstyle/profiles.py`) и пресетов профиля;
    [selftest] рендер      480p / 4 шага на **синтетической картинке**, собранной в памяти
                           (градиент + фигуры, 768×768), промт — `tests/texts/exp3.txt`.

Проверки рендера: кадр нужного размера (сетка conditioning), `Image.getextrema()` не нулевые
и кадр не одноцветный, файл сохранён в `out/tests/selftest-<stamp>.png`, тайминги напечатаны.
Ядро не переписывается: это обычный `pipeline.run(cfg, log=None)` с `RunConfig.from_profile`
(черновой пресет + `generation.steps=4`), поэтому selftest проверяет **тот же путь**, которым
потом идёт пользовательская генерация.

Коды возврата: 0 — всё в порядке; 2 — нет весов/кэша (печатается готовая команда
`models download --profile … --yes`) или нечем рендерить (нет torch/diffusers); 1 — рендер
прошёл, но не прошёл проверки (размер/пустой кадр/файл не сохранился).

Кэши selftest не вытесняет: прогон идёт с `cache_supersede=none` (AGENTS.md R2), поэтому
пользовательские эмбеддинги рядом с пробным кадром не удаляются.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..backends.base import REPO_ROOT, hardware_resources
from ..profiles import PROFILE_ORDER, profile as machine_profile

NAME = "selftest"
HELP = ("проверка установки: версии/CUDA/VRAM/RAM/диск, веса профиля и один рендер 480p/4 шага "
        "на синтетической картинке")

#: Пресет для пробного рендера: базовое семейство (Qwen-Image-2.1, fp8-кэш `cache/models/fp8_transformer`,
#: энкодер из base.yaml). Размер и шаги опускаются ниже: 480p и 4 шага — проба весов, а не кадр.
RENDER_PRESET = "base"
#: Высота пробного кадра (`size_px`): 480p — дешёвая проба на любом профиле.
RENDER_SIZE = 480
#: Групповой offload — чтобы проба влезала и в 8 ГиБ VRAM (`base` по умолчанию без offload).
RENDER_OFFLOAD = "group"
#: Промт лежит в репозитории (SFW-текст), кадр — нет: вход собирается кодом.
PROMPT_FILE = "tests/texts/exp3.txt"
#: Сторона синтетического входа: пропорция 1:1 -> conditioning 480×480 (высота = `size_px`).
SYNTHETIC_SIZE = 768
RENDER_STEPS = 4
#: Куда кладётся пробный кадр (`out/` целиком в `.gitignore`).
OUT_DIR = REPO_ROOT / "out" / "tests"


@dataclass
class Row:
    """Строка секции: что проверяли, статус (`OK`/`WARN`/`FAIL`) и подробность."""

    name: str
    status: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


@dataclass
class Section:
    title: str
    rows: list[Row] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> None:
        self.rows.append(Row(name, status, detail))

    @property
    def failed(self) -> bool:
        return any(row.status == "FAIL" for row in self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "rows": [row.to_dict() for row in self.rows]}


# --------------------------------------------------------------------------- #
# Окружение
# --------------------------------------------------------------------------- #

def _version(module: str) -> str:
    """`<version>` или бросок исключения — вызывающий решает, FAIL это или WARN."""
    imported = __import__(module)
    return str(getattr(imported, "__version__", "?"))


def _env_section() -> Section:
    section = Section("окружение")
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    section.add("python", "OK" if sys.version_info >= (3, 10) else "WARN",
                f"{version} (нужен 3.12)" if sys.version_info < (3, 12) else version)

    imports: dict[str, bool] = {}
    for module in ("torch", "diffusers", "transformers", "PIL"):
        name = "pillow" if module == "PIL" else module.lower()
        try:
            section.add(name, "OK", _version(module))
            imports[module] = True
        except Exception as exc:                         # noqa: BLE001 — отсутствие пакета не падение
            section.add(name, "FAIL", f"нет: {type(exc).__name__}: {exc}")
            imports[module] = False
    try:
        section.add("psutil", "OK", _version("psutil"))
    except Exception as exc:                             # noqa: BLE001 — RAM тогда «нет данных»
        section.add("psutil", "WARN", f"нет ({type(exc).__name__}): размер RAM не измерить")

    if imports.get("torch"):
        import torch

        available = torch.cuda.is_available()
        cuda = f"cuda {torch.version.cuda} available={available}"
        if available:
            section.add("cuda", "OK", f"{cuda}, {torch.cuda.get_device_name(0)}")
        else:
            section.add("cuda", "WARN", f"{cuda}: рендер пойдёт на CPU (медленно)")

    hw = hardware_resources(REPO_ROOT)
    vram_ok = hw.vram_total_gib is None or hw.vram_total_gib >= 8.0
    ram_ok = hw.ram_total_gib is None or hw.ram_total_gib >= 16.0
    section.add("vram", "OK" if vram_ok else "WARN", hw.summary())
    if not vram_ok:
        section.rows[-1].detail += " (меньше 8 GiB — берите профиль VRAM8)"
    if not ram_ok:
        section.rows[-1].detail += " (меньше 16 GiB RAM)"

    for base in ((REPO_ROOT / "cache" / "models"), (REPO_ROOT / "cache")):
        found = False
        for directory in sorted(base.glob("fp8_transformer*")):
            if not directory.is_dir():
                continue
            found = True
            size = sum(item.stat().st_size for item in directory.rglob("*") if item.is_file())
            section.add("fp8-кэш", "OK", f"{_rel(directory)}: {size / 1024 ** 3:.1f} GiB")
        if found:
            break
    return section


def _rel(path: str | Path) -> str:
    try:
        return Path(path).resolve().relative_to(REPO_ROOT).as_posix()
    except (ValueError, OSError):
        return str(path)


# --------------------------------------------------------------------------- #
# Профиль
# --------------------------------------------------------------------------- #

def _profile_section(name: str) -> tuple[Section, list[str]]:
    """Наличие репозиториев профиля и его пресетов; возвращает секцию и список недостающих."""
    from ..cmd.models import repo_available_locally
    from ..config import available_presets

    prof = machine_profile(name)
    section = Section(f"профиль {prof.name} ({prof.hardware}, целевой рендер {prof.target_height}p)")
    missing: list[str] = []
    for entry in prof.repos:
        found = repo_available_locally(entry.repo, entry.local_dir)
        if found:
            section.add(entry.repo, "OK", _rel(found))
        else:
            section.add(entry.repo, "FAIL",
                        "нет локально" + (" (gated: нужен HF_TOKEN)" if entry.gated else ""))
            missing.append(entry.repo)
    known = set(available_presets())
    for preset in prof.presets:
        if preset in known:
            section.add(f"пресет {preset}", "OK", "есть")
        else:
            section.add(f"пресет {preset}", "WARN", "ещё не создан (шаг пресетов)")
    return section, missing


# --------------------------------------------------------------------------- #
# Рендер
# --------------------------------------------------------------------------- #

def synthetic_image(size: int = SYNTHETIC_SIZE) -> Any:
    """Вход для проверки: градиент + простые фигуры. Ни одного файла из репозитория."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)
    for y in range(size):
        t = y / max(1, size - 1)
        draw.line([(0, y), (size, y)],
                  fill=(int(20 + 120 * t), int(60 + 70 * (1.0 - t)), int(170 - 90 * t)))
    draw.ellipse([size * 0.12, size * 0.10, size * 0.48, size * 0.46], fill=(246, 206, 96))
    draw.rectangle([size * 0.55, size * 0.18, size * 0.92, size * 0.52],
                   fill=(34, 34, 46), outline=(250, 250, 250), width=max(2, size // 128))
    draw.polygon([(size * 0.30, size * 0.92), (size * 0.70, size * 0.86),
                  (size * 0.52, size * 0.58)], fill=(206, 62, 96))
    return image


def _extrema_problem(image: Any) -> str | None:
    """Почему кадр считаем «пустым» (`None` = всё в порядке)."""
    extrema = image.getextrema()
    bands = [extrema] if isinstance(extrema[0], int) else list(extrema)
    if all(int(low) == 0 and int(high) == 0 for low, high in bands):
        return "getextrema() нулевые: кадр полностью чёрный"
    if all(int(low) == int(high) for low, high in bands):
        return f"кадр одноцветный (getextrema()={extrema})"
    return None


def _render_section(args: argparse.Namespace) -> tuple[Section, bool, dict[str, Any]]:
    """Один рендер `RENDER_PRESET` / `RENDER_STEPS` шагов; возвращает секцию, code_ok и данные."""
    from PIL import Image

    from ..config import RunConfig
    from ..pipeline import run as run_once

    section = Section(f"рендер (пресет {RENDER_PRESET}, {RENDER_STEPS} шага, вход {SYNTHETIC_SIZE}×"
                      f"{SYNTHETIC_SIZE} синтетический)")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = OUT_DIR / f"selftest-{stamp}-input.png"
    out_path = OUT_DIR / f"selftest-{stamp}.png"
    image = synthetic_image()
    image.save(input_path)
    cfg = RunConfig.from_profile(RENDER_PRESET).merged(**{
        "generation.steps": RENDER_STEPS,
        "generation.offload": RENDER_OFFLOAD,
        "size_px": RENDER_SIZE,
        "input": str(input_path),
        "out": str(out_path),
        "prompt_file": str(REPO_ROOT / PROMPT_FILE),
        "preview": True,
        # R2: selftest — проверка, а не обслуживание кэша: ни одной чужой записи не вытесняем.
        "cache_supersede": "none",
        "logs": False,
    })
    expected = cfg.crop(*image.size)
    section.add("вход", "OK", f"{_rel(input_path)} {image.size[0]}×{image.size[1]} (в памяти)")

    started = time.time()
    try:
        result = run_once(cfg, log=None)
    except Exception as exc:                             # noqa: BLE001 — показываем, а не трейсбек
        if not getattr(args, "keep", False):
            try:
                input_path.unlink()                      # вход нужен только на сам прогон
            except OSError:
                pass
        section.add("прогон", "FAIL", f"{type(exc).__name__}: {exc}")
        return section, False, {"error": f"{type(exc).__name__}: {exc}"}
    wall = time.time() - started

    timings = {key: round(value, 1) for key, value in (result.timings or {}).items()}
    section.add("тайминги", "OK", ", ".join(f"{key} {value} s" for key, value in timings.items())
                or f"{wall:.1f} s")

    saved = Path(result.out_path or out_path)
    if saved != out_path and saved.is_file():
        # Ядро дописывает к имени `-<seed>-<steps>`; у selftest каноничное имя без тегов
        # (`out/tests/selftest-<stamp>.png`), иначе путь в отчёте/acceptance не совпадал бы.
        try:
            saved.replace(out_path)
            saved = out_path
        except OSError:
            pass
    if not saved.is_file():
        section.add("файл", "FAIL", f"кадр не сохранился: {saved}")
        return section, False, {"timings": timings, "wall_s": round(wall, 1)}

    with Image.open(saved) as stored:
        stored.load()
        size = (int(stored.size[0]), int(stored.size[1]))
        problem = _extrema_problem(stored)
        extrema = stored.getextrema()

    size_ok = size == (int(expected[0]), int(expected[1]))
    section.add("размер", "OK" if size_ok else "FAIL",
                f"{size[0]}×{size[1]} (ожидалось {expected[0]}×{expected[1]})")
    section.add("getextrema", "OK" if problem is None else "FAIL", problem or str(extrema))
    section.add("файл", "OK", _rel(saved))

    if not getattr(args, "keep", False):
        try:
            input_path.unlink()                          # синтетический вход нужен только на прогон
        except OSError:
            pass
    else:
        section.add("вход сохранён", "OK", _rel(input_path))

    payload = {
        "frame": _rel(saved),
        "gen_size": list(result.gen_size),
        "out_size": list(result.out_size),
        "expected": list(expected),
        "extrema": [list(band) for band in (extrema if isinstance(extrema[0], tuple) else [extrema])],
        "timings": timings,
        "wall_s": round(wall, 1),
    }
    return section, (not section.failed), payload


def _render_requirements() -> Section:
    """Веса, без которых пробный рендер не пойдёт (repo/lora/энкодер берутся из пресета)."""
    from ..cmd.models import repo_available_locally
    from ..config import RunConfig

    cfg = RunConfig.from_profile(RENDER_PRESET)
    section = Section(f"веса для рендера ({RENDER_PRESET})")
    wanted: list[tuple[str, str | None]] = [(str(cfg.generation.repo), None)]
    lora = cfg.generation.lora
    if lora is not None:
        wanted.append((str(lora.repo), None))
    if getattr(cfg.encoder, "local_dir", None):
        wanted.append((str(cfg.encoder.repo), str(cfg.encoder.local_dir)))
    seen: set[tuple[str, str | None]] = set()
    for repo, local_dir in wanted:
        key = (repo, local_dir)
        if key in seen:
            continue
        seen.add(key)
        found = repo_available_locally(repo, local_dir)
        if found:
            section.add(repo, "OK", _rel(found))
        else:
            section.add(repo, "FAIL", "нет локально")
    return section


# --------------------------------------------------------------------------- #
# Печать и CLI
# --------------------------------------------------------------------------- #

def _print_sections(sections: list[Section]) -> None:
    for section in sections:
        print(f"[selftest] {section.title}")
        width = max((len(row.name) for row in section.rows), default=0)
        for row in section.rows:
            print(f"  {row.name.ljust(width)}  {row.status:<4}  {row.detail}".rstrip())


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", metavar="NAME",
                        help="проверить ещё и веса профиля (" + " | ".join(PROFILE_ORDER) + ")")
    parser.add_argument("--dry-run", action="store_true",
                        help="только проверки, без рендера (весов не грузит)")
    parser.add_argument("--keep", action="store_true",
                        help="оставить синтетический входной кадр рядом с результатом "
                             "(сам кадр остаётся в out/tests/ в любом случае)")
    parser.add_argument("--json", action="store_true", help="вывести результат как JSON")


def run(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    started = time.time()

    env_section = _env_section()
    sections: list[Section] = [env_section]

    missing: list[str] = []
    profile_name = str(getattr(args, "profile", "") or "").strip()
    if profile_name:
        try:
            profile_section, missing = _profile_section(profile_name)
        except ValueError as exc:
            print(f"[selftest] {exc}")
            return 2
        sections.append(profile_section)
    weights_section = _render_requirements()
    sections.append(weights_section)

    # Нет torch/diffusers или весов — рендера не будет: это код 2 с готовой командой загрузки.
    fatal = env_section.failed or weights_section.failed
    payload: dict[str, Any] = {"profile": profile_name or None, "render": None}
    render_ok = True
    if getattr(args, "dry_run", False):
        sections.append(Section("рендер", [Row("пропуск", "OK", "--dry-run: только проверки")]))
    elif fatal:
        sections.append(Section("рендер", [Row(
            "пропуск", "WARN", "нет весов или зависимостей — рендер не запускался")]))
    else:
        section, render_ok, render_payload = _render_section(args)
        sections.append(section)
        payload["render"] = render_payload

    payload["ok"] = bool(not (env_section.failed or weights_section.failed)
                        and render_ok and not sections[-1].failed)
    payload["missing"] = missing
    payload["sections"] = [section.to_dict() for section in sections]
    payload["wall_s"] = round(time.time() - started, 1)

    if json_mode:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        _print_sections(sections)
        if missing:
            hint = ("python -m qwenstyle models download --profile "
                    f"{profile_name or 'VRAM12'} --yes")
            print(f"[selftest] не хватает весов ({len(missing)}): {', '.join(missing)}")
            print(f"  скачать: {hint}")
        print(f"[selftest] {'PASS' if payload['ok'] else 'FAIL'} за {payload['wall_s']} s")

    if env_section.failed or weights_section.failed:
        return 2
    return 0 if payload["ok"] else 1


__all__ = ["NAME", "HELP", "RENDER_PRESET", "add_args", "run", "synthetic_image"]
