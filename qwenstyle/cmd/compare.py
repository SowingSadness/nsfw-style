"""`python -m qwenstyle compare-upscale` — сравнение апскейл-бекендов на одном кадре.

Перенос `tools/upscale_compare.py` на реестр бекендов: имена берутся из
`--backends` (по умолчанию `esrgan,seedvr2`), каждый бекенд доводит кадр до заданной **высоты**
(`--height`, по умолчанию 1440) с сохранением пропорций, кадр сохраняется рядом с исходником или
в `--out-dir`, и печатается одна строка с метриками:

    [esrgan] 864x480 -> 1728x960  0.4s -> canyon--esrgan-960p.png · резкость 12.34 · зерно 3.45 …

Плюс попарное расхождение (`mean|d|`, p99, max), если отработало >= 2 бекенда.

Недоступный бекенд — не ошибка (AGENTS.md R2): печатается
`seedvr2: not installed (<причина>)` и код возврата остаётся 0, если хоть один бекенд отработал.
Ничего не скачивается: `status()` бекендов проверяет только локальные файлы (ESRGAN при
отсутствии весов скачает 17.9 МиБ — это его собственный быстрый путь).

Запуск:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m qwenstyle compare-upscale \\
        --input out/canyon-edit-640-lora1.0-s777.png --height 960 --out-dir out/_cmp
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..config import ConfigError, UpscaleConfig
from ..metrics import image_metrics, pair_diff, target_size
from ..registry import discover

NAME = "compare-upscale"
HELP = "сравнить апскейл-бекенды на одном кадре: метрики, кадры и попарное расхождение"
#: Порядок по умолчанию = как в старом `tools/upscale_compare.py`.
DEFAULT_BACKENDS = "esrgan,seedvr2"
DEFAULT_HEIGHT = 1440
DEFAULT_SEED = 84


# --------------------------------------------------------------------------- #
# Помощники
# --------------------------------------------------------------------------- #

def parse_backends(raw: str | None) -> list[str]:
    """`'esrgan, seedvr2'` -> `['esrgan', 'seedvr2']` (порядок сохраняется, дубли убираются)."""
    names: list[str] = []
    for item in str(raw or "").split(","):
        name = item.strip()
        if name and name not in names:
            names.append(name)
    if not names:
        raise ConfigError("--backends: нужен хотя бы один бекенд, например 'esrgan,seedvr2'")
    return names


def _fmt(value: float | None, key: str) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}" if key == "seconds" else f"{value:.2f}"


def _metric_line(name: str, big: Image.Image, img: Image.Image, saved: str,
                 m: dict[str, float]) -> str:
    """Одна строка на бекенд: размеры, время, файл и метрики."""
    metrics = (f"резкость {_fmt(m['sharpness_lap'], 'sharpness_lap')} · "
               f"зерно {_fmt(m['detail_rms'], 'detail_rms')} · "
               f"насыщенность {_fmt(m['saturation'], 'saturation')}% · "
               f"контраст {_fmt(m['contrast_std'], 'contrast_std')} · "
               f"яркость {_fmt(m['mean_lum'], 'mean_lum')}")
    return (f"[{name}] {big.width}x{big.height} -> {img.width}x{img.height}  "
            f"{_fmt(m['seconds'], 'seconds')}s -> {saved} · {metrics}")


def _not_installed(name: str, reason: str) -> str:
    return f"{name}: not installed ({reason})"


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #

def compare(src: Image.Image, names: list[str], height: int, seed: int,
            out_dir: Path | None, stem: str, color_fix: str = "wavelet",
            ) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray], int]:
    """Прогоняет бекенды и печатает результат.

    Возвращает (метрики по бекендам, массивы кадров, число неудач). Ничего не роняет:
    исключение одного бекенда — строка в лог, остальные продолжают работу.
    """
    registry = discover()
    target_w, target_h = target_size(src.width, src.height, int(height))
    folder = Path(out_dir) if out_dir else Path(".")
    folder.mkdir(parents=True, exist_ok=True)

    print(f"[compare] {src.width}x{src.height} -> {target_w}x{target_h}, "
          f"бекенды: {','.join(names)}", flush=True)

    results: dict[str, dict[str, float]] = {}
    frames: dict[str, np.ndarray] = {}
    failures = 0

    for name in names:
        entry = registry.upscale.get(name)
        if entry is None:
            errors = "; ".join(f"{mod}: {why}" for mod, why in sorted(registry.errors.items()))
            reason = f"бекенд не зарегистрирован; доступны: {', '.join(registry.names('upscale')) or '—'}"
            print(_not_installed(name, reason + (f"; ошибки импорта: {errors}" if errors else "")),
                  flush=True)
            failures += 1
            continue

        backend = entry()
        try:
            ok, why = backend.status()
        except Exception as exc:                        # noqa: BLE001 — статус не должен ронять CLI
            ok, why = False, f"{type(exc).__name__}: {exc}"
        if not ok:
            print(_not_installed(name, why), flush=True)
            failures += 1
            continue

        cfg = UpscaleConfig(backend=name, target_height=target_h, color_fix=color_fix)
        started = time.time()
        try:
            big = backend.upscale(src, factor=target_h / max(1, src.height), cfg=cfg,
                                  seed=int(seed), source=src)
        except Exception as exc:                        # noqa: BLE001 — один бекенд не роняет прогон
            print(f"[{name}] ошибка: {type(exc).__name__}: {exc}", flush=True)
            failures += 1
            continue
        elapsed = time.time() - started

        img = big if big.size == (target_w, target_h) else \
            big.resize((target_w, target_h), Image.LANCZOS)
        path = folder / f"{stem}--{name}-{target_h}p.png"
        saved = path.name
        try:
            img.save(path)
        except OSError as exc:                          # права/диск: кадр не сохранился
            saved = f"(не сохранён: {exc})"
            print(f"[{name}] не удалось сохранить {path}: {exc}", file=sys.stderr, flush=True)

        arr = np.asarray(img)
        m = image_metrics(arr)
        m["seconds"] = round(elapsed, 1)
        results[name] = m
        frames[name] = arr
        print(_metric_line(name, big, img, saved, m), flush=True)

    if len(frames) >= 2:
        first, second = list(frames)[:2]
        d = pair_diff(frames[first], frames[second])
        print(f"\n[разница] {first} vs {second}: mean|d| {d['mean']:.2f}, "
              f"p99 {d['p99']:.0f}, max {d['max']:.0f}", flush=True)

    return results, frames, failures


def _cmd_compare(args: argparse.Namespace) -> int:
    path = Path(args.input)
    if not path.is_file():
        print(f"[compare] нет входного кадра {path}", file=sys.stderr)
        return 2
    try:
        with Image.open(path) as im:
            src = im.convert("RGB").copy()
    except Exception as exc:                            # noqa: BLE001
        print(f"[compare] не удалось прочитать {path}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 2

    names = parse_backends(args.backends)
    out_dir = Path(args.out_dir) if args.out_dir else path.parent
    results, frames, failures = compare(src, names, args.height, args.seed, out_dir, path.stem,
                                       color_fix=args.color_fix)

    if not results:
        print(f"[compare] ни один бекенд не отработал ({failures} недоступно/с ошибкой)",
              file=sys.stderr)
        return 1
    saved = ", ".join(f"{name} {frames[name].shape[1]}x{frames[name].shape[0]}"
                      for name in frames)
    print(f"\n[compare] кадры в {out_dir}: {saved}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# CLI-контракт подкоманды
# --------------------------------------------------------------------------- #

def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, metavar="PATH",
                        help="входной кадр (например out/canyon-edit-640-lora1.0-s777.png)")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, metavar="N",
                        help=f"целевая высота кадра в пикселях (по умолчанию {DEFAULT_HEIGHT}); "
                             "ширина считается по пропорциям и округляется до кратности 16")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, metavar="N",
                        help=f"seed для генеративных бекендов (по умолчанию {DEFAULT_SEED}); "
                             "ESRGAN его игнорирует")
    parser.add_argument("--out-dir", default=None, metavar="DIR",
                        help="куда сохранять кадры (по умолчанию — рядом с --input)")
    parser.add_argument("--backends", default=DEFAULT_BACKENDS, metavar="LIST",
                        help=f"через запятую, по умолчанию {DEFAULT_BACKENDS}; "
                             f"зарегистрированные: см. `models list`")
    parser.add_argument("--color-fix", dest="color_fix",
                        choices=("none", "wavelet", "reinhard"), default="wavelet",
                        help="цветокоррекция по исходнику для генеративных бекендов "
                             "(по умолчанию wavelet; ESRGAN её не использует)")


def run(args: argparse.Namespace) -> int:
    return _cmd_compare(args)


__all__ = ["DEFAULT_BACKENDS", "DEFAULT_HEIGHT", "DEFAULT_SEED", "HELP", "NAME", "add_args",
           "compare", "parse_backends", "run"]
