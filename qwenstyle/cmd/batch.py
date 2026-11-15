"""`python -m qwenstyle batch` — N seed'ов за одну загрузку модели + пакетный апскейл.

Перенос `tools/batch_seeds.py` на каркас (шаг 8 плана `qwenstyle-framework`). Здесь **только**
CLI: загрузка модели, генерация, освобождение VRAM и апскейл живут в `qwenstyle/pipeline.py`
(`run_batch`), имена файлов — в `qwenstyle/io.py` (`batch_paths`), параметры — в `RunConfig`
(`batch.seeds`, `batch.out_dir`, `batch.keep_gen`, `batch.resume`, `upscale.*`).
Ни одного `torch`/`diffusers`/апскейл-импорта в этом модуле нет.

Зачем батч: одиночный `run` платит старт пайплайна (~5–6 s: fp8-кэш + `.to("cuda")` + LoRA)
на каждый seed. Здесь пайплайн грузится **один раз**, генерируется N кадров на черновом
разрешении, потом пайплайн выгружается и все кадры апскейлятся до целевой высоты.

Кэш эмбеддингов (AGENTS.md, R1) должен уже существовать: батч его **не создаёт** (иначе платил бы
55–255 с CPU-энкода). Если `dry_run()["need_encoder"]` — печатается готовая команда одиночного
прогона и возвращается код 2, как в старом скрипте.

Логи (`qwenstyle/runlog.py`): одна папка на сессию — `<out-dir>/_logs/<YYYYmmdd-HHMMSS>/run.log` и
`meta.json` (кадров в `_logs` нет — только лог и параметры). `--no-logs` отключает и папку, и tee.

Маппинг старых флагов `tools/batch_seeds.py` (все значения идут в `RunConfig`, а не в argparse):

    --seeds SPEC          -> batch.seeds          (== `--seed-range`)
    --gen-res N           -> resolution           (sqrt площади crop, как раньше; шаги 8 при N<=640)
    --out-res N           -> upscale.target_height (через высоту профиля: 1024 -> 768 при 16:9)
    --final-h N           -> upscale.target_height (== `--upscale-to`)
    --upscale F           -> upscale.factor
    --out-dir DIR         -> batch.out_dir
    --weights PATH        -> upscale.weights      (== --upscale-weights)
    --tile N              -> upscale.tile         (== --upscale-tile)
    --lora-scale F        -> generation.lora.scale
    --steps N             -> generation.steps
    новые/общие: --config/--model/--set, --gen-size/--size (высота кадра), --upscale-to (высота),
    --offload, --text-encoder, --upscale-backend, --keep-gen/--no-keep-gen, --resume/--no-resume.

Запуск:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m qwenstyle batch \\
        --config preview-qwen --seed-range 0-1 --input out/canyon-edit-640-lora1.0-s777.png \\
        --prompt-file promts/exp3.txt --out-dir out/batch-check --upscale-to 768p
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..config import (
    DEFAULT_STEPS,
    PREVIEW_STEPS,
    ConfigError,
    RunConfig,
    available_presets,
)
from . import parse_sets

NAME = "batch"
HELP = ("N seed'ов за одну загрузку модели: черновик (--config preview-qwen/--gen-size), затем "
        "пакетный апскейл до --upscale-to (перенос tools/batch_seeds.py)")
DEFAULT_CONFIG = "base"
DEFAULT_SEEDS = "0-9"
DEFAULT_OUT_DIR = "out/batch"


# --------------------------------------------------------------------------- #
# Парсинг размеров
# --------------------------------------------------------------------------- #

def parse_height(spec: str) -> int:
    """`'768p'` / `'768'` -> 768 (высота кадра в пикселях), как `--size` в run_style.py."""
    text = str(spec).strip().lower().removesuffix("p").strip()
    try:
        height = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"ожидалась высота вида 768p или 768, получено {spec!r}"
        ) from None
    if not 256 <= height <= 4096:
        raise argparse.ArgumentTypeError(f"высота {height} вне диапазона 256..4096")
    return height


def out_res_height(resolution: int, in_size: tuple[int, int]) -> int:
    """Старый `--out-res N` (resolution профиля) -> высота кадра в пикселях.

    Тот же расчёт, что делает `calculate_dimensions(N**2, ratio)`: высота округляется до
    кратной 32, поэтому `--out-res 1024` при 16:9 даёт 768 — эквивалент `--upscale-to 768p`.
    Формула повторена здесь, чтобы `cmd/batch.py` не тянул diffusers ради одного числа.
    """
    width, height = int(in_size[0]), int(in_size[1])
    if width <= 0 or height <= 0:
        raise ConfigError(f"--out-res: некорректный размер входа {width}x{height}")
    ratio = width / height
    px = int(resolution) / (ratio ** 0.5)
    return max(32, int(round(px / 32)) * 32)


def _input_size(path: str | Path) -> tuple[int, int]:
    """Размер входа без декодирования пикселей (PIL открывает лениво)."""
    from PIL import Image

    with Image.open(path) as image:
        return int(image.size[0]), int(image.size[1])


# --------------------------------------------------------------------------- #
# Сборка конфига
# --------------------------------------------------------------------------- #

def pick_preset(args: argparse.Namespace) -> str:
    """`--config` и `--model` — один и тот же выбор пресета (base/turbo/preview/…)."""
    config = getattr(args, "config", None)
    model = getattr(args, "model", None)
    if config and model and config != model:
        raise ConfigError(
            f"--config {config!r} и --model {model!r} выбирают разные пресеты; оставьте один "
            f"(доступны: {', '.join(available_presets()) or '—'})"
        )
    return str(model or config or DEFAULT_CONFIG)


def build_config(args: argparse.Namespace) -> RunConfig:
    """Флаги старого скрипта -> пути `RunConfig`; пресет главнее, флаги — поверх него."""
    overrides: dict[str, Any] = dict(parse_sets(getattr(args, "sets", None)))
    preset = pick_preset(args)

    # батч-секция задаётся целиком: пресеты её не содержат, а `run_batch` требует seeds/out_dir
    batch: dict[str, Any] = {
        "seeds": args.seed_range or DEFAULT_SEEDS,
        "out_dir": args.out_dir or DEFAULT_OUT_DIR,
    }
    if args.keep_gen is not None:
        batch["keep_gen"] = bool(args.keep_gen)
    if args.resume is not None:
        batch["resume"] = bool(args.resume)
    overrides["batch"] = batch
    overrides["input"] = args.input
    overrides["prompt_file"] = args.prompt_file

    # черновое разрешение: --gen-size/--size — высота, --gen-res/--resolution — старый resolution
    if args.gen_size is not None:
        overrides["size_px"] = int(args.gen_size)
    if args.gen_res is not None:
        overrides["resolution"] = int(args.gen_res)
    if getattr(args, "aspect", None) is not None:
        overrides["aspect"] = str(args.aspect)

    if args.steps is not None:
        overrides["generation.steps"] = int(args.steps)
    elif args.gen_res is not None and preset != "turbo" \
            and "generation.steps" not in overrides:
        # правило старого скрипта: 8 шагов на черновом 640, иначе 25
        overrides["generation.steps"] = (
            PREVIEW_STEPS if args.gen_res <= 640 else DEFAULT_STEPS
        )

    targets = [value for value in (args.upscale_to, args.final_h, args.out_res)
               if value is not None]
    if len(targets) > 1:
        raise ConfigError("--upscale-to / --final-h / --out-res задают одно и то же — оставьте один")
    if args.upscale_to is not None:
        overrides["upscale.target_height"] = int(args.upscale_to)
    elif args.final_h is not None:
        overrides["upscale.target_height"] = int(args.final_h)
    elif args.out_res is not None:
        height = out_res_height(int(args.out_res), _input_size(args.input))
        print(f"[batch] --out-res {args.out_res} (старый resolution профиля) -> высота {height}px",
              flush=True)
        overrides["upscale.target_height"] = height
    if args.upscale is not None:
        overrides["upscale.factor"] = float(args.upscale)

    if args.lora_scale is not None:
        overrides["generation.lora.scale"] = float(args.lora_scale)
    if args.offload is not None:
        overrides["generation.offload"] = args.offload
    if args.text_encoder is not None:
        overrides["encoder.name"] = args.text_encoder      # cache_tag следует за именем (D7)
    if args.upscale_backend is not None:
        overrides["upscale.backend"] = args.upscale_backend
    if args.upscale_weights is not None:
        overrides["upscale.weights"] = args.upscale_weights
    if args.upscale_tile is not None:
        overrides["upscale.tile"] = int(args.upscale_tile)

    # логи: одна папка на сессию батча (<out-dir>/_logs/<stamp>/); --no-logs — выкл
    if getattr(args, "no_logs", False):
        overrides["logs"] = False

    return RunConfig.from_profile(preset, **overrides)


# --------------------------------------------------------------------------- #
# Печать
# --------------------------------------------------------------------------- #

def _fmt_scale(value: Any) -> str:
    return f"{float(value):g}" if isinstance(value, (int, float)) else "—"


def describe(cfg: RunConfig, info: dict[str, Any]) -> str:
    """Одна строка конфига: что именно будет сгенерировано и где это лежит."""
    batch = info.get("batch") or {}
    lora = (info.get("lora") or {}).get("scale")
    target = info.get("upscale_target_height")
    final = "без апскейла" if not info.get("upscale_factor") else \
        f"{info.get('out_size')} ({'высота ' + str(target) + 'px' if target else 'factor ' + _fmt_scale(info.get('upscale_factor'))})"
    return (f"[batch] пресет {cfg.preset or '-'} · gen {info.get('crop')} ({info.get('steps')} шагов, "
            f"resolution {info.get('resolution')}) · seeds {batch.get('seeds')} · финал {final} · "
            f"lora {_fmt_scale(lora)} · embeds {Path(info.get('embeds_path', '')).name} · "
            f"out {batch.get('out_dir')}")


def _missing_embeds(cfg: RunConfig, info: dict[str, Any]) -> None:
    """Что делать при промахе кэша: батч не энкодит, нужен один одиночный прогон."""
    stream = sys.stdout
    print(f"[abort] нет кэша эмбеддингов: {info.get('embeds_path')} "
          f"(image {info.get('image_sha')} @ {info.get('crop')})", file=stream, flush=True)
    print("        батч не кодирует промт/картинку сам: CPU-энкод (55 с @480p … 255 с @768p) "
          "платится один раз, дальше все seed'ы идут из кэша", file=stream, flush=True)
    command = ["  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m qwenstyle run",
               f"      --config {cfg.preset or DEFAULT_CONFIG}"]
    if cfg.input:
        command.append(f"      --input {cfg.input}")
    if cfg.prompt_file:
        command.append(f"      --prompt-file {cfg.prompt_file}")
    if cfg.size_px:
        command.append(f"      --size {cfg.size_px}p")
    elif cfg.resolution:
        command.append(f"      --resolution {cfg.resolution}")
    if cfg.encoder.cache_tag not in ("", "base"):
        command.append(f"      --text-encoder {cfg.encoder.name}")
    command.append("      --out out/<любое>.png")
    print("        создать его одним одиночным прогоном с теми же промтом/входом/разрешением:",
          file=stream, flush=True)
    print("\n".join(command), file=stream, flush=True)
    print("        после этого повторный запуск batch подхватит готовый кэш (resume — по умолчанию)",
          file=stream, flush=True)


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #

def _check_files(args: argparse.Namespace) -> str | None:
    """Понятная ошибка до тяжёлых импортов: нет входа/файла промта -> код 2 без трейсбека."""
    if not Path(args.input).is_file():
        return f"нет входного кадра {args.input} (--input)"
    if args.prompt_file and not Path(args.prompt_file).is_file():
        return f"нет файла промта {args.prompt_file} (--prompt-file)"
    return None


def run(args: argparse.Namespace) -> int:
    problem = _check_files(args)
    if problem:
        print(f"[batch] {problem}", file=sys.stderr, flush=True)
        return 2
    try:
        cfg = build_config(args)
    except (ConfigError, OSError) as exc:
        print(f"[batch] {exc}", file=sys.stderr, flush=True)
        return 2

    from .. import pipeline      # локальный импорт: `batch --help` не должен тянуть каркас
    from ..runlog import RunLog

    # Tee с самого начала: `describe` и строки `dry_run` (план вытеснения) — часть лога.
    with RunLog(cfg.logs) as log:
        info = pipeline.dry_run(cfg)
        if info.get("need_encoder"):
            _missing_embeds(cfg, info)
            return 2

        print(describe(cfg, info), flush=True)
        results = pipeline.run_batch(cfg, log=log)
        if not results:         # всё пропущено через --resume: итоговая строка всё равно нужна
            print(f"[done] 0 кадров в {cfg.batch.out_dir if cfg.batch else '-'} "
                  f"(всё уже готово)", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# CLI-контракт подкоманды
# --------------------------------------------------------------------------- #

def add_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_argument_group("что и чем генерировать")
    source.add_argument("--config", default=None, metavar="PRESET",
                        help=f"пресет ({', '.join(available_presets())}) или путь к YAML; "
                             f"по умолчанию {DEFAULT_CONFIG}")
    source.add_argument("--model", default=None, metavar="PRESET",
                        help="синоним --config для выбора модели (base/turbo)")
    source.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                        help="точечный параметр RunConfig, например generation.sampler_kwargs.offload_blocks=4 "
                             "(можно повторять)")
    source.add_argument("--input", required=True, metavar="PATH",
                        help="входной кадр (он же conditioning; ключ кэша зависит от его sha256)")
    source.add_argument("--prompt-file", required=True, metavar="PATH",
                        help="промт из файла (promts/*.txt): смена текста = новый кэш эмбеддингов")
    source.add_argument("--seed-range", "--seeds", dest="seed_range", default=DEFAULT_SEEDS,
                        metavar="SPEC",
                        help=f"seed'ы: '0-9', '1,7,42', '0-2,10' (по умолчанию {DEFAULT_SEEDS})")
    source.add_argument("--out-dir", default=DEFAULT_OUT_DIR, metavar="DIR",
                        help=f"куда писать s{{N}}-gen.png и s{{N}}.png (по умолчанию {DEFAULT_OUT_DIR})")
    source.add_argument("--gen-size", "--size", dest="gen_size", type=parse_height, default=None,
                        metavar="NNNp",
                        help="высота черновика в пикселях (480p/768p); иначе — из --config/--gen-res")
    source.add_argument("--aspect", choices=("16:9", "4:3", "1:1"), default=None, metavar="R",
                        help="пропорция кадра черновика (по умолчанию — как у входа)")
    source.add_argument("--gen-res", "--resolution", dest="gen_res", type=int, default=None,
                        metavar="N",
                        help="старый resolution профиля (640 = 480p, 1024 = 768p); при N<=640 "
                             "шаги по умолчанию 8, иначе 25")

    upscale = parser.add_argument_group("апскейл (после выгрузки пайплайна)")
    upscale.add_argument("--upscale-to", type=parse_height, default=None, metavar="NNNp",
                         help="целевая высота финального кадра (768p); ширина — по пропорциям")
    upscale.add_argument("--out-res", type=int, default=None, metavar="N",
                         help="старый флаг tools/batch_seeds.py: resolution профиля, "
                              "пересчитывается в высоту (1024 -> 768p при 16:9)")
    upscale.add_argument("--final-h", type=int, default=None, metavar="N",
                         help="старый --final-h: точная высота в пикселях (== --upscale-to)")
    upscale.add_argument("--upscale", type=float, default=None, metavar="F",
                         help="множитель апскейла (0 = выкл); игнорируется, если задана высота")
    upscale.add_argument("--upscale-backend", default=None, metavar="NAME",
                         help="esrgan (по умолчанию) или seedvr2 — из реестра бекендов")
    upscale.add_argument("--upscale-weights", "--weights", dest="upscale_weights", default=None,
                         metavar="PATH",
                         help="веса ESRGAN (старый --weights); по умолчанию из пресета")
    upscale.add_argument("--upscale-tile", "--tile", dest="upscale_tile", type=int, default=None,
                         metavar="N",
                         help="тайл ESRGAN (старый --tile; 256 даёт результат бит-в-бит с целым кадром)")

    common = parser.add_argument_group("общие оси (бесплатные: не меняют кэш эмбеддингов)")
    common.add_argument("--steps", type=int, default=None, metavar="N",
                        help="шаги сэмплера; по умолчанию из пресета (8 в preview, 25 в base)")
    common.add_argument("--lora-scale", type=float, default=None, metavar="F",
                        help="сила LoRA (0.7 база, 0.85–1.0 структура, 0.65–0.70 правка картинки)")
    common.add_argument("--offload", choices=("none", "group"), default=None,
                        help="group = блоки трансформера стримятся с CPU (960p+ на 16 ГБ VRAM)")
    common.add_argument("--text-encoder", choices=("base", "heretic"), default=None,
                        help="heretic = DreamFast/Qwen3-VL-8B-Heretic-1.3.0 (другой ключ кэша!)")

    files = parser.add_argument_group("файлы и повторный запуск")
    files.add_argument("--keep-gen", action=argparse.BooleanOptionalAction, default=None,
                       help="хранить s{N}-gen.png рядом с апскейлом (по умолчанию да)")
    files.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None,
                       help="пропускать seed'ы, у которых s{N}.png уже есть (по умолчанию да)")

    logs = parser.add_argument_group("логи")
    logs.add_argument("--no-logs", dest="no_logs", action="store_true",
                      help="не создавать подпапку с отладочным логом (по умолчанию "
                           "<out-dir>/_logs/<время>/run.log)")


__all__ = ["DEFAULT_CONFIG", "DEFAULT_OUT_DIR", "DEFAULT_SEEDS", "HELP", "NAME", "add_args",
           "build_config", "describe", "out_res_height", "parse_height", "pick_preset", "run"]
