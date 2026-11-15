"""`python -m qwenstyle run` — одиночный кадр: все флаги `run_style.py` 1:1 + каркасные.

Здесь **только** CLI: загрузка модели, кэш эмбеддингов, генерация, апскейл и сохранение живут
в `qwenstyle/pipeline.py`; имена файлов — в `qwenstyle/io.py`; параметры — в `RunConfig`.
Ни одного `torch`/`diffusers`-импорта в этом модуле нет: `run --help` мгновенный.

Маппинг старых флагов `run_style.py` (значение -> путь в `RunConfig`):

    --input PATH               -> inputs (кортеж; повтор флага = несколько референсов),
                                  а input остаётся алиасом первого (inputs[0])
    --out PATH                 -> out                      (теги -x{factor} и -s{seed} добавит io)
    --lora-scale F             -> generation.lora.scale     (бесплатная ось: ключ кэша не меняет)
    --model base|turbo         -> пресет (base/turbo)       (turbo — только по явному флагу)
    --preview                  -> preview=True + resolution=640 + steps=8
    --steps N                  -> generation.steps
    --size NNNp                -> size_px                   (высота кадра; resolution из пропорций)
    --aspect 16:9|4:3|1:1      -> aspect                    (пропорция кадра вместо пропорций входа)
    --resolution N             -> resolution
    --seed N                   -> seed
    --fp8 0|1                  -> generation.precision = fp8|bf16
    --fp8-cast-cache DIR       -> generation.precision_cache  ('' = не кэшировать)
    --rebuild-fp8-cache        -> generation.sampler_kwargs.rebuild_fp8_cache
    --offload none|group       -> generation.offload
    --offload-blocks N         -> generation.sampler_kwargs.offload_blocks
    --offload-stream 0|1       -> generation.sampler_kwargs.offload_stream
    --prompt-file PATH         -> prompt_file
    --upscale F                -> upscale.factor
    --upscale-backend NAME     -> upscale.backend
    --upscale-weights PATH     -> upscale.weights
    --upscale-tile N           -> upscale.tile
    --upscale-color-fix MODE   -> upscale.color_fix
    --upscale-color-strength F -> upscale.color_strength
    --upscale-color-sigma F    -> upscale.color_sigma
    --text-encoder NAME        -> encoder.name            (cache_tag следует за именем, D7)

Три «модельных» ключа (`rebuild_fp8_cache`/`offload_blocks`/`offload_stream`) не про сэмплер,
но в `RunConfig` они живут в `generation.sampler_kwargs` — так их читает бекенд
`qwen_image21`, и так они не попадают в ключ кэша эмбеддингов. Объявлены эти ключи
бекендом (`parameters()`), а не ядром: неизвестный `generation.sampler_kwargs.*` отбивается до
загрузки весов, а таблица всех параметров — `config show <preset> --explain-params`.

Новое (не было в `run_style.py`):

    --config PRESET|PATH       -> пресет `qwenstyle/presets/*.yaml` (base/turbo/preview/flux2/…)
    --set KEY=VALUE            -> точечный параметр RunConfig (повторяемый)
    --dry-run                  -> печать плана (crop, ключ кэша, нужен ли энкодер, шаги, фактор)
                                  без загрузки весов; exit 0
    --no-logs                  -> не создавать подпапку кадра с логом (`RunConfig.logs=False`)

Логи (`qwenstyle/runlog.py`): каждый прогон пишет свой `run.log` и `meta.json` в папку кадра
(`out/x-s7.png` -> `out/x-s7/run.log`). Tee включается **до** первой печати (с `[profile]`),
а файл открывается в `pipeline.run` сразу после `_plan` — только там известно имя кадра с тегами.

Поведение веток сохранено: профиль по умолчанию base; `--size <= 480p` включает черновые
8 шагов (как `--preview`); `steps` по умолчанию 8 в preview и 25 в base (у turbo — 8 из пресета
и расписания `sample_sigmas`); `--resolution` по умолчанию 640 в preview и 1024 в base
(в turbo-пресете — 640 при `--preview`).

Все «значимые» флаги (seed, lora-scale, fp8, offload*, upscale*) объявлены с `default=None`:
их действующие значения живут в пресете, а флаг добавляет override только когда его явно указали.
Иначе дефолт argparse перебивал бы пресет — например `--lora-scale 0.7` врезался бы в
`generation.lora: null` у `flux2` (ConfigError).

Запуск:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m qwenstyle run --preview \\
        --input out/canyon-edit-640-lora1.0-s777.png --prompt-file promts/exp3.txt \\
        --out out/p.png
    .venv/Scripts/python.exe -m qwenstyle run --dry-run --config preview --steps 8 \\
        --input out/canyon-edit-640-lora1.0-s777.png --prompt-file promts/exp3.txt --out out/p.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ..config import (
    ASPECT_RATIOS,
    PREVIEW_MAX_SIZE_PX,
    PREVIEW_RESOLUTION,
    PREVIEW_STEPS,
    ConfigError,
    RunConfig,
    available_presets,
    readonly_params,
)
from . import parse_sets

NAME = "run"
HELP = "одиночный кадр: все флаги run_style.py + --config/--set/--dry-run"
DEFAULT_CONFIG = "base"
DEFAULT_OUT = "out/prison-school-style.png"
DEFAULT_SEED = 20261113
DEFAULT_LORA_SCALE = 0.7
DEFAULT_WEIGHTS = "cache/models/weights/RealESRGAN_x4plus_anime_6B.pth"
#: Адаптер базового профиля (FUSAL R80 под Qwen-Image-2.1): по нему и по трансформеру
#: определяется «тот же самый base», а не отдельная модель.
BASE_LORA = "fusal-qwen-image-2.1-v2-exact-r80.safetensors"


# --------------------------------------------------------------------------- #
# Парсинг и сборка конфига
# --------------------------------------------------------------------------- #

def parse_size(spec: str) -> int:
    """`'864p'` / `'864'` -> 864 (высота кадра), как `parse_size_px` в run_style.py."""
    text = str(spec).strip().lower().removesuffix("p").strip()
    try:
        height = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--size ждёт высоту вида 864p или 864, получено {spec!r}"
        ) from None
    if not 256 <= height <= 4096:
        raise argparse.ArgumentTypeError(f"--size: высота {height} вне диапазона 256..4096")
    return height


def pick_preset(args: argparse.Namespace) -> str:
    """`--config` и `--model` — один и тот же выбор пресета; `--preview` без них = preview."""
    config = getattr(args, "config", None)
    model = getattr(args, "model", None)
    if config and model and config != model:
        raise ValueError(
            f"--config {config!r} и --model {model!r} выбирают разные пресеты; оставьте один "
            f"(доступны: {', '.join(available_presets()) or '—'})"
        )
    if model or config:
        return str(model or config)
    return "preview" if getattr(args, "preview", False) else DEFAULT_CONFIG


def model_label(cfg: RunConfig) -> str:
    """Метка семейства для строки `[profile]`.

    Возвращает `base`, если прогон идёт на **той же** модели, что и профиль по умолчанию
    (Qwen-Image-2.1 + FUSAL R80): другой размер/шаги/апскейл — это всё ещё `base`, поэтому
    `preview-qwen`, `full-qwen`, `qwen`, `anime` печатаются как `base`.
    Всё остальное возвращается именем пресета: `turbo` (другой трансформер), а также FLUX.2-профили.

    Решается по конфигу, а не списком имён: новый пресет на базовой модели получает `base`
    автоматически, и его не надо регистрировать здесь (так был допущен баг: `preview` остался
    в старом списке `BASE_PRESETS` после того, как стал FLUX.2, и печатался неверно).
    """
    preset = str(cfg.preset or DEFAULT_CONFIG)
    gen = cfg.generation
    if str(gen.backend) != "qwen_image21":
        return preset                                   # FLUX.2 и любые другие семейства
    transformer = gen.transformer or ()
    if transformer and str(transformer[0]) != str(gen.repo):
        return preset                                   # turbo/hybrid: трансформер из другого репозитория
    lora = getattr(gen.lora, "filename", None) or ""
    if lora and lora != BASE_LORA:
        return preset                                   # другой адаптер = другой профиль
    return "base"


def build_config(args: argparse.Namespace) -> RunConfig:
    """Все флаги -> `RunConfig`: пресет главнее умолчаний, явные флаги главнее пресета."""
    preset = pick_preset(args)
    overrides: dict[str, Any] = dict(parse_sets(getattr(args, "sets", None)))
    preview = bool(getattr(args, "preview", False))
    size_px = getattr(args, "size", None)
    resolution = getattr(args, "resolution", None)
    steps = getattr(args, "steps", None)

    # размеры: --size (высота) главнее --resolution; --preview = 640, если не задано ничего
    if size_px is not None:
        overrides["size_px"] = int(size_px)
    if resolution is not None:
        overrides["resolution"] = int(resolution)
    aspect = getattr(args, "aspect", None)
    if aspect is not None:
        overrides["aspect"] = str(aspect)          # "16:9" | "4:3" | "1:1"
    elif preview and size_px is None:
        overrides["resolution"] = PREVIEW_RESOLUTION

    # шаги: явные главнее; turbo берёт свои 8 из пресета (и расписания sample_sigmas),
    # 480p и ниже — черновой режим, как --preview
    if steps is not None:
        overrides["generation.steps"] = int(steps)
    elif "generation.steps" not in overrides and (
        preview or (size_px is not None and int(size_px) <= PREVIEW_MAX_SIZE_PX)
    ) and preset != "turbo" and preset != "preview":
        overrides["generation.steps"] = PREVIEW_STEPS
    if preview:
        overrides["preview"] = True

    if getattr(args, "inputs", None):
        overrides["inputs"] = tuple(args.inputs)
    if getattr(args, "out", None):
        overrides["out"] = args.out
    else:
        # --out не указан: путь тоже берём из пресета, а если его там нет — исторический дефолт
        overrides.setdefault("out", DEFAULT_OUT)
    if getattr(args, "prompt_file", None):
        overrides["prompt_file"] = args.prompt_file
    if getattr(args, "seed", None) is not None:
        overrides["seed"] = int(args.seed)
    if getattr(args, "lora_scale", None) is not None:
        overrides["generation.lora.scale"] = float(args.lora_scale)

    # точность и fp8-кэш трансформера
    if getattr(args, "fp8", None) is not None:
        overrides["generation.precision"] = "fp8" if int(args.fp8) else "bf16"
    if getattr(args, "fp8_cast_cache", None) is not None:
        overrides["generation.precision_cache"] = str(args.fp8_cast_cache)
    if getattr(args, "rebuild_fp8_cache", False):
        overrides["generation.sampler_kwargs.rebuild_fp8_cache"] = True

    # раскладка памяти
    if getattr(args, "offload", None) is not None:
        overrides["generation.offload"] = str(args.offload)
    if getattr(args, "offload_blocks", None) is not None:
        overrides["generation.sampler_kwargs.offload_blocks"] = int(args.offload_blocks)
    if getattr(args, "offload_stream", None) is not None:
        overrides["generation.sampler_kwargs.offload_stream"] = int(args.offload_stream)

    # постобработка
    if getattr(args, "upscale", None) is not None:
        overrides["upscale.factor"] = float(args.upscale)
    if getattr(args, "upscale_backend", None) is not None:
        overrides["upscale.backend"] = str(args.upscale_backend)
    if getattr(args, "upscale_weights", None) is not None:
        overrides["upscale.weights"] = str(args.upscale_weights)
    if getattr(args, "upscale_tile", None) is not None:
        overrides["upscale.tile"] = int(args.upscale_tile)
    if getattr(args, "upscale_color_fix", None) is not None:
        overrides["upscale.color_fix"] = str(args.upscale_color_fix)
    if getattr(args, "upscale_color_strength", None) is not None:
        overrides["upscale.color_strength"] = float(args.upscale_color_strength)
    if getattr(args, "upscale_color_sigma", None) is not None:
        overrides["upscale.color_sigma"] = float(args.upscale_color_sigma)

    # энкодер: cache_tag следует за именем (D7: записи two энкодеров не смешиваются)
    if getattr(args, "text_encoder", None):
        overrides["encoder.name"] = str(args.text_encoder)

    # логи: отладочный лог в подпапке кадра (`qwenstyle/runlog.py`); `--no-logs` — выкл
    if getattr(args, "no_logs", False):
        overrides["logs"] = False

    cfg = RunConfig.from_profile(preset, **overrides)
    # `merged()` приводит строку 'none' к None (это удобно для `--set`, но не для флагов, где
    # 'none' — валидное значение перечисления). Возвращаем строку, чтобы конфиг и логи
    # (`config show`, `--dry-run`) показывали реальное значение, а не '-'.
    from dataclasses import replace

    generation, upscale = cfg.generation, cfg.upscale
    if not generation.offload:                      # 'none' из флага, `--set` или пресета
        generation = replace(generation, offload="none")
    if not upscale.color_fix:
        upscale = replace(upscale, color_fix="none")
    if generation is not cfg.generation or upscale is not cfg.upscale:
        cfg = replace(cfg, generation=generation, upscale=upscale)
    # Параметры объявляет адаптер модели (`parameters()`): если флаг/`--set` попал в readonly
    # (например `steps` у MiniMax-H3: расписание PDD задаёт 6 шагов), предупреждаем.
    for spec in readonly_params(cfg, overrides):
        print(f"[run] {spec.readonly_note()}, значение проигнорировано", flush=True)
    return cfg


# --------------------------------------------------------------------------- #
# Печать плана (--dry-run)
# --------------------------------------------------------------------------- #

def _yes_no(value: Any) -> str:
    return "yes" if value else "no"


def print_plan(cfg: RunConfig, info: dict[str, Any]) -> None:
    """План прогона без весов: сетка, ключ кэша, нужен ли энкодер, шаги, фактор, имя файла."""
    lora = (info.get("lora") or {}).get("scale")
    lora_text = f"{float(lora):g}" if lora is not None else "—"
    print(f"[plan] backend={info.get('backend')} preset={cfg.preset or '-'} seed={info.get('seed')} "
          f"size_px={info.get('size_px') or '-'} resolution={info.get('resolution')}", flush=True)
    inputs = info.get("inputs") or []
    refs = f", {len(inputs)} референсов" if len(inputs) > 1 else ""
    print(f"[plan] input {info.get('input')} ({info.get('in_size')}{refs}) -> crop {info.get('crop')}",
          flush=True)
    if len(inputs) > 1:                              # одиночный путь печатается как раньше
        print(f"[plan] референсы ({len(inputs)}): {', '.join(inputs)}", flush=True)
    print(f"[plan] steps={info.get('steps')} precision={info.get('precision')} "
          f"offload={info.get('offload')} guidance={info.get('guidance')} lora={lora_text}", flush=True)
    print(f"[plan] cache key={info.get('embeds_key')} (stage={info.get('embeds_stage')}, "
          f"cached={_yes_no(info.get('embeds_cached'))}) -> encoder needed: "
          f"{_yes_no(info.get('need_encoder'))}", flush=True)
    print(f"[plan] cache file {info.get('embeds_path')} (dir {info.get('cache_dir')})", flush=True)
    target = info.get("upscale_target_height")
    print(f"[plan] upscale {info.get('upscale_backend')} factor={float(info.get('upscale_factor') or 0):g} "
          f"target_height={target if target else '-'} -> out {info.get('out_size')}", flush=True)
    print(f"[plan] out file {info.get('out_path')}", flush=True)
    if info.get("batch"):
        batch = info["batch"]
        print(f"[plan] batch seeds={batch.get('seeds')} out_dir={batch.get('out_dir')} "
              f"keep_gen={_yes_no(batch.get('keep_gen'))} resume={_yes_no(batch.get('resume'))} "
              f"pending={batch.get('pending')}", flush=True)
    mode = "dry-run" if info.get("dry_run") else "plan"
    print(f"[plan] {mode}: модель не загружается, CPU-энкод не платится "
          f"(offline={'yes' if info.get('offline') else 'no'})", flush=True)


def _profile_line(cfg: RunConfig) -> str:
    """Строка `[profile]` как в run_style.py: что именно выбрано и чем это стоит."""
    gen, up = cfg.generation, cfg.upscale
    lora = gen.lora.filename if gen.lora else "—"
    transformer = f"{gen.transformer[0]}" if gen.transformer else gen.repo
    line = (f"[profile] {model_label(cfg)}: repo={gen.repo}, transformer={transformer}, lora={lora}, "
            f"steps={cfg.steps}, text_encoder={cfg.encoder.name}, precision={gen.precision}")
    if up.factor or up.target_height:
        factor = f"x{float(up.factor):g}" if up.factor else f"height={up.target_height}px"
        line += f", upscale={up.backend} {factor}"
    return line


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #

def _check_files(args: argparse.Namespace) -> str | None:
    """Понятные ошибки по путям до тяжёлых импортов (exit 2 без трейсбека).

    `--input` больше не обязателен безусловно: нужен ли референс, знает только бекенд
    (`Capabilities.requires_image`), и решает это `pipeline.input_problem()` — уже после того,
    как собран `RunConfig`. Здесь остаются только «путь указан, а файла нет», иначе
    T2I-модели не смогли бы работать без `--input`.
    """
    raw = getattr(args, "inputs", None) or []
    for path in ([raw] if isinstance(raw, str) else raw):
        if path and not Path(path).is_file():
            return f"нет входного кадра {path} (--input)"
    prompt_file = getattr(args, "prompt_file", None)
    if prompt_file and not Path(prompt_file).is_file():
        return f"нет файла промта {prompt_file} (--prompt-file)"
    return None


def run(args: argparse.Namespace) -> int:
    problem = _check_files(args)
    if problem:
        print(f"[run] {problem}", file=sys.stderr, flush=True)
        return 2
    try:
        cfg = build_config(args)
    except (ValueError, OSError) as exc:              # ConfigError — подкласс ValueError
        print(f"[run] {exc}", file=sys.stderr, flush=True)
        return 2

    from .. import pipeline                            # локально: `run --help` без каркаса

    # Финальный вердикт по референсу даёт `pipeline.input_problem` (внутри `_plan`):
    # "модель X требует референс" / "без --input нужен --aspect" — это конфиг, а не файлы.
    try:
        return _execute(args, cfg, pipeline)
    except ConfigError as exc:
        print(f"[run] {exc}", file=sys.stderr, flush=True)
        return 2


def _execute(args: argparse.Namespace, cfg: RunConfig, pipeline: Any) -> int:
    """`--dry-run` или реальный прогон; `ConfigError` ловит `run()` (один текст, exit 2)."""
    if getattr(args, "dry_run", False):
        info = pipeline.dry_run(cfg)
        info["dry_run"] = True
        print_plan(cfg, info)
        return 0

    from ..runlog import RunLog

    # Tee включается до первой печати: `[profile]`, `[size]` и `[in]` тоже входят в лог.
    with RunLog(cfg.logs) as log:
        print(_profile_line(cfg), flush=True)
        pipeline.run(cfg, log=log)
    return 0


# --------------------------------------------------------------------------- #
# CLI-контракт подкоманды
# --------------------------------------------------------------------------- #

def add_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_argument_group("что и чем генерировать")
    source.add_argument("--input", dest="inputs", action="append", default=None, metavar="PATH",
                        help="входной кадр (референс: он же conditioning и ключ кэша эмбеддингов); "
                             "можно повторить для нескольких референсов (--input a --input b; "
                             "сколько принимает модель — см. `models list`/`Capabilities.max_images`); "
                             "не обязателен для моделей с Capabilities.requires_image=False — "
                             "там без него работает T2I, а пропорцию кадра задаёт --aspect")
    source.add_argument("--out", default=None, metavar="PATH",
                        help=f"куда сохранить кадр (по умолчанию {DEFAULT_OUT}); к имени добавится "
                             f"-s{{seed}} и -x{{factor}}")
    source.add_argument("--prompt-file", default=None, metavar="PATH",
                        help="взять промт из файла (promts/*.txt) вместо встроенного в конфиг")
    source.add_argument("--config", default=None, metavar="PRESET",
                        help=f"пресет ({', '.join(available_presets())}) или путь к YAML; "
                             f"по умолчанию {DEFAULT_CONFIG} (с --preview — preview)")
    source.add_argument("--model", default=None, choices=("base", "turbo"), metavar="PROFILE",
                        help="профиль генерации: base (по умолчанию) = R80/25 шагов; turbo = "
                             "Turbo-трансформер (8 шагов, тот же text-encoder и VAE) — только явно")
    source.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                        help="точечный параметр RunConfig, например generation.steps=8 (повторяемый)")
    source.add_argument("--preview", action="store_true",
                        help="быстрый профиль: 640p и 8 шагов")
    source.add_argument("--steps", type=int, default=None, metavar="N",
                        help="шаги сэмплера; по умолчанию turbo 8 (расписание), base 8 в --preview / 25")
    source.add_argument("--size", type=parse_size, default=None, metavar="NNNp",
                        help="высота кадра в пикселях: 480p / 768p / 864p / 960p / 1088p; "
                             "переопределяет --resolution (оно считается по пропорциям входа)")
    source.add_argument("--resolution", type=int, default=None, metavar="N",
                        help="sqrt(площадь crop): 640=480p, 1024=768p, 1280=960p, 1440=1080p; "
                             "960p+ только с --offload group")
    source.add_argument("--aspect", choices=tuple(ASPECT_RATIOS), default=None, metavar="R",
                        help="пропорция кадра: 16:9 | 4:3 | 1:1 (по умолчанию — как у входного "
                             "изображения); при отличии от входа референс подгоняется scale-to-cover")
    source.add_argument("--seed", type=int, default=None, metavar="N",
                        help=f"seed; по умолчанию из пресета ({DEFAULT_SEED}); смена не трогает кэш")
    source.add_argument("--lora-scale", type=float, default=None, metavar="F",
                        help=f"сила LoRA; по умолчанию из пресета ({DEFAULT_LORA_SCALE}, у профилей "
                             "без адаптера — не задаётся): 0.7 база, 0.85–1.0 структура, "
                             "0.65–0.70 правка картинки")

    memory = parser.add_argument_group("память и fp8")
    memory.add_argument("--fp8", type=int, default=None, choices=(0, 1), metavar="0|1",
                        help="1 = fp8 layerwise casting трансформера (для 16 ГБ VRAM), 0 = bf16; "
                             "по умолчанию из пресета (fp8)")
    memory.add_argument("--fp8-cast-cache", default=None, metavar="DIR",
                        help="папка с fp8-трансформером (по умолчанию своя у каждого профиля; "
                             "'' = не кэшировать)")
    memory.add_argument("--rebuild-fp8-cache", action="store_true",
                        help="переконвертировать fp8 и перезаписать кэш")
    memory.add_argument("--offload", choices=("none", "group"), default=None,
                        help="group = блоки трансформера стримятся с CPU (нужно для 960p+, 2–4x "
                             "медленнее); по умолчанию из пресета (none)")
    memory.add_argument("--offload-blocks", type=int, default=None, metavar="N",
                        help="сколько блоков трансформера держать на GPU (по умолчанию 1: минимум "
                             "VRAM, больше = быстрее)")
    memory.add_argument("--offload-stream", type=int, default=None, choices=(0, 1), metavar="0|1",
                        help="1 = асинхронная префетч-стрим загрузки блоков (по умолчанию 1)")

    upscale = parser.add_argument_group("постобработка (после выгрузки пайплайна)")
    upscale.add_argument("--upscale", type=float, default=None, metavar="F",
                         help="множитель апскейла: 0 = выкл, 2/3/4 или дробный (1.2 -> 640p в 768p); "
                              "seedvr2 x4 = OOM на 16 ГБ; по умолчанию из пресета (0 = выкл)")
    upscale.add_argument("--upscale-backend", choices=("esrgan", "seedvr2"), default=None,
                         help="esrgan = Real-ESRGAN anime6B (по умолчанию); seedvr2 = SeedVR2-3B "
                              "(отдельное окружение seedvr2/)")
    upscale.add_argument("--upscale-weights", default=None, metavar="PATH",
                         help=f"веса ESRGAN; по умолчанию из пресета ({DEFAULT_WEIGHTS}; у seedvr2 "
                              "путь фиксирован: seedvr2/SeedVR/ckpts/)")
    upscale.add_argument("--upscale-tile", type=int, default=None, metavar="N",
                         help="тайл ESRGAN; по умолчанию из пресета (256; при pad 32 на текущем "
                              "стеке даёт max diff 7 к целому кадру)")
    upscale.add_argument("--upscale-color-fix", choices=("none", "wavelet", "reinhard"),
                         default=None,
                         help="цветокоррекция после seedvr2 (он уводит яркость/насыщенность): "
                              "wavelet — как в оригинале (по умолчанию из пресета)")
    upscale.add_argument("--upscale-color-strength", type=float, default=None, metavar="F",
                         help="0..1: доля коррекции (по умолчанию 0.75 — насыщенность ближе всего "
                              "к исходнику)")
    upscale.add_argument("--upscale-color-sigma", type=float, default=None, metavar="F",
                         help="радиус разделения низких/высоких частот в wavelet-коррекции "
                              "(по умолчанию 12)")

    encoder = parser.add_argument_group("текст-энкодер")
    encoder.add_argument("--text-encoder", choices=("base", "heretic"), default=None,
                         help="base = штатный Qwen3VL-8B (по умолчанию); heretic = "
                              "DreamFast/Qwen3-VL-8B-Heretic-1.3.0 (conditioning как в трейне LoRA, "
                              "другой ключ кэша!)")

    checks = parser.add_argument_group("проверки без генерации")
    checks.add_argument("--dry-run", action="store_true",
                        help="напечатать план (crop, ключ кэша, нужен ли энкодер, шаги, фактор) "
                             "и выйти, не загружая модель")

    logs = parser.add_argument_group("логи")
    logs.add_argument("--no-logs", dest="no_logs", action="store_true",
                      help="не создавать подпапку с отладочным логом (по умолчанию рядом с кадром "
                           "лежит <кадр>/run.log и <кадр>/meta.json)")


__all__ = ["BASE_LORA", "DEFAULT_CONFIG", "DEFAULT_OUT", "HELP", "NAME", "add_args",
           "build_config", "model_label", "parse_size", "pick_preset", "print_plan", "run"]
