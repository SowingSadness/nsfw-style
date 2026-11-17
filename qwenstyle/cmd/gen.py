"""`python -m qwenstyle gen` — главная команда: проект промта + N seed'ов в одну сессию.

Один проект — одна сцена; внутри несколько промтов (Qwen/H3), общий набор референсов и свой
каталог результатов:

    prompts/my-project/
        refs/                      # все png|jpg|jpeg|webp, имена любые, порядок = natural sort
            01-face.png
            02-room.jpg
        prompt-qwen.txt            # tag = "qwen"
        prompt-h3.txt              # tag = "h3"
        project.yaml               # опционально: preset + поля RunConfig + prompt/refs/out/seeds
        out/
            202611162054-qwen/     # сессия
                84-25.png          # кадр (seed 84, 25 шагов)
                84-25-gen.png      # чистая генерация — только если апскейл реально работает
                84-25.log          # весь вывод этого кадра (`qwenstyle/runlog.py`)
                84-25-meta.json    # параметры и тайминги кадра

    python -m qwenstyle gen --preset VRAM12 --prompt my-project/prompt-qwen --seeds 84
    python -m qwenstyle gen --project prompts/my-project/project.yaml --count 3
    python -m qwenstyle gen --project prompts/my-project/project.yaml --prompt prompt-h3

`--prompt` — shorthand внутри `prompts/`: `<project>/<stem>` (или `<project>` при единственном
`*.txt`). Проект, референсы и `out/` выводятся из пути до промта, поэтому `--preset` обязателен
(если его нет в `project.yaml`). `--project` читает yaml (пресет + любые параметры `RunConfig`)
и **не** подхватывается автоматически: без него настройки берутся только из флагов.

Сессия — `<out>/<YYYYMMDDHHMM>-<tag>/`; `--out` задаёт каталог сессий (по умолчанию
`prompts/<project>/out`). Файлы кадра — `<seed>-<steps>.png`, `<seed>-<steps>-gen.png`,
`<seed>-<steps>.log`, `<seed>-<steps>-meta.json`.

`--preset` принимает шорткаты (`VRAM8`/`VRAM12`/`VRAM16` = `<family>-release`,
`<family>-preview`, `-release`, `-preview-anime`, `-release-anime`, `-anime` = `-release-anime`,
`VRAM16-4k-fast`) и **точное имя любого** зарегистрированного пресета, включая dev-пресеты.
Регистр не важен: `--preset vram12` = `--preset VRAM12` (а два файла, отличающиеся только
регистром, — ошибка, а не угадывание). Строка с `/` или `.yaml` считается путём к файлу.
`--seeds` (`"84"`, `"32-38"`, `"1,7,42"`) и `--count N` (N случайных сидов) — взаимоисключающие;
ни одного — один случайный сид (печатается). Без картинок: если пресет требует референс —
ошибка, иначе T2I (`noref`), а пропорцию кадра задаёт `--aspect`.

Тяжёлых импортов здесь нет: `torch`/`diffusers` тянут бекенды, `pipeline` — только при запуске.
CLI-часть прежнего `batch` переехала сюда, `pipeline.run_batch`/`io.batch_paths` остались
внутренним API.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ..config import (
    ASPECT_RATIOS,
    ConfigError,
    RunConfig,
    available_presets,
    load_prompt,
    prompt_md5,
    readonly_params,
)
from ..io import batch_paths
from ..profiles import PROFILE_ORDER, PROFILES
from ..runlog import write_meta
from . import parse_sets
from . import _project as project
from ._args import UPSCALE_HELP, parse_upscale

NAME = "gen"
HELP = ("сессия кадров из проекта промта: prompts/<project>/refs + prompt-<model>.txt, "
        "--seeds SPEC | --count N -> <project>/out/<штамп>-<tag>/")
#: Расширения, по которым `--preset` — путь к файлу, а не имя пресета.
PATH_SUFFIXES = (".yaml", ".yml")
#: Диапазон случайных сидов (`--count`): положительные 32-битные, как `--seed`.
SEED_MIN = 1
SEED_MAX = 2 ** 31 - 1


# --------------------------------------------------------------------------- #
# Пресеты: шорткаты и точные имена
# --------------------------------------------------------------------------- #

def public_presets() -> tuple[str, ...]:
    """15 публичных пресетов в порядке профилей (`VRAM8` -> `VRAM12` -> `VRAM16`).

    Источник — манифесты `qwenstyle/profiles.py`, поэтому список не расходится с
    `models download --profile` (dev-пресеты сюда не попадают и не рекламируются).
    """
    seen: list[str] = []
    for profile in PROFILE_ORDER:
        for preset in PROFILES[profile].presets:
            if preset not in seen:
                seen.append(preset)
    return tuple(seen)


def preset_aliases() -> dict[str, str]:
    """Шорткаты: `<family>` -> `<family>-release`, `<family>-anime` -> `-release-anime`."""
    aliases: dict[str, str] = {"VRAM16-4k-fast": "VRAM16-4k-fast",
                               "4K-slide": "VRAM16-4k-slide"}
    for family in PROFILE_ORDER:
        aliases[family] = f"{family}-release"
        for suffix in ("preview", "release", "preview-anime", "release-anime"):
            aliases[f"{family}-{suffix}"] = f"{family}-{suffix}"
        aliases[f"{family}-anime"] = f"{family}-release-anime"
    return aliases


def looks_like_path(raw: str) -> bool:
    """`--preset` с `/`, обратным слэшем или `.yaml` — путь к файлу: текст и регистр не трогаем."""
    text = str(raw or "").strip()
    return "/" in text or "\\" in text or text.lower().endswith(PATH_SUFFIXES)


def casefold_index(names: Iterable[str], what: str) -> dict[str, str]:
    """Имя без учёта регистра -> каноническое имя; различия только регистром -> `ConfigError`.

    Порядок `glob()` (а значит и `available_presets()`) не стабилен между файловыми системами,
    поэтому два имени, отличающиеся только регистром, нельзя разрешать «как получится»: это
    ошибка с перечнем конфликтующих файлов, а не молчаливый выбор.
    """
    index: dict[str, str] = {}
    clashes: dict[str, list[str]] = {}
    for name in names:
        key = name.casefold()
        known = index.get(key)
        if known is None:
            index[key] = name
        elif known != name:
            clashes.setdefault(key, [known]).append(name)
    if clashes:
        detail = "; ".join(f"{key}: {', '.join(vals)}" for key, vals in sorted(clashes.items()))
        raise ConfigError(f"{what}: имена отличаются только регистром — {detail}")
    return index


def resolve_preset(raw: str) -> str:
    """Шорткат или точное имя -> имя пресета; регистр не важен; неизвестное — `ConfigError`.

    Приоритет: точное имя -> точное без учёта регистра -> шорткат (тоже без учёта регистра),
    поэтому dev-пресет (`base`, `turbo`, `h3-image`) важнее одноимённого шортката. Строка с `/`
    или `.yaml` — путь к файлу и возвращается как есть (его проверяет `config.preset_path`).
    """
    name = str(raw or "").strip()
    if looks_like_path(name):
        return name
    registered = available_presets()
    if name in registered:
        return name
    folded = name.casefold()
    canonical = casefold_index(registered, "пресеты qwenstyle/presets/").get(folded)
    if canonical:
        return canonical
    aliases = preset_aliases()
    alias_key = (name if name in aliases
                 else casefold_index(aliases, "шорткаты --preset").get(folded))
    alias = aliases.get(alias_key) if alias_key else None
    if alias and alias in registered:
        return alias
    raise ConfigError(
        f"--preset {raw!r} не найден. Публичные пресеты: {', '.join(public_presets())}. "
        f"Шорткаты: VRAM8|VRAM12|VRAM16 (== <family>-release), "
        f"<family>-preview|-release|-preview-anime|-release-anime|-anime (== -release-anime), "
        f"VRAM16-4k-fast. Принимается и точное имя любого зарегистрированного пресета: "
        f"{', '.join(registered)}"
    )


def preset_hint() -> str:
    """Строка для ошибки «нет пресета»: что вообще можно передать."""
    return (f"укажите --preset, {project.DEFAULT_PROJECT_FILE} с preset: в проекте, "
            f"или --project <yaml>: {', '.join(public_presets())} "
            f"(шорткат VRAM8|VRAM12|VRAM16 = <family>-release)")


def preset_source(args: argparse.Namespace, spec: project.ProjectSpec | None) -> str | Mapping[str, Any]:
    """Пресет: `--preset` -> `project.yaml: preset` -> ошибка. Шорткаты разрешаются здесь."""
    raw = getattr(args, "preset", None)
    source = raw if raw else (spec.preset if spec is not None else None)
    if source is None or (not isinstance(source, Mapping) and not str(source).strip()):
        raise ConfigError(preset_hint())
    if isinstance(source, Mapping):
        return source
    text = str(source).strip()
    return text if looks_like_path(text) else resolve_preset(text)


# --------------------------------------------------------------------------- #
# Сиды
# --------------------------------------------------------------------------- #

def random_seeds(count: int) -> list[int]:
    """`count` разных случайных сидов (`SystemRandom`: не повторяются внутри сессии)."""
    rng = random.SystemRandom()
    seeds: list[int] = []
    while len(seeds) < count:
        seed = rng.randint(SEED_MIN, SEED_MAX)
        if seed not in seeds:
            seeds.append(seed)
    return seeds


def _unique(seeds: list[int], source: str) -> list[int]:
    """Убрать повторы (иначе два одинаковых seed'а писали бы один и тот же файл дважды)."""
    seen: set[int] = set()
    unique: list[int] = []
    for seed in seeds:
        if int(seed) in seen:
            continue
        seen.add(int(seed))
        unique.append(int(seed))
    if len(unique) != len(seeds):
        print(f"[gen] повторяющиеся сиды в {source} убраны: {len(seeds)} -> {len(unique)}",
              flush=True)
    return unique


def _parse_seed_spec(spec: str, source: str) -> list[int]:
    """`--seeds`/`seeds:` из yaml -> список сидов (логика `pipeline.parse_seeds`)."""
    from ..pipeline import parse_seeds             # локально: `gen --help` без оркестратора

    seeds = parse_seeds(str(spec))
    if not seeds:
        raise ConfigError(f"{source} {spec!r}: пустой список сидов")
    return _unique(seeds, source)


def resolve_seeds(spec: str | None, count: int | None,
                  project_spec: project.ProjectSpec | None = None) -> list[int]:
    """CLI `--seeds` XOR `--count`; иначе `seeds/count` из `project.yaml`; иначе 1 случайный.

    CLI главнее yaml: если задан любой из флагов, значения из yaml не участвуют.
    """
    if spec and count is not None:
        raise ConfigError("--seeds и --count взаимоисключающие: оставьте одно "
                          "(--seeds '84' или --count 3)")
    if count is not None:
        if int(count) < 1:
            raise ConfigError(f"--count ждёт N >= 1, получено {count}")
        return random_seeds(int(count))
    if spec:
        return _parse_seed_spec(spec, "--seeds")
    if project_spec is not None:
        if project_spec.count is not None:
            if int(project_spec.count) < 1:
                raise ConfigError(f"count в {project.DEFAULT_PROJECT_FILE} ждёт N >= 1")
            return random_seeds(int(project_spec.count))
        if project_spec.seeds:
            return _parse_seed_spec(project_spec.seeds, f"seeds в {project.DEFAULT_PROJECT_FILE}")
    return random_seeds(1)


def seeds_text(seeds: list[int]) -> str:
    """Компактная запись сидов для лога: `84` / `1, 7, 42`."""
    return ", ".join(str(int(seed)) for seed in seeds)


# --------------------------------------------------------------------------- #
# Конфиг
# --------------------------------------------------------------------------- #

def build_config(args: argparse.Namespace, *, spec: project.ProjectSpec | None,
                 prompt: project.ProjectPrompt, images: list[Path], seeds: list[int],
                 session: Path) -> tuple[RunConfig, dict[str, Any]]:
    """Флаги `gen` -> `RunConfig`: project.yaml -> CLI -> производные сессии.

    Возвращает `(cfg, user_sets)`, где `user_sets` — только явные `--set` (для предупреждения
    о readonly-параметрах модели, которые перекрыл пресет).
    """
    overrides: dict[str, Any] = dict(parse_sets(getattr(args, "sets", None)))
    user_sets = dict(overrides)
    cfg = project.build_run_config(spec, preset_source(args, spec))
    steps = int(cfg.steps)
    first_seed = int(seeds[0])
    _gen_path, up_path = batch_paths(posix(session), first_seed, steps)
    overrides.update({
        "prompt_file": posix(prompt.prompt_file),
        "inputs": tuple(posix(path) for path in images),
        "seed": first_seed,
        "out": up_path,
        "batch": {
            "seeds": ",".join(str(int(seed)) for seed in seeds),
            "out_dir": posix(session),
            "keep_gen": True,
            "resume": False,
        },
    })
    if getattr(args, "aspect", None):
        overrides["aspect"] = str(args.aspect)
    if getattr(args, "upscale", None) is not None:
        value = parse_upscale(args.upscale)          # мусор -> ValueError (exit 2 в `run()`)
        if value is None:
            overrides["upscale"] = None              # пустой блок = апскейл выключен целиком
        else:
            overrides["upscale.factor"] = float(value)
    if getattr(args, "no_logs", False):
        overrides["logs"] = False
    cfg = cfg.merged(**overrides)
    # Параметры объявляет адаптер модели: если `--set` попал в readonly (например `steps` у H3,
    # где расписание PDD задаёт ровно 6 шагов), предупреждаем, а не делаем вид, что применилось.
    for item in readonly_params(cfg, user_sets):
        print(f"[gen] {item.readonly_note()}, значение проигнорировано", flush=True)
    return cfg, user_sets


def backend_caps(cfg: RunConfig) -> Any:
    """`Capabilities` бекенда пресета (реестр импортирует torch лениво — весов не касается)."""
    from ..registry import get_generation

    return getattr(get_generation(cfg.generation.backend), "caps", None)


def apply_image_limit(cfg: RunConfig, images: list[Path], preset: str,
                      caps: Any) -> tuple[RunConfig, list[Path]]:
    """Guard по `Capabilities.max_images`: лишние референсы отбрасываются с одной строкой.

    Повторяет правило `pipeline._plan` (первое место, знающее caps), но здесь — **до** плана,
    чтобы сообщение говорило о пресете и настоящих именах файлов:
    `[gen] картинок 12, пресет VRAM12-release поддерживает 10 — взяты 01-a.png …`.
    """
    limit = getattr(caps, "max_images", None)
    if not limit or len(images) <= int(limit):
        return cfg, images
    keep = int(limit)
    kept, dropped = images[:keep], images[keep:]
    print(f"[gen] картинок {len(images)}, пресет {preset} поддерживает {keep} — "
          f"взяты {kept[0].name}..{kept[-1].name} "
          f"({dropped[0].name}..{dropped[-1].name} отброшены)", flush=True)
    return cfg.merged(inputs=tuple(posix(path) for path in kept)), kept


# --------------------------------------------------------------------------- #
# Печать
# --------------------------------------------------------------------------- #

def posix(path: str | Path) -> str:
    """Путь с прямыми `/` (в логах и `meta.json` Windows-бэкслэши читаются плохо)."""
    return Path(path).as_posix()


def _prompt_text(cfg: RunConfig) -> str:
    """Текст промта (для md5 в шапке и meta); ошибка чтения не роняет сессию."""
    try:
        return load_prompt(cfg)
    except OSError:
        return ""


def print_session(cfg: RunConfig, session: Path, prompt: project.ProjectPrompt,
                  project_path: Path | None, images: list[Path], seeds: list[int]) -> None:
    """Шапка сессии: куда пишем, чем и с какими сидами (без весов и без сети)."""
    gen = cfg.generation
    lora = gen.lora.scale if gen.lora else None
    lora_text = f"{float(lora):g}" if lora is not None else "—"
    source = f"project.yaml: {posix(project_path)}" if project_path else "только CLI"
    print(f"[gen] проект {posix(prompt.project_dir)} ({source})", flush=True)
    print(f"[gen] пресет {cfg.preset or '-'} · backend {gen.backend} · {cfg.steps} шагов · "
          f"lora {lora_text} · энкодер {cfg.encoder.name}", flush=True)
    print(f"[gen] промт {posix(cfg.prompt_file or '')} (md5 {prompt_md5(_prompt_text(cfg))})",
          flush=True)
    if images:
        refs = ", ".join(f"{path.name} → <image{index}>"
                         for path, index in project.placeholder_map(images))
        print(f"[gen] референсы {len(images)}: {refs}", flush=True)
    else:
        print(f"[gen] референсов 0 — T2I (noref), пропорция кадра: "
              f"{cfg.aspect or '1:1 (--aspect не задан)'}", flush=True)
    print(f"[gen] сессия {posix(session)}", flush=True)
    print(f"[gen] сиды {len(seeds)}: {seeds_text(seeds)}", flush=True)


def print_plan(cfg: RunConfig, info: dict[str, Any], session: Path, seeds: list[int],
               steps: int) -> None:
    """`--dry-run`: сетка, ключ кэша, нужен ли энкодер, фактор и имена файлов сессии."""
    lora = (info.get("lora") or {}).get("scale")
    lora_text = f"{float(lora):g}" if lora is not None else "—"
    factor = float(info.get("upscale_factor") or 0)
    print(f"[plan] backend={info.get('backend')} preset={cfg.preset or '-'} "
          f"size_px={info.get('size_px') or '-'} steps={info.get('steps')} lora={lora_text}",
          flush=True)
    inputs = info.get("inputs") or []
    refs = f", {len(inputs)} референсов" if len(inputs) > 1 else (f", {len(inputs)} референс"
                                                                if len(inputs) == 1 else "")
    print(f"[plan] input {info.get('input') or '—'} ({info.get('in_size')}{refs}) -> "
          f"crop {info.get('crop')}", flush=True)
    print(f"[plan] cache key={info.get('embeds_key')} (stage={info.get('embeds_stage')}, "
          f"cached={'yes' if info.get('embeds_cached') else 'no'}) -> encoder needed: "
          f"{'yes' if info.get('need_encoder') else 'no'}", flush=True)
    target = info.get("upscale_target_height")
    print(f"[plan] upscale {info.get('upscale_backend')} factor={factor:g} "
          f"target_height={target if target else '-'} -> out {info.get('out_size')}"
          + (" (пропущен)" if info.get("upscale_skipped") else ""), flush=True)
    frame = f"{posix(session)}/<seed>-{int(steps)}.png"
    files = f"{frame} + <seed>-{int(steps)}.log + <seed>-{int(steps)}-meta.json"
    if factor > 0:
        files += f" + <seed>-{int(steps)}-gen.png (апскейл включён)"
    print(f"[plan] файлы сессии: {files}", flush=True)
    print(f"[plan] сиды {len(seeds)}: {seeds_text(seeds)}", flush=True)
    print(f"[plan] dry-run: модель не загружается, энкод не платится "
          f"(offline={'yes' if info.get('offline') else 'no'})", flush=True)


# --------------------------------------------------------------------------- #
# meta.json кадров
# --------------------------------------------------------------------------- #

def _versions() -> dict[str, str]:
    """Версии пакета и библиотек — из метаданных дистрибутивов (без импорта torch)."""
    from importlib.metadata import version as dist_version

    from .. import __version__

    data: dict[str, str] = {"qwenstyle": __version__}
    for dist in ("torch", "diffusers", "transformers", "peft", "safetensors", "numpy"):
        try:
            data[dist] = dist_version(dist)
        except Exception:                           # noqa: BLE001 — метаданных может не быть
            continue
    return data


def _image_sha(path: Path) -> str:
    """Короткий sha256 файла — тот же, что в ключе кэша эмбеддингов (`io.image_sha`)."""
    from ..io import image_sha

    try:
        return image_sha(path)
    except OSError:
        return ""


def write_session_meta(cfg: RunConfig, *, prompt: project.ProjectPrompt,
                       project_path: Path | None, refs_dir: Path, images: list[Path],
                       seeds: list[int], results: list[Any] | None = None,
                       elapsed: float = 0.0) -> None:
    """`<кадр>-meta.json` для каждого кадра сессии (общая часть + тайминги своего seed'а).

    Кадр, его `.log` и meta лежат рядом; общих `meta.json` в корне сессии больше нет.
    `pipeline.run_batch` уже записал в meta параметры и seed до генерации — здесь запись
    дополняется сессией (проект, промт, референсы, версии) и таймингами.
    """
    session_info: dict[str, Any] = {
        "project": posix(project_path) if project_path else None,
        "prompt_dir": posix(prompt.project_dir),
        "prompt_file": posix(prompt.prompt_file),
        "prompt_stem": prompt.prompt_stem,
        "tag": prompt.tag,
        "prompt_md5": prompt_md5(_prompt_text(cfg)),
        "refs_dir": posix(refs_dir),
        "images": [{"file": path.name, "sha": _image_sha(path),
                    "placeholder": f"<image{index}>"}
                   for path, index in project.placeholder_map(images)],
        "preset": cfg.preset,
        "seeds": [int(seed) for seed in seeds],
        "versions": _versions(),
    }
    frames: list[dict[str, Any]] = []
    for result in results or []:
        frames.append({
            "seed": int(result.seed),
            "file": Path(result.out_path).name if result.out_path else None,
            "gen_size": "%dx%d" % result.gen_size,
            "out_size": "%dx%d" % result.out_size,
            "timings": {key: round(float(value), 2) for key, value in result.timings.items()},
        })
    count = len(frames)
    session_info["frames"] = frames
    session_info["totals"] = {
        "frames": count,
        "total_s": round(float(elapsed), 1),
        "per_frame_s": round(float(elapsed) / count, 1) if count else None,
    }
    for result in results or []:
        if result.out_path is None:
            continue
        write_meta(result.out_path, cfg, extra={
            "seed": int(result.seed),
            "steps": int(cfg.steps),
            "gen_size": "%dx%d" % result.gen_size,
            "out_size": "%dx%d" % result.out_size,
            "timings": {key: round(float(value), 2) for key, value in result.timings.items()},
            "session": session_info,
        })


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #

def run(args: argparse.Namespace) -> int:
    try:
        project_path: Path | None = None
        spec: project.ProjectSpec | None = None
        if getattr(args, "project", None):
            project_path = project.resolve_project(args.project)
            spec = project.load_project(project_path)

        prompt: project.ProjectPrompt | None = None
        if getattr(args, "prompt", None):
            prompt = project.resolve_prompt(args.prompt)
        elif spec is not None and spec.prompt is not None:
            prompt = spec.prompt
        if prompt is None:
            print(f"[gen] укажите --prompt <project>/<prompt> или --project "
                  f"<{project.DEFAULT_PROJECT_FILE}>", file=sys.stderr, flush=True)
            return 2

        refs_dir = (spec.refs_dir if spec is not None and spec.refs_dir is not None
                    else prompt.project_dir / project.DEFAULT_REFS_DIR)
        out_dir = (Path(args.out) if getattr(args, "out", None)
                   else (spec.out_dir if spec is not None and spec.out_dir is not None
                         else prompt.project_dir / project.DEFAULT_OUT_DIR))
        images = project.discover_refs(refs_dir)
        seeds = resolve_seeds(getattr(args, "seeds", None), getattr(args, "count", None), spec)
        session = project.session_dir(out_dir, prompt.tag)
        cfg, _ = build_config(args, spec=spec, prompt=prompt, images=images,
                              seeds=seeds, session=session)
        preset_label = str(cfg.preset or (spec.preset if spec is not None else "-"))
        caps = backend_caps(cfg)
        cfg, images = apply_image_limit(cfg, images, preset_label, caps)
    except (ConfigError, ValueError) as exc:         # ValueError — от --upscale (мусор/пусто)
        print(f"[gen] {exc}", file=sys.stderr, flush=True)
        return 2

    requires_image = bool(getattr(caps, "requires_image", True))
    if not images and requires_image:
        print(f"[gen] в {posix(refs_dir)} нет картинок (png|jpg|jpeg|webp), а пресет "
              f"{preset_label} требует референс — положите файлы в refs/",
              file=sys.stderr, flush=True)
        return 2

    print_session(cfg, session, prompt, project_path, images, seeds)

    from .. import pipeline                          # локально: `gen --help` без каркаса и torch

    if getattr(args, "dry_run", False):
        try:
            info = pipeline.dry_run(cfg)
        except ConfigError as exc:
            print(f"[gen] {exc}", file=sys.stderr, flush=True)
            return 2
        print_plan(cfg, info, session, seeds, int(cfg.steps))
        return 0

    from ..runlog import RunLog

    results: list[Any] = []
    started = time.time()
    try:
        with RunLog(cfg.logs) as log:
            try:
                results = pipeline.run_batch(cfg, log=log)
            except ConfigError as exc:
                print(f"[gen] {exc}", file=sys.stderr, flush=True)
                return 2
            except (OSError, RuntimeError) as exc:       # веса/VRAM/диск: понятная строка + код 2
                print(f"[gen] сессия не завершена: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
                return 2
            elapsed = time.time() - started
            write_session_meta(cfg, prompt=prompt, project_path=project_path,
                               refs_dir=refs_dir, images=images, seeds=seeds,
                               results=results, elapsed=elapsed)
            count = len(results)
            average = (elapsed / count) if count else 0.0
            print(f"[gen] итог: {count} кадр(ов) в {posix(session)} · всего {elapsed:.1f}s · "
                  f"среднее {average:.1f}s/кадр", flush=True)
    except OSError as exc:                           # лог/диск упали вне run_batch
        print(f"[gen] сессия не завершена: {type(exc).__name__}: {exc}", file=sys.stderr,
              flush=True)
        return 2
    return 0


# --------------------------------------------------------------------------- #
# CLI-контракт подкоманды
# --------------------------------------------------------------------------- #

def add_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_argument_group("что и чем генерировать")
    source.add_argument("--preset", default=None, metavar="PRESET",
                        help="VRAM8|VRAM12|VRAM16 (== <family>-release), <family>-preview|"
                             "-release|-preview-anime|-release-anime|-anime, VRAM16-4k-fast, 4K-slide "
                             "или точное имя любого пресета из qwenstyle/presets/; регистр не важен")
    source.add_argument("--prompt", default=None, metavar="NAME",
                        help="промт проекта: <project>/<prompt-stem> (или <project> при "
                             "единственном *.txt); референсы — <project>/refs/, результаты — "
                             "<project>/out/")
    source.add_argument("--project", default=None, metavar="PATH",
                        help=f"project.yaml (файл, каталог или имя проекта в "
                             f"{project.PROMPT_ROOT}/): preset + любые поля RunConfig поверх него; "
                             f"флаги CLI главнее yaml")
    source.add_argument("--out", default=None, metavar="DIR",
                        help="каталог сессий (по умолчанию <project>/out); сессия — "
                             "<YYYYMMDDHHMM>-<tag> внутри него")
    source.add_argument("--aspect", choices=tuple(ASPECT_RATIOS), default=None, metavar="R",
                        help="пропорция кадра (по умолчанию — пропорция первого референса; "
                             "для T2I без флага — 1:1)")
    source.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                        help="точечный параметр RunConfig, например generation.steps=8 "
                             "(можно повторять)")
    source.add_argument("--upscale", type=str, default=None, metavar="F|null",
                        help=UPSCALE_HELP)

    seeds = parser.add_argument_group("сколько кадров")
    seeds.add_argument("--seeds", default=None, metavar="SPEC",
                       help="сиды: '84', '32-38', '1,7,42' (взаимоисключающе с --count)")
    seeds.add_argument("--count", type=int, default=None, metavar="N",
                       help="N случайных сидов (печатаются); по умолчанию 1 случайный сид")

    run_group = parser.add_argument_group("режим")
    run_group.add_argument("--dry-run", action="store_true",
                           help="только план (сетка, ключ кэша, сессия) без загрузки весов")
    run_group.add_argument("--no-logs", dest="no_logs", action="store_true",
                           help="не писать <кадр>.log (кадры и <кадр>-meta.json остаются)")


__all__ = ["HELP", "NAME", "PATH_SUFFIXES", "add_args", "build_config", "casefold_index",
           "looks_like_path", "preset_aliases", "preset_source", "public_presets",
           "resolve_preset", "resolve_seeds", "run", "write_session_meta"]
