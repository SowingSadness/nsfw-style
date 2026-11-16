"""`python -m qwenstyle gen` — главная команда: папка промта + N seed'ов в одну сессию.

Один вход, один выход, ничего не надо собирать из флагов:

    prompts/<имя>/
        prompt.txt          # ровно это имя: текст промта (FUSAL, см. PROMPT_RULES.md)
        image1.png          # …image10.* — референсы (до `Capabilities.max_images`)
        image2.jpg

    python -m qwenstyle gen --preset VRAM12 --prompt rainforest --seeds 84
    python -m qwenstyle gen --preset VRAM8-preview --prompt _demo --count 3

Сессия — `out/<имя>/gen-YYYYMMDD-HHMM/` (+`-2`, `-3` при коллизии минуты), в её **корне**:

    s{seed}.png             # готовый кадр (после апскейла, если он включён пресетом)
    s{seed}-gen.png         # чистая генерация — только если апскейл реально работает
    run.log                 # весь вывод прогона с метками времени (`qwenstyle/runlog.py`)
    meta.json               # сессия: пресет, промт, картинки с sha, сиды, тайминги, версии

`--preset` принимает шорткаты (`VRAM8`/`VRAM12`/`VRAM16` = `<family>-release`,
`<family>-preview`, `-release`, `-preview-anime`, `-release-anime`, `-anime` = `-release-anime`,
`VRAM16-4k-fast`) и **точное имя любого** зарегистрированного пресета, включая dev-пресеты.
Регистр не важен: `--preset vram12` = `--preset VRAM12` (а два файла, отличающиеся только
регистром, — ошибка, а не угадывание). Строка с `/` или `.yaml` считается путём к файлу и
не трогается.
`--seeds` (`"84"`, `"32-38"`, `"1,7,42"`) и `--count N` (N случайных сидов) — взаимоисключающие;
ни одного — один случайный сид (печатается). Без картинок: если пресет требует референс —
ошибка, иначе T2I (`noref`), а пропорцию кадра задаёт `--aspect`.

Тяжёлых импортов здесь нет: `torch`/`diffusers` тянут бекенды, `pipeline` — только при запуске.
CLI-часть прежнего `batch` переехала сюда, `pipeline.run_batch`/`io.batch_paths` остались
внутренним API (шаг 8 плана публикации).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections.abc import Iterable
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
from ..profiles import PROFILE_ORDER, PROFILES
from ..runlog import META_NAME
from . import parse_sets
from ._args import UPSCALE_HELP, parse_upscale

NAME = "gen"
HELP = ("сессия кадров из папки промта: prompts/<имя>/prompt.txt + image*, "
        "--seeds SPEC | --count N -> out/<имя>/gen-<штамп>/")
#: Файл промта — ровно это имя (план публикации, решение 11).
PROMPT_FILE = "prompt.txt"
#: Каталог пользовательских папок промтов (в `.gitignore`).
PROMPT_ROOT = "prompts"
#: Корень результатов: сессия = `<out-root>/<имя>/gen-<штамп>/`.
OUT_ROOT = "out"
#: Расширения референсов (регистронезависимо).
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
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
    aliases: dict[str, str] = {"VRAM16-4k-fast": "VRAM16-4k-fast"}
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
    """Строка для ошибки «нет --preset»: что вообще можно передать."""
    return (f"укажите --preset: {', '.join(public_presets())} "
            f"(или шорткат VRAM8|VRAM12|VRAM16 = <family>-release)")


# --------------------------------------------------------------------------- #
# Папка промта и референсы
# --------------------------------------------------------------------------- #

def prompt_dir(name: str) -> Path:
    """`prompts/<имя>` — папка промта (пользовательская, в `.gitignore`).

    Имя обязано остаться внутри `prompts/`: `..`, абсолютные пути и всё, что после `resolve()`
    выходит из корня, отклоняется (`ConfigError`), иначе `--prompt ../../etc` читал бы промт
    и писал сессию вне проекта.
    """
    raw = str(name or "").strip()
    root = Path(PROMPT_ROOT)
    inside = False
    if raw and not Path(raw).is_absolute() and ".." not in Path(raw).parts:
        try:
            inside = (root / raw).resolve().is_relative_to(root.resolve())
        except (OSError, ValueError):             # нерезолвимый путь — считаем выходом из prompts/
            inside = False
    if not inside:
        raise ConfigError(f"имя папки промта должно быть внутри {PROMPT_ROOT}/ "
                          f"(без .. и абсолютных путей)")
    return root / raw


def _natural_key(path: Path) -> tuple:
    """Естественная сортировка: `image2` раньше `image10`; файлы без числа — в конце.

    Ключ — кортеж из (флаг «нет числа», куски имени, полное имя), поэтому сравнение не зависит
    от длины имён и не падает на смешанных типах (`image1` vs `image10`, `image` vs `image1`).
    """
    parts = [part for part in re.split(r"(\d+)", path.stem.lower()) if part]
    has_number = any(part.isdigit() for part in parts)
    chunks = tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)
    return (0 if has_number else 1, chunks, path.name.lower())


def discover_images(folder: Path) -> list[Path]:
    """Все `image*.{png,jpg,jpeg,webp}` папки промта в естественном порядке.

    Другие файлы (`prompt.txt`, заметки, `.gitkeep`) игнорируются: референс — это только файл
    с префиксом `image`. Регистр расширения и имени не важен (`IMAGE1.PNG` — тоже референс).
    """
    if not folder.is_dir():
        return []
    found = [
        item for item in folder.iterdir()
        if item.is_file()
        and item.suffix.lower() in IMAGE_SUFFIXES
        and item.stem.lower().startswith("image")
    ]
    return sorted(found, key=_natural_key)


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


def resolve_seeds(spec: str | None, count: int | None) -> list[int]:
    """`--seeds SPEC` XOR `--count N`; ни одного — один случайный сид (печатается)."""
    from ..pipeline import parse_seeds             # локально: `gen --help` без оркестратора

    if spec and count is not None:
        raise ConfigError("--seeds и --count взаимоисключающие: оставьте одно "
                          "(--seeds '84' или --count 3)")
    if count is not None:
        if int(count) < 1:
            raise ConfigError(f"--count ждёт N >= 1, получено {count}")
        return random_seeds(int(count))
    if spec:
        seeds = parse_seeds(str(spec))
        if not seeds:
            raise ConfigError(f"--seeds {spec!r}: пустой список сидов")
        return _unique(seeds, "--seeds")
    return random_seeds(1)


def seeds_text(seeds: list[int]) -> str:
    """Компактная запись сидов для лога: `84` / `1, 7, 42`."""
    return ", ".join(str(int(seed)) for seed in seeds)


# --------------------------------------------------------------------------- #
# Сессия
# --------------------------------------------------------------------------- #

def session_dir(prompt: str, out_root: str = OUT_ROOT, *, now: time.struct_time | None = None) -> Path:
    """`out/<имя>/gen-YYYYMMDD-HHMM`, при коллизии минуты — `-2`, `-3`, …

    Каталог **не создаётся** (это делает `run_batch`): `--dry-run` не должен оставлять следов.
    """
    stamp = time.strftime("%Y%m%d-%H%M", now or time.localtime())
    base = Path(out_root) / str(prompt)
    candidate = base / f"gen-{stamp}"
    index = 2
    while candidate.exists():
        candidate = base / f"gen-{stamp}-{index}"
        index += 1
    return candidate


def posix(path: str | Path) -> str:
    """Путь с прямыми `/` (в логах и `meta.json` Windows-бэкслэши читаются плохо)."""
    return Path(path).as_posix()


# --------------------------------------------------------------------------- #
# Конфиг
# --------------------------------------------------------------------------- #

def build_config(args: argparse.Namespace, *, preset: str, prompt_file: Path,
                 images: list[Path], seeds: list[int], session: Path) -> RunConfig:
    """Флаги `gen` -> `RunConfig`: пресет + сессия (кадры, seeds, лог, промт, референсы).

    Батч-секция задаётся целиком: пресеты её не содержат, а `run_batch` требует `seeds`/`out_dir`.
    `resume=False` — внутри одной сессии нечего пропускать, а старое имя не должно молча
    оставлять кадр от прошлого прогона.
    """
    overrides: dict[str, Any] = dict(parse_sets(getattr(args, "sets", None)))
    user_sets = dict(overrides)                      # только то, что задал пользователь
    first_seed = int(seeds[0])
    overrides.update({
        "prompt_file": posix(prompt_file),
        "inputs": tuple(posix(path) for path in images),
        "seed": first_seed,
        # `out` нужен только для плана (`_plan`/`--dry-run`): кадры пишет `run_batch` по batch_paths
        "out": posix(Path(session) / f"s{first_seed}.png"),
        "log_dir": posix(session),
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
    cfg = RunConfig.from_profile(preset, **overrides)
    # Параметры объявляет адаптер модели: если `--set` попал в readonly (например `steps` у H3,
    # где расписание PDD задаёт ровно 6 шагов), предупреждаем, а не делаем вид, что применилось.
    for spec in readonly_params(cfg, user_sets):
        print(f"[gen] {spec.readonly_note()}, значение проигнорировано", flush=True)
    return cfg


def backend_caps(cfg: RunConfig) -> Any:
    """`Capabilities` бекенда пресета (реестр импортирует torch лениво — весов не касается)."""
    from ..registry import get_generation

    return getattr(get_generation(cfg.generation.backend), "caps", None)


def apply_image_limit(cfg: RunConfig, images: list[Path], preset: str,
                      caps: Any) -> tuple[RunConfig, list[Path]]:
    """Guard по `Capabilities.max_images`: лишние референсы отбрасываются с одной строкой.

    Повторяет правило `pipeline._plan` (первое место, знающее caps), но здесь — **до** плана,
    чтобы сообщение говорило о пресете и настоящих именах файлов:
    `[gen] картинок 12, пресет VRAM12-release поддерживает 10 — взяты image1..image10 …`.
    """
    limit = getattr(caps, "max_images", None)
    if not limit or len(images) <= int(limit):
        return cfg, images
    keep = int(limit)
    kept, dropped = images[:keep], images[keep:]
    print(f"[gen] картинок {len(images)}, пресет {preset} поддерживает {keep} — "
          f"взяты {kept[0].stem}..{kept[-1].stem} "
          f"({dropped[0].stem}..{dropped[-1].stem} отброшены)", flush=True)
    return cfg.merged(inputs=tuple(posix(path) for path in kept)), kept


# --------------------------------------------------------------------------- #
# Печать
# --------------------------------------------------------------------------- #

def print_session(cfg: RunConfig, preset: str, session: Path, images: list[Path],
                  seeds: list[int]) -> None:
    """Шапка сессии: куда пишем, чем и с какими сидами (без весов и без сети)."""
    gen = cfg.generation
    lora = gen.lora.scale if gen.lora else None
    lora_text = f"{float(lora):g}" if lora is not None else "—"
    print(f"[gen] сессия {posix(session)}", flush=True)
    print(f"[gen] пресет {preset} · backend {gen.backend} · {cfg.steps} шагов · "
          f"lora {lora_text} · энкодер {cfg.encoder.name}", flush=True)
    print(f"[gen] промт {posix(cfg.prompt_file or '')} (md5 {prompt_md5(_prompt_text(cfg))})",
          flush=True)
    if images:
        print(f"[gen] референсов {len(images)}: "
              f"{', '.join(path.name for path in images)}", flush=True)
    else:
        print(f"[gen] референсов 0 — T2I (noref), пропорция кадра: "
              f"{cfg.aspect or '1:1 (--aspect не задан)'}", flush=True)
    print(f"[gen] сиды {len(seeds)}: {seeds_text(seeds)}", flush=True)


def _prompt_text(cfg: RunConfig) -> str:
    """Текст промта (для md5 в шапке и `meta.json`); ошибка чтения не роняет сессию."""
    try:
        return load_prompt(cfg)
    except OSError:
        return ""


def print_plan(cfg: RunConfig, info: dict[str, Any], session: Path, seeds: list[int]) -> None:
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
    files = f"{posix(session)}/s{{N}}.png"
    if factor > 0:
        files += " + s{N}-gen.png (апскейл включён)"
    print(f"[plan] файлы сессии: {files}", flush=True)
    print(f"[plan] сиды {len(seeds)}: {seeds_text(seeds)}", flush=True)
    print(f"[plan] dry-run: модель не загружается, энкод не платится "
          f"(offline={'yes' if info.get('offline') else 'no'})", flush=True)


# --------------------------------------------------------------------------- #
# meta.json сессии
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


def write_session_meta(directory: Path, cfg: RunConfig, *, preset: str, images: list[Path],
                       seeds: list[int], results: list[Any] | None = None,
                       elapsed: float = 0.0) -> Path | None:
    """`meta.json` в корне сессии: чем сделан кадр + его сиды, картинки, тайминги, версии.

    `runlog._write_meta` кладёт туда же `created`/`argv`/`params` в момент открытия лога —
    здесь запись дополняется данными сессии (`session`) и таймингами, уже известными после
    генерации. Существующий файл не теряется: читается, дополняется, пишется атомарно.
    """
    path = Path(directory) / META_NAME
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded
        except (OSError, ValueError):
            data = {}
    data.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
    data["argv"] = list(sys.argv)
    try:
        data["params"] = cfg.to_mapping()
    except Exception:                               # noqa: BLE001 — метаданные не роняют сессию
        pass

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
    data["session"] = {
        "preset": preset,
        "dir": posix(directory),
        "prompt_dir": posix(Path(cfg.prompt_file or "").parent),
        "prompt_file": cfg.prompt_file,
        "prompt_md5": prompt_md5(_prompt_text(cfg)),
        "images": [{"file": path.name, "sha": _image_sha(path)} for path in images],
        "seeds": [int(seed) for seed in seeds],
        "frames": frames,
        "totals": {
            "frames": count,
            "total_s": round(float(elapsed), 1),
            "per_frame_s": round(float(elapsed) / count, 1) if count else None,
        },
        "versions": _versions(),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    except OSError as exc:
        print(f"[gen] не удалось записать {posix(path)}: {exc}", file=sys.stderr, flush=True)
        return None
    return path


def _image_sha(path: Path) -> str:
    """Короткий sha256 файла — тот же, что в ключе кэша эмбеддингов (`io.image_sha`)."""
    from ..io import image_sha

    try:
        return image_sha(path)
    except OSError:
        return ""


# --------------------------------------------------------------------------- #
# Прогон
# --------------------------------------------------------------------------- #

def run(args: argparse.Namespace) -> int:
    if not str(getattr(args, "preset", "") or "").strip():
        print(f"[gen] {preset_hint()}", file=sys.stderr, flush=True)
        return 2
    try:
        preset = resolve_preset(args.preset)
    except ConfigError as exc:
        print(f"[gen] {exc}", file=sys.stderr, flush=True)
        return 2

    prompt_name = str(getattr(args, "prompt", "") or "").strip()
    if not prompt_name:
        print(f"[gen] укажите --prompt: имя папки в {PROMPT_ROOT}/ "
              f"({PROMPT_ROOT}/<имя>/{PROMPT_FILE} + image*)", file=sys.stderr, flush=True)
        return 2
    try:
        folder = prompt_dir(prompt_name)
    except ConfigError as exc:                       # `..`/абсолютный путь: выход из prompts/
        print(f"[gen] {exc}", file=sys.stderr, flush=True)
        return 2
    prompt_file = folder / PROMPT_FILE
    if not prompt_file.is_file():
        print(f"[gen] нет {posix(prompt_file)} — положите текст промта в этот файл",
              file=sys.stderr, flush=True)
        return 2

    images = discover_images(folder)
    try:
        seeds = resolve_seeds(getattr(args, "seeds", None), getattr(args, "count", None))
    except ConfigError as exc:
        print(f"[gen] {exc}", file=sys.stderr, flush=True)
        return 2

    session = session_dir(prompt_name)
    try:
        cfg = build_config(args, preset=preset, prompt_file=prompt_file, images=images,
                           seeds=seeds, session=session)
        caps = backend_caps(cfg)
        cfg, images = apply_image_limit(cfg, images, preset, caps)
    except (ConfigError, ValueError) as exc:         # ValueError — от --upscale (мусор/пусто)
        print(f"[gen] {exc}", file=sys.stderr, flush=True)
        return 2

    requires_image = bool(getattr(caps, "requires_image", True))
    if not images and requires_image:
        print(f"[gen] в {posix(folder)} нет картинок (image*.png|jpg|jpeg|webp), а пресет "
              f"{preset} требует референс — положите image1.png рядом с {PROMPT_FILE}",
              file=sys.stderr, flush=True)
        return 2

    print_session(cfg, preset, session, images, seeds)

    from .. import pipeline                          # локально: `gen --help` без каркаса и torch

    if getattr(args, "dry_run", False):
        try:
            info = pipeline.dry_run(cfg)
        except ConfigError as exc:
            print(f"[gen] {exc}", file=sys.stderr, flush=True)
            return 2
        print_plan(cfg, info, session, seeds)
        return 0

    from ..runlog import RunLog

    results: list[Any] = []
    started = time.time()
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
        write_session_meta(session, cfg, preset=preset, images=images, seeds=seeds,
                           results=results, elapsed=elapsed)
        count = len(results)
        average = (elapsed / count) if count else 0.0
        print(f"[gen] итог: {count} кадр(ов) в {posix(session)} · всего {elapsed:.1f}s · "
              f"среднее {average:.1f}s/кадр", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# CLI-контракт подкоманды
# --------------------------------------------------------------------------- #

def add_args(parser: argparse.ArgumentParser) -> None:
    source = parser.add_argument_group("что и чем генерировать")
    source.add_argument("--preset", default=None, metavar="PRESET",
                        help="VRAM8|VRAM12|VRAM16 (== <family>-release), <family>-preview|"
                             "-release|-preview-anime|-release-anime|-anime, VRAM16-4k-fast "
                             "или точное имя любого пресета из qwenstyle/presets/; регистр не важен")
    source.add_argument("--prompt", default=None, metavar="NAME",
                        help=f"папка промта {PROMPT_ROOT}/NAME/: {PROMPT_FILE} (текст) + "
                             f"image*.png|jpg|jpeg|webp (референсы, до max_images модели)")
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
                           help="не писать run.log в папку сессии (кадры и meta.json остаются)")


__all__ = ["HELP", "NAME", "PATH_SUFFIXES", "add_args", "build_config", "casefold_index",
           "discover_images", "looks_like_path", "preset_aliases", "public_presets",
           "resolve_preset", "resolve_seeds", "run", "session_dir"]
