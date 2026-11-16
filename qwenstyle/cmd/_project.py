"""Проекты промтов: `prompts/<project>/{refs,prompt-*.txt,project.yaml,out}`.

Схема (см. AGENTS.md, R2/§5):

    prompts/my-project/
      refs/                      # все png|jpg|jpeg|webp, имена любые, порядок = natural sort
      prompt-qwen.txt            # промт Qwen (tag = "qwen")
      prompt-h3.txt              # промт H3 (tag = "h3")
      project.yaml               # опционально: preset + поля RunConfig + prompt/refs/out/seeds
      out/<YYYYMMDDHHMM>-<tag>/  # сессия: <seed>-<steps>.png|.log|-meta.json

Модуль намеренно лёгкий (stdlib + `config` + `yaml` только внутри загрузчика): `gen --help`
и `--dry-run` не должны тянуть torch.

Имя начинается с `_`, поэтому `qwenstyle/cli.py` не регистрирует его как подкоманду
(как `_args.py`).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..config import ConfigError, RunConfig

#: Каталог пользовательских проектов промтов (в `.gitignore`).
PROMPT_ROOT = "prompts"
#: Имя конфига проекта по умолчанию (может быть любым при явном `--project`).
DEFAULT_PROJECT_FILE = "project.yaml"
#: Каталог референсов внутри проекта.
DEFAULT_REFS_DIR = "refs"
#: Каталог сессий внутри проекта.
DEFAULT_OUT_DIR = "out"
#: Расширения референсов (регистронезависимо).
REFS_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
#: Мета-ключи `project.yaml`: они не попадают в `RunConfig`, а разбираются здесь.
PROJECT_META_KEYS = ("preset", "prompt", "refs", "out", "seeds", "count")


@dataclass(frozen=True)
class ProjectPrompt:
    """Разрешённый промт: проект, файл и tag для имени сессии."""

    project_dir: Path
    prompt_file: Path
    prompt_stem: str
    tag: str


@dataclass(frozen=True)
class ProjectSpec:
    """Содержимое `project.yaml` (пути уже разрешены относительно каталога yaml)."""

    path: Path | None = None
    preset: str | Mapping[str, Any] | None = None
    prompt: ProjectPrompt | None = None
    refs_dir: Path | None = None
    out_dir: Path | None = None
    seeds: str | None = None
    count: int | None = None
    config: dict[str, Any] = field(default_factory=dict)
    base_dir: Path | None = None


# --------------------------------------------------------------------------- #
# Пути и имена
# --------------------------------------------------------------------------- #

def _norm(raw: Any) -> str:
    """Путь одной строкой с прямыми `/` (Windows-бэкслэши в логах/сравнениях мешают)."""
    return str(raw or "").strip().replace("\\", "/")


def _root() -> Path:
    """Корень проектов: `prompts/` (константа читается в момент вызова — тесты патчат её)."""
    return Path(PROMPT_ROOT)


def _inside(root: Path, path: Path) -> bool:
    """`path` после `resolve()` остаётся внутри `root` (без `..` и абсолютных выходов)."""
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def _txt_candidates(folder: Path) -> list[str]:
    """Имена `*.txt` проекта — для подсказки в ошибках."""
    if not folder.is_dir():
        return []
    return sorted(path.name for path in folder.glob("*.txt") if path.is_file())


def _natural_key(path: Path) -> tuple:
    """Естественная сортировка: `2-x` раньше `10-x`; файлы без числа — в конце."""
    parts = [part for part in re.split(r"(\d+)", path.stem.lower()) if part]
    has_number = any(part.isdigit() for part in parts)
    chunks = tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)
    return (0 if has_number else 1, chunks, path.name.lower())


def tag_for(stem: str, project_dir: Path) -> str:
    """Tag сессии из имени промта: `prompt-qwen` -> `qwen`, `canyon-iren` -> `canyon-iren`.

    Префикс `prompt-` отбрасывается (он общий для всех промтов проекта), остальное приводится
    к нижнему регистру и безопасным символам: tag попадает в имя каталога сессии, где он должен
    сортироваться вместе со штампом времени.
    """
    name = str(stem or "").strip()
    if name.lower().startswith("prompt-"):
        name = name[len("prompt-"):]
    elif name.lower() == "prompt":
        name = project_dir.name
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._").lower()
    return name or project_dir.name.lower() or "prompt"


def _project_prompt(path: Path) -> ProjectPrompt:
    """`ProjectPrompt` по файлу промта: проект = каталог файла."""
    project_dir = path.parent
    return ProjectPrompt(project_dir=project_dir, prompt_file=path,
                         prompt_stem=path.stem, tag=tag_for(path.stem, project_dir))


# --------------------------------------------------------------------------- #
# `--prompt` (shorthand внутри prompts/)
# --------------------------------------------------------------------------- #

def resolve_prompt(raw: Any) -> ProjectPrompt:
    """`my-project/prompt-qwen` -> проект + файл + tag.

    Формы (все внутри `prompts/`, иначе `ConfigError`):

    * `<project>/<stem>` (или `<project>/<stem>.txt`) — файл `prompts/<project>/<stem>.txt`;
    * `<project>` — если в каталоге ровно один `*.txt`, берётся он; иначе ошибка со списком.
    """
    text = _norm(raw)
    if not text:
        raise ConfigError("--prompt: пустое имя (укажите <project>/<prompt> или <project>)")
    root = _root()
    if text.lower().endswith(".txt"):
        path = root / text
        if not _inside(root, path):
            raise ConfigError(f"промт должен лежать внутри {PROMPT_ROOT}/ (без .. и абсолютных путей)")
        if not path.is_file():
            raise ConfigError(f"нет файла промта {path.as_posix()}")
        return _project_prompt(path)
    if "/" in text:
        path = root / f"{text}.txt"
        if not _inside(root, path):
            raise ConfigError(f"промт должен лежать внутри {PROMPT_ROOT}/ (без .. и абсолютных путей)")
        if path.is_file():
            return _project_prompt(path)
        hints = _txt_candidates(path.parent)
        tail = f"; в проекте есть: {', '.join(hints)}" if hints else ""
        raise ConfigError(f"нет промта {path.as_posix()}{tail}")
    folder = root / text
    if not _inside(root, folder) or not folder.is_dir():
        raise ConfigError(
            f"нет проекта {folder.as_posix()} (ожидался каталог с *.txt и refs/)"
        )
    txts = [path for path in folder.glob("*.txt") if path.is_file()]
    if len(txts) == 1:
        return _project_prompt(txts[0])
    names = ", ".join(sorted(path.name for path in txts)) or "—"
    raise ConfigError(
        f"в {folder.as_posix()} несколько/нет .txt ({names}); укажите <project>/<stem>"
    )


# --------------------------------------------------------------------------- #
# `--project` (project.yaml)
# --------------------------------------------------------------------------- #

def resolve_project(raw: Any) -> Path:
    """Путь к `project.yaml`: файл, каталог с ним или имя проекта в `prompts/`."""
    text = _norm(raw)
    if not text:
        raise ConfigError("--project: пустой путь (укажите project.yaml, каталог или имя проекта)")
    path = Path(text)
    if path.is_file():
        return path
    for candidate in (path / DEFAULT_PROJECT_FILE, _root() / text / DEFAULT_PROJECT_FILE):
        if candidate.is_file():
            return candidate
    if path.is_dir():
        raise ConfigError(f"в {path.as_posix()} нет {DEFAULT_PROJECT_FILE}")
    available = sorted(item.as_posix() for item in _root().glob(f"*/{DEFAULT_PROJECT_FILE}"))
    tail = f"; доступны: {', '.join(available)}" if available else ""
    raise ConfigError(f"нет {DEFAULT_PROJECT_FILE} по пути {text!r}{tail}")


def _yaml_path(raw: Any, base_dir: Path) -> Path | None:
    """Путь из yaml: относительный считается от каталога yaml, абсолютный — как есть."""
    text = _norm(raw)
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else base_dir / path


def _yaml_prompt(raw: Any, base_dir: Path) -> ProjectPrompt:
    """Промт из `project.yaml` (расширение `.txt` можно опустить)."""
    path = _yaml_path(raw, base_dir)
    assert path is not None
    if not path.is_file():
        alternative = path.with_name(f"{path.name}.txt")
        if alternative.is_file():
            path = alternative
        else:
            raise ConfigError(f"{base_dir.as_posix()}: нет файла промта {path.as_posix()}")
    return _project_prompt(path)


def load_project(path: str | Path) -> ProjectSpec:
    """Прочитать `project.yaml`: preset + мета-ключи + поля `RunConfig`."""
    import yaml  # локальный импорт: `gen --help` без yaml-зависимости

    target = Path(path)
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{target.as_posix()}: некорректный YAML ({exc})") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{target.as_posix()}: ожидался словарь параметров, "
                          f"получено {type(raw).__name__}")
    data = dict(raw)
    base_dir = target.parent
    preset = data.pop("preset", None)
    prompt_raw = data.pop("prompt", None)
    refs_raw = data.pop("refs", None)
    out_raw = data.pop("out", None)
    seeds = data.pop("seeds", None)
    count = data.pop("count", None)
    if seeds is not None and count is not None:
        raise ConfigError(f"{target.as_posix()}: seeds и count взаимоисключающие")
    prompt = _yaml_prompt(prompt_raw, base_dir) if prompt_raw else None
    refs_dir = _yaml_path(refs_raw, base_dir) or base_dir / DEFAULT_REFS_DIR
    out_dir = _yaml_path(out_raw, base_dir) or base_dir / DEFAULT_OUT_DIR
    try:
        count_value = None if count is None else int(count)
    except (TypeError, ValueError):
        raise ConfigError(f"{target.as_posix()}: count ждёт целое, получено {count!r}") from None
    return ProjectSpec(path=target, preset=preset, prompt=prompt, refs_dir=refs_dir,
                       out_dir=out_dir, seeds=None if seeds is None else str(seeds),
                       count=count_value, config=data, base_dir=base_dir)


def build_run_config(spec: ProjectSpec | None, preset: str | Mapping[str, Any] | None,
                     overrides: Mapping[str, Any] | None = None) -> RunConfig:
    """`project.yaml` -> `RunConfig`: preset -> ключи yaml -> CLI-переопределения.

    `preset` уже разрешён вызывающим (шорткат `VRAM12` -> `VRAM12-release`); здесь только
    сборка. `None` — ошибка: без пресета запускать нечего.
    """
    data: dict[str, Any] = dict(spec.config) if spec is not None else {}
    source = preset if preset is not None else (spec.preset if spec else None)
    if source is None or (not isinstance(source, Mapping) and not str(source).strip()):
        raise ConfigError("нужен preset: --preset или preset в project.yaml")
    data["preset"] = source
    origin = spec.path.name if spec is not None and spec.path is not None else "project.yaml"
    cfg = RunConfig.from_mapping(data, origin=origin)
    if overrides:
        cfg = cfg.merged(**dict(overrides))
    return cfg


# --------------------------------------------------------------------------- #
# Референсы и сессии
# --------------------------------------------------------------------------- #

def discover_refs(folder: str | Path) -> list[Path]:
    """Все картинки `refs/` в natural-порядке; одинаковый стем -> первый.

    Нерекурсивно: референсы — плоский набор проекта, и порядок в нём = порядок `<imageN>`
    в промте. Дубликат стема (`a.png` + `a.jpg`) не ошибка: берётся первый по natural sort,
    остальные печатаются одной строкой (детерминированно на любой ФС).
    """
    target = Path(folder)
    if not target.is_dir():
        return []
    found = sorted((path for path in target.iterdir()
                    if path.is_file() and path.suffix.lower() in REFS_SUFFIXES),
                   key=_natural_key)
    seen: dict[str, Path] = {}
    result: list[Path] = []
    for path in found:
        key = path.stem.casefold()
        known = seen.get(key)
        if known is not None:
            print(f"[gen] refs: дубликат {path.name} — взят {known.name}", flush=True)
            continue
        seen[key] = path
        result.append(path)
    return result


def placeholder_map(paths: list[Path]) -> list[tuple[Path, int]]:
    """Порядок референсов = порядок плейсхолдеров `<image1>`, `<image2>`, …"""
    return [(path, index) for index, path in enumerate(paths, start=1)]


def session_dir(out_dir: str | Path, tag: str,
                now: time.struct_time | None = None) -> Path:
    """`<out_dir>/<YYYYMMDDHHMM>-<tag>`, при коллизии минуты — `-2`, `-3`, …

    Каталог **не создаётся** (это делает `run_batch`): `--dry-run` не должен оставлять следов.
    `out_dir` — каталог сессий (`--out` или `out:` в `project.yaml`), а не конкретная сессия.
    """
    base = Path(out_dir)
    if base.exists() and not base.is_dir():
        raise ConfigError(f"--out {base.as_posix()}: это файл, а нужен каталог для сессий")
    stamp = time.strftime("%Y%m%d%H%M", now or time.localtime())
    tag_text = str(tag or "prompt")
    candidate = base / f"{stamp}-{tag_text}"
    index = 2
    while candidate.exists():
        candidate = base / f"{stamp}-{tag_text}-{index}"
        index += 1
    return candidate


__all__ = [
    "DEFAULT_OUT_DIR", "DEFAULT_PROJECT_FILE", "DEFAULT_REFS_DIR", "PROJECT_META_KEYS",
    "PROMPT_ROOT", "ProjectPrompt", "ProjectSpec", "REFS_SUFFIXES", "build_run_config",
    "discover_refs", "load_project", "placeholder_map", "resolve_project", "resolve_prompt",
    "session_dir", "tag_for",
]
