"""`python -m qwenstyle models list|probe|download` — реестр бекендов и работа с репозиториями.

* `models list` — что зарегистрировано (`qwenstyle/backends/*.py`), с ключевыми полями
  `Capabilities`/`ResourceBudget` и секцией ошибок импорта. Пустой реестр — не ошибка.
* `models probe <repo>` — читает `model_index.json` и `*/config.json` **без загрузки весов**:
  сначала из локального `hf-cache`, иначе (если сеть разрешена) через HuggingFace Hub. Печатает
  компоненты с размерами, найденный бекенд и вердикт по бюджету (`hardware_budget_verdict`).
* `models download <repo>` — **по умолчанию только план**: список файлов с размерами, итог в GiB,
  свободное место на диске и требуемый `--yes`; ничего не скачивается. `--dry-run` — тот же план,
  `--yes` — `snapshot_download` с `allow_patterns` из `--include` (без `assets/` и `*.gguf`).

Оффлайн по умолчанию (`HF_HUB_OFFLINE=1`, AGENTS.md R3): сеть — только явным `--offline`-снятием,
т.е. `HF_HUB_OFFLINE=0 python -m qwenstyle models probe <repo>`. Скачивание весов требует
явного `--yes` **и** сети — без них команда печатает только план.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..backends.base import (
    GIB,
    REPO_ROOT,
    ResourceBudget,
    hardware_budget_verdict,
    hardware_resources,
)
from ..registry import KINDS, discover, generation_for_repo

NAME = "models"
HELP = ("зарегистрированные бекенды (list), зондирование репозитория без скачивания (probe) "
        "и план/скачивание весов по явному --yes (download)")

_TRUTHY = {"1", "true", "yes", "on"}
_KIND_TITLES = {
    "generation": "модели генерации",
    "upscale": "апскейл",
    "encoder": "текст-энкодеры",
}
# --------------------------------------------------------------------------- #
# Вспомогательное форматирование
# --------------------------------------------------------------------------- #

def _fmt_size(size: int | None) -> str:
    if size is None:
        return "?"
    for unit, div in (("GiB", GIB), ("MiB", 1024 ** 2), ("KiB", 1024)):
        if size >= div:
            return f"{size / div:.1f} {unit}"
    return f"{size} B"


def _fmt_gib(size: int | None) -> str:
    return "?" if size is None else f"{(size or 0) / GIB:.1f} GiB"


def _table(headers: list[str], rows: list[list[str]], indent: str = "  ") -> None:
    widths = [max([len(str(h))] + [len(str(r[i])) for r in rows]) for i, h in enumerate(headers)]
    for row in [headers] + rows:
        line = "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        print(indent + line)


def _budget_short(budget: ResourceBudget) -> str:
    weights = f"{budget.weights_bf16_gib:.1f}"
    if budget.weights_fp8_gib is not None:
        weights += f"/{budget.weights_fp8_gib:.1f}"
    return f"вес {weights} GiB, enc {budget.encoder_gib:.1f}, диск {budget.disk_gib:.1f}"


# --------------------------------------------------------------------------- #
# `models list`
# --------------------------------------------------------------------------- #

def _list_rows(kind: str, table: dict[str, type]) -> tuple[list[str], list[list[str]]]:
    if kind == "generation":
        headers = ["имя", "conditioning", "image_inputs", "encoder", "precision", "offload",
                   "steps", "lora", "бюджет (bf16/fp8, enc, диск)"]
        rows = []
        for name in sorted(table):
            cls = table[name]
            caps = getattr(cls, "caps", None)
            budget = getattr(cls, "budget", None)
            rows.append([
                name,
                getattr(caps, "conditioning", "-"),
                getattr(caps, "image_inputs", "-"),
                getattr(caps, "encoder", "-"),
                "|".join(getattr(caps, "precision", ()) or ()) or "-",
                "|".join(getattr(caps, "offload", ()) or ()) or "-",
                "-" if getattr(caps, "fixed_steps", None) is None else str(caps.fixed_steps),
                ("yes" if getattr(caps, "supports_lora", False) else "no") if caps else "-",
                _budget_short(budget) if isinstance(budget, ResourceBudget) else "-",
            ])
        return headers, rows
    if kind == "encoder":
        headers = ["имя", "cache_tag", "requires_vision"]
        return headers, [[name, getattr(table[name], "cache_tag", "-"),
                          "yes" if getattr(table[name], "requires_vision", False) else "no"]
                         for name in sorted(table)]
    headers = ["имя"]
    return headers, [[name] for name in sorted(table)]


def _cmd_list(args: argparse.Namespace) -> int:
    registry = discover()
    if getattr(args, "json", False):
        print(json.dumps(registry.to_dict(), ensure_ascii=False, indent=2))
        return 0
    if registry.empty:
        print("зарегистрированных бекендов нет")
        print("  бекенд добавляется модулем qwenstyle/backends/<name>.py "
              "с @register_generation / @register_upscale / @register_encoder")
    else:
        for kind in KINDS:
            table = registry.table(kind)
            print(f"\n{_KIND_TITLES[kind]} ({len(table)}):")
            if not table:
                print("  (нет)")
                continue
            headers, rows = _list_rows(kind, table)
            _table(headers, rows)
    if registry.errors:
        print(f"\nошибки импорта модулей бекендов ({len(registry.errors)}):")
        for modname, reason in registry.errors.items():
            print(f"  {modname}: {reason}")
    return 0


# --------------------------------------------------------------------------- #
# Зондирование репозитория
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RemoteFile:
    """Файл репозитория: путь внутри снапшота + размер (None = неизвестен)."""

    path: str
    size: int | None = None


@dataclass
class Probe:
    """Результат зондирования: откуда взято, компоненты, конфиги, размеры."""

    repo: str
    source: str
    commit: str | None = None
    snapshot: str | None = None
    files: list[RemoteFile] = field(default_factory=list)
    model_index: dict[str, Any] | None = None
    configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    note: str | None = None
    error: str | None = None

    @property
    def total_size(self) -> int:
        return sum(int(f.size or 0) for f in self.files)

    def component_sizes(self) -> dict[str, int]:
        """Сумма размеров по верхнему каталогу компонента; корневые файлы — ключ '(корень)'."""
        sizes: dict[str, int] = {}
        for item in self.files:
            head, sep, _ = item.path.partition("/")
            key = head if sep else "(корень)"
            sizes[key] = sizes.get(key, 0) + int(item.size or 0)
        return sizes

    def components(self) -> list[tuple[str, str, str]]:
        """[(имя, библиотека, класс)] из `model_index.json`: канонический diffusers-порядок."""
        index = self.model_index or {}
        out: list[tuple[str, str, str]] = []
        for name, value in index.items():
            if name.startswith("_"):
                continue
            if isinstance(value, (list, tuple)) and len(value) == 2:
                out.append((str(name), str(value[0]), str(value[1])))
        return out

    def to_dict(self) -> dict[str, Any]:
        sizes = self.component_sizes()
        return {
            "repo": self.repo,
            "source": self.source,
            "commit": self.commit,
            "snapshot": self.snapshot,
            "note": self.note,
            "error": self.error,
            "total_size": self.total_size,
            "pipeline": (self.model_index or {}).get("_class_name"),
            "components": [
                {
                    "name": name,
                    "library": library,
                    "class": cls,
                    "config_class": (self.configs.get(name) or {}).get("_class_name")
                    or ";".join((self.configs.get(name) or {}).get("architectures") or []),
                    "size": sizes.get(name, 0),
                }
                for name, library, cls in self.components()
            ],
        }


def cache_dirs() -> list[Path]:
    """Каталоги hf-cache, которые просматривает probe (HF_HUB_CACHE -> HF_HOME/hub -> repo/hf-cache/hub)."""
    candidates: list[Path] = []
    for raw in (os.environ.get("HF_HUB_CACHE"),):
        if raw:
            candidates.append(Path(raw))
    home = os.environ.get("HF_HOME")
    if home:
        candidates.append(Path(home) / "hub")
    candidates.append(REPO_ROOT / "hf-cache" / "hub")
    unique: list[Path] = []
    for path in candidates:
        if all(path.resolve() != other.resolve() for other in unique):
            unique.append(path)
    return unique


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _probe_from_cache_dir(cache_dir: Path, repo: str) -> Probe | None:
    """Через `huggingface_hub.scan_cache_dir`: точные размеры блобов, симлинки учтены."""
    try:
        from huggingface_hub import scan_cache_dir
    except Exception:                                    # noqa: BLE001 — hub может отсутствовать
        return None
    try:
        info = scan_cache_dir(cache_dir=str(cache_dir))
    except Exception:                                    # noqa: BLE001 — пустой/битый кэш
        return None
    target = repo.strip().lower()
    entry = None
    for item in info.repos:
        if item.repo_type == "model" and item.repo_id.lower() == target:
            entry = item
            break
    if entry is None or not entry.revisions:
        return None
    revisions = sorted(entry.revisions, key=lambda rev: ("main" not in rev.refs, rev.commit_hash))
    rev = revisions[0]
    root = Path(rev.snapshot_path)
    files: list[RemoteFile] = []
    for item in rev.files:
        try:
            rel = Path(item.file_path).relative_to(root).as_posix()
        except ValueError:
            rel = item.file_name
        files.append(RemoteFile(rel, int(item.size_on_disk)))
    return Probe(repo=repo, source=f"hf-cache ({cache_dir})", commit=rev.commit_hash,
                 snapshot=str(root), files=sorted(files, key=lambda f: f.path))


def _probe_from_snapshot_dir(cache_dir: Path, repo: str) -> Probe | None:
    """Фолбэк без huggingface_hub: обход `models--org--name/snapshots/<commit>/` (symlink -> цель)."""
    root_dir = Path(cache_dir) / ("models--" + repo.replace("/", "--"))
    snapshots = root_dir / "snapshots"
    if not snapshots.is_dir():
        return None
    commit = None
    ref = root_dir / "refs" / "main"
    if ref.is_file():
        try:
            commit = ref.read_text(encoding="utf-8").strip() or None
        except OSError:
            commit = None
    dirs = sorted(p for p in snapshots.iterdir() if p.is_dir())
    if not dirs:
        return None
    root = next((p for p in dirs if p.name == commit), dirs[0])
    files: list[RemoteFile] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        try:
            size: int | None = path.stat().st_size          # stat следует по симлинку
        except OSError:
            size = None
        files.append(RemoteFile(path.relative_to(root).as_posix(), size))
    return Probe(repo=repo, source=f"hf-cache ({cache_dir})", commit=root.name,
                 snapshot=str(root), files=files)


def _local_probe(repo: str) -> Probe | None:
    for cache_dir in cache_dirs():
        if not Path(cache_dir).is_dir():
            continue
        found = _probe_from_cache_dir(Path(cache_dir), repo) \
            or _probe_from_snapshot_dir(Path(cache_dir), repo)
        if found is not None:
            return found
    return None


def _remote_json(repo: str, filename: str) -> dict[str, Any] | None:
    """Крошечный JSON с Хаба (без весов и без записи в hf-cache)."""
    try:
        from huggingface_hub import hf_hub_url
        from huggingface_hub.utils import build_hf_headers, get_session

        response = get_session().get(hf_hub_url(repo_id=repo, filename=filename),
                                     headers=build_hf_headers(), timeout=20)
        if response.status_code != 200:
            return None
        data = response.json()
    except Exception:                                    # noqa: BLE001 — сеть не обязательна
        return None
    return data if isinstance(data, dict) else None


def _online_probe(repo: str) -> Probe:
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo, files_metadata=True)
    except Exception as exc:                             # noqa: BLE001
        return Probe(repo=repo, source="нет данных",
                     error=f"Hub недоступен: {type(exc).__name__}: {exc}")
    files = [RemoteFile(str(getattr(s, "rfilename", "")), getattr(s, "size", None))
             for s in (getattr(info, "siblings", None) or [])]
    probe = Probe(repo=repo, source="hub (метаданные + конфиги, веса не скачиваются)",
                  commit=getattr(info, "sha", None), files=files)
    probe.model_index = _remote_json(repo, "model_index.json")
    if probe.model_index is None:
        probe.note = "model_index.json не получен с Хаба (сеть/репозиторий)"
        return probe
    for name, _library, _cls in probe.components():
        config = _remote_json(repo, f"{name}/config.json")
        if config is not None:
            probe.configs[name] = config
    return probe


def _fill_local_configs(probe: Probe) -> None:
    if not probe.snapshot:
        return
    root = Path(probe.snapshot)
    probe.model_index = _read_json(root / "model_index.json")
    for name, _library, _cls in probe.components():
        config = _read_json(root / name / "config.json")
        if config is not None:
            probe.configs[name] = config


def probe_repo(repo: str, offline: bool = True) -> Probe:
    """Локальный кэш -> (если разрешена сеть) Hub. Ничего не скачивает в `hf-cache`."""
    repo = repo.strip().strip("/")
    local = _local_probe(repo)
    if local is not None:
        _fill_local_configs(local)
        return local
    if offline:
        return Probe(repo=repo, source="нет данных",
                     error="нет данных: не в hf-cache, сеть запрещена (HF_HUB_OFFLINE=1)")
    return _online_probe(repo)


def _budget_from_probe(probe: Probe) -> ResourceBudget:
    """Оценка бюджета по размерам файлов репозитория (когда бекенд ещё не зарегистрирован)."""
    sizes = probe.component_sizes()
    denoiser = sizes.get("transformer") or sizes.get("unet") or 0
    encoder = sizes.get("text_encoder") or 0
    disk = round(probe.total_size / GIB, 1)
    if not denoiser:
        return ResourceBudget(encoder_gib=round(encoder / GIB, 1), disk_gib=disk,
                              notes="генератор (transformer/unet) не найден — вес в VRAM не оценён")
    weights_bf16 = denoiser / GIB
    return ResourceBudget(
        weights_bf16_gib=round(weights_bf16, 1),
        weights_fp8_gib=round(weights_bf16 / 2, 1),
        encoder_gib=round(encoder / GIB, 1),
        disk_gib=disk,
        notes="бюджет оценён по размерам файлов репозитория (бекенд не зарегистрирован)",
    )


def _cmd_probe(args: argparse.Namespace) -> int:
    offline = bool(getattr(args, "offline", False)) or _env_offline()
    probe = probe_repo(args.repo, offline=offline)
    matched = generation_for_repo(probe.repo) if not probe.error else None
    budget = matched[1].budget if matched and isinstance(
        getattr(matched[1], "budget", None), ResourceBudget) else None
    if budget is None:
        budget = _budget_from_probe(probe)
    hw = hardware_resources()
    verdict = hardware_budget_verdict(budget, hw)

    if getattr(args, "json", False):
        payload = probe.to_dict()
        payload["offline"] = offline
        payload["backend"] = None if matched is None else {
            "name": matched[0],
            "caps": getattr(matched[1], "caps", None).to_dict()
            if getattr(matched[1], "caps", None) is not None else None,
            "budget": budget.to_dict(),
        }
        payload["hardware"] = hw.to_dict()
        payload["verdict"] = verdict
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if probe.error:
        print(probe.error)
        print("  искали в: " + ", ".join(str(p) for p in cache_dirs()))
        if offline:
            print(f"  метаданные Хаба: HF_HUB_OFFLINE=0 python -m qwenstyle models probe {probe.repo}")
        print(f"железо: {hw.summary()}")
        print("вердикт: нет данных (состав репозитория неизвестен — бюджет не оценить)")
        return 0

    print(f"repo={probe.repo}")
    print(f"источник={probe.source}" + (f", commit={probe.commit[:12]}" if probe.commit else ""))
    if probe.note:
        print(f"примечание={probe.note}")

    if probe.model_index:
        pipeline = probe.model_index.get("_class_name", "-")
        version = probe.model_index.get("_diffusers_version")
        print(f"pipeline={pipeline}" + (f" (diffusers {version})" if version else ""))
        sizes = probe.component_sizes()
        table: dict[str, list[str]] = {}
        for name, library, cls in probe.components():
            config = probe.configs.get(name) or {}
            config_cls = (config.get("_class_name")
                          or ";".join(config.get("architectures") or []) or "-")
            size = sizes.get(name, 0)
            table[name] = [name, library, cls, config_cls,
                           _fmt_size(size) if size else "нет в файлах"]
        print("компоненты:")
        rows: list[list[str]] = []
        for name, _library, _cls in probe.components():
            rows.append(table.get(name, [name, "-", "-", "-", "нет в файлах"]))
        for name in sizes:
            if name not in table:
                rows.append([name, "-", "-", "-", _fmt_size(sizes[name])])
        _table(["компонент", "библиотека", "класс", "config._class_name", "размер"], rows)
        print(f"  итого по репозиторию: {_fmt_size(probe.total_size)}")
    else:
        print("model_index.json: нет данных")
        if probe.snapshot:
            print(f"  снапшот: {probe.snapshot}")

    if matched is not None:
        caps = getattr(matched[1], "caps", None)
        print(f"бекенд={matched[0]} (зарегистрирован)")
        print(f"  caps: {caps.summary() if caps else '-'}")
    else:
        print("бекенд=не зарегистрирован (ни один модуль qwenstyle/backends/*.py не объявил "
              f"repos=('{probe.repo}',))")
    print(f"бюджет: {budget}")
    print(f"железо: {hw.summary()}")
    print(f"вердикт: {verdict}")
    return 0


# --------------------------------------------------------------------------- #
# План и скачивание весов (`models download`)
# --------------------------------------------------------------------------- #

#: Всегда пропускается: картинки/превью репозитория и GGUF-кванты (пайплайну нужны веса diffusers).
_SKIP_SUFFIXES = (".gguf",)
_SKIP_TOP_DIRS = ("assets",)
#: Корневые файлы весов (одиночные копии трансформера/энкодера) — только без `--include`: в них
#: нет смысла, когда берём компоненты по папкам, а весят они десятки GiB.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".msgpack")
#: Сколько строк таблицы файлов печатать максимум (остальное — одной строкой).
_MAX_FILE_ROWS = 200


@dataclass(frozen=True)
class DownloadPlan:
    """Что будет скачано: репозиторий, отбор файлов и откуда взяты размеры.

    Строится и по локальному `hf-cache`, и по метаданным Хаба — тогда `--dry-run` работает без сети.
    """

    repo: str
    source: str
    commit: str | None
    files: list[RemoteFile]
    skipped: list[RemoteFile] = field(default_factory=list)
    include: tuple[str, ...] = ()
    target: str = ""

    @property
    def size(self) -> int:
        return sum(int(f.size or 0) for f in self.files)

    @property
    def skipped_size(self) -> int:
        return sum(int(f.size or 0) for f in self.skipped)

    def component_sizes(self) -> dict[str, int]:
        """Сумма размеров по верхнему каталогу компонента; корневые файлы — ключ '(корень)'."""
        sizes: dict[str, int] = {}
        for item in self.files:
            head, sep, _ = item.path.partition("/")
            key = head if sep else "(корень)"
            sizes[key] = sizes.get(key, 0) + int(item.size or 0)
        return sizes

    def allow_patterns(self) -> list[str]:
        """`snapshot_download(allow_patterns=…)`: глоб по каталогам компонентов + точные корневые файлы."""
        dirs: list[str] = []
        root_files: list[str] = []
        for item in self.files:
            head, sep, _ = item.path.partition("/")
            bucket = dirs if sep else root_files
            value = head if sep else item.path
            if value not in bucket:
                bucket.append(value)
        patterns = [f"{d}/*" for d in dirs] + root_files
        return patterns or ["*"]


def _is_skipped(path: str) -> bool:
    lowered = path.lower()
    if lowered.endswith(_SKIP_SUFFIXES):
        return True
    return lowered.split("/", 1)[0] in _SKIP_TOP_DIRS


def _include_items(raw: Any) -> tuple[str, ...]:
    """`--include transformer,text_encoder` (можно повторять и разделять ';') -> кортеж имён."""
    items: list[str] = []
    for chunk in raw or ():
        for part in str(chunk).replace(";", ",").split(","):
            part = part.strip().strip("/")
            if part and part not in items:
                items.append(part)
    return tuple(items)


def _hf_home() -> Path:
    return Path(os.environ.get("HF_HOME") or (REPO_ROOT / "hf-cache"))


def _download_target(local_dir: str | None) -> Path:
    return Path(local_dir).expanduser() if local_dir else _hf_home() / "hub"


def _disk_space_gib(path: Path) -> tuple[float, float] | None:
    """(свободно, всего) для ближайшего существующего предка `path`."""
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    try:
        usage = shutil.disk_usage(str(current))
    except OSError:
        return None
    return usage.free / GIB, usage.total / GIB


def plan_download(probe: Probe, include: tuple[str, ...] = (),
                  target: Path | None = None) -> DownloadPlan:
    """Отбор файлов: `assets/` и `*.gguf` — всегда мимо; `include` — по верхним каталогам.

    Без `include` берётся всё (включая одиночные веса в корне репозитория). С `include`
    корневые конфиги (`model_index.json` и пр.) остаются, а корневые *веса* пропускаются.
    """
    wanted = [item for item in include if item]
    everything = not wanted or any(item in {"*", "all"} for item in wanted)
    files: list[RemoteFile] = []
    skipped: list[RemoteFile] = []
    for item in probe.files:
        head, sep, _ = item.path.partition("/")
        if _is_skipped(item.path):
            keep = False
        elif not sep:                        # корень: model_index.json нужен всегда
            keep = everything or not item.path.lower().endswith(_WEIGHT_SUFFIXES)
        else:
            keep = everything or head in wanted
        (files if keep else skipped).append(item)
    return DownloadPlan(repo=probe.repo, source=probe.source, commit=probe.commit,
                        files=sorted(files, key=lambda f: f.path), skipped=skipped,
                        include=tuple(wanted),
                        target=str(target or _download_target(None)))


def _print_plan(plan: DownloadPlan, budget: ResourceBudget,
                free: tuple[float, float] | None, yes: bool = False) -> None:
    print("[download] план (веса не скачиваются)")
    print(f"repo={plan.repo}")
    print(f"источник размеров={plan.source}"
          + (f", commit={plan.commit[:12]}" if plan.commit else ""))
    if plan.include:
        print(f"включая: {', '.join(plan.include)}")
    else:
        print("включая: все компоненты (assets/ и *.gguf пропускаются всегда)")
    print(f"цель={plan.target}")
    if not plan.files:
        print("файлов к скачиванию нет (метаданные пусты или --include ничего не выбрал)")
        print("  проверьте состав: python -m qwenstyle models probe " + plan.repo)
        return
    sizes = plan.component_sizes()
    print("компоненты: " + ", ".join(f"{name} {_fmt_gib(size)}"
                                     for name, size in sorted(sizes.items(),
                                                               key=lambda kv: -kv[1])))
    print(f"файлы ({len(plan.files)}):")
    _table(["файл", "размер"],
           [[item.path, _fmt_size(item.size)] for item in plan.files[:_MAX_FILE_ROWS]])
    if len(plan.files) > _MAX_FILE_ROWS:
        print(f"  … ещё {len(plan.files) - _MAX_FILE_ROWS} файлов")
    if plan.skipped:
        print(f"пропущено: {len(plan.skipped)} файлов, {_fmt_gib(plan.skipped_size)} "
              "(assets/, *.gguf, корневые веса без --include или не вошли в --include)")
    print(f"итого к скачиванию: {_fmt_gib(plan.size)} ({len(plan.files)} файлов)")
    if free is not None:
        need = plan.size / GIB
        verdict = "хватает" if need <= free[0] else "НЕ ХВАТАЕТ — освободите место"
        print(f"диск: свободно {free[0]:.1f} из {free[1]:.1f} GiB, нужно {need:.1f} → {verdict}")
    print(f"бюджет: {budget}")
    if yes:
        return
    print("ничего не скачано: нужен явный флаг --yes")
    print(f"  скачать: python -m qwenstyle models download {plan.repo} --yes"
          + (f" --include {','.join(plan.include)}" if plan.include else ""))


def _do_download(plan: DownloadPlan, local_dir: str | None) -> int:
    """`snapshot_download` с allow_patterns плана. Вызывается только при `--yes` и без оффлайна."""
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:                             # noqa: BLE001 — hub может отсутствовать
        print(f"[download] huggingface_hub недоступен: {type(exc).__name__}: {exc}")
        return 2
    patterns = plan.allow_patterns()
    print(f"[download] snapshot_download {plan.repo} -> {plan.target} "
          f"({_fmt_gib(plan.size)}, allow_patterns={len(patterns)})")
    started = time.perf_counter()
    try:
        path = snapshot_download(repo_id=plan.repo, allow_patterns=patterns,
                                 local_dir=local_dir or None)
    except Exception as exc:                             # noqa: BLE001 — сеть/диск/авторизация
        print(f"[download] ошибка: {type(exc).__name__}: {exc}")
        return 2
    print(f"[download] готово за {time.perf_counter() - started:.1f} s: {path}")
    return 0


def _cmd_download(args: argparse.Namespace) -> int:
    repo = str(getattr(args, "repo", "")).strip().strip("/")
    offline = bool(getattr(args, "offline", False)) or _env_offline()
    yes = bool(getattr(args, "yes", False))
    local_dir = getattr(args, "local_dir", None)
    target = _download_target(local_dir)

    probe = probe_repo(repo, offline=offline)
    if probe.error:
        print(f"[download] нет данных о {repo}: {probe.error}")
        print("  искали в: " + ", ".join(str(p) for p in cache_dirs()))
        if offline:
            print("  для списка файлов и размеров нужен HF_HUB_OFFLINE=0:")
            print(f"  HF_HUB_OFFLINE=0 python -m qwenstyle models download {repo} --dry-run")
        if yes:
            print("[download] скачивание не начато: список файлов неизвестен")
            return 2
        return 0

    plan = plan_download(probe, _include_items(getattr(args, "include", None)), target)
    matched = generation_for_repo(probe.repo)
    registered = matched[1].budget if matched and isinstance(
        getattr(matched[1], "budget", None), ResourceBudget) else None
    budget = registered or _budget_from_probe(probe)
    _print_plan(plan, budget, _disk_space_gib(target), yes=yes)

    if not yes:
        return 0
    if offline:
        print("[download] скачивание невозможно при HF_HUB_OFFLINE=1 — снимите его явно:")
        print(f"  HF_HUB_OFFLINE=0 python -m qwenstyle models download {repo} --yes")
        return 2
    if not plan.files:
        print("[download] скачивание не начато: нечего качать")
        return 2
    return _do_download(plan, local_dir)


def _env_offline() -> bool:
    return str(os.environ.get("HF_HUB_OFFLINE", "")).strip().lower() in _TRUTHY


# --------------------------------------------------------------------------- #
# CLI-контракт подкоманды
# --------------------------------------------------------------------------- #

def add_args(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="models_action", metavar="<action>")
    list_cmd = actions.add_parser(
        "list",
        help="зарегистрированные бекенды + ошибки импорта",
        description="Печатает бекенды генерации/апскейла/энкодеров с ключевыми полями "
                    "Capabilities и ResourceBudget; пустой реестр — не ошибка.",
        prog=f"{parser.prog} list",
    )
    list_cmd.add_argument("--json", action="store_true", help="вывести реестр как JSON")
    list_cmd.set_defaults(_models_action="list")

    probe_cmd = actions.add_parser(
        "probe",
        help="что за модель в репозитории и влезает ли она (без скачивания весов)",
        description="Читает model_index.json и */config.json из локального hf-cache (оффлайн), "
                    "иначе с Hub; печатает компоненты, бекенд и вердикт по бюджету.",
        prog=f"{parser.prog} probe",
    )
    probe_cmd.add_argument("repo", metavar="REPO",
                           help="репозиторий модели, например Qwen/Qwen-Image-2.1")
    probe_cmd.add_argument("--offline", action="store_true",
                           help="только локальный hf-cache (по умолчанию так же при HF_HUB_OFFLINE=1)")
    probe_cmd.add_argument("--json", action="store_true", help="вывести результат как JSON")
    probe_cmd.set_defaults(_models_action="probe")

    download_cmd = actions.add_parser(
        "download",
        help="план скачивания весов; сама загрузка — только по явному --yes",
        description="По умолчанию (и с --dry-run) печатает список файлов с размерами, итог в GiB "
                    "и свободное место; ничего не скачивает. С --yes вызывает snapshot_download "
                    "с allow_patterns по --include и требует сети (HF_HUB_OFFLINE=0).",
        prog=f"{parser.prog} download",
    )
    download_cmd.add_argument("repo", metavar="REPO",
                              help="репозиторий модели, например Qwen/Qwen-Image-2.1")
    download_cmd.add_argument("--dry-run", action="store_true",
                              help="только план (то же, что и без флагов)")
    download_cmd.add_argument("--include", action="append", metavar="A,B",
                              help="компоненты через запятую (например transformer,text_encoder,vae); "
                                   "по умолчанию все, всегда без assets/ и *.gguf; с --include корневые "
                                   "конфиги остаются, а одиночные корневые веса не качаются")
    download_cmd.add_argument("--local-dir", metavar="DIR",
                              help="качать в DIR вместо HF_HOME/hub")
    download_cmd.add_argument("--offline", action="store_true",
                              help="размеры только из локального hf-cache "
                                   "(по умолчанию так же при HF_HUB_OFFLINE=1)")
    download_cmd.add_argument("--yes", action="store_true",
                              help="действительно скачать веса (без него — только план)")
    download_cmd.set_defaults(_models_action="download")


def run(args: argparse.Namespace) -> int:
    action = getattr(args, "_models_action", None) or "list"
    if action == "list":
        return _cmd_list(args)
    if action == "probe":
        return _cmd_probe(args)
    if action == "download":
        return _cmd_download(args)
    print(f"[models] неизвестное действие {action!r}; доступны: list, probe, download")
    return 2


__all__ = [
    "NAME", "HELP", "DownloadPlan", "Probe", "RemoteFile", "add_args", "cache_dirs",
    "plan_download", "probe_repo", "run",
]
