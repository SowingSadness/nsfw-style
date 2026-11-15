"""Единый дисковый кэш проекта: `BlobCache` + разбор и политика кэша эмбеддингов.

Контракт (план `qwenstyle-framework`, шаг 3):

    @dataclass(frozen=True)
    class Blob:
        stage: str          # "embeds" | "text_embeds" | "ref_latents" | ...
        key: str
        ext: str = "pt"

    class BlobCache:
        def __init__(self, root: Path | str = "cache")
        def path(self, blob) -> Path
        def get(self, blob) -> dict | None
        def put(self, blob, payload, meta) -> Path
        def list(self) -> list[tuple[Blob, dict | None, int, float]]

Имя файла — `<stage>_<key>.<ext>`, сайдкар — `<stage>_<key>.json`. Ключ эмбеддингов
(`RunConfig.embeds_key`) даёт `<md5>_<sha>[_te-<энкодер>][-<dtype>][-dev-<device>]_<W>x<H>`,
поэтому запись остаётся буквально прежней: `cache/embeds/embeds_<md5>_<sha>_<W>x<H>.pt` (AGENTS.md, R1:
формат менять нельзя). Сегменты dtype/device появляются только для неумолчаний (`bfloat16`/`cpu`).

Инварианты:
  * запись атомарная: `*.tmp` рядом и `os.replace()` — прерванный прогон не оставляет битый кэш;
  * сайдкар `.json` опционален при чтении (старые записи живут без него) и всегда пишется при `put`;
  * битый `.pt` не роняет вызывающего: `get()` возвращает `None` и печатает предупреждение;
  * `torch` импортируется лениво — `cache list` и импорт `qwenstyle` не тянут 2 ГБ библиотек;
  * `scan_embeds`/`plan_prune`/`delete_embeds` — политика обслуживания 1:1 с `tools/cli.py`:
    удаляются только `cache/embeds/embeds_*.pt` (+ их `.json`), `cache/models/fp8_transformer*`,
    `cache/models/weights/*` и чужие файлы не трогаются никогда.

Раскладка кешей (шаг 4 публикации): `cache/embeds/` — вычисления (conditioning промта/картинок),
`cache/models/` — веса и квантование. `BlobCache`/`scan_embeds`/`delete_embeds` работают только
с первым; легаси-корень `cache` (его передаёт `pipeline.blob_cache`) разрешается в `cache/embeds`,
поэтому ключ кэша и существующие записи остаются валидными — миграция это путь, а не математика.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import (
    CACHE_ROOT,
    DEFAULT_ENCODER_DEVICE,
    DEFAULT_ENCODER_DTYPE,
    EMBEDS_DIR,
    MODELS_DIR,
    encoder_device_canon,
    encoder_dtype_canon,
)

#: Корень кеша вычислений (эмбеддинги). Историческое имя: `BlobCache()` по умолчанию пишет сюда.
CACHE_DIR = EMBEDS_DIR


def embeds_root(root: Path | str | None = None) -> Path:
    """Корень кеша вычислений: легаси-корень `cache` -> `cache/embeds` (миграция шага 4).

    `pipeline.blob_cache()` исторически передаёт `cache`; переписывать его ради новой раскладки
    не нужно — записи найдёт тот же вызов. Всё, что уже указывает на `cache/embeds` (или любой
    другой каталог), возвращается как есть.
    """
    path = Path(root) if root is not None else Path(CACHE_DIR)
    if path.name == CACHE_ROOT.name:            # `cache` -> `cache/embeds`, `.../cache` -> `.../cache/embeds`
        return path / EMBEDS_DIR.name
    return path


# Этапы (stage), которые `BlobCache.list()` ищет в корне кэша. Новый этап кэша = добавить сюда.
DEFAULT_STAGES: tuple[str, ...] = ("embeds", "text_embeds", "ref_latents")

# Схема ключа кэша эмбеддингов. Скопирована из tools/cli.py без изменения семантики:
# md5(промт)[:12] + sha256(вход)[:12] (опционально) + [_te-<энкодер>]
# + [-<dtype>] + [-dev-<device>] + WxH + .pt.
# Сегменты dtype/device появляются только для неумолчаний (`bfloat16`/`cpu`), поэтому старые
# имена `embeds_<md5>_<sha>[_te-<enc>]_<W>x<H>.pt` работают как раньше (AGENTS.md R1).
# Записи без `sha` — старая схема (мёртвые, AGENTS.md D4).
#
# Тег энкодера не должен съедать `-dev-<device>`: `[A-Za-z0-9._-]+` жаден и иначе разобрал бы
# `te-heretic-dev-cuda` целиком в `enc`, а `dtype`/`dev` остались бы пустыми.
EMBEDS_RE = re.compile(
    r"^embeds_(?P<md5>[0-9a-f]{12})"
    r"(?:_(?P<sha>[0-9a-f]{12}))?"
    r"(?:_te-(?P<enc>[A-Za-z0-9._]+(?:-(?!dev-)[A-Za-z0-9._]+)*))?"
    r"(?:-(?P<dtype>[a-z0-9]+))?"
    r"(?:-dev-(?P<dev>[a-z0-9]+))?"
    r"_(?P<w>\d+)x(?P<h>\d+)\.pt$"
)


# --------------------------------------------------------------------------- #
# Blob / BlobCache
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Blob:
    """Одна запись кэша: этап, ключ и расширение.

    `key` — часть имени после `<stage>_`; для эмбеддингов это `"<md5>_<sha>_<W>x<H>"`.
    """

    stage: str
    key: str
    ext: str = "pt"

    def __post_init__(self) -> None:
        for field_name, value in (("stage", self.stage), ("key", self.key)):
            if not str(value or "").strip():
                raise ValueError(f"Blob.{field_name} не может быть пустым")
            if any(ch in str(value) for ch in ("/", "\\", ":", "\0")):
                raise ValueError(f"Blob.{field_name} не должен содержать разделителей пути: {value!r}")
        if self.ext.startswith("."):
            object.__setattr__(self, "ext", self.ext[1:])
        if not self.ext:
            raise ValueError("Blob.ext не может быть пустым")

    @property
    def name(self) -> str:
        """Имя файла записи: `<stage>_<key>.<ext>`."""
        return f"{self.stage}_{self.key}.{self.ext}"

    def __str__(self) -> str:                       # для логов
        return self.name


class BlobCache:
    """Stage-кэш в одном каталоге: атомарная запись, опциональный `.json`-сайдкар.

    `root` — каталог кеша вычислений: по умолчанию `cache/embeds`; легаси-корень `cache`
    разрешается в `cache/embeds` (`embeds_root`), поэтому старые вызывающие не меняются.
    """

    def __init__(self, root: Path | str = CACHE_DIR):
        self.root = embeds_root(root)

    # -- пути ------------------------------------------------------------- #

    def path(self, blob: Blob) -> Path:
        return self.root / blob.name

    def sidecar(self, blob: Blob) -> Path:
        """`<stage>_<key>.json` рядом с бинарём (имя без учёта `ext`)."""
        return self.path(blob).with_suffix(".json")

    # -- чтение ----------------------------------------------------------- #

    def get(self, blob: Blob, *, with_meta: bool = False) -> Any:
        """payload из `torch.load(map_location="cpu")` или `None`, если записи нет/она битая.

        `with_meta=True` возвращает пару `(payload, meta)`, где `meta` — содержимое `.json`
        или `None` (сайдкар опционален: старые записи его не имеют).
        """
        payload, meta = self._load(blob)
        return (payload, meta) if with_meta else payload

    def get_meta(self, blob: Blob) -> dict[str, Any] | None:
        """Сайдкар `.json` или `None` (нет файла / не читается — с предупреждением)."""
        return self._read_meta(blob)

    def _load(self, blob: Blob) -> tuple[Any | None, dict[str, Any] | None]:
        path = self.path(blob)
        meta = self._read_meta(blob)
        if not path.is_file():
            return None, meta
        try:
            import torch                                  # ленивый импорт: cache list не грузит torch
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:                          # noqa: BLE001 — битый кэш не должен ронять прогон
            print(f"[cache] не читается {path}: {type(exc).__name__}: {exc}; "
                  f"запись будет пересоздана", file=sys.stderr)
            return None, meta
        return payload, meta

    def _read_meta(self, blob: Blob) -> dict[str, Any] | None:
        side = self.sidecar(blob)
        if not side.is_file():
            return None
        try:
            data = json.loads(side.read_text(encoding="utf-8"))
        except Exception as exc:                          # noqa: BLE001
            print(f"[cache] сайдкар не читается {side}: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            return None
        return data if isinstance(data, dict) else None

    # -- запись ----------------------------------------------------------- #

    def put(self, blob: Blob, payload: Any, meta: Mapping[str, Any] | None = None) -> Path:
        """Атомарно сохранить payload и сайдкар (`meta` + `created`). Возвращает путь `.pt`."""
        path = self.path(blob)
        meta_data: dict[str, Any] = dict(meta or {})
        meta_data.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
        self.root.mkdir(parents=True, exist_ok=True)

        import torch
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(payload, tmp)
            os.replace(tmp, path)                         # атомарная подмена на NTFS/POSIX
        finally:
            tmp.unlink(missing_ok=True)

        side = self.sidecar(blob)
        side_tmp = side.with_name(f"{side.name}.{os.getpid()}.tmp")
        try:
            side_tmp.write_text(json.dumps(meta_data, ensure_ascii=False, indent=2),
                                encoding="utf-8")
            os.replace(side_tmp, side)
        finally:
            side_tmp.unlink(missing_ok=True)
        return path

    def remove(self, blob: Blob) -> list[Path]:
        """Удалить запись и её сайдкар (только внутри root). Возвращает удалённые пути.

        ⚠️ Это **не** путь вытеснения кэша эмбеддингов: ни CLI, ни `cleanup.DeferredCleanup`
        его не вызывают. Единственный путь удаления `embeds_*` — `delete_embeds` (её проверки
        пути/имени и сайдкары), он же стоит в `cleanup.flush` и `cache prune --apply`. Метод
        оставлен как низкоуровневый помощник для точечной инспекции/отладки.
        """
        removed: list[Path] = []
        for p in (self.path(blob), self.sidecar(blob)):
            if p.is_file() and p.parent.resolve() == self.root.resolve():
                p.unlink()
                removed.append(p)
        return removed

    # -- инвентарь -------------------------------------------------------- #

    def list(self, stage: str | None = None) -> list[tuple[Blob, dict[str, Any] | None, int, float]]:
        """Все записи известных этапов: `[(Blob, meta|None, size, mtime)]`.

        Сайдкары `.json`, `*.tmp` и всё, что не похоже на `<stage>_<key>.<ext>`, пропускается;
        подкаталоги (`_stale/`) не читаются вовсе, а `cache/models/` лежит вне `root`.
        """
        if not self.root.is_dir():
            return []
        stages = (stage,) if stage else DEFAULT_STAGES
        found: list[tuple[Blob, dict[str, Any] | None, int, float]] = []
        for candidate in sorted(self.root.iterdir()):
            if not candidate.is_file():
                continue
            prefix = next((f"{st}_" for st in sorted(stages, key=len, reverse=True)
                           if candidate.name.startswith(f"{st}_")), None)
            if prefix is None:
                continue
            stem, dot, ext = candidate.name.rpartition(".")
            if not dot or ext in ("json", "tmp") or not stem.startswith(prefix):
                continue
            key = stem[len(prefix):]
            if not key:
                continue
            try:
                st = candidate.stat()
                blob = Blob(stage=prefix[:-1], key=key, ext=ext)
            except (OSError, ValueError):
                continue
            found.append((blob, self._read_meta(blob), st.st_size, st.st_mtime))
        found.sort(key=lambda item: item[0].name)
        return found


# --------------------------------------------------------------------------- #
# Политика обслуживания кэша эмбеддингов (перенос из tools/cli.py, 1:1)
# --------------------------------------------------------------------------- #

@dataclass
class EmbedsEntry:
    """Разобранная запись `cache/embeds/embeds_*.pt`."""

    path: Path
    size: int
    mtime: float
    md5: str
    sha: str | None
    enc: str
    crop: str
    legacy: bool                       # старая схема ключа: без хеша картинки (AGENTS.md, D4)
    dtype: str = "bfloat16"            # канон `encoder.dtype` (умолчание — bf16, без сегмента)
    device: str = "cpu"                # канон `encoder.device` (умолчание — cpu, без сегмента)

    @property
    def group(self) -> tuple[str, str | None, str, str, str]:
        return (self.md5, self.sha, self.enc, self.dtype, self.device)

    def group_key(self, by: str = "prompt") -> tuple:
        """Группа для политики «один кеш на разрешение».

        by="prompt" (по умолчанию) — разные промты не мешают друг другу;
        by="image"  — на каждое разрешение остаётся только самый свежий кадр
                     (включая записи других промтов той же картинки).

        dtype/device входят в группу всегда: CPU- и GPU-эмбеддинги, как и bf16/fp16, —
        разные величины (GPU ≠ CPU по округлениям), поэтому одна запись не смеет вытеснить
        другую как «дубль» (R1: ключ обязан различать их).
        """
        if by == "image":
            return (self.sha, self.enc, self.dtype, self.device)
        return self.group

    @property
    def age_days(self) -> float:
        return (time.time() - self.mtime) / 86400

    @property
    def resolution(self) -> int:
        w, h = self.crop.split("x")
        return int(round((int(w) * int(h)) ** 0.5))

    @property
    def created(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.mtime))

    @property
    def sidecar(self) -> Path:
        return self.path.with_suffix(".json")


def scan_embeds(root: Path | str = CACHE_DIR) -> list[EmbedsEntry]:
    """Все `cache/embeds/embeds_*.pt`, подходящие под `EMBEDS_RE` (остальное игнорируется).

    Легаси-корень `cache` разрешается в `cache/embeds` (`embeds_root`), а `cache/models`
    не читается никогда: веса/квантование не вытесняются.
    """
    root = embeds_root(root)
    out: list[EmbedsEntry] = []
    if not root.is_dir():
        return out
    for p in sorted(root.glob("embeds_*.pt")):
        m = EMBEDS_RE.match(p.name)
        if not m:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        out.append(EmbedsEntry(path=p, size=st.st_size, mtime=st.st_mtime,
                               md5=m["md5"], sha=m["sha"], enc=m["enc"] or "base",
                               crop=f'{m["w"]}x{m["h"]}', legacy=m["sha"] is None,
                               dtype=encoder_dtype_canon(m["dtype"] or DEFAULT_ENCODER_DTYPE),
                               device=encoder_device_canon(m["dev"] or DEFAULT_ENCODER_DEVICE)))
    return out


def plan_prune(entries: Iterable[EmbedsEntry], *, keep_per_resolution: int = 1,
               group_by: str = "prompt", older_than: float | None = None,
               legacy: bool = False, all: bool = False,
               now: float | None = None) -> list[EmbedsEntry]:
    """План очистки — точная копия политики `tools/cli.py cache prune`.

    `older_than` — возраст в секундах; `all` — всё; иначе на каждое разрешение внутри группы
    остаётся `keep_per_resolution` самых свежих записей. Группа (`EmbedsEntry.group_key`)
    включает dtype/device: CPU- и GPU-варианты одной пары не вытесняют друг друга.
    Legacy-записи (без хеша картинки) в «живой» политике не участвуют и удаляются только
    с `legacy=True`.
    """
    if group_by not in ("prompt", "image"):
        raise ValueError(f"group_by: ожидалось 'prompt' или 'image', получено {group_by!r}")
    entries = list(entries)
    now = time.time() if now is None else now
    if older_than is not None:
        targets = [e for e in entries if (now - e.mtime) > older_than]
    elif all:
        targets = list(entries)
    else:
        keep = max(1, int(keep_per_resolution))
        targets = []
        groups: dict[tuple, list[EmbedsEntry]] = {}
        for e in entries:                   # legacy-записи не участвуют в «живой» политике
            if e.legacy and not legacy:
                continue
            groups.setdefault(e.group_key(group_by), []).append(e)
        for group_entries in groups.values():
            per_crop: dict[str, list[EmbedsEntry]] = {}
            for e in group_entries:
                per_crop.setdefault(e.crop, []).append(e)
            for crop_entries in per_crop.values():
                crop_entries.sort(key=lambda e: e.mtime, reverse=True)
                targets.extend(crop_entries[keep:])
        if legacy:
            targets.extend(e for e in entries if e.legacy)
    unique = {e.path: e for e in targets}
    return sorted(unique.values(), key=lambda e: str(e.path))


def delete_embeds(entries: Iterable[EmbedsEntry], root: Path | str = CACHE_DIR) -> tuple[int, int]:
    """Удалить записи и их `.json`-сайдкары. Возвращает `(сколько, освобождено байт)`.

    Защита: путь обязан лежать прямо в `root` (по умолчанию `cache/embeds`), а имя — быть
    `embeds_*.pt`/`embeds_*.json`. Всё остальное (fp8-кэш, веса, чужие файлы) не удаляется никогда.
    """
    root = embeds_root(root)
    deleted = 0
    freed = 0
    for e in entries:
        path = e.path
        if path.parent.resolve() != root.resolve():
            print(f"[cache] пропущен (не в {root}): {path}", file=sys.stderr)
            continue
        if not EMBEDS_RE.match(path.name):
            print(f"[cache] пропущен (не embeds_*.pt): {path}", file=sys.stderr)
            continue
        side = path.with_suffix(".json")
        for p in (path, side):
            if not p.is_file():
                continue
            if p != path and not re.fullmatch(r"embeds_.+\.json", p.name):
                continue
            freed += p.stat().st_size
            p.unlink()
            print(f"  удалён {p.name}")
        deleted += 1
    return deleted, freed


# --------------------------------------------------------------------------- #
# Форматирование для CLI
# --------------------------------------------------------------------------- #

def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def parse_age(spec: str) -> float:
    """'30d' / '12h' / '90m' -> секунды. ValueError с подсказкой при опечатке."""
    m = re.fullmatch(r"(\d+)([dhm])", str(spec).strip())
    if not m:
        raise ValueError(f"ожидалось вида 30d / 12h / 90m, получено {spec!r}")
    n, unit = int(m.group(1)), m.group(2)
    return n * {"d": 86400, "h": 3600, "m": 60}[unit]


__all__ = [
    "Blob", "BlobCache", "CACHE_DIR", "CACHE_ROOT", "DEFAULT_STAGES", "EMBEDS_DIR",
    "EMBEDS_RE", "MODELS_DIR", "EmbedsEntry", "embeds_root",
    "scan_embeds", "plan_prune", "delete_embeds", "human_bytes", "parse_age",
]
