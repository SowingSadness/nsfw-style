"""Унифицированное хранилище квантованных весов: `QuantSpec`/`QuantStore` для любой модели.

Зачем (план `cube-architecture`, шаг 5b): `precision.py` складывал в одном файле две разные вещи —
**хранение** (чтение заголовков шардов, дайджест источника, пошардовая запись с потолком буфера,
сайдкар `precision.json`) и **адаптер diffusers** (meta-модель, точные не-fp8 тензоры, инъекция
буферов). Хранение модельно-агностично: тем же слоем будет пользоваться fp8-кэш LM (PE) и
fp8 для энкодеров (долг D9). Здесь — только хранение.

Формат кэша (совместим с сегодняшним `cache/models/fp8_transformer*`):

```
cache_dir/
  diffusion_pytorch_model-00001-of-000NN.safetensors   # шарды, имя как у save_pretrained
  diffusion_pytorch_model.safetensors.index.json       # weight_map: имя тензора -> шард
  precision.json                                       # сайдкар: schema, source digest, dtype, skip
  non_persistent.buffers.pt                            # (опционально) persistent=False буферы
```

`block_bytes` (бывший `MAX_PENDING_BYTES`) — потолок «незаписанного» буфера шарда; он же пик RAM
при касте. Уменьшили параметром — получили больше шардов и меньше памяти.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = [
    "BUFFERS_NAME",
    "CACHE_SCHEMA",
    "INDEX_NAME",
    "MAX_PENDING_BYTES",
    "SHARD_NAME",
    "SIDECAR_NAME",
    "WEIGHTS_NAME",
    "QuantSpec",
    "QuantStore",
    "ShardWriter",
    "entries_digest",
    "human_bytes",
    "read_header",
    "replace_dir",
    "resolve_source",
    "source_entries",
    "source_revision",
    "source_shards",
]

#: Версия формата кэша. Bump = все записи пересобираются один раз.
CACHE_SCHEMA = 1
SIDECAR_NAME = "precision.json"
WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
SHARD_NAME = "diffusion_pytorch_model-{index:05d}-of-{count:05d}.safetensors"
BUFFERS_NAME = "non_persistent.buffers.pt"
#: Потолок незаписанного буфера шарда (он же пик RAM при касте).
MAX_PENDING_BYTES = 2 * 1024 * 1024 * 1024

#: safetensors-имена целевых dtype.
TARGET_TOKENS: dict[str, str] = {
    "fp8_e4m3": "F8_E4M3",
    "fp8_e5m2": "F8_E5M2",
    "bf16": "BF16",
    "f16": "F16",
}


def human_bytes(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


@dataclass(frozen=True)
class QuantSpec:
    """Что именно делает кэш: целевая точность, что оставить точным, размер блока.

    `skip` — regex-паттерны имён модулей, которые остаются в исходной точности (у DiT это
    `pos_embed`/`patch_embed`/`norm`/`proj_in|out`, у LM — `embed_tokens`/`lm_head`/`norm`). Список
    приходит от адаптера модели, потому что зависит от архитектуры.
    """

    target: str = "fp8_e4m3"
    skip: tuple[str, ...] = ()
    block_bytes: int = MAX_PENDING_BYTES
    keep_state: tuple[str, ...] = ()

    @property
    def target_token(self) -> str:
        try:
            return TARGET_TOKENS[self.target]
        except KeyError:
            raise ValueError(f"неизвестная целевая точность {self.target!r}; "
                             f"допустимые: {', '.join(TARGET_TOKENS)}") from None

    def is_skipped(self, name: str) -> bool:
        """Попадает ли тензор под пропуск (остаётся точным)."""
        return any(re.search(pattern, name) for pattern in self.skip)


def replace_dir(source: Path, target: Path) -> None:
    """Атомарно подменить каталог: старый уходит в `.old-<pid>` и удаляется."""
    suffix = f".old-{os.getpid()}"
    stale = target.with_name(target.name + suffix)
    if stale.exists():
        shutil.rmtree(stale, ignore_errors=True)
    if target.exists():
        os.replace(target, stale)
    os.replace(source, target)
    shutil.rmtree(stale, ignore_errors=True)


def resolve_source(source: str | Path) -> Path:
    """Каталог весов: локальный путь или репозиторий Хаба, найденный в локальном кэше."""
    path = Path(source).expanduser()
    if path.is_dir():
        return path
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(str(source), local_files_only=True))
    except Exception as exc:                                     # noqa: BLE001
        raise FileNotFoundError(
            f"{source}: нет локального каталога и не нашлось в кэше Хаба ({exc}); "
            f"скачайте веса: python -m qwenstyle models download {source} --yes") from None


def source_revision(path: Path) -> str | None:
    """Commit снапшота Хаба (из пути `snapshots/<commit>`), если это кэш Хаба."""
    parts = Path(path).parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    meta = Path(path) / ".cache" / "huggingface" / "download" / "config.json.metadata"
    if meta.is_file():
        lines = meta.read_text(encoding="utf-8").splitlines()
        if lines:
            return lines[0].strip()
    return None


def read_header(file: Path) -> dict[str, dict[str, Any]]:
    """Заголовок safetensors без чтения данных: `{имя: {dtype, shape, data_offsets}}`."""
    import struct

    with open(file, "rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    header.pop("__metadata__", None)
    return header


def source_shards(source: Path) -> list[Path]:
    """Шарды весов источника: `index.json` -> перечисленные файлы, иначе одиночный файл."""
    source = Path(source)
    for index_name in (INDEX_NAME, "model.safetensors.index.json"):
        index_path = source / index_name
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            names = sorted(set((index.get("weight_map") or {}).values()))
            shards = [source / name for name in names]
            missing = [str(p) for p in shards if not p.is_file()]
            if missing:
                raise FileNotFoundError(f"{source}: нет шардов из {index_name}: {missing}")
            return shards
    for pattern in (WEIGHTS_NAME, "model.safetensors", "*.safetensors"):
        found = sorted(source.glob(pattern))
        if found:
            return found
    raise FileNotFoundError(f"{source}: не найдено ни {INDEX_NAME}, ни {WEIGHTS_NAME}")


def source_entries(shards: Iterable[Path]) -> list[dict[str, Any]]:
    """Плоский список тензоров источника: имя, dtype, форма, байты, файл."""
    entries: list[dict[str, Any]] = []
    for shard in shards:
        header = read_header(shard)
        for name, info in header.items():
            offsets = info.get("data_offsets") or [0, 0]
            entries.append({"key": name, "dtype": str(info["dtype"]),
                            "shape": list(info.get("shape") or ()),
                            "nbytes": int(offsets[1]) - int(offsets[0]),
                            "shard": Path(shard).name})
    entries.sort(key=lambda item: item["key"])
    return entries


def entries_digest(entries: Iterable[Mapping[str, Any]]) -> str:
    """Дайджест источника: имена, формы, dtype и байты всех тензоров (без чтения данных)."""
    hasher = hashlib.sha256()
    for entry in entries:
        hasher.update(f"{entry['key']}|{entry['dtype']}|{entry['shape']}|{entry['nbytes']}\n"
                      .encode("utf-8"))
    return hasher.hexdigest()


class ShardWriter:
    """Пошардовая запись state dict с потолком незаписанного буфера.

    Использование:

        writer = ShardWriter(cache_dir, block_bytes=spec.block_bytes)
        for name, tensor in items: writer.add(name, tensor)
        index = writer.close()          # -> {"weight_map": {...}, "shards": [...]}

    Имена шардов — как у `save_pretrained` (`diffusion_pytorch_model-00001-of-000NN`), чтобы
    `from_pretrained` читал кэш штатным путём.
    """

    def __init__(self, cache_dir: str | Path, *, block_bytes: int = MAX_PENDING_BYTES) -> None:
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.block_bytes = max(1, int(block_bytes))
        self._pending: dict[str, Any] = {}
        self._pending_bytes = 0
        self._shards: list[dict[str, Any]] = []
        self._weight_map: dict[str, str] = {}

    # -- внутреннее -------------------------------------------------------- #
    def _flush(self) -> None:
        if not self._pending:
            return
        from safetensors.torch import save_file

        index = len(self._shards) + 1
        name = SHARD_NAME.format(index=index, count=99999)     # count уточняется в close()
        path = self.cache / name
        save_file(self._pending, str(path))
        for tensor_name in self._pending:
            self._weight_map[tensor_name] = name
        self._shards.append({"name": name, "keys": len(self._pending),
                             "nbytes": int(path.stat().st_size)})
        self._pending = {}
        self._pending_bytes = 0

    # -- публичное --------------------------------------------------------- #
    def add(self, name: str, tensor: Any) -> None:
        self._pending[name] = tensor
        self._pending_bytes += int(tensor.numel() * tensor.element_size())
        if self._pending_bytes >= self.block_bytes:
            self._flush()

    def close(self) -> dict[str, Any]:
        self._flush()
        total = len(self._shards)
        renamed: list[dict[str, Any]] = []
        for index, shard in enumerate(self._shards, start=1):
            old = self.cache / shard["name"]
            new_name = SHARD_NAME.format(index=index, count=total)
            new = self.cache / new_name
            if old != new:
                os.replace(old, new)
                for key, value in self._weight_map.items():
                    if value == shard["name"]:
                        self._weight_map[key] = new_name
            renamed.append({**shard, "name": new_name})
        self._shards = renamed
        index = {"metadata": {"format": "pt"}, "weight_map": dict(sorted(self._weight_map.items()))}
        (self.cache / INDEX_NAME).write_text(json.dumps(index, indent=1), encoding="utf-8")
        return {"weight_map": index["weight_map"], "shards": self._shards}


class QuantStore:
    """Каталог кэша квантованных весов: сайдкар, проверка годности, чтение таблиц."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()

    # -- сайдкар ----------------------------------------------------------- #
    @property
    def sidecar_path(self) -> Path:
        return self.path / SIDECAR_NAME

    def sidecar(self) -> dict[str, Any]:
        return json.loads(self.sidecar_path.read_text(encoding="utf-8")) if self.sidecar_path.is_file() else {}

    def write_sidecar(self, payload: Mapping[str, Any]) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        tmp = self.sidecar_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.sidecar_path)

    def index(self) -> dict[str, Any]:
        index_path = self.path / INDEX_NAME
        return json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}

    # -- проверка ---------------------------------------------------------- #
    def verify(self, *, digest: str, config_sha: str, spec: QuantSpec) -> str | None:
        """`None` = кэш годен; иначе строка с причиной (что именно изменилось)."""
        if not self.path.is_dir():
            return "каталога кэша нет"
        side = self.sidecar()
        if not side:
            return f"нет {SIDECAR_NAME}"
        if int(side.get("schema", -1)) != CACHE_SCHEMA:
            return f"схема кэша {side.get('schema')} != {CACHE_SCHEMA}"
        if side.get("source_digest") != digest:
            return "дайджест источника весов изменился"
        if side.get("config_sha256") != config_sha:
            return "config.json источника изменился"
        if side.get("fp8_dtype") != spec.target_token:
            return f"точность кэша {side.get('fp8_dtype')} != {spec.target_token}"
        if tuple(side.get("skip_patterns") or ()) != tuple(spec.skip):
            return "набор точных модулей (skip) изменился"
        index = self.index()
        if not index.get("weight_map"):
            return f"нет {INDEX_NAME} или пустой weight_map"
        for name, shard in index["weight_map"].items():
            if not (self.path / shard).is_file():
                return f"нет шарда {shard} (тензор {name})"
        return None


def _field_names(cls: Any) -> tuple[str, ...]:                        # pragma: no cover - хелпер
    return tuple(field.name for field in cls.__dataclass_fields__.values())
