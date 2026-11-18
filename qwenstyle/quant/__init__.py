"""Квантование весов: унифицированные хранилище и чтение (план `cube-architecture`, шаг 5b).

Слой разделён на хранилище и адаптеры моделей:

* `store.py` — `QuantSpec` (что кастовать, что оставить точным, размер блока), `ShardWriter`
  (пошардовая запись с потолком буфера), `QuantStore` (сайдкар `precision.json`, проверка годности);
* `reader.py` — `QuantReader` (позиционное чтение + де-квант);
* адаптеры (`diffusers`, `transformers`) живут отдельно: они знают, как собрать модель и куда
  подставить прочитанные тензоры.
"""
from __future__ import annotations

from .reader import QuantReader
from .store import (
    BUFFERS_NAME,
    CACHE_SCHEMA,
    INDEX_NAME,
    MAX_PENDING_BYTES,
    SHARD_NAME,
    SIDECAR_NAME,
    WEIGHTS_NAME,
    QuantSpec,
    QuantStore,
    ShardWriter,
    entries_digest,
    human_bytes,
    read_header,
    resolve_source,
    source_entries,
    source_revision,
    source_shards,
)

__all__ = [
    "BUFFERS_NAME", "CACHE_SCHEMA", "INDEX_NAME", "MAX_PENDING_BYTES", "QuantReader", "QuantSpec",
    "QuantStore", "SHARD_NAME", "SIDECAR_NAME", "ShardWriter", "WEIGHTS_NAME", "entries_digest",
    "human_bytes", "read_header", "resolve_source", "source_entries", "source_revision",
    "source_shards",
]
