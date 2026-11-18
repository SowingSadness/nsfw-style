"""Фасад над адаптером diffusers квантованного кэша (план `cube-architecture`, шаг 5b.2).

Раньше модуль совмещал хранение и адаптер diffusers. Теперь:

* хранение — `qwenstyle/quant/store.py` (`QuantSpec`, `ShardWriter`, `QuantStore`);
* чтение — `qwenstyle/quant/reader.py` (`QuantReader`);
* адаптер diffusers — `qwenstyle/quant/adapters/diffusers.py` (`cast_and_cache`, `load_cached`).

Здесь остались только реэкспорты, чтобы существующие вызовы (`backends/qwen_image21.py`,
`backends/minimax_h3_image.py`, тесты) не менялись.
"""
from __future__ import annotations

from .quant.adapters.diffusers import (                       # noqa: F401
    BUFFERS_NAME,
    CACHE_SCHEMA,
    INDEX_NAME,
    MAX_PENDING_BYTES,
    SIDECAR_NAME,
    cast_and_cache,
    load_cached,
)

__all__ = [
    "BUFFERS_NAME",
    "CACHE_SCHEMA",
    "INDEX_NAME",
    "MAX_PENDING_BYTES",
    "SIDECAR_NAME",
    "cast_and_cache",
    "load_cached",
]
