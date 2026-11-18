"""Кубики пайплайна (`qwenstyle/cube.py`): по модулю на стадию или стратегию.

Импорт пакета регистрирует все кубики в реестре; порядок импорта фиксирован, чтобы реестр был
одинаковым вне зависимости от того, кто первым потянул пакет.
"""
from __future__ import annotations

from . import decoder_vae  # noqa: F401  — decoder.vae.qwen21
from . import encoder_qwen3vl  # noqa: F401  — encoder.qwen3vl
from . import strategy_kv  # noqa: F401  — strategy.kv_offload
from . import strategy_precision  # noqa: F401  — strategy.precision.fp8
from . import strategy_window_attention  # noqa: F401  — strategy.window_attention

__all__ = ["decoder_vae", "encoder_qwen3vl", "strategy_kv", "strategy_precision",
           "strategy_window_attention"]
