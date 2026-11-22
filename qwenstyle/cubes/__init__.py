"""Кубики пайплайна (`qwenstyle/cube.py`): по модулю на стадию или стратегию.

Импорт пакета регистрирует все кубики в реестре; порядок импорта фиксирован, чтобы реестр был
одинаковым вне зависимости от того, кто первым потянул пакет.
"""
from __future__ import annotations

from . import decoder_vae  # noqa: F401  — decoder.vae.qwen21
from . import encoder_qwen3vl  # noqa: F401  — encoder.qwen3vl
from . import post_colorfix  # noqa: F401  — post.colorfix
from . import post_esrgan  # noqa: F401  — post.upscale.esrgan
from . import post_seedvr2  # noqa: F401  — post.upscale.seedvr2
from . import postprocess_image  # noqa: F401  — decoder.postprocess.image
from . import rewrite_pe  # noqa: F401  — rewrite.pe_i2i, rewrite.pe_t2i
from . import rewrite_pe_llamacpp  # noqa: F401  — rewrite.pe_llamacpp
from . import source_frame  # noqa: F401  — source.frame
from . import source_txtimage  # noqa: F401  — source.txtFileImage
from . import strategy_kv  # noqa: F401  — strategy.kv_offload
from . import strategy_precision  # noqa: F401  — strategy.precision.fp8
from . import strategy_window_attention  # noqa: F401  — strategy.window_attention
from . import transformer_qwen21  # noqa: F401  — transformer.qwen_image21

__all__ = ["decoder_vae", "encoder_qwen3vl", "post_colorfix", "post_esrgan",
           "post_seedvr2", "postprocess_image", "rewrite_pe", "rewrite_pe_llamacpp",
           "source_frame", "source_txtimage", "strategy_kv", "strategy_precision",
           "strategy_window_attention", "transformer_qwen21"]
