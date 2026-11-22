"""Таблица устаревших путей `--set` -> канонические пути кубиков (план `workflow-layout`).

Зачем: `--set generation.steps=8`, `--set encoder.device=cpu`, `--set
generation.sampler_kwargs.offload_blocks=4` встречаются в документации, тестах и привычках. После
разложения `RunConfig` на кубики у каждого параметра есть один канонический путь
(`transformer.qwen_image21.steps`, `encoder.qwen3vl.device`, …), и старые написания должны
продолжать работать — иначе сломается всё, что написано до cut-over.

**Это единственное место в пакете, где встречается слово `sampler_kwargs`** после шага 4 плана:
самих `sampler_kwargs` больше нет, остались их пути как алиасы.
"""
from __future__ import annotations

__all__ = ["LEGACY_PATHS", "SAMPLER_PREFIX", "canonical"]

#: Префикс старых «свободных» параметров модели (`generation.sampler_kwargs.*`).
SAMPLER_PREFIX = "generation.sampler_kwargs."

#: Устаревший путь `--set` -> путь параметра кубика (или `run.*`).
LEGACY_PATHS: dict[str, str] = {
    # -- модель генерации (transformer.qwen_image21) ---------------------------------- #
    "generation.steps": "transformer.qwen_image21.steps",
    "generation.guidance": "transformer.qwen_image21.guidance",
    "generation.offload": "transformer.qwen_image21.offload",
    "generation.repo": "transformer.qwen_image21.repo",
    "generation.revision": "transformer.qwen_image21.revision",
    "generation.lora.scale": "transformer.qwen_image21.lora_scale",
    "generation.lora.repo": "transformer.qwen_image21.lora_repo",
    "generation.lora.filename": "transformer.qwen_image21.lora_file",
    # -- стратегии -------------------------------------------------------------------- #
    "generation.precision_cache": "strategy.precision.fp8.cache_dir",
    "generation.kv_cache": "strategy.kv_offload.mode",
    SAMPLER_PREFIX + "offload_blocks": "transformer.qwen_image21.offload_blocks",
    SAMPLER_PREFIX + "offload_stream": "transformer.qwen_image21.offload_stream",
    SAMPLER_PREFIX + "window_attn_px": "strategy.window_attention.window_attn_px",
    SAMPLER_PREFIX + "window_attn_margin_px": "strategy.window_attention.window_attn_margin_px",
    SAMPLER_PREFIX + "ref_px": "source.frame.ref_crop_side",
    SAMPLER_PREFIX + "vae_tile_px": "decoder.vae.qwen21.tile_px",
    SAMPLER_PREFIX + "rebuild_precision_cache": "strategy.precision.fp8.rebuild",
    SAMPLER_PREFIX + "rebuild_fp8_cache": "strategy.precision.fp8.rebuild",
    # -- энкодер ---------------------------------------------------------------------- #
    "encoder.name": "encoder.qwen3vl.name",
    "encoder.cache_tag": "encoder.qwen3vl.cache_tag",
    "encoder.repo": "encoder.qwen3vl.repo",
    "encoder.revision": "encoder.qwen3vl.revision",
    "encoder.local_dir": "encoder.qwen3vl.local_dir",
    "encoder.dtype": "encoder.qwen3vl.dtype",
    "encoder.device": "encoder.qwen3vl.device",
    "encoder.stream": "encoder.qwen3vl.stream",
    "encoder.prefetch": "encoder.qwen3vl.prefetch",
    "encoder.pin_memory": "encoder.qwen3vl.pin_memory",
    "encoder.lookahead": "encoder.qwen3vl.lookahead",
    # -- постобработка ---------------------------------------------------------------- #
    "upscale.factor": "post.upscale.esrgan.factor",
    "upscale.target_height": "post.upscale.esrgan.target_height",
    "upscale.tile": "post.upscale.esrgan.tile",
    "upscale.pad": "post.upscale.esrgan.pad",
    "upscale.weights": "post.upscale.esrgan.weights",
    "upscale.color_fix": "post.colorfix.mode",
    "upscale.color_strength": "post.colorfix.strength",
    "upscale.color_sigma": "post.colorfix.sigma",
    # -- геометрия кадра -------------------------------------------------------------- #
    "size_px": "source.frame.height",
    "resolution": "source.frame.resolution",
}


def canonical(path: str) -> str:
    """Канонический путь кубика для устаревшего или уже нового написания."""
    text = str(path).strip()
    if text in LEGACY_PATHS:
        return LEGACY_PATHS[text]
    if text.startswith(SAMPLER_PREFIX):
        return LEGACY_PATHS.get(text, text)
    return text
