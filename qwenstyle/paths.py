"""Единственные (канонические) пути `--set` и их перевод в поля `RunConfig`.

Правило проекта (без обратной совместимости): пользователь пишет **путь параметра кубика** —
`transformer.qwen_image21.steps`, `encoder.qwen3vl.device`, `decoder.vae.qwen21.tile_px`,
`source.frame.height`, `post.upscale.esrgan.target_height`. Старых написаний
(`generation.steps`, `generation.sampler_kwargs.offload_blocks`, `upscale.tile`) больше нет:
неизвестный ключ отвергается списком допустимых.

Зачем тогда таблица внутри: прогон пока читает `RunConfig` (переходный режим до плана `cutover`),
и канонический путь нужно перевести в поле конфига. Это **не** совместимость, а трансляция
единственного публичного имени во внутреннюю форму; уйдёт вместе с `RunConfig`.
"""
from __future__ import annotations

from typing import Final

__all__ = ["CUBE_TO_CONFIG", "EXTRA_CONFIG_ALIASES", "PATHS", "RUN_KEYS", "RUN_PREFIXES",
           "config_path", "is_run_key", "paths", "validate"]

#: Канонический путь параметра кубика -> путь поля `RunConfig` (переходный рантайм).
CUBE_TO_CONFIG: Final[dict[str, str]] = {
    # -- source.frame ------------------------------------------------------------------ #
    "source.frame.height": "size_px",
    "source.frame.ref_crop_side": "generation.sampler_kwargs.ref_px",
    # -- encoder.qwen3vl --------------------------------------------------------------- #
    "encoder.qwen3vl.name": "encoder.name",
    "encoder.qwen3vl.cache_tag": "encoder.cache_tag",
    "encoder.qwen3vl.repo": "encoder.repo",
    "encoder.qwen3vl.revision": "encoder.revision",
    "encoder.qwen3vl.local_dir": "encoder.local_dir",
    "encoder.qwen3vl.dtype": "encoder.dtype",
    "encoder.qwen3vl.device": "encoder.device",
    "encoder.qwen3vl.stream": "encoder.stream",
    "encoder.qwen3vl.prefetch": "encoder.prefetch",
    "encoder.qwen3vl.pin_memory": "encoder.pin_memory",
    "encoder.qwen3vl.lookahead": "encoder.lookahead",
    # -- strategy.precision.fp8 -------------------------------------------------------- #
    "strategy.precision.fp8.cache_dir": "generation.precision_cache",
    "strategy.precision.fp8.rebuild": "generation.sampler_kwargs.rebuild_fp8_cache",
    # -- strategy.kv_offload ----------------------------------------------------------- #
    "strategy.kv_offload.mode": "generation.kv_cache",
    # -- strategy.window_attention ----------------------------------------------------- #
    "strategy.window_attention.window_attn_px": "generation.sampler_kwargs.window_attn_px",
    "strategy.window_attention.window_attn_margin_px":
        "generation.sampler_kwargs.window_attn_margin_px",
    # -- transformer.qwen_image21 ------------------------------------------------------ #
    "transformer.qwen_image21.repo": "generation.repo",
    "transformer.qwen_image21.revision": "generation.revision",
    "transformer.qwen_image21.steps": "generation.steps",
    "transformer.qwen_image21.guidance": "generation.guidance",
    "transformer.qwen_image21.offload": "generation.offload",
    "transformer.qwen_image21.offload_blocks": "generation.sampler_kwargs.offload_blocks",
    "transformer.qwen_image21.offload_stream": "generation.sampler_kwargs.offload_stream",
    "transformer.qwen_image21.lora_repo": "generation.lora.repo",
    "transformer.qwen_image21.lora_file": "generation.lora.filename",
    "transformer.qwen_image21.lora_scale": "generation.lora.scale",
    "transformer.qwen_image21.lora_strip_alpha": "generation.lora.strip_alpha",
    "transformer.qwen_image21.transformer_repo": "generation.transformer",
    # -- decoder.vae.qwen21 ------------------------------------------------------------ #
    "decoder.vae.qwen21.tile_px": "generation.sampler_kwargs.vae_tile_px",
    "decoder.vae.qwen21.tiling": "cube.decoder.vae.qwen21.tiling",
    # -- post.upscale.esrgan ----------------------------------------------------------- #
    "post.upscale.esrgan.factor": "upscale.factor",
    "post.upscale.esrgan.target_height": "upscale.target_height",
    "post.upscale.esrgan.tile": "upscale.tile",
    "post.upscale.esrgan.pad": "upscale.pad",
    "post.upscale.esrgan.weights": "upscale.weights",
    # -- post.colorfix ----------------------------------------------------------------- #
    "post.colorfix.mode": "upscale.color_fix",
    "post.colorfix.strength": "upscale.color_strength",
    "post.colorfix.sigma": "upscale.color_sigma",
}

#: Поля `RunConfig`, у которых есть канонический путь, но они не покрыты таблицей выше
#: (один параметр кубика = два старых ключа: `--rebuild-fp8-cache` исторически имел два имени).
EXTRA_CONFIG_ALIASES: Final[dict[str, str]] = {
    "generation.sampler_kwargs.rebuild_precision_cache": "strategy.precision.fp8.rebuild",
}

#: Ключи **запуска** (не параметры пайплайна): их `--set` пропускает без перевода.
#: `batch.*` — вложенные поля батча (`--set batch.seeds=1-3`).
RUN_KEYS: Final[tuple[str, ...]] = ("input", "inputs", "out", "prompt", "prompt_file", "seed",
                                    "preview", "logs", "cache_dir", "preview_keep",
                                    "cache_supersede")
#: Префиксы ключей запуска (вложенные секции RunConfig).
RUN_PREFIXES: Final[tuple[str, ...]] = ("batch.",)

#: Все допустимые пути (для ошибок и `--help`).
PATHS: Final[tuple[str, ...]] = tuple(sorted(CUBE_TO_CONFIG))


def paths() -> tuple[str, ...]:
    """Все канонические пути `--set` (отсортированы: так их и печатаем)."""
    return PATHS


def is_run_key(key: str) -> bool:
    """Ключ запуска (`out`, `seed`, `batch.seeds`, …) — не параметр кубика, перевода не нужно."""
    text = str(key).strip()
    return text in RUN_KEYS or text.startswith(RUN_PREFIXES)


def validate(key: str) -> str:
    """Проверить канонический путь: вернуть его же или `ValueError` со списком допустимых.

    Ключи запуска (`run.seeds`, `run.out`, …) проверяет `RunSpec` — здесь только параметры
    пайплайна.
    """
    text = str(key).strip()
    if text in CUBE_TO_CONFIG or is_run_key(text):
        return text
    raise ValueError(f"--set: неизвестный параметр {text!r}; допустимые пути: "
                     f"{', '.join(PATHS)}; ключи запуска: {', '.join(RUN_KEYS)}, batch.*")


def config_path(key: str) -> str:
    """Канонический путь -> путь поля `RunConfig` (переходный рантайм)."""
    return CUBE_TO_CONFIG[validate(key)]
