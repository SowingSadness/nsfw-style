"""Адаптеры моделей: как собрать модель и подставить в неё тензоры из квантованного кэша.

* `diffusers` — meta-модель класса + `from_pretrained(dtype=fp8)` (Qwen-Image, MiniMax-H3);
* `transformers` — LM (PE 9.41B): fp8-веса в VRAM + де-квант модуля на forward (шаг 5b.3).
"""
from __future__ import annotations

from . import diffusers  # noqa: F401  — регистрирует cast_and_cache/load_cached

__all__ = ["diffusers"]
