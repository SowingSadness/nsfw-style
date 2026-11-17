"""Скользящее контекстное окно внимания для Qwen-Image-2.1 (Context Windows).

В отличие от тайлового денойзинга модель запускается **один раз на весь
канвас**, латент и позиции глобальные, режется только сам attention — каждая полоса запросов
видит префикс (текст + референсы) и свою окрестность по строкам. Никакого усреднения выходов
окон нет, поэтому «сшить разные картинки» физически не из чего: x0 один.

Геометрия: целевые токены идут растром (`h` внешний, `w` внутренний — так же, как их нумерует
RoPE в `QwenImage21PosEmbed`), поэтому полоса по строкам — непрерывный срез токенов.

Хук: штатный `QwenImage21AttnProcessor` уже режет prefill на сегменты (по вызову на каждый
префикс + один на целевое изображение) и получает `segments`/`key_valid`. Здесь этот последний
вызов заменяется циклом по полосам — ровно то место, которое в diffusers предусмотрено под
контекстные окна.
"""
from __future__ import annotations

from typing import Any

#: Латентная клетка Qwen-Image-2.1: VAE 16x (16 px на клетку, 64 канала).
VAE_SCALE = 16


def _torch() -> Any:
    import torch

    return torch


class SlidingWindowAttnProcessor:
    """Внимание в полосах: запросы полосы видят префикс + окрестность ± margin.

    Полоса (`band_px`) задаёт высоту блока запросов, `margin_px` — сколько строк сверху и снизу
    эта полоса видит дополнительно. При `band_px` не меньше высоты канваса и любом margin
    остаётся ровно один блок = штатное плотное внимание, поэтому такой режим — контроль
    корректности (результат обязан совпасть с нативным бит-в-бит).
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(self, grid: tuple[int, int], band_px: int, margin_px: int) -> None:
        self.height = max(1, int(grid[1]) // VAE_SCALE)
        self.width = max(1, int(grid[0]) // VAE_SCALE)
        self.band = max(1, int(band_px) // VAE_SCALE)
        self.margin = max(0, int(margin_px) // VAE_SCALE)

    # -- внутреннее -------------------------------------------------------- #
    def _keep_index(self, prefix_len: int, row0: int, row1: int) -> Any:
        """Индексы ключей: весь префикс + строки `[row0, row1)` целевого изображения."""
        torch = _torch()
        width = self.width
        prefix = torch.arange(prefix_len)
        target = torch.arange(prefix_len + row0 * width, prefix_len + row1 * width)
        return torch.cat([prefix, target])

    def _windowed(self, query: Any, key: Any, value: Any, prefix_len: int,
                  mask: Any) -> Any:
        """Внимание полосами: `query` — только целевые токены, `key` — префикс + цель."""
        torch = _torch()
        from diffusers.models.attention_dispatch import dispatch_attention_fn

        height, width = self.height, self.width
        whole = prefix_len == 0
        outputs = []
        for row0 in range(0, height, self.band):
            row1 = min(height, row0 + self.band)
            key0 = max(0, row0 - self.margin)
            key1 = min(height, row1 + self.margin)
            query_slice = query[:, row0 * width:row1 * width]
            if whole and key0 == 0 and key1 >= height:
                keys, values, band_mask = key, value, mask
            else:
                index = self._keep_index(prefix_len, key0, key1).to(key.device)
                keys = key.index_select(1, index)
                values = value.index_select(1, index)
                band_mask = None if mask is None else mask.index_select(-1, index)
            outputs.append(
                dispatch_attention_fn(
                    query_slice, keys, values,
                    attn_mask=band_mask,
                    dropout_p=0.0,
                    backend=None,
                    parallel_config=self._parallel_config,
                )
            )
        return torch.cat(outputs, dim=1)

    # -- вызов ------------------------------------------------------------- #
    def __call__(
        self,
        attn: Any,
        hidden_states: Any,
        attention_mask: Any = None,
        rotary_emb: Any = None,
        layer_cache: Any = None,
        kv_cache_mode: str | None = None,
        cache_write_slice: slice | None = None,
        segments: list[tuple[int, int, bool]] | None = None,
        key_valid: Any = None,
    ) -> Any:
        torch = _torch()
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        from diffusers.models.transformers.transformer_qwenimage21 import (
            _qwenimage21_prepare_qkv,
        )

        query, key, value, seq_len_q = _qwenimage21_prepare_qkv(
            attn, hidden_states, rotary_emb, layer_cache, kv_cache_mode, cache_write_slice)

        target_len = self.height * self.width
        if segments is None:
            # decode: запросы — только целевое изображение, ключи — префикс из кэша + цель.
            prefix_len = max(0, int(key.shape[1]) - target_len)
            mask = attention_mask if attention_mask is None else attention_mask.to(torch.bool)
            hidden_states = self._windowed(query, key, value, prefix_len, mask)
        else:
            # prefill: префиксные сегменты считаются как в штатном процессоре (текст каузальный,
            # каждый блок изображения двунаправленный), целевое изображение — полосами.
            prefix_len = segments[-1][1] if segments else 0
            outputs = []
            for start, end, is_text in segments:
                seg_mask = None
                if is_text:
                    seg_len = end - start
                    seg_mask = torch.cat(
                        [
                            torch.ones(seg_len, start, dtype=torch.bool, device=query.device),
                            torch.tril(torch.ones(seg_len, seg_len, dtype=torch.bool,
                                                 device=query.device)),
                        ],
                        dim=1,
                    )[None, None]
                if key_valid is not None:
                    seg_key_valid = key_valid[:, None, None, :end]
                    seg_mask = seg_key_valid if seg_mask is None else (seg_mask & seg_key_valid)
                outputs.append(
                    dispatch_attention_fn(
                        query[:, start:end], key[:, :end], value[:, :end],
                        attn_mask=seg_mask,
                        dropout_p=0.0,
                        backend=None,
                        parallel_config=self._parallel_config,
                    )
                )
            target_mask = None if key_valid is None else key_valid[:, None, None, :]
            outputs.append(
                self._windowed(query[:, prefix_len:], key, value, prefix_len, target_mask))
            hidden_states = torch.cat(outputs, dim=1)

        hidden_states = hidden_states[:, :seq_len_q]
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        return attn.to_out[1](hidden_states)


def install(transformer: Any, *, grid: tuple[int, int], band_px: int,
            margin_px: int) -> dict[str, Any]:
    """Поставить оконный процессор на все attention-блоки; вернуть прежние для отката."""
    previous = dict(transformer.attn_processors)
    processor = SlidingWindowAttnProcessor(grid=grid, band_px=band_px, margin_px=margin_px)
    transformer.set_attn_processor(processor)
    return previous


def uninstall(transformer: Any, previous: dict[str, Any]) -> None:
    """Вернуть штатные процессоры (иначе режим утёк бы в следующий кадр)."""
    from diffusers.models.transformers.transformer_qwenimage21 import (
        QwenImage21AttnProcessor,
    )

    transformer.set_attn_processor(previous or QwenImage21AttnProcessor())


def describe(grid: tuple[int, int], band_px: int, margin_px: int) -> str:
    """Строка для лога: сколько полос и сколько ключей видит каждая."""
    height = max(1, int(grid[1]) // VAE_SCALE)
    width = max(1, int(grid[0]) // VAE_SCALE)
    band = max(1, int(band_px) // VAE_SCALE)
    margin = max(0, int(margin_px) // VAE_SCALE)
    bands = (height + band - 1) // band
    keys = min(height, band + 2 * margin) * width
    return (f"окно внимания {band * VAE_SCALE}px + запас {margin * VAE_SCALE}px: "
            f"{bands} полос по {width} латентов, каждая видит {keys} целевых токенов "
            f"из {height * width}")
