"""Стратегия `strategy.window_attention`: скользящее окно внимания по строкам канваса.

Кубик-декоратор (план `cube-architecture`, шаг 3), `provides={"window_attention"}`.

Механика (`qwenstyle/backends/qwen_sliding.py`): латент один и глобальный, позиции глобальные,
режется только сам attention — полосы по `band` строк, каждая видит префикс (текст + референсы)
и окрестность ± `margin` строк. Никакого усреднения окон, поэтому швов нет по построению.

Ограничения, зафиксированные замерами (`README`, `AGENTS.md`):

* `band_px >= высоты канваса` → одна полоса = плотное внимание, кадр бит-в-бит как нативный;
* покрытие `(band + 2*margin) / height` должно быть ≳ ⅔ канваса, иначе полоса дорисовывает свою
  сцену и кадр размножается;
* VRAM режим не экономит в рабочей зоне — выигрыш в скорости (~1.2×) и чуть меньше ряби.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..backends.base import ParameterSpec, ResourceBudget
from ..cube import CubeContext, CubeSpec, Stage, register_cube

__all__ = ["CUBE", "WindowParams", "coverage"]

#: Клетка латента Qwen-Image-2.1.
VAE_SCALE = 16


@dataclass(frozen=True)
class WindowParams:
    """Разобранные параметры окна: высота полосы и запас по строкам (в пикселях)."""

    band_px: int = 0
    margin_px: int = 0

    @property
    def active(self) -> bool:
        return self.band_px > 0

    @property
    def band_rows(self) -> int:
        return max(1, self.band_px // VAE_SCALE)

    @property
    def margin_rows(self) -> int:
        return max(0, self.margin_px // VAE_SCALE)

    def geometry(self, canvas: tuple[int, int]) -> dict[str, Any]:
        """Число полос, видимые токены и доля покрытия для канваса (W, H) в пикселях."""
        width = max(1, int(canvas[0]) // VAE_SCALE)
        height = max(1, int(canvas[1]) // VAE_SCALE)
        band = self.band_rows
        margin = self.margin_rows
        bands = (height + band - 1) // band
        visible_rows = min(height, band + 2 * margin)
        return {"width": width, "height": height, "band": band, "margin": margin,
                "bands": bands, "visible_rows": visible_rows,
                "tokens_visible": visible_rows * width, "tokens_total": height * width,
                "coverage": visible_rows / height if height else 0.0}

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> "WindowParams":
        values = {p.name: p.default for p in CUBE.params}
        values.update({k: v for k, v in (params or {}).items() if k in values})
        return cls(band_px=int(values["window_attn_px"] or 0),
                   margin_px=int(values["window_attn_margin_px"] or 0))


def coverage(canvas: tuple[int, int], band_px: int, margin_px: int) -> float:
    """Доля канваса, видимая одной полосой (0…1)."""
    return WindowParams(band_px, margin_px).geometry(canvas)["coverage"]


def _self_test(ctx: CubeContext) -> None:
    """Без весов и GPU: арифметика полос на реальных числах наших прогонов."""
    off = WindowParams.from_mapping({})
    assert not off.active, "0 = плотное внимание"

    # 2912x2176, окно 1088 + 448 — прогон nita-tokyo (2 полосы, 91 %)
    live = WindowParams.from_mapping({"window_attn_px": 1088, "window_attn_margin_px": 448})
    geo = live.geometry((2912, 2176))
    assert geo["bands"] == 2, geo
    assert geo["visible_rows"] == 124, geo
    assert geo["tokens_visible"] == 22568 and geo["tokens_total"] == 24752, geo
    assert abs(geo["coverage"] - 22568 / 24752) < 1e-9, geo

    # гейт паритета: полоса не меньше высоты -> одна полоса и 100 % покрытия
    whole = WindowParams.from_mapping({"window_attn_px": 4096, "window_attn_margin_px": 0})
    geo_whole = whole.geometry((1024, 768))
    assert geo_whole["bands"] == 1 and geo_whole["coverage"] == 1.0, geo_whole

    # порог режима: 36 % покрытия -> меньше 2/3 (кадр размножается)
    bad = WindowParams.from_mapping({"window_attn_px": 384, "window_attn_margin_px": 128})
    assert bad.geometry((3200, 1792))["coverage"] < 2 / 3, bad.geometry((3200, 1792))


CUBE: CubeSpec = register_cube(CubeSpec(
    id="strategy.window_attention",
    stage=Stage.TRANSFORMER,
    provides=frozenset({"window_attention"}),
    params=(
        ParameterSpec("window_attn_px", "cube", "int", 0, minimum=0, maximum=4096,
                      doc="высота полосы внимания, px: 0 = плотное внимание; кадр один и "
                          "глобальный, режется только attention"),
        ParameterSpec("window_attn_margin_px", "cube", "int", 0, minimum=0, maximum=2048,
                      doc="запас строк сверху и снизу к полосе, px (шире контекст, дороже память); "
                          "покрытие (полоса+2*запас)/высота должно быть >~2/3"),
    ),
    budget=ResourceBudget(weights_bf16_gib=0.0, notes="attention режется по строкам, латент один"),
    self_test=_self_test,
    doc="скользящее окно внимания по строкам: полосы видят префикс и окрестность ± запас",
))
