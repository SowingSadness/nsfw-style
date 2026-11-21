"""Кубик `post.colorfix`: цветокоррекция кадра по исходнику (стадия POST, `pixels -> pixels`).

Зачем отдельный кубик: коррекция нужна не только SeedVR2 (для неё её и писали — у официального
SeedVR2 перед сохранением идёт `wavelet_reconstruction`), но и любому другому апскейлеру, а
включать её руками в бекенде — значит терять её при смене бекенда. Здесь она одна на всех.

    params = colorfix_cube.ColorfixParams.from_mapping({"mode": "wavelet", "strength": 0.8})
    pixels = colorfix_cube.apply(pixels, params, source=refs_pixels)

`source` — исходный кадр (референс/база), к которому подтягивается цвет. Без него кубик
возвращает кадр без изменений: источники есть не в каждой композиции.

Режимы (формулы не менялись, см. `qwenstyle/colorfix.py`):

* `wavelet` (умолчание) — низкие частоты (цвет, яркость, контраст) от источника, высокие
  (детали, микротекстура) от результата; `sigma` — радиус размытия, `strength` — сила смеси;
* `reinhard` — перегон mean/std по каналам, мягче и тупее;
* `none` — выключено.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..backends.base import ParameterSpec, ResourceBudget
from ..cube import CubeContext, CubeSpec, Stage, register_cube
from ..ports import Pixels

__all__ = ["CUBE", "MODES", "ColorfixParams", "apply"]

#: Режимы коррекции (`colorfix.apply_color_fix`).
MODES = ("none", "wavelet", "reinhard")


@dataclass(frozen=True)
class ColorfixParams:
    """Разобранные параметры коррекции."""

    mode: str = "wavelet"
    sigma: float = 12.0
    strength: float = 1.0
    match_saturation: bool = True

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> "ColorfixParams":
        values = {p.name: p.default for p in CUBE.params}
        values.update({k: v for k, v in (params or {}).items() if k in values})
        return cls(mode=str(values["mode"]), sigma=float(values["sigma"]),
                   strength=float(values["strength"]),
                   match_saturation=bool(values["match_saturation"]))


def apply(pixels: Pixels, params: ColorfixParams, *, source: Any = None,
          log: Any = None) -> Pixels:
    """Подтянуть цвет кадра к `source`. Без источника или при `mode=none` — без изменений."""
    say = log or (lambda _msg: None)
    if params.mode == "none" or source is None:
        return pixels
    from ..colorfix import apply_color_fix

    image = pixels.image
    source_image = getattr(source, "image", source)
    fixed = apply_color_fix(image, source_image, mode=params.mode, sigma=params.sigma,
                           strength=params.strength, match_saturation=params.match_saturation)
    say(f"[post] colorfix {params.mode}: sigma={params.sigma:g}, strength={params.strength:g}")
    return Pixels(image=fixed, stage_trace=pixels.stage_trace + ("post.colorfix",))


def _self_test(ctx: CubeContext) -> None:
    """Без весов: разбор параметров и реальная коррекция на 8x8 картинках."""
    from PIL import Image

    params = ColorfixParams.from_mapping({})
    assert params.mode == "wavelet" and params.strength == 1.0, params
    # source=None и mode=none — тождественны по объекту (кубик ничего не делает)
    image = Image.new("RGB", (8, 8), (10, 20, 30))
    pixels = Pixels(image=image)
    assert apply(pixels, ColorfixParams.from_mapping({"mode": "none"}),
                 source=image).image is image
    assert apply(pixels, params, source=None).image is image
    # wavelet тянет яркость к источнику: серый кадр на светлом источнике светлеет
    gray = Image.new("RGB", (8, 8), (40, 40, 40))
    bright = Image.new("RGB", (8, 8), (200, 200, 200))
    fixed = apply(Pixels(image=gray), params, source=bright, log=ctx.log).image
    assert fixed.size == gray.size, fixed.size
    assert fixed.getpixel((4, 4))[0] > 60, fixed.getpixel((4, 4))
    # reinhard тоже принимается
    same = apply(Pixels(image=gray), ColorfixParams.from_mapping({"mode": "reinhard"}),
                 source=bright, log=ctx.log).image
    assert same.size == gray.size, same.size


CUBE: CubeSpec = register_cube(CubeSpec(
    id="post.colorfix",
    stage=Stage.POST,
    consumes=frozenset({"pixels"}),
    produces="pixels",
    params=(
        ParameterSpec("mode", "post", "enum", "wavelet", choices=MODES,
                      doc="none = выключено; wavelet = низкие частоты от источника (как у SeedVR2); "
                          "reinhard = перегон mean/std по каналам"),
        ParameterSpec("sigma", "post", "float", 12.0, minimum=0.0, maximum=64.0,
                      doc="радиус размытия для разделения частот (только wavelet), px"),
        ParameterSpec("strength", "post", "float", 1.0, minimum=0.0, maximum=1.0,
                      doc="сила смеси с результатом: 1 = полная коррекция, 0 = как было"),
        ParameterSpec("match_saturation", "post", "bool", True,
                      doc="подтянуть среднюю насыщенность к источнику (только wavelet)"),
    ),
    budget=ResourceBudget(notes="numpy + PIL, весов нет; пик на 2912x2176 ~ 100 МиБ RAM"),
    self_test=_self_test,
    doc="цветокоррекция кадра по исходнику: pixels -> pixels (wavelet/reinhard/none)",
))
