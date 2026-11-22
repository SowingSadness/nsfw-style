"""Кубик `source.frame`: геометрия кадра -> порт `frame` (план `workflow-layout`, шаг 1).

Зачем отдельный кубик: канвас нужен **двум** стадиям сразу — трансформеру (сетка латентов, в которой
идёт денойзинг) и энкодеру (сетка, на которой считаются vision-токены референсов). Если положить
`width`/`height` в трансформер, энкодеру пришлось бы читать чужие параметры; поэтому геометрия —
**данные, текущие по порту `frame`**:

    params = source_frame.FrameParams.from_mapping({"width": 2912, "height": 2176,
                                                    "ref_crop_side": 768})
    frame = source_frame.frame_of(params, ref_size=(1728, 1152))   # -> Frame

Семантика (фиксирована планом, менять только с миграцией шаблонов):

* `width`/`height` — канвас в пикселях, **кратны 32**; `0` = «вывести из пропорции референса по
  другой стороне» (`height: 768, width: 0` = прежний `--size 768p`). Больше одной нулевой стороны
  быть не может; оба нуля → 1024×1024 (T2I без референса и без геометрии);
* `ref_crop_side` — сторона сетки референсов **по высоте кадра** (как прежний `ref_px`): высота
  сетки = `min(ref_crop_side, height)`, ширина — по пропорции канваса, обе кратны 32; `0` = сетка
  канваса. Меньше канваса нужно на больших кадрах: на сетке канваса энкодер Qwen3VL-8B падает в OOM
  задолго до генерации.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from ..backends.base import ParameterSpec, ResourceBudget
from ..cube import CubeContext, CubeSpec, Stage, register_cube
from ..ports import Frame

__all__ = ["CUBE", "GRID_STEP", "FrameParams", "canvas", "embedding_grid", "frame_of", "resolution"]

#: Кратность сторон: `image_processor` пайплайна усекает кадр до 32, иначе латенты разъедутся.
GRID_STEP = 32
#: Канвас по умолчанию, когда нет ни геометрии, ни референса (T2I без `width`/`height`).
DEFAULT_CANVAS = (1024, 1024)
#: Минимум для сетки референсов (как в `RunConfig.embedding_crop`).
MIN_REF_SIDE = 256


@dataclass(frozen=True)
class FrameParams:
    """Разобранные параметры геометрии."""

    width: int = 0
    height: int = 0
    ref_crop_side: int = 0

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> "FrameParams":
        values = {p.name: p.default for p in CUBE.params}
        values.update({k: v for k, v in (params or {}).items() if k in values})
        return cls(width=int(values["width"] or 0), height=int(values["height"] or 0),
                   ref_crop_side=int(values["ref_crop_side"] or 0))


def _round32(value: float) -> int:
    """Округлить до кратного 32 (вверх-вниз по арифметике, как в `calculate_dimensions`)."""
    return max(GRID_STEP, int(round(float(value) / GRID_STEP)) * GRID_STEP)


def _floor32(value: float) -> int:
    """Срез до кратного 32 — ровно так `RunConfig.embedding_crop` считает сетку референсов."""
    return max(MIN_REF_SIDE, int(int(value) // GRID_STEP) * GRID_STEP)


def _check_side(name: str, value: int) -> None:
    if value and (value % GRID_STEP or value < GRID_STEP):
        raise ValueError(f"source.frame.{name}={value}: нужна кратность {GRID_STEP} и минимум "
                         f"{GRID_STEP} (иначе image_processor усечёт кадр и латенты разъедутся)")


def canvas(params: FrameParams, ref_size: tuple[int, int] = (0, 0)) -> tuple[int, int]:
    """Канвас (W, H) по параметрам и пропорции референса.

    `0` в одной из сторон — «достроить по пропорции референса»; оба нуля — 1024×1024. Результат
    всегда кратен 32, поэтому одинаков на всех машинах и совпадает с `calculate_dimensions`
    diffusers для тех же входах.
    """
    _check_side("width", params.width)
    _check_side("height", params.height)
    if params.width and params.height:
        return params.width, params.height
    ref_w, ref_h = (int(ref_size[0]), int(ref_size[1]))
    ratio = (ref_w / ref_h) if ref_w > 0 and ref_h > 0 else 1.0
    if params.width and not params.height:
        return params.width, _round32(params.width / ratio)
    if params.height and not params.width:
        return _round32(params.height * ratio), params.height
    return DEFAULT_CANVAS


def resolution(params: FrameParams, size: tuple[int, int]) -> int:
    """`output_resolution` пайплайна: `sqrt(W*H)`, как `RunConfig.effective_resolution` из `--size`."""
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        return 1024
    return max(256, int(round((width * height) ** 0.5)))


def embedding_grid(params: FrameParams, size: tuple[int, int]) -> tuple[int, int]:
    """Сетка, на которой энкодер видит референсы (кратна 32), по канвасу `size`.

    Совпадает с прежним `RunConfig.embedding_crop`: `ref_h = min(ref_crop_side, height)`,
    `ref_w = min(round(ref_h * ratio), width)`; `ref_crop_side = 0` → сетка канваса.
    """
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        return DEFAULT_CANVAS
    side = int(params.ref_crop_side or 0)
    if side <= 0:
        return width, height
    ref_h = max(MIN_REF_SIDE, min(_round32(side), height))
    ratio = width / max(1, height)
    ref_w = max(MIN_REF_SIDE, min(int(round(ref_h * ratio)), width))
    # срез, а не округление: так же считает `RunConfig.embedding_crop` (иначе сетка разъедется
    # на 32 px и ключ кэша эмбеддингов перестанет совпадать)
    return _floor32(ref_w), _floor32(ref_h)


def frame_of(params: FrameParams, ref_size: tuple[int, int] = (0, 0)) -> Frame:
    """Собрать порт `Frame`: канвас, сетка референсов и `output_resolution` в одном месте."""
    width, height = canvas(params, ref_size)
    grid = embedding_grid(params, (width, height))
    return Frame(width=width, height=height, ref_crop_side=int(params.ref_crop_side or 0),
                 resolution=resolution(params, (width, height)))


def _self_test(ctx: CubeContext) -> None:
    """Проверка без весов: авто-сторона, кратность 32, сетка референсов как у `ref_px`."""
    # `height` + авто-ширина = прежний `--size 768p` на 16:9-референсе
    auto = FrameParams.from_mapping({"height": 768})
    assert canvas(auto, (864, 480)) == (1376, 768), canvas(auto, (864, 480))
    # явный канвас приоритетнее референса
    explicit = FrameParams.from_mapping({"width": 2912, "height": 2176})
    assert canvas(explicit, (1728, 1152)) == (2912, 2176)
    # оба нуля -> квадрат (T2I без геометрии)
    assert canvas(FrameParams.from_mapping({}), (0, 0)) == DEFAULT_CANVAS
    # сетка референсов: тот же результат, что давал `ref_px` при канвасе 2912x2176
    grid = embedding_grid(FrameParams.from_mapping({"ref_crop_side": 768}), (2912, 2176))
    assert grid == (1024, 768), grid
    assert embedding_grid(FrameParams.from_mapping({}), (2912, 2176)) == (2912, 2176)
    # resolution = sqrt(area)
    assert resolution(explicit, (2912, 2176)) == 2517, resolution(explicit, (2912, 2176))
    # некратная 32 сторона отвергается
    for bad in ({"width": 100}, {"height": 300}, {"width": 33}):
        try:
            canvas(FrameParams.from_mapping(bad))
        except ValueError:
            pass
        else:                                                    # pragma: no cover
            raise AssertionError(f"{bad} обязан отвергаться")
    frame = frame_of(explicit, (1728, 1152))
    assert (frame.width, frame.height, frame.resolution) == (2912, 2176, 2517), frame
    assert ctx is not None


CUBE: CubeSpec = register_cube(CubeSpec(
    id="source.frame",
    stage=Stage.SOURCE,
    produces="frame",
    params=(
        ParameterSpec("width", "cube", "int", 0, minimum=0, maximum=8192,
                      doc="канвас по ширине, px (кратно 32); 0 = вывести из пропорции референса "
                          "по высоте; оба 0 = 1024x1024"),
        ParameterSpec("height", "cube", "int", 0, minimum=0, maximum=8192,
                      doc="канвас по высоте, px (кратно 32); 0 = вывести из пропорции референса "
                          "по ширине (768 — прежний --size 768p)"),
        ParameterSpec("ref_crop_side", "cube", "int", 0, minimum=0, maximum=2048,
                      doc="сторона сетки референсов по высоте кадра, px (кратно 32): меньше "
                          "канваса на больших кадрах, иначе энкодер падает в OOM; 0 = сетка канваса"),
    ),
    budget=ResourceBudget(notes="только арифметика сеток: ни весов, ни VRAM"),
    self_test=_self_test,
    doc="геометрия кадра: канвас (width/height) и сетка референсов (ref_crop_side) -> порт frame",
))
