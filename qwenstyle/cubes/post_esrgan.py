"""Кубик `post.upscale.esrgan`: постобработка Real-ESRGAN (стадия POST, `pixels -> pixels`).

Роль: увеличить готовый кадр x4-сетью (anime6B) и, если нужен дробный множитель, вернуть его
до целевой высоты LANCZOS. Реализация — `backends/esrgan.py` (своя RRDBNet, без basicSR);
кубик добавляет к ней только **декларацию** и чистое решение «какой множитель» — то место, где
раньше путались `factor` и `target_height`.

Правила множителя (сохранены 1:1 с `--upscale`, см. AGENTS.md §4):

* `target_height` главнее `factor`: `factor = target_height / высота_кадра`;
* фактор ≈ 1 или кадр уже не ниже цели — **пропуск**, а не апскейл (суффикс `-x1` не появляется);
* `factor = 4` — x4-сеть как есть, `factor = 2` — x4 и уменьшение вдвое, `N < 4` — LANCZOS;
* `factor = 0` без `target_height` — апскейл выключен.

Тайлы: на текущем стеке тайловый режим даёт `max diff 7` из 255 к полному кадру — это свойство
режима (старый `upscale.py` давал то же), `tile=256`/`pad=32` — умолчание и «бит-в-бит» к прежним
прогонам.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..backends.base import ParameterSpec, ResourceBudget
from ..cube import CubeContext, CubeSpec, Stage, register_cube
from ..ports import Pixels

__all__ = ["CUBE", "DEFAULT_WEIGHTS", "EsrganParams", "plan_factor", "resolve_factor", "upscale"]

#: Путь весов по умолчанию — тот же, что в пресетах (`upscale.weights`).
DEFAULT_WEIGHTS = "cache/models/weights/RealESRGAN_x4plus_anime_6B.pth"

#: Признак «апскейл не нужен»: фактор ниже этого считается ~1 и кадр возвращается как есть.
MIN_FACTOR = 1.01


@dataclass(frozen=True)
class EsrganParams:
    """Разобранные параметры апскейла."""

    weights: str = ""
    factor: float = 0.0
    target_height: int | None = None
    tile: int = 256
    pad: int = 32

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> "EsrganParams":
        values = {p.name: p.default for p in CUBE.params}
        values.update({k: v for k, v in (params or {}).items() if k in values})
        target = values["target_height"]
        return cls(weights=str(values["weights"] or ""), factor=float(values["factor"] or 0.0),
                   target_height=None if target in (None, 0) else int(target),
                   tile=int(values["tile"]), pad=int(values["pad"]))


def resolve_factor(height: int, params: EsrganParams) -> float:
    """Множитель для кадра высотой `height`: `target_height` главнее `factor`."""
    if params.target_height:
        return float(params.target_height) / max(1, int(height))
    return float(params.factor or 0.0)


def plan_factor(height: int, params: EsrganParams) -> float:
    """Фактор с учётом пропуска: `0.0` означает «апскейл не нужен» (бекенд не грузится)."""
    factor = resolve_factor(height, params)
    if factor <= 0.0 or factor < MIN_FACTOR:
        return 0.0
    return factor


def upscale(pixels: Pixels, params: EsrganParams, *, log: Any = None) -> Pixels:
    """Увеличить кадр. Сеть грузится только при ненулевом факторе (см. `plan_factor`)."""
    say = log or (lambda _msg: None)
    image = pixels.image
    factor = plan_factor(int(image.height), params)
    if factor <= 0.0:
        say(f"[post] upscale пропущен: цель {params.target_height} px при высоте {image.height}")
        return pixels
    from ..backends import esrgan

    path = esrgan.ensure_weights(params.weights or None)
    device, fp16 = esrgan._device_and_dtype()
    model = esrgan.load_upscaler(path, device=device, fp16=fp16)
    try:
        out = esrgan.upscale_image(image, model, factor=factor, tile=params.tile, pad=params.pad)
    finally:
        del model
        esrgan._empty_cuda_cache()
    say(f"[post] esrgan x{factor:.3g}: {image.width}x{image.height} -> {out.width}x{out.height}")
    return Pixels(image=out, stage_trace=pixels.stage_trace + ("post.upscale.esrgan",))


def _self_test(ctx: CubeContext) -> None:
    """Без весов: правила множителя, пропуск и разбор параметров."""
    params = EsrganParams.from_mapping({"target_height": 1440})
    assert abs(resolve_factor(1088, params) - 1440 / 1088) < 1e-9, resolve_factor(1088, params)
    assert plan_factor(1088, params) > 1.0, "1088 -> 1440 обязан апскейлиться"
    # кадр уже не ниже цели -> пропуск (0.0), сеть не грузится
    assert plan_factor(1440, params) == 0.0, plan_factor(1440, params)
    assert plan_factor(2176, params) == 0.0
    # явный factor без цели
    assert plan_factor(480, EsrganParams.from_mapping({"factor": 2.0})) == 2.0
    # выключено: factor=0 и нет target_height
    assert plan_factor(480, EsrganParams.from_mapping({})) == 0.0
    # тайлы по умолчанию — 256/32 (бит-в-бит к прежним прогонам)
    default = EsrganParams.from_mapping({})
    assert (default.tile, default.pad) == (256, 32), (default.tile, default.pad)


CUBE: CubeSpec = register_cube(CubeSpec(
    id="post.upscale.esrgan",
    stage=Stage.POST,
    consumes=frozenset({"pixels"}),
    produces="pixels",
    params=(
        ParameterSpec("weights", "cube", "str", DEFAULT_WEIGHTS,
                      doc="путь к .pth (по умолчанию cache/models/weights/"
                          "RealESRGAN_x4plus_anime_6B.pth, скачивается автоматически)"),
        ParameterSpec("factor", "upscale", "float", 0.0, minimum=0.0, maximum=4.0,
                      doc="множитель; 0 = выключено; target_height главнее factor"),
        ParameterSpec("target_height", "upscale", "int", None, minimum=0,
                      doc="целевая высота кадра, px: factor считается из неё "
                          "(1440 у VRAM16, 2160 у 4K-пресетов)"),
        ParameterSpec("tile", "upscale", "int", 256, minimum=64, maximum=1024,
                      doc="тайл инференса, px: 256 (умолчание, pad 32) — бит-в-бит к прежним прогонам"),
        ParameterSpec("pad", "upscale", "int", 32, minimum=0, maximum=128,
                      doc="перекрытие тайлов, px: при tile=128 нужен 32"),
    ),
    budget=ResourceBudget(weights_bf16_gib=0.02, disk_gib=0.02,
                          notes="Real-ESRGAN x4plus_anime_6B, 17.9 МиБ"),
    self_test=_self_test,
    doc="постобработка кадра сетью Real-ESRGAN: pixels -> pixels (пропуск, если кадр уже не ниже цели)",
))
