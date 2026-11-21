"""Кубик `post.upscale.seedvr2`: реставрация/апскейл SeedVR2-3B (стадия POST, dev-only).

Роль: `pixels -> pixels`. Кубик тонкий: вся работа — во внешнем окружении `seedvr2/` (torch 2.3
cu121), запуск через `seedvr2/run_image.py` подпроцессом (`backends/seedvr2.py`). Цветокоррекция
из бекенда **не** применяется здесь: за неё отвечает отдельный кубик `post.colorfix`, поэтому
SeedVR2 не тянет её дважды.

Почему dev-only: на 16 ГБ VRAM x4 (3456×1920) падает в OOM по активациям даже с fp8-весами,
поэтому в публичные пресеты он не входит (см. AGENTS.md §4).

Параметры: `width`/`height` — целевой кадр (кратен 16; 0 = считать из `factor`), `factor` —
множитель, если размеры не заданы, `fp8` — хранение весов (bf16 на 16 ГБ не влезает).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..backends.base import ParameterSpec, ResourceBudget
from ..cube import CubeContext, CubeSpec, Stage, register_cube
from ..ports import Pixels

__all__ = ["CUBE", "Seedvr2Params", "status", "target_size", "upscale"]

#: Кратность стороны кадра у SeedVR2 (см. `backends/seedvr2.py::_target_size`).
SIZE_MULTIPLE = 16


@dataclass(frozen=True)
class Seedvr2Params:
    """Разобранные параметры SeedVR2."""

    width: int = 0
    height: int = 0
    factor: float = 0.0
    fp8: bool = True

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> "Seedvr2Params":
        values = {p.name: p.default for p in CUBE.params}
        values.update({k: v for k, v in (params or {}).items() if k in values})
        return cls(width=int(values["width"] or 0), height=int(values["height"] or 0),
                   factor=float(values["factor"] or 0.0), fp8=bool(values["fp8"]))


def target_size(size: tuple[int, int], params: Seedvr2Params) -> tuple[int, int]:
    """Целевой размер кадра: явные `width`/`height` главнее `factor`; итог кратен 16."""
    width, height = size
    if params.width and params.height:
        return (max(SIZE_MULTIPLE, round(params.width / SIZE_MULTIPLE) * SIZE_MULTIPLE),
                max(SIZE_MULTIPLE, round(params.height / SIZE_MULTIPLE) * SIZE_MULTIPLE))
    factor = params.factor or 2.0
    return (max(SIZE_MULTIPLE, round(width * factor / SIZE_MULTIPLE) * SIZE_MULTIPLE),
            max(SIZE_MULTIPLE, round(height * factor / SIZE_MULTIPLE) * SIZE_MULTIPLE))


def status() -> tuple[bool, str]:
    """(установлен?, причина) — без запуска процесса и без загрузки весов."""
    from ..backends.seedvr2 import Seedvr2Backend

    return Seedvr2Backend().status()


def upscale(pixels: Pixels, params: Seedvr2Params, *, source: Any = None,
            seed: int = 0, log: Any = None) -> Pixels:
    """Прогнать SeedVR2 до целевого размера. Ошибка установки — `RuntimeError` с подсказкой."""
    say = log or (lambda _msg: None)
    ok, why = status()
    if not ok:
        from ..backends.seedvr2 import Seedvr2Backend

        raise RuntimeError(Seedvr2Backend().missing_help(why))
    from ..backends import seedvr2

    image = pixels.image
    width, height = target_size((image.width, image.height), params)
    factor = height / max(1, image.height)
    out = seedvr2.Seedvr2Backend().upscale(image, factor=factor, seed=seed,
                                           source=getattr(source, "image", source),
                                           fp8=params.fp8)
    say(f"[post] seedvr2 {image.width}x{image.height} -> {out.width}x{out.height} (fp8={params.fp8})")
    return Pixels(image=out, stage_trace=pixels.stage_trace + ("post.upscale.seedvr2",))


def _self_test(ctx: CubeContext) -> None:
    """Без весов: целевой размер (кратность 16), приоритет width/height, наличие окружения."""
    assert target_size((864, 480), Seedvr2Params.from_mapping({"factor": 2.0})) == (1728, 960)
    assert target_size((864, 480), Seedvr2Params.from_mapping({"factor": 4.0})) == (3456, 1920)
    explicit = Seedvr2Params.from_mapping({"width": 1728, "height": 961})
    assert target_size((864, 480), explicit) == (1728, 960), target_size((864, 480), explicit)
    # дефолт — x2 (как у бекенда), fp8 включён (bf16 на 16 ГБ не влезает)
    default = Seedvr2Params.from_mapping({})
    assert default.fp8 and target_size((640, 480), default) == (1280, 960)
    ok, why = status()
    assert isinstance(ok, bool) and isinstance(why, str), (ok, why)


CUBE: CubeSpec = register_cube(CubeSpec(
    id="post.upscale.seedvr2",
    stage=Stage.POST,
    consumes=frozenset({"pixels"}),
    produces="pixels",
    params=(
        ParameterSpec("width", "cube", "int", 0, minimum=0, maximum=8192,
                      doc="целевая ширина, px (кратна 16); 0 вместе с height = считать из factor"),
        ParameterSpec("height", "cube", "int", 0, minimum=0, maximum=8192,
                      doc="целевая высота, px (кратна 16); 0 = из factor"),
        ParameterSpec("factor", "cube", "float", 0.0, minimum=0.0, maximum=4.0,
                      doc="множитель, если width/height не заданы; 0 = x2 (умолчание бекенда); "
                          "x4 (3456x1920) на 16 ГиБ падает по активациям"),
        ParameterSpec("fp8", "cube", "bool", True,
                      doc="fp8-хранение весов (3.64 ГиБ вместо 12.6); bf16 на 16 ГиБ не влезает"),
    ),
    budget=ResourceBudget(weights_bf16_gib=12.6, weights_fp8_gib=3.64, disk_gib=17.0,
                          notes="отдельное окружение seedvr2/ (torch 2.3 cu121) + веса 3B"),
    self_test=_self_test,
    doc="постобработка SeedVR2-3B (dev-only): pixels -> pixels, реставрация до целевого размера",
))
