"""Кубик `decoder.vae.qwen21`: декодер VAE с настройками тайлинга и слайсинга.

Зачем отдельный кубик (план `cube-architecture`, шаг 2): раньше настройка VAE жила внутри
`backends/qwen_image21.py::prepare()` и читала `generation.sampler_kwargs["vae_tile_px"]` руками.
Из-за этого UI не мог показать «параметры декодера», а опечатка в пресете ловилась в рантайме.

Теперь у кубика своя спека (`CUBE`) со своим `ParameterSpec`-набором и `self_test` без весов:

    cube = decoder_vae.CUBE
    params = decoder_vae.DecodeParams.from_mapping(cube, {"tile_px": 512})
    decoder_vae.setup(vae, params)           # slicing + tiling по параметрам
    pixels = decoder_vae.decode(vae, latents)

Семантика `tile_px` сохранена по замерам (`AGENTS.md`, «тайлинг декодера VAE»):

* `512` (умолчание) — тайл 512 px, шаг 384: 16-px сетка на асфальте ×1.69 против ×2.71,
  пик VRAM 7.4 вместо 7.2 ГиБ;
* `0` — историческое поведение diffusers (256/192);
* `N > 0` — свой тайл, шаг считается как `0.75·N` (та же доля перекрытия, что у diffusers);
* `-1` — без тайлинга (декод всего кадра; на 2912×2176 это OOM).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from ..backends.base import ParameterSpec, ResourceBudget
from ..cube import CubeContext, CubeSpec, Stage, register_cube
from ..ports import Frames, Latents

__all__ = ["CUBE", "DecodeParams", "Qwen21Decoder", "decode", "setup", "tiling_for"]

#: Доля шага от размера тайла: у diffusers по умолчанию 192/256 = 0.75.
STRIDE_RATIO = 0.75


@dataclass(frozen=True)
class DecodeParams:
    """Разобранные параметры декодера (уже провалидированы по дескрипторам кубика)."""

    tile_px: int = 512
    tiling: bool = True
    slicing: bool = True
    dtype: str = "bfloat16"
    device: str = "auto"

    @property
    def stride_px(self) -> int:
        """Шаг тайла в пикселях; для `tile_px <= 0` не используется."""
        if self.tile_px <= 0:
            return 0
        return max(1, int(round(self.tile_px * STRIDE_RATIO)))

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> "DecodeParams":
        """Собрать из «словаря параметров кубика». Дефолты берутся из `CUBE.params`."""
        values = {p.name: p.default for p in CUBE.params}
        values.update({k: v for k, v in (params or {}).items() if k in values})
        return cls(tile_px=int(values["tile_px"]),
                   tiling=bool(values["tiling"]),
                   slicing=bool(values["slicing"]),
                   dtype=str(values["dtype"]),
                   device=str(values["device"]))

    def enable_kwargs(self) -> dict[str, int]:
        """Аргументы `vae.enable_tiling(...)` для текущего `tile_px` (пусто = дефолты diffusers)."""
        if self.tile_px <= 0:
            return {}
        stride = self.stride_px
        return {"tile_sample_min_height": self.tile_px, "tile_sample_min_width": self.tile_px,
                "tile_sample_stride_height": stride, "tile_sample_stride_width": stride}


def tiling_for(tile_px: int) -> dict[str, int]:
    """Короткий путь для старого кода: `tile_px` -> аргументы `enable_tiling` (или пусто)."""
    return DecodeParams(tile_px=int(tile_px)).enable_kwargs()


def setup(vae: Any, params: DecodeParams, *, log: Any = None) -> DecodeParams:
    """Применить настройки декодера к объекту VAE (diffusers `AutoencoderKL*`).

    Ничего не грузит и не перемещает: вызывающая сторона уже держит VAE на нужном устройстве.
    """
    say = log or (lambda _msg: None)
    if params.slicing:
        vae.enable_slicing()
    if not params.tiling or params.tile_px < 0:
        vae.use_tiling = False
        say("[vae] тайлинг декодера выключен (tile_px=-1)")
        return params
    kwargs = params.enable_kwargs()
    if kwargs:
        vae.enable_tiling(**kwargs)
        say(f"[vae] тайл декодера {params.tile_px} px, шаг {params.stride_px} px")
    else:
        vae.enable_tiling()
        say("[vae] тайл декодера 256 px, шаг 192 px (историческое умолчание)")
    return params


def decode(vae: Any, latents: Latents, params: DecodeParams | None = None, *,
           log: Any = None) -> Frames:
    """Декодировать латент в **тензор** кадра (`Frames`). `params=None` — умолчания кубика.

    Шаги повторяют блок декода diffusers (`QwenImage21DecodeStep`) один-в-один, иначе кадр не совпадёт
    с прогоном через пайплайн:

    1. латент `(B, C, T, H, W)` или `(B, C, H, W)` (последний получает ось `T=1`);
    2. **денормализация** по `vae.config.latents_mean`/`latents_std` — без неё картинка
       разъезжается по яркости и цвету (в пайплайне это делает блочный код, а не VAE);
    3. `vae.decode(..., return_dict=False)[0]` — тензор `(B, C, T, H, W)` (или 4D).

    В `PIL.Image` (и вообще в контейнер) тензор превращает **следующий** кубик —
    `decoder.postprocess.image`: у картинки и видео разные форматы, и декодер не должен их знать.
    """
    import torch

    params = params or DecodeParams()
    x = latents.x
    with torch.no_grad():
        if x.ndim == 4:
            x = x.unsqueeze(dim=2)
        x = x.to(vae.dtype)
        config = getattr(vae, "config", None)
        mean = getattr(config, "latents_mean", None)
        std = getattr(config, "latents_std", None)
        if mean is not None and std is not None:
            z_dim = int(getattr(config, "z_dim", x.shape[1]))
            mean_t = torch.tensor(mean).view(1, z_dim, 1, 1, 1).to(x.device, x.dtype)
            std_t = torch.tensor(std).view(1, z_dim, 1, 1, 1).to(x.device, x.dtype)
            x = x * std_t + mean_t
        decoded = vae.decode(x, return_dict=False)[0]
    # `vae.decode` уже отдаёт пиксельную сетку (проверено замером паритета: латент 30x54 ->
    # декод 480x864 при VAE-факторе 16), поэтому размер берётся из тензора как есть.
    temporal = int(decoded.shape[2]) if decoded.ndim == 5 else 1
    height = int(decoded.shape[-2])
    width = int(decoded.shape[-1])
    return Frames(x=decoded, size=(width, height), temporal=temporal,
                  source="decoder.vae.qwen21")


class Qwen21Decoder:
    """Декодер VAE как объект: **параметры в конструкторе**, декод — через `__call__`.

    Почему объект, а не свободная функция (шаг 7b плана `cube-architecture`):

    * параметры (`tile_px`/`tiling`/`slicing`/`dtype`/`device`) принадлежат экземпляру, а не вызову —
      их нельзя случайно перепутать или перезаписать на середине батча;
    * VAE — тяжёлый компонент: `setup(vae)` один раз применяет tiling/slicing и держит ссылку,
      а `__call__` остаётся коротким (его зовёт `pipeline.run` и замеры паритета).

    Пример:

        dec = Qwen21Decoder.from_mapping({"tile_px": 512})
        dec.setup(pipe.vae)
        frames = dec(latents)               # -> Frames (тензор, ещё не PIL)
    """

    def __init__(self, params: DecodeParams | None = None) -> None:
        self.params = params or DecodeParams()
        self.vae: Any = None
        self._ready = False

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any] | None = None) -> "Qwen21Decoder":
        return cls(DecodeParams.from_mapping(params or {}))

    def setup(self, vae: Any, *, log: Any = None) -> "Qwen21Decoder":
        """Привязать VAE и применить настройки тайлинга/слайсинга (один раз на прогон)."""
        setup(vae, self.params, log=log)
        self.vae = vae
        self._ready = True
        return self

    def __call__(self, latents: Latents) -> Frames:
        if not self._ready or self.vae is None:
            raise RuntimeError("Qwen21Decoder: сначала setup(vae) — декодер без VAE не работает")
        return decode(self.vae, latents, self.params)


def _self_test(ctx: CubeContext) -> None:
    """Проверка без весов: параметры разбираются, `enable_tiling` получает верные аргументы."""
    class _Stub:
        use_tiling = True

        def __init__(self) -> None:
            self.kwargs: dict[str, int] | None = None
            self.slicing = False

        def enable_slicing(self) -> None:
            self.slicing = True

        def enable_tiling(self, **kwargs: int) -> None:
            self.kwargs = kwargs

    stub = _Stub()
    setup(stub, DecodeParams.from_mapping({"tile_px": 512}), log=ctx.log)
    assert stub.slicing, "slicing не включён"
    assert stub.kwargs == {"tile_sample_min_height": 512, "tile_sample_min_width": 512,
                           "tile_sample_stride_height": 384, "tile_sample_stride_width": 384},\
        stub.kwargs
    legacy = _Stub()
    setup(legacy, DecodeParams.from_mapping({"tile_px": 0}), log=ctx.log)
    assert legacy.kwargs == {}, "tile_px=0 обязан оставить дефолты diffusers (256/192)"
    off = _Stub()
    setup(off, DecodeParams.from_mapping({"tile_px": -1}), log=ctx.log)
    assert off.use_tiling is False, "tile_px=-1 обязан выключить тайлинг"


CUBE: CubeSpec = register_cube(CubeSpec(
    id="decoder.vae.qwen21",
    stage=Stage.DECODER,
    consumes=frozenset({"latents"}),
    produces="frames",
    params=(
        ParameterSpec("tile_px", "cube", "int", 512, minimum=-1, maximum=8192,
                      doc="тайл декодера, px: 512 (умолчание) = 512/384; 0 = историческое 256/192; "
                          "-1 = без тайлинга (на 2912x1632 декод не влезает в 16 ГиБ)"),
        ParameterSpec("tiling", "cube", "bool", True,
                      doc="включить пространственный тайлинг декодера (иначе декод всего кадра)"),
        ParameterSpec("slicing", "cube", "bool", True,
                      doc="слайсинг декодера (diffusers enable_slicing): память под батч"),
        ParameterSpec("dtype", "cube", "enum", "bfloat16", choices=("bfloat16", "float16", "float32"),
                      doc="точность декодера; bf16 — как в пайплайне"),
        ParameterSpec("device", "cube", "enum", "auto", choices=("auto", "cuda", "cpu"),
                      doc="устройство декодера; auto = cuda при наличии"),
    ),
    budget=ResourceBudget(weights_bf16_gib=1.35, notes="VAE Qwen-Image-2.1 (238 тензоров)"),
    self_test=_self_test,
    doc="VAE-декодер Qwen-Image-2.1: латент -> тензор кадра (frames), с тайлингом/слайсингом; в PIL его превращает decoder.postprocess.image",
))
