"""Кубик `decoder.postprocess.image`: тензор кадра -> `PIL.Image` (`frames -> pixels`).

Зачем отдельный кубик (шаг 7b плана `cube-architecture`): в diffusers «постпроцесс» — это вызов
`image_processor.postprocess(tensor, output_type=...)`, то есть **формат результата**. Декодер VAE
может отдавать не только изображение, но и временную последовательность, поэтому контейнер и
формат — отдельная стадия: здесь картинка, в будущем рядом появится `decoder.postprocess.video`
на том же порту `frames`, и декодер при этом не меняется.

    params = postprocess_image.ImagePostprocess.from_mapping({"output_type": "pil"})
    params.setup(pipe.image_processor)          # VaeImageProcessor
    pixels = params(frames)                      # -> Pixels(image=PIL.Image)

Важно про форму тензора: `VaeImageProcessor.postprocess` ждёт **4D** `B x C x H x W`, а декодер
отдаёт 5D `B x C x T x H x W` (у Qwen-Image VAE есть временная ось). Для одного кадра берётся
`[:, :, 0]` — ровно так делает блок декода diffusers (`QwenImage21DecodeStep`). Если `temporal > 1`,
это видео: кубик картинки обязан сказать об этом явно, а не молча взять первый кадр.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..backends.base import ParameterSpec, ResourceBudget
from ..cube import CubeContext, CubeSpec, Stage, register_cube
from ..ports import Frames, Pixels

__all__ = ["CUBE", "ImagePostprocess", "PostprocessParams"]

#: Форматы, которые понимает `VaeImageProcessor.postprocess`.
OUTPUT_TYPES = ("pil", "np")


@dataclass(frozen=True)
class PostprocessParams:
    """Разобранные параметры постпроцесса."""

    output_type: str = "pil"

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> "PostprocessParams":
        values = {p.name: p.default for p in CUBE.params}
        values.update({k: v for k, v in (params or {}).items() if k in values})
        return cls(output_type=str(values["output_type"]))


class ImagePostprocess:
    """Превращает тензор кадра в `PIL.Image`: параметры в конструкторе, работа — в `__call__`.

    `image_processor` (обычно `pipe.image_processor`) приходит через `setup()`: это компонент
    пайплайна, и кубик не должен его создавать сам — иначе нормализация разъедется с той, которой
    пользуется пайплайн (а она и определяет паритет кадра).
    """

    def __init__(self, params: PostprocessParams | None = None) -> None:
        self.params = params or PostprocessParams()
        self.image_processor: Any = None

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any] | None = None) -> "ImagePostprocess":
        return cls(PostprocessParams.from_mapping(params or {}))

    def setup(self, image_processor: Any) -> "ImagePostprocess":
        """Привязать `VaeImageProcessor` (или совместимый объект с `postprocess`)."""
        if image_processor is None or not hasattr(image_processor, "postprocess"):
            raise RuntimeError("decoder.postprocess.image: нужен image_processor с методом "
                               "postprocess (у пайплайна это pipe.image_processor)")
        self.image_processor = image_processor
        return self

    def __call__(self, frames: Frames) -> Pixels:
        if self.image_processor is None:
            raise RuntimeError("decoder.postprocess.image: сначала setup(image_processor) — "
                               "без него тензор не станет картинкой")
        x = frames.x
        if x is None:
            raise ValueError("decoder.postprocess.image: пустой порт frames (x=None)")
        if getattr(x, "ndim", 0) == 5:
            if int(frames.temporal) > 1:
                raise ValueError(
                    f"decoder.postprocess.image: frames содержат {frames.temporal} кадров "
                    f"(видео) — это задача decoder.postprocess.video, а не кубика картинки")
            x = x[:, :, 0]
        elif getattr(x, "ndim", 0) != 4:
            raise ValueError(f"decoder.postprocess.image: нужен тензор BxCxHxW или BxCxTxHxW, "
                             f"получено {tuple(getattr(x, 'shape', ()))!r}")
        image = self.image_processor.postprocess(x, output_type=self.params.output_type)[0]
        return Pixels(image=image, stage_trace=("decoder.postprocess.image",))


def _self_test(ctx: CubeContext) -> None:
    """Проверка без весов: 4D/5D, `output_type`, отказ на видео и без `setup`."""
    import torch

    class StubProcessor:
        def __init__(self) -> None:
            self.calls: list[tuple[tuple[int, ...], str]] = []

        def postprocess(self, image: Any, output_type: str = "pil") -> list[str]:
            self.calls.append((tuple(image.shape), output_type))
            return ["PIL"]

    processor = StubProcessor()
    cube = ImagePostprocess.from_mapping({"output_type": "pil"}).setup(processor)
    pixels = cube(Frames(x=torch.zeros(1, 3, 8, 8), size=(128, 128), temporal=1))
    assert isinstance(pixels, Pixels) and pixels.image == "PIL", pixels
    assert processor.calls == [((1, 3, 8, 8), "pil")], processor.calls
    assert pixels.stage_trace == ("decoder.postprocess.image",), pixels.stage_trace

    # 5D с T=1 — берётся единственный кадр
    cube(Frames(x=torch.zeros(1, 3, 1, 8, 8), size=(128, 128), temporal=1))
    assert processor.calls[-1][0] == (1, 3, 8, 8), processor.calls[-1]
    # output_type=np доходит до процессора
    ImagePostprocess.from_mapping({"output_type": "np"}).setup(processor)(
        Frames(x=torch.zeros(1, 3, 8, 8), temporal=1))
    assert processor.calls[-1][1] == "np", processor.calls[-1]
    # видео — это не задача кубика картинки
    try:
        cube(Frames(x=torch.zeros(1, 3, 4, 8, 8), temporal=4))
    except ValueError as exc:
        assert "video" in str(exc), exc
    else:                                                       # pragma: no cover
        raise AssertionError("видео обязано отвергаться кубиком картинки")
    # без setup — понятная ошибка, а не молчаливый тензор в порту pixels
    try:
        ImagePostprocess.from_mapping({})(Frames(x=torch.zeros(1, 3, 8, 8)))
    except RuntimeError as exc:
        assert "setup" in str(exc), exc
    else:                                                       # pragma: no cover
        raise AssertionError("без setup обязан быть RuntimeError")


CUBE: CubeSpec = register_cube(CubeSpec(
    id="decoder.postprocess.image",
    stage=Stage.DECODER,
    consumes=frozenset({"frames"}),
    produces="pixels",
    params=(
        ParameterSpec("output_type", "cube", "enum", "pil", choices=OUTPUT_TYPES,
                      doc="форма результата: pil = PIL.Image (обычный путь, дальше post.*); "
                          "np = numpy-массив (замеры/сравнения)"),
    ),
    budget=ResourceBudget(notes="чистая конвертация тензора в картинку: весов нет, "
                                "память ~3 байта на пиксель результата"),
    self_test=_self_test,
    doc="постпроцесс декодера: тензор кадра (frames) -> PIL.Image (pixels); видео — отдельный кубик",
))
