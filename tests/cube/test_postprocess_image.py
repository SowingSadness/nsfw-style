"""Кубик `decoder.postprocess.image`: тензор кадра -> `PIL.Image` (без весов, без CUDA).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/cube/test_postprocess_image.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (шаг 7b плана `cube-architecture`): постпроцесс diffusers (`image_processor.postprocess`) вынесен
из декодера в отдельный кубик, потому что это **формат результата**, а не декодирование: декодер VAE
может отдавать и последовательность кадров. Тест фиксирует контракт стадии:

1. 4D `B×C×H×W` уходит в процессор как есть, `output_type` из параметров (`pil`/`np`);
2. 5D `B×C×T×H×W` с `temporal=1` — берётся единственный кадр (`[:, :, 0]`, как в diffusers);
3. 5D с `temporal>1` — это видео: кубик картинки обязан отказать с указанием на `postprocess.video`;
4. без `setup(image_processor)` — `RuntimeError` (в порту `pixels` не может оказаться тензор);
5. `stage_trace` пишется в результат, кубик стоит на стадии DECODER и потребляет `frames`.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

import torch                                                          # noqa: E402

from qwenstyle.cube import CubeContext, Stage, get_cube               # noqa: E402
from qwenstyle.cubes import postprocess_image                          # noqa: E402
from qwenstyle.ports import PORT_TYPES, Frames, Pixels                 # noqa: E402


class StubProcessor:
    """`VaeImageProcessor.postprocess` в миниатюре: запоминает форму и `output_type`."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], str]] = []

    def postprocess(self, image, output_type: str = "pil"):
        self.calls.append((tuple(image.shape), output_type))
        return [f"image-{output_type}"]


def case_four_dim() -> None:
    processor = StubProcessor()
    cube = postprocess_image.ImagePostprocess.from_mapping({}).setup(processor)
    pixels = cube(Frames(x=torch.zeros(1, 3, 8, 8), size=(128, 128)))
    assert isinstance(pixels, Pixels), type(pixels)
    assert pixels.image == "image-pil", pixels.image
    assert processor.calls == [((1, 3, 8, 8), "pil")], processor.calls
    assert pixels.stage_trace == ("decoder.postprocess.image",), pixels.stage_trace


def case_five_dim_single_frame() -> None:
    processor = StubProcessor()
    cube = postprocess_image.ImagePostprocess.from_mapping({}).setup(processor)
    cube(Frames(x=torch.zeros(1, 3, 1, 8, 8), size=(128, 128), temporal=1))
    assert processor.calls[-1][0] == (1, 3, 8, 8), processor.calls[-1]


def case_output_type_np() -> None:
    processor = StubProcessor()
    cube = postprocess_image.ImagePostprocess.from_mapping({"output_type": "np"}).setup(processor)
    pixels = cube(Frames(x=torch.zeros(1, 3, 8, 8)))
    assert pixels.image == "image-np", pixels.image
    assert processor.calls[-1][1] == "np", processor.calls[-1]


def case_video_is_rejected() -> None:
    cube = postprocess_image.ImagePostprocess.from_mapping({}).setup(StubProcessor())
    try:
        cube(Frames(x=torch.zeros(1, 3, 4, 8, 8), temporal=4))
    except ValueError as exc:
        assert "video" in str(exc), exc
        assert "decoder.postprocess.video" in str(exc), exc
    else:                                                       # pragma: no cover
        raise AssertionError("видео обязано отвергаться кубиком картинки")


def case_requires_setup() -> None:
    cube = postprocess_image.ImagePostprocess.from_mapping({})
    try:
        cube(Frames(x=torch.zeros(1, 3, 8, 8)))
    except RuntimeError as exc:
        assert "setup" in str(exc), exc
    else:                                                       # pragma: no cover
        raise AssertionError("без setup обязан быть RuntimeError")
    try:
        cube.setup(None)
    except RuntimeError as exc:
        assert "image_processor" in str(exc), exc
    else:                                                       # pragma: no cover
        raise AssertionError("setup(None) обязан быть RuntimeError")


def case_bad_tensor() -> None:
    cube = postprocess_image.ImagePostprocess.from_mapping({}).setup(StubProcessor())
    for frames in (Frames(x=torch.zeros(1, 3, 8)), Frames(x=None)):
        try:
            cube(frames)
        except ValueError:
            pass
        else:                                                   # pragma: no cover
            raise AssertionError(f"плохой тензор обязан отвергаться: {frames}")


def case_contract() -> None:
    assert "frames" in PORT_TYPES and PORT_TYPES["frames"] is Frames, PORT_TYPES.keys()
    cube = get_cube("decoder.postprocess.image")
    assert cube.stage is Stage.DECODER, cube.stage
    assert cube.consumes == frozenset({"frames"}), cube.consumes
    assert cube.outputs == frozenset({"pixels"}), cube.outputs
    assert [p.name for p in cube.params] == ["output_type"], [p.name for p in cube.params]
    assert cube.param("output_type").choices == postprocess_image.OUTPUT_TYPES
    assert cube.self_test is not None
    cube.self_test(CubeContext())


CASES = [
    ("4D тензор -> PIL", case_four_dim),
    ("5D с T=1 -> единственный кадр", case_five_dim_single_frame),
    ("output_type=np", case_output_type_np),
    ("видео отвергается", case_video_is_rejected),
    ("нужен setup(image_processor)", case_requires_setup),
    ("плохой тензор отвергается", case_bad_tensor),
    ("контракт кубика и порта", case_contract),
]


def main() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            fn()
        except Exception as exc:                                    # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    if failed:
        print(f"\n{failed} из {len(CASES)} кейсов провалено")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
