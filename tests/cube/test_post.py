"""Пост-кубики: ESRGAN, colorfix, SeedVR2 (без весов, без CUDA).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/cube/test_post.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (план `cube-architecture`, шаг 6): постобработка была размазана по `pipeline.py` и бекендам,
и ошибки в ней дорогие — «апскейл не тот множитель» замечается только глазами на 4K-кадре.
Тест фиксирует чистую арифметику и правила пропуска:

1. `post.upscale.esrgan`: `target_height` главнее `factor`; кадр уже не ниже цели -> пропуск (0.0);
   `factor=0` без цели -> выключено; тайлы по умолчанию 256/32 («бит-в-бит» к прежним прогонам);
2. `post.colorfix`: `mode=none` и отсутствие источника — тождественны (тот же объект кадра);
   `wavelet`/`reinhard` меняют картинку и сохраняют размер;
3. `post.upscale.seedvr2`: размеры кратны 16, явные `width`/`height` главнее `factor`, `fp8` включён;
4. у всех трёх кубиков `self_test` проходит на синтетике (это часть контракта).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from qwenstyle.cube import CubeContext, Stage, get_cube                # noqa: E402
from qwenstyle.cubes import post_colorfix, post_esrgan, post_seedvr2   # noqa: E402
from qwenstyle.ports import Pixels                                     # noqa: E402

CTX = CubeContext()


def image(width: int, height: int, color: tuple[int, int, int]):
    from PIL import Image

    return Image.new("RGB", (width, height), color)


def case_esrgan_target_height_wins() -> None:
    params = post_esrgan.EsrganParams.from_mapping({"target_height": 1440, "factor": 2.0})
    assert abs(post_esrgan.resolve_factor(1088, params) - 1440 / 1088) < 1e-9
    assert post_esrgan.plan_factor(1088, params) > 1.0


def case_esrgan_skip() -> None:
    params = post_esrgan.EsrganParams.from_mapping({"target_height": 1440})
    assert post_esrgan.plan_factor(1440, params) == 0.0, "кадр равен цели — пропуск"
    assert post_esrgan.plan_factor(2176, params) == 0.0, "кадр выше цели — пропуск"
    # фактор ~1 (1441/1440) тоже пропуск, иначе суффикс -x1 в имени файла
    assert post_esrgan.plan_factor(1441, params) == 0.0
    # пропуск не грузит сеть и возвращает тот же кадр (цель = высота кадра)
    pixels = Pixels(image=image(64, 32, (10, 20, 30)))
    out = post_esrgan.upscale(pixels, post_esrgan.EsrganParams.from_mapping({"target_height": 32}))
    assert out.image is pixels.image, "пропуск обязан вернуть тот же объект"


def case_esrgan_disabled_and_tiles() -> None:
    assert post_esrgan.plan_factor(480, post_esrgan.EsrganParams.from_mapping({})) == 0.0
    default = post_esrgan.EsrganParams.from_mapping({})
    assert (default.tile, default.pad) == (256, 32), (default.tile, default.pad)
    explicit = post_esrgan.EsrganParams.from_mapping({"tile": 128, "pad": 32, "factor": 4.0})
    assert (explicit.tile, explicit.pad) == (128, 32)
    assert post_esrgan.plan_factor(480, explicit) == 4.0


def case_colorfix_identity() -> None:
    plain = image(8, 8, (10, 20, 30))
    pixels = Pixels(image=plain)
    assert post_colorfix.apply(pixels, post_colorfix.ColorfixParams.from_mapping(
        {"mode": "none"}), source=plain).image is plain
    assert post_colorfix.apply(pixels, post_colorfix.ColorfixParams.from_mapping({}),
                               source=None).image is plain


def case_colorfix_changes_frame() -> None:
    gray = image(8, 8, (40, 40, 40))
    bright = image(8, 8, (200, 200, 200))
    for mode in ("wavelet", "reinhard"):
        params = post_colorfix.ColorfixParams.from_mapping({"mode": mode})
        out = post_colorfix.apply(Pixels(image=gray), params, source=bright, log=CTX.log)
        assert out.image.size == gray.size, out.image.size
        assert out.image.getpixel((4, 4))[0] > 40, (mode, out.image.getpixel((4, 4)))
        assert out.stage_trace[-1] == "post.colorfix", out.stage_trace
    # strength=0 — «как было»
    zero = post_colorfix.ColorfixParams.from_mapping({"strength": 0.0})
    out = post_colorfix.apply(Pixels(image=gray), zero, source=bright, log=CTX.log)
    assert out.image.getpixel((4, 4))[0] == 40, out.image.getpixel((4, 4))


def case_seedvr2_target_size() -> None:
    assert post_seedvr2.target_size((864, 480), post_seedvr2.Seedvr2Params.from_mapping(
        {"factor": 2.0})) == (1728, 960)
    assert post_seedvr2.target_size((864, 480), post_seedvr2.Seedvr2Params.from_mapping(
        {"factor": 4.0})) == (3456, 1920)
    explicit = post_seedvr2.Seedvr2Params.from_mapping({"width": 1728, "height": 961})
    assert post_seedvr2.target_size((864, 480), explicit) == (1728, 960), "обязано стать кратным 16"
    default = post_seedvr2.Seedvr2Params.from_mapping({})
    assert default.fp8 is True, "bf16-веса 12.6 ГиБ на 16 ГиБ не влезают"
    assert post_seedvr2.target_size((640, 480), default) == (1280, 960), "дефолт — x2"


def case_cube_contracts() -> None:
    for cube_id, module in (("post.upscale.esrgan", post_esrgan),
                            ("post.colorfix", post_colorfix),
                            ("post.upscale.seedvr2", post_seedvr2)):
        cube = get_cube(cube_id)
        assert cube.stage is Stage.POST, cube.stage
        assert cube.consumes == frozenset({"pixels"}), cube.consumes
        assert cube.outputs == frozenset({"pixels"}), cube.outputs
        assert cube.self_test is not None
        cube.self_test(CTX)
        assert module.CUBE is cube, cube_id


CASES = [
    ("esrgan: target_height главнее factor", case_esrgan_target_height_wins),
    ("esrgan: пропуск без загрузки сети", case_esrgan_skip),
    ("esrgan: выключен и тайлы", case_esrgan_disabled_and_tiles),
    ("colorfix: тождественность", case_colorfix_identity),
    ("colorfix: меняет кадр", case_colorfix_changes_frame),
    ("seedvr2: целевой размер", case_seedvr2_target_size),
    ("контракты пост-кубиков", case_cube_contracts),
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
