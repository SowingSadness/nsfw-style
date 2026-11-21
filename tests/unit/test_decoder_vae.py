"""Кубик `decoder.vae.qwen21`: параметры тайлинга и self_test без весов.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/unit/test_decoder_vae.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (план `cube-architecture`, шаг 2): настройки VAE переехали из `qwen_image21.prepare()` в
кубик со своей спекой. Тест фиксирует семантику замеров из `AGENTS.md`:

1. `tile_px=512` (умолчание) → `enable_tiling(512/512/384/384)`, `slicing` включён.
2. `tile_px=0` → историческое поведение diffusers (`enable_tiling()` без аргументов = 256/192).
3. `tile_px=-1` → `use_tiling=False`, без вызова `enable_tiling`.
4. `tile_px=N` → шаг `0.75·N` (доля перекрытия как у diffusers).
5. `tiling=False` → тайлинг выключен независимо от `tile_px`.
6. Дефолты берутся из спеки кубика, неизвестные ключи игнорируются.
7. Контракт кубика: порт `latents` -> `frames` (тензор), декодер объявлен на стадии DECODER.
8. `Qwen21Decoder` — параметры в конструкторе, `setup(vae)` применяет тайлинг, `__call__` декодирует
   и отдаёт `Frames` (размер в пикселях, `temporal`); модульные `setup()/decode()` дают тот же тензор.
9. Постпроцесс (`PIL`) — **не** здесь: `decode()` больше не принимает `image_processor` (это кубик
   `decoder.postprocess.image`), иначе стадии разъезжаются.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from qwenstyle.cube import CubeContext, Stage, get_cube            # noqa: E402
from qwenstyle.cubes import decoder_vae                            # noqa: E402
from qwenstyle.ports import Frames                                 # noqa: E402


class StubVAE:
    """Заглушка diffusers-VAE: запоминает вызовы, ничего не загружает."""

    use_tiling = True

    def __init__(self) -> None:
        self.slicing = False
        self.tiling_calls: list[dict] = []

    def enable_slicing(self) -> None:
        self.slicing = True

    def enable_tiling(self, **kwargs) -> None:
        self.tiling_calls.append(kwargs)


def _setup(**params):
    vae = StubVAE()
    parsed = decoder_vae.DecodeParams.from_mapping(params)
    decoder_vae.setup(vae, parsed)
    return vae, parsed


def case_default_tile() -> None:
    vae, parsed = _setup()
    assert parsed.tile_px == 512, parsed.tile_px
    assert parsed.stride_px == 384, parsed.stride_px
    assert vae.slicing, "slicing обязан включаться"
    assert vae.tiling_calls == [{"tile_sample_min_height": 512, "tile_sample_min_width": 512,
                                 "tile_sample_stride_height": 384,
                                 "tile_sample_stride_width": 384}], vae.tiling_calls


def case_legacy_tile() -> None:
    vae, _ = _setup(tile_px=0)
    assert vae.tiling_calls == [{}], "tile_px=0 = дефолты diffusers (256/192)"


def case_tiling_off() -> None:
    vae, parsed = _setup(tile_px=-1)
    assert parsed.tile_px == -1
    assert vae.use_tiling is False, "tile_px=-1 обязан выключить тайлинг"
    assert vae.tiling_calls == [], "enable_tiling не должен вызываться"


def case_stride_ratio() -> None:
    _, parsed = _setup(tile_px=1024)
    assert parsed.stride_px == 768, parsed.stride_px          # 0.75 * 1024
    vae, _ = _setup(tile_px=256)
    assert vae.tiling_calls[0]["tile_sample_stride_height"] == 192


def case_flag_off() -> None:
    vae, _ = _setup(tile_px=512, tiling=False)
    assert vae.use_tiling is False, "tiling=False обязан выключить тайлинг даже при tile_px>0"


def case_defaults_and_unknown() -> None:
    parsed = decoder_vae.DecodeParams.from_mapping({"nope": 1})
    assert parsed == decoder_vae.DecodeParams(), parsed
    assert parsed.dtype == "bfloat16" and parsed.device == "auto"


def case_decode_denormalizes_like_pipeline() -> None:
    """`decode` повторяет блок декода diffusers: денормализация + срез временной оси.

    Заглушка VAE воспроизводит контракт настоящего: `config.latents_mean`/`latents_std`,
    `decode(x, return_dict=False)` возвращает `(B, C, T, H, W)`.
    """
    import torch

    class StubConfig:
        z_dim = 2
        latents_mean = (1.0, 2.0)
        latents_std = (2.0, 3.0)

    class StubVAE:
        dtype = torch.float32
        config = StubConfig()

        def __init__(self) -> None:
            self.seen = None

        def decode(self, x, return_dict: bool = True):
            self.seen = x.detach().clone()
            return (x,)

    vae = StubVAE()
    # (B, C, H, W) -> ось T=1 добавляется внутри; денормализация: x*std + mean
    latents = decoder_vae.Latents(x=torch.ones(1, 2, 1, 1))
    frames = decoder_vae.decode(vae, latents)
    assert vae.seen.shape == (1, 2, 1, 1, 1), vae.seen.shape
    assert torch.allclose(vae.seen, torch.tensor([[[[[3.0]]], [[[5.0]]]]])), vae.seen
    assert frames.x.shape == (1, 2, 1, 1, 1), frames.x.shape      # тензор, а не PIL
    assert frames.temporal == 1, frames.temporal
    assert frames.size == (1, 1), frames.size                     # VAE отдал 1x1 пиксель
    assert frames.source == "decoder.vae.qwen21", frames.source


def case_decoder_class_api() -> None:
    """`Qwen21Decoder`: параметры в конструкторе, `setup(vae)` + `__call__(latents)` -> `Frames`."""
    import torch

    class StubVAE:
        dtype = torch.float32

        def __init__(self) -> None:
            self.kwargs = None
            self.slicing = False
            self.seen = None

        def enable_slicing(self) -> None:
            self.slicing = True

        def enable_tiling(self, **kwargs) -> None:
            self.kwargs = kwargs

        def decode(self, x, return_dict: bool = True):
            self.seen = x.detach().clone()
            return (x,)

    vae = StubVAE()
    dec = decoder_vae.Qwen21Decoder.from_mapping({"tile_px": 1024})
    assert dec.params.tile_px == 1024 and dec.params.stride_px == 768, dec.params
    # без setup — понятная ошибка, а не «декод неизвестно чем»
    try:
        dec(decoder_vae.Latents(x=torch.zeros(1, 2, 1, 4, 4)))
    except RuntimeError as exc:
        assert "setup" in str(exc), exc
    else:                                                          # pragma: no cover
        raise AssertionError("без setup обязан быть RuntimeError")
    dec.setup(vae)
    assert vae.slicing, "setup обязан включить slicing"
    assert vae.kwargs == {"tile_sample_min_height": 1024, "tile_sample_min_width": 1024,
                          "tile_sample_stride_height": 768, "tile_sample_stride_width": 768},         vae.kwargs
    frames = dec(decoder_vae.Latents(x=torch.zeros(1, 2, 1, 4, 4)))
    assert isinstance(frames, decoder_vae.Frames), type(frames)
    assert frames.size == (4, 4), frames.size
    # модульная функция даёт тот же тензор, что класс (один код)
    via_function = decoder_vae.decode(vae, decoder_vae.Latents(x=torch.zeros(1, 2, 1, 4, 4)),
                                      dec.params)
    assert torch.equal(frames.x, via_function.x), "класс и функция разошлись"


def case_decode_rejects_image_processor() -> None:
    """Постпроцесс — отдельный кубик: `decode()` не должен тайком делать PIL."""
    import torch

    class StubVAE:
        dtype = torch.float32

        def decode(self, x, return_dict: bool = True):
            return (x,)

    try:
        decoder_vae.decode(StubVAE(), decoder_vae.Latents(x=torch.zeros(1, 2, 1, 4, 4)),
                           image_processor=object())
    except TypeError:
        pass                                   # параметра больше нет — тоже приемлемый отказ
    else:                                                          # pragma: no cover
        raise AssertionError("decode() не должен принимать image_processor")


def case_contract() -> None:
    cube = get_cube("decoder.vae.qwen21")
    assert cube.stage is Stage.DECODER, cube.stage
    assert cube.consumes == frozenset({"latents"}), cube.consumes
    assert cube.outputs == frozenset({"frames"}), cube.outputs
    assert [p.name for p in cube.params] == ["tile_px", "tiling", "slicing", "dtype", "device"]
    assert cube.self_test is not None
    cube.self_test(CubeContext())                               # self_test проходит без весов


CASES = [
    ("tile_px=512 по умолчанию", case_default_tile),
    ("tile_px=0 = историческое 256/192", case_legacy_tile),
    ("tile_px=-1 выключает тайлинг", case_tiling_off),
    ("шаг = 0.75 * тайл", case_stride_ratio),
    ("tiling=False", case_flag_off),
    ("дефолты и неизвестные ключи", case_defaults_and_unknown),
    ("decode: денормализация и Frames", case_decode_denormalizes_like_pipeline),
    ("класс Qwen21Decoder: setup + __call__", case_decoder_class_api),
    ("decode не делает постпроцесс", case_decode_rejects_image_processor),
    ("контракт кубика", case_contract),
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
