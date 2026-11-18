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
7. Контракт кубика: порт `latents` -> `pixels`, декодер объявлен на стадии DECODER.
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


def case_contract() -> None:
    cube = get_cube("decoder.vae.qwen21")
    assert cube.stage is Stage.DECODER, cube.stage
    assert cube.consumes == frozenset({"latents"}), cube.consumes
    assert cube.produces == "pixels", cube.produces
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
