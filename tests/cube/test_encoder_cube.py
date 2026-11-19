"""Кубик `encoder.qwen3vl`: ключ conditioning-кэша и контракт (без torch и весов).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/unit/test_encoder_cube.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (план `cube-architecture`, шаг 4): ключ кэша эмбеддингов теперь строится из **портов**
(`Prompt.md5`, `RefImages.sha`, сетка) и параметров кубика, а не из `RunConfig`. Главное требование
— имя файла остаётся **байт-в-байт** прежним, иначе весь существующий кэш станет промахом.

Кейсы:

1. Дефолт (heretic/cuda/bf16) → `_te-heretic-dev-cuda`, имя как в `cache/embeds/`.
2. `base`/`cpu`/`bfloat16` → хвоста нет, историческое имя `embeds_<md5>_<sha>_WxH.pt`.
3. Неумолчания: `float16` → `-fp16`, `cuda` → `-dev-cuda`.
4. T2I (`refs=None`) → сегмент `noref` (как `io.images_sha` для пустого списка).
5. Совпадение с `RunConfig.embeds_key` на одинаковых входах (parity ключей).
6. Контракт кубика: стадия ENCODER, `consumes={"prompt"}`, `reads={"refs"}`, `produces="conditioning"`.
7. `load`/`save` на заглушке хранилища: пустой кэш -> None, запись -> чтение Conditioning.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from qwenstyle.config import RunConfig                                    # noqa: E402
from qwenstyle.cube import CubeContext, Stage, get_cube                    # noqa: E402
from qwenstyle.cubes import encoder_qwen3vl                                # noqa: E402
from qwenstyle.ports import Conditioning, Prompt, RefImages                # noqa: E402

MD5 = "9f7c48df7961"
SHA = "c531e6946c07"


class StubStore:
    """Минимальный сервис кэша: `get_name`/`put_name` в памяти."""

    def __init__(self) -> None:
        self.data: dict[str, dict] = {}

    def get_name(self, name: str):
        return self.data.get(name)

    def put_name(self, name: str, payload: dict, meta: dict):
        self.data[name] = payload
        return name


def case_default_key() -> None:
    params = encoder_qwen3vl.EncoderParams.from_mapping({})
    assert params.key_suffix == "_te-heretic-dev-cuda", params.key_suffix
    name = params.cache_name(MD5, SHA, (1024, 768))
    assert name == f"embeds_{MD5}_{SHA}_te-heretic-dev-cuda_1024x768.pt", name


def case_base_legacy_key() -> None:
    params = encoder_qwen3vl.EncoderParams.from_mapping(
        {"name": "base", "cache_tag": "base", "device": "cpu"})
    assert params.key_suffix == "", params.key_suffix
    assert params.cache_name(MD5, SHA, (1376, 768)) == f"embeds_{MD5}_{SHA}_1376x768.pt"


def case_non_default_segments() -> None:
    fp16 = encoder_qwen3vl.EncoderParams.from_mapping(
        {"name": "base", "cache_tag": "base", "dtype": "float16", "device": "cuda"})
    assert fp16.key_suffix == "-fp16-dev-cuda", fp16.key_suffix
    cpu_heretic = encoder_qwen3vl.EncoderParams.from_mapping({"device": "cpu"})
    assert cpu_heretic.key_suffix == "_te-heretic", cpu_heretic.key_suffix


def case_t2i_noref() -> None:
    params = encoder_qwen3vl.EncoderParams.from_mapping({})
    name = encoder_qwen3vl.conditioning_key(params, Prompt(text="x", md5=MD5), None,
                                            grid=(1024, 768))
    assert name == f"embeds_{MD5}_noref_te-heretic-dev-cuda_1024x768.pt", name


def case_parity_with_runconfig() -> None:
    """Ключ кубика обязан совпасть с `RunConfig.embeds_key` для тех же входов."""
    for preset in ("base", "VRAM16-preview-anime", "VRAM12-release"):
        cfg = RunConfig.from_profile(preset)
        enc = cfg.encoder
        params = encoder_qwen3vl.EncoderParams.from_mapping({
            "name": enc.name, "cache_tag": enc.cache_tag, "dtype": enc.dtype,
            "device": enc.device,
        })
        for grid in ((1024, 768), (1376, 768), (2912, 2176)):
            expected = cfg.embeds_key(grid, MD5, SHA)
            got = params.cache_name(MD5, SHA, grid)
            assert got == expected, f"{preset} {grid}: {got} != {expected}"


def case_contract() -> None:
    cube = get_cube("encoder.qwen3vl")
    assert cube.stage is Stage.ENCODER, cube.stage
    assert cube.consumes == frozenset({"prompt"}), cube.consumes
    assert cube.reads == frozenset({"refs"}), cube.reads
    assert cube.produces == "conditioning", cube.produces
    names = [p.name for p in cube.params]
    assert names[:3] == ["name", "cache_tag", "repo"], names
    assert cube.self_test is not None
    cube.self_test(CubeContext())


def case_store_roundtrip() -> None:
    store = StubStore()
    params = encoder_qwen3vl.EncoderParams.from_mapping({})
    prompt = Prompt(text="x", md5=MD5)
    refs = RefImages(images=("img",), grid=(1024, 768), sha=SHA)
    name = encoder_qwen3vl.conditioning_key(params, prompt, refs, grid=(1024, 768))
    assert encoder_qwen3vl.load(store, name) is None, "пустой кэш обязан давать None"
    cond = Conditioning(embeds="E", mask="M", img_mask="I", ref_grid=(1024, 768))
    encoder_qwen3vl.save(store, name, cond, {"prompt_md5": MD5})
    back = encoder_qwen3vl.load(store, name)
    assert isinstance(back, Conditioning), type(back)
    assert (back.embeds, back.mask, back.img_mask) == ("E", "M", "I"), back
    assert back.ref_grid == (1024, 768), back.ref_grid


CASES = [
    ("дефолтный ключ (heretic/cuda)", case_default_key),
    ("легаси-ключ base/cpu", case_base_legacy_key),
    ("сегменты fp16/dev-cuda", case_non_default_segments),
    ("T2I -> noref", case_t2i_noref),
    ("parity с RunConfig.embeds_key", case_parity_with_runconfig),
    ("контракт кубика", case_contract),
    ("load/save на заглушке", case_store_roundtrip),
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
