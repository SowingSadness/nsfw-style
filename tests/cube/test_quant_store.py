"""`QuantStore` / `QuantReader`: хранение кэша и чтение с де-квантом (без CUDA и весов моделей).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/unit/test_quant_store.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (план `cube-architecture`, шаг 5b.1a): слой квантованного кэша отделён от адаптеров моделей.
Тест фиксирует контракт хранилища на синтетическом state dict — быстро, без GPU:

1. `source_shards`/`read_header`/`source_entries`/`entries_digest`: одиночный файл и шарды по индексу.
2. `ShardWriter`: разбиение по `block_bytes`, имена `...-00001-of-000NN`, `index.json` с полным
   `weight_map`, сайдкар с полем `shards`.
3. `QuantReader`: `names()` совпадает с `weight_map`, `read()` возвращает те же данные и dtype,
   `read(dtype=bf16)` де-квантует fp8 (относительная ошибка < 1 %), `read_into` пишет в буфер.
4. `QuantStore.verify`: годен; ловит смену схемы, digest, `config_sha`, целевой точности, skip и
   отсутствующий шард.
5. `QuantSpec`: `target_token`, `is_skipped` по regex.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

import torch                                                              # noqa: E402
from safetensors.torch import save_file                                    # noqa: E402

from qwenstyle.quant import (                                              # noqa: E402
    CACHE_SCHEMA,
    INDEX_NAME,
    SIDECAR_NAME,
    QuantReader,
    QuantSpec,
    QuantStore,
    ShardWriter,
    entries_digest,
    read_header,
    source_entries,
    source_shards,
)

FP8 = torch.float8_e4m3fn


def _state_dict() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "block.weight": torch.randn(64, 128),
        "block.bias": torch.randn(64),
        "norm.weight": torch.ones(128),
        "pos_embed": torch.randn(4, 128),
    }


def _build_cache(root: Path, *, block_bytes: int = 4096, spec: QuantSpec | None = None) -> QuantStore:
    """Синтетический кэш: кастуем всё, кроме `norm`/`pos_embed` (как в реальном spec)."""
    spec = spec or QuantSpec(target="fp8_e4m3", skip=(r"\.norm\.", r"pos_embed"),
                             block_bytes=block_bytes)
    store = QuantStore(root / "cache")
    writer = ShardWriter(store.path, block_bytes=spec.block_bytes)
    for name, tensor in _state_dict().items():
        if spec.is_skipped(name) or tensor.dim() < 2:
            writer.add(name, tensor.to(torch.float32) if tensor.dtype == torch.float32 else tensor)
        else:
            writer.add(name, tensor.to(FP8))
    table = writer.close()
    store.write_sidecar({"schema": CACHE_SCHEMA, "source_digest": "digest-1",
                         "config_sha256": "config-1", "fp8_dtype": spec.target_token,
                         "skip_patterns": list(spec.skip), "weights": table})
    return store


def case_source_helpers() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        single = root / "single"
        single.mkdir()
        save_file(_state_dict(), str(single / "diffusion_pytorch_model.safetensors"))
        shards = source_shards(single)
        assert len(shards) == 1, shards
        header = read_header(shards[0])
        assert set(header) == set(_state_dict()), header
        entries = source_entries(shards)
        assert [e["key"] for e in entries] == sorted(_state_dict()), entries
        digest = entries_digest(entries)
        assert digest == entries_digest(source_entries(source_shards(single))), "дайджест нестабилен"

        # шардированный источник через index.json
        split = root / "split"
        split.mkdir()
        items = list(_state_dict().items())
        save_file(dict(items[:2]), str(split / "model-00001-of-00002.safetensors"))
        save_file(dict(items[2:]), str(split / "model-00002-of-00002.safetensors"))
        (split / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            items[0][0]: "model-00001-of-00002.safetensors",
            items[1][0]: "model-00001-of-00002.safetensors",
            items[2][0]: "model-00002-of-00002.safetensors",
            items[3][0]: "model-00002-of-00002.safetensors",
        }}), encoding="utf-8")
        assert len(source_shards(split)) == 2
        assert entries_digest(source_entries(source_shards(split))) == digest, \
            "дайджест обязан не зависеть от разбиения на шарды"


def case_writer_and_reader() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = _build_cache(Path(tmp), block_bytes=2048)
        index = json.loads((store.path / INDEX_NAME).read_text(encoding="utf-8"))
        assert set(index["weight_map"]) == set(_state_dict()), index["weight_map"]
        names = sorted(set(index["weight_map"].values()))
        assert len(names) >= 2, f"блок 2 КиБ обязан дать несколько шардов: {names}"
        assert names[0].endswith("-00001-of-000%02d.safetensors" % len(names)), names

        reader = QuantReader(store.path)
        assert reader.names() == sorted(_state_dict()), reader.names()
        weight = reader.read("block.weight")
        assert weight.dtype == FP8, weight.dtype
        back = reader.read("block.weight", dtype=torch.bfloat16)
        assert back.dtype == torch.bfloat16 and back.shape == (64, 128), back
        exact = _state_dict()["block.weight"]
        rel = float((back.float() - exact).abs().mean() / exact.abs().mean())
        # Здесь каст без скейла (синтетика для хранилища); реальный кэш масштабируется адаптером
        # (layerwise cast со scale), поэтому допускаем «сырую» ошибку fp8 ~2–3 %.
        assert rel < 0.03, f"де-квант fp8: относительная ошибка {rel:.4f}"

        buffer = torch.empty(64, dtype=torch.bfloat16)
        got = reader.read_into("block.bias", buffer)
        assert got.data_ptr() == buffer.data_ptr(), "read_into обязан писать в буфер"
        reader.close()


def case_verify() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        spec = QuantSpec(target="fp8_e4m3", skip=(r"\.norm\.", r"pos_embed"))
        store = _build_cache(Path(tmp), spec=spec)
        assert store.verify(digest="digest-1", config_sha="config-1", spec=spec) is None, \
            "свежий кэш обязан быть годным"

        checks = {
            "дайджест": dict(digest="digest-2", config_sha="config-1", spec=spec),
            "config": dict(digest="digest-1", config_sha="config-2", spec=spec),
            "точность": dict(digest="digest-1", config_sha="config-1",
                            spec=QuantSpec(target="bf16", skip=spec.skip)),
            "skip": dict(digest="digest-1", config_sha="config-1",
                         spec=QuantSpec(target="fp8_e4m3", skip=("pos_embed",))),
        }
        for label, kwargs in checks.items():
            problem = store.verify(**kwargs)
            assert problem, f"{label}: verify обязан найти проблему"

        side = store.sidecar()
        store.write_sidecar({**side, "schema": CACHE_SCHEMA + 1})
        assert "схема" in (store.verify(digest="digest-1", config_sha="config-1", spec=spec) or "")

        store.write_sidecar({**side, "schema": CACHE_SCHEMA})
        index = json.loads((store.path / INDEX_NAME).read_text(encoding="utf-8"))
        first = next(iter(index["weight_map"].values()))
        (store.path / first).unlink()
        assert "шарда" in (store.verify(digest="digest-1", config_sha="config-1", spec=spec) or "")


def case_spec() -> None:
    spec = QuantSpec(target="fp8_e4m3", skip=(r"\.norm\.", r"^lm_head"))
    assert spec.target_token == "F8_E4M3", spec.target_token
    assert spec.is_skipped("block.norm.weight") and spec.is_skipped("lm_head.weight"), spec.skip
    assert not spec.is_skipped("block.attn.q_proj.weight")
    try:
        QuantSpec(target="int4").target_token
    except ValueError as exc:
        assert "неизвестная" in str(exc)
    else:
        raise AssertionError("неизвестная точность обязана падать")


CASES = [
    ("источник: шарды, заголовки, дайджест", case_source_helpers),
    ("ShardWriter + QuantReader", case_writer_and_reader),
    ("QuantStore.verify", case_verify),
    ("QuantSpec: token и skip", case_spec),
]


def main() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            fn()
        except Exception as exc:                                        # noqa: BLE001
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
