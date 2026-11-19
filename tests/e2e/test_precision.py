"""Тест `qwenstyle/precision.py` — fp8-каст и дисковый кэш трансформера. Обычный скрипт, без pytest.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_precision.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Кейсы (по плану шага 3):

1. мини-`MiniMaxH3ImageTransformer2DModel` (hidden 128, 2 слоя, 4 головы) в **смешанной точности**,
   как реальный чекпойнт (`proj_in`, `time_embedder`, `pdd_proj_out` — float32, остальное bf16),
   сохраняется `save_pretrained` -> `cast_and_cache`;
2. в кэше появляются `*.safetensors` + индекс + `config.json` + `precision.json` +
   `non_persistent.buffers.pt`; сайдкар несёт все параметры каста (R1);
3. повторный вызов `cast_and_cache` — reuse: mtime всех файлов не меняется, каста нет;
4. `load_cached`: fp8-тензоры в `float8_e4m3fn`, точные — в исходном dtype (bf16/f32) и **бит-в-бит**
   равны исходной модели; буфер `rope.inv_freq` восстановлен из сайдкара и `torch.equal` исходному;
5. forward на загруженной fp8-модели с `enable_layerwise_casting` (как это делает бекенд) даёт
   конечный тензор нужной формы; для контроля — тот же forward у исходной bf16-модели.

Модель крошечная, поэтому тест идёт секунды и не читает веса проекта. Временные каталоги — только
в системном temp (`tempfile.TemporaryDirectory`), в репозитории не остаётся ничего.
"""
from __future__ import annotations

import json
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:              # запуск из любого cwd
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                  # тест не должен оставлять __pycache__ в пакете

from qwenstyle.precision import (                # noqa: E402 — после правки sys.path
    BUFFERS_NAME,
    CACHE_SCHEMA,
    INDEX_NAME,
    SIDECAR_NAME,
    cast_and_cache,
    load_cached,
)

#: Мини-конфиг: hidden 128 = 2 головы x 64, 2 блока, 1 refiner, 6 голов выхода (как pdd).
#: `attention_head_dim` обязан быть ≥ `2 * 3 * rope_freq_dim` (RoPE крутит 48 каналов из 64).
TINY = dict(
    num_attention_heads=2,
    attention_head_dim=64,
    hidden_size=128,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=256,
    in_channels=4,
    patch_size=(1, 2, 2),
    text_dim=32,
    freq_dim=32,
    time_embed_hidden_dim=128,
    time_embed_dim=64,
    rope_freq_dim=8,
    num_pdd_heads=6,
)
#: Модули, которые реальный чекпойнт держит в float32 (`_keep_in_fp32_modules`).
FLOAT32_MODULES = ("proj_in", "time_embedder", "pdd_proj_out", "rope")


def temp_dir() -> tempfile.TemporaryDirectory:
    """Системный temp из `tempfile` (под песочницей без прав на запись вне workspace он = cwd).

    Своих кандидатов не перебираем: под песочницей `mkdtemp` по недоступному пути не падает, а
    **зависает** (проверено), поэтому единственный безопасный выбор — штатный `gettempdir()`.
    Каталог в любом случае удаляется контекстным менеджером.
    """
    return tempfile.TemporaryDirectory(prefix="precision-test-")


# --------------------------------------------------------------------------- #
# Фикстуры
# --------------------------------------------------------------------------- #

def tiny_model(torch: Any) -> Any:
    """Мини-модель в смешанной точности: bf16 + float32 у `_keep_in_fp32_modules`."""
    from diffusers.models import MiniMaxH3ImageTransformer2DModel

    torch.manual_seed(0)
    model = MiniMaxH3ImageTransformer2DModel(**TINY)
    # `.to(bfloat16)` округлил бы буфер `rope.inv_freq` (он float32 по контракту модели) — сохраняем
    # точное значение, как это делает `from_pretrained(torch_dtype=bf16)` с `_keep_in_fp32_modules`.
    inv_freq = model.rope.inv_freq.clone()
    model.to(torch.bfloat16)
    for name in FLOAT32_MODULES:
        getattr(model, name).to(torch.float32)
    model.rope.inv_freq = inv_freq
    model.eval()
    return model


def forward_inputs(torch: Any, model: Any) -> dict[str, Any]:
    """Один валидный вход `forward`: короткая упакованная последовательность (текст + 8 image-строк)."""
    generator = torch.Generator().manual_seed(11)
    patch_dim = TINY["in_channels"] * TINY["patch_size"][0] * TINY["patch_size"][1] * TINY["patch_size"][2]
    num_text, num_image, timesteps = 3, 8, 2
    seq = num_text + num_image
    timestep_indices = torch.cat([
        torch.zeros(num_text, dtype=torch.long),
        torch.ones(num_image, dtype=torch.long),
    ])
    token_tags = torch.cat([
        torch.ones(num_text, dtype=torch.long),
        torch.zeros(num_image, dtype=torch.long),
    ])
    position_ids = torch.arange(seq, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
    return dict(
        hidden_states=torch.randn(1, num_image, patch_dim, generator=generator, dtype=torch.float32),
        encoder_hidden_states=torch.randn(1, num_text, TINY["text_dim"], generator=generator,
                                          dtype=torch.float32),
        timestep=torch.linspace(1.0, 0.5, timesteps),
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        image_indices=torch.arange(num_text, seq, dtype=torch.long),
        text_indices=torch.arange(0, num_text, dtype=torch.long),
        pdd_step_index=2,
    )


def forward_once(torch: Any, model: Any, inputs: dict[str, Any]) -> Any:
    with torch.no_grad():
        return model(**inputs).sample


def fp8_dtypes(torch: Any) -> tuple[Any, ...]:
    return (torch.float8_e4m3fn, torch.float8_e5m2)


# --------------------------------------------------------------------------- #
# Кейсы
# --------------------------------------------------------------------------- #

def check_cast_and_reuse(tmp: Path, torch: Any) -> None:
    """Кейсы 1-3: каст, состав кэша, сайдкар и повторный вызов без пересоздания."""
    from diffusers.models import MiniMaxH3ImageTransformer2DModel

    source = tmp / "src"
    cache = tmp / "fp8"
    model = tiny_model(torch)
    model.save_pretrained(str(source))
    assert (source / "config.json").is_file(), "save_pretrained не создал config.json"

    returned = cast_and_cache(MiniMaxH3ImageTransformer2DModel, source, cache)
    assert Path(returned) == cache.resolve(), f"вернулся путь {returned}, ожидался {cache.resolve()}"

    shards = sorted(cache.glob("*.safetensors"))
    assert shards, f"в кэше нет *.safetensors: {sorted(p.name for p in cache.iterdir())}"
    assert (cache / INDEX_NAME).is_file(), "нет индекса шардов"
    assert (cache / "config.json").is_file(), "нет config.json"
    sidecar = json.loads((cache / SIDECAR_NAME).read_text(encoding="utf-8"))
    assert (cache / BUFFERS_NAME).is_file(), "нет сайдкара буферов non_persistent.buffers.pt"
    assert sidecar["buffers"]["names"] == ["rope.inv_freq"], (
        f"буферы в сайдкаре: {sidecar['buffers']['names']}"
    )

    # R1: в сайдкаре — все параметры каста, а не только «есть fp8».
    required = {"schema", "source", "source_digest", "config_sha256", "fp8_dtype",
                "torch_version", "diffusers_version", "skip_patterns", "weights",
                "num_fp8", "num_exact", "non_fp8", "buffers"}
    missing = sorted(required - set(sidecar))
    assert not missing, f"в сайдкаре нет полей: {missing}"
    assert sidecar["schema"] == CACHE_SCHEMA
    assert sidecar["fp8_dtype"] == "float8_e4m3fn"
    assert sidecar["num_fp8"] > 0 and sidecar["num_exact"] > 0, (
        sidecar["num_fp8"], sidecar["num_exact"]
    )

    # Нормы и float32-модули объявлены точными, block-линеары — fp8.
    exact = set(sidecar["non_fp8"])
    for key in ("proj_in.weight", "time_embedder.linear_1.weight",
                "pdd_proj_out.0.weight", "transformer_blocks.0.norm1.weight",
                "transformer_blocks.0.attn.norm_q.weight", "norm_out.norm.weight"):
        assert key in exact, f"{key} должен остаться точным, а он в fp8"
    for key in ("transformer_blocks.0.attn.to_q.weight", "transformer_blocks.0.ff.net.2.weight",
                "context_embedder.weight"):
        assert key not in exact, f"{key} должен быть в fp8"
    assert len(exact) + sidecar["num_fp8"] == sidecar["num_tensors"], "списки не покрывают все тензоры"

    # Повторный вызов: reuse, ни один файл не пересоздан.
    before = sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in cache.iterdir())
    time.sleep(0.05)                             # mtime разрешение файловой системы
    cast_and_cache(MiniMaxH3ImageTransformer2DModel, source, cache)
    after = sorted((p.name, p.stat().st_mtime_ns, p.stat().st_size) for p in cache.iterdir())
    assert before == after, "повторный cast_and_cache пересоздал кэш"

    # Смена fp8_dtype инвалидирует кэш и **заменяет** непустой каталог (атомарная подмена).
    cast_and_cache(MiniMaxH3ImageTransformer2DModel, source, cache, fp8_dtype="float8_e5m2")
    changed = json.loads((cache / SIDECAR_NAME).read_text(encoding="utf-8"))
    assert changed["fp8_dtype"] == "float8_e5m2", changed["fp8_dtype"]
    leftovers = sorted(p.name for p in cache.parent.glob(f"{cache.name}.old-*"))
    assert not leftovers, f"после подмены остались каталоги: {leftovers}"
    cast_and_cache(MiniMaxH3ImageTransformer2DModel, source, cache)      # вернуть e4m3
    back = json.loads((cache / SIDECAR_NAME).read_text(encoding="utf-8"))
    assert back["fp8_dtype"] == "float8_e4m3fn", back["fp8_dtype"]


def check_load(tmp: Path, torch: Any) -> Any:
    """Кейс 4: fp8 у весов, точный dtype у норм/fp32-модулей, буферы бит-в-бит."""
    from diffusers.models import MiniMaxH3ImageTransformer2DModel

    cache = tmp / "fp8"
    original = tiny_model(torch)
    src_state = original.state_dict()
    exact_names = set(json.loads((cache / SIDECAR_NAME).read_text(encoding="utf-8"))["non_fp8"])
    loaded = load_cached(MiniMaxH3ImageTransformer2DModel, cache)

    state = loaded.state_dict()
    assert set(state) == set(src_state), (
        f"состав state_dict разошёлся: лишние {sorted(set(state) - set(src_state))[:3]}, "
        f"пропали {sorted(set(src_state) - set(state))[:3]}"
    )

    fp8 = fp8_dtypes(torch)
    for name, ref in src_state.items():
        got = state[name]
        if name in exact_names:
            assert got.dtype == ref.dtype, f"{name}: dtype {got.dtype} != {ref.dtype}"
            assert torch.equal(got, ref), f"{name}: значения не совпали бит-в-бит"
        else:
            assert got.dtype in fp8, f"{name}: ожидался fp8, пришёл {got.dtype}"
    assert len(exact_names) == len(src_state) - sum(
        1 for n, t in state.items() if t.dtype in fp8
    ), "сайдкар non_fp8 не совпадает с тем, что лежит в файлах весов"

    # Буфер persistent=False восстановлен из сайдкара (на диске его нет).
    ref_buffer = original.rope.inv_freq
    assert torch.equal(loaded.rope.inv_freq, ref_buffer), "rope.inv_freq не восстановлен"
    assert loaded.rope.inv_freq.dtype == torch.float32, loaded.rope.inv_freq.dtype
    assert not any("inv_freq" in name for name in state), "inv_freq неожиданно попал в файл весов"
    return loaded


def check_forward(tmp: Path, torch: Any, loaded: Any) -> None:
    """Кейс 5: forward fp8-модели с хуками апкаста — конечный тензор нужной формы."""
    original = tiny_model(torch)
    inputs = forward_inputs(torch, original)

    ref = forward_once(torch, original, inputs)
    loaded.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
    out = forward_once(torch, loaded, inputs)

    assert out.shape == ref.shape == (1, 8, TINY["in_channels"] * 2 * 2), (out.shape, ref.shape)
    assert torch.isfinite(out).all(), "в fp8-выходе есть NaN/Inf"
    # fp8 e4m3 (~2 значащих бита мантиссы) — сравнение только по величине расхождения.
    scale = float(ref.abs().max()) or 1.0
    rel = float((out - ref).abs().max()) / scale
    assert rel < 0.5, f"расхождение fp8-выхода {rel:.3f} от bf16 больше 50 % шкалы"
    print(f"    forward: fp8 vs bf16 max|diff|/scale = {rel:.3f}", flush=True)


# --------------------------------------------------------------------------- #
# Раннер
# --------------------------------------------------------------------------- #

def main() -> int:
    import torch

    results: list[tuple[str, str]] = []

    def run(name: str, func: Callable[[], Any]) -> Any:
        try:
            value = func()
        except Exception as exc:                             # noqa: BLE001 — тест печатает всё
            results.append((name, f"FAIL {type(exc).__name__}: {exc}"))
            traceback.print_exc()
            return None
        results.append((name, "PASS"))
        return value

    with temp_dir() as raw:
        tmp = Path(raw)
        run("каст + reuse", lambda: check_cast_and_reuse(tmp, torch))
        loaded = run("загрузка кэша", lambda: check_load(tmp, torch))
        if loaded is not None:
            run("forward fp8", lambda: check_forward(tmp, torch, loaded))

    failed = 0
    for name, status in results:
        print(f"{status} — {name}", flush=True)
        failed += status != "PASS"
    print("ALL PASS" if not failed else f"FAILED {failed}", flush=True)
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
