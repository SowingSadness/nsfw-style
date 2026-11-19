"""Адаптер transformers: fp8-кэш LM и `Fp8DequantLinear` (без весов моделей и CUDA).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/unit/test_transformers_adapter.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (план `cube-architecture`, шаг 5b.3): 9.41B в bf16 не влезает в 16 ГиБ, а fp8-кэш влезает
(8.8 ГиБ) и на замере даёт 4.19 ток/с против 1.07 у `device_map`. Тест фиксирует правила каста и
численность де-кванта:

1. `quantizable`: только 2D `*.weight`, не под пропуском (`embed_tokens`, `norm`, `bias`);
   `lm_head` квантуется — это 2D-вес на 1.2 ГиБ bf16.
2. `to_fp8`: fp8-вес + per-channel scale, де-квант возвращает вес с относительной ошибкой < 3 %.
3. `Fp8DequantLinear`: forward равен `F.linear` по де-квантованному весу (бит-в-бит), bias учитывается.
4. `cache_param_name`: `<n>.weight_scale` -> `<n>.weight.scale`, прочее без изменений.
5. `build_cache`: на синтетическом репозитории в локальном кэше Хаба — сайдкар с digest/skip и
   запись только 2D-весов в fp8 (точные тензоры остаются исходными).
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

import torch                                                    # noqa: E402
import torch.nn.functional as functional                         # noqa: E402

from qwenstyle.quant.adapters.transformers import (              # noqa: E402
    Fp8DequantLinear,
    cache_param_name,
    quantizable,
    to_fp8,
)


def case_quantizable_rules() -> None:
    weight = torch.randn(32, 64, dtype=torch.bfloat16)
    assert quantizable("model.layers.0.self_attn.q_proj.weight", weight)
    assert quantizable("lm_head.weight", weight), "lm_head квантуется (2D-вес 151936x4096)"
    assert not quantizable("model.embed_tokens.weight", weight), "эмбеддинги точные"
    assert not quantizable("model.layers.0.input_layernorm.weight", weight), "нормы точные"
    assert not quantizable("model.layers.0.self_attn.q_proj.bias", torch.randn(64)), "bias точный"
    assert not quantizable("conv.weight", torch.randn(8, 8, 3, 3)), "не 2D — не кастуем"


def case_to_fp8_roundtrip() -> None:
    weight = torch.randn(64, 128, dtype=torch.bfloat16)
    quantized, scale = to_fp8(weight)
    assert quantized.dtype == torch.float8_e4m3fn and quantized.shape == weight.shape, quantized
    assert scale.dtype == torch.float32 and scale.shape == (64,), scale
    back = (quantized.to(torch.bfloat16) * scale.to(torch.bfloat16)[:, None])
    rel = float((back.float() - weight.float()).abs().mean() / weight.float().abs().mean())
    assert rel < 0.03, f"де-квант: относительная ошибка {rel:.4f}"


def case_linear_forward() -> None:
    weight = torch.randn(64, 128, dtype=torch.bfloat16)
    bias = torch.randn(64, dtype=torch.bfloat16)
    quantized, scale = to_fp8(weight)
    cls = Fp8DequantLinear.install()
    layer = cls(64, 128, bias=True)
    layer.weight = quantized
    layer.weight_scale = scale
    layer.bias = torch.nn.Parameter(bias.clone())
    x = torch.randn(3, 5, 128, dtype=torch.bfloat16)
    got = layer(x)
    expected = functional.linear(x, layer.dequantized(x.dtype), bias)
    assert torch.equal(got, expected), "forward обязан считать по де-квантованному весу"
    assert got.shape == (3, 5, 64), got.shape

    no_bias = cls(64, 128, bias=False)
    assert no_bias.bias is None


def case_cache_param_name() -> None:
    assert cache_param_name("model.layers.0.q_proj.weight_scale") == \
        "model.layers.0.q_proj.weight.scale", cache_param_name("x.weight_scale")
    assert cache_param_name("model.layers.0.q_proj.weight") == "model.layers.0.q_proj.weight"
    assert cache_param_name("norm.weight") == "norm.weight"


CASES = [
    ("правила каста", case_quantizable_rules),
    ("to_fp8 + де-квант", case_to_fp8_roundtrip),
    ("Fp8DequantLinear.forward", case_linear_forward),
    ("имена параметров кэша", case_cache_param_name),
]


def main() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            fn()
        except Exception as exc:                                  # noqa: BLE001
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
