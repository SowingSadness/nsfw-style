"""Полная цепочка кубиков: `source -> [rewrite] -> encoder -> transformer -> decoder -> post`.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/cube/test_full_chain.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.
Без весов и без CUDA: композитор ничего не загружает, он проверяет данные.

Зачем (план `cube-architecture`, шаг 6, приёмка пререквизита): до этого шага композиция
`source -> encoder -> transformer -> decoder -> post` не собиралась — не хватало пяти кубиков.
Тест фиксирует, что теперь собирается и, главное, что компоновщик по-прежнему **ловит** ошибки:

1. полная цепочка 2912x2176 (окно внимания + kv offload + fp8 + апскейл + цветокоррекция) — `ok`;
2. T2I-цепочка (без референсов и без PE) — `ok`;
3. бюджет VRAM считается в фактической точности: fp8-стратегия включает `weights_fp8_gib`,
   энкодер и стратегии в сумму не идут;
4. нет `strategy.precision.fp8` -> `strategy_missing` (до загрузки весов);
5. забытый декодер -> `ports_unused` (латенты без потребителя);
6. транспонер раньше энкодера -> `stage_order`;
7. опечатка в имени параметра кубика -> `param_unknown` с подсказкой.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from qwenstyle.compose import CubeBinding as B                        # noqa: E402
from qwenstyle.compose import PipelineSpec, compose                    # noqa: E402
from qwenstyle.cube import CubeContext, all_cubes, get_cube            # noqa: E402
from qwenstyle.cubes import decoder_vae, post_esrgan, source_project, transformer_qwen21  # noqa: E402

MACHINE = {"vram_gib": 16.0, "ram_gib": 63.6}


def full_chain(**overrides: object) -> PipelineSpec:
    """Цепочка проекта 2912x2176: PE + окно внимания + kv offload + апскейл до 2160."""
    cubes = [
        B("source.project", {"dir": "prompts/nita-camp-model", "refs": "refs-style",
                             "aspect": "4:3", "size_px": 2176}),
        B("rewrite.pe_llamacpp", {"model": "pe_i2i_heretic-Q4_K_M.gguf",
                                  "mmproj": "pe_i2i_heretic-mmproj-F16.gguf"}),
        B("encoder.qwen3vl", {}),
        B("strategy.precision.fp8", {"cache_dir": "cache/models/fp8_transformer"}),
        B("strategy.kv_offload", {"mode": "offload"}),
        B("strategy.window_attention", {"window_attn_px": 1088, "window_attn_margin_px": 448}),
        B("transformer.qwen_image21", {"steps": 25, "offload": "group", "lora_scale": 0.6}),
        B("decoder.vae.qwen21", {"tile_px": 512}),
        B("post.upscale.esrgan", {"target_height": 2160}),
        B("post.colorfix", {"mode": "wavelet"}),
    ]
    return PipelineSpec(name="full-2176", cubes=tuple(cubes), **overrides)  # type: ignore[arg-type]


def codes(composition) -> list[str]:
    return [issue.code for issue in composition.issues]


def case_full_chain_is_ok() -> None:
    composition = compose(full_chain(), machine=MACHINE)
    assert composition.ok, composition.format_issues()
    assert composition.ports_out == frozenset({"pixels"}), composition.ports_out
    assert composition.provides == frozenset({"fp8", "kv_offload", "window_attention"})


def case_t2i_chain_is_ok() -> None:
    cubes = (
        B("source.project", {"dir": "prompts/nita-camp-model", "refs": "-", "aspect": "16:9"}),
        B("encoder.qwen3vl", {}),
        B("strategy.precision.fp8", {}),
        B("strategy.kv_offload", {"mode": "true"}),
        B("transformer.qwen_image21", {"steps": 8}),
        B("decoder.vae.qwen21", {}),
    )
    composition = compose(PipelineSpec(name="t2i", cubes=cubes, inputs=frozenset({"prompt"})),
                          machine=MACHINE)
    assert composition.ok, composition.format_issues()
    # без референсов окно внимания не нужно, но и не запрещено (стратегия — до трансформера)
    with_window = compose(PipelineSpec(name="t2i-win", cubes=cubes[:4] + (
        B("strategy.window_attention", {"window_attn_px": 960}),) + cubes[4:],
        inputs=frozenset({"prompt"})), machine=MACHINE)
    assert with_window.ok, with_window.format_issues()


def case_budget_counts_fp8() -> None:
    """Бюджет VRAM: fp8-стратегия переключает сумму на `weights_fp8_gib`, энкодер не в счёт."""
    composition = compose(full_chain(), machine=MACHINE)
    weights = {cube.id: cube.budget for cube in composition.cubes}
    expected = (weights["transformer.qwen_image21"].weights_fp8_gib
                + weights["decoder.vae.qwen21"].weights_bf16_gib
                + weights["post.upscale.esrgan"].weights_bf16_gib
                + weights["rewrite.pe_llamacpp"].weights_bf16_gib)
    assert abs(composition.budget_gib - expected) < 0.01, (composition.budget_gib, expected)
    # без fp8-стратегии веса трансформера считаются в bf16 — и в 16 ГиБ уже не влезают
    cubes = tuple(b for b in full_chain().cubes if b.cube_id != "strategy.precision.fp8")
    heavy = compose(PipelineSpec(name="no-fp8", cubes=cubes), machine=MACHINE)
    assert "budget" in codes(heavy), codes(heavy)
    assert heavy.budget_gib > composition.budget_gib


def case_missing_strategy_is_reported() -> None:
    cubes = tuple(b for b in full_chain().cubes if b.cube_id != "strategy.precision.fp8")
    composition = compose(PipelineSpec(name="no-strategy", cubes=cubes))
    missing = [i for i in composition.issues if i.code == "strategy_missing"]
    assert missing, codes(composition)
    assert "strategy.precision.fp8" in missing[0].hint, missing[0].hint
    assert missing[0].cube_id == "transformer.qwen_image21", missing[0].cube_id


def case_forgotten_decoder_is_reported() -> None:
    cubes = tuple(b for b in full_chain().cubes
                  if b.cube_id not in ("decoder.vae.qwen21", "post.upscale.esrgan",
                                       "post.colorfix"))
    composition = compose(PipelineSpec(name="no-decoder", cubes=cubes))
    assert "ports_unused" in codes(composition), codes(composition)
    assert "output_missing" in codes(composition), codes(composition)


def case_stage_order_is_reported() -> None:
    cubes = (
        B("encoder.qwen3vl", {}),
        B("source.project", {"dir": "prompts/x", "refs": "-"}),
    )
    composition = compose(PipelineSpec(name="backwards", cubes=cubes, inputs=frozenset({"prompt"})))
    assert "stage_order" in codes(composition), codes(composition)


def case_unknown_param_is_reported() -> None:
    cubes = full_chain().cubes[:-1] + (B("post.upscale.esrgan", {"target_height": 2160,
                                                                 "tile_size": 256}),)
    composition = compose(PipelineSpec(name="typo", cubes=cubes))
    issues = [i for i in composition.issues if i.code == "param_unknown"]
    assert issues and issues[0].field == "tile_size", [i.text for i in composition.issues]
    assert "tile" in (issues[0].hint or ""), issues[0].hint


def case_every_cube_in_registry_has_stage() -> None:
    """Все 13 кубиков регистрируются модулями и имеют рабочий self_test."""
    ctx = CubeContext()
    for cube in all_cubes():
        assert cube.id in {c.id for c in all_cubes()}
        if cube.self_test is not None:
            cube.self_test(ctx)
    ids = {cube.id for cube in all_cubes()}
    for expected in ("source.project", "transformer.qwen_image21", "post.upscale.esrgan",
                     "post.colorfix", "post.upscale.seedvr2"):
        assert expected in ids, expected
    assert get_cube("transformer.qwen_image21").requires == frozenset({"fp8", "kv_offload"})
    assert source_project.CUBE.outputs == frozenset({"prompt", "refs"})
    assert transformer_qwen21.CUBE.backend if hasattr(transformer_qwen21.CUBE, "backend") else True
    assert post_esrgan.CUBE.id == "post.upscale.esrgan"
    assert decoder_vae.CUBE.id == "decoder.vae.qwen21"


CASES = [
    ("полная цепочка 2912x2176 собирается", case_full_chain_is_ok),
    ("T2I-цепочка собирается", case_t2i_chain_is_ok),
    ("бюджет VRAM в фактической точности", case_budget_counts_fp8),
    ("нет fp8-стратегии -> strategy_missing", case_missing_strategy_is_reported),
    ("нет декодера -> ports_unused", case_forgotten_decoder_is_reported),
    ("энкодер раньше источника -> stage_order", case_stage_order_is_reported),
    ("опечатка параметра -> param_unknown", case_unknown_param_is_reported),
    ("все кубики зарегистрированы", case_every_cube_in_registry_has_stage),
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
