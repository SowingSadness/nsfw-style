"""Кубики-стратегии: точность fp8, KV-offload, скользящее окно внимания (без torch).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/unit/test_strategies.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (план `cube-architecture`, шаг 3): раньше `precision`, `kv_cache`, `window_attn_px` и
`window_attn_margin_px` читались из `generation.sampler_kwargs` руками. Теперь это три
кубика-декоратора со своими `ParameterSpec` и `self_test`, а `qwen_image21` вызывает их
`from_mapping()` — то есть семантика в одном месте и её можно показать в UI.

Тест фиксирует:

1. `strategy.precision.fp8`: дефолт fp8, `bf16` без кэша, пустая строка отключает кэш, `rebuild` из строки.
2. `strategy.kv_offload`: `auto-offload-ref2` → `true` при 1 рефе и `offload` при 2+; явные режимы.
3. `strategy.window_attention`: гейт паритета (полоса >= высоты -> 1 полоса, 100 %), реальная
   геометрия 2912x2176 (2 полосы, 22568/24752) и порог покрытия ~2/3.
4. Композитор: трансформер с `requires={"fp8"}` без кубика-стратегии — `strategy_missing`;
   со стратегией — `ok`; `in-place` кубик (pixels -> pixels) не считается дубликатом.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from qwenstyle.backends.base import ParameterSpec                    # noqa: E402
from qwenstyle.compose import CubeBinding, PipelineSpec, compose      # noqa: E402
from qwenstyle.cube import CubeContext, CubeSpec, Stage, register_cube, reset_registry  # noqa: E402
from qwenstyle.cubes import (                                        # noqa: E402
    strategy_kv,
    strategy_precision,
    strategy_window_attention,
)

CTX = CubeContext()


def case_precision() -> None:
    default = strategy_precision.PrecisionParams.from_mapping({})
    assert default.precision == "fp8" and default.is_fp8, default
    assert default.cache_path is not None, "по умолчанию fp8-кэш включён"
    bf16 = strategy_precision.PrecisionParams.from_mapping({"precision": "bf16"})
    assert not bf16.is_fp8 and bf16.cache_path is None, bf16
    off = strategy_precision.PrecisionParams.from_mapping({"cache_dir": ""})
    assert off.cache_path is None, "пустая строка отключает кэш"
    assert strategy_precision.PrecisionParams.from_mapping({"rebuild": "true"}).rebuild is True
    assert strategy_precision.PrecisionParams.from_mapping({"rebuild": "off"}).rebuild is False


def case_kv() -> None:
    auto = strategy_kv.KvParams.from_mapping({})
    assert auto.resolve(1) == "true", auto.resolve(1)
    assert auto.resolve(2) == "offload", auto.resolve(2)
    assert auto.resolve(5) == "offload"
    assert strategy_kv.KvParams.from_mapping({"mode": "false"}).resolve(3) == "false"
    assert strategy_kv.KvParams.from_mapping({"mode": "offload"}).in_ram is True
    assert set(strategy_kv.KV_MODES) == set(strategy_kv.CUBE.param("mode").choices or ())


def case_window() -> None:
    off = strategy_window_attention.WindowParams.from_mapping({})
    assert not off.active, "0 px = плотное внимание"

    live = strategy_window_attention.WindowParams.from_mapping(
        {"window_attn_px": 1088, "window_attn_margin_px": 448})
    geo = live.geometry((2912, 2176))
    assert geo["bands"] == 2 and geo["visible_rows"] == 124, geo
    assert geo["tokens_visible"] == 22568 and geo["tokens_total"] == 24752, geo

    whole = strategy_window_attention.WindowParams.from_mapping(
        {"window_attn_px": 4096, "window_attn_margin_px": 0})
    geo_whole = whole.geometry((1024, 768))
    assert geo_whole["bands"] == 1 and geo_whole["coverage"] == 1.0, geo_whole

    thin = strategy_window_attention.WindowParams.from_mapping(
        {"window_attn_px": 384, "window_attn_margin_px": 128})
    assert strategy_window_attention.coverage((3200, 1792), 384, 128) < 2 / 3, "порог 2/3"


def case_composer_with_strategies() -> None:
    """Трансформер требует fp8: без кубика-стратегии — ошибка, со стратегией — ok."""
    reset_registry()
    register_cube(CubeSpec(id="source.toy", stage=Stage.SOURCE, produces="prompt",
                           params=(ParameterSpec("text", "cube", "str", ""),)))
    register_cube(CubeSpec(id="encoder.toy", stage=Stage.ENCODER,
                           consumes=frozenset({"prompt"}), produces="conditioning"))
    register_cube(CubeSpec(id="transformer.toy", stage=Stage.TRANSFORMER,
                           consumes=frozenset({"conditioning"}), produces="latents",
                           requires=frozenset({"fp8"})))
    register_cube(CubeSpec(id="decoder.toy", stage=Stage.DECODER,
                           consumes=frozenset({"latents"}), produces="pixels"))
    register_cube(CubeSpec(id="post.toy", stage=Stage.POST,
                           consumes=frozenset({"pixels"}), produces="pixels"))
    register_cube(strategy_precision.CUBE)

    base = [CubeBinding("source.toy", {"text": "x"}), CubeBinding("encoder.toy"),
            CubeBinding("transformer.toy"), CubeBinding("decoder.toy")]
    broken = compose(PipelineSpec("toy", tuple(base), frozenset({"prompt"})))
    codes = [issue.code for issue in broken.issues]
    assert "strategy_missing" in codes, codes
    assert "strategy.precision.fp8" in broken.format_issues(), broken.format_issues()

    chain = tuple(base[:2] + [CubeBinding("strategy.precision.fp8")] + base[2:])
    fixed = compose(PipelineSpec("toy", chain, frozenset({"prompt"})))
    assert fixed.ok, fixed.format_issues()

    # in-place кубик (pixels -> pixels) — не дубликат порта
    with_post = compose(PipelineSpec("toy", chain + (CubeBinding("post.toy"),),
                                     frozenset({"prompt"})))
    assert with_post.ok, with_post.format_issues()


CASES = [
    ("strategy.precision.fp8", case_precision),
    ("strategy.kv_offload", case_kv),
    ("strategy.window_attention", case_window),
    ("композитор со стратегиями", case_composer_with_strategies),
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
    # все зарегистрированные self_test — без весов
    try:
        from qwenstyle.cube import all_cubes
        for cube in all_cubes():
            if cube.self_test is not None:
                cube.self_test(CTX)
    except Exception as exc:                                        # noqa: BLE001
        failed += 1
        print(f"FAIL self_test всех кубиков: {exc}")
        traceback.print_exc()
    else:
        print("PASS self_test всех кубиков")
    if failed:
        print(f"\n{failed} из {len(CASES) + 1} кейсов провалено")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
