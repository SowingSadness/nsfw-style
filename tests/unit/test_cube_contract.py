"""Контракт кубика и композитор пайплайна: порты, параметры, стратегии (без torch и CUDA).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/unit/test_cube_contract.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем: план `cube-architecture`, шаг 1. Кубик объявляет порты/параметры/стратегии **данными**, а
композитор отсекает несовместимые цепочки до загрузки весов и без трейсбека Python. Тест фиксирует
контракт:

1. Регистрация: неизвестный порт, пустой id, стратегия без `provides`, дубли параметров — `CubeError`.
2. Валидная цепочка `source -> encoder -> transformer -> decoder` даёт `ok`, выход `pixels`.
3. Несходящиеся порты (`decoder` без `transformer`) — ошибка `port_missing`, а не исключение.
4. Неизвестный параметр, неверный тип и выход за диапазон — `param_unknown`/`param_type`/`param_range`.
5. Отсутствующая стратегия (`requires`) — `strategy_missing` с подсказкой имени кубика.
6. Нарушение порядка стадий — `stage_order`.
7. Неиспользованные порты на выходе (`latents` без декодера) — `ports_unused`.
8. `readonly`-параметр с чужим значением — `param_readonly`.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                        # тест не должен оставлять __pycache__ в пакете

from qwenstyle.backends.base import ParameterSpec, ResourceBudget   # noqa: E402
from qwenstyle.compose import CubeBinding, PipelineSpec, compose    # noqa: E402
from qwenstyle.cube import (                                       # noqa: E402
    CubeError,
    CubeSpec,
    Stage,
    register_cube,
    reset_registry,
)

SCHEMA = (
    ParameterSpec("steps", "cube", "int", 25, minimum=1, maximum=200),
    ParameterSpec("precision", "cube", "enum", "fp8", choices=("bf16", "fp8")),
    ParameterSpec("fixed", "cube", "int", 6, readonly=True),
)


def _build() -> None:
    """Игрушечный набор кубиков: source/encoder/transformer/decoder + стратегия fp8."""
    register_cube(CubeSpec(id="source.plain", stage=Stage.SOURCE, produces="prompt",
                           params=(ParameterSpec("text", "cube", "str", ""),)))
    register_cube(CubeSpec(id="encoder.toy", stage=Stage.ENCODER,
                           consumes=frozenset({"prompt"}), produces="conditioning"))
    register_cube(CubeSpec(id="strategy.precision.fp8", stage=Stage.TRANSFORMER,
                           provides=frozenset({"fp8"})))
    register_cube(CubeSpec(id="transformer.toy", stage=Stage.TRANSFORMER,
                           consumes=frozenset({"conditioning"}), produces="latents",
                           params=SCHEMA, provides=frozenset({"needs_fp8"}),
                           requires=frozenset({"fp8"}),
                           budget=ResourceBudget(weights_bf16_gib=13.0)))
    register_cube(CubeSpec(id="decoder.toy", stage=Stage.DECODER,
                           consumes=frozenset({"latents"}), produces="pixels"))
    register_cube(CubeSpec(id="post.toy", stage=Stage.POST,
                           consumes=frozenset({"pixels"}), produces="pixels"))


def _spec(*bindings: CubeBinding, inputs: frozenset[str] = frozenset({"prompt"})) -> PipelineSpec:
    return PipelineSpec(name="toy", cubes=bindings, inputs=inputs)


def _codes(result) -> list[str]:
    return [issue.code for issue in result.issues]


# --------------------------------------------------------------------------- кейсы

def case_registration_errors() -> None:
    """Ошибки контракта видны при регистрации, а не в рантайме."""
    reset_registry()
    bad = [
        CubeSpec(id="x bad", stage=Stage.SOURCE, produces="prompt"),            # пробел в id
        CubeSpec(id="x.bad_port", stage=Stage.SOURCE, produces="nope"),         # неизвестный порт
        CubeSpec(id="x.bad_in", stage=Stage.ENCODER,
                 consumes=frozenset({"nope"}), produces="conditioning"),
        CubeSpec(id="x.strategy_empty", stage=Stage.TRANSFORMER),               # стратегия без provides
        CubeSpec(id="x.dup", stage=Stage.TRANSFORMER, consumes=frozenset({"conditioning"}),
                 produces="latents",
                 params=(ParameterSpec("a", "cube", "int", 1),
                         ParameterSpec("a", "cube", "int", 2))),
        CubeSpec(id="x.self_required", stage=Stage.TRANSFORMER,
                 consumes=frozenset({"conditioning"}), produces="latents",
                 provides=frozenset({"fp8"}), requires=frozenset({"fp8"})),
    ]
    for spec in bad:
        try:
            register_cube(spec)
        except CubeError:
            continue
        raise AssertionError(f"регистрация {spec.id} должна была упасть")
    register_cube(CubeSpec(id="ok.cube", stage=Stage.SOURCE, produces="prompt"))
    try:
        register_cube(CubeSpec(id="ok.cube", stage=Stage.SOURCE, produces="prompt"))
    except CubeError:
        pass
    else:
        raise AssertionError("повторная регистрация того же id должна падать")


def case_valid_chain() -> None:
    """Валидная цепочка собирается, выход — `pixels`, бюджет суммируется."""
    reset_registry()
    _build()
    result = compose(_spec(CubeBinding("source.plain", {"text": "hi"}),
                           CubeBinding("encoder.toy"),
                           CubeBinding("strategy.precision.fp8"),
                           CubeBinding("transformer.toy", {"steps": 8}),
                           CubeBinding("decoder.toy")))
    assert result.ok, result.format_issues()
    assert result.ports_out == frozenset({"pixels"}), result.ports_out
    assert result.cube_ids() == ("source.plain", "encoder.toy", "strategy.precision.fp8",
                                 "transformer.toy", "decoder.toy")
    assert abs(result.budget_gib - 13.0) < 1e-9, result.budget_gib
    assert "fp8" in result.provides


def case_broken_ports() -> None:
    """Несходящиеся порты — данные, а не исключение."""
    reset_registry()
    _build()
    result = compose(_spec(CubeBinding("encoder.toy"), CubeBinding("decoder.toy")))
    assert not result.ok
    assert "port_missing" in _codes(result), _codes(result)
    assert "latents" in result.format_issues()
    assert isinstance(result.format_issues(), str)


def case_parameter_checks() -> None:
    """Чужой параметр, неверный тип, выход за диапазон, readonly."""
    reset_registry()
    _build()
    base = [CubeBinding("source.plain", {"text": "hi"}), CubeBinding("encoder.toy"),
            CubeBinding("strategy.precision.fp8")]
    cases = {
        "param_unknown": CubeBinding("transformer.toy", {"nope": 1}),
        "param_type": CubeBinding("transformer.toy", {"steps": "много"}),
        "param_range": CubeBinding("transformer.toy", {"steps": 999}),
        "param_readonly": CubeBinding("transformer.toy", {"fixed": 8}),
    }
    for code, binding in cases.items():
        result = compose(_spec(*base, binding, CubeBinding("decoder.toy")))
        assert code in _codes(result), f"{code}: {_codes(result)}"
    # неверное значение enum тоже ловится
    result = compose(_spec(*base, CubeBinding("transformer.toy", {"precision": "int8"}),
                           CubeBinding("decoder.toy")))
    assert "param_type" in _codes(result), _codes(result)


def case_missing_strategy() -> None:
    """`requires` без соответствующего кубика — понятная ошибка с подсказкой."""
    reset_registry()
    _build()
    result = compose(_spec(CubeBinding("source.plain", {"text": "x"}),
                           CubeBinding("encoder.toy"),
                           CubeBinding("transformer.toy"),
                           CubeBinding("decoder.toy")))
    assert "strategy_missing" in _codes(result), _codes(result)
    assert "strategy.precision.fp8" in result.format_issues(), result.format_issues()


def case_stage_order() -> None:
    """Стадии не должны идти назад."""
    reset_registry()
    _build()
    result = compose(_spec(CubeBinding("source.plain", {"text": "x"}),
                           CubeBinding("decoder.toy"),
                           CubeBinding("transformer.toy")))
    assert "stage_order" in _codes(result), _codes(result)


def case_unused_ports() -> None:
    """Порт, который никто не забрал, — ошибка (кадр не соберётся)."""
    reset_registry()
    _build()
    result = compose(_spec(CubeBinding("source.plain", {"text": "x"}),
                           CubeBinding("encoder.toy"),
                           CubeBinding("strategy.precision.fp8"),
                           CubeBinding("transformer.toy")))
    assert "ports_unused" in _codes(result), _codes(result)
    assert "latents" in result.format_issues()


def case_machine_budget() -> None:
    """Бюджет весов больше VRAM машины — ошибка композиции."""
    reset_registry()
    _build()
    chain = (CubeBinding("source.plain", {"text": "x"}), CubeBinding("encoder.toy"),
             CubeBinding("strategy.precision.fp8"), CubeBinding("transformer.toy"),
             CubeBinding("decoder.toy"))
    fits = compose(_spec(*chain), machine={"vram_gib": 16.0})
    assert fits.ok, fits.format_issues()
    small = compose(_spec(*chain), machine={"vram_gib": 8.0})
    assert "budget" in _codes(small), _codes(small)


CASES = [
    ("регистрация: ошибки контракта", case_registration_errors),
    ("валидная цепочка -> pixels", case_valid_chain),
    ("несходящиеся порты", case_broken_ports),
    ("параметры: чужой/тип/диапазон/readonly", case_parameter_checks),
    ("стратегия требуется и отсутствует", case_missing_strategy),
    ("порядок стадий", case_stage_order),
    ("неиспользованные порты", case_unused_ports),
    ("бюджет против VRAM машины", case_machine_budget),
]


def main() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            fn()
        except Exception as exc:                                     # noqa: BLE001
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
