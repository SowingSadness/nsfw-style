"""Контракт кубика: декларация (`CubeSpec`) == реализация (`*Params.from_mapping`).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/contract/test_cube_declarations.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.
Без torch-CUDA, без весов и без сети — поэтому идёт в срезе `contract` (секунды).

Зачем: в шагах 1–5 спек и код кубика писались вместе, и разъезжались они молча. Типичные поломки,
которые ловит этот файл:

* параметр объявлен в `CubeSpec.params`, но `from_mapping` его не знает — значение из `--set`
  или из пресета просто исчезает (у нас так было бы с `parse_mode` у `rewrite.pe_t2i`);
* в датаклассе параметров больше полей, чем объявлено (поле-сирота: в UI его нет, а поведение зависит);
* домен `enum` не совпадает со списком, который проверяет код (`strategy.kv_offload` держит
  `KV_MODES` отдельно от `spec.choices`);
* кубик объявил `provides`, но стратегией не является, или наоборот;
* `self_test` у стратегии требует CUDA — тогда «быстрая» проверка тянет за собой весь стенд.

Проверки намеренно **структурные**: они не загружают модели, а сверяют данные с кодом.
"""
from __future__ import annotations

import dataclasses
import inspect
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

import qwenstyle.cubes as cubes_pkg                               # noqa: E402  — регистрирует кубики
from qwenstyle.backends.base import ParameterSpec                  # noqa: E402
from qwenstyle.compose import (                                    # noqa: E402
    CubeBinding, PipelineSpec, compose, validate_parameters,
)
from qwenstyle.cube import (                                       # noqa: E402
    CubeContext, CubeSpec, Stage, all_cubes, get_cube,
)
from qwenstyle.ports import PORT_TYPES                             # noqa: E402

#: Допустимые типы значений параметра (`ParameterSpec.type`).
KNOWN_TYPES = frozenset({"int", "float", "bool", "str", "enum"})
#: Допустимые `scope` (`ParameterSpec.scope`).
KNOWN_SCOPES = frozenset({"cube", "generation", "sampler", "encoder", "upscale", "post"})
#: Порядок стадий: кубик-источник идёт первым, постобработка — последней.
STAGE_ORDER = [s.value for s in Stage]
#: Контекст без устройства и кэша: `self_test` обязан работать и на нём.
CTX = CubeContext()


# --------------------------------------------------------------------------- обнаружение

def _cube_specs(module: Any) -> list[CubeSpec]:
    """Спеки, объявленные модулем: `CUBE`, `CUBE_I2I`, `CUBE_T2I`, … (по соглашению)."""
    found = []
    for name, value in vars(module).items():
        if isinstance(value, CubeSpec) and (name == "CUBE" or name.startswith("CUBE_")):
            found.append(value)
    return found


def _params_class(module: Any) -> type | None:
    """Класс параметров модуля — тот, что объявляет `from_mapping` (в нём живёт реализация)."""
    for value in vars(module).values():
        if (inspect.isclass(value) and value.__module__ == module.__name__
                and hasattr(value, "from_mapping") and dataclasses.is_dataclass(value)):
            return value
    return None


def _modules() -> list[Any]:
    return [getattr(cubes_pkg, name) for name in cubes_pkg.__all__]


def _declared(spec: CubeSpec) -> dict[str, Any]:
    """Значения по умолчанию из спеки — то, что увидит UI и что подставит пресет."""
    return {p.name: p.default for p in spec.params}


def _alternative(spec: ParameterSpec) -> Any:
    """Другое допустимое значение параметра (для проверки «декларация == поведение»)."""
    if spec.type == "bool":
        return not spec.default
    if spec.type == "enum":
        for choice in spec.choices or ():
            if choice != spec.default:
                return choice
        return None
    if spec.type in ("int", "float"):
        base = spec.default if isinstance(spec.default, (int, float)) else 0
        up = base + 1 if spec.type == "int" else base + 0.1
        if spec.maximum is None or up <= spec.maximum:
            return up
        down = base - 1 if spec.type == "int" else base - 0.1
        if spec.minimum is None or down >= spec.minimum:
            return down
        return None
    return f"{spec.default or ''}_x"


# --------------------------------------------------------------------------- кейсы

def case_every_cube_is_declared_by_its_module() -> None:
    """Каждый зарегистрированный кубик объявлен модулем пакета, и ссылка та же самая."""
    declared: dict[str, int] = {}
    for module in _modules():
        specs = _cube_specs(module)
        assert specs, f"{module.__name__}: нет ни одного CUBE/CUBE_*"
        for spec in specs:
            declared[spec.id] = declared.get(spec.id, 0) + 1
            assert get_cube(spec.id) is spec, f"{spec.id}: в реестре другой объект"
    for cube_id, count in declared.items():
        assert count == 1, f"{cube_id}: объявлен {count} раз (два модуля с одним id)"
    registered = {spec.id for spec in all_cubes()}
    assert registered == set(declared), (f"реестр и модули разошлись: "
                                         f"лишние {sorted(registered - set(declared))}, "
                                         f"нет {sorted(set(declared) - registered)}")
    for cube_id in registered:
        assert "." in cube_id, f"{cube_id}: id обязан быть составным (`stage.name`), как в пресетах"


def case_parameter_specs_are_sane() -> None:
    """Дескрипторы параметров: имя, тип, домен, диапазон, док, путь для `--set`."""
    for spec in all_cubes():
        names = [p.name for p in spec.params]
        assert len(names) == len(set(names)), f"{spec.id}: повторяющиеся параметры {names}"
        for p in spec.params:
            where = f"{spec.id}.{p.name}"
            assert p.name.isidentifier(), f"{where}: имя не идентификатор"
            assert p.type in KNOWN_TYPES, f"{where}: неизвестный тип {p.type!r}"
            assert p.scope in KNOWN_SCOPES, f"{where}: неизвестный scope {p.scope!r}"
            assert p.doc.strip(), f"{where}: пустой doc — UI покажет пустую строку"
            if p.type == "enum":
                assert p.choices, f"{where}: enum без вариантов"
                assert p.default in p.choices, f"{where}: default {p.default!r} вне choices"
            else:
                assert not p.choices, f"{where}: choices у не-enum типа {p.type!r}"
            if p.type in ("int", "float") and p.default is not None:
                if p.minimum is not None:
                    assert p.default >= p.minimum, f"{where}: default ниже минимума"
                if p.maximum is not None:
                    assert p.default <= p.maximum, f"{where}: default выше максимума"
            assert " " not in p.path, f"{where}: путь для --set содержит пробел"
            assert p.path.endswith(p.name), f"{where}: путь {p.path} не заканчивается именем"


def case_params_reach_implementation() -> None:
    """Спека и датакласс параметров совпадают по количеству, а значения доходят до полей."""
    for module in _modules():
        cls = _params_class(module)
        assert cls is not None, f"{module.__name__}: нет датакласса параметров с from_mapping"
        fields = [f.name for f in dataclasses.fields(cls)]
        for spec in _cube_specs(module):
            declared = _declared(spec)
            assert len(fields) == len(spec.params), (
                f"{spec.id}: в спеке {len(spec.params)} параметров, в {cls.__name__} "
                f"{len(fields)} полей — одно из двух объявлено впустую")
            params = cls.from_mapping(declared)
            by_name = {f.name: getattr(params, f.name) for f in dataclasses.fields(cls)}
            for p in spec.params:
                if p.name in by_name:
                    assert by_name[p.name] == p.default, (
                        f"{spec.id}.{p.name}: спека обещает {p.default!r}, "
                        f"from_mapping отдаёт {by_name[p.name]!r}")
                else:
                    # Имена полей могут отличаться (спекой правит конфиг, полями — геометрия:
                    # `window_attn_px` -> `band_px`), тогда обязательна честная передача значения.
                    alt = _alternative(p)
                    if alt is None:
                        continue
                    changed = cls.from_mapping({**declared, p.name: alt})
                    assert changed != params, (
                        f"{spec.id}.{p.name}: значение {alt!r} не изменило ни одного поля — "
                        f"параметр объявлен, но не доходит до кода")


def case_unknown_parameter_is_rejected_by_composer() -> None:
    """Чужое имя параметра ловит композитор (сам `from_mapping` намеренно молчит)."""
    for spec in all_cubes():
        issues = validate_parameters(spec, {"__nope__": 1})
        assert any(i.code == "param_unknown" for i in issues), (
            f"{spec.id}: `__nope__` не пойман композитором")


def case_ports_stage_and_strategies_agree() -> None:
    """Порты известны, стадии в правильном порядке, стратегии — только `provides`."""
    for spec in all_cubes():
        assert isinstance(spec.stage, Stage), f"{spec.id}: stage не Stage"
        unknown = (spec.consumes | spec.reads) - set(PORT_TYPES)
        assert not unknown, f"{spec.id}: неизвестные порты {sorted(unknown)}"
        if spec.produces is not None:
            assert spec.produces in PORT_TYPES, f"{spec.id}: неизвестный порт {spec.produces!r}"
        if spec.is_strategy:
            assert spec.provides, f"{spec.id}: стратегия без provides"
            assert spec.produces is None and not spec.consumes, (
                f"{spec.id}: стратегия не должна потреблять/производить порты")
        else:
            assert spec.produces, f"{spec.id}: не-стратегия без produces"
        assert not (spec.requires & spec.provides), (
            f"{spec.id}: требует и сам даёт {sorted(spec.requires & spec.provides)}")


def case_budget_is_non_negative() -> None:
    """Бюджет: числа не отрицательные, fp8 не больше bf16 (fp8 — упаковка, а не новая копия)."""
    for spec in all_cubes():
        budget = spec.budget
        for field_name in ("weights_bf16_gib", "encoder_gib", "disk_gib"):
            value = getattr(budget, field_name)
            assert value >= 0, f"{spec.id}: {field_name}={value}"
        if budget.weights_fp8_gib is not None:
            assert budget.weights_fp8_gib >= 0, f"{spec.id}: weights_fp8_gib отрицательный"
            if budget.weights_bf16_gib:
                assert budget.weights_fp8_gib <= budget.weights_bf16_gib, (
                    f"{spec.id}: fp8 {budget.weights_fp8_gib} > bf16 {budget.weights_bf16_gib}")


def case_self_test_runs_without_weights() -> None:
    """`self_test` есть у всех и проходит на синтетике без весов, CUDA и сети."""
    for spec in all_cubes():
        assert spec.self_test is not None, f"{spec.id}: нет self_test"
        spec.self_test(CTX)


def case_cube_binds_in_composition() -> None:
    """Кубик с объявленными умолчаниями связывается композитором без замечаний по параметрам."""
    for spec in all_cubes():
        pipeline = PipelineSpec(name=f"contract-{spec.id}",
                                cubes=(CubeBinding(spec.id, _declared(spec)),))
        composition = compose(pipeline)
        param_issues = [i for i in composition.issues
                        if i.code.startswith("param_") and i.cube_id == spec.id]
        assert not param_issues, (f"{spec.id}: "
                                  f"{'; '.join(f'{i.code}: {i.message}' for i in param_issues)}")


CASES = [
    ("кубик объявлен своим модулем", case_every_cube_is_declared_by_its_module),
    ("дескрипторы параметров корректны", case_parameter_specs_are_sane),
    ("параметры доходят до реализации", case_params_reach_implementation),
    ("чужой параметр ловит композитор", case_unknown_parameter_is_rejected_by_composer),
    ("порты/стадии/стратегии согласованы", case_ports_stage_and_strategies_agree),
    ("бюджет не отрицательный", case_budget_is_non_negative),
    ("self_test без весов", case_self_test_runs_without_weights),
    ("кубик связывается в композиции", case_cube_binds_in_composition),
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
