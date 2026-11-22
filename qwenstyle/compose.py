"""Композитор пайплайна: статическая проверка цепочки кубиков (план `cube-architecture`, шаг 1).

Задача — не дать собрать несовместимую цепочку **до** загрузки весов: порты стыкуются, параметры
принадлежат спеке, значения в допустимых диапазонах, требуемые стратегии включены, стадии идут
по возрастанию, бюджет укладывается в профиль машины. Ошибки возвращаются **данными**
(`Composition.issues`), а не исключениями: CLI печатает их списком, UI подсвечивает поле.

Пример:

    spec = PipelineSpec(name="t2i", cubes=(
        CubeBinding("source.plain"), CubeBinding("encoder.qwen3vl"),
        CubeBinding("transformer.qwen_image21"), CubeBinding("decoder.vae.qwen21"),
    ))
    result = compose(spec)
    assert result.ok, result.format_issues()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .backends.base import ParameterSpec
from .cube import CubeError, CubeSpec, Stage, get_cube
from .ports import PORT_TYPES

__all__ = [
    "Composition",
    "CubeBinding",
    "Issue",
    "PipelineSpec",
    "compose",
    "validate_parameters",
]


@dataclass(frozen=True)
class Issue:
    """Одна проблема композиции. `code` — машиночитаемый ярлык для UI."""

    code: str
    message: str
    cube_id: str | None = None
    field: str | None = None
    hint: str = ""

    @property
    def text(self) -> str:
        where = self.cube_id or "pipeline"
        if self.field:
            where = f"{where}.{self.field}"
        tail = f" ({self.hint})" if self.hint else ""
        return f"{where}: {self.message}{tail}"


@dataclass(frozen=True)
class CubeBinding:
    """Кубик в композиции: id + значения его параметров (уже без имён чужих кубиков)."""

    cube_id: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineSpec:
    """Описание пайплайна: какие кубики и с какими параметрами.

    `inputs` — порты, доступные на входе (у проекта с референсами это `prompt`+`refs`, у T2I —
    только `prompt`); `profile` — имя профиля машины для проверки бюджета (необязательно).
    """

    name: str
    cubes: tuple[CubeBinding, ...] = ()
    inputs: frozenset[str] = frozenset({"prompt", "refs"})
    profile: str | None = None

    def with_cube(self, cube_id: str, **params: Any) -> "PipelineSpec":
        return PipelineSpec(name=self.name, cubes=self.cubes + (CubeBinding(cube_id, params),),
                            inputs=self.inputs, profile=self.profile)


@dataclass(frozen=True)
class Composition:
    """Результат `compose`: разрешённые спеки, порядок, итоговые порты и проблемы."""

    spec: PipelineSpec
    cubes: tuple[CubeSpec, ...] = ()
    issues: tuple[Issue, ...] = ()
    ports_in: frozenset[str] = frozenset()
    ports_out: frozenset[str] = frozenset()
    bindings: tuple[tuple[str, dict[str, Any]], ...] = ()
    provides: frozenset[str] = frozenset()
    budget_gib: float = 0.0

    @property
    def errors(self) -> tuple[Issue, ...]:
        return self.issues

    @property
    def ok(self) -> bool:
        return not self.issues

    def format_issues(self) -> str:
        return "\n".join(f"  - {issue.text}" for issue in self.issues)

    def cube_ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.cubes)


def _check_value(spec: ParameterSpec, value: Any, cube_id: str) -> list[Issue]:
    """Проверка одного значения по дескриптору параметра."""
    issues: list[Issue] = []
    where = (cube_id, spec.name)
    if spec.readonly and value != spec.default:
        issues.append(Issue("param_readonly",
                            f"параметр зафиксирован моделью (значение {spec.default!r})",
                            cube_id, spec.name, f"получено {value!r}"))
        return issues
    if value is None and spec.default is None:
        # «не задан» — законное значение для параметра с дефолтом `None`
        # (например `target_height`: апскейл включается только явной целью).
        return issues
    try:
        if spec.type == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("нужно целое")
            checked = value
        elif spec.type == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("нужно число")
            checked = float(value)
        elif spec.type == "bool":
            if not isinstance(value, bool):
                raise TypeError("нужно true/false")
            checked = value
        elif spec.type == "enum":
            if not isinstance(value, str) or (spec.choices and value not in spec.choices):
                raise ValueError(f"нужно одно из {list(spec.choices or ())}")
            checked = value
        else:                                            # "str" и всё остальное
            checked = value
    except (TypeError, ValueError) as exc:
        issues.append(Issue("param_type", f"неверный тип: {exc}", *where,
                            f"ожидалось {spec.type}"))
        return issues

    if spec.type in ("int", "float"):
        if spec.minimum is not None and checked < spec.minimum:
            issues.append(Issue("param_range", f"меньше минимума {spec.minimum}", *where,
                                f"получено {checked}"))
        if spec.maximum is not None and checked > spec.maximum:
            issues.append(Issue("param_range", f"больше максимума {spec.maximum}", *where,
                                f"получено {checked}"))
    return issues


def validate_parameters(cube: CubeSpec, params: Mapping[str, Any]) -> list[Issue]:
    """Проверить значения параметров кубика: чужие имена, типы, диапазоны, обязательные."""
    issues: list[Issue] = []
    known = cube.param_names
    for name in sorted(set(params) - known):
        issues.append(Issue("param_unknown", "у кубика нет такого параметра",
                            cube.id, name, f"допустимые: {', '.join(sorted(known)) or '—'}"))
    for spec in cube.params:
        if spec.name in params:
            issues.extend(_check_value(spec, params[spec.name], cube.id))
        elif spec.default is None and spec.readonly:
            issues.append(Issue("param_required", "параметр обязателен и не имеет значения",
                                cube.id, spec.name))
    return issues


def compose(spec: PipelineSpec, *, machine: Mapping[str, Any] | None = None) -> Composition:
    """Собрать и проверить композицию. Ничего не загружает и не трогает GPU."""
    issues: list[Issue] = []

    unknown_inputs = set(spec.inputs) - set(PORT_TYPES)
    if unknown_inputs:
        issues.append(Issue("input_unknown", f"неизвестные входные порты {sorted(unknown_inputs)}",
                            hint=f"допустимые: {sorted(PORT_TYPES)}"))

    available: set[str] = set(spec.inputs)
    consumed: set[str] = set()
    read_ports: set[str] = set()
    provides: set[str] = set()
    cube_specs: list[CubeSpec] = []
    bindings: list[tuple[str, dict[str, Any]]] = []
    stage_cursor = Stage.SOURCE.order - 1
    produced: set[str] = set()
    # Последнее событие по порту: `produced` — порт никем не забран (значит, он и есть выход
    # композиции), `consumed`/`read` — им уже воспользовались. Нужен из-за fan-out: `frame`
    # производят один раз, а потребляют два кубика, поэтому «просто множество» тут врёт.
    last_event: dict[str, str] = {}

    for binding in spec.cubes:
        try:
            cube = get_cube(binding.cube_id)
        except CubeError as exc:
            issues.append(Issue("cube_unknown", str(exc), binding.cube_id))
            continue
        cube_specs.append(cube)
        bindings.append((cube.id, dict(binding.params)))

        # стадии не должны идти назад (стратегии живут на стадии своего кубика)
        if cube.stage.order < stage_cursor:
            issues.append(Issue("stage_order",
                                f"стадия {cube.stage.value} идёт после более поздней",
                                cube.id, hint="порядок: " + " -> ".join(s.value for s in Stage)))
        stage_cursor = max(stage_cursor, cube.stage.order)

        # порты
        needed = cube.consumes | cube.reads
        missing = sorted(needed - available)
        if missing:
            issues.append(Issue("port_missing", f"нет входных портов {missing}", cube.id, None,
                                f"доступны: {', '.join(sorted(available)) or '—'}"))
        # кубик-преобразователь (`pixels -> pixels`, `prompt -> prompt`) законно переиспользует
        # свой порт: дубликатом считается только «производство» порта, который он не потребляет.
        outputs = cube.outputs
        fresh = outputs - cube.consumes
        for port in sorted(fresh & produced):
            issues.append(Issue("port_duplicate", f"порт {port!r} уже произведён", cube.id))
        # Порт может потребляться **несколькими** кубиками (геометрия `frame` нужна и энкодеру, и
        # трансформеру), поэтому потребление не изымает порт из `available`: оно лишь перестаёт
        # считаться «неиспользованным». Дубликаты ловятся выше, по `produced`.
        consumed |= set(cube.consumes)
        read_ports |= set(cube.reads)
        available |= outputs
        produced |= fresh
        for port in cube.consumes:
            last_event[port] = "consumed"
        for port in cube.reads:
            # чтение уже произведённого порта = им воспользовались (иначе `refs` висел бы выходом)
            last_event[port] = "consumed" if port in produced else "read"
        for port in outputs:
            last_event[port] = "produced"

        # стратегии
        unmet = sorted(cube.requires - provides)
        if unmet:
            issues.append(Issue("strategy_missing",
                                f"нужны стратегии {unmet}", cube.id, None,
                                "добавьте кубики: " + ", ".join(_strategy_hint(unmet))))
        provides |= set(cube.provides)

        # параметры
        if cube.is_strategy and binding.params and not cube.params:
            issues.append(Issue("param_unknown", "у стратегии нет параметров",
                                cube.id, sorted(binding.params)[0]))
        issues.extend(validate_parameters(cube, binding.params))

    if "pixels" not in available:
        issues.append(Issue("output_missing", "на выходе композиции нет порта 'pixels'",
                            hint=f"сейчас на выходе: {', '.join(sorted(available)) or '—'}"))
    # Неиспользованным считается только **произведённый** порт, который никто не забрал:
    # «неиспользованный референс» на входе — норма (энкодер читает его через `reads`), а вот
    # латенты без декодера или conditioning без трансформера — ошибка композиции.
    # `reads` считается использованием: `encoder.qwen3vl` именно читает `refs` (не забирает),
    # и `source.project` производит его ради энкодера.
    leftover = sorted(produced - consumed - read_ports - {"pixels"})
    if leftover:
        issues.append(Issue("ports_unused", f"на выходе остались неиспользованные порты {leftover}",
                            hint="добавьте кубик-потребитель (например decoder для 'latents')"))

    # Бюджет VRAM: суммарные веса в **фактической** точности. Включённая стратегия `fp8` берёт
    # `weights_fp8_gib` (упаковка тех же весов), а энкодер стримится по модулям и объявляет
    # `encoder_gib` (RAM) — его веса в VRAM не живут, поэтому в сумму VRAM не идут.
    use_fp8 = "fp8" in provides
    budget = 0.0
    for cube in cube_specs:
        if cube.stage is Stage.ENCODER or cube.is_strategy:
            continue                        # энкодер в RAM/стримингом, стратегия ничего не держит
        packed = cube.budget.weights_fp8_gib
        budget += packed if (use_fp8 and packed is not None) else cube.budget.weights_bf16_gib
    if machine:
        vram = machine.get("vram_gib")
        if isinstance(vram, (int, float)) and budget > float(vram):
            issues.append(Issue("budget", f"веса не влезают в VRAM машины: {budget:.1f} > {vram:.1f} GiB",
                                hint="смените кубик или стратегию (fp8/offload)"))

    return Composition(spec=spec, cubes=tuple(cube_specs), issues=tuple(issues),
                       ports_in=frozenset(spec.inputs),
                       ports_out=frozenset(p for p, event in last_event.items()
                                           if event == "produced"),
                       bindings=tuple(bindings), provides=frozenset(provides), budget_gib=budget)


def _strategy_hint(names: Iterable[str]) -> list[str]:
    """Подсказка для UI: какие зарегистрированные кубики закрывают стратегию.

    Ищем по реестру (`provides` содержит имя), иначе — синтетическое имя `strategy.<name>`.
    """
    from .cube import all_cubes

    registry = all_cubes()
    hints: list[str] = []
    for name in sorted(names):
        known = [cube.id for cube in registry if name in cube.provides]
        hints.extend(known or [f"strategy.{name.replace('_', '.')}"])
    return hints
