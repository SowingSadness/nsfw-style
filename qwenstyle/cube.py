"""Контракт кубика: `Stage`, `CubeSpec`, `CubeContext` и реестр кубиков (план `cube-architecture`, шаг 1).

Кубик — это стадия пайплайна, а не модель. Он объявляет **данными**:

* какие порты потребляет (`consumes`), какие читает, но не забирает (`reads`), что производит;
* свои параметры (`ParameterSpec` — тот же дескриптор, что у бекендов, поэтому UI/`--set`
  валидируются одинаково);
* бюджет ресурсов (`ResourceBudget`) и стратегии, которые он включает/требует
  (`provides`/`requires`: `fp8`, `kv_offload`, `window_attention`, …);
* `self_test` — проверку на синтетике, без весов и без CUDA.

Реестр проверяет контракт **при регистрации** (не в рантайме): опечатка в порте или параметре
видна сразу при импорте модуля кубиков.

Сервисы (кэш, лог, планировщик памяти) кубиками не являются и в реестр не попадают — они приходят
в `CubeContext`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, ClassVar, Iterable, Mapping

from .backends.base import ParameterSpec, ResourceBudget
from .ports import PORT_TYPES

__all__ = [
    "CubeContext",
    "CubeError",
    "CubeSpec",
    "Stage",
    "all_cubes",
    "cubes_by_stage",
    "get_cube",
    "register_cube",
    "reset_registry",
]


class CubeError(LookupError):
    """Кубик не зарегистрирован / нарушен контракт регистрации."""


class Stage(str, Enum):
    """Стадия пайплайна. Порядок значений — порядок выполнения (см. `compose`)."""

    SOURCE = "source"
    REWRITE = "rewrite"
    ENCODER = "encoder"
    TRANSFORMER = "transformer"
    DECODER = "decoder"
    POST = "post"

    @property
    def order(self) -> int:
        return list(Stage).index(self)


@dataclass(frozen=True)
class CubeContext:
    """То, что кубику разрешено знать о мире: устройство, кэш, лог, сид.

    Намеренно **не** содержит `RunConfig`: кубик получает свой типизированный набор параметров
    (проверенный композитором) и этот контекст.
    """

    device: str = "auto"
    cache: Any = None
    log: Callable[[str], None] = lambda _msg: None
    seed: int = 0
    machine: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CubeSpec:
    """Декларация кубика. Всё, что нужно композитору и UI, живёт здесь."""

    id: str
    stage: Stage
    consumes: frozenset[str] = frozenset()
    produces: str | frozenset[str] | None = None
    reads: frozenset[str] = frozenset()
    params: tuple[ParameterSpec, ...] = ()
    budget: ResourceBudget = ResourceBudget()
    provides: frozenset[str] = frozenset()
    requires: frozenset[str] = frozenset()
    self_test: Callable[[CubeContext], None] | None = None
    doc: str = ""

    # -- производные свойства ---------------------------------------------- #
    @property
    def outputs(self) -> frozenset[str]:
        """Порты, которые кубик производит (одна строка — частный случай набора).

        Много портов бывает у источника: `source.project` отдаёт и `prompt`, и `refs` — они приходят
        из одного проекта, и разделять их на два кубика значило бы дважды разбирать один yaml.
        """
        if self.produces is None:
            return frozenset()
        if isinstance(self.produces, str):
            return frozenset({self.produces})
        return frozenset(self.produces)

    @property
    def is_strategy(self) -> bool:
        """Декоратор-стратегия: ничего не потребляет и не производит, только `provides`."""
        return not self.outputs and not self.consumes

    @property
    def param_names(self) -> frozenset[str]:
        return frozenset(p.name for p in self.params)

    def param(self, name: str) -> ParameterSpec | None:
        for spec in self.params:
            if spec.name == name:
                return spec
        return None

    def summary(self) -> str:
        ports_in = ", ".join(sorted(self.consumes | self.reads)) or "—"
        ports_out = "|".join(sorted(self.outputs)) or "—"
        return (f"{self.id:28} {self.stage.value:11} {ports_in} -> {ports_out}"
                f"  params={len(self.params)}"
                f"{'  provides=' + '|'.join(sorted(self.provides)) if self.provides else ''}")


# --------------------------------------------------------------------------- реестр

_REGISTRY: dict[str, CubeSpec] = {}


def _validate(spec: CubeSpec) -> None:
    """Проверка контракта при регистрации: ошибки видны при импорте, а не в прогоне."""
    if not spec.id or " " in spec.id:
        raise CubeError(f"id кубика обязан быть непустым и без пробелов: {spec.id!r}")
    if not isinstance(spec.stage, Stage):
        raise CubeError(f"{spec.id}: stage обязан быть Stage, получено {spec.stage!r}")
    unknown = (spec.consumes | spec.reads) - set(PORT_TYPES)
    if unknown:
        raise CubeError(f"{spec.id}: неизвестные порты входа {sorted(unknown)}; "
                        f"допустимые: {sorted(PORT_TYPES)}")
    unknown_out = spec.outputs - set(PORT_TYPES)
    if unknown_out:
        raise CubeError(f"{spec.id}: неизвестные порты выхода {sorted(unknown_out)}; "
                        f"допустимые: {sorted(PORT_TYPES)}")
    if not spec.outputs and spec.consumes:
        raise CubeError(f"{spec.id}: кубик с consumes обязан объявить produces")
    if spec.is_strategy and not spec.provides:
        raise CubeError(f"{spec.id}: стратегия без provides бесполезна")
    if not spec.is_strategy and not spec.outputs:
        raise CubeError(f"{spec.id}: кубик стадии {spec.stage.value} обязан объявить produces")
    names = [p.name for p in spec.params]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise CubeError(f"{spec.id}: повторяющиеся параметры {duplicates}")
    for p in spec.params:
        if p.scope not in ("cube", "generation", "sampler", "encoder", "upscale", "post"):
            raise CubeError(f"{spec.id}.{p.name}: неизвестный scope {p.scope!r}")
    if spec.requires & spec.provides:
        raise CubeError(f"{spec.id}: требует и сам предоставляет {sorted(spec.requires & spec.provides)}")


def register_cube(spec: CubeSpec) -> CubeSpec:
    """Зарегистрировать кубик. Повторная регистрация того же id — ошибка (импорт дважды?).

    Здесь же параметры со `scope="cube"` получают **полный путь с id кубика**
    (`decoder.vae.qwen21.tile_px`): без этого `--set cube.repo=…` был бы неоднозначным — `repo`
    объявляют энкодер, трансформер и оба PE-кубика. Путь — часть данных кубика (UI берёт его
    оттуда же), поэтому переписывается один раз при регистрации.
    """
    _validate(spec)
    if spec.id in _REGISTRY:
        raise CubeError(f"кубик {spec.id!r} уже зарегистрирован")
    spec = _qualify_paths(spec)
    _REGISTRY[spec.id] = spec
    return spec


def _qualify_paths(spec: CubeSpec) -> CubeSpec:
    """Проставить `--set`-пути параметрам `scope="cube"` (id кубика + имя параметра)."""
    from dataclasses import replace

    if not any(param.scope == "cube" and not param.cube_id for param in spec.params):
        return spec
    return replace(spec, params=tuple(
        replace(param, cube_id=spec.id) if param.scope == "cube" and not param.cube_id else param
        for param in spec.params))


def get_cube(cube_id: str) -> CubeSpec:
    try:
        return _REGISTRY[cube_id]
    except KeyError:
        raise CubeError(f"кубик {cube_id!r} не зарегистрирован; "
                        f"известные: {', '.join(sorted(_REGISTRY)) or '—'}") from None


def all_cubes() -> tuple[CubeSpec, ...]:
    return tuple(_REGISTRY[k] for k in sorted(_REGISTRY))


def cubes_by_stage(stage: Stage | None = None) -> tuple[CubeSpec, ...]:
    items = all_cubes()
    return tuple(c for c in items if stage is None or c.stage is stage)


def reset_registry() -> None:
    """Только для тестов: опустошить реестр (боевые модули регистрируют кубики при импорте)."""
    _REGISTRY.clear()


def _validate_iterable(specs: Iterable[CubeSpec]) -> None:      # pragma: no cover - хелпер
    for spec in specs:
        _validate(spec)
