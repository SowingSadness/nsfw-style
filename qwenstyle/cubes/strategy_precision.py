"""Стратегия `strategy.precision.fp8`: точность трансформера и fp8-кэш.

Кубик-декоратор (план `cube-architecture`, шаг 3): ничего не потребляет и не производит, объявляет
`provides={"fp8"}` — трансформер, которому нужна fp8-раскладка, указывает это в `requires`.

Семантика (замеры в `AGENTS.md`): `fp8` — 6.6 ГиБ на GPU и кэш `cache/models/fp8_transformer*`
(каст 26 с однократно); `bf16` — 14.2 ГиБ в VRAM, без кэша. Строка кэша попадает в имя записи,
поэтому пересборка — только явным флагом.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..backends.base import ParameterSpec, ResourceBudget
from ..cube import CubeContext, CubeSpec, Stage, register_cube

__all__ = ["CUBE", "PrecisionParams"]

TRUTHY = ("1", "true", "yes", "on")


@dataclass(frozen=True)
class PrecisionParams:
    """Разобранные параметры стратегии точности."""

    precision: str = "fp8"
    cache_dir: str | None = None
    rebuild: bool = False

    @property
    def is_fp8(self) -> bool:
        return self.precision == "fp8"

    @property
    def cache_path(self) -> Path | None:
        """Папка кэша: `None`, если кэш отключён пустой строкой или точность не fp8."""
        if not self.is_fp8 or self.cache_dir is None:
            return None
        raw = str(self.cache_dir).strip()
        return Path(raw) if raw else None

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> "PrecisionParams":
        values = {p.name: p.default for p in CUBE.params}
        values.update({k: v for k, v in (params or {}).items() if k in values})
        rebuild = values["rebuild"]
        if isinstance(rebuild, str):
            rebuild = rebuild.strip().lower() in TRUTHY
        return cls(precision=str(values["precision"]), cache_dir=values["cache_dir"],
                   rebuild=bool(rebuild))


def _self_test(ctx: CubeContext) -> None:
    """Без весов: дефолты, fp8/bf16 и отключение кэша пустой строкой."""
    default = PrecisionParams.from_mapping({})
    assert default.precision == "fp8" and default.is_fp8, default
    expected = str(CUBE.param("cache_dir").default).replace("\\", "/")
    assert default.cache_path is not None and default.cache_path.as_posix() == expected,         default.cache_path
    bf16 = PrecisionParams.from_mapping({"precision": "bf16"})
    assert not bf16.is_fp8 and bf16.cache_path is None, bf16
    off = PrecisionParams.from_mapping({"cache_dir": ""})
    assert off.cache_path is None, "пустая строка обязана отключать кэш"
    rebuild = PrecisionParams.from_mapping({"rebuild": "true"})
    assert rebuild.rebuild is True, rebuild


CUBE: CubeSpec = register_cube(CubeSpec(
    id="strategy.precision.fp8",
    stage=Stage.TRANSFORMER,
    provides=frozenset({"fp8"}),
    params=(
        ParameterSpec("precision", "cube", "enum", "fp8", choices=("bf16", "fp8"),
                      doc="fp8 = 6.6 ГиБ в VRAM и кэш каста (26 с однократно); bf16 = 14.2 ГиБ"),
        ParameterSpec("cache_dir", "cube", "str", "cache/models/fp8_transformer",
                      doc="папка fp8-кэша; пустая строка = не кэшировать"),
        ParameterSpec("rebuild", "cube", "bool", False,
                      doc="пересобрать fp8-кэш вместо чтения готового"),
    ),
    budget=ResourceBudget(weights_fp8_gib=6.6, notes="fp8 layerwise cast одного трансформера"),
    self_test=_self_test,
    doc="точность трансформера: fp8 layerwise cast + дисковый кэш (или bf16 без кэша)",
))
