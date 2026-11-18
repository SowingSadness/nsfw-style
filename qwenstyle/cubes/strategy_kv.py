"""Стратегия `strategy.kv_offload`: где живёт префиксный KV-кеш (текст + референсы).

Кубик-декоратор (план `cube-architecture`, шаг 3), `provides={"kv_offload"}`.

Семантика (замеры `docs/bench/kv-cache-offload.md` и `AGENTS.md`):

* `true` — K/V в VRAM (быстро; цена — 512 КиБ на префиксный токен);
* `offload` — K/V в RAM с послойной подкачкой (бит-в-бит, +12–14 % времени);
* `false` — кеша нет, префикс пересчитывается каждый шаг (~2× медленнее);
* `auto-offload-ref2` — `true` при одном референсе, `offload` при двух и более (умолчание
  `presets/base.yaml`): ровно там, где `true` падает в OOM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..backends.base import ParameterSpec, ResourceBudget
from ..cube import CubeContext, CubeSpec, Stage, register_cube

__all__ = ["CUBE", "KV_MODES", "KvParams"]

KV_MODES: tuple[str, ...] = ("auto-offload-ref2", "true", "offload", "false")


@dataclass(frozen=True)
class KvParams:
    """Разобранный режим KV-кеша."""

    mode: str = "auto-offload-ref2"

    @property
    def declared(self) -> str:
        return self.mode

    def resolve(self, refs: int) -> str:
        """Режим с учётом числа референсов (`auto-offload-ref2`)."""
        if self.mode == "auto-offload-ref2":
            return "offload" if int(refs) > 1 else "true"
        return self.mode

    @property
    def in_ram(self) -> bool:
        """Нужен ли патч offload (только для разрешённого режима)."""
        return self.mode == "offload"

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> "KvParams":
        values = {p.name: p.default for p in CUBE.params}
        values.update({k: v for k, v in (params or {}).items() if k in values})
        return cls(mode=str(values["mode"]))


def _self_test(ctx: CubeContext) -> None:
    """Без весов: разрешение auto-режима и домен значений."""
    auto = KvParams.from_mapping({})
    assert auto.resolve(1) == "true", auto.resolve(1)
    assert auto.resolve(2) == "offload", auto.resolve(2)
    assert auto.resolve(3) == "offload", auto.resolve(3)
    assert KvParams.from_mapping({"mode": "false"}).resolve(2) == "false"
    assert set(KV_MODES) == set(CUBE.param("mode").choices or ()), "домен режимов разошёлся"


CUBE: CubeSpec = register_cube(CubeSpec(
    id="strategy.kv_offload",
    stage=Stage.TRANSFORMER,
    provides=frozenset({"kv_offload"}),
    params=(
        ParameterSpec("mode", "cube", "enum", "auto-offload-ref2", choices=KV_MODES,
                      doc="true = K/V в VRAM; offload = в RAM послойно (бит-в-бит, +13 % времени); "
                          "false = без кеша (~2x медленнее); auto-offload-ref2 = true при 1 рефе, "
                          "offload при 2+"),
    ),
    budget=ResourceBudget(weights_bf16_gib=0.0, notes="K/V префикса: 512 КиБ на токен (VRAM или RAM)"),
    self_test=_self_test,
    doc="префиксный KV-кеш референсов: VRAM, RAM-offload или без кеша",
))
