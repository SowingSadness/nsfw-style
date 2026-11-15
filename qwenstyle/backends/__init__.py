"""Бекенды `qwenstyle`: интерфейсы + реэкспорт реестра.

Модуль бекенда — обычный файл `qwenstyle/backends/<name>.py`, который на импорте вызывает один
из декораторов (он же выполняет проверку контракта):

```python
from .base import Capabilities, ResourceBudget
from ..registry import register_generation

@register_generation
class QwenImage21:
    name = "qwen_image21"
    caps = Capabilities(conditioning="precomputed", image_inputs="single", encoder="optional",
                        supports_lora=True, precision=("bf16", "fp8"), offload=("none", "group"))
    budget = ResourceBudget(weights_bf16_gib=14.2, weights_fp8_gib=6.7, encoder_gib=17.5,
                            disk_gib=53.0)
```

`discover()` импортирует все такие модули и не падает на отсутствующих зависимостях
(SeedVR2 в отдельном venv, FLUX.2 без весов): причины попадают в `Discovered.errors`.

Имена реестра (`discover`, `register*`, `get_*`, `Discovered`, `RegistryError`) отдаются лениво
через `__getattr__`: `qwenstyle.registry` сам импортирует этот пакет (нужны `Capabilities`/
`ResourceBudget`), поэтому жадный реэкспорт дал бы цикл.
"""
from __future__ import annotations

from typing import Any

from .base import (
    Capabilities,
    ConditioningMode,
    EncoderBackend,
    EncoderRole,
    GenerationBackend,
    Hardware,
    ImageInputs,
    ResourceBudget,
    UpscaleBackend,
    hardware_budget_verdict,
    hardware_resources,
)

#: Имена из `qwenstyle/registry.py`, доступные как `qwenstyle.backends.<name>` (ленивый реэкспорт).
_REGISTRY_NAMES = frozenset({
    "Discovered", "RegistryError", "available", "discover", "generation_for_repo", "get_encoder",
    "get_generation", "get_upscale", "register", "register_encoder", "register_generation",
    "register_upscale", "reset",
})


def __getattr__(name: str) -> Any:
    if name in _REGISTRY_NAMES:
        from .. import registry

        return getattr(registry, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _REGISTRY_NAMES)


__all__ = [
    "Capabilities", "ConditioningMode", "Discovered", "EncoderBackend", "EncoderRole",
    "GenerationBackend", "Hardware", "ImageInputs", "RegistryError", "ResourceBudget",
    "UpscaleBackend", "available", "discover", "generation_for_repo", "get_encoder",
    "get_generation", "get_upscale", "hardware_budget_verdict", "hardware_resources",
    "register", "register_encoder", "register_generation", "register_upscale", "reset",
]
