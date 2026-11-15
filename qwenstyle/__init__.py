"""qwenstyle — пакет для NSFW-style (локальная стилизация/правка/генерация на Qwen-Image-2.1).

Стилевой адаптер FUSAL R80 — опция, не требование.

Единый источник параметров запуска и ключей кэша — `RunConfig`/`BatchConfig` из
`qwenstyle.config`; подкоманды лежат в `qwenstyle/cmd/`, бекенды — в `qwenstyle/backends/`.
Публичный API: `run(cfg)` (один кадр) и `run_batch(cfg)` (N seed'ов за одну загрузку).
Оба реэкспортируются из `qwenstyle.pipeline` **лениво** (`__getattr__`): `import qwenstyle`
остаётся дешёвым и не тянет torch/PIL, пока не вызван сам прогон.

Запуск: `python -m qwenstyle <command>` (`gen`, `run`, `selftest`, `env`, `cache`, `models`,
`compare-upscale`, `bench`, `config`).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Оффлайн по умолчанию (AGENTS.md, R3/инвариант 5): сеть — только явным HF_HUB_OFFLINE=0.
# setdefault, чтобы значения из окружения пользователя всегда были главнее.
os.environ.setdefault("HF_HOME", str(_REPO_ROOT / "hf-cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .config import (  # noqa: E402 — env-настройка должна произойти до тяжёлых импортов
    BatchConfig,
    ConfigError,
    EncoderConfig,
    GenerationConfig,
    LoraConfig,
    RunConfig,
    UpscaleConfig,
)

if TYPE_CHECKING:                                  # только типы: рантайм идёт через __getattr__
    from .pipeline import Result, run, run_batch

#: Имена, отдаваемые из `qwenstyle.pipeline` по требованию (ленивый реэкспорт).
_LAZY = {"run": "run", "run_batch": "run_batch", "Result": "Result", "Plan": "Plan"}


def __getattr__(name: str) -> Any:
    """Ленивый реэкспорт `run`/`run_batch` (+`Result`/`Plan`) из `qwenstyle.pipeline`.

    `import qwenstyle` не должен тянуть torch/diffusers: пайплайн импортируется при первом
    обращении, поэтому `qwenstyle.run(cfg)` работает, а `from qwenstyle import RunConfig` —
    мгновенный.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import pipeline

    return getattr(pipeline, target)


def __dir__() -> list[str]:
    return sorted(set(__all__))


__all__ = [
    "BatchConfig", "ConfigError", "EncoderConfig", "GenerationConfig", "LoraConfig",
    "Plan", "Result", "RunConfig", "UpscaleConfig", "__version__", "run", "run_batch",
]
