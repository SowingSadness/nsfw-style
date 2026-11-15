"""Реестр бекендов: регистрация в модулях + `discover()` с изоляцией ошибок импорта.

Подстановка модели = новый файл `qwenstyle/backends/<name>.py` с одним из декораторов
(`@register_generation`, `@register_upscale`, `@register_encoder`). Ядро (`pipeline.py`, CLI) знает
только имена: `get_generation(cfg.generation.backend)`.

Ключевое свойство: `discover()` НЕ пробрасывает ошибки импорта. Отсутствующие опциональные
зависимости (seedvr2 в отдельном venv, FLUX.2 без весов, torch-расширения) попадают в
`Discovered.errors`, а CLI продолжает работать.
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Literal

from .backends.base import Capabilities, ResourceBudget

BACKENDS_PACKAGE = "qwenstyle.backends"
#: Модули пакета бекендов, которые не являются бекендами и не импортируются как они.
SKIP_MODULES = frozenset({"base"})
#: Методы, без которых бекенд бесполезен (ровно контракты из `backends/base.py`).
REQUIRED_METHODS: dict[str, tuple[str, ...]] = {
    "generation": ("prepare", "encode", "generate", "release", "probe"),
    "upscale": ("status", "upscale"),
    "encoder": ("load", "encode"),
}

Kind = Literal["generation", "upscale", "encoder"]
KINDS: tuple[Kind, ...] = ("generation", "upscale", "encoder")


class RegistryError(LookupError):
    """Бекенд не зарегистрирован / неизвестный kind."""


@dataclass(frozen=True)
class Discovered:
    """Снимок реестра: зарегистрированные классы по видам + ошибки импорта модулей."""

    generation: dict[str, type] = field(default_factory=dict)
    upscale: dict[str, type] = field(default_factory=dict)
    encoder: dict[str, type] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def table(self, kind: Kind) -> dict[str, type]:
        return {"generation": self.generation, "upscale": self.upscale,
                "encoder": self.encoder}[kind]

    def names(self, kind: Kind) -> list[str]:
        return sorted(self.table(kind))

    @property
    def empty(self) -> bool:
        return not (self.generation or self.upscale or self.encoder)

    def to_dict(self) -> dict[str, Any]:
        """Для `models list --json`: имена и ключевые поля caps/budget."""
        out: dict[str, Any] = {}
        for kind in KINDS:
            out[kind] = {
                name: {
                    "caps": getattr(cls, "caps", None).to_dict()
                    if isinstance(getattr(cls, "caps", None), Capabilities) else None,
                    "budget": getattr(cls, "budget", None).to_dict()
                    if isinstance(getattr(cls, "budget", None), ResourceBudget) else None,
                    "repos": list(getattr(cls, "repos", ()) or ()),
                    "cache_tag": getattr(cls, "cache_tag", None),
                    "requires_vision": getattr(cls, "requires_vision", None),
                }
                for name, cls in sorted(self.table(kind).items())
            }
        out["errors"] = dict(sorted(self.errors.items()))
        return out


_TABLES: dict[str, dict[str, type]] = {kind: {} for kind in KINDS}
_MODULES: dict[str, ModuleType] = {}
_ERRORS: dict[str, str] = {}
_DISCOVERED = False


# --------------------------------------------------------------------------- #
# Регистрация
# --------------------------------------------------------------------------- #

def _check_kind(kind: str) -> Kind:
    if kind not in KINDS:
        raise RegistryError(f"неизвестный вид бекенда {kind!r}; допустимые: {', '.join(KINDS)}")
    return kind          # type: ignore[return-value]


def _register_cls(kind: str, cls: type) -> type:
    kind = _check_kind(kind)
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise TypeError(f"{cls.__module__}.{cls.__qualname__}: нужен атрибут name: ClassVar[str]")
    name = name.strip()
    missing = [m for m in REQUIRED_METHODS[kind] if not callable(getattr(cls, m, None))]
    if missing:
        raise TypeError(f"{name} (kind={kind}): нет методов {', '.join(missing)} "
                        f"(контракт: qwenstyle/backends/base.py)")
    caps = getattr(cls, "caps", None)
    if caps is not None and not isinstance(caps, Capabilities):
        raise TypeError(f"{name}: caps должен быть Capabilities, получено {type(caps).__name__}")
    budget = getattr(cls, "budget", None)
    if budget is not None and not isinstance(budget, ResourceBudget):
        raise TypeError(f"{name}: budget должен быть ResourceBudget, получено {type(budget).__name__}")
    _TABLES[kind][name] = cls
    return cls


def register(kind: str, cls: type | None = None):
    """`@register("generation")` / `register("generation", MyBackend)`; см. три декоратора ниже."""
    if cls is not None:
        return _register_cls(kind, cls)

    def deco(target: type) -> type:
        return _register_cls(kind, target)

    return deco


def register_generation(cls: type | None = None):
    """`@register_generation` (или `@register_generation()`) для модели генерации."""
    return register("generation", cls)


def register_upscale(cls: type | None = None):
    """`@register_upscale` для бекенда апскейла."""
    return register("upscale", cls)


def register_encoder(cls: type | None = None):
    """`@register_encoder` для текст-энкодера."""
    return register("encoder", cls)


# --------------------------------------------------------------------------- #
# Обнаружение
# --------------------------------------------------------------------------- #

def discover(package: str = BACKENDS_PACKAGE, reload: bool = False) -> Discovered:
    """Импортирует все модули пакета бекендов (кроме `base`, `__init__` и `_*`).

    Ошибки импорта не пробрасываются: `qwenstyle.backends.seedvr2` без внешнего venv или
    `flux2` без весов просто окажутся в `errors`. `reload=True` переимпортирует уже загруженные
    модули (нужно после правки бекенда в текущем процессе).
    """
    global _DISCOVERED
    try:
        pkg = importlib.import_module(package)
    except Exception as exc:                      # noqa: BLE001 — CLI не должен падать на импорте
        _ERRORS[package] = f"{type(exc).__name__}: {exc}"
        _DISCOVERED = True
        return _snapshot()

    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_") or info.name in SKIP_MODULES:
            continue
        modname = f"{package}.{info.name}"
        try:
            if reload and modname in sys.modules:
                mod = importlib.reload(sys.modules[modname])
            else:
                mod = importlib.import_module(modname)
        except Exception as exc:                  # noqa: BLE001 — опциональные зависимости
            _ERRORS[modname] = f"{type(exc).__name__}: {exc}"
            continue
        _ERRORS.pop(modname, None)
        _MODULES[modname] = mod

    _DISCOVERED = True
    return _snapshot()


def _snapshot() -> Discovered:
    return Discovered(
        generation=dict(_TABLES["generation"]),
        upscale=dict(_TABLES["upscale"]),
        encoder=dict(_TABLES["encoder"]),
        errors=dict(sorted(_ERRORS.items())),
    )


def _ensure_discovered() -> None:
    if not _DISCOVERED:
        discover()


def get_generation(name: str) -> type:
    """Класс бекенда генерации по `cfg.generation.backend`."""
    return _get("generation", name)


def get_upscale(name: str) -> type:
    """Класс бекенда апскейла по `cfg.upscale.backend`."""
    return _get("upscale", name)


def get_encoder(name: str) -> type:
    """Класс текст-энкодера по `cfg.encoder.name`."""
    return _get("encoder", name)


def _get(kind: Kind, name: str) -> type:
    kind = _check_kind(kind)
    _ensure_discovered()
    table = _TABLES[kind]
    if name in table:
        return table[name]
    errors = "".join(f"\n  {mod}: {reason}" for mod, reason in sorted(_ERRORS.items()))
    raise RegistryError(
        f"бекенд {kind} {name!r} не зарегистрирован; доступны: {', '.join(sorted(table)) or '—'}"
        + (f"\nошибки импорта модулей бекендов:{errors}" if errors else "")
    )


def available(kind: Kind) -> list[str]:
    """Имена зарегистрированных бекендов вида `kind`."""
    _ensure_discovered()
    return sorted(_TABLES[_check_kind(kind)])


def generation_for_repo(repo: str) -> tuple[str, type] | None:
    """Бекенд, объявивший `repos = (repo, ...)` — для `models probe <repo>`."""
    _ensure_discovered()
    target = str(repo).strip().lower()
    for name, cls in sorted(_TABLES["generation"].items()):
        repos = getattr(cls, "repos", ()) or ()
        if target in {str(r).strip().lower() for r in repos}:
            return name, cls
    return None


def reset() -> None:
    """Чистое состояние реестра (нужно только тестам/REPL, не CLI)."""
    global _DISCOVERED
    for table in _TABLES.values():
        table.clear()
    _MODULES.clear()
    _ERRORS.clear()
    _DISCOVERED = False


__all__ = [
    "BACKENDS_PACKAGE", "Capabilities", "Discovered", "KINDS", "Kind", "REQUIRED_METHODS",
    "RegistryError", "ResourceBudget", "SKIP_MODULES", "available", "discover",
    "generation_for_repo", "get_encoder", "get_generation", "get_upscale", "register",
    "register_encoder", "register_generation", "register_upscale", "reset",
]
