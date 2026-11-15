"""Корневой CLI: argparse + автосборка подкоманд из `qwenstyle/cmd/*.py`.

Новая подкоманда = новый модуль в `qwenstyle/cmd/` с `NAME`, `HELP`, `add_args()`, `run()`;
этот файл при этом не правится (план `qwenstyle-framework`, шаг 1).
Запуск: `python -m qwenstyle <command> [...]`.
"""
from __future__ import annotations

import argparse
import importlib
import pkgutil
import sys
from types import ModuleType

from . import __version__
from .config import ConfigError

COMMAND_PACKAGE = "qwenstyle.cmd"


def discover_commands(package: str = COMMAND_PACKAGE) -> tuple[dict[str, ModuleType], dict[str, str]]:
    """Импортирует все `qwenstyle/cmd/*.py`.

    Возвращает (commands {name: module}, errors {module: причина}). Неимпортируемый модуль
    (например, опциональная зависимость вроде seedvr2/flux2) не роняет CLI: он просто
    попадает в errors и печатается в stderr.
    """
    commands: dict[str, ModuleType] = {}
    errors: dict[str, str] = {}
    try:
        pkg = importlib.import_module(package)
    except Exception as exc:                      # noqa: BLE001 — CLI не должен падать на импорте
        errors[package] = f"{type(exc).__name__}: {exc}"
        return commands, errors
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name.startswith("_"):
            continue
        modname = f"{package}.{info.name}"
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:                  # noqa: BLE001
            errors[modname] = f"{type(exc).__name__}: {exc}"
            continue
        name = getattr(mod, "NAME", None)
        missing = [attr for attr in ("NAME", "HELP", "add_args", "run")
                   if not hasattr(mod, attr)]
        if missing or not callable(getattr(mod, "run", None)) \
                or not callable(getattr(mod, "add_args", None)):
            errors[modname] = f"нет NAME/HELP/add_args/run (не хватает: {', '.join(missing) or 'callable'})"
            continue
        commands[str(name)] = mod
    return commands, errors


def build_parser(commands: dict[str, ModuleType]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m qwenstyle",
        description=("NSFW-style: локальная стилизация, правка и генерация на Qwen-Image-2.1; "
                     "стилевой адаптер (FUSAL R80) — опция."),
        epilog=f"qwenstyle {__version__} · подкоманда без аргументов печатает свою справку",
    )
    parser.add_argument("--version", action="version", version=f"qwenstyle {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for name in sorted(commands):
        mod = commands[name]
        help_text = str(getattr(mod, "HELP", "") or "")
        p = sub.add_parser(name, help=help_text, description=help_text,
                           prog=f"python -m qwenstyle {name}")
        mod.add_args(p)
        p.set_defaults(_run=mod.run, _command=name)
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands, errors = discover_commands()
    for modname, reason in errors.items():
        print(f"[cli] подкоманда {modname} недоступна: {reason}", file=sys.stderr)
    parser = build_parser(commands)
    if not argv:
        parser.print_help()
        return 0
    if argv[0] not in commands and not argv[0].startswith("-"):
        print(f"[cli] неизвестная подкоманда {argv[0]!r}; доступны: "
              f"{', '.join(sorted(commands)) or '—'}", file=sys.stderr)
        parser.print_help(sys.stderr)
        return 2
    args = parser.parse_args(argv)
    runner = getattr(args, "_run", None)
    if runner is None:                            # справка/-h или пустая подкоманда
        parser.print_help()
        return 0
    try:
        return int(runner(args) or 0)
    except ConfigError as exc:
        print(f"[config] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[cli] прервано", file=sys.stderr)
        return 130
    except NotImplementedError as exc:
        print(f"[todo] {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
