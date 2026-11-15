"""Подкоманды `python -m qwenstyle`.

Каждый модуль этого пакета = одна подкоманда и отдаёт четыре имени:

    NAME: str                                  # имя подкоманды в CLI
    HELP: str                                  # одна строка для --help
    add_args(parser: argparse.ArgumentParser)  # объявление флагов
    run(args: argparse.Namespace) -> int       # код возврата

`qwenstyle/cli.py` собирает подкоманды сам (`discover_commands()`), поэтому новая команда =
новый файл здесь, без правок ядра. Модули с именем `_*` пропускаются; ошибка импорта модуля
не роняет CLI (попадает в список ошибок и печатается).
"""
from __future__ import annotations

from typing import Iterable

from ..config import ConfigError


def parse_sets(items: Iterable[str] | None) -> dict[str, object]:
    """`['generation.steps=8', 'encoder.name=heretic']` -> {'generation.steps': '8', ...}.

    Значения остаются строками: приведение типов делает `RunConfig.merged()`
    (единый путь для CL и --set).
    """
    out: dict[str, object] = {}
    for item in items or ():
        key, sep, value = str(item).partition("=")
        key = key.strip()
        if not sep or not key:
            raise ConfigError(f"--set ждёт вид KEY=VALUE, получено {item!r}")
        out[key] = value.strip()
    return out


__all__ = ["parse_sets"]
