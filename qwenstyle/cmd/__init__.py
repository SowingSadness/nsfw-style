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
    """`['transformer.qwen_image21.steps=8', 'encoder.qwen3vl.device=cpu']` -> поля `RunConfig`.

    Пути **канонические** — параметры кубиков (`qwenstyle/paths.py`): старых написаний вроде
    `generation.steps` больше нет, неизвестный ключ отвергается списком допустимых. Значения
    остаются строками: приведение типов делает `RunConfig.merged()`; перевод пути в поле конфига —
    `paths.config_path` (уйдёт вместе с `RunConfig` в плане `cutover`).
    """
    from ..paths import config_path, is_run_key

    out: dict[str, object] = {}
    for item in items or ():
        key, sep, value = str(item).partition("=")
        key = key.strip()
        if not sep or not key:
            raise ConfigError(f"--set ждёт вид KEY=VALUE, получено {item!r}")
        if is_run_key(key):
            out[key] = value.strip()          # запуск: путь совпадает с полем RunConfig
            continue
        try:
            mapped = config_path(key)
        except ValueError as exc:
            raise ConfigError(str(exc)) from None
        out[mapped] = value.strip()
    return out


__all__ = ["parse_sets"]
