"""`compose` — проверка композиции кубиков из файла (`compose --spec FILE --check`).

Зачем отдельная подкоманда (план `cube-architecture`, шаг 8): UI и человек должны уметь проверить
YAML-описание пайплайна **до** запуска — без весов, без CUDA и без трейсбека Python. Ошибки
компоновщика печатаются строками `[код] сообщение`, код возврата 1.

    python -m qwenstyle compose --spec out/tests/pipeline-ok.yaml --check
    python -m qwenstyle compose --spec out/tests/pipeline-ok.yaml --check --json

Формат файла — `{name, inputs, cubes: [{id, params}]}` (см. `cubes schema` → `pipeline`).
"""
from __future__ import annotations

import argparse

from . import cubes as cubes_cmd

NAME = "compose"
HELP = "проверить композицию кубиков из файла (без весов и GPU)"


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--spec", required=True, metavar="FILE",
                        help="YAML/JSON с ключом cubes (описание стадий и их параметров)")
    parser.add_argument("--check", action="store_true",
                        help="только проверить (по умолчанию это и делается); флаг для явности в CI")
    parser.add_argument("--json", action="store_true", help="машинный формат (для UI)")


def run(args: argparse.Namespace) -> int:
    return cubes_cmd.check_file(args.spec, as_json=bool(args.json))


__all__ = ["HELP", "NAME", "add_args", "run"]
