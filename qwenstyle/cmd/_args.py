"""Разбор флагов, которые объявлены строкой, но значат «число» или «выключено целиком».

`--upscale` принимает либо множитель (`2`, `1.5`), либо явный выключатель (`null`/`none`/`off`).
Второе через `type=float` не выразить: argparse ответил бы `invalid float value`, а пустой блок
`upscale` (= апскейл выключен целиком, `UpscaleConfig()` на умолчаниях) — легальное состояние
`RunConfig`. Отдельно от `useless`: `factor: 0` сам апскейл **не** выключает, если в пресете задан
`target_height` — приоритет у высоты (`RunConfig.upscale_factor`), поэтому нужен именно `null`.

Модуль начинается с `_`, поэтому `qwenstyle/cli.py` (`discover_commands`, `if info.name.startswith("_")`)
не регистрирует его как подкоманду.
"""
from __future__ import annotations

import math

#: Значения `--upscale`, которые выключают апскейл целиком (регистр и пробелы не важны).
UPSCALE_OFF: tuple[str, ...] = ("null", "none", "off")

#: Одинаковый `--help` у `run` и `gen`: флаг объявлен в двух командах.
UPSCALE_HELP = ("число = множитель апскейла (target_height пресета главнее); "
                "null = выключить апскейл целиком")

#: Один текст на все отказы: мусор, пусто, `nan`/`inf`, отрицательное.
UPSCALE_ERR = "--upscale: ожидается число > 0 или null|none|off"


def parse_upscale(text: str | None) -> float | None:
    """`null|none|off` -> `None`; конечное число >= 0 -> `float`; всё остальное -> `ValueError`.

    `None` на входе — «флаг не задан» (вызывающий код и так проверяет `args.upscale is not None`)
    и возвращается как `None` без разбора; пустая строка и одни пробелы — ошибка, а не выключение:
    `--upscale ""` в командной строке почти наверняка опечатка, и молча выключить апскейл хуже,
    чем сказать об этом. `0` — «фактор 0», допустим (сам апскейл выключает только `null`).
    `nan`/`inf`/`-1` не числа множителя, поэтому тоже отказ.
    """
    if text is None:
        return None
    raw = str(text).strip().lower()
    if raw in UPSCALE_OFF:
        return None
    if not raw:
        raise ValueError(UPSCALE_ERR)
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(UPSCALE_ERR) from None
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(UPSCALE_ERR)
    return value


__all__ = ["UPSCALE_ERR", "UPSCALE_HELP", "UPSCALE_OFF", "parse_upscale"]
