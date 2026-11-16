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

#: Значения `--upscale`, которые выключают апскейл целиком (регистр и пробелы не важны).
UPSCALE_OFF: tuple[str, ...] = ("null", "none", "off")

#: Одинаковый `--help` у `run` и `gen`: флаг объявлен в двух командах.
UPSCALE_HELP = ("число = множитель апскейла (target_height пресета главнее); "
                "null = выключить апскейл целиком")


def parse_upscale(text: str | None) -> float | None:
    """`null|none|off` (и пусто) -> `None`; число -> `float`; мусор -> `ValueError`.

    `None` означает «выключить апскейл целиком» (вызывающий код ставит `upscale: null` в пресет),
    а не «флаг не задан»: отсутствие флага проверяется до вызова (`args.upscale is not None`).
    """
    raw = str("" if text is None else text).strip().lower()
    if not raw or raw in UPSCALE_OFF:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError("--upscale: ожидается число или null|none|off") from None


__all__ = ["UPSCALE_HELP", "UPSCALE_OFF", "parse_upscale"]
