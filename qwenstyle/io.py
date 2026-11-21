"""Имена выходных файлов и атомарное сохранение PNG (шаг 7 плана `qwenstyle-framework`).

Правила:

* имя кадра = `<seed>-<steps>` (без префикса `s`), поэтому seed и число шагов видны прямо
  в имени, а смена `--steps` не перезаписывает прежний кадр: `84-25.png`, `84-25-gen.png`;
* одиночный прогон (`run`) в файл-режиме: теги `-<seed>-<steps>` (+ `-x{factor}` при апскейле)
  дописываются к имени: `out/x.png` + seed 84 + 25 шагов -> `out/x-84-25.png`, с апскейлом
  `out/x-84-25-x4.png`. Теги **не дублируются**, если они уже есть в имени;
* батч/сессия: `<out-dir>/<seed>-<steps>-gen.png` (чистая генерация) и
  `<out-dir>/<seed>-<steps>.png` (после апскейла);
* PNG пишется **без** `pnginfo`: в файле не остаётся ни промта, ни параметров (приватность);
* запись атомарная: `*.tmp` в том же каталоге + `os.replace()`, иначе прерванный прогон
  оставит полуфайл, который потом молча прочитается как готовый кадр.

Почему функции возвращают `str`, а не `Path`: имена печатаются в логах и в acceptance-командах,
а `str(WindowsPath('out/x.png'))` на Windows даёт `out\\x.png`. Строки хранят путь с прямыми `/`,
поэтому печать совпадает с тем, что видно в плане/acceptance, а `Path(out_path(cfg, factor))`
работает как обычно (Windows `/` понимает).
"""
from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:                       # только типы: io не должен тянуть PIL ради имён файлов
    from PIL import Image

    from .config import RunConfig

DEFAULT_SUFFIX = ".png"
#: Sentinel `image_sha` для прогона без референса (T2I). Не hex — это осознанно: ключ кэша
#: `embeds_*` допускает любой сегмент, а `EMBEDS_RE` относится только к legacy-записям.
NOREF_SHA = "noref"
# Известные расширения картинок: нужны, чтобы отличить реальный формат от тега в имени
# (`out/x-s20261113`: Path(...).suffix == ".s20261113", но это НЕ расширение).
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


def natural_key(path: str | Path) -> tuple:
    """Естественная сортировка: `2-x` раньше `10-x`; файлы без числа — в конце.

    Референсы проекта нумеруются людьми (`1-...`, `2-...`, `10-...`), и лексикографический порядок
    поставил бы `10-` вторым — а с ним поехали бы плейсхолдеры `<imageN>` в промте.
    """
    path = Path(path)
    parts = [part for part in re.split(r"(\d+)", path.stem.lower()) if part]
    has_number = any(part.isdigit() for part in parts)
    chunks = tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts)
    return (0 if has_number else 1, chunks, path.name.lower())


def discover_images(folder: str | Path, *, suffixes: Sequence[str] = IMAGE_SUFFIXES,
                    on_duplicate: Callable[[Path, Path], None] | None = None) -> list[Path]:
    """Картинки каталога (не рекурсивно) в natural-порядке; одинаковый стем -> первый.

    Один порядок для всего проекта: `refs*/` проекта (`cmd/_project.py`) и `source.project`
    считают файлы одинаково, иначе `<image1>` в промте и conditioning разъехались бы.
    `on_duplicate(путь, уже_взятый)` — что сделать с дубликатом стема (`a.png` + `a.jpg`);
    по умолчанию просто берётся первый по natural sort.
    """
    root = Path(folder)
    if not root.is_dir():
        return []
    found = sorted((p for p in root.iterdir()
                    if p.is_file() and p.suffix.lower() in tuple(suffixes)),
                   key=natural_key)
    seen: dict[str, Path] = {}
    result: list[Path] = []
    for path in found:
        key = path.stem.casefold()
        known = seen.get(key)
        if known is not None:
            if on_duplicate is not None:
                on_duplicate(path, known)
            continue
        seen[key] = path
        result.append(path)
    return result


# --------------------------------------------------------------------------- #
# Ключ входа
# --------------------------------------------------------------------------- #

def image_sha(image_path: str | Path | Sequence[str | Path]) -> str:
    """Ключ входа: 12 hex sha256.

    Один путь — **историческое** поведение (`run_style.py`): sha256 файла, первые 12 hex
    (`imgs/a.png` -> `9f2c…`). Пустой путь = прогон без референса (T2I, план `minimax-h3-image`):
    возвращается sentinel `NOREF_SHA`, а не исключение.

    Список/кортеж путей — составной ключ (`images_sha`): `sha256("|".join(полные sha256))[:12]`.
    Список из одного элемента **обязан** совпадать с одиночным путём — иначе существующие записи
    `cache/embeds/embeds_<md5>_<sha>_<W>x<H>.pt` перестали бы попадать в кэш (AGENTS.md R1),
    поэтому в этом случае вызывается та же ветка одного файла.

    Сегмент попадает в имя `.pt`-кэша, поэтому записи T2I, одиночного кадра и набора референсов
    никогда не смешиваются.
    """
    if isinstance(image_path, (list, tuple)):
        return images_sha(image_path)
    text = str(image_path or "").strip()
    if not text:
        return NOREF_SHA
    return _file_sha256(text)[:12]


def images_sha(paths: Sequence[str | Path] | str | Path | None) -> str:
    """Составной ключ набора референсов: `sha256("|".join(full sha256 каждого файла))[:12]`.

    Порядок значим: `--input a --input b` и `--input b --input a` — разные conditioning, как и
    разные плейсхолдеры `<image1>`/`<image2>` в промте Qwen-Image-2.1. Пустой список/`None`/только
    пустые строки -> `NOREF_SHA`. Один элемент -> `image_sha(путь)` (см. докстринг выше).
    """
    if paths is None or isinstance(paths, (str, Path)):
        return image_sha(paths or "")
    items = [str(item or "").strip() for item in paths]
    items = [item for item in items if item]
    if not items:
        return NOREF_SHA
    if len(items) == 1:
        return image_sha(items[0])
    joined = "|".join(_file_sha256(item) for item in items)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def _file_sha256(path: str) -> str:
    """Полный sha256 файла (hex): в составной ключ идёт целиком, в имя кэша — первые 12."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Теги имени и пути
# --------------------------------------------------------------------------- #

def factor_tag(factor: float | None) -> str:
    """`-x4` для 4.0, `-x2.5` для 2.5, `''` при factor <= 0/None (апскейл выключен)."""
    try:
        value = float(factor or 0.0)
    except (TypeError, ValueError):
        return ""
    if value <= 0:
        return ""
    return f"-x{value:g}"               # 4.0 -> '4', 2.5 -> '2.5', 1.6 -> '1.6'


def frame_stem(seed: int, steps: int) -> str:
    """`84-25` — имя кадра: seed и число шагов через дефис (`gen` и `run`)."""
    return f"{int(seed)}-{int(steps)}"


def _norm(raw: str | Path) -> str:
    """Путь одной строкой с прямыми '/' (Windows-бэкслэши в логах читаются плохо)."""
    return str(raw).replace("\\", "/")


def _split(raw: str) -> tuple[str, str, str]:
    """('out', '/', 'x.png'): каталог, разделитель, имя файла."""
    text = _norm(raw)
    cut = text.rfind("/")
    if cut < 0:
        return "", "", text
    return text[:cut], "/", text[cut + 1:]


def _insert_tags(raw: str, tags: list[str]) -> str:
    """Вставить теги перед расширением, не меняя каталог и разделители."""
    head, sep, name = _split(raw)
    suffix = Path(name).suffix
    stem = name[: -len(suffix)] if suffix else name
    return f"{head}{sep}{stem}{''.join(tags)}{suffix}"


def out_path(cfg: "RunConfig", upscale_factor: float = 0.0) -> str:
    """Имя файла одиночного прогона: `--out` + `-<seed>-<steps>` (+ `-x{factor}`).

    Отсутствующие теги дописываются, уже присутствующие не дублируются (идемпотентность:
    `out/x-84-25.png` при seed 84 и 25 шагах вернётся без изменений).
    """
    raw = _norm(getattr(cfg, "out", "") or "")
    if not raw:
        raise ValueError("out_path: в конфиге нет --out (cfg.out пустой)")
    _head, _sep, name = _split(raw)
    stem = frame_stem(cfg.seed, cfg.steps)
    tags: list[str] = []
    # идемпотентность: `20261113-8.png` и `x-20261113-8.png` уже тегированы, второй раз не надо
    if not (name.startswith(stem) or f"-{stem}" in name):
        tags.append(f"-{stem}")
    factor = factor_tag(upscale_factor)
    if factor and factor not in name:
        tags.append(factor)
    result = _insert_tags(raw, tags) if tags else raw
    # --out без расширения (например 'out/flux2-640') — дописываем DEFAULT_SUFFIX,
    # иначе PIL не знает формат при сохранении. Проверяем по списку известных расширений,
    # а не по `Path.suffix`: теги вида `-84-25`/`-x1.5` тоже дают «suffix».
    if not result.lower().endswith(IMAGE_SUFFIXES):
        result = f"{result}{DEFAULT_SUFFIX}"
    return result


def batch_paths(out_dir: str | Path, seed: int, steps: int) -> tuple[str, str]:
    """`(gen, out)` для батча: `<out-dir>/<seed>-<steps>-gen.png` и `<out-dir>/<seed>-<steps>.png`.

    `gen` — чистая генерация (её переиспользует апскейл и `--resume` смотрит `out`).
    Возвращаются строки (см. докстринг модуля); `Path(gen)`/`Path(out)` работают.
    """
    directory = _norm(out_dir).rstrip("/")
    base = f"{directory}/" if directory else ""
    stem = frame_stem(seed, steps)
    return (f"{base}{stem}-gen{DEFAULT_SUFFIX}",
            f"{base}{stem}{DEFAULT_SUFFIX}")


# --------------------------------------------------------------------------- #
# Сохранение
# --------------------------------------------------------------------------- #

def save_atomic(image: "Image.Image", path: str | Path,
                meta: Mapping[str, Any] | None = None) -> Path:
    """Сохранить кадр атомарно и **без метаданных в PNG**.

    `meta` принимается ради вызывающего кода (что именно сгенерировано), но сознательно не
    попадает ни в `pnginfo`, ни в отдельный `.json`: иначе рядом с кадром появлялись бы файлы,
    которых не ждут ни батч (`ровно 4 файла на 2 seed'а`), ни `out/`-раскладка. Параметр
    оставлен как место для будущего opt-in-сайдкара, а не как скрытая запись метаданных.

    Запись идёт в `*.tmp` в том же каталоге (свой для каждого процесса) и подменяется через
    `os.replace()`, поэтому прерванный прогон не оставляет полуфайл вместо готового кадра.
    """
    del meta                        # см. докстринг: в PNG ничего не пишем
    target = Path(path)
    if not target.name.lower().endswith(IMAGE_SUFFIXES):   # страховка: без расширения PIL падает
        target = target.with_name(f"{target.name}{DEFAULT_SUFFIX}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.stem}.{os.getpid()}.tmp{target.suffix}")
    try:
        image.save(tmp, format="PNG")
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


__all__ = ["DEFAULT_SUFFIX", "IMAGE_SUFFIXES", "NOREF_SHA", "batch_paths", "factor_tag",
           "frame_stem", "image_sha", "images_sha", "out_path", "save_atomic"]
