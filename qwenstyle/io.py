"""Имена выходных файлов и атомарное сохранение PNG (шаг 7 плана `qwenstyle-framework`).

Правила взяты из `run_style.py`/`tools/batch_seeds.py` дословно (AGENTS.md, инвариант 3):

* одиночный прогон: `--out` + `-x{factor}` (только при факторе > 0) + `-s{seed}`;
  `out/x.png` + seed 777 + factor 4 -> `out/x-x4-s777.png`. Теги **не дублируются**, если они
  уже есть в имени: `out/x-x4-s777.png` остаётся собой (повторный прогон того же кадра —
  одна и та же запись, а не `x-x4-s777-x4-s777`);
* батч: `<out-dir>/s{N}-gen.png` (чистая генерация) и `<out-dir>/s{N}.png` (после апскейла);
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
from collections.abc import Sequence
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


def seed_tag(seed: int) -> str:
    """`-s777` — по нему кадр перезапускается один в один (смена seed не трогает кэш)."""
    return f"-s{int(seed)}"


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
    """Имя файла одиночного прогона: `--out` + `-x{factor}` (если factor > 0) + `-s{seed}`.

    Отсутствующие теги дописываются, уже присутствующие не дублируются (идемпотентность:
    `out/x-x4-s777.png` при seed 777 и factor 4 вернётся без изменений).
    """
    raw = _norm(getattr(cfg, "out", "") or "")
    if not raw:
        raise ValueError("out_path: в конфиге нет --out (cfg.out пустой)")
    tags = [tag for tag in (factor_tag(upscale_factor), seed_tag(cfg.seed)) if tag]
    missing = [tag for tag in tags if tag not in raw]
    result = _insert_tags(raw, missing) if missing else raw
    # --out без расширения (например 'out/flux2-640') — дописываем DEFAULT_SUFFIX,
    # иначе PIL не знает формат при сохранении. Проверяем по списку известных расширений,
    # а не по `Path.suffix`: теги вида `-s20261113`/`-x1.5` тоже дают «suffix».
    if not result.lower().endswith(IMAGE_SUFFIXES):
        result = f"{result}{DEFAULT_SUFFIX}"
    return result


def log_dir(raw: str | Path) -> Path:
    """Папка отладочного лога прогона: каталог кадра + имя кадра без расширения.

    `out/x-s7.png` -> `out/x-s7/` (`out/x-s7/run.log` и `out/x-s7/meta.json`). Имя кадра уже
    содержит `-s{seed}` и `-x{factor}`, поэтому у каждой генерации своя папка, а лог всегда
    лежит рядом со своим кадром (см. `qwenstyle/runlog.py`).
    """
    text = _norm(raw)
    head, sep, name = _split(text)
    suffix = Path(name).suffix
    stem = name[: -len(suffix)] if suffix.lower() in IMAGE_SUFFIXES else name
    return Path(f"{head}{sep}{stem}")


def batch_paths(out_dir: str | Path, seed: int) -> tuple[str, str]:
    """`(gen, out)` для батча: `<out-dir>/s{N}-gen.png` и `<out-dir>/s{N}.png`.

    `gen` — чистая генерация (её переиспользует апскейл и `--resume` смотрит `out`).
    Возвращаются строки (см. докстринг модуля); `Path(gen)`/`Path(out)` работают.
    """
    directory = _norm(out_dir).rstrip("/")
    base = f"{directory}/" if directory else ""
    return (f"{base}s{int(seed)}-gen{DEFAULT_SUFFIX}",
            f"{base}s{int(seed)}{DEFAULT_SUFFIX}")


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
           "image_sha", "images_sha", "log_dir", "out_path", "save_atomic", "seed_tag"]
