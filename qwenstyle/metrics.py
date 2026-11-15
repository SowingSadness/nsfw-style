"""Метрики качества апскейла — перенос формул из `tools/upscale_compare.py`, без PIL.

Чистые функции: на входе numpy-массив (HxWx3, uint8/float) или 2D-яркость, на выходе числа и
dict. Никакого чтения/записи файлов и никакого PIL — I/O живёт в `qwenstyle/cmd/compare.py`.

Что считается (все прокси, не «настоящее» качество):

* `laplacian_energy` — средняя |4-соседний лапласиан| по яркости: резкость/микрорельеф линий;
* `detail_rms` — RMS отличия от блочного блюра 4x4: «зерно»/шум реставрации;
* `saturation` — средняя `(max-min)/max` в процентах;
* `contrast_std`, `mean_lum` — стандартное отклонение и среднее яркости.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "contrast_std", "detail_rms", "gray", "image_metrics", "laplacian_energy", "mean_lum",
    "metrics", "pair_diff", "saturation", "target_size",
]

#: Кратность целевого размера при сравнении бекендов (SeedVR2 требует кратность 16).
SIZE_MULTIPLE = 16


def gray(arr: np.ndarray) -> np.ndarray:
    """Яркость как среднее по каналам (та же формула, что в `tools/upscale_compare.py`)."""
    return np.asarray(arr, dtype=np.float32).mean(axis=2)


def laplacian_energy(g: np.ndarray) -> float:
    """Средняя абсолютная энергия 4-соседнего лапласиана; 0.0 на кадрах меньше 3x3."""
    if g.ndim != 2 or min(g.shape) < 3:
        return 0.0
    lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
    return float(np.abs(lap).mean())


def detail_rms(g: np.ndarray, block: int = 4) -> float:
    """RMS отличия яркости от блочного блюра `block`x`block` («зерно»)."""
    if g.ndim != 2 or block < 2:
        return 0.0
    h, w = g.shape
    hh, ww = (h // block) * block, (w // block) * block
    if hh < block or ww < block:
        return 0.0
    blurred = g[:hh, :ww].reshape(hh // block, block, ww // block, block).mean(axis=(1, 3))
    detail = g[:hh, :ww] - np.repeat(np.repeat(blurred, block, axis=0), block, axis=1)
    return float(np.sqrt((detail ** 2).mean()))


def saturation(arr: np.ndarray) -> float:
    """Насыщенность в процентах: средняя `(max-min)/max` по пикселям."""
    a = np.asarray(arr, dtype=np.float32)
    mx, mn = a.max(axis=2), a.min(axis=2)
    return float(((mx - mn) / (mx + 1e-6)).mean() * 100)


def contrast_std(g: np.ndarray) -> float:
    """Контраст как стандартное отклонение яркости."""
    return float(np.asarray(g, dtype=np.float32).std())


def mean_lum(g: np.ndarray) -> float:
    """Средняя яркость 0..255."""
    return float(np.asarray(g, dtype=np.float32).mean())


def image_metrics(arr: np.ndarray) -> dict[str, float]:
    """Все метрики одним dict (ключи исторические: как в `tools/upscale_compare.py`)."""
    g = gray(arr)
    return {
        "sharpness_lap": laplacian_energy(g),
        "detail_rms": detail_rms(g),
        "saturation": saturation(arr),
        "contrast_std": contrast_std(g),
        "mean_lum": mean_lum(g),
    }


#: Совместимое с `tools/upscale_compare.py` имя.
metrics = image_metrics


def pair_diff(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Попарное расхождение двух кадров одинаковой формы: mean|d|, p99, max (в уровнях 0..255)."""
    x = np.asarray(a, dtype=np.int16)
    y = np.asarray(b, dtype=np.int16)
    if x.shape != y.shape:
        raise ValueError(f"pair_diff: разные размеры {x.shape} и {y.shape}")
    diff = np.abs(x - y)
    return {
        "mean": float(diff.mean()),
        "p99": float(np.percentile(diff, 99)),
        "max": float(diff.max()),
    }


def target_size(width: int, height: int, target_height: int,
                multiple: int = SIZE_MULTIPLE) -> tuple[int, int]:
    """Целевой размер кадра по заданной высоте с сохранением пропорций.

    Ширина округляется вниз/вверх до кратности `multiple` (16) — требование препроцессинга
    SeedVR2 (`DivisibleCrop(16)`), высота берётся ровно как задана.
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"target_size: некорректный исходный размер {width}x{height}")
    if multiple > 1:
        w = int(round(target_height * width / height / multiple) * multiple)
        w = max(multiple, w)
    else:
        w = int(round(target_height * width / height))
    return w, int(target_height)
