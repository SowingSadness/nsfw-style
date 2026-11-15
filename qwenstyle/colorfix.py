"""Цветокоррекция результата реставрации (SeedVR2) по исходнику — перенос `colorfix.py` 1:1.

Зачем: официальный SeedVR2 прогоняет `wavelet_reconstruction(sample, input)` перед сохранением
(5-уровневый вейвлет: низкие частоты — от исходника, высокие — от реставрации). В нашей сборке
`projects/video_diffusion_sr/color_fix.py` отсутствует (в коде он под `if os.path.exists(...)`),
поэтому вывод уезжает по яркости и насыщенности. Здесь два варианта той же идеи без basicSR:

  * `wavelet`  — низкие частоты (цвет, яркость, общий контраст) берутся из исходника,
                 высокие (детали, микротекстура) — из реставрации. Дефолт, как в оригинале.
  * `reinhard` — перегон статистик по каналам (mean/std), мягче и тупее.

Обе функции работают с PIL.Image и возвращают PIL.Image того же размера.
Формулы не менялись относительно корневого `colorfix.py`: `sigma`/`strength`/`match_saturation`
означают то же самое.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def _as_float(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _sat(arr: np.ndarray) -> float:
    """Средняя насыщенность (max-min)/max на клипнутом 0..255 массиве.

    Клип обязателен: без него в почти чёрных пикселях знаменатель уходит в ~0
    и метрика разлетается на порядки (проверено — ломало масштаб цветности).
    """
    a = np.clip(arr, 0.0, 255.0)
    mx, mn = a.max(axis=2), a.min(axis=2)
    return float(((mx - mn) / np.maximum(mx, 1e-6)).mean())


def _to_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).round().astype(np.uint8), mode="RGB")


def _blur(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Гауссово размытие по трём каналам через PIL (быстро и без scipy)."""
    return _as_float(_to_image(arr).filter(ImageFilter.GaussianBlur(radius=sigma)))


def _chroma(out: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Подтянуть среднюю насыщенность out к ref (масштаб цветности вокруг серого)."""
    k = _sat(ref) / max(_sat(out), 1e-6)
    gray = out.mean(axis=2, keepdims=True)
    return gray + (out - gray) * k


def wavelet_fix(result: Image.Image, source: Image.Image, sigma: float = 12.0,
                strength: float = 1.0, match_saturation: bool = True) -> Image.Image:
    """Низкие частоты от source, высокие от result (аналог wavelet_reconstruction).

    strength < 1 — частичная коррекция (смесь с исходным результатом реставрации).
    """
    src = source.convert("RGB").resize(result.size, Image.LANCZOS)
    rec = _as_float(result)
    ref = _as_float(src)
    detail = rec - _blur(rec, sigma)
    out = _blur(ref, sigma) + detail
    if match_saturation:
        out = _chroma(out, ref)
    if strength < 1.0:
        out = rec + (out - rec) * strength
    return _to_image(out)


def reinhard_fix(result: Image.Image, source: Image.Image) -> Image.Image:
    """Перегон mean/std по каналам (Reinhard)."""
    src = source.convert("RGB").resize(result.size, Image.LANCZOS)
    rec = _as_float(result)
    ref = _as_float(src)
    out = np.empty_like(rec)
    for c in range(3):
        r, f = rec[..., c], ref[..., c]
        std_r = r.std() or 1.0
        std_f = f.std() or 1.0
        out[..., c] = (r - r.mean()) / std_r * std_f + f.mean()
    return _to_image(out)


def apply_color_fix(result: Image.Image, source: Image.Image | None, mode: str = "wavelet",
                    sigma: float = 12.0, strength: float = 1.0,
                    match_saturation: bool = True) -> Image.Image:
    """mode: none | wavelet | reinhard (при `source=None` возвращает кадр без изменений)."""
    if mode in (None, "", "none") or source is None:
        return result
    if mode == "wavelet":
        return wavelet_fix(result, source, sigma=sigma, strength=strength,
                           match_saturation=match_saturation)
    if mode == "reinhard":
        return reinhard_fix(result, source)
    raise ValueError(f"неизвестный режим цветокоррекции: {mode!r} (none|wavelet|reinhard)")


def stats(a: Image.Image) -> dict:
    """Яркость/насыщенность/контраст кадра (для логов и сравнения бекендов)."""
    arr = np.clip(_as_float(a), 0.0, 255.0)
    return {
        "lum": float(arr.mean()),
        "saturation": _sat(arr) * 100,
        "contrast": float(arr.mean(axis=2).std()),
    }


__all__ = ["apply_color_fix", "reinhard_fix", "stats", "wavelet_fix"]
