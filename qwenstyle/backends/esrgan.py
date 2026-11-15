"""Real-ESRGAN x4plus_anime_6B как `UpscaleBackend` — перенос из корневого `upscale.py` 1:1.

Своя реализация `RRDBNet` на чистом torch (без `basicsr`: он не собирается под torch 2.8 /
Windows), тайловый инференс и автоскачивание весов с проверкой sha256.

Ключевые свойства, которые нельзя менять:

* **тайлы 256 / pad >= 16 дают бит-в-бит тот же кадр, что прогон всего кадра** (`_tiled`);
* x4 — родной масштаб сети; `factor` 2/1.5/… = прогнать x4 и уменьшить LANCZOS;
* веса: `cache/models/weights/RealESRGAN_x4plus_anime_6B.pth` (17.9 МиБ, sha256 `f872d837…`), при
  отсутствии качаются из официального GitHub-релиза (сеть — только в этом случае);
* `seed` не влияет на результат (сеть детерминирована) — параметр есть только ради общего
  контракта `UpscaleBackend`.

Тяжёлые зависимости (`torch`, `numpy`) импортируются лениво: `models list`/`config show` не
должны платить ~1.8 с за импорт torch. Классы сети строятся при первом обращении и доступны как
обычные атрибуты модуля (`from qwenstyle.backends.esrgan import RRDBNet`) через `__getattr__`.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ..registry import register_upscale
from .base import REPO_ROOT, ParameterSpec, ResourceBudget

if TYPE_CHECKING:                                   # только типы: PIL/torch не тянем на импорте
    from PIL import Image

    from ..config import UpscaleConfig

#: Официальный релиз Real-ESRGAN (тот же URL, что в корневом `upscale.py`).
WEIGHT_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/download/"
              "v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth")
#: sha256 файла весов (17.9 МиБ) — проверяется сразу после скачивания.
WEIGHT_SHA256 = "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da"
#: Путь по умолчанию относительно корня репозитория (а не cwd: в пакете `chdir` недопустим).
#: Шаг 4 разделил кеши: веса и квантование живут в `cache/models/`, вычисления — в `cache/embeds/`.
DEFAULT_WEIGHTS = "cache/models/weights/RealESRGAN_x4plus_anime_6B.pth"

SCALE = 4                                           # родной масштаб сети
NUM_BLOCK = 6                                       # anime_6B
DEFAULT_TILE = 256
DEFAULT_PAD = 32

_NET_NAMES = ("ResidualDenseBlock", "RRDB", "RRDBNet")
_TORCH: Any = None
_NETS: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Ленивые torch и классы сети
# --------------------------------------------------------------------------- #

def _torch() -> Any:
    """Модуль torch (импорт ~1.8 с — только когда апскейл реально нужен)."""
    global _TORCH
    if _TORCH is None:
        import torch

        _TORCH = torch
    return _TORCH


def _nets() -> dict[str, Any]:
    """Строит `ResidualDenseBlock`/`RRDB`/`RRDBNet` при первом обращении (математика 1:1)."""
    if _NETS:
        return _NETS
    torch = _torch()
    nn = torch.nn
    F = torch.nn.functional

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat: int, num_grow_ch: int = 32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb3(self.rdb2(self.rdb1(x)))
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        """x4-апскейлер Real-ESRGAN. num_block=6 — версия anime_6B."""

        def __init__(self, num_in_ch: int = 3, num_out_ch: int = 3, scale: int = 4,
                     num_feat: int = 64, num_block: int = 6):
            super().__init__()
            self.scale = scale
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            for conv in (self.conv_up1, self.conv_up2):
                feat = self.lrelu(conv(F.interpolate(feat, scale_factor=2, mode="nearest")))
            out = self.conv_last(self.lrelu(self.conv_hr(feat)))
            return out

    _NETS.update({
        "ResidualDenseBlock": ResidualDenseBlock,
        "RRDB": RRDB,
        "RRDBNet": RRDBNet,
        "torch": torch,
        "F": F,
    })
    return _NETS


def __getattr__(name: str) -> Any:
    """`RRDBNet`/`RRDB`/`ResidualDenseBlock` доступны как атрибуты, но строятся лениво."""
    if name in _NET_NAMES:
        return _nets()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------- #
# Веса
# --------------------------------------------------------------------------- #

def weights_path(path: str | Path | None = None) -> Path:
    """Путь весов: абсолютный как есть, относительный — от корня репозитория."""
    raw = Path(path or DEFAULT_WEIGHTS).expanduser()
    return raw if raw.is_absolute() else (REPO_ROOT / raw)


def sha256_file(path: str | Path) -> str:
    """sha256 файла (потоково; 17.9 МБ — можно и целиком, но так честнее)."""
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def ensure_weights(path: str | Path | None = None) -> Path:
    """Возвращает путь к весам, скачивая их один раз при отсутствии (с проверкой sha256).

    Уже лежащий файл не перепроверяется (как в корневом `upscale.py`): быстрый путь не платит
    за хеш 17.9 МБ на каждый кадр.
    """
    target = weights_path(path)
    if target.is_file():
        return target
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"    веса ESRGAN отсутствуют, качаю {WEIGHT_URL}", flush=True)
    tmp = target.with_name(target.name + ".tmp")
    urllib.request.urlretrieve(WEIGHT_URL, tmp)
    sha = sha256_file(tmp)
    if sha != WEIGHT_SHA256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 весов не совпал: {sha} != {WEIGHT_SHA256}")
    os.replace(tmp, target)
    return target


# --------------------------------------------------------------------------- #
# Инференс
# --------------------------------------------------------------------------- #

def load_upscaler(weights: str | Path, device: str = "cuda", fp16: bool = True,
                  num_block: int = NUM_BLOCK) -> Any:
    """Собирает RRDBNet и грузит weights (формат basicsr: params_ema / params)."""
    torch = _torch()
    model_cls = _nets()["RRDBNet"]
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=True)
    state = ckpt.get("params_ema") or ckpt.get("params") or ckpt
    model = model_cls(scale=SCALE, num_block=num_block)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    if fp16 and device.startswith("cuda"):
        model = model.half()
    return model


def _tiled(model: Any, x: Any, tile: int, pad: int) -> Any:
    """Тайловый forward; при `tile >= max(h, w)` это ровно прогон всего кадра."""
    scale = model.scale
    _, _, h, w = x.shape
    out = x.new_zeros((1, 3, h * scale, w * scale))
    with _torch().inference_mode():
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                py0, px0 = max(0, y0 - pad), max(0, x0 - pad)
                py1, px1 = min(h, y1 + pad), min(w, x1 + pad)
                patch = x[:, :, py0:py1, px0:px1]
                up = model(patch)
                oy, ox = (y0 - py0) * scale, (x0 - px0) * scale
                out[:, :, y0 * scale:y1 * scale, x0 * scale:x1 * scale] = \
                    up[:, :, oy:oy + (y1 - y0) * scale, ox:ox + (x1 - x0) * scale]
    return out


def upscale_image(image: "Image.Image", model: Any, factor: float = 4.0,
                  tile: int = DEFAULT_TILE, pad: int = DEFAULT_PAD) -> "Image.Image":
    """x4-апскейл тайлами; `factor != 4` — уменьшение LANCZOS до `round(w*factor)`."""
    import numpy as np
    from PIL import Image as _Image

    src = image.convert("RGB")
    torch = _torch()
    with torch.inference_mode():
        arr = torch.from_numpy(np.array(src)).permute(2, 0, 1).float().div_(255.0)[None]
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        arr = arr.to(device=device, dtype=dtype)
        out = _tiled(model, arr, tile=int(tile), pad=int(pad))
        out = out.clamp_(0, 1).mul_(255).round_().byte().to("cpu")
    res = _Image.fromarray(out[0].permute(1, 2, 0).numpy())
    if float(factor) != float(model.scale):
        # factor бывает дробным (1.5 -> 1440p из 960p): размер обязан быть целым
        size = (int(round(src.width * float(factor))), int(round(src.height * float(factor))))
        res = res.resize(size, _Image.LANCZOS)
    return res


def _device_and_dtype() -> tuple[str, bool]:
    """cuda+fp16 как в корневом `upscale.py`; без CUDA — CPU/fp32 (числа могут отличаться)."""
    try:
        return ("cuda", True) if _torch().cuda.is_available() else ("cpu", False)
    except Exception:                               # noqa: BLE001 — диагностика не критична
        return "cpu", False


# --------------------------------------------------------------------------- #
# Бекенд
# --------------------------------------------------------------------------- #

@register_upscale
class EsrganBackend:
    """Real-ESRGAN anime6B: x4-сеть + тайлы, дробный `factor` через LANCZOS."""

    name: ClassVar[str] = "esrgan"
    budget: ClassVar[ResourceBudget] = ResourceBudget(
        weights_bf16_gib=0.02, disk_gib=0.02,
        notes="Real-ESRGAN x4plus_anime_6B, 17.9 МиБ в cache/models/weights, тайлы 256/pad 32",
    )

    @staticmethod
    def parameters() -> tuple[ParameterSpec, ...]:
        """Настраиваемые параметры апскейла (шаг 9 плана публикации).

        Объявляет сам бекенд: `config show --explain-params` печатает из этого списка, а ядро
        валидирует ими `--set` для `upscale.*` (поля `UpscaleConfig`).
        """
        return (
            ParameterSpec("factor", "upscale", "float", 0.0, minimum=0.0, maximum=4.0,
                          doc="множитель; 0 = выключено; target_height главнее factor"),
            ParameterSpec("target_height", "upscale", "int", None, minimum=0,
                          doc="целевая высота кадра: factor считается из неё (1440 для strong)"),
            ParameterSpec("tile", "upscale", "int", DEFAULT_TILE, minimum=64, maximum=1024,
                          doc="размер тайла (только esrgan; 256/pad 32 — бит-в-бит)"),
            ParameterSpec("pad", "upscale", "int", DEFAULT_PAD, minimum=0, maximum=128,
                          doc="перекрытие тайлов (только esrgan; при tile=128 нужен 32)"),
            ParameterSpec("backend", "upscale", "enum", "esrgan", choices=("esrgan",),
                          doc="бекенд постобработки (seedvr2 — вне публичных профилей)"),
        )

    # -- контракт UpscaleBackend ------------------------------------------ #

    def status(self) -> tuple[bool, str]:
        """(доступен?, причина). Отсутствие весов не ошибка: они скачаются при первом апскейле."""
        path = weights_path()
        if path.is_file():
            return True, "ok"
        return False, (f"нет весов {path} (скачаются автоматически при первом апскейле: "
                       f"{WEIGHT_URL})")

    def upscale(self, image: "Image.Image", factor: float = SCALE,
                cfg: "UpscaleConfig | None" = None, seed: int = 666,
                source: "Image.Image | None" = None, tile: int | None = None,
                pad: int | None = None) -> "Image.Image":
        """Увеличить кадр: `cfg.target_height` главнее `factor`, `tile`/`pad` главнее cfg.

        `seed` и `source` в математику не входят (сеть детерминирована, цветокоррекция —
        только у SeedVR2), параметры приняты ради общего контракта.
        """
        if cfg is None:
            from ..config import UpscaleConfig

            cfg = UpscaleConfig()
        scale = _resolve_factor(image, factor, cfg)
        if scale <= 0:
            raise ValueError(f"esrgan: нужен factor > 0 или cfg.target_height, получено {scale!r}")
        path = ensure_weights(cfg.weights)
        device, fp16 = _device_and_dtype()
        use_tile = tile if tile is not None else (cfg.tile or DEFAULT_TILE)
        use_pad = pad if pad is not None else (DEFAULT_PAD if cfg.pad is None else cfg.pad)
        model = load_upscaler(path, device=device, fp16=fp16)
        try:
            return upscale_image(image, model, factor=scale,
                                 tile=int(use_tile), pad=int(use_pad))
        finally:
            del model
            _empty_cuda_cache()


def _resolve_factor(image: "Image.Image", factor: float, cfg: "UpscaleConfig") -> float:
    """Множитель: из `cfg.target_height` (главнее), иначе явный `factor`."""
    if getattr(cfg, "target_height", None):
        return float(cfg.target_height) / max(1, int(image.height))
    return float(factor or 0.0)


def _empty_cuda_cache() -> None:
    try:
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                               # noqa: BLE001 — освобождение не критично
        pass


__all__ = [
    "DEFAULT_PAD", "DEFAULT_TILE", "DEFAULT_WEIGHTS", "EsrganBackend", "RRDB", "RRDBNet",
    "ResidualDenseBlock", "SCALE", "WEIGHT_SHA256", "WEIGHT_URL", "ensure_weights",
    "load_upscaler", "sha256_file", "upscale_image", "weights_path",
]
