"""SeedVR2-3B (ByteDance-Seed/SeedVR) как `UpscaleBackend` — перенос из `upscale.py` 1:1.

Модель живёт в **отдельном окружении** `seedvr2/` (torch 2.3 cu121 / diffusers 0.29 /
transformers 4.38), потому что её требования несовместимы с основным стеком (torch 2.8,
diffusers 0.43, transformers 5). Поэтому бекенд не импортирует её код, а запускает
`seedvr2/run_image.py` интерпретатором `seedvr2/.venv/Scripts/python.exe` через `subprocess`
(cwd = `seedvr2/SeedVR`: иначе не находятся `configs_3b/`, `ckpts/`, `pos_emb.pt`).

Что нельзя менять:

* вызов тот же: `--input/--output/--width/--height/--seed` + `--fp8` (bf16-веса 12.6 ГиБ в 16 ГБ
  не влезают);
* целевой размер кратен 16 (`DivisibleCrop(16)` в их препроцессинге) — округление как в
  корневом `upscale.py`;
* `HF_HOME=<repo>/hf-cache` в окружении дочернего процесса (оффлайн, без похода в сеть);
* после реставрации — цветокоррекция `qwenstyle/colorfix.py` (в их сборке отсутствует
  `wavelet_reconstruction`, без неё уезжают яркость и насыщенность). По умолчанию `wavelet`,
  режим берётся из `UpscaleConfig.color_fix`; `none` — без коррекции.

Ничего не скачивает: `status()` только проверяет наличие venv/репозитория/весов.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from ..colorfix import apply_color_fix
from ..registry import register_upscale
from .base import REPO_ROOT, ResourceBudget

if TYPE_CHECKING:                                   # только типы: PIL не тянем на импорте
    from PIL import Image

    from ..config import UpscaleConfig

#: Каталог изолированного окружения и репозитория SeedVR.
SEEDVR2_DIR = REPO_ROOT / "seedvr2"
SEEDVR2_REPO = SEEDVR2_DIR / "SeedVR"
SEEDVR2_PY = SEEDVR2_DIR / ".venv" / "Scripts" / "python.exe"
SEEDVR2_INFER = SEEDVR2_REPO / "projects" / "video_diffusion_sr" / "infer.py"
SEEDVR2_CKPT = SEEDVR2_REPO / "ckpts" / "seedvr2_ema_3b.pth"
SEEDVR2_VAE = SEEDVR2_REPO / "ckpts" / "ema_vae.pth"
SEEDVR2_RUNNER = SEEDVR2_DIR / "run_image.py"
#: Кэш весов HF для дочернего процесса (тот же, что у проекта: оффлайн, R3).
HF_HOME = REPO_ROOT / "hf-cache"

#: Кратность целевого кадра (DivisibleCrop(16) в препроцессинге SeedVR2).
SIZE_MULTIPLE = 16
DEFAULT_SEED = 666

INSTALL_HINT = ("Установить: python seedvr2/setup.py   (~17 ГБ, torch 2.3 cu121 + веса 3B bf16)\n"
                "Либо оставить по умолчанию: --upscale-backend esrgan")


def _target_size(image: "Image.Image", factor: float) -> tuple[int, int]:
    """Целевой размер, кратный 16 (как `round(w * factor / 16) * 16` в `upscale.py`)."""
    w = max(SIZE_MULTIPLE, round(image.width * float(factor) / SIZE_MULTIPLE) * SIZE_MULTIPLE)
    h = max(SIZE_MULTIPLE, round(image.height * float(factor) / SIZE_MULTIPLE) * SIZE_MULTIPLE)
    return int(w), int(h)


@register_upscale
class Seedvr2Backend:
    """SeedVR2-3B через внешний venv: реставрация + цветокоррекция по исходнику."""

    name: ClassVar[str] = "seedvr2"
    budget: ClassVar[ResourceBudget] = ResourceBudget(
        weights_bf16_gib=12.6, weights_fp8_gib=3.64, disk_gib=17.0,
        notes=("отдельное окружение seedvr2/ (torch 2.3 cu121), веса 3B bf16 + VAE; "
               "fp8-хранение весов включено всегда (bf16 на 16 ГБ OOM)"),
    )
    #: venv + репозиторий + веса, которые обязаны существовать (для `models probe`/диагностики).
    required_paths: ClassVar[tuple[Path, ...]] = (SEEDVR2_PY, SEEDVR2_INFER, SEEDVR2_CKPT)

    # -- контракт UpscaleBackend ------------------------------------------ #

    def status(self) -> tuple[bool, str]:
        """(доступен?, причина). Перечисляет ровно то, чего не хватает."""
        missing: list[str] = []
        if not SEEDVR2_PY.is_file():
            missing.append(str(SEEDVR2_PY))
        if not SEEDVR2_INFER.is_file():
            missing.append(str(SEEDVR2_INFER))
        if not SEEDVR2_CKPT.is_file():
            missing.append(str(SEEDVR2_CKPT))
        if not SEEDVR2_RUNNER.is_file():
            missing.append(str(SEEDVR2_RUNNER))
        if missing:
            return False, "не хватает: " + ", ".join(missing)
        return True, "готов"

    def missing_help(self, reason: str) -> str:
        """Текст ошибки при недоступном бекенде (как `RuntimeError` в корневом `upscale.py`)."""
        return f"SeedVR2 не установлен ({reason}).\n{INSTALL_HINT}"

    def upscale(self, image: "Image.Image", factor: float = 2.0,
                cfg: "UpscaleConfig | None" = None, seed: int = DEFAULT_SEED,
                source: "Image.Image | None" = None, fp8: bool = True) -> "Image.Image":
        """Прогнать SeedVR2 до `image.size * factor` (или до `cfg.target_height`) и вернуть кадр.

        `source` — кадр для цветокоррекции (по умолчанию сам вход): низкие частоты/статистики
        берутся у него, высокие — у реставрации.
        """
        if cfg is None:
            from ..config import UpscaleConfig

            cfg = UpscaleConfig()
        scale = _resolve_factor(image, factor, cfg)
        if scale <= 0:
            raise ValueError(f"seedvr2: нужен factor > 0 или cfg.target_height, получено {scale!r}")

        ok, why = self.status()
        if not ok:
            raise RuntimeError(self.missing_help(why))

        width, height = _target_size(image, scale)
        with tempfile.TemporaryDirectory(prefix="seedvr2_") as tmp:
            src = Path(tmp) / "in.png"
            dst = Path(tmp) / "out.png"
            image.convert("RGB").save(src)
            cmd = [str(SEEDVR2_PY), str(SEEDVR2_RUNNER), "--input", str(src),
                   "--output", str(dst), "--width", str(width), "--height", str(height),
                   "--seed", str(seed)]
            if fp8:
                cmd.append("--fp8")          # bf16-веса 12.6 ГиБ + активации в 16 ГБ не влезают
            env = dict(os.environ, HF_HOME=str(HF_HOME))
            print("    SeedVR2: " + " ".join(cmd), flush=True)
            subprocess.run(cmd, cwd=str(SEEDVR2_REPO), check=True, env=env)
            if not dst.is_file():
                raise RuntimeError("SeedVR2 не создал выходной файл")
            out = _open_rgb(dst)

        mode = getattr(cfg, "color_fix", "none")
        if mode and mode != "none":
            reference = source if source is not None else image
            out = apply_color_fix(out, reference, mode=mode,
                                 sigma=float(getattr(cfg, "color_sigma", 12.0)),
                                 strength=float(getattr(cfg, "color_strength", 1.0)))
            print(f"    цветокоррекция {mode} (strength {getattr(cfg, 'color_strength', 1.0)}, "
                  f"sigma {getattr(cfg, 'color_sigma', 12.0)})", flush=True)
        return out


def _resolve_factor(image: "Image.Image", factor: float, cfg: "UpscaleConfig") -> float:
    """Множитель: из `cfg.target_height` (главнее), иначе явный `factor`."""
    if getattr(cfg, "target_height", None):
        return float(cfg.target_height) / max(1, int(image.height))
    return float(factor or 0.0)


def _open_rgb(path: str | Path) -> Any:
    """Читает PNG из временного каталога целиком (файл живёт только до выхода из `with`)."""
    from PIL import Image as _Image

    with _Image.open(path) as im:
        return im.convert("RGB").copy()


def status() -> tuple[bool, str]:
    """Функциональная обёртка над `Seedvr2Backend.status()` (удобно для скриптов)."""
    return Seedvr2Backend().status()


__all__ = [
    "DEFAULT_SEED", "HF_HOME", "INSTALL_HINT", "SEEDVR2_CKPT", "SEEDVR2_DIR", "SEEDVR2_INFER",
    "SEEDVR2_PY", "SEEDVR2_REPO", "SEEDVR2_RUNNER", "SEEDVR2_VAE", "Seedvr2Backend", "status",
]
