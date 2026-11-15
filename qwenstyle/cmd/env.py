"""`python -m qwenstyle env` — окружение и состояние кэшей (перенос `tools/cli.py env`).

Отвечает на вопросы «чем я вообще работаю» и «сколько уже лежит на диске»:

    [python]     версия, интерпретатор
    [versions]   torch / diffusers / transformers / peft / safetensors / huggingface_hub
    [cuda]       версия CUDA, доступность, GPU, занято/всего VRAM
    [ram]        свободно/всего RAM (psutil; без него — прочерк)
    [offline]    HF_HOME, HF_HUB_OFFLINE, телеметрия, PYTORCH_CUDA_ALLOC_CONF
    [cache]      cache/embeds (записей, размер, разрешения, legacy) и отдельно cache/models
                 (fp8-квантование + веса энкодеров — не вытесняется никогда), hf-cache
    [disk]       свободно/всего на диске репозитория

Только чтение, ничего не грузит и не удаляет (AGENTS.md R2: read-only без подтверждений).
Версии импортируются лениво и по одной: отсутствующий/битый пакет печатается как «нет», а не
роняет команду.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from ..backends.base import REPO_ROOT
from ..cache import human_bytes, scan_embeds
from ..config import EMBEDS_DIR, MODELS_DIR

NAME = "env"
HELP = "версии, CUDA/GPU/VRAM/RAM, offline-режим, счётчики кэшей и свободный диск"
#: Пакеты, чьи версии важны для воспроизводимости кадров и кэшей (AGENTS.md §1).
VERSION_MODULES = ("torch", "diffusers", "transformers", "peft", "safetensors", "huggingface_hub")


# --------------------------------------------------------------------------- #
# Помощники
# --------------------------------------------------------------------------- #

def _dir_size(path: Path) -> int:
    """Суммарный размер файлов в каталоге (0, если его нет)."""
    if not path.is_dir():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            continue
    return total


def _version(module: str) -> str:
    """`<version>` или `нет (ImportError)` — импорт одного пакета, без побочных эффектов."""
    try:
        imported = __import__(module)
    except Exception as exc:                       # noqa: BLE001 — отсутствие пакета не ошибка
        return f"нет ({type(exc).__name__})"
    return str(getattr(imported, "__version__", "?"))


def _cuda_lines() -> list[str]:
    """CUDA/GPU/VRAM: строка на факт; без torch — одна строка «torch нет»."""
    try:
        import torch
    except Exception as exc:                       # noqa: BLE001
        return [f"torch нет ({type(exc).__name__}: {exc})"]
    lines = [f"cuda {torch.version.cuda} available={torch.cuda.is_available()}"]
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        lines.append(f"gpu  {torch.cuda.get_device_name(0)}  free {human_bytes(free)} / "
                     f"total {human_bytes(total)}")
    return lines


def _ram_line() -> str:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return f"свободно {human_bytes(vm.available)} из {human_bytes(vm.total)}"
    except Exception:                              # noqa: BLE001
        return "psutil нет"


def _models_cache() -> list[tuple[str, int]]:
    """Содержимое `cache/models` (путь относительно репозитория, размер в байтах).

    Кеш моделей — другая природа, чем `cache/embeds`: fp8-квантование (`fp8_transformer*`) и
    локальные веса энкодеров/апскейлера (`weights/`) пересоздаются один раз в жизни, поэтому
    `cache prune` их не трогает; пересборка — только `--rebuild-fp8-cache` (AGENTS.md, R2).
    """
    models_dir = REPO_ROOT / MODELS_DIR
    if not models_dir.is_dir():
        return []
    out: list[tuple[str, int]] = []
    for path in sorted(models_dir.iterdir()):
        try:
            size = _dir_size(path) if path.is_dir() else path.stat().st_size
        except OSError:
            continue
        out.append((path.relative_to(REPO_ROOT).as_posix(), size))
    return out


def _cache_lines() -> list[str]:
    """Счётчики кэшей: `cache/embeds` (вычисления) и `cache/models` (веса), плюс hf-cache.

    Две ветки кеша печатаются раздельно, потому что управляются по-разному: эмбеддинги
    вытесняются (`cache prune`, `cache_supersede`), а кеш моделей — никогда.
    """
    embeds_dir = REPO_ROOT / EMBEDS_DIR
    entries = scan_embeds(embeds_dir)
    total = sum(entry.size for entry in entries)
    crops = sorted({entry.crop for entry in entries})
    lines = [f"{EMBEDS_DIR.as_posix()}: {len(entries)} записей, {human_bytes(total)}, "
             f"сеток: {len(crops)} ({', '.join(crops) or '—'}) — вытесняется "
             f"(`cache prune`, `cache_supersede`)"]
    legacy = [entry for entry in entries if entry.legacy]
    if legacy:
        lines.append(f"из них по старой схеме ключа (без хеша картинки): {len(legacy)} — мёртвые, "
                     f"чистятся `python -m qwenstyle cache prune --legacy --apply`")

    models = _models_cache()
    models_total = sum(size for _, size in models)
    lines.append(f"{MODELS_DIR.as_posix()}: {len(models)} объект(ов), {human_bytes(models_total)} — "
                 f"не вытесняется никогда (пересборка: --rebuild-fp8-cache)")
    if not models:
        lines.append(f"  нет каталога {MODELS_DIR.as_posix()} (fp8-кэш/веса ещё не собраны)")
    for name, size in models:
        lines.append(f"  {name}: {human_bytes(size)}")

    hf_home = Path(os.environ.get("HF_HOME", str(REPO_ROOT / "hf-cache")))
    if hf_home.is_dir():
        lines.append(f"{hf_home}: {human_bytes(_dir_size(hf_home))}")
    else:
        lines.append(f"{hf_home}: нет каталога (веса не скачаны?)")
    return lines


def _disk_lines() -> list[str]:
    """Свободное место на диске репозитория (для скачивания весов и записи кадров)."""
    try:
        usage = shutil.disk_usage(REPO_ROOT)
    except OSError as exc:
        return [f"нет данных ({exc})"]
    return [f"{REPO_ROOT} свободно {human_bytes(usage.free)} из {human_bytes(usage.total)}"]


def _offline_lines() -> list[str]:
    """Переменные оффлайна/памяти: значения из окружения или «не задан» с ожидаемым default."""
    defaults = {
        "HF_HOME": str(REPO_ROOT / "hf-cache"),
        "HF_HUB_OFFLINE": "1 (оффлайн по умолчанию, AGENTS.md R3)",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True (на Windows не поддержан; "
                                   "замер: эффекта нет)",
    }
    lines = []
    for key, default in defaults.items():
        value = os.environ.get(key)
        lines.append(f"{key:26} {value if value is not None else '<не задан: ' + default + '>'}")
    return lines


# --------------------------------------------------------------------------- #
# Печать
# --------------------------------------------------------------------------- #

def report() -> list[str]:
    """Все строки отчёта (без печати — удобно проверять в тестах)."""
    lines = ["[python]", f"  {sys.version.split()[0]}  {sys.executable}", "[versions]"]
    lines += [f"  {module:16} {_version(module)}" for module in VERSION_MODULES]
    lines.append("[cuda]")
    lines += [f"  {line}" for line in _cuda_lines()]
    lines += ["[ram]", f"  {_ram_line()}"]
    lines += ["[offline]"] + [f"  {line}" for line in _offline_lines()]
    lines += ["[cache]"] + [f"  {line}" for line in _cache_lines()]
    lines += ["[disk]"] + [f"  {line}" for line in _disk_lines()]
    return lines


# --------------------------------------------------------------------------- #
# CLI-контракт подкоманды
# --------------------------------------------------------------------------- #

def add_args(parser: argparse.ArgumentParser) -> None:
    """Флагов нет: `env` — read-only снимок окружения."""
    parser.add_argument("--json", action="store_true", dest="env_json",
                        help="вывести те же данные как JSON (для скриптов)")


def _json_report() -> int:
    """Машиночитаемый снимок тех же данных: версии/GPU/кэши/диск/оффлайн."""
    import json

    entries = scan_embeds(REPO_ROOT / EMBEDS_DIR)
    models = _models_cache()
    usage = shutil.disk_usage(REPO_ROOT)
    data = {
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "versions": {module: _version(module) for module in VERSION_MODULES},
        "offline": {key: os.environ.get(key) for key in
                    ("HF_HOME", "HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY",
                     "PYTORCH_CUDA_ALLOC_CONF")},
        "cache": {
            "embeds_dir": EMBEDS_DIR.as_posix(),
            "embeds": len(entries),
            "embeds_bytes": sum(entry.size for entry in entries),
            "resolutions": sorted({entry.resolution for entry in entries}),
            "crops": sorted({entry.crop for entry in entries}),
            "legacy": sum(1 for entry in entries if entry.legacy),
            "models_dir": MODELS_DIR.as_posix(),
            "models_bytes": sum(size for _, size in models),
            "models": [{"path": name, "bytes": size} for name, size in models],
            "models_evictable": False,
        },
        "disk": {"repo": str(REPO_ROOT), "free": usage.free, "total": usage.total},
    }
    try:
        import torch

        data["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
                         "cuda_available": bool(torch.cuda.is_available())}
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            data["torch"]["gpu"] = torch.cuda.get_device_name(0)
            data["torch"]["vram_free"] = free
            data["torch"]["vram_total"] = total
    except Exception as exc:                       # noqa: BLE001
        data["torch"] = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


def run(args: argparse.Namespace) -> int:
    if getattr(args, "env_json", False):
        return _json_report()
    for line in report():
        print(line, flush=True)
    return 0


__all__ = ["HELP", "NAME", "VERSION_MODULES", "add_args", "report", "run"]
