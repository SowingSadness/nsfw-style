"""Бекенд генерации MiniMax-H3-Image (T2I + edit, 6 фиксированных шагов).

Модель входит как адаптер (R6): ядро знает её только по имени `minimax_h3_image`, вся специфика
живёт здесь. Шаг 4 плана `minimax-h3-image`: `prepare`/`encode`/`generate`/`release` работают на
модульном пайплайне diffusers (`MiniMaxH3ImageModularPipeline`), `probe` читает hf-cache без torch.

Как собирается пайплайн (и почему не `from_pretrained`):
в снапшоте **нет** `modular_model_index.json`, а `model_index.json` описывает нативные компоненты
(`transformer/` содержит только `config.json`, класс `MiniMaxH3DiTModel`), поэтому
`ModularPipeline.from_pretrained` получил бы пустые веса. Компоненты грузятся вручную
(`transformer_diffusers/` через `precision.py`, `vae_diffusers/`, `scheduler/`, `tokenizer/`,
`processor/`), а пайплайн собирается из **официальных блоков** диффузеров
(`SequentialPipelineBlocks.from_blocks_dict`) — своего сэмплера, паковки и расписания здесь нет.
Блоков два набора, потому что текст-энкодер (Qwen3-VL-32B, 62.1 GiB) не должен работать на каждом
прогоне:

* `ENCODE_BLOCKS` = `before_encode` + `text_encoder` → промт (и референсы) в `prompt_embeds` /
  `text_token_tags` (кэш `text_embeds`, R1);
* `GENERATE_BLOCKS` = `before_encode` + `vae_encoder` + `denoise` + `decode` → принимает
  `prompt_embeds`/`text_token_tags` как обычные входы, поэтому попадание в кэш не грузит энкодер
  вовсе. Режим (`text2image` или `image_conditioned`) пайплайн выбирает сам по наличию `image`.

Порядок загрузки зависит от режима (`prepare` -> `encode` -> `generate`):

* попадание в кэш эмбеддингов (энкодер не нужен) — как на шаге 4: fp8-трансформер -> размещение
  (`group offload`) -> VAE на GPU -> `generate`;
* промах (энкодер нужен) — сначала только фаза `[3/4]` (в отдельном процессе, см. ниже), а
  трансформер грузится уже в `generate`, после неё. Причина — commit-charge Windows: на этой
  машине **mmap файла списывается в commit целиком** (замер: `safe_open` трёх шардов энкодера =
  +62 GiB commit при rss 0.4 GiB), а `precision.load_cached` держит **~65 GiB** под 30.9 GiB
  fp8-весов (commit 48.4 -> 113.2 GiB при rss 0.7 GiB). Commit-limit здесь 115–125 GiB, поэтому
  обе фазы в **одном** процессе не сходятся: первый прогон падал на `safe_open` с
  `OSError: Файл подкачки слишком мал (os error 1455)`, а после разведения по времени —
  access violation уже при загрузке VAE (занято 123.3/125.0 GiB).

Поэтому фаза `[3/4]` идёт в **отдельном процессе** (`encode` -> `_encode_child_main`): её commit
(62 GiB mmap + pinned-стейджинг) умирает вместе с процессом, а родитель встречает fp8-трансформер
с чистым адресным пространством и платит ровно те же 113/118 GiB, что и проверенный прогон шага 4.
Энкодер собирает `prepare` и только в этом процессе (`_is_encode_child`): сборочный код один,
`handle.encode_pipe` остаётся единственным путём кодирования (своей склейки промта в бекенде нет).

Почему такие объявления (проверено по установленному снапшоту, см. `.pi/runs/…/plan.md`):

* `conditioning="internal"` — у трансформера **нет cross-attention**: текст, референсы и кадр идут
  одной «упакованной» последовательностью (`hidden_states`, `encoder_hidden_states`,
  `timestep_indices`, `text_indices`, `text_dim=5120`). Поэтому промт нельзя «зашить» в `.pt` так, как это делает
  Qwen-Image-2.1: пайплайн принимает его целиком. Кэшируется только этап `text_embeds`
  (эмбеддинги Qwen3-VL-32B после 50-го слоя).
* `image_inputs="multi"` + `requires_image=False` — `model_index.json` объявляет задачи `t2i` и
  `edit`, а модульный пайплайн сам выбирает режим (`AutoPipelineBlocks` с
  `block_trigger_inputs = ["image", None]`): есть референс — image_conditioned, нет — text2image.
  `requires_image=False` поэтому разрешает `run` без `--input` (проверка — `pipeline.input_problem`).
* `fixed_steps=6` — не рекомендация: `pdd_config.json` задаёт 6 шагов и сигмы
  `[1.0, 0.9675…, 0.0]`, а `pdd_step_index` выбирает выходную голову (`pdd_fused_heads.safetensors`).
  `num_inference_steps` пайплайн игнорирует.
* `supports_lora=False` — FUSAL R80 обучен на Qwen-Image-2.1 (другая архитектура), Turbo-адаптеров нет.
* `precision=("bf16", "fp8")` — 33.13B bf16 (61.7 GiB) в 16 GiB VRAM не влезает; fp8-кэш
  (`qwenstyle/precision.py`) даёт 30.9 GiB, что тоже больше VRAM, поэтому рабочий режим —
  `offload="group"` (блоки живут в RAM, на GPU одна группа).

Тяжёлые импорты (`torch`, `diffusers`, `transformers`, `safetensors`) не импортируются на уровне
модуля: `models list`/`models probe` и импорт бекенда обязаны работать без CUDA и без torch (как у
`qwen_image21`/`flux2`).

Отсутствие текст-энкодера (шаг 5 плана) — `EncoderUnavailableError`: он наследует и `RuntimeError`
(контракт бекенда), и `ConfigError`, поэтому `run` печатает одну строку `[run] …` с именем/репо/
тегом энкодера и `exit 2` без трейсбека — до загрузки 30.9 GiB fp8-весов.
"""
from __future__ import annotations

import gc
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace as _replace
from pathlib import Path
from typing import Any, ClassVar

from ..config import ConfigError
from ..registry import RegistryError, get_encoder, register_generation
from .base import GIB, REPO_ROOT, Capabilities, ParameterSpec, ResourceBudget, as_image_list

#: Репозиторий пайплайна: `transformer_diffusers/`, `vae_diffusers/`, `pdd_*`, `processor/`,
#: `tokenizer/`, `scheduler/`, `model_index.json`. Веса уже установлены в `hf-cache`.
DEFAULT_REPO = "MiniMaxAI/MiniMax-H3-Image"
#: Снапшот, на котором сняты все числа в агентском плане (probe показывает коммит сам).
KNOWN_COMMIT = "5f90473023975e6271b0ed1dd84c4e88e4bbc6d3"

#: Компоненты, без которых бекенд не соберётся. Именно `*_diffusers/` (нативные `transformer/`
#: и `vae/` из репозитория — другие веса, их бекенд не читает; `transformer/` у нас содержит
#: только `config.json`).
REQUIRED_COMPONENTS: tuple[str, ...] = (
    "model_index.json",
    "pdd_config.json",
    "pdd_fused_heads.safetensors",
    "transformer_diffusers",
    "vae_diffusers",
    "processor",
    "tokenizer",
    "scheduler",
)

#: Параметры из `pdd_config.json`, если файла нет (значения установленного снапшота).
DEFAULT_STEPS = 6
DEFAULT_SIGMAS: tuple[float, ...] = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75,
                                    0.5327739119529724, 0.23200887441635132, 0.0)
#: Текст: hidden state после 50-го decoder layer Qwen3-VL-32B (hidden 5120) — не настраивается.
TEXT_HIDDEN_DIM = 5120
TEXT_LAYER_INDEX = 49

#: Маркер дочернего процесса фазы `[3/4]` (см. `encode` и `_encode_child_main`).
ENCODE_CHILD_ENV = "QWENSTYLE_H3_ENCODE_CHILD"

#: Минимальная площадь канвы, при которой пайплайн рисует ровно запрошенный размер.
#: Умолчание блоков (`2048*2048`) на 512p подняло бы канву до 2048x2048 — это не тот прогон,
#: который просили (`min_generation_area=1024` = «всегда в запрошенном размере»).
DEFAULT_MIN_GENERATION_AREA = 1024

#: Блоки генерации: всё после текст-энкодера (`prompt_embeds` приходит из кэша или `encode`).
GENERATE_BLOCKS: tuple[str, ...] = ("before_encode", "vae_encoder", "denoise", "decode")
#: Блоки энкода: подготовка референсов (edit) + текст-энкодер Qwen3-VL-32B.
ENCODE_BLOCKS: tuple[str, ...] = ("before_encode", "text_encoder")


# --------------------------------------------------------------------------- #
# Handle
# --------------------------------------------------------------------------- #

@dataclass
class MiniMaxH3ImageHandle:
    """Загруженные пайплайны + состояние, нужное `encode`/`generate`/`release`.

    `pipe` — генерация (`GENERATE_BLOCKS`), `encode_pipe` — только текст-энкодер
    (`ENCODE_BLOCKS`, создаётся лишь при промахе кэша `text_embeds`). Оба пайплайна держат
    одни и те же объекты компонентов, поэтому трансформер в памяти один.
    """

    pipe: Any = None
    encode_pipe: Any = None
    transformer: Any = None
    text_encoder: Any = None
    repo: str = DEFAULT_REPO
    fp8: bool = False
    steps: int = DEFAULT_STEPS
    sigmas: tuple[float, ...] = DEFAULT_SIGMAS
    precision_cache: Path | None = None
    snapshot: Path | None = None
    #: тег энкодера (часть ключа кэша) и признак стриминговой загрузки (его можно отпустить)
    encoder_tag: str = ""
    encoder_stream: bool = False
    #: `prepare` отложил трансформер до `generate` (промах кэша эмбеддингов: энкодер идёт первым)
    deferred: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Локальный hf-cache (для probe; без torch)
# --------------------------------------------------------------------------- #

def _hf_cache_dirs() -> list[Path]:
    """Каталоги hub-кэша: HF_HUB_CACHE -> HF_HOME/hub -> <repo>/hf-cache/hub."""
    candidates: list[Path] = []
    raw = os.environ.get("HF_HUB_CACHE")
    if raw:
        candidates.append(Path(raw))
    home = os.environ.get("HF_HOME")
    if home:
        candidates.append(Path(home) / "hub")
    candidates.append(REPO_ROOT / "hf-cache" / "hub")
    unique: list[Path] = []
    seen: list[Path] = []
    for path in candidates:
        key = path.resolve()
        if key in seen:
            continue
        seen.append(key)
        unique.append(path)
    return unique


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _dir_size(path: Path) -> int:
    """Размер каталога/файла с переходом по симлинкам `blobs/` (схема hf-cache, AGENTS.md R5)."""
    if path.is_file():
        try:
            return int(path.stat().st_size)
        except OSError:
            return 0
    total = 0
    for item in sorted(path.rglob("*")):
        if item.is_dir():
            continue
        try:
            total += int(item.stat().st_size)
        except OSError:
            continue
    return total


def _local_snapshot(repo: str) -> dict[str, Any] | None:
    """Снапшот репозитория из hf-cache: путь, коммит и `model_index.json`.

    Как `qwen_image21._local_snapshot`, но без `libcst`-тонкостей: H3-`model_index.json` пишет
    компоненты списками из 3 элементов (`[library, class, {type_hint…}]`), поэтому второй элемент
    берётся по индексу, а лишние игнорируются. Ничего не импортирует и не качает.
    """
    for cache_dir in _hf_cache_dirs():
        root_dir = Path(cache_dir) / ("models--" + repo.replace("/", "--"))
        snapshots = root_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        ref = root_dir / "refs" / "main"
        commit = None
        if ref.is_file():
            try:
                commit = ref.read_text(encoding="utf-8").strip() or None
            except OSError:
                commit = None
        dirs = sorted(p for p in snapshots.iterdir() if p.is_dir())
        if not dirs:
            continue
        root = next((p for p in dirs if p.name == commit), dirs[0])
        return {"cache_dir": str(cache_dir), "commit": root.name, "snapshot": root,
                "model_index": _read_json(root / "model_index.json")}
    return None


def _components_of(model_index: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Компоненты из `model_index.json`: `[library, class, {…}]` (3-й элемент необязателен)."""
    out: dict[str, dict[str, Any]] = {}
    for name, value in (model_index or {}).items():
        if name.startswith("_") or not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        out[str(name)] = {"library": str(value[0]), "class_name": str(value[1])}
    return out


def _pdd(path: Path) -> dict[str, Any]:
    """Шаги и сигмы из `pdd_config.json`; отсутствие файла — не ошибка (значения снапшота)."""
    data = _read_json(path)
    steps = DEFAULT_STEPS
    sigmas: tuple[float, ...] = DEFAULT_SIGMAS
    if data:
        try:
            steps = int(data.get("num_inference_steps", DEFAULT_STEPS))
        except (TypeError, ValueError):
            steps = DEFAULT_STEPS
        raw = data.get("sigmas")
        if isinstance(raw, (list, tuple)) and raw:
            try:
                sigmas = tuple(float(v) for v in raw)
            except (TypeError, ValueError):
                sigmas = DEFAULT_SIGMAS
    return {"path": str(path), "present": path.is_file(), "steps": steps,
            "sigmas": list(sigmas)}


# --------------------------------------------------------------------------- #
# Ленивые импорты и мелочи (torch/diffusers только внутри функций)
# --------------------------------------------------------------------------- #

def _torch() -> Any:
    import torch

    return torch


def _sampler_kwarg(cfg: Any, key: str, default: Any) -> Any:
    kwargs = getattr(cfg.generation, "sampler_kwargs", None) or {}
    return kwargs.get(key, default)


def _sampler_int(cfg: Any, key: str, default: int) -> int:
    try:
        return int(_sampler_kwarg(cfg, key, default))
    except (TypeError, ValueError):
        return int(default)


def _rebuild_requested(cfg: Any) -> bool:
    """`--rebuild-fp8-cache` / `rebuild_precision_cache` живут в `generation.sampler_kwargs`."""
    kwargs = getattr(cfg.generation, "sampler_kwargs", None) or {}
    for key in ("rebuild_precision_cache", "rebuild_fp8_cache"):
        value = kwargs.get(key)
        if isinstance(value, str):
            if value.strip().lower() in ("1", "true", "yes", "on"):
                return True
        elif value:
            return True
    return False


def _precision_cache_dir(cfg: Any) -> Path | None:
    """`generation.precision_cache` (относительный путь — от корня репозитория)."""
    raw = str(getattr(cfg.generation, "precision_cache", "") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _snapshot_root(cfg: Any) -> Path:
    """Локальный снапшот репозитория пайплайна (оффлайн, R3): без него бекенд не собирается."""
    repo = str(cfg.generation.repo or DEFAULT_REPO)
    info = _local_snapshot(repo)
    if info is None:
        raise RuntimeError(
            f"нет локального снапшота {repo} в hf-cache "
            f"(HF_HOME={os.environ.get('HF_HOME', '') or '—'}): скачай веса явно "
            f"(`python -m qwenstyle models download {repo} --yes`) — "
            f"в оффлайне бекенд ничего не качает (R3)"
        )
    root = Path(info["snapshot"])
    missing = [name for name in ("transformer_diffusers", "vae_diffusers", "scheduler",
                                 "tokenizer", "processor") if not (root / name).is_dir()]
    if missing:
        raise RuntimeError(f"{root}: нет компонентов {', '.join(missing)} — снапшот неполный")
    return root


# --------------------------------------------------------------------------- #
# Текст-энкодер (Qwen3-VL-32B) — шаг 5 плана: ошибки отсутствующего энкодера
# --------------------------------------------------------------------------- #

class EncoderUnavailableError(ConfigError, RuntimeError):
    """Энкодера нет в реестре/на диске — прогон не может начаться вовсе.

    Наследники выбраны сознательно: `RuntimeError` — контракт бекенда (так этот случай ловит
    `tests/test_h3_image_caps.py`, и так же падает `pipeline._preflight_encoder`), а `ConfigError`
    (`ValueError`) — то, что `cmd/run.py`/`cli.py` печатают **одной строкой** (`[run] …`, exit 2)
    вместо трейсбека: отсутствующий 62 GiB-энкодер — это состояние пресета, а не баг внутри кадра.
    """


def _encoder_unregistered_message(cfg: Any) -> str:
    """Точное объяснение, чего не хватает для генерации (шаг 5: веса 32B-энкодера)."""
    enc = cfg.encoder
    return (
        f"текст-энкодер {enc.name!r} не зарегистрирован (`register_encoder`, шаг 5 плана "
        f"`minimax-h3-image`): MiniMax-H3-Image читает hidden state Qwen3-VL-32B (hidden 5120, "
        f"vocab 151936, `hidden_states[50]`), а все энкодеры проекта — 8B (hidden 4096), они "
        f"не подходят по ширине `text_dim`.\n"
        f"Нужны веса {enc.repo} (62.1 GiB bf16, тег {enc.cache_tag}) и класс энкодера в "
        f"qwenstyle/backends/encoders.py:\n"
        f"  snapshot_download(repo_id='{enc.repo}', "
        f"allow_patterns=['model-*.safetensors', '*.index.json', 'config.json', "
        f"'generation_config.json'])\n"
        f"До этого генерация H3-Image невозможна: fp8-каст 33B-трансформера и декод VAE "
        f"проверяются без энкодера (структурный smoke — out/tests/h3i-probe.py --stage generate)."
    )


class _PinnedEncoderConfig:
    """Прокси `RunConfig`, в котором `encoder_pin_memory` — не принудительный `False`.

    `RunConfig.encoder_pin_memory` выключает pinned-staging энкодера **всегда** при
    `generation.offload=group`: в pinned-память ложатся 30.9 GiB fp8-весов трансформера, а на
    Windows это тот же ресурс, что VRAM (AGENTS.md §4). В отложенном пути (`_prepare_for_encode`)
    трансформера в памяти ещё нет, а стриминг 62.1 GiB 32B-энкодера **без** pinned-staging на этой
    машине падает с `Windows fatal exception: access violation` в продюсере
    (`safetensors.get_tensor`, traceback — `streaming.py:470 __getitem__`, exit 0xC0000005):
    неблокирующая копия H2D из pageable-памяти живёт дольше своего буфера. Замер на 218 токенах:
    `pin=0` — падение через ~40 s, `pin=1` — encode **122 s** и `prompt_embeds (1, 218, 5120)`.

    Прокси, а не правка `config.py`/`encoders.py`: правило шага — чинить только в этом файле, а
    `RunConfig` не должен знать, что бекенд грузит энкодер раньше трансформера.
    """

    def __init__(self, cfg: Any) -> None:
        object.__setattr__(self, "_cfg", cfg)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_cfg"), name)

    @property
    def encoder_pin_memory(self) -> bool:
        """Явное `encoder.pin_memory` пресета (по умолчанию `True`), без поправки на offload."""
        return bool(getattr(self._cfg.encoder, "pin_memory", True))


def _is_encode_child() -> bool:
    """Этот процесс — рабочий фазы `[3/4]` (запущен `_encode_in_child`)? """
    return os.environ.get(ENCODE_CHILD_ENV, "").strip() not in ("", "0")


def _encode_child_main(job_path: str) -> int:
    """Точка входа дочернего процесса фазы `[3/4]`: `prepare(need_encoder=True)` + `encode`.

    Задание — pickle-файл `{cfg, prompt, images, out}` (см. `_encode_in_child`): `RunConfig`
    забирает с собой пресет, значит дочерний процесс собирает **тот же** энкодер (та же
    ревизия/локальный каталог/dtype/stream/prefetch), что и родитель, и пишет результат
    в `out` тем же форматом, который ждёт `pipeline._conditioning`.
    """
    job = pickle.loads(Path(job_path).read_bytes())
    cfg = job["cfg"]
    sync = bool(job.get("sync"))
    if sync:
        cfg = _sync_encoder_cfg(cfg)
        print("    [3/4] запасной режим: синхронный стриминг энкодера (prefetch=0, без pinned)",
              flush=True)
    images: list[Any] = []
    paths = job.get("images") or ([job["image"]] if job.get("image") else [])
    if paths:
        from PIL import Image

        images = [Image.open(path) for path in paths]   # как `pipeline._plan`: без convert()
    backend = MiniMaxH3ImageBackend()
    handle = backend.prepare(cfg, need_encoder=True)
    try:
        payload = backend.encode(cfg, handle, images, str(job["prompt"]))
    finally:
        backend.release(handle)
    _torch().save(payload, job["out"])
    return 0


def _sync_encoder_cfg(cfg: Any) -> Any:
    """Конфиг с синхронным стримингом энкодера (`prefetch=0`, без pinned-стейджинга).

    Запасной режим фазы `[3/4]` (второй заход `_encode_in_child`): конвейер с pinned-буферами и
    неблокирующими H2D на этой машине падает `CUDA error: resource already mapped` / access
    violation, когда рядом лежат 62 GiB mmap весов (Windows/WDDM, AGENTS.md §4). Синхронный путь
    читает модуль, копирует и сразу считает — медленнее на доли секунды (узкое место — диск,
    0.3–0.5 GB/s), но не требует ни pinned-памяти, ни фонового потока.

    `device` берётся уже разрешённым: `EncoderConfig.resolved()` вернул бы `prefetch=4` обратно
    для `auto`+cuda, а явный `device` этого не делает.
    """
    enc = cfg.encoder.resolved()
    enc = _replace(enc, prefetch=0, pin_memory=False)
    return _replace(cfg, encoder=enc)


def _encoder_preflight(cfg: Any) -> None:
    """Проверка энкодера **до** загрузки 30.9 GiB: нет регистрации/весов — понятная ошибка."""
    name = str(cfg.encoder.name or "").strip()
    try:
        cls = get_encoder(name)
    except RegistryError as exc:
        raise EncoderUnavailableError(_encoder_unregistered_message(cfg)) from exc
    status = getattr(cls, "status", None)
    if callable(status):
        try:
            ok, why = cls().status(cfg)
        except Exception:                        # noqa: BLE001 — статус не должен ронять прогон
            ok, why = True, ""
        if not ok:
            raise EncoderUnavailableError(f"текст-энкодер {name} недоступен: {why}")


def _load_text_encoder(cfg: Any) -> Any:
    """`Qwen3VLForConditionalGeneration` из реестра энкодеров (стриминг весов — там же)."""
    name = str(cfg.encoder.name or "").strip()
    try:
        cls = get_encoder(name)
    except RegistryError as exc:
        raise EncoderUnavailableError(_encoder_unregistered_message(cfg)) from exc
    print(f"    энкодер {name}: загрузка {cfg.encoder.repo} "
          f"(device={cfg.encoder.device}, stream={cfg.encoder.stream})", flush=True)
    started = time.time()
    model = cls().load(cfg)
    print(f"    энкодер {name} загружен за {time.time() - started:.1f}s", flush=True)
    return model


# --------------------------------------------------------------------------- #
# Компоненты и сборка модульного пайплайна
# --------------------------------------------------------------------------- #

def _load_vae_and_scheduler(root: Path, torch: Any) -> dict[str, Any]:
    """VAE (2.2 GiB, 24 латентных канала) и планировщик из `*_diffusers/`-раскладки снапшота."""
    from diffusers import AutoencoderKLMiniMaxH3Image, MiniMaxH3Scheduler

    vae = AutoencoderKLMiniMaxH3Image.from_pretrained(
        str(root / "vae_diffusers"), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True,
    )
    scheduler = MiniMaxH3Scheduler.from_pretrained(str(root / "scheduler"))
    return {"vae": vae, "scheduler": scheduler}


def _load_text_components(root: Path) -> dict[str, Any]:
    """Токенизатор/процессор остаются из репозитория H3-Image (интерфейс энкодера меняется, они — нет)."""
    from transformers import Qwen2TokenizerFast, Qwen3VLProcessor

    return {
        "tokenizer": Qwen2TokenizerFast.from_pretrained(str(root / "tokenizer")),
        "processor": Qwen3VLProcessor.from_pretrained(str(root / "processor")),
    }


def _blocks(names: tuple[str, ...]) -> dict[str, Any]:
    """Инстансы официальных блоков diffusers по именам.

    Своего сэмплера, паковки последовательности и расписания здесь нет сознательно (R6):
    6 шагов и `pdd_step_index` — это `MiniMaxH3ImageDenoiseStep` с головами `pdd_proj_out`.
    """
    from diffusers.modular_pipelines.minimax_h3_image.modular_blocks_minimax_h3_image import (
        MiniMaxH3ImageAutoBeforeEncodeStep,
        MiniMaxH3ImageAutoDenoiseStep,
        MiniMaxH3ImageAutoTextEncoderStep,
        MiniMaxH3ImageAutoVaeEncoderStep,
        MiniMaxH3ImageDecodeStep,
    )

    available = {
        "before_encode": MiniMaxH3ImageAutoBeforeEncodeStep,
        "text_encoder": MiniMaxH3ImageAutoTextEncoderStep,
        "vae_encoder": MiniMaxH3ImageAutoVaeEncoderStep,
        "denoise": MiniMaxH3ImageAutoDenoiseStep,
        "decode": MiniMaxH3ImageDecodeStep,
    }
    return {name: available[name]() for name in names}


def _modular_pipeline(names: tuple[str, ...]) -> Any:
    """`MiniMaxH3ImageModularPipeline` из подмножества блоков (свой сэмплер не нужен)."""
    from diffusers import MiniMaxH3ImageModularPipeline
    from diffusers.modular_pipelines.modular_pipeline import SequentialPipelineBlocks

    blocks = SequentialPipelineBlocks.from_blocks_dict(_blocks(names))
    return MiniMaxH3ImageModularPipeline(blocks=blocks)


def _load_transformer(cfg: Any, root: Path, *, fp8: bool) -> tuple[Any, Path | None]:
    """Трансформер 33.13B: fp8 из дискового кэша (`precision.py`) либо bf16 из снапшота.

    fp8-веса не считаются напрямую: `enable_layerwise_casting` вешает хуки, которые
    апкастят модуль в bf16 на forward и возвращают fp8 на место. Набор «точных» модулей тот же,
    что у `cast_and_cache` (объявления класса), иначе проверка кэша и хуки разъедутся.
    """
    torch = _torch()
    from diffusers import MiniMaxH3ImageTransformer2DModel

    source = root / "transformer_diffusers"
    if not (source / "config.json").is_file():
        raise RuntimeError(f"{source}: нет config.json — снапшот неполный")
    if not fp8:
        print("[precision] bf16-трансформер (61.7 GiB в RAM) — режим для другой карты/сравнения",
              flush=True)
        model = MiniMaxH3ImageTransformer2DModel.from_pretrained(
            str(source), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            local_files_only=True,
        )
        return model, None

    from .. import precision                     # лёгкий модуль: torch внутри функций

    cache = _precision_cache_dir(cfg)
    if cache is None:
        raise RuntimeError(
            "generation.precision_cache пуст: каст 33B на каждом старте стоит 62 GiB чтения и "
            "30.9 GiB записи — укажи каталог кэша (cache/models/fp8_transformer_h3_image)"
        )
    if _rebuild_requested(cfg):
        print(f"[precision] пересборка запрошена, но кэш {cache} переиспользуется: удалять "
              f"30.9 GiB внутри бекенда — не его дело (R1); сотри каталог вручную", flush=True)
    started = time.time()
    precision.cast_and_cache(MiniMaxH3ImageTransformer2DModel, source, cache)
    model = precision.load_cached(MiniMaxH3ImageTransformer2DModel, cache)
    model.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn,
                                  compute_dtype=torch.bfloat16)
    print(f"    трансформер готов за {time.time() - started:.1f}s (хуки upcast fp8->bf16 включены)",
          flush=True)
    return model, cache


def _place_transformer(cfg: Any, pipe: Any, torch: Any) -> None:
    """Размещение: `offload=group` либо целиком на GPU (30.9 GiB fp8 в 16 GiB не влезают)."""
    transformer = pipe.transformer
    offload = str(cfg.generation.offload or "none").strip().lower()
    if offload == "group":
        blocks = _sampler_int(cfg, "offload_blocks", 1)
        stream = bool(_sampler_int(cfg, "offload_stream", 1))
        # pinned-память на Windows — тот же ресурс, что VRAM (AGENTS.md §4): 30.9 GiB fp8
        # пинуем группами на месте загрузки (`low_cpu_mem_usage=1`), а не целиком заранее.
        low_cpu = bool(_sampler_int(cfg, "offload_low_cpu_mem", 1))
        print(f"    размещение: transformer group-offload (блоков в группе {blocks}, "
              f"stream={stream}, low_cpu_mem={low_cpu}), vae -> cuda", flush=True)
        transformer.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=blocks,
            use_stream=stream,
            non_blocking=True,
            low_cpu_mem_usage=low_cpu,
        )
    else:
        print("    размещение: transformer -> cuda целиком (для fp8 30.9 GiB нужен "
              "--offload group)", flush=True)
        transformer.to("cuda")


def _frame_size(cfg: Any, image: Any) -> tuple[int, int]:
    """Размер кадра из `--size`/`--aspect` — тем же `RunConfig.crop`, что и план прогона."""
    in_w, in_h = image.size if image is not None else (0, 0)
    width, height = cfg.crop(int(in_w), int(in_h))
    return int(width), int(height)


# --------------------------------------------------------------------------- #
# Бекенд
# --------------------------------------------------------------------------- #

@register_generation
class MiniMaxH3ImageBackend:
    """MiniMax-H3-Image: T2I и edit одним пайплайном, 6 фиксированных шагов (PDD).

    `conditioning="internal"` + `requires_image=False`: ядро не требует `--input` (проверка —
    `pipeline.input_problem`), а промт и референсы уходят в пайплайн целиком; кэш `.pt` — только
    для `text_embeds` (эмбеддинги Qwen3-VL-32B, hidden 5120, `hidden_states[50]`).

    `probe` работает без torch (только `stat`/`read_text` по hf-cache).
    """

    name: ClassVar[str] = "minimax_h3_image"
    #: Репозиторий пайплайна (`models probe <repo>` -> `generation_for_repo`).
    repos: ClassVar[tuple[str, ...]] = (DEFAULT_REPO,)
    #: Этап кэша conditioning: эмбеддинги текста (`cache/embeds/text_embeds_<…>.pt`), картинки не кэшируются.
    cache_stage: ClassVar[str] = "text_embeds"
    caps: ClassVar[Capabilities] = Capabilities(
        conditioning="internal",          # packed-последовательность: текст + референсы + кадр
        image_inputs="multi",             # референсов несколько: по латенту на каждый (edit)
        max_images=10,                    # как у Qwen-Image-2.1: больше 10 блоки не рассчитаны
        requires_image=False,             # T2I разрешён: image-блоки пайплайн пропускает сам
        encoder="required",               # Qwen3-VL-32B (hidden 5120); без него новый промт не закодить
        fixed_steps=DEFAULT_STEPS,        # pdd_config.json: 6 шагов, сигмы фиксированы
        supports_lora=False,              # FUSAL R80 — адаптер Qwen-Image-2.1, тут неприменим
        precision=("bf16", "fp8"),
        offload=("none", "group"),
        max_pixels=None,                  # предел площади в конфиге модели не объявлен
        extra_sampling=("height", "width"),   # пайплайн принимает размер кадра явно
    )
    budget: ClassVar[ResourceBudget] = ResourceBudget(
        weights_bf16_gib=61.7,            # transformer_diffusers, 33.13B bf16 (12 шардов)
        weights_fp8_gib=30.9,             # fp8-кэш `qwenstyle/precision.py` (30.93 GiB, 17 шардов)
        encoder_gib=62.1,                 # Qwen3-VL-32B bf16 — только через `stream=True` (>63.6 GiB RAM)
        disk_gib=64.0,                    # весь снапшот в hf-cache (`du`: 64 GiB, замер на месте)
        notes=("33.13B трансформер: 61.7 GiB bf16 -> 30.9 GiB fp8 (дисковый кэш обязателен); "
               "fp8-веса всё равно больше 16 GiB VRAM, поэтому рабочий режим — offload=group; "
               "текст-энкодер Qwen3-VL-32B (62.1 GiB bf16) считается только стримингом по слоям "
               "(`encoder.stream=true`, иначе в 63.6 GiB RAM вместе с активациями не влезает); "
               "VAE 2.2 GiB / 584M, 24 латентных канала; 6 шагов задаёт pdd_config.json"),
    )

    @staticmethod
    def parameters() -> tuple[ParameterSpec, ...]:
        """Настраиваемые параметры модели (шаг 9 плана публикации).

        `steps` объявлен `readonly`: расписание PDD (`pdd_config.json`) задаёт ровно 6 шагов и
        сигмы, `num_inference_steps` пайплайн игнорирует — CLI предупреждает об этом, вместо того
        чтобы молча принять `--set generation.steps=N`. `guidance` не объявлен вовсе: у H3 нет
        guider'а и `negative_prompt`, CFG-ветки в пайплайне нет.
        """
        caps = MiniMaxH3ImageBackend.caps
        return (
            ParameterSpec("steps", "generation", "int", caps.fixed_steps, readonly=True,
                          minimum=caps.fixed_steps, maximum=caps.fixed_steps,
                          doc="PDD: 6 шагов и сигмы фиксированы pdd_config.json (--steps игнорируется)"),
            ParameterSpec("offload", "generation", "enum", "group", choices=tuple(caps.offload),
                          doc="group = блоки стримятся с CPU (fp8-веса 30.9 GiB в 16 GiB VRAM)"),
            ParameterSpec("precision", "generation", "enum", "fp8", choices=tuple(caps.precision),
                          doc="fp8 = кэш cache/models/fp8_transformer_h3_image (30.93 GiB)"),
            ParameterSpec("max_images", "generation", "int", caps.max_images, readonly=True,
                          doc="предел модели: столько референсов принимает edit "
                              "(Capabilities.max_images)"),
            ParameterSpec("offload_blocks", "sampler", "int", 1, minimum=1, maximum=64,
                          doc="блоков трансформера на GPU одновременно (offload=group)"),
            ParameterSpec("offload_stream", "sampler", "bool", True,
                          doc="асинхронная префетч-загрузка блоков (offload=group)"),
            ParameterSpec("offload_low_cpu_mem", "sampler", "bool", True,
                          doc="экономный вариант group offload (меньше pinned-памяти)"),
            ParameterSpec("min_generation_area", "sampler", "int", DEFAULT_MIN_GENERATION_AREA,
                          minimum=256,
                          doc="минимальная площадь кадра (px²), ниже которой размер поднимается"),
            ParameterSpec("rebuild_precision_cache", "sampler", "bool", False,
                          doc="пересобрать fp8-кэш вместо чтения готового (202.8 s каста)"),
            ParameterSpec("rebuild_fp8_cache", "sampler", "bool", False,
                          doc="алиас rebuild_precision_cache (флаг --rebuild-fp8-cache)"),
        )

    # -- загрузка / размещение -------------------------------------------- #

    def prepare(self, cfg: Any, need_encoder: bool, *,
                announce: bool = True) -> MiniMaxH3ImageHandle:
        """Сборка: при промахе кэша — сначала энкодер, при попадании — сразу трансформер.

        Порядок загрузки в двух режимах разный, и это не случайность:

        * **промах кэша эмбеддингов** (`need_encoder=True`) — только текст-энкодер (`encode` идёт
          следующим шагом ядра), трансформер откладывается до `generate` (`_ensure_pipeline`).
          Замер на этой машине: `precision.load_cached` занимает **~65 GiB commit** под 30.9 GiB
          fp8-весов (commit 48.4 -> 113.2 GiB при rss 0.7 GiB, commit-limit 118.5 GiB), а
          стримингу 62.1 GiB энкодера нужны те же адреса — до правки первый прогон падал на
          `safe_open` шарда с `OSError: Файл подкачки слишком мал (os error 1455)`. После
          `del model` commit возвращается к 51.5 GiB, т.е. лечится именно порядок, а не объём
          (смена `offload_stream`/`low_cpu_mem` даёт те же 114.8 GiB).
        * **попадание в кэш** (`need_encoder=False`) — порядок шага 4 без изменений: fp8-трансформер
          -> размещение (`group offload`) -> VAE на GPU.

        Энкодер (62.1 GiB) грузится **лишь** при промахе кэша и **после** проверки, что он вообще
        есть: иначе прогон упал бы через 20 минут каста (R4 — сначала цена, потом работа).

        `announce=False` — не печатать `[1/4]` (повторный вызов из `_ensure_pipeline`).
        """
        resolved = cfg.encoder.resolved()
        if resolved is not cfg.encoder:
            cfg = _replace(cfg, encoder=resolved)
        if need_encoder:
            _encoder_preflight(cfg)              # падаем до весов, а не после каста

        root = _snapshot_root(cfg)
        fp8 = str(cfg.generation.precision or "").strip().lower() == "fp8"
        cache = _precision_cache_dir(cfg) if fp8 else None
        child = _is_encode_child()
        if announce and not child:          # в дочернем процессе эти строки уже напечатал родитель
            print(f"[1/4] пайплайн MiniMaxH3ImageModularPipeline (блоки diffusers) из {root}",
                  flush=True)

        if need_encoder:
            # Трансформер (30.9 GiB fp8 = ~65 GiB commit) здесь намеренно НЕ грузится: 62.1 GiB
            # энкодера стримятся с диска в те же адреса и падают с `os error 1455` (docstring).
            if not child:
                print("[2/4] трансформер 33.13B: отложен до генерации — 30.9 GiB fp8 держат "
                      "~65 GiB commit, а 62.1 GiB энкодера стримятся с диска в то же время "
                      "(Windows: os error 1455 иначе)", flush=True)
            handle = MiniMaxH3ImageHandle(
                repo=str(cfg.generation.repo or DEFAULT_REPO),
                fp8=fp8,
                steps=DEFAULT_STEPS,
                sigmas=DEFAULT_SIGMAS,
                precision_cache=cache,
                snapshot=root,
                encoder_tag=str(cfg.encoder.cache_tag or ""),
                deferred=True,
            )
            if not child:
                # Фаза `[3/4]` пойдёт отдельным процессом (`encode` -> `_encode_in_child`): здесь
                # её энкодер собирать незачем — 62 GiB его commit нужны только там.
                return handle
            # Дочерний процесс: собираем энкодер ровно так, как это делал in-process путь.
            text_parts = _load_text_components(root)
            # pinned-staging включается явно (прокси): трансформера с group-offload в памяти
            # ещё нет, а без pin стриминг 32B падает access violation (см. `_PinnedEncoderConfig`).
            text_encoder = _load_text_encoder(_PinnedEncoderConfig(cfg))
            encode_pipe = _modular_pipeline(ENCODE_BLOCKS)
            encode_pipe.register_components(text_encoder=text_encoder, **text_parts)
            from .. import streaming

            handle.encode_pipe = encode_pipe
            handle.text_encoder = text_encoder
            handle.encoder_stream = bool(streaming.is_streamed(text_encoder))
            return handle

        torch = _torch()
        print(f"[2/4] трансформер 33.13B: "
              + (f"fp8-кэш {cache}" if fp8 else "bf16 из снапшота"), flush=True)
        transformer, fp8_dir = _load_transformer(cfg, root, fp8=fp8)

        started = time.time()
        diffusers_parts = _load_vae_and_scheduler(root, torch)
        text_parts = _load_text_components(root)
        pipe = _modular_pipeline(GENERATE_BLOCKS)
        pipe.register_components(transformer=transformer, **diffusers_parts)
        sigmas = tuple(float(v) for v in pipe.config.sample_sigmas)
        # сигм всегда на одну больше, чем шагов (`[1.0 … 0.0]` = 7 сигм = 6 шагов PDD), поэтому
        # число шагов — `len(sigmas) - 1`, а не длина списка (иначе в лог уходило «7 шагов»).
        steps = max(1, len(sigmas) - 1)
        print(f"    компоненты (vae/scheduler/tokenizer/processor) за {time.time() - started:.1f}s; "
              f"шагов {steps} (сигм {len(sigmas)}; совпадают с pdd_config.json: "
              f"{list(sigmas) == list(DEFAULT_SIGMAS)})", flush=True)

        _place_transformer(cfg, pipe, torch)
        pipe.vae.to("cuda")
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"    vram после загрузки: занято {(total - free) / GIB:.1f}/{total / GIB:.1f} GiB",
              flush=True)

        return MiniMaxH3ImageHandle(
            pipe=pipe,
            transformer=transformer,
            repo=str(cfg.generation.repo or DEFAULT_REPO),
            fp8=fp8,
            steps=steps,
            sigmas=sigmas,
            precision_cache=fp8_dir,
            snapshot=root,
            encoder_tag=str(cfg.encoder.cache_tag or ""),
        )

    def _ensure_pipeline(self, cfg: Any, handle: MiniMaxH3ImageHandle) -> Any:
        """Пайплайн генерации; если `prepare` его отложил — собрать здесь (после энкода).

        Переиспользуется тот же `prepare(need_encoder=False)` (без повторной печати `[1/4]`),
        поэтому сборка и размещение описаны в одном месте.
        """
        if handle.pipe is not None:
            return handle.pipe
        if handle.snapshot is None:
            raise RuntimeError("пайплайн отложен, но путь снапшота не сохранён в handle")
        print("    энкодер отпущен -> сборка генерации (трансформер подгружается сейчас)",
              flush=True)
        built = self.prepare(cfg, need_encoder=False, announce=False)
        handle.pipe = built.pipe
        handle.transformer = built.transformer
        handle.steps = built.steps
        handle.sigmas = built.sigmas
        handle.precision_cache = built.precision_cache
        handle.deferred = False
        del built
        return handle.pipe

    # -- conditioning ------------------------------------------------------ #

    def encode(self, cfg: Any, handle: MiniMaxH3ImageHandle, image: Any,
               prompt: str) -> dict[str, Any]:
        """Промт (+ референсы) -> `prompt_embeds` (hidden 5120) и `text_token_tags`.

        `image` — список референсов (одиночная `PIL.Image` тоже принимается, `base.as_image_list`);
        блоки `image_conditioned` формируют метку `Picture i: ` и vision-блок для **каждого**
        референса и считают по латенту на каждый — многореференсный режим есть в самом пайплайне,
        поэтому кодируем именно блоками пайплайна, а не своей склейкой.

        Формат `.pt`-кэша — R1: ключевые тензоры лежат под теми же именами, что и у других
        бекендов (`prompt_embeds` есть в `pipeline._SHAPE_KEYS`), плюс `prompt` дописывает
        оркестратор. Режим (t2i/edit) выбирают сами блоки по наличию `image`.

        **Где считается.** Если `prepare` собрал энкодер в этом процессе (так делает только
        дочерний процесс фазы, `_is_encode_child`) — считаем здесь; иначе (родитель) запускаем
        `_encode_in_child`. Это не «оптимизация»: 62.1 GiB энкодера дают +62 GiB commit на mmap
        шардов, а следующий шаг — fp8-трансформер с его ~65 GiB; одновременно они не влезают
        (commit-limit 115–125 GiB, см. docstring модуля).
        """
        images = as_image_list(image)
        if handle.encode_pipe is None or handle.text_encoder is None:
            return self._encode_in_child(cfg, images, prompt)
        torch = _torch()
        mode = (f"edit: {len(images)} референсов + промт" if len(images) > 1
                else ("edit: референс + промт" if images else "t2i: промт"))
        print(f"[3/4] текстовые эмбеддинги Qwen3-VL-32B (hidden 5120, hidden_states[50], {mode})",
              flush=True)
        started = time.time()
        try:
            out = handle.encode_pipe(prompt=prompt, image=images or None,
                                     output=["prompt_embeds", "text_token_tags"])
        finally:
            # VRAM/RAM нужны трансформеру: стриминговый энкодер больше не пригодится
            self._drop_encoder(handle)
        embeds = out["prompt_embeds"].detach().to("cpu", torch.bfloat16).contiguous()
        tags = out["text_token_tags"].detach().to("cpu", torch.long).contiguous()
        if int(embeds.shape[0]) != 1 or int(embeds.shape[-1]) != TEXT_HIDDEN_DIM:
            raise RuntimeError(f"prompt_embeds {tuple(embeds.shape)}: ожидалось (1, N, "
                               f"{TEXT_HIDDEN_DIM}) — не тот энкодер?")
        print(f"    ok, {time.time() - started:.1f}s prompt_embeds {tuple(embeds.shape)}, "
              f"text_token_tags {tuple(tags.shape)}", flush=True)
        return {"prompt_embeds": embeds, "text_token_tags": tags}

    def _encode_in_child(self, cfg: Any, images: Any, prompt: str) -> dict[str, Any]:
        """Фаза `[3/4]` отдельным процессом: `_encode_child_main` -> `.pt` -> тензоры.

        `images` — список референсов (пустой = T2I). Все сохраняются в рабочий каталог задачи
        без потерь (`PNG`), потому что дочерний процесс получает только пути: PIL-объект через
        pickle не передаётся.

        Зачем процесс, а не «просто аккуратнее жить с памятью»: commit-charge Windows на этой
        машине — общий ресурс (RAM 63.6 GiB + pagefile ~55 GiB), и mmap шардов энкодера
        списывается в него **целиком** (замер: 3 шарда = +62 GiB commit при rss 0.4 GiB).
        Родитель к этому моменту ещё чист (трансформер отложен), поэтому fp8-фаза получает
        ровно те же 113/118 GiB, что и прогон, в котором энкодера не было вовсе.

        Побочная польза: AV-падение safetensors в фазе энкода остаётся в дочернем процессе и
        приходит сюда кодом возврата, а не убивает весь прогон без диагноза.
        """
        work = Path(tempfile.mkdtemp(prefix="qwenstyle-h3i-encode-"))
        job_path = work / "job.pickle"
        out_path = work / "embeds.pt"
        job: dict[str, Any] = {"cfg": cfg, "prompt": str(prompt), "images": [],
                              "out": str(out_path)}
        try:
            for index, item in enumerate(as_image_list(images), start=1):
                image_path = work / f"image{index}.png"
                item.save(image_path, format="PNG")   # без потерь: те же пиксели, что у plan.images
                job["images"].append(str(image_path))
            job_path.write_bytes(pickle.dumps(job))
            env = dict(os.environ)
            env[ENCODE_CHILD_ENV] = "1"
            env.setdefault("PYTHONIOENCODING", "utf-8")
            bootstrap = (
                "import sys;"
                f"sys.path.insert(0, {str(REPO_ROOT)!r});"
                "from qwenstyle.backends.minimax_h3_image import _encode_child_main;"
                "raise SystemExit(_encode_child_main(sys.argv[1]))"
            )
            started = time.time()
            print(f"    фаза [3/4] -> отдельный процесс ({sys.executable})", flush=True)
            # Одна повторная попытка: на этой машине фаза энкодера сидит на пределе commit-charge
            # (62 GiB mmap шардов + pinned-стейджинг при лимите 115–125 GiB), и Windows иногда
            # не успевает расширить pagefile — прогон падает на AV/CUDA-OOM, а со второй попытки
            # (pagefile уже больше) проходит. Настоящая ошибка конфигурации повторится и даст тот
            # же текст, поэтому диагностика не страдает.
            rc = 0
            for attempt, sync in ((1, False), (2, True)):
                job["sync"] = sync
                job_path.write_bytes(pickle.dumps(job))
                rc = subprocess.run([sys.executable, "-c", bootstrap, str(job_path)],
                                    cwd=str(REPO_ROOT), env=env).returncode
                if rc == 0:
                    break
                if attempt == 1:
                    print(f"    попытка {attempt} упала (код {rc}) — повтор с синхронным "
                          f"стримингом (память/commit-charge), вывод выше", flush=True)
            if rc != 0:
                raise RuntimeError(
                    f"фаза [3/4] (энкодер 32B) упала в дочернем процессе: код {rc} "
                    f"(0xC0000005 — access violation, 0xC0000142 — DLL init: обе = исчерпан "
                    f"commit-charge; CUDA OOM с «свободной» VRAM — то же самое). Фаза требует "
                    f"~70 GiB commit под 62.1 GiB весов, лимит машины 115–125 GiB при ~46 GiB "
                    f"уже занятых: закройте тяжёлые приложения (браузер) или увеличьте pagefile — "
                    f"сам бекенд в этом месте ничего не теряет (in-process путь тот же код)"
                )
            payload = _torch().load(out_path, map_location="cpu")
            print(f"    отдельный процесс: {time.time() - started:.1f}s", flush=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return payload

    @staticmethod
    def _drop_encoder(handle: MiniMaxH3ImageHandle) -> None:
        """Отпустить энкодер после энкода: `text_encoder=None` + del + gc + `empty_cache`.

        У стримингового энкодера (62.1 GiB по слоям) дополнительно закрываются mmap шардов —
        иначе на Windows дескрипторы живут до конца процесса.
        """
        model = getattr(handle, "text_encoder", None)
        if model is None:
            return
        streamed = bool(getattr(handle, "encoder_stream", False))
        handle.text_encoder = None
        handle.encoder_stream = False
        if getattr(handle, "encode_pipe", None) is not None:
            handle.encode_pipe.text_encoder = None
        if streamed:
            weights = getattr(model, "_stream_weights", None)
            close = getattr(weights, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:                # noqa: BLE001 — уборка не должна ронять прогон
                    pass
        del model
        gc.collect()
        try:
            _torch().cuda.empty_cache()
        except Exception:                        # noqa: BLE001 — без CUDA чистить нечего
            pass
        print("    энкодер отпущен (VRAM/RAM отданы трансформеру)", flush=True)

    # -- сэмплирование ----------------------------------------------------- #

    def generate(self, cfg: Any, handle: MiniMaxH3ImageHandle, cond: Any, image: Any,
                 seed: int) -> Any:
        """6 шагов PDD (`pdd_step_index` внутри пайплайна) -> PIL-кадр; референсы опциональны.

        `image` — список референсов (одиночная `PIL.Image` тоже принимается): пустой -> блоки
        `text2image` (T2I без `--input`), непустой -> `image_conditioned` (по латенту на каждый
        референс, размер канвы — по первому).
        Размер канвы берётся из `crop` плана и округляется вверх до `canvas_multiple` (32) —
        этого требует `PrepareLayoutStep`, а `RunConfig.crop` гарантирует только кратность 16.
        """
        torch = _torch()
        pipe = self._ensure_pipeline(cfg, handle)
        refs = as_image_list(image)
        width, height = _frame_size(cfg, refs[0] if refs else None)
        multiple = int(pipe.canvas_multiple)
        snapped_w = max(multiple, int(round(width / multiple)) * multiple)
        snapped_h = max(multiple, int(round(height / multiple)) * multiple)
        if (snapped_w, snapped_h) != (width, height):
            print(f"    канва {width}x{height} -> {snapped_w}x{snapped_h} (кратность {multiple})",
                  flush=True)
        width, height = snapped_w, snapped_h

        device = pipe._execution_device
        embeds = cond["prompt_embeds"].to(device=device, dtype=torch.bfloat16)
        # `text_token_tags` остаётся на CPU: `build_image_packed_sequence` индексирует им CPU-тензор
        # (`token_tags[text_indices] = text_token_tags`), и блоки энкодера создают его без device —
        # перенос на GPU даёт `Expected all tensors to be on the same device`.
        tags = cond["text_token_tags"].to("cpu", torch.long)
        area = _sampler_int(cfg, "min_generation_area", DEFAULT_MIN_GENERATION_AREA)
        generator = torch.Generator(device=str(device)).manual_seed(int(seed))
        mode = "edit" if refs else "t2i"
        print(f"[4/4] генерация: {handle.steps} шагов @ {width}x{height} ({mode}), "
              f"min_generation_area={area}, seed={int(seed)}", flush=True)
        started = time.time()
        images = pipe(
            prompt_embeds=embeds,
            text_token_tags=tags,
            image=refs or None,
            height=int(height),
            width=int(width),
            min_generation_area=int(area),
            generator=generator,
            output="images",
        )
        frame = images[0]
        used = None
        try:
            free, total = torch.cuda.mem_get_info()
            used = (total - free) / GIB
        except Exception:                        # noqa: BLE001 — печать не критична
            used = None
        print(f"    ok, {time.time() - started:.1f}s {frame.size[0]}x{frame.size[1]}"
              + (f" peak-ish vram {used:.1f} GiB" if used is not None else ""), flush=True)
        return frame

    def release(self, handle: MiniMaxH3ImageHandle | None) -> None:
        """Освободить VRAM/RAM перед апскейлом (`del` + `gc` + `empty_cache`)."""
        if handle is None:
            return
        self._drop_encoder(handle)
        pipe, encode_pipe = handle.pipe, handle.encode_pipe
        handle.pipe = None
        handle.encode_pipe = None
        handle.transformer = None
        handle.snapshot = None
        del pipe, encode_pipe
        gc.collect()
        try:
            _torch().cuda.empty_cache()
        except Exception:                        # noqa: BLE001 — без CUDA чистить нечего
            pass

    # -- диагностика (работает уже на шаге 2) ------------------------------- #

    def probe(self, cfg: Any) -> dict[str, Any]:
        """Наличие компонентов в локальном hf-cache, шаги/сигмы PDD, состояние fp8-кэша.

        Ничего не грузит и не качает: только `stat`/`read_text` по снапшоту (симлинки `blobs/`
        читаются как обычные файлы) и по каталогу fp8-кэша. `torch`/`diffusers` не импортируются
        вовсе — иначе `models probe`/тесты зависели бы от CUDA.
        """
        gen = cfg.generation
        repo = str(gen.repo or DEFAULT_REPO)
        info: dict[str, Any] = {
            "backend": self.name,
            "repo": repo,
            "caps": self.caps.to_dict(),
            "budget": self.budget.to_dict(),
            "text_hidden_dim": TEXT_HIDDEN_DIM,
            "text_layer_index": TEXT_LAYER_INDEX,
            "precision": gen.precision,
            "offload": gen.offload,
            "offline": str(os.environ.get("HF_HUB_OFFLINE", "")).strip().lower() in
                       ("1", "true", "yes", "on"),
        }

        snapshot = _local_snapshot(repo)
        if snapshot is None:
            info["source"] = "нет данных: репозитория нет в локальном hf-cache"
            info["components"] = {}
            info["missing"] = list(REQUIRED_COMPONENTS)
        else:
            root: Path = snapshot["snapshot"]
            info["source"] = f"hf-cache ({snapshot['cache_dir']})"
            info["commit"] = snapshot["commit"]
            info["snapshot"] = str(root)
            index = snapshot["model_index"]
            info["pipeline"] = (index or {}).get("_class_name")
            declared = _components_of(index)
            sizes: dict[str, int] = {}
            for path in sorted(root.iterdir()):
                sizes[path.name] = _dir_size(path)
            components: dict[str, dict[str, Any]] = {}
            for name in REQUIRED_COMPONENTS:
                path = root / name
                present = path.is_file() or path.is_dir()
                size = sizes.get(name, 0)
                entry: dict[str, Any] = {"present": present, "size_bytes": int(size)}
                entry.update(declared.get(name, {}))
                components[name] = entry
            info["components"] = components
            info["missing"] = [n for n in REQUIRED_COMPONENTS if not components[n]["present"]]
            # `*_diffusers/` — компоненты diffusers-раскладки; нативные `transformer/`/`vae/`
            # в репозитории лежат отдельными весами и бекендом не читаются.
            info["declared_components"] = declared
            info["pdd"] = _pdd(root / "pdd_config.json")

        precision_cache = str(gen.precision_cache or "").strip()
        if not precision_cache:
            info["precision_cache"] = {"path": None, "ready": False,
                                       "status": "выключен (generation.precision_cache='')"}
        else:
            path = Path(precision_cache)
            sidecar = path / "precision.json"
            files = sorted(p.name for p in path.glob("*")) if path.is_dir() else []
            info["precision_cache"] = {
                "path": str(path),
                "ready": sidecar.is_file(),
                "files": files,
                "status": ("готов (старт читает fp8 вместо bf16 + каста)" if sidecar.is_file()
                           else "нет — первый запуск сконвертирует bf16 -> fp8 и запишет его"),
            }
        return info


__all__ = ["DEFAULT_MIN_GENERATION_AREA", "DEFAULT_REPO", "ENCODE_BLOCKS",
           "EncoderUnavailableError", "GENERATE_BLOCKS", "KNOWN_COMMIT", "MiniMaxH3ImageBackend",
           "MiniMaxH3ImageHandle", "REQUIRED_COMPONENTS"]
