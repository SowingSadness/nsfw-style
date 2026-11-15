"""Бекенд генерации FLUX.2 (diffusers `Flux2Pipeline` / `Flux2KleinPipeline`) — internal multi-reference.

Главное отличие от `qwenstyle/backends/qwen_image21.py`: у FLUX.2 **референсы уходят внутрь
пайплайна**, а не в наш кэш эмбеддингов. Поэтому:

* `encode()` считает и кэширует **только текстовую часть** (`encode_prompt` -> `prompt_embeds`);
  референсные картинки в эмбеддинги не входят — их VAE-кодирование и «time offsets» делает сам
  пайплайн на каждом прогоне (`prepare_image_latents`);
* `generate()` передаёт референсы списком в `pipe(image=[...])` (или `image=None` для text-to-image);
* кэш референсных латентов сознательно не делается: они зависят от VAE и от набора картинок,
  а не от промта (этап `ref_latents` в `qwenstyle/cache.py` — задел на будущее, не для этого шага).

Сигнатуры сверены по исходникам `.venv` (diffusers 0.43.0.dev0 / transformers 5.22.2):

* `Flux2Pipeline.__init__(scheduler, vae, text_encoder, tokenizer, transformer)` — аргумента
  `system_message` в конструкторе НЕТ (системные сообщения живут атрибутами пайплайна);
  `text_encoder: Mistral3ForConditionalGeneration`, `text_encoder_out_layers=(10, 20, 30)`,
  есть `upsample_prompt`/`caption_upsample_temperature`;
* `Flux2KleinPipeline.__init__(scheduler, vae, text_encoder, tokenizer, transformer, is_distilled)`
  — `text_encoder: Qwen3ForCausalLM`, `text_encoder_out_layers=(9, 18, 27)`, **нет**
  `caption_upsample_temperature`/`upsample_prompt`, зато есть `negative_prompt_embeds`
  (настоящий CFG, если `is_distilled=False`);
* у обоих `encode_prompt(prompt, device=None, num_images_per_prompt=1, prompt_embeds=None,
  max_sequence_length=512, text_encoder_out_layers=...) -> (prompt_embeds, text_ids)`, где
  `text_ids` — детерминированная функция длины последовательности (в кэш идёт только
  `prompt_embeds`);
* у обоих `__call__(image=PIL|list[PIL]|None, prompt=None, height=None, width=None,
  num_inference_steps=50, sigmas=None, guidance_scale=4.0, generator=None, latents=None,
  prompt_embeds=None, …, max_sequence_length=512, text_encoder_out_layers=...)`;
* `check_inputs()` **запрещает** одновременно `prompt` и `prompt_embeds`;
* LoRA — общий миксин `Flux2LoraLoaderMixin` (`load_lora_weights` + `set_adapters`).

**Что показал `models probe` (метаданные Хаба, веса не скачивались, commit 3bfbda104230):**
репозиторий `darknight9121/FLUX.2-klein-base-9B-bucket-uncensored` объявляет
`_class_name=Flux2KleinPipeline`, `text_encoder=Qwen3ForCausalLM` (15.3 GiB),
`transformer=Flux2Transformer2DModel` (16.9 GiB), `vae=AutoencoderKLFlux2` (160 MiB); всего 49.3 GiB.
Поэтому бекенд **выбирает класс пайплайна и класс энкодера по `model_index.json`** из локального
hf-cache (`PIPELINE_SPECS` ниже), а не хардкодом: `Flux2Pipeline`+Mistral3 остаётся вариантом
по умолчанию для репозиториев, где `model_index.json` недоступен или неизвестен.

**Никаких скачиваний весов.** Все `from_pretrained` получают `local_files_only=HF_HUB_OFFLINE`
(в проекте `HF_HUB_OFFLINE=1` по умолчанию, AGENTS.md R3): если чего-то нет в `hf-cache`, будет
понятная ошибка, а не поход в сеть. В сеть (только метаданные, без весов) ходит лишь
`models probe` при явном `HF_HUB_OFFLINE=0`.

Тяжёлые зависимости (`torch`/`diffusers`/`transformers`/`safetensors`) импортируются лениво:
`discover()` и `models list` не должны тянуть 2 ГБ библиотек.

Ключи `generation.sampler_kwargs` (в `RunConfig` нет отдельных полей):

* `guidance_scale` (по умолчанию `generation.guidance`) — аргумент `pipe(...)`; у дистиллированной
  Klein-модели (`is_distilled=True`) пайплайн его игнорирует и предупреждает;
* `caption_upsample_temperature` — LLM-«дописывание» промта (рекомендуется 0.15). Есть только у
  `Flux2Pipeline`; применяется в `encode()` ДО записи эмбеддингов в кэш, поэтому попадает в ключ
  кэша (`config.CACHE_SAMPLER_KEYS`) и действует одинаково при попадании и промахе;
* `text_encoder_out_layers` / `max_sequence_length` — параметры энкодинга; по умолчанию берутся
  из `PIPELINE_SPECS` (у Klein и Flux2 они разные). В `CACHE_SAMPLER_KEYS` их нет, поэтому их
  значения пишутся в payload кэша, а `generate()` берёт их оттуда — иначе прогон с другими слоями
  молча разъехался бы с закэшированными эмбеддингами;
* `offload_blocks`/`offload_stream` — параметры `enable_group_offload`;
* `rebuild_precision_cache` (он же `rebuild_fp8_cache`) — пересобрать fp8-кэш трансформера.

**Что не проверено (осознанно, шаг 12 плана):** ни один вес здесь не скачивался, поэтому пути
`enable_layerwise_casting`-каста в fp8, `enable_group_offload` и первый реальный `pipe(...)`
проверяются только на реальном запуске FLUX.2 (решение пользователя, шаг 12). Логика 1:1 повторяет
проверенный на Qwen-Image-2.1 порядок.
"""
from __future__ import annotations

import gc
import importlib
import inspect
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ..cache import Blob
from ..registry import register_generation
from .base import REPO_ROOT, Capabilities, ParameterSpec, ResourceBudget, hardware_budget_verdict

#: Репозиторий пайплайна (transformer + text-encoder + VAE + tokenizer). Веса НЕ скачаны.
DEFAULT_REPO = "darknight9121/FLUX.2-klein-base-9B-bucket-uncensored"
#: Запасной путь fp8-кэша, если пресет не задал `generation.precision_cache` (как у Qwen).
DEFAULT_PRECISION_CACHE = "cache/models/fp8_transformer_flux2"
#: Если `model_index.json` недоступен: считаем репозиторий обычным FLUX.2 (Flux2Pipeline + Mistral3).
DEFAULT_PIPELINE = "Flux2Pipeline"
#: Дефолты из исходников пайплайнов: `num_inference_steps=50`, `guidance_scale=4.0`.
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 4.0
DEFAULT_MAX_SEQUENCE_LENGTH = 512
#: Сторона квадратного кадра для text-to-image, если `--size`/`--resolution` не заданы.
DEFAULT_SIDE = 1024
#: Этап кэша: текстовые эмбеддинги (референсы не кэшируются, см. docstring).
CACHE_STAGE = "text_embeds"
#: Пайплайн сам ужимает референс до 1024*1024 px (`_resize_to_target_area`).
MAX_PIXELS = 1024 * 1024
#: Кратность кадра: `vae_scale_factor * 2` (иначе diffusers ресайзит с предупреждением).
SIZE_MULTIPLE = 16
_TRUTHY = {"1", "true", "yes", "on"}

#: Различия двух пайплайнов FLUX.2, которые видны в `model_index.json` и на сигнатурах.
PIPELINE_SPECS: dict[str, dict[str, Any]] = {
    "Flux2Pipeline": {
        "encoder_class": "Mistral3ForConditionalGeneration",
        "out_layers": (10, 20, 30),
        "caption_upsample": True,
    },
    "Flux2KleinPipeline": {
        "encoder_class": "Qwen3ForCausalLM",
        "out_layers": (9, 18, 27),
        "caption_upsample": False,
    },
}


# --------------------------------------------------------------------------- #
# Handle
# --------------------------------------------------------------------------- #

@dataclass
class Flux2Handle:
    """Загруженный пайплайн + параметры энкодинга, общие для `encode`/`generate`."""

    pipe: Any
    transformer: Any
    repo: str
    fp8: bool
    pipeline_name: str = DEFAULT_PIPELINE
    encoder_class: str = PIPELINE_SPECS[DEFAULT_PIPELINE]["encoder_class"]
    #: Слои энкодера по умолчанию: у Flux2Pipeline и Flux2KleinPipeline они РАЗНЫЕ.
    text_encoder_out_layers: tuple[int, ...] = PIPELINE_SPECS[DEFAULT_PIPELINE]["out_layers"]
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    #: Есть ли у пайплайна `caption_upsample_temperature`/`upsample_prompt` (у Klein — нет).
    supports_caption_upsample: bool = True
    precision_cache: Path | None = None


# --------------------------------------------------------------------------- #
# Ленивые импорты и мелкие помощники
# --------------------------------------------------------------------------- #

def _torch():
    import torch

    return torch


def _sampler_kwarg(cfg: Any, key: str, default: Any) -> Any:
    kwargs = getattr(cfg.generation, "sampler_kwargs", None) or {}
    return kwargs.get(key, default)


def _sampler_float(cfg: Any, key: str, default: Any) -> Any:
    value = _sampler_kwarg(cfg, key, default)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sampler_int(cfg: Any, key: str, default: int) -> int:
    value = _sampler_kwarg(cfg, key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _rebuild_requested(cfg: Any) -> bool:
    """`--rebuild-fp8-cache` из run_style.py живёт в `generation.sampler_kwargs`."""
    kwargs = getattr(cfg.generation, "sampler_kwargs", None) or {}
    for key in ("rebuild_precision_cache", "rebuild_fp8_cache"):
        value = kwargs.get(key)
        if isinstance(value, str):
            if value.strip().lower() in _TRUTHY:
                return True
        elif value:
            return True
    return False


def _hub_offline() -> bool:
    """Оффлайн-режим проекта: `HF_HUB_OFFLINE=1` (AGENTS.md R3) -> ни одного запроса к Хабу."""
    return str(os.environ.get("HF_HUB_OFFLINE", "")).strip().lower() in _TRUTHY


def _precision_cache_dir(cfg: Any) -> Path | None:
    """Папка fp8-кэша: `generation.precision_cache`; пустая строка = не кэшировать (None)."""
    raw = cfg.generation.precision_cache
    if raw is None:
        raw = DEFAULT_PRECISION_CACHE
    text = str(raw).strip()
    return Path(text) if text else None


def _out_layers(cfg: Any, default: tuple[int, ...]) -> tuple[int, ...]:
    """`sampler_kwargs.text_encoder_out_layers` -> кортеж int (в YAML это список)."""
    raw = _sampler_kwarg(cfg, "text_encoder_out_layers", None)
    if raw is None:
        return tuple(default)
    if isinstance(raw, str):
        raw = [item for item in raw.replace(" ", "").split(",") if item]
    try:
        layers = tuple(int(item) for item in raw)
    except (TypeError, ValueError):
        return tuple(default)
    return layers or tuple(default)


def _max_sequence_length(cfg: Any) -> int:
    value = _sampler_kwarg(cfg, "max_sequence_length", DEFAULT_MAX_SEQUENCE_LENGTH)
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return DEFAULT_MAX_SEQUENCE_LENGTH


def _reference_images(image: Any) -> list[Any] | None:
    """PIL | list[PIL] | None -> список референсов (или None для text-to-image).

    Модель multi-reference: порядок референсов значим, `None`-элементы отбрасываются.
    """
    if image is None:
        return None
    if isinstance(image, (list, tuple)):
        refs = [item for item in image if item is not None]
        return refs or None
    return [image]


def _text_device(pipe: Any):
    """Устройство текст-энкодера: энкодер живёт на CPU (15-48 ГБ в 16 ГБ VRAM не влезают)."""
    torch = _torch()
    encoder = getattr(pipe, "text_encoder", None)
    device = getattr(encoder, "device", None) if encoder is not None else None
    return device if device is not None else torch.device("cpu")


def _pipeline_device(pipe: Any):
    """Устройство, на котором считает трансформер (`_execution_device` diffusers)."""
    device = getattr(pipe, "_execution_device", None)
    return device if device is not None else _torch().device("cpu")


def _module_gib(module: Any) -> float:
    """Сколько занимают параметры/буферы модуля (для печати, а не для бюджета)."""
    if module is None:
        return 0.0
    sizes = {
        "float32": 4, "float": 4, "float16": 2, "half": 2, "bfloat16": 2,
        "float8_e4m3fn": 1, "float8_e5m2": 1,
    }
    try:
        total = 0
        for tensor in list(module.parameters()) + list(module.buffers()):
            total += tensor.numel() * sizes.get(str(tensor.dtype).replace("torch.", ""), 2)
    except Exception:                                # noqa: BLE001 — диагностика не обязательна
        return 0.0
    return total / 1024 ** 3


def _filter_kwargs(pipe: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Оставить только те аргументы, которые `pipe.__call__` действительно принимает.

    `Flux2Pipeline` и `Flux2KleinPipeline` отличаются набором (`caption_upsample_temperature`
    есть только у первого), а ошибиться молча нельзя — TypeError на середине прогона дороже.
    """
    try:
        parameters = inspect.signature(type(pipe).__call__).parameters
    except (TypeError, ValueError):                  # noqa: BLE001 — нестандартный __call__
        return kwargs
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs
    dropped = sorted(set(kwargs) - set(parameters))
    if dropped:
        print(f"[flux2] пайплайн {type(pipe).__name__} не принимает {', '.join(dropped)} — "
              f"пропускаю", flush=True)
    return {key: value for key, value in kwargs.items() if key in parameters}


# --------------------------------------------------------------------------- #
# Локальный hf-cache: какой класс пайплайна/энкодера объявляет model_index.json
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
    for path in candidates:
        if all(path.resolve() != other.resolve() for other in unique):
            unique.append(path)
    return unique


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _local_model_index(repo: str) -> dict[str, Any] | None:
    """`model_index.json` из локального hf-cache — без сети (обход `snapshots/<commit>/`)."""
    for cache_dir in _hf_cache_dirs():
        root_dir = Path(cache_dir) / ("models--" + str(repo).replace("/", "--"))
        snapshots = root_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        commit = None
        ref = root_dir / "refs" / "main"
        if ref.is_file():
            try:
                commit = ref.read_text(encoding="utf-8").strip() or None
            except OSError:
                commit = None
        dirs = sorted(p for p in snapshots.iterdir() if p.is_dir())
        if not dirs:
            continue
        root = next((p for p in dirs if p.name == commit), dirs[0])
        index = _read_json(root / "model_index.json")
        if index:
            return index
    return None


def _pipeline_spec(model_index: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    """(класс пайплайна, спека) по `model_index.json`; неизвестный -> обычный Flux2Pipeline."""
    name = str((model_index or {}).get("_class_name") or DEFAULT_PIPELINE).strip()
    spec = PIPELINE_SPECS.get(name)
    if spec is None:
        if model_index:
            print(f"[flux2] неизвестный пайплайн {name!r} — работаю как {DEFAULT_PIPELINE}",
                  flush=True)
        return DEFAULT_PIPELINE, dict(PIPELINE_SPECS[DEFAULT_PIPELINE])
    return name, dict(spec)


def _encoder_class_name(model_index: dict[str, Any] | None, spec: dict[str, Any]) -> str:
    """Класс текст-энкодера: из `model_index.json` (`text_encoder: [transformers, Class]`)."""
    value = (model_index or {}).get("text_encoder")
    if isinstance(value, (list, tuple)) and len(value) == 2 and str(value[0]) == "transformers":
        return str(value[1])
    return str(spec.get("encoder_class") or PIPELINE_SPECS[DEFAULT_PIPELINE]["encoder_class"])


def _import_pipeline(name: str) -> type:
    """`diffusers.<name>`; отсутствие класса — понятная ошибка, а не AttributeError."""
    module = importlib.import_module("diffusers")
    cls = getattr(module, name, None)
    if cls is None:
        raise RuntimeError(
            f"flux2: в diffusers {getattr(module, '__version__', '?')} нет класса {name}; "
            f"доступны: {', '.join(sorted(PIPELINE_SPECS))}"
        )
    return cls


def _import_encoder(name: str) -> type:
    """`transformers.<name>` (Mistral3ForConditionalGeneration / Qwen3ForCausalLM / …)."""
    module = importlib.import_module("transformers")
    cls = getattr(module, name, None)
    if cls is None:
        raise RuntimeError(
            f"flux2: в transformers {getattr(module, '__version__', '?')} нет класса {name}"
        )
    return cls


# --------------------------------------------------------------------------- #
# Веса: текст-энкодер, fp8-кэш трансформера, LoRA
# --------------------------------------------------------------------------- #

def _dtype(name: str):
    torch = _torch()
    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(str(name or "bfloat16").strip().lower(), torch.bfloat16)


def _load_text_encoder(cfg: Any, encoder_class: str) -> Any:
    """Текст-энкодер (`Mistral3ForConditionalGeneration` / `Qwen3ForCausalLM`) — CPU, bf16.

    Нужен только при промахе кэша текстовых эмбеддингов (`prepare(need_encoder=True)`).
    Порядок как в проекте: сначала локальный `hf-cache`/`local_dir`, сеть — только если
    `HF_HUB_OFFLINE=0`; иначе понятная ошибка с командой, что делать.

    `cfg.encoder.stream=True` -> веса читаются по слоям с диска (`qwenstyle/streaming.py`): модель
    приходит уже `eval()` на устройстве с пустыми параметрами (0 байт, `device=cpu`, dtype весов) —
    веса приезжают хуками на время forward, поэтому `.to()/.eval()` здесь не нужны. `prefetch`/
    `pin_memory`/`lookahead` включают конвейер H2D только при `device=cuda` (`prefetch=0` = прежний
    синхронный путь). Это раскладка памяти, а не другая математика — ключ кэша эмбеддингов не меняется.
    """
    where = str(cfg.encoder.local_dir or cfg.encoder.repo)
    offline = _hub_offline()
    # `resolved()` обязателен: `device: auto` (умолчание пресетов) разрешается в cuda/cpu
    # и на машине с CUDA сам включает GPU-стриминг (`stream=True`, `prefetch=4`). Без этого
    # `.to("auto")` падал `RuntimeError: ... device type ... 'auto'`.
    enc = cfg.encoder.resolved() if hasattr(cfg.encoder, "resolved") else cfg.encoder
    device = str(enc.device or "cpu")
    stream = bool(getattr(enc, "stream", False) or False)
    # параметры конвейера стриминга (нет в старых пресетах -> безопасные умолчания)
    prefetch = int(getattr(enc, "prefetch", 0) or 0)
    # `cfg.encoder_pin_memory` — это уже разрешённое поле `encoder.pin_memory`: pin включён
    # и при `offload=group`, а упавшая pinned-аллокация деградирует в `streaming._PinnedPool`.
    pin_memory = bool(getattr(cfg, "encoder_pin_memory", getattr(enc, "pin_memory", True)))
    lookahead = int(getattr(enc, "lookahead", 1) or 1)
    if not pin_memory:
        print(f"[te] энкодер {cfg.encoder.name}: pin_memory отключён пользователем "
              f"(encoder.pin_memory=false)", flush=True)
    model_class = _import_encoder(encoder_class)
    # В diffusers-репозиториях энкодер лежит в subfolder='text_encoder', а не в корне
    # (так у FLUX.2: text_encoder/model.safetensors). Для локального каталога-энкодера
    # (например cache/models/weights/heretic-te) subfolder не нужен.
    subfolder = None
    if not cfg.encoder.local_dir:
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                cfg.encoder.repo,
                "text_encoder/config.json",
                revision=cfg.encoder.revision,
                local_files_only=offline,
            )
            subfolder = "text_encoder"
        except Exception:  # noqa: BLE001 — нет такого файла → грузим из корня
            subfolder = None
    print(f"[te] энкодер {cfg.encoder.name} ({encoder_class}) из {where}"
          f"{f'/{subfolder}' if subfolder else ''} "
          f"({cfg.encoder.dtype}, {device}, offline={offline})...", flush=True)
    started = time.time()
    try:
        if stream:
            from .. import streaming                   # лениво: без torch при `models list`

            encoder = streaming.load_text_encoder(
                model_class,
                where,
                subfolder=subfolder,
                revision=cfg.encoder.revision,
                dtype=cfg.encoder.dtype,
                local_files_only=offline,
                device=device,
                stream=True,
                prefetch=prefetch,
                pin_memory=pin_memory,
                lookahead=lookahead,
            )
        else:
            encoder = model_class.from_pretrained(
                where,
                revision=cfg.encoder.revision,
                subfolder=subfolder,
                torch_dtype=_dtype(cfg.encoder.dtype),
                low_cpu_mem_usage=True,
                local_files_only=offline,
            )
    except Exception as exc:                       # noqa: BLE001 — нужна команда, а не трассировка
        raise RuntimeError(
            f"flux2: текст-энкодер {where} не загружается ({type(exc).__name__}: {exc}). "
            f"Веса не скачиваются автоматически: либо положи их в hf-cache (HF_HUB_OFFLINE=0 "
            f"для одного раза), либо укажи encoder.local_dir на готовый каталог. "
            f"Состав репозитория покажет: python -m qwenstyle models probe {where}"
        ) from exc
    if stream:                     # уже eval() на device, параметры пустые (веса дают хуки)
        print(f"    ok, {time.time() - started:.1f}s", flush=True)
        return encoder
    encoder.eval()
    encoder.to(device)
    print(f"    ok, {time.time() - started:.1f}s", flush=True)
    return encoder


def _load_fp8_transformer(path: Path) -> Any:
    """fp8-трансформер из дискового кэша с точными dtype не-fp8 тензоров.

    Логика 1:1 с `qwen_image21._load_fp8_transformer` (проверена на Qwen-Image-2.1):
    `from_pretrained(torch_dtype=fp8)` кастует ВСЁ, включая нормы и patch/pos-embed, которые
    `enable_layerwise_casting` держит в bf16; обратный каст невосстановим, а на буферах (freqs)
    forward падает с `Promotion for Float8 Types is not supported`. Поэтому не-fp8 тензоры
    перечитываются с диска в исходном dtype, а `persistent=False` буферы — из сайдкара.
    """
    from diffusers.models import Flux2Transformer2DModel
    from safetensors import safe_open

    torch = _torch()
    transformer = Flux2Transformer2DModel.from_pretrained(
        str(path), torch_dtype=torch.float8_e4m3fn, low_cpu_mem_usage=True,
    )
    params = dict(transformer.named_parameters())
    buffers = dict(transformer.named_buffers())
    exact = 0
    for file in sorted(path.glob("*.safetensors")):
        with file.open("rb") as handle:
            header = json.loads(handle.read(int.from_bytes(handle.read(8), "little")))
        keys = [name for name, meta in header.items()
                if name != "__metadata__" and not meta["dtype"].startswith("F8")]
        if not keys:
            continue
        with safe_open(str(file), framework="pt", device="cpu") as state:
            for name in keys:
                tensor = state.get_tensor(name)          # ровно тот dtype, что на диске
                if name in params:
                    params[name].data = tensor
                elif name in buffers:
                    buffers[name] = tensor
                exact += 1
    sidecar = path / "non_persistent_buffers.pt"
    if sidecar.is_file():
        for name, tensor in torch.load(sidecar, map_location="cpu").items():
            parent_path, _, leaf = name.rpartition(".")
            parent = transformer.get_submodule(parent_path) if parent_path else transformer
            if hasattr(parent, "_buffers"):
                parent._buffers[leaf] = tensor
    print(f"    точных не-fp8 тензоров из кэша: {exact}", flush=True)
    return transformer


def _save_fp8_cache(pipe: Any, path: Path) -> None:
    """Записать fp8-трансформер на диск (первый запуск; дальше старт читает fp8 вместо bf16).

    Аргумент `R1`: каст 9B bf16 -> fp8 стоит минуты и десятки ГБ RAM, повторять его на каждом
    старте нельзя. Рядом кладётся `non_persistent_buffers.pt` — буферы, которых нет в
    `state_dict` (`freqs` таймстеп-эмбеддинга), но которые нужны для forward.
    """
    torch = _torch()
    started = time.time()
    pipe.transformer.save_pretrained(str(path))
    state_keys = set(pipe.transformer.state_dict())
    extra = {name: buf.clone() for name, buf in pipe.transformer.named_buffers()
             if name not in state_keys}
    torch.save(extra, path / "non_persistent_buffers.pt")
    print(f"      fp8-кэш записан в {path} ({time.time() - started:.1f}s), "
          f"следующий старт быстрее", flush=True)


def _load_lora(pipe: Any, cfg: Any) -> None:
    """LoRA через `Flux2LoraLoaderMixin` (`load_lora_weights` + `set_adapters`).

    Адаптер берётся из `cache/models/weights/<filename>`, если он там уже есть, иначе
    `hf_hub_download` (в оффлайне — понятная ошибка, а не тихое «без LoRA»).
    """
    lora = getattr(cfg.generation, "lora", None)
    if lora is None:
        print("    адаптер не задан в конфиге — LoRA не применяется", flush=True)
        return
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    local = Path("cache") / "models" / "weights" / lora.filename
    payload: Any
    if local.is_file():
        payload = load_file(str(local), device="cpu")
    else:
        payload = hf_hub_download(lora.repo, lora.filename)
    if isinstance(payload, dict) and getattr(lora, "strip_alpha", False):
        alpha_keys = [key for key in payload if key.endswith(".alpha")]
        for key in alpha_keys:
            payload.pop(key)
        if alpha_keys:
            print(f"    alpha-ключей вырезано: {len(alpha_keys)}", flush=True)
    pipe.load_lora_weights(payload, adapter_name="flux2_lora")
    pipe.set_adapters(["flux2_lora"], adapter_weights=[float(lora.scale)])


def _place(pipe: Any, offload: str, cfg: Any) -> None:
    """Размещение компонентов: `none` (всё на GPU, кроме энкодера) / `group` / `model_cpu`."""
    from .. import streaming                     # лёгкий модуль: torch внутри функций

    torch = _torch()
    if offload not in ("none", "group", "model_cpu"):
        raise ValueError(f"flux2: offload={offload!r}; допустимо 'none', 'group', 'model_cpu'")
    if getattr(pipe, "text_encoder", None) is not None:
        # 15-48 ГБ bf16 в 16 ГБ VRAM не влезают; стриминговая модель уже размещена хуками -> no-op
        streaming.move_to(pipe.text_encoder, "cpu")
    if offload == "model_cpu":
        print("[3/4] размещение: model_cpu_offload (по компонентам)", flush=True)
        pipe.enable_model_cpu_offload()
        return
    if offload == "group":
        blocks = _sampler_int(cfg, "offload_blocks", 1)
        stream = bool(_sampler_int(cfg, "offload_stream", 1))
        print(f"[3/4] размещение: transformer group-offload (блоков в группе {blocks}, "
              f"stream={stream}), vae -> cuda", flush=True)
        pipe.transformer.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=blocks,
            use_stream=stream,
            non_blocking=True,
        )
        pipe.vae.to("cuda")
        return
    print("[3/4] размещение: transformer+vae -> cuda", flush=True)
    pipe.transformer.to("cuda")
    pipe.vae.to("cuda")
    for method in ("enable_slicing", "enable_tiling"):     # декод в 16 ГБ впритык
        if hasattr(pipe.vae, method):
            getattr(pipe.vae, method)()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Бекенд
# --------------------------------------------------------------------------- #

@register_generation
class Flux2Backend:
    """FLUX.2: `Flux2Transformer2DModel` + текстовый энкодер, `conditioning="internal"`.

    `internal` — единственный признак, по которому оркестратор понимает, что референсы нельзя
    «зашить» в `.pt`: он кэширует только `prompt_embeds` (этап `text_embeds`), а картинки отдаёт
    в `generate` -> `pipe(image=[...])`. Поэтому `encoder="required"`: без энкодера пайплайн
    соберётся (`text_encoder=None`), но новый промт закодировать нечем.

    Какой именно пайплайн (`Flux2Pipeline` + Mistral3 или `Flux2KleinPipeline` + Qwen3) — решает
    `model_index.json` репозитория, прочитанный из локального hf-cache; см. `PIPELINE_SPECS`.

    `budget` — из метаданных Хаба для объявленного репозитория (веса не скачивались,
    commit 3bfbda104230): transformer 16.9 GiB -> fp8 ~8.5 GiB, text-encoder 15.3 GiB (только
    CPU/RAM), диск 49.3 GiB. Точные числа по любому другому репозиторию —
    `python -m qwenstyle models probe <repo>`.
    """

    name: ClassVar[str] = "flux2"
    #: Репозитории, которые обслуживает бекенд (для `models probe`).
    repos: ClassVar[tuple[str, ...]] = (DEFAULT_REPO,)
    #: Этап кэша для `encode` (референсы не кэшируются): `cache/embeds/text_embeds_*.pt` + `.json`.
    cache_stage: ClassVar[str] = CACHE_STAGE
    caps: ClassVar[Capabilities] = Capabilities(
        conditioning="internal",
        image_inputs="multi",
        encoder="required",
        fixed_steps=None,
        supports_lora=True,
        precision=("bf16", "fp8"),
        offload=("none", "group", "model_cpu"),
        max_pixels=MAX_PIXELS,
        extra_sampling=("guidance_scale", "caption_upsample_temperature",
                        "text_encoder_out_layers"),
    )
    budget: ClassVar[ResourceBudget] = ResourceBudget(
        weights_bf16_gib=16.9,
        weights_fp8_gib=8.5,
        encoder_gib=15.3,
        disk_gib=49.3,
        notes=("размеры — из метаданных Хаба (`models probe`), веса НЕ скачаны: "
               "transformer 16.9 GiB bf16 -> ~8.5 GiB fp8; text-encoder Qwen3ForCausalLM "
               "15.3 GiB держится только на CPU/RAM (вместе с весами в 16 GiB VRAM не влезает); "
               "диск 49.3 GiB — в репозитории трансформер лежит дважды (root + transformer/). "
               "Из этого следует: fp8-веса влезают в 16 ГБ, а энкод идёт по CPU (десятки секунд "
               "на промт). `caption_upsample_temperature` у Flux2KleinPipeline отсутствует."),
    )

    @staticmethod
    def parameters() -> tuple[ParameterSpec, ...]:
        """Настраиваемые параметры модели (шаг 9 плана публикации).

        Кроме полей `GenerationConfig` (`steps`/`guidance`/`offload`/`precision`) модель читает
        ключи `generation.sampler_kwargs`:

        * `max_sequence_length`, `text_encoder_out_layers` — параметры энкодинга (входят в ключ
          кэша `text_embeds`; `text_encoder_out_layers` по умолчанию берётся из `PIPELINE_SPECS`);
        * `guidance_scale`/`caption_upsample_temperature` — аргументы `pipe(...)` (`extra_sampling`);
        * `offload_blocks`/`offload_stream` — `enable_group_offload`;
        * `rebuild_precision_cache`/`rebuild_fp8_cache` — пересборка fp8-кэша.
        """
        caps = Flux2Backend.caps
        return (
            ParameterSpec("steps", "generation", "int", 25, minimum=8, maximum=50,
                          doc="шагов сэмплера (для klein 8 даёт черновик)"),
            ParameterSpec("guidance", "generation", "float", DEFAULT_GUIDANCE, minimum=0.0,
                          maximum=10.0,
                          doc="guidance пайплайна; аргумент pipe(...) — sampler_kwargs.guidance_scale"),
            ParameterSpec("offload", "generation", "enum", "none", choices=tuple(caps.offload),
                          doc="none = веса в VRAM; group = блоки с CPU; model_cpu = целиком на CPU"),
            ParameterSpec("precision", "generation", "enum", "fp8", choices=tuple(caps.precision),
                          doc="fp8 = кэш cache/models/fp8_transformer_flux2; bf16 = 16.9 GiB в VRAM"),
            ParameterSpec("max_sequence_length", "sampler", "int", DEFAULT_MAX_SEQUENCE_LENGTH,
                          minimum=256, maximum=2048,
                          doc="токенов промта; входит в ключ кэша text_embeds"),
            ParameterSpec("text_encoder_out_layers", "sampler", "str", None,
                          doc="слои энкодера через запятую (напр. 9,18,27); не задано = из "
                              "PIPELINE_SPECS по model_index.json (Klein 9,18,27 / Flux2 10,20,30)"),
            ParameterSpec("guidance_scale", "sampler", "float", DEFAULT_GUIDANCE, minimum=0.0,
                          maximum=10.0, doc="аргумент pipe(...): дублирует guidance намеренно"),
            ParameterSpec("caption_upsample_temperature", "sampler", "float", None,
                          minimum=0.0, maximum=2.0,
                          doc="upsample промта; есть только у Flux2Pipeline (Klein игнорирует), "
                              "входит в ключ кэша"),
            ParameterSpec("offload_blocks", "sampler", "int", 1, minimum=1, maximum=64,
                          doc="блоков трансформера на GPU одновременно (offload=group)"),
            ParameterSpec("offload_stream", "sampler", "bool", True,
                          doc="асинхронная префетч-загрузка блоков (offload=group)"),
            ParameterSpec("rebuild_precision_cache", "sampler", "bool", False,
                          doc="пересобрать fp8-кэш вместо чтения готового"),
            ParameterSpec("rebuild_fp8_cache", "sampler", "bool", False,
                          doc="алиас rebuild_precision_cache (флаг --rebuild-fp8-cache)"),
        )

    # -- загрузка ---------------------------------------------------------- #

    def prepare(self, cfg: Any, need_encoder: bool) -> Flux2Handle:
        """Ленивая сборка пайплайна: model_index -> класс -> fp8-кэш -> энкодер -> LoRA -> раскладка.

        `need_encoder=False` (эмбеддинги из кэша) -> `text_encoder=None`: энкодер не грузится.
        """
        torch = _torch()
        gen = cfg.generation
        repo = str(gen.repo)
        offline = _hub_offline()
        precision = str(gen.precision or "bf16").strip().lower()
        if precision not in ("bf16", "fp8"):
            raise ValueError(f"flux2: precision={precision!r}; допустимо 'bf16' или 'fp8'")
        fp8 = precision == "fp8"
        cache_dir = _precision_cache_dir(cfg) if fp8 else None
        rebuild = _rebuild_requested(cfg)

        assert torch.cuda.is_available(), "torch собрана без CUDA"

        model_index = _local_model_index(repo)
        if model_index is None and not offline:
            # Локального снапшота нет, но сеть разрешена: класс пайплайна/энкодера берём из
            # model_index.json репозитория (только метаданные, веса качает сам from_pretrained).
            try:
                from ..cmd.models import probe_repo

                model_index = probe_repo(repo, offline=False).model_index
            except Exception:                    # noqa: BLE001 — остаются дефолты бекенда
                model_index = None
        pipeline_name, spec = _pipeline_spec(model_index)
        encoder_class = _encoder_class_name(model_index, spec)
        pipeline_class = _import_pipeline(pipeline_name)
        print(f"[1/4] пайплайн {pipeline_name}, энкодер {encoder_class}"
              f"{'' if model_index else ' (model_index.json не найден в hf-cache — дефолты)'}",
              flush=True)

        kwargs: dict[str, Any] = {}
        transformer = None
        tr_is_fp8 = False
        t0 = time.time()
        if cache_dir is not None and not rebuild and (cache_dir / "config.json").is_file():
            transformer = _load_fp8_transformer(cache_dir)
            kwargs["transformer"] = transformer
            tr_is_fp8 = True
            print(f"[1/4] fp8-трансформер из кэша {cache_dir}", flush=True)

        if need_encoder:
            kwargs["text_encoder"] = _load_text_encoder(cfg, encoder_class)
        else:
            kwargs["text_encoder"] = None
            print("[1/4] текст-энкодер не нужен (текстовые эмбеддинги из кэша)", flush=True)

        print(f"[1/4] загрузка {repo} ({precision}, offline={offline})...", flush=True)
        try:
            pipe = pipeline_class.from_pretrained(
                repo,
                revision=gen.revision,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                local_files_only=offline,      # никаких скачиваний: только уже закэшированное
                **kwargs,
            )
        except Exception as exc:               # noqa: BLE001 — понятная деградация вместо трейса
            raise RuntimeError(
                f"flux2: пайплайн {repo} ({pipeline_name}) не загружается "
                f"({type(exc).__name__}: {exc}). Веса не скачиваются автоматически "
                f"(HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE')}). Состав и размеры покажет: "
                f"python -m qwenstyle models probe {repo}"
            ) from exc
        print(f"    ok, {time.time() - t0:.1f}s", flush=True)

        if fp8:
            t = time.time()
            pipe.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=torch.bfloat16,
            )
            if tr_is_fp8:
                print(f"[2/4] хуки upcast fp8->bf16 включены ({time.time() - t:.1f}s)", flush=True)
            else:
                print(f"[2/4] bf16 -> fp8 cast ({time.time() - t:.1f}s), "
                      f"трансформер {_module_gib(pipe.transformer):.1f} GiB", flush=True)
                if cache_dir is not None:
                    _save_fp8_cache(pipe, cache_dir)
        else:
            print("[2/4] precision=bf16: трансформер ~17 GiB — на 16 GiB VRAM только с --offload",
                  flush=True)

        print("[2/4] LoRA...", flush=True)
        _load_lora(pipe, cfg)

        _place(pipe, str(gen.offload or "none").strip().lower(), cfg)
        free, total = torch.cuda.mem_get_info()
        print(f"    vram после загрузки: занято {(total - free) / 2 ** 30:.1f}/{total / 2 ** 30:.1f} GiB",
              flush=True)

        return Flux2Handle(
            pipe=pipe,
            transformer=pipe.transformer,
            repo=repo,
            fp8=fp8,
            pipeline_name=pipeline_name,
            encoder_class=encoder_class,
            text_encoder_out_layers=tuple(spec["out_layers"]),
            max_sequence_length=_max_sequence_length(cfg),
            supports_caption_upsample=bool(spec["caption_upsample"]),
            precision_cache=cache_dir,
        )

    # -- conditioning ------------------------------------------------------ #

    def encode(self, cfg: Any, handle: Flux2Handle, image: Any, prompt: str) -> dict[str, Any]:
        """ТОЛЬКО текстовые эмбеддинги (`pipe.encode_prompt`), формат `.pt`-кэша.

        Референсы в эмбеддинги не входят: их кодирует VAE внутри пайплайна на каждом прогоне.
        Исключение — `caption_upsample_temperature` (только `Flux2Pipeline`): пайплайн «дописывает»
        промт LLM-частью энкодера с учётом референсов, поэтому апсэмплинг делается здесь, ДО
        энкодинга, и его результат попадает в кэш (в ключе уже есть `sha256` входного файла).
        """
        torch = _torch()
        pipe = handle.pipe
        if getattr(pipe, "text_encoder", None) is None:
            raise RuntimeError(
                "flux2: текст-энкодер не загружен (prepare(need_encoder=False)), а кэша текстовых "
                "эмбеддингов нет — нужен прогон с need_encoder=True"
            )

        device = _text_device(pipe)
        layers = _out_layers(cfg, tuple(handle.text_encoder_out_layers))
        max_seq = int(handle.max_sequence_length)
        temperature = _sampler_float(cfg, "caption_upsample_temperature", None)
        text: Any = prompt or ""

        if temperature:
            if not handle.supports_caption_upsample:
                print(f"[flux2] {handle.pipeline_name} не умеет caption upsampling — "
                      f"temperature={temperature} игнорируется", flush=True)
                temperature = None
            else:
                refs = _reference_images(image)
                print(f"[3/4] caption upsampling (temperature={temperature}, "
                      f"референсов {0 if not refs else len(refs)})...", flush=True)
                started = time.time()
                text = pipe.upsample_prompt(text, images=refs, temperature=float(temperature),
                                            device=device)
                print(f"    ok, {time.time() - started:.1f}s", flush=True)

        print(f"[3/4] кодирование промта ({handle.encoder_class}, layers={layers}, "
              f"seq={max_seq})...", flush=True)
        started = time.time()
        with torch.no_grad():
            prompt_embeds, _text_ids = pipe.encode_prompt(
                prompt=text,
                device=device,
                max_sequence_length=max_seq,
                text_encoder_out_layers=layers,
            )
            # CFG: при guidance_scale > 1 пайплайн требует negative_prompt_embeds и иначе
            # полез бы к text_encoder (а его в кэшированном режиме нет). Пустой промт от
            # промта не зависит — кодируем один раз и кладём в тот же .pt.
            negative_embeds, _neg_ids = pipe.encode_prompt(
                prompt="",
                device=device,
                max_sequence_length=max_seq,
                text_encoder_out_layers=layers,
            )
        print(f"    ok, {time.time() - started:.1f}s {tuple(prompt_embeds.shape)}", flush=True)
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_embeds,
            "backend": self.name,
            "pipeline": handle.pipeline_name,
            "text_encoder_out_layers": list(layers),
            "max_sequence_length": int(max_seq),
            "caption_upsample_temperature": None if not temperature else float(temperature),
        }

    @classmethod
    def cache_sidecar_values(cls, cfg: Any) -> dict[str, Any]:
        """Поля сайдкара, которые обязаны совпасть при попадании в кэш (R1).

        `max_sequence_length` реально меняет форму и содержимое эмбеддингов (512 vs 1024 токена —
        усечение промта), поэтому без этой сверки смена контекста молча брала бы старый `.pt`.
        """
        return {"max_sequence_length": int(_max_sequence_length(cfg))}

    @classmethod
    def cache_blob(cls, cfg: Any, crop: tuple[int, int], prompt_md5: str,
                   img_sha: str) -> Blob:
        """Запись кэша текстовых эмбеддингов: `text_embeds_<md5>_<sha>[_te-<enc>]_<W>x<H>.pt`.

        Ключ — тот же, что у `cfg.embeds_key()` (формат AGENTS.md R1), но без префикса `embeds_`:
        его даёт сам этап. Тег энкодера в ключе обязателен — иначе эмбеддинги FLUX.2 и
        Qwen-Image-2.1 перемешаются.
        """
        key = str(cfg.embeds_key(crop, prompt_md5, img_sha))
        if key.startswith("embeds_"):
            key = key[len("embeds_"):]
        if key.endswith(".pt"):
            key = key[:-3]
        return Blob(stage=cls.cache_stage, key=key)

    # -- сэмплирование ----------------------------------------------------- #

    def generate(self, cfg: Any, handle: Flux2Handle, cond: Any, image: Any,
                 seed: int) -> Any:
        """Денойзинг. Референсы -> `pipe(image=[...])` (порядок значим), `None` — text-to-image."""
        torch = _torch()
        pipe = handle.pipe
        gen = cfg.generation
        cond = cond or {}
        prompt_embeds = cond.get("prompt_embeds")

        layers = tuple(handle.text_encoder_out_layers)
        stored = cond.get("text_encoder_out_layers")
        if stored:
            layers = tuple(int(item) for item in stored)
            requested = _out_layers(cfg, tuple(handle.text_encoder_out_layers))
            if layers != requested:
                print(f"[warn] text_encoder_out_layers: в кэше {layers}, в конфиге {requested} — "
                      f"беру значения из кэша (иначе кадр не совпадёт с эмбеддингами)", flush=True)
        max_seq = int(cond.get("max_sequence_length") or handle.max_sequence_length)

        steps = int(gen.steps) if gen.steps else DEFAULT_STEPS
        guidance = _sampler_float(cfg, "guidance_scale", gen.guidance or DEFAULT_GUIDANCE)
        refs = _reference_images(image)

        if refs is not None:
            width, height = (int(value) for value in cfg.crop(*refs[0].size))
        else:
            side = int(cfg.size_px or cfg.resolution or DEFAULT_SIDE)
            side = max(SIZE_MULTIPLE, (side // SIZE_MULTIPLE) * SIZE_MULTIPLE)
            width = height = side

        kwargs: dict[str, Any] = {
            "image": refs,
            "height": height,
            "width": width,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "max_sequence_length": max_seq,
            "text_encoder_out_layers": layers,
            "generator": torch.Generator("cpu").manual_seed(int(seed)),
        }
        if prompt_embeds is not None:
            # `check_inputs` запрещает prompt+prompt_embeds одновременно. Температуру апсэмплинга
            # сюда не передаём: она уже применена в `encode` (иначе кадр разъедется с кэшем).
            kwargs["prompt"] = None
            kwargs["prompt_embeds"] = prompt_embeds.to(_pipeline_device(pipe))
            # CFG (guidance > 1) требует и негативные эмбеддинги; они лежат в том же .pt.
            if guidance is not None and float(guidance) > 1.0:
                negative = cond.get("negative_prompt_embeds")
                if negative is None:
                    raise RuntimeError(
                        "flux2: в кэше нет negative_prompt_embeds, а CFG включён "
                        f"(guidance_scale={guidance}). Пересоздай кэш текстовых эмбеддингов "
                        "одним прогоном без кэша (например с другим --seed-тегом в имени "
                        "или удалив cache/embeds/text_embeds_*.pt для этой пары)."
                    )
                kwargs["negative_prompt_embeds"] = negative.to(_pipeline_device(pipe))
        else:
            kwargs["prompt"] = cond.get("prompt") or cfg.prompt
            if not kwargs["prompt"]:
                raise ValueError("flux2: нет ни prompt_embeds, ни промта")
            temperature = _sampler_float(cfg, "caption_upsample_temperature", None)
            if temperature:
                kwargs["caption_upsample_temperature"] = float(temperature)

        mode = "text-to-image" if refs is None else f"multi-reference ({len(refs)} шт.)"
        print(f"[4/4] генерация: {steps} шагов @ {width}x{height}, guidance {guidance}, {mode}"
              f"{' (preview)' if getattr(cfg, 'is_preview', False) else ''}", flush=True)
        started = time.time()
        with torch.inference_mode():
            result = pipe(**_filter_kwargs(pipe, kwargs)).images[0]
        print(f"    ok, {time.time() - started:.0f}s", flush=True)
        return result

    # -- освобождение ------------------------------------------------------ #

    def release(self, handle: Flux2Handle) -> None:
        """Отпускает пайплайн перед апскейлом: del + gc + empty_cache."""
        torch = _torch()
        pipe = getattr(handle, "pipe", None)
        handle.pipe = None
        handle.transformer = None
        del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- диагностика ------------------------------------------------------- #

    def probe(self, cfg: Any) -> dict[str, Any]:
        """Компоненты из локального hf-cache (или метаданных Хаба при `HF_HUB_OFFLINE=0`).

        Веса не скачиваются: `probe_repo` читает только `model_index.json`, `*/config.json`
        и размеры файлов; в оффлайне без локальной копии отдаёт «нет данных».
        """
        gen = cfg.generation
        repo = str(gen.repo)
        info: dict[str, Any] = {
            "backend": self.name,
            "repo": repo,
            "precision": gen.precision,
            "offload": gen.offload,
            "steps": gen.steps,
            "caps": self.caps.to_dict(),
            "budget": self.budget.to_dict(),
            "offline": _hub_offline(),
        }

        model_index = _local_model_index(repo)
        pipeline_name, spec = _pipeline_spec(model_index)
        encoder_class = _encoder_class_name(model_index, spec)
        info["pipeline"] = pipeline_name
        info["encoder_class"] = encoder_class
        info["text_encoder_out_layers"] = list(spec["out_layers"])
        info["caption_upsample"] = bool(spec["caption_upsample"])
        if model_index is None:
            info["pipeline_source"] = "model_index.json нет в hf-cache — дефолты бекенда"
        else:
            info["pipeline_source"] = "model_index.json из hf-cache"

        try:
            from ..cmd.models import probe_repo

            probe = probe_repo(repo, offline=_hub_offline())
        except Exception as exc:                     # noqa: BLE001 — диагностика не обязана работать
            info["source"] = f"нет данных: probe недоступен ({type(exc).__name__}: {exc})"
        else:
            if probe.error:
                info["source"] = probe.error
                info["components"] = {}
            else:
                payload = probe.to_dict()
                info["source"] = payload["source"]
                info["commit"] = payload["commit"]
                info["declared_pipeline"] = payload["pipeline"]
                info["components"] = {item["name"]: item for item in payload["components"]}
                info["total_bytes"] = payload["total_size"]
                if probe.model_index:            # точнее локальных дефолтов (напр. из Хаба)
                    pipeline_name, spec = _pipeline_spec(probe.model_index)
                    info.update(
                        pipeline=pipeline_name,
                        encoder_class=_encoder_class_name(probe.model_index, spec),
                        text_encoder_out_layers=list(spec["out_layers"]),
                        caption_upsample=bool(spec["caption_upsample"]),
                        pipeline_source=f"model_index.json из {payload['source']}",
                    )
            if probe.note:
                info["note"] = probe.note

        cache_dir = _precision_cache_dir(cfg)
        if cache_dir is None:
            info["precision_cache"] = {"path": None, "ready": False,
                                       "status": "выключен (generation.precision_cache='')"}
        else:
            config_file = cache_dir / "config.json"
            files = sorted(item.name for item in cache_dir.glob("*")) if cache_dir.is_dir() else []
            size = 0
            for name in files:
                try:
                    size += (cache_dir / name).stat().st_size
                except OSError:
                    pass
            info["precision_cache"] = {
                "path": str(cache_dir),
                "ready": config_file.is_file(),
                "bytes": size,
                "files": files,
                "status": ("готов (старт читает fp8 вместо bf16 + каста)" if config_file.is_file()
                           else "нет — первый запуск сконвертирует bf16 -> fp8 и запишет его"),
            }

        versions: dict[str, str] = {}
        for module_name in ("torch", "diffusers", "transformers"):
            try:
                versions[module_name] = __import__(module_name).__version__
            except Exception:                        # noqa: BLE001
                continue
        info["versions"] = versions
        info["verdict"] = hardware_budget_verdict(self.budget)
        return info


__all__ = [
    "CACHE_STAGE", "DEFAULT_GUIDANCE", "DEFAULT_MAX_SEQUENCE_LENGTH", "DEFAULT_PIPELINE",
    "DEFAULT_REPO", "PIPELINE_SPECS", "Flux2Backend", "Flux2Handle",
]
