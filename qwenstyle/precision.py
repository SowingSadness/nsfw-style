"""fp8-каст и дисковый кэш для больших трансформеров (R1, R4).

Зачем отдельный модуль: в `backends/qwen_image21.py` уже живёт рабочая схема «каст в fp8 + дисковый
кэш + точные не-fp8 тензоры + буферы из сайдкара», но она вшита в бекенд и рассчитана на 14 ГиБ
трансформер. 33B DiT MiniMax-H3-Image (61.7 ГиБ bf16 → 30.9 ГиБ fp8) требует того же смысла, но с
**потоковой** записью, иначе `save_pretrained` держит в RAM весь fp8-модель. Поэтому здесь
переиспользуемая версия, а `qwen_image21.py` не трогается (нулевой риск регресса).

Что именно кастуется (и почему именно так)
------------------------------------------
Классификация «fp8 vs исходный dtype» повторяет то, что сделал бы
`ModelMixin.enable_layerwise_casting(storage_dtype=fp8, compute_dtype=bf16)` для этой же модели,
а не эвристику по именам: набор пропусков = `DEFAULT_SKIP_MODULES_PATTERN` диффузеров
(`pos_embed`, `patch_embed`, `norm`, `proj_in`, `proj_out`) + объявления самого класса
(`_keep_in_fp32_modules`, `_skip_layerwise_casting_patterns`) + `extra_dtype_patterns`. Каст
применяется только к весам/байесам модулей из `_GO_LC_SUPPORTED_PYTORCH_LAYERS` (Linear/Conv/
Embedding) — ровно там, где диффузеры вешают хуки.

Отсюда следует то, что обязано остаться точным:

* нормы (`norm1`, `norm2`, `norm_q`, `norm_k`, `norm_out.norm`, `final_norm`) — `norm` в пропусках;
* `proj_in` (patch-проекция), `time_embedder`, `pdd_proj_out` (головы по шагам) — у MiniMax-H3-Image
  они **float32 в чекпойнте** (`_keep_in_fp32_modules`), каст в fp8 их бы испортил;
* `rope.inv_freq` — единственный буфер, `persistent=False`, в файл весов не попадает вообще.

Буферы `persistent=False` восстанавливаются из сайдкара `non_persistent.buffers.pt`: диффузеры
считают их из конфига, а общий `.to(fp8)` их портит (e4m3 не держит 1e-4…1e0), поэтому в fp8-модели
они читаются как fp8 и обязаны быть заменены точными значениями.

Гарантии
--------
* **Пик RAM** при касте не зависит от размера модели: тензоры читаются по одному из mmap, буфер
  готового шарда ограничен `MAX_PENDING_BYTES` (2 ГиБ) — на 30.9 ГиБ fp8 это десятки шардов вместо
  одного 30-гигабайтного словаря (и вместо «весь bf16 в RAM» у `from_pretrained`). Замер на
  MiniMax-H3-Image: 644 тензора, крупнейший 248 МиБ, кэш 30.93 ГиБ в 17 шардах → пик каста ≈ 2.2 ГиБ
  вместо 61.7 ГиБ (bf16 всей модели) или 30.9 ГиБ (`save_pretrained` после `from_pretrained(fp8)`).
* **Атомарность (R1)**: пишем в `<cache>.tmp-<pid>`, потом `os.replace`; старый каталог уводится в
  `<cache>.old-<pid>` и удаляется. Прерванный каст не оставляет валидного кэша.
* **Reuse**: повторный вызов при готовом кэше ничего не пересоздаёт — сверяются схема, версии
  torch/diffusers, `fp8_dtype`, набор пропусков, digest источника (имена/формы/dtype всех тензоров)
  и размеры шардов кэша.
* Метаданные (R1) — `precision.json` рядом: `source` (+ commit, если путь из `snapshots/<sha>`),
  версии, `fp8_dtype`, `skip_patterns`, списки fp8/точных тензоров, список буферов, дата.

Тяжёлые зависимости (`torch`, `safetensors`, `diffusers`, `accelerate`) импортируются **лениво**,
внутри функций: модуль обязан импортироваться без torch (как `streaming.py`), иначе `models list`
и `--dry-run` потянут 2.8 ГиБ DLL ни за что.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "BUFFERS_NAME",
    "CACHE_SCHEMA",
    "INDEX_NAME",
    "MAX_PENDING_BYTES",
    "SIDECAR_NAME",
    "cast_and_cache",
    "load_cached",
]

#: Версия формата кэша. Менять при смене раскладки/сайдкара: старые записи перестанут считаться
#: валидными и будут пересобраны (bump = один разовый каст).
CACHE_SCHEMA = 1

#: Сайдкар с параметрами каста (R1).
SIDECAR_NAME = "precision.json"
#: Буферы `persistent=False`, которых нет в файле весов (`rope.inv_freq` и пр.).
BUFFERS_NAME = "non_persistent.buffers.pt"
#: Имена как у `save_pretrained`, чтобы `from_pretrained` читал кэш штатным путём.
WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
SHARD_NAME = "diffusion_pytorch_model-{index:05d}-of-{count:05d}.safetensors"

#: Потолок «незаписанного» буфера шарда: он же пик RAM каста (минус один тензор).
MAX_PENDING_BYTES = 2 * 1024 * 1024 * 1024

#: Пропуски по умолчанию — `diffusers.hooks.layerwise_casting.DEFAULT_SKIP_MODULES_PATTERN`.
DEFAULT_SKIP_PATTERNS: tuple[str, ...] = ("pos_embed", "patch_embed", "norm", "^proj_in$", "^proj_out$")

#: safetensors dtype -> байт на элемент (для оценки размера кэша до чтения данных).
_ITEMSIZE = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
}
#: dtypes, которые каст-план переводит в fp8 (уже-fp8 и целочисленные не трогаем).
_CASTABLE_DTYPES = ("F16", "BF16", "F32")
_FP8_DTYPE_NAMES = ("F8_E4M3", "F8_E5M2")


# --------------------------------------------------------------------------- #
# Ленивые импорты и мелочи
# --------------------------------------------------------------------------- #

def _torch():
    import torch

    return torch


def _diffusers_version() -> str:
    import diffusers

    return str(diffusers.__version__)


def _dtype_name(dtype: Any) -> str:
    """`torch.bfloat16` -> `"bfloat16"` (стабильная строка для имени кэша и сайдкара)."""
    return str(dtype).replace("torch.", "")


def _resolve_fp8_dtype(fp8_dtype: Any, torch: Any) -> Any:
    """`None` -> `float8_e4m3fn`; принимает и `torch.dtype`, и строку (`"float8_e4m3fn"`)."""
    if fp8_dtype is None:
        return torch.float8_e4m3fn
    if isinstance(fp8_dtype, torch.dtype):
        return fp8_dtype
    name = str(fp8_dtype).strip()
    value = getattr(torch, name, None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"неизвестный fp8_dtype: {fp8_dtype!r}")
    return value


def _safetensors_name(dtype: Any, torch: Any) -> str:
    """torch dtype -> имя dtype в заголовке safetensors (`F8_E4M3`)."""
    if dtype == torch.float8_e4m3fn:
        return "F8_E4M3"
    if dtype == torch.float8_e5m2:
        return "F8_E5M2"
    return {
        torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16",
        torch.float64: "F64", torch.int64: "I64", torch.int32: "I32",
        torch.int16: "I16", torch.int8: "I8", torch.uint8: "U8", torch.bool: "BOOL",
    }[dtype]


def _human(nbytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024 or unit == "TiB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{int(nbytes)} B"
        nbytes /= 1024
    return f"{nbytes:.1f} TiB"


def _replace_dir(source: Path, target: Path) -> None:
    """`os.replace` каталога: на Windows цели быть не должно, поэтому старую уводим в `.old-<pid>`."""
    if target.exists():
        stale = target.with_name(f"{target.name}.old-{os.getpid()}")
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)
        os.replace(target, stale)
        os.replace(source, target)
        shutil.rmtree(stale, ignore_errors=True)
    else:
        os.replace(source, target)


# --------------------------------------------------------------------------- #
# Источник: каталог, шарды, digest, meta-модель
# --------------------------------------------------------------------------- #

def _resolve_source(source: str | Path) -> Path:
    """Локальный каталог весов. Если путь не каталог — пробуем HF-кэш (оффлайн, R3)."""
    path = Path(source).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.exists():
        raise NotADirectoryError(f"{source} — не каталог с весами трансформера")
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:                                   # noqa: BLE001 — hub есть, но не обязателен
        raise FileNotFoundError(
            f"нет каталога {source} и нет huggingface_hub для поиска в кэше"
        ) from exc
    try:
        return Path(snapshot_download(str(source), local_files_only=True)).resolve()
    except Exception as exc:                                   # noqa: BLE001 — оффлайн-промах
        raise FileNotFoundError(
            f"нет локальных весов {source}: скачай их явно "
            f"(`python -m qwenstyle models download {source} --yes`) и повтори"
        ) from exc


def _source_revision(path: Path) -> str | None:
    """Commit, если каталог пришёл из HF-кэша (`.../snapshots/<sha>/...`)."""
    parts = path.parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def _read_header(file: Path) -> dict[str, dict[str, Any]]:
    """Заголовок safetensors без чтения данных: `имя -> {dtype, shape, data_offsets}`."""
    with open(file, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise ValueError(f"{file}: не safetensors (файл короче 8 байт)")
        length = int.from_bytes(raw, "little")
        payload = fh.read(length)
    if len(payload) != length:
        raise ValueError(f"{file}: обрезанный заголовок safetensors")
    header = json.loads(payload)
    header.pop("__metadata__", None)
    return header


def _source_shards(source: Path) -> list[Path]:
    """Шарды источника в порядке индекса (или единственный файл весов)."""
    index = source / INDEX_NAME
    if index.is_file():
        try:
            data = json.loads(index.read_text(encoding="utf-8"))
            names = sorted(set(data["weight_map"].values()))
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"{index}: не читается карта весов ({exc})") from exc
        shards = [source / name for name in names]
    else:
        single = source / WEIGHTS_NAME
        if not single.is_file():
            raise FileNotFoundError(
                f"{source}: нет ни {WEIGHTS_NAME}, ни {INDEX_NAME} — это не каталог весов"
            )
        shards = [single]
    missing = [str(f) for f in shards if not f.is_file()]
    if missing:
        raise FileNotFoundError(f"{source}: нет шардов весов: {', '.join(missing)}")
    return shards


def _source_entries(shards: list[Path]) -> list[dict[str, Any]]:
    """Плоский список тензоров источника (порядок шардов/файла сохраняется).

    Читаются только заголовки: `nbytes` берётся из `data_offsets`, поэтому размер будущего кэша
    известен до чтения данных.
    """
    entries: list[dict[str, Any]] = []
    for file in shards:
        for name, meta in _read_header(file).items():
            offsets = meta.get("data_offsets") or [0, 0]
            entries.append({
                "key": name,
                "file": file,
                "dtype": str(meta["dtype"]),
                "shape": list(meta["shape"]),
                "nbytes": int(offsets[1]) - int(offsets[0]),
            })
    return entries


def _entries_digest(entries: list[dict[str, Any]]) -> str:
    """Отпечаток источника: имена + dtype + формы (раскладка по шардам не участвует)."""
    digest = hashlib.sha256()
    for key, dtype, shape in sorted((e["key"], e["dtype"], tuple(e["shape"])) for e in entries):
        digest.update(f"{key}|{dtype}|{shape}\n".encode("utf-8"))
    return digest.hexdigest()


def _build_meta_model(model_cls: Any, source: Path) -> Any:
    """Модель на `meta`-параметрах и с **настоящими** буферами (`include_buffers=False`).

    Нужна ровно для двух вещей: план каста (какие веса диффузеры держали бы в fp8, а какие — точными)
    и значения буферов `persistent=False`. Ни один вес при этом не читается и не аллоцируется.
    """
    from accelerate import init_empty_weights

    config = model_cls.load_config(str(source))
    with init_empty_weights():
        return model_cls.from_config(config)


def _supported_layers(torch: Any) -> tuple[type, ...]:
    """Те же классы, на которые диффузеры вешают LayerwiseCastingHook."""
    try:
        from diffusers.hooks.layerwise_casting import _GO_LC_SUPPORTED_PYTORCH_LAYERS

        return tuple(_GO_LC_SUPPORTED_PYTORCH_LAYERS)
    except Exception:                                          # noqa: BLE001 — другая версия diffusers
        return (
            torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
            torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d,
            torch.nn.Linear, torch.nn.Embedding,
        )


def _skip_patterns(model_cls: Any, extra: tuple[str, ...]) -> tuple[str, ...]:
    """Пропуски каста: умолчания диффузеров + объявления класса + `extra_dtype_patterns`."""
    patterns: list[str] = list(DEFAULT_SKIP_PATTERNS)
    for attr in ("_keep_in_fp32_modules", "_skip_layerwise_casting_patterns"):
        for pattern in getattr(model_cls, attr, None) or ():
            if str(pattern).strip():
                patterns.append(str(pattern))
    for pattern in extra:
        if str(pattern).strip():
            patterns.append(str(pattern))
    unique: list[str] = []
    for pattern in patterns:
        if pattern not in unique:
            unique.append(pattern)
    return tuple(unique)


def _cast_plan(model: Any, patterns: tuple[str, ...]) -> dict[str, bool]:
    """`имя параметра -> кастовать ли в fp8`; повторяет обход `_apply_layerwise_casting`."""
    torch = _torch()
    layers = _supported_layers(torch)
    plan: dict[str, bool] = {}
    for module_name, module in model.named_modules():
        cast = isinstance(module, layers) and not any(
            re.search(pattern, module_name) for pattern in patterns
        )
        for param_name, param in module._parameters.items():
            if param is None:                    # `Linear(bias=False)` тоже регистрирует ключ `bias`
                continue
            key = f"{module_name}.{param_name}" if module_name else param_name
            plan[key] = cast
    return plan


def _non_persistent_buffers(model: Any) -> dict[str, Any]:
    """Буферы, которых не будет в файле весов (`persistent=False`), — как есть, с CPU-значениями."""
    persisted = set(model.state_dict())
    out: dict[str, Any] = {}
    for name, buffer in model.named_buffers():
        if name in persisted or buffer is None:
            continue
        out[name] = buffer.detach().clone()
    return out


# --------------------------------------------------------------------------- #
# Самостоятельность кэша: проверка перед переиспользованием
# --------------------------------------------------------------------------- #

def _cache_problem(
    cache: Path,
    *,
    digest: str,
    config_sha: str,
    fp8_dtype: Any,
    patterns: tuple[str, ...],
    torch: Any,
) -> str | None:
    """`None` — кэш валиден; иначе строка «что именно изменилось» (R1: печатать причину промаха)."""
    if not cache.is_dir():
        return "кэша нет"
    sidecar = cache / SIDECAR_NAME
    if not sidecar.is_file():
        return f"нет сайдкара {SIDECAR_NAME}"
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return f"сайдкар не читается ({exc})"
    if data.get("schema") != CACHE_SCHEMA:
        return f"схема кэша {data.get('schema')} != {CACHE_SCHEMA}"
    if data.get("fp8_dtype") != _dtype_name(fp8_dtype):
        return f"изменился fp8_dtype ({data.get('fp8_dtype')} -> {_dtype_name(fp8_dtype)})"
    if data.get("torch_version") != str(torch.__version__):
        return f"изменилась версия torch ({data.get('torch_version')} -> {torch.__version__})"
    current = _diffusers_version()
    if data.get("diffusers_version") != current:
        return f"изменилась версия diffusers ({data.get('diffusers_version')} -> {current})"
    if tuple(data.get("skip_patterns") or ()) != patterns:
        return "изменился набор пропусков каста (skip_patterns)"
    if data.get("source_digest") != digest:
        return "изменился источник (digest весов)"
    if data.get("config_sha256") != config_sha:
        return "изменился config.json источника"
    if not (cache / "config.json").is_file():
        return "в кэше нет config.json"
    index_name = str((data.get("weights") or {}).get("index") or INDEX_NAME)
    if not (cache / index_name).is_file():
        return f"в кэше нет индекса шардов {index_name}"
    for shard in (data.get("weights") or {}).get("shards") or []:
        path = cache / str(shard.get("name"))
        if not path.is_file():
            return f"нет шарда {shard.get('name')}"
        if path.stat().st_size != int(shard.get("size", -1)):
            return f"шард {shard.get('name')} другого размера"
    buffers = data.get("buffers") or {}
    if buffers.get("names") and not (cache / str(buffers.get("file") or BUFFERS_NAME)).is_file():
        return f"нет сайдкара буферов {buffers.get('file') or BUFFERS_NAME}"
    return None


# --------------------------------------------------------------------------- #
# Каст
# --------------------------------------------------------------------------- #

def cast_and_cache(
    model_cls: Any,
    source: str | Path,
    cache_dir: str | Path,
    *,
    fp8_dtype: Any = None,
    extra_dtype_patterns: tuple[str, ...] = (),
) -> Path:
    """Собирает fp8-кэш трансформера из `source` и возвращает каталог кэша.

    Args:
        model_cls: класс модели diffusers (`MiniMaxH3ImageTransformer2DModel` и любой другой):
            нужны `load_config`/`from_config`/`from_pretrained` и классовое объявление точных модулей
            (`_keep_in_fp32_modules`, `_skip_layerwise_casting_patterns`).
        source: каталог с `config.json` и весами bf16/fp32 (`diffusion_pytorch_model[.safetensors
            .index.json]`) либо репозиторий Хаба, который ищется в локальном кэше (оффлайн).
        cache_dir: каталог кэша. Готовый и совпадающий по параметрам кэш **переиспользуется**.
        fp8_dtype: dtype хранения (по умолчанию `torch.float8_e4m3fn`); принимает строку.
        extra_dtype_patterns: дополнительные regex-пропуски (модуль -> держать исходный dtype),
            например `("adaln_proj",)` если качество потребует точной модуляции.

    Возвращает путь кэша. Печатает одну строку на промах (что именно инвалидировало) и одну на hit.
    """
    torch = _torch()
    from safetensors import safe_open
    from safetensors.torch import save_file

    started = time.time()
    out_dtype = _resolve_fp8_dtype(fp8_dtype, torch)
    src = _resolve_source(source)
    cache = Path(cache_dir).expanduser().resolve()
    shards = _source_shards(src)
    entries = _source_entries(shards)
    digest = _entries_digest(entries)
    config_bytes = (src / "config.json").read_bytes()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    patterns = _skip_patterns(model_cls, extra_dtype_patterns)

    problem = _cache_problem(cache, digest=digest, config_sha=config_sha, fp8_dtype=out_dtype,
                             patterns=patterns, torch=torch)
    if problem is None:
        sidecar = json.loads((cache / SIDECAR_NAME).read_text(encoding="utf-8"))
        print(
            f"[precision] fp8-кэш переиспользован: {cache} "
            f"(fp8-тензоров {sidecar.get('num_fp8')}, точных {sidecar.get('num_exact')}, "
            f"буферов {len((sidecar.get('buffers') or {}).get('names') or [])})",
            flush=True,
        )
        return cache

    print(f"[precision] каста нет: {problem}; каст {_human(sum(e['nbytes'] for e in entries))} "
          f"-> fp8 ({_dtype_name(out_dtype)})", flush=True)

    model = _build_meta_model(model_cls, src)
    plan = _cast_plan(model, patterns)
    buffers = _non_persistent_buffers(model)
    params_total = 0
    # Что получится на диске: fp8 только для каст-плана и только для float-dtypes.
    out_names: dict[str, str] = {}                    # ключ -> dtype тензора в кэше
    unknown = 0
    for entry in entries:
        key, disk_dtype = entry["key"], entry["dtype"]
        if key not in plan:
            unknown += 1                              # неожиданный ключ: держим точным
        if plan.get(key) and disk_dtype in _CASTABLE_DTYPES:
            out_names[key] = _safetensors_name(out_dtype, torch)
        else:
            out_names[key] = disk_dtype
    fp8_keys = sorted(k for k, v in out_names.items() if v in _FP8_DTYPE_NAMES)
    exact_keys = sorted(k for k, v in out_names.items() if v not in _FP8_DTYPE_NAMES)

    # Размер будущего кэша известен заранее (shape x itemsize): по нему же считается число шардов —
    # имена файлов содержат `-NNNNN-of-MMMMM`, поэтому M нужно знать до записи первого шарда.
    sizes = []
    for entry in entries:
        numel = 1
        for dim in entry["shape"]:
            numel *= int(dim)
        params_total += numel
        sizes.append(numel * _ITEMSIZE[out_names[entry["key"]]])
    total_out = sum(sizes)
    shard_count = 1
    pending = 0
    for size in sizes:
        if pending and pending + size > MAX_PENDING_BYTES:
            shard_count += 1
            pending = 0
        pending += size

    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(f"{cache.name}.tmp-{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        write_started = time.time()
        shard_index = 0
        written: list[dict[str, Any]] = []
        pending_tensors: dict[str, Any] = {}
        pending_bytes = 0
        weight_map: dict[str, str] = {}

        def flush() -> None:
            nonlocal pending_tensors, pending_bytes, shard_index
            if not pending_tensors:
                return
            name = SHARD_NAME.format(index=shard_index + 1, count=shard_count)
            save_file(pending_tensors, str(tmp / name), metadata={"format": "pt"})
            written.append({"name": name, "size": (tmp / name).stat().st_size,
                            "tensors": len(pending_tensors)})
            for key in pending_tensors:
                weight_map[key] = name
            shard_index += 1
            # Освобождаем ссылки до следующего шарда: держать их = вернуть пик RAM.
            pending_tensors = {}
            pending_bytes = 0

        for file in shards:
            with safe_open(str(file), framework="pt", device="cpu") as st:
                for key in st.keys():
                    stored = out_names.get(key)
                    if stored is None:
                        raise KeyError(f"{key}: тензор исчез между чтением заголовка и данных")
                    tensor = st.get_tensor(key)
                    wanted = _dtype_from_name(stored, torch)
                    if tensor.dtype != wanted:
                        tensor = tensor.to(wanted)
                    tensor = tensor.contiguous()
                    size = tensor.numel() * tensor.element_size()
                    if pending_bytes and pending_bytes + size > MAX_PENDING_BYTES:
                        flush()
                    pending_tensors[key] = tensor
                    pending_bytes += size
        flush()

        if shard_index != shard_count:
            raise RuntimeError(f"шардов записано {shard_index}, ожидалось {shard_count}")
        if len(weight_map) != len(entries):
            raise RuntimeError(f"в кэш попало {len(weight_map)} тензоров из {len(entries)}")

        (tmp / INDEX_NAME).write_text(
            json.dumps({"metadata": {"total_size": total_out}, "weight_map": weight_map},
                       ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(src / "config.json", tmp / "config.json")
        buffers_file: str | None = None
        if buffers:
            torch.save(buffers, tmp / BUFFERS_NAME)
            buffers_file = BUFFERS_NAME

        sidecar = {
            "schema": CACHE_SCHEMA,
            "kind": "fp8-transformer-cache",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": str(src),
            "source_revision": _source_revision(src),
            "source_digest": digest,
            "config_sha256": config_sha,
            "fp8_dtype": _dtype_name(out_dtype),
            "torch_version": str(torch.__version__),
            "diffusers_version": _diffusers_version(),
            "skip_patterns": list(patterns),
            "extra_dtype_patterns": [p for p in extra_dtype_patterns if str(p).strip()],
            "weights": {
                "index": INDEX_NAME,
                "shards": written,
                "total_bytes": total_out,
                "max_pending_bytes": MAX_PENDING_BYTES,
            },
            "num_fp8": len(fp8_keys),
            "num_exact": len(exact_keys),
            "non_fp8": exact_keys,
            "buffers": {"file": buffers_file, "names": sorted(buffers)},
            "num_tensors": len(entries),
            "unknown_keys": unknown,
            "params_estimate": params_total,            "seconds": round(time.time() - started, 1),
        }
        (tmp / SIDECAR_NAME).write_text(json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
        _replace_dir(tmp, cache)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    print(
        f"[precision] fp8-кэш записан: {cache} — {_human(total_out)} в {len(written)} шардах, "
        f"fp8-тензоров {len(fp8_keys)}, точных {len(exact_keys)}, буферов {len(buffers)}; "
        f"каст {time.time() - write_started:.1f}s, пик RAM каста ≈ буфер шарда "
        f"({_human(MAX_PENDING_BYTES)})",
        flush=True,
    )
    return cache


def _dtype_from_name(name: str, torch: Any) -> Any:
    """Имя dtype из заголовка safetensors -> torch dtype."""
    return {
        "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
        "BF16": torch.bfloat16, "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
        "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
        "U8": torch.uint8, "BOOL": torch.bool,
    }[name]


# --------------------------------------------------------------------------- #
# Загрузка
# --------------------------------------------------------------------------- #

def load_cached(model_cls: Any, cache_dir: str | Path, *, fp8_dtype: Any = None) -> Any:
    """Читает fp8-кэш: fp8-тензоры из файла, не-fp8 — точно с диска, буферы — из сайдкара.

    `from_pretrained(torch_dtype=fp8)` конвертирует ВСЁ, включая нормы, `proj_in`,
    `time_embedder` и `pdd_proj_out`, которые чекпойнт держит в float32/bf16: обратный каст
    bf16 -> fp8 -> bf16 невосстановим, а на буферах (`rope.inv_freq`) forward падает с
    `Promotion for Float8 Types is not supported`. Поэтому не-fp8 тензоры (их единицы-десятки)
    перечитываются с диска в исходном dtype — так же, как это делает `qwen_image21`.

    Модель возвращается **без** хуков layerwise casting: включить их —
    `model.enable_layerwise_casting(storage_dtype=fp8, compute_dtype=bf16)` (иначе `addmm` для fp8
    в torch не реализован и первый же forward упадёт).
    """
    torch = _torch()
    from safetensors import safe_open

    cache = Path(cache_dir).expanduser().resolve()
    if not (cache / "config.json").is_file():
        raise FileNotFoundError(f"нет fp8-кэша {cache}: сначала cast_and_cache()")
    out_dtype = _resolve_fp8_dtype(fp8_dtype, torch)

    sidecar: dict[str, Any] = {}
    sidecar_path = cache / SIDECAR_NAME
    if sidecar_path.is_file():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            sidecar = {}
    stored = sidecar.get("fp8_dtype")
    if stored and stored != _dtype_name(out_dtype):
        print(f"[precision] внимание: кэш собран в {stored}, грузим как {_dtype_name(out_dtype)}",
              flush=True)

    model = model_cls.from_pretrained(
        str(cache), torch_dtype=out_dtype, low_cpu_mem_usage=True, local_files_only=True,
    )

    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    exact = 0
    fp8_seen = 0
    for file in sorted(cache.glob("*.safetensors")):
        header = _read_header(file)
        keys = [name for name, meta in header.items() if not str(meta["dtype"]).startswith("F8")]
        with safe_open(str(file), framework="pt", device="cpu") as st:
            for name in keys:
                tensor = st.get_tensor(name)                  # ровно тот dtype, что на диске
                if name in params:
                    params[name].data = tensor
                elif name in buffers:
                    # persistent-буфер в файле (float dtype): `from_pretrained(torch_dtype=fp8)`
                    # кастит и его, поэтому значение возвращается точным на место.
                    parent_path, _, leaf = name.rpartition(".")
                    parent = model.get_submodule(parent_path) if parent_path else model
                    setattr(parent, leaf, tensor)
                exact += 1
            fp8_seen += len(header) - len(keys)

    buffers_file = (sidecar.get("buffers") or {}).get("file") or BUFFERS_NAME
    np_path = cache / str(buffers_file)
    restored = 0
    if np_path.is_file():
        for name, tensor in torch.load(np_path, map_location="cpu").items():
            parent_path, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, leaf, tensor)
            restored += 1
        fallback = 0
    else:
        # Старый кэш без сайдкара: хоть что-то вместо fp8 (лучше пересобрать каст).
        fallback = 0
        fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        for module in model.modules():
            for buffer_name, buffer in list(module._buffers.items()):
                if buffer is not None and buffer.dtype in fp8_dtypes:
                    module._buffers[buffer_name] = buffer.to(torch.float32)
                    fallback += 1

    print(
        f"[precision] fp8-кэш прочитан: {cache} — fp8-тензоров {fp8_seen}, точных {exact}, "
        f"буферов из сайдкара {restored}"
        + (f", fallback-конверсий {fallback}" if fallback else ""),
        flush=True,
    )
    return model
