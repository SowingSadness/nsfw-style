"""Бекенд генерации Qwen-Image-2.1 (base + turbo) — перенос Qwen-специфики из `run_style.py` 1:1.

Модуль знает про Qwen всё, что раньше жило в `main()`/`load_fp8_transformer`/`load_lora` монолита:
fp8-кэш трансформера (включая точные не-fp8 тензоры и sidecar `non_persistent_buffers.pt`),
`enable_layerwise_casting`, LoRA через state dict со стрипом 224 `alpha` (=128), bf16-каст
LoRA-тензоров (fp8 `addmm` не реализован), `enable_group_offload`, регистрацию turbo `sample_sigmas`
из `model_index.json`, подмену `encode_prompt` готовыми эмбеддингами и `text_encoder=None`
при попадании в кэш эмбеддингов.

**Порядок операций менять нельзя** (каждый шаг влияет на картинку, проверка — паритет sha256):
`prepare` = fp8-кэш → bf16-трансформер (hybrid/turbo) → `from_pretrained` → turbo sigmas →
`enable_layerwise_casting` (+запись fp8-кэша) → LoRA → bf16-каст LoRA → размещение
(`--offload group` либо `.to("cuda")`) → VAE cuda/slicing/tiling; затем `encode` → `generate`.
Внутри `generate` — тоже по порядку: эмбеддинги на GPU → подмена `encode_prompt` → `pipe(...)`.

Тяжёлые зависимости (`torch`, `diffusers`, `safetensors`, `huggingface_hub`) импортируются лениво
внутри функций: `discover()`/`models list` не должны тянуть 2 ГБ библиотек, а импорт модуля не
должен падать на машине без CUDA.

Два ключа `generation.sampler_kwargs` — не про сэмплер, а про эту модель (в `RunConfig` нет
отдельных полей, а `cache_key_parts()` смотрит только на `CACHE_SAMPLER_KEYS`, поэтому ключ
эмбеддингов они не меняют):

* `offload_blocks` (по умолчанию 1) и `offload_stream` (1) — параметры `enable_group_offload`;
* `rebuild_precision_cache` (он же `rebuild_fp8_cache`) — эквивалент `--rebuild-fp8-cache`
  из `run_style.py`: пересобрать fp8-кэш вместо чтения готового.

Оба объявлены в `parameters()` вместе с `steps`/`guidance`/`offload`/`precision`/`max_images`:
это единственный источник, по которому ядро проверяет `--set` и печатает `--explain-params`.
"""
from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ..config import encoder_device_canon
from ..registry import register_generation
from .base import REPO_ROOT, Capabilities, ParameterSpec, ResourceBudget, as_image_list

#: Репозиторий пайплайна (text-encoder + VAE + processor + transformer).
DEFAULT_REPO = "Qwen/Qwen-Image-2.1"
#: Turbo: берём только transformer (архитектура та же), энкодер и VAE остаются базовые.
TURBO_REPO = "Qwen/Qwen-Image-2.1-Turbo"
LORA_REPO = "MicksHF/Qwen-Image-2.1-Anime-Fusal"
BASE_LORA = "fusal-qwen-image-2.1-v2-exact-r80.safetensors"
TURBO_LORA = "fusal-qwen-image-2.1-turbo-e6-preview-r128.safetensors"

#: Запасные пути fp8-кэша (кеш моделей, шаг 4: `cache/models/...`); пресеты главнее.
DEFAULT_FP8_CACHE = {"base": "cache/models/fp8_transformer",
                     "turbo": "cache/models/fp8_transformer_turbo"}


# --------------------------------------------------------------------------- #
# Handle
# --------------------------------------------------------------------------- #

@dataclass
class QwenImage21Handle:
    """Загруженный пайплайн + всё состояние, нужное `encode`/`generate`/`release`.

    `steps` — разрешённое число шагов (turbo-расписание из `model_index.json` главнее конфига),
    `input_image`/`crop`/`resolution` — кэш сетки conditioning (считаются один раз на кадр).
    """

    pipe: Any
    transformer: Any
    label: str                              # "base" | "turbo"
    repo: str
    fp8: bool
    steps: int
    sigmas: tuple[float, ...] | None = None
    precision_cache: Path | None = None
    input_image: Any = None
    crop: tuple[int, int] | None = None
    resolution: int | None = None
    #: где считает энкодер (`cuda` = наш GPU-стриминг, `cpu` = обычная загрузка diffusers/реестра)
    encoder_device: str = "cpu"
    #: энкодер собран нашим стримингом (пустые параметры) — его можно отпустить после энкода
    encoder_from_stream: bool = False


# --------------------------------------------------------------------------- #
# Ленивые импорты и мелкие помощники
# --------------------------------------------------------------------------- #

def _torch():
    import torch

    return torch


def _sampler_kwarg(cfg: Any, key: str, default: Any) -> Any:
    kwargs = getattr(cfg.generation, "sampler_kwargs", None) or {}
    return kwargs.get(key, default)


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
            if value.strip().lower() in ("1", "true", "yes", "on"):
                return True
        elif value:
            return True
    return False


def _is_hybrid(cfg: Any) -> bool:
    """`generation.transformer` — подмена трансформера (turbo), а не тот же repo пайплайна."""
    spec = cfg.generation.transformer
    return bool(spec) and str(spec[0]) != str(cfg.generation.repo)


def _precision_cache_dir(cfg: Any, label: str) -> Path | None:
    """Папка fp8-кэша: `generation.precision_cache`; пустая строка = не кэшировать (None)."""
    raw = cfg.generation.precision_cache
    if raw is None:
        raw = DEFAULT_FP8_CACHE.get(label)
    text = str(raw).strip()
    return Path(text) if text else None


def _load_text_encoder(cfg: Any) -> Any | None:
    """Энкодер из реестра (`encoder:heretic` и пр.). Для `base` — None: грузит сам бекенд.

    Подключается только при промахе кэша эмбеддингов (иначе `prepare` передаёт `text_encoder=None`).
    Бекенд энкодера (`qwenstyle/backends/encoders.py`) отвечает за веса, dtype и устройство.
    `cfg.encoder` приходит уже разрешённым (`EncoderConfig.resolved()`): `auto` -> `cpu`/`cuda`.
    """
    name = str(cfg.encoder.name or "base").strip()
    if name in ("", "base"):
        return None                     # штатный Qwen3VL-8B — его собирает `_load_base_encoder_gpu`
    from .. import registry

    try:
        cls = registry.get_encoder(name)
    except Exception as exc:             # noqa: BLE001 — понятная ошибка вместо RegistryError
        raise RuntimeError(f"текст-энкодер {name!r} не зарегистрирован: {exc}") from exc
    where = cfg.encoder.local_dir or cfg.encoder.repo
    print(f"[te] энкодер {name} из {where} ({cfg.encoder.dtype}, {cfg.encoder.device})...", flush=True)
    return cls().load(cfg)


def _encoder_device(cfg: Any) -> str:
    """Фактическое устройство энкода: `encoder.device` с разрешённым `auto` (`cuda`/`cpu`).

    `encoder_device_canon` кэширует проверку CUDA на процесс, тут это просто чтение строки.
    """
    return encoder_device_canon(getattr(cfg.encoder, "device", None))


def _load_base_encoder_gpu(cfg: Any) -> Any:
    """Собрать штатный Qwen3VL-8B нашим стриминговым загрузчиком на GPU (`pipe.text_encoder`).

    Зачем подмена: `encoder.name=base` раньше не проходил через `streaming.py` — 17.5 ГБ весов грузил
    сам diffusers на CPU, и энкод промта стоил 255–682 s. Стриминг отдаёт модель с пустыми
    параметрами: на время forward модуля веса читаются с диска в VRAM и отпускаются, поэтому
    энкод укладывается в ~15–25 s и не требует 17.5 ГБ RAM.
    """
    from huggingface_hub.constants import HF_HUB_OFFLINE
    from transformers import Qwen3VLForConditionalGeneration   # лениво, ~0.5 GiB RSS

    from .. import streaming

    enc = cfg.encoder.resolved()
    device = _encoder_device(cfg)
    pin = bool(cfg.encoder_pin_memory)          # offload=group -> pinned уже занят (см. RunConfig)
    if pin != bool(enc.pin_memory):
        print("[te] pin_memory выключен: трансформер уже держит pinned-память (offload group), "
              "а на Windows pin — тот же ресурс (см. AGENTS.md §4)", flush=True)
    print(f"[te] энкодер base: {device}-стриминг (stream=True, prefetch={enc.prefetch}, "
          f"pin_memory={int(pin)}, lookahead={enc.lookahead}) — 17.5 ГБ не грузим "
          f"целиком", flush=True)
    return streaming.load_text_encoder(
        Qwen3VLForConditionalGeneration,
        str(cfg.encoder.repo),
        subfolder="text_encoder",
        revision=cfg.encoder.revision,
        dtype=cfg.encoder.dtype,
        local_files_only=HF_HUB_OFFLINE,
        device=device,
        stream=True,
        prefetch=int(enc.prefetch),
        pin_memory=pin,
        lookahead=int(enc.lookahead),
    )


# --------------------------------------------------------------------------- #
# fp8-кэш трансформера
# --------------------------------------------------------------------------- #

def _load_fp8_transformer(path: Path) -> Any:
    """Загружает fp8-трансформер из кэша, сохраняя исходные dtype не-fp8 тензоров.

    Перенос `run_style.load_fp8_transformer` 1:1. `from_pretrained(dtype=fp8)` конвертирует ВСЁ,
    включая нормы и patch/pos-embed, которые `enable_layerwise_casting` оставляет в bf16 —
    обратный каст bf16 -> fp8 -> bf16 невосстановим, а на буферах (freqs) forward вообще падает
    с `Promotion for Float8 Types is not supported`. Поэтому не-fp8 тензоры (32 МиБ) читаются
    с диска в оригинальном dtype.
    """
    from diffusers.models import QwenImage21Transformer2DModel
    from safetensors import safe_open

    torch = _torch()
    tr = QwenImage21Transformer2DModel.from_pretrained(
        str(path), torch_dtype=torch.float8_e4m3fn, low_cpu_mem_usage=True,
    )
    params = dict(tr.named_parameters())
    buffers = dict(tr.named_buffers())
    exact = 0
    for f in sorted(path.glob("*.safetensors")):
        with f.open("rb") as fh:
            header = json.loads(fh.read(int.from_bytes(fh.read(8), "little")))
        keys = [n for n, m in header.items() if n != "__metadata__" and not m["dtype"].startswith("F8")]
        if not keys:
            continue
        with safe_open(str(f), framework="pt", device="cpu") as st:
            for name in keys:
                tensor = st.get_tensor(name)          # ровно тот dtype, что записан на диске
                if name in params:
                    params[name].data = tensor
                elif name in buffers:
                    buffers[name] = tensor
                exact += 1
    # Буферы persistent=False (синусоидальные freqs таймстеп-эмбеддинга) в файл весов не попадают,
    # а общий .to(fp8) их портит (fp8 e4m3 не держит 1e-4..1e0). Берём их из сайдкара.
    np_file = path / "non_persistent_buffers.pt"
    if np_file.is_file():
        for name, tensor in torch.load(np_file, map_location="cpu").items():
            parent_path, _, leaf = name.rpartition(".")
            parent = tr.get_submodule(parent_path) if parent_path else tr
            setattr(parent, leaf, tensor)
    # Запасной вариант для старых кэшей: хоть что-то вместо fp8. Лучше --rebuild-fp8-cache.
    buffers_fixed = 0
    if not np_file.is_file():
        for module in tr.modules():
            for bname, buf in list(module._buffers.items()):
                if buf is not None and buf.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                    module._buffers[bname] = buf.to(torch.float32)
                    buffers_fixed += 1
    print(f"    точных не-fp8 тензоров из файла: {exact}, sidecar-буферов: {np_file.is_file()}, "
          f"fallback-конверсий: {buffers_fixed}", flush=True)
    return tr


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #

def _load_lora(pipe: Any, cfg: Any) -> None:
    """LoRA. Turbo-адаптер несёт 224 per-module alpha (=128): их нужно вырезать и отдать state dict —
    иначе альфы попадают в state dict и масштаб едет (так предписывает модель-карточка).
    Перенос `run_style.load_lora` 1:1 (поведение не зависит от `lora.strip_alpha`: alpha-ключи
    определяются по факту наличия в файле).
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    lora = cfg.generation.lora
    if lora is None:
        print("    адаптер не задан в конфиге — LoRA не применяется", flush=True)
        return
    path = hf_hub_download(lora.repo, lora.filename)
    state = load_file(path, device="cpu")
    alpha_keys = [k for k in state if k.endswith(".alpha")]
    if alpha_keys:
        assert len(alpha_keys) == 224, f"неожиданное число alpha-ключей: {len(alpha_keys)}"
        for k in alpha_keys:
            assert float(state.pop(k)) == 128.0, f"{k}: alpha != 128"
        print(f"    alpha-ключей вырезано: {len(alpha_keys)} (rank/alpha = 128/128)", flush=True)
        pipe.load_lora_weights(state, adapter_name="fusal_anime")
    else:
        pipe.load_lora_weights(lora.repo, weight_name=lora.filename, adapter_name="fusal_anime")
    pipe.set_adapters(["fusal_anime"], adapter_weights=[lora.scale])


# --------------------------------------------------------------------------- #
# Локальный hf-cache (для probe; без импорта qwenstyle/cmd/*)
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


def _local_snapshot(repo: str) -> dict[str, Any] | None:
    """Снапшот репозитория из hf-cache: `model_index.json`, компоненты и их размеры.

    Обход `models--org--name/snapshots/<commit>/` (stat идёт по симлинку) — без Hub и без
    `scan_cache_dir`, чтобы probe работал оффлайн на любой раскладке кэша.
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
        sizes: dict[str, int] = {}
        for path in sorted(root.rglob("*")):
            if path.is_dir():
                continue
            try:
                size = path.stat().st_size     # stat следует по симлинку на blobs/
            except OSError:
                size = 0
            head, sep, _ = path.relative_to(root).as_posix().partition("/")
            key = head if sep else "(корень)"
            sizes[key] = sizes.get(key, 0) + int(size)
        index = _read_json(root / "model_index.json")
        components: dict[str, dict[str, Any]] = {}
        for name, value in (index or {}).items():
            if name.startswith("_") or not (isinstance(value, (list, tuple)) and len(value) == 2):
                continue
            config = _read_json(root / name / "config.json") or {}
            components[str(name)] = {
                "library": str(value[0]),
                "class": str(value[1]),
                "config_class": (config.get("_class_name")
                                 or ";".join(config.get("architectures") or []) or "-"),
                "size_bytes": int(sizes.get(str(name), 0)),
            }
        return {
            "source": f"hf-cache ({cache_dir})",
            "commit": root.name,
            "snapshot": str(root),
            "pipeline": (index or {}).get("_class_name"),
            "components": components,
            "component_sizes": sizes,
            "total_bytes": sum(sizes.values()),
        }
    return None


# --------------------------------------------------------------------------- #
# Бекенд
# --------------------------------------------------------------------------- #

@register_generation
class QwenImage21Backend:
    """Qwen-Image-2.1: base (R80/25 шагов) и turbo (свой transformer + R128/8 шагов).

    `conditioning="precomputed"`: эмбеддинги считаются здесь (`encode`) и кэшируются `.pt`
    оркестратором; попадание в кэш -> `prepare(need_encoder=False)` не грузит text-encoder (17.5 ГБ).
    """

    name: ClassVar[str] = "qwen_image21"
    caps: ClassVar[Capabilities] = Capabilities(
        conditioning="precomputed",
        # до 10 референсов в одном запросе: шаблон промта подставляет <image1>…<imageN>,
        # а vision-токены считаются по всем картинкам сразу (`encode_prompt(image=[...])`)
        image_inputs="multi",
        max_images=10,
        encoder="optional",
        fixed_steps=None,                 # turbo 8 шагов известно только после чтения sigmas
        supports_lora=True,
        precision=("bf16", "fp8"),
        offload=("none", "group"),
        max_pixels=None,
        extra_sampling=(),
    )
    budget: ClassVar[ResourceBudget] = ResourceBudget(
        weights_bf16_gib=14.2,
        weights_fp8_gib=6.6,
        encoder_gib=17.5,
        disk_gib=31.0,
        notes="transformer 14.2 GiB bf16 -> 6.6 GiB fp8 в VRAM; text-encoder 17.5 ГБ только в RAM "
              "и только при промахе кэша эмбеддингов; turbo добавляет +14 GiB (+6.7 GiB fp8-кэша); "
              "960p+ работает только с offload=group",
    )
    #: Репозитории, которые обслуживает бекенд (`models probe <repo>` -> `generation_for_repo`).
    repos: ClassVar[tuple[str, ...]] = (DEFAULT_REPO, TURBO_REPO)

    @staticmethod
    def parameters() -> tuple[ParameterSpec, ...]:
        """Настраиваемые параметры модели (шаг 9 плана публикации).

        Ядро этим списком валидирует `--set generation.sampler_kwargs.*` (неизвестный ключ —
        ошибка), печатает `config show --explain-params` и считает `models list`. В списке обязаны
        быть **все** ключи, которые бекенд читает сам: `offload_blocks`/`offload_stream`
        (`_sampler_kwarg`) и `rebuild_precision_cache`/`rebuild_fp8_cache` (флаг
        `--rebuild-fp8-cache`, читается `_rebuild_requested`).

        `guidance`/`steps` у turbo-трансформера фиксированы расписанием `sample_sigmas` (8 шагов,
        CFG-ветки нет), но в статике это не видно — предупреждение печатает `pipeline`, а не спека.
        """
        caps = QwenImage21Backend.caps
        return (
            ParameterSpec("steps", "generation", "int", 25, minimum=8, maximum=50,
                          doc="шагов сэмплера; у turbo расписание фиксировано (8, sample_sigmas)"),
            ParameterSpec("guidance", "generation", "float", 1.0, minimum=0.0, maximum=10.0,
                          doc="true_cfg_scale; у turbo расписание фиксировано, отрицательный промт "
                              "не используется"),
            ParameterSpec("offload", "generation", "enum", "none", choices=tuple(caps.offload),
                          doc="none = веса в VRAM; group = блоки стримятся с CPU (960p и выше)"),
            ParameterSpec("precision", "generation", "enum", "fp8", choices=tuple(caps.precision),
                          doc="fp8 = кэш cache/models/fp8_transformer*; bf16 = 14.2 GiB в VRAM"),
            ParameterSpec("max_images", "generation", "int", caps.max_images, readonly=True,
                          doc="предел модели: столько референсов принимает запрос "
                              "(Capabilities.max_images)"),
            ParameterSpec("offload_blocks", "sampler", "int", 1, minimum=1, maximum=64,
                          doc="блоков трансформера на GPU одновременно (offload=group)"),
            ParameterSpec("offload_stream", "sampler", "bool", True,
                          doc="асинхронная префетч-загрузка блоков (offload=group)"),
            ParameterSpec("rebuild_precision_cache", "sampler", "bool", False,
                          doc="пересобрать fp8-кэш вместо чтения готового"),
            ParameterSpec("rebuild_fp8_cache", "sampler", "bool", False,
                          doc="алиас rebuild_precision_cache (флаг --rebuild-fp8-cache)"),
        )

    # -- подготовка -------------------------------------------------------- #

    def prepare(self, cfg: Any, need_encoder: bool) -> QwenImage21Handle:
        """Грузит пайплайн в порядке run_style.py: fp8-кэш -> LoRA -> bf16 LoRA -> размещение.

        Порядок важен: каст в fp8 обязан случиться ДО LoRA (peft создаёт тензоры в dtype слоя,
        и bf16-адаптер поверх fp8 базы — единственная рабочая комбинация), а размещение на GPU —
        ПОСЛЕ обоих, иначе хуки offload/кастинга лягут на другие веса.

        Устройство энкода разрешается здесь один раз (`encoder.device=auto` -> `cuda`/`cpu`):
        при `cuda` штатный Qwen3VL-8B не попадает в `from_pretrained` (там его грузил бы diffusers —
        17.5 ГБ в RAM и CPU-энкод), а собирается нашим стримингом после размещения трансформера.
        """
        torch = _torch()
        from dataclasses import replace as _replace

        from diffusers import QwenImage21Pipeline
        from diffusers.models import QwenImage21Transformer2DModel
        from huggingface_hub.constants import HF_HUB_OFFLINE

        # `auto` разрешается один раз: дальше и загрузка, и ключи кэша видят фактическое устройство
        resolved = cfg.encoder.resolved()
        if resolved is not cfg.encoder:
            cfg = _replace(cfg, encoder=resolved)
        encoder_device = _encoder_device(cfg)
        gpu_encoder = encoder_device == "cuda"
        base_encoder = str(cfg.encoder.name or "base").strip() in ("", "base")

        gen = cfg.generation
        hybrid = _is_hybrid(cfg)                     # turbo: свой transformer, базовые TE/VAE
        label = "turbo" if hybrid or str(gen.repo) == TURBO_REPO else "base"
        fp8 = str(gen.precision or "").strip().lower() == "fp8"
        fp8_dir = _precision_cache_dir(cfg, label) if fp8 else None
        rebuild = _rebuild_requested(cfg)

        assert torch.cuda.is_available(), "torch собрана без CUDA"

        # base + GPU: энкодер собираем сами после пайплайна -> diffusers его не грузит вовсе
        defer_encoder = bool(need_encoder and gpu_encoder and base_encoder)
        if need_encoder and base_encoder and not gpu_encoder and resolved.stream:
            # не молчим: флаг рабочий только на GPU (CPU-стриминг для base не подключён)
            print("[te] streaming: для энкодера base он включается только при device=cuda — "
                  "на CPU веса грузит diffusers", flush=True)
        kwargs: dict[str, Any] = {}
        if not need_encoder:
            kwargs["text_encoder"] = None            # эмбеддинги в кэше -> 17.5 ГБ не грузим
        elif defer_encoder:
            kwargs["text_encoder"] = None            # 17.5 ГБ (CPU) не нужны: GPU-стриминг ниже
        else:
            text_encoder = _load_text_encoder(cfg)
            if text_encoder is not None:
                kwargs["text_encoder"] = text_encoder

        transformer = None
        tr_is_fp8 = False
        t = time.time()
        if fp8 and fp8_dir is not None and not rebuild and (fp8_dir / "config.json").is_file():
            transformer = _load_fp8_transformer(fp8_dir)
            kwargs["transformer"] = transformer
            tr_is_fp8 = True
            print(f"[1/5] fp8-трансформер из кэша {fp8_dir}", flush=True)
        elif hybrid:
            repo, sub = gen.transformer
            print(f"[1/5] трансформер из {repo} (bf16)...", flush=True)
            transformer = QwenImage21Transformer2DModel.from_pretrained(
                repo, subfolder=sub, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                # в оффлайне diffusers иначе дергает model_info() для проверки шардов
                local_files_only=HF_HUB_OFFLINE,
            )
            kwargs["transformer"] = transformer
        else:
            print("[1/5] загрузка базовой модели...", flush=True)
        pipe = QwenImage21Pipeline.from_pretrained(
            gen.repo,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            **kwargs,
        )
        te_label = ("не нужен" if not need_encoder
                    else "не грузим: GPU-стриминг после сборки" if defer_encoder
                    else "загружен, CPU")
        print(f"    ok, {time.time() - t:.1f}s (text_encoder: {te_label})", flush=True)

        # Turbo: расписание 8 шагов живёт в model_index.json его репозитория, а пайплайн собран из
        # базового (нужны его text-encoder, VAE и processor) — поэтому регистрируем sigmas вручную.
        if label == "turbo":
            try:
                from huggingface_hub import hf_hub_download

                src = str(gen.transformer[0]) if hybrid else str(gen.repo)
                index = json.loads(Path(hf_hub_download(src, "model_index.json")).read_text())
                if index.get("sample_sigmas"):
                    pipe.register_to_config(sample_sigmas=index["sample_sigmas"])
                    print(f"[schedule] turbo sample_sigmas ({len(index['sample_sigmas'])} шагов): "
                          f"{index['sample_sigmas']}", flush=True)
            except Exception as exc:                 # noqa: BLE001 — оффлайн без метаданных
                print(f"[schedule] не удалось прочитать turbo model_index.json "
                      f"({type(exc).__name__}) — запусти один раз с HF_HUB_OFFLINE=0", flush=True)

        # Порядок важен: каст bf16 -> fp8 идёт до LoRA (см. docstring prepare).
        if fp8:
            t = time.time()
            pipe.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn,
                compute_dtype=torch.bfloat16,
            )
            if tr_is_fp8:
                print(f"[1/5] хуки upcast fp8->bf16 включены ({time.time() - t:.1f}s)", flush=True)
            else:
                print(f"[1/5] bf16 -> fp8 cast ({time.time() - t:.1f}s)", flush=True)
                if fp8_dir is not None:
                    t = time.time()
                    pipe.transformer.save_pretrained(str(fp8_dir))
                    # буферы, которых нет в state_dict (freqs и пр.), но которые нужны для forward
                    state_keys = set(pipe.transformer.state_dict())
                    np_buffers = {n: b.clone() for n, b in pipe.transformer.named_buffers()
                                  if n not in state_keys}
                    torch.save(np_buffers, fp8_dir / "non_persistent_buffers.pt")
                    print(f"      fp8-кэш записан в {fp8_dir} ({time.time() - t:.1f}s), "
                          f"следующий старт быстрее", flush=True)

        print("[2/5] LoRA...", flush=True)
        _load_lora(pipe, cfg)

        # peft создаёт LoRA-тензоры в dtype базового слоя (fp8) — fp8 addmm не реализован,
        # поэтому адаптер держим в bf16.
        fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        cast = 0
        for name, param in pipe.transformer.named_parameters():
            if "lora" in name and param.dtype in fp8_dtypes:
                param.data = param.data.to(torch.bfloat16)
                cast += 1
        print(f"    LoRA-тензоров переведено в bf16: {cast}", flush=True)

        from .. import streaming                     # лёгкий модуль: torch внутри функций
        offload = str(gen.offload or "none").strip().lower()
        if offload == "group":
            # 960p+ в 16 ГБ не влезает целиком: держим на GPU только одну группу блоков,
            # остальные лежат в fp8 на CPU и подгружаются на каждом шаге.
            blocks = _sampler_int(cfg, "offload_blocks", 1)
            stream = bool(_sampler_int(cfg, "offload_stream", 1))
            print(f"[3/5] размещение: transformer group-offload (блоков в группе {blocks}, "
                  f"stream={stream}), vae -> cuda", flush=True)
            if pipe.text_encoder is not None:
                # Qwen3VL-8B, 17.5 ГБ — в 16 ГБ VRAM не влезает; стриминговый энкодер уже на месте
                streaming.move_to(pipe.text_encoder, "cpu")
            pipe.transformer.enable_group_offload(
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=blocks,
                use_stream=stream,
                non_blocking=True,
            )
        else:
            print("[3/5] размещение: transformer+vae -> cuda", flush=True)
            if pipe.text_encoder is not None:
                # Qwen3VL-8B, 17.5 ГБ — в 16 ГБ VRAM не влезает; стриминговый энкодер уже на месте
                streaming.move_to(pipe.text_encoder, "cpu")
            pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"    vram после загрузки: занято {(total - free) / 2**30:.1f}/{total / 2**30:.1f} GiB",
              flush=True)

        # Энкодер на GPU — только теперь, после размещения трансформера: стриминговые веса живут
        # в VRAM лишь на время forward модуля, но порядок всё равно держим прежним
        # (fp8 -> LoRA -> размещение), чтобы не ловить два пика одновременно.
        encoder_from_stream = False
        if defer_encoder:
            pipe.text_encoder = _load_base_encoder_gpu(cfg)
            encoder_from_stream = True
        elif pipe.text_encoder is not None and encoder_device != "cpu":
            # энкодер из реестра на GPU: этот же путь, но стриминговый (см. `encoders.load`)
            encoder_from_stream = streaming.is_streamed(pipe.text_encoder)

        # Расписание Turbo (sample_sigmas) главнее, чем num_inference_steps: число шагов = длина sigmas.
        steps = cfg.steps
        sigmas = pipe.config.get("sample_sigmas") if label == "turbo" else None
        if sigmas:
            if steps != len(sigmas):
                print(f"    steps {steps} -> {len(sigmas)} "
                      f"(расписание turbo из model_index.json)", flush=True)
            steps = len(sigmas)

        return QwenImage21Handle(
            pipe=pipe,
            transformer=pipe.transformer,
            label=label,
            repo=str(gen.repo),
            fp8=fp8,
            steps=int(steps),
            sigmas=tuple(sigmas) if sigmas else None,
            precision_cache=fp8_dir if fp8 else None,
            encoder_device=encoder_device,
            encoder_from_stream=encoder_from_stream,
        )

    # -- conditioning ------------------------------------------------------ #

    def resize_input(self, cfg: Any, handle: QwenImage21Handle, image: Any) -> Any:
        """Входной кадр, приведённый к сетке conditioning (`image_processor.resize`).

        Делается один раз на кадр: `encode` и `generate` обязаны видеть один и тот же тензор,
        иначе картинка поедет относительно run_style.py.
        """
        crop = tuple(int(v) for v in cfg.crop(*image.size))
        if handle.input_image is None or handle.crop != crop:
            handle.input_image = handle.pipe.image_processor.resize(
                image, width=crop[0], height=crop[1])
            handle.crop = crop
            handle.resolution = int(cfg.effective_resolution(*image.size))
        return handle.input_image

    def resize_inputs(self, cfg: Any, handle: QwenImage21Handle, images: list[Any]) -> list[Any]:
        """Входные кадры, приведённые к сетке conditioning (`image_processor.resize`).

        Сетка считается по **первой** картинке (как `RunConfig.crop`), остальные референсы
        приводятся к тому же WxH: `encode_prompt` и `encode_prompt`-замыкание в `generate`
        обязаны видеть одни и те же тензоры, иначе картинка поедет относительно одиночного пути.

        Одиночный путь кэшируется в `handle` (`input_image`/`crop`/`resolution`): `encode` и
        `generate` вызываются по разу на кадр, а повторный resize — лишняя работа и лишний тензор.
        При нескольких референсах список пересчитывается каждый раз (кэш держит только первый,
        его же видит старый код через `handle.input_image`).
        """
        first = images[0]
        crop = tuple(int(v) for v in cfg.crop(*first.size))
        if len(images) == 1:
            return [self.resize_input(cfg, handle, first)]
        resized = [handle.pipe.image_processor.resize(image, width=crop[0], height=crop[1])
                   for image in images]
        handle.input_image = resized[0]
        handle.crop = crop
        handle.resolution = int(cfg.effective_resolution(*first.size))
        return resized

    def encode(self, cfg: Any, handle: QwenImage21Handle, image: Any, prompt: str) -> dict[str, Any]:
        """Эмбеддинги промта и **всех** референсов (`encode_prompt`) на устройстве энкодера.

        `image` — список картинок входа (одиночная `PIL.Image` тоже принимается, `base.as_image_list`).
        Vision-токены считаются по всем: `QwenImage21Pipeline._get_qwen_prompt_embeds` строит
        плейсхолдеры `<image1>`…`<imageN>` ровно по числу переданных картинок и отдаёт processor'у
        весь набор, а `image_pad_mask` отмечает все vision-позиции — то есть conditioning обучен
        на многореференсном режиме, а не склеен вручную.

        Устройство берётся из `handle` (разрешённое `encoder.device`): по умолчанию на машине
        с CUDA это `cuda` — стриминговые веса Qwen3VL-8B идут через VRAM (~15–25 s) вместо CPU
        (255–682 s). Явный `--set encoder.device=cpu` возвращает прежний путь.

        Печать `[4/5]` (промах кэша) — здесь; попадание в кэш печатает оркестратор, потому что
        про кэш знает он, а не бекенд.
        """
        torch = _torch()
        images = as_image_list(image)
        if not images:
            raise ValueError("qwen_image21: нужен входной кадр (conditioning=img2img)")

        input_images = self.resize_inputs(cfg, handle, images)
        device = torch.device(handle.encoder_device or "cpu")
        on_gpu = device.type == "cuda"
        if on_gpu:
            print("[4/5] кодирование промта на GPU (Qwen3VL-8B, стриминг)...", flush=True)
        else:
            print("[4/5] кодирование промта на CPU (Qwen3VL-8B, ~1-3 мин)...", flush=True)
        if len(input_images) > 1:
            print(f"    референсов: {len(input_images)} (vision-токены по всем)", flush=True)
        started = time.time()
        try:
            with torch.no_grad():
                prompt_embeds, prompt_embeds_mask, image_pad_mask = handle.pipe.encode_prompt(
                    prompt=prompt,
                    image=input_images,
                    device=device,
                )
        finally:
            # VRAM нужна трансформеру: стриминговый энкодер больше не пригодится (эмбеддинги есть)
            self._drop_encoder(handle)
        print(f"    ok, {time.time() - started:.0f}s {tuple(prompt_embeds.shape)}", flush=True)
        return {"embeds": prompt_embeds, "mask": prompt_embeds_mask, "img_mask": image_pad_mask}

    @staticmethod
    def _drop_encoder(handle: QwenImage21Handle) -> None:
        """Отпустить стриминговый энкодер после энкода: `text_encoder=None` + del + gc.

        Стриминговый энкодер держит в VRAM буферы и пустые параметры (веса приезжают на время
        forward), а его `ShardedWeights` — открытые дескрипторы шардов (чтение позиционное,
        `seek + readinto`, шард целиком в память не тянется). После энкода он не нужен ни одному
        шагу сэмплера (эмбеддинги подменяются замыканием), поэтому место отдаётся трансформеру.
        """
        pipe = getattr(handle, "pipe", None)
        model = getattr(pipe, "text_encoder", None) if pipe is not None else None
        if model is None or not handle.encoder_from_stream:
            return
        handle.encoder_from_stream = False
        pipe.text_encoder = None
        weights = getattr(model, "_stream_weights", None)
        close = getattr(weights, "close", None)
        if callable(close):                          # Windows: закрытые дескрипторы не держат файл шардов
            try:
                close()
            except Exception:                        # noqa: BLE001 — уборка не должна ронять прогон
                pass
        del model, weights
        gc.collect()
        try:
            _torch().cuda.empty_cache()
        except Exception:                            # noqa: BLE001 — без CUDA чистить нечего
            pass
        print("    энкодер отпущен (VRAM отдана трансформеру)", flush=True)

    # -- сэмплирование ----------------------------------------------------- #

    def generate(self, cfg: Any, handle: QwenImage21Handle, cond: Any, image: Any,
                 seed: int) -> Any:
        """Денойзинг с готовыми эмбеддингами. Порядок важен: перенос на GPU -> подмена
        `encode_prompt` -> `pipe(...)`: `__call__` не принимает `image_pad_mask`, а с `image` +
        `prompt_embeds` падает, поэтому эмбеддинги отдаются через замыкание.

        `image` — список референсов (одиночная `PIL.Image` тоже принимается). В `pipe(...)` он
        уходит списком только при >1: `__call__` для одной картинки ведёт себя как раньше, а для
        списка считает латенты по каждому референсу (`prepare_latents(vae_images)`) и склеивает
        их перед латентами кадра.
        """
        torch = _torch()
        images = as_image_list(image)
        if images:
            input_images = self.resize_inputs(cfg, handle, images)
        elif handle.input_image is not None:
            input_images = [handle.input_image]
        else:
            raise ValueError("qwen_image21: нет входного кадра (conditioning=img2img)")

        first = images[0] if images else None
        crop = handle.crop or tuple(int(v) for v in cfg.crop(*first.size))
        crop_w, crop_h = int(crop[0]), int(crop[1])
        resolution = int(handle.resolution or cfg.effective_resolution(*first.size))

        prompt_embeds = cond["embeds"].to("cuda", dtype=torch.bfloat16)
        mask = cond.get("mask")
        prompt_embeds_mask = mask.to("cuda") if mask is not None else None
        image_pad_mask = cond["img_mask"].to("cuda")

        def cached_encode_prompt(*_args: Any, **_kwargs: Any):
            return prompt_embeds, prompt_embeds_mask, image_pad_mask

        handle.pipe.encode_prompt = cached_encode_prompt

        steps = int(handle.steps)
        refs = f", референсов {len(input_images)}" if len(input_images) > 1 else ""
        print(f"[5/5] генерация: {steps} шагов @ res {resolution}{refs}"
              f"{' (preview)' if cfg.is_preview else ''}", flush=True)
        started = time.time()
        with torch.inference_mode():
            result = handle.pipe(
                prompt=None,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                image=input_images[0] if len(input_images) == 1 else input_images,
                height=crop_h,
                width=crop_w,
                num_inference_steps=steps,
                true_cfg_scale=float(cfg.generation.guidance or 1.0),
                output_resolution=resolution,
                generator=torch.Generator("cpu").manual_seed(int(seed)),
            ).images[0]
        print(f"    ok, {time.time() - started:.0f}s", flush=True)
        return result

    # -- освобождение ------------------------------------------------------ #

    def release(self, handle: QwenImage21Handle) -> None:
        """Отпускает пайплайн (до 16 ГБ VRAM) перед апскейлом: del + gc + empty_cache."""
        torch = _torch()
        pipe, transformer = handle.pipe, handle.transformer
        handle.input_image = None
        handle.pipe = None
        handle.transformer = None
        del pipe, transformer
        gc.collect()
        torch.cuda.empty_cache()

    # -- диагностика ------------------------------------------------------- #

    def probe(self, cfg: Any) -> dict[str, Any]:
        """Компоненты и fp8-кэш из локального hf-cache, версии стека, бюджет, caps.

        Ничего не загружает и не скачивает: только чтение снапшота `hf-cache` и `stat` fp8-кэша.
        """
        gen = cfg.generation
        hybrid = _is_hybrid(cfg)
        label = "turbo" if hybrid or str(gen.repo) == TURBO_REPO else "base"
        repo = str(gen.transformer[0]) if hybrid else str(gen.repo)

        info: dict[str, Any] = {
            "backend": self.name,
            "label": label,
            "repo": str(gen.repo),
            "transformer_repo": repo,
            "precision": gen.precision,
            "offload": gen.offload,
            "caps": self.caps.to_dict(),
            "budget": self.budget.to_dict(),
        }

        snapshot = _local_snapshot(repo)
        if snapshot is None:
            info["source"] = "нет данных: репозитория нет в локальном hf-cache"
            info["components"] = {}
        else:
            info.update(snapshot)

        precision_cache = _precision_cache_dir(cfg, label)
        if precision_cache is None:
            info["precision_cache"] = {"path": None, "ready": False,
                                       "status": "выключен (generation.precision_cache='')"}
        else:
            config_file = precision_cache / "config.json"
            files = sorted(precision_cache.glob("*")) if precision_cache.is_dir() else []
            size = 0
            for item in files:
                try:
                    size += item.stat().st_size
                except OSError:
                    pass
            info["precision_cache"] = {
                "path": str(precision_cache),
                "ready": config_file.is_file(),
                "bytes": size,
                "files": [item.name for item in files],
                "status": ("готов (старт читает fp8 вместо bf16 + каста)" if config_file.is_file()
                           else "нет — первый запуск сконвертирует bf16 -> fp8 и запишет его"),
            }

        versions: dict[str, str] = {}
        for module_name, attr in (("torch", "torch"), ("diffusers", "diffusers"),
                                  ("transformers", "transformers")):
            try:
                versions[attr] = __import__(module_name).__version__
            except Exception:                        # noqa: BLE001 — диагностика не обязана работать
                continue
        info["versions"] = versions
        info["offline"] = str(os.environ.get("HF_HUB_OFFLINE", "")).strip().lower() in (
            "1", "true", "yes", "on")
        return info

    # -- удобные имена (для логов/CLI; не часть протокола) ------------------ #

    @staticmethod
    def lora_filename(cfg: Any) -> str | None:
        """Имя LoRA-файла профиля — для логов `[profile]` в CLI."""
        return cfg.generation.lora.filename if cfg.generation.lora else None

    @staticmethod
    def default_lora(repo_label: str = "base") -> tuple[str, str]:
        """(repo, filename) штатного адаптера профиля — для пресетов/диагностики."""
        filename = TURBO_LORA if repo_label == "turbo" else BASE_LORA
        return LORA_REPO, filename


__all__ = [
    "BASE_LORA", "DEFAULT_FP8_CACHE", "DEFAULT_REPO", "LORA_REPO", "QwenImage21Backend",
    "QwenImage21Handle", "TURBO_LORA", "TURBO_REPO",
]
