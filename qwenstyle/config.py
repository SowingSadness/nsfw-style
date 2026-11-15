"""Единый конфиг запуска: единственный источник параметров и ключей кэша.

Правила, которые здесь зафиксированы (см. AGENTS.md, R1/R3 и план `qwenstyle-framework`):

* `RunConfig` — frozen dataclass; любой параметр запуска живёт здесь, а не в `argparse` подкоманды.
* пресеты (`qwenstyle/presets/*.yaml`) — только данные; `extends:` даёт deep-merge, без кода.
* ключ кэша эмбеддингов строится ТОЛЬКО через `embeds_key()`/`cache_key_parts()` и обязан
  совпадать с историческим форматом `embeds_<md5>_<sha>[_te-<enc>][-<dtype>][-dev-<device>]_<W>x<H>.pt`
  — иначе все существующие `cache/embeds/embeds_*.pt` станут мёртвыми (255 с CPU-энкода за промт).
  Сегменты dtype/device есть только у неумолчаний (`bfloat16`/`cpu`), сами умолчания
  дают байт-в-байт прежнее имя.
* `seed` и `lora.scale` в ключ НЕ входят: это бесплатные оси (AGENTS.md, §5.6/§6).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

# `image_sha` живёт в `io.py` (хеш файла — это ввод/вывод, а не параметр конфига), но реэкспортируется
# отсюда: `pipeline`/сайдкары исторически берут её из `qwenstyle.config`.
from .io import NOREF_SHA, image_sha  # noqa: F401 — реэкспорт для существующих вызывающих

PRESETS_DIR = Path(__file__).resolve().parent / "presets"

# Раскладка кешей (пути относительны корня репозитория):
#   cache/embeds/  — вычисления: conditioning промта/картинок (мелкое, вытесняется `cache prune`);
#   cache/models/  — веса и квантование (fp8-трансформеры, веса энкодеров, ESRGAN).
# Кеш моделей не вытесняется никогда: пересборка = `--rebuild-fp8-cache`.
CACHE_ROOT = Path("cache")
EMBEDS_DIR = CACHE_ROOT / "embeds"
MODELS_DIR = CACHE_ROOT / "models"

# Схема ключа кэша. Менять только вместе с миграцией содержимого cache/ (AGENTS.md, R1).
CACHE_SCHEMA = "qwenstyle-1"

DEFAULT_SEED = 20261113
# Допустимые пропорции кадра (--aspect). None = как у входного изображения.
ASPECT_RATIOS: dict[str, float] = {"16:9": 16 / 9, "4:3": 4 / 3, "1:1": 1.0}
DEFAULT_RESOLUTION = 1024      # 768p при 16:9 (run_style.py: --resolution по умолчанию 1024)
PREVIEW_RESOLUTION = 640       # 480p при 16:9 (run_style.py: --preview)
PREVIEW_MAX_SIZE_PX = 480      # 480p и ниже — черновой режим, как `--preview`
PREVIEW_STEPS = 8
DEFAULT_STEPS = 25

#: sampler kwargs, которые влияют на conditioning/кадр и потому обязаны попасть в ключ кэша.
CACHE_SAMPLER_KEYS = ("caption_upsample_temperature",)

#: Политика вытеснения кэша эмбеддингов (`qwenstyle/cleanup.py`). Значения — данные, логика там.
CACHE_SUPERSEDE_POLICIES: tuple[str, ...] = ("none", "prompt", "image", "pair")
DEFAULT_CACHE_SUPERSEDE = "pair"


class ConfigError(ValueError):
    """Некорректный пресет, неизвестный путь в `--set` или несовместимое значение."""


def models_path(raw: str | Path | None) -> str | None:
    """Канонический путь кеша моделей: легаси `cache/<x>` -> `cache/models/<x>`.

    Данные пресетов миграцией шага 4 не переписываются: старые значения
    (`cache/fp8_transformer`, `cache/weights/heretic-te`, …) разрешаются в новое место
    (`cache/models/...`), поэтому существующие fp8-кэши и веса находятся без пересборки.
    Каноничные пути (`cache/models/...`, `cache/embeds/...`) и любые пути вне `cache/`
    (абсолютные, `~`, относительные от cwd) возвращаются как есть — но всегда `str`.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return text                             # '' = «не кэшировать», а не None
    parts = Path(text).expanduser().parts
    # Берём последний сегмент `cache`: работает и для относительного пути от корня репозитория,
    # и для абсолютного `<repo>/cache/<x>`.
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] != CACHE_ROOT.name:
            continue
        tail = parts[idx + 1:]
        if not tail or tail[0] in (MODELS_DIR.name, EMBEDS_DIR.name):
            return text
        # as_posix(): путь попадает в config show/логи и не должен менять '/' на '\\'
        return (Path(*parts[: idx + 1]) / MODELS_DIR.name / Path(*tail)).as_posix()
    return text


# --------------------------------------------------------------------------- #
# Дампы/загрузки датаклассов
# --------------------------------------------------------------------------- #

def _build_dataclass(cls: type, data: Mapping[str, Any], where: str) -> Any:
    """Собирает датакласс из mapping, рекурсивно и со строгой проверкой ключей."""
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(
            f"{where}: неизвестные параметры {unknown}; "
            f"допустимые: {sorted(known)}"
        )
    nested = _NESTED.get(cls.__name__, {})
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        sub = nested.get(name)
        if sub is not None and isinstance(value, Mapping):
            kwargs[name] = _build_dataclass(sub, value, f"{where}.{name}" if where else name)
        elif sub is not None and value is None:
            kwargs[name] = None
        else:
            kwargs[name] = value
    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise ConfigError(f"{where or cls.__name__}: {exc}") from exc


_NESTED: dict[str, dict[str, type]] = {}


def _register_nested(cls: type, **names: type) -> None:
    _NESTED[cls.__name__] = names


def _plain(value: Any) -> Any:
    """Датакласс -> dict, tuple -> tuple, всё остальное как есть (для merged/set)."""
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_plain(v) for v in value)
    return value


def _deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Пресеты: вложенные словари сливаются, скаляры/списки перезаписываются."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(value: Any) -> Any:
    """Значение `--set`: строка -> bool/int/float/None, остальное без изменений."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    low = text.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("none", "null", "~"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
    """`target['generation']['steps'] = value` по пути 'generation.steps'."""
    keys = [k for k in path.split(".") if k]
    if not keys:
        raise ConfigError("--set: пустой путь")
    node = target
    for key in keys[:-1]:
        child = node.get(key)
        if child is None:
            child = node[key] = {}
        elif not isinstance(child, dict):
            raise ConfigError(f"--set {path}: '{key}' — не секция, туда нельзя вложить {keys[-1]}")
        node = child
    node[keys[-1]] = value


# --------------------------------------------------------------------------- #
# Конфиги
# --------------------------------------------------------------------------- #

#: Канонические умолчания энкодера: для них в имени файла кэша НЕ появляется сегмент
#: (иначе все существующие `cache/embeds/embeds_*.pt` стали бы мёртвыми — AGENTS.md R1).
DEFAULT_ENCODER_DTYPE = "bfloat16"
#: Канон «cpu» — именно он не добавляет `-dev-…` в имя записи (легаси-записи остаются валидными).
DEFAULT_ENCODER_DEVICE = "cpu"
#: `encoder.device=auto` (умолчание): разрешается в cuda, если она доступна, иначе в cpu.
AUTO_ENCODER_DEVICE = "auto"
#: Глубина конвейера чтения весов при авто-GPU (см. `qwenstyle/streaming.py`).
AUTO_ENCODER_PREFETCH = 4
#: Кэш разрешённого `auto`: torch импортируется только при первом разрешении (и только здесь).
_RESOLVED_AUTO_DEVICE: str | None = None

_DTYPE_ALIASES: dict[str, str] = {
    "bf16": "bfloat16",
    "fp16": "float16",
    "half": "float16",
    "fp32": "float32",
    "float": "float32",
}
#: Короткий токен для имени файла (в имени не бывает точек/двоеточий, `EMBEDS_RE` — [a-z0-9]).
_DTYPE_TOKENS: dict[str, str] = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "float64": "fp64",
}


def encoder_dtype_canon(dtype: str | None) -> str:
    """Канон `encoder.dtype`: `bf16` -> `bfloat16`, `fp16` -> `float16`, пусто -> умолчание.

    Один и тот же канон используется и в ключе кэша, и в сверке сайдкара, поэтому
    `--set encoder.dtype=bf16` и `--set encoder.dtype=bfloat16` не дают промаха.
    """
    value = str(dtype or "").strip().lower() or DEFAULT_ENCODER_DTYPE
    return _DTYPE_ALIASES.get(value, value)


def auto_encoder_device() -> str:
    """Разрешение `encoder.device=auto`: `cuda`, если CUDA доступна, иначе `cpu`.

    `torch` импортируется **только здесь** и результат кэшируется на процесс: модуль обязан
    импортироваться без torch (`qwenstyle models list` не должен тянуть 2 ГБ библиотек), а
    многократные вызовы канона (ключ кэша, сайдкар, `cache_key_parts`) не платят за импорт дважды.
    """
    global _RESOLVED_AUTO_DEVICE
    if _RESOLVED_AUTO_DEVICE is None:
        try:
            import torch                            # лениво: см. докстринг

            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:                           # noqa: BLE001 — без torch считаем, что CUDA нет
            resolved = "cpu"
        _RESOLVED_AUTO_DEVICE = resolved
    return _RESOLVED_AUTO_DEVICE


def encoder_device_canon(device: str | None) -> str:
    """Канон `encoder.device`: `cuda:0` -> `cuda` (та же карта), `auto` -> фактическое, пусто -> `cpu`.

    `auto` разрешается в фактическое устройство (см. `auto_encoder_device`), поэтому имя записи
    кэша и сверка сайдкара говорят правду: на машине с CUDA `auto` даёт `-dev-cuda`.
    """
    value = str(device or "").strip().lower() or DEFAULT_ENCODER_DEVICE
    if value == AUTO_ENCODER_DEVICE:
        return auto_encoder_device()
    return "cuda" if value.startswith("cuda") else value


def _dtype_token(canon: str) -> str:
    """Токен dtype в имени файла: `float16` -> `fp16`, `int8` -> `int8`."""
    token = _DTYPE_TOKENS.get(canon, canon)
    return "".join(ch for ch in token if ch.isalnum()) or "unknown"


def _device_token(canon: str) -> str:
    """Токен device в имени файла: `cuda` -> `cuda`, `xpu:0` -> `xpu0`."""
    return "".join(ch for ch in canon if ch.isalnum()) or "unknown"


def _config_int(path: str, value: Any) -> int:
    """Целое из конфига; строка допустима (`--set` через YAML), мусор — понятная `ConfigError`."""
    if isinstance(value, bool):                      # bool — подкласс int, но это ошибка ввода
        raise ConfigError(f"{path} должен быть целым числом, получено {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{path} должен быть целым числом, получено {value!r}") from None


@dataclass(frozen=True)
class EncoderConfig:
    """Текст-энкодер. `cache_tag` — часть ключа кэша эмбеддингов.

    `device` — где считать энкод: `"auto"` (умолчание) разрешается в `cuda`, если она доступна,
    иначе в `cpu`; явные `--set encoder.device=cpu|cuda` продолжают работать как раньше. При
    `auto`+cuda `resolved()` дополнительно включает GPU-стриминг (`stream=True`, `prefetch=4`):
    веса 17.5 ГБ не влезают в 16 ГБ VRAM целиком, но влезает один модуль (`qwenstyle/streaming.py`).
    CPU-эскапей — явный `--set encoder.device=cpu` (именно он даёт легаси-имя записи кэша).

    `stream` — читать веса по слоям с диска (модель больше RAM), см. `qwenstyle/streaming.py`.
    На ключ кэша НЕ влияет — это раскладка памяти, а не математика.

    `prefetch`/`pin_memory`/`lookahead` — параметры конвейера стриминга и действуют **только**
    при `stream=True` + `device=cuda` (`qwenstyle/streaming.py::build_streamed`):

    * `prefetch` — глубина очереди продюсер-потока (сколько слоёв читается с диска заранее);
      `0` = синхронный путь, ровно прежнее поведение для CPU и GPU (`resolved()` ставит `4`
      для умолчания `device=auto` на машине с CUDA);
    * `pin_memory` — держать staging-буферы в pinned-памяти (быстрый H2D);
    * `lookahead` — сколько копий H2D выдавать заранее (`torch.cuda.Stream` + `Event`).

    На ключ кэша эмбеддингов эти три поля НЕ влияют: это раскладка памяти, а не математика
    (результат бит-в-бит тот же), поэтому `embeds_*` не пересоздаются. На `device=cpu` конвейер
    не включается и печатается одна строка «игнорируется».
    """

    name: str = "base"
    repo: str = "Qwen/Qwen-Image-2.1"
    revision: str | None = None
    local_dir: str | None = None
    # "base" -> в имени файла нет тега; "heretic" -> "_te-heretic" (D7).
    cache_tag: str = "base"
    requires_vision: bool = True
    dtype: str = "bfloat16"
    # "auto" -> cuda на машине с CUDA (см. `resolved()`/`auto_encoder_device`), иначе cpu
    device: str = AUTO_ENCODER_DEVICE
    stream: bool = False
    # конвейер стриминга GPU (действует только при stream=True + device=cuda, ключ кэша не трогает)
    prefetch: int = 0
    pin_memory: bool = True
    lookahead: int = 1

    def __post_init__(self) -> None:
        if not self.cache_tag:                       # cache_tag не задан -> берём имя
            object.__setattr__(self, "cache_tag", self.name or "base")
        # prefetch/lookahead: целые с проверкой диапазона; нормализуем, чтобы `to_mapping()`
        # и логи показывали int даже если значение пришло строкой из пресета.
        prefetch = _config_int("encoder.prefetch", self.prefetch)
        if prefetch < 0:
            raise ConfigError(
                f"encoder.prefetch должен быть >= 0 (0 = синхронный путь без конвейера), "
                f"получено {self.prefetch!r}"
            )
        lookahead = _config_int("encoder.lookahead", self.lookahead)
        if lookahead < 1:
            raise ConfigError(
                f"encoder.lookahead должен быть >= 1 (сколько копий H2D выдавать заранее), "
                f"получено {self.lookahead!r}"
            )
        object.__setattr__(self, "prefetch", prefetch)
        object.__setattr__(self, "lookahead", lookahead)
        object.__setattr__(self, "pin_memory", bool(self.pin_memory))

    @property
    def is_auto_device(self) -> bool:
        """`device` не задан явно (умолчание `auto`) — разрешается при первом обращении."""
        return str(self.device or "").strip().lower() == AUTO_ENCODER_DEVICE

    def resolved(self) -> "EncoderConfig":
        """Копия с фактическим `device` (`auto` -> `cuda`/`cpu`) и GPU-стримингом для `auto`+cuda.

        Зачем копия, а не разрешение в `__post_init__`: `EncoderConfig` строится и на лёгких путях
        (`--dry-run`, `config show`), а `torch` в них не нужен, пока никто не спросил устройство.
        Разрешение поэтому ленивое, с кэшем на процесс (`auto_encoder_device`).

        При `auto`+cuda включается `stream=True` и `prefetch=4` (веса 17.5 ГБ в 16 ГБ VRAM не
        влезают, но влезает один модуль). Явные `device=cpu|cuda` не трогают `stream`/`prefetch`:
        это прежнее поведение флагов.
        """
        if not self.is_auto_device:
            return self
        from dataclasses import replace as _replace

        device = auto_encoder_device()
        changes: dict[str, Any] = {"device": device}
        if device == "cuda":
            if not self.stream:
                changes["stream"] = True
            if not int(self.prefetch):
                changes["prefetch"] = AUTO_ENCODER_PREFETCH
        return _replace(self, **changes)

    @property
    def key_suffix(self) -> str:
        """Хвост ключа кэша после `<md5>_<sha>`: тег энкодера и неумолчания dtype/device.

        Умолчания (`cpu`/`bfloat16`) сегментов НЕ добавляют, поэтому имя записи остаётся
        байт-в-байт историческим: `embeds_<md5>_<sha>[_te-<tag>]_<W>x<H>.pt`.
        Примеры: base/cpu/bf16 -> ''; heretic/cpu/bf16 -> '_te-heretic';
        heretic/cuda/bf16 -> '_te-heretic-dev-cuda'; base/cuda/fp16 -> '-fp16-dev-cuda'.
        """
        suffix = "" if self.cache_tag in ("", "base") else f"_te-{self.cache_tag}"
        dtype = encoder_dtype_canon(self.dtype)
        if dtype != DEFAULT_ENCODER_DTYPE:
            suffix += f"-{_dtype_token(dtype)}"
        device = encoder_device_canon(self.device)
        if device != DEFAULT_ENCODER_DEVICE:
            suffix += f"-dev-{_device_token(device)}"
        return suffix


@dataclass(frozen=True)
class LoraConfig:
    repo: str
    filename: str
    scale: float = 0.7
    # Turbo-адаптер несёт 224 per-module alpha (=128): их вырезают и грузят state dict.
    strip_alpha: bool = False


@dataclass(frozen=True)
class GenerationConfig:
    """Модель генерации. `repo` — репозиторий пайплайна, `transformer` — его подмена (turbo)."""

    backend: str = "qwen_image21"
    repo: str = "Qwen/Qwen-Image-2.1"
    revision: str | None = None
    transformer: tuple[str, str] | None = None      # (repo, subfolder) или None = из repo
    lora: LoraConfig | None = None
    steps: int | None = None
    guidance: float = 1.0
    precision: str = "fp8"
    precision_cache: str | None = None
    offload: str = "none"
    sampler_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpscaleConfig:
    backend: str = "esrgan"
    factor: float = 0.0
    target_height: int | None = None                # если задан — factor считается из высоты
    weights: str | None = None
    tile: int = 256
    pad: int = 32
    color_fix: str = "wavelet"                      # как --upscale-color-fix в run_style.py
    color_strength: float = 0.75
    color_sigma: float = 12.0


@dataclass(frozen=True)
class BatchConfig:
    seeds: str = "0-9"                              # "0-9" | "1,7,42" (как tools/batch_seeds.py)
    out_dir: str = "out/batch"                      # <out-dir>/s{N}-gen.png и <out-dir>/s{N}.png
    keep_gen: bool = True
    resume: bool = True


@dataclass(frozen=True)
class RunConfig:
    """Полное описание прогона: всё, что влияет на кадр и на ключи кэша."""

    encoder: EncoderConfig
    generation: GenerationConfig
    upscale: UpscaleConfig = field(default_factory=UpscaleConfig)
    batch: BatchConfig | None = None
    input: str = ""
    out: str = "out/out.png"
    prompt: str = ""
    prompt_file: str | None = None
    seed: int = DEFAULT_SEED
    size_px: int | None = None                      # высота кадра в пикселях (--size 768p -> 768)
    resolution: int | None = None                   # sqrt(площадь crop), если задан явно
    preview: bool = False
    aspect: str | None = None                       # "16:9" | "4:3" | "1:1"; None = пропорция входа
    preset: str | None = None                       # откуда взято (для логов/отчётов)
    # Политика вытеснения устаревших эмбеддингов: none|prompt|image|pair (`qwenstyle/cleanup.py`).
    # На ключ кэша НЕ влияет: это обслуживание диска, а не параметр conditioning.
    cache_supersede: str = DEFAULT_CACHE_SUPERSEDE
    # Отладочный лог прогона в подпапке кадра (`qwenstyle/runlog.py`). `--no-logs` -> False.
    # На кадр и на ключ кэша НЕ влияет: это наблюдение, а не параметр генерации.
    logs: bool = True

    # -- построение -------------------------------------------------------- #

    @classmethod
    def from_profile(cls, name: str, **overrides: Any) -> "RunConfig":
        """`preset` (имя в qwenstyle/presets или путь к YAML) + точечные overrides."""
        cfg = _build_run_config(load_profile(name))
        return cfg.merged(**overrides) if overrides else cfg

    def merged(self, **overrides: Any) -> "RunConfig":
        """Копия с точечными правками по путям: `merged(**{"generation.steps": 8})`.

        Ключи — плоские пути ('generation.steps', 'encoder.name'), значения coerced из строк,
        поэтому `--set a.b=c` и код используют один и тот же механизм.
        """
        if not overrides:
            return self
        data = self.to_mapping()
        touched: set[str] = set()
        for path, value in overrides.items():
            _set_path(data, path, _coerce(value))
            touched.add(path)
        return _build_run_config(data, touched=touched)

    def to_mapping(self) -> dict[str, Any]:
        """Полный dict параметров (для JSON-вывода, логов и merged())."""
        return _plain(self)

    # -- производные значения ---------------------------------------------- #

    @property
    def encoder_pin_memory(self) -> bool:
        """Pinned staging энкодера с учётом уже занятой pinned-памяти (`offload=group`).

        На Windows pinned-память CUDA — это блокировка страниц **и** маппинг в адресное
        пространство GPU (WDDM), то есть те же ресурсы, что тратит VRAM. Групповой offload
        держит в ней 6.6–8.5 ГиБ весов fp8-трансформера, после чего первый же крупный
        staging-буфер энкодера (`embed_tokens` 1.16 ГиБ) падал `CUDA error: out of memory`
        при совершенно свободной VRAM, а следом умирал и CUDA-контекст
        (`shared object initialization failed`) — прогон терялся целиком.

        Выигрыш pin на этом профиле — 1.3× на H2D, то есть 0.05–0.31 s из 8–10 s энкода
        (AGENTS.md §4), поэтому при `offload=group` пиннинг отключается **принудительно**:
        поле `encoder.pin_memory` и `--set` его не включают обратно. Цена ошибки — упавший
        прогон, цена отказа — доли секунды.
        """
        offload = str(getattr(self.generation, "offload", "") or "").strip().lower()
        if offload == "group":
            return False
        return bool(getattr(self.encoder, "pin_memory", True))

    @property
    def is_preview(self) -> bool:
        """Черновой режим: явный `preview` или высота кадра <= 480p (как в run_style.py)."""
        if self.preview:
            return True
        return self.size_px is not None and self.size_px <= PREVIEW_MAX_SIZE_PX

    @property
    def steps(self) -> int:
        """Разрешённое число шагов: явное значение, иначе 8 в preview / 25 в финале."""
        if self.generation.steps is not None:
            return int(self.generation.steps)
        return PREVIEW_STEPS if self.is_preview else DEFAULT_STEPS

    def frame_ratio(self, in_w: int, in_h: int) -> float:
        """Пропорция кадра: из `--aspect`, иначе из входного изображения.

        Прогон без референса (`in_w`/`in_h` = 0, T2I) не падает и не даёт нулевую пропорцию:
        берётся `--aspect`, а если он не задан — квадрат 1:1. Ошибку «без --input нужен --aspect»
        решает не здесь, а `pipeline.input_problem()`: конфиг обязан уметь посчитать сетку для
        любого из трёх состояний, чтобы `--dry-run` не падал трейсбеком.
        """
        if self.aspect:
            key = str(self.aspect).strip()
            if key not in ASPECT_RATIOS:
                raise ConfigError(
                    f"aspect={self.aspect!r}: допустимо {', '.join(ASPECT_RATIOS)} или не задавать"
                )
            return ASPECT_RATIOS[key]
        if int(in_w) <= 0 or int(in_h) <= 0:
            return 1.0                            # нет картинки и нет --aspect: кадр квадратный
        return in_w / max(1, in_h)

    def effective_resolution(self, in_w: int, in_h: int) -> int:
        """`--resolution`: явный, иначе из `--size` (по пропорции кадра), иначе профиль."""
        if self.resolution:
            return int(self.resolution)
        if self.size_px:
            # area = W*H, W = H*ratio -> resolution = sqrt(area) = H * sqrt(ratio)
            ratio = self.frame_ratio(in_w, in_h)
            return max(256, int(round(self.size_px * ratio ** 0.5)))
        return PREVIEW_RESOLUTION if self.preview else DEFAULT_RESOLUTION

    def crop(self, in_w: int, in_h: int) -> tuple[int, int]:
        """Размер conditioning-кадра: сетка WxH из diffusers `calculate_dimensions`."""
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

        res = self.effective_resolution(in_w, in_h)
        width, height, _ = calculate_dimensions(res ** 2, self.frame_ratio(in_w, in_h))
        return int(width), int(height)

    def upscale_factor(self, gen_h: int) -> float:
        """Множитель апскейла: из `target_height/gen_h`, иначе явный `factor`."""
        if self.upscale.target_height:
            return self.upscale.target_height / max(1, int(gen_h))
        return float(self.upscale.factor or 0.0)

    # -- ключи кэша -------------------------------------------------------- #

    def cache_key_parts(self, crop: tuple[int, int], prompt_md5: str, img_sha: str) -> dict[str, str]:
        """Всё, что влияет на эмбеддинги; НЕ включает seed и lora.scale (бесплатные оси)."""
        width, height = int(crop[0]), int(crop[1])
        gen = self.generation
        lora = gen.lora
        parts: dict[str, str] = {
            "schema": CACHE_SCHEMA,
            "prompt_md5": str(prompt_md5),
            "image_sha": str(img_sha),
            "crop": f"{width}x{height}",
            "encoder": self.encoder.cache_tag,
            "encoder_repo": self.encoder.repo,
            "encoder_revision": self.encoder.revision or "",
            # dtype/device меняют округления энкода (GPU != CPU), значит влияют и на запись:
            # входят в ключ и сверяются по сайдкару при попадании (шаг 2 плана te-gpu-streaming).
            "encoder_dtype": encoder_dtype_canon(self.encoder.dtype),
            "encoder_device": encoder_device_canon(self.encoder.device),
            "backend": gen.backend,
            "repo": gen.repo,
            "revision": gen.revision or "",
            "transformer": f"{gen.transformer[0]}:{gen.transformer[1]}" if gen.transformer else "",
            "lora": lora.filename if lora else "",
            "lora_repo": lora.repo if lora else "",
            "precision": gen.precision,
            "steps": str(self.steps),
            "guidance": f"{gen.guidance:g}",
        }
        for key in CACHE_SAMPLER_KEYS:
            if key in gen.sampler_kwargs:
                parts[f"sampler.{key}"] = str(gen.sampler_kwargs[key])
        return parts

    def embeds_key(self, crop: tuple[int, int], prompt_md5: str, img_sha: str) -> str:
        """Имя файла кэша эмбеддингов (`tools/cli.EMBEDS_RE`).

        Для умолчаний (base/cpu/bfloat16) строка байт-в-байт историческая:
        `embeds_<md5>_<sha>_<W>x<H>.pt` (AGENTS.md R1: формат ломать нельзя).
        """
        width, height = int(crop[0]), int(crop[1])
        return f"embeds_{prompt_md5}_{img_sha}{self.encoder.key_suffix}_{width}x{height}.pt"

    def embeds_path(self, crop: tuple[int, int], prompt_md5: str, img_sha: str,
                    cache_dir: str | Path = EMBEDS_DIR) -> Path:
        return Path(cache_dir) / self.embeds_key(crop, prompt_md5, img_sha)


def _build_run_config(data: Mapping[str, Any], touched: set[str] | None = None) -> RunConfig:
    """Сборка RunConfig + пересчёт производных полей (например `cache_tag` из `encoder.name`)."""
    touched = touched or set()
    data = dict(data)
    encoder_raw = dict(data.get("encoder") or {})
    # encoder.name изменён, а cache_tag — нет: тег обязан следовать имени (иначе кэш энкодеров
    # смешается: D7/heretic пишет conditioning другого энкодера).
    if "encoder.cache_tag" not in touched and (
        "encoder.name" in touched or not encoder_raw.get("cache_tag")
    ):
        encoder_raw["cache_tag"] = encoder_raw.get("name") or "base"
    data["encoder"] = encoder_raw
    generation_raw = dict(data.get("generation") or {})
    transformer = generation_raw.get("transformer")
    if isinstance(transformer, list) and len(transformer) == 2:      # YAML: [repo, subfolder]
        generation_raw["transformer"] = (str(transformer[0]), str(transformer[1]))
    elif isinstance(transformer, str):                               # --set generation.transformer=repo:sub
        sep = "," if "," in transformer else ":"
        repo, _, sub = transformer.partition(sep)
        if not repo.strip() or not sub.strip():
            raise ConfigError(
                "generation.transformer ждёт 'repo:subfolder' (или список [repo, subfolder]), "
                f"получено {transformer!r}"
            )
        generation_raw["transformer"] = (repo.strip(), sub.strip())
    upscale_raw = dict(data.get("upscale") or {})
    # Пути кеша моделей приводятся к новой раскладке (`cache/models/...`) в одном месте:
    # пресеты и CLI могут по-прежнему писать `cache/fp8_transformer`/`cache/weights/...`.
    for container, key in ((generation_raw, "precision_cache"),
                           (encoder_raw, "local_dir"),
                           (upscale_raw, "weights")):
        if key in container:
            container[key] = models_path(container[key])
    data["generation"] = generation_raw
    data["encoder"] = encoder_raw
    data["upscale"] = upscale_raw
    cfg = _build_dataclass(RunConfig, data, "config")
    # Политика вытеснения: единственное место, где `--set cache_supersede=…` проверяется.
    # `--set cache_supersede=none` приходит как None (`_coerce`): это явная политика `none`,
    # а не «не задано» — умолчание поля всегда `pair`.
    raw_policy = cfg.cache_supersede
    if raw_policy is None:
        raw_policy = "none"
    policy = str(raw_policy).strip().lower() or DEFAULT_CACHE_SUPERSEDE
    if policy not in CACHE_SUPERSEDE_POLICIES:
        raise ConfigError(
            f"cache_supersede={cfg.cache_supersede!r}: допустимо "
            f"{', '.join(CACHE_SUPERSEDE_POLICIES)} (умолчание {DEFAULT_CACHE_SUPERSEDE})"
        )
    if policy != cfg.cache_supersede:
        from dataclasses import replace as _replace

        cfg = _replace(cfg, cache_supersede=policy)
    return cfg


_register_nested(RunConfig, encoder=EncoderConfig, generation=GenerationConfig,
                 upscale=UpscaleConfig, batch=BatchConfig)
_register_nested(GenerationConfig, lora=LoraConfig)


# --------------------------------------------------------------------------- #
# Пресеты
# --------------------------------------------------------------------------- #

def available_presets() -> list[str]:
    """Имена пресетов из `qwenstyle/presets/*.yaml`."""
    return sorted(p.stem for p in PRESETS_DIR.glob("*.yaml"))


def preset_path(name_or_path: str) -> Path:
    """'base' -> qwenstyle/presets/base.yaml; путь к файлу тоже принимается."""
    raw = str(name_or_path)
    candidate = Path(raw)
    if candidate.suffix.lower() in (".yaml", ".yml") and candidate.is_file():
        return candidate
    candidate = PRESETS_DIR / f"{raw}.yaml"
    if candidate.is_file():
        return candidate
    raise ConfigError(
        f"пресет {raw!r} не найден в {PRESETS_DIR}; доступны: {', '.join(available_presets()) or '—'}"
    )


def load_profile(name_or_path: str) -> dict[str, Any]:
    """Читает YAML-пресет (+ `extends:`) и возвращает словарь параметров RunConfig."""
    import yaml  # локальный импорт: конфиг должен собираться и без yaml, если пресет не нужен

    path = preset_path(name_or_path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: некорректный YAML ({exc})") from exc
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path}: ожидался словарь параметров, получено {type(raw).__name__}")
    data = dict(raw)
    parent = data.pop("extends", None)
    if parent:
        data = _deep_merge(load_profile(str(parent)), data)
    data.pop("preset", None)            # preset всегда = имя текущего файла, не родителя
    data["preset"] = path.stem
    return data


def prompt_md5(prompt: str) -> str:
    """Первые 12 hex md5 промта — как в run_style.py."""
    return hashlib.md5(prompt.encode()).hexdigest()[:12]


def load_prompt(cfg: RunConfig) -> str:
    """Промт: из файла (promts/*.txt), иначе встроенный в конфиг."""
    if cfg.prompt_file:
        return Path(cfg.prompt_file).read_text(encoding="utf-8").strip()
    return cfg.prompt


__all__ = [
    "BatchConfig", "CACHE_SCHEMA", "CACHE_SUPERSEDE_POLICIES", "ConfigError",
    "AUTO_ENCODER_DEVICE", "AUTO_ENCODER_PREFETCH", "auto_encoder_device",
    "DEFAULT_CACHE_SUPERSEDE", "EncoderConfig", "GenerationConfig",
    "DEFAULT_ENCODER_DEVICE", "DEFAULT_ENCODER_DTYPE", "encoder_device_canon",
    "encoder_dtype_canon", "EMBEDS_DIR", "MODELS_DIR", "CACHE_ROOT", "models_path",
    "LoraConfig", "NOREF_SHA", "RunConfig", "UpscaleConfig", "available_presets", "image_sha",
    "load_profile", "load_prompt", "preset_path", "prompt_md5",
]
