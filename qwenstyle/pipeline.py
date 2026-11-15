"""Оркестратор прогона: единственное место, где живёт последовательность шагов.

    вход -> ключ кэша -> prepare -> encode/кэш -> generate -> release -> upscale -> save

Ничего специфичного для Qwen здесь нет: модель выбирается по `cfg.generation.backend` из
реестра (`qwenstyle/registry.py`), текст-энкодер — по `cfg.encoder.name`, апскейл — по
`cfg.upscale.backend`. Поэтому новая модель = новый модуль `qwenstyle/backends/<name>.py`,
а не правка этого файла.

Порядок и печати повторяют `run_style.py` 1:1 (`[1/5]`…`[5/5]` печатают бекенды; `[size]`,
`[in]`, `[4/5] embeddings из кэша`, `[post]`, `[done]` — здесь), плюс `run_batch` из
`tools/batch_seeds.py` (один `prepare`, N seed'ов, апскейл после `release`).

Кэш эмбеддингов (AGENTS.md, R1) — ответственность оркестратора, а не бекенда: бекенд только
считает payload (`encode`), а ключ/`Blob`/сайдкар/атомарную запись делает этот модуль. Иначе
строки «из кэша» и «записан» дублировались бы у каждой модели (шаг 4 плана это зафиксировал).

Тяжёлые импорты (`torch`, `diffusers`, `PIL`) — только внутри функций: `import
qwenstyle.pipeline` обязан быть мгновенным и работать без CUDA.
"""
from __future__ import annotations

import gc
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .cache import Blob, BlobCache
from .cleanup import CacheWrite, DeferredCleanup, normalize_policy, plan_names
from .config import (
    CACHE_SCHEMA,
    DEFAULT_ENCODER_DEVICE,
    DEFAULT_ENCODER_DTYPE,
    BatchConfig,
    ConfigError,
    RunConfig,
    encoder_device_canon,
    encoder_dtype_canon,
    load_prompt,
    prompt_md5,
)
from .io import batch_paths, images_sha, log_dir, out_path, save_atomic
from .registry import RegistryError, get_encoder, get_generation, get_upscale
from .runlog import batch_log_dir

if TYPE_CHECKING:                       # только типы: в рантайме runlog импортируется ниже
    from .runlog import RunLog

GIB = 1024 ** 3
#: Каталог кэша по умолчанию (как `qwenstyle.cache.CACHE_DIR`), но считается от корня репозитория.
DEFAULT_CACHE_DIR = "cache"
#: Ключи `.pt`-кэша, из которых печатается форма эмбеддингов (у Qwen `embeds`, у FLUX.2 — свой).
_SHAPE_KEYS = ("embeds", "prompt_embeds", "mask", "img_mask")


# --------------------------------------------------------------------------- #
# Результат и план
# --------------------------------------------------------------------------- #

@dataclass
class Result:
    """Один готовый кадр: генерация, финал и тайминги фаз."""

    image: Any                                   # PIL.Image финального кадра
    gen_size: tuple[int, int]                    # размер чистой генерации (до апскейла)
    out_size: tuple[int, int]                    # размер сохранённого кадра
    seed: int
    timings: dict[str, float] = field(default_factory=dict)
    out_path: Path | None = None                 # куда сохранён (None = ещё не сохранялся)


@dataclass(frozen=True)
class Plan:
    """Всё, что можно посчитать без моделей: вход, сетка, ключ кэша, апскейл и имя файла."""

@dataclass(frozen=True)
class Plan:
    """Всё, что можно посчитать без моделей: вход, сетка, ключ кэша, апскейл и имя файла."""

    #: Все референсы входа (PIL.Image RGBA, порядок значим); пустой список = T2I (`--aspect`).
    images: list[Any]
    in_size: tuple[int, int]
    crop: tuple[int, int]                        # conditioning-сетка (WxH)
    resolution: int                              # sqrt(площадь crop), как `--resolution`
    prompt: str
    prompt_md5: str
    img_sha: str
    blob: Blob                                   # запись кэша эмбеддингов/текстовых эмбеддингов
    cache_path: Path
    cached: Any                                  # payload из кэша или None
    cache_meta: dict[str, Any] | None
    need_encoder: bool                           # промах кэша на бекенде, которому нужен энкодер
    factor: float                                # множитель апскейла (0 = выключен/не нужен)
    out_path: str

    @property
    def image(self) -> Any | None:
        """Первый референс (прежнее поле `Plan.image`): `None` при T2I.

        Оставлено свойством, чтобы старые вызовы (`plan.image`) работали без правок: сетка
        conditioning и пропорция кадра всегда считаются по **первой** картинке (`RunConfig.crop`).
        """
        return self.images[0] if self.images else None


# --------------------------------------------------------------------------- #
# Помощники: кэш, формы, VRAM
# --------------------------------------------------------------------------- #

def blob_cache(root: str | Path | None = None) -> BlobCache:
    """`BlobCache` от корня репозитория: относительный путь не зависит от текущего каталога."""
    from .backends.base import REPO_ROOT

    raw = Path(root) if root is not None else Path(DEFAULT_CACHE_DIR)
    return BlobCache(raw if raw.is_absolute() else REPO_ROOT / raw)


# --------------------------------------------------------------------------- #
# Отложенная очистка кэша (`qwenstyle/cleanup.py`)
# --------------------------------------------------------------------------- #

def _cleanup_for(cfg: RunConfig) -> DeferredCleanup | None:
    """Отложенная задача вытеснения или `None`, если политика выключена (`none`).

    Создание дешёвое (ничего не читает и не удаляет): файлы трогает только `flush()` в конце
    прогона. `none` = полный no-op, поэтому задача даже не создаётся (поведение до шага 3).
    """
    if normalize_policy(cfg.cache_supersede) == "none":
        return None
    return DeferredCleanup(blob_cache().root)


def _cache_write(cfg: RunConfig, plan: Plan) -> CacheWrite | None:
    """Что этот прогон записал в кэш — в терминах политики вытеснения.

    Вытеснение живёт только для этапа `embeds` (Qwen-Image): у FLUX.2 записи лежат в других
    этапах (`text_embeds`/`ref_latents`) с другим ключом, и их уборка остаётся за `cache prune`
    (AGENTS.md, D2).
    """
    if str(plan.blob.stage) != "embeds":
        return None
    return CacheWrite(
        md5=plan.prompt_md5,
        sha=plan.img_sha,
        crop=f"{int(plan.crop[0])}x{int(plan.crop[1])}",
        enc=str(cfg.encoder.cache_tag),
        dtype=encoder_dtype_canon(cfg.encoder.dtype),
        device=encoder_device_canon(cfg.encoder.device),
        path=plan.cache_path,
    )


def _flush_cleanup(cfg: RunConfig, cleanup: DeferredCleanup | None, *,
                  apply: bool = True) -> tuple[int, int]:
    """Исполнить политику вытеснения в самом конце прогона (кадр уже сохранён).

    Задач нет или политика `none` — полный no-op. Ошибка удаления не отменяет готовый кадр:
    печатаем причину и продолжаем (файл на диске ценнее уборки кэша), но молча не глотаем.
    """
    if cleanup is None or not len(cleanup):
        return (0, 0)
    try:
        return cleanup.flush(cfg.cache_supersede, apply=apply)
    except Exception as exc:                        # noqa: BLE001 — см. докстринг
        print(f"[cache] вытеснение ({cfg.cache_supersede}) не выполнено: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return (0, 0)


def _cache_blob(backend_cls: type, cfg: RunConfig, crop: tuple[int, int],
                prompt_md5_: str, img_sha: str) -> Blob:
    """Запись кэша conditioning.

    Бекенд, который знает свой этап/ключ (FLUX.2: `text_embeds`), объявляет `cache_blob()` —
    спрашиваем его, чтобы знание ключа не дублировалось в оркестраторе. Остальные (Qwen-Image-2.1)
    получают исторический `embeds_<md5>_<sha>[_te-…]_<W>x<H>.pt` 1:1 с `run_style.py`.
    """
    factory = getattr(backend_cls, "cache_blob", None)
    if callable(factory):
        return factory(cfg, crop, prompt_md5_, img_sha)

    key = str(cfg.embeds_key(crop, prompt_md5_, img_sha))
    if key.startswith("embeds_"):
        key = key[len("embeds_"):]
    if key.endswith(".pt"):
        key = key[:-3]
    stage = str(getattr(backend_cls, "cache_stage", "") or "embeds")
    return Blob(stage=stage, key=key)


#: Поля сайдкара, которые ОБЯЗАНЫ совпасть с ожидаемыми при попадании в кэш (AGENTS.md, R1).
#: Всё остальное (`steps`, `seed`, `lora*`, `precision`, `guidance`, `sampler.*`, `backend`,
#: `repo`, `transformer`) — бесплатные оси: их смена не должна вызывать CPU-энкод.
EMBEDS_HIT_FIELDS: tuple[str, ...] = (
    "schema", "prompt_md5", "image_sha", "crop", "encoder",
    "encoder_repo", "encoder_revision", "encoder_dtype", "encoder_device",
    # Бекенд-зависимое: FLUX.2 кодирует с усечением по `max_sequence_length`, и 512 vs 1024 —
    # это разная форма тензора и другой кадр (R1). У Qwen-Image-2.1 поля нет -> "".
    "max_sequence_length",
)
#: Отсутствующее в сайдкаре поле трактуется как это умолчание (старые записи остаются валидными).
EMBEDS_HIT_DEFAULTS: dict[str, str] = {
    "schema": CACHE_SCHEMA,
    "encoder_dtype": DEFAULT_ENCODER_DTYPE,
    "encoder_device": DEFAULT_ENCODER_DEVICE,
    # 512 — дефолт бекенда FLUX.2 (`DEFAULT_MAX_SEQUENCE_LENGTH`): запись без этого поля снята
    # именно на нём, поэтому любое явное отличие (1024/2048) остаётся корректным промахом.
    "max_sequence_length": "512",
}
#: Старый сайдкар (`run_style.py`) писал те же значения под другими именами.
EMBEDS_HIT_ALIASES: dict[str, tuple[str, ...]] = {
    "image_sha": ("image_sha256_12",),
    "encoder": ("text_encoder",),
}


def embeds_expected(cfg: RunConfig, crop: tuple[int, int], prompt_md5_: str,
                    img_sha: str, backend_cls: Any = None) -> dict[str, str]:
    """Ожидаемые значения белого списка `EMBEDS_HIT_FIELDS` для текущего прогона.

    Бекенд может добавить свои поля хуком `cache_sidecar_values(cfg)` — так `max_sequence_length`
    FLUX.2 попадает в сверку без того, чтобы ядро знало про конкретную модель.
    """
    expected = {
        "schema": CACHE_SCHEMA,
        "prompt_md5": str(prompt_md5_),
        "image_sha": str(img_sha),
        "crop": f"{int(crop[0])}x{int(crop[1])}",
        "encoder": str(cfg.encoder.cache_tag),
        "encoder_repo": str(cfg.encoder.repo),
        "encoder_revision": str(cfg.encoder.revision or ""),
        "encoder_dtype": encoder_dtype_canon(cfg.encoder.dtype),
        "encoder_device": encoder_device_canon(cfg.encoder.device),
        "max_sequence_length": "",
    }
    hook = getattr(backend_cls, "cache_sidecar_values", None)
    if callable(hook):
        expected.update({str(k): str(v) for k, v in (hook(cfg) or {}).items()})
    return expected


def _sidecar_value(meta: Mapping[str, Any], field: str) -> Any:
    """Значение поля из сайдкара: верхний уровень -> `cache_key_parts` -> старые имена."""
    if field in meta:
        return meta[field]
    parts = meta.get("cache_key_parts")
    if isinstance(parts, Mapping) and field in parts:
        return parts[field]
    for alias in EMBEDS_HIT_ALIASES.get(field, ()):
        if alias in meta:
            return meta[alias]
        if isinstance(parts, Mapping) and alias in parts:
            return parts[alias]
    return None


def embeds_sidecar_mismatch(meta: Mapping[str, Any], expected: Mapping[str, str],
                            ) -> tuple[str, str, str] | None:
    """Первое расхождение сайдкара с ожидаемым: `(поле, было, стало)` или `None`.

    Сверка идёт по УЗКОМУ белому списку `EMBEDS_HIT_FIELDS`: смена `--seed`, `--steps`,
    `--lora-scale`, `--fp8` или `--offload` остаётся бесплатной. Пропущенное поле берётся
    из `EMBEDS_HIT_DEFAULTS`; если умолчания нет — считается, что доказательств изменения нет
    (так живут сайдкары старого формата, где `encoder_repo` не писался вовсе).
    """
    for field in EMBEDS_HIT_FIELDS:
        want = str(expected.get(field, ""))
        raw = _sidecar_value(meta, field)
        if raw is None:
            raw = EMBEDS_HIT_DEFAULTS.get(field)
            # Умолчание принадлежит модели, у которой поле есть (FLUX.2: 512), а не ядру: если
            # бекенд поля не объявляет (`want` пусто, Qwen-Image-2.1 такой записи не писал),
            # сверять нечего — иначе дефолт 512 «изменял» бы каждый сайдкар Qwen.
            if raw is not None and not want:
                continue
        if raw is None:
            continue
        got = str(raw).strip()
        if field in ("encoder_dtype", "encoder_device"):
            canon = encoder_dtype_canon if field == "encoder_dtype" else encoder_device_canon
            got, want = canon(got), canon(want)
        if got != want:
            return (field, got, want)
    return None


def embeds_miss_message(mismatch: tuple[str, str, str], sidecar: Path) -> str:
    """Строка «какой именно параметр изменился» — промах обязан быть объяснён (R1)."""
    field, old, new = mismatch
    return f"[cache] miss: изменился {field} ({old} -> {new}): {sidecar}"


def embeds_cache_hit(meta: Mapping[str, Any] | None, expected: Mapping[str, str],
                     sidecar: Path) -> tuple[bool, str]:
    """`(попадание, причина промаха)`: сайдкар сверяется по белому списку, если он есть.

    Сайдкара нет — попадание (так жили записи до его появления). `.pt` при расхождении
    НЕ удаляется: он просто перестаёт находиться этим ключом (удаление — только `cache prune`).
    """
    if not meta:
        return True, ""
    mismatch = embeds_sidecar_mismatch(meta, expected)
    if mismatch is None:
        return True, ""
    return False, embeds_miss_message(mismatch, sidecar)


def embeds_shape(payload: Any) -> str:
    """`(1, 512, 3584)` главного тензора payload или `'?'` — для логов и сайдкара."""
    for key in _SHAPE_KEYS:
        shape = getattr((payload or {}).get(key), "shape", None)
        if shape is not None:
            return str(tuple(int(v) for v in shape))
    return "?"


def _shapes(payload: Any) -> dict[str, list[int]]:
    """Все тензоры payload с формой: в сайдкар, чтобы видеть формат без `torch.load`."""
    return {key: [int(v) for v in value.shape]
            for key, value in dict(payload or {}).items()
            if getattr(value, "shape", None) is not None}


def vram_used_gib() -> float | None:
    """Занятая VRAM по данным CUDA (None — нет torch/CUDA: логи не должны падать)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return (total - free) / GIB
    except Exception:                            # noqa: BLE001 — диагностика не обязательна
        return None


def _release(backend: Any, handle: Any) -> None:
    """Отпустить модель до апскейла: бекенд делает `del`, здесь — `gc` + `empty_cache`."""
    backend.release(handle)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:                            # noqa: BLE001 — освобождение не критично
        pass


# --------------------------------------------------------------------------- #
# План прогона (без весов)
# --------------------------------------------------------------------------- #

def fit_cover(image: Any, width: int, height: int) -> Any:
    """Привести кадр к `width x height` без искажения пропорций: scale-to-cover + центральный кроп.

    Нужно для `--aspect`: когда пропорция кадра задана вручную и отличается от входной,
    обычный `resize` сплющил бы картинку, а так референс сохраняет геометрию.
    """
    from PIL import Image as _Image

    if image.size == (width, height):
        return image
    src_ratio = image.width / image.height
    dst_ratio = width / height
    if abs(src_ratio - dst_ratio) < 1e-6:
        return image.resize((width, height), _Image.LANCZOS)
    if src_ratio > dst_ratio:                 # шире кадра — режем по бокам
        new_w, new_h = max(width, round(image.width * height / image.height)), height
    else:                                     # выше кадра — режем сверху/снизу
        new_w, new_h = width, max(height, round(image.height * width / image.width))
    resized = image.resize((new_w, new_h), _Image.LANCZOS)
    left, top = (new_w - width) // 2, (new_h - height) // 2
    return resized.crop((left, top, left + width, top + height))


def input_problem(cfg: RunConfig, caps: Any) -> str | None:
    """Почему входной кадр непригоден для этого бекенда (или `None` = всё в порядке).

    Чистая и дешёвая функция: картинка не открывается (только `Path.is_file()`), тяжёлые
    импорты (torch/transformers/diffusers) не трогаются. Единственное место, которое знает,
    нужен ли `--input`: ядро читает `Capabilities.requires_image` бекенда, а не держит список
    имён моделей (R6). Вызывается в `_plan` до `Image.open`, чтобы `run --dry-run` и реальный
    прогон падали одним и тем же текстом **до** загрузки весов (а не трейсбеком внутри).

    Порядок веток: сначала «референс обязателен», потом «нечем задать пропорцию», потом
    «файл указан, но его нет» — так ошибка про `--input` всегда говорит о том, чего не хватает.
    """
    has_input = bool(str(cfg.input or "").strip())
    if not has_input:
        if bool(getattr(caps, "requires_image", True)):
            backend = str(cfg.generation.backend or "бекенд")
            return f"модель {backend} требует референс: передайте --input <png>"
        if not str(cfg.aspect or "").strip():
            return ("без --input нужен --aspect (16:9|4:3|1:1), иначе не из чего взять "
                    "пропорцию кадра")
        return None
    for path in (cfg.inputs or (cfg.input,)):      # все референсы, не только первый
        if not Path(str(path)).is_file():
            return f"нет входного кадра {path} (--input)"
    return None


def _plan(cfg: RunConfig, backend_cls: type, *, verbose: bool = False) -> Plan:
    """Вход -> сетка conditioning -> ключ кэша -> попадание/промах -> имя файла.

    Ничего не грузит: ни модели, ни энкодер. Именно поэтому `dry_run()` отвечает на вопрос
    «сколько это будет стоить» (энкод 55–255 с или только шаги сэмплера) без единого тензора.

    Без `--input` (T2I) картинок нет (`images = []`, `in_size = (0, 0)`, `img_sha = "noref"`),
    а пропорция кадра берётся из `--aspect` (`RunConfig.frame_ratio`); нужен ли референс вовсе,
    решает `input_problem()` по `Capabilities`.

    **Несколько референсов** (`--input a --input b`): открываются все, но сетка `crop` и `in_size`
    считаются по **первой** картинке (так же и `fit_cover` при `--aspect`), а ключ кэша берётся
    составной (`io.images_sha`) — одиночный путь даёт прежний хеш. Если картинок больше, чем
    `Capabilities.max_images`, лишние отбрасываются здесь (единственное место, знающее caps)
    с одной строкой в логе — прогон не падает.
    """
    caps = getattr(backend_cls, "caps", None)
    problem = input_problem(cfg, caps)
    if problem:
        raise ConfigError(problem)

    from PIL import Image

    prompt = load_prompt(cfg)
    paths = [str(p) for p in (cfg.inputs or ((cfg.input,) if cfg.input else ()))]
    limit = getattr(caps, "max_images", None)
    if limit and len(paths) > int(limit):
        total, keep = len(paths), int(limit)
        name = str(getattr(backend_cls, "name", "") or cfg.generation.backend)
        print(f"[in] картинок {total}, бекенд {name} поддерживает {keep} — "
              f"взяты image1..image{keep} (image{keep + 1}..image{total} отброшены)", flush=True)
        paths = paths[:keep]
    images = [Image.open(path).convert("RGBA") for path in paths]
    image = images[0] if images else None
    in_w, in_h = image.size if image is not None else (0, 0)
    resolution = int(cfg.effective_resolution(in_w, in_h))
    if verbose and cfg.size_px:
        ratio = cfg.frame_ratio(in_w, in_h)
        aspect = f"aspect={cfg.aspect}" if cfg.aspect else f"ratio {ratio:.3f}"
        print(f"[size] {cfg.size_px}p при {aspect} -> --resolution {resolution}", flush=True)
    crop = tuple(int(v) for v in cfg.crop(in_w, in_h))
    if cfg.aspect and image is not None:
        # пропорция кадра задана вручную: референс подгоняем без искажений (scale-to-cover),
        # иначе resize в бекенде сплющил бы картинку под чужую пропорцию.
        images = [fit_cover(item, crop[0], crop[1]) for item in images]
        image = images[0]
    md5 = prompt_md5(prompt)
    sha = images_sha(paths)

    if verbose:
        source = f"{in_w}x{in_h}" if image is not None else "нет референса"
        if len(images) > 1:                 # одиночный путь печатает историческую строку 1:1
            source += f" ({len(images)} референсов)"
        print(f"[in] {source} -> conditioning {crop[0]}x{crop[1]} | prompt {md5} | image {sha}",
              flush=True)

    blob = _cache_blob(backend_cls, cfg, crop, md5, sha)
    cache = blob_cache()
    path = cache.path(blob)
    cached: Any = None
    meta: dict[str, Any] | None = None
    if path.is_file():
        # Само имя файла уже гарантирует промт/картинку/crop/тег/dtype/device, но старые
        # записи писались без части этих сегментов — их догоняет сверка сайдкара.
        meta = cache.get_meta(blob)
        hit, reason = embeds_cache_hit(meta, embeds_expected(cfg, crop, md5, sha, backend_cls),
                                       cache.sidecar(blob))
        if hit:
            cached = cache.get(blob)
        else:
            meta = None
            if reason:
                print(reason, flush=True)     # R1: промах обязан называть изменившийся параметр

    encoder_role = str(getattr(caps, "encoder", "optional") or "optional")
    need_encoder = cached is None and encoder_role != "none"

    # Фактор для имени файла: 0.0, если апскейл ничего не делает (иначе в имени появился бы `-x1`,
    # см. `upscale_effective_factor`). На решение о вызове бекенда это тоже влияет: `run()`
    # вызывает `_postprocess` только при `factor > 0`.
    factor = upscale_effective_factor(cfg, crop)
    return Plan(images=images, in_size=(in_w, in_h), crop=crop, resolution=resolution,
                prompt=prompt, prompt_md5=md5, img_sha=sha, blob=blob, cache_path=path,
                cached=cached, cache_meta=meta, need_encoder=need_encoder, factor=factor,
                out_path=out_path(cfg, factor))


def _preflight_encoder(cfg: RunConfig, need_encoder: bool) -> type | None:
    """Класс текст-энкодера из реестра — без загрузки весов (17.5 ГБ грузит бекенд).

    Сам handle загружает бекенд генерации: он владеет processor/tokenizer и подменяет
    `pipe.encode_prompt`. Здесь только `get_encoder()` + `status()` — чтобы промах кэша падал
    понятным «нет весов в cache/weights/heretic-te», а не трейсом transformers.

    Незарегистрированное имя не ошибка: у FLUX.2 энкодер локальный (`encoder.name: qwen3`
    в пресете), его грузит бекенд из `generation.repo`.
    """
    name = str(cfg.encoder.name or "base").strip()
    if not need_encoder or name in ("", "base"):
        return None
    try:
        cls = get_encoder(name)
    except RegistryError:
        return None
    status = getattr(cls, "status", None)
    if callable(status):
        try:
            ok, why = cls().status(cfg)
        except Exception:                        # noqa: BLE001 — статус не должен ронять прогон
            ok, why = True, ""
        if not ok:
            raise RuntimeError(f"текст-энкодер {name!r} недоступен: {why}")
    return cls


# --------------------------------------------------------------------------- #
# Conditioning: кэш или энкод
# --------------------------------------------------------------------------- #

def _conditioning(cfg: RunConfig, backend: Any, handle: Any, plan: Plan,
                  timings: dict[str, float], cleanup: DeferredCleanup | None = None) -> Any:
    """Эмбеддинги: попадание в кэш печатаем и не пересчитываем, промах — энкодим и пишем.

    `[4/5] кодирование промта…` печатает бекенд (он знает, чем и на каком устройстве кодирует),
    строку про кэш — только этот модуль, иначе у каждой модели была бы своя копия логирования.

    Успешная запись сразу регистрирует отложенную задачу вытеснения (`cleanup`), но НИЧЕГО не
    удаляет: старые записи этой пары нужны, пока кадр не сохранён.
    """
    if plan.cached is not None:
        print(f"[4/5] embeddings из кэша {plan.blob.name} {embeds_shape(plan.cached)}", flush=True)
        return plan.cached

    started = time.time()
    payload = backend.encode(cfg, handle, plan.images, plan.prompt)
    timings["encode"] = time.time() - started

    stored = dict(payload or {})
    stored.setdefault("prompt", plan.prompt)      # как в run_style.py: в .pt лежит и промт
    cache = blob_cache()
    path = cache.put(plan.blob, stored, _embeds_meta(cfg, plan, payload))
    print(f"    эмбеддинги сохранены: {path}", flush=True)
    if cleanup is not None:
        write = _cache_write(cfg, plan)
        if write is not None:
            cleanup.register(write)                 # удаление — только в конце прогона
    return payload


def _embeds_meta(cfg: RunConfig, plan: Plan, payload: Any) -> dict[str, Any]:
    """Сайдкар `.json`: чем именно сгенерирован кэш (поля run_style.py + ключ целиком)."""
    gen = cfg.generation
    lora = gen.lora
    tensors = _shapes(payload)
    return {
        # поля 1:1 с run_style.py: по ним старые записи инспектируются без чтения бинаря
        "prompt_md5": plan.prompt_md5,
        "prompt_file": cfg.prompt_file,
        "image": str(cfg.input),                   # первый референс (прежнее поле)
        "images": [str(p) for p in cfg.inputs][: len(plan.images)],   # все референсы в порядке промта
        "image_sha256_12": plan.img_sha,
        "text_encoder": cfg.encoder.name,
        "encoder_dtype": encoder_dtype_canon(cfg.encoder.dtype),
        "encoder_device": encoder_device_canon(cfg.encoder.device),
        "crop": f"{plan.crop[0]}x{plan.crop[1]}",
        "resolution": plan.resolution,
        "steps": cfg.steps,
        "seed": cfg.seed,
        "lora_scale": lora.scale if lora else None,
        "fp8": str(gen.precision or "").strip().lower() == "fp8",
        "base_model": gen.repo,
        "lora": lora.filename if lora else None,
        "embeds_shape": tensors.get("embeds") or tensors.get("prompt_embeds") or [],
        # дополнительно: всё, что влияет на кадр, и формы остальных тензоров
        "backend": gen.backend,
        "preset": cfg.preset,
        "key": plan.blob.name,
        "tensors": tensors,
        "cache_key_parts": cfg.cache_key_parts(plan.crop, plan.prompt_md5, plan.img_sha),
        # бекенд-зависимые поля, которые участвуют в сверке попадания (R1): у FLUX.2 —
        # длина контекста текстового энкодера (512/1024 дают разную форму эмбеддингов)
        **({"max_sequence_length": str(payload["max_sequence_length"])}
           if isinstance(payload, Mapping) and payload.get("max_sequence_length") is not None else {}),
    }


# --------------------------------------------------------------------------- #
# Апскейл
# --------------------------------------------------------------------------- #

def _fit_height(image: Any, height: int) -> Any:
    """Точная высота (ширина по пропорциям, чётная) — как `tools/batch_seeds.py` (`--final-h`)."""
    from PIL import Image

    target_w = int(round(int(height) * image.width / image.height))
    target_w -= target_w % 2
    target = (max(2, target_w), int(height))
    if image.size == target:
        return image
    return image.resize(target, Image.LANCZOS)


#: Допуск сравнения фактора с 1.0: `1±1e-6` — апскейл ничего не меняет (дробные 1.333/1.5 не задеты).
FACTOR_ONE_EPS = 1e-6


def upscale_skip_reason(cfg: RunConfig, size: tuple[int, int], factor: float) -> str:
    """Почему бекенд апскейла вызывать не нужно: `''` — нужен, иначе текст причины.

    Проверяется ДО загрузки весов апскейлера: `esrgan x1` на 1920x1440 стоил 9.6 s и не менял
    ни одного пикселя (план `qwen1440p-perf`, шаг 3). Порядок веток важен: `target_height`
    главнее `factor` (как и в `RunConfig.upscale_factor`), поэтому «уже достигнуто» и
    «выключено пользователем» (`--upscale 0` без цели по высоте) — разные сообщения.
    """
    width, height = int(size[0]), int(size[1])
    target = cfg.upscale.target_height
    try:
        value = float(factor)
    except (TypeError, ValueError):
        value = 0.0
    if target is not None and int(target) <= height:
        return f"кадр {width}x{height} уже не ниже целевой {int(target)}"
    if abs(value - 1.0) <= FACTOR_ONE_EPS:
        return f"factor={value:g}"
    if value <= 0 and not target:
        return f"апскейл выключен (factor={value:g})"
    return ""


def upscale_effective_factor(cfg: RunConfig, gen_size: tuple[int, int]) -> float:
    """Фактор апскейла для решения и для имени файла: `0.0`, если бекенд вызывать не нужно.

    `io.out_path` дописывает `-x<factor>` только при factor > 0, поэтому no-op апскейл не должен
    получать `-x1`: иначе один и тот же кадр лежит под двумя именами (`out/x-s7.png` и
    `out/x-x1-s7.png`). `gen_size` — размер сгенерированного кадра (у Qwen он равен `crop`).
    """
    width, height = int(gen_size[0]), int(gen_size[1])
    factor = float(cfg.upscale_factor(height))
    if factor <= 0 and not cfg.upscale.target_height:
        return factor                            # выключено пользователем — так и остаётся 0.0
    if upscale_skip_reason(cfg, (width, height), factor):
        return 0.0
    return factor


def _print_upscale_skip(cfg: RunConfig, size: tuple[int, int]) -> None:
    """Одна строка «почему апскейла не будет» вместо загрузки бекенда."""
    width, height = int(size[0]), int(size[1])
    factor = float(cfg.upscale_factor(height))
    reason = upscale_skip_reason(cfg, (width, height), factor) or f"factor={factor:g}"
    print(f"[post] upscale пропущен: {reason}", flush=True)


def _postprocess(cfg: RunConfig, image: Any, factor: float, seed: int) -> tuple[Any, float]:
    """Апскейл кадра выбранным бекендом; `source` — сам кадр (цветокоррекция SeedVR2).

    Если кадр уже не ниже `upscale.target_height` или фактор равен 1, бекенд **не загружается**
    вовсе: печатается `[post] upscale пропущен: …` и кадр возвращается как есть с нулевым
    временем апскейла (`timings["upscale"] = 0`). Имя файла при этом уже не получает `-x1`:
    фактор для `io.out_path` обнуляет `upscale_effective_factor` в `_plan`.
    """
    size = (int(image.size[0]), int(image.size[1]))
    reason = upscale_skip_reason(cfg, size, factor)
    if reason:
        print(f"[post] upscale пропущен: {reason}", flush=True)
        return image, 0.0
    cls = get_upscale(cfg.upscale.backend)
    started = time.time()
    big = cls().upscale(image, factor, cfg.upscale, seed, source=image)
    if cfg.upscale.target_height:
        big = _fit_height(big, int(cfg.upscale.target_height))
    elapsed = time.time() - started
    print(f"[post] {cfg.upscale.backend} x{factor:g}: {image.size[0]}x{image.size[1]} -> "
          f"{big.size[0]}x{big.size[1]}  ({elapsed:.1f}s)", flush=True)
    return big, elapsed


def _print_done(path: Path, gen_size: tuple[int, int], out_size: tuple[int, int],
                timings: Mapping[str, float]) -> None:
    used = vram_used_gib()
    peak = f"  peak vram {used:.1f} GiB" if used is not None else ""
    print(f"[done] {path} gen {gen_size[0]}x{gen_size[1]} -> out {out_size[0]}x{out_size[1]}"
          f"{peak}  total {timings.get('total', 0.0):.0f}s", flush=True)


# --------------------------------------------------------------------------- #
# Батч: парсинг seed'ов
# --------------------------------------------------------------------------- #

def parse_seeds(spec: str) -> list[int]:
    """`'0-9'` / `'1,7,42'` / `'0-2,10'` -> список seed'ов (как `tools/batch_seeds.py`)."""
    seeds: list[int] = []
    for part in str(spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            first, _, last = part.partition("-")
            try:
                start, stop = int(first), int(last)
            except ValueError as exc:
                raise ConfigError(f"seeds: {part!r} — ожидалось 'A-B' с целыми A, B") from exc
            if stop < start:
                raise ConfigError(f"seeds: {part!r} — конец диапазона меньше начала")
            seeds.extend(range(start, stop + 1))
        else:
            try:
                seeds.append(int(part))
            except ValueError as exc:
                raise ConfigError(f"seeds: {part!r} — не целое число") from exc
    return seeds


# --------------------------------------------------------------------------- #
# Одиночный прогон
# --------------------------------------------------------------------------- #

def _open_log(log: "RunLog | None", directory: Path, cfg: RunConfig) -> None:
    """Открыть лог прогона в конкретной папке (no-op без лога — `--no-logs` или вызов из теста).

    Вызывается **после** `_plan`: только там известно настоящее имя кадра (`out_path` с
    `-s{seed}` и `-x{factor}`), а значит и папка лога (`io.log_dir`). Строки, напечатанные
    раньше (`[plan]`), уже лежат в буфере `RunLog` и потянутся в файл в момент `attach`.

    `cfg.log_dir` перебивает `directory`: `gen` кладёт `run.log`/`meta.json` в корень сессии
    (`out/<имя>/gen-<stamp>/`), а не в подпапку кадра. `None` — прежняя раскладка.
    """
    if log is None:
        return
    explicit = getattr(cfg, "log_dir", None)
    log.attach(Path(explicit) if explicit else directory, cfg=cfg)


def run(cfg: RunConfig, log: "RunLog | None" = None) -> Result:
    """Один кадр: prepare -> encode/кэш -> generate -> release -> upscale -> save.

    `log` — открытый `qwenstyle.runlog.RunLog` (его включает `cmd/run.py`); `None` — без лога
    (так ходят `selftest` и тесты: они не должны оставлять папок рядом с чужими кадрами).
    """
    t0 = time.time()
    backend_cls = get_generation(cfg.generation.backend)
    plan = _plan(cfg, backend_cls, verbose=True)
    _open_log(log, log_dir(plan.out_path), cfg)      # папка лога = папка кадра без расширения
    _preflight_encoder(cfg, plan.need_encoder)      # до загрузки модели: понятная ошибка
    cleanup = _cleanup_for(cfg)                     # отложенная задача вытеснения кэша

    timings: dict[str, float] = {}
    backend = backend_cls()
    started = time.time()
    handle = backend.prepare(cfg, plan.need_encoder)
    timings["prepare"] = time.time() - started

    cond = _conditioning(cfg, backend, handle, plan, timings, cleanup)

    started = time.time()
    gen_image = backend.generate(cfg, handle, cond, plan.images, cfg.seed)
    timings["generate"] = time.time() - started
    gen_size = (int(gen_image.size[0]), int(gen_image.size[1]))

    # пайплайн на пике держит до 16 ГБ: перед апскейлом его обязательно отпустить
    _release(backend, handle)

    result_image = gen_image
    if plan.factor > 0:
        result_image, timings["upscale"] = _postprocess(cfg, gen_image, plan.factor, cfg.seed)
    else:
        # Пропуск решён ещё в плане (`upscale_effective_factor`): печатаем причину здесь, чтобы
        # бекенд апскейла не только не работал, но и не загружался.
        _print_upscale_skip(cfg, gen_size)
    out_size = (int(result_image.size[0]), int(result_image.size[1]))

    target = save_atomic(result_image, plan.out_path)
    timings["total"] = time.time() - t0
    _print_done(target, gen_size, out_size, timings)
    # Отложенная очистка — строго после сохранения кадра: удалять кэш до этого нельзя (упади
    # декод/апскейл — остались бы и без кадра, и без старого кэша). При исключении выше
    # до сюда просто не доходит.
    _flush_cleanup(cfg, cleanup)
    return Result(image=result_image, gen_size=gen_size, out_size=out_size, seed=int(cfg.seed),
                  timings=timings, out_path=target)


# --------------------------------------------------------------------------- #
# Батч по seed'ам
# --------------------------------------------------------------------------- #

def run_batch(cfg: RunConfig, log: "RunLog | None" = None) -> list[Result]:
    """N seed'ов за одну загрузку модели (`tools/batch_seeds.py`), затем пакетный апскейл.

    Один `prepare` + один `encode`: старт fp8-пайплайна и CPU-энкод платятся ровно раз.
    Генерация каждого seed'а идёт тем же `backend.generate`, что и в `run()`, поэтому один и
    тот же seed даёт тот же кадр (`s{N}-gen.png`). Модель отпускается **после** всех генераций
    и **до** первого апскейла: ESRGAN/SeedVR2 должны получить всю VRAM.

    Файлы: `<out-dir>/s{N}-gen.png` (чистая генерация, только если апскейл включён и `keep_gen`)
    и `<out-dir>/s{N}.png` (после апскейла; при выключенном апскейле — сама генерация, без дубля
    `-gen`). `resume` пропускает seed'ы, у которых `s{N}.png` уже есть.
    """
    batch: BatchConfig | None = cfg.batch
    if batch is None:
        raise ConfigError("run_batch: в конфиге нет секции batch (нужны seeds/out_dir)")

    t0 = time.time()
    backend_cls = get_generation(cfg.generation.backend)
    plan = _plan(cfg, backend_cls, verbose=True)
    # одна папка лога на сессию: `cfg.log_dir` (gen) или прежний `<out-dir>/_logs/<stamp>`
    _open_log(log, batch_log_dir(batch.out_dir), cfg)
    seeds = parse_seeds(batch.seeds)
    if not seeds:
        raise ConfigError(f"run_batch: пустой список seed'ов ({batch.seeds!r})")

    out_dir = Path(batch.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo: list[int] = []
    for index, seed in enumerate(seeds, start=1):
        _gen_path, up_path = batch_paths(batch.out_dir, seed)
        if batch.resume and Path(up_path).is_file():
            print(f"[batch {index}/{len(seeds)}] seed={seed} — пропуск: уже есть {up_path}",
                  flush=True)
            continue
        todo.append(seed)
    if not todo:
        print(f"[batch] все кадры уже готовы в {out_dir} ({len(seeds)} шт.), генерации нет",
              flush=True)
        return []

    _preflight_encoder(cfg, plan.need_encoder)
    cleanup = _cleanup_for(cfg)                     # отложенная задача вытеснения кэша
    timings: dict[str, float] = {}
    backend = backend_cls()
    started = time.time()
    handle = backend.prepare(cfg, plan.need_encoder)
    timings["prepare"] = time.time() - started

    cond = _conditioning(cfg, backend, handle, plan, timings, cleanup)

    frames: list[tuple[int, Any, float]] = []
    for index, seed in enumerate(seeds, start=1):
        gen_path, _up_path = batch_paths(batch.out_dir, seed)
        if seed not in todo:
            continue
        print(f"[batch {index}/{len(seeds)}] seed={seed}", flush=True)
        started = time.time()
        image = backend.generate(cfg, handle, cond, plan.images, seed)
        elapsed = time.time() - started
        used = vram_used_gib()
        print(f"[gen] seed {seed}: {elapsed:.1f}s {image.size[0]}x{image.size[1]}"
              + (f" peak {used:.1f} GiB" if used is not None else ""), flush=True)
        if batch.keep_gen and plan.factor > 0:
            # только если апскейл реально изменит кадр: иначе s{N}.png и есть генерация,
            # и -gen был бы байт-в-байт дублем.
            save_atomic(image, gen_path)          # сразу на диск: прерванный батч не теряет кадры
        frames.append((seed, image, elapsed))

    _release(backend, handle)                      # до апскейла: освобождаем всю VRAM

    factor = float(cfg.upscale_factor(frames[0][1].size[1]))
    results: list[Result] = []
    up_seconds: list[float] = []
    for seed, image, gen_elapsed in frames:
        gen_size = (int(image.size[0]), int(image.size[1]))
        final = image
        elapsed = 0.0
        if factor > 0:
            final, elapsed = _postprocess(cfg, image, factor, seed)
            up_seconds.append(elapsed)
        _gen_path, up_path = batch_paths(batch.out_dir, seed)
        target = save_atomic(final, up_path)
        results.append(Result(image=final, gen_size=gen_size,
                              out_size=(int(final.size[0]), int(final.size[1])),
                              seed=int(seed),
                              timings={"generate": gen_elapsed, "upscale": elapsed},
                              out_path=target))

    timings["generate"] = sum(item[2] for item in frames)
    timings["upscale"] = sum(up_seconds)
    timings["total"] = time.time() - t0
    print(f"[done] {len(results)} кадров в {out_dir} за {timings['total']:.0f}s "
          f"(генерация {timings['generate']:.0f}s, апскейл {timings['upscale']:.0f}s)", flush=True)
    # Отложенная очистка — после сохранения всех кадров (сейчас батч берёт эмбеддинги из кэша,
    # поэтому задач обычно нет: `flush` тогда ничего не печатает и ничего не трогает).
    _flush_cleanup(cfg, cleanup)
    return results



# --------------------------------------------------------------------------- #
# План без загрузки моделей
# --------------------------------------------------------------------------- #

def dry_run(cfg: RunConfig) -> dict[str, Any]:
    """Что именно произойдёт: сетка, ключ кэша, нужен ли энкодер, шаги, фактор, имя файла.

    Ничего не загружает и не скачивает (весов не касается вовсе). Сетка conditioning считается
    через `RunConfig.crop()` — единственное место, где вызывается diffusers (чистая функция
    `calculate_dimensions`), поэтому torch может оказаться в `sys.modules`, но веса — нет.

    `need_encoder=True` — прямой ответ на «заплатим ли 55–255 с CPU-энкода»: промах кэша.

    Заодно печатается план вытеснения (`cache_supersede`): какие записи этой пары
    устарели бы в конце прогона. Печать идёт через `flush(apply=False)` — файлы не трогаются.

    План печатается **только при промахе кэша**: при попадании новый `.pt` не пишется, задача
    вытеснения не регистрируется (`_conditioning` вызывается лишь на промахе) и удалять в конце
    прогона нечего — вместо плана печатается одна строка «вытеснение: нечего».
    """
    backend_cls = get_generation(cfg.generation.backend)
    plan = _plan(cfg, backend_cls, verbose=False)
    out_w, out_h = _estimate_out(plan.crop, plan.factor, cfg.upscale.target_height)
    raw_factor = float(cfg.upscale_factor(int(plan.crop[1])))
    skip = upscale_skip_reason(cfg, (int(plan.crop[0]), int(plan.crop[1])), raw_factor)
    if skip:
        print(f"[plan] upscale пропущен: {skip}", flush=True)
    info: dict[str, Any] = {
        "backend": cfg.generation.backend,
        "caps": (getattr(backend_cls, "caps", None).to_dict()
                 if hasattr(getattr(backend_cls, "caps", None), "to_dict") else None),
        "preset": cfg.preset,
        "seed": cfg.seed,
        "input": str(cfg.input),
        "inputs": [str(p) for p in cfg.inputs][: len(plan.images)],   # уже без отброшенных
        "in_size": f"{plan.in_size[0]}x{plan.in_size[1]}",
        "size_px": cfg.size_px,
        "resolution": plan.resolution,
        "crop": f"{plan.crop[0]}x{plan.crop[1]}",
        "gen_size": f"{plan.crop[0]}x{plan.crop[1]}",
        "steps": cfg.steps,
        "prompt_file": cfg.prompt_file,
        "prompt_md5": plan.prompt_md5,
        "image_sha": plan.img_sha,
        "encoder": cfg.encoder.name,
        "embeds_key": cfg.embeds_key(plan.crop, plan.prompt_md5, plan.img_sha),
        "embeds_stage": plan.blob.stage,
        "embeds_path": plan.cache_path.as_posix(),
        "embeds_cached": plan.cached is not None,
        "embeds_created": (plan.cache_meta or {}).get("created"),
        "need_encoder": plan.need_encoder,
        "cache_dir": blob_cache().root.as_posix(),
        "cache_supersede": normalize_policy(cfg.cache_supersede),
        "precision": cfg.generation.precision,
        "offload": cfg.generation.offload,
        "guidance": cfg.generation.guidance,
        "lora": ({"repo": cfg.generation.lora.repo, "filename": cfg.generation.lora.filename,
                  "scale": cfg.generation.lora.scale} if cfg.generation.lora else None),
        "upscale_backend": cfg.upscale.backend,
        "upscale_factor": round(plan.factor, 4),
        "upscale_skipped": bool(skip),
        "upscale_target_height": cfg.upscale.target_height,
        "out_size": f"{out_w}x{out_h}",
        "out_path": plan.out_path,
        "offline": str(os.environ.get("HF_HUB_OFFLINE", "")).strip().lower()
        in ("1", "true", "yes", "on"),
    }
    if cfg.batch is not None:
        seeds = parse_seeds(cfg.batch.seeds)
        info["batch"] = {
            "seeds": seeds,
            "out_dir": str(cfg.batch.out_dir),
            "keep_gen": cfg.batch.keep_gen,
            "resume": cfg.batch.resume,
            "pending": [seed for seed in seeds
                        if not Path(batch_paths(cfg.batch.out_dir, seed)[1]).is_file()],
            "files": [batch_paths(cfg.batch.out_dir, seed) for seed in seeds],
        }
    # План вытеснения без удаления: что устарело бы при появлении этой пары (crop не участвует).
    # Только на промахе: попадание в кэш ничего не пишет, поэтому и вытеснять нечего — иначе
    # dry-run обещал бы удаление, которого реальный прогон не сделает (см. `_conditioning`).
    cleanup = _cleanup_for(cfg)
    write = _cache_write(cfg, plan) if plan.cached is None else None
    targets: list = []
    if cleanup is not None and write is not None:
        cleanup.register(write)
        targets = cleanup.plan(cfg.cache_supersede, cleanup.root)
        info["cleanup_plan"] = plan_names(targets)
        cleanup.flush(cfg.cache_supersede, apply=False)     # только печать: файлы не трогаются
    else:
        info["cleanup_plan"] = []
        if cleanup is not None and plan.cached is not None:
            print(f"[cache] вытеснение ({normalize_policy(cfg.cache_supersede)}): нечего — "
                  f"кадр из кэша ({plan.blob.name}), новый .pt не пишется", flush=True)
    return info


def _estimate_out(crop: tuple[int, int], factor: float, target_height: int | None) -> tuple[int, int]:
    """Размер финального кадра без генерации (для `--dry-run`): как считает апскейл."""
    if factor <= 0:
        return int(crop[0]), int(crop[1])
    if target_height:
        width = int(round(int(target_height) * crop[0] / crop[1]))
        width -= width % 2
        return max(2, width), int(target_height)
    return int(round(crop[0] * factor)), int(round(crop[1] * factor))


__all__ = [
    "DEFAULT_CACHE_DIR", "Plan", "Result", "blob_cache", "dry_run", "embeds_shape",
    "input_problem", "parse_seeds", "run", "run_batch", "vram_used_gib",
]
