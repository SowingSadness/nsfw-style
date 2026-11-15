"""Контракты бекендов: `Capabilities`, `ResourceBudget`, протоколы и вердикт по железу.

Здесь нет ни одной реализации модели: только интерфейсы, которые обязан соблюсти любой модуль
`qwenstyle/backends/<name>.py` (план `qwenstyle-framework`, шаг 2). Тяжёлые зависимости
(torch/psutil) импортируются лениво внутри функций, поэтому `models list` работает без CUDA.

Правила (см. AGENTS.md):
* бюджет ресурсов объявляет сам бекенд (`ResourceBudget`), а проверяет оркестратор/`models probe`
  через `hardware_budget_verdict()` — до скачивания весов;
* conditioning (`precomputed`/`internal`) — единственный признак, по которому оркестратор решает,
  кэшировать ли эмбеддинги `.pt` (Qwen-Image-2.1) или отдавать промт/референсы в пайплайн (FLUX.2).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol

if TYPE_CHECKING:                                   # только для типов: не тянем PIL/diffusers
    from PIL import Image

    from ..config import RunConfig, UpscaleConfig

#: `precomputed` — бекенд сам считает эмбеддинги (кэш `.pt`); `internal` — пайплайн принимает промт
#: и список референсов целиком (FLUX.2: текстовые эмбеддинги кэшируются, картинки кодируются внутри).
ConditioningMode = Literal["precomputed", "internal"]
#: Сколько картинок на входе: `none` — чисто T2I, `single` — img2img, `multi` — набор референсов.
ImageInputs = Literal["none", "single", "multi"]
#: Нужен ли внешний текст-энкодер: `required` — без него пайплайн не соберётся,
#: `optional` — грузим только при промахе кэша эмбеддингов, `none` — энкодер не нужен вовсе.
EncoderRole = Literal["required", "optional", "none"]

GIB = 1024 ** 3
#: Корень репозитория: от него считаются `cache/`, `hf-cache/` и свободное место на диске.
REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# Дескрипторы возможностей и бюджета
# --------------------------------------------------------------------------- #

def _fmt_gib(value: float | None) -> str:
    return "нет данных" if value is None else f"{value:.1f} GiB"


def _fmt_steps(value: int | None) -> str:
    return "-" if value is None else str(value)


@dataclass(frozen=True)
class Capabilities:
    """Что умеет бекенд. Объявляется классом бекенда (`caps: ClassVar[Capabilities]`).

    Оркестратор и CLI читают только эти поля: подстановка новой модели не требует правок ядра.
    """

    conditioning: ConditioningMode
    image_inputs: ImageInputs = "single"
    #: Сколько референсов модель принимает в **одном** запросе (`None` = без явного предела).
    #: Ядро читает поле в `pipeline._plan`: лишние картинки отбрасываются с одной строкой в логе
    #: («взяты image1..image10»), а не роняют прогон. У Qwen-Image-2.1 предел 10 (в промте модели
    #: плейсхолдеры `<image1>`…`<image10>`), у FLUX.2/MiniMax-H3 предел не объявлен.
    max_images: int | None = None
    #: Обязателен ли референс на входе. `True` (умолчание) — генерация без `--input` падает
    #: preflight'ом `pipeline.input_problem()` до загрузки весов; `False` — чистое T2I
    #: (H3-Image: image-блоки пайплайна просто не запускаются). `image_inputs` этого не заменяет:
    #: `multi` у FLUX.2 по-прежнему требует кадр, а у H3 референсов может быть несколько *или ноль*.
    requires_image: bool = True
    encoder: EncoderRole = "optional"
    fixed_steps: int | None = None                 # 8 для turbo (расписание sigmas фиксировано)
    supports_lora: bool = False
    precision: tuple[str, ...] = ("bf16",)         # ("bf16",) | ("bf16", "fp8")
    offload: tuple[str, ...] = ("none",)           # ("none", "group", "model_cpu")
    max_pixels: int | None = None                  # предел площади кадра, если у модели он есть
    extra_sampling: tuple[str, ...] = ()           # ("guidance_scale", "caption_upsample_temperature")

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def summary(self) -> str:
        """Одна строка для `models list`."""
        parts = [
            f"conditioning={self.conditioning}",
            f"image_inputs={self.image_inputs}",
            f"max_images={_fmt_steps(self.max_images)}",
            f"encoder={self.encoder}",
            f"fixed_steps={_fmt_steps(self.fixed_steps)}",
            f"precision={'|'.join(self.precision) or '-'}",
            f"offload={'|'.join(self.offload) or '-'}",
            f"lora={'yes' if self.supports_lora else 'no'}",
            f"max_pixels={_fmt_steps(self.max_pixels)}",
            f"sampling={'|'.join(self.extra_sampling) or '-'}",
        ]
        return " ".join(parts)


@dataclass(frozen=True)
class ResourceBudget:
    """Объявленный бюджет: что бекенд займёт на диске, в VRAM и в RAM.

    `weights_bf16_gib`/`weights_fp8_gib` — веса генератора в VRAM, `encoder_gib` — текст-энкодер
    (в этом проекте живёт в RAM), `disk_gib` — сколько нужно скачать/держать в кэше.
    """

    weights_bf16_gib: float = 0.0
    weights_fp8_gib: float | None = None
    encoder_gib: float = 0.0
    disk_gib: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def summary(self) -> str:
        """Короткая строка для `models list` (без notes)."""
        weights = f"вес bf16 {self.weights_bf16_gib:.1f} GiB"
        if self.weights_fp8_gib is not None:
            weights += f" / fp8 {self.weights_fp8_gib:.1f} GiB"
        return (f"{weights}, энкодер {self.encoder_gib:.1f} GiB, диск {self.disk_gib:.1f} GiB")

    def __str__(self) -> str:                       # удобно печатать целиком, с notes
        text = self.summary()
        return f"{text} — {self.notes}" if self.notes else text


# --------------------------------------------------------------------------- #
# Протоколы (структурные, без runtime-checkable: проверку делает registry)
# --------------------------------------------------------------------------- #

class GenerationBackend(Protocol):
    """Модель генерации. Регистрация: `@register_generation` в `qwenstyle/backends/<name>.py`."""

    name: ClassVar[str]
    caps: ClassVar[Capabilities]
    budget: ClassVar[ResourceBudget]
    #: Репозитории пайплайна, которые обслуживает бекенд (для `models probe`); может отсутствовать.
    repos: ClassVar[tuple[str, ...]]

    def prepare(self, cfg: "RunConfig", need_encoder: bool) -> Any:
        """Загружает всё нужное (пайплайн/трансформер/энкодер) и возвращает handle."""

    def encode(self, cfg: "RunConfig", handle: Any, images: "list[Image.Image] | None",
               prompt: str) -> Any:
        """Conditioning: эмбеддинги промта и **всех** референсов (или `None` при `internal`/T2I).

        `images` — список картинок входа (пустой/T2I — `None` или `[]`); список длины 1 — обычный
        режим img2img, и бекенд обязан обрабатывать его так же, как раньше одиночную картинку.
        Больше `Capabilities.max_images` сюда не придёт: лишнее отбрасывает `pipeline._plan`.
        """

    def generate(self, cfg: "RunConfig", handle: Any, cond: Any,
                 images: "list[Image.Image] | None", seed: int) -> "Image.Image":
        """Сэмплирование: тот же seed и параметры обязаны давать побайтово тот же кадр.

        `images` — тот же список референсов, что у `encode` (список длины 1 — обычный режим;
        `None`/`[]` — text-to-image). Одиночная `PIL.Image` тоже принимается: так бекенд зовут
        `bench`/одноразовые проверки, и это не должно требовать от них знать про список.
        """

    def release(self, handle: Any) -> None:
        """Освобождает VRAM/RAM перед апскейлом (`del` + `gc` + `empty_cache`)."""

    def probe(self, cfg: "RunConfig") -> dict[str, Any]:
        """Диагностика без генерации: версии, размер весов, состояние fp8-кэша."""


def as_image_list(value: Any) -> list[Any]:
    """`PIL | list[PIL] | tuple[PIL] | None` -> список референсов (пустой для `None`).

    Единственное место, где бекенд приводит вход к списку: ядро передаёт `list` (даже для одного
    кадра), но `bench`/тесты/старые обёртки зовут `encode`/`generate` одиночной картинкой, и это
    должно оставаться рабочим. `None`-элементы отбрасываются (T2I-ветка H3).
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [item for item in value if item is not None]
    return [value]


class UpscaleBackend(Protocol):
    """Постобработка (ESRGAN/SeedVR2/…). Регистрация: `@register_upscale`."""

    name: ClassVar[str]

    def status(self) -> tuple[bool, str]:
        """(доступен?, причина). Недоступный бекенд не должен ронять CLI."""

    def upscale(self, image: "Image.Image", factor: float, cfg: "UpscaleConfig", seed: int,
                source: "Image.Image | None" = None) -> "Image.Image":
        """Увеличивает кадр в `factor` раз; `source` — исходник для color-fix."""


class EncoderBackend(Protocol):
    """Текст-энкодер. `cache_tag` — часть ключа кэша эмбеддингов (R1, байт-в-байт)."""

    name: ClassVar[str]
    cache_tag: ClassVar[str]
    requires_vision: ClassVar[bool]

    def load(self, cfg: "RunConfig") -> Any:
        """Грузит веса (в этом проекте — bf16 на CPU) и отдаёт handle."""

    def encode(self, cfg: "RunConfig", handle: Any, prompt: str,
               image: "Image.Image | None") -> dict[str, Any]:
        """-> {"embeds", "mask", "img_mask"} (формат `.pt`-кэша)."""


# --------------------------------------------------------------------------- #
# Железо и вердикт по бюджету
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Hardware:
    """Измеренные ресурсы машины. `None` = измерить не удалось (нет torch/psutil/CUDA)."""

    vram_free_gib: float | None = None
    vram_total_gib: float | None = None
    ram_free_gib: float | None = None
    ram_total_gib: float | None = None
    disk_free_gib: float | None = None
    disk_total_gib: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def summary(self) -> str:
        vram = ("нет данных" if self.vram_free_gib is None
                else f"{self.vram_free_gib:.1f}/{self.vram_total_gib:.1f} GiB")
        ram = ("нет данных" if self.ram_free_gib is None
               else f"{self.ram_free_gib:.1f}/{self.ram_total_gib:.1f} GiB")
        disk = ("нет данных" if self.disk_free_gib is None
                else f"{self.disk_free_gib:.1f}/{self.disk_total_gib:.1f} GiB")
        return f"VRAM {vram}, RAM {ram}, диск {disk}"


def hardware_resources(path: str | Path | None = None) -> Hardware:
    """Свободно/всего для VRAM (текущая CUDA-карта), RAM и диска под `path` (по умолчанию корень)."""
    vram_free = vram_total = None
    try:
        import torch                                     # лениво: без CUDA CLI должен работать

        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            vram_free, vram_total = free / GIB, total / GIB
    except Exception:                                    # noqa: BLE001 — диагностика не обязана работать
        pass

    ram_free = ram_total = None
    try:
        import psutil

        mem = psutil.virtual_memory()
        ram_free, ram_total = mem.available / GIB, mem.total / GIB
    except Exception:                                    # noqa: BLE001
        pass

    disk_free = disk_total = None
    try:
        usage = shutil.disk_usage(str(path or REPO_ROOT))
        disk_free, disk_total = usage.free / GIB, usage.total / GIB
    except OSError:
        pass

    return Hardware(vram_free_gib=vram_free, vram_total_gib=vram_total,
                    ram_free_gib=ram_free, ram_total_gib=ram_total,
                    disk_free_gib=disk_free, disk_total_gib=disk_total)


def hardware_budget_verdict(budget: ResourceBudget, resources: Hardware | None = None,
                            path: str | Path | None = None) -> str:
    """Вердикт по объявленному бюджету: `OK …` / `НЕ ВЛЕЗАЕТ: …` / `нет данных: …`.

    Сравнивает веса (fp8, если объявлен, иначе bf16) со свободной VRAM, энкодер — со свободной RAM,
    `disk_gib` — со свободным местом. Неизмеримое считается «нет данных», а не ошибкой: CLI должен
    работать и на машине без CUDA.
    """
    hw = resources if resources is not None else hardware_resources(path)
    problems: list[str] = []
    details: list[str] = []
    measured = 0

    vram_need = budget.weights_fp8_gib if budget.weights_fp8_gib is not None else budget.weights_bf16_gib
    if vram_need:
        mode = "fp8" if budget.weights_fp8_gib is not None else "bf16"
        text = f"VRAM {vram_need:.1f} GiB {mode} (bf16 {budget.weights_bf16_gib:.1f})"
        if hw.vram_free_gib is None:
            details.append(f"{text}: нет данных")
        else:
            measured += 1
            text += f" при свободных {hw.vram_free_gib:.1f} из {hw.vram_total_gib:.1f} GiB"
            (details if vram_need <= hw.vram_free_gib else problems).append(text)

    if budget.encoder_gib:
        text = f"RAM {budget.encoder_gib:.1f} GiB под энкодер"
        if hw.ram_free_gib is None:
            details.append(f"{text}: нет данных")
        else:
            measured += 1
            text += f" при свободных {hw.ram_free_gib:.1f} из {hw.ram_total_gib:.1f} GiB"
            (details if budget.encoder_gib <= hw.ram_free_gib else problems).append(text)

    if budget.disk_gib:
        text = f"диск {budget.disk_gib:.1f} GiB"
        if hw.disk_free_gib is None:
            details.append(f"{text}: нет данных")
        else:
            measured += 1
            text += f" при свободных {hw.disk_free_gib:.1f} из {hw.disk_total_gib:.1f} GiB"
            (details if budget.disk_gib <= hw.disk_free_gib else problems).append(text)

    if not (vram_need or budget.encoder_gib or budget.disk_gib):
        return f"нет данных: бюджет не объявлен (всё нулевое); железо {hw.summary()}"
    if not measured:
        return "нет данных: не удалось измерить VRAM/RAM/диск (нет torch/psutil/CUDA)"
    head = "НЕ ВЛЕЗАЕТ: " + "; ".join(problems) if problems else "OK"
    tail = f" ({'; '.join(details)})" if details else ""
    notes = f" — {budget.notes}" if budget.notes else ""
    return f"{head}{tail}{notes}"


__all__ = [
    "Capabilities", "ConditioningMode", "EncoderBackend", "EncoderRole", "GIB", "GenerationBackend",
    "Hardware", "ImageInputs", "REPO_ROOT", "ResourceBudget", "UpscaleBackend", "as_image_list",
    "hardware_budget_verdict", "hardware_resources",
]
