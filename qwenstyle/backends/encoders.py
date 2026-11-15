"""Текст-энкодеры: штатный Qwen3VL-8B (`base`), Heretic-1.3.0 (`heretic`) и Qwen3-VL-32B
heretic для MiniMax-H3-Image (`h3qwen32-heretic`).

Шаг 5 плана `qwenstyle-framework`: выбор и загрузка text-encoder'а вынесены из `run_style.py`
в два бекенда с объявленными `cache_tag` и `requires_vision`.

Почему это бекенды, а не хардкод в пайплайне:
* `cache_tag` — часть ключа кэша эмбеддингов (AGENTS.md R1): `EncoderConfig.key_suffix` даёт
  `""` для `base` и `_te-<tag>` иначе. Ошибка в теге = тихая подмена conditioning одного
  энкодера другим (D7), поэтому тег объявлен рядом с весами, а не в месте сборки ключа.
* `requires_vision` — conditioning Qwen-Image-2.1 кодируется VLM с vision-башней (картинка
  уходит в энкодер как image-токены, отсюда `img_mask`). Это признак для бекенда генерации:
  каркас сразу поддерживает модели и с vision-башней, и без неё.

Чего здесь СОЗНАТЕЛЬНО нет:
* никакой загрузки весов на импорте — `discover()` обязан работать без torch/transformers и без
  Heretic-весов (16.7 GiB могут отсутствовать), поэтому тяжёлые импорты только внутри `load()`;
* реального `encode()`: кодирование промта+картинки делает бекенд генерации, он владеет
  processor/tokenizer и подменяет `pipe.encode_prompt` (`qwen_image21.py`). См. `encode()` ниже.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping

from ..config import AUTO_ENCODER_PREFETCH, encoder_device_canon
from ..registry import register_encoder

if TYPE_CHECKING:                                   # только типы: не тянем PIL ради `models list`
    from PIL import Image

#: Корень репозитория: относительные `local_dir` считаются от него, а не от текущего каталога.
REPO_ROOT = Path(__file__).resolve().parents[2]
_TRUTHY = frozenset({"1", "true", "yes", "on"})
#: Имя dtype в конфиге -> имя атрибута `torch`; `auto` = оставить как в конфиге модели.
_DTYPES: dict[str, str | None] = {
    "bfloat16": "bfloat16", "bf16": "bfloat16",
    "float16": "float16", "fp16": "float16", "half": "float16",
    "float32": "float32", "fp32": "float32", "float": "float32",
    "auto": None,
}
#: Файлы весов в локальном каталоге (D7: 4 шарда `model-0000N-of-00004.safetensors`).
_WEIGHTS_GLOBS = ("model-*.safetensors", "model.safetensors")
#: Ключи `.pt`-кэша эмбеддингов (формат cache/embeds/embeds_*.pt, см. qwenstyle/cache.py).
_EMBED_KEYS = ("embeds", "mask", "img_mask")

#: Сигнатура пользовательского кодировщика: `encode_fn(handle, prompt, image) -> dict`.
EncodeFn = Callable[[Any, str, "Image.Image | None"], Mapping[str, Any]]


# --------------------------------------------------------------------------- #
# Вспомогательное
# --------------------------------------------------------------------------- #

def _offline() -> bool:
    """`HF_HUB_OFFLINE=1` (значение по умолчанию, qwenstyle/__init__.py) -> только локальный кэш."""
    return str(os.environ.get("HF_HUB_OFFLINE", "")).strip().lower() in _TRUTHY


def _pick(obj: Any, attr: str, default: Any) -> Any:
    """Поле из датакласса-конфига, если оно задано; `None`/пустая строка = «взять умолчание»."""
    value = getattr(obj, attr, None)
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return value


def _torch_dtype(name: str) -> Any:
    """`"bfloat16"` -> `torch.bfloat16`; `"auto"` -> None (пусть transformers решит по config.json)."""
    key = str(name or "auto").strip().lower()
    if key not in _DTYPES:
        raise ValueError(f"неизвестный dtype {name!r} для текст-энкодера; "
                         f"допустимые: {', '.join(sorted(_DTYPES))}")
    torch_name = _DTYPES[key]
    if torch_name is None:
        return None
    import torch                                     # лениво: импорт модуля не тянет torch

    return getattr(torch, torch_name)


def _local_dir_problem(path: Path) -> str | None:
    """Чего не хватает в локальном каталоге весов; `None` = каталог полный."""
    if not path.is_dir():
        return "каталога нет"
    if not (path / "config.json").is_file():
        return "нет config.json"
    if not any(p.is_file() for glob in _WEIGHTS_GLOBS for p in path.glob(glob)):
        return "нет весов model-*.safetensors"
    return None


@dataclass(frozen=True)
class _Source:
    """Разрешённый источник весов: локальный каталог (D7) или репозиторий Hub."""

    target: str                                   # что передать в from_pretrained
    repo: str
    revision: str | None
    subfolder: str | None
    local_dir: Path | None
    dtype: str
    device: str
    local_files_only: bool

    def summary(self) -> str:
        """Короткая строка для логов/диагностики."""
        if self.local_dir is not None:
            return f"{self.local_dir} ({self.dtype}, {self.device.upper()})"
        where = self.repo + (f"/{self.subfolder}" if self.subfolder else "")
        return f"{where} ({self.dtype}, {self.device.upper()})"


# --------------------------------------------------------------------------- #
# Общий предок: Qwen3VL-8B в bf16 на CPU
# --------------------------------------------------------------------------- #

class _Qwen3VlEncoder:
    """Общая часть для `Qwen3VLForConditionalGeneration`: разрешение источника + `load()`.

    Наследник объявляет `name`/`cache_tag`/`requires_vision` и значения по умолчанию
    (repo/subfolder/revision/local_dir); `EncoderConfig` из `RunConfig` может их переопределить.
    """

    name: ClassVar[str] = ""
    cache_tag: ClassVar[str] = ""
    requires_vision: ClassVar[bool] = True
    #: Репозитории, к которым относится бекенд (диагностика/`models probe`).
    repos: ClassVar[tuple[str, ...]] = ()
    # --- умолчания, которые переопределяет EncoderConfig ------------------- #
    repo: ClassVar[str] = ""
    subfolder: ClassVar[str | None] = None
    revision: ClassVar[str | None] = None
    local_dir: ClassVar[str | None] = None        # задан -> грузим с диска, без Hub (D7)
    #: Веса живут только в `local_dir`: `repo`/`revision` — подсказка для скачивания, а не выбор
    #: модели. Иначе базовый пресет (там `encoder.repo` = Qwen/Qwen-Image-2.1) после
    #: `--set encoder.name=heretic` подставил бы Heretic-адаптеру чужой репозиторий.
    local_weights: ClassVar[bool] = False
    #: Ожидаемый RSS процесса после загрузки (AGENTS.md §4: энкодер живёт на CPU), GiB.
    memory_gib: ClassVar[float | None] = None
    notes: ClassVar[str] = ""

    # -- метаданные -------------------------------------------------------- #

    def resolve(self, cfg: Any = None) -> _Source:
        """`RunConfig`/`EncoderConfig` (или `None`) -> конкретный источник весов.

        `repo`/`revision`/`local_dir` берутся из конфига только если конфиг про этот же энкодер:
        `--set encoder.name=heretic` не переписывает `encoder.repo` (в base.yaml он базовый), и
        подставлять оттуда `Qwen/Qwen-Image-2.1` Heretic-адаптеру было бы ошибкой.
        """
        enc = getattr(cfg, "encoder", cfg)
        own = str(_pick(enc, "name", self.name)) == self.name
        trust_repo = own and not self.local_weights
        repo = _pick(enc, "repo", self.repo) if trust_repo else self.repo
        revision = _pick(enc, "revision", self.revision) if trust_repo else self.revision
        local_dir = _pick(enc, "local_dir", self.local_dir) if own else self.local_dir

        if own:
            tag = str(_pick(enc, "cache_tag", self.cache_tag))
            if tag != self.cache_tag:
                print(f"[te] внимание: у энкодера {self.name} cache_tag={tag!r}, ожидается "
                      f"{self.cache_tag!r}: ключ кэша эмбеддингов будет другим (R1)", flush=True)

        local_path: Path | None = None
        if local_dir:
            path = Path(str(local_dir)).expanduser()
            local_path = path if path.is_absolute() else REPO_ROOT / path

        return _Source(
            target=str(local_path) if local_path is not None else str(repo),
            repo=str(repo),
            revision=revision,
            # из локального каталога subfolder не нужен, веса лежат в его корне
            subfolder=None if local_path is not None else self.subfolder,
            local_dir=local_path,
            dtype=str(_pick(enc, "dtype", "bfloat16")),
            # `auto` -> `cuda`/`cpu`: на ключ кэша это попадает через тот же канон, а здесь — в устройство
            device=encoder_device_canon(_pick(enc, "device", "cpu")),
            local_files_only=local_path is not None or _offline(),
        )

    def status(self, cfg: Any = None) -> tuple[bool, str]:
        """(доступен?, источник/причина) — без загрузки весов (для `--dry-run` и логов)."""
        src = self.resolve(cfg)
        if src.local_dir is None:
            return True, src.summary()
        problem = _local_dir_problem(src.local_dir)
        if problem:
            return False, f"{src.local_dir}: {problem}"
        return True, src.summary()

    def missing_message(self, src: _Source, problem: str) -> str:
        """Текст `RuntimeError` с готовой командой скачивания (D7)."""
        command = (f"snapshot_download(repo_id='{src.repo}'"
                   + (f", revision='{src.revision}'" if src.revision else "")
                   + ", allow_patterns=['model-*.safetensors', '*.index.json', 'config.json', "
                     "'generation_config.json']"
                   + (f", local_dir='{src.local_dir}'" if src.local_dir is not None else "")
                   + ")")
        size = f"{self.memory_gib:.1f} GiB" if self.memory_gib else "тяжёлые веса"
        return (f"{self.name}-энкодер не найден в {src.local_dir} ({problem}; {size}, "
                f"см. AGENTS.md D7): {command}")

    # -- загрузка ---------------------------------------------------------- #

    def load(self, cfg: Any = None) -> Any:
        """`Qwen3VLForConditionalGeneration` в `cfg.encoder.dtype` на `cfg.encoder.device`.

        Ленивый импорт transformers: импорт модуля энкодеров не должен платить за torch.
        Локальный каталог без весов -> `RuntimeError` с командой `snapshot_download`, а не
        невнятная ошибка transformers/Hub.

        `cfg.encoder.stream=True` -> веса читаются по слоям с диска (`qwenstyle/streaming.py`):
        модель приходит уже `eval()` на устройстве с пустыми параметрами (0 байт, `device=cpu`,
        dtype весов) — веса приезжают хуками на время forward, поэтому `.to()`/`.eval()` не нужны.
        `prefetch`/`pin_memory`/`lookahead` берутся из `cfg.encoder` и включают конвейер H2D
        только при `device=cuda` (`prefetch=0` = прежний синхронный путь).

        `cfg.encoder=auto` разрешается здесь (`EncoderConfig.resolved()`): на машине с CUDA это
        `cuda` + `stream=True` + `prefetch=4`, поэтому 16.7 GiB heretic-энкода не грузятся в RAM
        и не уводятся в VRAM целиком. Если устройство всё-таки `cuda`, а `stream=False` задан явно,
        стриминг включается принудительно: обычный `from_pretrained(...).to("cuda")` такой энкодер
        в 16 GiB не поднимет.
        """
        src = self.resolve(cfg)
        enc = getattr(cfg, "encoder", cfg)
        resolved = getattr(enc, "resolved", None)
        if callable(resolved):
            enc = resolved()
        stream = bool(_pick(enc, "stream", False))
        prefetch = int(_pick(enc, "prefetch", 0) or 0)
        # `cfg.encoder_pin_memory` — это уже разрешённое поле `encoder.pin_memory` (pin включён
        # и при `offload=group`); страховка от упавшей аллокации живёт в `streaming._PinnedPool`.
        pin_memory = bool(getattr(cfg, "encoder_pin_memory", _pick(enc, "pin_memory", True)))
        lookahead = int(_pick(enc, "lookahead", 1) or 1)
        if not pin_memory:
            print(f"[te] {self.name}: pin_memory отключён пользователем "
                  f"(encoder.pin_memory=false)", flush=True)
        if src.device != "cpu" and not stream:
            print(f"[te] {self.name}: device={src.device} -> включаю стриминг весов "
                  f"(без него энкодер не влезает в RAM/VRAM целиком)", flush=True)
            stream = True
            if not prefetch:
                prefetch = AUTO_ENCODER_PREFETCH
        if src.local_dir is not None:
            problem = _local_dir_problem(src.local_dir)
            if problem:
                raise RuntimeError(self.missing_message(src, problem))

        from transformers import Qwen3VLForConditionalGeneration   # лениво, ~1-2 s и ~0.5 GiB RSS

        if stream:
            from .. import streaming                      # лениво: без torch при `models list`

            # раскладка памяти, а не другая математика: ключ кэша эмбеддингов тот же
            return streaming.load_text_encoder(
                Qwen3VLForConditionalGeneration,
                src.target,
                subfolder=src.subfolder,
                revision=src.revision,
                dtype=src.dtype,
                local_files_only=src.local_files_only,
                device=src.device,
                stream=True,
                prefetch=prefetch,
                pin_memory=pin_memory,
                lookahead=lookahead,
            )

        kwargs: dict[str, Any] = {"local_files_only": src.local_files_only}
        dtype = _torch_dtype(src.dtype)
        if dtype is not None:
            # в transformers 5.x актуальный аргумент — dtype; torch_dtype оставлен для BC
            kwargs["dtype"] = dtype
        if src.subfolder:
            kwargs["subfolder"] = src.subfolder
        if src.revision:
            kwargs["revision"] = src.revision

        model = Qwen3VLForConditionalGeneration.from_pretrained(src.target, **kwargs)
        if src.device and src.device != "cpu":
            model.to(src.device)
        model.eval()
        return model

    # -- кодирование -------------------------------------------------------- #

    def encode(self, cfg: Any, handle: Any, prompt: str, image: "Image.Image | None" = None,
               *, encode_fn: EncodeFn | None = None) -> dict[str, Any]:
        """`{"embeds","mask","img_mask"}` — через `encode_fn`, иначе `NotImplementedError`.

        Кодирование промта и картинки не работа этого класса: процессором/токенизатором владеет
        пайплайн генерации (`qwen_image21.py`), он же подменяет `pipe.encode_prompt` и пишет
        `cache/embeds/embeds_*.pt`. Второй путь кодирования разошёлся бы с ним (другой prompt template
        или `img_mask`) и испортил бы кэш при одинаковом имени файла. Поэтому здесь либо явный
        `encode_fn(handle, prompt, image)` от вызывающего (переиспользует уже загруженный handle),
        либо честная ошибка с объяснением, где кодирование живёт.
        """
        if encode_fn is None:
            raise NotImplementedError(
                f"{self.name}.encode(): кодирование выполняет бекенд генерации — он владеет "
                "processor/tokenizer и подменяет pipe.encode_prompt "
                "(см. qwenstyle/backends/qwen_image21.py). Энкодер отдаёт только загруженный "
                "компонент: load(cfg) -> QwenImage21Pipeline.from_pretrained(..., "
                "text_encoder=<handle>). Если расчёт нужен здесь, передайте "
                "encode_fn(handle, prompt, image) -> {'embeds','mask','img_mask'}."
            )
        payload = encode_fn(handle, prompt, image)
        if not isinstance(payload, Mapping):
            raise TypeError(f"encode_fn вернул {type(payload).__name__}, ожидался dict вида "
                            f"{{{', '.join(repr(k) for k in _EMBED_KEYS)}}}")
        missing = [key for key in _EMBED_KEYS if key not in payload]
        if missing:
            raise ValueError(f"encode_fn вернул dict без ключей {', '.join(missing)} "
                             f"(формат cache/embeds/embeds_*.pt: {', '.join(_EMBED_KEYS)})")
        return dict(payload)


# --------------------------------------------------------------------------- #
# Бекенды
# --------------------------------------------------------------------------- #

@register_encoder
class BaseEncoder(_Qwen3VlEncoder):
    """Штатный Qwen3VL-8B из пайплайна: `Qwen/Qwen-Image-2.1`, subfolder `text_encoder`."""

    name = "base"
    cache_tag = "base"                 # key_suffix -> "": имя кэша как в run_style.py
    requires_vision = True
    repos = ("Qwen/Qwen-Image-2.1", "Qwen/Qwen-Image-2.1-Turbo")   # turbo-база: тот же энкодер
    repo = "Qwen/Qwen-Image-2.1"
    subfolder = "text_encoder"
    memory_gib = 17.5
    notes = "bf16 на CPU; грузится только при промахе кэша эмбеддингов"


@register_encoder
class HereticEncoder(_Qwen3VlEncoder):
    """Abliterated Qwen3-VL-8B Heretic-1.3.0 — conditioning как в трейне LoRA (AGENTS.md D7).

    Отличается от штатного MLP `down_proj` слоёв 15 и 30; processor/tokenizer берутся базовые —
    так предписывает карточка LoRA. Веса лежат в `cache/models/weights/heretic-te` (16.7 GiB, 4 шарда).
    """

    name = "heretic"
    cache_tag = "heretic"              # key_suffix -> "_te-heretic" (записи энкодеров не смешать)
    requires_vision = True
    repos = ("DreamFast/Qwen3-VL-8B-Heretic-1.3.0",)
    repo = "DreamFast/Qwen3-VL-8B-Heretic-1.3.0"
    revision = "28dc0129b4c7c16304bc2ed3697c9437ae8ac2f3"
    local_dir = "cache/models/weights/heretic-te"
    local_weights = True             # веса только в local_dir; repo — лишь подсказка для скачивания
    memory_gib = 16.7
    notes = "только локальные веса (D7); processor/tokenizer — базовые"


@register_encoder
class H3HereticEncoder(_Qwen3VlEncoder):
    """Qwen3-VL-**32B** heretic — conditioning MiniMax-H3-Image (шаг 5 плана `minimax-h3-image`).

    Почему отдельный бекенд, а не `--set encoder.repo=…`: H3-Image читает hidden state после
    50-го decoder layer и ждёт `text_dim` = **5120**. У всех энкодеров проекта (Qwen3VL-8B,
    Heretic-8B, FLUX.2 `Qwen3ForCausalLM`) hidden 4096 — они не подходят по ширине, поэтому
    молчаливая подстановка недопустима (R1: другой энкодер = другие эмбеддинги).

    Своих умолчаний у адаптера фактически три: `repo`/`revision` (62.1 GiB bf16 в `hf-cache`,
    полный HF-layout), `memory_gib` и `local_dir`. Загрузка, стриминг (`stream`/`prefetch`),
    dtype и device — общий код `_Qwen3VlEncoder.load()`; processor/tokenizer для H3-Image берутся
    из его собственного репозитория (их держит пайплайн), здесь только веса.

    Почему веса читаются из `cache/models/weights/h3qwen32-heretic`, а не из `hf-cache` напрямую
    (как heretic-8B, D7): в hf-cache лежит шард `model-00001-of-00002.safetensors` на **46.5 GiB**,
    а `safetensors` 0.8.0 (последняя версия) на Windows падает с access violation на любом
    `get_tensor` из файла ≥ ~40 GiB (проверено синтетической разрежённой матрицей: 39 GiB — ок,
    40 GiB — segfault; на шардах 16.8 GiB ошибки нет). Поэтому снапшот пересобран в 3 шарда
    по ~24 GiB побайтовым копированием (`torch.equal`-эквивалент: байты те же, заголовки новые),
    оригинал в `hf-cache` не тронут — при следующей загрузке шарды можно пересобрать заново.
    """

    name = "h3qwen32-heretic"
    cache_tag = "h3qwen32-heretic"     # key_suffix -> "_te-h3qwen32-heretic" (записи не смешать)
    requires_vision = True
    repos = ("llmfan46/Qwen3-VL-32B-Instruct-ultra-uncensored-heretic",)
    repo = "llmfan46/Qwen3-VL-32B-Instruct-ultra-uncensored-heretic"
    revision = "c44b949b30d111666a5ed9851c5cd633ed39b070"
    local_dir = "cache/models/weights/h3qwen32-heretic"   # пересобранные шарды < 40 GiB (см. выше)
    local_weights = True             # веса только в local_dir; repo/revision — для текста ошибки
    memory_gib = 62.1
    notes = ("Qwen3VLForConditionalGeneration 32B, hidden 5120, 64 слоя; 62.1 GiB bf16 "
             "в 63.6 GiB RAM не влезает -> stream=True обязателен; шарды пересобраны до <40 GiB "
             "(safetensors 0.8.0 падает на файлах ≥40 GiB на Windows)")


__all__ = ["BaseEncoder", "EncodeFn", "H3HereticEncoder", "HereticEncoder"]
