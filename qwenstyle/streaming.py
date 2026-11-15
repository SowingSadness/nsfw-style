"""Диск-стриминг весов: модель на `meta` + подгрузка весов модуля перед его forward.

Зачем это здесь: текст-энкодер 32B (Qwen3-VL-32B и heretic-производные, ~62 GiB bf16) в 63.6 GiB RAM
целиком не живёт, а 4-битное квантование меняет математику. Альтернатива — держать **в RAM только
тот модуль, который сейчас считает**, а остальные веса читать с диска по требованию:

1. модель собирается на `meta`-параметрах (`accelerate.init_empty_weights(include_buffers=False)`);
   буферы при этом **настоящие** (у Qwen3 `rotary_emb.inv_freq` считается в `__init__` и обязан жить);
2. на каждый модуль с параметрами прямого владения (`named_parameters(recurse=False)`) вешается пара
   хуков: pre-forward читает веса модуля позиционно с диска (`seek + readinto`, `qwenstyle/shards.py`),
   post-forward снимает хранилище (см. `_free_parameter`) — RAM освобождается, а дескриптор шарда
   остаётся открытым до `close()`;
   `load_text_encoder(stream=True)` дополнительно избавляет модель от meta-параметров, чтобы
   `model.device`/`get_module_execution_device()` отдавали `cpu`, а не `meta` (B1-bis);
3. `ShardedWeights` — ленивая карта `имя -> тензор`: `keys()`/`__contains__` не читают данные,
   `__getitem__` читает один тензор и кастит его в `StreamSource.dtype`, а `read_into` отдаёт байты
   тензора прямо в pinned-буфер конвейера (без промежуточной копии в pageable RAM).

Инвариант, ради которого всё написано: **это раскладка памяти, а не другая математика.** Тот же файл
с теми же dtype даёт бит-в-бит те же активации, поэтому `encoder.stream` не входит в ключ кэша
эмбеддингов (см. AGENTS.md R1 и `EncoderConfig.key_suffix`).

Почему не `param.data = tensor`: в torch 2.8 присваивание между `meta` и обычным устройством падает
(`Attempted to call variable.set_data(tensor), but variable and tensor have incompatible tensor type`),
причём в обе стороны. Поэтому параметр заменяется объектом `Parameter` того же класса — ровно так,
как это делает accelerate в `accelerate/utils/modeling.py::set_module_tensor_to_device`, на котором
стоит штатный `AlignDevicesHook(offload=True)`.

Тяжёлые зависимости (`torch`/`accelerate`/`transformers`) импортируются **лениво**, внутри функций:
модуль обязан импортироваться без torch (как `backends/encoders.py`, который `registry.discover()`
импортирует ради `models list`). `safetensors` в этом модуле не используется вовсе: заголовки шардов
и байты тензоров читает `qwenstyle/shards.py` (`open` + `seek` + `readinto`).

Сознательные ограничения:

* связанные веса (`config.tie_word_embeddings=True`) не поддерживаются — освобождение `embed_tokens`
  сломало бы `lm_head`; такой конфиг отвергается `ValueError` до сборки;
* буферы не стримятся и не освобождаются: они малы и посчитаны из конфига;
* карта `ShardedWeights` рассчитана на чтение из одного потока за раз (форвард: либо consumer,
  либо поток-продюсер конвейера — одновременно они её не трогают); сам слой чтения потокобезопасен
  (замок на `seek + readinto` в `qwenstyle/shards.py`), но кэша прочитанных тензоров нет — повторное
  чтение того же имени снова читает файл (в этом и смысл: RAM не растёт);
* post-forward хук регистрируется с `always_call=True`, поэтому освобождает веса модуля **и тогда,
  когда forward бросил исключение** (веса не остаются в RAM до конца процесса). Единственная
  лазейка — падение самого pre-хука (например, шард/имя исчезли между проверкой и чтением):
  тогда параметры этого модуля остаются материализованными до конца процесса;
* стриминг — **только инференс**: параметры подменяются и освобождаются под `torch.no_grad()`,
  а создаются с `requires_grad=False` (графу некуда расти: вес живёт один forward);
* устройств два — `cpu` (вес живёт в RAM только на время forward модуля) и `cuda` (то же, но в VRAM,
  плюс конвейер ниже). GPU-энкод ≠ CPU-энкод по округлениям, поэтому `encoder.device` входит в ключ
  кэша эмбеддингов, а `encoder.stream`/`prefetch`/`pin_memory` — нет (это раскладка, не математика);
* конвейер (`prefetch > 0`, только `device="cuda"`): фоновый поток читает слой N+1 с NVMe в
  **pinned**-буфер RAM, пока GPU считает слой N, а копии H2D идут асинхронно на отдельном
  `torch.cuda.Stream` с `Event`; глубина очереди — `prefetch`, копий вперёд — `lookahead`.
  Плечи меряются `StreamStats` (`read`/`pin`/`copy`/`compute`/`wait`/`total`, см. `summary()`).
  `prefetch=0` (умолчание) — прежний синхронный путь «читаем → `.to(device)` → forward», он же
  baseline для сравнений; на `cpu` конвейер не включается (prefetch игнорируется, с одной строкой
  в логе), результат от этого не зависит — `torch.equal` с `prefetch=0`.
* `file_dtype()` без имени отвечает только для **однородного** набора весов: при нескольких dtype
  поднимается `ValueError` (иначе потребитель получил бы dtype первого тензора и соврал бы
  про остальные).
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from functools import partial, wraps
from pathlib import Path
from typing import Any, Callable

from .shards import ShardSet, TensorInfo

__all__ = [
    "ShardedWeights",
    "StreamSource",
    "StreamStats",
    "build_streamed",
    "is_streamed",
    "load_text_encoder",
    "move_to",
    "resolve_stream_source",
    "stream_stats",
]

GIB = 1024 ** 3
_TRUTHY = frozenset({"1", "true", "yes", "on"})
#: Имена dtype из конфигов/CLI -> канонические имена `torch`.
_DTYPE_ALIASES = {
    "bf16": "bfloat16", "bfloat16": "bfloat16",
    "f16": "float16", "fp16": "float16", "half": "float16", "float16": "float16",
    "f32": "float32", "fp32": "float32", "float": "float32", "float32": "float32",
}
#: Что ищем в каталоге весов и что просим у Хаба скачать.
_WEIGHT_PATTERNS = ("*.safetensors", "*.index.json", "config.json")


# --------------------------------------------------------------------------- #
# Источник весов
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StreamSource:
    """Разрешённый набор шардов: файлы safetensors + карта имён + целевой dtype."""

    shards: tuple[Path, ...]                 # файлы safetensors (в том порядке, в каком их читать)
    weight_map: dict[str, str] | None        # имя тензора -> имя шарда; None, если шард один
    dtype: str | None                        # "bfloat16" | None = как в файле

    def summary(self) -> str:
        """Короткая строка для логов."""
        dtype = self.dtype or "как в файле"
        return f"{len(self.shards)} шард(ов), {dtype}"


def _download_hint(repo: str, revision: str | None = None,
                   local_dir: str | Path | None = None) -> str:
    """Готовая команда скачивания — вместо голого трейса (R3: сеть только явным решением)."""
    parts = [f"repo_id={repo!r}"]
    if revision:
        parts.append(f"revision={revision!r}")
    parts.append(f"allow_patterns={list(_WEIGHT_PATTERNS)!r}")
    if local_dir is not None:
        parts.append(f"local_dir={str(local_dir)!r}")
    return "snapshot_download(" + ", ".join(parts) + ")"


def _missing_message(root: Path, problems: list[str], *, repo: str | None,
                     revision: str | None) -> str:
    """Текст `RuntimeError`: чего не хватает и какой командой это докачать."""
    return (
        f"streaming: в {root} нет весов текст-энкодера: {', '.join(problems)}. "
        f"Нужны config.json и *.safetensors (мульти-шард — вместе с *.index.json); "
        f"скачать: {_download_hint(repo or '<repo>', revision=revision, local_dir=root)}"
    )


def _config_dtype(path: Path) -> str | None:
    """`dtype`/`torch_dtype` из config.json (transformers 5.x пишет `dtype`) -> имя или None."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get("dtype", data.get("torch_dtype"))
    if isinstance(value, dict):                  # бывает {"dtype": "bfloat16", ...}
        value = value.get("dtype")
    return _dtype_name(value)


def _parse_index(path: Path) -> dict[str, str]:
    """`*.index.json` -> `{имя тензора: имя шарда}` (значения — как в файле, обычно basename)."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"streaming: не читается индекс шардов {path}: {exc}") from exc
    raw = data.get("weight_map")
    if not isinstance(raw, dict) or not raw:
        raise RuntimeError(f"streaming: в {path} нет непустого weight_map — это не индекс "
                           f"safetensors-шардов")
    return {str(name): str(shard) for name, shard in raw.items()}


def _source_from_dir(root: Path, *, repo: str | None, revision: str | None) -> StreamSource:
    """Каталог весов (корень энкодера, не репозиторий) -> `StreamSource`."""
    problems: list[str] = []
    if not root.is_dir():
        problems.append("нет такого каталога")
    if not (root / "config.json").is_file():
        problems.append("нет config.json")
    if problems:
        raise RuntimeError(_missing_message(root, problems, repo=repo, revision=revision))

    weight_map: dict[str, str] | None = None
    shards: tuple[Path, ...] = ()
    index = root / "model.safetensors.index.json"
    if not index.is_file():
        indexes = sorted(root.glob("*.index.json"))
        index = indexes[0] if indexes else None
    if index is not None:
        weight_map = _parse_index(index)
        names = sorted(set(weight_map.values()))
        shards = tuple(root / Path(name).name for name in names)
    else:
        single = root / "model.safetensors"
        if single.is_file():
            shards = (single,)
        else:
            shards = tuple(sorted(root.glob("*.safetensors")))
    if not shards:
        problems.append("нет *.safetensors")
    missing = [p.name for p in shards if not p.is_file()]
    if missing:
        problems.append(f"нет шардов {missing[:5]}")
    if problems:
        raise RuntimeError(_missing_message(root, problems, repo=repo, revision=revision))
    return StreamSource(shards=shards, weight_map=weight_map,
                        dtype=_config_dtype(root / "config.json"))


def _looks_local(target: str) -> bool:
    """Похоже ли `target` на путь в файловой системе, а не на id репозитория Хаба.

    На Windows id репозитория содержит `/` (это `os.altsep`), поэтому признак локальности —
    абсолютный путь, ведущий `.`/`~`, обратный слеш или буква диска, а не разделитель вообще
    (иначе `Qwen/Qwen3-VL-32B` на POSIX считался бы путём).
    """
    text = str(target).strip()
    if not text:
        return False
    if text.startswith((".", "~")):
        return True
    path = Path(text)
    if path.is_absolute():
        return True
    return os.name == "nt" and ("\\" in text or bool(path.drive))


def _offline() -> bool:
    """`HF_HUB_OFFLINE=1` (умолчание проекта, R3) -> ни одного запроса к Хабу."""
    return str(os.environ.get("HF_HUB_OFFLINE", "")).strip().lower() in _TRUTHY


def _hub_cache_dirs() -> tuple[Path, ...]:
    """Возможные каталоги hub-кэша (`HF_HUB_CACHE`, `HF_HOME/hub`, проектное умолчание)."""
    dirs: list[Path] = []
    explicit = os.environ.get("HF_HUB_CACHE")
    if explicit:
        dirs.append(Path(explicit).expanduser())
    home = os.environ.get("HF_HOME")
    if home:
        dirs.append(Path(home).expanduser() / "hub")
    dirs.append(Path(__file__).resolve().parents[1] / "hf-cache" / "hub")
    seen: set[str] = set()
    unique: list[Path] = []
    for item in dirs:
        key = str(item)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return tuple(unique)


def _cached_snapshot(repo: str, *, revision: str | None = None,
                     subfolder: str | None = None) -> Path | None:
    """Каталог snapshot'а в локальном hf-cache — обходом раскладки, без Hub API.

    `snapshot_download(local_files_only=True)` из huggingface_hub 2.x не годится: для частично
    скачанного репозитория он бросает `IncompleteSnapshotError`, хотя нужный шард лежит на диске
    (так у `darknight9121/FLUX.2-…`: в кэше есть `text_encoder/model.safetensors`, но весь
    репозиторий — 49 GiB). Поэтому ищем так же, как `cmd/models.py`: `models--org--name/
    snapshots/<commit из refs/main>` и проверяем, что в `subfolder` есть config + веса.
    `revision` — commit; если это ветка (её нет среди каталогов `snapshots/`), берётся любой
    snapshot с весами в `subfolder`.
    """
    folder = "models--" + repo.replace("/", "--")
    for cache_dir in _hub_cache_dirs():
        repo_dir = cache_dir / folder
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        candidates: list[Path] = []
        if revision:
            candidates.append(snapshots / revision)
        else:
            ref = repo_dir / "refs" / "main"
            commit = ""
            if ref.is_file():
                try:
                    commit = ref.read_text(encoding="utf-8").strip()
                except OSError:
                    commit = ""
            if commit:
                candidates.append(snapshots / commit)
        candidates.extend(sorted(p for p in snapshots.iterdir() if p.is_dir()))
        for snapshot in candidates:
            root = snapshot / subfolder if subfolder else snapshot
            if (root / "config.json").is_file() and any(root.glob("*.safetensors")):
                return snapshot
    return None


def resolve_stream_source(target: str | Path, *, subfolder: str | None = None,
                          revision: str | None = None,
                          local_files_only: bool = True) -> StreamSource:
    """Найти шарды весов: локальный каталог (config.json + *.safetensors) или snapshot Хаба.

    Локальный каталог (`target` — существующий каталог) не требует сети: `local_files_only`
    на него не влияет. Иначе `target` — id репозитория: сначала ищем готовый snapshot прямо
    в `hf-cache` (раскладка `models--org--name/snapshots/<commit>`), и только если его нет
    и сеть разрешена (`local_files_only=False` и не `HF_HUB_OFFLINE=1`) идём в `snapshot_download`.
    Нет весов -> `RuntimeError` с перечнем того, чего не хватает, и командой `snapshot_download`.
    """
    path = Path(str(target)).expanduser()
    if path.is_dir():
        root = path / subfolder if subfolder else path
        return _source_from_dir(root, repo=None, revision=revision)
    if _looks_local(str(target)):                # путь задан, но каталога нет — не путать с repo id
        root = path / subfolder if subfolder else path
        raise RuntimeError(_missing_message(root, ["нет такого каталога"], repo=None,
                                            revision=revision))

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:                   # pragma: no cover — huggingface_hub в стеке
        raise RuntimeError(
            f"streaming: {target} не локальный каталог, а huggingface_hub не установлен; "
            f"нужен путь к готовым весам или {_download_hint(str(target), revision)}"
        ) from exc

    repo = str(target)
    cached = _cached_snapshot(repo, revision=revision, subfolder=subfolder)
    if cached is not None:                       # R3: кэш раньше сети, даже если сеть разрешена
        root = cached / subfolder if subfolder else cached
        return _source_from_dir(root, repo=repo, revision=revision)

    where = repo + (f"/{subfolder}" if subfolder else "")
    offline = _offline()
    reason = "HF_HUB_OFFLINE=1" if offline else "local_files_only=True"
    if not (local_files_only or offline):
        try:
            snapshot = snapshot_download(
                repo,
                revision=revision,
                local_files_only=False,
                allow_patterns=list(_WEIGHT_PATTERNS),
            )
        except Exception as exc:                 # noqa: BLE001 — нужна команда, а не трейс Хаба
            reason = f"{type(exc).__name__}: {exc}"
        else:
            root = Path(snapshot) / subfolder if subfolder else Path(snapshot)
            return _source_from_dir(root, repo=repo, revision=revision)

    raise RuntimeError(
        f"streaming: весов {where} нет в локальном кэше ({reason}; искали "
        f"{', '.join(str(p) for p in _hub_cache_dirs())}). Скачать: "
        f"{_download_hint(repo, revision=revision)}"
    )


# --------------------------------------------------------------------------- #
# Ленивая карта тензоров
# --------------------------------------------------------------------------- #

class ShardedWeights(Mapping[str, Any]):
    """`имя -> Tensor` поверх шардов safetensors: позиционное чтение, без загрузки файла целиком.

    Чтение делает `qwenstyle/shards.py`: заголовок шарда разбирается один раз в память, а байты
    тензора читаются по абсолютному смещению (`seek + readinto` под замком) — ровно столько,
    сколько занимает тензор, без загрузки файла целиком и без списания шарда в commit (D10).

    * `keys()`/`__contains__`/`len()` — по заголовкам шардов, данные не читаются;
    * `__getitem__` читает **один** тензор и кастит его в `dtype` источника; кэша нет, поэтому
      повторное чтение (следующий forward) снова берёт байты с диска;
    * `read_into` читает байты тензора **прямо в готовый буфер** (pinned-стейдж конвейера): одна
      копия из page cache вместо «читать в pageable-тензор -> копировать в pinned»;
    * `info()` — `TensorInfo` (файл, смещение, dtype, форма) без чтения данных;
    * `close()` закрывает дескрипторы шардов (`ShardSet.close()`); после закрытия имена, формы и
      `file_dtype()` остаются доступны, а чтение даёт `RuntimeError`.

    Потокобезопасность: замок на `seek + readinto` живёт в `ShardFile`, поэтому карта безопасна
    для чтения из разных потоков, но фактически читает один за раз (consumer или поток-продюсер
    конвейера, не оба одновременно). Ошибка в имени -> `KeyError` с примерами.
    """

    #: Текст ошибки чтения после `close()` — прежний, на него опираются тесты.
    _CLOSED_MESSAGE = "streaming: ShardedWeights уже закрыт (close()) — веса больше не читаются"

    def __init__(self, source: StreamSource) -> None:
        if not source.shards:
            raise ValueError("streaming: ShardedWeights без шардов — сначала resolve_stream_source()")
        self.source = source
        self.shards: tuple[Path, ...] = tuple(Path(p) for p in source.shards)
        self._weight_map = dict(source.weight_map) if source.weight_map else None
        self._dtype = _dtype_name(source.dtype)
        # Заголовки всех шардов разбираются здесь (`ShardSet.__init__`), данные — по требованию.
        self._set = ShardSet(self.shards, self._weight_map)
        self._lock = threading.Lock()
        self._closed = False
        # N12: dtype читается из заголовков шардов один раз на инстанс (400-750 имён иначе
        # обходились бы на каждый вызов). `_file_dtype_known` — прошла ли проверка на однородность.
        self._dtype_cache: dict[str, str | None] = {}
        self._file_dtype_cache: str | None = None
        self._file_dtype_known = False
        self._names = tuple(self._set)
        self._name_set = frozenset(self._names)

    # -- интроспекция ------------------------------------------------------ #

    @property
    def dtype(self) -> str | None:
        """Целевой dtype в имени `torch` (`"bfloat16"`) или None = как в файле."""
        return self._dtype

    @property
    def closed(self) -> bool:
        """`True` после `close()`: файлы шардов закрыты, чтение даёт `RuntimeError`."""
        return self._closed

    def file_dtype(self, name: str | None = None) -> str | None:
        """dtype тензора **в файле** (`"bfloat16"`), по заголовку safetensors — данные не читаются.

        Нужен, когда `dtype` источника = None ("как в файле"): потребителю (`_free_meta_parameters`)
        надо знать, в каком dtype придут веса, не материализуя их.

        `name` — dtype одного тензора. Без имени — dtype всего набора: обходятся заголовки всех
        шардов, набор обязан быть однородным, иначе `ValueError` с перечнем встреченных dtype и
        примерами имён (смешанный набор нельзя описать одним dtype честно). Результат кэшируется
        на инстансе: повторный вызов заголовки уже не читает.
        """
        if name is not None:
            return self._tensor_dtype(name)
        if self._file_dtype_known:
            return self._file_dtype_cache
        if not self._names:
            self._file_dtype_known = True
            return None
        seen: dict[str | None, str] = {}          # dtype -> первое имя с этим dtype
        file_dtypes = self._set.file_dtypes()     # имя -> dtype файла ("BF16"), только заголовки
        for item in self._names:
            seen.setdefault(_dtype_name(file_dtypes[item]), item)
        if len(seen) > 1:
            shown = ", ".join(f"{dtype or '<неизвестно>'} (например {item!r})"
                              for dtype, item in seen.items())
            raise ValueError(
                f"streaming: набор весов смешан по dtype (тензоров: {len(self._names)}, "
                f"шардов: {len(self.shards)}): {shown}. Одного dtype у набора нет — задайте dtype "
                f"явно (`--set encoder.dtype=…`), иначе `_free_meta_parameters` создал бы параметры "
                f"не того типа"
            )
        self._file_dtype_cache = next(iter(seen))
        self._file_dtype_known = True
        return self._file_dtype_cache

    def _tensor_dtype(self, name: str) -> str | None:
        """dtype одного тензора по заголовку шарда (данные не читаются), с кэшем на инстансе."""
        if name not in self._name_set:
            raise KeyError(self._unknown_message(name))
        if name in self._dtype_cache:
            return self._dtype_cache[name]
        # `TensorInfo.file_dtype` — имя из заголовка ("BF16"); `_dtype_name` знает эти алиасы
        dtype = _dtype_name(self._set.info(name).file_dtype)
        self._dtype_cache[name] = dtype
        return dtype

    def __repr__(self) -> str:
        state = "закрыт" if self._closed else "открыт"
        return (f"ShardedWeights({len(self._names)} тензоров, {len(self.shards)} шардов, "
                f"dtype={self._dtype or 'как в файле'}, {state})")

    # -- Mapping ----------------------------------------------------------- #

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._name_set

    def __getitem__(self, name: str) -> Any:
        """Прочитать один тензор (с кастом в `dtype` источника, если он задан)."""
        self._check_name(name)
        tensor = self._read(name)
        target = self._target_dtype()
        if target is not None and tensor.dtype != target:
            tensor = tensor.to(target)
        # `clone` для выравнивания больше не нужен: `torch.empty` в `shards.read()` уже выровнен
        # (>= 64 байт), поэтому раскладка та же, что у `from_pretrained` (бит-в-бит).
        return tensor

    def info(self, name: str) -> TensorInfo:
        """Описание тензора (файл, смещение, dtype, форма) — без чтения данных."""
        self._check_name(name)
        return self._set.info(name)

    def read_into(self, name: str, view: Any, *, at: int = 0) -> int:
        """Прочитать байты тензора **прямо в буфер** `view` (pinned-стейдж конвейера).

        Возвращает число прочитанных байт (`info(name).nbytes`). Каст тут не делается: байты
        кладутся как в файле, поэтому прямой путь конвейера допустим только при совпадении
        `dtype` источника с dtype файла (иначе — обычный `__getitem__` + `take`).
        """
        self._check_name(name)
        self._ensure_open()
        return self._set.read_into(name, view, at=at)

    # -- закрытие ---------------------------------------------------------- #

    def close(self) -> None:
        """Закрыть дескрипторы всех шардов (идемпотентно; заголовки остаются в памяти)."""
        with self._lock:
            self._closed = True
            self._set.close()

    # -- внутреннее -------------------------------------------------------- #

    def _where(self, name: str) -> str:
        """Где искать имя (для сообщений об ошибках)."""
        shard = self._weight_map.get(name) if self._weight_map else None
        if shard:
            return f"шарде {shard}"
        return "шардах " + ", ".join(p.name for p in self.shards[:3])

    def _unknown_message(self, name: str) -> str:
        shown = ", ".join(repr(item) for item in self._names[:5])
        return (f"streaming: тензора {name!r} нет в {self._where(name)} "
                f"({len(self._names)} тензоров). Примеры имён: {shown}"
                f"{' …' if len(self._names) > 5 else ''}")

    def _check_name(self, name: object) -> None:
        """Имя обязано быть строкой и быть в наборе (иначе `KeyError` с примерами)."""
        if not isinstance(name, str):
            raise KeyError(f"streaming: имя тензора должно быть строкой, получено {type(name).__name__}")
        if name not in self._name_set:
            raise KeyError(self._unknown_message(name))

    def _ensure_open(self) -> None:
        """Чтение после `close()` — `RuntimeError` с прежним текстом."""
        if self._closed:
            raise RuntimeError(self._CLOSED_MESSAGE)

    def _read(self, name: str) -> Any:
        """Прочитать тензор через `ShardSet` (каста нет — его делает `__getitem__`)."""
        self._ensure_open()
        return self._set.read(name)

    def _target_dtype(self) -> Any:
        """Целевой dtype torch из `StreamSource.dtype`; None — как в файле."""
        if self._dtype is None:
            return None
        target = getattr(_torch(), self._dtype, None)
        if target is None:
            raise ValueError(f"streaming: неизвестный dtype {self._dtype!r} "
                             f"в источнике {self.source.summary()}")
        return target


# --------------------------------------------------------------------------- #
# dtype-помощники (torch лениво)
# --------------------------------------------------------------------------- #

def _torch() -> Any:
    """Ленивый `import torch`: модуль обязан импортироваться без torch."""
    import torch

    return torch


def _dtype_name(value: Any) -> str | None:
    """`torch.bfloat16`/`"bf16"`/`None` -> `"bfloat16"`/None (`auto` = None)."""
    if value is None:
        return None
    if isinstance(value, str):
        key = value.strip().lower()
        if key in ("", "auto", "none"):
            return None
        return _DTYPE_ALIASES.get(key, key)
    return str(value).replace("torch.", "")


def _torch_dtype(value: Any) -> Any:
    """`"bfloat16"`/`torch.bfloat16`/None -> dtype torch (None = не задавать)."""
    name = _dtype_name(value)
    if name is None:
        return None
    torch = _torch()
    dtype = getattr(torch, name, None)
    if dtype is None:
        raise ValueError(f"streaming: неизвестный dtype {value!r}; ожидается torch-имя вроде bfloat16")
    return dtype


# --------------------------------------------------------------------------- #
# Статистика и конвейер (чтение / пиннинг / H2D / compute)
# --------------------------------------------------------------------------- #

@dataclass
class StreamStats:
    """Замеры плеч стриминга: чтение, пиннинг, H2D, compute и ожидание данных.

    Все времена — секунды стены, накапливаются по всем `forward()` модели. Плечи:

    * `t_read`  — `ShardedWeights` -> RAM (позиционное чтение с диска, `seek + readinto`, плюс каст
      dtype если он задан); в синхронном режиме его платит сам consumer, в конвейере — фоновый
      поток (а consumer ждёт на очереди продюсера; это ожидание **не** попадает в `t_wait`, см.
      ниже); при `pin_memory=True` байты читаются сразу в pinned-буфер, и `t_read` включает эту
      запись (отдельной копии `t_pin` тогда нет);
    * `t_pin`   — копия в pinned-буфер RAM (только `cuda` при `pin_memory=True`; на прямом пути
      «`read_into` в pinned-буфер» её нет, там пишет диск);
    * `t_copy`  — **выдача** асинхронной копии H2D (не ожидание её завершения);
    * `t_compute` — forward модулей, стена от pre-хука до post-хука;
    * `t_wait`  — только ожидание `Event` уже выданной копии на compute-стриме
      (`record.event.wait()` в `_pre_forward_pipeline`). Блокировка на очереди продюсера
      (`ready.get()` без таймаута, пока фоновый поток читает слой) сюда **не** входит, поэтому
      на конвейере `ratio_wait = t_wait / t_total` почти всегда **0.00** и индикатором I/O-bound
      не является: для стриминга смотрите долю чтения (`t_read / t_total`) и строку `[stream]
      budget` в `bench encoder` (там ожидание данных = `total - compute - copy`);
    * `t_total` — стена всего `forward()` модели. GPU-работа, выданная, но ещё не исполненная к
      моменту возврата (последние ядра), сюда не входит — её синхронизирует вызывающий.

    `tensors` — сколько тензоров прочитано с диска, `bytes_read` — сколько байт,
    `modules` — сколько модулей под стримингом (заполняется при сборке модели).
    """

    tensors: int = 0
    modules: int = 0
    bytes_read: int = 0
    t_read: float = 0.0
    t_pin: float = 0.0
    t_copy: float = 0.0
    t_compute: float = 0.0
    t_wait: float = 0.0
    t_total: float = 0.0

    def summary(self) -> str:
        """Одна строка для логов: `[stream] read=… copy=… pin=… compute=… wait=… total=…s ratio_wait=…`.

        `ratio_wait` — это `t_wait / t_total`, то есть доля **только** `Event.wait()` на compute-стриме.
        Блокировку на очереди продюсера он не видит (на конвейере обычно 0.00), поэтому для
        I/O-bound-вывода нужны `read / total` и строка `[stream] budget` в `bench encoder`.
        """
        ratio = self.t_wait / self.t_total if self.t_total > 0 else 0.0
        return (f"[stream] read={self.t_read:.2f} copy={self.t_copy:.2f} pin={self.t_pin:.2f} "
                f"compute={self.t_compute:.2f} wait={self.t_wait:.2f} total={self.t_total:.2f}s "
                f"ratio_wait={ratio:.2f}")


#: «Продюсер упал» в очереди готовых слоёв: иначе consumer ждал бы вечно.
_ABORT = object()
#: «Продюсеру пора завершаться» в очереди заявок.
_STOP = object()


class _PinnedPool:
    """Пул pinned-буферов RAM: буфер переиспользуется, а не держится по одному на каждый слой.

    Зачем пул: `.pin_memory()` — новая блокировка страниц (лишняя копия и locked-память). Если
    аллоцировать её на каждый тензор каждого forward, за прогон 8B-энкодера набирались бы
    десятки гигабайт pinned-памяти. Поэтому буферы нумеруются по dtype и переиспользуются по
    ёмкости: берём самый маленький подходящий, а когда копия H2D завершилась — возвращаем обратно.

    Два входа: `take(tensor)` — тензор уже прочитан в pageable-память, нужна копия в pinned;
    `reserve(nbytes)` — вес ещё не читался, и диск пишется **прямо в pinned-буфер**
    (`ShardFile.read_into`), без промежуточной pageable-копии.

    Если pinned-память выделить не удалось (Windows/WDDM, свободная VRAM кончилась — см.
    `_allocate`), пул **выключается навсегда для этой сессии** и стриминг идёт из обычной RAM.
    Обратно не включаемся: гарантия — один сбой на прогон, а не предупреждение на каждый слой.

    Потокобезопасность: `take`/`reserve` зовёт продюсер, `give` — consumer (в `_sweep`/`stop`),
    поэтому все операции под замком.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        self._free: dict[str, list[Any]] = {}
        self._lock = threading.Lock()
        self.bytes = 0                           # занято pinned-памяти сейчас (для логов/замеров)
        #: Резерв: True — pinned-буфер не выделился и режим деградировал до pageable RAM.
        self.fallback = False

    def take(self, tensor: Any) -> tuple[Any, Any]:
        """`tensor` (pageable) -> `(view в pinned-буфере, сам буфер)`; без пула — тот же тензор."""
        if not self.enabled:
            return tensor, None
        key = str(tensor.dtype)
        numel = tensor.numel()
        with self._lock:
            bucket = self._free.get(key) or []
            buffer = None
            best = None
            # Best-fit, а не first-fit: иначе мелкий тензор занял бы большой буфер, а следующему
            # крупному пришлось бы выделять новый — пул медленно рос бы на каждом forward.
            for index, candidate in enumerate(bucket):
                if candidate.numel() >= numel and (best is None or candidate.numel() < best[0]):
                    best = (candidate.numel(), index)
            if best is not None:
                buffer = bucket.pop(best[1])
            if buffer is None:
                # pin_memory=True сразу при аллокации: без промежуточной pageable-копии
                buffer = self._allocate(numel, tensor.dtype, tensor.element_size())
                if buffer is None:               # pinned недоступен: тот же тензор из обычной RAM
                    return tensor, None
                self.bytes += buffer.numel() * buffer.element_size()
        view = buffer[:numel].view(tensor.shape)
        view.copy_(tensor)                       # в продюсере: параллельно с compute на GPU
        return view, buffer

    def reserve(self, nbytes: int) -> tuple[Any, Any] | None:
        """`nbytes` байт pinned-памяти -> `(uint8-view, буфер)`; `None` — пинить некуда/не вышло.

        В отличие от `take`, тензора ещё нет: буфер отдаётся под позиционное чтение с диска
        (`ShardFile.read_into`), поэтому получатель — пара из **view длины `nbytes`** (в него
        пишет диск; `buffer[:nbytes]` разделяет хранилище с буфером) и **самого буфера** (его
        возвращают в `give`). Best-fit по байтам среди свободных uint8-буферов, иначе новая
        pinned-аллокация. Неудача — та же деградация, что в `take` (две строки `[stream]`,
        `enabled=False`, `fallback=True`, `bytes=0`), и `None`: вызывающий идёт обычным путём.
        """
        if not self.enabled:
            return None
        nbytes = int(nbytes)
        if nbytes <= 0:                          # pragma: no cover — вырожденный тензор
            return None
        torch = _torch()
        key = str(torch.uint8)                   # тот же ключ, что у `give` (str(buffer.dtype))
        with self._lock:
            bucket = self._free.get(key) or []
            buffer = None
            best = None
            # Best-fit, а не first-fit: та же логика, что в `take` — мелкий тензор не должен
            # занимать большой буфер и заставлять пул расти.
            for index, candidate in enumerate(bucket):
                if candidate.numel() >= nbytes and (best is None or candidate.numel() < best[0]):
                    best = (candidate.numel(), index)
            if best is not None:
                buffer = bucket.pop(best[1])
            if buffer is None:
                buffer = self._allocate(nbytes, torch.uint8, 1)
                if buffer is None:               # pinned недоступен: читаем в pageable RAM
                    return None
                self.bytes += buffer.numel() * buffer.element_size()
        return buffer[:nbytes], buffer

    def _allocate(self, numel: int, dtype: Any, element_size: int) -> Any | None:
        """`torch.empty(..., pin_memory=True)`; `None` = pinned недоступен, работаем без него.

        На Windows pinned-память CUDA — это блокировка страниц **и** маппинг в адресное
        пространство GPU (WDDM), поэтому она упирается в те же ресурсы, что и VRAM: если рядом
        уже лежат pinned-параметры группового offload (6.6 ГиБ fp8-трансформера), большой
        staging-буфер падает `CUDA error: out of memory` при свободной VRAM.

        Pinned — оптимизация транспорта (26.7 против 20.5 GB/s, AGENTS.md §4), а не условие
        корректности: числа те же бит-в-бит. Поэтому не падаем, а деградируем до pageable RAM и
        печатаем причину — режим disk-streaming продолжает работать, как требует R4.
        """
        torch = _torch()
        try:
            return torch.empty(numel, dtype=dtype, pin_memory=True)
        except (RuntimeError, MemoryError) as exc:
            # Вызывается под `self._lock` (из `take`/`reserve`): замок здесь не берём — он не
            # реентрантный.
            self.enabled = False
            self.fallback = True
            self._free.clear()
            self.bytes = 0
            print(f"[stream] pinned-буфер {numel * element_size / GIB:.2f} ГиБ не выделился "
                  f"({type(exc).__name__}: {exc})", flush=True)
            print("[stream] pin_memory выключен: staging в обычной RAM (H2D медленнее, "
                  "арифметика бит-в-бит)", flush=True)
            return None

    def give(self, buffers: Any) -> None:
        """Вернуть буферы в пул (только после завершения копии H2D из них)."""
        if not self.enabled or not buffers:
            return
        with self._lock:
            for buffer in buffers:
                self._free.setdefault(str(buffer.dtype), []).append(buffer)


@dataclass
class _StageCopy:
    """Готовая к использованию копия слоя: параметры, их H2D-тензоры и event завершения копии."""

    entries: list[tuple[str, str]]           # (локальное имя, полное имя в весах)
    tensors: list[Any]                       # те же параметры, уже на устройстве модели
    event: Any                               # torch.cuda.Event после последней копии слоя
    keep: list[Any] = field(default_factory=list)      # host-тензоры: живут до конца копии
    pooled: list[Any] = field(default_factory=list)    # pinned-буферы: вернуть в пул после копии


class _StreamSession:
    """Состояние стриминга одной модели: замеры, хуки модулей и (опционально) конвейер.

    Хуки — методы этого класса: так индекс модуля и его имена живут в одном месте, а не в
    замыканиях на каждый модуль. Синхронный путь (`prefetch=0` или `device="cpu"`) — ровно
    прежний: читаем тензор, зовём `on_load`, `.to(device)`, ставим параметр.

    Конвейер (`pipelined`): продюсер-поток читает слой по заявке из `self._requests`, пинит и кладёт
    в `self._ready` (`Queue(maxsize=prefetch)` — это и есть глубина буфера RAM); consumer в
    pre-хуке берёт свой слой, выдаёт H2D-копии на `self._copy_stream()` с `Event`, ждёт event своего
    слоя в compute-стриме и ставит параметры. Копии слоёв N+1..N+lookahead выдаются пока считается
    слой N — они перекрываются с его forward.
    """

    def __init__(self, weights: Any, stages: list[tuple[tuple[str, str], ...]], *,
                 device: str, on_load: Callable[[str, Any], None] | None, prefetch: int,
                 pin_memory: bool, lookahead: int, stats: StreamStats, pipelined: bool) -> None:
        self._weights = weights
        self._stages = stages
        self._device = device
        self._on_load = on_load
        self._prefetch = max(1, int(prefetch))
        self._lookahead = max(0, int(lookahead))
        #: на сколько слоёв вперёд просим продюсера читать: `prefetch` задаёт глубину очереди
        #: (буфер RAM), `lookahead` — сколько копий H2D выдаётся заранее.
        self._ahead = max(self._prefetch, self._lookahead + 1)
        self._stats = stats
        self.pipelined = bool(pipelined)
        self._pin = bool(pin_memory) and self.pipelined
        self._pool = _PinnedPool(self._pin)
        self._stats_lock = threading.Lock()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._requests: queue.Queue | None = None
        self._ready: queue.Queue | None = None
        self._error: BaseException | None = None
        self._pending: dict[int, list] = {}
        self._requested: set[int] = set()
        self._copies: dict[int, _StageCopy] = {}
        self._inflight: list[tuple[Any, list[Any]]] = []
        self._compute_t0: dict[int, float] = {}
        self._copy_stream_obj: Any = None

    # -- замеры -------------------------------------------------------------- #

    def add_total(self, elapsed: float) -> None:
        """Добавить стену прошедшего `forward()` (зовётся обёрткой `model.forward`)."""
        self._add("t_total", elapsed)

    def _add(self, name: str, value: float) -> None:
        with self._stats_lock:
            setattr(self._stats, name, getattr(self._stats, name) + value)

    def _note_read(self, tensor: Any, elapsed: float) -> None:
        with self._stats_lock:
            self._stats.tensors += 1
            self._stats.t_read += elapsed
            self._stats.bytes_read += tensor.numel() * tensor.element_size()

    # -- хуки модулей -------------------------------------------------------- #

    def pre_forward(self, module: Any, args: Any, index: int) -> None:
        """Pre-forward модуля: материализовать веса (синхронно или из конвейера)."""
        torch = _torch()
        with torch.no_grad():
            if self.pipelined:
                self._pre_forward_pipeline(module, index)
            else:
                self._pre_forward_sync(module, index)
        self._compute_t0[index] = time.perf_counter()

    def post_forward(self, module: Any, args: Any, output: Any, index: int) -> None:
        """Post-forward модуля: учёт compute, освобождение весов, возврат буферов в пул."""
        started = self._compute_t0.pop(index, None)
        if started is not None:
            self._add("t_compute", time.perf_counter() - started)
        with _torch().no_grad():
            for local, _ in self._stages[index]:
                # Не `to("meta")`: модель обязана и после forward сообщать `device` (B1-bis),
                # пустой параметр на том же устройстве даёт `device` и ноль байт хранилища.
                _free_parameter(module, local)
        if self.pipelined:
            self._release(index)

    def _pre_forward_sync(self, module: Any, index: int) -> None:
        """Прежний путь: читаем -> on_load -> `.to(device)` -> ставим параметр (без конвейера)."""
        for local, full in self._stages[index]:
            started = time.perf_counter()
            tensor = self._weights[full]
            self._note_read(tensor, time.perf_counter() - started)
            if self._on_load is not None:
                self._on_load(full, tensor)
            started = time.perf_counter()
            _assign_parameter(module, local, tensor.to(self._device))
            self._add("t_copy", time.perf_counter() - started)

    def _pre_forward_pipeline(self, module: Any, index: int) -> None:
        """Взять готовую копию слоя, дождаться её event'а и поставить параметры."""
        self._request(index)
        record = self._copies.get(index)
        if record is None:
            record = self._issue(index)
        if record is None:                       # pragma: no cover — продюсер обязан отдать слой
            raise RuntimeError(f"streaming: продюсер не отдал веса слоя {index} "
                               f"({len(self._stages)} модулей)")
        # Копии следующих слоёв выдаём до ожидания своего: они пойдут по копирующему стриму
        # сразу за нашей и перекроются с forward этого слоя.
        last = min(index + self._lookahead, len(self._stages) - 1)
        for offset in range(1, last - index + 1):
            if (index + offset) not in self._copies:
                self._issue(index + offset, block=False)
        started = time.perf_counter()
        record.event.wait()                       # ожидание на текущем (compute) стриме
        self._add("t_wait", time.perf_counter() - started)
        current = self._compute_stream()
        for (local, _), tensor in zip(record.entries, record.tensors):
            # record_stream: блок занят копирующим стримом — allocator не должен отдать его под
            # новое выделение (иначе память под копией переиспользуется и получим мусор).
            tensor.record_stream(current)
            _assign_parameter(module, local, tensor)

    # -- конвейер ------------------------------------------------------------ #

    def start(self) -> None:
        """Запустить продюсер на один `forward()` (идемпотентно; без конвейера — no-op)."""
        if not self.pipelined:
            return
        with self._lock:
            if self._thread is not None:
                return
            requests: queue.Queue = queue.Queue()
            ready: queue.Queue = queue.Queue(maxsize=self._prefetch)
            thread = threading.Thread(target=self._produce, args=(requests, ready),
                                      name="qwenstyle-stream-producer", daemon=True)
            self._requests = requests
            self._ready = ready
            self._pending = {}
            self._requested = set()
            self._copies = {}
            self._error = None
            self._thread = thread
        thread.start()

    def stop(self) -> None:
        """Остановить продюсер (sentinel + join), добрать буферы и отпустить копии.

        Вызывается из `finally` обёртки `model.forward`, то есть и на исключении, и на
        `KeyboardInterrupt`: утечек потоков не остаётся. Демон-флаг у потока — только страховка
        на случай, если `join` всё же не дождался (печатаем предупреждение).
        """
        if not self.pipelined:
            return
        with self._lock:
            thread = self._thread
            requests = self._requests
            ready = self._ready
            self._thread = None
            self._requests = None
            self._ready = None
        if thread is not None and requests is not None and ready is not None:
            requests.put(_STOP)
            deadline = time.monotonic() + 30.0
            while thread.is_alive():
                try:
                    # Разбираем очередь готовых слоёв: иначе продюсер, застрявший в `put`,
                    # никогда не дойдёт до sentinel'а.
                    ready.get_nowait()
                except queue.Empty:
                    thread.join(0.005)
                if time.monotonic() > deadline:
                    print("[stream] продюсер не завершился за 30 s (оставлен демоном)", flush=True)
                    break
            if not thread.is_alive():
                thread.join()
        self._pending.clear()
        self._requested.clear()
        self._compute_t0.clear()
        self._drain_buffers()

    def _drain_buffers(self) -> None:
        """Дождаться выданных копий и вернуть в пул **все** pinned-буферы.

        Сюда попадают и буферы слоёв, до которых forward не дошёл (`_pending`/`_copies` после
        исключения): их копий в полёте нет или они уже отмечены event'ом — иначе за каждый
        прерванный/короткий прогон терялся бы кусок pinned-памяти.
        """
        pending, self._inflight = self._inflight, []
        copies = list(self._copies.values())
        self._copies.clear()
        for record in copies:                    # копии слоёв, до которых forward не дошёл
            pending.append((record.event, record.pooled))
        for data in self._pending.values():      # прочитано, но копия не выдавалась
            self._pool.give([buffer for _, _, _, buffer in data if buffer is not None])
        self._pending.clear()
        for event, buffers in pending:
            try:
                event.synchronize()              # единственное место, где ждём CPU-блокировкой
            except Exception:                    # noqa: BLE001 — закрытие не должно падать
                pass
            self._pool.give(buffers)

    def _produce(self, requests: queue.Queue, ready: queue.Queue) -> None:
        """Тело продюсера: по заявке прочитать слой, запинить и положить в `ready`."""
        try:
            while True:
                index = requests.get()
                if index is _STOP:
                    return
                ready.put((index, self._load(index)))
        except BaseException as exc:              # noqa: BLE001 — об ошибке сообщаем consumer'у
            self._error = exc
            try:
                ready.put(_ABORT)
            except BaseException:                 # noqa: BLE001 — очередь уже разобрана
                pass
            return

    def _load(self, index: int) -> list[tuple[str, str, Any, Any]]:
        """Прочитать тензоры слоя (в продюсере); при `pin_memory` — прямо в pinned-буфер.

        Если каст не нужен (`dtype` источника совпадает с dtype файла), байты читаются сразу в
        pinned-буфер пула (`read_into`), без промежуточного pageable-тензора. Если нужен каст
        (например `--set encoder.dtype=float32` на bf16-файлах) — прежний путь: `__getitem__`
        кастует, `take` копирует уже правильный тензор (в pinned-буфер чужой dtype не попадёт).
        """
        data: list[tuple[str, str, Any, Any]] = []
        for local, full in self._stages[index]:
            prepared = self._load_pinned(full) if self._pin else None
            if prepared is not None:
                host, buffer = prepared
                data.append((local, full, host, buffer))
                continue
            started = time.perf_counter()
            tensor = self._weights[full]
            self._note_read(tensor, time.perf_counter() - started)
            if self._on_load is not None:
                self._on_load(full, tensor)
            if not self._pin:
                data.append((local, full, tensor, None))
                continue
            started = time.perf_counter()
            host, buffer = self._pool.take(tensor)
            self._add("t_pin", time.perf_counter() - started)
            data.append((local, full, host, buffer))
        return data

    def _load_pinned(self, full: str) -> tuple[Any, Any] | None:
        """Прочитать вес **прямо в pinned-буфер** -> `(host, buffer)`; `None` — общий путь.

        `None` возвращается, когда прямой путь неприменим: нужен каст dtype (тогда в буфер легли бы
        байты файлового dtype) или пул деградировал до pageable RAM. Счётчики те же, что на общем
        пути: `t_read` — чтение с диска (включая запись в pinned-буфер), отдельного `t_pin` нет —
        второй копии тоже нет.
        """
        weights = self._weights
        info = weights.info(full)
        target = weights.dtype
        if target is not None and target != info.dtype:
            return None                          # каст сделает `__getitem__`, потом `take`
        got = self._pool.reserve(info.nbytes)
        if got is None:                          # пул выключен/деградировал: читаем в pageable RAM
            return None
        host_u8, buffer = got
        started = time.perf_counter()
        try:
            weights.read_into(full, memoryview(host_u8.numpy()))
        except BaseException:                    # буфер не теряем: пул считает его занятым
            self._pool.give([buffer])
            raise
        # `view`, а не копия: байты уже лежат в pinned-памяти, а аллокация torch выровнена
        host = host_u8.view(getattr(_torch(), info.dtype)).view(info.shape)
        self._note_read(host, time.perf_counter() - started)
        if self._on_load is not None:
            self._on_load(full, host)
        return host, buffer

    def _request(self, index: int) -> None:
        """Попросить продюсера прочитать слои `index..index+ahead` (в возрастающем порядке)."""
        requests = self._requests
        if requests is None:                     # pre-хук раньше обёртки forward (порядок хуков)
            self.start()
            requests = self._requests
        if requests is None:                     # pragma: no cover — конвейер не запущен
            raise RuntimeError("streaming: конвейер не запущен — start() не был вызван")
        last = min(index + self._ahead, len(self._stages) - 1)
        for offset in range(index, last + 1):
            if offset in self._requested:
                continue
            self._requested.add(offset)
            requests.put(offset)

    def _take(self, index: int, *, block: bool) -> list | None:
        """Забрать готовые тензоры слоя `index` (blocking для своего, non-blocking для будущих)."""
        if index in self._pending:
            return self._pending.pop(index)
        ready = self._ready
        if ready is None:                        # pragma: no cover — конвейер не запущен
            raise RuntimeError("streaming: конвейер остановлен — данные слоя больше не читаются")
        while True:
            if block:
                item = ready.get()
            else:
                try:
                    item = ready.get_nowait()
                except queue.Empty:
                    return None
            if item is _ABORT:
                error = self._error
                raise RuntimeError(f"streaming: чтение весов в потоке-продюсере упало: "
                                   f"{type(error).__name__}: {error}") from error
            key, data = item
            if key == index:
                return data
            # Порядок исполнения модулей не совпал с порядком сборки — придержим чужой слой.
            self._pending[key] = data

    def _issue(self, index: int, *, block: bool = True) -> _StageCopy | None:
        """Выдать асинхронную копию H2D слоя `index` на копирующем стриме (+ его `Event`)."""
        data = self._take(index, block=block)
        if data is None:
            return None
        torch = _torch()
        self._sweep()                             # вернуть в пул буферы завершённых копий
        stream = self._copy_stream()
        started = time.perf_counter()
        with torch.cuda.stream(stream):
            tensors = []
            for _, _, host, _ in data:
                tensor = host.to(self._device, non_blocking=True)
                tensor.record_stream(stream)       # выделено/используется копирующим стримом
                tensors.append(tensor)
            event = torch.cuda.Event()
            event.record(stream)
        self._add("t_copy", time.perf_counter() - started)
        record = _StageCopy(
            entries=[(local, full) for local, full, _, _ in data],
            tensors=tensors,
            event=event,
            keep=[host for _, _, host, _ in data],
            pooled=[buffer for _, _, _, buffer in data if buffer is not None],
        )
        self._copies[index] = record
        return record

    def _sweep(self) -> None:
        """Вернуть в пул буферы тех копий, что уже завершились (без блокировки CPU)."""
        if not self._inflight:
            return
        rest = []
        for event, buffers in self._inflight:
            if bool(event.query()):
                self._pool.give(buffers)
            else:
                rest.append((event, buffers))
        self._inflight = rest

    def _release(self, index: int) -> None:
        """Слой отработал: его pinned-буферы можно вернуть в пул, как только копия завершится."""
        record = self._copies.pop(index, None)
        if record is None:
            return
        self._inflight.append((record.event, record.pooled))
        self._sweep()

    def _compute_stream(self) -> Any:
        """Текущий (compute) стрим устройства модели — на нём ждём `Event` и считаем forward."""
        torch = _torch()
        device = torch.device(self._device)
        index = device.index
        return torch.cuda.current_stream(None if index is None else torch.device("cuda", index))

    def _copy_stream(self) -> Any:
        """Отдельный стрим для H2D-копий (создаётся при первом использовании)."""
        if self._copy_stream_obj is None:
            self._copy_stream_obj = _torch().cuda.Stream(device=self._device)
        return self._copy_stream_obj


def _wrap_forward(model: Any, session: _StreamSession) -> None:
    """Обернуть `model.forward`: старт/остановка конвейера и `t_total` — всегда, даже на исключении.

    Почему не хук: `always_call`-хуки torch исполняются только в `except Exception:`, то есть
    `KeyboardInterrupt` их обходит, и продюсер остался бы жить. `finally` ловит всё.
    """
    original = model.forward

    @wraps(original)                             # `inspect.signature`/`__name__` остаются прежними
    def _forward(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        session.start()
        try:
            return original(*args, **kwargs)
        finally:
            session.stop()
            session.add_total(time.perf_counter() - started)

    model.forward = _forward


# --------------------------------------------------------------------------- #
# Сборка модели на meta + хуки
# --------------------------------------------------------------------------- #

def _assign_parameter(module: Any, name: str, tensor: Any, *,
                      requires_grad: bool = False) -> None:
    """Поставить тензор в `module._parameters[name]`, сменив устройство/dtype при необходимости.

    `param.data = tensor` здесь неприменим: в torch 2.8 присваивание между `meta` и обычным
    устройством бросает `RuntimeError: ... variable and tensor have incompatible tensor type`
    (в обе стороны). Рабочий путь — новый `Parameter` того же класса, как в accelerate.

    `requires_grad=False` по умолчанию: стриминг — только инференс (вес живёт один forward, графу
    некуда расти). Подмена идёт под `torch.no_grad()`, чтобы граф не тянулся в `_parameters` и не
    всплывал на следующем `backward`/`requires_grad_(True)` у чужого кода.
    """
    torch = _torch()
    old = module._parameters[name]
    with torch.no_grad():
        module._parameters[name] = type(old)(tensor, requires_grad=requires_grad)


def _free_parameter(module: Any, name: str, *, device: Any = None, dtype: Any = None) -> None:
    """Отпустить хранилище параметра: заменить его пустым тензором (0 байт).

    `device=None` / `dtype=None` — остаться на текущих (так делает post-forward хук: веса
    модуля уже материализованы нужным dtype на `device`); иначе параметр переезжает туда
    **пустым** (так снимаются оставшиеся `meta`-параметры, см. `_free_meta_parameters`).

    Формы `torch.empty(0)` достаточно: `numel() == 0`, хранилища нет, а `torch.empty(<полная форма>)`
    выделил бы все 15 GiB — ровно то, чего стриминг избегает.
    """
    param = module._parameters[name]
    torch = _torch()
    target_device = param.device if device is None else device
    target_dtype = param.dtype if dtype is None else dtype
    _assign_parameter(module, name, torch.empty(0, dtype=target_dtype, device=target_device))


def _free_meta_parameters(model: Any, device: str, *, dtype: Any = None) -> int:
    """Снять с модели оставшиеся `meta`-параметры: пустые параметры на `device` в dtype весов.

    Зачем: `transformers`/`diffusers` определяют устройство и dtype модели по её параметрам
    (`PreTrainedModel.device` / `.dtype` = `next(param.…)`, `get_module_execution_device()` = то же
    при отсутствии offload-хуков). С `meta`-параметрами они получают `meta` и уводят на него
    входы: `input_ids.to(meta)` -> `NotImplementedError: Cannot copy out of meta tensor` внутри
    `transformers/masking_utils.py` (B1-bis). Пустой параметр на `device` (0 байт хранилища) даёт
    `device=cpu` и не занимает RAM: веса, как и раньше, приезжают хуком на время forward.

    `dtype` обязателен по смыслу: модель собрана из конфига в его dtype (`float32`), а веса приезжают
    в dtype шардов; без явного dtype `model.dtype` соврал бы, и diffusers отдал бы эмбеддинги
    не того типа (проверено: промт-эмбеддинги float32 против bf16-трансформера).

    Возвращает число снятых параметров; `RuntimeError`, если остались параметры с хранилищем,
    не на `device` или не в `dtype`.
    """
    torch = _torch()
    expected = torch.device(device).type
    freed = 0
    for _, module in model.named_modules():
        for local, param in list(module.named_parameters(recurse=False)):
            if param.device.type == "meta":
                _free_parameter(module, local, device=device, dtype=dtype)
                freed += 1
    bad = [(name, tuple(param.shape), str(param.device), str(param.dtype))
           for name, param in model.named_parameters()
           if param.numel() != 0 or param.device.type != expected
           or (dtype is not None and param.dtype != dtype)]
    if bad:
        raise RuntimeError(
            f"streaming: meta-параметры не сняты (device={device}, dtype={dtype}, accelerate "
            f"{_accelerate_version()}): с хранилищем/не на {expected}/не в {dtype}: {bad[:3]}"
        )
    return freed


def _accelerate_version() -> str:
    """Версия accelerate для сообщений (ленивый импорт, модуль обязан жить без него)."""
    try:
        import accelerate
    except ImportError:                          # pragma: no cover
        return "не установлен"
    return str(getattr(accelerate, "__version__", "?"))


def _move_buffers(model: Any, device: Any) -> None:
    """Буферы остались настоящими (include_buffers=False) — переносим их на `device` вручную.

    `model.to(device)` вызывать нельзя: параметры на `meta`, и `.to()` на них падает.
    """
    if str(device) == "cpu":
        return
    for _, module in model.named_modules():
        for buffer_name, buffer in list(module.named_buffers(recurse=False)):
            if module._buffers.get(buffer_name) is None or buffer is None:
                continue
            module._buffers[buffer_name] = buffer.to(device)


def _model_label(model_cls: Any) -> str:
    return str(getattr(model_cls, "__name__", model_cls))


def _missing_params_message(model_cls: Any, missing: list[str]) -> str:
    shown = ", ".join(missing[:5])
    tail = f" и ещё {len(missing) - 5}" if len(missing) > 5 else ""
    return (f"streaming: в весах нет {len(missing)} параметров модели {_model_label(model_cls)}: "
            f"{shown}{tail}. Похоже, набор шардов не от этой модели/конфига: имена параметров "
            f"сверяются по всей модели до первого forward.")


def build_streamed(model_cls: Any, config: Any, weights: ShardedWeights, *,
                   device: str = "cpu",
                   on_load: Callable[[str, Any], None] | None = None,
                   prefetch: int = 0,
                   pin_memory: bool = True,
                   lookahead: int = 1,
                   stats: StreamStats | None = None) -> Any:
    """Собрать модель на `meta` с весами из `weights` и хуками подгрузки на каждый модуль.

    Возвращает модель в `eval()`: буферы — настоящие и на `device`, параметры — meta-тензоры без
    хранилища (веса приходят на время forward модуля). Имена параметров сверяются с `weights` сразу:
    недостающие -> `KeyError` (первые 5 имён + подсказка), связанные веса -> `ValueError`.

    Режимы (умолчания = прежнее поведение):

    * `prefetch=0` — синхронно: в pre-хуке читаем тензор, `.to(device)` и ставим параметр;
    * `prefetch>0` и `device="cuda"` — конвейер: фоновый поток читает слой и пинит его в RAM
      (`pin_memory`), а consumer в pre-хуке выдаёт асинхронную H2D-копию с отдельным
      `torch.cuda.Stream` и ждёт только `Event` своего слоя; глубина очереди — `prefetch`,
      копий вперёд — `lookahead` (они перекрываются с forward текущего слоя);
    * `prefetch>0` при `device="cpu"` — тот же синхронный путь плюс одна строка в лог: конвейер и
      pinned-стейджинг имеют смысл только для VRAM (результат от `prefetch` не зависит).

    `stats` — куда писать замеры (по умолчанию новый `StreamStats`); статистика заполняется всегда,
    в том числе в синхронном режиме, и остаётся доступной по `stream_stats(model)`.
    """
    prefetch = int(prefetch)
    lookahead = int(lookahead)
    if prefetch < 0:
        raise ValueError(f"streaming: prefetch не может быть отрицательным, получено {prefetch}")
    if lookahead < 0:
        raise ValueError(f"streaming: lookahead не может быть отрицательным, получено {lookahead}")

    from accelerate import init_empty_weights      # лениво: ~0.1 s и без torch на импорте модуля
    import accelerate

    if getattr(config, "tie_word_embeddings", False):
        raise ValueError(
            "стриминг не поддерживает связанные веса: config.tie_word_embeddings=True "
            "(embed_tokens и lm_head делят одно хранилище; освобождение одного ломает другой). "
            "Для такой модели нужна обычная загрузка (encoder.stream=false)."
        )

    with init_empty_weights(include_buffers=False):
        model = model_cls(config)

    # Полные имена — из `named_parameters()` всей модели (канонический источник), а не склейка
    # префиксов: так сопоставление с весами не зависит от того, как модули названы внутри.
    named_params = dict(model.named_parameters())
    owned: list[tuple[Any, tuple[tuple[str, str], ...]]] = []
    seen_modules: set[int] = set()
    for module_name, module in model.named_modules():
        if id(module) in seen_modules:            # модуль зарегистрирован дважды — хуки один раз
            continue
        seen_modules.add(id(module))
        direct = list(module.named_parameters(recurse=False))
        if not direct:
            continue
        entries = tuple(
            (local, f"{module_name}.{local}" if module_name else local) for local, _ in direct
        )
        owned.append((module, entries))

    have = set(weights.keys())
    missing = [name for name in named_params if name not in have]
    if missing:
        raise KeyError(_missing_params_message(model_cls, missing))

    stats = stats if stats is not None else StreamStats()
    stats.modules = len(owned)
    device_kind = str(device).split(":", 1)[0].strip().lower()
    pipelined = prefetch > 0 and device_kind == "cuda"
    if prefetch > 0 and not pipelined:
        print(f"[stream] prefetch={prefetch} игнорируется при device={device}: конвейер и "
              f"pinned-стейджинг имеют смысл только для VRAM (порядок чтения и математика те же)",
              flush=True)
    session = _StreamSession(
        weights,
        [entries for _, entries in owned],
        device=device,
        on_load=on_load,
        prefetch=prefetch,
        pin_memory=pin_memory,
        lookahead=lookahead,
        stats=stats,
        pipelined=pipelined,
    )

    hooks: list[Any] = []
    for index, (module, _entries) in enumerate(owned):
        # `partial` обязателен: без него хук получил бы index=0 и материализовал бы чужой слой
        hooks.append(module.register_forward_pre_hook(
            partial(session.pre_forward, index=index)))
        # N5: `always_call=True` — веса освобождаются даже если forward бросил исключение,
        # иначе RAM держала бы их до конца процесса.
        hooks.append(module.register_forward_hook(
            partial(session.post_forward, index=index), always_call=True))

    # N14: стриминг — только инференс. `init_empty_weights` сохраняет `requires_grad` из __init__
    # (у свежей модели это True у всех весов), поэтому флаг снимаем сами и сообщаем об этом одной
    # строкой: иначе вызывающий решил бы, что через подмену весов можно считать градиенты.
    grad_params = [name for name, param in named_params.items() if param.requires_grad]
    if grad_params:
        print(f"[stream] {len(grad_params)} параметров пришли с requires_grad=True -> снят "
              f"(стриминг только для инференса)", flush=True)
        for param in named_params.values():
            param.requires_grad_(False)

    _move_buffers(model, device)
    model.eval()

    # Здесь (на выходе сборщика) инвариант прежний: все параметры на `meta`. Пустыми на `device`
    # их делает `load_text_encoder` (`_free_meta_parameters`) — единственный потребитель,
    # отдающий модель в transformers/diffusers; см. докстринг `load_text_encoder`.
    bad_params = [name for name, param in model.named_parameters() if param.device.type != "meta"]
    bad_buffers = [name for name, buffer in model.named_buffers() if buffer.device.type == "meta"]
    if bad_params or bad_buffers:
        raise RuntimeError(
            f"streaming: сборка на meta не удалась (accelerate {getattr(accelerate, '__version__', '?')}: "
            f"init_empty_weights(include_buffers=False)). Параметры не на meta: {bad_params[:3]}; "
            f"буферы на meta: {bad_buffers[:3]}. Подходящая версия accelerate — 1.15+"
        )

    # держим ссылки: хуки должны жить вместе с моделью, `weights` — чтобы файлы не закрылись раньше
    model._stream_hooks = hooks
    model._stream_weights = weights
    model._stream_session = session
    model._stream_stats = stats
    # Старт/остановка конвейера и `t_total` — здесь, а не хуком: `finally` держит и KeyboardInterrupt.
    _wrap_forward(model, session)

    shards = tuple(getattr(weights, "shards", ()) or ())
    size = 0
    for path in shards:
        try:
            size += Path(path).stat().st_size
        except OSError:
            pass
    extra = ""
    if pipelined:
        extra = (f", prefetch={prefetch}, lookahead={lookahead}, "
                 f"pin_memory={int(bool(pin_memory))}")
    print(f"[stream] {len(owned)} модулей, {len(named_params)} тензоров, "
          f"{size / GIB:.2f} GiB из {len(shards)} шардов (device={device}{extra})", flush=True)
    return model


# --------------------------------------------------------------------------- #
# Взаимодействие с моделью, собранной стримингом
# --------------------------------------------------------------------------- #

def is_streamed(model: Any) -> bool:
    """`True`, если модель собрана `build_streamed` (признак — атрибут `_stream_hooks`).

    Нужен потребителям, чтобы отличить стриминговую модель от обычной: у первой параметры без
    хранилища (`meta` до `load_text_encoder`, пустые на `device` после; веса приходят на время
    forward модуля), поэтому с ней нельзя делать `.to()`/`state_dict()`.
    """
    return getattr(model, "_stream_hooks", None) is not None


def stream_stats(model: Any) -> StreamStats | None:
    """Замеры стриминга модели (`None`, если модель не собрана `build_streamed`).

    Объект тот же, что передавали в `build_streamed(stats=…)` (или созданный там), и он продолжает
    накапливаться по всем `forward()`: читать его имеет смысл после `torch.cuda.synchronize()`,
    если нужны полные GPU-времена.
    """
    stats = getattr(model, "_stream_stats", None)
    return stats if isinstance(stats, StreamStats) else None


def move_to(model: Any, device: "str | torch.device") -> None:
    """Переместить модель на `device`; для стриминговой модели (`is_streamed`) — no-op.

    No-op, потому что стриминговая модель уже размещена своими хуками: параметры либо на `meta`
    (сразу после `build_streamed`), либо пустые на `device` (после `load_text_encoder`), буферы — на
    том `device`, который передали в `build_streamed`, а веса приезжают на время forward модуля.
    `nn.Module.to()` на `meta`-параметрах падает (`NotImplementedError: Cannot copy out of meta
    tensor`), а на пустых параметрах он бы «сработал» и создал ложное впечатление, что веса
    переехали, — поэтому вызов остаётся no-op в обоих состояниях.

    Используется там, где пайплайн раскладывает компоненты по устройствам и получает энкодер,
    загруженный либо обычным путём, либо через `load_text_encoder(stream=True)`:
    `backends/flux2.py::_place` и `backends/qwen_image21.py::prepare` (два места).
    """
    if is_streamed(model):
        return
    model.to(device)


# --------------------------------------------------------------------------- #
# Единая точка загрузки текст-энкодера
# --------------------------------------------------------------------------- #

def load_text_encoder(model_cls: Any, target: str | Path, *, subfolder: str | None = None,
                      revision: str | None = None, dtype: str | None = "bfloat16",
                      local_files_only: bool = True, device: str = "cpu",
                      stream: bool = False, prefetch: int = 0, pin_memory: bool = True,
                      lookahead: int = 1, stats: StreamStats | None = None) -> Any:
    """Загрузить текст-энкодер: обычным `from_pretrained` или диск-стримингом.

    `stream=False` — ровно прежнее поведение: `model_cls.from_pretrained(target, ...)` без
    `.to()`/`.eval()` (это делает вызывающий, у него свой порядок освобождения памяти).
    `stream=True` — `resolve_stream_source` + `AutoConfig` + `build_streamed`: веса читаются из
    safetensors по модулям, модель уже в `eval()` на `device`, а оставшиеся `meta`-параметры
    заменяются пустыми на `device` в dtype весов (`_free_meta_parameters`). Режим — только инференс.

    `device` — `"cpu"` (веса живут в RAM только на время forward модуля) или `"cuda"` (то же в VRAM:
    веса не влезают в неё целиком, но влезает один модуль). Для GPU конвейер включается `prefetch>0`:
    фоновое чтение с диска + pinned-стейджинг + асинхронная H2D-копия перекрываются с forward
    (см. `build_streamed`); при `prefetch=0` GPU работает тем же синхронным путём, что и CPU.
    Другие устройства отвергаются `ValueError` до чтения шардов. CPU-путь от этих флагов не зависит
    (результат бит-в-бит тот же — это раскладка памяти, а не математика).

    Если после `resolve_stream_source` что-то падает, дескрипторы шардов закрываются
    (`ShardedWeights.close()`), чтобы они не пережили исключение.

    Замена `meta` -> пустой параметр делается **здесь**, а не в `build_streamed`, сознательно:
    `build_streamed` — низкоуровневый сборщик, его контракт («все параметры на `meta`») закреплён
    тестом `tests/test_streaming.py`, а модель, готовая к transformers/diffusers (и отдающая
    `device=cpu`), — результат именно этой функции, единственной точки загрузки.
    """
    if not stream:
        kwargs: dict[str, Any] = {"local_files_only": bool(local_files_only)}
        target_dtype = _torch_dtype(dtype)
        if target_dtype is not None:
            kwargs["dtype"] = target_dtype
        if subfolder:
            kwargs["subfolder"] = subfolder
        if revision:
            kwargs["revision"] = revision
        return model_cls.from_pretrained(str(target), **kwargs)

    if str(device).split(":", 1)[0].strip().lower() not in ("cpu", "cuda"):
        raise ValueError(f"streaming поддерживает device='cpu' и device='cuda', получено {device!r}")
    device_kind = str(device).split(":", 1)[0].strip().lower()
    if device_kind == "cuda" and not _torch().cuda.is_available():
        raise ValueError("streaming: device='cuda', но CUDA недоступна — используйте device='cpu'")

    from transformers import AutoConfig              # лениво: импорт модуля без transformers

    source = resolve_stream_source(target, subfolder=subfolder, revision=revision,
                                   local_files_only=local_files_only)
    wanted = _dtype_name(dtype)
    if wanted is not None and wanted != source.dtype:
        # dtype из вызова главнее config.json: from_pretrained(dtype=...) кастует так же
        source = replace(source, dtype=wanted)
    # Конфиг читаем из того же каталога, где шарды: у частично скачанного репозитория
    # `AutoConfig.from_pretrained(<repo id>, subfolder=...)` в оффлайне не находит файл
    # (hf_hub 2.x ищет его через trees/blobs), хотя config.json рядом с весами лежит.
    config_dir = source.shards[0].parent
    config = AutoConfig.from_pretrained(str(config_dir), local_files_only=True)
    weights = ShardedWeights(source)
    try:
        # N11: любой сбой ниже (в т.ч. в _free_meta_parameters) обязан закрыть дескрипторы шардов —
        # иначе они живут до конца процесса и файлы нельзя удалить/перезаписать.
        model = build_streamed(model_cls, config, weights, device=device, prefetch=prefetch,
                              pin_memory=pin_memory, lookahead=lookahead, stats=stats)
        # B1-bis: без этого `model.device` и `get_module_execution_device()` отдают `meta`, и
        # diffusers уводит `input_ids`/`attention_mask` на meta -> `Cannot copy out of meta tensor`.
        # dtype — тот, в котором веса реально придут (цель каста или dtype шардов), а не dtype конфига.
        weight_dtype = _torch_dtype(weights.dtype or weights.file_dtype())
        freed = _free_meta_parameters(model, device, dtype=weight_dtype)
        # GPU: веса живут в VRAM только на время forward модуля; RAM держит один слой + очередь.
        where = "VRAM" if device_kind == "cuda" else "RAM"
        print(f"[te] streaming: {config_dir} — веса читаются по слоям с диска "
              f"(dtype={weights.dtype or 'как в файле'}, {where} = только текущий модуль); "
              f"{freed} параметров без хранилища на {device} "
              f"(device модели = {device}, dtype модели = {_dtype_name(weight_dtype) or '?'})",
              flush=True)
    except BaseException:
        weights.close()
        raise
    return model
