"""Позиционное чтение весов из шардов safetensors: `seek + readinto`, без mmap и без `safetensors`.

Зачем это здесь (замеры этой машины, RTX 4080 / 63.6 GiB RAM, лимит коммита 111.5 GiB):

* `safetensors.safe_open` списывает в commit **размер файла целиком при нуле прочитанных тензоров**:
  **+23.88 / +23.95 / +14.54 GiB** на трёх шардах энкодера 32B (Σ **+62.37 GiB**) при rss 0.4 GiB.
  Базовый commit фазы ~41.5 GiB + 62.4 GiB упирается в лимит 111.5 GiB, отсюда 4 падения из 8
  на холодном энкоде (долг D10 в AGENTS.md §7). `ShardSet` — **+0.00 GiB** на каждом шарде
  (33 KiB заголовка + дескриптор), `read()` прибавляет ровно байты тензора. Замеры —
  `out/bench/shards-readinto.md`;
* питоновский `mmap.mmap` того же файла стоит **+0.09 GiB**, `open()` — **+0.00 GiB**. То есть дорог
  не mmap как таковой, а `safetensors`-меммаппинг: дескриптор файла бесплатен, и «когда закрывать
  шард» перестаёт быть проблемой;
* тензоры в safetensors адресуются точно: заголовок шарда 23.79 GiB — это **33.0 KiB JSON,
  256 тензоров, открытие 2.3–3.3 ms**, абсолютное смещение данных =
  `8 + len(header) + data_offsets[0]`. Читать 24 ГБ целиком не нужно никогда;
* скорость (измерено, `out/bench/shards-readinto.md`). Холодный проход упирается в диск и не зависит
  от способа чтения: новый путь **2.58** GiB/s против **2.53** GiB/s у `safe_open`+`clone`. Тёплый
  (page cache) — новый **6.5–8.1** GiB/s против **19.45–21.24** GiB/s у `old`: mmap+`clone` — это
  один memcpy из page cache, а `readinto` — копия через системный вызов. Целый слой (11 тензоров,
  930 MiB) — **8.90** GiB/s. Зато уходит одна лишняя копия: раньше view из mmap обязательно
  клонировался ради выравнивания, теперь байты читаются сразу в pinned-буфер конвейера
  (`new` и `new-pinned` неразличимы по скорости).
* `safetensors` 0.8.0 падает access violation на `get_tensor` из файла ≥ ~40 GiB (D12). Убирая
  `safe_open` из пути данных, снимаем и это ограничение.

Математика не меняется: те же байты в буфер torch (выровнен ≥ 64 байт) → те же активации. Это
раскладка памяти, а не другая арифметика, поэтому ключ кэша эмбеддингов от этого слоя не зависит.

Почему замок. `os.pread` на Windows нет (`hasattr(os, "pread") is False`, Python 3.12.10), а чтение
идёт по одному общему дескриптору: без замка на **весь** цикл `seek + readinto` поток A получает
байты потока B (воспроизводится детерминированно). Цена замка — 115 ns на захват, 704 захвата на
энкод ≈ 0.08 ms, два потока с общим замком дают ту же полосу, что один (6.84 против 6.81 GB/s).
Поэтому аллокации, логирование и каст в замок не попадают — только `seek` и `readinto`.

Почему цикл короткого чтения. `readinto` вправе вернуть меньше запрошенного, а `0` — это не «конец
данных», а обрыв. Молча усечённый тензор — не исключение, а «просто плохая картинка», поэтому
`got < nbytes` дотягивается циклом, `n == 0` даёт `ShardFormatError`.

Зависимости: `torch` импортируется **лениво** внутри `read()` (модуль обязан импортироваться без
torch — его берёт `registry.discover()` ради `python -m qwenstyle models list`), `safetensors` не
импортируется вовсе, `mmap` не используется.
"""
from __future__ import annotations

import json
import struct
import threading
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "GIB",
    "ShardFile",
    "ShardFormatError",
    "ShardSet",
    "TensorInfo",
]

GIB = 1024 ** 3

#: Имя dtype в заголовке safetensors -> имя dtype в `torch` (`getattr(torch, value)`).
_DTYPE_TO_TORCH: dict[str, str] = {
    "F64": "float64",
    "F32": "float32",
    "F16": "float16",
    "BF16": "bfloat16",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}

#: Размер элемента в байтах — нужен для сверки `end - begin == numel * element_size`, а через
#: `torch` это делать нельзя (модуль обязан работать без torch).
_ELEMENT_SIZE: dict[str, int] = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}

#: Разумный предел длины JSON-заголовка: у 750-тензорного набора 16.33 GiB это ~100 KiB, у
#: 256-тензорного шарда 23.79 GiB — 33 KiB. Предел нужен только чтобы не читать «заголовок» в
#: сотни мегабайт из мусорного файла.
_MAX_HEADER_BYTES = 256 * 1024 * 1024


class ShardFormatError(RuntimeError):
    """Файл не является корректным safetensors (битый заголовок, размер не совпал, короткое чтение)."""


@dataclass(frozen=True)
class TensorInfo:
    """Где именно лежит тензор: файл, смещение в байтах, размер, dtype, форма.

    Данные по этому описанию читаются одним `seek + readinto` — без разбора заголовка на каждый
    тензор и без загрузки шарда целиком.
    """

    name: str                       # полное имя тензора
    shard: Path                     # файл-шард
    file_dtype: str                 # как в заголовке: "BF16"
    dtype: str                      # torch-имя: "bfloat16"
    shape: tuple[int, ...]
    offset: int                     # абсолютное смещение данных в файле: 8 + len(header) + data_offsets[0]
    nbytes: int                     # произведение shape * element_size (сверено с data_offsets)


def _element_size(file_dtype: str, path: Path, name: str) -> int:
    """Размер элемента по имени dtype из заголовка; неизвестный dtype — ошибка формата."""
    try:
        return _ELEMENT_SIZE[file_dtype]
    except KeyError:
        raise ShardFormatError(
            f"shards: {path}: тензор {name!r} объявлен с dtype {file_dtype!r}, которого нет в "
            f"safetensors-спецификации (известны: {', '.join(sorted(_ELEMENT_SIZE))})"
        ) from None


def _read_header(fh: Any, path: Path) -> tuple[dict[str, Any], int]:
    """Прочитать `8 байт длины + JSON` из открытого файла; вернуть (заголовок, длина JSON)."""
    prefix = fh.read(8)
    if len(prefix) != 8:
        raise ShardFormatError(
            f"shards: {path}: файл короче 8 байт (получено {len(prefix)}) — это не safetensors"
        )
    (header_len,) = struct.unpack("<Q", prefix)
    if header_len == 0 or header_len > _MAX_HEADER_BYTES:
        raise ShardFormatError(
            f"shards: {path}: длина JSON-заголовка {header_len} байт вне разумных пределов "
            f"(1 … {_MAX_HEADER_BYTES}) — файл не safetensors или битый"
        )
    raw = fh.read(header_len)
    if len(raw) != header_len:
        # Заголовок короче объявленного: файл обрезан (частый след прерванной записи).
        raise ShardFormatError(
            f"shards: {path}: заголовок объявлен как {header_len} байт, прочитано {len(raw)} — "
            f"файл обрезан"
        )
    try:
        header = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ShardFormatError(
            f"shards: {path}: JSON-заголовок не разбирается ({exc}): {raw[:120]!r}"
        ) from None
    if not isinstance(header, dict):
        raise ShardFormatError(
            f"shards: {path}: заголовок должен быть JSON-объектом, получено {type(header).__name__}"
        )
    return header, header_len


def _parse_tensors(header: dict[str, Any], path: Path, header_len: int) -> dict[str, TensorInfo]:
    """Собрать `имя -> TensorInfo` и сверить объявленные размеры с фактическим размером файла."""
    infos: dict[str, TensorInfo] = {}
    max_end = 0
    for name, entry in header.items():
        if name == "__metadata__":           # метаданные файла — не тензор
            continue
        if not isinstance(entry, dict):
            raise ShardFormatError(
                f"shards: {path}: описание тензора {name!r} должно быть объектом, "
                f"получено {type(entry).__name__}"
            )
        try:
            file_dtype = entry["dtype"]
            shape = entry["shape"]
            begin, end = entry["data_offsets"]
        except (KeyError, TypeError, ValueError):
            raise ShardFormatError(
                f"shards: {path}: у тензора {name!r} нет полей dtype/shape/data_offsets: {entry!r}"
            ) from None
        if not isinstance(file_dtype, str) or not isinstance(shape, (list, tuple)):
            raise ShardFormatError(
                f"shards: {path}: у тензора {name!r} некорректные dtype/shape: {entry!r}"
            )
        dims: list[int] = []
        for dim in shape:
            if not isinstance(dim, int) or isinstance(dim, bool) or dim < 0:
                raise ShardFormatError(
                    f"shards: {path}: у тензора {name!r} некорректная форма {shape!r}"
                )
            dims.append(dim)
        numel = 1
        for dim in dims:
            numel *= dim
        nbytes = numel * _element_size(file_dtype, path, name)
        if not isinstance(begin, int) or not isinstance(end, int) or begin < 0 or end < begin:
            raise ShardFormatError(
                f"shards: {path}: у тензора {name!r} некорректные data_offsets {begin!r}…{end!r}"
            )
        if end - begin != nbytes:
            # Несовпадение объёма данных и формы = тихая порча весов при чтении, режем сразу.
            raise ShardFormatError(
                f"shards: {path}: тензор {name!r} ({file_dtype}, форма {tuple(dims)}) занимает "
                f"{end - begin} байт по data_offsets, а ожидается {nbytes} байт "
                f"({numel} элементов × {_ELEMENT_SIZE[file_dtype]} байт)"
            )
        infos[name] = TensorInfo(
            name=name,
            shard=path,
            file_dtype=file_dtype,
            dtype=_DTYPE_TO_TORCH.get(file_dtype) or _unknown_dtype(file_dtype, path, name),
            shape=tuple(dims),
            offset=8 + header_len + begin,
            nbytes=nbytes,
        )
        max_end = max(max_end, end)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ShardFormatError(f"shards: {path}: размер файла не читается ({exc})") from None
    expected = 8 + header_len + max_end
    if expected != size:
        # Обрезанный или «дописанный» файл: смещения после такого разбора попали бы в чужие байты
        # или за конец файла, то есть короткое/мусорное чтение вместо весов.
        raise ShardFormatError(
            f"shards: {path}: размер файла {size} байт не совпадает с расчётным {expected} байт "
            f"(8 + заголовок {header_len} + данные {max_end}) — файл обрезан или дописан"
        )
    return infos


def _unknown_dtype(file_dtype: str, path: Path, name: str) -> str:
    """Неизвестный dtype — ошибка формата (а не молчаливое чтение «как получится»)."""
    raise ShardFormatError(
        f"shards: {path}: тензор {name!r} объявлен с dtype {file_dtype!r}, которого нет в "
        f"safetensors-спецификации (известны: {', '.join(sorted(_DTYPE_TO_TORCH))})"
    )


def _as_bytes_view(view: Any) -> memoryview:
    """Нормализовать буфер к `memoryview` формата `B` (иначе `readinto` писал бы в копию среза)."""
    buffer = view if isinstance(view, memoryview) else memoryview(view)
    if buffer.itemsize != 1 or buffer.format != "B":
        buffer = buffer.cast("B")
    return buffer


class ShardFile:
    """Один файл: заголовок разобран в память, дескриптор открыт, чтение под замком.

    Заголовок читается один раз в `__init__` (33 KiB и 2 ms на шард 23.79 GiB), поэтому `info()`,
    `__contains__` и `len()` данных не трогают и после `close()` продолжают работать. Данные
    читаются позиционно: `seek(offset)` + `readinto(буфер)` под замком.

    Дескриптор открыт **без буферизации** (`buffering=0`): буферизованный `read` требует, чтобы
    позиция файла велась самим объектом, и добавляет лишнюю копию данных в user-space — ровно то,
    чего мы избегаем.

    Потокобезопасно: `seek + readinto` целиком внутри замка (на Windows нет `os.pread`, и гонка на
    общем дескрипторе перепутывает байты). Всё остальное — вне замка, чтобы чтение из хука
    (например, из `on_load`) не упиралось в собственный замок.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._closed = False
        self._infos: dict[str, TensorInfo] = {}
        self._names: tuple[str, ...] = ()
        self._fh: Any = None
        fh = open(self.path, "rb", buffering=0)   # дескриптор живёт до close(), не до конца файла
        try:
            header, header_len = _read_header(fh, self.path)
            self._infos = _parse_tensors(header, self.path, header_len)
        except BaseException:
            fh.close()                            # битый файл не оставляет открытый дескриптор
            raise
        self._fh = fh
        self._names = tuple(self._infos)

    # -- интроспекция ------------------------------------------------------ #

    @property
    def tensor_names(self) -> tuple[str, ...]:
        """Имена тензоров в порядке заголовка (данные не читаются)."""
        return self._names

    @property
    def closed(self) -> bool:
        """`True` после `close()`: дескриптор закрыт, чтение даёт `RuntimeError`."""
        return self._closed

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._infos

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        state = "закрыт" if self._closed else "открыт"
        size = ", ".join(sorted({info.file_dtype for info in self._infos.values()})) or "—"
        return f"ShardFile({self.path.name}: {len(self._names)} тензоров, {size}, {state})"

    def info(self, name: str) -> TensorInfo:
        """Описание тензора (`TensorInfo`) — из заголовка, без чтения данных."""
        try:
            return self._infos[name]
        except KeyError:
            raise KeyError(self._unknown_message(name)) from None

    def _unknown_message(self, name: str) -> str:
        shown = ", ".join(repr(item) for item in self._names[:5])
        return (f"shards: тензора {name!r} нет в {self.path.name} ({len(self._names)} тензоров). "
                f"Примеры имён: {shown}{' …' if len(self._names) > 5 else ''}")

    # -- чтение ------------------------------------------------------------ #

    def read_into(self, name: str, view: Any, *, at: int = 0) -> int:
        """Прочитать байты тензора в готовый буфер `view` со смещением `at`; вернуть число байт.

        `view` — любой буфер с интерфейсом `memoryview` (например `memoryview(t.numpy())` у
        `torch.uint8`-тензора): конвейер читает **прямо в pinned-буфер**, без промежуточной копии.
        Короткий `view` — `ValueError` **до** `seek`, то есть без движения позиции файла.
        """
        info = self.info(name)
        nbytes = info.nbytes
        buffer = _as_bytes_view(view)             # нормализация — вне замка
        if at < 0:
            raise ValueError(f"shards: отрицательное смещение {at} в буфере для {name!r}")
        if len(buffer) - at < nbytes:
            raise ValueError(
                f"shards: буфер короче тензора {name!r}: нужно {nbytes} байт со смещением {at}, "
                f"а в буфере {len(buffer)} байт"
            )
        with self._lock:                          # `seek + readinto` обязаны быть атомарны
            fh = self._fh
            if fh is None or self._closed:
                raise RuntimeError(
                    f"shards: {self.path.name} уже закрыт (close()) — веса больше не читаются"
                )
            fh.seek(info.offset)
            got = 0
            while got < nbytes:
                # Короткое чтение дотягивается циклом: усечённый тензор — это не исключение, а
                # тихая порча весов, заметная только по картинке.
                n = fh.readinto(buffer[at + got: at + nbytes])
                if not n:
                    raise ShardFormatError(
                        f"shards: {self.path.name}: короткое чтение тензора {name!r}: прочитано "
                        f"{got} из {nbytes} байт (файл обрезан или изменился на диске)"
                    )
                got += n
        return nbytes

    def read(self, name: str) -> Any:
        """Прочитать тензор в `torch` нужного dtype и формы (буфер выровнен, `clone` не нужен).

        `torch` импортируется здесь, а не на уровне модуля: `qwenstyle/shards.py` обязан
        импортироваться без torch (`registry.discover()` в `models list`).
        """
        import torch

        info = self.info(name)
        try:
            dtype = getattr(torch, info.dtype)
        except AttributeError:                 # pragma: no cover — старый torch без fp8-типов
            raise ShardFormatError(
                f"shards: {self.path}: dtype {info.dtype!r} в этой сборке torch отсутствует"
            ) from None
        tensor = torch.empty(info.nbytes, dtype=torch.uint8)   # выровнен ≥ 64 байт
        # `memoryview(tensor.untyped_storage())` в torch 2.8 падает с TypeError, а `torch.frombuffer`
        # дал бы буфер без гарантии выравнивания — поэтому `numpy()`-вид на CPU-тензор uint8.
        self.read_into(name, memoryview(tensor.numpy()))
        return tensor.view(dtype).view(info.shape)

    # -- закрытие ---------------------------------------------------------- #

    def close(self) -> None:
        """Закрыть дескриптор (идемпотентно, под тем же замком, что и чтение)."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            fh, self._fh = self._fh, None
            if fh is not None:
                try:
                    fh.close()
                except OSError:                    # закрытие не должно падать
                    pass


class ShardSet(Mapping[str, TensorInfo]):
    """Набор шардов: `имя -> TensorInfo`; чтение само выбирает файл; `close()` закрывает все.

    Как `Mapping` набор отдаёт **описания** (`TensorInfo`), а не тензоры: `__getitem__` не читает
    данные (`set[name].shape`, `.dtype`, `.offset` доступны бесплатно), а тензор берётся явно —
    `read(name)`. Так `file_dtypes()` и проверки форм не платят за чтение гигабайтов.

    `weight_map` (значения — имена файлов, как в `*.index.json`) определяет, какой шард держит имя.
    Без карты имена собираются из всех шардов; повтор имени в двух шардах — ошибка формата
    (неоднозначность молча разрешать нельзя).
    """

    def __init__(self, shards: Sequence[str | Path],
                 weight_map: Mapping[str, str] | None = None) -> None:
        paths: list[Path] = []
        for item in shards:
            path = Path(item)
            if path not in paths:
                paths.append(path)
        if not paths:
            raise ValueError("shards: ShardSet без шардов — сначала resolve_stream_source()")
        self._files: dict[Path, ShardFile] = {}
        self._by_name: dict[str, ShardFile] = {}
        try:
            for path in paths:
                self._files[path] = ShardFile(path)
            # имя файла -> файл: значения `weight_map` в `*.index.json` записаны именами
            self._by_name = {path.name: shard for path, shard in self._files.items()}
            self._infos, self._owner = self._build_map(weight_map)
        except BaseException:
            self.close()                          # не оставляем открытые дескрипторы при сбое
            raise
        self._names = tuple(self._infos)

    def _build_map(self, weight_map: Mapping[str, str] | None) -> tuple[dict[str, TensorInfo],
                                                                       dict[str, ShardFile]]:
        """Собрать `имя -> TensorInfo` и `имя -> файл` (по карте или по всем шардам)."""
        infos: dict[str, TensorInfo] = {}
        owner: dict[str, ShardFile] = {}
        if weight_map is not None:
            for name, shard_name in weight_map.items():
                shard = self._file_for(str(shard_name))
                try:
                    infos[name] = shard.info(name)
                except KeyError:
                    # Битая карта — ошибка формата, а не отсутствующий тензор: остальные ветки
                    # `weight_map` уже говорят `ShardFormatError`, и `KeyError` здесь утёк бы
                    # наружу как «нет имени», хотя проблема — в файле, указанном картой.
                    raise ShardFormatError(
                        f"shards: weight_map указывает файл {shard.path.name!r} для тензора "
                        f"{name!r}, но такого тензора в нём нет; доступные шарды: "
                        f"{', '.join(path.name for path in self._files)}"
                    ) from None
                owner[name] = shard
            return infos, owner
        for path, shard in self._files.items():
            for name in shard.tensor_names:
                if name in owner:
                    raise ShardFormatError(
                        f"shards: тензор {name!r} объявлен и в {owner[name].path.name}, и в "
                        f"{path.name} — без weight_map нельзя решить, какой файл верный"
                    )
                infos[name] = shard.info(name)
                owner[name] = shard
        return infos, owner

    def _file_for(self, shard_name: str) -> ShardFile:
        """Файл по имени из `weight_map` (значение — имя файла или путь)."""
        if shard_name in self._by_name:
            return self._by_name[shard_name]
        candidate = Path(shard_name)
        if not candidate.is_absolute():
            candidate = Path(next(iter(self._files))).parent / candidate
        for path, shard in self._files.items():
            if path == candidate or path.resolve() == candidate.resolve():
                return shard
        raise ShardFormatError(
            f"shards: weight_map ссылается на файл {shard_name!r}, которого нет среди шардов "
            f"({', '.join(path.name for path in self._files)})"
        )

    # -- Mapping ----------------------------------------------------------- #

    def __getitem__(self, name: str) -> TensorInfo:
        return self.info(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._infos

    def __repr__(self) -> str:
        state = "закрыт" if self.closed else "открыт"
        return (f"ShardSet({len(self._names)} тензоров, {len(self._files)} шардов, {state})")

    # -- интроспекция ------------------------------------------------------ #

    @property
    def closed(self) -> bool:
        """`True`, когда закрыты все файлы набора."""
        return all(shard.closed for shard in self._files.values())

    def info(self, name: str) -> TensorInfo:
        """Описание тензора — без чтения данных."""
        try:
            return self._infos[name]
        except KeyError:
            raise KeyError(self._unknown_message(name)) from None

    def _unknown_message(self, name: str) -> str:
        shown = ", ".join(repr(item) for item in self._names[:5])
        where = ", ".join(path.name for path in self._files)
        return (f"shards: тензора {name!r} нет в {where} ({len(self._names)} тензоров). "
                f"Примеры имён: {shown}{' …' if len(self._names) > 5 else ''}")

    def file_dtypes(self) -> dict[str, str]:
        """`имя тензора -> dtype файла` (`"BF16"`), только по заголовкам, без чтения данных."""
        return {name: info.file_dtype for name, info in self._infos.items()}

    def shard_path(self, name: str) -> Path:
        """Файл, в котором лежит тензор (нужен для сообщений и для `weight_map`-диагностики)."""
        return self._owner[name].path

    # -- чтение ------------------------------------------------------------ #

    def read(self, name: str) -> Any:
        """Прочитать тензор (`torch`) нужного dtype и формы из его шарда."""
        return self._owner_or_unknown(name).read(name)

    def read_into(self, name: str, view: Any, *, at: int = 0) -> int:
        """Прочитать байты тензора прямо в готовый буфер (см. `ShardFile.read_into`)."""
        return self._owner_or_unknown(name).read_into(name, view, at=at)

    def _owner_or_unknown(self, name: str) -> ShardFile:
        try:
            return self._owner[name]
        except KeyError:
            raise KeyError(self._unknown_message(name)) from None

    # -- закрытие ---------------------------------------------------------- #

    def close(self) -> None:
        """Закрыть все файлы набора (идемпотентно)."""
        for shard in self._files.values():
            shard.close()
