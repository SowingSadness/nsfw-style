"""Чтение квантованного кэша: позиционно (shard + safetensors) и с де-квантом на месте.

Зачем (план `cube-architecture`, шаг 5b): адаптерам моделей нужен одинаковый способ достать тензор
из кэша — неважно, diffusers это (`from_pretrained(dtype=fp8)`) или transformers-LM с подменой
`Linear`. `QuantReader` даёт:

* `names()` — список тензоров кэша (из `index.json`);
* `read(name)` — тензор как он лежит на диске (fp8 остаётся fp8);
* `read(name, dtype=torch.bfloat16)` — с приведением, то есть де-квант;
* `read_into(name, buffer)` — запись прямо в буфер (для потокового загрузчика).

Файлы открываются лениво и закрываются в `close()`; это дешевле, чем `safe_open` на каждый тензор
(замер: 2.3–3.3 мс на открытие шарда).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .store import INDEX_NAME, WEIGHTS_NAME, QuantStore, read_header

__all__ = ["QuantReader"]


class QuantReader:
    """Позиционное чтение тензоров из шардов кэша по `weight_map`."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.store = QuantStore(cache_dir)
        index = self.store.index()
        self._weight_map: dict[str, str] = dict(index.get("weight_map") or {})
        if not self._weight_map:
            # Легаси-раскладка: один файл без index.json (`cache/models/fp8_transformer`).
            single = self.store.path / WEIGHTS_NAME
            if not single.is_file():
                raise FileNotFoundError(
                    f"{self.store.path}: нет ни {INDEX_NAME} с weight_map, ни {WEIGHTS_NAME}")
            self._weight_map = {name: WEIGHTS_NAME for name in read_header(single)}
        self._handles: dict[str, Any] = {}
        self._shard_paths = sorted(set(self._weight_map.values()))

    # -- метаданные -------------------------------------------------------- #
    def names(self) -> list[str]:
        return sorted(self._weight_map)

    def shards(self) -> list[str]:
        return list(self._shard_paths)

    def shard_of(self, name: str) -> str:
        try:
            return self._weight_map[name]
        except KeyError:
            raise KeyError(f"{name!r} нет в кэше {self.store.path}") from None

    def sidecar(self) -> dict[str, Any]:
        return self.store.sidecar()

    # -- чтение ------------------------------------------------------------ #
    def _handle(self, shard: str) -> Any:
        handle = self._handles.get(shard)
        if handle is None:
            from safetensors import safe_open

            handle = safe_open(str(self.store.path / shard), framework="pt", device="cpu")
            self._handles[shard] = handle
        return handle

    def read(self, name: str, *, dtype: Any = None) -> Any:
        """Тензор из кэша; `dtype` задан — привести (для fp8 это де-квант)."""
        shard = self.shard_of(name)
        tensor = self._handle(shard).get_tensor(name)
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        return tensor

    def read_into(self, name: str, buffer: Any) -> Any:
        """Прочитать тензор в заранее выделенный буфер (по возможности без копии)."""
        tensor = self.read(name)
        if tensor.shape != buffer.shape:
            raise ValueError(f"{name}: форма {tuple(tensor.shape)} != буфера {tuple(buffer.shape)}")
        buffer.copy_(tensor.to(buffer.dtype))
        return buffer

    def iter_shard(self, shard: str) -> Iterator[tuple[str, Any]]:
        """Все тензоры одного шарда (для потоковой загрузки модулей)."""
        names = [n for n in self.names() if self._weight_map[n] == shard]
        for name in names:
            yield name, self.read(name)

    # -- уборка ------------------------------------------------------------ #
    def close(self) -> None:
        self._handles.clear()

    def __enter__(self) -> "QuantReader":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:                                        # pragma: no cover
        return f"QuantReader({self.store.path.as_posix()}, {len(self._weight_map)} тензоров)"
