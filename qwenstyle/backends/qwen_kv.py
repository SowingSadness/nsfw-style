"""RAM-offload префиксного KV-кеша Qwen-Image-2.1 — раскладка памяти, а не математика.

Зачем. При `causal_condition` (у Qwen-Image-2.1 `true`) `QwenImage21Pipeline` считает префикс
(текст + все референсы) **один раз** на шаге 0 и держит K/V каждого из 32 слоёв до конца
денойзинга (`QwenImage21KVLayerCache`, `transformer_qwenimage21.py`). Префикс не зависит от шага,
поэтому кеш — это просто два тензора формы `(1, prefix, heads, head_dim)`; цена — **512 КиБ на
префиксный токен** (`32 слоя x 2 x 4096 x 2 Б` bf16). Для 3 рефов @1376x768 это ~6.6 ГиБ VRAM,
для 4 — ~8.6 ГиБ, чего на 16 ГБ вместе с весами и активациями уже нет.

Что делает патч. `store()` (шаг 0, K/V живут в VRAM) копирует K и V в **обычную (pageable) RAM** и
отпускает GPU-копии, `get()` (каждый следующий шаг, каждый слой) возвращает их на то же устройство,
что было у `store()`. В VRAM живёт только текущий слой, сумма по слоям остаётся в RAM — то есть
работает ровно как `--offload group` у весов, только для кеша, а не для весов.

Почему бит-в-бит. Меняется только место хранения тензоров: `store`/`get` отдают те же значения в
том же dtype/shape/устройстве, поэтому `torch.cat([cached_k, key], dim=1)` получает байт-в-байт
те же тензоры, что и при кеше в VRAM. Никакой арифметики (в т.ч. fp8) здесь нет.

Почему **не pinned**. На Windows pinned-память CUDA — это блокировка страниц **и** маппинг в
адресное пространство GPU (WDDM), то есть тот же ресурс, что VRAM (AGENTS.md §4: pinned staging
энкодера и group-offload конфликтуют). Поэтому кеш кладётся в pageable RAM, а `non_blocking=True`
оставлен как безопасный no-op: без pinned он ничего не ускоряет, но и не мешает.

Использование (`QwenImage21Backend.generate`):

    from .qwen_kv import install_offload, uninstall
    if install_offload():            # False = diffusers без этого класса, режим не включён
        try:
            pipe(...)                # use_kv_cache=True
        finally:
            uninstall()              # обязательно: патч живёт в процессе глобально

`install_offload`/`uninstall` идемпотентны; `uninstall` обязан вызываться всегда (в `finally`),
иначе в батче по seed'ам второй кадр поедет с чужим режимом раскладки. Модуль не импортирует
torch на уровне модуля: единственная тяжёлая зависимость (`diffusers`) подтягивается лениво внутри
`install_offload`.
"""
from __future__ import annotations

import importlib
from typing import Any

#: Модуль diffusers, в котором живут классы кеша (единственная точка перехвата).
TRANSFORMER_MODULE = "diffusers.models.transformers.transformer_qwenimage21"
#: Per-layer кеш: два поля `k`/`v` и методы `store`/`get` — их и подменяем.
LAYER_CACHE_CLASS = "QwenImage21KVLayerCache"
#: Контейнер кешей по слоям: нужен только чтобы узнать число слоёв для строки `[kv] offload: …`.
CACHE_CLASS = "QwenImage21KVCache"

#: Состояние патча (модульное: патч живёт в процессе, `install`/`uninstall` — идемпотентны).
_installed = False
_verbose = True
_reported = False
_orig_store: Any = None
_orig_get: Any = None
_orig_cache_init: Any = None
_patched_layer_cls: Any = None
_patched_cache_cls: Any = None
#: Число блоков трансформера в последнем созданном `QwenImage21KVCache` (для «всего … GiB»).
_layer_count = 0


def _human_bytes(size: int) -> tuple[float, str]:
    """Байты в MiB/GiB (`2**20`/`2**30`), KiB только для крошечных тензоров (тесты, 7 токенов)."""
    if size >= 2 ** 30:
        return size / 2 ** 30, "GiB"
    if size >= 2 ** 20:
        return size / 2 ** 20, "MiB"
    return size / 2 ** 10, "KiB"


# --------------------------------------------------------------------------- #
# Подменённые реализации
# --------------------------------------------------------------------------- #

def _store_offload(self: Any, k: Any, v: Any) -> None:
    """Шаг 0: K/V уходят в pageable RAM; устройство запоминается для `get()`.

    Присваивание `self.k`/`self.v` само отпускает предыдущие тензоры: перезапись без `get()`
    (второй `extract` того же кеша) допустима и не течёт.
    """
    self._device = k.device
    self.k = k.detach().to("cpu", dtype=k.dtype, copy=True)
    self.v = v.detach().to("cpu", dtype=v.dtype, copy=True)
    _report_once(self)


def _get_offload(self: Any) -> tuple[Any, Any]:
    """Каждый шаг/слой: K/V возвращаются на исходное устройство (`cuda`) для `torch.cat`."""
    if self.k is None or self.v is None:
        raise RuntimeError(
            "kv offload: get() до store() — префиксный KV-кеш не заполнен"
        )
    device = getattr(self, "_device", None)
    return self.k.to(device, non_blocking=True), self.v.to(device, non_blocking=True)


def _cache_init_counted(self: Any, *args: Any, **kwargs: Any) -> None:
    """`QwenImage21KVCache(num_layers)`: запомнить число слоёв для строки `[kv] offload: …`."""
    global _layer_count
    _orig_cache_init(self, *args, **kwargs)
    try:
        _layer_count = len(getattr(self, "layer_caches", None) or ())
    except Exception:                                # noqa: BLE001 — лог не должен ронять прогон
        _layer_count = 0


def _report_once(cache: Any) -> None:
    """Одна строка `[kv] offload: …` на установку патча; числа — из реальных тензоров."""
    global _reported
    if _reported or not _verbose:
        return
    _reported = True
    try:
        tokens = int(cache.k.shape[1])
        layer_bytes = int(cache.k.numel() * cache.k.element_size()
                          + cache.v.numel() * cache.v.element_size())
    except Exception:                                    # noqa: BLE001 — печать не обязательна
        return
    layer_value, layer_unit = _human_bytes(layer_bytes)
    total = f", всего {layer_bytes * _layer_count / 2 ** 30:.2f} GiB в RAM" if _layer_count else ""
    print(f"[kv] offload: prefix {tokens} токенов/слой, K+V {layer_value:.1f} {layer_unit}/слой"
          f"{total}; подкачка {layer_value:.1f} {layer_unit} на слой и шаг", flush=True)


# --------------------------------------------------------------------------- #
# Установка / снятие патча
# --------------------------------------------------------------------------- #

def install_offload(verbose: bool = True) -> bool:
    """Подменить `store`/`get` per-layer кеша на RAM-offload. `False` — патч недоступен.

    Идемпотентно: повторный вызов ничего не делает и возвращает `True`. Тяжёлый импорт
    (`diffusers` -> `torch`) — только здесь, поэтому модуль можно импортировать где угодно.
    """
    global _installed, _verbose, _reported
    global _orig_store, _orig_get, _orig_cache_init, _patched_layer_cls, _patched_cache_cls

    if _installed:
        return True
    try:
        module: Any = importlib.import_module(TRANSFORMER_MODULE)
        QwenImage21KVLayerCache = getattr(module, LAYER_CACHE_CLASS)
    except Exception as exc:                             # noqa: BLE001 — режим не обязателен
        print(f"[kv] offload недоступен: {type(exc).__name__}: {exc}", flush=True)
        return False

    cache_cls = getattr(module, CACHE_CLASS, None)

    _orig_store = QwenImage21KVLayerCache.store
    _orig_get = QwenImage21KVLayerCache.get
    QwenImage21KVLayerCache.store = _store_offload
    QwenImage21KVLayerCache.get = _get_offload
    _patched_layer_cls = QwenImage21KVLayerCache
    # Число слоёв знает только контейнер; без него строка лога печатается без «всего … GiB».
    _orig_cache_init = None
    _patched_cache_cls = None
    if isinstance(cache_cls, type):
        _orig_cache_init = cache_cls.__init__
        cache_cls.__init__ = _cache_init_counted
        _patched_cache_cls = cache_cls

    _installed = True
    _verbose = bool(verbose)
    _reported = False
    return True


def uninstall() -> None:
    """Вернуть оригинальные `store`/`get` (и `__init__` контейнера). Идемпотентно."""
    global _installed, _orig_store, _orig_get, _orig_cache_init
    global _patched_layer_cls, _patched_cache_cls

    if not _installed:
        return
    if _patched_layer_cls is not None:
        if _orig_store is not None:
            _patched_layer_cls.store = _orig_store
        if _orig_get is not None:
            _patched_layer_cls.get = _orig_get
    if _patched_cache_cls is not None and _orig_cache_init is not None:
        _patched_cache_cls.__init__ = _orig_cache_init

    _orig_store = _orig_get = _orig_cache_init = None
    _patched_layer_cls = _patched_cache_cls = None
    _installed = False


__all__ = ["CACHE_CLASS", "LAYER_CACHE_CLASS", "TRANSFORMER_MODULE", "install_offload", "uninstall"]
