"""RAM-offload префиксного KV-кеша (`qwenstyle/backends/qwen_kv.py`) — обычный скрипт, без pytest.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_kv_cache_offload.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем: `kv_cache=offload` держит префиксный KV-кеш Qwen-Image-2.1 (текст + все референсы, шаг 0,
K/V каждого из 32 слоёв) в pageable RAM и возвращает K/V на исходное устройство в `get()`.
Это **раскладка памяти, а не математика**, поэтому единственное, что нужно доказать без весов, —
`torch.equal` (бит-в-бит), исходное устройство, dtype и shape, плюс что патч снимается и не течёт
между прогонами (в батче по seed'ам второй кадр не должен поехать с чужим режимом).

Тест идёт без CUDA: кейс 5 (реальный `cuda`) печатает `SKIP`, если GPU нет, остальные — только CPU.

Кейсы:

1. roundtrip бит-в-бит на CPU: `store` → `get` возвращает те же значения (`torch.equal`), device
   `cpu`, dtype/shape сохранены;
2. `store` дважды без `get` (перезапись, второй `extract` того же кеша) — не падает, старые тензоры
   отпущены (`id` сменился), `get()` отдаёт **новые** значения, а не первые;
3. `get()` без `store()` — внятный `RuntimeError` с упоминанием `store`, а не `AttributeError`;
4. идемпотентность: `install_offload()` дважды → оба раза `True` (второй — no-op), `uninstall()`
   дважды не падает, после `uninstall` методы оригинальные — `get()` отдаёт **те же объекты**,
   что положил `store` (то есть кеш снова в VRAM, без копии на CPU);
5. CUDA: `store` кладёт K/V на `cpu` (`c.k.device.type == "cpu"`), `get()` возвращает на `cuda`
   бит-в-бит (иначе `SKIP` — машина без GPU);
6. строка `[kv] offload: …` печатается **один раз** на установку патча (второй `store` молчит);
7. `_kv_cache_mode` (режим из `generation.sampler_kwargs.kv_cache`): умолчание `true`, `offload`,
   `false` в bool/`"0"`/`"off"`, регистр, неизвестное значение → `ValueError` с `true|false|offload`.
"""
from __future__ import annotations

import contextlib
import io
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:              # запуск из любого cwd
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                  # тест не должен оставлять __pycache__ в пакете

from qwenstyle.backends.qwen_image21 import _kv_cache_mode      # noqa: E402
from qwenstyle.backends.qwen_kv import (                        # noqa: E402
    LAYER_CACHE_CLASS,
    TRANSFORMER_MODULE,
    install_offload,
    uninstall,
)


# --------------------------------------------------------------------------- #
# Мелкие помощники
# --------------------------------------------------------------------------- #

def _torch() -> Any:
    import torch

    return torch


def _layer_cls() -> Any:
    """Класс per-layer кеша из diffusers (единственная точка перехвата патча)."""
    import importlib

    return getattr(importlib.import_module(TRANSFORMER_MODULE), LAYER_CACHE_CLASS)


def _has_cuda() -> bool:
    try:
        return bool(_torch().cuda.is_available())
    except Exception:                                    # noqa: BLE001 — тест не должен врать
        return False


def _kv(tokens: int = 7, *, device: str = "cpu", seed: int = 0) -> tuple[Any, Any]:
    """K/V формы `(1, tokens, 2, 4)` bf16 — как у пайплайна `(batch, prefix, heads, head_dim)`."""
    torch = _torch()
    generator = torch.Generator().manual_seed(seed)
    shape = (1, tokens, 2, 4)
    k = torch.randn(shape, generator=generator, dtype=torch.bfloat16).to(device)
    v = torch.randn(shape, generator=generator, dtype=torch.bfloat16).to(device)
    return k, v


@contextlib.contextmanager
def _patched(verbose: bool = False):
    """`install_offload` на время кейса; `uninstall()` в `finally` — патч не течёт между кейсами."""
    if not install_offload(verbose=verbose):
        raise AssertionError(f"install_offload() вернул False: в diffusers нет {LAYER_CACHE_CLASS}")
    try:
        yield
    finally:
        uninstall()                                  # идемпотентно: кейс 4 зовёт его сам


def _fake_cfg(sampler_kwargs: Any) -> Any:
    """Минимальный RunConfig-подобный объект: бекенду нужен только `generation.sampler_kwargs`."""
    return SimpleNamespace(generation=SimpleNamespace(sampler_kwargs=sampler_kwargs))


# --------------------------------------------------------------------------- #
# Кейсы
# --------------------------------------------------------------------------- #

def case_roundtrip_cpu() -> None:
    """1. `store` → `get` на CPU: бит-в-бит, device/shape/dtype сохранены."""
    k, v = _kv()
    with _patched():
        cache = _layer_cls()()
        cache.store(k, v)
        k2, v2 = cache.get()
        if k2.device.type != "cpu":
            raise AssertionError(f"get() вернул не cpu: {k2.device}")
        if k2.shape != k.shape or v2.shape != v.shape:
            raise AssertionError(f"shape изменился: {tuple(k2.shape)} vs {tuple(k.shape)}")
        if k2.dtype != k.dtype or v2.dtype != v.dtype:
            raise AssertionError(f"dtype изменился: {k2.dtype}/{v2.dtype} vs {k.dtype}/{v.dtype}")
        if not _torch().equal(k, k2) or not _torch().equal(v, v2):
            raise AssertionError("значения после roundtrip не бит-в-бит (torch.equal == False)")
        if cache.k.device.type != "cpu" or cache.v.device.type != "cpu":
            raise AssertionError("в кеше остались не-CPU тензоры — экономии VRAM нет")
    print(f"  ok: roundtrip cpu {tuple(k.shape)} bf16 — torch.equal, device/shape/dtype сохранены")


def case_store_twice() -> None:
    """2. Второй `store` без `get` — перезапись: старые тензоры отпущены, `get` отдаёт новые."""
    k1, v1 = _kv(seed=1)
    k2, v2 = _kv(seed=2)
    with _patched():
        cache = _layer_cls()()
        cache.store(k1, v1)
        first_k, first_v = cache.k, cache.v
        cache.store(k2, v2)                          # второй extract того же кеша
        if cache.k is first_k or cache.v is first_v:
            raise AssertionError("перезапись не создала новые тензоры (старые держатся)")
        got_k, got_v = cache.get()
        if not _torch().equal(got_k, k2) or not _torch().equal(got_v, v2):
            raise AssertionError("get() вернул не последние значения (перезапись потерялась)")
        if _torch().equal(got_k, k1):
            raise AssertionError("get() вернул первые значения — store не перезаписал кеш")
    print("  ok: повторный store перезаписал кеш без падения, get() отдаёт новые значения")


def case_get_without_store() -> None:
    """3. `get()` без `store()` — внятный `RuntimeError` с подсказкой про `store`."""
    with _patched():
        cache = _layer_cls()()
        try:
            cache.get()
        except RuntimeError as exc:
            text = str(exc)
            if "store" not in text:
                raise AssertionError(f"RuntimeError не упоминает store: {text!r}") from None
            print(f"  ok: get() без store -> RuntimeError: {text}")
            return
        except Exception as exc:                     # noqa: BLE001
            raise AssertionError(f"ожидался RuntimeError, получено {type(exc).__name__}: {exc}") from None
        raise AssertionError("get() без store не упал — кеш отдал неинициализированные тензоры")


def case_idempotent_and_restore() -> None:
    """4. `install`×2 / `uninstall`×2 идемпотентны, после снятия — оригинальные методы."""
    cls = _layer_cls()
    original_store, original_get = cls.store, cls.get
    try:
        if install_offload(verbose=False) is not True:
            raise AssertionError("первый install_offload() вернул не True")
        patched_store = cls.store
        if install_offload(verbose=False) is not True:
            raise AssertionError("повторный install_offload() вернул не True (должен быть no-op)")
        if cls.store is not patched_store or cls.store is original_store:
            raise AssertionError("повторный install переустановил патч (должен быть no-op)")

        uninstall()
        uninstall()                                  # второй uninstall не должен падать
        if cls.store is not original_store or cls.get is not original_get:
            raise AssertionError("uninstall() не вернул оригинальные store/get")

        # Оригинал кеширует тензоры в VRAM как есть: get() отдаёт те же объекты, без копии.
        k, v = _kv()
        cache = cls()
        cache.store(k, v)
        if cache.k is not k or cache.v is not v:
            raise AssertionError("после uninstall() store всё ещё копирует тензоры (патч не снят)")
        got_k, got_v = cache.get()
        if got_k is not k or got_v is not v:
            raise AssertionError("после uninstall() get() отдаёт не те же объекты")
        print("  ok: install×2 / uninstall×2 идемпотентны, после снятия get() отдаёт те же объекты")
    finally:
        uninstall()


def case_roundtrip_cuda() -> None:
    """5. CUDA: `store` → CPU, `get` → `cuda` бит-в-бит (иначе SKIP)."""
    if not _has_cuda():
        print("  SKIP (нет CUDA): кеш в VRAM проверять не на чем")
        return
    torch = _torch()
    k, v = _kv(13, device="cuda")
    with _patched():
        cache = _layer_cls()()
        cache.store(k, v)
        if cache.k.device.type != "cpu" or cache.v.device.type != "cpu":
            raise AssertionError(f"в кеше остались GPU-тензоры: {cache.k.device}, {cache.v.device}")
        k2, v2 = cache.get()
        if k2.device.type != "cuda" or v2.device.type != "cuda":
            raise AssertionError(f"get() вернул не cuda: {k2.device}, {v2.device}")
        if not torch.equal(k, k2) or not torch.equal(v, v2):
            raise AssertionError("значения после roundtrip cuda -> cpu -> cuda не бит-в-бит")
        if k2.dtype != k.dtype or k2.shape != k.shape:
            raise AssertionError(f"dtype/shape изменились: {k2.dtype} {tuple(k2.shape)}")
        del k2, v2
        torch.cuda.empty_cache()
    print(f"  ok: cuda {tuple(k.shape)} bf16 -> кеш на cpu -> get() на cuda, torch.equal")


def case_prints_once() -> None:
    """6. Строка `[kv] offload: …` — один раз на установку патча, второй `store` молчит."""
    k1, v1 = _kv(seed=3)
    k2, v2 = _kv(seed=4)
    with _patched(verbose=True):
        cache = _layer_cls()()
        first = io.StringIO()
        with contextlib.redirect_stdout(first):
            cache.store(k1, v1)
        second = io.StringIO()
        with contextlib.redirect_stdout(second):
            cache.store(k2, v2)
        text, again = first.getvalue(), second.getvalue()
    if "[kv] offload:" not in text:
        raise AssertionError(f"первый store не напечатал строку [kv] offload:, stdout: {text!r}")
    if again.strip():
        raise AssertionError(f"второй store напечатал ещё раз: {again!r}")
    print(f"  ok: одна строка на установку патча: {text.strip()}")


def case_mode_from_config() -> None:
    """7. `_kv_cache_mode`: умолчание `true`, три режима, алиасы, регистр, ошибка на мусоре."""
    expected: tuple[tuple[Any, str], ...] = (
        ({}, "true"),
        ({"kv_cache": "offload"}, "offload"),
        ({"kv_cache": False}, "false"),
        ({"kv_cache": True}, "true"),
        ({"kv_cache": "0"}, "false"),
        ({"kv_cache": "off"}, "false"),
        ({"kv_cache": "OFFLOAD"}, "offload"),
        ({"kv_cache": "  Offload "}, "offload"),
        ({"kv_cache": "on"}, "true"),
    )
    for kwargs, want in expected:
        got = _kv_cache_mode(_fake_cfg(dict(kwargs)))
        if got != want:
            raise AssertionError(f"sampler_kwargs={kwargs} -> {got!r}, ожидалось {want!r}")
    print(f"  ok: {len(expected)} значений sampler_kwargs.kv_cache разобраны в true/false/offload")

    try:
        _kv_cache_mode(_fake_cfg({"kv_cache": "bad"}))
    except ValueError as exc:
        text = str(exc)
        if "true|false|offload" not in text:
            raise AssertionError(f"ValueError без перечня режимов: {text!r}")
        print(f"  ok: неизвестное значение -> ValueError: {text}")
        return
    raise AssertionError("значение kv_cache=bad не отвергнуто (нужен ValueError)")


def main() -> int:
    cases = (
        ("1. roundtrip KV в RAM (бит-в-бит, cpu)", case_roundtrip_cpu),
        ("2. повторный store без get (перезапись)", case_store_twice),
        ("3. get без store (RuntimeError)", case_get_without_store),
        ("4. install/uninstall идемпотентны, оригинал восстановлен", case_idempotent_and_restore),
        ("5. roundtrip cuda -> cpu -> cuda", case_roundtrip_cuda),
        ("6. строка [kv] offload печатается один раз", case_prints_once),
        ("7. режим из sampler_kwargs.kv_cache", case_mode_from_config),
    )
    failed = 0
    for label, case in cases:
        try:
            case()
            print(f"PASS {label}", flush=True)
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {label}: {exc}", flush=True)
        except Exception:                               # noqa: BLE001 — тест не должен врать
            failed += 1
            print(f"FAIL {label}: исключение\n{traceback.format_exc()}", flush=True)
        finally:
            uninstall()                                 # патч живёт в процессе глобально
    if failed:
        print(f"{failed} FAIL", flush=True)
        return 1
    print("ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
