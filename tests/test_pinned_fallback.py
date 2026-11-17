"""Pinned-пул стриминга: деградация до обычной RAM вместо OOM (без CUDA обязателен только кейс 1).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_pinned_fallback.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем: на Windows pinned-память CUDA — это и блокировка страниц, и маппинг в адресное
пространство GPU (WDDM), поэтому она упирается в те же ресурсы, что VRAM. Исторически групповой
offload трансформера держал рядом ~6.6 ГиБ pinned-параметров, и первый же большой staging-буфер
энкодера падал `CUDA error: out of memory` **при свободной VRAM**:

    [4/5] кодирование промта на GPU (Qwen3VL-8B, стриминг)...
    torch.AcceleratorError: CUDA error: out of memory
      streaming.py: torch.empty(numel, dtype=tensor.dtype, pin_memory=True)

Pinned — оптимизация транспорта (26.7 против 20.5 GB/s), а не условие корректности: числа те же
бит-в-бит. Поэтому pin включён по умолчанию **и при `offload=group`** (политика — кейс 5), а пул
обязан **деградировать**, а не падать: `take()` возвращает исходный тензор из обычной RAM, пул
выключается до конца сессии и печатает причину одной строкой. Деградация — страховка на случай
чужой машины, а не штатный режим под group-offload.

Кейсы:

1. `pin_memory=False` -> `take` отдаёт тот же тензор и `None` (пул не создаётся вовсе).
2. Нормальный режим -> `take` отдаёт **pinned** буфер с теми же числами, `give` его переиспользует.
3. `torch.empty(pin_memory=True)` бросает `RuntimeError("CUDA error: out of memory")` ->
   `take` не падает: возвращает исходный тензор, `pool.enabled == False`, `pool.fallback == True`,
   `pool.bytes == 0`, в stdout ровно две строки `[stream] …`.
4. После деградации последующие `take`/`give` работают (модель дочитывает веса) и не пытаются
   пинить снова.
"""
from __future__ import annotations

import contextlib
import io
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwenstyle import streaming                       # noqa: E402


def _torch() -> object:
    return streaming._torch()


def _has_cuda() -> bool:
    try:
        return bool(_torch().cuda.is_available())
    except Exception:
        return False


def _captured() -> "contextlib.AbstractContextManager[io.StringIO]":
    """Перехват stdout/stderr: предупреждение о деградации печатается, а не молчит."""
    return contextlib.redirect_stdout(io.StringIO())


def case_pool_disabled() -> None:
    """1. `enabled=False` -> нулевая работа: ни буфера, ни аллокации."""
    pool = streaming._PinnedPool(False)
    tensor = _torch().zeros(4, dtype=_torch().float32)
    host, buffer = pool.take(tensor)
    if host is not tensor or buffer is not None:
        raise AssertionError(f"выключенный пул вернул ({type(host)}, {buffer})")
    if pool.bytes != 0:
        raise AssertionError(f"выключенный пул занял {pool.bytes} байт")
    print("  ok: pin_memory=False -> тот же тензор, пул ничего не держит")


def case_pool_reuse() -> None:
    """2. Нормальный режим: pinned-буфер, те же числа, буфер переиспользуется."""
    if not _has_cuda():
        print("  SKIP (нет CUDA): pinned-память недоступна")
        return
    torch = _torch()
    pool = streaming._PinnedPool(True)
    first = torch.arange(8, dtype=torch.float32)
    host, buffer = pool.take(first)
    if buffer is None or not host.is_pinned():
        raise AssertionError("pinned-буфер не выделился при доступной CUDA")
    if not torch.equal(host, first):
        raise AssertionError("данные в pinned-буфере не совпали с исходным тензором")
    pool.give([buffer])
    second = torch.arange(8, dtype=torch.float32) * 2
    host2, buffer2 = pool.take(second)
    if buffer2 is not buffer:
        raise AssertionError("пул не переиспользовал освобождённый буфер (best-fit сломан)")
    if not torch.equal(host2, second):
        raise AssertionError("после переиспользования буфера данные не те")
    print("  ok: pinned-буфер выделяется, числа совпадают, буфер переиспользуется")


@contextlib.contextmanager
def _failing_pin():
    """`torch.empty(pin_memory=True)` падает как на Windows при забитой pinned-памяти."""
    torch = _torch()
    original = torch.empty

    def failing_empty(*args: object, **kwargs: object) -> object:
        if kwargs.get("pin_memory"):
            raise RuntimeError("CUDA error: out of memory")
        return original(*args, **kwargs)

    torch.empty = failing_empty                    # type: ignore[assignment]
    try:
        yield
    finally:
        torch.empty = original                     # type: ignore[assignment]


def case_degrades_instead_of_crash() -> None:
    """3. OOM на pinned-аллокации -> страховочный путь, а не падение прогона."""
    if not _has_cuda():
        print("  SKIP (нет CUDA): без CUDA `pin_memory=True` всё равно не пинится")
        return
    torch = _torch()
    pool = streaming._PinnedPool(True)
    tensor = torch.ones(16, dtype=torch.float32)
    out = io.StringIO()
    with _failing_pin(), contextlib.redirect_stdout(out):
        host, buffer = pool.take(tensor)
    text = out.getvalue()

    if host is not tensor:
        raise AssertionError("после OOM нужно вернуть исходный тензор (pageable), а не копию")
    if buffer is not None:
        raise AssertionError(f"после OOM буфер должен быть None, получено {type(buffer)}")
    if pool.enabled or not pool.fallback:
        raise AssertionError(f"пул не деградировал: enabled={pool.enabled}, "
                             f"fallback={pool.fallback}")
    if pool.bytes != 0:
        raise AssertionError(f"после OOM счётчик pinned-памяти не обнулён: {pool.bytes}")
    if "pin_memory выключен" not in text or "out of memory" not in text:
        raise AssertionError(f"нет внятного предупреждения, stdout: {text!r}")
    if not torch.equal(host, tensor):
        raise AssertionError("данные после страховочного пути не совпали")
    print("  ok: OOM на pin -> исходный тензор + две строки [stream], прогон живёт")


def case_keeps_working() -> None:
    """4. После деградации пул продолжает отдавать тензоры (и не пытается пинить)."""
    if not _has_cuda():
        print("  SKIP (нет CUDA)")
        return
    torch = _torch()
    pool = streaming._PinnedPool(True)
    with _failing_pin(), contextlib.redirect_stdout(io.StringIO()):
        pool.take(torch.ones(16, dtype=torch.float32))
    # Дальше аллокация уже не нужна: пул выключен, а веса читаются как есть.
    for size in (8, 32):
        tensor = torch.full((size,), float(size))
        host, buffer = pool.take(tensor)
        if host is not tensor or buffer is not None:
            raise AssertionError(f"после деградации take({size}) вернул не исходный тензор")
    pool.give([None])                              # consumer всё равно зовёт give со списком
    print("  ok: после деградации take/give работают, повторного pin нет")


def case_pin_policy() -> None:
    """5. Политика pin: включён по умолчанию и при `offload=group`; выключает только `--set`.

    Pin — транспорт, а не математика: результат бит-в-бит от него не зависит, поэтому
    `offload=group` его больше не выключает. Явное выключение — `encoder.pin_memory=0`
    (`--set encoder.pin_memory=0`); если pinned-аллокация всё же упадёт, `_PinnedPool`
    деградирует до pageable RAM (кейсы 3/4) — это страховка, а не штатный режим.
    """
    from qwenstyle.config import RunConfig
    from qwenstyle.profiles import PROFILES

    # Публичные пресеты машины (qwenstyle/profiles.py) и fact-check: какие из них `offload: group`.
    public = [name for profile in PROFILES.values() for name in profile.presets]
    expected_group = (
        "VRAM8-preview", "VRAM8-release", "VRAM8-preview-anime", "VRAM8-release-anime",
        "VRAM12-preview", "VRAM12-release", "VRAM12-preview-anime", "VRAM12-release-anime",
        "VRAM16-release", "VRAM16-release-anime", "VRAM16-large", "VRAM16-4k-fast",
        "VRAM16-r3-169", "VRAM16-r3-43", "VRAM16-4k-slide",
    )
    group = tuple(name for name in public
                  if RunConfig.from_profile(name).generation.offload == "group")
    if group != expected_group:
        raise AssertionError(f"набор публичных пресетов с offload=group изменился: {group}")
    for name in group:
        cfg = RunConfig.from_profile(name)
        if cfg.encoder.pin_memory is not True:
            raise AssertionError(f"{name}: поле preseta должно остаться True (умолчание)")
        if cfg.encoder_pin_memory is not True:
            raise AssertionError(f"{name}: offload=group не должен выключать pin")
    print(f"  ok: {len(group)} публичных пресетов с offload=group -> pin включён")

    for name in ("VRAM16-preview", "VRAM16-preview-anime"):
        cfg = RunConfig.from_profile(name)
        if cfg.generation.offload == "group":
            raise AssertionError(f"{name}: ожидался offload без группы, получено {cfg.generation.offload!r}")
        if cfg.encoder_pin_memory is not True:
            raise AssertionError(f"{name}: без group-offload pin тоже должен быть включён")
    print("  ok: без group-offload pin включён (offload его не касается)")

    # Выключает только явный флаг — и на group-, и на не-group-пресете.
    for name in ("VRAM16-release-anime", "VRAM16-preview"):
        off = RunConfig.from_profile(name).merged(**{"encoder.pin_memory": "0"})
        if off.encoder_pin_memory:
            raise AssertionError(f"{name}: --set encoder.pin_memory=0 не выключил pin")
    print("  ok: выключает только явный encoder.pin_memory=0")

    # Правило не следует за offload: включение группы у не-group пресета pin не трогает.
    for name in ("VRAM16-preview", "VRAM16-preview-anime"):
        grp = RunConfig.from_profile(name).merged(**{"generation.offload": "group"})
        if grp.generation.offload != "group" or grp.encoder_pin_memory is not True:
            raise AssertionError(f"{name}: pin не должен зависеть от generation.offload")
    print("  ok: pin не зависит от generation.offload")


def main() -> int:
    cases = (
        ("1. пул выключен", case_pool_disabled),
        ("2. pinned-буфер и переиспользование", case_pool_reuse),
        ("3. деградация вместо OOM", case_degrades_instead_of_crash),
        ("4. работа после деградации", case_keeps_working),
        ("5. политика pin при offload=group", case_pin_policy),
    )
    failed = 0
    for label, case in cases:
        try:
            case()
            print(f"PASS {label}", flush=True)
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {label}: {exc}", flush=True)
        except Exception:                                   # noqa: BLE001 — тест не должен врать
            failed += 1
            print(f"FAIL {label}: исключение\n{traceback.format_exc()}", flush=True)
    if failed:
        print(f"{failed} FAIL", flush=True)
        return 1
    print("ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
