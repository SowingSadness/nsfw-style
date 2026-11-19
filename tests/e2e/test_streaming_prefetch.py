"""Тесты конвейера стриминга (`qwenstyle/streaming.py`, шаг «GPU-стриминг с RAM-стейджингом»).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_streaming_prefetch.py`
Успех: строка `PASS <кейс>` на кейс, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.
Без CUDA печатается `SKIP (нет CUDA)`, GPU-кейс не считается провалом и exit всё равно 0.

Кейсы (по плану шага 1):

1. `cuda`: `prefetch=0` vs `prefetch=4` — **бит-в-бит** те же `hidden_states` и `logits`
   (`torch.equal`, а не `allclose`): конвейер — раскладка памяти, а не другая математика;
2. `cpu`: тот же прогон с `prefetch=4` (конвейер на CPU игнорируется, печатается одна строка) —
   результат равен `prefetch=0`;
3. `StreamStats`: `bytes_read > 0`, `t_total > 0`, `t_wait <= t_total + 1e-3` (и просто строка
   `summary()` с ожидаемыми полями);
4. продюсер завершается: после двух последовательных `forward()` число живых потоков не растёт,
   а поток `qwenstyle-stream-producer` не остаётся жить.

Модель крошечная и собирается **импортом** билдера из `tests/test_streaming.py` (не копией):
`build_tiny`/`tiny_config`/`temp_base` — те же, что в тесте диск-стриминга.
"""
from __future__ import annotations

import gc
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, TESTS_DIR):            # запуск из любого cwd; билдер — из соседнего теста
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
sys.dont_write_bytecode = True                  # тест не должен оставлять __pycache__ в пакете

import test_streaming as donor                 # noqa: E402 — билдер мини-модели и temp_base

from qwenstyle.streaming import (              # noqa: E402 — после правки sys.path
    StreamStats,
    load_text_encoder,
    stream_stats,
)

PRODUCER = "qwenstyle-stream-producer"
TINY = donor.TINY


# --------------------------------------------------------------------------- #
# Общие помощники
# --------------------------------------------------------------------------- #

def _cuda() -> bool:
    """Есть ли CUDA (без неё GPU-кейс печатает `SKIP`)."""
    import torch

    return bool(torch.cuda.is_available())


def _run(root: Path, *, dtype: str, device: str, prefetch: int,
         pin_memory: bool = True, stats: StreamStats | None = None) -> Any:
    """Загрузить мини-модель стримингом с заданными флагами конвейера."""
    return load_text_encoder(
        _model_cls(), root, dtype=dtype, local_files_only=True, stream=True,
        device=device, prefetch=prefetch, pin_memory=pin_memory, lookahead=1, stats=stats,
    )


def _model_cls() -> Any:
    from transformers import Qwen3ForCausalLM

    return Qwen3ForCausalLM


def _inputs(device: str) -> Any:
    """Те же `input_ids` на каждом прогоне (сравнение идёт бит-в-бит)."""
    import torch

    torch.manual_seed(123)
    return torch.randint(0, TINY["vocab_size"], (1, 7), device=device)


def _forward(model: Any, input_ids: Any) -> tuple[tuple[Any, ...], Any]:
    """Один прогон: hidden states + logits (отсоединённые копии на CPU)."""
    import torch

    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)
    hidden = tuple(state.detach() for state in out.hidden_states)
    logits = out.logits.detach()
    del out
    if str(input_ids.device).split(":")[0] != "cpu":   # держать ссылки на GPU-активации незачем
        torch.cuda.synchronize()
    return hidden, logits


def _compare(label: str, expected: tuple[tuple[Any, ...], Any],
             actual: tuple[tuple[Any, ...], Any]) -> None:
    """`torch.equal` по каждому hidden state и по logits (не allclose: нужна бит-в-бит)."""
    import torch

    hidden_a, logits_a = expected
    hidden_b, logits_b = actual
    if len(hidden_a) != len(hidden_b):
        raise AssertionError(f"{label}: слоёв {len(hidden_b)}, ожидалось {len(hidden_a)}")
    for layer, (left, right) in enumerate(zip(hidden_a, hidden_b)):
        if not torch.equal(left, right):
            diff = (left.float() - right.float()).abs().max().item()
            raise AssertionError(f"{label}: hidden_states[{layer}] не бит-в-бит "
                                 f"(max abs diff {diff})")
    if not torch.equal(logits_a, logits_b):
        diff = (logits_a.float() - logits_b.float()).abs().max().item()
        raise AssertionError(f"{label}: logits не бит-в-бит (max abs diff {diff})")
    print(f"  ok: {label}: {len(hidden_b)} hidden states и logits бит-в-бит", flush=True)


def _check_stats(label: str, stats: StreamStats | None) -> None:
    """`StreamStats` заполняется всегда: байты, время, и `t_wait` не больше целого прогона."""
    if stats is None:
        raise AssertionError(f"{label}: stream_stats(model) не вернул StreamStats")
    if stats.bytes_read <= 0:
        raise AssertionError(f"{label}: bytes_read={stats.bytes_read} (веса не читались?)")
    if stats.t_total <= 0:
        raise AssertionError(f"{label}: t_total={stats.t_total} (forward не измерен)")
    if stats.t_wait > stats.t_total + 1e-3:
        raise AssertionError(f"{label}: t_wait={stats.t_wait} > t_total={stats.t_total}")
    if stats.tensors <= 0 or stats.modules <= 0:
        raise AssertionError(f"{label}: tensors={stats.tensors}, modules={stats.modules}")
    line = stats.summary()
    for needle in ("[stream] ", "read=", "copy=", "pin=", "compute=", "wait=", "total=",
                   "ratio_wait="):
        if needle not in line:
            raise AssertionError(f"{label}: в summary() нет {needle!r}: {line}")
    print(f"  ok: {label}: {line}", flush=True)


def _close(model: Any) -> None:
    """Закрыть mmap-шарды и отпустить модель (Windows не даёт удалить открытый файл)."""
    donor._close(model)


# --------------------------------------------------------------------------- #
# Кейс 1: cuda, prefetch=0 vs prefetch=4 — бит-в-бит
# --------------------------------------------------------------------------- #

def case_cuda_pipeline(root: Path, shared: dict[str, Any]) -> None:
    """`prefetch=4` на GPU обязан дать `torch.equal` с `prefetch=0` (baseline)."""
    if not _cuda():
        print("SKIP (нет CUDA)", flush=True)
        return
    import torch

    device = "cuda"
    dtype = "bfloat16"
    input_ids = _inputs(device)
    stats_sync = StreamStats()
    stats_pipe = StreamStats()
    plain = _run(root, dtype=dtype, device=device, prefetch=0, stats=stats_sync)
    piped = None
    try:
        piped = _run(root, dtype=dtype, device=device, prefetch=4, stats=stats_pipe)
        expected = _forward(plain, input_ids)
        actual = _forward(piped, input_ids)
        _compare(f"cuda {dtype}: prefetch=0 vs prefetch=4", expected, actual)
        again = _forward(piped, input_ids)                 # второй прогон: продюсер снова жив
        _compare(f"cuda {dtype}: повторный прогон prefetch=4", expected, again)
        _check_stats("cuda prefetch=0", stream_stats(plain))
        _check_stats("cuda prefetch=4", stream_stats(piped))
        shared["piped"] = piped                            # для кейса 4 (потоки)
        shared["input_ids"] = input_ids
        print(f"  ok: cuda {dtype}: два прогона конвейера бит-в-бит, "
              f"peak VRAM={torch.cuda.max_memory_allocated() / 1024 ** 2:.1f} MiB", flush=True)
    finally:
        if piped is not None and shared.get("piped") is not piped:
            _close(piped)
        _close(plain)
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Кейс 2: cpu, prefetch игнорируется, но не падает
# --------------------------------------------------------------------------- #

def case_cpu_ignores_prefetch(root: Path, shared: dict[str, Any]) -> None:
    """`prefetch=4` на CPU — тот же синхронный путь: результат равен `prefetch=0`."""
    device = "cpu"
    dtype = "float32"
    input_ids = _inputs(device)
    stats_sync = StreamStats()
    stats_pipe = StreamStats()
    plain = _run(root, dtype=dtype, device=device, prefetch=0, stats=stats_sync)
    piped = None
    try:
        piped = _run(root, dtype=dtype, device=device, prefetch=4, stats=stats_pipe)
        expected = _forward(plain, input_ids)
        actual = _forward(piped, input_ids)
        _compare(f"cpu {dtype}: prefetch=0 vs prefetch=4", expected, actual)
        _check_stats("cpu prefetch=0", stream_stats(plain))
        _check_stats("cpu prefetch=4", stream_stats(piped))
    finally:
        if piped is not None:
            _close(piped)
        _close(plain)


# --------------------------------------------------------------------------- #
# Кейс 3: статистика плеч (на том же конвейере, что в кейсе 1)
# --------------------------------------------------------------------------- #

def case_stats(root: Path, shared: dict[str, Any]) -> None:
    """`StreamStats` заполняется всегда; на GPU с конвейером есть и `pin`, и `copy`."""
    piped = shared.get("piped")
    if piped is None:
        # CUDA нет: проверяем заполнение статистики на CPU-пути (он есть всегда)
        model = _run(root, dtype="float32", device="cpu", prefetch=0, stats=StreamStats())
        try:
            _forward(model, _inputs("cpu"))
            _check_stats("cpu baseline", stream_stats(model))
        finally:
            _close(model)
        print("SKIP (нет CUDA): плечи pin/copy конвейера не замерены", flush=True)
        return
    stats = stream_stats(piped)
    _check_stats("cuda prefetch=4", stats)
    if stats.t_copy <= 0:
        raise AssertionError(f"конвейер не выдавал H2D-копий: copy={stats.t_copy}")
    ratio = stats.t_wait / stats.t_total if stats.t_total > 0 else 0.0
    print(f"  ok: плечи конвейера: pin={stats.t_pin:.3f}s copy={stats.t_copy:.3f}s "
          f"ratio_wait={ratio:.2f}", flush=True)


# --------------------------------------------------------------------------- #
# Кейс 4: продюсер не течёт
# --------------------------------------------------------------------------- #

def case_no_thread_leak(root: Path, shared: dict[str, Any]) -> None:
    """После двух подряд forward число живых потоков не растёт, продюсер завершён."""
    piped = shared.get("piped")
    if piped is None:
        model = _run(root, dtype="float32", device="cpu", prefetch=4, stats=StreamStats())
        device = "cpu"
    else:
        model = piped
        device = "cuda"
    try:
        input_ids = shared["input_ids"] if "input_ids" in shared else _inputs(device)
        before = threading.active_count()
        _forward(model, input_ids)
        _forward(model, input_ids)
        after = threading.active_count()
        alive = [thread.name for thread in threading.enumerate() if thread.name == PRODUCER]
        if after > before:
            raise AssertionError(f"потоков стало больше: {before} -> {after} "
                                 f"(живые: {[t.name for t in threading.enumerate()]})")
        if alive:
            raise AssertionError(f"поток-продюсер остался жить: {alive}")
        print(f"  ok: потоков до={before} после={after}, продюсер завершён", flush=True)
    finally:
        _close(model)
        shared.pop("piped", None)
        gc.collect()


# --------------------------------------------------------------------------- #
# Запуск
# --------------------------------------------------------------------------- #

CASES: tuple[tuple[str, Callable[[Path, dict[str, Any]], None]], ...] = (
    ("1. cuda: prefetch=0 vs prefetch=4 бит-в-бит (и повторный прогон)", case_cuda_pipeline),
    ("2. cpu: prefetch игнорируется, результат равен prefetch=0", case_cpu_ignores_prefetch),
    ("3. StreamStats: bytes_read/t_total/t_wait и summary()", case_stats),
    ("4. продюсер завершается: после двух forward потоков не больше", case_no_thread_leak),
)


def main() -> int:
    failed = 0
    shared: dict[str, Any] = {}
    base = donor.temp_base()
    with tempfile.TemporaryDirectory(prefix="qwenstyle-prefetch-test-", dir=str(base)) as raw:
        tmp = Path(raw)
        print(f"[tmp] {tmp} (TemporaryDirectory, будет удалён)", flush=True)
        root = tmp / "tiny"
        donor.build_tiny(root)                  # одна мини-модель на все кейсы
        try:
            for name, case in CASES:
                try:
                    case(root, shared)
                except Exception as exc:        # noqa: BLE001 — тест-скрипт печатает и идёт дальше
                    print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
                    failed += 1
                else:
                    print(f"PASS {name}", flush=True)
        finally:
            leftover = shared.pop("piped", None)
            if leftover is not None:
                _close(leftover)
    if failed:
        print(f"FAIL {failed} из {len(CASES)} кейсов", flush=True)
        return 1
    print("ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
