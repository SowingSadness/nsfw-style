"""Тесты позиционного чтения шардов (`qwenstyle/shards.py`) — обычный скрипт, без pytest.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_shard_reader.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Кейсы (по плану шага 1 «Модуль позиционного чтения шардов»):

1. round-trip: `save_file` пишет мини-тензоры (float32, bfloat16, int64, float16) -> `read` даёт
   **бит-в-бит** те же данные (`torch.equal`, не `allclose`), формы и dtype совпадают;
2. `weight_map` на 2 шарда (имена перемешаны, как в `tests/test_streaming.py::shard_copy`) -> каждое
   имя читается из своего файла; `keys()`/`__len__`/`__contains__`/`__getitem__` (описание, не тензор);
3. `read_into` в буфер большего размера читает точно `nbytes` и не задевает хвост; `at`-смещение
   работает; короткий `view` -> `ValueError`;
4. битые файлы (обрезка, `data_offsets` шире файла, `data_offsets` не по форме) -> `ShardFormatError`
   с путём; `save_file` с `metadata` читается, `__metadata__` в имена не попадает;
5. два потока читают один `ShardFile` разные тензоры по 200 раз -> все результаты `torch.equal`
   эталону (замок держит `seek + readinto` целиком);
6. `close()` идемпотентен, после него `read`/`read_into` -> `RuntimeError`, `info()` работает;
7. отсутствующее имя -> `KeyError` с примерами имён;
8. битая `weight_map` (тензора нет в указанном ею файле / файла из карты нет среди шардов) ->
   `ShardFormatError` с именем искомого файла и перечнем доступных шардов.

Шарды пишутся `safetensors.torch.save_file` в системный temp (`TemporaryDirectory`), веса проекта
и сеть не трогаются.
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:              # запуск из любого cwd
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                  # тест не должен оставлять __pycache__ в пакете

from qwenstyle.shards import (                  # noqa: E402 — после правки sys.path
    ShardFile,
    ShardFormatError,
    ShardSet,
    TensorInfo,
)


# --------------------------------------------------------------------------- #
# Фикстуры и утилиты
# --------------------------------------------------------------------------- #

def temp_base() -> Path:
    """База для `TemporaryDirectory`: системный temp, если он вне репозитория.

    На этой машине `%TEMP%` может оказаться внутри репозитория (песочница), и
    `TemporaryDirectory()` без `dir=` создал бы каталог в корне. Поэтому базу выбираем явно.
    """
    candidate = Path(tempfile.gettempdir())
    try:
        resolved = candidate.resolve()
    except OSError:                             # pragma: no cover — недоступный путь
        resolved = candidate
    if resolved != REPO_ROOT and REPO_ROOT not in resolved.parents:
        return resolved
    return REPO_ROOT / "tests"


def sample_tensors() -> dict[str, Any]:
    """Мини-тензоры всех «обычных» dtype набора весов (bf16/fp16/fp32/int64)."""
    import torch

    torch.manual_seed(7)
    return {
        "f32": torch.randn(7, 5, dtype=torch.float32),
        "bf16": torch.randn(3, 11, dtype=torch.bfloat16),
        "i64": torch.randint(-1000, 1000, (4, 3), dtype=torch.int64),
        "f16": torch.randn(2, 3, 4, dtype=torch.float16),
    }


def raw_bytes(tensor: Any) -> bytes:
    """Байты тензора как есть (сравнение идёт по байтам, dtype не приводится)."""
    import torch

    flat = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    return bytes(memoryview(flat.numpy()))


def write_shards(directory: Path, state: dict[str, Any], shards: int = 2,
                 metadata: dict[str, str] | None = None) -> tuple[list[Path], dict[str, str]]:
    """Разложить тензоры на `shards` файлов + вернуть (файлы, weight_map)."""
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    names = list(state)
    files: list[Path] = []
    weight_map: dict[str, str] = {}
    for index in range(shards):
        chunk = names[index::shards]                  # перемешиваем имена между шардами
        filename = f"model-{index + 1:05d}-of-{shards:05d}.safetensors"
        path = directory / filename
        save_file({name: state[name] for name in chunk}, str(path), metadata=metadata)
        files.append(path)
        weight_map.update({name: filename for name in chunk})
    return files, weight_map


def write_header_only(path: Path, header: dict[str, Any], payload: bytes) -> None:
    """Записать safetensors «руками»: битый заголовок не соберёт `save_file`."""
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def expect_format_error(path: Path) -> ShardFormatError:
    """Открыть заведомо битый файл и вернуть `ShardFormatError` (падать, если его нет)."""
    try:
        ShardFile(path)
    except ShardFormatError as exc:
        if path.name not in str(exc):
            raise AssertionError(f"в тексте ошибки нет имени файла {path.name!r}: {exc}") from exc
        return exc
    raise AssertionError(f"{path.name}: битый файл открылся без ShardFormatError")


# --------------------------------------------------------------------------- #
# Кейс 1: round-trip байт-в-байт
# --------------------------------------------------------------------------- #

def case_roundtrip(tmp: Path) -> None:
    """`read` возвращает те же байты, форму и dtype, что писал `save_file`."""
    import torch

    root = tmp / "case1"
    state = sample_tensors()
    files, _ = write_shards(root, state, shards=1)
    shards = ShardSet(files)
    try:
        for name, expected in state.items():
            info = shards.info(name)
            got = shards.read(name)
            if got.dtype != expected.dtype:
                raise AssertionError(f"{name}: dtype {got.dtype}, ожидался {expected.dtype}")
            if tuple(got.shape) != tuple(expected.shape):
                raise AssertionError(f"{name}: форма {tuple(got.shape)}, ожидалась "
                                     f"{tuple(expected.shape)}")
            if not torch.equal(got, expected):
                raise AssertionError(f"{name}: данные не бит-в-бит")
            if got.stride() != expected.stride():
                raise AssertionError(f"{name}: stride {got.stride()}, ожидался "
                                     f"{expected.stride()} — тензор прочитан не в layout файла")
            if info.nbytes != expected.numel() * expected.element_size():
                raise AssertionError(f"{name}: nbytes={info.nbytes} не совпал с формой")
            print(f"  ok: {name} ({info.file_dtype} -> {got.dtype}, форма {tuple(got.shape)}, "
                  f"{info.nbytes} байт) бит-в-бит")
        if len(shards) != len(state):
            raise AssertionError(f"в наборе {len(shards)} имён, ожидалось {len(state)}")
    finally:
        shards.close()


# --------------------------------------------------------------------------- #
# Кейс 2: weight_map на два шарда
# --------------------------------------------------------------------------- #

def case_weight_map(tmp: Path) -> None:
    """`weight_map` выбирает файл по имени; `__getitem__` отдаёт описание, а не тензор."""
    import torch

    root = tmp / "case2"
    state = sample_tensors()
    files, weight_map = write_shards(root, state, shards=2)
    shards = ShardSet(files, weight_map)
    try:
        if len(shards) != len(state):
            raise AssertionError(f"в наборе {len(shards)} имён, ожидалось {len(state)}")
        if set(shards.keys()) != set(state):
            raise AssertionError(f"имена набора {sorted(shards.keys())} != {sorted(state)}")
        if "f32" not in shards or "нет.такого" in shards:
            raise AssertionError("__contains__ отвечает неверно")
        for name, expected in state.items():
            info = shards[name]
            if not isinstance(info, TensorInfo):
                raise AssertionError(f"{name}: __getitem__ вернул {type(info).__name__}, "
                                     f"ожидался TensorInfo")
            if info.shard.name != weight_map[name]:
                raise AssertionError(f"{name}: лежит в {info.shard.name}, а weight_map говорит "
                                     f"{weight_map[name]}")
            if shards.shard_path(name) != info.shard:
                raise AssertionError(f"{name}: shard_path() расходится с TensorInfo.shard")
            if not torch.equal(shards.read(name), expected):
                raise AssertionError(f"{name}: данные из {info.shard.name} не бит-в-бит")
        dtypes = shards.file_dtypes()
        if dtypes != {"f32": "F32", "bf16": "BF16", "i64": "I64", "f16": "F16"}:
            raise AssertionError(f"file_dtypes() вернул {dtypes}")
        print(f"  ok: {len(shards)} имён на 2 шарда, каждое читается из своего файла "
              f"({', '.join(sorted({info.shard.name for info in shards.values()}))})")
        print(f"  ok: file_dtypes() = {dtypes}")
    finally:
        shards.close()


# --------------------------------------------------------------------------- #
# Кейс 3: read_into в готовый буфер
# --------------------------------------------------------------------------- #

def case_read_into(tmp: Path) -> None:
    """Точно `nbytes` в буфер большего размера, `at`-смещение, короткий буфер -> ValueError."""
    import torch

    root = tmp / "case3"
    state = sample_tensors()
    files, _ = write_shards(root, state, shards=1)
    shard = ShardFile(files[0])
    try:
        name = "bf16"
        info = shard.info(name)
        reference = raw_bytes(shard.read(name))

        guard = 64
        buffer = torch.full((info.nbytes + guard,), 0xAA, dtype=torch.uint8)
        got = shard.read_into(name, memoryview(buffer.numpy()))
        if got != info.nbytes:
            raise AssertionError(f"read_into вернул {got}, ожидалось {info.nbytes}")
        if raw_bytes(buffer[:info.nbytes]) != reference:
            raise AssertionError("байты в буфере не совпали с read()")
        if not bool((buffer[info.nbytes:] == 0xAA).all()):
            raise AssertionError("read_into задел хвост буфера за пределами тензора")
        print(f"  ok: read_into в буфер +{guard} байт: ровно {info.nbytes} байт, хвост цел")

        offset = 16
        shifted = torch.full((offset + info.nbytes + guard,), 0xBB, dtype=torch.uint8)
        shard.read_into(name, memoryview(shifted.numpy()), at=offset)
        if raw_bytes(shifted[offset:offset + info.nbytes]) != reference:
            raise AssertionError("read_into с at= не совпал с read()")
        if not bool((shifted[:offset] == 0xBB).all()) or \
                not bool((shifted[offset + info.nbytes:] == 0xBB).all()):
            raise AssertionError("read_into с at= задел байты до/после окна")
        print(f"  ok: read_into(at={offset}) пишет только своё окно")

        short = memoryview(torch.empty(info.nbytes - 1, dtype=torch.uint8).numpy())
        for label, action in (
            ("короче тензора на 1 байт", lambda: shard.read_into(name, short)),
            ("at съедает место в буфере",
             lambda: shard.read_into(name, memoryview(torch.empty(info.nbytes,
                                                                 dtype=torch.uint8).numpy()),
                                    at=8)),
        ):
            try:
                action()
            except ValueError as exc:
                print(f"  ok: короткий буфер ({label}) -> ValueError: {str(exc)[:60]}…")
            else:
                raise AssertionError(f"короткий буфер ({label}) не дал ValueError")
    finally:
        shard.close()


# --------------------------------------------------------------------------- #
# Кейс 4: битые файлы и метаданные
# --------------------------------------------------------------------------- #

def case_broken_and_metadata(tmp: Path) -> None:
    """Обрезка/несовпадение offsets -> ShardFormatError; `__metadata__` в имена не попадает."""
    from safetensors.torch import save_file

    root = tmp / "case4"
    root.mkdir(parents=True, exist_ok=True)

    truncated = root / "truncated.safetensors"
    save_file({"a": sample_tensors()["f32"]}, str(truncated))
    truncated.write_bytes(truncated.read_bytes()[:-9])        # обрезка последних байт
    error = expect_format_error(truncated)
    print(f"  ok: обрезанный файл -> ShardFormatError: {str(error)[:80]}…")

    wide = root / "offsets-wide.safetensors"
    write_header_only(wide, {"t": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}},
                      b"\x00" * 8)                            # данных меньше, чем объявлено
    error = expect_format_error(wide)
    print(f"  ok: data_offsets шире файла -> ShardFormatError: {str(error)[:80]}…")

    wrong_shape = root / "offsets-shape.safetensors"
    write_header_only(wrong_shape,
                      {"t": {"dtype": "F32", "shape": [4], "data_offsets": [0, 8]}},
                      b"\x00" * 16)                           # форма просит 16 байт, дано 8
    error = expect_format_error(wrong_shape)
    print(f"  ok: data_offsets не по форме -> ShardFormatError: {str(error)[:80]}…")

    unknown_dtype = root / "unknown-dtype.safetensors"
    write_header_only(unknown_dtype,
                      {"t": {"dtype": "F4", "shape": [4], "data_offsets": [0, 16]}},
                      b"\x00" * 16)
    error = expect_format_error(unknown_dtype)
    print(f"  ok: неизвестный dtype -> ShardFormatError: {str(error)[:80]}…")

    with_meta = root / "with-metadata.safetensors"
    state = sample_tensors()
    save_file(state, str(with_meta), metadata={"format": "pt", "note": "metadata не тензор"})
    shard = ShardFile(with_meta)
    try:
        if "__metadata__" in shard:
            raise AssertionError("__metadata__ попал в имена тензоров")
        if set(shard.tensor_names) != set(state):
            raise AssertionError(f"имена с metadata: {shard.tensor_names}")
        print(f"  ok: save_file(metadata=…) читается, __metadata__ в имена не попал "
              f"({len(shard)} тензоров)")
    finally:
        shard.close()

    empty_description = root / "no-fields.safetensors"
    write_header_only(empty_description, {"t": {"dtype": "F32"}}, b"\x00" * 16)
    expect_format_error(empty_description)
    print("  ok: описание без data_offsets -> ShardFormatError")


# --------------------------------------------------------------------------- #
# Кейс 5: два потока на одном файле
# --------------------------------------------------------------------------- #

THREAD_ROUNDS = 200


def case_threads(tmp: Path) -> None:
    """Два потока читают один и тот же файл разные тензоры — замок держит `seek + readinto`."""
    import torch

    root = tmp / "case5"
    root.mkdir(parents=True, exist_ok=True)
    size = 64 * 1024
    state = {
        "a": (torch.arange(size, dtype=torch.int64) % 251).to(torch.uint8),
        "b": (255 - (torch.arange(size, dtype=torch.int64) % 241)).to(torch.uint8),
    }
    files, _ = write_shards(root, state, shards=1)
    shard = ShardFile(files[0])
    try:
        reference = {name: shard.read(name) for name in state}
        failures: list[str] = []
        barrier = threading.Barrier(2)

        def worker(name: str) -> None:
            barrier.wait()                        # стартуем одновременно: гонка наиболее вероятна
            for _ in range(THREAD_ROUNDS):
                try:
                    got = shard.read(name)
                except Exception as exc:          # noqa: BLE001 — тест собирает все сбои
                    failures.append(f"{name}: {type(exc).__name__}: {exc}")
                    return
                if not torch.equal(got, reference[name]):
                    failures.append(f"{name}: данные не бит-в-бит (перепутались с другим потоком?)")
                    return

        threads = [threading.Thread(target=worker, args=(name,)) for name in state]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if failures:
            raise AssertionError("; ".join(failures[:3]))
        print(f"  ok: 2 потока × {THREAD_ROUNDS} чтений одного файла "
              f"({size} байт × 2 тензора) — все бит-в-бит")
    finally:
        shard.close()


# --------------------------------------------------------------------------- #
# Кейс 6: close()
# --------------------------------------------------------------------------- #

def case_close(tmp: Path) -> None:
    """`close()` идемпотентен; после него `info()` жив, а чтение -> RuntimeError."""
    root = tmp / "case6"
    state = sample_tensors()
    files, weight_map = write_shards(root, state, shards=2)
    shards = ShardSet(files, weight_map)
    try:
        info = shards.info("bf16")
        shards.close()
        shards.close()                            # идемпотентно
        if not shards.closed:
            raise AssertionError("после close() набор считает себя открытым")
        for shard in shards._files.values():      # noqa: SLF001 — проверяем закрытие дескрипторов
            if not shard.closed:
                raise AssertionError(f"файл {shard.path.name} остался открытым")
        if shards.info("bf16") != info:
            raise AssertionError("info() после close() вернул другое описание")
        if len(shards) != len(state):
            raise AssertionError("после close() пропали имена")
        buffer = memoryview(bytearray(info.nbytes))
        for label, action in (
            ("read", lambda: shards.read("bf16")),
            ("read_into", lambda: shards.read_into("bf16", buffer)),
        ):
            try:
                action()
            except RuntimeError as exc:
                print(f"  ok: после close() {label} -> RuntimeError: {str(exc)[:60]}…")
            else:
                raise AssertionError(f"после close() {label} продолжает читать веса")
        print(f"  ok: close() идемпотентен, info()/len() после него работают "
              f"({len(shards)} имён, {len(files)} шардов)")
    finally:
        shards.close()


# --------------------------------------------------------------------------- #
# Кейс 7: неизвестное имя
# --------------------------------------------------------------------------- #

def case_unknown_name(tmp: Path) -> None:
    """`KeyError` с примерами имён — и у файла, и у набора."""
    root = tmp / "case7"
    state = sample_tensors()
    files, weight_map = write_shards(root, state, shards=2)
    shards = ShardSet(files, weight_map)
    single = ShardFile(files[0])
    try:
        for label, action in (
            ("ShardSet.info", lambda: shards.info("no.such.tensor")),
            ("ShardSet.read", lambda: shards.read("no.such.tensor")),
            ("ShardFile.info", lambda: single.info("no.such.tensor")),
        ):
            try:
                action()
            except KeyError as exc:
                text = str(exc)
                if "Примеры имён" not in text:
                    raise AssertionError(f"{label}: KeyError без примеров имён: {text}") from exc
                if not any(name in text for name in state):
                    raise AssertionError(f"{label}: в KeyError нет ни одного реального имени: "
                                         f"{text}") from exc
                print(f"  ok: {label} с неизвестным именем -> KeyError с примерами")
            else:
                raise AssertionError(f"{label}: неизвестное имя не дало KeyError")
    finally:
        single.close()
        shards.close()


# --------------------------------------------------------------------------- #
# Кейс 8: битая weight_map
# --------------------------------------------------------------------------- #

def case_broken_weight_map(tmp: Path) -> None:
    """Карта, которая врёт о файле, -> `ShardFormatError`, а не `KeyError` и не пустой набор."""
    from safetensors.torch import save_file

    root = tmp / "case8"
    root.mkdir(parents=True, exist_ok=True)
    state = sample_tensors()
    save_file({"w1": state["f32"]}, str(root / "a.safetensors"))
    save_file({"w2": state["bf16"]}, str(root / "b.safetensors"))
    files = [root / "a.safetensors", root / "b.safetensors"]
    available = ", ".join(path.name for path in files)

    # (а) w2 объявлен в a.safetensors, но лежит в b.safetensors
    for label, weight_map, sought in (
        ("тензора нет в указанном картой файле",
         {"w1": "a.safetensors", "w2": "a.safetensors"}, "a.safetensors"),
        ("файла из карты нет среди шардов",
         {"w1": "a.safetensors", "w2": "c.safetensors"}, "c.safetensors"),
    ):
        try:
            shards = ShardSet(files, weight_map)
        except ShardFormatError as exc:
            text = str(exc)
            if sought not in text:
                raise AssertionError(f"{label}: в ошибке нет имени файла {sought!r}: "
                                     f"{text}") from exc
            for name in files:
                if name.name not in text:
                    raise AssertionError(f"{label}: в ошибке нет доступного шарда {name.name!r}: "
                                         f"{text}") from exc
            if available not in text:
                raise AssertionError(f"{label}: перечень доступных шардов не совпал: "
                                     f"{text}") from exc
            print(f"  ok: {label} -> ShardFormatError: {text[:90]}…")
        except KeyError as exc:
            raise AssertionError(f"{label}: битая weight_map дала KeyError вместо "
                                 f"ShardFormatError: {exc}") from exc
        else:
            shards.close()
            raise AssertionError(f"{label}: битая weight_map принята молча "
                                 f"({len(shards)} имён в наборе)")

    # карта, которая не врёт, по-прежнему работает (сбой выше — про карту, а не про файлы)
    shards = ShardSet(files, {"w1": "a.safetensors", "w2": "b.safetensors"})
    try:
        if sorted(shards.keys()) != ["w1", "w2"]:
            raise AssertionError(f"верная карта дала имена {sorted(shards.keys())}")
        print("  ok: верная weight_map на тех же 2 шардах собирается (w1, w2)")
    finally:
        shards.close()


# --------------------------------------------------------------------------- #
# Запуск
# --------------------------------------------------------------------------- #

CASES: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("1. round-trip: read() бит-в-бит для f32/bf16/i64/f16", case_roundtrip),
    ("2. weight_map на 2 шарда: имя -> свой файл, __getitem__ отдаёт TensorInfo", case_weight_map),
    ("3. read_into: точный размер, at-смещение, короткий буфер -> ValueError", case_read_into),
    ("4. битые файлы -> ShardFormatError; __metadata__ не тензор", case_broken_and_metadata),
    ("5. два потока на одном ShardFile — данные не перепутываются", case_threads),
    ("6. close() идемпотентен, после него чтение -> RuntimeError, info() жив", case_close),
    ("7. неизвестное имя -> KeyError с примерами", case_unknown_name),
    ("8. битая weight_map -> ShardFormatError с именем файла и списком шардов",
     case_broken_weight_map),
)


def main() -> int:
    failed = 0
    base = temp_base()
    with tempfile.TemporaryDirectory(prefix="qwenstyle-shards-test-", dir=str(base)) as raw:
        tmp = Path(raw)
        print(f"[tmp] {tmp} (TemporaryDirectory, будет удалён)", flush=True)
        for name, case in CASES:
            try:
                case(tmp)
            except Exception as exc:            # noqa: BLE001 — тест-скрипт печатает и продолжает
                print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                failed += 1
            else:
                print(f"PASS {name}", flush=True)
    if failed:
        print(f"FAIL {failed} из {len(CASES)} кейсов", flush=True)
        return 1
    print("ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
