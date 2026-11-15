"""Политика вытеснения кэша эмбеддингов: отложенная очистка (без torch и GPU).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_cleanup.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (план `te-gpu-streaming`, шаг 3): новая пара (промт, картинка) делает прошлые записи этой
пары ненужными, но удалять их в момент записи нельзя — генерация ещё не прошла, кадр не сохранён.
Поэтому `qwenstyle/cleanup.py` копит `CacheWrite` и исполняет план в самом конце прогона.
Тест фиксирует таблицу «политика -> что попало в план удаления» на временном каталоге.

Кейсы:

1. `pair` = объединение `prompt` и `image`: пишем (md5₂, sha₁) при наличии (md5₁, sha₁) и
   (md5₂, sha₂) -> в плане **обе** записи (старая картинка того же промта тоже устарела);
   запись (md5₃, sha₃), не делящая ни промт, ни картинку, остаётся.
2. `prompt` не трогает записи того же промта с другой картинкой; `image` — наоборот; `none` — пусто.
3. Текущая пара и текущий `.pt` не удаляются никогда, даже при `pair`.
4. Варианты: `-dev-cuda` не попадает в план при записи `cpu`-варианта и наоборот; `fp16` — тоже.
5. Разные `crop` одной пары -> план пуст; `crop` в вытеснении не участвует.
6. `flush(apply=False)` печатает план и ничего не удаляет; `cache/fp8_transformer/dummy.pt`
   и прочие не-`embeds` файлы не трогаются ни при какой политике; план пересчитывается в `flush`.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                   # тест не должен оставлять __pycache__ в пакете

from qwenstyle.cache import EmbedsEntry, scan_embeds          # noqa: E402 — после правки sys.path
from qwenstyle.cleanup import CacheWrite, DeferredCleanup, normalize_policy  # noqa: E402

MD5_1 = "aaaa1111bbbb"
MD5_2 = "eeee2222ffff"
MD5_3 = "9999aaaabbbb"
SHA_1 = "cccc2222dddd"
SHA_2 = "1234aaaa5678"
SHA_3 = "fedc00001111"

POLICIES = ("none", "prompt", "image", "pair")


# --------------------------------------------------------------------------- #
# Фикстуры
# --------------------------------------------------------------------------- #

def _embeds_name(md5: str, sha: str | None, crop: str = "1376x768", *,
                 enc: str = "base", dtype: str = "bfloat16", device: str = "cpu") -> str:
    """Имя записи в формате шага 2 (сегменты dtype/device — только для неумолчаний)."""
    tail = "" if enc in ("", "base") else f"_te-{enc}"
    if dtype != "bfloat16":
        tail += "-fp16" if dtype == "float16" else f"-{dtype}"
    if device != "cpu":
        tail += f"-dev-{device}"
    sha_part = f"_{sha}" if sha else ""
    return f"embeds_{md5}{sha_part}{tail}_{crop}.pt"


def _make(root: Path, name: str, *, size: int = 1024) -> Path:
    """Создать запись `cache/<name>` (+ `.json`) заданного размера."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_bytes(b"x" * size)
    path.with_suffix(".json").write_text(json.dumps({"key": name}), encoding="utf-8")
    return path


def _names(entries: Iterable[EmbedsEntry]) -> list[str]:
    return sorted(e.path.name for e in entries)


def _expect(label: str, entries: Iterable[EmbedsEntry], expected: Iterable[str]) -> None:
    """Печать строки таблицы «политика -> что попало в план удаления» и проверка ожидания."""
    got = _names(entries)
    want = sorted(expected)
    mark = "ничего" if not got else ", ".join(got)
    print(f"    {label:34} -> {mark}")
    if got != want:
        raise AssertionError(f"{label}: план {got}, ожидалось {want}")


def _write(md5: str, sha: str, *, crop: str = "1376x768", enc: str = "base",
           dtype: str = "bfloat16", device: str = "cpu",
           path: Path | None = None) -> CacheWrite:
    return CacheWrite(md5=md5, sha=sha, crop=crop, enc=enc, dtype=dtype, device=device, path=path)


def _cleanup(root: Path, write: CacheWrite) -> DeferredCleanup:
    cleanup = DeferredCleanup(root)
    cleanup.register(write)
    return cleanup


def _capture(func: Callable[[], object]) -> tuple[object, str]:
    """Выполнить функцию, вернуть `(результат, весь stdout)`."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = func()
    return result, buf.getvalue()


# --------------------------------------------------------------------------- #
# Кейс 1: pair
# --------------------------------------------------------------------------- #

def case_pair(tmp: Path) -> None:
    root = tmp / "pair"
    old_prompt = _embeds_name(MD5_1, SHA_1)            # старый промт, та же картинка
    old_image = _embeds_name(MD5_2, SHA_2)             # тот же промт, старая картинка
    unrelated = _embeds_name(MD5_3, SHA_3)
    for name in (old_prompt, old_image, unrelated):
        _make(root, name)

    write = _write(MD5_2, SHA_1)                       # сменили и промт, и картинку сразу
    cleanup = _cleanup(root, write)
    # `pair` = объединение `prompt` и `image` (plan.md): старая картинка ТОГО ЖЕ промта
    # (md5₂, sha₂) тоже устарела — она делит промт с текущей парой.
    _expect("pair (объединение)", cleanup.plan("pair"), [old_prompt, old_image])
    _expect("prompt (только та же картинка)", cleanup.plan("prompt"), [old_prompt])
    _expect("image (только тот же промт)", cleanup.plan("image"), [old_image])
    _expect("none", cleanup.plan("none"), [])
    if not (root / unrelated).is_file():
        raise AssertionError(f"{unrelated} не делит ни промт, ни картинку — удалять нельзя")
    print(f"  ok: текущая пара ({write.md5}, {write.sha}) вытесняет старую картинку и старый промт,")
    print(f"      а запись, не делящая ни промт, ни картинку, остаётся")


# --------------------------------------------------------------------------- #
# Кейс 2: prompt не трогает другой промт, image — другую картинку
# --------------------------------------------------------------------------- #

def case_prompt_vs_image(tmp: Path) -> None:
    root = tmp / "prompt-image"
    same_image = _embeds_name(MD5_1, SHA_1)            # тот же промт? нет: другой md5, та же картинка
    same_prompt = _embeds_name(MD5_2, SHA_2)           # тот же промт, другая картинка
    for name in (same_image, same_prompt):
        _make(root, name)

    cleanup = _cleanup(root, _write(MD5_2, SHA_1))
    _expect("prompt", cleanup.plan("prompt"), [same_image])
    _expect("image", cleanup.plan("image"), [same_prompt])
    _expect("none", cleanup.plan("none"), [])
    _expect("pair", cleanup.plan("pair"), [same_image, same_prompt])
    for name in (same_image, same_prompt):
        if not (root / name).is_file():
            raise AssertionError(f"plan() не должен удалять файлы ({name} пропал)")
    print("  ok: prompt трогает только смену промта, image — только смену картинки")


# --------------------------------------------------------------------------- #
# Кейс 3: текущая пара и текущий .pt не удаляются никогда
# --------------------------------------------------------------------------- #

def case_current_pair(tmp: Path) -> None:
    root = tmp / "current"
    current_crop = _embeds_name(MD5_2, SHA_1, "1376x768")
    other_crop = _embeds_name(MD5_2, SHA_1, "864x480")
    stale = _embeds_name(MD5_1, SHA_1)
    for name in (current_crop, other_crop, stale):
        _make(root, name)

    write = _write(MD5_2, SHA_1, path=root / current_crop)
    cleanup = _cleanup(root, write)
    _expect("pair (текущая пара + текущий путь)", cleanup.plan("pair"), [stale])

    # тот же путь, но путь в CacheWrite не задан — запись всё равно защищена текущей парой
    _expect("pair (path=None, пара та же)",
            _cleanup(root, _write(MD5_2, SHA_1)).plan("pair"), [stale])
    # другая пара, но путь — текущий .pt: правило пути тоже обязано сработать
    stranger = _cleanup(root, _write(MD5_3, SHA_3, path=root / stale)).plan("pair")
    if current_crop in _names(stranger) or other_crop in _names(stranger):
        raise AssertionError(f"удалена запись текущей пары: {_names(stranger)}")
    print("  ok: текущий .pt и текущая пара остаются, даже под политикой pair")


# --------------------------------------------------------------------------- #
# Кейс 4: варианты (enc, dtype, device) — свои группы
# --------------------------------------------------------------------------- #

def case_variants(tmp: Path) -> None:
    root = tmp / "variants"
    cpu = _embeds_name(MD5_1, SHA_1)
    gpu = _embeds_name(MD5_1, SHA_1, device="cuda")
    fp16 = _embeds_name(MD5_1, SHA_1, dtype="float16")
    heretic = _embeds_name(MD5_1, SHA_1, enc="heretic")
    for name in (cpu, gpu, fp16, heretic):
        _make(root, name)

    _expect("pair, пишем cpu/bf16", _cleanup(root, _write(MD5_2, SHA_1)).plan("pair"), [cpu])
    _expect("pair, пишем cuda/bf16",
            _cleanup(root, _write(MD5_2, SHA_1, device="cuda")).plan("pair"), [gpu])
    _expect("pair, пишем cpu/fp16",
            _cleanup(root, _write(MD5_2, SHA_1, dtype="float16")).plan("pair"), [fp16])
    _expect("pair, пишем heretic/cpu/bf16",
            _cleanup(root, _write(MD5_2, SHA_1, enc="heretic")).plan("pair"), [heretic])
    # canonical-поля: 'bf16'/'cuda:0' в CacheWrite — тот же вариант, что и в имени
    _expect("pair, пишем bf16/cuda:0 (канон)",
            _cleanup(root, _write(MD5_2, SHA_1, dtype="bf16", device="cuda:0")).plan("pair"), [gpu])
    for name in (cpu, gpu, fp16, heretic):
        if not (root / name).is_file():
            raise AssertionError(f"plan() не должен удалять файлы ({name} пропал)")
    print("  ok: cuda не вытесняет cpu, fp16 не вытесняет bf16, heretic — отдельный вариант")


# --------------------------------------------------------------------------- #
# Кейс 5: crop не участвует в вытеснении
# --------------------------------------------------------------------------- #

def case_crops(tmp: Path) -> None:
    root = tmp / "crops"
    crops = ("864x480", "1376x768", "2560x1440")
    for crop in crops:
        _make(root, _embeds_name(MD5_1, SHA_1, crop))

    write = _write(MD5_1, SHA_1, crop="1376x768")
    cleanup = _cleanup(root, write)
    for policy in POLICIES:
        _expect(f"{policy} (одна пара, 3 crop)", cleanup.plan(policy), [])

    # смена картинки убивает старые записи ВСЕХ разрешений (crop их не защищает)
    switched = _cleanup(root, _write(MD5_1, SHA_2, crop="1376x768"))
    _expect("pair, сменили картинку",
            switched.plan("pair"), [_embeds_name(MD5_1, SHA_1, crop) for crop in crops])
    print("  ok: разные --size/--resolution/--aspect одной пары остаются все")


# --------------------------------------------------------------------------- #
# Кейс 6: flush(apply=False) ничего не удаляет, fp8-кэш цел
# --------------------------------------------------------------------------- #

def case_flush(tmp: Path) -> None:
    root = tmp / "flush"
    stale = _embeds_name(MD5_1, SHA_1)
    kept = _embeds_name(MD5_3, SHA_3)
    _make(root, stale)
    _make(root, kept)
    fp8 = root / "fp8_transformer" / "dummy.pt"
    fp8.parent.mkdir(parents=True, exist_ok=True)
    fp8.write_bytes(b"weights")
    other = root / "notes.txt"
    other.write_text("не кэш", encoding="utf-8")

    write = _write(MD5_2, SHA_1)
    cleanup = _cleanup(root, write)

    (result, out) = _capture(lambda: cleanup.flush("pair", apply=False))
    print(f"    flush(apply=False) -> {result}")
    if result != (0, 0):
        raise AssertionError(f"flush(apply=False) вернул {result}, ожидалось (0, 0)")
    if "ничего не удалено" not in out or stale not in out:
        raise AssertionError(f"flush(apply=False) не напечатал план: {out!r}")
    for name in (stale, kept):
        if not (root / name).is_file() or not (root / name.replace(".pt", ".json")).is_file():
            raise AssertionError(f"flush(apply=False) удалил {name} (и/или его .json)")

    # план пересчитывается в flush по фактическому содержимому cache/, а не по снимку register
    late = _make(root, _embeds_name(MD5_2, SHA_2))
    if late.name not in _names(cleanup.plan("pair", root)):
        raise AssertionError("plan() не увидел запись, появившуюся после register()")
    deleted, freed = cleanup.flush("pair", apply=True)
    print(f"    flush(apply=True) -> удалено {deleted}, освобождено {freed} B")
    if deleted != 2 or freed <= 0:
        raise AssertionError(f"flush(apply=True) -> ({deleted}, {freed}), ожидалось (2, >0)")
    for name in (stale, late.name):
        if (root / name).exists() or (root / name.replace(".pt", ".json")).exists():
            raise AssertionError(f"{name} и его .json должны быть удалены")
    if not (root / kept).is_file():
        raise AssertionError(f"{kept} не делит промт/картинку с текущей парой — удалять нельзя")
    if not fp8.is_file() or not other.is_file():
        raise AssertionError("удалён файл вне embeds_*.pt (fp8-кэш / чужой файл)")

    # политика none — полный no-op: ни печати, ни удаления
    none_cleanup = _cleanup(root, _write(MD5_1, SHA_2))
    (result, out) = _capture(lambda: none_cleanup.flush("none", apply=True))
    if result != (0, 0) or out.strip():
        raise AssertionError(f"policy=none не no-op: {result}, вывод {out!r}")
    for policy in POLICIES:
        before = _names(scan_embeds(root))
        (result, _out) = _capture(lambda: _cleanup(root, write).flush(policy, apply=False))
        if result != (0, 0) or _names(scan_embeds(root)) != before:
            raise AssertionError(f"{policy}: dry-run изменил cache/ или вернул {result}")
    if not fp8.is_file():
        raise AssertionError("cache/fp8_transformer/dummy.pt удалён при какой-то политике")
    print("  ok: --dry-run печатает план, fp8-кэш и не-embeds файлы целы при всех политиках")


# --------------------------------------------------------------------------- #
# Кейс 7: служебное (summary, policy name, регистрация нескольких записей)
# --------------------------------------------------------------------------- #

def case_misc(tmp: Path) -> None:
    root = tmp / "misc"
    empty = DeferredCleanup(root)
    if "задач нет" not in empty.summary():
        raise AssertionError(f"summary() пустой задачи: {empty.summary()!r}")
    (result, out) = _capture(lambda: empty.flush("pair", apply=True))
    if result != (0, 0) or out.strip():
        raise AssertionError(f"без регистраций flush не должен ничего делать: {result}, {out!r}")

    for raw, want in ((" pair ", "pair"), ("none", "none"), ("IMAGE", "image"), (None, "none")):
        got = normalize_policy(raw)
        if got != want:
            raise AssertionError(f"normalize_policy({raw!r}) = {got!r}, ожидалось {want!r}")
    try:
        normalize_policy("bogus")
    except ValueError as exc:
        print(f"    normalize_policy('bogus') -> ValueError ({str(exc)[:40]}…)")
    else:
        raise AssertionError("normalize_policy('bogus') не поднял ValueError")

    # две записи в одной политике: план — объединение
    root.mkdir(parents=True, exist_ok=True)
    first = _make(root, _embeds_name(MD5_1, SHA_1))
    second = _make(root, _embeds_name(MD5_2, SHA_2))
    cleanup = DeferredCleanup(root)
    cleanup.register(_write(MD5_2, SHA_1))
    cleanup.register(_write(MD5_1, SHA_2))
    _expect("pair по двум записям", cleanup.plan("pair"), [first.name, second.name])
    summary = cleanup.summary()
    print(f"    summary(): {summary}")
    if "задач 2" not in summary or "пар 2" not in summary or "вариантов 1" not in summary:
        raise AssertionError(f"summary() не описывает состояние: {summary!r}")
    deleted, _freed = cleanup.flush("pair", apply=True)
    if deleted != 2:
        raise AssertionError(f"flush по двум записям удалил {deleted}, ожидалось 2")
    print("  ok: summary/normalize_policy/несколько записей")


# --------------------------------------------------------------------------- #

CASES: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("1. pair: объединение prompt и image", case_pair),
    ("2. prompt не трогает другой промт", case_prompt_vs_image),
    ("3. текущая пара и текущий .pt", case_current_pair),
    ("4. варианты enc/dtype/device", case_variants),
    ("5. crop не участвует в вытеснении", case_crops),
    ("6. flush(apply=False) и fp8-кэш", case_flush),
    ("7. summary/normalize_policy/несколько записей", case_misc),
)


def main() -> int:
    failed = 0
    with tempfile.TemporaryDirectory(prefix="qwenstyle-cleanup-") as raw:
        tmp = Path(raw)
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
