"""Политика кэша эмбеддингов: разбор новых имён и группировка `prune` (без torch и GPU).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_cache_variants.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (план `te-gpu-streaming`, шаг 2): в имени записи появились хвосты `-<dtype>` и
`-dev-<device>`. Если их не разобрать, запись выпадает из `cache list`/`prune` и становится
неубираемым мусором; если не учесть в группе, `prune --group-by image` удалит CPU-запись как
«дубль» GPU-шной. Тест фиксирует матрицу «режим prune -> что удаляется».

Кейсы:

1. **Парсинг имён:** `embeds_<md5>_<sha>-fp16-dev-cuda_1376x768.pt` и
   `embeds_<md5>_<sha>_te-heretic-dev-cuda_1376x768.pt` -> верные dtype/device/enc; старое имя
   -> `bfloat16`/`cpu`/`legacy=False`; имя без `sha` -> `legacy=True`.
2. **Смена разрешения/аспекта ничего не удаляет:** три crop'а одной пары -> план пуст
   (и в дефолтном режиме, и с `group_by="image"`).
3. **Два промта на одной картинке:** дефолтный prune пуст; `group_by="image"` удаляет только
   старую по mtime запись (это и есть смысл режима `image`), а `delete_embeds` трогает только
   `embeds_*.pt` + их `.json`.
4. **CPU и GPU не вытесняют друг друга:** варианты одной пары на одном crop ->
   `plan_prune(group_by="image", keep_per_resolution=1)` пуст.
5. **Явные режимы:** `--older-than 0d` и `--all` возвращают все живые записи; `--legacy`
   без legacy-записей -> пусто.
"""
from __future__ import annotations

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

from qwenstyle.cache import (                    # noqa: E402 — после правки sys.path
    EmbedsEntry,
    delete_embeds,
    parse_age,
    plan_prune,
    scan_embeds,
)

MD5_A = "aaaa1111bbbb"
MD5_B = "eeee3333ffff"
SHA_1 = "cccc2222dddd"


# --------------------------------------------------------------------------- #
# Фикстуры
# --------------------------------------------------------------------------- #

def _make(root: Path, name: str, *, mtime: float | None = None, sidecar: bool = True) -> Path:
    """Создать запись `cache/<name>` (+ `.json`) размером 1 KiB с заданным mtime."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_bytes(b"x" * 1024)
    if sidecar:
        (root / name.replace(".pt", ".json")).write_text(
            json.dumps({"prompt_md5": name.split("_")[1]}, ensure_ascii=False),
            encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
        side = path.with_suffix(".json")
        if side.is_file():
            os.utime(side, (mtime, mtime))
    return path


def _names(entries: Iterable[EmbedsEntry]) -> list[str]:
    return sorted(e.path.name for e in entries)


def _expect(label: str, entries: Iterable[EmbedsEntry], expected: Iterable[str]) -> None:
    """Печать строки матрицы «режим -> что удаляется» и проверка ожидания."""
    got = _names(entries)
    want = sorted(expected)
    mark = "ничего" if not got else ", ".join(got)
    print(f"    {label:44} -> {mark}")
    if got != want:
        raise AssertionError(f"{label}: план {got}, ожидалось {want}")


def _require(entries: list[EmbedsEntry], name: str) -> EmbedsEntry:
    for e in entries:
        if e.path.name == name:
            return e
    raise AssertionError(f"запись {name} не найдена среди {_names(entries)}")


# --------------------------------------------------------------------------- #
# Кейс 1: парсинг имён
# --------------------------------------------------------------------------- #

GPU_FP16 = f"embeds_{MD5_A}_{SHA_1}-fp16-dev-cuda_1376x768.pt"
GPU_HERETIC = f"embeds_{MD5_B}_{SHA_1}_te-heretic-dev-cuda_1376x768.pt"
OLD_HERETIC = f"embeds_{MD5_B}_{SHA_1}_te-heretic_1376x768.pt"
NO_SHA = f"embeds_{MD5_B}_1376x768.pt"


def case_parse(tmp: Path) -> None:
    root = tmp / "parse"
    for name in (GPU_FP16, GPU_HERETIC, OLD_HERETIC, NO_SHA):
        _make(root, name)
    entries = scan_embeds(root)
    if len(entries) != 4:
        raise AssertionError(f"разобрано {len(entries)} из 4 записей: {_names(entries)}")

    checks = (
        (GPU_FP16, dict(md5=MD5_A, sha=SHA_1, enc="base", dtype="float16",
                        device="cuda", crop="1376x768", legacy=False)),
        (GPU_HERETIC, dict(md5=MD5_B, sha=SHA_1, enc="heretic", dtype="bfloat16",
                           device="cuda", crop="1376x768", legacy=False)),
        (OLD_HERETIC, dict(md5=MD5_B, sha=SHA_1, enc="heretic", dtype="bfloat16",
                           device="cpu", crop="1376x768", legacy=False)),
        (NO_SHA, dict(md5=MD5_B, sha=None, enc="base", dtype="bfloat16",
                      device="cpu", crop="1376x768", legacy=True)),
    )
    for name, want in checks:
        e = _require(entries, name)
        for field, expected in want.items():
            got = getattr(e, field)
            if got != expected:
                raise AssertionError(f"{name}: {field}={got!r}, ожидалось {expected!r}")
        print(f"  ok: {name} -> md5={e.md5} sha={e.sha} enc={e.enc} "
              f"dtype={e.dtype} device={e.device} legacy={e.legacy}")


# --------------------------------------------------------------------------- #
# Кейс 2: разрешение/аспект ничего не удаляет
# --------------------------------------------------------------------------- #

def case_crops(tmp: Path) -> None:
    root = tmp / "crops"
    base_mtime = 1_700_000_000.0
    for i, crop in enumerate(("864x480", "1376x768", "2560x1440")):
        _make(root, f"embeds_{MD5_A}_{SHA_1}_{crop}.pt", mtime=base_mtime + i)
    entries = scan_embeds(root)
    _expect("plan_prune(keep_per_resolution=1)", plan_prune(entries), [])
    _expect("plan_prune(group_by=image, keep=1)",
            plan_prune(entries, group_by="image"), [])


# --------------------------------------------------------------------------- #
# Кейс 3: два промта на одной картинке
# --------------------------------------------------------------------------- #

def case_two_prompts(tmp: Path) -> None:
    root = tmp / "prompts"
    old = f"embeds_{MD5_A}_{SHA_1}_1376x768.pt"
    new = f"embeds_{MD5_B}_{SHA_1}_1376x768.pt"
    _make(root, old, mtime=1_700_000_000.0)
    _make(root, new, mtime=1_700_100_000.0)
    entries = scan_embeds(root)
    _expect("plan_prune(keep_per_resolution=1)", plan_prune(entries), [])
    _expect("plan_prune(group_by=image, keep=1)",
            plan_prune(entries, group_by="image"), [old])

    # `delete_embeds` трогает только embeds_*.pt/embeds_*.json: рядом лежащий fp8-кэш цел.
    fp8 = root / "fp8_transformer" / "dummy.pt"
    fp8.parent.mkdir(parents=True, exist_ok=True)
    fp8.write_bytes(b"weights")
    targets = plan_prune(entries, group_by="image")
    deleted, freed = delete_embeds(targets, root)
    if deleted != 1 or freed < 1024:
        raise AssertionError(f"delete_embeds -> ({deleted}, {freed}), ожидалось (1, >=1024)")
    if (root / old).exists() or (root / old.replace(".pt", ".json")).exists():
        raise AssertionError(f"{old} и его .json должны быть удалены")
    if not (root / new).exists():
        raise AssertionError(f"{new} — текущая запись, удалять нельзя")
    if not fp8.exists():
        raise AssertionError("delete_embeds удалил cache/fp8_transformer/dummy.pt")
    print(f"  ok: удалена только старая запись {old} (+ её .json), fp8-кэш цел")


# --------------------------------------------------------------------------- #
# Кейс 4: CPU и GPU варианты не вытесняют друг друга
# --------------------------------------------------------------------------- #

def case_device_variants(tmp: Path) -> None:
    root = tmp / "variants"
    cpu_name = f"embeds_{MD5_A}_{SHA_1}_1376x768.pt"
    _make(root, cpu_name, mtime=1_700_000_000.0)
    _make(root, f"embeds_{MD5_A}_{SHA_1}-dev-cuda_1376x768.pt", mtime=1_700_100_000.0)
    _make(root, f"embeds_{MD5_A}_{SHA_1}-fp16_1376x768.pt", mtime=1_700_200_000.0)
    entries = scan_embeds(root)
    if len({e.group for e in entries}) != 3:
        raise AssertionError(f"варианты слиплись в одну группу: {[e.group for e in entries]}")
    _expect("plan_prune(group_by=image, keep=1)",
            plan_prune(entries, group_by="image"), [])
    _expect("plan_prune(keep_per_resolution=1)", plan_prune(entries), [])


# --------------------------------------------------------------------------- #
# Кейс 5: явные режимы
# --------------------------------------------------------------------------- #

def case_explicit(tmp: Path) -> None:
    root = tmp / "explicit"
    names = [f"embeds_{MD5_A}_{SHA_1}_864x480.pt",
             f"embeds_{MD5_A}_{SHA_1}_1376x768.pt",
             f"embeds_{MD5_B}_{SHA_1}-dev-cuda_1696x960.pt"]
    for i, name in enumerate(names):
        _make(root, name, mtime=1_700_000_000.0 + i)
    entries = scan_embeds(root)
    _expect("plan_prune(older_than=parse_age('0d'))",
            plan_prune(entries, older_than=parse_age("0d")), names)
    _expect("plan_prune(all=True)", plan_prune(entries, all=True), names)
    _expect("plan_prune(legacy=True) без legacy-записей", plan_prune(entries, legacy=True), [])
    if parse_age("0d") != 0:
        raise AssertionError(f"parse_age('0d') = {parse_age('0d')!r}, ожидалось 0")


# --------------------------------------------------------------------------- #

CASES: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("1. парсинг новых и старых имён", case_parse),
    ("2. смена разрешения/аспекта ничего не удаляет", case_crops),
    ("3. два промта на одной картинке", case_two_prompts),
    ("4. CPU и GPU варианты не вытесняют друг друга", case_device_variants),
    ("5. явные режимы --older-than/--all/--legacy", case_explicit),
)


def main() -> int:
    failed = 0
    with tempfile.TemporaryDirectory(prefix="qwenstyle-cache-variants-") as raw:
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
