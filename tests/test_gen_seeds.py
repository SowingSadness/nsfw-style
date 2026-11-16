"""Сиды и имя сессии `gen`: `--seeds SPEC` XOR `--count N`, один случайный по умолчанию.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_gen_seeds.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем: `gen` — сессия из N кадров, и её цена считается кадрами, а не минутами. Ошибка в разборе
сидов не падает: она делает лишние кадры (дорого), дубли (`84-8.png` дважды) или — хуже — молча
игнорирует `--seeds`, отдавая «случайный» набор вместо заказанного. Тест фиксирует контракт
`qwenstyle/cmd/gen.py` без весов и генерации:

1. `--seeds "84,32-38"` -> ровно `[84, 32, 33, 34, 35, 36, 37, 38]` (порядок сохраняется);
2. `--count 3` -> три **разных** положительных сида;
3. `--seeds` вместе с `--count` -> ошибка (`ConfigError`), а не тихий приоритет одного из них;
4. ничего не задано -> ровно один случайный сид (он печатается в шапке сессии);
5. повторы в `--seeds` убираются (один файл `<seed>-<steps>.png` на сид);
6. имя сессии — `<YYYYMMDDHHMM>-<tag>`, при коллизии минуты — `-2`, `-3`; каталог не создаётся
   (`--dry-run` не оставляет следов).
"""
from __future__ import annotations

import re
import sys
import tempfile
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                   # тест не должен оставлять __pycache__ в пакете

from qwenstyle.cmd import _project as project     # noqa: E402
from qwenstyle.cmd import gen as gen_cmd          # noqa: E402
from qwenstyle.config import ConfigError          # noqa: E402

#: Момент времени для детерминированных имён сессий (2026-11-15 19:40 по локальному времени).
FIXED_NOW = time.strptime("20261115-1940", "%Y%m%d-%H%M")
#: Имя сессии: `<YYYYMMDDHHMM>-<tag>` + необязательный суффикс коллизии (`-2`, `-3`, …).
SESSION_RE = re.compile(r"^\d{12}-[a-z0-9._-]+(-\d+)?$")


def case_seeds_spec() -> None:
    """1. `--seeds "84,32-38"` разворачивается в список сидов в указанном порядке."""
    got = gen_cmd.resolve_seeds("84,32-38", None)
    want = [84, 32, 33, 34, 35, 36, 37, 38]
    if got != want:
        raise AssertionError(f"--seeds '84,32-38' -> {got}, ожидалось {want}")
    for spec, want in (("7", [7]), ("1,7,42", [1, 7, 42]), ("0-2,10", [0, 1, 2, 10])):
        got = gen_cmd.resolve_seeds(spec, None)
        if got != want:
            raise AssertionError(f"--seeds {spec!r} -> {got}, ожидалось {want}")
    print(f"  ok: '84,32-38' -> {[84, 32, 33, 34, 35, 36, 37, 38]}; '1,7,42' и '0-2,10' разобраны так же")


def case_count() -> None:
    """2. `--count N` даёт N разных сидов (положительные 32-битные, без повторов)."""
    for count in (1, 3, 5):
        seeds = gen_cmd.resolve_seeds(None, count)
        if len(seeds) != count:
            raise AssertionError(f"--count {count} -> {len(seeds)} сидов: {seeds}")
        if len(set(seeds)) != count:
            raise AssertionError(f"--count {count} -> повторы среди {seeds}")
        for seed in seeds:
            if not isinstance(seed, int):
                raise AssertionError(f"сид не int: {seed!r}")
            if not (gen_cmd.SEED_MIN <= seed <= gen_cmd.SEED_MAX):
                raise AssertionError(f"сид {seed} вне диапазона "
                                     f"{gen_cmd.SEED_MIN}..{gen_cmd.SEED_MAX}")
    # два вызова подряд не обязаны совпадать, но каждый обязан быть полным и уникальным
    first = gen_cmd.resolve_seeds(None, 4)
    second = gen_cmd.resolve_seeds(None, 4)
    if len(set(first) | set(second)) < 4:
        raise AssertionError(f"случайные сиды выродились: {first} и {second}")
    print(f"  ok: --count 1/3/5 -> по N разных сидов; пример count=3: {gen_cmd.resolve_seeds(None, 3)}")


def case_both_specified() -> None:
    """3. `--seeds` и `--count` вместе — ошибка (цена кадра зависит от этого выбора)."""
    for spec, count in (("84", 3), ("84", 1), ("84", 0)):
        try:
            gen_cmd.resolve_seeds(spec, count)
        except ConfigError as exc:
            text = str(exc)
            for token in ("--seeds", "--count"):
                if token not in text:
                    raise AssertionError(f"в ошибке нет {token!r}: {text}")
            continue
        raise AssertionError(f"--seeds {spec!r} + --count {count} не отбиты")
    try:
        gen_cmd.resolve_seeds(None, 0)
    except ConfigError:
        pass
    else:
        raise AssertionError("--count 0 не отбит")
    print("  ok: --seeds + --count -> ConfigError, --count 0 -> ConfigError")


def case_default_one_seed() -> None:
    """4. Ничего не задано: ровно один случайный сид (его печатает шапка сессии)."""
    seeds = gen_cmd.resolve_seeds(None, None)
    if len(seeds) != 1:
        raise AssertionError(f"по умолчанию {len(seeds)} сидов: {seeds}")
    if not (gen_cmd.SEED_MIN <= seeds[0] <= gen_cmd.SEED_MAX):
        raise AssertionError(f"сид {seeds[0]} вне диапазона")
    text = gen_cmd.seeds_text(seeds)
    if not text.isdigit():
        raise AssertionError(f"сид не печатается числом: {text!r}")
    print(f"  ok: без флагов -> один сид ({seeds[0]}), печатается как {text!r}")


def case_duplicates() -> None:
    """5. Повторы в `--seeds` убираются: один файл `<seed>-<steps>.png` на сид, без дублей."""
    got = gen_cmd.resolve_seeds("5,5,6,5", None)
    if got != [5, 6]:
        raise AssertionError(f"--seeds '5,5,6,5' -> {got}, ожидалось [5, 6]")
    if gen_cmd.seeds_text([84]) != "84" or gen_cmd.seeds_text([1, 7, 42]) != "1, 7, 42":
        raise AssertionError(f"seeds_text: {gen_cmd.seeds_text([84])!r}, "
                             f"{gen_cmd.seeds_text([1, 7, 42])!r}")
    print("  ok: '5,5,6,5' -> [5, 6]; seeds_text -> '84' / '1, 7, 42'")


def case_invalid_spec() -> None:
    """6. Кривой `--seeds` — ошибка с указанием испорченной части, а не пустой список."""
    for spec in ("a-b", "7-3", "84,,x"):
        try:
            gen_cmd.resolve_seeds(spec, None)
        except ConfigError:
            continue
        raise AssertionError(f"--seeds {spec!r} не отбит")
    try:
        gen_cmd.resolve_seeds("", None)
    except ConfigError:
        pass
    else:
        # пустая строка = «флаг не задан»: допустим один случайный сид
        seeds = gen_cmd.resolve_seeds(None, None)
        if len(seeds) != 1:
            raise AssertionError(f"пустой --seeds дал {seeds}")
    print("  ok: 'a-b', '7-3', '84,,x' -> ConfigError с разбором части")


def case_session_name() -> None:
    """7. Имя сессии `<YYYYMMDDHHMM>-<tag>`, коллизия минуты -> `-2`, `-3`; без каталога."""
    with tempfile.TemporaryDirectory(prefix="qwenstyle-gen-seeds-") as tmp:
        first = project.session_dir(tmp, "qwen", now=FIXED_NOW)
        if first.name != "202611151940-qwen":
            raise AssertionError(f"имя сессии {first.name!r}, ожидалось '202611151940-qwen'")
        if not SESSION_RE.match(first.name):
            raise AssertionError(f"имя сессии не подходит под <YYYYMMDDHHMM>-<tag>: {first.name!r}")
        if Path(first).parent != Path(tmp):
            raise AssertionError(f"сессия вне <out>: {first}")
        if first.exists():
            raise AssertionError(f"session_dir создал каталог сам: {first}")

        first.mkdir(parents=True)
        second = project.session_dir(tmp, "qwen", now=FIXED_NOW)
        if second.name != "202611151940-qwen-2":
            raise AssertionError(f"при коллизии минуты ожидалось '-2', дано {second.name!r}")
        second.mkdir(parents=True)
        third = project.session_dir(tmp, "qwen", now=FIXED_NOW)
        if third.name != "202611151940-qwen-3":
            raise AssertionError(f"третья коллизия: ожидалось '-3', дано {third.name!r}")

        # другая минута — независимое имя, суффикса нет
        other = project.session_dir(tmp, "qwen", now=time.strptime("20261115-1941", "%Y%m%d-%H%M"))
        if other.name != "202611151941-qwen":
            raise AssertionError(f"новая минута получила {other.name!r}")
        # другой промт — другой tag, чужие коллизии не влияют
        other_prompt = project.session_dir(tmp, "h3", now=FIXED_NOW)
        if other_prompt.name != "202611151940-h3":
            raise AssertionError(f"tag другого промта: {other_prompt.name!r}")
    print("  ok: 202611151940-qwen -> -2 -> -3; каталог создаёт только `run_batch`, не session_dir")


CASES = (
    ("1. --seeds '84,32-38'", case_seeds_spec),
    ("2. --count N -> N разных", case_count),
    ("3. --seeds + --count -> ошибка", case_both_specified),
    ("4. без флагов -> один сид", case_default_one_seed),
    ("5. повторы убираются", case_duplicates),
    ("6. кривой --seeds", case_invalid_spec),
    ("7. имя сессии и коллизии", case_session_name),
)


def main() -> int:
    failed = 0
    for label, case in CASES:
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
