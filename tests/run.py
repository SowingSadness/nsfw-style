"""Запуск тестов репозитория: `python tests/run.py [--tag ...] [--changed <path> ...]`.

Зачем отдельный раннер, а не pytest (его в venv нет и ставить нельзя, см. AGENTS.md §1):

* тесты — обычные скрипты (`ALL PASS` + exit 0), раннер их не импортирует, а запускает
  подпроцессом: падение одного не уносит остальные и не оставляет грязи в `sys.modules`;
* `--changed` берёт только тесты, отвечающие за изменённые файлы (`tests/registry.py`), то есть
  правка кубика проверяется за секунды, без CUDA;
* `--tag` даёт явные срезы: `unit` (без torch), `contract` (декларация == реализация),
  `cube` (кубик на заглушках), `e2e` (CUDA + веса).

Примеры:

    .venv/Scripts/python.exe tests/run.py --tag unit
    .venv/Scripts/python.exe tests/run.py --changed qwenstyle/cubes/decoder_vae.py
    .venv/Scripts/python.exe tests/run.py --tag e2e
    .venv/Scripts/python.exe tests/run.py --map            # карта «исходник -> тесты»

Код возврата: 0 — все файлы `PASS`/`SKIP`, 1 — есть `FAIL`, 2 — нечего запускать.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import registry                                                     # noqa: E402


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("HF_HUB_OFFLINE", "1")
    return env


def _collect(tags: list[str], changed: list[str]) -> tuple[list[str], bool]:
    """Список тестов: явные `--tag` плюс (или вместо них) результат по `--changed`."""
    selected: list[str] = []
    wide = False
    for tag in tags:
        if tag not in registry.TAGS:
            raise SystemExit(f"неизвестная категория {tag!r}; есть: {', '.join(registry.TAGS)}")
        for test in registry.tests_in_tag(tag):
            if test not in selected:
                selected.append(test)
    if changed:
        found, wide = registry.tests_for(changed)
        for test in found:
            if test not in selected:
                selected.append(test)
    if not tags and not changed:                      # по умолчанию — всё, кроме e2e
        for tag in ("unit", "contract", "cube"):
            for test in registry.tests_in_tag(tag):
                if test not in selected:
                    selected.append(test)
    return sorted(selected), wide


def _run_one(test: str, verbose: bool) -> tuple[str, int, float, str]:
    """Запустить один файл: `(статус, код, секунды, хвост вывода)`."""
    path = TESTS_DIR / test
    started = time.perf_counter()
    proc = subprocess.run([sys.executable, str(path)], cwd=str(REPO_ROOT), env=_env(),
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.perf_counter() - started
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        status = "FAIL"
    elif "SKIP" in output and "ALL PASS" not in output:
        status = "SKIP"
    else:
        status = "PASS"
    tail = output if verbose else "\n".join(output.strip().splitlines()[-12:])
    return status, proc.returncode, elapsed, tail


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tests/run.py",
        description="Раннер тестов: скрипты, subprocess, срезы по категориям и по изменённым файлам.",
    )
    parser.add_argument("--tag", action="append", default=[], choices=list(registry.TAGS),
                        help="категория тестов (можно несколько раз)")
    parser.add_argument("--changed", action="append", default=[], metavar="PATH",
                        help="изменённый путь; тесты берутся из tests/registry.py")
    parser.add_argument("--list", action="store_true", help="только напечатать список тестов")
    parser.add_argument("--map", action="store_true", help="напечатать карту «исходник -> тесты»")
    parser.add_argument("--verbose", "-v", action="store_true", help="печатать весь вывод файла")
    parser.add_argument("--stop", action="store_true", help="остановиться на первом FAIL")
    args = parser.parse_args(argv)

    if args.map:
        print(registry.describe())
        return 0

    tests, wide = _collect(args.tag, args.changed)
    if wide:
        print("[map] ни одно правило не совпало — беру unit + contract целиком")
    if not tests:
        print("нет тестов под указанные условия")
        return 2
    if args.list:
        for test in tests:
            print(test)
        return 0

    print(f"[run] {len(tests)} файл(ов): {', '.join(sorted(args.tag) or ['по карте'])}")
    failed: list[str] = []
    for test in tests:
        status, code, elapsed, tail = _run_one(test, args.verbose)
        print(f"{status} {test}  ({elapsed:.1f} с)")
        if status == "FAIL":
            failed.append(test)
            print("\n".join("    " + line for line in tail.splitlines()))
            if args.stop:
                break
    if failed:
        print(f"\n{len(failed)} из {len(tests)} файлов провалено: {', '.join(failed)}")
        return 1
    print(f"\nALL PASS ({len(tests)} файл(ов))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
