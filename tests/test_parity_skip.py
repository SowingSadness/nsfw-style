"""Тесты skip-механизма `check-parity` (`requires` в записи эталона) — обычный скрипт, без pytest.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_parity_skip.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем: в `tests/reference.json` появляются эталоны, которые физически не воспроизвести без чужих
весов (FLUX.2 — 49 GiB). На машине без них `check-parity` не должен падать, но и не смеет выдавать
пропуск за PASS: печатается `[skip] <entry>: <причина>`, запись считается отдельно, FAIL — только у
реально не совпавших кадров.

Кейсы:

1. `requires` отсутствует / пустой список / пустые строки -> `None`;
2. существующий репозиторий Хаба (`Qwen/Qwen-Image-2.1` есть в hf-cache) -> `None`;
3. несуществующий репозиторий (`example/definitely-absent-model-xyz`) -> строка с причиной,
   и в причине есть сам текст из `requires`;
4. явные пути: отсутствующий (`./cache/_bench/definitely-absent-dir`, то есть с `./`,
   и абсолютный) -> причина; существующий (`cache/weights/heretic-te`, он же в Windows-форме и
   абсолютным путём) -> `None`. Строка без `./`, которой нет на диске, по правилу `parity` —
   id репозитория, а не путь (так же решается «репозиторий или путь» в `check-parity`);
5. FLUX.2-репозиторий (`darknight9121/FLUX.2-klein-base-9B-bucket-uncensored`) -> `None` (веса есть);
6. `hf_cache`/`root` — параметры: пустой временный кэш -> причина (сеть не нужна, R3),
   `root` подменяет корень для относительных путей;
7. сквозная проверка `_run_all` на синтетическом эталоне: запись с невыполнимым `requires` даёт
   `[skip]` и `skipped 1` в итоге, код возврата 0, ни одного `PASS`/`FAIL` для неё; запись с
   существующим `requires` и без совпадения sha — по-прежнему FAIL (exit 1). Это защита от
   «skip, который скрывает поломки».

Сеть не нужна: `HF_HUB_OFFLINE=1`, всё локальное. `sys.dont_write_bytecode` — чтобы в пакете не
осталось `__pycache__`.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT,):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
sys.dont_write_bytecode = True                   # тест не должен оставлять __pycache__ в пакете
os.environ.setdefault("HF_HUB_OFFLINE", "1")     # R3: только локальные веса, никакой сети

from qwenstyle.cmd import parity                 # noqa: E402 — после правки sys.path

#: Репозиторий, который на этой машине точно есть в `hf-cache` (Qwen-Image-2.1, 30.86 GiB).
PRESENT_REPO = "Qwen/Qwen-Image-2.1"
#: Репозиторий FLUX.2: веса энкодера/трансформера уже лежат локально (шаг 4 добавит его эталон).
FLUX2_REPO = "darknight9121/FLUX.2-klein-base-9B-bucket-uncensored"
#: Заведомо отсутствующий id — не должен найтись ни в одном кэше.
ABSENT_REPO = "example/definitely-absent-model-xyz"
#: Существующий путь в репозитории (4 шарда heretic-энкодера).
PRESENT_PATH = "cache/weights/heretic-te"
#: Заведомо отсутствующий путь (с `./` — явная форма; без него строка трактуется как репозиторий).
ABSENT_PATH = "./cache/_bench/definitely-absent-dir"


def _entry(requires: Any = ..., **extra: Any) -> dict[str, Any]:
    """Минимальная запись эталона: только `requires` (и, при необходимости, прочее)."""
    entry: dict[str, Any] = {"kind": "run", "sha256": "0" * 64}
    if requires is not ...:
        entry["requires"] = requires
    entry.update(extra)
    return entry


def case_empty_requires(tmp: Path) -> None:
    """Нет поля / пустой список / пустые строки -> `None` (require'ов нет, значит всё на месте)."""
    for label, requires in (("нет поля", ...), ("пустой список", []),
                            ("только пустые строки", ["", "  "])):
        reason = parity.missing_requirements(_entry(requires))
        if reason is not None:
            raise AssertionError(f"{label}: ожидалось None, получено {reason!r}")
        print(f"  ok: {label} -> None")


def case_present_repo(tmp: Path) -> None:
    """Репозиторий, который есть в hf-cache, -> `None` (раскладка models--org--name/snapshots)."""
    reason = parity.missing_requirements(_entry([PRESENT_REPO]))
    if reason is not None:
        raise AssertionError(f"{PRESENT_REPO} есть в hf-cache, а причина {reason!r}")
    print(f"  ok: {PRESENT_REPO} -> None")


def case_absent_repo(tmp: Path) -> None:
    """Несуществующий репозиторий -> причина с его именем (и это не PASS)."""
    reason = parity.missing_requirements(_entry([ABSENT_REPO]))
    if not isinstance(reason, str) or ABSENT_REPO not in reason:
        raise AssertionError(f"для {ABSENT_REPO} ожидалась причина с его именем, получено {reason!r}")
    print(f"  ok: причина {reason!r}")


def case_paths(tmp: Path) -> None:
    """Явные пути: отсутствующий -> причина, существующий -> None (в т.ч. Windows-форма)."""
    reason = parity.missing_requirements(_entry([ABSENT_PATH]))
    if not isinstance(reason, str) or ABSENT_PATH not in reason:
        raise AssertionError(f"для отсутствующего пути {ABSENT_PATH} ожидалась причина, "
                             f"получено {reason!r}")
    if "hf-cache" in reason:                        # путь проверяется как путь, а не как репозиторий
        raise AssertionError(f"{ABSENT_PATH} принят за репозиторий: {reason!r}")
    print(f"  ok: {ABSENT_PATH} -> {reason!r}")

    reason = parity.missing_requirements(_entry([PRESENT_PATH]))
    if reason is not None:
        raise AssertionError(f"{PRESENT_PATH} существует, а причина {reason!r}")
    print(f"  ok: {PRESENT_PATH} -> None")

    win_form = PRESENT_PATH.replace("/", "\\")            # «cache\\weights\\heretic-te»
    reason = parity.missing_requirements(_entry([win_form]))
    if reason is not None:
        raise AssertionError(f"Windows-путь {win_form!r} не распознан: {reason!r}")
    print(f"  ok: {win_form!r} -> None")

    absolute = str(tmp / "absent-dir")
    reason = parity.missing_requirements(_entry([absolute]))
    if not isinstance(reason, str) or absolute not in reason or "hf-cache" in reason:
        raise AssertionError(f"абсолютный отсутствующий путь {absolute} не распознан: {reason!r}")
    print(f"  ok: абсолютный отсутствующий путь -> {reason!r}")

    absolute = str((REPO_ROOT / PRESENT_PATH).resolve())
    reason = parity.missing_requirements(_entry([absolute]))
    if reason is not None:
        raise AssertionError(f"абсолютный путь {absolute} не распознан: {reason!r}")
    print(f"  ok: абсолютный существующий путь -> None")


def case_flux2_repo(tmp: Path) -> None:
    """Запись FLUX.2: веса (49 GiB-репозиторий частично) есть в hf-cache -> `None`."""
    reason = parity.missing_requirements(_entry([FLUX2_REPO]))
    if reason is not None:
        raise AssertionError(f"{FLUX2_REPO} есть в hf-cache, а причина {reason!r}")
    print(f"  ok: {FLUX2_REPO} -> None")


def case_hf_cache_and_root(tmp: Path) -> None:
    """`hf_cache`/`root` — параметры: чужой (пустой) кэш -> причина, `root` меняет базу путей."""
    empty_cache = tmp / "empty-hf-cache"
    empty_cache.mkdir(exist_ok=True)
    reason = parity.missing_requirements(_entry([PRESENT_REPO]), hf_cache=empty_cache)
    if not isinstance(reason, str) or PRESENT_REPO not in reason:
        raise AssertionError(f"при пустом hf_cache ожидалась причина, получено {reason!r}")
    if str(empty_cache / "hub") not in reason:
        raise AssertionError(f"в причине нет каталога кэша: {reason!r}")
    print(f"  ok: пустой hf_cache -> {reason!r}")

    # относительный путь считается от `root`: в temp его нет -> причина, в корне репозитория -> None
    reason = parity.missing_requirements(_entry([PRESENT_PATH]), root=tmp)
    if not isinstance(reason, str):
        raise AssertionError(f"путь {PRESENT_PATH} не должен найтись в {tmp}, получено {reason!r}")
    reason = parity.missing_requirements(_entry([PRESENT_PATH]), root=REPO_ROOT)
    if reason is not None:
        raise AssertionError(f"в корне репозитория {PRESENT_PATH} есть, а причина {reason!r}")
    print("  ok: root подменяет базу относительных путей")


def case_run_all_skip(tmp: Path) -> None:
    """`_run_all`: невыполнимый `requires` -> `[skip]` и `skipped 1`, exit 0; FAIL не подменяется.

    Синтетический эталон из двух записей: одна пропускается (нет весов), вторая — без `requires`,
    но с неверным sha256 (файл не создаётся, `_compare` даёт FAIL). Проверяем, что пропуск
    печатается отдельной строкой, не попадает ни в PASS, ни в FAIL, и что FAIL всё равно роняет код.
    """
    reference: dict[str, Any] = {"entries": {
        "skip-me": _entry([ABSENT_REPO], sha256="0" * 64, image_size=[8, 8]),
        "fail-me": _entry(sha256="f" * 64, image_size=[8, 8]),
    }}
    out_root = str(tmp / "parity-out")

    def fake_run_entry(name: str, entry: Any, *, out_root: str = "") -> list[dict[str, Any]]:
        """Заглушка прогона: файла нет, поэтому `_compare` честно вернёт FAIL."""
        return [parity._compare(name, str(entry.get("sha256")), str(Path(out_root) / f"{name}.png"))]

    original = parity.run_entry
    parity.run_entry = fake_run_entry               # type: ignore[assignment] — подмена на время кейса
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            both = parity._run_all(reference, ["skip-me"], out_root)
        first = buf.getvalue()
    finally:
        parity.run_entry = original

    if both != 0:
        raise AssertionError(f"только пропуск: ожидался exit 0, получено {both}\n{first}")
    if "[skip] skip-me:" not in first:
        raise AssertionError(f"нет строки [skip]: {first!r}")
    if "PASS" in first.split("[check-parity]")[0] or "FAIL" in first.split("[check-parity]")[0]:
        raise AssertionError(f"пропуск выдаётся за PASS/FAIL: {first!r}")
    if "skipped 1" not in first:
        raise AssertionError(f"в итоговой строке нет «skipped 1»: {first!r}")
    if "[entry] skip-me" not in first:
        raise AssertionError(f"нет строки [entry] (формат изменился): {first!r}")
    print(f"  ok: [skip] без PASS/FAIL, exit 0, итог «skipped 1»")

    parity.run_entry = fake_run_entry
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            failed = parity._run_all(reference, ["skip-me", "fail-me"], out_root)
        second = buf.getvalue()
    finally:
        parity.run_entry = original

    if failed == 0:
        raise AssertionError(f"FAIL рядом с пропуском должен давать exit != 0\n{second}")
    if "skipped 1" not in second or "FAIL 1" not in second:
        raise AssertionError(f"итог не разделяет FAIL и skipped: {second!r}")
    print("  ok: FAIL рядом с пропуском по-прежнему роняет прогон (exit != 0)")


CASES: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("1. пустой/отсутствующий requires -> None", case_empty_requires),
    ("2. существующий репозиторий Хаба -> None", case_present_repo),
    ("3. несуществующий репозиторий -> причина", case_absent_repo),
    ("4. явные пути: отсутствующий -> причина, существующий -> None", case_paths),
    ("5. FLUX.2-репозиторий -> None", case_flux2_repo),
    ("6. hf_cache/root: пустой кэш -> причина, root меняет базу путей", case_hf_cache_and_root),
    ("7. _run_all: [skip] не PASS и не FAIL, FAIL всё равно роняет код", case_run_all_skip),
)


def main() -> int:
    failed = 0
    with tempfile.TemporaryDirectory(prefix="qwenstyle-parity-skip-") as raw:
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
