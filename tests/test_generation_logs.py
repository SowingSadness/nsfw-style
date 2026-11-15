"""Логи прогона и раскладка результатов тестов: `out/tests/<кадр>/run.log`, `--no-logs`.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_generation_logs.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.
Без CUDA печатается `SKIP (нет CUDA)` — генерация невозможна, exit 0.

Зачем: прогон уже печатает всю диагностику (`[profile]`, `[size]`, `[in]`, `[1/4]`…`[4/4]`, ключ
кэша, время фаз, `[done]`), но жила она только в консоли — по кадру нельзя было понять, чем он
сделан. Теперь у каждого прогона своя папка рядом с кадром (`qwenstyle/runlog.py`), а `--no-logs`
её отключает. Тест фиксирует три вещи, которые легко сломать незаметно:

1. **папка лога появилась там, где кадр**: `out/tests/<кадр>/run.log` + `meta.json`, и в логе есть
   и ранние строки (`[profile]` — он печатается до открытия файла, значит буфер работает),
   и финальная `[done]`;
2. **`--no-logs` ничего не создаёт** (ни папки, ни файла), но кадр генерируется как обычно;
3. **результаты тестов лежат только в `out/tests/`**: корень `out/` принадлежит пользователю,
   туда тест писать не должен (промт теста — `tests/texts/exp1.txt`, а не `promts/`).

Генерация настоящая, но дешёвая: пресет `preview` (FLUX.2 klein, 1 шаг, 864x480), вход —
синтетический градиент, промт — `tests/texts/exp1.txt`. Второй прогон идёт с попаданием в кэш
эмбеддингов (R1), поэтому платится только генерация. `cache_supersede=none` — тест не должен
вытеснять чужие записи кэша.
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import _shared as shared


#: Seed прогонов: из имени кадра видно, что тег `-s…` и папка лога — про один и тот же кадр.
SEED = 7


def _frame(out: str) -> Path:
    """Файл кадра: `out` + `-s{seed}` + `.png` (теги дописывает `io.out_path`)."""
    return Path(out).with_name(f"{Path(out).stem}-s{SEED}.png")


def _run(*extra: str, out: str) -> "tuple[int, str]":
    """Прогон `run` с общими флагами теста; вернуть (код возврата, консольный вывод)."""
    result = shared.run_cli(
        "run",
        "--preview",                     # FLUX.2 klein, 8 шагов по 1, 1376x768-база
        "--steps", "1",                  # ... но нам хватает одного: проверяем логи, а не картинку
        "--seed", str(SEED),
        "--prompt-file", str(shared.TEST_PROMPT),
        "--input", str(shared.out_tests("input-864x480.png")),
        "--out", out,
        "--set", "cache_supersede=none",  # регресс-тест не вытесняет записи кэша (R2)
        *extra,
    )
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def case_log_folder() -> None:
    """1. Прогон по умолчанию: кадр + папка `<кадр>/` с `run.log` и `meta.json` рядом."""
    out = str(shared.out_tests("logs", "frame.png"))
    frame_out = _frame(out)
    folder = frame_out.with_suffix("")           # `out/tests/logs/frame-s7`
    for stale in (frame_out, folder / "run.log", folder / "meta.json"):
        stale.unlink(missing_ok=True)

    code, console = _run(out=out)
    if code != 0:
        raise AssertionError(f"run завершился с кодом {code}: {console[-800:]}")
    if not frame_out.is_file():
        raise AssertionError(f"кадр не создан: {frame_out}")

    log = folder / "run.log"
    meta = folder / "meta.json"
    if not log.is_file():
        raise AssertionError(f"нет лога {log} (папка {folder} существует: {folder.is_dir()})")
    if not meta.is_file():
        raise AssertionError(f"нет сайдкара параметров {meta}")
    if f"[log] {str(log).replace(chr(92), '/')}" not in console:
        raise AssertionError("в консоли нет строки [log] с путём лога")

    text = log.read_text(encoding="utf-8", errors="replace")
    for marker in ("[profile]", "[in]", "[log]", "[done]"):
        if marker not in text:
            raise AssertionError(f"в логе нет {marker}: начало файла {text[:300]!r}")
    # Ранняя строка [profile] печатается ДО открытия файла: она обязана попасть в лог из буфера.
    if not text.lstrip().startswith("[profile]"):
        raise AssertionError(f"лог начинается не с [profile]: {text[:200]!r}")
    print("  ok: кадр и папка <кадр>/{run.log,meta.json} на месте, [profile] не потерялся")

    params = json.loads(meta.read_text(encoding="utf-8")).get("params") or {}
    want_prompt = str(shared.TEST_PROMPT).replace("\\", "/")
    got_prompt = str(params.get("prompt_file") or "").replace("\\", "/")
    if got_prompt != want_prompt:
        raise AssertionError(f"meta.json: prompt_file={params.get('prompt_file')!r}, "
                             f"ожидался {shared.TEST_PROMPT}")
    if params.get("logs") is not True:
        raise AssertionError(f"meta.json: logs={params.get('logs')!r}, ожидалось True")
    if params.get("seed") != SEED:
        raise AssertionError(f"meta.json: seed={params.get('seed')!r}, ожидался {SEED}")
    print("  ok: meta.json описывает прогон (промт из tests/texts, seed, logs=True)")


def case_out_root_untouched() -> None:
    """2. Корень `out/` не тронут: результаты тестов живут только в `out/tests/`."""
    root = shared.REPO_ROOT / "out"
    suspicious = sorted(path.name for path in root.iterdir()
                        if path.name.startswith(("frame", "nologs", "logs")))
    if suspicious:
        raise AssertionError(f"тест писал в корень out/: {suspicious}")
    if not str(shared.OUT_TESTS).startswith(str(root)):
        raise AssertionError(f"OUT_TESTS вне out/: {shared.OUT_TESTS}")
    if not shared.TEST_PROMPT.is_file():
        raise AssertionError(f"нет промта тестов {shared.TEST_PROMPT}")
    if str(shared.TEST_PROMPT).replace("\\", "/").find("/promts/") >= 0:
        raise AssertionError("промт тестов не должен жить в promts/ (он в .gitignore)")
    print("  ok: out/ чист (кадры и логи — в out/tests, промт — в tests/texts)")


def case_no_logs() -> None:
    """3. `--no-logs`: кадр есть, папки лога нет, в консоли нет `[log]`."""
    out = str(shared.out_tests("logs", "nologs.png"))
    frame_out = _frame(out)
    folder = frame_out.with_suffix("")
    for stale in (frame_out, folder / "run.log", folder / "meta.json"):
        stale.unlink(missing_ok=True)

    code, console = _run("--no-logs", out=out)
    if code != 0:
        raise AssertionError(f"run --no-logs завершился с кодом {code}: {console[-800:]}")
    if not frame_out.is_file():
        raise AssertionError(f"--no-logs сломал генерацию: нет кадра {frame_out}")
    if folder.exists():
        raise AssertionError(f"--no-logs создал папку лога {folder}")
    if "[log]" in console:
        raise AssertionError("--no-logs: в консоли всё равно есть строка [log]")
    print("  ok: --no-logs -> кадр без папки лога")


def main() -> int:
    if not shared.has_cuda():
        print("SKIP (нет CUDA): генерация недоступна, проверять логи нечего")
        return 0

    shared.out_tests("logs")
    shared.make_input_image(shared.out_tests("input-864x480.png"))

    cases = (
        ("1. папка лога рядом с кадром", case_log_folder),
        ("2. результаты тестов только в out/tests", case_out_root_untouched),
        ("3. --no-logs отключает лог", case_no_logs),
    )
    failed = 0
    for label, case in cases:
        try:
            print(f"PASS {label}" if case() is None else f"FAIL {label}: неизвестный результат",
                  flush=True)
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
