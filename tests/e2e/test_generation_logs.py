"""Логи и meta прогона: `<кадр>.log` и `<кадр>-meta.json` рядом с кадром (run и gen).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_generation_logs.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.
Без CUDA печатается `SKIP (нет CUDA)` — генерация невозможна, exit 0; логика раскладки
(`case_out_root_untouched`) при этом всё равно проверяется.

Зачем: прогон печатает всю диагностику (`[profile]`, `[size]`, `[in]`, `[1/4]`…`[4/4]`, ключ
кэша, время фаз, `[post]`, `[done]`), но жила она только в консоли — по кадру нельзя было понять,
чем он сделан. Теперь лог и meta лежат **рядом с кадром и называются его именем**:

* `run` — `out/tests/x-7-1.png`, `out/tests/x-7-1.log`, `out/tests/x-7-1-meta.json`;
* `gen` — в каталоге сессии на каждый кадр `<seed>-<steps>.png` и `<seed>-<steps>.log`, а
  `meta.json` **один на всю сессию** (20 сидов = 20 логов + 1 meta, а не 20 копий общей части).

Тест фиксирует то, что легко сломать незаметно:

1. **`run`**: лог и meta рядом с кадром, подпапки нет; в логе есть и ранние строки (`[profile]`
   печатается до открытия файла — значит буфер работает), и финальная `[done]`;
2. **метки времени**: каждая содержательная строка лога начинается с `[HH:MM:SS] ` (прогресс-бары
   tqdm — исключение: они перерисовывают одну строку через `\\r`);
3. **`--no-logs`** убирает `.log`, но `<кадр>-meta.json` остаётся (так обещает help);
4. **`gen`**: у каждого кадра свой `.log`, `meta.json` один на сессию; в логе есть общий заголовок (`[in]`,
   `[size]`) и `[gen] итог`; сессия — `<out>/<штамп>-<tag>/`;
5. **результаты тестов лежат только в `out/tests/`**: корень `out/` принадлежит пользователю.

Генерация настоящая, но дешёвая: пресет `VRAM12-preview` (Qwen-Image-2.1, 1 шаг, 864×480), вход —
синтетический градиент, промт — `tests/texts/exp1.txt`. Первый прогон платит энкод, остальные идут
с попаданием в кэш эмбеддингов (R1). Сессия `gen` выполняется в **временном каталоге** (там свои
`prompts/`), поэтому `out/` репозитория не трогается. `cache_supersede=none` — тест не вытесняет
чужие записи кэша.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

# `_shared.py` лежит на уровень выше (`tests/`): тесты живут в `tests/e2e/`, а запускаются скриптом,
# поэтому `sys.path[0]` — сам `tests/e2e/`, и без этого импорта `import _shared` не найдётся.
_TESTS_DIR = Path(__file__).resolve().parents[1]
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import _shared as shared

#: Seed прогонов: из имени кадра видно, что тег и лог — про один и тот же кадр.
SEED = 7
#: Шагов в тестовых прогонах (`--steps 1`): имя кадра `<seed>-<steps>.png`.
STEPS = 1
#: Метка времени строки лога (`qwenstyle/runlog._stamp`).
STAMP_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\] ")
#: Строка прогресс-бара tqdm: перерисовывается через `\r`, метку времени не получает.
PROGRESS_RE = re.compile(r"\d+%\|")
#: Логичная и дешёвая сессия `gen`: 480p, один шаг, без апскейла (у `VRAM12-preview` его нет).
GEN_PRESET = "VRAM12-preview"
GEN_SETS = ("size_px=480", "generation.steps=1", "cache_supersede=none")


def _frame(out: str) -> Path:
    """Файл кадра: `out` + `-<seed>-<steps>` + `.png` (теги дописывает `io.out_path`)."""
    return Path(out).with_name(f"{Path(out).stem}-{SEED}-{STEPS}.png")


def _log(frame: Path) -> Path:
    return frame.with_suffix(".log")


def _meta(frame: Path) -> Path:
    return frame.with_name(f"{frame.stem}-meta.json")


def _run(*extra: str, out: str) -> "tuple[int, str]":
    """Прогон `run` с общими флагами теста; вернуть (код возврата, консольный вывод)."""
    result = shared.run_cli(
        "run",
        "--preview",                     # черновой режим: 480p, дешёвый кадр
        "--steps", str(STEPS),           # ... но нам хватает одного: проверяем логи, а не картинку
        "--seed", str(SEED),
        "--prompt-file", str(shared.TEST_PROMPT),
        "--input", str(shared.out_tests("input-864x480.png")),
        "--out", out,
        "--set", "cache_supersede=none",  # регресс-тест не вытесняет записи кэша (R2)
        *extra,
    )
    return result.returncode, (result.stdout or "") + (result.stderr or "")


def _log_lines(text: str) -> list[str]:
    """Непустые строки лога (файл читается в universal-newline режиме: `\\r` тоже разделитель)."""
    return [line for line in text.split("\n") if line.strip()]


def _assert_stamped(text: str, where: str) -> None:
    """Каждая строка с текстом получает метку времени; исключение — прогресс-бар tqdm."""
    lines = _log_lines(text)
    if not lines:
        raise AssertionError(f"{where}: пустой лог")
    unstamped = [line for line in lines
                 if not STAMP_RE.match(line) and not PROGRESS_RE.search(line)]
    if unstamped:
        raise AssertionError(f"{where}: строки без метки [HH:MM:SS] (не прогресс-бар): "
                             f"{unstamped[:3]!r}")
    if not STAMP_RE.match(lines[0]):
        raise AssertionError(f"{where}: первая строка без метки времени: {lines[0]!r}")
    stamped = sum(1 for line in lines if STAMP_RE.match(line))
    print(f"  ok: {where}: {stamped} строк с меткой [HH:MM:SS], прогресс-бары без метки")


def case_run_logs() -> None:
    """1. `run`: лог и meta рядом с кадром (без подпапки), [profile] не потерялся."""
    out = str(shared.out_tests("logs", "frame.png"))
    frame = _frame(out)
    for stale in (frame, _log(frame), _meta(frame)):
        stale.unlink(missing_ok=True)

    code, console = _run(out=out)
    if code != 0:
        raise AssertionError(f"run завершился с кодом {code}: {console[-800:]}")
    if not frame.is_file():
        raise AssertionError(f"кадр не создан: {frame}")
    log, meta = _log(frame), _meta(frame)
    if not log.is_file():
        raise AssertionError(f"нет лога {log}")
    if not meta.is_file():
        raise AssertionError(f"нет meta {meta}")
    if f"[log] {log.as_posix()}" not in console:
        raise AssertionError("в консоли нет строки [log] с путём лога")
    if frame.with_suffix("").is_dir():
        raise AssertionError("run создал подпапку кадра — лог обязан быть рядом с кадром")

    text = log.read_text(encoding="utf-8", errors="replace")
    for marker in ("[profile]", "[in]", "[log]", "[done]"):
        if marker not in text:
            raise AssertionError(f"в логе нет {marker}: начало файла {text[:300]!r}")
    # Ранняя строка [profile] печатается ДО открытия файла: она обязана попасть в лог из буфера.
    first = re.sub(STAMP_RE, "", _log_lines(text)[0])
    if not first.startswith("[profile]"):
        raise AssertionError(f"лог начинается не с [profile]: {_log_lines(text)[0]!r}")
    _assert_stamped(text, "frame.log")

    payload = json.loads(meta.read_text(encoding="utf-8"))
    params = payload.get("params") or {}
    want_prompt = str(shared.TEST_PROMPT).replace("\\", "/")
    got_prompt = str(params.get("prompt_file") or "").replace("\\", "/")
    if got_prompt != want_prompt:
        raise AssertionError(f"meta: prompt_file={params.get('prompt_file')!r}, "
                             f"ожидался {shared.TEST_PROMPT}")
    if params.get("seed") != SEED:
        raise AssertionError(f"meta: seed={params.get('seed')!r}, ожидался {SEED}")
    if payload.get("seed") != SEED or payload.get("steps") != STEPS:
        raise AssertionError(f"meta: seed/steps={payload.get('seed')!r}/{payload.get('steps')!r}")
    if "timings" not in payload:
        raise AssertionError("meta: нет таймингов кадра")
    print("  ok: кадр + .log + -meta.json в одном каталоге; [profile] из буфера, тайминги есть")


def case_no_logs() -> None:
    """2. `--no-logs`: кадра нет в логе, `.log` не создан, meta осталась."""
    out = str(shared.out_tests("logs", "nologs.png"))
    frame = _frame(out)
    for stale in (frame, _log(frame), _meta(frame)):
        stale.unlink(missing_ok=True)

    code, console = _run("--no-logs", out=out)
    if code != 0:
        raise AssertionError(f"run --no-logs завершился с кодом {code}: {console[-800:]}")
    if not frame.is_file():
        raise AssertionError(f"--no-logs сломал генерацию: нет кадра {frame}")
    if _log(frame).exists():
        raise AssertionError(f"--no-logs создал лог {_log(frame)}")
    if "[log]" in console:
        raise AssertionError("--no-logs: в консоли всё равно есть строка [log]")
    if not _meta(frame).is_file():
        raise AssertionError("--no-logs убрал meta, хотя help обещает её сохранить")
    print("  ok: --no-logs -> без .log, но -meta.json на месте")


def case_gen_session_log() -> None:
    """3. `gen`: у каждого кадра свой `.log`, `meta.json` — один на всю сессию."""
    from qwenstyle.cmd import _project as project    # импорт не тянет torch
    from qwenstyle.cmd import gen as gen_cmd

    with tempfile.TemporaryDirectory(prefix="qwenstyle-gen-log-") as tmp:
        root = Path(tmp)
        prompt_root = root / "prompts"
        out_root = root / "out"
        folder = prompt_root / "case"
        (folder / "refs").mkdir(parents=True)
        shutil.copyfile(shared.TEST_PROMPT, folder / "prompt-qwen.txt")
        shutil.copyfile(shared.out_tests("input-864x480.png"), folder / "refs" / "01-input.png")

        args = argparse.Namespace(preset=GEN_PRESET, prompt="case/prompt-qwen", project=None,
                                  out=str(out_root), aspect=None, sets=list(GEN_SETS),
                                  upscale=None, seeds=str(SEED), count=None,
                                  dry_run=False, no_logs=False)
        original_prompt_root = project.PROMPT_ROOT
        project.PROMPT_ROOT = prompt_root.as_posix()
        try:
            code = gen_cmd.run(args)
        finally:
            project.PROMPT_ROOT = original_prompt_root
        if code != 0:
            raise AssertionError(f"gen завершился с кодом {code}")

        sessions = sorted(out_root.glob("*-qwen"))
        if len(sessions) != 1:
            raise AssertionError(f"сессия не создана (найдено {len(sessions)})")
        session = sessions[0]
        if not re.fullmatch(r"\d{12}-qwen(-\d+)?", session.name):
            raise AssertionError(f"имя сессии не <YYYYMMDDHHMM>-qwen: {session.name!r}")

        frame = session / f"{SEED}-{STEPS}.png"
        log = session / f"{SEED}-{STEPS}.log"
        meta = session / "meta.json"
        if not frame.is_file() or not log.is_file() or not meta.is_file():
            raise AssertionError(f"нет кадра/лога/meta: "
                                 f"{sorted(path.name for path in session.iterdir())}")
        per_frame_meta = list(session.glob("*-meta.json"))
        if per_frame_meta:
            raise AssertionError(f"meta.json обязан быть один на сессию, а найдено "
                                 f"{[path.name for path in per_frame_meta]}")
        if (session / f"{SEED}-{STEPS}").is_dir():
            raise AssertionError("gen создал подпапку на кадр — лог обязан быть рядом")

        text = log.read_text(encoding="utf-8", errors="replace")
        for marker in ("[size]", "[in]", "[log]", "[done]", "[gen] итог"):
            if marker not in text:
                raise AssertionError(f"в логе кадра нет {marker}: {text[:300]!r}")
        _assert_stamped(text, "session/<frame>.log")

        payload = json.loads(meta.read_text(encoding="utf-8"))
        if [int(seed) for seed in payload.get("seeds") or []] != [SEED]:
            raise AssertionError(f"meta.seeds={payload.get('seeds')!r}, ожидалось [{SEED}]")
        if Path(payload.get("prompt_file") or "").name != "prompt-qwen.txt":
            raise AssertionError(f"meta.prompt_file={payload.get('prompt_file')!r}")
        frames = payload.get("frames") or []
        if len(frames) != 1 or int(frames[0].get("seed", -1)) != SEED:
            raise AssertionError(f"meta.frames={frames!r}, ожидался один кадр seed={SEED}")
        if Path(frames[0].get("file") or "").name != frame.name:
            raise AssertionError(f"meta.frames[0].file={frames[0].get('file')!r}, "
                                 f"ожидалось {frame.name}")
        if int((payload.get("totals") or {}).get("frames", 0)) != 1:
            raise AssertionError(f"meta.totals={payload.get('totals')!r}")
        print(f"  ok: gen -> {session.name}/{{{frame.name},{log.name},{meta.name}}}")


def case_out_root_untouched() -> None:
    """4. Корень `out/` не тронут: результаты тестов живут только в `out/tests/`."""
    root = shared.REPO_ROOT / "out"
    suspicious = sorted(path.name for path in root.iterdir()
                        if path.name.startswith(("frame", "nologs", "logs", "case")))
    if suspicious:
        raise AssertionError(f"тест писал в корень out/: {suspicious}")
    if not str(shared.OUT_TESTS).startswith(str(root)):
        raise AssertionError(f"OUT_TESTS вне out/: {shared.OUT_TESTS}")
    if not shared.TEST_PROMPT.is_file():
        raise AssertionError(f"нет промта тестов {shared.TEST_PROMPT}")
    if str(shared.TEST_PROMPT).replace("\\", "/").find("/prompts/") >= 0:
        raise AssertionError("промт тестов не должен жить в prompts/ (он в .gitignore)")
    print("  ok: out/ чист (кадры и логи — в out/tests, промт — в tests/texts)")


def main() -> int:
    shared.out_tests("logs")
    if shared.has_cuda():
        shared.make_input_image(shared.out_tests("input-864x480.png"))
        cases = (
            ("1. run: .log и -meta.json рядом с кадром", case_run_logs),
            ("2. --no-logs отключает .log, meta остаётся", case_no_logs),
            ("3. gen: .log у кадра, meta.json один на сессию", case_gen_session_log),
            ("4. результаты тестов только в out/tests", case_out_root_untouched),
        )
    else:
        print("SKIP (нет CUDA): генерация недоступна, проверяется только раскладка каталогов")
        cases = (("4. результаты тестов только в out/tests", case_out_root_untouched),)
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
