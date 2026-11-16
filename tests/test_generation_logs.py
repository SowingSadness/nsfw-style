"""Логи прогона: две схемы раскладки (`run` -> `<кадр>/run.log`, `gen` -> корень сессии).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_generation_logs.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.
Без CUDA печатается `SKIP (нет CUDA)` — генерация невозможна, exit 0; логика раскладки
(`case_out_root_untouched`) при этом всё равно проверяется.

Зачем: прогон печатает всю диагностику (`[profile]`, `[size]`, `[in]`, `[1/4]`…`[4/4]`, ключ
кэша, время фаз, `[post]`, `[done]`), но жила она только в консоли — по кадру нельзя было понять,
чем он сделан. Теперь лог пишется в файл (`qwenstyle/runlog.py`) и у раскладки **две схемы**:

* `run` — папка кадра: `out/tests/<кадр>/run.log` + `meta.json`;
* `gen` — папка сессии: `out/<имя>/gen-YYYYMMDD-HHMM/run.log` + `meta.json` **в корне**, рядом с
  кадрами `s{N}.png` (один лог на сессию из N seed'ов, подпапок на кадр нет).

Тест фиксирует то, что легко сломать незаметно:

1. **схема `run`**: папка лога там, где кадр, в логе есть и ранние строки (`[profile]`, он печатается
   до открытия файла — значит буфер работает), и финальная `[done]`;
2. **метки времени**: каждая содержательная строка лога начинается с `[HH:MM:SS] ` (прогресс-бары
   tqdm — исключение: они перерисовывают одну строку через `\\r`);
3. **`--no-logs`** ничего не создаёт, но кадр генерируется как обычно;
4. **схема `gen`**: `run.log`/`meta.json` в корне сессии, кадр `s{N}.png` там же, подпапки на кадр нет;
5. **результаты тестов лежат только в `out/tests/`**: корень `out/` принадлежит пользователю.

Генерация настоящая, но дешёвая: пресет `strong-preview` (Qwen-Image-2.1, 1 шаг, 864×480), вход —
синтетический градиент, промт — `tests/texts/exp1.txt`. Первый прогон платит энкод, остальные идут
с попаданием в кэш эмбеддингов (R1). Сессия `gen` выполняется в **временном каталоге** (там свои
`prompts/` и `out/`), поэтому `out/` репозитория не трогается. `cache_supersede=none` — тест не
вытесняет чужие записи кэша.
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

import _shared as shared

#: Seed прогонов: из имени кадра видно, что тег `-s…` и папка лога — про один и тот же кадр.
SEED = 7
#: Метка времени строки лога (`qwenstyle/runlog._stamp`).
STAMP_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\] ")
#: Строка прогресс-бара tqdm: перерисовывается через `\r`, метку времени не получает.
PROGRESS_RE = re.compile(r"\d+%\|")
#: Логичная и дешёвая сессия `gen`: 480p, один шаг, без апскейла (у `strong-preview` его нет).
GEN_PRESET = "strong-preview"
GEN_SETS = ("size_px=480", "generation.steps=1", "cache_supersede=none")


def _frame(out: str) -> Path:
    """Файл кадра: `out` + `-s{seed}` + `.png` (теги дописывает `io.out_path`)."""
    return Path(out).with_name(f"{Path(out).stem}-s{SEED}.png")


def _run(*extra: str, out: str) -> "tuple[int, str]":
    """Прогон `run` с общими флагами теста; вернуть (код возврата, консольный вывод)."""
    result = shared.run_cli(
        "run",
        "--preview",                     # черновой режим: 480p, дешёвый кадр
        "--steps", "1",                  # ... но нам хватает одного: проверяем логи, а не картинку
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


def case_log_folder() -> None:
    """1. `run` по умолчанию: кадр + папка `<кадр>/` с `run.log` и `meta.json` рядом."""
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
    first = re.sub(STAMP_RE, "", _log_lines(text)[0])
    if not first.startswith("[profile]"):
        raise AssertionError(f"лог начинается не с [profile]: {_log_lines(text)[0]!r}")
    _assert_stamped(text, "run.log")
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


def case_no_logs() -> None:
    """2. `--no-logs`: кадр есть, папки лога нет, в консоли нет `[log]`."""
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


def case_gen_session_log() -> None:
    """3. Схема `gen`: `run.log`/`meta.json` в корне сессии, кадры `s{N}.png` рядом, без подпапок.

    Корни пользовательских каталогов (`prompts/`, `out/`) подменяются на **временный** каталог:
    тест не засоряет репозиторий, а схема лога (`cfg.log_dir` -> `RunLog.attach`) остаётся той
    же, что у пользователя. `cwd` не меняется — иначе относительные пути кешей (`cache/models/…`,
    `precision.resolve`) уехали бы во временный каталог и тест пересобрал бы 8.5 GiB fp8-кэша.
    Пресет — `strong-preview` с `size_px=480` и `steps=1`: та же сетка, промт и вход, что у кейса 1,
    поэтому энкодер не платится.
    """
    import functools

    from qwenstyle.cmd import gen as gen_cmd            # импорт не тянет torch

    with tempfile.TemporaryDirectory(prefix="qwenstyle-gen-log-") as tmp:
        root = Path(tmp)
        prompt_root = root / "prompts"
        out_root = root / "out"
        folder = prompt_root / "case"
        folder.mkdir(parents=True)
        shutil.copyfile(shared.TEST_PROMPT, folder / gen_cmd.PROMPT_FILE)
        shutil.copyfile(shared.out_tests("input-864x480.png"), folder / "image1.png")

        args = argparse.Namespace(preset=GEN_PRESET, prompt="case", aspect=None,
                                  sets=list(GEN_SETS), seeds=str(SEED), count=None,
                                  dry_run=False, no_logs=False)
        original_prompt_root = gen_cmd.PROMPT_ROOT
        original_session_dir = gen_cmd.session_dir
        gen_cmd.PROMPT_ROOT = str(prompt_root)
        gen_cmd.session_dir = functools.partial(original_session_dir, out_root=str(out_root))
        try:
            code = gen_cmd.run(args)
        finally:
            gen_cmd.PROMPT_ROOT = original_prompt_root
            gen_cmd.session_dir = original_session_dir
        if code != 0:
            raise AssertionError(f"gen завершился с кодом {code}")

        sessions = sorted((out_root / "case").glob("gen-*"))
        if len(sessions) != 1:
            raise AssertionError(f"сессия не создана (найдено {len(sessions)})")
        session = sessions[0]
        if not re.fullmatch(r"gen-\d{8}-\d{4}(-\d+)?", session.name):
            raise AssertionError(f"имя сессии не gen-YYYYMMDD-HHMM: {session.name!r}")

        log = session / "run.log"
        meta = session / "meta.json"
        if not log.is_file() or not meta.is_file():
            raise AssertionError(f"в корне сессии нет run.log/meta.json: "
                                 f"{sorted(p.name for p in session.iterdir())}")
        frames = sorted(path.name for path in session.glob("s*.png"))
        if f"s{SEED}.png" not in frames:
            raise AssertionError(f"в корне сессии нет кадра s{SEED}.png: {frames}")
        if (session / f"s{SEED}").is_dir():
            raise AssertionError("ген создал подпапку на кадр — лог обязан быть один на сессию")
        if not (session / f"s{SEED}.png").is_file():
            raise AssertionError(f"кадр не файл: {session / f's{SEED}.png'}")

        text = log.read_text(encoding="utf-8", errors="replace")
        for marker in ("[size]", "[in]", "[log]", "[done]", "[gen] итог"):
            if marker not in text:
                raise AssertionError(f"в логе сессии нет {marker}: {text[:300]!r}")
        _assert_stamped(text, "session/run.log")

        payload = json.loads(meta.read_text(encoding="utf-8"))
        section = payload.get("session") or {}
        if [int(seed) for seed in section.get("seeds") or []] != [SEED]:
            raise AssertionError(f"meta.json session.seeds={section.get('seeds')!r}, "
                                 f"ожидалось [{SEED}]")
        if Path(section.get("dir") or "").name != session.name:
            raise AssertionError(f"meta.json session.dir={section.get('dir')!r}, "
                                 f"ожидалась сессия {session.name}")
        print(f"  ok: gen -> {session.name}/{{run.log,meta.json,s{SEED}.png}}, кадров {len(frames)}")


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
            ("1. run: папка лога рядом с кадром", case_log_folder),
            ("2. --no-logs отключает лог", case_no_logs),
            ("3. gen: run.log в корне сессии", case_gen_session_log),
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
