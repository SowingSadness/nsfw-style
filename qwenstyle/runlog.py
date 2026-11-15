"""Отладочный лог прогона: подпапка рядом с кадром (`out/<кадр>/run.log` + `meta.json`).

Зачем: stdout прогона уже содержит всю диагностику — `[profile]`, `[plan]` (сетка, ключ кэша,
попадание/промах), `[1/5]`…`[5/5]` от бекенда, время фаз, `[post]`, `[done]`. Но живёт он только
в консоли: при ручном редиректе (`> out/run.log`) теряется привязка к кадру, а прогоны разных
seed'ов/резолюций пишут в один и тот же файл и затирают друг друга. Поэтому лог кладётся
**в свою папку на каждый кадр**:

    out/rainforest-s20261113.png          <- кадр (путь не изменился)
    out/rainforest-s20261113/run.log      <- весь вывод этого прогона
    out/rainforest-s20261113/meta.json    <- параметры прогона (чем именно он сделан)

Правила:

* **Имя папки** = путь кадра без расширения (`io.log_dir`). В имени кадра уже есть `-s{seed}`
  и `-x{factor}`, поэтому две генерации не смешиваются и лог всегда лежит рядом со своим кадром.
* **Метки времени**: каждая непустая строка лога (и в консоли, и в файле) получает
  префикс `[HH:MM:SS] ` — по нему видно, сколько шла каждая фаза кадра.
* **Tee, а не редирект**: stdout и stderr одновременно идут в консоль и в файл — прогон остаётся
  интерактивным (tqdm-прогресс виден), а traceback падения попадает в лог.
* **Лог открывается позже, чем начинается печать**: имя кадра известно только после `_plan`
  (там же решается `-x{factor}`), поэтому ранние строки копятся в буфере и дописываются в файл
  в момент `attach`. Порядок строк в логе совпадает с консолью.
* **`--no-logs`** (`RunConfig.logs=False`) выключает всё: ни папки, ни файла, ни tee — вывод
  только в консоль.
* **Лог — наблюдение, а не результат**: любая ошибка файловой системы не роняет прогон,
  печатается одна строка `[log] …`, и тест/генерация продолжаются.
* **Сессия (`gen`)** — один процесс на N seed'ов, поэтому **одна папка на сессию**, а не на кадр:
  `cfg.log_dir` (`out/<имя>/gen-YYYYMMDD-HHMM/`) задаёт её явно (`pipeline._open_log`), и
  `run.log`/`meta.json` лежат в корне сессии рядом с кадрами `s{N}.png`. Без `cfg.log_dir`
  `run_batch` падает в прежнюю папку `<out-dir>/_logs/<YYYYmmdd-HHMMSS>/` (`batch_log_dir`).

Модуль без тяжёлых импортов: только stdlib, чтобы `--help` и `dry-run` оставались мгновенными.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, TextIO

#: Имя файла лога внутри папки прогона.
LOG_NAME = "run.log"
#: Имя сайдкара с параметрами прогона (рядом с логом, не рядом с кадром).
META_NAME = "meta.json"
#: Подпапка батча: `<out-dir>/_logs/<stamp>`. Ведущее `_` — чтобы не путать с кадрами `s{N}.png`.
BATCH_LOGS_DIR = "_logs"


class _Sink:
    """Приёмник строк лога: сначала буфер в памяти, после `attach` — открытый файл.

    Буфер нужен потому, что первые строки (`[profile]`, `[plan]`) печатаются до того, как
    известно имя кадра. Строки не теряются и не переставляются: `attach` списывает буфер
    в файл и дальше пишет напрямую.
    """

    def __init__(self) -> None:
        self.file: TextIO | None = None
        self.pending: list[str] = []

    def write(self, text: str) -> None:
        if self.file is not None:
            self.file.write(text)
            self.file.flush()
        else:
            self.pending.append(text)

    def flush(self) -> None:
        if self.file is not None:
            self.file.flush()

    def attach(self, file: TextIO) -> None:
        """Открыть файл и списать в него всё, что накопилось в буфере."""
        self.file = file
        if self.pending:
            file.write("".join(self.pending))
            self.pending.clear()
        file.flush()

    def close(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None


def _stamp() -> str:
    """Метка времени для строки лога: `[HH:MM:SS] ` (локальное время машины)."""
    return time.strftime("[%H:%M:%S] ")


class _Tee:
    """Поток-обёртка: пишет и в консоль, и в лог, добавляя каждой строке метку времени.

    Метка ставится **построчно** (буферизуется неполная строка): `print()` может прийти двумя
    вызовами `write` («текст», затем перевод строки), и тогда штамп получила бы только первая половина.
    Пустые строки не штампуются — это разделители, а не сообщения.

    `__getattr__` пробрасывает `isatty`/`encoding`/`fileno`/`buffer` к исходному потоку, иначе
    tqdm решил бы, что вывод перенаправлен в файл, и перестал рисовать прогресс.
    """

    def __init__(self, console: TextIO, sink: _Sink) -> None:
        object.__setattr__(self, "_console", console)
        object.__setattr__(self, "_sink", sink)
        object.__setattr__(self, "_partial", "")

    def _stamped(self, text: str) -> str:
        """Дописать текст к буферу и вернуть готовые строки с метками времени."""
        buf = self._partial + text
        if chr(10) not in buf:
            object.__setattr__(self, "_partial", buf)
            return ""
        out: list[str] = []
        *lines, rest = buf.split(chr(10))
        object.__setattr__(self, "_partial", rest)
        for line in lines:
            if line.strip():
                out.append(_stamp() + line + chr(10))
            else:
                out.append(line + chr(10))
        return "".join(out)

    def write(self, text: str) -> int:
        if chr(13) in text:
            # Прогресс-бар tqdm перерисовывает одну строку через возврат каретки:
            # метку не ставим, иначе каждый кадр полосы получил бы свой штамп.
            self._flush_partial()
            try:
                self._sink.write(text)
            except Exception:
                pass
            self._console.write(text)
            return len(text)
        stamped = self._stamped(text)
        if stamped:
            try:
                self._sink.write(stamped)
            except Exception:                   # лог не имеет права ронять прогон
                pass
            self._console.write(stamped)
        return len(text)

    def _flush_partial(self) -> None:
        """Отдать последнюю неполную строку (без перевода строки) — при закрытии лога."""
        rest = self._partial
        if not rest:
            return
        object.__setattr__(self, "_partial", "")
        if not rest.strip():
            return
        line = _stamp() + rest
        try:
            self._sink.write(line)
        except Exception:
            pass
        self._console.write(line)

    def writelines(self, lines: Any) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._flush_partial()
        try:
            self._sink.flush()
        except Exception:
            pass
        self._console.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_console"), name)


class RunLog:
    """Перехват вывода прогона в `<папка кадра>/run.log`; `enabled=False` — ничего не делает.

    Использование (`cmd/run.py`)::

        with RunLog(cfg.logs) as log:                 # tee включён до первой печати
            print(_profile_line(cfg), flush=True)
            result = pipeline.run(cfg, log=log)       # pipeline вызовет `log.attach(...)`

    `attach` вызывается **из pipeline** сразу после `_plan`: только там известно реальное имя
    кадра (`out_path` с `-s{seed}` и `-x{factor}`).
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.directory: Path | None = None
        self.path: Path | None = None
        self._sink = _Sink()
        self._saved: tuple[Any, Any] | None = None

    # -- жизненный цикл ---------------------------------------------------- #

    def __enter__(self) -> "RunLog":
        if self.enabled:
            # Сохраняем и stdout, и stderr: traceback тоже должен попасть в лог.
            self._saved = (sys.stdout, sys.stderr)
            sys.stdout = _Tee(self._saved[0], self._sink)      # type: ignore[assignment]
            sys.stderr = _Tee(self._saved[1], self._sink)      # type: ignore[assignment]
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        # Падение тоже должно остаться в логе: трассировка печатается уже после восстановления
        # потоков, поэтому пишем её в файл здесь, пока лог ещё открыт.
        if exc is not None and self._sink.file is not None:
            try:
                self._sink.write("".join(traceback.format_exception(exc_type, exc, tb)))
            except Exception:
                pass
        self.close()
        return False

    def close(self) -> None:
        """Вернуть исходные потоки и закрыть файл (идемпотентно)."""
        if self._saved is not None:
            sys.stdout, sys.stderr = self._saved
            self._saved = None
        try:
            self._sink.close()
        except Exception:
            pass

    # -- открытие файла ---------------------------------------------------- #

    def attach(self, directory: str | Path, *, cfg: Any = None) -> Path | None:
        """Создать папку лога и открыть `<directory>/run.log`; вернуть путь к логу.

        Вызывается один раз, сразу после `_plan`. Повторный вызов — no-op (возвращает уже
        открытый путь): батч может захотеть переоткрыть лог, но второго файла не появится.
        """
        if not self.enabled:
            return None
        if self._sink.file is not None:
            return self.path
        target = Path(directory)
        try:
            target.mkdir(parents=True, exist_ok=True)
            handle = open(target / LOG_NAME, "w", encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"[log] не удалось создать лог в {target}: {exc} — вывод только в консоль",
                  flush=True)
            self.enabled = False
            return None
        self.directory = target
        self.path = target / LOG_NAME
        self._sink.attach(handle)
        if cfg is not None:
            _write_meta(target, cfg)
        print(f"[log] {str(self.path).replace(chr(92), '/')}", flush=True)
        return self.path


def _write_meta(directory: Path, cfg: Any) -> None:
    """`meta.json` рядом с логом: чем именно сделан кадр (без чтения консоли).

    Пишем параметры прогона (`RunConfig.to_mapping()`), время и argv. Ошибка записи не важна:
    метаданные — удобство, кадр и лог от них не зависят.
    """
    data: dict[str, Any] = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "argv": list(sys.argv),
    }
    to_mapping = getattr(cfg, "to_mapping", None)
    if callable(to_mapping):
        try:
            data["params"] = to_mapping()
        except Exception:
            pass
    try:
        (directory / META_NAME).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        pass


def batch_log_dir(out_dir: str | Path) -> Path:
    """Папка лога сессии: `<out-dir>/_logs/<YYYYmmdd-HHMMSS>` — одна на прогон, не на seed.

    Запасной путь для `run_batch`: `gen` задаёт каталог явно (`cfg.log_dir`), а этот вариант
    остаётся для вызова без `log_dir` (тесты, скрипты, `run --set batch.*`).
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return Path(str(out_dir).replace("\\", "/")) / BATCH_LOGS_DIR / stamp


__all__ = ["BATCH_LOGS_DIR", "LOG_NAME", "META_NAME", "RunLog", "batch_log_dir"]
