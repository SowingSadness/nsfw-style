"""Отладочный лог и meta прогона: `<кадр>.log` и `<кадр>-meta.json` **рядом с кадром**.

Зачем: stdout прогона уже содержит всю диагностику — `[profile]`, `[plan]` (сетка, ключ кэша,
попадание/промах), `[1/5]`…`[5/5]` от бекенда, время фаз, `[post]`, `[done]`. Но живёт он только
в консоли: при ручном редиректе (`> run.log`) теряется привязка к кадру, а прогоны разных
seed'ов/резолюций пишут в один и тот же файл и затирают друг друга. Поэтому лог кладётся
**рядом с кадром и называется его именем**:

    out/84-25.png             <- кадр (seed 84, 25 шагов)
    out/84-25.log             <- весь вывод этого кадра
    out/84-25-meta.json       <- параметры и тайминги кадра

Правила:

* **Имя** = путь кадра с расширением `.log` / `-meta.json` (`runlog.log_path`/`meta_path`).
  В имени кадра уже есть seed и число шагов (`84-25`), поэтому две генерации не смешиваются.
* **Метки времени**: каждая непустая строка лога (и в консоли, и в файле) получает
  префикс `[HH:MM:SS] ` — по нему видно, сколько шла каждая фаза кадра.
* **Tee, а не редирект**: stdout и stderr одновременно идут в консоль и в файл — прогон остаётся
  интерактивным (tqdm-прогресс виден), а traceback падения попадает в лог.
* **Лог открывается позже, чем начинается печать**: имя кадра известно только после `_plan`
  (там же решается `-x{factor}`), поэтому ранние строки копятся в заголовке и дописываются в файл
  в момент `attach`/`switch`. Порядок строк в логе совпадает с консолью.
* **Сессия (`gen`)**: лог **свой у каждого кадра**, но общий заголовок (план, prepare/encode)
  переписывается в каждый новый `.log` (`switch`), а повторное открытие того же кадра
  (апскейл после освобождения модели) — дозапись без дублирования заголовка.
* **`--no-logs`** (`RunConfig.logs=False`) выключает `.log`, но `<кадр>-meta.json` пишется всегда
  (`write_meta` вызывается командами, а не `RunLog`).
* **Лог — наблюдение, а не результат**: любая ошибка файловой системы не роняет прогон,
  печатается одна строка `[log] …`, и тест/генерация продолжаются.

Модуль без тяжёлых импортов: только stdlib, чтобы `--help` и `dry-run` оставались мгновенными.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, TextIO

#: Суффикс имени лога рядом с кадром (`84-25.png` -> `84-25.log`).
LOG_SUFFIX = ".log"
#: Суффикс meta рядом с кадром (`84-25.png` -> `84-25-meta.json`).
META_SUFFIX = "-meta.json"


def log_path(frame: str | Path) -> Path:
    """`out/84-25.png` -> `out/84-25.log` (лог лежит рядом с кадром, без подпапки)."""
    target = Path(frame)
    return target.with_suffix(LOG_SUFFIX)


def meta_path(frame: str | Path) -> Path:
    """`out/84-25.png` -> `out/84-25-meta.json` (meta лежит рядом с кадром)."""
    target = Path(frame)
    return target.with_name(f"{target.stem}{META_SUFFIX}")


class _Sink:
    """Приёмник строк лога: заголовок в памяти, текущий файл — после `switch`.

    Буфер нужен потому, что первые строки (`[profile]`, `[plan]`) печатаются до того, как
    известно имя кадра. Строки не теряются и не переставляются: `switch` списывает заголовок
    в файл и дальше пишет напрямую.

    В батче (`gen`) кадров несколько, поэтому заголовок **переписывается** в каждый новый файл:
    каждый `<seed>-<steps>.log` самодостаточен (в нём есть план/энкод/тайминги подготовки),
    а посевные строки других seed'ов в него не попадают.
    """

    def __init__(self) -> None:
        self.file: TextIO | None = None
        self.path: Path | None = None
        self.header: list[str] = []
        self.seen: set[Path] = set()

    def write(self, text: str) -> None:
        if self.file is not None:
            self.file.write(text)
            self.file.flush()
        else:
            self.header.append(text)

    def flush(self) -> None:
        if self.file is not None:
            self.file.flush()

    def switch(self, path: Path) -> None:
        """Открыть `path` и писать дальше туда; новый файл получает накопленный заголовок."""
        if self.path == path and self.file is not None:
            return
        self.close()
        path.parent.mkdir(parents=True, exist_ok=True)
        append = path in self.seen
        handle = open(path, "a" if append else "w", encoding="utf-8", errors="replace")
        self.seen.add(path)
        self.file = handle
        self.path = path
        if not append and self.header:
            handle.write("".join(self.header))
        handle.flush()

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
    """Перехват вывода прогона в `<кадр>.log`; `enabled=False` — ничего не делает.

    Использование (`cmd/run.py`)::

        with RunLog(cfg.logs) as log:                 # tee включён до первой печати
            print(_profile_line(cfg), flush=True)
            result = pipeline.run(cfg, log=log)       # pipeline вызовет `log.attach(...)`

    `attach` вызывается **из pipeline** сразу после `_plan`: только там известно реальное имя
    кадра (`out_path` с `<seed>-<steps>` и `-x{factor}`). В батче `run_batch` вызывает `attach`
    перед каждым кадром — файл у каждого seed'а свой.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
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

    def attach(self, frame: str | Path) -> Path | None:
        """Открыть `<кадр>.log` рядом с кадром; вернуть путь к логу.

        Вызывается сразу после `_plan` (когда имя кадра известно). Повторный вызов с тем же
        кадром — no-op; с другим кадром (`run_batch`) — новый файл с копией заголовка.
        """
        if not self.enabled:
            return None
        target = log_path(frame)
        repeat = target == self.path
        try:
            self._sink.switch(target)
        except OSError as exc:
            print(f"[log] не удалось создать лог {target}: {exc} — вывод только в консоль",
                  flush=True)
            self.enabled = False
            return None
        self.path = target
        if not repeat:
            print(f"[log] {str(target).replace(chr(92), '/')}", flush=True)
        return self.path


def write_meta(frame: str | Path, cfg: Any, extra: dict[str, Any] | None = None) -> Path | None:
    """`<кадр>-meta.json` рядом с кадром: чем и когда сделан кадр (без чтения консоли).

    Пишется **всегда**, независимо от `logs`/`--no-logs` (это обещает help): только сам `.log`
    выключается флагом. Ошибка записи не важна: метаданные — удобство, кадр и лог от них
    не зависят. Запись атомарная (`*.tmp` + `os.replace`).
    """
    target = meta_path(frame)
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
    if extra:
        data.update(extra)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)
    except OSError as exc:
        print(f"[meta] не удалось записать {target.as_posix()}: {exc}", flush=True)
        return None
    return target


__all__ = ["LOG_SUFFIX", "META_SUFFIX", "RunLog", "log_path", "meta_path", "write_meta"]
