"""Пропуск избыточного апскейла: бекенд не вызывается, если апскейл ничего не делает.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_upscale_skip.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (план `qwen1440p-perf`, шаг 3): пресет с `upscale.target_height: 1440` (ныне — `strong-release`),
а `--size 1440p` уже даёт кадр 1920x1440 — код всё равно поднимал Real-ESRGAN и тратил 9.6 s
на no-op (`[post] esrgan x1: 1920x1440 -> 1920x1440 (9.6s)`), плюс дописывал `-x1` в имя файла.
Тест фиксирует матрицу «целевая высота / фактор -> бекенд вызван или нет» без GPU и без весов:
`pipeline.get_upscale` подменяется заглушкой, которая считает вызовы и фактор.

Кейсы:

1. `target_height == gen_h` -> бекенд не вызван (0 вызовов), печатается `[post] upscale пропущен`.
2. `target_height < gen_h` -> тоже пропущен.
3. `target_height > gen_h` -> бекенд вызван с фактором `target/gen_h` (точность 1e-6) и печатает
   обычную строку `[post] <backend> x<factor>: WxH -> WxH`.
4. `factor=1.0` без `target_height` -> пропущен с сообщением про `factor=1`.
5. `--upscale 0` (factor=0, без цели) -> пропущен, сообщение про выключение, а не про высоту.
6. Имя файла в случае 1 не содержит `-x1` (пропуск возвращает фактор 0.0), а при `factor > 1`
   суффикс `-x1.5` на месте — имена не поехали.
7. Дробный фактор (`target/gen_h` = 1.333) — не считается «единицей» (порог 1±1e-6).
"""
from __future__ import annotations

import contextlib
import io
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                   # тест не должен оставлять __pycache__ в пакете

from qwenstyle import pipeline                   # noqa: E402 — после правки sys.path
from qwenstyle.config import (                   # noqa: E402
    EncoderConfig,
    GenerationConfig,
    RunConfig,
    UpscaleConfig,
)
from qwenstyle.io import out_path                # noqa: E402

GEN_W, GEN_H = 1920, 1440


# --------------------------------------------------------------------------- #
# Фикстуры
# --------------------------------------------------------------------------- #

class FakeBackend:
    """Заглушка бекенда апскейла: считает вызовы и запоминает фактор (никаких весов)."""

    calls: list[float] = []

    def upscale(self, image: Any, factor: float = 4.0, cfg: Any = None, seed: int = 0,
                source: Any = None, **_kwargs: Any) -> Any:
        FakeBackend.calls.append(float(factor))
        size = (max(2, int(round(image.width * float(factor)))),
                max(2, int(round(image.height * float(factor)))))
        return image.resize(size)


def _factory() -> Callable[[str], type]:
    def get_upscale(name: str) -> type:
        return FakeBackend

    return get_upscale


def _cfg(*, factor: float = 0.0, target_height: int | None = None) -> RunConfig:
    """Конфиг без пресетов: только то, что решает судьбу апскейла.

    `out` — внутри `out/tests/`: результаты тестов не должны попадать в пользовательский `out/`.
    """
    return RunConfig(
        encoder=EncoderConfig(),
        generation=GenerationConfig(),
        upscale=UpscaleConfig(backend="esrgan", factor=factor, target_height=target_height),
        out="out/tests/x.png",
        seed=7,
    )


def _frame(width: int = GEN_W, height: int = GEN_H) -> Any:
    from PIL import Image

    return Image.new("RGB", (width, height), (10, 20, 30))


@contextlib.contextmanager
def _captured() -> Any:
    """Перехватить stdout прогона (`[post] …`) и вернуть список строк."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield buf


def _postprocess(cfg: RunConfig, image: Any, factor: float, seed: int = 7,
                 ) -> tuple[Any, float, str]:
    """Вызвать `_postprocess` с подменённым `get_upscale`; вернуть (кадр, фактор, лог)."""
    original = pipeline.get_upscale
    FakeBackend.calls = []
    pipeline.get_upscale = _factory()
    try:
        with _captured() as buf:
            result = pipeline._postprocess(cfg, image, factor, seed)
    finally:
        pipeline.get_upscale = original
    return result[0], result[1], buf.getvalue()


def _line_checks(log: str, *, needle: str, forbid: str | None = None) -> None:
    if needle not in log:
        raise AssertionError(f"в логе нет {needle!r}: {log!r}")
    if forbid is not None and forbid in log:
        raise AssertionError(f"в логе не должно быть {forbid!r}: {log!r}")


# --------------------------------------------------------------------------- #
# Кейсы
# --------------------------------------------------------------------------- #

def case_target_reached(tmp: Path) -> None:
    """(а) `target_height == gen_h` -> бекенд не вызван, печать «пропущен»."""
    cfg = _cfg(target_height=GEN_H)
    factor = float(cfg.upscale_factor(GEN_H))
    if abs(factor - 1.0) > 1e-9:
        raise AssertionError(f"предусловие: factor={factor}, ожидалось 1.0")
    image, used, log = _postprocess(cfg, _frame(), factor)
    if FakeBackend.calls:
        raise AssertionError(f"бекенд вызван {len(FakeBackend.calls)} раз(а): {FakeBackend.calls}")
    if used != 0.0:
        raise AssertionError(f"пропуск должен вернуть фактор 0.0, получено {used!r}")
    if image.size != (GEN_W, GEN_H):
        raise AssertionError(f"кадр изменился: {image.size}")
    _line_checks(log, needle="[post] upscale пропущен:", forbid="[post] esrgan")
    _line_checks(log, needle=f"кадр {GEN_W}x{GEN_H} уже не ниже целевой {GEN_H}")
    if log.count("[post]") != 1:
        raise AssertionError(f"ожидалась ровно одна строка [post]: {log!r}")
    print(f"  ok: 0 вызовов бекенда, лог {log.strip()!r}")


def case_target_below(tmp: Path) -> None:
    """(б) `target_height < gen_h` -> тоже пропущен (даунскейлом апскейлер не занимаем)."""
    cfg = _cfg(target_height=1080)
    image, used, log = _postprocess(cfg, _frame(), float(cfg.upscale_factor(GEN_H)))
    if FakeBackend.calls:
        raise AssertionError(f"бекенд вызван: {FakeBackend.calls}")
    if used != 0.0 or image.size != (GEN_W, GEN_H):
        raise AssertionError(f"ожидался пропуск без изменений, получено {used!r} {image.size}")
    _line_checks(log, needle="[post] upscale пропущен:", forbid="[post] esrgan")
    _line_checks(log, needle="уже не ниже целевой 1080")
    print(f"  ok: target 1080 < {GEN_H} -> пропуск")


def case_target_above(tmp: Path) -> None:
    """(в) `target_height > gen_h` -> бекенд вызван с фактором `target/gen_h` (1e-6)."""
    target = 2160
    cfg = _cfg(target_height=target)
    factor = float(cfg.upscale_factor(GEN_H))
    image, elapsed, log = _postprocess(cfg, _frame(), factor)
    if len(FakeBackend.calls) != 1:
        raise AssertionError(f"ожидался 1 вызов бекенда, получено {FakeBackend.calls}")
    if abs(FakeBackend.calls[0] - target / GEN_H) > 1e-6:
        raise AssertionError(f"фактор {FakeBackend.calls[0]!r} != {target / GEN_H!r}")
    if not isinstance(elapsed, float) or elapsed < 0:
        raise AssertionError(f"второе значение — время апскейла, получено {elapsed!r}")
    if image.size != (int(round(GEN_W * factor)), target):
        raise AssertionError(f"кадр {image.size}, ожидался {(int(round(GEN_W * factor)), target)}")
    _line_checks(log, needle="[post] esrgan x1.5:", forbid="пропущен")
    print(f"  ok: вызов с фактором {FakeBackend.calls[0]:.6f} ({target}/{GEN_H})")


def case_factor_one(tmp: Path) -> None:
    """(г) `factor=1.0` без `target_height` -> пропущен с сообщением про factor=1."""
    cfg = _cfg(factor=1.0)
    image, used, log = _postprocess(cfg, _frame(), float(cfg.upscale_factor(GEN_H)))
    if FakeBackend.calls:
        raise AssertionError(f"бекенд вызван: {FakeBackend.calls}")
    if used != 0.0 or image.size != (GEN_W, GEN_H):
        raise AssertionError(f"ожидался пропуск, получено {used!r} {image.size}")
    _line_checks(log, needle="[post] upscale пропущен:", forbid="[post] esrgan")
    _line_checks(log, needle="factor=1")
    print(f"  ok: factor=1 без цели -> пропуск ({log.strip()!r})")


def case_disabled(tmp: Path) -> None:
    """(д) `--upscale 0` -> пропуск; сообщение про выключение, а не про высоту."""
    cfg = _cfg()                                  # factor=0.0, target_height=None
    image, used, log = _postprocess(cfg, _frame(), float(cfg.upscale_factor(GEN_H)))
    if FakeBackend.calls:
        raise AssertionError(f"бекенд вызван: {FakeBackend.calls}")
    if used != 0.0 or image.size != (GEN_W, GEN_H):
        raise AssertionError(f"ожидался пропуск, получено {used!r} {image.size}")
    _line_checks(log, needle="[post] upscale пропущен:", forbid="[post] esrgan")
    _line_checks(log, needle="выключен", forbid="уже не ниже")
    print(f"  ok: выключено пользователем -> пропуск ({log.strip()!r})")


def case_name_without_x1(tmp: Path) -> None:
    """(е) пропуск не добавляет `-x1` в имя; при реальном апскейле суффикс на месте."""
    reached = _cfg(target_height=GEN_H)
    factor = pipeline.upscale_effective_factor(reached, (GEN_W, GEN_H))
    if factor != 0.0:
        raise AssertionError(f"effective factor для достигнутой цели = {factor!r}, ожидался 0.0")
    name = out_path(reached, factor)
    if "-x1" in name:
        raise AssertionError(f"в имени no-op апскейла появился суффикс: {name!r}")
    if not name.endswith("-s7.png"):
        raise AssertionError(f"имя потеряло seed-тег: {name!r}")

    above = _cfg(target_height=2160)
    big_factor = pipeline.upscale_effective_factor(above, (GEN_W, GEN_H))
    big_name = out_path(above, big_factor)
    if abs(big_factor - 2160 / GEN_H) > 1e-6 or "-x1.5" not in big_name:
        raise AssertionError(f"реальный апскейл потерял суффикс: {big_factor!r} {big_name!r}")

    disabled = _cfg()
    if pipeline.upscale_effective_factor(disabled, (GEN_W, GEN_H)) != 0.0:
        raise AssertionError("выключенный апскейл должен остаться фактором 0.0")
    if "-x" in out_path(disabled, 0.0):
        raise AssertionError("выключенный апскейл не должен добавлять -x в имя")
    print(f"  ok: {name} (без -x1) / {big_name}")


def case_fractional_factor(tmp: Path) -> None:
    """(7) 1440/1080 = 1.333 — не «единица»: бекенд вызывается."""
    cfg = _cfg(target_height=GEN_H)
    frame = _frame(1440, 1080)
    factor = float(cfg.upscale_factor(frame.height))
    if pipeline.upscale_skip_reason(cfg, frame.size, factor):
        raise AssertionError(f"дробный фактор {factor!r} принят за пропуск")
    image, elapsed, log = _postprocess(cfg, frame, factor)
    if len(FakeBackend.calls) != 1 or abs(FakeBackend.calls[0] - factor) > 1e-6:
        raise AssertionError(f"ожидался вызов с {factor!r}: {FakeBackend.calls}")
    if image.size != (GEN_W, GEN_H):
        raise AssertionError(f"кадр {image.size}, ожидался {(GEN_W, GEN_H)}")
    _line_checks(log, needle="[post] esrgan x1.33333:", forbid="пропущен")
    if elapsed < 0:
        raise AssertionError(f"время апскейла не может быть отрицательным: {elapsed!r}")
    print(f"  ok: 1080 -> 1440 вызвал бекенд с {FakeBackend.calls[0]:.6f}")


def case_dry_run_fields(tmp: Path) -> None:
    """Фактор/имя в плане: `_estimate_out` и `upscale_effective_factor` согласованы."""
    reached = _cfg(target_height=GEN_H)
    crop = (GEN_W, GEN_H)
    factor = pipeline.upscale_effective_factor(reached, crop)
    if pipeline._estimate_out(crop, factor, reached.upscale.target_height) != crop:
        raise AssertionError("_estimate_out должен вернуть размер генерации при пропуске")
    above = _cfg(target_height=2160)
    big = pipeline.upscale_effective_factor(above, crop)
    expected = (int(round(2160 * GEN_W / GEN_H)), 2160)        # 2880x2160 в пропорциях кадра
    if pipeline._estimate_out(crop, big, above.upscale.target_height) != expected:
        raise AssertionError(" _estimate_out для реального апскейла посчитан неверно")
    print("  ok: _estimate_out согласован с upscale_effective_factor")


def case_run_skip_line(tmp: Path) -> None:
    """Гейт `run()`: при `factor == 0` печатается `_print_upscale_skip`, реестр не доргается."""
    cfg = _cfg(target_height=GEN_H)
    original = pipeline.get_upscale

    def boom(name: str) -> type:
        raise AssertionError(f"get_upscale вызван в режиме пропуска: {name!r}")

    pipeline.get_upscale = boom
    try:
        with _captured() as buf:
            pipeline._print_upscale_skip(cfg, (GEN_W, GEN_H))
    finally:
        pipeline.get_upscale = original
    log = buf.getvalue()
    if log.count("\n") != 1:
        raise AssertionError(f"ожидалась ровно одна строка: {log!r}")
    _line_checks(log, needle="[post] upscale пропущен: кадр 1920x1440 уже не ниже целевой 1440")
    print("  ok: run-гейт печатает ту же строку без обращения к бекенду")


# --------------------------------------------------------------------------- #

CASES: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("1. target_height == gen_h -> бекенд не вызван", case_target_reached),
    ("2. target_height < gen_h -> пропуск", case_target_below),
    ("3. target_height > gen_h -> вызов с target/gen_h", case_target_above),
    ("4. factor=1.0 без цели -> пропуск", case_factor_one),
    ("5. upscale 0 -> пропуск «выключен»", case_disabled),
    ("6. имя файла без -x1", case_name_without_x1),
    ("7. дробный фактор 1.333 не пропускается", case_fractional_factor),
    ("8. _estimate_out и dry-run поля", case_dry_run_fields),
    ("9. run-гейт печатает «пропущен» без бекенда", case_run_skip_line),
)


def main() -> int:
    failed = 0
    for name, case in CASES:
        try:
            case(REPO_ROOT)
        except Exception as exc:                # noqa: BLE001 — тест-скрипт печатает и продолжает
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
