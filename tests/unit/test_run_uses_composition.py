"""Прогон проверяет композицию до загрузки весов (план `cube-architecture`, шаг 7a).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/unit/test_run_uses_composition.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем: компоновщик появился в шаге 1, но прогон о нём не знал — расхождение стадий или
отсутствующая стратегия обнаруживались после 6.7 ГиБ fp8-кэша и первого шага. Теперь `pipeline.run`
вызывает `verify_composition` **до** `prepare`, и тест фиксирует три вещи:

1. все публичные пресеты и H3 проходят проверку (H3 не выражен кубиками — это не ошибка);
2. блокирующие замечания компоновщика поднимают `ConfigError` (а не тонут в логе);
3. `cube_unsupported` — не блокирует: иначе MiniMax-H3 и FLUX.2 перестали бы запускаться.

Плюс `cubes_view` (печать композиции для `--cubes`): rc=0 у чистой композиции и текст стадий.
"""
from __future__ import annotations

import contextlib
import io
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from qwenstyle import composition, pipeline                            # noqa: E402
from qwenstyle.cmd import config as config_cmd                         # noqa: E402
from qwenstyle.compose import Issue                                    # noqa: E402
from qwenstyle.config import ConfigError, RunConfig, available_presets  # noqa: E402


def case_all_presets_pass() -> None:
    for preset in available_presets():
        cfg = RunConfig.from_profile(preset)
        pipeline.verify_composition(cfg)                # не должно бросить ни на одном пресете


def case_blocking_issue_raises() -> None:
    """Блокирующее замечание (например не сходятся стадии) обязано остановить прогон."""
    cfg = RunConfig.from_profile("VRAM16-preview")
    broken = composition.check(cfg)
    from dataclasses import replace

    patched = replace(broken, issues=(Issue("port_missing", "нет входных портов ['pixels']",
                                            "post.colorfix"),))
    original = composition.check
    composition.check = lambda *_a, **_k: patched          # type: ignore[assignment]
    try:
        pipeline.verify_composition(cfg)
    except ConfigError as exc:
        assert "post.colorfix" in str(exc), exc
        assert "не сходится" in str(exc), exc
    else:                                                   # pragma: no cover
        raise AssertionError("блокирующее замечание обязано поднимать ConfigError")
    finally:
        composition.check = original                        # type: ignore[assignment]


def case_cube_unsupported_does_not_raise() -> None:
    """H3 не выражен кубиками — это не повод отказывать в прогоне."""
    cfg = RunConfig.from_profile("h3-image")
    assert composition.unsupported_reason(cfg) is not None, "H3 пока без кубиков"
    pipeline.verify_composition(cfg)                        # молча проходит


def case_cubes_view_prints_tree() -> None:
    cfg = RunConfig.from_profile("VRAM16-4k-slide")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = config_cmd.cubes_view(cfg)
    text = out.getvalue()
    assert code == 0, text
    assert "composition=VRAM16-4k-slide (ok)" in text, text[:200]
    for stage in ("source.frame", "source.txtFileImage", "encoder.qwen3vl",
                  "strategy.precision.fp8",
                  "strategy.kv_offload", "strategy.window_attention",
                  "transformer.qwen_image21", "decoder.vae.qwen21",
                  "decoder.postprocess.image"):
        assert stage in text, f"{stage} нет в выводе --cubes"
    # H3: rc=1 и одна ошибка (без каскада port_missing)
    h3 = RunConfig.from_profile("h3-image")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = config_cmd.cubes_view(h3)
    assert code == 1, out.getvalue()
    assert "cube_unsupported" in out.getvalue(), out.getvalue()


CASES = [
    ("все пресеты проходят проверку композиции", case_all_presets_pass),
    ("блокирующее замечание -> ConfigError", case_blocking_issue_raises),
    ("cube_unsupported не блокирует прогон", case_cube_unsupported_does_not_raise),
    ("cubes_view печатает дерево стадий", case_cubes_view_prints_tree),
]


def main() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            fn()
        except Exception as exc:                                    # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    if failed:
        print(f"\n{failed} из {len(CASES)} кейсов провалено")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
