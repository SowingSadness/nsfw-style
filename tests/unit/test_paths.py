"""Канонические пути `--set` (`qwenstyle/paths.py`) — единственное написание, без совместимости.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/unit/test_paths.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем: пользователь пишет только пути параметров кубиков, а прогон пока читает `RunConfig`.
Ошибка в таблице трансляции — это либо молча потерянный `--set` (значение не доехало до модели),
либо ключ, который «работает» в документации и падает в CLI. Поэтому проверяется:

1. **каждый** канонический путь — реальный параметр реального кубика (нет висячих целей);
2. таблица `CUBE_TO_CONFIG` не содержит двух путей на одно поле конфига (иначе `--set` неоднозначен);
3. `parse_sets` переводит канонические пути в поля `RunConfig` и отвергает старые написания
   (`generation.steps`, `generation.sampler_kwargs.*`, `encoder.device`, `upscale.tile`, `size_px`)
   со списком допустимых;
4. `RunConfig.merged` реально принимает переведённые поля (трансляция не врёт);
5. `config show --explain-params` печатает канонические пути, а не поля конфига.
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

import qwenstyle.cubes                                              # noqa: E402,F401 — реестр
from qwenstyle.cmd import parse_sets                                # noqa: E402
from qwenstyle.cmd import config as config_cmd                      # noqa: E402
from qwenstyle.config import ConfigError, RunConfig                 # noqa: E402
from qwenstyle.cube import get_cube                                 # noqa: E402
from qwenstyle.paths import CUBE_TO_CONFIG, PATHS, config_path, validate  # noqa: E402

#: Старые написания, которые обязаны отвергаться.
OLD = ("generation.steps", "generation.guidance", "generation.offload",
       "generation.kv_cache", "generation.precision_cache", "generation.lora.scale",
       "generation.sampler_kwargs.offload_blocks", "generation.sampler_kwargs.ref_px",
       "generation.sampler_kwargs.vae_tile_px", "generation.sampler_kwargs.window_attn_px",
       "encoder.device", "encoder.name", "encoder.stream", "encoder.pin_memory",
       "upscale.factor", "upscale.target_height", "upscale.tile", "upscale.color_fix",
       "size_px")


def case_paths_are_real_params() -> None:
    known = {f"{cube.id}.{param.name}" for cube in __import__("qwenstyle.cube", fromlist=["all_cubes"]).all_cubes()
             for param in cube.params}
    missing = sorted(set(PATHS) - known)
    assert not missing, f"в таблице есть пути, которых нет у кубиков: {missing}"
    assert len(PATHS) >= 35, len(PATHS)


def case_no_two_paths_per_config_field() -> None:
    seen: dict[str, str] = {}
    for cube_path, config_path_ in CUBE_TO_CONFIG.items():
        assert config_path_ not in seen, (cube_path, config_path_, seen[config_path_])
        seen[config_path_] = cube_path


def case_validate_and_config_path() -> None:
    assert validate("transformer.qwen_image21.steps") == "transformer.qwen_image21.steps"
    assert config_path("transformer.qwen_image21.steps") == "generation.steps"
    assert config_path("encoder.qwen3vl.device") == "encoder.device"
    assert config_path("source.frame.height") == "size_px"
    try:
        validate("generation.steps")
    except ValueError as exc:
        assert "допустимые пути" in str(exc), exc
        assert "transformer.qwen_image21.steps" in str(exc), exc
    else:                                                       # pragma: no cover
        raise AssertionError("старый путь обязан отвергаться")


def case_parse_sets_translates() -> None:
    parsed = parse_sets(["transformer.qwen_image21.steps=8", "encoder.qwen3vl.device=cpu",
                         "strategy.kv_offload.mode=offload"])
    assert parsed == {"generation.steps": "8", "encoder.device": "cpu",
                      "generation.kv_cache": "offload"}, parsed
    for old in OLD:
        try:
            parse_sets([f"{old}=1"])
        except ConfigError as exc:
            assert "допустимые пути" in str(exc), (old, exc)
        else:                                                   # pragma: no cover
            raise AssertionError(f"--set {old}=1 обязан отвергаться")
    try:
        parse_sets(["без равно"])
    except ConfigError as exc:
        assert "KEY=VALUE" in str(exc), exc
    else:                                                       # pragma: no cover
        raise AssertionError("--set без `=` обязан отвергаться")


def case_translated_fields_apply() -> None:
    """Переведённые поля реально принимает `RunConfig.merged` (а не только «есть в таблице»)."""
    parsed = parse_sets(["transformer.qwen_image21.steps=8",
                         "transformer.qwen_image21.offload=group",
                         "strategy.kv_offload.mode=offload",
                         "decoder.vae.qwen21.tile_px=1024",
                         "post.upscale.esrgan.target_height=1440",
                         "source.frame.height=1088",
                         "source.frame.ref_crop_side=960"])
    cfg = RunConfig.from_profile("VRAM16-preview", **parsed)
    assert cfg.steps == 8, cfg.steps
    assert cfg.generation.offload == "group", cfg.generation.offload
    assert cfg.generation.kv_cache == "offload", cfg.generation.kv_cache
    assert cfg.generation.sampler_kwargs["vae_tile_px"] == 1024, cfg.generation.sampler_kwargs
    assert cfg.upscale.target_height == 1440, cfg.upscale.target_height
    assert cfg.size_px == 1088, cfg.size_px
    assert cfg.generation.sampler_kwargs["ref_px"] == 960, cfg.generation.sampler_kwargs


def case_explain_params_canonical() -> None:
    cfg = RunConfig.from_profile("base")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        config_cmd.explain_params(cfg)
    text = buffer.getvalue()
    assert "transformer.qwen_image21.steps" in text, text[:400]
    assert "post.upscale.esrgan.factor" in text, text[:400]
    assert "generation.sampler_kwargs." not in text.replace("(модель без кубиков", ""), \
        "поля конфига в таблице --set быть не должно"


CASES = [
    ("все пути — реальные параметры", case_paths_are_real_params),
    ("одно поле конфига — один путь", case_no_two_paths_per_config_field),
    ("validate/config_path", case_validate_and_config_path),
    ("parse_sets: перевод и отказ старым", case_parse_sets_translates),
    ("переведённые поля применяются", case_translated_fields_apply),
    ("--explain-params печатает канонические пути", case_explain_params_canonical),
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
