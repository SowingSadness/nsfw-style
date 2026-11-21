"""Проекция пресета в композицию кубиков: `RunConfig -> PipelineSpec` (без весов и CUDA).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/unit/test_composition_from_config.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (план `cube-architecture`, шаг 6, приёмка): до этого шага пресет и композиция жили
в разных мирах, и никто не проверял, что все 17 публичных пресетов вообще выражаются кубиками.
Тест фиксирует:

1. каждый публичный пресет (`VRAM8/12/16-*`) собирается композитором **без ошибок**;
2. H3 (`h3-image`, `max-h3`) даёт ровно одну ошибку `cube_unsupported` — без каскада
   `port_missing`, иначе UI показывал бы три бессмысленные строки;
3. проекция не теряет параметры: `offload`, `kv_cache`, окно внимания, `tile_px`, апскейл
   и цветокоррекция переезжают в кубики;
4. `precision=bf16` убирает fp8-стратегию (и бюджет VRAM растёт);
5. референсы на входе (`--input`) дают порты `prompt+refs`, без них — только `prompt`;
6. разрез декодера (шаг 7b): за `decoder.vae.qwen21` (`latents -> frames`) всегда идёт
   `decoder.postprocess.image` (`frames -> pixels`), иначе композиция не сходится.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from qwenstyle.composition import check, spec_of, unsupported_reason   # noqa: E402
from qwenstyle.config import RunConfig, available_presets               # noqa: E402

PUBLIC = tuple(p for p in available_presets()
               if p.startswith(("VRAM8-", "VRAM12-", "VRAM16-")))
H3_PRESETS = ("h3-image", "max-h3")


def params_of(spec, cube_id: str) -> dict:
    for binding in spec.cubes:
        if binding.cube_id == cube_id:
            return dict(binding.params)
    return {}


def ids(spec) -> list[str]:
    return [binding.cube_id for binding in spec.cubes]


def case_all_public_presets_compose() -> None:
    assert len(PUBLIC) == 17, f"публичных пресетов должно быть 17, найдено {len(PUBLIC)}: {PUBLIC}"
    for preset in PUBLIC:
        cfg = RunConfig.from_profile(preset)
        composition = check(cfg)
        assert composition.ok, f"{preset}: {composition.format_issues()}"
        assert composition.ports_out == frozenset({"pixels"}), (preset, composition.ports_out)
        assert "source.project" in ids(spec_of(cfg)), preset


def case_h3_is_unsupported_with_one_issue() -> None:
    for preset in H3_PRESETS:
        cfg = RunConfig.from_profile(preset)
        assert unsupported_reason(cfg) is not None, preset
        composition = check(cfg)
        assert not composition.ok, preset
        assert [i.code for i in composition.issues] == ["cube_unsupported"], \
            (preset, [i.code for i in composition.issues])
        assert "minimax_h3_image" in composition.issues[0].message


def case_slide_preset_params() -> None:
    cfg = RunConfig.from_profile("VRAM16-4k-slide")
    spec = spec_of(cfg)
    assert "strategy.window_attention" in ids(spec), ids(spec)
    assert params_of(spec, "strategy.window_attention") == {"window_attn_px": 1088,
                                                            "window_attn_margin_px": 448}
    assert params_of(spec, "strategy.kv_offload")["mode"] == "offload"
    transformer = params_of(spec, "transformer.qwen_image21")
    assert transformer["offload"] == "group", transformer
    assert transformer["steps"] == 25, transformer
    assert params_of(spec, "decoder.vae.qwen21")["tile_px"] == 512
    # декодер отдаёт тензор, в картинку его превращает постпроцесс — цепочка обязана это показать
    ids_in_order = ids(spec)
    decoder_at = ids_in_order.index("decoder.vae.qwen21")
    assert ids_in_order[decoder_at + 1] == "decoder.postprocess.image", ids_in_order
    assert params_of(spec, "decoder.postprocess.image")["output_type"] == "pil"
    # `upscale: null` — пост-апскейла в композиции нет, цветокоррекция есть
    assert "post.upscale.esrgan" not in ids(spec), ids(spec)
    assert params_of(spec, "post.colorfix")["mode"] == "wavelet"


def case_4k_fast_upscale() -> None:
    cfg = RunConfig.from_profile("VRAM16-4k-fast")
    spec = spec_of(cfg)
    assert params_of(spec, "post.upscale.esrgan")["target_height"] == 2160, params_of(
        spec, "post.upscale.esrgan")
    assert "strategy.window_attention" not in ids(spec), "у 4k-fast окна внимания нет"


def case_precision_toggle() -> None:
    fp8 = RunConfig.from_profile("VRAM16-preview")
    assert "strategy.precision.fp8" in ids(spec_of(fp8))
    bf16 = RunConfig.from_profile("VRAM16-preview", generation={"precision": "bf16"})
    spec = spec_of(bf16)
    assert "strategy.precision.fp8" not in ids(spec), ids(spec)
    assert check(bf16).budget_gib > check(fp8).budget_gib, "bf16 обязан быть дороже по VRAM"


def case_refs_ports() -> None:
    t2i = spec_of(RunConfig.from_profile("VRAM16-preview"))
    assert t2i.inputs == frozenset({"prompt"}), t2i.inputs
    edit = spec_of(RunConfig.from_profile("VRAM16-preview", input="out/tests/in.png"))
    assert edit.inputs == frozenset({"prompt", "refs"}), edit.inputs
    assert params_of(edit, "source.project")["refs"] == "out/tests/in.png"


def case_encoder_device_and_stream() -> None:
    vram8 = spec_of(RunConfig.from_profile("VRAM8-preview"))
    encoder = params_of(vram8, "encoder.qwen3vl")
    assert encoder["device"] == "cpu", encoder
    assert encoder["cache_tag"] == "heretic", encoder
    assert params_of(vram8, "transformer.qwen_image21")["offload"] == "group", "VRAM8 = group"


CASES = [
    ("17 публичных пресетов собираются", case_all_public_presets_compose),
    ("H3 -> одна ошибка cube_unsupported", case_h3_is_unsupported_with_one_issue),
    ("4k-slide: окно/kv/offload/пост", case_slide_preset_params),
    ("4k-fast: апскейл до 2160", case_4k_fast_upscale),
    ("precision=bf16 убирает fp8-стратегию", case_precision_toggle),
    ("референсы дают порт refs", case_refs_ports),
    ("VRAM8: энкодер на CPU, offload group", case_encoder_device_and_stream),
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
