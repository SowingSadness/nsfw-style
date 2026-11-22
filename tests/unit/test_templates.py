"""Шаблоны `qwenstyle/templates/<модель>/`: без наследования, значения 1:1 с пресетами.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/unit/test_templates.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.
Без весов, без CUDA, без сети.

Зачем (план `workflow-layout`, шаг 4): шаблоны — не «пресеты с extends», а самодостаточные файлы,
которые копируют и правят. Тест заменяет наследование проверками:

1. ни в одном шаблоне нет `extends`/`template`/`preset` и устаревших секций `RunConfig`;
2. каждый шаблон проходит `compose` (порты, стадии, стратегии) и грузится `Workflow`;
3. значения шаблона **равны** пресету с тем же именем: обратная проекция `config_mapping()` ->
   `RunConfig` сравнивается поле за полем со снимком `RunConfig.from_profile(preset)`; у пресета
   с выключенным апскейлом кубика апскейла в композиции нет — там сверяется только «выключен»;
4. инварианты тарифа держатся на уровне кубиков: VRAM8 -> энкодер на CPU + `offload=group`;
   VRAM16-* -> `kv_offload=offload`; все -> `strategy.precision.fp8`; `4k-slide`/`large` ->
   окно внимания > 0 и `ref_crop_side > 0`;
5. имена публичных шаблонов остаются в тарифной семье (`VRAM8/12/16-*`), `base`/`turbo` — dev.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from qwenstyle.config import RunConfig, available_presets              # noqa: E402
from qwenstyle.workflow import load                                   # noqa: E402

TEMPLATES = REPO_ROOT / "qwenstyle" / "templates" / "qwen-image-2.1"
#: Пресеты без кубиков (H3): шаблонов у них нет до появления кубиков (долг D13).
LEGACY_PRESETS = ("h3-image", "max-h3")
#: Поля запуска, которые в шаблоне и в пресете расходятся по определению.
RUN_FIELDS = ("preset", "out", "prompt_file", "inputs", "input", "seed", "batch",
              "cache_dir", "cache_supersede", "logs", "preview", "resolution")


def template_files() -> list[Path]:
    return sorted(TEMPLATES.glob("*.yaml"))


def params_of(workflow, cube_id: str) -> dict:
    for binding in workflow.pipeline.cubes:
        if binding.cube_id == cube_id:
            return dict(binding.params)
    return {}


def ids_of(workflow) -> list[str]:
    return [binding.cube_id for binding in workflow.pipeline.cubes]


def case_no_inheritance() -> None:
    files = template_files()
    assert files, f"нет шаблонов в {TEMPLATES.as_posix()}"
    for path in files:
        body = path.read_text(encoding="utf-8")
        for forbidden in ("extends:", "template:", "preset:", "generation:", "encoder:",
                          "upscale:"):
            assert forbidden not in body, f"{path.name}: в шаблоне не должно быть {forbidden!r}"


def case_all_templates_compose() -> None:
    for path in template_files():
        workflow = load(path)
        composition = workflow.composition()
        assert composition.ok, f"{path.name}: {composition.format_issues()}"
        assert composition.ports_out == frozenset({"pixels"}), (path.name, composition.ports_out)


def case_values_match_presets() -> None:
    """Значения шаблона == значения пресета (кроме полей запуска)."""
    checked = 0
    for path in template_files():
        preset = path.stem
        if preset not in available_presets():
            continue
        workflow = load(path)
        from_preset = RunConfig.from_profile(preset).to_mapping()
        from_template = RunConfig.from_mapping(workflow.config_mapping()).to_mapping()
        for section in ("encoder", "generation"):
            left = from_preset.get(section) or {}
            right = from_template.get(section) or {}
            assert left == right, (preset, section, left, right)
        # апскейл: если он в пресете выключен (factor 0 и нет target_height), кубика апскейла в
        # композиции нет вовсе — сравниваем только сам факт «выключен», а не его настройки
        preset_up = from_preset.get("upscale") or {}
        template_up = from_template.get("upscale")
        if (preset_up.get("factor") or preset_up.get("target_height")):
            assert template_up == preset_up, (preset, preset_up, template_up)
        else:
            assert not template_up or not (template_up.get("factor")
                                           or template_up.get("target_height")),                 (preset, template_up)
        # геометрия: если пресет не задаёт `size_px`, канвас берётся из `resolution` профиля —
        # шаблон обязан быть конкретным, поэтому height записан в нём явно; сверяем только когда
        # пресет задаёт размер сам
        if from_preset.get("size_px"):
            assert from_preset.get("size_px") == from_template.get("size_px"), \
                (preset, from_preset.get("size_px"), from_template.get("size_px"))
        assert from_preset.get("aspect") == from_template.get("aspect"), \
            (preset, from_preset.get("aspect"), from_template.get("aspect"))
        checked += 1
    assert checked >= 17, f"сверено шаблонов: {checked}"


def case_tariff_invariants() -> None:
    for path in template_files():
        workflow = load(path)
        ids = ids_of(workflow)
        assert "strategy.precision.fp8" in ids, f"{path.name}: нет fp8-стратегии"
        name = path.stem
        if name.startswith("VRAM8"):
            assert params_of(workflow, "encoder.qwen3vl")["device"] == "cpu", name
            assert params_of(workflow, "transformer.qwen_image21")["offload"] == "group", name
        if name.startswith("VRAM16"):
            assert params_of(workflow, "strategy.kv_offload")["mode"] == "offload", name
        if name in ("VRAM16-4k-slide", "VRAM16-large"):
            window = params_of(workflow, "strategy.window_attention")
            assert window and window["window_attn_px"] > 0, (name, window)
            assert params_of(workflow, "source.frame")["ref_crop_side"] > 0, name


def case_public_names() -> None:
    names = {path.stem for path in template_files()}
    for preset in available_presets():
        if preset in LEGACY_PRESETS:
            assert preset not in names, f"{preset}: H3 живёт пресетом до появления кубиков"
            continue
        assert preset in names, f"{preset}: нет шаблона в {TEMPLATES.as_posix()}"
    public = {name for name in names if name.startswith(("VRAM8-", "VRAM12-", "VRAM16-"))}
    assert len(public) == 17, sorted(public)


def case_workflow_new_uses_template() -> None:
    """`workflow new` берёт именно шаблон (а не пресет), когда шаблон уже есть."""
    from qwenstyle.cmd.workflow import new_file

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        code = new_file(source="VRAM16-preview", name="copied", out_dir=str(Path(tmp) / "w"),
                        force=False)
        assert code == 0, code
        path = Path(tmp) / "w" / "workflow.yaml"
        body = path.read_text(encoding="utf-8")
        assert "из пресета" not in body, "должен использоваться шаблон, а не пресет"
        workflow = load(path)
        assert workflow.composition().ok, workflow.composition().format_issues()
        assert params_of(workflow, "transformer.qwen_image21")["offload"] == "none", \
            "VRAM16-preview: offload=none"


CASES = [
    ("шаблоны без наследования", case_no_inheritance),
    ("все шаблоны собираются", case_all_templates_compose),
    ("значения == пресеты", case_values_match_presets),
    ("инварианты тарифа", case_tariff_invariants),
    ("публичные имена", case_public_names),
    ("workflow new берёт шаблон", case_workflow_new_uses_template),
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
