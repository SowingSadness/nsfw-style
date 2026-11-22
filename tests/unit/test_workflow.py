"""`Workflow` + `RunSpec`: файл пайплайна без наследования и маршрутизация `--set`.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/unit/test_workflow.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.
Без torch, без CUDA, без сети.

Зачем (план `workflow-layout`, шаг 2): файл воркфлоу — единственный источник пайплайна, и он
намеренно простой: **никакого наследования**. Тест фиксирует ровно это и то, что вокруг него легко
сломать:

1. `version` обязателен и сверяется; ключи `template`/`extends`/`preset` отвергаются с подсказкой
   «шаблоны копируют», устаревшие секции `RunConfig` — с подсказкой про кубики;
2. `cubes` непустой, дубликат `id` — ошибка, неизвестный `id` — ошибка со списком известных;
3. `run`: `seeds` и `count` взаимоисключающие, политика кэша из домена, `seeds` строкой/списком;
4. `--set`: единственное написание — путь параметра кубика (`encoder.qwen3vl.device`);
   старые (`generation.steps`, `generation.sampler_kwargs.*`, `upscale.tile`) отвергаются со
   списком допустимых;
5. `inputs` — только то, что никто не производит (полный воркфлоу = пусто);
6. обратная проекция `config_mapping()` восстанавливает `RunConfig` (переходный режим);
   внешними входами могут быть только `prompt`/`refs` — латент и канвас производит композиция;
7. `load_template` находит шаблон по имени и просит модель при неоднозначности.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

import yaml                                                             # noqa: E402

from qwenstyle.config import RunConfig                                  # noqa: E402
from qwenstyle.runspec import RunSpec                                   # noqa: E402
from qwenstyle.workflow import Workflow, load, load_template, parse      # noqa: E402

CUBES = [
    {"id": "source.frame", "params": {"width": 2912, "height": 2176, "ref_crop_side": 768}},
    {"id": "source.txtFileImage", "params": {"prompt": "prompt-tokyo.txt",
                                            "Image1": "refs-style/nita-style.jpg"}},
    {"id": "encoder.qwen3vl"},
    {"id": "strategy.precision.fp8"},
    {"id": "strategy.kv_offload", "params": {"mode": "offload"}},
    {"id": "strategy.window_attention", "params": {"window_attn_px": 1088,
                                                   "window_attn_margin_px": 448}},
    {"id": "transformer.qwen_image21", "params": {"steps": 25, "offload": "group"}},
    {"id": "decoder.vae.qwen21", "params": {"tile_px": 512}},
    {"id": "decoder.postprocess.image"},
    {"id": "post.colorfix", "params": {"mode": "wavelet"}},
]


def params_of(workflow: Workflow, cube_id: str) -> dict:
    for binding in workflow.pipeline.cubes:
        if binding.cube_id == cube_id:
            return dict(binding.params)
    return {}


def case_parses_full_file() -> None:
    workflow = parse({"version": 1, "cubes": [dict(c) for c in CUBES],
                      "run": {"seeds": [142857], "out": "out"}})
    assert workflow.name == "workflow", workflow.name
    assert len(workflow.pipeline.cubes) == len(CUBES), len(workflow.pipeline.cubes)
    assert workflow.run.seeds == (142857,), workflow.run.seeds
    assert workflow.composition().ok, workflow.composition().format_issues()
    # полный воркфлоу не требует внешних входов: источники всё производят
    assert workflow.pipeline.inputs == frozenset(), workflow.pipeline.inputs


def case_partial_workflow_needs_inputs() -> None:
    workflow = parse({"version": 1, "cubes": [
        {"id": "encoder.qwen3vl"},
        {"id": "strategy.precision.fp8"},
        {"id": "strategy.kv_offload"},
        {"id": "transformer.qwen_image21"},
        {"id": "decoder.vae.qwen21"},
        {"id": "decoder.postprocess.image"},
    ]})
    assert workflow.pipeline.inputs == frozenset({"prompt", "refs"}), \
        workflow.pipeline.inputs
    # `frame` внешним быть не может: канвас обязаны произвести кубики (`source.frame`)
    composition = workflow.composition()
    assert "port_missing" in [issue.code for issue in composition.issues], composition.issues


def case_version_and_forbidden() -> None:
    for bad, needle in (
        ({"cubes": [{"id": "source.frame"}]}, "version"),
        ({"version": 2, "cubes": [{"id": "source.frame"}]}, "version"),
        ({"version": 1, "template": "VRAM16-preview", "cubes": [{"id": "source.frame"}]},
         "копию"),
        ({"version": 1, "extends": "base", "cubes": [{"id": "source.frame"}]}, "самодостаточен"),
        ({"version": 1, "preset": "base", "cubes": [{"id": "source.frame"}]}, "шаблон"),
        ({"version": 1, "generation": {"steps": 25}, "cubes": [{"id": "source.frame"}]},
         "transformer.qwen_image21"),
        ({"version": 1, "size_px": 768, "cubes": [{"id": "source.frame"}]}, "source.frame"),
        ({"version": 1, "cube": [], "cubes": [{"id": "source.frame"}]}, "неизвестные ключи"),
    ):
        try:
            parse(bad)
        except ValueError as exc:
            assert needle in str(exc), (bad, needle, exc)
        else:                                                       # pragma: no cover
            raise AssertionError(f"{bad} обязан отвергаться")


def case_cubes_validation() -> None:
    for bad, needle in (
        ({"version": 1, "cubes": []}, "непустым"),
        ({"version": 1, "cubes": {"id": "source.frame"}}, "непустым"),
        ({"version": 1, "cubes": [{"params": {}}]}, "id"),
        ({"version": 1, "cubes": [{"id": "nope.cube"}]}, "не зарегистрирован"),
        ({"version": 1, "cubes": [{"id": "source.frame"}, {"id": "source.frame"}]}, "дважды"),
        ({"version": 1, "cubes": [{"id": "source.frame", "params": 5}]}, "params"),
        ({"version": 1, "cubes": [{"id": "source.frame", "extra": 1}]}, "неизвестные ключи"),
    ):
        try:
            parse(bad)
        except ValueError as exc:
            assert needle in str(exc), (bad, needle, exc)
        else:                                                       # pragma: no cover
            raise AssertionError(f"{bad} обязан отвергаться")


def case_run_spec() -> None:
    assert RunSpec.from_mapping({}).seeds == ()
    assert RunSpec.from_mapping({"seeds": "32-34"}).seeds == (32, 33, 34)
    assert RunSpec.from_mapping({"seeds": "1,7,42"}).seed_spec() == "1,7,42"
    assert RunSpec.from_mapping({"seeds": [84]}).with_seed(7).seeds == (7,)
    for bad, needle in (
        ({"seeds": [1], "count": 3}, "взаимоисключающие"),
        ({"cache_supersede": "all"}, "cache_supersede"),
        ({"nope": 1}, "неизвестные ключи"),
        ({"seeds": "5-1"}, "наоборот"),
    ):
        try:
            RunSpec.from_mapping(bad)
        except ValueError as exc:
            assert needle in str(exc), (bad, needle, exc)
        else:                                                       # pragma: no cover
            raise AssertionError(f"{bad} обязан отвергаться")
    merged = RunSpec.from_mapping({"seeds": [1], "out": "out"}).merged(seeds="7-8", logs="false")
    assert merged.seeds == (7, 8) and merged.logs is False, merged


def case_set_new_paths() -> None:
    workflow = parse({"version": 1, "cubes": [dict(c) for c in CUBES]})
    changed = workflow.apply_sets(["transformer.qwen_image21.steps=8",
                                   "encoder.qwen3vl.device=cpu",
                                   "strategy.kv_offload.mode=true",
                                   "decoder.vae.qwen21.tile_px=1024",
                                   "run.seeds=1-3"])
    assert params_of(changed, "transformer.qwen_image21")["steps"] == 8, changed
    assert params_of(changed, "encoder.qwen3vl")["device"] == "cpu"
    assert params_of(changed, "strategy.kv_offload")["mode"] == "true"
    assert params_of(changed, "decoder.vae.qwen21")["tile_px"] == 1024
    assert changed.run.seeds == (1, 2, 3), changed.run.seeds


def case_set_rejects_old_spellings() -> None:
    """Старых написаний нет: `--set generation.steps=8` отвергается списком канонических путей."""
    workflow = parse({"version": 1, "cubes": [dict(c) for c in CUBES]})
    for old in ("generation.steps", "generation.kv_cache",
                "generation.sampler_kwargs.offload_blocks", "generation.sampler_kwargs.ref_px",
                "encoder.device", "upscale.target_height", "size_px"):
        try:
            workflow.apply_sets([f"{old}=1"])
        except ValueError as exc:
            assert "допустимые пути" in str(exc), exc
            assert "transformer.qwen_image21.steps" in str(exc), exc
        else:                                                       # pragma: no cover
            raise AssertionError(f"{old} обязан отвергаться")


def case_set_unknown_lists_paths() -> None:
    workflow = parse({"version": 1, "cubes": [dict(c) for c in CUBES]})
    try:
        workflow.apply_sets(["generation.nope=1"])
    except ValueError as exc:
        assert "transformer.qwen_image21.steps" in str(exc), exc
    else:                                                           # pragma: no cover
        raise AssertionError("неизвестный ключ обязан отвергаться")
    try:
        workflow.apply_sets(["generation.steps"])
    except ValueError as exc:
        assert "key=value" in str(exc), exc
    else:                                                           # pragma: no cover
        raise AssertionError("--set без `=` обязан отвергаться")
    # снятые ключи дают подсказку, а не «неизвестный ключ»
    try:
        workflow.apply_sets(["resolution=1024"])
    except ValueError as exc:
        assert "source.frame" in str(exc), exc
    else:                                                           # pragma: no cover
        raise AssertionError("resolution обязан давать подсказку")


def case_config_mapping_roundtrip() -> None:
    """Обратная проекция восстанавливает значения пресета (переходный режим до cutover)."""
    from qwenstyle.composition import config_mapping_of, spec_of

    cfg = RunConfig.from_profile("VRAM16-4k-slide")
    back = RunConfig.from_mapping(config_mapping_of(spec_of(cfg)))
    assert back.generation.offload == cfg.generation.offload, back.generation.offload
    assert back.generation.kv_cache == cfg.generation.kv_cache, back.generation.kv_cache
    assert back.steps == cfg.steps, (back.steps, cfg.steps)
    assert back.encoder.name == cfg.encoder.name, back.encoder.name
    assert back.size_px == cfg.size_px, (back.size_px, cfg.size_px)
    assert back.upscale.color_fix == cfg.upscale.color_fix, back.upscale.color_fix
    sampler = back.generation.sampler_kwargs
    assert sampler.get("window_attn_px") == 1088, sampler
    assert sampler.get("ref_px") == 768, sampler
    # ключи, совпадающие с умолчаниями кубиков, в обратную проекцию не попадают (шум)
    assert "vae_tile_px" not in sampler, sampler
    assert back.generation.sampler_kwargs.get("offload_blocks") is None, sampler


def case_load_and_templates() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / "workflow.yaml"
        path.write_text(yaml.safe_dump({"version": 1, "cubes": [dict(c) for c in CUBES],
                                        "run": {"seeds": [7]}}, allow_unicode=True),
                        encoding="utf-8")
        workflow = load(path)
        assert workflow.path == path, workflow.path
        assert workflow.base_dir == root, workflow.base_dir
        assert workflow.run.seeds == (7,), workflow.run.seeds
        # шаблоны: <root>/<модель>/<имя>.yaml, имя без модели при неоднозначности — ошибка
        folder = root / "qwen-image-2.1"
        folder.mkdir()
        (folder / "tpl.yaml").write_text(yaml.safe_dump({"version": 1,
                                                         "cubes": [dict(c) for c in CUBES]}),
                                         encoding="utf-8")
        found = load_template("tpl", root=root)
        assert found.name == "tpl", found.name
        # то же имя у второй модели -> нужна квалификация «модель/имя»
        other = root / "h3-image"
        other.mkdir()
        (other / "tpl.yaml").write_text(yaml.safe_dump({"version": 1,
                                                        "cubes": [dict(c) for c in CUBES]}),
                                        encoding="utf-8")
        try:
            load_template("tpl", root=root)
        except ValueError as exc:
            assert "нескольких моделей" in str(exc), exc
        else:                                                       # pragma: no cover
            raise AssertionError("неоднозначный шаблон обязан требовать модель")
        assert load_template("qwen-image-2.1/tpl", root=root).name == "tpl"
        try:
            load_template("nope", root=root)
        except FileNotFoundError as exc:
            assert "tpl" in str(exc), exc
        else:                                                       # pragma: no cover
            raise AssertionError("неизвестный шаблон обязан давать FileNotFoundError")


CASES = [
    ("полный файл разбирается", case_parses_full_file),
    ("частичный воркфлоу требует входы", case_partial_workflow_needs_inputs),
    ("version и запрещённые ключи", case_version_and_forbidden),
    ("проверка cubes", case_cubes_validation),
    ("RunSpec: seeds/count/домены", case_run_spec),
    ("--set: новые пути", case_set_new_paths),
    ("--set: старые написания отвергаются", case_set_rejects_old_spellings),
    ("--set: неизвестный ключ", case_set_unknown_lists_paths),
    ("обратная проекция в RunConfig", case_config_mapping_roundtrip),
    ("load и шаблоны", case_load_and_templates),
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
