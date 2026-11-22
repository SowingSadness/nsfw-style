"""`cubes` и `compose`: API кубиков для UI (плана `cube-architecture`, шаг 8).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/unit/test_cubes_cli.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.
Без torch, без CUDA, без сети (всё через `cubes_view`-подобные функции и `run_cli`).

Зачем этот тест: подкоманда `cubes` — это контракт с UI. Если схема разъедется с `ParameterSpec`,
конструктор начнёт принимать параметры, которые ядро отвергает. Поэтому проверяется:

1. `list`/`list --json` — все 14 кубиков, стадии и порты на месте (включая `frames`);
2. `show --json` — параметры, диапазоны, `--set`-путь и doc для каждого;
3. `test` — `self_test` всех кубиков проходит без CUDA (`self_test` вообще не трогает torch-CUDA);
4. `schema` — draft-07: у каждого параметра есть `type`, enum-параметры дают `enum`,
   `additionalProperties=false`, схема файла композиции ссылается на реальные id;
5. `check`/`compose --spec` — сходящаяся композиция rc 0, сломанная rc 1 **с кодами ошибок и без
   трейсбека**, отсутствующий файл rc 2;
6. `cubes list` не падает при пустом окружении (реестр наполняется импортом пакета).
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from qwenstyle.cmd import compose as compose_cmd                      # noqa: E402
from qwenstyle.cmd import cubes as cubes_cmd                          # noqa: E402
from qwenstyle.cube import Stage, all_cubes                           # noqa: E402

VALID = """\
name: full-4k
inputs: [prompt, refs]
cubes:
  - id: source.frame
    params: {width: 2912, height: 2176, ref_crop_side: 768}
  - id: source.txtFileImage
    params: {prompt: prompt-qwen.txt, Image1: refs/a.png}
  - id: encoder.qwen3vl
  - id: strategy.precision.fp8
  - id: strategy.kv_offload
    params: {mode: offload}
  - id: transformer.qwen_image21
    params: {steps: 25, offload: group}
  - id: decoder.vae.qwen21
  - id: decoder.postprocess.image
  - id: post.colorfix
    params: {mode: wavelet}
"""

BROKEN = """\
name: broken
cubes:
  - id: source.frame
  - id: source.txtFileImage
  - id: decoder.vae.qwen21
"""


def capture(fn, *args, **kwargs) -> tuple[int, str]:
    """Вызвать функцию, перехватив stdout: `(код, текст)`."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = fn(*args, **kwargs)
    return code, buffer.getvalue()


def namespace(**kwargs) -> object:
    import types

    return types.SimpleNamespace(**kwargs)


def case_list() -> None:
    code, text = capture(cubes_cmd.run, namespace(cubes_action="list", stage=None, json=False))
    assert code == 0, text
    assert "decoder.postprocess.image" in text and "source.frame" in text, text[:200]
    rows = [line for line in text.splitlines() if line.startswith("  ") and "." in line]
    assert len(rows) == len(all_cubes()), (len(rows), len(all_cubes()))
    code, payload = capture(cubes_cmd.run, namespace(cubes_action="list", stage=None, json=True))
    data = json.loads(payload)
    assert code == 0 and len(data) == len(all_cubes()), len(data)
    ids = {entry["id"] for entry in data}
    assert "decoder.vae.qwen21" in ids and "decoder.postprocess.image" in ids, ids
    decoder = next(entry for entry in data if entry["id"] == "decoder.vae.qwen21")
    assert decoder["outputs"] == ["frames"], decoder
    # стадия как фильтр
    code, payload = capture(cubes_cmd.run,
                            namespace(cubes_action="list", stage=Stage.POST.value, json=True))
    assert code == 0
    assert {entry["stage"] for entry in json.loads(payload)} == {"post"}, payload


def case_show() -> None:
    code, payload = capture(cubes_cmd.run,
                            namespace(cubes_action="show", id="transformer.qwen_image21", json=True))
    assert code == 0, payload
    cube = json.loads(payload)
    assert cube["id"] == "transformer.qwen_image21", cube
    assert cube["requires"] == ["fp8", "kv_offload"], cube["requires"]
    names = [param["name"] for param in cube["params"]]
    assert names[:4] == ["backend", "repo", "revision", "steps"], names
    steps = next(param for param in cube["params"] if param["name"] == "steps")
    assert steps["type"] == "int" and steps["minimum"] == 1 and steps["maximum"] == 50, steps
    assert steps["set_path"] == "generation.steps", steps
    assert steps["doc"], "у параметра обязан быть doc для UI"
    # несуществующий кубик — код 2 и текст, без трейсбека
    code, payload = capture(cubes_cmd.run, namespace(cubes_action="show", id="nope", json=True))
    assert code == 2, code


def case_test_all() -> None:
    code, payload = capture(cubes_cmd.run,
                            namespace(cubes_action="test", id=None, json=True))
    assert code == 0, payload
    results = json.loads(payload)
    assert len(results) == len(all_cubes()), len(results)
    assert all(row["status"] == "ok" for row in results), results
    assert all(row["s"] < 5.0 for row in results), [(r["id"], r["s"]) for r in results]


def case_schema_draft07() -> None:
    code, payload = capture(cubes_cmd.run, namespace(cubes_action="schema", id=None))
    assert code == 0, payload
    bundle = json.loads(payload)
    assert bundle["pipeline"]["$schema"].endswith("draft-07/schema#"), bundle["pipeline"]["$schema"]
    ids = {cube.id for cube in all_cubes()}
    assert set(bundle["cubes"]) == ids, set(bundle["cubes"]) ^ ids
    for cube_id, schema in bundle["cubes"].items():
        assert schema["type"] == "object", cube_id
        assert schema["additionalProperties"] is False, cube_id
        for name, prop in schema["properties"].items():
            assert prop["type"] in ("integer", "number", "boolean", "string"), (cube_id, name)
            if "enum" in prop:
                assert prop["type"] == "string", (cube_id, name)
            assert "default" in prop, (cube_id, name)
    # enum-параметр действительно даёт enum
    post = bundle["cubes"]["decoder.postprocess.image"]["properties"]["output_type"]
    assert post["enum"] == ["pil", "np"], post
    # схема файла композиции ссылается на реальные id
    pipeline = bundle["pipeline"]
    item_ids = set(pipeline["properties"]["cubes"]["items"]["properties"]["id"]["enum"])
    assert item_ids == ids, item_ids ^ ids
    code, payload = capture(cubes_cmd.run, namespace(cubes_action="schema",
                                                     id="decoder.postprocess.image"))
    assert code == 0, payload
    single = json.loads(payload)
    assert single["title"] == "decoder.postprocess.image", single["title"]


def case_check_ok_and_broken() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        good = Path(tmp) / "ok.yaml"
        good.write_text(VALID, encoding="utf-8")
        bad = Path(tmp) / "bad.yaml"
        bad.write_text(BROKEN, encoding="utf-8")
        code, text = capture(compose_cmd.run,
                             namespace(spec=str(good), check=True, json=False))
        assert code == 0, text
        assert "(ok)" in text, text
        code, text = capture(compose_cmd.run, namespace(spec=str(bad), check=True, json=False))
        assert code == 1, text
        assert "[port_missing]" in text, text
        assert "Traceback" not in text, "пользователю не показывают трейсбек"
        code, text = capture(compose_cmd.run,
                             namespace(spec=str(Path(tmp) / "missing.yaml"), check=True, json=False))
        assert code == 2, text


def case_check_rejects_bad_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cases = {
            "no_cubes.yaml": "name: x\n",
            "unknown_key.yaml": "cubes: [{id: source.frame}]\nextra: 1\n",
            "bad_entry.yaml": "cubes: [source.frame]\n",
            "bad_params.yaml": "cubes: [{id: source.frame, params: 5}]\n",
        }
        for name, text in cases.items():
            path = Path(tmp) / name
            path.write_text(text, encoding="utf-8")
            try:
                cubes_cmd.load_pipeline_spec(str(path))
            except ValueError as exc:
                assert str(exc), name
            else:                                               # pragma: no cover
                raise AssertionError(f"{name}: обязан быть ValueError")


def case_loader_defaults() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p.yaml"
        path.write_text("cubes:\n  - id: source.frame\n", encoding="utf-8")
        spec = cubes_cmd.load_pipeline_spec(str(path))
        assert spec.name == "p", spec.name
        assert spec.inputs == frozenset({"prompt", "refs"}), spec.inputs
        assert spec.cubes[0].params == {}, spec.cubes[0].params


CASES = [
    ("cubes list (текст и json)", case_list),
    ("cubes show: параметры и домены", case_show),
    ("cubes test: self_test всех", case_test_all),
    ("cubes schema: draft-07", case_schema_draft07),
    ("compose --check: ok / ошибки / нет файла", case_check_ok_and_broken),
    ("loader отвергает плохой файл", case_check_rejects_bad_file),
    ("loader: дефолты имени и портов", case_loader_defaults),
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
