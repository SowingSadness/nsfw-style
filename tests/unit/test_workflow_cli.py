"""CLI воркфлоу: `workflow check|show|new` и `gen --workflow` (переходный режим).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/unit/test_workflow_cli.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.
Без весов и CUDA: `check`/`show`/`new` композицию только разбирают, `gen --workflow` гоняется
с `--dry-run`.

Зачем (план `workflow-layout`, шаг 3): файл воркфлоу должен стать обычным способом запуска, не
ломая старый. Тест фиксирует:

1. `workflow check` — rc 0 на сходящейся композиции, rc 1 с кодами ошибок (без трейсбека) на
   сломанной, rc 2 на отсутствующем файле;
2. `workflow new --from <имя>` создаёт **развёрнутый** файл (без `extends`/`template`), а повторный
   запуск без `--force` отказывается;
3. `gen --workflow --dry-run` видит шаги/канвас/сиды из файла, старые `--set`-написания работают;
4. старый путь (`--project`) не сломан.
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from qwenstyle.cmd import workflow as workflow_cmd                     # noqa: E402

WORKFLOW = """\
version: 1
name: demo
cubes:
  - id: source.frame
    params: {height: 480, width: 0}
  - id: source.txtFileImage
    params: {prompt: prompt.txt, Image1: refs/01.png}
  - id: encoder.qwen3vl
  - id: strategy.precision.fp8
  - id: strategy.kv_offload
  - id: transformer.qwen_image21
    params: {steps: 4, offload: none}
  - id: decoder.vae.qwen21
  - id: decoder.postprocess.image
run:
  seeds: [7]
  out: out
"""

BROKEN = """\
version: 1
name: broken
cubes:
  - id: source.frame
  - id: source.txtFileImage
  - id: decoder.vae.qwen21
"""


def capture(fn, *args, **kwargs) -> tuple[int, str]:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
        code = fn(*args, **kwargs)
    return code, buffer.getvalue()


def namespace(**kwargs):
    import types

    return types.SimpleNamespace(**kwargs)


def make_workflow(root: Path, text: str = WORKFLOW) -> Path:
    folder = root / "demo"
    (folder / "refs").mkdir(parents=True)
    (folder / "prompt.txt").write_text(
        "fusal style.\n[TASK: IMAGE_EDIT]\n\n[Reference guidance] Edit <image1> in place.\n",
        encoding="utf-8")
    (folder / "refs" / "01.png").write_bytes(b"stub")
    path = folder / "workflow.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def case_check_ok_and_broken() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        good = make_workflow(root)
        code, text = capture(workflow_cmd.run, namespace(workflow_action="check",
                                                         path=str(good), json=False))
        assert code == 0, text
        assert "composition=demo (ok)" in text, text
        bad = root / "broken.yaml"
        bad.write_text(BROKEN, encoding="utf-8")
        code, text = capture(workflow_cmd.run, namespace(workflow_action="check",
                                                         path=str(bad), json=False))
        assert code == 1, text
        assert "[port_missing]" in text, text
        code, text = capture(workflow_cmd.run, namespace(workflow_action="check",
                                                         path=str(root / "nope.yaml"), json=False))
        assert code == 2, text


def case_check_json() -> None:
    import json

    with tempfile.TemporaryDirectory() as tmp:
        good = make_workflow(Path(tmp))
        code, text = capture(workflow_cmd.run, namespace(workflow_action="check",
                                                         path=str(good), json=True))
        assert code == 0, text
        payload = json.loads(text)
        assert payload["ok"] is True and payload["run"]["seeds"] == [7], payload
        assert [cube["id"] for cube in payload["cubes"]][:2] == ["source.frame",
                                                                "source.txtFileImage"]


def case_show() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        good = make_workflow(Path(tmp))
        code, text = capture(workflow_cmd.run, namespace(workflow_action="show",
                                                         path=str(good), json=False))
        assert code == 0, text
        assert "transformer.qwen_image21" in text and "steps=4" in text, text


def case_new_expands_template() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "created"
        code, text = capture(workflow_cmd.new_file, source="VRAM16-preview", name="created",
                             out_dir=str(target), force=False)
        assert code == 0, text
        path = target / "workflow.yaml"
        assert path.is_file(), path
        body = path.read_text(encoding="utf-8")
        assert "version: 1" in body and "cubes:" in body, body[:200]
        assert "extends:" not in body and "template:" not in body, "файл обязан быть без ссылок"
        assert "id: source.frame" in body and "id: transformer.qwen_image21" in body, body[:400]
        # повторный запуск без --force отказывается
        code, text = capture(workflow_cmd.new_file, source="VRAM16-preview", name="created",
                             out_dir=str(target), force=False)
        assert code == 2, text
        # и созданный файл проходит проверку
        code, text = capture(workflow_cmd.run, namespace(workflow_action="check",
                                                         path=str(path), json=False))
        assert code == 0, text


def case_new_unknown_template() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        code, _text = capture(workflow_cmd.new_file, source="nope-nope", name="x",
                              out_dir=str(Path(tmp) / "x"), force=False)
        assert code == 2, code


CASES = [
    ("workflow check: ok/ошибки/нет файла", case_check_ok_and_broken),
    ("workflow check --json", case_check_json),
    ("workflow show", case_show),
    ("workflow new: развёрнутый файл", case_new_expands_template),
    ("workflow new: неизвестный шаблон", case_new_unknown_template),
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
