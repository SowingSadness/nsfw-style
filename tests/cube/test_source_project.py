"""Кубик `source.project`: чтение проекта промта (без весов, без CUDA).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/cube/test_source_project.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (план `cube-architecture`, шаг 6): раньше промт и референсы собирались в `cmd/gen.py`
и `cmd/_project.py`, и композитор не видел этих входов. Теперь это стадия SOURCE с двумя выходами
(`prompt` + `refs`), и тест фиксирует то, на чём легко разъехаться с прежним поведением:

1. natural-порядок референсов (`2-a` раньше `10-b`) — от него зависит `<imageN>` в промте;
2. дубликат стема (`a.png` + `a.jpg`) берётся один, первый;
3. `refs = '-'`/пусто = без референсов, но промт всё равно читается;
4. ключи (`md5` промта, `sha` референсов) те же, что у `config.prompt_md5` и `io.images_sha`;
5. явные ошибки: нет файла промта, несколько `*.txt` без `prompt=`, пустой промт;
6. тот же порядок, что у `cmd/_project.discover_refs` (одна реализация на проект).
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

from qwenstyle.cmd import _project as project_cmd                     # noqa: E402
from qwenstyle.config import prompt_md5                                # noqa: E402
from qwenstyle.cube import CubeContext, Stage, get_cube                # noqa: E402
from qwenstyle.cubes import source_project                             # noqa: E402
from qwenstyle.io import images_sha, natural_key                       # noqa: E402

PROMPT_TEXT = "fusal style.\n[TASK: TEXT_TO_IMAGE]\n\n[Subject] one young woman\n"


def make_project(root: Path, refs: dict[str, bytes] | None = None,
                 prompts: dict[str, str] | None = None) -> Path:
    """Каталог проекта: `*.txt` + `refs/` (содержимое картинок не важно — кубик их не открывает)."""
    root.mkdir(parents=True, exist_ok=True)
    for name, text in (prompts or {"prompt-test.txt": PROMPT_TEXT}).items():
        (root / name).write_text(text, encoding="utf-8")
    folder = root / "refs"
    folder.mkdir(exist_ok=True)
    for name, data in (refs or {}).items():
        (folder / name).write_bytes(data)
    return root


def case_reads_prompt_and_refs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = make_project(Path(tmp), refs={"10-b.png": b"b", "2-a.png": b"a", "1-c.webp": b"c"})
        params = source_project.SourceParams.from_mapping({"dir": str(root)})
        prompt, refs = source_project.read(params)
        assert prompt.text.strip() == PROMPT_TEXT.strip(), prompt.text[:40]
        # ключ считается по тексту после `.strip()`, как `config.load_prompt`
        assert prompt.md5 == prompt_md5(PROMPT_TEXT.strip()), prompt.md5
        assert refs is not None, "референсы обязаны прочитаться"
        assert refs.grid == (0, 0), refs.grid          # сетку решает энкодер, не источник
        names = [Path(p).name for p in refs.images]
        assert names == ["1-c.webp", "2-a.png", "10-b.png"], names
        assert refs.sha == images_sha([Path(root) / "refs" / n for n in names]), refs.sha


def case_natural_order_matches_project_cmd() -> None:
    """Порядок один на проект: `cmd/_project.discover_refs` и кубик дают одинаковый список."""
    with tempfile.TemporaryDirectory() as tmp:
        root = make_project(Path(tmp), refs={"9-z.png": b"z", "10-y.png": b"y", "1-x.jpg": b"x"})
        from_cmd = [p.name for p in project_cmd.discover_refs(root / "refs")]
        params = source_project.SourceParams.from_mapping({"dir": str(root)})
        _, refs = source_project.read(params)
        from_cube = [Path(p).name for p in refs.images]
        assert from_cmd == from_cube, (from_cmd, from_cube)
        assert from_cube == ["1-x.jpg", "9-z.png", "10-y.png"], from_cube
        # ключ сортировки ведёт себя как обещает имя: числа сравниваются как числа
        assert natural_key("2-a.png") < natural_key("10-a.png")


def case_duplicate_stem_takes_first() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = make_project(Path(tmp), refs={"2-a.png": b"png", "2-a.jpg": b"jpg"})
        params = source_project.SourceParams.from_mapping({"dir": str(root)})
        _, refs = source_project.read(params)
        assert refs is not None and len(refs.images) == 1, refs
        assert Path(refs.images[0]).name == "2-a.jpg", refs.images  # .jpg < .png по имени


def case_no_refs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = make_project(Path(tmp))
        for value in ("-", "", "none"):
            params = source_project.SourceParams.from_mapping({"dir": str(root), "refs": value})
            prompt, refs = source_project.read(params)
            assert refs is None, (value, refs)
            assert prompt.md5 == prompt_md5(PROMPT_TEXT.strip()), value
        empty = source_project.SourceParams.from_mapping({"dir": str(root), "refs": "refs-empty"})
        prompt, refs = source_project.read(empty)          # каталога нет -> референсов нет
        assert refs is None and prompt.text, (refs, prompt.text[:20])


def case_explicit_prompt_name() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = make_project(Path(tmp), prompts={"prompt-a.txt": PROMPT_TEXT,
                                                "prompt-b.txt": PROMPT_TEXT + "b"})
        named = source_project.SourceParams.from_mapping({"dir": str(root), "prompt": "prompt-b"})
        assert source_project.read(named)[0].text.endswith("b"), "стем обязан найти .txt"
        with_ext = source_project.SourceParams.from_mapping({"dir": str(root),
                                                             "prompt": "prompt-a.txt"})
        assert source_project.read(with_ext)[0].text.strip() == PROMPT_TEXT.strip()


def case_errors_are_explicit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = make_project(Path(tmp), prompts={"a.txt": PROMPT_TEXT, "b.txt": PROMPT_TEXT})
        ambiguous = source_project.SourceParams.from_mapping({"dir": str(root)})
        try:
            source_project.read(ambiguous)
        except FileNotFoundError as exc:
            assert "prompt=" in str(exc), exc
        else:                                               # pragma: no cover
            raise AssertionError("два .txt без prompt= обязаны давать ошибку")
        missing = source_project.SourceParams.from_mapping({"dir": str(root), "prompt": "nope"})
        try:
            source_project.read(missing)
        except FileNotFoundError as exc:
            assert "nope" in str(exc), exc
        else:                                               # pragma: no cover
            raise AssertionError("нет файла промта — обязана быть ошибка")
        blank = make_project(Path(tmp) / "blank", prompts={"p.txt": "   \n"})
        params = source_project.SourceParams.from_mapping({"dir": str(blank)})
        try:
            source_project.read(params)
        except ValueError:
            pass
        else:                                               # pragma: no cover
            raise AssertionError("пустой промт обязан давать ошибку")


def case_cube_contract() -> None:
    cube = get_cube("source.project")
    assert cube.stage is Stage.SOURCE, cube.stage
    assert cube.outputs == frozenset({"prompt", "refs"}), cube.outputs
    assert not cube.consumes and not cube.reads, (cube.consumes, cube.reads)
    assert cube.self_test is not None
    cube.self_test(CubeContext())


CASES = [
    ("промт + референсы", case_reads_prompt_and_refs),
    ("natural-порядок совпадает с cmd", case_natural_order_matches_project_cmd),
    ("дубликат стема -> первый", case_duplicate_stem_takes_first),
    ("без референсов", case_no_refs),
    ("явное имя промта", case_explicit_prompt_name),
    ("явные ошибки", case_errors_are_explicit),
    ("контракт кубика", case_cube_contract),
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
