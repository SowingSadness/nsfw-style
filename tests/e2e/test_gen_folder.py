"""Проект промта `prompts/<project>/` для `gen`: refs/, `prompt-<model>.txt`, сессия.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_gen_folder.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем: `gen` — главная точка входа пользователя, и весь её вход описывается проектом:
`refs/` (любые имена, порядок = natural sort) + `prompt-<model>.txt` + опциональный
`project.yaml`. Ошибка здесь не падает — она молча берёт не те файлы, отдаёт модели не тот
набор референсов или просит пользователя гадать, почему «ничего не происходит».

Тест фиксирует правила раскладки **без весов и без генерации**:

1. `refs/`: только png|jpg|jpeg|webp, natural sort, дубликат стема -> первый + предупреждение;
2. `--prompt`: `<project>/<stem>`, `<project>/<stem>.txt`, `<project>` при единственном `*.txt`;
   ошибки с перечнем кандидатов; выход из `prompts/` отбит;
3. tag сессии: `prompt-qwen` -> `qwen`, `canyon-iren` -> `canyon-iren`;
4. имена кадров `<seed>-<steps>[-gen].png`, лог у кадра, `meta.json` — один на сессию;
5. `Capabilities.max_images` соблюдается (12 -> 10), `<imageN>` = порядок refs;
6. нет `refs/` при `requires_image=True` -> exit 2 с текстом про refs; T2I-пресет — законный
   `noref`;
7. переименование референсов с сохранением порядка **не меняет ключ кэша** (`io.images_sha`).

Пользовательские каталоги не трогаются: корень `prompts/` подменяется на временный
(`_project.PROMPT_ROOT`), поэтому тест не оставляет следов в репозитории.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                   # тест не должен оставлять __pycache__ в пакете

from qwenstyle import io as qio                  # noqa: E402
from qwenstyle.cmd import _project as project    # noqa: E402
from qwenstyle.cmd import gen as gen_cmd         # noqa: E402
from qwenstyle.config import ConfigError, RunConfig   # noqa: E402

TESTS_DIR = Path(__file__).resolve().parents[1]      # `tests/` (файл лежит в `tests/e2e/`)
#: Готовый FUSAL-промт тестов (лежит в репозитории, а не в `prompts/`).
TEST_PROMPT = TESTS_DIR / "texts" / "exp1.txt"


# --------------------------------------------------------------------------- #
# Временный корень projects
# --------------------------------------------------------------------------- #

@contextlib.contextmanager
def temp_root():
    """`prompts/` во временном каталоге: `gen` не пишет в репозиторий."""
    with tempfile.TemporaryDirectory(prefix="qwenstyle-gen-folder-") as tmp:
        root = Path(tmp) / "prompts"
        root.mkdir()
        original = project.PROMPT_ROOT
        project.PROMPT_ROOT = root.as_posix()
        try:
            yield root
        finally:
            project.PROMPT_ROOT = original


def png_bytes(width: int = 64, height: int = 32) -> bytes:
    """Минимальный настоящий PNG (PIL открывает файл по содержимому, а не по расширению)."""
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (len(data).to_bytes(4, "big") + body
                + zlib.crc32(body).to_bytes(4, "big"))

    raw = b"".join(b"\x00" + bytes([40, 80, 120]) * width for _ in range(height))
    header = (b"\x89PNG\r\n\x1a\n"
              + chunk(b"IHDR", width.to_bytes(4, "big") + height.to_bytes(4, "big")
                      + b"\x08\x02\x00\x00\x00"))
    return header + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def make_project(root: Path, name: str, refs: list[str] | None = None,
                 prompts: tuple[str, ...] = ("prompt-qwen.txt",),
                 refs_dir: str = "refs") -> Path:
    """Проект с `refs/` и промт-файлами (текст — готовый FUSAL-промт тестов)."""
    folder = root / name
    folder.mkdir(parents=True)
    if refs:
        (folder / refs_dir).mkdir()
        for index, filename in enumerate(refs, start=1):
            (folder / refs_dir / filename).write_bytes(png_bytes(width=32 + index))
    for filename in prompts:
        shutil.copyfile(TEST_PROMPT, folder / filename)
    return folder


def gen_args(**overrides) -> argparse.Namespace:
    """Namespace со всеми полями `gen` (значения по умолчанию — как у CLI)."""
    values = dict(preset=None, prompt=None, project=None, out=None, aspect=None, sets=[],
                  upscale=None, seeds="1", count=None, dry_run=True, no_logs=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def call_gen(args: argparse.Namespace) -> tuple[int, str, str]:
    """`gen.run` с перехватом stdout/stderr (без весов: `--dry-run`)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = gen_cmd.run(args)
    return code, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- #
# Кейсы
# --------------------------------------------------------------------------- #

def case_refs_discovery() -> None:
    """1. `refs/`: фильтр расширений, natural sort, дубликат стема -> первый."""
    with temp_root() as root:
        names = ["10-c.webp", "2-b.JPG", "01-a.png", "notes.md", "anim.gif", "dup.jpg", "dup.png"]
        folder = make_project(root, "p", names)
        got = [path.name for path in project.discover_refs(folder / "refs")]
        want = ["01-a.png", "2-b.JPG", "10-c.webp", "dup.jpg"]
        if got != want:
            raise AssertionError(f"discover_refs -> {got}, ожидалось {want}")
        if project.discover_refs(folder / "none") != []:
            raise AssertionError("discover_refs непустого несуществующего каталога")
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            project.discover_refs(folder / "refs")
        if "дубликат dup.png" not in captured.getvalue():
            raise AssertionError(f"нет предупреждения о дубликате: {captured.getvalue()!r}")
        print("  ok: natural sort (01, 2, 10), фильтр расширений, дубликат -> первый + warning")


def case_resolve_prompt_forms() -> None:
    """2. `--prompt`: стем, `.txt`, каталог с единственным `.txt`; ошибки с перечнем."""
    with temp_root() as root:
        make_project(root, "one", ["01.png"])
        make_project(root, "two", ["01.png"], prompts=("prompt-qwen.txt", "prompt-h3.txt"))
        for raw in ("one/prompt-qwen", "one/prompt-qwen.txt"):
            prompt = project.resolve_prompt(raw)
            if prompt.prompt_file.name != "prompt-qwen.txt" or prompt.tag != "qwen":
                raise AssertionError(f"{raw} -> {prompt}")
            if prompt.project_dir.name != "one":
                raise AssertionError(f"{raw}: проект {prompt.project_dir}")
        single = project.resolve_prompt("one")
        if single.prompt_file.name != "prompt-qwen.txt":
            raise AssertionError(f"каталог с одним .txt -> {single.prompt_file}")
        for raw, token in (("one/nope", "prompt-qwen.txt"), ("two", "prompt-h3.txt")):
            try:
                project.resolve_prompt(raw)
            except ConfigError as exc:
                if token not in str(exc):
                    raise AssertionError(f"в ошибке {raw!r} нет {token!r}: {exc}") from None
            else:
                raise AssertionError(f"{raw!r} не отбит")
        for raw in ("../outside", "one/../../x", "D:/absolute/prompt-qwen"):
            try:
                project.resolve_prompt(raw)
            except ConfigError:
                continue
            raise AssertionError(f"выход из prompts/ не отбит: {raw!r}")
        print("  ok: стем/.txt/каталог; несколько .txt и выход из prompts/ -> ConfigError")


def case_tags() -> None:
    """3. Tag сессии: `prompt-qwen` -> `qwen`, прочее — как есть, мусор — в `-`."""
    cases = {
        "prompt-qwen": "qwen",
        "prompt-H3": "h3",
        "canyon-iren": "canyon-iren",
        "My Scene!": "my-scene",
    }
    for stem, want in cases.items():
        got = project.tag_for(stem, Path("prompts/fallback"))
        if got != want:
            raise AssertionError(f"tag_for({stem!r}) -> {got!r}, ожидалось {want!r}")
    if project.tag_for("prompt", Path("prompts/camp")) != "camp":
        raise AssertionError("tag_for('prompt') должен брать имя проекта")
    print("  ok: prompt-qwen->qwen, prompt-H3->h3, мусор -> безопасный tag")


def case_frame_names() -> None:
    """4. Имена: `<seed>-<steps>[-gen].png`; лог — у кадра, `meta.json` — один на сессию."""
    gen_path, up_path = qio.batch_paths("d", 84, 25)
    if (gen_path, up_path) != ("d/84-25-gen.png", "d/84-25.png"):
        raise AssertionError(f"batch_paths -> {(gen_path, up_path)}")
    cfg = RunConfig.from_mapping({"preset": "base", "seed": 84, "out": "out/x.png"})
    for factor, want in ((0.0, "out/x-84-25.png"), (2.0, "out/x-84-25-x2.png")):
        got = qio.out_path(cfg, factor)
        if got != want:
            raise AssertionError(f"out_path(factor={factor}) -> {got}, ожидалось {want}")
        if qio.out_path(cfg.merged(out=got), factor) != got:
            raise AssertionError(f"out_path не идемпотентен для {got}")
    from qwenstyle.runlog import log_path, meta_path
    if log_path("d/84-25.png").name != "84-25.log":
        raise AssertionError(f"log_path -> {log_path('d/84-25.png')}")
    if meta_path("d/84-25.png").name != "84-25-meta.json":
        raise AssertionError(f"meta_path -> {meta_path('d/84-25.png')}")
    print("  ok: 84-25[.log|-meta.json|-gen.png]; out_path идемпотентен")


def case_placeholders_and_limit() -> None:
    """5. `<imageN>` = порядок refs; `max_images` отбрасывает хвост с явной строкой."""
    with temp_root() as root:
        folder = make_project(root, "p", ["01.png", "02.png", "03.png"])
        refs = project.discover_refs(folder / "refs")
        mapping = project.placeholder_map(refs)
        if [index for _, index in mapping] != [1, 2, 3]:
            raise AssertionError(f"placeholder_map -> {mapping}")
        cfg = RunConfig.from_mapping({"preset": "base"})

        class Caps:
            max_images = 2

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            limited, kept = gen_cmd.apply_image_limit(cfg, refs, "base", Caps())
        if len(kept) != 2 or len(limited.inputs) != 2:
            raise AssertionError(f"limit не сработал: kept={kept}, inputs={limited.inputs}")
        if "01.png..02.png" not in captured.getvalue():
            raise AssertionError(f"нет строки об отброшенных: {captured.getvalue()!r}")
        print("  ok: <image1..3> по natural sort; max_images=2 -> взяты 01, 02")


def case_requires_image() -> None:
    """6. Без refs: бекенд, требующий референс, отбивается; T2I — законный `noref`."""
    from qwenstyle.pipeline import input_problem

    cfg = RunConfig.from_mapping({"preset": "VRAM12-preview"})

    class NeedsImage:
        requires_image = True
        max_images = 10

    problem = input_problem(cfg, NeedsImage())
    if not problem or "требует референс" not in problem:
        raise AssertionError(f"input_problem(requires_image=True) -> {problem!r}")
    with temp_root() as root:
        make_project(root, "noref", refs=None)
        # у Qwen T2I разрешён, но без --aspect пропорцию взять неоткуда
        code, _, err = call_gen(gen_args(preset="VRAM12-preview", prompt="noref/prompt-qwen"))
        if code != 2 or "--aspect" not in err:
            raise AssertionError(f"T2I без --aspect: code={code}, stderr={err!r}")
        code, out, err = call_gen(gen_args(preset="VRAM12-preview", prompt="noref/prompt-qwen",
                                           aspect="1:1"))
        if code != 0:
            raise AssertionError(f"T2I с --aspect: code={code}, out={out[-300:]}, err={err[-300:]}")
        if "T2I (noref)" not in out:
            raise AssertionError(f"шапка не сказала про noref: {out[:300]!r}")
        print("  ok: requires_image -> 'требует референс'; без refs+aspect -> exit 2; с aspect -> noref")


def case_cache_key_stable() -> None:
    """7. Переименование refs с сохранением порядка не меняет ключ кэша."""
    with temp_root() as root:
        folder = make_project(root, "p", ["01-a.png", "02-b.png"])
        before = [folder / "refs" / "01-a.png", folder / "refs" / "02-b.png"]
        old = qio.images_sha(before)
        for source, target in (("01-a.png", "a.png"), ("02-b.png", "b.png")):
            (folder / "refs" / source).rename(folder / "refs" / target)
        after = [folder / "refs" / "a.png", folder / "refs" / "b.png"]
        new = qio.images_sha(after)
        if old != new:
            raise AssertionError(f"ключ кэша изменился: {old} -> {new}")
        if qio.images_sha(list(reversed(after))) == new:
            raise AssertionError("порядок референсов перестал влиять на ключ")
        print(f"  ok: {old} == после переименования; обратный порядок даёт другой ключ")


CASES = (
    ("1. refs/: имена, порядок, дубликаты", case_refs_discovery),
    ("2. --prompt: формы и ошибки", case_resolve_prompt_forms),
    ("3. tag сессии", case_tags),
    ("4. имена кадров/лога/meta", case_frame_names),
    ("5. <imageN> и max_images", case_placeholders_and_limit),
    ("6. нет refs/ -> ошибка, T2I -> noref", case_requires_image),
    ("7. ключ кэша от имён не зависит", case_cache_key_stable),
)


def main() -> int:
    failed = 0
    for label, case in CASES:
        try:
            case()
            print(f"PASS {label}", flush=True)
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {label}: {exc}", flush=True)
        except Exception:                                   # noqa: BLE001 — тест не должен врать
            failed += 1
            print(f"FAIL {label}: исключение\n{traceback.format_exc()}", flush=True)
    if failed:
        print(f"{failed} FAIL", flush=True)
        return 1
    print("ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
