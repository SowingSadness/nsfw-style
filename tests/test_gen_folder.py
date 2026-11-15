"""Папка промта `prompts/<имя>/` для `gen`: `prompt.txt`, фильтр картинок, порядок, guard.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_gen_folder.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем: `gen` — главная точка входа пользователя (план публикации, решение 7), и весь её вход
описывается папкой: `prompt.txt` + `image1.png`…`image10.*`. Ошибка здесь не падает — она молча
берёт не те файлы, отдаёт модели не тот набор референсов или просит пользователя гадать, почему
«ничего не происходит». Тест фиксирует пять правил раскладки **без весов и без генерации**
(только функции `qwenstyle/cmd/gen.py` и `pipeline.input_problem`):

1. файл промта — **ровно** `prompt.txt`; нет его — понятная ошибка с путём (exit 2);
2. референс — только файл с префиксом `image` и расширением `.png/.jpg/.jpeg/.webp`
   (регистр букв и расширения не важен; `prompt.txt`, заметки и `.gif` игнорируются);
3. порядок — **естественный**: `image2.png` раньше `image10.png` (`sorted()` дал бы наоборот);
4. `Capabilities.max_images` соблюдается: из 12 файлов берутся первые 10, в лог — кого отбросили;
5. папка без картинок при `requires_image=True` — ошибка (а для T2I-модели — законный `noref`).

Пользовательские каталоги не трогаются: корни `prompts/` и `out/` подменяются на временный каталог
(`gen.PROMPT_ROOT`/`gen.session_dir`), поэтому тест не оставляет следов в репозитории.
"""
from __future__ import annotations

import contextlib
import functools
import io
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                   # тест не должен оставлять __pycache__ в пакете

import argparse                                   # noqa: E402 — после правки sys.path

from qwenstyle.cmd import gen as gen_cmd          # noqa: E402
from qwenstyle.config import RunConfig   # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
#: Готовый FUSAL-промт тестов (лежит в репозитории, а не в `prompts/`).
TEST_PROMPT = TESTS_DIR / "texts" / "exp1.txt"
#: Пресеты с разными требованиями к входу: `strong-release` требует референс, `h3-image` — нет.
PRESET_NEEDS_IMAGE = "strong-release"
PRESET_T2I = "h3-image"
#: Модель qwen_image21 принимает 10 референсов (`Capabilities.max_images`).
MAX_IMAGES = 10


# --------------------------------------------------------------------------- #
# Временные корни пользовательских каталогов
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def temp_roots():
    """`prompts/` и `out/` во временном каталоге: `gen` не пишет в репозиторий.

    Подменяются только корни (`gen.PROMPT_ROOT` и `out_root` у `gen.session_dir`); сама логика
    поиска/сортировки/guard'а остаётся пользовательской. `cwd` не меняется: относительные пути
    кешей (`cache/models/…`) должны остаться репозиторными.
    """
    with tempfile.TemporaryDirectory(prefix="qwenstyle-gen-folder-") as tmp:
        root = Path(tmp)
        prompt_root = root / "prompts"
        prompt_root.mkdir()
        original_root = gen_cmd.PROMPT_ROOT
        original_session_dir = gen_cmd.session_dir
        gen_cmd.PROMPT_ROOT = str(prompt_root)
        gen_cmd.session_dir = functools.partial(original_session_dir, out_root=str(root / "out"))
        try:
            yield prompt_root
        finally:
            gen_cmd.PROMPT_ROOT = original_root
            gen_cmd.session_dir = original_session_dir


def png_bytes(width: int = 64, height: int = 32) -> bytes:
    """Минимальный настоящий PNG (PIL открывает файл по содержимому, а не по расширению)."""
    from PIL import Image

    image = Image.new("RGB", (width, height), (32, 64, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_case(root: Path, name: str, files: "list[str] | dict[str, bytes]",
              *, prompt: bool = True) -> Path:
    """Создать папку промта `root/<name>/`: `prompt.txt` (копия теста) + перечисленные файлы."""
    folder = root / name
    folder.mkdir(parents=True)
    if prompt:
        shutil.copyfile(TEST_PROMPT, folder / gen_cmd.PROMPT_FILE)
    payload = png_bytes() if files else b""
    if isinstance(files, dict):
        for file_name, data in files.items():
            (folder / file_name).write_bytes(data)
    else:
        for file_name in files:
            (folder / file_name).write_bytes(payload)
    return folder


def run_gen(prompt: str, *extra: str, preset: str = PRESET_NEEDS_IMAGE,
            **kwargs: object) -> "tuple[int, str]":
    """`gen.run()` в процессе (без подпроцесса): вернуть (код, stdout+stderr)."""
    namespace = dict(preset=preset, prompt=prompt, aspect=None, sets=["cache_supersede=none"],
                     seeds="1", count=None, dry_run=True, no_logs=True)
    namespace.update(kwargs)
    args = argparse.Namespace(**namespace)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        code = gen_cmd.run(args)
    return code, buffer.getvalue()


# --------------------------------------------------------------------------- #
# Кейсы
# --------------------------------------------------------------------------- #

def case_prompt_file() -> None:
    """1. Файл промта — ровно `prompt.txt`; его отсутствие даёт понятную ошибку (exit 2)."""
    if gen_cmd.PROMPT_FILE != "prompt.txt":
        raise AssertionError(f"gen.PROMPT_FILE={gen_cmd.PROMPT_FILE!r}, ожидалось 'prompt.txt'")
    with temp_roots() as root:
        # правильное имя файла — план строится (dry-run: ни весов, ни кадров)
        make_case(root, "ok", ["image1.png"])
        code, out = run_gen("ok")
        if code != 0:
            raise AssertionError(f"папка с prompt.txt не принята (код {code}): {out[-600:]}")
        # «похожие» имена не считаются файлом промта (`PROMPT.TXT` не проверяем: файловая
        # система Windows регистронезависима, и это фактически то же имя)
        make_case(root, "wrong", ["image1.png"], prompt=False)
        for file_name in ("prompt.md", "prompt.txt.bak", "prompt", "prompt-en.txt"):
            (root / "wrong" / file_name).write_text("fusal style.", encoding="utf-8")
        code, out = run_gen("wrong")
        if code != 2:
            raise AssertionError(f"без prompt.txt ожидался код 2, получено {code}: {out[-400:]}")
        if "prompt.txt" not in out:
            raise AssertionError(f"в ошибке нет имени нужного файла: {out[-400:]}")
        # несуществующая папка — та же ошибка, путь назван
        code, out = run_gen("nope")
        if code != 2 or "nope" not in out:
            raise AssertionError(f"нет папки промта: код {code}, вывод {out[-400:]}")
    print("  ok: только prompt.txt; иначе exit 2 с путём к отсутствующему файлу")


def case_extension_filter() -> None:
    """2. Референс = `image*` + png/jpg/jpeg/webp; регистр не важен, прочее игнорируется."""
    with temp_roots() as root:
        folder = make_case(root, "mixed", [
            "image1.png", "image2.JPG", "image3.jpeg", "IMAGE4.WebP",
            "prompt.txt", "notes.md", "image5.gif", "photo.png", "image", "image6.tiff",
            ".gitkeep",
        ])
        found = [path.name for path in gen_cmd.discover_images(folder)]
        want = ["image1.png", "image2.JPG", "image3.jpeg", "IMAGE4.WebP"]
        if found != want:
            raise AssertionError(f"фильтр расширений/префикса: {found}, ожидалось {want}")
        if gen_cmd.discover_images(folder / "нет-такой-папки"):
            raise AssertionError("discover_images несуществующей папки должен вернуть пустой список")
    print(f"  ok: из {len(found)} найденных — только image* с png/jpg/jpeg/webp (регистр не важен)")


def case_natural_sort() -> None:
    """3. Естественная сортировка: `image2` раньше `image10` (а не как у `sorted()`)."""
    names = ["image10.png", "image2.png", "image1.png"]
    plain = sorted(names)
    if plain == ["image1.png", "image2.png", "image10.png"]:
        raise AssertionError("sorted() уже даёт естественный порядок — тест ничего не проверяет")
    with temp_roots() as root:
        folder = make_case(root, "order", names + ["image3.png"])
        found = [path.name for path in gen_cmd.discover_images(folder)]
        want = ["image1.png", "image2.png", "image3.png", "image10.png"]
        if found != want:
            raise AssertionError(f"порядок картинок {found}, ожидался естественный {want}")
    print(f"  ok: {names} -> image1 < image2 < image10 (sorted() дал бы {plain})")


def case_max_images_guard() -> None:
    """4. Guard `Capabilities.max_images`: из 12 картинок берутся первые 10, отброшенные — в лог."""
    cfg = RunConfig.from_profile(PRESET_NEEDS_IMAGE)
    caps = gen_cmd.backend_caps(cfg)
    limit = getattr(caps, "max_images", None)
    if limit != MAX_IMAGES:
        raise AssertionError(f"caps.max_images={limit!r}, ожидалось {MAX_IMAGES} у {caps}")
    with temp_roots() as root:
        names = [f"image{index}.png" for index in range(1, 13)]
        folder = make_case(root, "many", names)
        images = gen_cmd.discover_images(folder)
        if len(images) != 12:
            raise AssertionError(f"discover_images вернул {len(images)} из 12")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            limited_cfg, kept = gen_cmd.apply_image_limit(cfg, images, PRESET_NEEDS_IMAGE, caps)
        out = buffer.getvalue()
        if [path.name for path in kept] != names[:MAX_IMAGES]:
            raise AssertionError(f"оставлены {[p.name for p in kept]}, ожидались "
                                 f"{names[:MAX_IMAGES]}")
        for token in ("12", str(MAX_IMAGES), "image11", "image12"):
            if token not in out:
                raise AssertionError(f"в сообщении guard'а нет {token!r}: {out.strip()!r}")
        if len(limited_cfg.inputs) != MAX_IMAGES:
            raise AssertionError(f"в конфиг ушло {len(limited_cfg.inputs)} референсов, "
                                 f"ожидалось {MAX_IMAGES}")
        # ровно 10 и меньше — ничего не отбрасывается и не печатается
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            _, kept_ten = gen_cmd.apply_image_limit(cfg, images[:MAX_IMAGES],
                                                    PRESET_NEEDS_IMAGE, caps)
        if len(kept_ten) != MAX_IMAGES or buffer.getvalue().strip():
            raise AssertionError(f"на 10 картинках guard молчит и ничего не режет: "
                                 f"{buffer.getvalue()!r}")
    print(f"  ok: 12 картинок -> взяты image1..image{MAX_IMAGES}, image11..image12 отброшены с логом")


def case_no_images() -> None:
    """5. Папка без картинок: ошибка для `requires_image=True`, законный `noref` для T2I."""
    with temp_roots() as root:
        make_case(root, "empty", [])
        code, out = run_gen("empty", preset=PRESET_NEEDS_IMAGE)
        if code != 2:
            raise AssertionError(f"без картинок при requires_image=True ожидался код 2, "
                                 f"получено {code}: {out[-400:]}")
        for token in ("image", PRESET_NEEDS_IMAGE):
            if token not in out:
                raise AssertionError(f"в ошибке нет {token!r}: {out[-400:]}")
        # модель, которой референс не нужен, строит T2I-план (noref) на том же промте
        code, out = run_gen("empty", preset=PRESET_T2I, aspect="16:9")
        if code != 0:
            raise AssertionError(f"T2I без картинок ({PRESET_T2I}) ожидал код 0, "
                                 f"получено {code}: {out[-600:]}")
        if "T2I" not in out:
            raise AssertionError(f"в выводе нет пометки T2I: {out[-400:]}")
    print(f"  ok: пустая папка -> exit 2 для {PRESET_NEEDS_IMAGE}, T2I (noref) для {PRESET_T2I}")


def case_prompt_dir() -> None:
    """6. `prompt_dir(name)` = `<PROMPT_ROOT>/<имя>` (никаких вложенных путей и `..`)."""
    with temp_roots() as root:
        path = gen_cmd.prompt_dir("my-scene")
        if Path(path).name != "my-scene" or Path(path).parent.name != root.name:
            raise AssertionError(f"prompt_dir вернул {path!r} при корне {root}")
        make_case(root, "case-name", ["image1.png"])
        code, out = run_gen("case-name")
        if code != 0:
            raise AssertionError(f"папка с дефисом в имени не принята: {out[-400:]}")
    print("  ok: prompt_dir -> prompts/<имя>, дефисы и подчёркивания в имени работают")


CASES = (
    ("1. файл промта — ровно prompt.txt", case_prompt_file),
    ("2. фильтр расширений и префикса", case_extension_filter),
    ("3. естественная сортировка image2 < image10", case_natural_sort),
    ("4. guard max_images", case_max_images_guard),
    ("5. папка без картинок", case_no_images),
    ("6. prompt_dir", case_prompt_dir),
)


def main() -> int:
    if not TEST_PROMPT.is_file():
        print(f"FAIL: нет промта тестов {TEST_PROMPT}")
        return 1
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
