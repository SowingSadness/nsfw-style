"""Кубик `source.txtFileImage`: промт из файла + явные референсы -> `prompt` + `refs`.

Зачем вместо каталога `refs/` (план `workflow-layout`, шаг 1): порядок `<image1>…<imageN>` в промте
обязан быть виден в файле пайплайна, а не выводиться из natural sort каталога. Поэтому референсы —
это параметры `Image1…Image10` (предел задаёт сама модель: `Capabilities.max_images` у
Qwen-Image 2.1 равно 10), и добавление референса = правка одной строки в `workflow.yaml`.

    params = source_txtimage.SourceParams.from_mapping({
        "prompt": "prompt-qwen.txt",
        "Image1": "refs/image1.png",
        "Image2": "refs-style/nita-style.jpg",
    })
    prompt, refs = source_txtimage.read(params, base_dir=Path("workflow/nita-camp-model"))

Правила (проверяются здесь, а не в прогоне):

* `prompt` — путь к `.txt` относительно каталога воркфлоу (или абсолютный); расширение можно опустить;
* `ImageN` — пути к картинкам; пустые игнорируются, нумерация обязана быть **без дыр**
  (`Image3` без `Image2` = ошибка: `<imageN>` в промте разъехался бы с фактическим порядком);
* `Image*` нет вовсе → референсов нет (T2I работает, если модель это позволяет);
* кэш: `md5` промта и `sha` референсов считаются здесь же — энкодер и политика вытеснения получают
  готовые ключи, а не пересчитывают их.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

from ..backends.base import ParameterSpec, ResourceBudget
from ..cube import CubeContext, CubeSpec, Stage, register_cube
from ..io import images_sha
from ..ports import Prompt, RefImages

__all__ = ["CUBE", "MAX_IMAGES", "IMAGE_NAMES", "SourceParams", "prompt_md5_of", "read"]

#: Сколько референсов принимает модель (`Capabilities.max_images` у Qwen-Image 2.1).
MAX_IMAGES = 10
#: Имена параметров в порядке `<image1>…<imageN>`.
IMAGE_NAMES = tuple(f"Image{index}" for index in range(1, MAX_IMAGES + 1))
#: Расширения, которые принимаются как референс (пайплайн читает их через PIL).
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


@dataclass(frozen=True)
class SourceParams:
    """Разобранные параметры источника: ровно то, что объявлено в спеке кубика.

    `Image1…Image10` — отдельные поля (а не список), чтобы набор параметров в коде совпадал со
    спекой: контрактный тест сверяет их один-к-одному, а `--set source.txtFileImage.Image2=…`
    работает без дополнительных типов. Упорядоченный список референсов — производное `images`.
    """

    prompt: str = ""
    Image1: str = ""
    Image2: str = ""
    Image3: str = ""
    Image4: str = ""
    Image5: str = ""
    Image6: str = ""
    Image7: str = ""
    Image8: str = ""
    Image9: str = ""
    Image10: str = ""

    @property
    def images(self) -> tuple[str, ...]:
        """Референсы в порядке `<image1>…<imageN>` (пустые параметры пропущены)."""
        filled = [getattr(self, name) for name in IMAGE_NAMES]
        return tuple(path for path in filled if path)

    @property
    def has_refs(self) -> bool:
        return bool(self.images)

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> "SourceParams":
        values = {p.name: p.default for p in CUBE.params}
        values.update({k: v for k, v in (params or {}).items() if k in values})
        filled = {name: str(values[name] or "").strip() for name in IMAGE_NAMES}
        used = [index for index, name in enumerate(IMAGE_NAMES, start=1) if filled[name]]
        # нумерация без дыр: Image3 без Image2 сдвинул бы <imageN> в промте
        missing = [n for n in range(1, (max(used) if used else 0) + 1) if n not in used]
        if missing:
            names = ", ".join(f"Image{index}" for index in missing)
            raise ValueError(f"source.txtFileImage: пропущены {names} — нумерация обязана быть "
                             f"без дыр (иначе <imageN> в промте не совпадёт с файлами)")
        return cls(prompt=str(values["prompt"] or "").strip(), **filled)


def prompt_md5_of(text: str) -> str:
    """Ключ промта: те же 12 hex, что в имени `.pt` (см. `config.prompt_md5`)."""
    import hashlib

    return hashlib.md5(text.encode()).hexdigest()[:12]


def _resolve(raw: str, base_dir: Path | None) -> Path:
    path = Path(raw)
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path


def _prompt_path(params: SourceParams, base_dir: Path | None) -> Path:
    if not params.prompt:
        raise ValueError("source.txtFileImage: не задан prompt (путь к .txt относительно "
                         "каталога воркфлоу)")
    path = _resolve(params.prompt, base_dir)
    if not path.suffix:
        path = path.with_suffix(".txt")
    if not path.is_file():
        raise FileNotFoundError(f"source.txtFileImage: нет файла промта {path.as_posix()}")
    return path


def _image_paths(params: SourceParams, base_dir: Path | None) -> tuple[Path, ...]:
    paths: list[Path] = []
    for raw in params.images:
        path = _resolve(raw, base_dir)
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"source.txtFileImage: {path.as_posix()} — не картинка "
                             f"({', '.join(IMAGE_SUFFIXES)})")
        if not path.is_file():
            raise FileNotFoundError(f"source.txtFileImage: нет референса {path.as_posix()}")
        paths.append(path)
    return tuple(paths)


def read(params: SourceParams, *, base_dir: Path | None = None,
         log: Any = None) -> tuple[Prompt, RefImages | None]:
    """Прочитать промт и референсы: `Prompt` (текст + md5) и `RefImages` (пути + sha) или `None`.

    Картинки не открываются: на какой сетке их увидит энкодер, решает `source.frame`; здесь только
    пути, `sha` для ключа кэша и порядок, который задал автор воркфлоу.
    """
    say = log or (lambda _msg: None)
    path = _prompt_path(params, base_dir)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"source.txtFileImage: пустой файл промта {path.as_posix()}")
    prompt = Prompt(text=text, md5=prompt_md5_of(text), origin=path.as_posix())
    say(f"[source] промт {path.as_posix()} ({len(text)} симв.)")
    paths = _image_paths(params, base_dir)
    if not paths:
        say("[source] референсов нет (Image1…Image10 пусты)")
        return prompt, None
    refs = RefImages(images=paths, grid=(0, 0), sha=images_sha(paths))
    say(f"[source] референсы: {len(paths)} шт. ({refs.sha})")
    return prompt, refs


def _self_test(ctx: CubeContext) -> None:
    """Проверка без весов: пути от каталога воркфлоу, порядок Image1…N, дырки, ключи."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "refs").mkdir()
        (root / "prompt-a.txt").write_text("fusal style.\n[TASK: IMAGE_EDIT]\n", encoding="utf-8")
        for name in ("a.png", "b.png", "c.webp"):
            (root / "refs" / name).write_bytes(b"stub")
        params = SourceParams.from_mapping({
            "prompt": "prompt-a", "Image1": "refs/a.png", "Image2": "refs/b.png",
            "Image3": "refs/c.webp"})
        prompt, refs = read(params, base_dir=root, log=ctx.log)
        assert prompt.text.startswith("fusal style."), prompt.text
        assert len(prompt.md5) == 12, prompt.md5
        assert refs is not None, "референсы обязаны найтись"
        assert [Path(p).name for p in refs.images] == ["a.png", "b.png", "c.webp"], refs.images
        assert len(refs.sha) >= 12, refs.sha
        # без Image* -> без референсов, промт читается
        only_prompt = SourceParams.from_mapping({"prompt": "prompt-a.txt"})
        _, none = read(only_prompt, base_dir=root, log=ctx.log)
        assert none is None, none
        # дырка в нумерации отвергается
        try:
            SourceParams.from_mapping({"prompt": "prompt-a.txt", "Image2": "refs/a.png"})
        except ValueError as exc:
            assert "Image1" in str(exc), exc   # имя пропущенного в тексте ошибки
        else:                                                    # pragma: no cover
            raise AssertionError("Image2 без Image1 обязан отвергаться")
        # пустой prompt отвергается
        try:
            read(SourceParams.from_mapping({"Image1": "refs/a.png"}), base_dir=root)
        except ValueError:
            pass
        else:                                                    # pragma: no cover
            raise AssertionError("без prompt обязан быть ValueError")
        # не-картинка отвергается
        try:
            read(SourceParams.from_mapping({"prompt": "prompt-a.txt",
                                            "Image1": "prompt-a.txt"}), base_dir=root)
        except ValueError:
            pass
        else:                                                    # pragma: no cover
            raise AssertionError("txt в Image1 обязан отвергаться")


CUBE: CubeSpec = register_cube(CubeSpec(
    id="source.txtFileImage",
    stage=Stage.SOURCE,
    produces=frozenset({"prompt", "refs"}),
    params=(
        ParameterSpec("prompt", "cube", "str", "",
                      doc="файл промта относительно каталога воркфлоу (prompt-qwen.txt; расширение "
                          "можно опустить)"),
        *(ParameterSpec(name, "cube", "str", "",
                        doc=f"референс {index}: путь к картинке относительно каталога воркфлоу; "
                            f"пусто = референса нет, нумерация без дыр, порядок = <image{index}> "
                            f"в промте")
          for index, name in enumerate(IMAGE_NAMES, start=1)),
    ),
    budget=ResourceBudget(notes="только чтение файлов: ни весов, ни VRAM"),
    self_test=_self_test,
    doc="источник входов: промт из .txt + явные референсы Image1…Image10 -> prompt + refs",
))
