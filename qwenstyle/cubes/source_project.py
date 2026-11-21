"""Кубик `source.project`: проект промта -> `prompt` + `refs` (план `cube-architecture`, шаг 6).

Зачем: до этого промт и референсы собирались в `cmd/gen.py` и `cmd/_project.py` до всякой
композиции, поэтому «откуда взялся кадр» знали два места, а композитор не видел входов вообще.
Теперь это стадия SOURCE с двумя выходами: она одна читает каталог проекта (`*.txt` + `refs*/`).

    params = source_project.SourceParams.from_mapping({"dir": "prompts/nita-camp-model",
                                                      "prompt": "prompt-tokyo-anime.txt"})
    prompt, refs = source_project.read(params)      # Prompt(md5) + RefImages(sha) или None

Семантика путей сохранена 1:1 с `--prompt`/`--project`:

* `dir` — каталог проекта; пусто = корень `prompts/` (тогда нужен `prompt`);
* `prompt` — имя файла промта (`prompt-qwen.txt`) или стем (`prompt-qwen`); пусто при непустом
  `dir` = единственный `*.txt` в каталоге (ошибка со списком, если их несколько);
* `refs` — подкаталог с референсами (`refs`, `refs-style`, …); `-` (или пусто) = без референсов;
  порядок = natural sort (`io._natural_key`), поэтому `<image1>` в промте всегда первый файл;
* `aspect`/`size_px` — целевой кадр (нужны композитору и энкодеру, сам кубик ничего не режет).

Кэш: `md5` промта и `sha` референсов считаются здесь же, чтобы энкодер (`encoder.qwen3vl`) и
политика вытеснения получали готовые ключи, а не пересчитывали их повторно.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

from ..backends.base import ParameterSpec, ResourceBudget
from ..cube import CubeContext, CubeSpec, Stage, register_cube
from ..io import discover_images, images_sha
from ..ports import Prompt, RefImages

__all__ = ["CUBE", "IMAGE_SUFFIXES", "SourceParams", "discover_images", "prompt_md5_of",
           "read"]

#: Расширения референсов (как в `cmd/_project.py`: набор проекта, не рекурсивно).
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
#: Значение `refs`, означающее «без референсов».
NO_REFS = ("", "-", "none")
#: Пропорции кадра; `auto` — брать пропорцию референса (как `--aspect` без значения).
ASPECTS = ("auto", "16:9", "4:3", "1:1", "2:3")


@dataclass(frozen=True)
class SourceParams:
    """Разобранные параметры источника (пути — как их задал пользователь)."""

    dir: str = ""
    prompt: str = ""
    refs: str = "refs"
    aspect: str = "auto"
    size_px: int = 1024
    seed: int = 0

    @property
    def project_dir(self) -> Path:
        """Каталог проекта (`prompts/<имя>`), либо текущий каталог при пустом `dir`."""
        return Path(self.dir) if self.dir else Path(".")

    @property
    def has_refs(self) -> bool:
        return self.refs.strip().lower() not in NO_REFS

    @property
    def refs_dir(self) -> Path:
        return self.project_dir / self.refs

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> "SourceParams":
        """Собрать из «словаря параметров кубика»; дефолты — из `CUBE.params`."""
        values = {p.name: p.default for p in CUBE.params}
        values.update({k: v for k, v in (params or {}).items() if k in values})
        return cls(dir=str(values["dir"] or ""), prompt=str(values["prompt"] or ""),
                   refs=str(values["refs"] if values["refs"] is not None else "refs"),
                   aspect=str(values["aspect"]), size_px=int(values["size_px"]),
                   seed=int(values["seed"]))


def prompt_md5_of(text: str) -> str:
    """Ключ промта: те же 12 hex, что в имени `.pt` (см. `config.prompt_md5`)."""
    import hashlib

    return hashlib.md5(text.encode()).hexdigest()[:12]


def _prompt_file(params: SourceParams) -> Path:
    """Файл промта: явное имя/стем, иначе единственный `*.txt` каталога проекта."""
    folder = params.project_dir
    name = params.prompt.strip()
    if name:
        path = folder / (name if name.lower().endswith(".txt") else f"{name}.txt")
        if not path.is_file():
            raise FileNotFoundError(f"нет файла промта {path.as_posix()}")
        return path
    candidates = sorted((p for p in folder.glob("*.txt") if p.is_file()),
                        key=lambda p: p.name)
    if len(candidates) == 1:
        return candidates[0]
    names = ", ".join(p.name for p in candidates) or "—"
    raise FileNotFoundError(
        f"в {folder.as_posix()} {len(candidates)} файл(ов) промта ({names}); "
        f"укажите prompt=<имя> в параметрах кубика")


def read(params: SourceParams, *, log: Any = None) -> tuple[Prompt, RefImages | None]:
    """Прочитать проект: `Prompt` (текст + md5) и `RefImages` (пути + sha) или `None`.

    Картинки не открываются: кубик не знает, на какой сетке их увидит энкодер, — он отдаёт пути и
    `sha`. Открытие/`fit_cover` делает `encoder.qwen3vl` (у референсов своя сетка `ref_px`).
    """
    say = log or (lambda _msg: None)
    path = _prompt_file(params)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"пустой файл промта {path.as_posix()}")
    prompt = Prompt(text=text, md5=prompt_md5_of(text), origin=path.as_posix())
    say(f"[source] промт {path.as_posix()} ({len(text)} симв.)")
    if not params.has_refs:
        say("[source] референсов нет (refs=-)")
        return prompt, None
    paths = discover_images(params.refs_dir, suffixes=IMAGE_SUFFIXES)
    if not paths:
        say(f"[source] каталог {params.refs_dir.as_posix()} без картинок — референсов нет")
        return prompt, None
    refs = RefImages(images=tuple(paths), grid=(0, 0), sha=images_sha(paths))
    say(f"[source] референсы {params.refs_dir.as_posix()}: {len(paths)} шт. ({refs.sha})")
    return prompt, refs


def _self_test(ctx: CubeContext) -> None:
    """Проверка без весов: чтение промта, natural-порядок референсов, ключи."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "refs").mkdir()
        (root / "prompt-test.txt").write_text("fusal style.\n[TASK: TEXT_TO_IMAGE]\n",
                                             encoding="utf-8")
        for name in ("10-b.png", "2-a.png", "1-c.webp"):
            (root / "refs" / name).write_bytes(b"stub")
        params = SourceParams.from_mapping({"dir": str(root), "refs": "refs"})
        prompt, refs = read(params, log=ctx.log)
        assert prompt.text.startswith("fusal style."), prompt.text
        assert len(prompt.md5) == 12 and int(prompt.md5, 16) >= 0, prompt.md5
        assert refs is not None, "референсы обязаны найтись"
        names = [Path(p).name for p in refs.images]
        assert names == ["1-c.webp", "2-a.png", "10-b.png"], names      # natural, не лексикограф
        assert len(refs.sha) >= 12, refs.sha
        # пустой `refs` -> референсов нет, промт всё равно читается
        no_refs = SourceParams.from_mapping({"dir": str(root), "refs": "-"})
        _, none_refs = read(no_refs, log=ctx.log)
        assert none_refs is None, none_refs
        # явное имя промта без расширения тоже работает
        named = SourceParams.from_mapping({"dir": str(root), "prompt": "prompt-test",
                                           "refs": "refs"})
        assert read(named, log=ctx.log)[0].md5 == prompt.md5, "стем и файл дали разные ключи"


CUBE: CubeSpec = register_cube(CubeSpec(
    id="source.project",
    stage=Stage.SOURCE,
    produces=frozenset({"prompt", "refs"}),
    params=(
        ParameterSpec("dir", "cube", "str", "",
                      doc="каталог проекта (например prompts/nita-camp-model); пусто = корень prompts/"),
        ParameterSpec("prompt", "cube", "str", "",
                      doc="файл промта или стем (prompt-qwen.txt / prompt-qwen); пусто = "
                          "единственный *.txt в каталоге проекта"),
        ParameterSpec("refs", "cube", "str", "refs",
                      doc="подкаталог референсов (refs, refs-style, …); '-' или пусто = без "
                          "референсов, порядок = natural sort файлов"),
        ParameterSpec("aspect", "cube", "enum", "auto", choices=ASPECTS,
                      doc="пропорция кадра: auto = как у референса; 16:9/4:3/1:1/2:3 = "
                          "fit_cover на неё (нужна T2I-моделям без референса)"),
        ParameterSpec("size_px", "cube", "int", 1024, minimum=256, maximum=4096,
                      doc="высота кадра в пикселях (480/768/960/1088/1440/2176)"),
        ParameterSpec("seed", "cube", "int", 0,
                      doc="сид батча — для трассировки; сэмплер получает его в CubeContext"),
    ),
    budget=ResourceBudget(notes="только чтение файлов: ни весов, ни VRAM"),
    self_test=_self_test,
    doc="источник проекта промта: каталог с *.txt и refs/ -> prompt + refs (natural sort, sha)",
))
