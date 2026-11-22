"""Кубики входов: `source.frame` (геометрия) и `source.txtFileImage` (промт + Image1…N).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/cube/test_source_cubes.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.
Без torch, без CUDA, без сети.

Зачем (план `workflow-layout`, шаг 1): файл пайплайна обязан быть самодостаточным — входы и
геометрия видны в нём, а не выводятся из содержимого каталогов. Тест фиксирует то, на чём легко
разъехаться с прежним поведением:

1. `source.frame`: `height` + авто-ширина == прежний `--size NNNp`; явные W/H главнее референса;
   стороны кратны 32, оба нуля — 1024×1024; `resolution = round(sqrt(W*H))`;
2. `ref_crop_side` даёт **ту же** сетку референсов, что прежний `ref_px` (`RunConfig.embedding_crop`);
3. `source.txtFileImage`: порядок `Image1…ImageN` = порядок в файле, дырка в нумерации — ошибка,
   пустые `Image*` пропускаются, `md5`/`sha` совпадают с `config.prompt_md5`/`io.images_sha`;
4. порты: `source.frame` -> `frame`, `source.txtFileImage` -> `prompt`+`refs`, их потребляют
   энкодер (`prompt`+`frame`, читает `refs`) и трансформер (`conditioning`+`frame`).
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

from qwenstyle.config import RunConfig, prompt_md5                    # noqa: E402
from qwenstyle.cube import CubeContext, Stage, get_cube                # noqa: E402
from qwenstyle.cubes import source_frame, source_txtimage              # noqa: E402
from qwenstyle.io import images_sha                                    # noqa: E402

PROMPT_TEXT = "fusal style.\n[TASK: IMAGE_EDIT]\n\n[Subject] one young woman\n"


def case_frame_auto_side() -> None:
    """`height` + авто-ширина воспроизводит прежний `--size` на пропорции референса."""
    params = source_frame.FrameParams.from_mapping({"height": 768})
    assert source_frame.canvas(params, (864, 480)) == (1376, 768)
    # 4:3 и 1:1
    assert source_frame.canvas(params, (1728, 1296)) == (1024, 768), \
        source_frame.canvas(params, (1728, 1296))
    assert source_frame.canvas(params, (600, 600)) == (768, 768)
    # явные обе стороны главнее референса
    explicit = source_frame.FrameParams.from_mapping({"width": 2912, "height": 2176})
    assert source_frame.canvas(explicit, (864, 480)) == (2912, 2176)
    # T2I без геометрии — квадрат
    assert source_frame.canvas(source_frame.FrameParams.from_mapping({}), (0, 0)) == (1024, 1024)


def case_frame_rules() -> None:
    for bad in ({"width": 100}, {"height": 300}, {"width": 33}):
        try:
            source_frame.canvas(source_frame.FrameParams.from_mapping(bad))
        except ValueError as exc:
            assert "32" in str(exc), exc
        else:                                                       # pragma: no cover
            raise AssertionError(f"{bad} обязан отвергаться")
    explicit = source_frame.FrameParams.from_mapping({"width": 2912, "height": 2176})
    assert source_frame.resolution(explicit, (2912, 2176)) == 2517
    frame = source_frame.frame_of(explicit, (1728, 1152))
    assert (frame.width, frame.height, frame.resolution) == (2912, 2176, 2517), frame
    assert frame.port == "frame", frame.port


def case_ref_crop_side_matches_ref_px() -> None:
    """Сетка референсов обязана совпасть с прежним `RunConfig.embedding_crop`.

    Прежний `ref_px` — это **высота** сетки; `ref_crop_side` держит ту же семантику, поэтому числа
    переносятся 1:1 и conditioning не меняется (иначе слетел бы кэш эмбеддингов).
    """
    cases = (
        ("VRAM16-4k-slide", 2912, 2176),
        ("VRAM16-large", 3200, 1792),
        ("VRAM16-r3-169", 1632, 896),
    )
    for preset, width, height in cases:
        cfg = RunConfig.from_profile(preset)
        ref_px = int((cfg.generation.sampler_kwargs or {}).get("ref_px") or 0)
        if not ref_px:
            continue
        expected = cfg.embedding_crop((width, height))
        params = source_frame.FrameParams.from_mapping({"width": width, "height": height,
                                                       "ref_crop_side": ref_px})
        got = source_frame.embedding_grid(params, (width, height))
        assert got == expected, f"{preset}: {got} != {expected} (ref_px={ref_px})"
    # 0 = сетка канваса
    assert source_frame.embedding_grid(source_frame.FrameParams.from_mapping({}),
                                       (2912, 2176)) == (2912, 2176)


def case_txtimage_order_and_keys() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "refs").mkdir()
        (root / "prompt-a.txt").write_text(PROMPT_TEXT, encoding="utf-8")
        for name in ("a.png", "b.png", "c.webp"):
            (root / "refs" / name).write_bytes(name.encode())
        params = source_txtimage.SourceParams.from_mapping({
            "prompt": "prompt-a", "Image1": "refs/b.png", "Image2": "refs/a.png",
            "Image3": "refs/c.webp"})
        prompt, refs = source_txtimage.read(params, base_dir=root)
        assert prompt.text == PROMPT_TEXT.strip(), prompt.text
        # ключ промта — как у config.load_prompt (текст после strip)
        assert prompt.md5 == prompt_md5(PROMPT_TEXT.strip()), prompt.md5
        assert refs is not None, "референсы обязаны найтись"
        names = [Path(p).name for p in refs.images]
        assert names == ["b.png", "a.png", "c.webp"], names      # порядок из файла, не sort
        assert refs.sha == images_sha([root / "refs" / n for n in names]), refs.sha


def case_txtimage_errors() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "prompt-a.txt").write_text(PROMPT_TEXT, encoding="utf-8")
        (root / "refs").mkdir()
        (root / "refs" / "a.png").write_bytes(b"a")
        # дырка в нумерации
        try:
            source_txtimage.SourceParams.from_mapping({"prompt": "prompt-a.txt",
                                                       "Image2": "refs/a.png"})
        except ValueError as exc:
            assert "Image1" in str(exc), exc
        else:                                                       # pragma: no cover
            raise AssertionError("Image2 без Image1 обязан отвергаться")
        # нет файла промта
        try:
            source_txtimage.read(source_txtimage.SourceParams.from_mapping(
                {"prompt": "nope.txt"}), base_dir=root)
        except FileNotFoundError:
            pass
        else:                                                       # pragma: no cover
            raise AssertionError("нет промта — обязана быть ошибка")
        # не картинка
        try:
            source_txtimage.read(source_txtimage.SourceParams.from_mapping(
                {"prompt": "prompt-a.txt", "Image1": "prompt-a.txt"}), base_dir=root)
        except ValueError:
            pass
        else:                                                       # pragma: no cover
            raise AssertionError("txt в Image1 обязан отвергаться")
        # без Image* — без референсов
        _, refs = source_txtimage.read(source_txtimage.SourceParams.from_mapping(
            {"prompt": "prompt-a.txt"}), base_dir=root)
        assert refs is None, refs


def case_contracts() -> None:
    frame = get_cube("source.frame")
    assert frame.stage is Stage.SOURCE, frame.stage
    assert frame.outputs == frozenset({"frame"}), frame.outputs
    assert [p.name for p in frame.params] == ["width", "height", "ref_crop_side"]
    assert frame.self_test is not None
    frame.self_test(CubeContext())

    source = get_cube("source.txtFileImage")
    assert source.stage is Stage.SOURCE, source.stage
    assert source.outputs == frozenset({"prompt", "refs"}), source.outputs
    names = [p.name for p in source.params]
    assert names[0] == "prompt" and names[1:] == list(source_txtimage.IMAGE_NAMES), names
    assert len(names) == source_txtimage.MAX_IMAGES + 1, len(names)
    assert source.self_test is not None
    source.self_test(CubeContext())

    # энкодер ест prompt+frame и читает refs; трансформер ест conditioning+frame
    encoder = get_cube("encoder.qwen3vl")
    assert encoder.consumes == frozenset({"prompt", "frame"}), encoder.consumes
    assert encoder.reads == frozenset({"refs"}), encoder.reads
    transformer = get_cube("transformer.qwen_image21")
    assert transformer.consumes == frozenset({"conditioning", "frame"}), transformer.consumes


CASES = [
    ("frame: авто-сторона", case_frame_auto_side),
    ("frame: правила и resolution", case_frame_rules),
    ("ref_crop_side == прежний ref_px", case_ref_crop_side_matches_ref_px),
    ("txtFileImage: порядок и ключи", case_txtimage_order_and_keys),
    ("txtFileImage: ошибки", case_txtimage_errors),
    ("контракты кубиков", case_contracts),
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
