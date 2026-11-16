"""Preflight входа: T2I без `--input` разрешён, модели с обязательным референсом падают до весов.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_t2i_preflight.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (план `minimax-h3-image`, шаг 1): `--input` был обязателен безусловно
(`cmd/run.py::_check_files`), а `pipeline._plan` сразу делал `Image.open(cfg.input)`. Появилась
модель, которая умеет **и** T2I, **и** edit (MiniMax-H3-Image), поэтому решение «нужен ли
референс» переехало в `Capabilities.requires_image` и проверяется одной чистой функцией
`pipeline.input_problem(cfg, caps)` — до открытия картинки и до torch/transformers.

Кейсы:

(а) `requires_image=True` без входа -> текст «модель <backend> требует референс … --input».
(б) `requires_image=False` без входа: `--aspect 1:1`/`size_px=1024` -> `None` и crop 1024x1024;
    `--aspect 16:9` -> `None`, ширина > высоты, обе кратны 16.
(в) `requires_image=False`, ни входа, ни `--aspect` -> причина про `--aspect` (пропорцию взять негде).
(г) вход задан, файла нет -> причина «нет входного кадра … --input».
(д) вход есть: `None`, и crop совпадает с `cfg.crop(*Image.open(...).size)` при `resolution 1024`
    (сверка с текущим поведением — ключ кэша у существующих прогонов не должен поехать).
(е) `io.image_sha("") == NOREF_SHA` и ключ эмбеддингов T2I содержит `noref` (записи T2I и записи
    с конкретным кадром не смешиваются).

Ничего не грузит: ни весов, ни сети, ни GPU; файлов не пишет (кадр для (д) — из `prompts/imgs/`).
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                   # тест не должен оставлять __pycache__ в пакете

from qwenstyle import io, pipeline                # noqa: E402 — после правки sys.path
from qwenstyle.config import RunConfig, prompt_md5  # noqa: E402
from qwenstyle.registry import get_generation     # noqa: E402

#: Вход с реальным файлом (2867x1612): пропорция 1.779 -> при `--resolution 1024` сетка 1376x768.
INPUT_IMAGE = REPO_ROOT / "prompts" / "imgs" / "ire0nita-store.png"
PROMPT = "fusal style. [TASK: TEXT_IMAGE] test prompt for preflight"
#: `qwen_image21` умеет T2I (`requires_image=False`), поэтому «референс обязателен» — это
#: заглушка того же бекенда: ветка проверяется на свойстве caps, а не на имени модели.
QWEN_CAPS = get_generation("qwen_image21").caps
REQUIRES_IMAGE_CAPS = replace(QWEN_CAPS, requires_image=True)
T2I_CAPS = replace(QWEN_CAPS, requires_image=False)


# --------------------------------------------------------------------------- #
# Кейсы
# --------------------------------------------------------------------------- #

def case_required_image_absent() -> None:
    """(а) модель с обязательным референсом без `--input` — понятная причина, не None."""
    cfg = RunConfig.from_profile("base", prompt=PROMPT, input="")
    reason = pipeline.input_problem(cfg, REQUIRES_IMAGE_CAPS)
    assert reason, "requires_image=True без --input должен давать причину"
    assert "требует референс" in reason, f"в тексте нет 'требует референс': {reason!r}"
    assert "--input" in reason, f"в тексте нет подсказки --input: {reason!r}"
    assert "qwen_image21" in reason, f"в тексте нет имени бекенда: {reason!r}"


def case_t2i_square() -> None:
    """(б) T2I без входа: `--aspect 1:1` даёт квадрат 1024x1024 при `size_px=1024`."""
    cfg = RunConfig.from_profile("base", prompt=PROMPT, input="", size_px=1024,
                                 aspect="1:1")
    assert pipeline.input_problem(cfg, T2I_CAPS) is None, "T2I с --aspect обязан быть валиден"
    crop = cfg.crop(0, 0)
    assert crop == (1024, 1024), f"crop 1:1 при size_px=1024: ожидалось (1024, 1024), дано {crop}"


def case_t2i_wide() -> None:
    """(б, продолжение) T2I без входа и `--aspect 16:9`: шире, чем выше, и кратно 16."""
    cfg = RunConfig.from_profile("base", prompt=PROMPT, input="", size_px=1024,
                                 aspect="16:9")
    assert pipeline.input_problem(cfg, T2I_CAPS) is None, "T2I с --aspect 16:9 обязан быть валиден"
    width, height = cfg.crop(0, 0)
    assert width > height, f"16:9 должен быть шире: {width}x{height}"
    assert width % 16 == 0 and height % 16 == 0, f"сетка не кратна 16: {width}x{height}"
    assert height <= 1024, f"высота выше size_px: {width}x{height}"


def case_t2i_without_aspect() -> None:
    """(в) T2I без входа и без `--aspect` — взять пропорцию неоткуда, причина про `--aspect`."""
    cfg = RunConfig.from_profile("base", prompt=PROMPT, input="", aspect=None)
    reason = pipeline.input_problem(cfg, T2I_CAPS)
    assert reason, "без --input и без --aspect причина обязана быть"
    assert "--aspect" in reason, f"причина должна называть --aspect: {reason!r}"
    # конфиг при этом не падает: dry-run должен уметь напечатать план до вердикта
    assert cfg.crop(0, 0)[0] % 16 == 0, "frame_ratio(0, 0) обязан давать кратную 16 сетку"


def case_missing_file() -> None:
    """(г) `--input` задан, файла нет — причина до открытия картинки."""
    missing = "prompts/imgs/__no_such_frame__.png"
    cfg = RunConfig.from_profile("base", prompt=PROMPT, input=missing)
    reason = pipeline.input_problem(cfg, REQUIRES_IMAGE_CAPS)
    assert reason, "несуществующий --input обязан давать причину"
    assert missing in reason, f"причина должна называть путь: {reason!r}"
    assert "--input" in reason, f"причина должна называть флаг: {reason!r}"


def case_existing_input_keeps_crop() -> None:
    """(д) рабочий вход: `None`, и crop — ровно как `cfg.crop(*Image.open(...).size)`."""
    from PIL import Image

    assert INPUT_IMAGE.is_file(), f"нет эталонного входа {INPUT_IMAGE}"
    cfg = RunConfig.from_profile("base", prompt=PROMPT, input=str(INPUT_IMAGE),
                                 resolution=1024, size_px=None)
    assert pipeline.input_problem(cfg, REQUIRES_IMAGE_CAPS) is None, "существующий вход валиден"
    assert pipeline.input_problem(cfg, T2I_CAPS) is None, "T2I-бекенд с входом тоже валиден"
    in_size = Image.open(INPUT_IMAGE).size
    assert cfg.crop(*in_size) == (1376, 768), \
        f"crop 2867x1612 при --resolution 1024 поехал: {cfg.crop(*in_size)}"
    assert cfg.frame_ratio(*in_size) == in_size[0] / in_size[1], "пропорция входа искажена"


def case_noref_key() -> None:
    """(е) `image_sha("")` = sentinel, а ключ кэша T2I содержит `noref`."""
    assert io.image_sha("") == io.NOREF_SHA == "noref", f"image_sha('') = {io.image_sha('')!r}"
    assert io.image_sha("   ") == "noref", "пробельный путь тоже считается «без референса»"
    cfg = RunConfig.from_profile("base", prompt=PROMPT, input="", size_px=1024,
                                 aspect="1:1")
    key = cfg.embeds_key(cfg.crop(0, 0), prompt_md5(PROMPT), io.image_sha(cfg.input))
    assert "noref" in key, f"в ключе кэша нет noref: {key}"
    parts = cfg.cache_key_parts(cfg.crop(0, 0), prompt_md5(PROMPT), io.image_sha(cfg.input))
    assert parts["image_sha"] == "noref", f"cache_key_parts.image_sha = {parts['image_sha']!r}"


CASES = (
    ("а: requires_image без --input", case_required_image_absent),
    ("б: T2I --aspect 1:1", case_t2i_square),
    ("б: T2I --aspect 16:9", case_t2i_wide),
    ("в: T2I без --aspect", case_t2i_without_aspect),
    ("г: --input без файла", case_missing_file),
    ("д: вход есть, crop как раньше", case_existing_input_keeps_crop),
    ("е: noref в ключе кэша", case_noref_key),
)


def main() -> int:
    failed = 0
    for title, func in CASES:
        try:
            func()
        except Exception as exc:                    # noqa: BLE001 — тест печатает всё сам
            failed += 1
            print(f"FAIL {title}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        else:
            print(f"PASS {title}", flush=True)
    if failed:
        print(f"FAIL: {failed} из {len(CASES)}", flush=True)
        return 1
    print("ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
