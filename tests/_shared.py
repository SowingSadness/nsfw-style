"""Общее для тестов: где лежат промт и куда тесты пишут результаты (`out/tests`).

Зачем: `prompts/` и `out/` — пользовательские каталоги (оба в `.gitignore`), поэтому тест не имеет
права ни читать промт оттуда, ни писать кадры в корень `out/`:

* **промт тестов** — `tests/texts/exp1.txt` (копия `prompts/exp1.txt`, эталонный FUSAL-промт из
  AGENTS.md §5): лежит в репозитории рядом с тестом, меняется только вместе с тестом;
* **результаты тестов** — только `out/tests/` (кадры, входные заглушки, логи прогонов). Корень
  `out/` принадлежит пользователю: тест, который пишет туда, затирает чужие кадры.

Файл называется `_shared` (ведущее `_`), чтобы его не приняли за тест: тесты запускаются как
скрипты (`python tests/test_x.py`), поэтому `import _shared` работает без установки пакета —
`sys.path[0]` в этом режиме и есть каталог `tests/`.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

#: Корень репозитория (каталог над `tests/`).
REPO_ROOT = Path(__file__).resolve().parents[1]
#: Каталог тестов.
TESTS_DIR = REPO_ROOT / "tests"
#: Промт для тестов: копия `prompts/exp1.txt`, живёт в репозитории.
TEST_PROMPT = TESTS_DIR / "texts" / "exp1.txt"
#: Куда тесты пишут результаты генерации. Корень `out/` — только для пользователя.
OUT_TESTS = REPO_ROOT / "out" / "tests"
#: Интерпретатор тестов (тот же venv, что запустил тест).
PYTHON = sys.executable


def out_tests(*parts: str) -> Path:
    """Путь внутри `out/tests/` (каталог создаётся): `out_tests("smoke")` -> `.../out/tests/smoke`."""
    path = OUT_TESTS.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def cli_env(**extra: str) -> dict[str, str]:
    """Окружение прогона: utf-8 в консоли (cp1251 ломает кириллицу) + оффлайн-Хаб как в проде."""
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.update(extra)
    return env


def run_cli(*args: str, timeout: float = 900.0) -> subprocess.CompletedProcess[str]:
    """`python -m qwenstyle <args>` из корня репозитория (так же, как это делает человек)."""
    return subprocess.run(
        [str(PYTHON), "-m", "qwenstyle", *map(str, args)],
        cwd=str(REPO_ROOT), env=cli_env(), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout,
    )


def has_cuda() -> bool:
    """Есть ли CUDA: тесты реальной генерации без неё печатают `SKIP` и не падают."""
    try:
        import torch
    except Exception:
        return False
    return bool(torch.cuda.is_available())


def make_input_image(path: Path, width: int = 864, height: int = 480) -> Path:
    """Детерминированная входная картинка теста (без файлов пользователя и без сети).

    Простой цветовой градиент: sha256 файла стабилен между прогонами, значит ключ кэша
    эмбеддингов тоже, и повторный запуск теста не платит энкод заново.
    """
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (x * 255 // max(1, width - 1),
                            y * 255 // max(1, height - 1),
                            128)
    if not path.is_file():
        image.save(path, format="PNG")
    return path


__all__ = ["OUT_TESTS", "PYTHON", "REPO_ROOT", "TESTS_DIR", "TEST_PROMPT", "cli_env", "has_cuda",
           "make_input_image", "out_tests", "run_cli"]
