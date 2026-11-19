"""Адаптер GGUF/llama.cpp: аргументы сервера, запрос, отказы (без сети и без бинарника).

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/unit/test_gguf_adapter.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (план `pe-int8-gguf`, шаг 2): torchao int8 в этом окружении не даёт fused-ядер, поэтому PE
переводится на llama.cpp. Тест фиксирует всё, что не требует запуска сервера:

1. `build_server_args`: модель, порт, `--n-gpu-layers`, `--ctx-size`, `--jinja`; `--mmproj`
   появляется только при заданном проекторе.
2. `chat_payload`: текстовые сообщения конвертируются в один text-парт, картинки — в `image_url`
   с base64 data-URL (llama.cpp ждёт именно так), сэмплинг едет в теле запроса.
3. `find_server`: без `llama_dir` и без бинарника — понятный `FileNotFoundError` с подсказкой;
   с `llama_dir`, где нет бинарника, — тоже.
4. `GgufServer.start`: отсутствие файла модели и отсутствие mmproj — `FileNotFoundError` до
   запуска процесса.
5. Кубик `rewrite.pe_llamacpp`: контракт (rewrite/prompt→prompt, reads refs), дефолты модели и
   `self_test` без сети.
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

from qwenstyle.cube import CubeContext, Stage, get_cube            # noqa: E402
from qwenstyle.quant.adapters.gguf import (                        # noqa: E402
    GgufParams,
    GgufServer,
    build_server_args,
    chat_payload,
    find_server,
)


class FakeImage:
    """Заглушка PIL: умеет только `save` в BMP-поток (base64 не важен, важна структура)."""

    def save(self, buffer, format: str) -> None:                   # noqa: A002 — как у PIL
        buffer.write(b"\x89PNG\r\n\x1a\n" + b"fake")


def case_server_args() -> None:
    params = GgufParams(model="m.gguf", gguf_dir="dir", ngl=42, ctx=4096, n_batch=256)
    args = build_server_args(params, binary=Path("llama-server"), port=8123)
    assert args[0] == "llama-server", args
    assert args[args.index("--model") + 1].endswith("m.gguf"), args
    assert args[args.index("--port") + 1] == "8123"
    assert args[args.index("--n-gpu-layers") + 1] == "42"
    assert args[args.index("--ctx-size") + 1] == "4096"
    assert args[-1] == "--jinja", args
    assert "--mmproj" not in args, "без проектора аргумента быть не должно"

    with_mmproj = GgufParams(model="m.gguf", mmproj="p.gguf", gguf_dir="dir")
    args2 = build_server_args(with_mmproj, binary=Path("s"), port=1)
    assert args2[args2.index("--mmproj") + 1].endswith("p.gguf"), args2


def case_payload() -> None:
    params = GgufParams(model="m.gguf", temperature=0.7, top_p=0.9, top_k=15, max_tokens=123)
    payload = chat_payload([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": [{"type": "image", "image": FakeImage()},
                                     {"type": "text", "text": "hello"}]},
    ], params)
    assert payload["temperature"] == 0.7 and payload["top_p"] == 0.9, payload
    assert payload["top_k"] == 15 and payload["max_tokens"] == 123 and payload["stream"] is False
    assert payload["messages"][0] == {"role": "system", "content": "sys"}, payload["messages"][0]
    parts = payload["messages"][1]["content"]
    assert parts[0]["type"] == "image_url", parts
    assert parts[0]["image_url"]["url"].startswith("data:image/png;base64,"), parts[0]
    assert parts[1] == {"type": "text", "text": "hello"}, parts[1]


def case_find_server() -> None:
    try:
        find_server(GgufParams(llama_dir=""))
    except FileNotFoundError as exc:
        assert "llama-server" in str(exc) and "llama_dir" in str(exc), exc
    else:
        raise AssertionError("без бинарника find_server обязан падать")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            find_server(GgufParams(llama_dir=tmp))
        except FileNotFoundError as exc:
            assert tmp in str(exc), exc
        else:
            raise AssertionError("пустой llama_dir обязан падать")


def case_start_checks() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        params = GgufParams(model="нет-такого.gguf", gguf_dir=tmp)
        try:
            GgufServer.start(params)
        except FileNotFoundError as exc:
            assert "нет файла модели" in str(exc), exc
        else:
            raise AssertionError("без файла модели старт обязан падать")

        (Path(tmp) / "m.gguf").write_bytes(b"x")
        params = GgufParams(model="m.gguf", mmproj="p.gguf", gguf_dir=tmp)
        try:
            GgufServer.start(params)
        except FileNotFoundError as exc:
            assert "mmproj" in str(exc), exc
        else:
            raise AssertionError("без mmproj vision-старт обязан падать")


def case_cube() -> None:
    from qwenstyle.cubes import rewrite_pe_llamacpp as cube_mod

    cube = get_cube("rewrite.pe_llamacpp")
    assert cube.stage is Stage.REWRITE, cube.stage
    assert cube.consumes == frozenset({"prompt"}) and cube.produces == "prompt", cube
    assert cube.reads == frozenset({"refs"}), "I2I-вариант читает референсы"
    assert cube.param("model").default.endswith(".gguf"), cube.param("model").default
    assert cube.param("mmproj").default is None, "по умолчанию текстовая модель"
    cube.self_test(CubeContext())

    params = cube_mod.LlamacppParams.from_mapping({"mmproj": "p.gguf"})
    assert params.needs_vision is True, params


CASES = [
    ("аргументы llama-server", case_server_args),
    ("тело запроса /v1/chat/completions", case_payload),
    ("поиск бинарника", case_find_server),
    ("проверки файлов при старте", case_start_checks),
    ("контракт кубика rewrite.pe_llamacpp", case_cube),
]


def main() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            fn()
        except Exception as exc:                                  # noqa: BLE001
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
