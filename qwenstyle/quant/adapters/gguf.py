"""Адаптер GGUF/llama.cpp: генеративные текстовые модели через внешний `llama-server`.

Зачем (план `pe-int8-gguf`, шаг 2): torchao int8 в этом окружении не даёт fused-CUDA-ядер
(`Skipping import of cpp extensions due to incompatible torch version 2.8.0+cu129`), поэтому
промт-энхансер на torch считает 1.87–4.19 ток/с. В llama.cpp квантованный GEMM слит с ядрами,
поэтому там те же 9.4B в Q4_K_M идут на порядок быстрее.

Границы применимости (важно): адаптер работает только с **генеративными** текстовыми моделями —
`rewrite.pe_t2i`, `rewrite.pe_i2i` (через `--mmproj`). Для кондишн-энкодера он не годится:
пайплайну нужны `prompt_embeds` + `img_mask` с конкретного слоя Qwen3-VL, а llama.cpp отдаёт
только текст (или эмбеддинги последнего слоя без vision-разметки). Для DiT ядер у llama.cpp нет.

Как устроено: `GgufServer` поднимает `llama-server` на свободном порту, ждёт `/health`, отдаёт
запросы через OpenAI-совместимый `/v1/chat/completions` и гасит процесс в `finally`. Разбор ответа
(think-блок, JSON, обёртка FUSAL) **переиспользует** `cubes/rewrite_pe.py`, чтобы формат промта
жил в одном месте.
"""
from __future__ import annotations

import base64
import io
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

__all__ = [
    "DEFAULT_GGUF_DIR",
    "GgufParams",
    "GgufServer",
    "build_server_args",
    "chat_payload",
    "find_server",
    "generate",
]

DEFAULT_GGUF_DIR = "cache/models/gguf"
SERVER_NAME = "llama-server.exe" if os.name == "nt" else "llama-server"


@dataclass(frozen=True)
class GgufParams:
    """Параметры запуска GGUF-модели (всё, что нужно `llama-server` и запросу)."""

    model: str = ""                     # имя файла .gguf в `gguf_dir`
    mmproj: str | None = None           # файл проектора для vision (PE-I2I)
    gguf_dir: str = DEFAULT_GGUF_DIR
    llama_dir: str = ""                 # каталог с llama-server; пусто = искать в PATH
    ngl: int = 99                       # слоёв на GPU
    ctx: int = 8192
    n_batch: int = 512
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 4096
    timeout: float = 900.0              # ожидание ответа (сек)
    threads: int = 0                    # 0 = по умолчанию llama.cpp
    log_dir: str = "cache/logs"         # логи llama-server (не в out/: корень out — пользователя)

    # -- пути -------------------------------------------------------------- #
    @property
    def model_path(self) -> Path:
        # Абсолютный путь обязателен: llama-server запускается с cwd = gguf_dir, иначе
        # относительное имя разрешилось бы дважды (`gguf_dir/gguf_dir/model`).
        return (Path(self.gguf_dir) / self.model).resolve()

    @property
    def mmproj_path(self) -> Path | None:
        return (Path(self.gguf_dir) / self.mmproj).resolve() if self.mmproj else None

    def sampling(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "top_p": self.top_p, "top_k": self.top_k,
                "max_tokens": self.max_tokens, "stream": False}


@dataclass
class GgufServer:
    """Поднятый `llama-server`: `url` для запросов и `close()` для остановки."""

    params: GgufParams
    port: int
    url: str
    process: Any = None
    log_path: Path | None = None
    _log_handle: Any = field(default=None, repr=False)

    # -- жизненный цикл ---------------------------------------------------- #
    @classmethod
    def start(cls, params: GgufParams, *, log: Any = None) -> "GgufServer":
        say = log or (lambda _m: None)
        model = params.model_path
        if not model.is_file():
            raise FileNotFoundError(
                f"[gguf] нет файла модели {model}; положите .gguf в {params.gguf_dir} "
                f"или укажите `model` с полным именем файла")
        mmproj = params.mmproj_path
        if mmproj is not None and not mmproj.is_file():
            raise FileNotFoundError(f"[gguf] нет mmproj {mmproj} (нужен для vision-входа)")
        binary = find_server(params)          # бинарник проверяем последним: файлы важнее
        port = _free_port()
        args = build_server_args(params, binary=binary, port=port)
        log_dir = Path(params.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"gguf-{port}.log"
        handle = log_path.open("wb")
        say(f"[gguf] {binary.name} :{port} {model.name}"
            f"{' + ' + mmproj.name if mmproj else ''} -> {log_path.as_posix()}")
        process = subprocess.Popen(args, stdout=handle, stderr=subprocess.STDOUT,
                                   cwd=str(Path(params.gguf_dir).resolve()))
        server = cls(params=params, port=port, url=f"http://127.0.0.1:{port}",
                     process=process, log_path=log_path, _log_handle=handle)
        try:
            server.wait_ready()
        except Exception:
            server.close()
            raise
        return server

    def wait_ready(self, *, timeout: float = 180.0) -> None:
        """Ждать `/health` (llama.cpp отвечает 200, когда модель загружена)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"[gguf] llama-server завершился с кодом {self.process.returncode}; "
                    f"лог: {self.log_path}")
            try:
                with urllib.request.urlopen(f"{self.url}/health", timeout=2) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
            time.sleep(0.5)
        raise TimeoutError(f"[gguf] llama-server не поднялся за {timeout} с; лог: {self.log_path}")

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:            # pragma: no cover — упрямый процесс
                self.process.kill()
        if self._log_handle is not None:
            try:
                self._log_handle.close()
            except OSError:                              # pragma: no cover
                pass

    def __enter__(self) -> "GgufServer":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # -- запрос ------------------------------------------------------------ #
    def chat(self, messages: Sequence[dict[str, Any]], params: GgufParams | None = None) -> str:
        params = params or self.params
        body = chat_payload(messages, params)
        request = urllib.request.Request(f"{self.url}/v1/chat/completions",
                                         data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=params.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:                # noqa: PERF203 — тело в ошибке
            detail = exc.read().decode("utf-8", "replace")[:400]
            raise RuntimeError(f"[gguf] llama-server ответил {exc.code}: {detail}") from None
        try:
            choice = payload["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"[gguf] неожиданный ответ сервера: {str(payload)[:400]}") from None
        content = str(message.get("content") or "")
        if content.strip():
            return content
        # Пустой `content` при непустом `reasoning_content` = модель остановилась внутри
        # рассуждения (кончился max_tokens). Это не ответ, и молча подставлять reasoning нельзя.
        reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
        raise RuntimeError(
            f"[gguf] ответ пуст (finish_reason={choice.get('finish_reason')!r}, "
            f"reasoning {len(reasoning)} симв.): модель не дошла до финального текста — "
            f"увеличьте max_new_tokens (сейчас {params.max_tokens}) или ctx")


# --------------------------------------------------------------------------- утилиты

def find_server(params: GgufParams) -> Path:
    """Найти `llama-server`: сначала `llama_dir`, потом PATH."""
    if params.llama_dir:
        candidate = Path(params.llama_dir) / SERVER_NAME
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(
            f"[gguf] в {params.llama_dir} нет {SERVER_NAME}; укажите каталог с бинарником")
    from shutil import which

    found = which(SERVER_NAME) or which("llama-server")
    if found:
        return Path(found)
    raise FileNotFoundError(
        f"[gguf] не найден {SERVER_NAME}: положите его в PATH или укажите параметр `llama_dir` "
        f"(например C:/Users/<user>/.docker/bin/inference)")


def build_server_args(params: GgufParams, *, binary: Path, port: int) -> list[str]:
    """Аргументы командной строки `llama-server` (вынесено отдельно — тестируется без процесса)."""
    args = [str(binary), "--model", str(params.model_path), "--host", "127.0.0.1",
            "--port", str(port), "--n-gpu-layers", str(params.ngl), "--ctx-size", str(params.ctx),
            "--batch-size", str(params.n_batch), "--jinja"]
    if params.mmproj_path is not None:
        args += ["--mmproj", str(params.mmproj_path)]
    if params.threads:
        args += ["--threads", str(params.threads)]
    return args


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _image_data_url(image: Any) -> str:
    """PIL-картинка -> data-URL (llama.cpp принимает base64 в поле `image_url`)."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def chat_payload(messages: Sequence[dict[str, Any]], params: GgufParams) -> dict[str, Any]:
    """Тело запроса к `/v1/chat/completions` (вынесено отдельно — тестируется без сервера)."""
    converted: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            converted.append({"role": message["role"], "content": content})
            continue
        parts: list[dict[str, Any]] = []
        for item in content or ():
            if item.get("type") == "image":
                parts.append({"type": "image_url",
                              "image_url": {"url": _image_data_url(item["image"])}})
            else:
                parts.append({"type": "text", "text": str(item.get("text", ""))})
        converted.append({"role": message["role"], "content": parts})
    return {"messages": converted, **params.sampling()}


def generate(params: GgufParams, messages: Sequence[dict[str, Any]], *, log: Any = None) -> str:
    """Один запрос к модели «под ключ»: поднять сервер, спросить, погасить."""
    with GgufServer.start(params, log=log) as server:
        return server.chat(messages, params)
