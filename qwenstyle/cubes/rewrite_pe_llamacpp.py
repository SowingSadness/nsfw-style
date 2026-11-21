"""Кубик `rewrite.pe_llamacpp`: промт-энхансер через llama.cpp/GGUF (план `pe-int8-gguf`, шаг 2).

Зачем: torchao int8 в этом окружении не даёт fused-ядер (замер шага 1: 1.87 ток/с против 4.19 у
нашего fp8), поэтому быстрый PE возможен только через llama.cpp, где квантованный GEMM слит с
ядрами. Кубик — тонкая обёртка: собирает сообщения (общий код `rewrite_pe`), поднимает
`llama-server` из `quant/adapters/gguf.py`, разбирает ответ и оборачивает его в скелет FUSAL.

Ограничение по входу: `mmproj` (vision) нужен только для I2I. Если модели без проектора, кубик
работает как T2I (см. `requires_refs`).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..backends.base import ParameterSpec, ResourceBudget
from ..cube import CubeContext, CubeSpec, Stage, register_cube
from ..ports import Prompt, RefImages
from .rewrite_pe import (RewriteParams, RewriteResult, apply_wrap, build_messages, cache_key,
                         parse_output, system_prompt_text)

__all__ = ["CUBE", "LlamacppParams", "rewrite"]

#: Репозитории с heretic-портами PE в GGUF (проверены по HF: файлы и размеры существуют).
GGUF_I2I_REPO = "darrellbest/Qwen-Image-2.1-PE-I2I-Heretic-GGUF"
GGUF_T2I_REPO = "pottokao/Qwen-Image-2.1-PE-T2I-Heretic-GGUF"


@dataclass(frozen=True)
class LlamacppParams:
    """Параметры кубика: что за GGUF поднять и как спросить."""

    model: str = "pe_t2i_heretic-Q4_K_M.gguf"
    mmproj: str | None = None
    gguf_dir: str = "cache/models/gguf"
    llama_dir: str = ""
    ngl: int = 99
    ctx: int = 8192
    n_batch: int = 512
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    max_new_tokens: int = 4096
    timeout: float = 900.0
    system_prompt_repo: str = "pottokao/Qwen-Image-2.1-PE-T2I-Heretic"
    system_prompt: str = "system_prompt.txt"
    mode: str = "idea"
    parse_mode: str = "text"
    wrap: str = "fusal"
    image_px: int = 896
    cache: bool = True

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> "LlamacppParams":
        values = {p.name: p.default for p in CUBE.params}
        values.update({k: v for k, v in (params or {}).items() if k in values})
        return cls(**{name: values[name] for name in cls.__dataclass_fields__})

    @property
    def needs_vision(self) -> bool:
        return bool(self.mmproj)


def _gguf_params(params: LlamacppParams) -> Any:
    from ..quant.adapters.gguf import GgufParams

    return GgufParams(model=params.model, mmproj=params.mmproj, gguf_dir=params.gguf_dir,
                      llama_dir=params.llama_dir, ngl=params.ngl, ctx=params.ctx,
                      n_batch=params.n_batch, temperature=params.temperature, top_p=params.top_p,
                      top_k=params.top_k, max_tokens=params.max_new_tokens, timeout=params.timeout)


def rewrite(params: LlamacppParams, instruction: str, *, images: Sequence[Any] = (),
            refs_sha: str = "", ctx: CubeContext | None = None,
            cache_dir: str | None = None) -> RewriteResult:
    """Спросить llama.cpp и вернуть разобранный ответ (кэш ответов — общий с torch-путём)."""
    from ..quant.adapters.gguf import GgufServer

    say = (ctx.log if ctx else (lambda _m: None))
    system_prompt = system_prompt_text(
        RewriteParams(repo=params.system_prompt_repo, system_prompt=params.system_prompt))
    key = cache_key(RewriteParams(repo=params.system_prompt_repo, mode=params.mode,
                                  wrap=params.wrap, thinking=False,
                                  max_new_tokens=params.max_new_tokens,
                                  temperature=params.temperature, top_p=params.top_p,
                                  top_k=params.top_k),
                    instruction, refs_sha, system_prompt=system_prompt)
    store = __import__("pathlib").Path(cache_dir or "cache/embeds")
    target = store / key
    if params.cache and target.is_file():
        try:
            import json

            data = json.loads(target.read_text(encoding="utf-8"))
            say(f"[rewrite] ответ из кэша {target.name}")
            return RewriteResult(**{k: data.get(k, "") for k in
                                    ("prompt", "thinking", "wh_ratio", "ratio_follow", "raw")})
        except Exception:                                        # noqa: BLE001 — битый кэш
            pass

    images = _fit(images, params.image_px)
    messages = build_messages(system_prompt, instruction, images)
    started = time.time()
    with GgufServer.start(_gguf_params(params), log=say) as server:
        raw = server.chat(messages)
    elapsed = time.time() - started
    # JSON пробуем всегда: и T2I-, и I2I-чекпоинты отвечают `{"rewritten_prompt": …}`, а различие
    # `parse_mode` нужно лишь как фоллбек на случай чисто текстового ответа.
    result = parse_output(raw, expect_thinking=False, expect_json=True)
    if not result.prompt.strip() and params.parse_mode == "text":
        result = parse_output(raw, expect_thinking=False, expect_json=False)
    body = result.prompt.strip()
    if not body:
        raise RuntimeError(f"[rewrite] llama.cpp не вернул промт за {params.max_new_tokens} ток "
                           f"({elapsed:.0f} с); raw={raw[-200:]!r}")
    result = RewriteResult(prompt=apply_wrap(body, params.wrap), thinking=result.thinking,
                           wh_ratio=result.wh_ratio, ratio_follow=result.ratio_follow, raw=raw)
    if params.cache:
        import json

        store.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps({**{f: getattr(result, f) for f in
                                      ("prompt", "thinking", "wh_ratio", "ratio_follow", "raw")},
                                   "params": {"backend": "llama.cpp", "model": params.model,
                                              "instruction": instruction, "refs_sha": refs_sha}},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(target)
        say(f"[rewrite] ответ сохранён {target.name}")
    return result


def _fit(images: Sequence[Any], max_side: int) -> list[Any]:
    from .rewrite_pe import _fit_images

    return _fit_images(images, max_side)


def _self_test(ctx: CubeContext) -> None:
    """Без сети и без бинарника: параметры, аргументы сервера и разбор ответа."""
    params = LlamacppParams.from_mapping({})
    assert params.model.endswith(".gguf"), params.model
    assert not params.needs_vision, "по умолчанию T2I-модель без mmproj"

    from ..quant.adapters.gguf import GgufParams, build_server_args, chat_payload

    gguf = _gguf_params(params)
    args = build_server_args(gguf, binary=__import__("pathlib").Path("llama-server"), port=1234)
    assert "--model" in args and "--n-gpu-layers" in args and args[-1] == "--jinja", args
    assert "--mmproj" not in args, "без mmproj аргумента быть не должно"

    payload = chat_payload([{"role": "user", "content": [
        {"type": "text", "text": "hello"}]}], gguf)
    assert payload["messages"][0]["content"] == [{"type": "text", "text": "hello"}], payload
    assert payload["stream"] is False and payload["max_tokens"] == 4096, payload

    parsed = parse_output('{"rewritten_prompt": "P", "wh_ratio": "4:3", "ratio_follow": ""}',
                          expect_thinking=False, expect_json=True)
    assert parsed.prompt == "P", parsed
    wrapped = apply_wrap("P", "fusal")
    assert wrapped.startswith("fusal style."), wrapped


CUBE: CubeSpec = register_cube(CubeSpec(
    id="rewrite.pe_llamacpp",
    stage=Stage.REWRITE,
    consumes=frozenset({"prompt"}),
    reads=frozenset({"refs"}),
    produces="prompt",
    params=(
        ParameterSpec("model", "cube", "str", "pe_t2i_heretic-Q4_K_M.gguf",
                      doc=f"файл GGUF в gguf_dir; T2I heretic: {GGUF_T2I_REPO}, "
                          f"I2I heretic: {GGUF_I2I_REPO} (+ mmproj)"),
        ParameterSpec("mmproj", "cube", "str", None,
                      doc="файл vision-проектора для PE-I2I (например "
                          "Qwen-Image-2.1-PE-I2I-Heretic-mmproj-F16.gguf); пусто = только текст"),
        ParameterSpec("gguf_dir", "cube", "str", "cache/models/gguf",
                      doc="каталог с .gguf весами"),
        ParameterSpec("llama_dir", "cube", "str", "",
                      doc="каталог с llama-server; пусто = искать в PATH"),
        ParameterSpec("ngl", "cube", "int", 99, minimum=0, maximum=99,
                      doc="слоёв на GPU (99 = все)"),
        ParameterSpec("ctx", "cube", "int", 8192, minimum=2048, maximum=32768,
                      doc="размер контекста llama.cpp (префикс + ответ)"),
        ParameterSpec("n_batch", "cube", "int", 512, minimum=64, maximum=4096,
                      doc="размер батча префилла"),
        ParameterSpec("temperature", "cube", "float", 1.0, minimum=0.0, maximum=2.0,
                      doc="сэмплинг ответа (официально 1.0)"),
        ParameterSpec("top_p", "cube", "float", 0.95, minimum=0.0, maximum=1.0, doc="сэмплинг"),
        ParameterSpec("top_k", "cube", "int", 20, minimum=0, maximum=200, doc="сэмплинг"),
        ParameterSpec("max_new_tokens", "cube", "int", 4096, minimum=64, maximum=24000,
                      doc="потолок токенов ответа"),
        ParameterSpec("timeout", "cube", "float", 900.0, minimum=10.0, maximum=7200.0,
                      doc="таймаут ожидания ответа, секунды"),
        ParameterSpec("system_prompt_repo", "cube", "str", "pottokao/Qwen-Image-2.1-PE-T2I-Heretic",
                      doc="репозиторий, откуда берётся system_prompt.txt"),
        ParameterSpec("system_prompt", "cube", "str", "system_prompt.txt",
                      doc="файл системного промта внутри репозитория"),
        ParameterSpec("mode", "cube", "enum", "idea", choices=("idea", "tighten"),
                      doc="idea = из короткой инструкции; tighten = переписать плотнее"),
        ParameterSpec("parse_mode", "cube", "enum", "text", choices=("text", "json"),
                      doc="text = ответ абзацем (PE-T2I); json = ждать JSON (PE-I2I)"),
        ParameterSpec("wrap", "cube", "enum", "fusal", choices=("none", "fusal"),
                      doc="fusal = дописать триггер `fusal style.` и [TASK: …]"),
        ParameterSpec("image_px", "cube", "int", 896, minimum=0, maximum=2048,
                      doc="длинная сторона референса (vision-токены префикса)"),
        ParameterSpec("cache", "cube", "bool", True, doc="кэшировать ответ по ключу инструкции"),
    ),
    budget=ResourceBudget(weights_bf16_gib=5.5, disk_gib=5.5,
                          notes="Q4_K_M GGUF: 5.5 ГиБ и на диске, и в VRAM (плюс mmproj 0.86); "
                                "bf16-копии в нашей памяти нет — её держит llama-server"),
    self_test=_self_test,
    doc="PE через llama.cpp/GGUF: Q4_K_M в VRAM, fused-ядра, ответ секунды вместо минут",
))
