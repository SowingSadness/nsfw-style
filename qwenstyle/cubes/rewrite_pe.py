"""Кубики промт-энхансера: `rewrite.pe_i2i` (картинка+инструкция) и `rewrite.pe_t2i` (только текст).

Это новая функциональность (план `cube-architecture`, шаг 5). Модель — Qwen3.5-VL 9B
(«Prompt Enhancer»): на вход короткая инструкция (+ референс у I2I), на выход **текст** — плотный
промт для Qwen-Image-2.1. Она не кодирует conditioning для сэмплера, поэтому живёт на стадии
REWRITE и производит порт `prompt`.

Почему два кубика, а не один с флагом: у них разные порты (`consumes={"prompt"}` против
`consumes={"prompt"}` + `reads={"refs"}`), и это видно композитору и UI.

Важно про FUSAL: PE выдаёт свободный описательный абзац в своём формате и **не знает** триггера
`fusal style.` и восьми разделов. Поэтому параметр `wrap`:

* `fusal` — дописать сверху `fusal style.` + `[TASK: …]`, иначе LoRA останется без триггера;
* `none` — отдать как есть (для базовой модели без LoRA).

Режим `wrap="skeleton"` (раскладка по восьми разделам) сознательно не реализован: PE по своему
system prompt выдаёт один непрерывный абзац, и раскидать его по `[Subject]/[Appearance]/…` можно
только додумав содержание — это противоречит правилу «не выдумывать».

Память: 9.41B bf16 = 17.55 ГиБ. В 16 ГиБ VRAM целиком не влезает, поэтому
`device="auto"` использует `device_map="auto"` (accelerate раскидывает слои между VRAM и RAM),
а `device="cpu"` — чистый CPU-резидент (17.6 ГиБ RAM, ~1–3 ток/с).
"""
from __future__ import annotations

import hashlib
import json
import re
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..backends.base import ParameterSpec, ResourceBudget
from ..cube import CubeContext, CubeSpec, Stage, register_cube
from ..ports import Prompt, RefImages

__all__ = [
    "CUBE_I2I",
    "CUBE_T2I",
    "RewriteParams",
    "RewriteResult",
    "apply_wrap",
    "build_messages",
    "cache_key",
    "parse_output",
    "rewrite",
    "system_prompt_text",
]

#: Репозитории heretic-портов PE (I2I — ARA-абляция rockerBOO, отказы 4/100 против 100/100).
REPO_I2I = "rockerBOO/Qwen-Image-2.1-PE-I2I-heretic-ara-seed2-t116"
REPO_T2I = "pottokao/Qwen-Image-2.1-PE-T2I-Heretic"

#: Ключи JSON, которые возвращает PE.
ANSWER_KEYS = ("rewritten_prompt", "wh_ratio", "ratio_follow")


@dataclass(frozen=True)
class RewriteParams:
    """Параметры кубика промт-энхансера (провалидированы по дескрипторам спеки)."""

    repo: str = REPO_I2I
    revision: str | None = None
    system_prompt: str = "system_prompt.txt"
    mode: str = "idea"                     # idea | tighten
    parse_mode: str = "json"                # json (I2I) | text (T2I: один абзац)
    wrap: str = "fusal"                    # none | fusal
    thinking: bool = True
    image_px: int = 896                    # длинная сторона референса для PE (vision-токены)
    max_new_tokens: int = 4096
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    dtype: str = "bfloat16"
    precision: str = "fp8"                 # fp8 = кэш cache/models/fp8_pe_*, bf16 = как есть
    quant_cache: str = "cache/models/fp8_pe_i2i"
    quant_rebuild: bool = False
    device: str = "auto"                   # auto = device_map (VRAM+RAM); cpu = 0.18 ток/с
    vram_gib: float = 10.0                 # потолок VRAM при device=auto (остальное -> RAM)
    seed: int = 0
    cache: bool = True

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> "RewriteParams":
        values = {p.name: p.default for p in CUBE_I2I.params}
        values.update({k: v for k, v in (params or {}).items() if k in values})
        return cls(repo=str(values["repo"]), revision=values["revision"],
                   system_prompt=str(values["system_prompt"]), mode=str(values["mode"]),
                   parse_mode=str(values["parse_mode"]),
                   wrap=str(values["wrap"]), thinking=bool(values["thinking"]),
                   image_px=int(values["image_px"]),
                   max_new_tokens=int(values["max_new_tokens"]),
                   temperature=float(values["temperature"]), top_p=float(values["top_p"]),
                   top_k=int(values["top_k"]), dtype=str(values["dtype"]),
                   precision=str(values["precision"]), quant_cache=str(values["quant_cache"]),
                   quant_rebuild=bool(values["quant_rebuild"]),
                   device=str(values["device"]), vram_gib=float(values["vram_gib"]),
                   seed=int(values["seed"]),
                   cache=bool(values["cache"]))

    def sampling(self) -> dict[str, Any]:
        return {"max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0,
                "temperature": max(self.temperature, 1e-5), "top_p": self.top_p,
                "top_k": self.top_k}


@dataclass(frozen=True)
class RewriteResult:
    """Результат переписывания: готовый промт плюс служебные поля ответа модели."""

    prompt: str
    thinking: str = ""
    wh_ratio: str = ""
    ratio_follow: str = ""
    raw: str = ""

    def as_prompt(self, md5_of: Any = None) -> Prompt:
        md5 = md5_of(self.prompt) if callable(md5_of) else _md5(self.prompt)
        return Prompt(text=self.prompt, md5=md5, origin="rewrite.pe")


# --------------------------------------------------------------------------- чистые части

def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def build_messages(system_prompt: str, instruction: str,
                   images: Sequence[Any] = ()) -> list[dict[str, Any]]:
    """Сообщения в формате Qwen-VL: system + user(картинки, затем текст инструкции)."""
    user: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
    user.append({"type": "text", "text": instruction})
    return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": user}]


def parse_output(raw: str, *, expect_thinking: bool = True,
                 expect_json: bool = True) -> RewriteResult:
    """Разобрать ответ PE: `</think>` отделяет рассуждение, дальше — JSON с промтом.

    Битый JSON — не падение: пробуем вытащить первый `{…}` и, если не вышло, считаем весь ответ
    (после think-блока) готовым промтом — это лучше, чем потерять результат.
    """
    thinking, _, answer = raw.partition("</think>")
    body = answer if answer.strip() else raw
    body = body.strip()
    data: dict[str, Any] | None = None
    for candidate in (body, _first_json_object(body)):
        if not candidate:
            continue
        try:
            loaded = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(loaded, dict) and any(key in loaded for key in ANSWER_KEYS):
            data = loaded
            break
    if data is None:
        # JSON нет. Если ждали thinking, а `</think>` так и не появился — ответ оборван по
        # max_new_tokens, и всё, что есть, это рассуждение: отдавать его как промт нельзя
        # (тихая подмена результата). Без thinking обычный текст — законный ответ.
        truncated = (expect_json and "</think>" not in raw) or                     (expect_thinking and "</think>" not in raw)
        if truncated:
            # Для I2I ответ обязан быть JSON: текст без JSON — это рассуждение, отдавать его как
            # промт нельзя. Для T2I (expect_json=False) абзац и есть ответ.
            return RewriteResult(prompt="", thinking=raw.strip(), raw=raw)
        return RewriteResult(prompt=body, thinking=thinking.strip(), raw=raw)
    return RewriteResult(prompt=str(data.get("rewritten_prompt", "")).strip(),
                         thinking=thinking.strip(),
                         wh_ratio=str(data.get("wh_ratio", "") or ""),
                         ratio_follow=str(data.get("ratio_follow", "") or ""),
                         raw=raw)


def _first_json_object(text: str) -> str:
    """Первый сбалансированный `{…}` (на случай «болтовни» вокруг JSON)."""
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return ""


def apply_wrap(text: str, wrap: str, *, task: str = "IMAGE_EDIT") -> str:
    """Обернуть промт в скелет FUSAL: триггер и `[TASK: …]` сверху (если их ещё нет)."""
    body = text.strip()
    if wrap == "none":
        return body
    lines = body.splitlines()
    has_trigger = bool(lines) and lines[0].strip().lower().startswith("fusal style")
    has_task = any(line.strip().upper().startswith("[TASK:") for line in lines)
    head: list[str] = []
    if not has_trigger:
        head.append("fusal style.")
    if not has_task:
        head.append(f"[TASK: {task}]")
    if not head:
        return body
    return "\n".join(head + ["", body])


def cache_key(params: RewriteParams, instruction: str, refs_sha: str = "", *,
              system_prompt: str = "") -> str:
    """Ключ кэша ответа: инструкция + референсы + модель/ревизия + параметры сэмплинга."""
    parts = [instruction, refs_sha, params.repo, params.revision or "", params.mode,
             params.wrap, str(int(params.thinking)), str(params.max_new_tokens),
             f"{params.temperature:.4f}", f"{params.top_p:.4f}", str(params.top_k),
             _md5(system_prompt)[:8]]
    digest = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"rewrite_{digest}.json"


# --------------------------------------------------------------------------- ввод-вывод

def system_prompt_text(params: RewriteParams, *, repo_root: Path | None = None) -> str:
    """Прочитать `system_prompt.txt` из локального снапшота репозитория (оффлайн)."""
    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(params.repo, revision=params.revision,
                                      allow_patterns=[params.system_prompt, "config.json",
                                                      "tokenizer_config.json"],
                                      local_files_only=True))
    path = snapshot / params.system_prompt
    if not path.is_file():
        raise FileNotFoundError(
            f"{params.repo}: нет {params.system_prompt} в {snapshot} "
            f"(скачайте веса: python -m qwenstyle models download {params.repo} --yes)")
    return path.read_text(encoding="utf-8").strip()


def load(params: RewriteParams) -> tuple[Any, Any]:
    """Загрузить модель и процессор (тяжёлые импорты только здесь, не на импорте модуля).

    `precision="fp8"` (умолчание) — веса берутся из fp8-кэша (`quant_cache`) и живут в VRAM:
    9.41B в 8.8 ГиБ вместо 17.55, замер на 4080 — **4.19 ток/с** против 1.07 у `device_map`
    в bf16 (и 0.18 на CPU). `precision="bf16"` — прежний путь через `device_map`/CPU.
    """
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(params.repo, revision=params.revision,
                                              local_files_only=True)
    if params.precision == "fp8":
        from ..quant.adapters import transformers as quant

        if params.quant_rebuild:
            quant.build_cache(params.repo, params.quant_cache, log=print)
        else:
            quant.build_cache(params.repo, params.quant_cache, log=print)   # внутри reuse по verify
        model, stats = quant.load_quantized(
            lambda: AutoModelForImageTextToText.from_pretrained(
                params.repo, revision=params.revision, dtype=torch.bfloat16,
                device_map="cpu", low_cpu_mem_usage=True, local_files_only=True),
            params.quant_cache, warmup=(),
        )
        print(f"    [rewrite] fp8-веса: {stats['linear_fp8']} Linear из кэша, "
              f"{stats['tensors']} тензоров", flush=True)
        return model, processor
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(params.dtype, torch.bfloat16)
    kwargs: dict[str, Any] = {"dtype": dtype, "local_files_only": True}
    if params.revision:
        kwargs["revision"] = params.revision
    if params.device == "auto":
        # Без потолка accelerate забивает VRAM весами (14.7 ГиБ) и активации на ~2К токенов
        # падают в OOM — замер на 4080: нужно оставить ~4–5 ГиБ свободными.
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: f"{params.vram_gib:.0f}GiB", "cpu": "44GiB"}
    model = AutoModelForImageTextToText.from_pretrained(params.repo, **kwargs)
    model.eval()
    if params.device == "cpu":
        model.to("cpu")
    return model, processor


def _fit_images(images: Sequence[Any], max_side: int) -> list[Any]:
    """Уменьшить референсы до `max_side` по длинной стороне.

    Vision-токены префикса — главная статья активаций: кадр 1728x1152 даёт ~1900 токенов и 2.5 ГиБ
    на attention при 2.6K контекста (OOM на 16 ГиБ), а для переписывания промта хватает 896 px
    (~560 токенов).
    """
    if not images or max_side <= 0:
        return list(images)
    out = []
    for image in images:
        try:
            width, height = image.size
        except Exception:                                     # noqa: BLE001 — не PIL
            out.append(image)
            continue
        longest = max(width, height)
        if longest <= max_side:
            out.append(image)
            continue
        ratio = max_side / float(longest)
        out.append(image.resize((max(1, int(width * ratio)), max(1, int(height * ratio)))))
    return out


def _generate(model: Any, processor: Any, params: RewriteParams, system_prompt: str,
              instruction: str, images: Sequence[Any]) -> str:
    """Один прогон модели: шаблон чата с thinking, сэмплинг, декод только новых токенов."""
    import torch

    images = _fit_images(images, params.image_px)
    messages = build_messages(system_prompt, instruction, images)
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt",
                                           enable_thinking=params.thinking)
    device = getattr(model, "device", None) or "cpu"
    inputs = {key: (value.to(device) if torch.is_tensor(value) else value)
              for key, value in inputs.items()}
    sampling = params.sampling()
    # `generator=` этот класс модели в transformes 5.22 не принимает (валидация kwargs),
    # поэтому сид сэмплинга ставим глобально — ответ всё равно кэшируется по ключу.
    torch.manual_seed(int(params.seed))
    with torch.inference_mode():
        out = model.generate(**inputs, **sampling)
    new_tokens = out[0, inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(new_tokens, skip_special_tokens=True)


def rewrite(params: RewriteParams, instruction: str, *, images: Sequence[Any] = (),
            refs_sha: str = "", ctx: CubeContext | None = None,
            cache_dir: Path | str | None = None,
            model_and_processor: tuple[Any, Any] | None = None) -> RewriteResult:
    """Переписать инструкцию промт-энхансером. Ответ кэшируется по `cache_key()`.

    `model_and_processor` — для повторных вызовов в одном прогоне (модель грузится один раз);
    `cache_dir` — каталог кэша ответов (по умолчанию `cache/embeds`, рядом с conditioning).
    """
    say = (ctx.log if ctx else (lambda _m: None))
    system_prompt = system_prompt_text(params)
    key = cache_key(params, instruction, refs_sha, system_prompt=system_prompt)
    store = Path(cache_dir) if cache_dir else Path("cache") / "embeds"
    target = store / key
    if params.cache and target.is_file():
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
            say(f"[rewrite] ответ из кэша {target.name}")
            return RewriteResult(**{k: data.get(k, "") for k in
                                    ("prompt", "thinking", "wh_ratio", "ratio_follow", "raw")})
        except (OSError, json.JSONDecodeError):
            pass                                          # битый кэш = промах
    model, processor = model_and_processor or load(params)
    started = _time.time()
    raw = _generate(model, processor, params, system_prompt, instruction, images)
    elapsed = _time.time() - started
    result = parse_output(raw, expect_thinking=params.thinking,
                          expect_json=params.parse_mode == "json")
    if not result.prompt.strip():
        approx_tokens = max(1, len(raw) // 4)
        raise RuntimeError(
            f"[rewrite] ответ оборван: JSON не получен за max_new_tokens={params.max_new_tokens}"
            f" (~{approx_tokens} ток, {elapsed:.0f} с, ~{approx_tokens / max(elapsed, 1e-9):.2f} ток/с)."
            f" PE с thinking пишет длинное рассуждение: поднимите max_new_tokens (официально до"
            f" 24000) или запускайте на fp8/GPU — device={params.device}, vram_gib={params.vram_gib:g}; "
            f"raw={raw[-200:]!r}")
    result = RewriteResult(prompt=apply_wrap(result.prompt, params.wrap),
                           thinking=result.thinking, wh_ratio=result.wh_ratio,
                           ratio_follow=result.ratio_follow, raw=result.raw)
    if params.cache:
        store.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps({**{f: getattr(result, f) for f in
                                      ("prompt", "thinking", "wh_ratio", "ratio_follow", "raw")},
                                   "params": {"repo": params.repo, "mode": params.mode,
                                              "wrap": params.wrap, "thinking": params.thinking,
                                              "max_new_tokens": params.max_new_tokens,
                                              "temperature": params.temperature,
                                              "top_p": params.top_p, "top_k": params.top_k,
                                              "instruction": instruction, "refs_sha": refs_sha}},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(target)
        say(f"[rewrite] ответ сохранён {target.name}")
    return result


# --------------------------------------------------------------------------- спеки

def _self_test(ctx: CubeContext) -> None:
    """Без весов: сообщения, парсер ответа, обёртка FUSAL и ключ кэша."""
    messages = build_messages("sys", "make it anime", images=["IMG"])
    assert messages[0]["role"] == "system" and messages[1]["role"] == "user", messages
    assert messages[1]["content"][0]["type"] == "image", messages[1]
    assert messages[1]["content"][-1] == {"type": "text", "text": "make it anime"}

    raw = ('пусть будет так</think>{"rewritten_prompt": "A red-haired fox girl on the Shibuya '
           'crossing at blue hour, painted anime background.", "wh_ratio": "4:3", '
           '"ratio_follow": "<image1>"}')
    parsed = parse_output(raw)
    assert parsed.prompt.startswith("A red-haired fox girl"), parsed
    assert parsed.wh_ratio == "4:3" and parsed.ratio_follow == "<image1>", parsed
    assert "пусть будет так" in parsed.thinking, parsed.thinking

    noisy = 'ok: {"rewritten_prompt": "X", "wh_ratio": "", "ratio_follow": ""} end'
    assert parse_output(noisy).prompt == "X", parse_output(noisy)

    broken = parse_output("</think>просто текст без JSON")
    assert broken.prompt == "просто текст без JSON", broken

    wrapped = apply_wrap("Some description.", "fusal")
    assert wrapped.startswith("fusal style.\n[TASK: IMAGE_EDIT]\n\nSome description."), wrapped
    assert apply_wrap(wrapped, "fusal") == wrapped, "повторная обёртка не должна дублировать триггер"
    assert apply_wrap("plain", "none") == "plain"

    params = RewriteParams.from_mapping({})
    first = cache_key(params, "make it anime", "abc123", system_prompt="sys")
    second = cache_key(params, "make it anime", "abc123", system_prompt="sys")
    assert first == second and first.startswith("rewrite_"), first
    assert cache_key(params, "make it anime", "abc124", system_prompt="sys") != first, \
        "смена референсов обязана менять ключ"
    assert cache_key(params, "make it bolder", "abc123", system_prompt="sys") != first


SPECS_COMMON = (
    ParameterSpec("repo", "cube", "str", REPO_I2I,
                  doc="репозиторий PE (heretic-порт); I2I по умолчанию ARA-абляция rockerBOO"),
    ParameterSpec("revision", "cube", "str", None, doc="ревизия весов (commit)"),
    ParameterSpec("system_prompt", "cube", "str", "system_prompt.txt",
                  doc="файл системного промта внутри репозитория"),
    ParameterSpec("mode", "cube", "enum", "idea", choices=("idea", "tighten"),
                  doc="idea = из короткой инструкции; tighten = наш промт переписать плотнее"),
    ParameterSpec("parse_mode", "cube", "enum", "json", choices=("json", "text"),
                  doc="json = ответ PE-I2I (rewritten_prompt/wh_ratio/ratio_follow); "
                      "text = один абзац (PE-T2I)"),
    ParameterSpec("wrap", "cube", "enum", "fusal", choices=("none", "fusal"),
                  doc="fusal = дописать триггер `fusal style.` и [TASK: …]; none = отдать как есть"),
    ParameterSpec("thinking", "cube", "bool", True,
                  doc="enable_thinking в шаблоне чата: чекпоинт обучен с блоком рассуждения"),
    ParameterSpec("image_px", "cube", "int", 896, minimum=0, maximum=2048,
                  doc="длинная сторона референса для PE: vision-токены префикса — главная статья "
                      "активаций (1728px ~1900 ток / 2.5 ГиБ, 896px ~560 ток); 0 = не менять"),
    ParameterSpec("max_new_tokens", "cube", "int", 4096, minimum=64, maximum=24000,
                  doc="потолок токенов ответа (вместе с рассуждением)"),
    ParameterSpec("temperature", "cube", "float", 1.0, minimum=0.0, maximum=2.0,
                  doc="сэмплинг ответа (официально 1.0); 0 = жадный"),
    ParameterSpec("top_p", "cube", "float", 0.95, minimum=0.0, maximum=1.0, doc="сэмплинг ответа"),
    ParameterSpec("top_k", "cube", "int", 20, minimum=0, maximum=200, doc="сэмплинг ответа"),
    ParameterSpec("dtype", "cube", "enum", "bfloat16", choices=("bfloat16", "float16"),
                  doc="точность весов (9.41B: bf16 = 17.55 ГиБ)"),
    ParameterSpec("precision", "cube", "enum", "fp8", choices=("fp8", "bf16"),
                  doc="fp8 = кэш fp8_pe_* (8.8 ГиБ в VRAM, 4.19 ток/с); bf16 = device_map/CPU"),
    ParameterSpec("quant_cache", "cube", "str", "cache/models/fp8_pe_i2i",
                  doc="каталог fp8-кэша весов PE (пересобирается при смене репозитория/паттернов)"),
    ParameterSpec("quant_rebuild", "cube", "bool", False,
                  doc="пересобрать fp8-кэш весов, даже если он годен"),
    ParameterSpec("device", "cube", "enum", "auto", choices=("auto", "cpu"),
                  doc="auto = device_map по VRAM+RAM (умолчание); cpu = резидентно в RAM "
                      "(17.6 ГиБ, замер 0.18 ток/с — только фоллбек)"),
    ParameterSpec("vram_gib", "cube", "float", 10.0, minimum=2.0, maximum=15.0,
                  doc="потолок VRAM для device=auto (остальные слои уходят в RAM); "
                      "10 ГиБ оставляет место активациям на 2К токенов"),
    ParameterSpec("seed", "cube", "int", 0, doc="сид сэмплинга ответа"),
    ParameterSpec("cache", "cube", "bool", True, doc="кэшировать ответ по ключу инструкции"),
)

_BUDGET = ResourceBudget(weights_bf16_gib=17.55, disk_gib=17.55,
                         notes="Qwen3.5-VL 9B bf16 здесь считается как RAM-резидент")

CUBE_I2I: CubeSpec = register_cube(CubeSpec(
    id="rewrite.pe_i2i",
    stage=Stage.REWRITE,
    consumes=frozenset({"prompt"}),
    reads=frozenset({"refs"}),
    produces="prompt",
    params=SPECS_COMMON,
    budget=_BUDGET,
    self_test=_self_test,
    doc="PE-I2I: картинка + инструкция -> точный промт (heretic-порт, thinking)",
))

def _t2i_spec() -> tuple[ParameterSpec, ...]:
    """Спека T2I: другой репозиторий и `parse_mode=text` (ответ — абзац, а не JSON)."""
    out: list[ParameterSpec] = []
    for spec in SPECS_COMMON:
        if spec.name == "repo":
            out.append(ParameterSpec("repo", "cube", "str", REPO_T2I,
                                     doc="репозиторий PE-T2I (heretic-порт pottokao)"))
        elif spec.name == "parse_mode":
            out.append(ParameterSpec("parse_mode", "cube", "enum", "text", choices=("json", "text"),
                                     doc="text = один абзац (PE-T2I); json = если порт отдаёт JSON"))
        else:
            out.append(spec)
    return tuple(out)


CUBE_T2I: CubeSpec = register_cube(CubeSpec(
    id="rewrite.pe_t2i",
    stage=Stage.REWRITE,
    consumes=frozenset({"prompt"}),
    produces="prompt",
    params=_t2i_spec(),
    budget=_BUDGET,
    self_test=_self_test,
    doc="PE-T2I: текст -> точный промт без референса (heretic-порт, thinking)",
))
