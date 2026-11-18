"""Кубики `rewrite.pe_i2i` / `rewrite.pe_t2i`: парсер, обёртка FUSAL, ключ кэша, композитор.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/unit/test_rewrite_pe.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (план `cube-architecture`, шаг 5): PE — отдельная стадия (вход текст(+картинка), выход текст),
а не энкодер для сэмплера. Тест фиксирует всё, что можно проверить **без весов**:

1. Сообщения: system + user(картинки, затем инструкция) — формат Qwen-VL.
2. Парсер: `</think>` отделяет рассуждение, JSON даёт `rewritten_prompt`/`wh_ratio`/`ratio_follow`.
3. Устойчивость: JSON посреди текста, битый JSON -> весь ответ как промт.
4. `wrap=fusal`: триггер + `[TASK: …]`, повторная обёртка не дублирует; `wrap=none` — как есть.
5. Ключ кэша: стабилен, меняется от инструкции, референсов, `mode`, `wrap` и параметров сэмплинга.
6. Контракт: `pe_i2i` читает refs, `pe_t2i` — нет; оба производят `prompt`; `repo` отличается.
7. Композитор: `pe_i2i -> encoder` валиден (refs остаются доступны энкодеру); `pe_t2i` в цепочке
   без референсов тоже валиден.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True

from qwenstyle.backends.base import ParameterSpec                     # noqa: E402
from qwenstyle.compose import CubeBinding, PipelineSpec, compose       # noqa: E402
from qwenstyle.cube import CubeContext, CubeSpec, Stage, register_cube, reset_registry  # noqa: E402
from qwenstyle.cubes import rewrite_pe                                # noqa: E402

CTX = CubeContext()


def case_messages() -> None:
    messages = rewrite_pe.build_messages("SYS", "make it anime", images=["IMG1", "IMG2"])
    assert [m["role"] for m in messages] == ["system", "user"], messages
    content = messages[1]["content"]
    assert [c["type"] for c in content] == ["image", "image", "text"], content
    assert content[0]["image"] == "IMG1" and content[-1]["text"] == "make it anime", content


def case_parse_standard() -> None:
    raw = ('рассуждение…</think>{"rewritten_prompt": "Long anime prompt", "wh_ratio": "4:3", '
           '"ratio_follow": "<image1>"}')
    result = rewrite_pe.parse_output(raw)
    assert result.prompt == "Long anime prompt", result
    assert result.wh_ratio == "4:3" and result.ratio_follow == "<image1>", result
    assert result.thinking == "рассуждение…", result.thinking


def case_parse_noisy_and_broken() -> None:
    noisy = 'вот ответ: {"rewritten_prompt": "X", "wh_ratio": "", "ratio_follow": ""} конец'
    assert rewrite_pe.parse_output(noisy).prompt == "X", rewrite_pe.parse_output(noisy)
    broken = rewrite_pe.parse_output("</think>текст без JSON")
    assert broken.prompt == "текст без JSON", broken
    # обрезанный JSON: в JSON-режиме это промах (промт пуст), в текстовом — текст как есть
    cut = '{"rewritten_prompt": "не закрыт'
    assert rewrite_pe.parse_output(cut, expect_thinking=False, expect_json=True).prompt == ""
    assert rewrite_pe.parse_output(cut, expect_thinking=False, expect_json=False).prompt == cut


def case_truncated() -> None:
    """Обрыв по max_new_tokens: для JSON-режима пустой промт, для текстового — сам текст."""
    cut = "рассуждение без терминатора"
    assert rewrite_pe.parse_output(cut).prompt == "", "оборванный think нельзя отдавать как промт"
    assert rewrite_pe.parse_output(cut, expect_json=True).prompt == ""
    assert rewrite_pe.parse_output(cut, expect_thinking=False, expect_json=False).prompt == cut


def case_wrap() -> None:
    wrapped = rewrite_pe.apply_wrap("Some description.", "fusal")
    assert wrapped == "fusal style.\n[TASK: IMAGE_EDIT]\n\nSome description.", wrapped
    assert rewrite_pe.apply_wrap(wrapped, "fusal") == wrapped, "триггер не дублируется"
    assert rewrite_pe.apply_wrap("fusal style.\n[TASK: X]\nbody", "fusal").startswith("fusal style.")
    assert rewrite_pe.apply_wrap("plain", "none") == "plain"
    assert rewrite_pe.apply_wrap("plain", "fusal", task="TEXT_TO_IMAGE").startswith(
        "fusal style.\n[TASK: TEXT_TO_IMAGE]")


def case_cache_key() -> None:
    params = rewrite_pe.RewriteParams.from_mapping({})
    base = rewrite_pe.cache_key(params, "make it anime", "sha1", system_prompt="sys")
    assert base == rewrite_pe.cache_key(params, "make it anime", "sha1", system_prompt="sys")
    assert base.startswith("rewrite_") and base.endswith(".json"), base
    variants = [
        rewrite_pe.cache_key(params, "make it bolder", "sha1", system_prompt="sys"),
        rewrite_pe.cache_key(params, "make it anime", "sha2", system_prompt="sys"),
        rewrite_pe.cache_key(rewrite_pe.RewriteParams.from_mapping({"thinking": False}),
                             "make it anime", "sha1", system_prompt="sys"),
        rewrite_pe.cache_key(rewrite_pe.RewriteParams.from_mapping({"max_new_tokens": 512}),
                             "make it anime", "sha1", system_prompt="sys"),
        rewrite_pe.cache_key(params, "make it anime", "sha1", system_prompt="другой система"),
    ]
    assert len({base, *variants}) == len(variants) + 1, "ключ обязан различать параметры"


def case_contract() -> None:
    i2i = rewrite_pe.CUBE_I2I
    t2i = rewrite_pe.CUBE_T2I
    assert i2i.stage is Stage.REWRITE and t2i.stage is Stage.REWRITE
    assert i2i.consumes == frozenset({"prompt"}) and i2i.reads == frozenset({"refs"}), i2i
    assert t2i.consumes == frozenset({"prompt"}) and not t2i.reads, t2i
    assert i2i.produces == t2i.produces == "prompt"
    assert i2i.param("repo").default == rewrite_pe.REPO_I2I
    assert t2i.param("repo").default == rewrite_pe.REPO_T2I
    assert {p.name for p in i2i.params} == {p.name for p in t2i.params}
    assert i2i.param("parse_mode").default == "json", "I2I отвечает JSON"
    assert t2i.param("parse_mode").default == "text", "T2I отвечает абзацем"
    i2i.self_test(CTX)
    t2i.self_test(CTX)


def case_composer() -> None:
    """PE перед энкодером: refs остаются доступны, цепочка валидна."""
    reset_registry()
    register_cube(rewrite_pe.CUBE_I2I)
    register_cube(rewrite_pe.CUBE_T2I)
    register_cube(CubeSpec(id="encoder.toy", stage=Stage.ENCODER,
                           consumes=frozenset({"prompt"}), reads=frozenset({"refs"}),
                           produces="conditioning"))
    register_cube(CubeSpec(id="transformer.toy", stage=Stage.TRANSFORMER,
                           consumes=frozenset({"conditioning"}), produces="latents"))
    register_cube(CubeSpec(id="decoder.toy", stage=Stage.DECODER,
                           consumes=frozenset({"latents"}), produces="pixels"))

    with_i2i = compose(PipelineSpec("i2i", (
        CubeBinding("rewrite.pe_i2i", {"mode": "idea"}),
        CubeBinding("encoder.toy"), CubeBinding("transformer.toy"), CubeBinding("decoder.toy"),
    ), frozenset({"prompt", "refs"})))
    assert with_i2i.ok, with_i2i.format_issues()

    # T2I-цепочка: энкодер без референсов на входе
    register_cube(CubeSpec(id="encoder.plain", stage=Stage.ENCODER,
                           consumes=frozenset({"prompt"}), produces="conditioning"))
    with_t2i = compose(PipelineSpec("t2i", (
        CubeBinding("rewrite.pe_t2i"), CubeBinding("encoder.plain"),
        CubeBinding("transformer.toy"), CubeBinding("decoder.toy"),
    ), frozenset({"prompt"})))
    assert with_t2i.ok, with_t2i.format_issues()

    # pe_i2i без референсов на входе — ошибка композиции, а не рантайм
    without_refs = compose(PipelineSpec("i2i", (
        CubeBinding("rewrite.pe_i2i"), CubeBinding("encoder.toy"),
        CubeBinding("transformer.toy"), CubeBinding("decoder.toy"),
    ), frozenset({"prompt"})))
    assert "port_missing" in [issue.code for issue in without_refs.issues], \
        [issue.code for issue in without_refs.issues]


CASES = [
    ("сообщения Qwen-VL", case_messages),
    ("парсер: think + JSON", case_parse_standard),
    ("парсер: шум и битый JSON", case_parse_noisy_and_broken),
    ("обрыв по max_new_tokens", case_truncated),
    ("обёртка fusal", case_wrap),
    ("ключ кэша", case_cache_key),
    ("контракт pe_i2i/pe_t2i", case_contract),
    ("композитор: PE перед энкодером", case_composer),
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
