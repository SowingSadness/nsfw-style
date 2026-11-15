"""Тесты диск-стриминга весов (`qwenstyle/streaming.py`) — обычный скрипт, без pytest.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_streaming.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Кейсы (по плану шага 1):

1. модель, собранная `build_streamed` на meta, даёт **бит-в-бит** те же `hidden_states`, что обычная
   загрузка через `from_pretrained` (сравнение `torch.equal`, а не `allclose`), в fp32 и bf16,
   на одном и на двух шардах, и повторный forward (веса перечитываются с диска) не расходится;
2. после сборки все параметры на `meta`, ни один буфер не на meta; `tie_word_embeddings=True`
   отвергается `ValueError` до сборки;
3. `resolve_stream_source` на каталоге без весов -> `RuntimeError` с подсказкой и командой
   `snapshot_download`; недостающие в шарде имена -> `KeyError` со списком (и у модели, и у карты).

Модель крошечная (`Qwen3ForCausalLM`, hidden 64, 2 слоя, vocab 128), поэтому тест идёт секунды и не
трогает веса проекта. Временные каталоги — только в системном temp (`TemporaryDirectory`).
"""
from __future__ import annotations

import gc
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:              # запуск из любого cwd
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                  # тест не должен оставлять __pycache__ в пакете

from qwenstyle.streaming import (               # noqa: E402 — после правки sys.path
    ShardedWeights,
    build_streamed,
    load_text_encoder,
    resolve_stream_source,
)

TINY = dict(                                    # минимальный Qwen3, чтобы тест шёл секунды
    hidden_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    intermediate_size=128,
    vocab_size=128,
    tie_word_embeddings=False,
)
DTYPES = ("float32", "bfloat16")


# --------------------------------------------------------------------------- #
# Фикстуры
# --------------------------------------------------------------------------- #

def tiny_config(tie_word_embeddings: bool = False):
    """`Qwen3Config` из `TINY` (+ переопределение tie)."""
    from transformers import Qwen3Config

    return Qwen3Config(**{**TINY, "tie_word_embeddings": tie_word_embeddings})


def build_tiny(root: Path) -> None:
    """Случайная мини-модель -> `root` (`config.json` + `model.safetensors`)."""
    import torch
    from transformers import Qwen3ForCausalLM

    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(tiny_config()).eval()
    model.save_pretrained(str(root))
    del model


def shard_copy(src: Path, dst: Path, shards: int = 2) -> None:
    """Разложить `src` на `shards` шардов + `*.index.json` (проверка weight_map-ветки)."""
    from safetensors.torch import load_file, save_file

    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "config.json", dst / "config.json")
    state = load_file(str(src / "model.safetensors"))
    names = list(state)
    weight_map: dict[str, str] = {}
    for index in range(shards):
        chunk = names[index::shards]                  # перемешиваем имена между шардами
        filename = f"model-{index + 1:05d}-of-{shards:05d}.safetensors"
        save_file({name: state[name] for name in chunk}, str(dst / filename))
        weight_map.update({name: filename for name in chunk})
    payload = {"metadata": {"format": "pt"}, "weight_map": weight_map}
    (dst / "model.safetensors.index.json").write_text(json.dumps(payload), encoding="utf-8")


def _streamed_weights(model: Any) -> ShardedWeights | None:
    """Карта шардов, которую `build_streamed` держит на модели (чтобы закрыть файлы)."""
    return getattr(model, "_stream_weights", None)


def _close(model: Any) -> None:
    """Закрыть mmap-файлы модели и отпустить её (Windows не даёт удалить открытый файл)."""
    weights = _streamed_weights(model)
    if weights is not None:
        weights.close()
    del model
    gc.collect()


def temp_base() -> Path:
    """База для `TemporaryDirectory`: системный temp, если он вне репозитория.

    На этой машине запись в `%TEMP%` недоступна (песочница), и stdlib `tempfile.gettempdir()`
    отдаёт cwd — то есть `TemporaryDirectory()` без `dir=` создал бы каталог в корне репозитория.
    Поэтому базу выбираем явно: системный temp, а если он совпал с репозиторием — `tests/`
    (каталог всё равно удаляется, в корне после теста ничего не остаётся).
    """
    candidate = Path(tempfile.gettempdir())
    try:
        resolved = candidate.resolve()
    except OSError:                             # pragma: no cover — недоступный путь
        resolved = candidate
    if resolved != REPO_ROOT and REPO_ROOT not in resolved.parents:
        return resolved
    return REPO_ROOT / "tests"


# --------------------------------------------------------------------------- #
# Кейс 1: математика не меняется
# --------------------------------------------------------------------------- #

def case_forward_matches(tmp: Path) -> None:
    """`torch.equal` по каждому hidden state: обычная загрузка vs стриминг, fp32/bf16, 1 и 2 шарда."""
    import torch
    from transformers import Qwen3ForCausalLM

    root = tmp / "case1-source"
    build_tiny(root)
    multi = tmp / "case1-multi"
    shard_copy(root, multi, shards=2)

    torch.manual_seed(123)
    input_ids = torch.randint(0, TINY["vocab_size"], (1, 7))
    for dtype in DTYPES:
        for label, weights_dir in (("single", root), ("multi", multi)):
            plain = load_text_encoder(Qwen3ForCausalLM, weights_dir, dtype=dtype,
                                      local_files_only=True).eval()
            streamed = load_text_encoder(Qwen3ForCausalLM, weights_dir, dtype=dtype,
                                         local_files_only=True, stream=True)
            try:
                with torch.no_grad():
                    ref = plain(input_ids=input_ids, output_hidden_states=True)
                    got = streamed(input_ids=input_ids, output_hidden_states=True)
                    again = streamed(input_ids=input_ids, output_hidden_states=True)
                if len(ref.hidden_states) != len(got.hidden_states):
                    raise AssertionError(f"{dtype}/{label}: слоёв {len(got.hidden_states)}, "
                                         f"ожидалось {len(ref.hidden_states)}")
                for layer, (expected, actual) in enumerate(zip(ref.hidden_states, got.hidden_states)):
                    if not torch.equal(expected, actual):
                        diff = (expected.float() - actual.float()).abs().max().item()
                        raise AssertionError(f"{dtype}/{label}: hidden_states[{layer}] не "
                                             f"бит-в-бит (max abs diff {diff})")
                if not torch.equal(ref.logits, got.logits):
                    diff = (ref.logits.float() - got.logits.float()).abs().max().item()
                    raise AssertionError(f"{dtype}/{label}: logits не бит-в-бит "
                                         f"(max abs diff {diff})")
                for layer, (expected, actual) in enumerate(zip(got.hidden_states, again.hidden_states)):
                    if not torch.equal(expected, actual):
                        raise AssertionError(f"{dtype}/{label}: повторный forward разошёлся "
                                             f"на hidden_states[{layer}] (веса не перечитались?)")
            finally:
                _close(plain)
                _close(streamed)
            print(f"  ok: {dtype}/{label}: {len(got.hidden_states)} hidden states, "
                  f"logits и повторный forward бит-в-бит")


# --------------------------------------------------------------------------- #
# Кейс 2: meta-инвариант и связанные веса
# --------------------------------------------------------------------------- #

def case_meta_and_tie(tmp: Path) -> None:
    """Параметры на meta, буферы настоящие; `tie_word_embeddings=True` -> `ValueError`."""
    from transformers import Qwen3ForCausalLM

    root = tmp / "case2-source"
    build_tiny(root)
    weights = ShardedWeights(resolve_stream_source(root))
    try:
        assert weights.dtype is not None, "dtype из config.json не прочитался"
        model = build_streamed(Qwen3ForCausalLM, tiny_config(), weights)
        bad_params = [(name, str(param.device)) for name, param in model.named_parameters()
                      if param.device.type != "meta"]
        if bad_params:
            raise AssertionError(f"параметры не на meta: {bad_params[:5]}")
        meta_buffers = [name for name, buffer in model.named_buffers()
                        if buffer.device.type == "meta"]
        if meta_buffers:
            raise AssertionError(f"буферы оказались на meta: {meta_buffers[:5]}")
        buffers = list(model.named_buffers())
        if not buffers or not all(buffer.numel() > 0 for _, buffer in buffers):
            raise AssertionError("буферы обязаны быть настоящими (rotary inv_freq)")
        assert model.training is False, "build_streamed обязан вернуть eval()-модель"
        print(f"  ok: {len(list(model.named_parameters()))} параметров на meta, "
              f"буферы настоящие: {[name for name, _ in buffers]}")

        try:
            build_streamed(Qwen3ForCausalLM, tiny_config(tie_word_embeddings=True), weights)
        except ValueError as exc:
            if "связанные веса" not in str(exc):
                raise AssertionError(f"ValueError без пояснения: {exc}") from exc
            print("  ok: tie_word_embeddings=True -> ValueError про связанные веса")
        else:
            raise AssertionError("tie_word_embeddings=True не отвергнут")
    finally:
        weights.close()


# --------------------------------------------------------------------------- #
# Кейс 3: ошибки — нет весов, нет имён
# --------------------------------------------------------------------------- #

def case_errors(tmp: Path) -> None:
    """`resolve_stream_source` без весов -> RuntimeError с командой; недостающие имена -> KeyError."""
    from safetensors.torch import load_file, save_file
    from transformers import Qwen3ForCausalLM

    empty = tmp / "case3-empty"
    empty.mkdir()
    (empty / "config.json").write_text(json.dumps(tiny_config().to_dict()), encoding="utf-8")
    try:
        resolve_stream_source(empty)
    except RuntimeError as exc:
        text = str(exc)
        for needle in ("safetensors", "snapshot_download", "config.json"):
            if needle not in text:
                raise AssertionError(f"в тексте ошибки нет {needle!r}: {text}") from exc
        print("  ok: каталог без весов -> RuntimeError с командой snapshot_download(...)")
    else:
        raise AssertionError("каталог без весов разрешился без ошибки")

    root = tmp / "case3-partial"
    build_tiny(root)
    full = load_file(str(root / "model.safetensors"))
    kept = dict(list(full.items())[:3])
    removed = [name for name in full if name not in kept]
    save_file(kept, str(root / "model.safetensors"))

    weights = ShardedWeights(resolve_stream_source(root))
    try:
        try:
            build_streamed(Qwen3ForCausalLM, tiny_config(), weights)
        except KeyError as exc:
            text = str(exc)
            if not any(name in text for name in removed):
                raise AssertionError(f"в KeyError нет ни одного недостающего имени: {text}") from exc
            print(f"  ok: недостающие параметры -> KeyError со списком "
                  f"({len(removed)} имён, первые: {', '.join(removed[:2])})")
        else:
            raise AssertionError("неполный набор весов собрался без KeyError")

        try:
            weights["no.such.parameter"]
        except KeyError as exc:
            if "Примеры имён" not in str(exc):
                raise AssertionError(f"KeyError без примеров имён: {exc}") from exc
            print("  ok: ShardedWeights с неизвестным именем -> KeyError с примерами")
        else:
            raise AssertionError("неизвестное имя не дало KeyError")
    finally:
        weights.close()

    try:
        weights["model.embed_tokens.weight"]
    except RuntimeError as exc:
        if "закрыт" not in str(exc):
            raise AssertionError(f"после close() ожидался RuntimeError про закрытие: {exc}") from exc
        print("  ok: после close() чтение -> RuntimeError")
    else:
        raise AssertionError("после close() карта продолжает читать веса")


# --------------------------------------------------------------------------- #
# Запуск
# --------------------------------------------------------------------------- #

CASES: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("1. forward бит-в-бит (fp32/bf16, single/multi-shard, повторный forward)",
     case_forward_matches),
    ("2. после сборки параметры на meta, буферы настоящие; tie -> ValueError",
     case_meta_and_tie),
    ("3. нет весов -> RuntimeError с подсказкой; нет имён -> KeyError",
     case_errors),
)


def main() -> int:
    failed = 0
    base = temp_base()
    with tempfile.TemporaryDirectory(prefix="qwenstyle-stream-test-", dir=str(base)) as raw:
        tmp = Path(raw)
        print(f"[tmp] {tmp} (TemporaryDirectory, будет удалён)", flush=True)
        for name, case in CASES:
            try:
                case(tmp)
            except Exception as exc:            # noqa: BLE001 — тест-скрипт печатает и продолжает
                print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                failed += 1
            else:
                print(f"PASS {name}", flush=True)
    if failed:
        print(f"FAIL {failed} из {len(CASES)} кейсов", flush=True)
        return 1
    print("ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
