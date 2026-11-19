"""Тесты диск-стриминга весов (`qwenstyle/streaming.py`) — обычный скрипт, без pytest.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_streaming.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Кейсы (шаг 1 — 1–3, шаг 2 «ридер `qwenstyle/shards.py`» — 4–7):

1. модель, собранная `build_streamed` на meta, даёт **бит-в-бит** те же `hidden_states`, что обычная
   загрузка через `from_pretrained` (сравнение `torch.equal`, а не `allclose`), в fp32 и bf16,
   на одном и на двух шардах, и повторный forward (веса перечитываются с диска) не расходится;
2. после сборки все параметры на `meta`, ни один буфер не на meta; `tie_word_embeddings=True`
   отвергается `ValueError` до сборки;
3. `resolve_stream_source` на каталоге без весов -> `RuntimeError` с подсказкой и командой
   `snapshot_download`; недостающие в шарде имена -> `KeyError` со списком (и у модели, и у карты);
4. `close()` закрывает дескрипторы: чтение -> `RuntimeError`, а `info()`/ключи/размеры живут;
5. набор на 2 шарда читается новым путём (`__getitem__` и `read_into` прямо в буфер) — **бит-в-бит**
   с `from_pretrained`, в том числе после каста в bf16;
6. `file_dtype()`: однородный набор -> `"bfloat16"`/`"float32"`, смешанный -> `ValueError`;
7. прямой pinned-путь (`read_into` в буфер пула, без pageable-копии): все веса `is_pinned()`,
   `t_pin == 0`, forward бит-в-бит (CUDA, иначе `SKIP`);
8. cast-путь: файлы bf16 + `dtype="float32"` — модель бит-в-бит с `from_pretrained(dtype=float32)`,
   `ShardedWeights.dtype` = целевой, а `file_dtype()`/`info().file_dtype` = файловый; на конвейере
   каст идёт прежним путём (`t_pin > 0`), а не в pinned-буфер файлового dtype (CUDA, иначе `SKIP`).

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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:              # запуск из любого cwd
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                  # тест не должен оставлять __pycache__ в пакете

from qwenstyle.streaming import (               # noqa: E402 — после правки sys.path
    ShardedWeights,
    StreamSource,
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
    """Закрыть дескрипторы шардов модели и отпустить её (Windows не даёт удалить открытый файл)."""
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
# Кейс 4: close() + info() (новый ридер, без маппинга шардов)
# --------------------------------------------------------------------------- #

def case_close_and_info(tmp: Path) -> None:
    """4. После `close()` чтение -> `RuntimeError`, а `info()`/ключи/формы остаются живыми."""
    import torch
    from safetensors.torch import load_file

    root = tmp / "case4-source"
    build_tiny(root)
    name = "model.norm.weight"
    expected = load_file(str(root / "model.safetensors"))[name]

    weights = ShardedWeights(resolve_stream_source(root))
    weights.close()
    weights.close()                              # идемпотентно
    if not weights.closed:
        raise AssertionError("после close() weights.closed = False")

    try:
        weights[name]
    except RuntimeError as exc:
        if "закрыт" not in str(exc):
            raise AssertionError(f"после close() ожидался RuntimeError про закрытие: {exc}") from exc
        print("  ok: после close() __getitem__ -> RuntimeError (текст прежний)")
    else:
        raise AssertionError("после close() чтение вернуло тензор")

    info = weights.info(name)                    # заголовок уже в памяти — работает после close
    if tuple(info.shape) != tuple(expected.shape) or info.dtype != "float32":
        raise AssertionError(f"info('{name}') после close() = {info!r}, ожидалось float32 {tuple(expected.shape)}")
    if name not in weights or len(weights) == 0:
        raise AssertionError("после close() пропали имена/размер набора")
    try:
        weights.info("no.such.parameter")
    except KeyError as exc:
        if "Примеры имён" not in str(exc):
            raise AssertionError(f"KeyError без примеров имён: {exc}") from exc
    else:
        raise AssertionError("info(неизвестное имя) не дал KeyError")

    buffer = torch.empty(info.nbytes, dtype=torch.uint8)
    try:
        weights.read_into(name, memoryview(buffer.numpy()))
    except RuntimeError as exc:
        if "закрыт" not in str(exc):
            raise AssertionError(f"read_into после close(): ожидался RuntimeError: {exc}") from exc
        print(f"  ok: info({name!r}) работает после close(), read_into -> RuntimeError")
    else:
        raise AssertionError("read_into после close() прочитал байты")


# --------------------------------------------------------------------------- #
# Кейс 5: набор на 2 шарда через новый путь — бит-в-бит
# --------------------------------------------------------------------------- #

def case_two_shards_reader(tmp: Path) -> None:
    """5. Два шарда: `__getitem__` и `read_into` дают те же байты, что `from_pretrained`."""
    import torch
    from transformers import Qwen3ForCausalLM

    root = tmp / "case5-source"
    build_tiny(root)
    multi = tmp / "case5-multi"
    shard_copy(root, multi, shards=2)

    plain = load_text_encoder(Qwen3ForCausalLM, root, dtype="float32",
                              local_files_only=True).eval()
    expected = {name: tensor.detach().clone() for name, tensor in plain.state_dict().items()}
    _close(plain)

    weights = ShardedWeights(resolve_stream_source(multi))
    try:
        if len(weights.shards) != 2:
            raise AssertionError(f"шардов {len(weights.shards)}, ожидалось 2")
        read = 0
        for name, reference in expected.items():
            tensor = weights[name]
            if not torch.equal(tensor, reference):
                diff = (tensor.float() - reference.float()).abs().max().item()
                raise AssertionError(f"{name}: __getitem__ не бит-в-бит с from_pretrained "
                                     f"(max abs diff {diff})")
            info = weights.info(name)
            buffer = torch.full((info.nbytes + 8,), 0xAA, dtype=torch.uint8)
            got = weights.read_into(name, memoryview(buffer.numpy()))
            if got != info.nbytes:
                raise AssertionError(f"{name}: read_into вернул {got} байт, ожидалось {info.nbytes}")
            tail = buffer[info.nbytes:]
            if not torch.equal(tail, torch.full_like(tail, 0xAA)):
                raise AssertionError(f"{name}: read_into задел байты за концом тензора")
            direct = buffer[:info.nbytes].view(getattr(torch, info.dtype)).view(info.shape)
            if not torch.equal(direct, reference):
                raise AssertionError(f"{name}: read_into не бит-в-бит с from_pretrained")
            read += 1
        print(f"  ok: 2 шарда, {read} тензоров через __getitem__/read_into бит-в-бит с from_pretrained")

        cast = ShardedWeights(StreamSource(shards=weights.shards,
                                           weight_map=weights.source.weight_map,
                                           dtype="bfloat16"))
        try:
            if cast.dtype != "bfloat16":
                raise AssertionError(f"dtype источника = {cast.dtype!r}, ожидалось bfloat16")
            for name, reference in expected.items():
                tensor = cast[name]
                if tensor.dtype != torch.bfloat16:
                    raise AssertionError(f"{name}: после каста dtype {tensor.dtype}, ожидалось bfloat16")
                if not torch.equal(tensor, reference.to(torch.bfloat16)):
                    raise AssertionError(f"{name}: каст в bf16 не совпал с .to(bfloat16)")
            print("  ok: каст dtype источника (fp32 -> bf16) на месте и бит-в-бит с .to(bfloat16)")
        finally:
            cast.close()
    finally:
        weights.close()


# --------------------------------------------------------------------------- #
# Кейс 6: file_dtype на однородном и смешанном наборе
# --------------------------------------------------------------------------- #

def case_file_dtype(tmp: Path) -> None:
    """6. `file_dtype()`: однородный набор -> dtype, смешанный -> `ValueError` с перечнем."""
    import torch
    from safetensors.torch import save_file

    root = tmp / "case6"
    root.mkdir(parents=True, exist_ok=True)
    bf16_shard = root / "bf16.safetensors"
    save_file({"a.weight": torch.zeros(4, dtype=torch.bfloat16),
               "b.weight": torch.ones(3, dtype=torch.bfloat16)}, str(bf16_shard))
    fp32_shard = root / "fp32.safetensors"
    save_file({"c.weight": torch.zeros(2, dtype=torch.float32)}, str(fp32_shard))

    uniform = ShardedWeights(StreamSource(shards=(bf16_shard,), weight_map=None, dtype=None))
    try:
        if uniform.file_dtype() != "bfloat16":
            raise AssertionError(f"file_dtype() однородного bf16-набора = {uniform.file_dtype()!r}")
        if uniform.file_dtype("a.weight") != "bfloat16":
            raise AssertionError(f"file_dtype('a.weight') = {uniform.file_dtype('a.weight')!r}")
        plain = ShardedWeights(StreamSource(shards=(fp32_shard,), weight_map=None, dtype=None))
        try:
            if plain.file_dtype() != "float32":
                raise AssertionError(f"file_dtype() fp32-набора = {plain.file_dtype()!r}")
        finally:
            plain.close()
    finally:
        uniform.close()
    if uniform.file_dtype() != "bfloat16":
        raise AssertionError("кэш file_dtype не выжил close()")
    print("  ok: file_dtype() -> bfloat16 / float32 на однородных наборах, кэш живёт после close()")

    mixed = ShardedWeights(StreamSource(shards=(bf16_shard, fp32_shard),
                                        weight_map=None, dtype=None))
    try:
        try:
            mixed.file_dtype()
        except ValueError as exc:
            text = str(exc)
            for needle in ("float32", "bfloat16", "a.weight", "c.weight"):
                if needle not in text:
                    raise AssertionError(f"в ValueError нет {needle!r}: {text}") from exc
            print(f"  ok: смешанный набор -> ValueError: {text}")
        else:
            raise AssertionError("смешанный по dtype набор не дал ValueError")
        if mixed.file_dtype("a.weight") != "bfloat16" or mixed.file_dtype("c.weight") != "float32":
            raise AssertionError("file_dtype(имя) на смешанном наборе соврал")
        print("  ok: file_dtype(имя) на смешанном наборе отдаёт dtype конкретного тензора")
    finally:
        mixed.close()


# --------------------------------------------------------------------------- #
# Кейс 7: прямой pinned-путь (read_into в буфер пула, без pageable-копии)
# --------------------------------------------------------------------------- #

def case_pinned_direct(tmp: Path) -> None:
    """7. Файлы bf16 + dtype «как в файле» -> веса читаются сразу в pinned-буфер.

    Прямой путь включается только когда каст не нужен (dtype источника == dtype файла).
    Признаки, что он действительно включился: каждый отданный хук `on_load` тензор
    `is_pinned()`, а `t_pin` (копия в pinned) — ровно ноль: второй копии нет.
    """
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import Qwen3ForCausalLM

    from qwenstyle.streaming import StreamStats, build_streamed, stream_stats

    if not torch.cuda.is_available():
        print("  SKIP (нет CUDA): pinned-память недоступна")
        return

    root = tmp / "case7-source"
    build_tiny(root)
    # Веса в bf16 и config с тем же dtype: тогда dtype источника == dtype файла, каст не нужен.
    state = load_file(str(root / "model.safetensors"))
    save_file({name: tensor.to(torch.bfloat16) for name, tensor in state.items()},
              str(root / "model.safetensors"))
    config_path = root / "config.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    config_data["dtype"] = "bfloat16"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")

    torch.manual_seed(123)
    input_ids = torch.randint(0, TINY["vocab_size"], (1, 7), device="cuda")
    plain = load_text_encoder(Qwen3ForCausalLM, root, dtype="bfloat16",
                              local_files_only=True).eval().to("cuda")
    loaded: list[tuple[str, Any]] = []
    weights = ShardedWeights(resolve_stream_source(root))
    streamed = None
    try:
        if weights.dtype != "bfloat16":
            raise AssertionError(f"dtype источника = {weights.dtype!r}, ожидалось bfloat16")
        stats = StreamStats()
        streamed = build_streamed(Qwen3ForCausalLM, tiny_config(), weights, device="cuda",
                                  prefetch=4, pin_memory=True, lookahead=1, stats=stats,
                                  on_load=lambda name, tensor: loaded.append((name, tensor)))
        with torch.no_grad():
            expected = plain(input_ids=input_ids, output_hidden_states=True)
            actual = streamed(input_ids=input_ids, output_hidden_states=True)
        for layer, (left, right) in enumerate(zip(expected.hidden_states, actual.hidden_states)):
            if not torch.equal(left, right):
                diff = (left.float() - right.float()).abs().max().item()
                raise AssertionError(f"hidden_states[{layer}] не бит-в-бит (max abs diff {diff})")
        if not torch.equal(expected.logits, actual.logits):
            raise AssertionError("logits не бит-в-бит")
        stats = stream_stats(streamed) or stats
        if stats.t_pin != 0.0:
            raise AssertionError(f"t_pin={stats.t_pin:.2f}s: пошла вторая копия в pinned-буфер "
                                 f"(прямой путь не включился)")
        if stats.t_read <= 0.0 or stats.bytes_read <= 0:
            raise AssertionError(f"чтение не учтено: t_read={stats.t_read}, bytes={stats.bytes_read}")
        if not loaded:
            raise AssertionError("on_load не вызывался")
        unpinned = [name for name, tensor in loaded if not tensor.is_pinned()]
        if unpinned:
            raise AssertionError(f"веса не в pinned-памяти: {unpinned[:3]}")
        print(f"  ok: {len(loaded)} тензоров прочитаны прямо в pinned-буфер (t_pin=0, "
              f"read={stats.t_read:.2f}s), forward бит-в-бит с from_pretrained")
    finally:
        if streamed is not None:
            _close(streamed)
        weights.close()
        _close(plain)
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Кейс 8: cast-путь — файлы bf16, dtype источника float32
# --------------------------------------------------------------------------- #

def _bf16_files(root: Path) -> None:
    """Переписать веса мини-модели в bf16 и сказать то же в `config.json`.

    Так `dtype` источника (`resolve_stream_source` -> `config.dtype`) совпадает с файловым,
    и прямой pinned-путь вообще возможен — а `dtype="float32"` в вызове включает каст.
    """
    import torch
    from safetensors.torch import load_file, save_file

    state = load_file(str(root / "model.safetensors"))
    save_file({name: tensor.to(torch.bfloat16) for name, tensor in state.items()},
              str(root / "model.safetensors"))
    config_path = root / "config.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    config_data["dtype"] = "bfloat16"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")


def case_cast_path(tmp: Path) -> None:
    """8. Каст dtype источника: модель бит-в-бит, dtype-пространства не путаются, пул не читает в себя.

    Доказывает три вещи:
    * `load_text_encoder(dtype="float32", stream=True)` на bf16-файлах даёт `hidden_states` и
      `logits` **бит-в-бит** с `from_pretrained(dtype=torch.float32)`: каст в `__getitem__`
      эквивалентен касту при загрузке;
    * `ShardedWeights.dtype` — *целевой* dtype (`float32`), а `file_dtype()` и
      `info().file_dtype` — *файловый* (`bfloat16` / `BF16`): на этом различии и стоит guard
      `_load_pinned`;
    * на конвейере (CUDA) при касте признак прямого pinned-пути обратный кейсу 7: `t_pin > 0`,
      то есть байты bf16 не легли в pinned-буфер как float32, а прошли `__getitem__` -> `take`;
      все отданные хуком тензоры — уже `float32`.
    """
    import torch
    from transformers import Qwen3ForCausalLM

    root = tmp / "case8-source"
    build_tiny(root)
    _bf16_files(root)

    torch.manual_seed(123)
    input_ids = torch.randint(0, TINY["vocab_size"], (1, 7))
    plain = load_text_encoder(Qwen3ForCausalLM, root, dtype="float32",
                              local_files_only=True).eval()
    streamed = None
    try:
        streamed = load_text_encoder(Qwen3ForCausalLM, root, dtype="float32",
                                     local_files_only=True, stream=True, device="cpu",
                                     prefetch=0).eval()
        weights = _streamed_weights(streamed)
        if weights is None:
            raise AssertionError("stream=True не выставил _stream_weights")
        if weights.dtype != "float32":
            raise AssertionError(f"ShardedWeights.dtype = {weights.dtype!r}, ожидалось float32 "
                                 f"(файлы bf16 — проверка, что dtype не подменён файловым)")
        if weights.file_dtype() != "bfloat16":
            raise AssertionError(f"file_dtype() = {weights.file_dtype()!r}, ожидалось bfloat16")
        for name in ("model.norm.weight", "model.embed_tokens.weight"):
            info = weights.info(name)
            if info.file_dtype != "BF16" or info.dtype != "bfloat16":
                raise AssertionError(f"{name}: info.file_dtype={info.file_dtype!r} "
                                     f"info.dtype={info.dtype!r}, ожидалось BF16/bfloat16")
            if weights[name].dtype != torch.float32:
                raise AssertionError(f"{name}: __getitem__ отдал {weights[name].dtype}, "
                                     f"ожидалось torch.float32")
        print("  ok: dtype источника float32 vs файловый bfloat16 — два разных пространства имён")

        with torch.no_grad():
            expected = plain(input_ids=input_ids, output_hidden_states=True)
            actual = streamed(input_ids=input_ids, output_hidden_states=True)
        for layer, (left, right) in enumerate(zip(expected.hidden_states, actual.hidden_states)):
            if not torch.equal(left, right):
                diff = (left.float() - right.float()).abs().max().item()
                raise AssertionError(f"hidden_states[{layer}] не бит-в-бит с from_pretrained"
                                     f"(dtype=float32) (max abs diff {diff})")
        if not torch.equal(expected.logits, actual.logits):
            raise AssertionError("logits не бит-в-бит с from_pretrained(dtype=float32)")
        print(f"  ok: bf16-файлы + dtype=float32: {len(actual.hidden_states)} hidden states и "
              f"logits бит-в-бит с from_pretrained(dtype=torch.float32)")
    finally:
        if streamed is not None:
            _close(streamed)
        _close(plain)

    if not torch.cuda.is_available():
        print("  SKIP (нет CUDA): конвейер и pinned-стейджинг недоступны")
        return

    # Конвейер (CUDA) при касте: прямой pinned-путь запрещён — байты файлового dtype не должны
    # лечь в буфер, который потом будет прочитан как float32.
    from qwenstyle.streaming import StreamStats, build_streamed, stream_stats

    loaded: list[tuple[str, Any]] = []
    stats = StreamStats()
    source = resolve_stream_source(root)
    cast_weights = ShardedWeights(StreamSource(shards=source.shards,
                                               weight_map=source.weight_map,
                                               dtype="float32"))
    cast_model = None
    plain_gpu = load_text_encoder(Qwen3ForCausalLM, root, dtype="float32",
                                 local_files_only=True).eval().to("cuda")
    try:
        cast_model = build_streamed(Qwen3ForCausalLM, tiny_config(), cast_weights,
                                    device="cuda", prefetch=4, pin_memory=True, lookahead=1,
                                    stats=stats,
                                    on_load=lambda name, tensor: loaded.append((name, tensor)))
        gpu_ids = input_ids.to("cuda")
        with torch.no_grad():
            expected_gpu = plain_gpu(input_ids=gpu_ids, output_hidden_states=True)
            actual_gpu = cast_model(input_ids=gpu_ids, output_hidden_states=True)
        for layer, (left, right) in enumerate(zip(expected_gpu.hidden_states,
                                                  actual_gpu.hidden_states)):
            if not torch.equal(left, right):
                diff = (left.float() - right.float()).abs().max().item()
                raise AssertionError(f"конвейер+каст: hidden_states[{layer}] не бит-в-бит "
                                     f"(max abs diff {diff})")
        if not torch.equal(expected_gpu.logits, actual_gpu.logits):
            raise AssertionError("конвейер+каст: logits не бит-в-бит")
        stats = stream_stats(cast_model) or stats
        if stats.t_pin <= 0.0:
            raise AssertionError("t_pin=0 при касте: байты файлового dtype залезли в прямой "
                                 "pinned-путь (должно было быть __getitem__ -> take)")
        if not loaded:
            raise AssertionError("on_load не вызывался")
        wrong = [name for name, tensor in loaded if tensor.dtype != torch.float32]
        if wrong:
            raise AssertionError(f"хук получил не float32: {wrong[:3]}")
        print(f"  ok: конвейер+каст: {len(loaded)} тензоров через __getitem__ -> take "
              f"(t_pin={stats.t_pin * 1e3:.2f} ms > 0), forward бит-в-бит")
    finally:
        if cast_model is not None:
            _close(cast_model)
        cast_weights.close()
        _close(plain_gpu)
        torch.cuda.empty_cache()


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
    ("4. close() -> RuntimeError, info()/ключи/формы остаются",
     case_close_and_info),
    ("5. набор на 2 шарда: __getitem__/read_into бит-в-бит с from_pretrained (+ каст bf16)",
     case_two_shards_reader),
    ("6. file_dtype(): однородный -> dtype, смешанный -> ValueError",
     case_file_dtype),
    ("7. прямое чтение в pinned-буфер (t_pin=0) и forward бит-в-бит",
     case_pinned_direct),
    ("8. cast-путь: bf16-файлы + dtype=float32 (бит-в-бит, t_pin>0)",
     case_cast_path),
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
