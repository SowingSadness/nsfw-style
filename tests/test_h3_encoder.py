"""Тест энкодера Qwen3-VL-32B heretic для MiniMax-H3-Image (шаг 5) — скрипт, без pytest.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_h3_encoder.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Кейсы:

1. бекенд `h3qwen32-heretic` зарегистрирован: `cache_tag` совпадает (`_te-…` попадёт в ключ),
   `requires_vision is True`, репозиторий/ревизия — heretic-32B, `local_weights is False`;
2. `RunConfig.from_profile("h3-image").encoder` резолвится в те же repo/revision/stream/prefetch/
   cache_tag, что объявлены в пресете (иначе ключ кэша и веса разъедутся молча — R1);
3. ключ кэша эмбеддингов содержит `_te-h3qwen32-heretic` и **не зависит** от `stream` (стриминг —
   раскладка памяти, а не другая математика); 32B-тег не путается с 8B (`_te-heretic`);
4. `encoder.local_dir` на пустой каталог -> `RuntimeError` с готовой командой `snapshot_download`
   (а не трейсбек transformers/Hub) — энкодер не грузится, 62 ГиБ не скачиваются;
5. smoke без 62 ГиБ: мини-`Qwen3VL`-совместимая модель с `hidden_size=5120`, загруженная через
   `streaming.load_text_encoder(stream=True)`, отдаёт hidden state формы `(1, N, 5120)`.

Веса heretic-32B в тестах НЕ загружаются: он идёт секунды и не требует сети/GPU.
"""
from __future__ import annotations

import gc
import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:              # запуск из любого cwd
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                  # тест не должен оставлять __pycache__ в пакете

from qwenstyle import registry                    # noqa: E402 — после правки sys.path
from qwenstyle.config import RunConfig            # noqa: E402

NAME = "h3qwen32-heretic"
REPO = "llmfan46/Qwen3-VL-32B-Instruct-ultra-uncensored-heretic"
TAG_SUFFIX = "_te-h3qwen32-heretic"
#: Мини-модель с шириной настоящего энкодера: hidden 5120, но 1 слой и vocab 128 -> секунды.
MINI = dict(
    hidden_size=5120,
    num_hidden_layers=1,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=128,
    intermediate_size=128,
    vocab_size=128,
    tie_word_embeddings=False,
)


# --------------------------------------------------------------------------- #
# Вспомогательное
# --------------------------------------------------------------------------- #

def temp_base() -> Path:
    """База для `TemporaryDirectory`: системный temp, если он вне репозитория.

    На этой машине `tempfile.gettempdir()` может отдавать cwd (песочница), поэтому базу выбираем
    явно и в корне репозитория ничего не оставляем (каталог всё равно удаляется).
    """
    candidate = Path(tempfile.gettempdir())
    try:
        resolved = candidate.resolve()
    except OSError:                             # pragma: no cover — недоступный путь
        resolved = candidate
    if resolved != REPO_ROOT and REPO_ROOT not in resolved.parents:
        return resolved
    return REPO_ROOT / "tests"


def _embeds_key(cfg: RunConfig) -> str:
    """Имя записи кэша эмбеддингов для T2I-кадра (без референса, `img_sha="noref"`)."""
    return cfg.embeds_key((1024, 1024), "0123456789ab", "noref")


# --------------------------------------------------------------------------- #
# Кейс 1: бекенд зарегистрирован
# --------------------------------------------------------------------------- #

def case_registered() -> None:
    """`registry.get_encoder` находит бекенд, метаданные совпадают, torch не тянется импортом."""
    assert "torch" not in sys.modules, "импорт модуля энкодеров не должен тянуть torch"
    cls = registry.get_encoder(NAME)
    assert cls.cache_tag == NAME, f"cache_tag={cls.cache_tag!r}, ожидался {NAME!r}"
    assert cls.requires_vision is True, "Qwen3-VL кодирует картинку vision-башней"
    assert cls.repo == REPO, f"repo={cls.repo!r}"
    assert cls.revision, "ревизия обязана быть закреплена (R1: веса входят в ключ кэша)"
    assert cls.local_weights is True, "веса 32B лежат в cache/weights (пересобранные шарды)"
    assert cls.local_dir == "cache/weights/h3qwen32-heretic", f"local_dir={cls.local_dir!r}"
    assert (cls.memory_gib or 0) > 60, f"memory_gib={cls.memory_gib} — ожидались ~62 GiB"
    assert NAME in registry.available("encoder"), "бекенд не попал в список энкодеров"
    # 8B-энкодеры не тронуты
    assert registry.get_encoder("base").cache_tag == "base"
    assert registry.get_encoder("heretic").cache_tag == "heretic"
    print(f"  ok: {NAME} -> {cls.repo} @ {cls.revision[:12]}, cache_tag={cls.cache_tag}")


# --------------------------------------------------------------------------- #
# Кейс 2: пресет
# --------------------------------------------------------------------------- #

def case_profile() -> None:
    """`RunConfig.from_profile("h3-image").encoder` = объявленный в пресете heretic-32B."""
    cfg = RunConfig.from_profile("h3-image")
    enc = cfg.encoder
    assert enc.name == NAME, f"encoder.name={enc.name!r}"
    assert enc.repo == REPO, f"encoder.repo={enc.repo!r}"
    assert enc.revision, "encoder.revision пуст — веса не закреплены"
    assert enc.cache_tag == NAME, f"encoder.cache_tag={enc.cache_tag!r}"
    assert enc.dtype == "bfloat16", f"encoder.dtype={enc.dtype!r}"
    assert enc.device == "auto", f"encoder.device={enc.device!r}"
    assert enc.stream is True, "62.1 GiB без стриминга не влезают в 63.6 GiB RAM"
    assert int(enc.prefetch) == 4, f"encoder.prefetch={enc.prefetch}"
    assert enc.requires_vision is True
    # `local_dir` в пресете не задан (иначе он наследовался бы от base — cache/weights/heretic-te):
    # умолчание даёт сам бекенд, и оно ведёт в пересобранные шарды < 40 GiB.
    assert enc.local_dir is None, f"encoder.local_dir={enc.local_dir!r}"
    resolved = registry.get_encoder(NAME)().resolve(cfg).local_dir
    assert resolved is not None and "h3qwen32" in str(resolved), f"resolve().local_dir={resolved}"
    print(f"  ok: пресет h3-image -> {enc.repo} @ {enc.revision[:12]}, "
          f"{enc.dtype}/{enc.device}, stream={enc.stream}, prefetch={enc.prefetch}, "
          f"local_dir={resolved}")


# --------------------------------------------------------------------------- #
# Кейс 3: ключ кэша
# --------------------------------------------------------------------------- #

def case_cache_key() -> None:
    """Тег в ключе, `stream` ключ не меняет, 32B-записи не совпадают с 8B."""
    cfg = RunConfig.from_profile("h3-image")
    key = _embeds_key(cfg)
    assert TAG_SUFFIX in key, f"в ключе нет {TAG_SUFFIX}: {key}"
    parts = cfg.cache_key_parts((1024, 1024), "0123456789ab", "noref")
    assert parts["encoder"] == NAME, f"cache_key_parts.encoder={parts['encoder']!r}"
    assert parts["encoder_repo"] == REPO

    off = cfg.merged(**{"encoder.stream": False})
    assert _embeds_key(off) == key, (
        f"stream меняет ключ кэша: {_embeds_key(off)} != {key} "
        "(стриминг — раскладка памяти, не другая математика)")

    base = RunConfig.from_profile("base")
    assert _embeds_key(base) != key, "запись 32B совпала с записью базового энкодера"
    heretic = base.merged(**{"encoder.name": "heretic"})
    assert TAG_SUFFIX not in _embeds_key(heretic), "8B-heretic получил тег 32B-энкодера"
    print(f"  ok: ключ {key}; stream=False -> тот же ключ")


# --------------------------------------------------------------------------- #
# Кейс 4: нет весов — понятная ошибка
# --------------------------------------------------------------------------- #

def case_missing_weights(tmp: Path) -> None:
    """`encoder.local_dir` без весов -> `RuntimeError` с `snapshot_download` (без загрузки 62 ГиБ)."""
    empty = tmp / "case4-empty"
    empty.mkdir()
    cls = registry.get_encoder(NAME)
    cfg = RunConfig.from_profile("h3-image").merged(**{"encoder.local_dir": str(empty)})
    assert cfg.encoder.local_dir == str(empty), "local_dir не дошёл до EncoderConfig"

    ok, why = cls().status(cfg)
    assert ok is False, f"status без весов вернул доступность: {why!r}"

    try:
        cls().load(cfg)
    except RuntimeError as exc:
        text = str(exc)
        for needle in ("snapshot_download", REPO, "config.json"):
            assert needle in text, f"в тексте ошибки нет {needle!r}: {text}"
        print(f"  ok: пустой local_dir -> RuntimeError с командой snapshot_download "
              f"({text.split(':')[0][:60]}…)")
    else:
        raise AssertionError("загрузка из пустого каталога прошла без ошибки")


# --------------------------------------------------------------------------- #
# Кейс 5: smoke hidden 5120 через стриминг
# --------------------------------------------------------------------------- #

def case_stream_smoke(tmp: Path) -> None:
    """Мини-модель (hidden 5120) через `stream=True` отдаёт hidden state `(1, N, 5120)`."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from qwenstyle.streaming import load_text_encoder

    root = tmp / "case5-mini"
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(**MINI)).eval()
    model.save_pretrained(str(root))
    n_params = sum(p.numel() for p in model.parameters())
    del model
    gc.collect()

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    assert config["hidden_size"] == 5120, f"config.json: hidden_size={config['hidden_size']}"

    streamed = load_text_encoder(Qwen3ForCausalLM, root, dtype="bfloat16",
                                local_files_only=True, device="cpu", stream=True)
    try:
        input_ids = torch.randint(0, MINI["vocab_size"], (1, 7))
        with torch.no_grad():
            out = streamed(input_ids=input_ids, output_hidden_states=True)
        shape = tuple(out.hidden_states[-1].shape)
        assert shape == (1, 7, 5120), f"hidden state {shape}, ожидалось (1, 7, 5120)"
        empty = [name for name, param in streamed.named_parameters() if param.numel()]
        assert not empty, f"после forward веса не освобождены: {empty[:3]}"
        print(f"  ok: stream=True, {n_params / 1e6:.1f}M параметров, "
              f"hidden state {shape}, параметры снова пустые")
    finally:
        weights: Any = getattr(streamed, "_stream_weights", None)
        if weights is not None:
            weights.close()
        del streamed
        gc.collect()


# --------------------------------------------------------------------------- #
# Запуск
# --------------------------------------------------------------------------- #

CASES: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("1. бекенд h3qwen32-heretic зарегистрирован (cache_tag/vision/repo/revision)",
     lambda _tmp: case_registered()),
    ("2. пресет h3-image резолвится в heretic-32B (repo/revision/stream/prefetch)",
     lambda _tmp: case_profile()),
    ("3. ключ кэша содержит _te-h3qwen32-heretic и не зависит от stream",
     lambda _tmp: case_cache_key()),
    ("4. local_dir без весов -> RuntimeError с snapshot_download", case_missing_weights),
    ("5. smoke: stream=True на hidden 5120 -> hidden state (1, N, 5120)", case_stream_smoke),
)


def main() -> int:
    failed = 0
    base = temp_base()
    with tempfile.TemporaryDirectory(prefix="qwenstyle-h3-te-", dir=str(base)) as raw:
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
