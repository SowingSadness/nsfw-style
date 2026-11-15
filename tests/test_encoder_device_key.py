"""Ключ кэша эмбеддингов: `encoder.dtype`/`encoder.device` в имени и в сверке сайдкара.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_encoder_device_key.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (план `te-gpu-streaming`, шаг 2): GPU-энкод ≠ CPU-энкод по округлениям, поэтому один
`.pt` не может означать два разных тензора (AGENTS.md R1). До этой правки `encoder.dtype` и
`encoder.device` не попадали ни в имя, ни в сверку — cuda-прогон молча брал cpu-эмбеддинги.

Кейсы:

1. **Эталон имени.** `base/cpu/bfloat16` (явный `--set encoder.device=cpu`) + crop 1376x768 -> имя
   литералом совпадает с историческим `embeds_<md5>_<sha>_1376x768.pt` (иначе все CPU-записи
   в `cache/` стали бы мёртвыми). Умолчание теперь `device=auto` -> cuda, поэтому cpu-эталон
   проверяется явным значением.
2. **Матрица суффиксов** (`EncoderConfig.key_suffix`): base/cpu/bf16 -> `''`;
   heretic/cpu/bf16 -> `_te-heretic`; heretic/cuda/bf16 -> `_te-heretic-dev-cuda`;
   base/cuda/bf16 -> `-dev-cuda`; base/cuda/fp16 -> `-fp16-dev-cuda`; `auto` -> фактическое
   устройство машины (`cuda` на машине с CUDA).
3. **Матрица переиспользования.** Смена seed/steps/lora-scale/precision/stream имена НЕ меняет
   (бесплатные оси), смена encoder.dtype/encoder.device/encoder.name — меняет; `auto` совпадает
   с явным `cuda`/`cpu` по фактическому устройству.
4. **Сверка сайдкара** (узкий белый список): (а) сайдкар старого формата без новых полей —
   попадание; (б) `encoder_device: cpu` при ожидаемом `cuda` — промах, в тексте есть
   `encoder_device`; (в) другая `schema` — промах.

Без GPU и без torch (`qwenstyle.pipeline` импортируется без тяжёлых зависимостей).
"""
from __future__ import annotations

import itertools
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Пресет `base` с закреплённым штатным энкодером: с D7 дефолтом в `presets/base.yaml` стал
#: heretic, а этот тест проверяет именно матрицу тегов (`base` -> без сегмента, `heretic` ->
#: `_te-heretic`), поэтому энкодер фиксируется явно — так же, как это делают `selftest`
#: и остальные проверки пакета.
def _profile_base(**overrides):
    pinned = {"encoder.name": "base"}
    pinned.update(overrides)
    from qwenstyle.config import RunConfig

    return RunConfig.from_profile("base").merged(**pinned)


sys.dont_write_bytecode = True                   # тест не должен оставлять __pycache__ в пакете
os.environ.setdefault("HF_HUB_OFFLINE", "1")     # R3: только локальные веса, никакой сети

from qwenstyle import pipeline                    # noqa: E402 — после правки sys.path
from qwenstyle.config import (                    # noqa: E402
    CACHE_SCHEMA,
    EncoderConfig,
    RunConfig,
    auto_encoder_device,
)

#: Историческое имя записи из `cache/` — эталон байт-в-байт (AGENTS.md R1).
LEGACY_NAME = "embeds_2d183e29dc3e_c8cb6f3b62f8_1376x768.pt"
MD5 = "2d183e29dc3e"
SHA = "c8cb6f3b62f8"
CROP = (1376, 768)


# --------------------------------------------------------------------------- #
# Кейс 1: эталон имени
# --------------------------------------------------------------------------- #

def case_legacy_name() -> None:
    """Явный `encoder.device=cpu` (эскапей) даёт ровно прежнюю строку имени.

    Умолчание — `auto` -> cuda, его имя содержит `-dev-cuda` (проверяется в кейсе 2),
    поэтому cpu-эталон фиксируется явным значением, а не «умолчанием».
    """
    cfg = _profile_base(**{"encoder.device": "cpu"})
    for label, enc in (("явный --set encoder.device=cpu", cfg.encoder),
                       ("явный EncoderConfig", EncoderConfig(name="base", cache_tag="base",
                                                            device="cpu"))):
        key = f"embeds_{MD5}_{SHA}{enc.key_suffix}_{CROP[0]}x{CROP[1]}.pt"
        if key != LEGACY_NAME:
            raise AssertionError(f"{label}: имя {key!r} != эталона {LEGACY_NAME!r}")
    if cfg.embeds_key(CROP, MD5, SHA) != LEGACY_NAME:
        raise AssertionError(f"RunConfig.embeds_key -> {cfg.embeds_key(CROP, MD5, SHA)!r}, "
                             f"ожидалось {LEGACY_NAME!r}")
    print(f"  ok: base/cpu/bfloat16 (явный cpu) -> {LEGACY_NAME}")


# --------------------------------------------------------------------------- #
# Кейс 2: матрица суффиксов
# --------------------------------------------------------------------------- #

SUFFIXES: tuple[tuple[str, dict, str], ...] = (
    ("base/cpu/bf16", {"device": "cpu"}, ""),
    ("heretic/cpu/bf16", {"cache_tag": "heretic", "device": "cpu"}, "_te-heretic"),
    ("heretic/cuda/bf16", {"cache_tag": "heretic", "device": "cuda"}, "_te-heretic-dev-cuda"),
    ("base/cuda/bf16", {"device": "cuda"}, "-dev-cuda"),
    ("base/cuda/fp16", {"device": "cuda", "dtype": "fp16"}, "-fp16-dev-cuda"),
)


def case_suffix_matrix() -> None:
    """Сегменты появляются только для неумолчаний, порядок — тег, dtype, device.

    `auto` не литерал в имени, а фактическое устройство (`auto_encoder_device()`): иначе одна
    запись означала бы два разных тензора (GPU vs CPU округления) на разных машинах.
    """
    for label, kwargs, expected in SUFFIXES:
        enc = EncoderConfig(**kwargs)
        if enc.key_suffix != expected:
            raise AssertionError(f"{label}: key_suffix={enc.key_suffix!r}, ожидалось {expected!r}")
        if expected and f"embeds_{MD5}_{SHA}{expected}_{CROP[0]}x{CROP[1]}.pt" == LEGACY_NAME:
            raise AssertionError(f"{label}: неумолчание дало историческое имя")
        print(f"  ok: {label:16} -> {expected!r}")

    actual = auto_encoder_device()
    expected_auto = "" if actual == "cpu" else f"-dev-{actual}"
    auto_suffix = EncoderConfig(device="auto").key_suffix
    if auto_suffix != expected_auto:
        raise AssertionError(f"auto (фактическое {actual}) дал key_suffix={auto_suffix!r}, "
                             f"ожидалось {expected_auto!r}")
    print(f"  ok: auto (фактическое {actual}) -> {auto_suffix!r}")


# --------------------------------------------------------------------------- #
# Кейс 3: матрица переиспользования
# --------------------------------------------------------------------------- #

def _key(cfg: RunConfig) -> str:
    return cfg.embeds_key(CROP, MD5, SHA)


def case_reuse_matrix() -> None:
    """Бесплатные оси имя не меняют, dtype/device/name — меняют; `auto` = фактическое устройство."""
    # Базовая точка — явный cpu: умолчание (`auto` -> cuda) теперь тоже несёт `-dev-cuda`.
    base = _profile_base(**{"encoder.device": "cpu"})
    base_key = _key(base)

    free: tuple[tuple[str, dict], ...] = (
        ("seed=42", {"seed": 42}),
        ("steps=8", {"generation.steps": 8}),
        ("lora-scale=0.5", {"generation.lora.scale": 0.5}),
        ("precision=none", {"generation.precision": "none"}),
        ("encoder.stream=true", {"encoder.stream": "true"}),
    )
    for label, overrides in free:
        key = _key(base.merged(**overrides))
        if key != base_key:
            raise AssertionError(f"{label} изменил имя кэша: {key!r} != {base_key!r}")
        print(f"  ok: {label:22} имя то же ({key})")

    costly: tuple[tuple[str, dict], ...] = (
        ("encoder.dtype=fp16", {"encoder.dtype": "fp16"}),
        ("encoder.device=cuda", {"encoder.device": "cuda"}),
        ("encoder.name=heretic", {"encoder.name": "heretic"}),
    )
    seen = {base_key}
    for label, overrides in costly:
        key = _key(base.merged(**overrides))
        if key in seen:
            raise AssertionError(f"{label} НЕ изменил имя кэша: {key!r} (дыра R1)")
        if key == base_key:
            raise AssertionError(f"{label}: имя не отличается от умолчания")
        seen.add(key)
        print(f"  ok: {label:22} имя другое ({key})")

    # Все три оси независимы: 2*2*2 комбинации дают 8 разных имён.
    combos = {
        _key(base.merged(**{"encoder.dtype": dt, "encoder.device": dev, "encoder.name": name}))
        for dt, dev, name in itertools.product(("bfloat16", "fp16"), ("cpu", "cuda"),
                                               ("base", "heretic"))
    }
    if len(combos) != 8:
        raise AssertionError(f"комбинации dtype/device/name дали {len(combos)} имён вместо 8: "
                             f"{sorted(combos)}")
    print(f"  ok: dtype x device x name -> {len(combos)} различных имён")

    # `auto` в ключе равно явному фактическому устройству (на этой машине — cuda).
    actual = auto_encoder_device()
    auto_key = _key(base.merged(**{"encoder.device": "auto"}))
    if auto_key != _key(base.merged(**{"encoder.device": actual})):
        raise AssertionError(f"auto ({auto_key}) не равен явному {actual} "
                             f"({_key(base.merged(**{'encoder.device': actual}))})")
    print(f"  ok: encoder.device=auto -> то же имя, что явный {actual}")


# --------------------------------------------------------------------------- #
# Кейс 4: сверка сайдкара
# --------------------------------------------------------------------------- #

def _old_sidecar() -> dict:
    """Сайдкар, который реально лежит в `cache/` (формат `run_style.py`, без новых полей)."""
    return {
        "prompt_md5": MD5,
        "prompt_file": "tests/texts/exp1.txt",
        "image": "out/tests/input-864x480.png",
        "image_sha256_12": SHA,
        "text_encoder": "base",
        "crop": f"{CROP[0]}x{CROP[1]}",
        "resolution": 1024,
        "steps": 25,
        "seed": 20261113,
        "lora_scale": 0.7,
        "fp8": True,
        "base_model": "Qwen/Qwen-Image-2.1",
        "lora": "fusal-qwen-image-2.1-v2-exact-r80.safetensors",
        "embeds_shape": [1, 512, 3584],
        "created": "2026-01-01 00:00:00",
    }


def _hit(cfg: RunConfig, meta: dict | None, tmp: Path) -> tuple[bool, str]:
    sidecar = tmp / LEGACY_NAME.replace(".pt", ".json")
    return pipeline.embeds_cache_hit(meta, pipeline.embeds_expected(cfg, CROP, MD5, SHA), sidecar)


def case_sidecar_check(tmp: Path) -> None:
    """Старый сайдкар валиден; смена device/schema — промах с названной причиной."""
    cfg = _profile_base(**{"encoder.device": "cpu"})

    # (а) старый формат: нет schema/encoder_dtype/encoder_device/encoder_repo -> попадание
    hit, reason = _hit(cfg, _old_sidecar(), tmp)
    if not hit or reason:
        raise AssertionError(f"сайдкар старого формата стал промахом: {reason!r}")
    hit_no_sidecar, reason_no_sidecar = _hit(cfg, None, tmp)
    if not hit_no_sidecar or reason_no_sidecar:
        raise AssertionError(f"отсутствие сайдкара — тоже попадание, получено {reason_no_sidecar!r}")
    print("  ok: сайдкар без новых полей (и без сайдкара вовсе) -> попадание")

    # Бесплатные оси в сайдкаре не сверяются: смена steps/seed/lora/precision/guidance — попадание
    free = _old_sidecar()
    free.update({"steps": 8, "seed": 7, "lora_scale": 0.5, "fp8": False, "guidance": 2.5})
    hit, reason = _hit(cfg, free, tmp)
    if not hit:
        raise AssertionError(f"бесплатная ось дала промах: {reason!r}")
    print("  ok: steps/seed/lora_scale/fp8 в сайдкаре не инвалидируют запись")

    # (б) ожидаем cuda, в сайдкаре cpu -> промах с именем поля
    cuda_cfg = cfg.merged(**{"encoder.device": "cuda"})
    meta = _old_sidecar()
    meta["encoder_device"] = "cpu"
    hit, reason = _hit(cuda_cfg, meta, tmp)
    if hit:
        raise AssertionError("encoder_device=cpu при ожидаемом cuda не дал промаха")
    if "encoder_device" not in reason:
        raise AssertionError(f"в причине промаха нет имени поля: {reason!r}")
    if "[cache] miss: изменился" not in reason or "cpu -> cuda" not in reason:
        raise AssertionError(f"формат строки промаха не тот: {reason!r}")
    print(f"  ok: device cpu!=cuda -> промах: {reason}")

    # dtype: bf16 в сайдкаре vs fp16 в конфиге -> промах
    fp16_cfg = cfg.merged(**{"encoder.dtype": "fp16"})
    hit, reason = _hit(fp16_cfg, _old_sidecar(), tmp)
    if hit or "encoder_dtype" not in reason:
        raise AssertionError(f"encoder_dtype не сверяется: hit={hit}, reason={reason!r}")
    print(f"  ok: dtype bfloat16!=float16 -> промах: {reason}")

    # (в) другая схема -> промах
    meta = _old_sidecar()
    meta["schema"] = "qwenstyle-0"
    hit, reason = _hit(cfg, meta, tmp)
    if hit or "schema" not in reason:
        raise AssertionError(f"schema не сверяется: hit={hit}, reason={reason!r}")
    print(f"  ok: другая schema -> промах: {reason}")

    # тег энкодера: сайдкар heretic при конфиге base -> промах (D7 не смешивается)
    meta = _old_sidecar()
    meta["text_encoder"] = "heretic"
    hit, reason = _hit(cfg, meta, tmp)
    if hit or "encoder" not in reason:
        raise AssertionError(f"тег энкодера не сверяется: hit={hit}, reason={reason!r}")
    print(f"  ok: тег энкодера heretic!=base -> промах: {reason}")

    # Новый сайдкар (с cache_key_parts) при совпадении -> попадание
    meta = _old_sidecar()
    meta["cache_key_parts"] = cfg.cache_key_parts(CROP, MD5, SHA)
    meta["encoder_dtype"] = "bfloat16"
    meta["encoder_device"] = "cpu"
    hit, reason = _hit(cfg, meta, tmp)
    if not hit:
        raise AssertionError(f"новый сайдкар с cache_key_parts не прошёл сверку: {reason!r}")
    if meta["cache_key_parts"]["schema"] != CACHE_SCHEMA:
        raise AssertionError("в cache_key_parts нет схемы")
    print("  ok: новый сайдкар (cache_key_parts + dtype/device) -> попадание")


# --------------------------------------------------------------------------- #

CASES: tuple[tuple[str, Callable[..., None]], ...] = (
    ("1. эталон имени для base/cpu/bfloat16", case_legacy_name),
    ("2. матрица суффиксов dtype/device", case_suffix_matrix),
    ("3. матрица переиспользования (бесплатные vs платные оси)", case_reuse_matrix),
    ("4. сверка сайдкара по белому списку", case_sidecar_check),
)


def main() -> int:
    failed = 0
    with tempfile.TemporaryDirectory(prefix="qwenstyle-dev-key-") as raw:
        tmp = Path(raw)
        for name, case in CASES:
            try:
                if case is case_sidecar_check:
                    case(tmp)
                else:
                    case()
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
