"""GPU-энкод промта по умолчанию: `encoder.device=auto` и подмена `pipe.text_encoder`.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_encoder_base_gpu.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (шаг 2 плана `qwen1440p-perf`): CPU-энкод Qwen3VL-8B стоил 682 s на 1440p, потому что
`encoder.device` был `cpu` по умолчанию, а `encoder.stream` для `name=base` вообще не действовал —
веса грузил diffusers. Теперь умолчание `auto` (на машине с CUDA -> `cuda`) и стриминговый
загрузчик, поэтому энкод идёт на GPU без единого `--set`, а CPU остаётся явным эскапеем.

Кейсы:

1. **cpu — эталон имени.** `--set encoder.device=cpu` даёт ровно историческое имя записи
   `embeds_<md5>_<sha>_1376x768.pt` (иначе старые CPU-записи в `cache/` стали бы мёртвыми, R1).
2. **cuda в имени.** `encoder.device=cuda` -> `-dev-cuda` в имени и `encoder_device: cuda`
   в `cache_key_parts` (GPU-энкод != CPU-энкод по округлениям, R1).
3. **auto.** На машине с CUDA `auto` разрешается в `cuda`, включает `stream=True` и `prefetch=4`
   (веса 17.5 ГБ в 16 ГБ VRAM целиком не влезают); на машине без CUDA — в `cpu` (ветка `SKIP`).
4. **мини-модель на GPU.** `streaming.load_text_encoder(device="cuda")` отдаёт модель с
   `model.device.type == "cuda"`, пустыми параметрами на cuda и logits **бит-в-бит** как у
   обычной загрузки на ту же карту (стриминг — раскладка памяти, а не другая математика).
5. **`--set encoder.device=cpu` возвращает старое поведение.** Устройство `cpu`, `stream`/`prefetch`
   не включаются, имя записи — прежнее; и это ровно то, что читает бекенд (`_encoder_device`).

Кейсы 3-4 требуют CUDA: без неё печатают `SKIP` и не считаются провалом (exit 0).
Временные каталоги — только в системном temp; `sys.dont_write_bytecode` — чтобы в пакете
не осталось `__pycache__`.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, TESTS_DIR):
    if str(_path) not in sys.path:               # запуск из любого cwd + импорт фикстур теста
        sys.path.insert(0, str(_path))
sys.dont_write_bytecode = True                   # тест не должен оставлять __pycache__ в пакете
os.environ.setdefault("HF_HUB_OFFLINE", "1")     # R3: только локальные веса, никакой сети

import test_streaming as fixtures                # noqa: E402 — фикстуры мини-модели (не копия)

from qwenstyle.config import (                   # noqa: E402 — после правки sys.path
    AUTO_ENCODER_PREFETCH,
    EncoderConfig,
    RunConfig,
    auto_encoder_device,
    encoder_device_canon,
)

def _pinned_base():
    """Пресет `base` со штатным энкодером: с D7 дефолтом в base.yaml стал heretic, а тест
    проверяет именно поведение штатного Qwen3VL-8B (легаси-имя без `_te-`, `-dev-cuda` и т.п.)."""
    from qwenstyle.config import RunConfig as _RunConfig

    return _RunConfig.from_profile("base").merged(**{"encoder.name": "base"}).merged(**{"encoder.name": "base"})


#: Историческое имя записи из `cache/` — эталон байт-в-байт (AGENTS.md R1).
LEGACY_NAME = "embeds_2d183e29dc3e_c8cb6f3b62f8_1376x768.pt"
MD5 = "2d183e29dc3e"
SHA = "c8cb6f3b62f8"
CROP = (1376, 768)


def _has_cuda() -> bool:
    import torch

    return bool(torch.cuda.is_available())


# --------------------------------------------------------------------------- #
# Кейс 1: cpu — эталон имени
# --------------------------------------------------------------------------- #

def case_cpu_legacy(tmp: Path) -> None:
    """`encoder.device=cpu` — прежнее имя кэша и прежние значения флагов (эскапей)."""
    cfg = _pinned_base().merged(**{"encoder.device": "cpu"})
    if cfg.encoder.device != "cpu":
        raise AssertionError(f"encoder.device={cfg.encoder.device!r}, ожидалось cpu")
    if cfg.encoder.stream is not False or int(cfg.encoder.prefetch) != 0:
        raise AssertionError(f"cpu не должен включать GPU-конвейер: stream={cfg.encoder.stream!r}, "
                             f"prefetch={cfg.encoder.prefetch!r}")
    key = cfg.embeds_key(CROP, MD5, SHA)
    if key != LEGACY_NAME:
        raise AssertionError(f"cpu-имя {key!r} != эталона {LEGACY_NAME!r}")
    if cfg.embeds_path(CROP, MD5, SHA).name != LEGACY_NAME:
        raise AssertionError("embeds_path не совпадает с эталоном имени")
    parts = cfg.cache_key_parts(CROP, MD5, SHA)
    if parts["encoder_device"] != "cpu":
        raise AssertionError(f"cache_key_parts[encoder_device]={parts['encoder_device']!r}")
    resolved = cfg.encoder.resolved()
    if resolved is not cfg.encoder and resolved.device != "cpu":
        raise AssertionError(f"resolved() сдвинул явный cpu на {resolved.device!r}")
    print(f"  ok: device=cpu -> {LEGACY_NAME} (stream/prefetch не включаются)")


# --------------------------------------------------------------------------- #
# Кейс 2: cuda в имени
# --------------------------------------------------------------------------- #

def case_cuda_name(tmp: Path) -> None:
    """`encoder.device=cuda` — `-dev-cuda` в имени и в ключе, `auto` разрешается так же."""
    cfg = _pinned_base().merged(**{"encoder.device": "cuda"})
    key = cfg.embeds_key(CROP, MD5, SHA)
    if "-dev-cuda" not in key:
        raise AssertionError(f"в имени нет -dev-cuda: {key!r}")
    if key == LEGACY_NAME:
        raise AssertionError("cuda-запись получила историческое имя (дыра R1)")
    parts = cfg.cache_key_parts(CROP, MD5, SHA)
    if parts["encoder_device"] != "cuda":
        raise AssertionError(f"cache_key_parts[encoder_device]={parts['encoder_device']!r}")

    enc = EncoderConfig(device="cuda")
    if enc.key_suffix != "-dev-cuda":
        raise AssertionError(f"EncoderConfig(device=cuda).key_suffix={enc.key_suffix!r}")
    if encoder_device_canon("cuda:0") != "cuda":
        raise AssertionError("канон cuda:0 != cuda")
    print(f"  ok: device=cuda -> {key}")


# --------------------------------------------------------------------------- #
# Кейс 3: auto
# --------------------------------------------------------------------------- #

def case_auto_device(tmp: Path) -> None:
    """`auto` -> cuda + stream + prefetch=4 (или cpu без CUDA).

    Проверяется и ленивость: `torch` не должен импортироваться при сборке конфига, пока никто
    не спросил устройство (иначе `models list` тянул бы 2 ГБ библиотек).
    """
    import subprocess

    probe = (
        "import sys\n"
        "from qwenstyle.config import RunConfig\n"
        "cfg = RunConfig.from_profile('strong-release')\n"
        "assert cfg.encoder.device == 'auto', cfg.encoder.device\n"
        "assert 'torch' not in sys.modules, 'torch импортирован при сборке конфига'\n"
        "print(cfg.encoder.device)\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], cwd=str(REPO_ROOT),
                          capture_output=True, text=True, encoding="utf-8")
    if done.returncode != 0:
        raise AssertionError(f"сборка конфига с device=auto: код {done.returncode}, "
                             f"stderr={done.stderr.strip()[-500:]}")
    print(f"  ok: пресет strong-release держит device=auto и не тянет torch при сборке конфига")

    # штатный энкодер: кейс про `auto` у Qwen3VL-8B, а не про дефолт-heretic из base.yaml
    cfg = RunConfig.from_profile("strong-release").merged(**{"encoder.name": "base"})
    enc = cfg.encoder
    if not enc.is_auto_device:
        raise AssertionError(f"EncoderConfig().device={enc.device!r}, ожидалось auto")
    resolved = enc.resolved()
    actual = auto_encoder_device()
    if resolved.device != actual:
        raise AssertionError(f"auto разрешился в {resolved.device!r}, а auto_encoder_device() "
                             f"говорит {actual!r}")
    if actual != "cuda":
        print(f"  SKIP: CUDA недоступна, auto -> {actual}; GPU-ветка (stream/prefetch, имя записи) "
              f"не проверена")
        if resolved.stream is not False or int(resolved.prefetch) != 0:
            raise AssertionError(f"без CUDA включать конвейер нельзя: stream={resolved.stream!r}, "
                                 f"prefetch={resolved.prefetch!r}")
        if "-dev-" in resolved.key_suffix:
            raise AssertionError(f"без CUDA сегмент device не должен появляться: "
                                 f"{resolved.key_suffix!r}")
        return

    if resolved.device != "cuda":
        raise AssertionError(f"auto -> {resolved.device!r}, ожидалось cuda")
    if resolved.stream is not True:
        raise AssertionError(f"auto+cuda не включил stream: {resolved.stream!r}")
    if int(resolved.prefetch) != AUTO_ENCODER_PREFETCH:
        raise AssertionError(f"auto+cuda: prefetch={resolved.prefetch!r}, "
                             f"ожидалось {AUTO_ENCODER_PREFETCH}")
    if resolved.key_suffix != "-dev-cuda":
        raise AssertionError(f"auto+cuda: key_suffix={resolved.key_suffix!r}, ожидалось -dev-cuda")
    if cfg.embeds_key(CROP, MD5, SHA) != f"embeds_{MD5}_{SHA}-dev-cuda_{CROP[0]}x{CROP[1]}.pt":
        raise AssertionError(f"имя записи по auto: {cfg.embeds_key(CROP, MD5, SHA)!r}")

    # Явный prefetch главнее умолчания (пользователь сказал «иное»).
    custom = (RunConfig.from_profile("strong-release")
              .merged(**{"encoder.name": "base", "encoder.prefetch": 2}).encoder.resolved())
    if int(custom.prefetch) != 2:
        raise AssertionError(f"явный prefetch=2 перетёрт: {custom.prefetch!r}")
    print(f"  ok: auto -> cuda, stream=True, prefetch={resolved.prefetch} "
          f"(явный prefetch сохраняется), ключ ...-dev-cuda_...pt")


# --------------------------------------------------------------------------- #
# Кейс 4: мини-модель через стриминг на GPU
# --------------------------------------------------------------------------- #

def case_mini_encoder_cuda(tmp: Path) -> None:
    """`load_text_encoder(device="cuda")`: `model.device`=cuda, параметры без хранилища, logits бит-в-бит.

    Это тот же путь, которым бекенд собирает штатный Qwen3VL-8B (`_load_base_encoder_gpu`),
    поэтому проверяем и численность: стриминг — раскладка памяти, а не другая математика.
    """
    import torch
    from transformers import Qwen3ForCausalLM

    from qwenstyle.streaming import is_streamed, load_text_encoder

    if not torch.cuda.is_available():
        print("  SKIP: CUDA недоступна — GPU-стриминг мини-модели не проверен")
        return

    root = tmp / "encoder-gpu"
    fixtures.build_tiny(root)                    # config.json + model.safetensors (фикстура шага 1)
    streamed = load_text_encoder(Qwen3ForCausalLM, root, dtype="bfloat16",
                                 local_files_only=True, stream=True, device="cuda",
                                 prefetch=2, pin_memory=True, lookahead=1)
    plain = None
    try:
        if not is_streamed(streamed):
            raise AssertionError("stream=True + device=cuda вернул не стриминговую модель")
        if streamed.device.type != "cuda":
            raise AssertionError(f"model.device={streamed.device!r}, ожидалось cuda")
        with_storage = [(name, tuple(param.shape), str(param.device))
                        for name, param in streamed.named_parameters() if param.numel() != 0]
        if with_storage:
            raise AssertionError(f"параметры с хранилищем: {with_storage[:5]}")
        empty_devices = sorted({str(param.device) for _, param in streamed.named_parameters()})
        if not all(item.startswith("cuda") for item in empty_devices):
            raise AssertionError(f"пустые параметры не на cuda: {empty_devices}")

        torch.manual_seed(23)
        input_ids = torch.randint(0, fixtures.TINY["vocab_size"], (1, 6)).cuda()
        with torch.no_grad():
            got = streamed(input_ids=input_ids).logits
        if tuple(got.shape) != (1, 6, fixtures.TINY["vocab_size"]):
            raise AssertionError(f"logits {tuple(got.shape)}")

        # Контроль: обычная загрузка + `.to("cuda")` на ту же карту (та же математика, другой layout).
        plain = load_text_encoder(Qwen3ForCausalLM, root, dtype="bfloat16",
                                  local_files_only=True).eval().to("cuda")
        with torch.no_grad():
            want = plain(input_ids=input_ids).logits
        if not torch.equal(want, got):
            diff = (want.float() - got.float()).abs().max().item()
            raise AssertionError(f"стриминговый GPU-энкод не бит-в-бит обычному (max abs diff {diff})")
        print(f"  ok: model.device={streamed.device}, параметры без хранилища на cuda, "
              f"logits {tuple(got.shape)} бит-в-бит обычной загрузке на ту же карту")
    finally:
        fixtures._close(streamed)
        if plain is not None:
            fixtures._close(plain)


# --------------------------------------------------------------------------- #
# Кейс 5: `--set encoder.device=cpu` — старое поведение
# --------------------------------------------------------------------------- #

def case_escape_cpu(tmp: Path) -> None:
    """`--set encoder.device=cpu` даёт прежний конфиг и прежнее имя записи (обратная совместимость)."""
    from qwenstyle.backends.qwen_image21 import _encoder_device

    profile = _pinned_base()
    explicit = profile.merged(**{"encoder.device": "cpu"})
    forced = profile.merged(**{"encoder.device": "cuda"})
    if _encoder_device(profile) != auto_encoder_device():
        raise AssertionError(f"_encoder_device(профиль auto)={_encoder_device(profile)!r}, "
                             f"ожидалось {auto_encoder_device()!r}")
    if _encoder_device(explicit) != "cpu":
        raise AssertionError(f"_encoder_device(--set encoder.device=cpu)="
                             f"{_encoder_device(explicit)!r}")
    if _encoder_device(forced) != "cuda":
        raise AssertionError(f"_encoder_device(--set encoder.device=cuda)="
                             f"{_encoder_device(forced)!r}")
    if explicit.embeds_key(CROP, MD5, SHA) != LEGACY_NAME:
        raise AssertionError(f"cpu-эскапей изменил имя записи: "
                             f"{explicit.embeds_key(CROP, MD5, SHA)!r}")
    parts = explicit.cache_key_parts(CROP, MD5, SHA)
    if parts["encoder_device"] != "cpu":
        raise AssertionError(f"cache_key_parts[encoder_device]={parts['encoder_device']!r}")
    print(f"  ok: --set encoder.device=cpu -> поведение и имя записи прежние ({LEGACY_NAME})")


# --------------------------------------------------------------------------- #

CASES: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("1. device=cpu -> эталонное имя кэша и прежние флаги", case_cpu_legacy),
    ("2. device=cuda -> -dev-cuda в имени и в ключе", case_cuda_name),
    ("3. auto -> cuda (stream=True, prefetch=4) без импорта torch при сборке конфига",
     case_auto_device),
    ("4. мини-модель через streaming.load_text_encoder(device=cuda)", case_mini_encoder_cuda),
    ("5. --set encoder.device=cpu -> старое поведение и старое имя записи", case_escape_cpu),
)


def main() -> int:
    failed = 0
    base = fixtures.temp_base()                  # системный temp, а не корень репозитория
    with tempfile.TemporaryDirectory(prefix="qwenstyle-enc-gpu-", dir=str(base)) as raw:
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
