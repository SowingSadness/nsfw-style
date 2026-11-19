"""Тесты подключения режима `encoder.stream` (диск-стриминг весов) — обычный скрипт, без pytest.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_encoder_stream.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Кейсы (план шага 2):

1. `encoder.stream` есть в `EncoderConfig` и включается через `--set`-механизм (`merged`);
   у явного `device=cpu` по умолчанию — `False`; ключ кэша эмбеддингов (`embeds_key`, `cache_key_parts`)
   при `stream=True` **тот же**, что и при `stream=False` (стриминг — раскладка памяти, а не другая
   математика, R1); предупреждения «для base режим пропущен» больше нет — режим действует;
2. базовый пресет (`base`): `stream=False` по умолчанию, с `--set` — `True`, ключ кэша тоже не едет;
3. мини-энкодер из временного каталога, загруженный через `streaming.load_text_encoder(stream=True)`,
   отдаёт модель без хранилища в параметрах (пустые на `device`, 0 байт) и работающим forward,
   а его `hidden_states` **бит-в-бит** совпадают с обычной загрузкой;
4. **регрессия B1**: тот же мини-энкодер, но загруженный через `backends.flux2._load_text_encoder`
   (`encoder.stream=True`), переживает `streaming.move_to(...)` — который вызывают
   `flux2._place`/`qwen_image21.prepare` вместо `pipe.text_encoder.to("cpu")` — и считает forward;
   для обычной (не стриминговой) модели `move_to` реально переносит параметры;
5. **регрессия B1-bis**: стриминговая модель сообщает `device=cpu` (`model.device` и
   `diffusers…get_module_execution_device`), иначе diffusers уводит входы на `meta` и падает
   (`Cannot copy out of meta tensor`); после forward параметры снова без хранилища (`numel()==0`),
   а второй forward даёт те же logits;

Кейсы 6-10 (нит N11/N12/N13/N14/N5/N8 после ревью):

6. `load_text_encoder(stream=True)` принимает `device='cpu'` и `device='cuda'`/`'cuda:0'` (веса
   живут в VRAM только на время forward модуля; отсутствие CUDA -> `ValueError`); любое другое
   устройство (`mps`, `xpu`, `npu`) -> `ValueError` с внятным текстом до чтения шардов;
   `stream=False` при том же `device` работает как раньше (аргумент не используется);
7. если пост-обработка (`_free_meta_parameters`) упала, исключение пробрасывается, а mmap-файлы
   шардов закрыты (`ShardedWeights.closed == True`, чтение -> `RuntimeError`);
8. `ShardedWeights.file_dtype()`: без имени — dtype всего однородного набора (кэшируется, работает
   и после `close()`), с именем — dtype одного тензора; искусственный смешанный набор -> `ValueError`
   с перечнем dtype и примером имени;
9. `always_call=True`: forward модуля бросил исключение -> его параметры всё равно освобождены
   (`numel()==0`), а после восстановления forward модель считает снова;
10. предупреждение «для base веса грузит diffusers — режим пропущен» удалено вместе с `_STREAM_WARNED`:
    `stream` для `base` теперь действует (бекенд подменяет `pipe.text_encoder`), а умолчание
    `device=auto` -> cuda ещё и включает конвейер (`stream=True`, `prefetch=4`); заодно pin сборки
    конвейера (`encoder_pin_memory`) больше не зависит от `offload=group` — включён по умолчанию
    и выключается только `--set encoder.pin_memory=0`.

Билдер мини-модели переиспользуется из `tests/test_streaming.py` (импортом, не копией). Временные
каталоги — только в системном temp; `sys.dont_write_bytecode` — чтобы в пакете не осталось
`__pycache__`.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_DIR = Path(__file__).resolve().parent
for _path in (REPO_ROOT, TESTS_DIR):
    if str(_path) not in sys.path:               # запуск из любого cwd + импорт фикстур теста
        sys.path.insert(0, str(_path))
sys.dont_write_bytecode = True                   # тест не должен оставлять __pycache__ в пакете
os.environ.setdefault("HF_HUB_OFFLINE", "1")     # R3: только локальные веса, никакой сети

import test_streaming as fixtures                # noqa: E402 — фикстуры мини-модели (не копия)

from qwenstyle.config import CACHE_SCHEMA, RunConfig   # noqa: E402 — после правки sys.path

#: Параметры ключа кэша для сверки имён файлов.
CROP = (864, 480)
MD5 = "0123456789ab"
SHA = "ba9876543210"

def _pinned_base():
    """Пресет `base` со штатным энкодером: с D7 дефолтом в base.yaml стал heretic, а тест
    проверяет именно поведение штатного Qwen3VL-8B (легаси-имя без `_te-`, `-dev-cuda` и т.п.)."""
    from qwenstyle.config import RunConfig as _RunConfig

    return _RunConfig.from_profile("base").merged(**{"encoder.name": "base"}).merged(**{"encoder.name": "base"})



# --------------------------------------------------------------------------- #
# Кейс 1: конфиг + неизменность ключа кэша (Qwen-Image-2.1, `base`)
# --------------------------------------------------------------------------- #

def case_config_and_cache_key(tmp: Path) -> None:
    """`stream` включается через `merged`, у явного cpu выключен, ключ кэша не меняется."""
    plain = _pinned_base().merged(**{"encoder.device": "cpu"})
    if plain.encoder.stream is not False:
        raise AssertionError(f"у явного device=cpu encoder.stream={plain.encoder.stream!r}, "
                             f"ожидалось False")

    # `stream=true` для base больше не мусор: веса грузит наш стриминговый загрузчик, поэтому
    # предупреждения «режим пропущен» быть не должно (оно удалено вместе с `_STREAM_WARNED`).
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):        # ловим любые строки, которыми конфиг шумит
        streamed = _pinned_base().merged(**{"encoder.device": "cpu",
                                                             "encoder.stream": "true"})
    if streamed.encoder.stream is not True:
        raise AssertionError(f"--set encoder.stream=true -> {streamed.encoder.stream!r}, "
                             f"ожидалось True")
    noise = buf.getvalue()
    if "[te] streaming:" in noise:
        raise AssertionError(f"для encoder.name=base всё ещё печатается предупреждение, что "
                             f"режим пропущен (он действует): {noise!r}")
    print("  ok: base.encoder.stream: у явного cpu False, с --set True; предупреждения нет")

    alone = _pinned_base().merged(**{"encoder.device": "cpu",
                                                    "encoder.stream": "false"})
    if alone.encoder.stream is not False:
        raise AssertionError(f"--set encoder.stream=false -> {alone.encoder.stream!r}")

    key_plain = plain.embeds_key(CROP, MD5, SHA)
    key_stream = streamed.embeds_key(CROP, MD5, SHA)
    if key_stream != key_plain:
        raise AssertionError(f"ключ кэша изменился от stream: {key_stream!r} != {key_plain!r}")
    if "stream" in key_plain:
        raise AssertionError(f"в имени файла кэша появился 'stream': {key_plain!r}")

    parts_plain = plain.cache_key_parts(CROP, MD5, SHA)
    parts_stream = streamed.cache_key_parts(CROP, MD5, SHA)
    if parts_stream != parts_plain:
        diff = {k: (parts_plain.get(k), parts_stream.get(k))
                for k in set(parts_plain) | set(parts_stream)
                if parts_plain.get(k) != parts_stream.get(k)}
        raise AssertionError(f"cache_key_parts изменились от stream: {diff}")
    if any("stream" in key for key in parts_stream):
        raise AssertionError(f"в cache_key_parts появился stream: {sorted(parts_stream)}")
    if parts_plain.get("schema") != CACHE_SCHEMA:
        raise AssertionError(f"схема кэша {parts_plain.get('schema')!r} != {CACHE_SCHEMA!r}")
    print(f"  ok: embeds_key и cache_key_parts при stream=True те же: {key_plain}")


# --------------------------------------------------------------------------- #
# Кейс 2: базовый пресет (`base`) — `stream` по умолчанию выключен
# --------------------------------------------------------------------------- #

def case_preset_stream(tmp: Path) -> None:
    """`base`: `stream` по умолчанию False, с `--set` True, ключ кэша не едет."""
    plain = RunConfig.from_profile("base")
    if plain.encoder.stream is not False:
        raise AssertionError(f"base: encoder.stream={plain.encoder.stream!r} по умолчанию")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        streamed = RunConfig.from_profile("base").merged(**{"encoder.stream": "true"})
    if streamed.encoder.stream is not True:
        raise AssertionError(f"base --set encoder.stream=true -> {streamed.encoder.stream!r}")
    if buf.getvalue():
        raise AssertionError(f"для энкодера {plain.encoder.name!r} предупреждения быть не должно: "
                             f"{buf.getvalue()!r}")
    key_plain = plain.embeds_key((1376, 768), MD5, SHA)
    key_stream = streamed.embeds_key((1376, 768), MD5, SHA)
    if key_stream != key_plain:
        raise AssertionError(f"ключ кэша базового пресета изменился от stream: "
                             f"{key_stream!r} != {key_plain!r}")
    print(f"  ok: base (encoder={plain.encoder.name}): False -> True через --set, ключ {key_plain}")


# --------------------------------------------------------------------------- #
# Кейс 3: мини-энкодер через streaming.load_text_encoder(stream=True)
# --------------------------------------------------------------------------- #

def case_mini_encoder(tmp: Path) -> None:
    """Модель на `meta` + работающий forward, бит-в-бит как обычная загрузка."""
    import torch
    from transformers import Qwen3ForCausalLM

    from qwenstyle.streaming import load_text_encoder

    root = tmp / "encoder-mini"
    fixtures.build_tiny(root)                    # config.json + model.safetensors (фикстура шага 1)
    torch.manual_seed(7)
    input_ids = torch.randint(0, fixtures.TINY["vocab_size"], (1, 6))

    plain = load_text_encoder(Qwen3ForCausalLM, root, dtype="bfloat16",
                              local_files_only=True).eval()
    try:
        streamed = load_text_encoder(Qwen3ForCausalLM, root, dtype="bfloat16",
                                     local_files_only=True, stream=True)
        try:
            # Инвариант после `load_text_encoder(stream=True)`: параметров с хранилищем нет,
            # все пустые (0 байт) и на `device` — так модель сообщает device=cpu (B1-bis).
            not_meta = [(name, tuple(param.shape), str(param.device))
                        for name, param in streamed.named_parameters() if param.numel() != 0]
            if not_meta:
                raise AssertionError(f"у параметров осталось хранилище: {not_meta[:5]}")
            wrong_device = [(name, str(param.device)) for name, param in streamed.named_parameters()
                            if param.device.type != "cpu"]
            if wrong_device:
                raise AssertionError(f"пустые параметры не на device=cpu: {wrong_device[:5]}")
            if streamed.training:
                raise AssertionError("stream-загрузка обязана вернуть eval()-модель")
            with torch.no_grad():
                expected = plain(input_ids=input_ids, output_hidden_states=True)
                actual = streamed(input_ids=input_ids, output_hidden_states=True)
            if len(expected.hidden_states) != len(actual.hidden_states):
                raise AssertionError(f"слоёв {len(actual.hidden_states)}, "
                                     f"ожидалось {len(expected.hidden_states)}")
            for layer, (want, got) in enumerate(zip(expected.hidden_states, actual.hidden_states)):
                if not torch.equal(want, got):
                    diff = (want.float() - got.float()).abs().max().item()
                    raise AssertionError(f"hidden_states[{layer}] не бит-в-бит "
                                         f"(max abs diff {diff})")
            print(f"  ok: {len(list(streamed.named_parameters()))} параметров без хранилища "
                  f"(0 байт, device=cpu), forward "
                  f"{tuple(actual.logits.shape)} и {len(actual.hidden_states)} hidden states "
                  f"бит-в-бит как у обычной загрузки")
        finally:
            fixtures._close(streamed)            # закрыть mmap + отпустить модель
    finally:
        fixtures._close(plain)


# --------------------------------------------------------------------------- #
# Кейс 4 (регрессия B1): стриминговый энкодер проходит через бэкенд и `move_to`
# --------------------------------------------------------------------------- #

def case_streamed_encoder_placement(tmp: Path) -> None:
    """B1: `flux2._load_text_encoder(stream=True)` + `move_to` не падают на `meta`-модели.

    Раньше `flux2._place`/`qwen_image21.prepare` делали `pipe.text_encoder.to("cpu")`, что на
    стриминговой модели падает с `NotImplementedError: Cannot copy out of meta tensor`.
    """
    import torch

    from qwenstyle import streaming
    from qwenstyle.backends import flux2

    root = tmp / "encoder-b1"
    fixtures.build_tiny(root)                    # config.json + model.safetensors (фикстура шага 1)
    # `encoder.device` явно cpu: после перевода пресета на `auto` резолв дал бы cuda,
    # а этот кейс проверяет размещение/move_to, а не GPU-путь (тест должен работать и без CUDA).
    stream_cfg = RunConfig.from_profile("base").merged(**{
        "encoder.stream": "true",
        "encoder.device": "cpu",
        "encoder.local_dir": str(root),
    })

    model = flux2._load_text_encoder(stream_cfg, "Qwen3ForCausalLM")
    try:
        if not streaming.is_streamed(model):
            raise AssertionError(f"is_streamed({type(model).__name__}) после stream-загрузки == False")
        streaming.move_to(model, "cpu")          # до фикса здесь был NotImplementedError
        not_meta = [(name, tuple(param.shape), str(param.device))
                    for name, param in model.named_parameters() if param.numel() != 0]
        if not_meta:
            raise AssertionError(f"move_to материализовал параметры (ожидался no-op): {not_meta[:5]}")
        torch.manual_seed(11)
        input_ids = torch.randint(0, fixtures.TINY["vocab_size"], (1, 5))
        with torch.no_grad():
            out = model(input_ids=input_ids)
        expected = (1, 5, fixtures.TINY["vocab_size"])
        if not isinstance(out.logits, torch.Tensor) or tuple(out.logits.shape) != expected:
            raise AssertionError(f"forward стриминговой модели: logits "
                                 f"{tuple(out.logits.shape) if hasattr(out.logits, 'shape') else out.logits!r}, "
                                 f"ожидалось {expected}")
        print(f"  ok: flux2._load_text_encoder(stream=True) -> is_streamed=True, "
              f"move_to(no-op) без ошибки, forward {tuple(out.logits.shape)}")
    finally:
        fixtures._close(model)

    # Контроль: для обычной (не стриминговой) модели move_to — это настоящий `.to()`.
    # Грузим её тем же `load_text_encoder`, а не через flux2: у бэкенда в ветке `local_dir`
    # `subfolder=None`, а `from_pretrained(..., subfolder=None)` в transformers 5.22 падает.
    from transformers import Qwen3ForCausalLM

    plain = streaming.load_text_encoder(Qwen3ForCausalLM, root, dtype="bfloat16",
                                       local_files_only=True).eval()
    try:
        if streaming.is_streamed(plain):
            raise AssertionError("обычная загрузка помечена как стриминговая")
        streaming.move_to(plain, "cpu")          # контроль: для обычной модели это настоящий .to()
        devices = sorted({str(param.device) for param in plain.parameters()})
        if devices != ["cpu"]:
            raise AssertionError(f"после move_to(plain, 'cpu') параметры на {devices}")
        print(f"  ok: обычная модель — is_streamed=False, move_to('cpu') переносит веса ({devices})")
    finally:
        fixtures._close(plain)


# --------------------------------------------------------------------------- #
# Кейс 5 (регрессия B1-bis): стриминговая модель сообщает device=cpu
# --------------------------------------------------------------------------- #

def case_streamed_reports_device(tmp: Path) -> None:
    """B1-bis: `model.device`/`get_module_execution_device` = cpu, хранилище возвращается.

    `diffusers/pipelines/flux2/pipeline_flux2_klein.py::_get_qwen3_prompt_embeds` делает
    `model_device = get_module_execution_device(text_encoder)` и уводит на него `input_ids`
    и `attention_mask`. С `meta`-параметрами это `Cannot copy out of meta tensor` в
    `transformers/masking_utils.py`; поэтому модель обязана сообщать `cpu` — и после forward тоже
    (post-forward хук не возвращает параметры в `meta`).
    """
    import torch
    from diffusers.utils.torch_utils import get_module_execution_device
    from transformers import Qwen3ForCausalLM

    from qwenstyle.streaming import load_text_encoder

    root = tmp / "encoder-b1bis"
    fixtures.build_tiny(root)                    # config.json + model.safetensors (фикстура шага 1)
    model = load_text_encoder(Qwen3ForCausalLM, root, dtype="bfloat16",
                              local_files_only=True, stream=True)
    try:
        if str(model.device) != "cpu":
            raise AssertionError(f"model.device={model.device!r}, ожидалось cpu")
        if model.dtype != torch.bfloat16:
            # dtype тоже читается из параметров: модель собрана из конфига (float32), веса приезжают
            # bf16. Совравший dtype даёт эмбеддинги float32 и падение трансформера на matmul (bf16).
            raise AssertionError(f"model.dtype={model.dtype}, ожидалось torch.bfloat16 (dtype весов)")
        execution = get_module_execution_device(model)
        if execution.type != "cpu":
            raise AssertionError(f"get_module_execution_device(model)={execution}, ожидалось cpu")

        torch.manual_seed(13)
        input_ids = torch.randint(0, fixtures.TINY["vocab_size"], (1, 5))
        with torch.no_grad():
            first = model(input_ids=input_ids).logits
        leaked = [(name, tuple(param.shape), str(param.device))
                  for name, param in model.named_parameters() if param.numel() != 0]
        if leaked:
            raise AssertionError(f"после forward параметры остались с хранилищем: {leaked[:5]}")
        if str(model.device) != "cpu":
            raise AssertionError(f"после forward model.device={model.device!r}, ожидалось cpu")
        if get_module_execution_device(model).type != "cpu":
            raise AssertionError("после forward get_module_execution_device(model) != cpu")

        with torch.no_grad():
            second = model(input_ids=input_ids).logits
        if not torch.equal(first, second):
            diff = (first.float() - second.float()).abs().max().item()
            raise AssertionError(f"второй forward разошёлся с первым (max abs diff {diff})")
        print(f"  ok: model.device=cpu, model.dtype={model.dtype}, "
              f"get_module_execution_device=cpu, после forward параметры "
              f"снова 0 байт ({len(list(model.named_parameters()))} шт.), второй forward "
              f"{tuple(second.shape)} бит-в-бит")
    finally:
        fixtures._close(model)

# --------------------------------------------------------------------------- #
# Кейс 6 (N13): стриминг — device="cpu" и "cuda", прочие отвергаются
# --------------------------------------------------------------------------- #

def case_device_validation(tmp: Path) -> None:
    """N13: `stream=True` принимает `cpu` и `cuda`, отвергает прочие устройства; `stream=False` — как раньше.

    Контракт после снятия guard'а «только cpu»: веса и на CPU, и на GPU живут в памяти лишь на
    время forward модуля (на GPU — в VRAM, диск -> pinned RAM -> H2D), поэтому `device='cuda'`
    разрешён и не должен отвергаться ещё до чтения шардов. Всё, что не `cpu`/`cuda` (`mps`, `xpu`,
    `npu`), по-прежнему `ValueError`: у стриминга нет ни pinned-стейджинга, ни H2D-копии для них.
    """
    import torch
    from transformers import Qwen3ForCausalLM

    from qwenstyle.streaming import is_streamed, load_text_encoder

    root = tmp / "encoder-n13"
    fixtures.build_tiny(root)

    # Не-cpu/cuda отвергаются до чтения шардов, с внятным текстом (что поддерживается + что дали).
    for device in ("mps", "xpu", "npu"):
        try:
            load_text_encoder(Qwen3ForCausalLM, root, dtype="bfloat16", local_files_only=True,
                              stream=True, device=device)
        except ValueError as exc:
            text = str(exc)
            for needle in ("streaming", "cpu", "cuda", device):
                if needle not in text:
                    raise AssertionError(f"ValueError без упоминания {needle!r}: {text}") from exc
        else:
            raise AssertionError(f"stream=True с device={device!r} не отвергнут")
    print("  ok: stream=True + device=mps/xpu/npu -> ValueError про поддерживаемые cpu/cuda")

    # CUDA: теперь штатный режим (веса в VRAM только на время forward модуля). Тем же путём идёт
    # `device='cuda:0'` — вид устройства разбирается по префиксу до ':'.
    if not torch.cuda.is_available():
        print("  SKIP: stream=True + device='cuda' — CUDA недоступна в этой системе (ветка не проверена)")
    else:
        for device in ("cuda", "cuda:0"):
            model = load_text_encoder(Qwen3ForCausalLM, root, dtype="bfloat16",
                                      local_files_only=True, stream=True, device=device)
            try:
                if not is_streamed(model):
                    raise AssertionError(f"stream=True с device={device!r}: is_streamed=False")
                with_storage = [(name, tuple(param.shape), str(param.device))
                                for name, param in model.named_parameters() if param.numel() != 0]
                if with_storage:
                    raise AssertionError(f"stream=True с device={device!r}: параметры с хранилищем "
                                         f"в RAM/VRAM: {with_storage[:5]}")
                empty_devices = sorted({str(param.device)
                                       for _, param in model.named_parameters()})
                if not all(dev.startswith("cuda") for dev in empty_devices):
                    raise AssertionError(f"stream=True с device={device!r}: пустые параметры на "
                                         f"{empty_devices}, ожидалось cuda")
                if not str(model.device).startswith("cuda"):
                    raise AssertionError(f"stream=True с device={device!r}: model.device="
                                         f"{model.device!r}, ожидалось cuda")
                if model.training:
                    raise AssertionError(f"stream=True с device={device!r} вернул обучаемую модель")
                print(f"  ok: stream=True + device={device!r} принят: is_streamed=True, "
                      f"параметры без хранилища на {empty_devices}")
            finally:
                fixtures._close(model)

    # Контроль: `stream=False` не трогает device вовсе — поведение прежнее (обычный from_pretrained).
    plain = load_text_encoder(Qwen3ForCausalLM, root, dtype="bfloat16",
                              local_files_only=True, device="cuda")
    try:
        devices = sorted({str(param.device) for param in plain.parameters()})
        if devices != ["cpu"]:
            raise AssertionError(f"stream=False с device='cuda' затронул раскладку: {devices}")
        print(f"  ok: stream=False + device='cuda' — как раньше, параметры на {devices}")
    finally:
        fixtures._close(plain)


# --------------------------------------------------------------------------- #
# Кейс 7 (N11): mmap закрывается, если пост-обработка упала
# --------------------------------------------------------------------------- #

def case_closed_on_post_error(tmp: Path) -> None:
    """N11: исключение в `_free_meta_parameters` -> `weights.closed` True, чтение -> `RuntimeError`.

    Без этого дескрипторы шардов жили бы до конца процесса (на Windows файл нельзя удалить/
    перезаписать, пока mmap открыт), а исключение теряло бы за собой неосвобождённый ресурс.
    """
    from transformers import Qwen3ForCausalLM

    from qwenstyle import streaming

    root = tmp / "encoder-n11"
    fixtures.build_tiny(root)

    created: list[object] = []
    original_cls = streaming.ShardedWeights
    original_free = streaming._free_meta_parameters

    class TrackingWeights(original_cls):         # noqa: N801 — копия класса ради списка инстансов
        def __init__(self, source: object) -> None:
            super().__init__(source)
            created.append(self)

    def boom(model: object, device: str, *, dtype: object = None) -> int:
        raise RuntimeError("пост-обработка упала (тест N11)")

    streaming.ShardedWeights = TrackingWeights
    streaming._free_meta_parameters = boom
    try:
        try:
            streaming.load_text_encoder(Qwen3ForCausalLM, root, dtype="bfloat16",
                                        local_files_only=True, stream=True)
        except RuntimeError as exc:
            if "N11" not in str(exc):
                raise AssertionError(f"проброшено чужое исключение: {exc}") from exc
        else:
            raise AssertionError("исключение из _free_meta_parameters не проброшено")
    finally:
        streaming.ShardedWeights = original_cls
        streaming._free_meta_parameters = original_free

    if not created:
        raise AssertionError("ShardedWeights не создавался — тест не проверил закрытие mmap")
    weights = created[-1]
    if not getattr(weights, "closed", False):
        raise AssertionError("после исключения mmap остался открытым (weights.closed == False)")
    try:
        weights["model.norm.weight"]
    except RuntimeError as exc:
        if "закрыт" not in str(exc):
            raise AssertionError(f"после закрытия ожидался RuntimeError про close(): {exc}") from exc
    else:
        raise AssertionError("закрытая карта продолжает читать веса")
    print(f"  ok: исключение в пост-обработке проброшено, weights.closed=True "
          f"({type(weights).__name__}, инстансов: {len(created)}), чтение -> RuntimeError")


# --------------------------------------------------------------------------- #
# Кейс 8 (N12): file_dtype честен на смешанном dtype
# --------------------------------------------------------------------------- #

def case_file_dtype(tmp: Path) -> None:
    """N12: `file_dtype()` = dtype всего однородного набора; смешанный набор -> `ValueError`.

    Раньше возвращался dtype первого тензора: на модели с fp32-нормами и bf16-матрицами
    `_free_meta_parameters` делал бы параметры не того типа, а `model.dtype` врал бы.
    """
    import torch
    from safetensors.torch import save_file

    from qwenstyle.streaming import ShardedWeights, StreamSource, resolve_stream_source

    root = tmp / "encoder-n12"
    fixtures.build_tiny(root)                    # fp32: save_pretrained пишет dtype модели

    weights = ShardedWeights(resolve_stream_source(root))
    try:
        whole = weights.file_dtype()
        if whole != "float32":
            raise AssertionError(f"file_dtype() однородной мини-модели = {whole!r}, ожидалось float32")
        one = weights.file_dtype("model.norm.weight")
        if one != whole:
            raise AssertionError(f"file_dtype('model.norm.weight') = {one!r}, ожидалось {whole!r}")
        try:
            weights.file_dtype("no.such.weight")
        except KeyError:
            pass
        else:
            raise AssertionError("file_dtype(неизвестное имя) не дал KeyError")
    finally:
        weights.close()
    if weights.file_dtype() != whole:
        raise AssertionError("кэш file_dtype не выжил close() — повторный вызов пытается читать файл")
    print(f"  ok: file_dtype()={whole!r} для всех тензоров, кэшируется и работает после close()")

    mixed = tmp / "encoder-n12-mixed"
    mixed.mkdir(parents=True, exist_ok=True)
    shard = mixed / "model.safetensors"
    save_file({"a.weight": torch.zeros(2, dtype=torch.float32),
               "b.weight": torch.zeros(2, dtype=torch.bfloat16)}, str(shard))
    mixed_weights = ShardedWeights(StreamSource(shards=(shard,), weight_map=None, dtype=None))
    try:
        try:
            mixed_weights.file_dtype()
        except ValueError as exc:
            text = str(exc)
            for needle in ("float32", "bfloat16", "a.weight", "b.weight"):
                if needle not in text:
                    raise AssertionError(f"в ValueError нет {needle!r}: {text}") from exc
            print(f"  ok: смешанный набор -> ValueError: {text}")
        else:
            raise AssertionError("смешанный по dtype набор не дал ValueError (dtype первого тензора?)")
        # Проверка одного тензора по имени продолжает работать и на смешанном наборе.
        if mixed_weights.file_dtype("a.weight") != "float32":
            raise AssertionError("file_dtype('a.weight') != float32")
        if mixed_weights.file_dtype("b.weight") != "bfloat16":
            raise AssertionError("file_dtype('b.weight') != bfloat16")
        print("  ok: file_dtype(имя) на смешанном наборе отдаёт dtype конкретного тензора")
    finally:
        mixed_weights.close()


# --------------------------------------------------------------------------- #
# Кейс 9 (N5): always_call освобождает веса при исключении в forward
# --------------------------------------------------------------------------- #

def case_always_call(tmp: Path) -> None:
    """N5: forward модуля бросил исключение -> его параметры всё равно освобождены (0 байт).

    Без `always_call=True` post-forward хук не вызывался, и веса упавшего модуля оставались в RAM
    до конца процесса (у 32B-энкодера это гигабайты).
    """
    import torch
    from transformers import Qwen3ForCausalLM

    from qwenstyle.streaming import load_text_encoder

    root = tmp / "encoder-n5"
    fixtures.build_tiny(root)
    model = load_text_encoder(Qwen3ForCausalLM, root, dtype="bfloat16",
                              local_files_only=True, stream=True)
    try:
        head = model.lm_head
        original = head.forward

        def boom(*args: object, **kwargs: object) -> object:
            raise RuntimeError("forward упал (тест N5)")

        head.forward = boom
        torch.manual_seed(17)
        input_ids = torch.randint(0, fixtures.TINY["vocab_size"], (1, 4))
        try:
            try:
                with torch.no_grad():
                    model(input_ids=input_ids)
            except RuntimeError as exc:
                if "N5" not in str(exc):
                    raise AssertionError(f"проброшено чужое исключение: {exc}") from exc
            else:
                raise AssertionError("исключение из forward не проброшено")
        finally:
            head.forward = original

        leaked = [(name, tuple(param.shape)) for name, param in model.named_parameters()
                  if param.numel() != 0]
        if leaked:
            raise AssertionError(f"после исключения веса остались в RAM: {leaked[:5]}")
        leaked_head = [name for name, param in head.named_parameters() if param.numel() != 0]
        if leaked_head:
            raise AssertionError(f"веса упавшего модуля не освобождены: {leaked_head}")

        with torch.no_grad():
            out = model(input_ids=input_ids)
        expected = (1, 4, fixtures.TINY["vocab_size"])
        if tuple(out.logits.shape) != expected:
            raise AssertionError(f"после восстановления forward logits {tuple(out.logits.shape)}, "
                                 f"ожидалось {expected}")
        print(f"  ok: forward lm_head бросил исключение -> веса освобождены (0 байт), "
              f"следующий forward {tuple(out.logits.shape)}")
    finally:
        fixtures._close(model)


# --------------------------------------------------------------------------- #
# Кейс 10 (N8): предупреждение для base не зависит от порядка кейсов
# --------------------------------------------------------------------------- #

def case_base_stream_effective(tmp: Path) -> None:
    """10: предупреждение удалено, `stream` для base действует, `auto` включает конвейер.

    Раньше для `name=base` печаталось «веса грузит diffusers — режим пропущен», потому что энкодер
    собирал сам пайплайн. Теперь `qwen_image21.prepare` подменяет `pipe.text_encoder` стриминговой
    моделью, поэтому `stream` — рабочий флаг, а умолчание `device=auto` на машине с CUDA даёт
    `stream=True` и `prefetch=4` без единого `--set`. Там же проверяется pin сборки конвейера:
    `offload=group` его больше не выключает (pin — транспорт, а не математика).
    """
    from qwenstyle import config as qconfig

    if hasattr(qconfig, "_STREAM_WARNED"):
        raise AssertionError("в config.py остался _STREAM_WARNED — предупреждение не удалено")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cfg = _pinned_base().merged(**{"encoder.stream": "true"})
    if buf.getvalue():
        raise AssertionError(f"сборка конфига с stream=true что-то печатает: {buf.getvalue()!r}")
    if cfg.encoder.stream is not True:
        raise AssertionError(f"encoder.stream={cfg.encoder.stream!r}, ожидалось True")

    enc = cfg.encoder.resolved()
    if enc.stream is not True:
        raise AssertionError(f"resolved().stream={enc.stream!r}, ожидалось True")
    if enc.device == "cuda" and int(enc.prefetch) != qconfig.AUTO_ENCODER_PREFETCH:
        raise AssertionError(f"auto+cuda: prefetch={enc.prefetch!r}, "
                             f"ожидалось {qconfig.AUTO_ENCODER_PREFETCH}")
    if enc.device == "cpu" and int(enc.prefetch) != 0:
        raise AssertionError(f"без CUDA конвейер включать нельзя: prefetch={enc.prefetch!r}")

    # Pin — транспорт (бит-в-бит), поэтому `offload=group` его не выключает: иначе staging шёл бы
    # через pageable RAM даже там, где pinned выделяется. Явное выключение — только `--set`.
    group_cfg = _pinned_base().merged(**{"generation.offload": "group"})
    if group_cfg.generation.offload != "group" or group_cfg.encoder_pin_memory is not True:
        raise AssertionError(f"offload=group выключил pin: {group_cfg.encoder_pin_memory!r}")
    if group_cfg.merged(**{"encoder.pin_memory": "0"}).encoder_pin_memory is not False:
        raise AssertionError("--set encoder.pin_memory=0 не выключает pin")
    print("  ok: pin включён и при offload=group; выключает только encoder.pin_memory=0")
    print(f"  ok: _STREAM_WARNED удалён, stream=true работает; auto -> {enc.device}, "
          f"stream={enc.stream}, prefetch={enc.prefetch}")


# --------------------------------------------------------------------------- #
# Запуск
# --------------------------------------------------------------------------- #


CASES: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("1. encoder.stream: конфиг, умолчание, ключ кэша не меняется, предупреждения для base нет",
     case_config_and_cache_key),
    ("2. базовый пресет base: stream False -> True, ключ кэша не меняется",
     case_preset_stream),
    ("3. мини-энкодер: stream-загрузка даёт параметры без хранилища и бит-в-бит forward",
     case_mini_encoder),
    ("4. B1: стриминговый энкодер из flux2._load_text_encoder + move_to не падают",
     case_streamed_encoder_placement),
    ("5. B1-bis: стриминговая модель сообщает device=cpu, после forward параметры снова 0 байт",
     case_streamed_reports_device),
    ("6. N13: stream=True + device=cpu/cuda принят, mps/xpu/npu -> ValueError",
     case_device_validation),
    ("7. N11: ошибка в пост-обработке -> mmap закрыт (weights.closed=True)",
     case_closed_on_post_error),
    ("8. N12: file_dtype() однородного набора и ValueError на смешанном",
     case_file_dtype),
    ("9. N5: always_call освобождает веса модуля при исключении в forward",
     case_always_call),
    ("10. предупреждение для base удалено: stream действует, auto включает конвейер",
     case_base_stream_effective),
)


def main() -> int:
    failed = 0
    base = fixtures.temp_base()                  # системный temp, а не корень репозитория
    with tempfile.TemporaryDirectory(prefix="qwenstyle-enc-stream-", dir=str(base)) as raw:
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
