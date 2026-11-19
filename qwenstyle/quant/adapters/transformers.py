"""Адаптер transformers: fp8-веса LM в VRAM с де-квантом модуля на forward (план, шаг 5b.3).

Задача — дать промт-энхансеру (Qwen3.5-VL 9.41B) работать на 16 ГиБ: bf16 целиком в VRAM не влезает
(17.55 ГиБ), а `device_map="auto"` размазывает слои по RAM и даёт ~1.1 ток/с. Схема адаптера:

* веса на диске в **fp8** (per-output-channel scale, как layerwise cast у diffusers) — 8.8 ГиБ;
* при загрузке fp8-тензоры кладутся прямо в VRAM, точные (нормы, эмбеддинги, свёртки) — как есть;
* `nn.Linear` с квантованным весом заменяется на `Fp8DequantLinear`: на forward он де-квантует
  **свой** вес в bf16 и считает обычный `F.linear`. Ошибки GEMM нет (в отличие от `_scaled_mm`,
  который на этой сборке умеет только tensor-wise scale, см. прототип шага 5b), цена — де-квант
  веса модуля: ~0.68 мс на 64 МиБ, то есть ~0.1 с на полный проход по 8.8 ГиБ.

Почему не `_scaled_mm`: замер прототипа — 2× быстрее, но относительная ошибка слоя 3.7 % на
tensor-wise scale; для 36 слоёв это неприемлемо, а per-row/per-column эта сборка не поддерживает.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from ..reader import QuantReader
from ..store import QuantSpec, QuantStore, ShardWriter, source_entries, source_shards

__all__ = [
    "FP8_SPEC",
    "Fp8DequantLinear",
    "build_cache",
    "cache_param_name",
    "load_quantized",
    "quantizable",
    "to_fp8",
]

#: Пропуски для LM: нормы, эмбеддинги и lm_head остаются точными; кастуются только 2D `*.weight`.
FP8_SPEC = QuantSpec(
    target="fp8_e4m3",
    # `lm_head` тоже квантуем: это 2D-вес 151936x4096, +1.2 ГиБ bf16 на ровном месте.
    # Точными остаются только эмбеддинги, нормы, biases и «служебные» тензоры.
    skip=(r"embed_tokens", r"layernorm", r"\.norm\.", r"\.bias$", r"\.scales?$", r"rotary",
          r"inv_freq"),
)


def quantizable(name: str, tensor: Any, spec: QuantSpec = FP8_SPEC) -> bool:
    """Кастуем ли этот тензор: 2D-вес Linear и не в списке пропусков."""
    try:
        ndim = int(tensor.dim())
    except Exception:                                        # noqa: BLE001 — не тензор
        return False
    return (name.endswith(".weight") and ndim == 2 and tensor.dtype.is_floating_point
            and not spec.is_skipped(name))


def _scales_and_cast(tensor: Any, torch: Any) -> tuple[Any, Any]:
    """Per-output-channel скейл: `amax(dim=1)/448`, вес в fp8, скейл — float32 [N]."""
    weight = tensor.detach().float()
    amax = weight.abs().amax(dim=1).clamp(min=1e-8)
    scale = (amax / 448.0).to(torch.float32)
    quantized = (weight / scale[:, None]).to(torch.float8_e4m3fn)
    return quantized, scale


def build_cache(repo: str, cache_dir: str | Path, *, spec: QuantSpec = FP8_SPEC,
                log: Callable[[str], None] | None = None,
                revision: str | None = None) -> Path:
    """Собрать fp8-кэш для LM из репозитория (оффлайн, из локального кэша Хаба).

    Тензоры читаются по одному (пик RAM — один тензор + буфер шарда), кастуются в fp8 и пишутся
    шардами; рядом ложится `precision.json` с дайджестом источника и списком пропусков.
    """
    import time

    import torch
    from safetensors import safe_open

    say = log or (lambda _m: None)
    store = QuantStore(cache_dir)
    src_entries = source_entries(source_shards(_resolve(repo, revision)))
    digest = _digest(src_entries)
    config_sha = _config_sha(repo, revision)
    problem = store.verify(digest=digest, config_sha=config_sha, spec=spec)
    if problem is None:
        say(f"[quant] fp8-кэш переиспользован: {store.path}")
        return store.path

    say(f"[quant] каст {repo} -> fp8: {problem}")
    started = time.time()
    tmp = store.path.with_name(store.path.name + f".tmp-{__import__('os').getpid()}")
    writer = ShardWriter(tmp, block_bytes=spec.block_bytes)
    num_fp8 = num_exact = 0
    handles: dict[str, Any] = {}
    for entry in src_entries:
        name = entry["key"]
        shard = entry["shard"]
        handle = handles.get(shard)
        if handle is None:
            handle = safe_open(str(_shard_path(repo, revision, shard)), framework="pt", device="cpu")
            handles[shard] = handle
        tensor = handle.get_tensor(name)
        if quantizable(name, tensor, spec):
            quantized, scale = _scales_and_cast(tensor, torch)
            writer.add(name, quantized)
            writer.add(f"{name}.scale", scale)
            num_fp8 += 1
        else:
            writer.add(name, tensor)
            num_exact += 1
    table = writer.close()
    store_tmp = QuantStore(tmp)
    store_tmp.write_sidecar({
        "schema": __import__("qwenstyle.quant.store", fromlist=["CACHE_SCHEMA"]).CACHE_SCHEMA,
        "kind": "transformers_lm",
        "source": repo,
        "source_revision": revision,
        "source_digest": digest,
        "config_sha256": config_sha,
        "fp8_dtype": spec.target_token,
        "skip_patterns": list(spec.skip),
        "num_fp8": num_fp8,
        "num_exact": num_exact,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights": table,
    })
    from ..store import replace_dir

    if store.path.exists():
        replace_dir(tmp, store.path)
    else:
        tmp.replace(store.path)
    say(f"[quant] готово за {time.time() - started:.0f} с: fp8 {num_fp8}, точных {num_exact}, "
        f"шардов {len(table['shards'])}")
    return store.path


class Fp8DequantLinear:
    """`nn.Linear` с fp8-весом: на forward де-квантует вес в dtype входа.

    Создаётся фабрикой `make_linear(...)`, которая возвращает настоящий `nn.Module` (torch требует
    `nn.Module` в `setattr` контейнера), поэтому класс собирается один раз при импорте модуля.
    """

    @staticmethod
    def install() -> type:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional

        class _Fp8DequantLinear(nn.Module):
            def __init__(self, out_features: int, in_features: int, bias: bool = True) -> None:
                super().__init__()
                self.in_features = int(in_features)
                self.out_features = int(out_features)
                self.register_buffer("weight", torch.empty(out_features, in_features,
                                                           dtype=torch.float8_e4m3fn))
                self.register_buffer("weight_scale", torch.empty(out_features,
                                                                 dtype=torch.float32))
                if bias:
                    self.register_parameter("bias", nn.Parameter(torch.empty(out_features)))
                else:
                    self.register_parameter("bias", None)

            def dequantized(self, dtype: Any = None) -> Any:
                # Без float32-промежутка: каст fp8->bf16 и умножение на скейл в целевой точности
                # экономят вдвое больше трафика, чем `float32 * scale`.
                target = dtype or torch.bfloat16
                scale = self.weight_scale.to(target)[:, None]
                return self.weight.to(target) * scale

            def forward(self, x: Any) -> Any:
                weight = self.dequantized(x.dtype)
                bias = None if self.bias is None else self.bias.to(x.dtype)
                return functional.linear(x, weight, bias)

        return _Fp8DequantLinear


def cache_param_name(param: str) -> str:
    """Имя тензора в кэше для параметра модуля (`<n>.weight_scale` -> `<n>.weight.scale`)."""
    if param.endswith(".weight_scale"):
        return param[: -len("_scale")] + ".scale"
    return param


def to_fp8(tensor: Any) -> tuple[Any, Any]:
    """Публичная обёртка над кастом: `(fp8-вес, per-channel scale)` — для тестов и адаптеров."""
    import torch

    return _scales_and_cast(tensor, torch)


def _replace_linear(module: Any, reader: QuantReader, prefix: str = "",
                    cls: type | None = None, device: str = "cpu") -> int:
    """Заменить `nn.Linear` на fp8-версию там, где в кэше есть fp8-вес и скейл.

    Замены создаются в контексте `meta` (вызывать внутри `torch.device("meta")`), чтобы не
    выделять реальную память под 9.4B параметров до присваивания из кэша.
    """
    import torch.nn as nn

    cls = cls or Fp8DequantLinear.install()
    replaced = 0
    for name, child in list(module.named_children()):
        full = f"{prefix}{name}"
        weight_key = f"{full}.weight"
        scale_key = f"{weight_key}.scale"
        if isinstance(child, nn.Linear) and weight_key in reader.names() and scale_key in reader.names():
            proxy = cls(int(child.out_features), int(child.in_features), child.bias is not None)
            proxy.weight = reader.read(weight_key).to(device)
            proxy.weight_scale = reader.read(scale_key).to(device)
            if child.bias is not None:
                import torch.nn as nn

                proxy.bias = nn.Parameter(child.bias.detach().clone().to(device))
            setattr(module, name, proxy)
            del child                                            # освобождаем bf16-вес сразу
            replaced += 1
        else:
            replaced += _replace_linear(child, reader, prefix=f"{full}.", cls=cls, device=device)
    return replaced


def load_quantized(model_factory: Callable[[], Any], cache_dir: str | Path, *,
                   device: str = "cuda",
                   warmup: Iterable[str] = ()) -> tuple[Any, dict[str, Any]]:
    """Собрать модель из fp8-кэша: bf16-модель на CPU -> подмена Linear из кэша -> `.to(device)`.

    `model_factory` создаёт **настоящую** модель на CPU в bf16 (например
    `lambda: AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16,
    device_map="cpu", low_cpu_mem_usage=True)`): так буферы (`rotary.inv_freq` и пр.) считаются
    штатно — попытка собрать модель на meta-device упирается в «Cannot copy out of meta tensor».

    Квантованные Linear берут готовые fp8-тензоры и per-channel скейлы из нашего кэша, bf16-копия
    веса удаляется сразу, поэтому к моменту `.to(device)` в RAM остаются только точные тензоры,
    а в VRAM уезжает fp8 (8.8 ГиБ для 9.41B) вместо 17.55 ГиБ bf16.
    """
    if warmup:
        import transformers

        for attr in warmup:
            getattr(transformers, str(attr), None)

    reader = QuantReader(cache_dir)
    model = model_factory()
    replaced = _replace_linear(model, reader, device="cpu")
    model.eval()
    if device != "cpu":
        model.to(device)
    stats = {"linear_fp8": replaced, "tensors": len(reader.names())}
    return model, stats


# --------------------------------------------------------------------------- внутреннее

def _resolve(repo: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, local_files_only=True))


def _shard_path(repo: str, revision: str | None, shard: str) -> Path:
    return _resolve(repo, revision) / shard


def _config_sha(repo: str, revision: str | None) -> str:
    import hashlib

    config = _resolve(repo, revision) / "config.json"
    return hashlib.sha256(config.read_bytes()).hexdigest()


def _digest(entries: Iterable[dict[str, Any]]) -> str:
    from ..store import entries_digest

    return entries_digest(entries)
