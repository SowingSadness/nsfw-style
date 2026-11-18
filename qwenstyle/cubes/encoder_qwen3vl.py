"""Кубик `encoder.qwen3vl`: энкодер Qwen3-VL-8B и типизированный ключ conditioning-кэша.

Зачем отдельный кубик (план `cube-architecture`, шаг 4): раньше имя файла кэша эмбеддингов
собиралось из `RunConfig` (`cfg.embeds_key`), а сам энкод жил внутри бекенда генерации. Теперь

* ключ считается **из портов** (`Prompt.md5`, `RefImages.sha`, сетка) и параметров кубика
  (`name`/`dtype`/`device` → хвост `_te-<tag>[-dtype][-dev-device]`) — ровно та же строка, что и
  раньше, но без `RunConfig`;
* `load`/`save` работают с типизированным `Conditioning`, а не с `dict`;
* `self_test` без весов проверяет формат ключа на реальных именах из `cache/embeds/`.

Формат ключа зафиксирован (R1, ломать нельзя): `embeds_<md5>_<sha><suffix>_<W>x<H>.pt`, где
`suffix` пуст для `base`/`cpu`/`bfloat16`, иначе `_te-<tag>` + `-<dtype>` + `-dev-<device>`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from ..backends.base import ParameterSpec, ResourceBudget
from ..cube import CubeContext, CubeSpec, Stage, register_cube
from ..ports import Conditioning, Prompt, RefImages

__all__ = ["CUBE", "EncoderParams", "conditioning_key", "load", "save"]

#: Умолчания, которые НЕ добавляют сегмент в имя файла (исторические имена записей).
DEFAULT_DTYPE = "bfloat16"
DEFAULT_DEVICE = "cpu"


def _dtype_token(dtype: str) -> str:
    return {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32"}.get(dtype, dtype)


def _device_token(device: str) -> str:
    return "cuda" if device.startswith("cuda") else device


@dataclass(frozen=True)
class EncoderParams:
    """Разобранные параметры энкодера (провалидированы по дескрипторам кубика)."""

    name: str = "heretic"
    cache_tag: str = "heretic"
    repo: str = "DreamFast/Qwen3-VL-8B-Heretic-1.3.0"
    revision: str | None = None
    local_dir: str | None = "cache/models/weights/heretic-te"
    dtype: str = DEFAULT_DTYPE
    device: str = "cuda"
    stream: bool = True
    prefetch: int = 4
    pin_memory: bool = True
    lookahead: int = 1

    @property
    def canon_device(self) -> str:
        """Канон устройства: `auto` -> фактическое (cuda при наличии), `cuda:0` -> `cuda`."""
        from ..config import encoder_device_canon

        return encoder_device_canon(self.device)

    @property
    def canon_dtype(self) -> str:
        from ..config import encoder_dtype_canon

        return encoder_dtype_canon(self.dtype)

    @property
    def key_suffix(self) -> str:
        """Хвост имени кэша: тег энкодера и неумолчания dtype/device (байт-в-байт как раньше)."""
        suffix = "" if self.cache_tag in ("", "base") else f"_te-{self.cache_tag}"
        if self.canon_dtype != DEFAULT_DTYPE:
            suffix += f"-{_dtype_token(self.canon_dtype)}"
        if self.canon_device != DEFAULT_DEVICE:
            suffix += f"-dev-{_device_token(self.canon_device)}"
        return suffix

    def cache_name(self, prompt_md5: str, refs_sha: str, grid: tuple[int, int]) -> str:
        """Имя записи кэша conditioning для пары (промт, референсы) на сетке `grid`."""
        width, height = int(grid[0]), int(grid[1])
        return f"embeds_{prompt_md5}_{refs_sha}{self.key_suffix}_{width}x{height}.pt"

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> "EncoderParams":
        values = {p.name: p.default for p in CUBE.params}
        values.update({k: v for k, v in (params or {}).items() if k in values})
        return cls(name=str(values["name"]), cache_tag=str(values["cache_tag"]),
                   repo=str(values["repo"]), revision=values["revision"],
                   local_dir=values["local_dir"], dtype=str(values["dtype"]),
                   device=str(values["device"]), stream=bool(values["stream"]),
                   prefetch=int(values["prefetch"]), pin_memory=bool(values["pin_memory"]),
                   lookahead=int(values["lookahead"]))


def conditioning_key(params: EncoderParams, prompt: Prompt, refs: RefImages | None, *,
                     grid: tuple[int, int]) -> str:
    """Типизированный ключ: строится из портов, а не из конфига прогона.

    `refs=None` (T2I) даёт сегмент `noref` — ровно как `io.images_sha` для пустого списка.
    """
    from ..io import NOREF_SHA

    sha = refs.sha if refs and refs.sha else NOREF_SHA
    return params.cache_name(prompt.md5, sha, grid)


def load(store: Any, name: str) -> Conditioning | None:
    """Прочитать conditioning из сервиса кэша (`BlobCache`-совместимый `get`)."""
    if store is None:
        return None
    payload = store.get_name(name) if hasattr(store, "get_name") else None
    if not payload:
        return None
    return Conditioning(embeds=payload.get("embeds"), mask=payload.get("mask"),
                        img_mask=payload.get("img_mask"), ref_grid=tuple(payload.get("grid") or (0, 0)))


def save(store: Any, name: str, cond: Conditioning, meta: Mapping[str, Any] | None = None) -> Any:
    """Записать conditioning в сервис кэша."""
    if store is None or not hasattr(store, "put_name"):
        return None
    payload = {"embeds": cond.embeds, "mask": cond.mask, "img_mask": cond.img_mask,
               "grid": tuple(cond.ref_grid)}
    return store.put_name(name, payload, dict(meta or {}))


def _self_test(ctx: CubeContext) -> None:
    """Без весов: формат ключа на реальных примерах из `cache/embeds/` и `AGENTS.md`."""
    heretic = EncoderParams.from_mapping({})
    assert heretic.key_suffix == "_te-heretic-dev-cuda", heretic.key_suffix
    name = heretic.cache_name("a" * 12, "b" * 12, (1024, 768))
    assert name == f"embeds_{'a' * 12}_{'b' * 12}_te-heretic-dev-cuda_1024x768.pt", name

    base = EncoderParams.from_mapping({"name": "base", "cache_tag": "base", "device": "cpu"})
    assert base.key_suffix == "", base.key_suffix
    assert base.cache_name("a" * 12, "b" * 12, (1376, 768)) == \
        f"embeds_{'a' * 12}_{'b' * 12}_1376x768.pt", base.cache_name("a" * 12, "b" * 12, (1376, 768))

    fp16_cuda = EncoderParams.from_mapping({"name": "base", "cache_tag": "base",
                                            "dtype": "float16", "device": "cuda"})
    assert fp16_cuda.key_suffix == "-fp16-dev-cuda", fp16_cuda.key_suffix

    t2i = conditioning_key(heretic, Prompt(text="x", md5="c" * 12), None, grid=(1024, 768))
    assert t2i.endswith("_noref_te-heretic-dev-cuda_1024x768.pt"), t2i


CUBE: CubeSpec = register_cube(CubeSpec(
    id="encoder.qwen3vl",
    stage=Stage.ENCODER,
    consumes=frozenset({"prompt"}),
    reads=frozenset({"refs"}),
    produces="conditioning",
    params=(
        ParameterSpec("name", "cube", "str", "heretic",
                      doc="имя энкодера в реестре (`heretic` по умолчанию, `base` — штатный)"),
        ParameterSpec("cache_tag", "cube", "str", "heretic",
                      doc="тег в имени кэша (`_te-<tag>`); обязан следовать имени энкодера"),
        ParameterSpec("repo", "cube", "str", "DreamFast/Qwen3-VL-8B-Heretic-1.3.0",
                      doc="репозиторий весов (подсказка для скачивания)"),
        ParameterSpec("revision", "cube", "str", None, doc="ревизия весов (commit)"),
        ParameterSpec("local_dir", "cube", "str", "cache/models/weights/heretic-te",
                      doc="каталог локальных шардов; грузим с диска, без Hub"),
        ParameterSpec("dtype", "cube", "enum", DEFAULT_DTYPE, choices=("bfloat16", "float16"),
                      doc="точность энкодера; bfloat16 — умолчание (без сегмента в имени кэша)"),
        ParameterSpec("device", "cube", "enum", "cuda", choices=("auto", "cuda", "cpu"),
                      doc="устройство энкода; cuda включает стриминг по модулям; cpu — сегмент не пишется"),
        ParameterSpec("stream", "cube", "bool", True,
                      doc="читать веса по модулям с диска (снимает требование «энкодер <= RAM»)"),
        ParameterSpec("prefetch", "cube", "int", 4, minimum=0, maximum=64,
                      doc="сколько модулей читать вперёд (конвейер pinned-буферов)"),
        ParameterSpec("pin_memory", "cube", "bool", True,
                      doc="pinned-буфер RAM для H2D; 0 = pageable, если pinned не выделяется"),
        ParameterSpec("lookahead", "cube", "int", 1, minimum=1, maximum=16,
                      doc="глубина конвейера: сколько модулей готовятся заранее"),
    ),
    budget=ResourceBudget(encoder_gib=17.5, disk_gib=17.5,
                          notes="Qwen3-VL-8B bf16 (750 тензоров, 4 шарда)"),
    self_test=_self_test,
    doc="энкодер Qwen3-VL-8B: prompt+refs -> conditioning, стриминг весов, ключ кэша из портов",
))
