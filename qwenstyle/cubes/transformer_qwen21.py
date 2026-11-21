"""Кубик `transformer.qwen_image21`: диффузионный трансформер Qwen-Image-2.1 (стадия TRANSFORMER).

Роль кубика: `conditioning -> latents`. Он держит веса трансформера (с fp8-кэшем или без), делает
шаги сэмплера и **требует** стратегии `strategy.precision.fp8` + `strategy.kv_offload` — раскладка
памяти и префиксный кэш объявляются отдельными кубиками, а не флагами внутри модели.

Границы на шаге 6 плана `cube-architecture` (важно, чтобы не было иллюзий):

* кубик **объявляет параметры и сверяет их с бекендом** (`verify()`), но сам forward пока живёт в
  `pipeline.py` — перенос кода в кубик делает шаг 7 (cut-over). Так композитор, `config show` и UI
  уже видят параметры трансформера, не дожидаясь переезда;
* VAE-декодер — отдельный кубик (`decoder.vae.qwen21`), поэтому `vae_tile_px` здесь нет;
* `kv_cache`/`window_attn_px`/`precision`-кэш — **не** параметры этого кубика: их объявляют
  стратегии (`strategy.kv_offload`, `strategy.window_attention`, `strategy.precision.fp8`).

Сверка с бекендом (`verify`) — не формальность: она ловит расхождение, из-за которого `--set`
начинает молча теряться (например бекенд переименовал ключ `sampler_kwargs`, а спека осталась).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..backends.base import ParameterSpec, ResourceBudget
from ..cube import CubeContext, CubeSpec, Stage, register_cube

__all__ = ["CUBE", "OFFLOAD_MODES", "TransformerParams", "lora_of", "sampler_kwargs", "verify"]

#: Раскладки весов (`--offload`): `none` — веса в VRAM, `group` — блоки стримятся с CPU.
OFFLOAD_MODES = ("none", "group")


@dataclass(frozen=True)
class TransformerParams:
    """Разобранные параметры трансформера."""

    backend: str = "qwen_image21"
    repo: str = "Qwen/Qwen-Image-2.1"
    revision: str | None = None
    steps: int = 25
    guidance: float = 1.0
    lora_repo: str = ""
    lora_file: str = ""
    lora_scale: float = 0.0
    offload: str = "none"
    offload_blocks: int = 1
    offload_stream: bool = True

    @property
    def has_lora(self) -> bool:
        """LoRA включается только парой (репозиторий, файл) и ненулевым масштабом."""
        return bool(self.lora_repo and self.lora_file and self.lora_scale > 0.0)

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> "TransformerParams":
        """Собрать из «словаря параметров кубика»; дефолты — из `CUBE.params`."""
        values = {p.name: p.default for p in CUBE.params}
        values.update({k: v for k, v in (params or {}).items() if k in values})
        return cls(backend=str(values["backend"]), repo=str(values["repo"]),
                   revision=None if values["revision"] in (None, "") else str(values["revision"]),
                   steps=int(values["steps"]), guidance=float(values["guidance"]),
                   lora_repo=str(values["lora_repo"] or ""),
                   lora_file=str(values["lora_file"] or ""),
                   lora_scale=float(values["lora_scale"] or 0.0),
                   offload=str(values["offload"]),
                   offload_blocks=int(values["offload_blocks"]),
                   offload_stream=bool(values["offload_stream"]))


def sampler_kwargs(params: TransformerParams) -> dict[str, Any]:
    """Ключи `generation.sampler_kwargs`, которые читает бекенд (без стратегий)."""
    return {"offload_blocks": params.offload_blocks, "offload_stream": params.offload_stream}


def lora_of(params: TransformerParams) -> dict[str, Any] | None:
    """LoRA в форме `config.LoraConfig` (или `None`, если адаптер не задан)."""
    if not params.has_lora:
        return None
    return {"repo": params.lora_repo, "filename": params.lora_file, "scale": params.lora_scale}


def verify(params: TransformerParams) -> None:
    """Сверить параметры кубика с тем, что объявляет сам бекенд (`parameters()`/`caps`).

    Ошибка здесь = «кубик и модель договорились по-разному»: например спека разрешает
    `offload=none|group`, а бекенд знает только `none`. Такое обязано падать в композиторе,
    до загрузки 6.7 ГиБ весов.
    """
    from ..backends.qwen_image21 import QwenImage21Backend

    caps = QwenImage21Backend.caps
    known = {p.name: p for p in QwenImage21Backend.parameters()}
    problems: list[str] = []

    for name, value in (("steps", params.steps), ("guidance", params.guidance),
                        ("offload", params.offload)):
        spec = known.get(name)
        if spec is None:
            continue                                    # ключа нет у бекенда — забота теста спеки
        if spec.choices and value not in spec.choices:
            problems.append(f"{name}={value!r} вне домена {list(spec.choices)}")
        # Минимум сверяется только по физическому полу (хотя бы один шаг): бекенд объявляет
        # `steps >= 8` как **рекомендуемый** диапазон (расписание 25 шагов), а `--steps 1..4` —
        # легальные смоук/черновые прогоны (так ходят selftest и тесты логов).
        if name == "steps" and value < 1:
            problems.append(f"{name}={value} меньше физического минимума 1")
        if spec.maximum is not None and isinstance(value, (int, float)) and value > spec.maximum:
            problems.append(f"{name}={value} больше максимума {spec.maximum}")
    if params.offload not in caps.offload:
        problems.append(f"offload={params.offload!r} не поддерживается моделью {caps.offload}")
    if params.has_lora and not caps.supports_lora:
        problems.append(f"модель {params.backend} не поддерживает LoRA")
    if problems:
        raise ValueError(f"transformer.qwen_image21: {'; '.join(problems)}")


def _self_test(ctx: CubeContext) -> None:
    """Без весов: параметры разбираются, домены совпадают с бекендом, LoRA включается парой."""
    defaults = TransformerParams.from_mapping({})
    assert defaults.backend == "qwen_image21", defaults
    assert defaults.steps == 25 and defaults.offload == "none", defaults
    verify(defaults)
    grouped = TransformerParams.from_mapping({"offload": "group", "offload_blocks": 4,
                                              "offload_stream": False})
    verify(grouped)
    assert sampler_kwargs(grouped) == {"offload_blocks": 4, "offload_stream": False}, grouped
    # LoRA без файла или с нулевым масштабом не включается: пустой repo/file — не адаптер
    assert not TransformerParams.from_mapping({"lora_repo": "r"}).has_lora
    assert not TransformerParams.from_mapping({"lora_repo": "r", "lora_file": "f"}).has_lora
    lora = TransformerParams.from_mapping(
        {"lora_repo": "MicksHF/Qwen-Image-2.1-Anime-Fusal", "lora_file": "a.safetensors",
         "lora_scale": 0.6})
    assert lora.has_lora and lora_of(lora)["scale"] == 0.6, lora_of(lora)
    verify(lora)
    # опечатка в раскладке обязана падать до загрузки весов
    try:
        verify(TransformerParams.from_mapping({"offload": "disk"}))
    except ValueError:
        pass
    else:                                               # pragma: no cover - защита от регресса
        raise AssertionError("offload=disk обязан отвергаться")


CUBE: CubeSpec = register_cube(CubeSpec(
    id="transformer.qwen_image21",
    stage=Stage.TRANSFORMER,
    consumes=frozenset({"conditioning"}),
    produces="latents",
    requires=frozenset({"fp8", "kv_offload"}),
    params=(
        ParameterSpec("backend", "cube", "enum", "qwen_image21", choices=("qwen_image21",),
                      doc="бекенд генерации: qwen_image21 (base/turbo выбираются репозиторием)"),
        ParameterSpec("repo", "cube", "str", "Qwen/Qwen-Image-2.1",
                      doc="репозиторий пайплайна; transformer может подменяться (turbo)"),
        ParameterSpec("revision", "cube", "str", None,
                      doc="ревизия репозитория (commit hash); пусто = последняя локальная"),
        ParameterSpec("steps", "generation", "int", 25, minimum=1, maximum=50,
                      doc="шагов сэмплера (1-4 — смоук/черновик, 25 — база, 8 — turbo по "
                          "расписанию sample_sigmas)"),
        ParameterSpec("guidance", "generation", "float", 1.0, minimum=0.0, maximum=10.0,
                      doc="true_cfg_scale; при 1.0 CFG-ветки нет (negative_prompt игнорируется)"),
        ParameterSpec("lora_repo", "cube", "str", "",
                      doc="репозиторий LoRA-адаптера (FUSAL R80); пусто = без адаптера"),
        ParameterSpec("lora_file", "cube", "str", "",
                      doc="файл адаптера внутри репозитория (.safetensors)"),
        ParameterSpec("lora_scale", "generation", "float", 0.0, minimum=0.0, maximum=2.0,
                      doc="сила адаптера: 0 = выключен; правка 0.65-0.70, генерация 0.85-1.0"),
        ParameterSpec("offload", "generation", "enum", "none", choices=OFFLOAD_MODES,
                      doc="none = веса в VRAM; group = блоки стримятся с CPU (нужно для 960p+)"),
        ParameterSpec("offload_blocks", "sampler", "int", 1, minimum=1, maximum=64,
                      doc="блоков трансформера на GPU одновременно (offload=group)"),
        ParameterSpec("offload_stream", "sampler", "bool", True,
                      doc="асинхронная префетч-загрузка блоков (offload=group)"),
    ),
    budget=ResourceBudget(weights_bf16_gib=14.2, weights_fp8_gib=6.6,
                          notes="Qwen-Image-2.1 transformer, 224 модуля; bf16 14.2 ГиБ, fp8-кэш 6.6 ГиБ"),
    self_test=_self_test,
    doc="диффузионный трансформер Qwen-Image-2.1: conditioning -> latents (требует fp8 + kv_offload)",
))
