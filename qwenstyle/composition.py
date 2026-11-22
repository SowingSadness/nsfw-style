"""Проекция `RunConfig` → композиция кубиков (`PipelineSpec`) и её проверка.

Зачем (план `cube-architecture`, шаг 6): пресеты сегодня — данные `RunConfig`
(`presets/*.yaml` + `extends:`), а композитор работает с кубиками. Чтобы UI и `config show`
видели дерево стадий **уже сейчас**, здесь односторонняя проекция: конфиг остаётся источником
истины для прогона (переезд `run`/`gen` на композицию — шаг 7), а проекция показывает, из каких
кубиков он состоит и какие стратегии включает.

    spec = spec_of(cfg)                    # PipelineSpec с кубиками и их параметрами
    composition = check(cfg)               # + проверка композитором (без загрузки весов)

Правила проекции (единственное место, где конфиг превращается в кубики):

* `source.frame` — всегда: канвас и сетка референсов (порт `frame`);
* `source.txtFileImage` — всегда: промт из файла и референсы (`--input a --input b` -> `Image1`,
  `Image2`)
  (тогда `refs` указывает на конкретные файлы, а не на каталог проекта);
* `encoder.qwen3vl` — только если бекенд объявляет `conditioning="precomputed"`: `internal`-модели
  (H3, FLUX.2) считают эмбеддинги сами, энкодер-кубик им не нужен;
* `strategy.precision.fp8` — только при `generation.precision=fp8` (bf16 идёт без стратегии);
* `strategy.kv_offload` — если бекенд заявляет поле `kv_cache` (сейчас `qwen_image21`);
* `strategy.window_attention` — если `sampler_kwargs.window_attn_px > 0`;
* `transformer.qwen_image21` + `decoder.vae.qwen21` + `decoder.postprocess.image` — только для
  бекендов, у которых есть кубик (сегодня `qwen_image21`; для остальных `unsupported_reason()`
  говорит почему); декодер отдаёт `frames`, в картинку их превращает постпроцесс;
* постобработка — `post.upscale.esrgan`/`post.upscale.seedvr2` при включённом апскейле,
  `post.colorfix` при `upscale.color_fix != none`.
"""
from __future__ import annotations

from typing import Any

from .compose import Composition, CubeBinding, PipelineSpec, compose
from .config import ASPECT_RATIOS
from .cube import all_cubes

#: Импорт пакета кубиков регистрирует все стадии в реестре (188 мс, без torch).
#: Без него `all_cubes()` пуст и любая проекция выглядела бы как «кубиков нет».
from . import cubes as _cubes  # noqa: F401  — регистрация кубиков

__all__ = ["CUBE_BACKENDS", "VAE_BACKENDS", "check", "spec_of", "unsupported_reason"]

#: Бекенды, у которых есть кубик трансформера.
CUBE_BACKENDS = ("qwen_image21",)
#: Бекенды с VAE Qwen-Image-2.1 (`decoder.vae.qwen21`).
VAE_BACKENDS = ("qwen_image21", "flux2")
#: Бекенды с предвычисленным conditioning (`.pt`-кэш): им нужен `encoder.qwen3vl`.
PRECOMPUTED_BACKENDS = ("qwen_image21", "flux2")
#: Бекенды, читающие `generation.kv_cache` (префиксный KV-кеш).
KV_BACKENDS = ("qwen_image21",)


def unsupported_reason(cfg: Any) -> str | None:
    """Почему композиция невозможна (или `None`, если всё поддержано)."""
    backend = cfg.generation.backend
    if backend not in CUBE_BACKENDS:
        return (f"бекенд {backend!r} ещё не выражен кубиками "
                f"(поддержаны: {', '.join(CUBE_BACKENDS)})")
    known = {cube.id for cube in all_cubes()}
    missing = [cid for cid in ("source.frame", "source.txtFileImage", "encoder.qwen3vl",
                               "transformer.qwen_image21", "decoder.vae.qwen21")
               if cid not in known]
    if missing:
        return f"в реестре нет кубиков {missing}"
    return None


def _sampler(cfg: Any, name: str, default: Any = None) -> Any:
    return (cfg.generation.sampler_kwargs or {}).get(name, default)


def _frame_params(cfg: Any) -> dict[str, Any]:
    """Геометрия из `RunConfig`: `size_px`/`aspect` -> `width`/`height`, `ref_px` -> `ref_crop_side`.

    Тонкость: если `aspect` задан явно, канвас известен целиком (W и H), и тогда прежний `ref_px`
    (высота сетки референсов) переводится в `ref_crop_side` тем же числом — семантика одна и та же.
    Если `aspect` не задан, канвас зависит от референса (`width: 0`), и `ref_crop_side` остаётся
    прежним `ref_px` — сетка считается от фактического канваса на прогоне.
    """
    ratio = None
    if cfg.aspect:
        ratio = ASPECT_RATIOS[str(cfg.aspect)] if str(cfg.aspect) in ASPECT_RATIOS else None
    size_px = int(cfg.size_px or 1024)
    ref_side = int(_sampler(cfg, "ref_px", 0) or 0)
    params: dict[str, Any] = {"ref_crop_side": ref_side}
    if ratio:
        params["height"] = size_px
        params["width"] = max(32, int(round(size_px * ratio / 32)) * 32)
    else:
        params["height"] = size_px
        params["width"] = 0
    return params


def spec_of(cfg: Any, *, name: str = "") -> PipelineSpec:
    """Собрать `PipelineSpec` из `RunConfig` (без диска, без весов, без CUDA)."""
    gen, enc, up = cfg.generation, cfg.encoder, cfg.upscale
    refs = list(cfg.inputs or ([cfg.input] if cfg.input else []))
    cubes: list[CubeBinding] = []

    # источник 1/2: геометрия кадра (канвас + сетка референсов) -> порт `frame`
    cubes.append(CubeBinding("source.frame", _frame_params(cfg)))

    # источник 2/2: промт из файла + явные референсы (`--input a --input b` -> Image1, Image2)
    source: dict[str, Any] = {"prompt": cfg.prompt_file or ""}
    for index, path in enumerate(refs[:10], start=1):
        source[f"Image{index}"] = str(path)
    if not cfg.prompt_file and cfg.prompt:
        source["prompt"] = ""               # промт встроен в конфиг, файла нет
    cubes.append(CubeBinding("source.txtFileImage", source))

    if gen.backend in PRECOMPUTED_BACKENDS:
        cubes.append(CubeBinding("encoder.qwen3vl", {
            "name": enc.name, "cache_tag": enc.cache_tag, "dtype": enc.dtype,
            "device": enc.device, "repo": enc.repo,
            "revision": enc.revision or "", "local_dir": enc.local_dir or "",
            "stream": bool(enc.stream), "prefetch": int(enc.prefetch),
            "pin_memory": bool(enc.pin_memory), "lookahead": int(enc.lookahead),
        }))

    if gen.precision == "fp8":
        cubes.append(CubeBinding("strategy.precision.fp8", {
            "cache_dir": gen.precision_cache or "cache/models/fp8_transformer"}))
    if gen.backend in KV_BACKENDS:
        cubes.append(CubeBinding("strategy.kv_offload", {"mode": gen.kv_cache}))
    window_px = int(_sampler(cfg, "window_attn_px", 0) or 0)
    if window_px > 0:
        cubes.append(CubeBinding("strategy.window_attention", {
            "window_attn_px": window_px,
            "window_attn_margin_px": int(_sampler(cfg, "window_attn_margin_px", 0) or 0)}))

    if gen.backend in CUBE_BACKENDS:
        lora = gen.lora
        cubes.append(CubeBinding("transformer.qwen_image21", {
            "backend": gen.backend,
            "repo": gen.repo,
            "revision": gen.revision or "",
            "steps": int(cfg.steps),
            "guidance": float(gen.guidance),
            "lora_repo": lora.repo if lora else "",
            "lora_file": lora.filename if lora else "",
            "lora_scale": float(lora.scale) if lora else 0.0,
            "offload": gen.offload,
            "offload_blocks": int(_sampler(cfg, "offload_blocks", 1) or 1),
            "offload_stream": bool(_sampler(cfg, "offload_stream", True)),
        }))
    if gen.backend in VAE_BACKENDS:
        cubes.append(CubeBinding("decoder.vae.qwen21", {
            "tile_px": int(_sampler(cfg, "vae_tile_px", 512)),
        }))
        # Декодер отдаёт тензор (`frames`), картинку делает следующий кубик: формат результата —
        # отдельная стадия (в будущем рядом встанет decoder.postprocess.video на том же порту).
        cubes.append(CubeBinding("decoder.postprocess.image", {"output_type": "pil"}))

    if up.backend == "esrgan" and (up.factor or up.target_height):
        cubes.append(CubeBinding("post.upscale.esrgan", {
            "weights": up.weights or "", "factor": float(up.factor or 0.0),
            "target_height": up.target_height, "tile": int(up.tile), "pad": int(up.pad)}))
    elif up.backend == "seedvr2" and (up.factor or up.target_height):
        cubes.append(CubeBinding("post.upscale.seedvr2", {
            "factor": float(up.factor or 0.0), "height": int(up.target_height or 0)}))
    if (up.color_fix or "none") != "none":
        cubes.append(CubeBinding("post.colorfix", {
            "mode": up.color_fix, "sigma": float(up.color_sigma),
            "strength": float(up.color_strength)}))

    inputs = frozenset({"prompt", "refs"}) if refs else frozenset({"prompt"})
    return PipelineSpec(name=name or (cfg.preset or "config"), cubes=tuple(cubes), inputs=inputs)


def check(cfg: Any, *, machine: dict[str, Any] | None = None) -> Composition:
    """Проверить композицию конфига (ничего не грузит).

    Для бекенда без кубика возвращается **одна** ошибка `cube_unsupported`: каскадные
    `port_missing`/`ports_unused` в этом случае только шумят — цепочка оборвана не по вине портов.
    """
    reason = unsupported_reason(cfg)
    if reason is not None:
        from dataclasses import replace

        from .compose import Issue

        spec = spec_of(cfg)
        return replace(compose(PipelineSpec(name=spec.name, cubes=(), inputs=spec.inputs),
                               machine=machine),
                       spec=spec,
                       issues=(Issue("cube_unsupported", reason,
                                     hint="PRESETS остаются источником истины для прогона до шага 7"),))
    return compose(spec_of(cfg), machine=machine)
