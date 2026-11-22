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
            "transformer_repo": (gen.transformer[0] if gen.transformer else ""),
            "transformer_subfolder": (gen.transformer[1] if gen.transformer
                                      else "transformer"),
            "lora_repo": lora.repo if lora else "",
            "lora_file": lora.filename if lora else "",
            "lora_scale": float(lora.scale) if lora else 0.0,
            "lora_strip_alpha": bool(lora.strip_alpha) if lora else False,
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


#: Допуск сравнения пропорций при обратной проекции (W/H -> `aspect` из `RunConfig`).
_RATIO_TOLERANCE = 0.01


def config_mapping_of(pipeline: PipelineSpec, run: "Any" = None) -> dict[str, Any]:
    """Обратная проекция: композиция -> словарь для `RunConfig.from_mapping` (до `cutover`).

    Нужна переходному режиму: файл воркфлоу — единственный источник, а рантайм ждёт привычную
    форму `RunConfig`. Точность важна, иначе `gen --workflow` поедет на других значениях:

    * `encoder.qwen3vl` -> `encoder.*`; `transformer.qwen_image21` -> `generation.*`;
    * стратегии -> `generation.precision`/`kv_cache`/`sampler_kwargs.{offload_*, window_attn_*,
      ref_px, vae_tile_px}` (словарь свободных ключей бекенда);
    * `source.frame` -> `size_px`/`aspect`/`sampler_kwargs.ref_px`. Явные `width`+`height`
      выражаются только через пропорцию из `ASPECT_RATIOS` (16:9, 4:3, 1:1, 2:3); иная пропорция
      честно отвергается до `cutover` (после него `RunConfig` не нужен вовсе);
    * `post.upscale.*`/`post.colorfix` -> `upscale.*`; `run` -> `seed`/`out`/`logs`/`cache_*`.
    """
    from .cube import get_cube

    data: dict[str, Any] = {}
    sampler: dict[str, Any] = {}
    generation: dict[str, Any] = {"sampler_kwargs": sampler}
    encoder: dict[str, Any] = {}
    upscale: dict[str, Any] = {}

    for binding in pipeline.cubes:
        cube_id = binding.cube_id
        # эффективные параметры: объявленные умолчания кубика + то, что написано в файле
        params = {param.name: param.default for param in get_cube(cube_id).params}
        params.update(binding.params)
        if cube_id == "source.frame":
            width = int(params.get("width") or 0)
            height = int(params.get("height") or 0)
            if int(params.get("ref_crop_side") or 0) > 0:
                sampler["ref_px"] = int(params["ref_crop_side"])
            if width and height:
                ratio = width / max(1, height)
                aspect = next((key for key, value in ASPECT_RATIOS.items()
                               if abs(value - ratio) / value <= _RATIO_TOLERANCE), None)
                if aspect is None:
                    raise ValueError(
                        f"source.frame: {width}x{height} не выражается пропорцией из "
                        f"{sorted(ASPECT_RATIOS)} — до плана cutover используйте "
                        f"`width: 0` (авто по референсу)")
                data["aspect"] = aspect
                data["size_px"] = height
            elif height:
                data["size_px"] = height                  # width=0: как прежний --size
            elif width:
                data["size_px"] = width                   # вырожденный случай: квадрат
        elif cube_id == "encoder.qwen3vl":
            encoder.update(params)
        elif cube_id == "strategy.precision.fp8":
            generation["precision"] = "fp8"
            generation["precision_cache"] = params.get("cache_dir") or "cache/models/fp8_transformer"
        elif cube_id == "strategy.kv_offload":
            generation["kv_cache"] = params.get("mode") or "auto-offload-ref2"
        elif cube_id == "strategy.window_attention":
            window_px = int(params.get("window_attn_px") or 0)
            margin_px = int(params.get("window_attn_margin_px") or 0)
            if window_px:
                sampler["window_attn_px"] = window_px
            if margin_px:
                sampler["window_attn_margin_px"] = margin_px
        elif cube_id == "transformer.qwen_image21":
            generation["backend"] = params.get("backend") or "qwen_image21"
            generation["repo"] = params.get("repo")
            generation["revision"] = params.get("revision") or None
            generation["steps"] = params.get("steps")
            generation["guidance"] = params.get("guidance")
            generation["offload"] = params.get("offload")
            # свободные ключи бекенда пишем только если отличаются от умолчаний кубика:
            # иначе обратная проекция тащит в конфиг `offload_blocks: 1`, `vae_tile_px: 512` и
            # прочий «шум», которого в пресете не было (сравнение со снимком это ловит)
            if int(params.get("offload_blocks") or 1) != 1:
                sampler["offload_blocks"] = int(params["offload_blocks"])
            if not bool(params.get("offload_stream", True)):
                sampler["offload_stream"] = False
            if params.get("transformer_repo"):
                generation["transformer"] = (str(params["transformer_repo"]),
                                             str(params.get("transformer_subfolder")
                                                 or "transformer"))
            if params.get("lora_repo") and params.get("lora_file"):
                generation["lora"] = {"repo": params.get("lora_repo") or "",
                                      "filename": params.get("lora_file") or "",
                                      "scale": float(params["lora_scale"]),
                                      "strip_alpha": bool(params.get("lora_strip_alpha"))}
            else:
                generation["lora"] = None
        elif cube_id == "decoder.vae.qwen21":
            if int(params.get("tile_px", 512)) != 512:
                sampler["vae_tile_px"] = int(params["tile_px"])
        elif cube_id == "post.upscale.esrgan":
            upscale.update({"backend": "esrgan", "factor": params.get("factor", 0.0),
                            "target_height": params.get("target_height"),
                            "tile": params.get("tile", 256), "pad": params.get("pad", 32)})
            if params.get("weights"):
                upscale["weights"] = params["weights"]
        elif cube_id == "post.upscale.seedvr2":
            upscale.update({"backend": "seedvr2", "factor": params.get("factor", 0.0),
                            "target_height": params.get("height") or None})
        elif cube_id == "post.colorfix":
            upscale.update({"color_fix": params.get("mode") or "none",
                            "color_strength": params.get("strength", 1.0),
                            "color_sigma": params.get("sigma", 12.0)})
        # source.txtFileImage / rewrite.* / decoder.postprocess.image рантайму пока не видны:
        # промт и референсы приходят из CLI-пути, PE ещё не подключён к прогону

    if generation.get("precision") is None:
        generation["precision"] = "bf16"
    if encoder:
        data["encoder"] = encoder
    data["generation"] = generation
    if upscale:
        # `upscale: null` (апскейл выключен целиком) — когда кубика апскейла нет вообще
        data["upscale"] = upscale if upscale.get("backend") else None
    else:
        data["upscale"] = None
    if run is not None:
        seeds = tuple(int(seed) for seed in getattr(run, "seeds", ()) or ())
        if seeds:
            data["seed"] = seeds[0]
        out = getattr(run, "out", "")
        if out:
            data["out"] = out
        data["logs"] = bool(getattr(run, "logs", True))
        cache_dir = getattr(run, "cache_dir", "")
        if cache_dir:
            data["cache_dir"] = cache_dir
        policy = getattr(run, "cache_supersede", "")
        if policy:
            data["cache_supersede"] = policy
    data["preset"] = pipeline.name
    return data
