"""`python -m qwenstyle config show <preset>` — печать резолвленного конфига.

Ничего не загружает и не считает: чистый вывод RunConfig после применения `--set`,
чтобы видеть, какие именно параметры (и какой ключ кэша) получит прогон.

`--explain-params` печатает вместо этого таблицу настраиваемых параметров выбранных бекендов:
её объявляют сами адаптеры моделей (`parameters()` -> `ParameterSpec`), поэтому ядро не хранит
имён параметров моделей и не может с ними разойтись.
"""
from __future__ import annotations

import argparse
import json

from ..config import RunConfig, available_presets

NAME = "config"
HELP = "показать резолвленный конфиг пресета (с учётом --set)"


def _fmt(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(str(v) for v in value) + "]"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False) if value else "{}"
    return str(value)


def describe(cfg: RunConfig) -> list[str]:
    """Человекочитаемый дамп: ровно те имена, на которые ссылаются acceptance-проверки."""
    gen, up, enc = cfg.generation, cfg.upscale, cfg.encoder
    lora = gen.lora
    lines = [
        f"preset={cfg.preset or '-'}",
        f"input={cfg.input or '-'}",
        f"out={cfg.out}",
        f"prompt_file={cfg.prompt_file or '-'}",
        f"seed={cfg.seed}",
        f"size_px={_fmt(cfg.size_px)}",
        f"resolution={_fmt(cfg.resolution)}",
        f"preview={_fmt(cfg.preview)}",
        f"logs={_fmt(cfg.logs)}",
        f"steps={cfg.steps}" + ("" if gen.steps is not None else " (по профилю)"),
        f"encoder.name={enc.name}",
        f"encoder.cache_tag={enc.cache_tag}",
        f"encoder.repo={enc.repo}",
        f"encoder.revision={_fmt(enc.revision)}",
        f"encoder.local_dir={_fmt(enc.local_dir)}",
        f"encoder.requires_vision={_fmt(enc.requires_vision)}",
        f"encoder.dtype={enc.dtype}",
        f"encoder.device={enc.device}",
        f"generation.backend={gen.backend}",
        f"generation.repo={gen.repo}",
        f"generation.revision={_fmt(gen.revision)}",
        f"generation.transformer={_fmt(gen.transformer)}",
        f"generation.lora.repo={lora.repo if lora else '-'}",
        f"generation.lora.filename={lora.filename if lora else '-'}",
        f"generation.lora.scale={_fmt(lora.scale if lora else None)}",
        f"generation.lora.strip_alpha={_fmt(lora.strip_alpha if lora else None)}",
        f"generation.guidance={_fmt(gen.guidance)}",
        f"generation.precision={gen.precision}",
        f"generation.precision_cache={_fmt(gen.precision_cache)}",
        f"generation.offload={gen.offload}",
        f"generation.kv_cache={gen.kv_cache}",
        f"generation.sampler_kwargs={_fmt(gen.sampler_kwargs)}",
        f"upscale.backend={up.backend}",
        f"upscale.factor={_fmt(up.factor)}",
        f"upscale.target_height={_fmt(up.target_height)}",
        f"upscale.weights={_fmt(up.weights)}",
        f"upscale.tile={up.tile}",
        f"upscale.pad={up.pad}",
        f"upscale.color_fix={up.color_fix}",
        f"upscale.color_strength={_fmt(up.color_strength)}",
        f"upscale.color_sigma={_fmt(up.color_sigma)}",
    ]
    batch = cfg.batch
    if batch is None:
        lines.append("batch=-")
    else:
        lines += [
            f"batch.seeds={batch.seeds}",
            f"batch.out_dir={batch.out_dir}",
            f"batch.keep_gen={_fmt(batch.keep_gen)}",
            f"batch.resume={_fmt(batch.resume)}",
        ]
    return lines


def _table(headers: list[str], rows: list[list[str]], indent: str = "  ") -> None:
    """Таблица с выравниванием по колонкам (как `models list`, без внешних зависимостей)."""
    widths = [max([len(str(h))] + [len(str(row[i])) for row in rows]) for i, h in enumerate(headers)]
    for row in [headers] + rows:
        line = "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        print(indent + line)


def canonical_of(spec: Any) -> str:
    """Канонический путь `--set` для параметра бекенда (или его поле конфига, если кубика нет).

    `--set` принимает только пути параметров кубиков (`qwenstyle/paths.py`), поэтому таблица
    `--explain-params` показывает именно их; у моделей без кубиков (H3, FLUX.2) остаётся поле
    `RunConfig` с пометкой.
    """
    from ..paths import CUBE_TO_CONFIG, EXTRA_CONFIG_ALIASES

    canonical = EXTRA_CONFIG_ALIASES.get(spec.path)
    if canonical:
        return canonical
    for cube_path, config_path in CUBE_TO_CONFIG.items():
        if config_path == spec.path:
            return cube_path
    return f"{spec.path} (нет параметра кубика: ключ только для RunConfig)"


def explain_params(cfg: RunConfig) -> None:
    """Таблица `scope | name | type | default | range/choices | readonly | doc` по бекендам.

    Источник — `parameters()` бекендов генерации и апскейла (`qwenstyle/backends/base.py`).
    В конце каждой таблицы — готовые пути для `--set` (они же попадают в ошибку неизвестного
    ключа `generation.sampler_kwargs.*`, см. `config.check_sampler_kwargs`).
    """
    from ..config import backend_parameters

    headers = ["scope", "name", "type", "default", "range/choices", "readonly", "doc"]
    pairs = (("generation", cfg.generation.backend), ("upscale", cfg.upscale.backend))
    for kind, name in pairs:
        specs = backend_parameters(kind, name)
        title = "модель генерации" if kind == "generation" else "апскейл"
        print(f"\n{title}: {name} — параметров {len(specs)}")
        if not specs:
            print("  (бекенд не объявил parameters(): настраиваемых параметров нет)")
            continue
        rows = [[
            spec.scope,
            spec.name,
            spec.type,
            spec.format_default(),
            spec.range_text(),
            "readonly" if spec.readonly else "-",
            (spec.doc or "-")[:100] + ("…" if len(spec.doc or "") > 100 else ""),
        ] for spec in specs]
        _table(headers, rows)
        settable = [canonical_of(spec) for spec in specs if not spec.readonly]
        fixed = [f"{canonical_of(spec)}={spec.format_default()}"
                 for spec in specs if spec.readonly]
        print("  --set ключи: " + (", ".join(settable) or "—"))
        if fixed:
            print("  фиксировано моделью (readonly): " + ", ".join(fixed))


def cubes_view(cfg: RunConfig, *, as_json: bool = False) -> int:
    """Печать композиции кубиков для конфига (план `cube-architecture`, шаг 6).

    Ничего не грузит: показывает, из каких стадий состоит пресет, какие стратегии он включает
    и какие ошибки видит композитор. Код возврата: 0 — композиция чистая, 1 — есть ошибки.
    """
    from ..composition import check, spec_of

    spec = spec_of(cfg)
    composition = check(cfg)
    if as_json:
        print(json.dumps({
            "name": spec.name,
            "ok": composition.ok,
            "budget_vram_gib": round(composition.budget_gib, 3),
            "provides": sorted(composition.provides),
            "ports_in": sorted(composition.ports_in),
            "ports_out": sorted(composition.ports_out),
            "cubes": [{"id": b.cube_id, "params": dict(b.params)} for b in spec.cubes],
            "issues": [{"code": i.code, "cube": i.cube_id, "field": i.field,
                        "text": i.text} for i in composition.issues],
        }, ensure_ascii=False, indent=2, default=str))
        return 0 if composition.ok else 1
    head = "ok" if composition.ok else f"{len(composition.issues)} ошиб."
    print(f"composition={spec.name} ({head})  бюджет VRAM {composition.budget_gib:.1f} ГиБ")
    print(f"inputs={', '.join(sorted(composition.ports_in)) or '—'} -> "
          f"outputs={', '.join(sorted(composition.ports_out)) or '—'}"
          f"  provides={', '.join(sorted(composition.provides)) or '—'}")
    for index, binding in enumerate(spec.cubes, start=1):
        cube = next((c for c in composition.cubes if c.id == binding.cube_id), None)
        stage = cube.stage.value if cube else "?"
        ports = cube.summary().split(None, 2)[-1] if cube else ""
        params = " ".join(f"{k}={_fmt(v)}" for k, v in binding.params.items())
        print(f"  {index:2}. {binding.cube_id:24} {stage:11} {ports}")
        if params:
            print(f"      params: {params}")
    if composition.issues:
        print("ошибки композиции:")
        for issue in composition.issues:
            print(f"  - [{issue.code}] {issue.text}")
    return 0 if composition.ok else 1


def add_args(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="config_action", required=True, metavar="<action>")
    show = actions.add_parser(
        "show",
        help="напечатать конфиг пресета",
        description="Печатает резолвленный RunConfig; --set позволяет переопределить точечно.",
        prog=f"{parser.prog} show",
    )
    show.add_argument("preset", nargs="?", default="base", metavar="PRESET",
                      help=f"имя пресета ({', '.join(available_presets())}) или путь к YAML; по умолчанию base")
    show.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                      help="переопределить параметр кубика, например "
                           "transformer.qwen_image21.steps=8 или encoder.qwen3vl.name=base "
                           "(можно повторять)")
    show.add_argument("--json", action="store_true", help="вывести конфиг как JSON (для скриптов)")
    show.add_argument("--explain-params", dest="explain_params", action="store_true",
                      help="таблица настраиваемых параметров бекендов (scope/name/type/default/"
                           "range/readonly/doc) вместо конфига; источник — parameters()")
    show.add_argument("--cubes", dest="cubes", action="store_true",
                      help="композиция кубиков пресета (стадии, стратегии, бюджет VRAM, ошибки); "
                           "с --json — машинный формат для UI")
    show.set_defaults(_config_action="show")


def run(args: argparse.Namespace) -> int:
    action = getattr(args, "_config_action", None) or getattr(args, "config_action", None)
    if action != "show":                       # argparse с required=True сюда не пустит
        print(f"[config] неизвестное действие {action!r}")
        return 2
    from . import parse_sets

    cfg = RunConfig.from_profile(args.preset, **parse_sets(args.sets))
    if getattr(args, "cubes", False):
        return cubes_view(cfg, as_json=args.json)
    if getattr(args, "explain_params", False):
        explain_params(cfg)
        return 0
    if args.json:
        print(json.dumps(cfg.to_mapping(), ensure_ascii=False, indent=2))
        return 0
    for line in describe(cfg):
        print(line)
    return 0


__all__ = ["NAME", "HELP", "add_args", "run", "describe", "explain_params", "cubes_view"]
