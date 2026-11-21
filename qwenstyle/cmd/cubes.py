"""`python -m qwenstyle cubes` — API кубиков для UI: list / show / test / schema.

Зачем отдельная подкоманда (план `cube-architecture`, шаг 8): конструктор пайплайна в UI не должен
импортировать Python и читать исходники — он спрашивает данные. Здесь они и отдаются:

    cubes list                 # id, стадия, порты, число параметров, provides, бюджет
    cubes show encoder.qwen3vl # порты, параметры (с диапазонами и doc), бюджет, требования
    cubes test decoder.vae.qwen21   # self_test без весов и CUDA (+ время)
    cubes schema [<id>]        # JSON Schema (draft-07) параметров кубика или всего набора
    cubes check pipeline.yaml  # проверка композиции из файла (то же, что `compose --check`)

Ничего не загружает: ни весов, ни CUDA, ни сети. Код возврата: 0 — успех/композиция сходится,
1 — ошибки композиции или проваленный `self_test`, 2 — ошибка вызова (нет такого кубика/файла).

Схема (`schema`) собирается из `ParameterSpec` — того же дескриптора, по которому валидируются
`--set` и пресеты, поэтому UI не может разойтись с ядром: тип, `enum`-варианты, минимум/максимум и
`default` берутся из одного места.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from .. import cubes as _cubes                      # noqa: F401 — регистрирует все кубики
from ..backends.base import ParameterSpec
from ..compose import CubeBinding, PipelineSpec, compose
from ..cube import CubeContext, Stage, all_cubes, cubes_by_stage, get_cube

NAME = "cubes"
HELP = "кубики пайплайна: list / show / test / schema / check (API для UI)"


# --------------------------------------------------------------------------- вывод

def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (tuple, list, frozenset)):
        return "|".join(str(v) for v in value) if value else "-"
    return str(value)


def _table(headers: list[str], rows: list[list[str]], indent: str = "  ") -> None:
    widths = [max([len(str(h))] + [len(str(row[i])) for row in rows]) for i, h in enumerate(headers)]
    for row in [headers] + rows:
        print(indent + "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)).rstrip())


def cube_brief(cube: Any) -> dict[str, Any]:
    """Краткая запись кубика для `list --json` (то, что нужно списку в UI)."""
    return {
        "id": cube.id,
        "stage": cube.stage.value,
        "consumes": sorted(cube.consumes),
        "reads": sorted(cube.reads),
        "outputs": sorted(cube.outputs),
        "provides": sorted(cube.provides),
        "requires": sorted(cube.requires),
        "params": len(cube.params),
        "budget_vram_gib": cube.budget.weights_fp8_gib if cube.budget.weights_fp8_gib is not None
        else cube.budget.weights_bf16_gib,
        "doc": cube.doc,
    }


def param_brief(spec: ParameterSpec) -> dict[str, Any]:
    """Параметр кубика в терминах UI: имя, тип, домен, дефолт, readonly, doc."""
    return {
        "name": spec.name,
        "scope": spec.scope,
        "type": spec.type,
        "default": spec.default,
        "choices": list(spec.choices) if spec.choices else None,
        "minimum": spec.minimum,
        "maximum": spec.maximum,
        "readonly": spec.readonly,
        "set_path": spec.path,
        "doc": spec.doc,
    }


def cube_full(cube: Any) -> dict[str, Any]:
    data = cube_brief(cube)
    data["params"] = [param_brief(p) for p in cube.params]
    data["budget"] = {
        "weights_bf16_gib": cube.budget.weights_bf16_gib,
        "weights_fp8_gib": cube.budget.weights_fp8_gib,
        "encoder_gib": cube.budget.encoder_gib,
        "disk_gib": cube.budget.disk_gib,
        "notes": cube.budget.notes,
    }
    return data


# --------------------------------------------------------------------------- JSON Schema

def param_schema(spec: ParameterSpec) -> dict[str, Any]:
    """JSON Schema (draft-07) одного параметра — из тех же дескрипторов, что валидируют прогон."""
    base: dict[str, Any] = {"title": spec.name, "description": spec.doc or ""}
    if spec.type == "int":
        base["type"] = "integer"
    elif spec.type == "float":
        base["type"] = "number"
    elif spec.type == "bool":
        base["type"] = "boolean"
    else:
        base["type"] = "string"
    if spec.choices:
        base["enum"] = list(spec.choices)
    if spec.minimum is not None:
        base["minimum"] = spec.minimum
    if spec.maximum is not None:
        base["maximum"] = spec.maximum
    base["default"] = spec.default
    if spec.readonly:
        base["readOnly"] = True
    return base


def cube_schema(cube: Any) -> dict[str, Any]:
    """Схема параметров кубика: объект без лишних ключей, `required` — readonly с `None`."""
    properties = {spec.name: param_schema(spec) for spec in cube.params}
    required = [spec.name for spec in cube.params if spec.readonly and spec.default is None]
    schema: dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": f"qwenstyle://cube/{cube.id}",
        "title": cube.id,
        "description": cube.doc or "",
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return schema


def pipeline_schema() -> dict[str, Any]:
    """Схема файла композиции: `{name, inputs, cubes: [{id, params}]}`."""
    ids = [cube.id for cube in all_cubes()]
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "qwenstyle://pipeline",
        "title": "pipeline",
        "type": "object",
        "required": ["cubes"],
        "additionalProperties": False,
        "properties": {
            "name": {"type": "string"},
            "profile": {"type": ["string", "null"]},
            "inputs": {"type": "array",
                       "items": {"type": "string",
                                 "enum": sorted({"prompt", "refs", "conditioning", "latents",
                                                 "frames", "pixels"})}},
            "cubes": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["id"],
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string", "enum": ids},
                        "params": {"type": "object"},
                    },
                },
            },
        },
    }


# --------------------------------------------------------------------------- композиция из файла

def load_pipeline_spec(path: str) -> PipelineSpec:
    """Прочитать файл композиции (`yaml` или `json`) в `PipelineSpec`."""
    from pathlib import Path

    import yaml

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"нет файла композиции {path}")
    raw = target.read_text(encoding="utf-8")
    data = json.loads(raw) if target.suffix.lower() == ".json" else yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: ожидался объект с ключом `cubes`")
    unknown = set(data) - {"name", "cubes", "inputs", "profile"}
    if unknown:
        raise ValueError(f"{path}: неизвестные ключи {sorted(unknown)}; "
                         f"допустимые: name, cubes, inputs, profile")
    cubes = data.get("cubes")
    if not isinstance(cubes, list) or not cubes:
        raise ValueError(f"{path}: `cubes` обязан быть непустым списком")
    bindings: list[CubeBinding] = []
    for index, entry in enumerate(cubes):
        if not isinstance(entry, dict) or "id" not in entry:
            raise ValueError(f"{path}: элемент cubes[{index}] обязан быть объектом с `id`")
        extra = set(entry) - {"id", "params"}
        if extra:
            raise ValueError(f"{path}: cubes[{index}] содержит неизвестные ключи {sorted(extra)}")
        params = entry.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(f"{path}: cubes[{index}].params обязан быть объектом")
        bindings.append(CubeBinding(str(entry["id"]), params))
    inputs = data.get("inputs")
    return PipelineSpec(
        name=str(data.get("name") or target.stem),
        cubes=tuple(bindings),
        inputs=frozenset(inputs) if inputs else frozenset({"prompt", "refs"}),
        profile=data.get("profile"),
    )


def check_file(path: str, *, as_json: bool = False) -> int:
    """Проверить композицию из файла и напечатать результат. 0 — сходится, 1 — нет."""
    try:
        spec = load_pipeline_spec(path)
    except (OSError, ValueError) as exc:
        print(f"[cubes] {exc}", file=sys.stderr, flush=True)
        return 2
    composition = compose(spec)
    if as_json:
        print(json.dumps({
            "name": spec.name,
            "ok": composition.ok,
            "budget_vram_gib": round(composition.budget_gib, 3),
            "ports_in": sorted(composition.ports_in),
            "ports_out": sorted(composition.ports_out),
            "provides": sorted(composition.provides),
            "cubes": [b.cube_id for b in spec.cubes],
            "issues": [{"code": i.code, "cube": i.cube_id, "field": i.field, "text": i.text}
                       for i in composition.issues],
        }, ensure_ascii=False, indent=2))
        return 0 if composition.ok else 1
    status = "ok" if composition.ok else f"{len(composition.issues)} ошиб."
    print(f"composition={spec.name} ({status})  бюджет VRAM {composition.budget_gib:.1f} ГиБ")
    for index, binding in enumerate(spec.cubes, start=1):
        print(f"  {index:2}. {binding.cube_id}")
    for issue in composition.issues:
        print(f"  - [{issue.code}] {issue.text}")
    return 0 if composition.ok else 1


# --------------------------------------------------------------------------- CLI

def add_args(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="cubes_action", required=True, metavar="<action>")

    listing = actions.add_parser("list", help="список кубиков",
                                description="Кубики реестра: стадия, порты, параметры, бюджет.")
    listing.add_argument("--stage", choices=[s.value for s in Stage], default=None,
                         help="только кубики этой стадии")
    listing.add_argument("--json", action="store_true", help="машинный формат (для UI)")

    show = actions.add_parser("show", help="один кубик подробно",
                              description="Порты, параметры (с диапазонами и doc), бюджет, требования.")
    show.add_argument("id", metavar="CUBE_ID", help="например encoder.qwen3vl")
    show.add_argument("--json", action="store_true", help="машинный формат (для UI)")

    test = actions.add_parser("test", help="self_test кубика без весов и CUDA",
                              description="Гоняет self_test на синтетике: секунды, без GPU.")
    test.add_argument("id", nargs="?", default=None, metavar="CUBE_ID",
                      help="кубик; без аргумента — все кубики")
    test.add_argument("--json", action="store_true", help="машинный формат (для UI)")

    schema = actions.add_parser("schema", help="JSON Schema (draft-07)",
                                description="Схема параметров кубика (или всего набора) для UI.")
    schema.add_argument("id", nargs="?", default=None, metavar="CUBE_ID",
                        help="кубик; без аргумента — все кубики плюс схема файла композиции")

    check = actions.add_parser("check", help="проверить композицию из файла",
                               description="То же, что `compose --spec <файл> --check`: без весов и GPU.")
    check.add_argument("spec", metavar="FILE", help="YAML/JSON с ключом cubes")
    check.add_argument("--json", action="store_true", help="машинный формат (для UI)")


def run(args: argparse.Namespace) -> int:
    action = args.cubes_action
    if action == "list":
        cubes = cubes_by_stage(Stage(args.stage)) if args.stage else all_cubes()
        if args.json:
            print(json.dumps([cube_brief(c) for c in cubes], ensure_ascii=False, indent=2))
            return 0
        rows = [[c.id, c.stage.value, _fmt(sorted(c.consumes | c.reads)),
                 _fmt(sorted(c.outputs)), str(len(c.params)), _fmt(sorted(c.provides)),
                 _fmt(cube_brief(c)["budget_vram_gib"])] for c in cubes]
        _table(["id", "stage", "in", "out", "params", "provides", "vram GiB"], rows)
        print(f"\nвсего: {len(cubes)}")
        return 0

    if action == "show":
        try:
            cube = get_cube(args.id)
        except LookupError as exc:
            print(f"[cubes] {exc}", file=sys.stderr, flush=True)
            return 2
        if args.json:
            print(json.dumps(cube_full(cube), ensure_ascii=False, indent=2))
            return 0
        print(f"{cube.id} ({cube.stage.value})")
        print(f"  {cube.doc}")
        print(f"  consumes={_fmt(sorted(cube.consumes))} reads={_fmt(sorted(cube.reads))} "
              f"outputs={_fmt(sorted(cube.outputs))}")
        print(f"  provides={_fmt(sorted(cube.provides))} requires={_fmt(sorted(cube.requires))}")
        print(f"  бюджет: vram bf16={_fmt(cube.budget.weights_bf16_gib)} "
              f"fp8={_fmt(cube.budget.weights_fp8_gib)} encoder={_fmt(cube.budget.encoder_gib)} "
              f"disk={_fmt(cube.budget.disk_gib)}")
        if cube.params:
            rows = [[p.name, p.type, p.format_default(), p.range_text(),
                     "readonly" if p.readonly else "-", p.path, (p.doc or "-")[:70]]
                    for p in cube.params]
            _table(["name", "type", "default", "range", "flags", "--set path", "doc"], rows)
        return 0

    if action == "test":
        targets = ([get_cube(args.id)] if args.id else list(all_cubes()))
        results = []
        for cube in targets:
            started = time.perf_counter()
            if cube.self_test is None:
                results.append({"id": cube.id, "status": "skip", "s": 0.0,
                                "error": "нет self_test"})
                continue
            try:
                cube.self_test(CubeContext())
            except Exception as exc:                                # noqa: BLE001
                results.append({"id": cube.id, "status": "fail",
                                "s": round(time.perf_counter() - started, 3),
                                "error": f"{type(exc).__name__}: {exc}"})
            else:
                results.append({"id": cube.id, "status": "ok",
                                "s": round(time.perf_counter() - started, 3), "error": ""})
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for row in results:
                tail = f"  {row['error']}" if row["error"] else ""
                print(f"{row['status'].upper():4} {row['id']:28} {row['s']:>6.2f} с{tail}")
        return 0 if all(row["status"] in ("ok", "skip") for row in results) else 1

    if action == "schema":
        if args.id:
            try:
                cube = get_cube(args.id)
            except LookupError as exc:
                print(f"[cubes] {exc}", file=sys.stderr, flush=True)
                return 2
            print(json.dumps(cube_schema(cube), ensure_ascii=False, indent=2))
            return 0
        bundle = {"pipeline": pipeline_schema(),
                  "cubes": {cube.id: cube_schema(cube) for cube in all_cubes()}}
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
        return 0

    if action == "check":
        return check_file(args.spec, as_json=args.json)

    print(f"[cubes] неизвестное действие {action!r}", file=sys.stderr, flush=True)
    return 2


__all__ = ["HELP", "NAME", "add_args", "check_file", "cube_brief", "cube_full", "cube_schema",
           "load_pipeline_spec", "param_schema", "pipeline_schema", "run"]
