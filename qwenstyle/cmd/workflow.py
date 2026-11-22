"""`python -m qwenstyle workflow` — файлы воркфлоу: `check`, `new`, `show`.

Зачем (план `workflow-layout`, шаг 3): `workflow/<проект>/workflow.yaml` — самодостаточное описание
пайплайна, и с ним нужны три операции без запуска тяжёлых стадий:

    workflow check out/tests/workflow-ok.yaml    # композиция + параметры, без весов и GPU
    workflow show  workflow/nita-camp-model/workflow.yaml --json
    workflow new --from VRAM16-preview --name my-project   # копия шаблона (развёрнутая, без extends)

`check` — тот же компоновщик, что `compose --check`, плюс проверка `run:`-секции; код возврата 1 и
строки `[код] сообщение` без трейсбека. `new` не подключает шаблон, а копирует его тело: файл
воркфлоу обязан быть самодостаточным (наследования в формате нет).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..workflow import Workflow, load, load_template

NAME = "workflow"
HELP = "файлы воркфлоу: check | show | new (композиция кубиков без весов и GPU)"


def _paths_of(workflow: Workflow) -> list[dict]:
    return [{"id": binding.cube_id, "params": dict(binding.params)}
            for binding in workflow.pipeline.cubes]


def check_file(path: str, *, as_json: bool = False) -> int:
    """Разобрать и проверить файл воркфлоу. 0 — всё сходится, 1 — ошибки, 2 — не читается."""
    try:
        workflow = load(path)
    except (OSError, ValueError) as exc:
        print(f"[workflow] {exc}", file=sys.stderr, flush=True)
        return 2
    composition = workflow.composition()
    if as_json:
        print(json.dumps({
            "path": str(path),
            "name": workflow.name,
            "ok": composition.ok,
            "inputs": sorted(workflow.pipeline.inputs),
            "ports_out": sorted(composition.ports_out),
            "provides": sorted(composition.provides),
            "budget_vram_gib": round(composition.budget_gib, 3),
            "run": workflow.run.to_mapping(),
            "cubes": _paths_of(workflow),
            "issues": [{"code": i.code, "cube": i.cube_id, "field": i.field, "text": i.text}
                       for i in composition.issues],
        }, ensure_ascii=False, indent=2, default=str))
        return 0 if composition.ok else 1
    status = "ok" if composition.ok else f"{len(composition.issues)} ошиб."
    print(f"composition={workflow.name} ({status})  бюджет VRAM {composition.budget_gib:.1f} ГиБ")
    print(f"  файл: {Path(path).as_posix()}")
    print(f"  входы: {', '.join(sorted(workflow.pipeline.inputs)) or '—'}"
          f"  выходы: {', '.join(sorted(composition.ports_out)) or '—'}")
    for index, binding in enumerate(workflow.pipeline.cubes, start=1):
        print(f"  {index:2}. {binding.cube_id}")
    if workflow.run.seeds or workflow.run.count:
        print(f"  run: seeds={workflow.run.seed_spec() or 'случайные'} "
              f"count={workflow.run.count or '-'} out={workflow.run.out}")
    for issue in composition.issues:
        print(f"  - [{issue.code}] {issue.text}")
    return 0 if composition.ok else 1


def show_file(path: str, *, as_json: bool = False) -> int:
    try:
        workflow = load(path)
    except (OSError, ValueError) as exc:
        print(f"[workflow] {exc}", file=sys.stderr, flush=True)
        return 2
    if as_json:
        print(json.dumps({"path": str(path), "name": workflow.name,
                          "inputs": sorted(workflow.pipeline.inputs),
                          "run": workflow.run.to_mapping(), "cubes": _paths_of(workflow)},
                         ensure_ascii=False, indent=2, default=str))
        return 0
    print(f"{workflow.name}  ({Path(path).as_posix()})")
    print(f"  run: seeds={workflow.run.seed_spec() or 'случайные'} out={workflow.run.out} "
          f"logs={workflow.run.logs} cache_supersede={workflow.run.cache_supersede}")
    for index, binding in enumerate(workflow.pipeline.cubes, start=1):
        params = " ".join(f"{key}={value}" for key, value in binding.params.items())
        print(f"  {index:2}. {binding.cube_id}" + (f"  {params}" if params else ""))
    return 0


def _template_or_preset(source: str) -> Workflow | None:
    """Шаблон по имени; пока шаблонов нет — тот же файл из пресета (переходный режим).

    До шага 4 (каталог `qwenstyle/templates/`) `workflow new --from VRAM16-preview` собирает
    воркфлоу из пресета через проекцию `RunConfig -> PipelineSpec`: результат — такой же
    самодостаточный файл, только значения пришли не из шаблона.
    """
    from ..composition import spec_of
    from ..config import ConfigError, RunConfig
    from ..runspec import RunSpec

    try:
        return load_template(source)
    except (OSError, ValueError) as exc:
        template_error = str(exc)
    try:
        cfg = RunConfig.from_profile(source)
    except (ConfigError, ValueError) as exc:
        print(f"[workflow] {template_error}; и пресет {source!r} не читается: {exc}",
              file=sys.stderr, flush=True)
        return None
    spec = spec_of(cfg, name=str(source).replace("/", "-"))
    run = RunSpec(seeds=(int(cfg.seed),) if cfg.seed else (), out="out", logs=bool(cfg.logs))
    print(f"[workflow] шаблонов пока нет: {source} собран из пресета "
          f"({len(spec.cubes)} кубиков)", flush=True)
    return Workflow(pipeline=spec, run=run, path=None)


def new_file(*, source: str, name: str, out_dir: str | None, force: bool = False) -> int:
    """Создать файл воркфлоу копией шаблона (тело развёрнуто, ссылок нет)."""
    template = _template_or_preset(source)
    if template is None:
        return 2
    target_dir = Path(out_dir) if out_dir else Path("workflow") / name
    target = target_dir / "workflow.yaml"
    if target.exists() and not force:
        print(f"[workflow] {target.as_posix()} уже существует (перезапись — --force)",
              file=sys.stderr, flush=True)
        return 2
    lines = [
        f"# workflow «{name}»: создан копией шаблона {source} ({template.name}).",
        "# Файл самодостаточен: правьте cubes/run прямо здесь, ссылок на шаблон нет.",
        "version: 1",
        f"name: {name}",
        "cubes:",
    ]
    for binding in template.pipeline.cubes:
        lines.append(f"  - id: {binding.cube_id}")
        if binding.params:
            lines.append("    params:")
            for key, value in binding.params.items():
                lines.append(f"      {key}: {_yaml_scalar(value)}")
    run = template.run.to_mapping()
    lines += ["run:", f"  seeds: {list(run['seeds']) or []}", f"  out: {run['out']}",
              f"  logs: {'true' if run['logs'] else 'false'}"]
    if run["count"]:
        lines.append(f"  count: {run['count']}")
    if run["cache_dir"]:
        lines.append(f"  cache_dir: {run['cache_dir']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".yaml.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(f"[workflow] создан {target.as_posix()} (шаблон {template.name}, "
          f"{len(template.pipeline.cubes)} кубиков)")
    return 0


def _yaml_scalar(value: object) -> str:
    """Скаляр в YAML-виде: строки с `:`/пробелами — в кавычках, bool — строчными."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if not text:
        return '""'
    if any(char in text for char in ":#") or text.strip() != text:
        return json.dumps(text, ensure_ascii=False)
    return text


def migrate(src: str | None = None, dst: str | None = None, *, apply: bool = False) -> int:
    """`prompts/` -> `workflow/`: переименовать каталоги и сконвертировать YAML.

    План по умолчанию (`apply=False`): печатается, что будет переименовано и какие `project.yaml`
    станут `workflow.yaml`. При `--apply`:

    * каталог проектов переименовывается (`prompts` -> `workflow`) одной операцией;
    * каждый `project.yaml` превращается в `workflow.yaml` (композиция кубиков + `run`), остальные
      YAML вариантов — тоже (`nita-tokyo.yaml` -> такой же самодостаточный файл);
    * `out/` и `cache/` внутри проектов не трогаются, `references`-каталоги остаются на месте;
    * запись атомарная (`*.tmp` + `os.replace`), старый YAML остаётся рядом как `.yaml.bak`.

    Данные пользователя (`prompts/` в `.gitignore`) — поэтому функция ничего не удаляет молча и
    всегда печатает итог: переименовано / сконвертировано / пропущено.
    """
    import shutil

    from ..composition import spec_of
    from ..config import ConfigError, RunConfig

    root = Path(src) if src else Path("prompts")
    target = Path(dst) if dst else Path("workflow")
    if not root.is_dir():
        print(f"[workflow] нет каталога {root.as_posix()}", file=sys.stderr, flush=True)
        return 2
    projects = sorted(path for path in root.iterdir() if path.is_dir())
    yamls = sorted(root.rglob("*.yaml"))
    print(f"[workflow] перенос {root.as_posix()} -> {target.as_posix()}")
    print(f"  проектов: {len(projects)}; yaml-файлов: {len(yamls)}")
    for path in yamls:
        print(f"  yaml: {path.relative_to(root).as_posix()}")
    if not apply:
        print("  (план; для выполнения — --apply)")
        return 0
    if target.exists():
        print(f"[workflow] {target.as_posix()} уже существует — миграция не делается",
              file=sys.stderr, flush=True)
        return 2
    converted = skipped = 0
    shutil.move(str(root), str(target))
    for path in sorted(target.rglob("*.yaml")):
        text = path.read_text(encoding="utf-8")
        if "version:" in text and "cubes:" in text:
            skipped += 1
            continue
        try:
            converted_data = _convert_yaml(path)
        except (ConfigError, ValueError, OSError) as exc:
            print(f"  пропущен {path.as_posix()}: {exc}", flush=True)
            skipped += 1
            continue
        new_path = path.with_name("workflow.yaml" if path.name == "project.yaml" else path.name)
        temporary = new_path.with_suffix(".yaml.tmp")
        temporary.write_text(converted_data, encoding="utf-8")
        temporary.replace(new_path)
        if new_path != path:
            path.rename(path.with_suffix(".yaml.bak"))
        converted += 1
        print(f"  сконвертирован {new_path.as_posix()}", flush=True)
    print(f"[workflow] готово: переименовано каталогов {len(projects)}, "
          f"сконвертировано {converted}, пропущено {skipped}")
    return 0


def _convert_yaml(path: Path) -> str:
    """`project.yaml` (RunConfig) -> текст файла воркфлоу."""
    import yaml

    from ..composition import spec_of
    from ..config import RunConfig

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("ожидался словарь")
    preset = data.get("preset")
    if not preset:
        raise ValueError("в файле нет `preset` — нечего разворачивать в кубики")
    overrides = {key: value for key, value in data.items()
                 if key not in ("preset", "prompt", "refs", "out", "seeds", "count")}
    # частичный `generation.lora` (например только `scale: 0.6`) — это правка адаптера пресета,
    # а не новое определение: дополняем repo/filename из пресета, иначе LoraConfig не собрать
    base = RunConfig.from_profile(str(preset))
    generation = dict(overrides.get("generation") or {})
    lora = generation.get("lora")
    if isinstance(lora, dict) and base.generation.lora is not None:
        merged_lora = {"repo": base.generation.lora.repo,
                       "filename": base.generation.lora.filename,
                       "scale": base.generation.lora.scale,
                       "strip_alpha": base.generation.lora.strip_alpha}
        merged_lora.update({key: value for key, value in lora.items() if value is not None})
        generation["lora"] = merged_lora
        overrides["generation"] = generation
    cfg = base.merged(**overrides)
    spec = spec_of(cfg, name=path.stem)
    lines = [f"# Воркфлоу «{path.stem}»: сконвертирован из {path.name} (пресет {preset}).",
             "# Файл самодостаточен: правится здесь, ссылок на пресеты/шаблоны нет.",
             "version: 1",
             f"name: {path.stem}",
             "cubes:"]
    # референсы: прежний проект брал их из каталога (`refs/` или `refs:` в yaml), новый файл
    # обязан перечислить их явно — иначе `gen --workflow` пойдёт без референсов
    from ..io import discover_images

    refs_name = str(data.get("refs") or "refs")
    refs_dir = path.parent / refs_name
    refs = discover_images(refs_dir, on_duplicate=lambda dup, kept: print(
        f"  refs: дубликат {dup.name} — взят {kept.name}", flush=True))
    for binding in spec.cubes:
        filled = {key: value for key, value in binding.params.items() if value not in ("", None)}
        if binding.cube_id == "source.txtFileImage":
            filled["prompt"] = str(data.get("prompt") or "")
            for index, image in enumerate(refs[:10], start=1):
                filled[f"Image{index}"] = image.relative_to(path.parent).as_posix()
        lines.append(f"  - id: {binding.cube_id}")
        if filled:
            lines.append("    params:")
            for key, value in filled.items():
                lines.append(f"      {key}: {_yaml_scalar(value)}")
    raw_seeds = data.get("seeds")
    if raw_seeds in (None, ""):
        seeds: list[int] = []
    elif isinstance(raw_seeds, (list, tuple)):
        seeds = [int(item) for item in raw_seeds]
    else:
        seeds = [int(part) for part in str(raw_seeds).replace(" ", "").split(",") if part]
    lines += ["run:", f"  seeds: {seeds}", f"  out: {data.get('out') or 'out'}"]
    if data.get("count"):
        lines.append(f"  count: {int(data['count'])}")
    return "\n".join(lines) + "\n"


def add_args(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="workflow_action", required=True, metavar="<action>")

    check = actions.add_parser("check", help="проверить файл воркфлоу",
                               description="Композиция + run без весов и GPU; rc 1 при ошибках.")
    check.add_argument("path", metavar="FILE", help="workflow.yaml (или его вариант)")
    check.add_argument("--json", action="store_true", help="машинный формат (для UI)")

    show = actions.add_parser("show", help="показать кубики файла",
                              description="Список кубиков и параметров запуска без проверки.")
    show.add_argument("path", metavar="FILE")
    show.add_argument("--json", action="store_true", help="машинный формат (для UI)")

    new = actions.add_parser("new", help="создать воркфлоу копией шаблона",
                             description="Тело шаблона разворачивается в файл: ссылок на шаблон нет.")
    new.add_argument("--from", dest="source", required=True, metavar="TEMPLATE",
                     help="имя шаблона (VRAM16-preview) или «модель/имя»")
    new.add_argument("--name", required=True, metavar="NAME", help="имя воркфлоу (каталог и title)")
    new.add_argument("--dir", default=None, metavar="DIR",
                     help="каталог (по умолчанию workflow/<name>); файл — DIR/workflow.yaml")
    new.add_argument("--force", action="store_true", help="перезаписать существующий файл")

    move = actions.add_parser("move", help="перенести prompts/ -> workflow/ (миграция)",
                              description="Переименование каталога и конвертация project.yaml в "
                                          "workflow.yaml; по умолчанию только план (--dry-run).")
    move.add_argument("--dir", default=None, metavar="DIR",
                      help="что переносить (по умолчанию prompts); удобно для проверки на копии")
    move.add_argument("--to", default=None, metavar="DIR",
                      help="куда (по умолчанию workflow рядом с исходным каталогом)")
    move.add_argument("--apply", action="store_true",
                      help="выполнить перенос (без флага — только план)")


def run(args: argparse.Namespace) -> int:
    action = args.workflow_action
    if action == "check":
        return check_file(args.path, as_json=bool(args.json))
    if action == "show":
        return show_file(args.path, as_json=bool(args.json))
    if action == "new":
        return new_file(source=args.source, name=args.name, out_dir=args.dir,
                        force=bool(args.force))
    if action == "move":
        return migrate(args.dir, args.to, apply=bool(args.apply))
    print(f"[workflow] неизвестное действие {action!r}", file=sys.stderr, flush=True)
    return 2


__all__ = ["HELP", "NAME", "add_args", "check_file", "new_file", "run", "show_file"]
