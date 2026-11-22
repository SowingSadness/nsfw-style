"""`Workflow` — файл воркфлоу как самодостаточное описание пайплайна (план `workflow-layout`).

Файл `workflow/<проект>/workflow.yaml` (или его вариант `tokyo-anime.yaml`) описывает **всё**:

```yaml
version: 1
cubes:
  - id: source.frame
    params: {width: 2912, height: 2176, ref_crop_side: 768}
  - id: source.txtFileImage
    params: {prompt: prompt-tokyo-anime.txt, Image1: refs-style/nita-style.jpg}
  # …
run:
  seeds: [142857]
  out: out
```

Правила (намеренно простые — никакого наследования):

* **ссылок на шаблоны нет**: `template:`/`extends:` — ошибка с подсказкой «шаблоны копируют».
  Шаблон — такой же файл этого формата, из него делают копию (`workflow new --from NAME`);
* `version` обязателен (сейчас `1`), несовпадение — явная ошибка, а не молчаливая догадка;
* `cubes` — упорядоченный список кубиков, порядок = порядок выполнения (проверяет компоновщик);
* `run` — параметры запуска (`RunSpec`), необязателен: без него один случайный сид;
* неизвестные верхнеуровневые ключи — ошибка (опечатка `cube:` вместо `cubes:` не должна
  выглядеть как «пустой пайплайн»).

Пути внутри `params` (`prompt`, `Image1…`) разрешаются **относительно каталога файла воркфлоу** —
это делает `source.txtFileImage` (он получает `base_dir` от вызывающего).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

#: Импорт пакета кубиков регистрирует все стадии в реестре (без него `get_cube` пуст).
from . import cubes as _cubes  # noqa: F401  — регистрация кубиков
from .compose import CubeBinding, PipelineSpec, compose
from .cube import get_cube
from .runspec import RunSpec

__all__ = ["VERSION", "Workflow", "load", "load_template", "parse"]

#: Версия формата файла (`workflow.yaml` и шаблона).
VERSION = 1
#: Допустимые верхнеуровневые ключи.
KEYS = ("version", "name", "cubes", "run")
#: Ключи, которые выглядят как наследование: их быть не должно.
FORBIDDEN = {
    "template": "шаблоны не подключаются ссылкой: сделайте копию (workflow new --from <шаблон>)",
    "extends": "наследования нет: файл воркфлоу самодостаточен (копируйте шаблон и правьте)",
    "preset": "пресеты ушли в шаблоны: скопируйте шаблон в файл воркфлоу",
}
#: Устаревшие «плоские» секции `RunConfig`: подсказка про кубики вместо молчаливого игнора.
LEGACY = {
    "generation": "transformer.qwen_image21 / strategy.*",
    "encoder": "encoder.qwen3vl",
    "upscale": "post.upscale.* / post.colorfix",
    "size_px": "source.frame.height",
    "aspect": "source.frame.width/height",
}


@dataclass(frozen=True)
class Workflow:
    """Разобранный файл воркфлоу: композиция + параметры запуска + путь к файлу."""

    pipeline: PipelineSpec
    run: RunSpec
    path: Path | None = None
    #: Сырой текст кубиков из файла (нужен `workflow new` и отладке).
    raw_cubes: tuple[Mapping[str, Any], ...] = ()

    @property
    def name(self) -> str:
        return self.pipeline.name

    @property
    def base_dir(self) -> Path:
        """Каталог, относительно которого разрешаются пути в параметрах."""
        return self.path.parent if self.path is not None else Path(".")

    @property
    def profile(self) -> str | None:
        return self.pipeline.profile

    # -- проверка ----------------------------------------------------------- #

    def composition(self):
        """Проверить композицию компоновщиком (без весов, без CUDA)."""
        return compose(self.pipeline)

    def apply_sets(self, paths: list[str] | tuple[str, ...]) -> "Workflow":
        """Применить `--set`-пути (новые или устаревшие написания) к композиции и запуску.

        Маршрутизация — по `ParameterSpec.path` кубиков: `--set encoder.qwen3vl.device=cpu`
        (новое), `--set encoder.device=cpu` (старое), `--set run.seeds=1-3` (запуск).
        """
        if not paths:
            return self
        cubes = [dict(binding.params) for binding in self.pipeline.cubes]
        ids = [binding.cube_id for binding in self.pipeline.cubes]
        run_overrides: dict[str, Any] = {}
        unknown: list[str] = []
        for raw in paths:
            key, sep, value = str(raw).partition("=")
            if not sep:
                raise ValueError(f"--set {raw!r}: нужен вид key=value")
            key = key.strip()
            value = value.strip()
            if key.startswith("run."):
                run_overrides[key] = value
                continue
            target = _resolve_path(key, ids)
            if target is None:
                unknown.append(key)
                continue
            index, name, spec = target
            cubes[index][name] = _coerce(value, spec.type)
        if unknown:
            available = sorted({f"{cube_id}.{param.name}" for cube_id in ids
                                for param in get_cube(cube_id).params})
            raise ValueError(f"--set: неизвестные ключи {unknown}; допустимые пути: "
                             f"{', '.join(available)}")
        pipeline = PipelineSpec(name=self.pipeline.name,
                                cubes=tuple(CubeBinding(cube_id, params)
                                            for cube_id, params in zip(ids, cubes)),
                                inputs=self.pipeline.inputs, profile=self.pipeline.profile)
        return Workflow(pipeline=pipeline, run=self.run.merged(**run_overrides),
                        path=self.path, raw_cubes=self.raw_cubes)

    # -- переходный режим (до плана `cutover`) ------------------------------ #

    def config_mapping(self) -> dict[str, Any]:
        """Обратная проекция: композиция -> словарь под `RunConfig.from_mapping`.

        Нужна, пока прогон читает `RunConfig`: файл воркфлоу остаётся единственным источником,
        а рантайм получает привычную форму. Уйдёт вместе с `RunConfig` в плане `cutover`.
        """
        from .composition import config_mapping_of

        return config_mapping_of(self.pipeline, self.run)


def _coerce(value: str, kind: str) -> Any:
    """Привести строку `--set` к типу параметра (`int`/`float`/`bool`/`enum`/`str`)."""
    text = value.strip()
    if kind == "int":
        return int(text)
    if kind == "float":
        return float(text)
    if kind == "bool":
        return text.lower() in ("true", "1", "yes", "on")
    return text                                    # enum и str остаются строками


def _resolve_path(path: str, ids: list[str]):
    """Найти `(индекс кубика, имя параметра, спека)` по пути `--set`.

    Принимаются обе формы: **квалифицированная** (`transformer.qwen_image21.steps`,
    `decoder.vae.qwen21.tile_px`) и **скоуп-форма прежнего конфига** (`generation.steps`,
    `upscale.tile`) — плюс все устаревшие написания из `preset_paths.LEGACY_PATHS`.
    """
    from .preset_paths import LEGACY_PATHS

    candidates = {path, LEGACY_PATHS.get(path, path)}
    for index, cube_id in enumerate(ids):
        for param in get_cube(cube_id).params:
            forms = {param.path, f"{cube_id}.{param.name}"}
            if forms & candidates:
                return index, param.name, param
    return None


def parse(data: Mapping[str, Any], *, path: Path | None = None,
          name: str | None = None) -> Workflow:
    """Разобрать уже прочитанный YAML воркфлоу (используется `load` и `workflow new`)."""
    if not isinstance(data, Mapping):
        raise ValueError(f"{_where(path)}: ожидался словарь, получено {type(data).__name__}")
    for key, hint in FORBIDDEN.items():
        if key in data:
            raise ValueError(f"{_where(path)}: ключ {key!r} не поддерживается — {hint}")
    for key, hint in LEGACY.items():
        if key in data:
            raise ValueError(f"{_where(path)}: секция {key!r} устарела; используйте кубики: {hint}")
    unknown = set(data) - set(KEYS)
    if unknown:
        raise ValueError(f"{_where(path)}: неизвестные ключи {sorted(unknown)}; допустимые: "
                         f"{', '.join(KEYS)}")
    version = data.get("version")
    if version != VERSION:
        raise ValueError(f"{_where(path)}: version={version!r}, поддерживается {VERSION}")
    cubes_raw = data.get("cubes")
    if not isinstance(cubes_raw, list) or not cubes_raw:
        raise ValueError(f"{_where(path)}: `cubes` обязан быть непустым списком")
    bindings: list[CubeBinding] = []
    seen: set[str] = set()
    for index, entry in enumerate(cubes_raw):
        if not isinstance(entry, Mapping) or "id" not in entry:
            raise ValueError(f"{_where(path)}: cubes[{index}] обязан быть объектом с `id`")
        extra = set(entry) - {"id", "params"}
        if extra:
            raise ValueError(f"{_where(path)}: cubes[{index}] содержит неизвестные ключи "
                             f"{sorted(extra)}")
        cube_id = str(entry["id"])
        try:
            get_cube(cube_id)
        except LookupError as exc:
            raise ValueError(f"{_where(path)}: {exc}") from None
        if cube_id in seen:
            raise ValueError(f"{_where(path)}: кубик {cube_id!r} указан дважды "
                             f"(порядок и один экземпляр на стадию)")
        seen.add(cube_id)
        params = entry.get("params") or {}
        if not isinstance(params, Mapping):
            raise ValueError(f"{_where(path)}: cubes[{index}].params обязан быть объектом")
        bindings.append(CubeBinding(cube_id, dict(params)))
    run = RunSpec.from_mapping(data.get("run") or {})
    pipeline_name = str(data.get("name") or name or (path.stem if path else "workflow"))
    inputs = _inputs_of(bindings)
    pipeline = PipelineSpec(name=pipeline_name, cubes=tuple(bindings), inputs=inputs)
    return Workflow(pipeline=pipeline, run=run, path=path,
                    raw_cubes=tuple(dict(entry) for entry in cubes_raw))


def _inputs_of(bindings: list[CubeBinding]) -> frozenset[str]:
    """Внешние входы композиции: порты, которые кто-то потребляет, но никто не производит.

    У полного воркфлоу (`source.frame` + `source.txtFileImage` + …) внешних входов нет — всё
    приходит из файлов. Если файл начинается, например, с `encoder.qwen3vl`, внешними окажутся
    `prompt`+`refs`, и компоновщик об этом скажет.
    """
    produced = {port for binding in bindings for port in get_cube(binding.cube_id).outputs}
    needed = {port for binding in bindings
              for port in (get_cube(binding.cube_id).consumes | get_cube(binding.cube_id).reads)}
    return frozenset(needed - produced)


def _where(path: Path | None) -> str:
    return path.as_posix() if path is not None else "workflow"


def load(path: str | Path, *, name: str | None = None) -> Workflow:
    """Прочитать файл воркфлоу (`yaml`/`json`)."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"нет файла воркфлоу {target.as_posix()}")
    text = target.read_text(encoding="utf-8")
    if target.suffix.lower() == ".json":
        import json

        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    return parse(data, path=target, name=name)


def load_template(name: str, *, root: str | Path | None = None) -> Workflow:
    """Прочитать шаблон по имени: `templates/<модель>/<имя>.yaml` (имя без пути)."""
    base = Path(root) if root is not None else Path(__file__).resolve().parent / "templates"
    stem = str(name)
    if "/" in stem:
        candidate = base / (stem if stem.endswith((".yaml", ".yml")) else f"{stem}.yaml")
        if candidate.is_file():
            return load(candidate)
        raise FileNotFoundError(f"нет шаблона {candidate.as_posix()}")
    matches = sorted(base.glob(f"*/{stem}.yaml"))
    if not matches:
        raise FileNotFoundError(f"нет шаблона {stem!r} в {base.as_posix()}/<модель>/ "
                                f"(список: {', '.join(p.stem for p in base.glob('*/*.yaml'))})")
    if len(matches) > 1:
        raise ValueError(f"шаблон {stem!r} есть у нескольких моделей: "
                         f"{', '.join(p.parent.name for p in matches)}; укажите «модель/имя»")
    return load(matches[0])
