"""`RunSpec` — параметры **запуска**, а не пайплайна (план `workflow-layout`, шаг 2).

Разделение, которое вводится вместе с файлом воркфлоу:

* **что считать** — `PipelineSpec` (кубики + их параметры: модель, шаги, точность, канвас, апскейл);
* **как запускать** — этот класс: сколько сидов, куда писать, писать ли лог, где кэш.

Геометрии кадра (`width`/`height`/`ref_crop_side`) здесь **нет** — она живёт в кубике
`source.frame`, потому что меняет картинку, а не способ запуска.

    run = RunSpec.from_mapping({"seeds": [142857], "out": "out"})
    run.merged(seeds="1-3")          # правки по путям, как --set
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping

__all__ = ["RunSpec", "SUPERSEDE_POLICIES", "SEED_PATTERN"]

#: Политики вытеснения кэша эмбеддингов (`RunConfig.cache_supersede`, AGENTS.md R2).
SUPERSEDE_POLICIES = ("none", "prompt", "image", "pair")
#: Как задаются сиды: список чисел или строка вида `84`, `32-38`, `1,7,42`.
SEED_PATTERN = "число | диапазон N-M | список через запятую"


@dataclass(frozen=True)
class RunSpec:
    """Параметры запуска одного воркфлоу."""

    #: Сиды кадров (пусто = один случайный, как `gen` по умолчанию).
    seeds: tuple[int, ...] = ()
    #: Каталог сессий относительно каталога воркфлоу (`out`) или абсолютный.
    out: str = "out"
    #: Писать ли `<кадр>.log` рядом с кадром (meta.json пишется всегда).
    logs: bool = True
    #: Каталог кэша эмбеддингов; пусто = `<workflow>/cache/embeds`.
    cache_dir: str = ""
    #: Политика вытеснения записей кэша этой пары (промт, картинка).
    cache_supersede: str = "pair"
    #: Сколько кадров, если сиды не заданы явно (`count` и `seeds` взаимоисключающие).
    count: int | None = None
    #: Произвольные служебные поля (печатаются в meta, ядром не читаются).
    extra: Mapping[str, Any] = field(default_factory=dict)

    # -- разбор ------------------------------------------------------------- #

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RunSpec":
        """Собрать из секции `run:` файла воркфлоу."""
        values = dict(data or {})
        unknown = set(values) - {"seeds", "out", "logs", "cache_dir", "cache_supersede", "count"}
        if unknown:
            raise ValueError(f"run: неизвестные ключи {sorted(unknown)}; допустимые: "
                             f"seeds, count, out, logs, cache_dir, cache_supersede")
        seeds = _parse_seeds(values.get("seeds"))
        count = values.get("count")
        if seeds and count:
            raise ValueError("run: seeds и count взаимоисключающие")
        policy = str(values.get("cache_supersede") or "pair")
        if policy not in SUPERSEDE_POLICIES:
            raise ValueError(f"run.cache_supersede={policy!r}: допустимо {SUPERSEDE_POLICIES}")
        return cls(seeds=seeds, out=str(values.get("out") or "out"),
                   logs=bool(values.get("logs", True)),
                   cache_dir=str(values.get("cache_dir") or ""), cache_supersede=policy,
                   count=None if count is None else int(count))

    def merged(self, **overrides: Any) -> "RunSpec":
        """Точечные правки (`--set run.seeds=…`, `--seed 7`): значения уже приведены вызывающим."""
        if not overrides:
            return self
        data = self.to_mapping()
        for key, value in overrides.items():
            key = key.removeprefix("run.")
            if key == "seeds":
                data["seeds"] = value
            elif key in ("out", "cache_dir", "cache_supersede"):
                data[key] = str(value)
            elif key == "logs":
                data["logs"] = _as_bool(value)
            elif key == "count":
                data["count"] = None if value in (None, "") else int(value)
            else:
                raise ValueError(f"run: неизвестный ключ {key!r}")
        return RunSpec.from_mapping(data)

    def to_mapping(self) -> dict[str, Any]:
        """Обратно в словарь (для `--json`, логов и `merged`)."""
        return {"seeds": list(self.seeds), "out": self.out, "logs": self.logs,
                "cache_dir": self.cache_dir, "cache_supersede": self.cache_supersede,
                "count": self.count}

    # -- производные -------------------------------------------------------- #

    @property
    def seed(self) -> int:
        """Первый сид (одиночный прогон `run` берёт его)."""
        return int(self.seeds[0]) if self.seeds else 0

    def with_seed(self, seed: int) -> "RunSpec":
        """Копия с единственным сидом (для `run`, где кадр всегда один)."""
        return replace(self, seeds=(int(seed),), count=None)

    def seed_spec(self) -> str:
        """Сиды в CLI-форме (`84`, `32-38`, `1,7,42`) для передачи в `pipeline.parse_seeds`."""
        if not self.seeds:
            return ""
        numbers = sorted(int(seed) for seed in self.seeds)
        spans: list[str] = []
        start = prev = numbers[0]
        for number in numbers[1:]:
            if number == prev + 1:
                prev = number
                continue
            spans.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = number
        spans.append(str(start) if start == prev else f"{start}-{prev}")
        return ",".join(spans)


def _as_bool(value: Any) -> bool:
    """`"false"`/`"0"`/`"no"` из CLI — это `False`, а не непустая строка (иначе `--set run.logs=false`
    включал бы логи)."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("false", "0", "no", "off", ""):
        return False
    return True


def _parse_seeds(raw: Any) -> tuple[int, ...]:
    """`[142857]`, `142857`, `"1-3,7"` -> кортеж чисел (порядок и диапазоны как в CLI)."""
    if raw is None or raw == "":
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(int(item) for item in raw)
    text = str(raw).strip()
    if not text:
        return ()
    seeds: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part[1:]:
            head, tail = part.split("-", 1)
            start, stop = int(head), int(tail)
            if stop < start:
                raise ValueError(f"run.seeds: диапазон {part!r} задан наоборот")
            seeds.extend(range(start, stop + 1))
        else:
            seeds.append(int(part))
    return tuple(seeds)
