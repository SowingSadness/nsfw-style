"""Политика вытеснения кэша эмбеддингов: отложенная задача удаления.

Смысл: новая пара (промт, картинка) делает прошлые записи этой пары ненужными, но удалять их
в момент записи **нельзя** — генерация ещё не прошла, кадр не сохранён, а при падении мы потеряли
бы и старое, и нового не получили бы. Поэтому `pipeline._conditioning()` только регистрирует
`CacheWrite`, а `DeferredCleanup.flush()` исполняется в самом конце `run`/`run_batch` — после
генерации, апскейла и сохранения файла.

Политика — `RunConfig.cache_supersede` (доступна как `--set cache_supersede=…`):

    none    ничего (поведение до этого шага)
    prompt  записи с тем же sha, но другим md5  (сменили промт у той же картинки)
    image   записи с тем же md5, но другим sha  (сменили картинку у того же промта)
    pair    объединение prompt и image (умолчание)

Жёсткие правила (покрыты `tests/test_cleanup.py`):

  * запись текущего прогона (`CacheWrite.path`) и текущая пара (md5+sha) не удаляются никогда;
  * вариант записи — `(enc, dtype, device)`: прогон на `cuda` не вытесняет `cpu`-записи, `fp16`
    не вытесняет `bfloat16` (нужны для сравнения и отката);
  * `crop` (`--size`/`--resolution`/`--aspect`) в вытеснении **не участвует**: разные разрешения
    и аспекты одного промта остаются все — иначе каждое переключение стоило бы 55–400 с энкода;
  * `legacy`-записи (без хеша картинки) в живую политику не входят (только `cache prune --legacy`);
  * план пересчитывается в `flush()` по фактическому содержимому `cache/embeds/`, а не по снимку времён
    регистрации;
  * удаляет только `cache.delete_embeds` (её защиты пути/имени сохраняются), `.json` уезжает
    вместе с `.pt`;
  * `cache/models/` (fp8-квантование и веса) не вытесняется никогда и не сканируется вовсе;
  * `apply=False` (`--dry-run`) печатает план и ничего не трогает; при исключении/прерывании
    прогона задача не исполняется — `flush` вызывается только в самом конце.

Импортов `torch`/`PIL` здесь нет: модуль обязан работать без GPU, весов и CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from .cache import (
    CACHE_DIR,
    EmbedsEntry,
    delete_embeds,
    embeds_root,
    human_bytes,
    scan_embeds,
)
from .config import (
    CACHE_SUPERSEDE_POLICIES,
    DEFAULT_CACHE_SUPERSEDE,
    encoder_device_canon,
    encoder_dtype_canon,
)

#: Политика вытеснения. Значения — `CACHE_SUPERSEDE_POLICIES` из `config.py` (один источник).
SupersedePolicy = Literal["none", "prompt", "image", "pair"]

#: Вариант записи: разные энкодеры/dtype/device — разные величины (GPU != CPU по округлениям).
Variant = tuple[str, str, str]


def normalize_policy(policy: str | None) -> SupersedePolicy:
    """Строка политики -> одно из `CACHE_SUPERSEDE_POLICIES`; пусто -> умолчание.

    Неизвестное значение — `ValueError` со списком допустимых (в `RunConfig` это же значение
    ловится как `ConfigError` ещё при сборке конфига).
    """
    # `--set cache_supersede=none` приходит как None (`_coerce`): это явная политика `none`.
    raw = "none" if policy is None else str(policy).strip().lower()
    if not raw:
        raw = DEFAULT_CACHE_SUPERSEDE
    if raw not in CACHE_SUPERSEDE_POLICIES:
        raise ValueError(
            f"cache_supersede={policy!r}: допустимо {', '.join(CACHE_SUPERSEDE_POLICIES)}"
        )
    return raw                        # type: ignore[return-value]


@dataclass(frozen=True)
class CacheWrite:
    """Что только что записал этот прогон (одна запись `cache/embeds_*.pt`).

    `md5` — хеш промта, `sha` — хеш входной картинки, `crop` — `WxH` conditioning-сетки,
    `enc` — тег энкодера (`cache_tag`), `path` — путь записанного `.pt` (его нельзя удалять).
    `dtype`/`device` приводятся к канону конфига в `__post_init__`, чтобы сравнение с
    `EmbedsEntry` (тоже канон) не зависело от написания (`bf16`/`bfloat16`, `cuda:0`/`cuda`).
    """

    md5: str
    sha: str
    crop: str
    enc: str
    dtype: str = "bfloat16"
    device: str = "cpu"
    path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "enc", str(self.enc or "base").strip() or "base")
        object.__setattr__(self, "dtype", encoder_dtype_canon(self.dtype))
        object.__setattr__(self, "device", encoder_device_canon(self.device))
        if self.path is not None:
            object.__setattr__(self, "path", Path(self.path))

    @property
    def variant(self) -> Variant:
        """Группа варианта: `(enc, dtype, device)` — вытеснение живёт только внутри неё."""
        return (self.enc, self.dtype, self.device)

    @property
    def pair(self) -> tuple[str, str]:
        return (str(self.md5), str(self.sha))

    def __str__(self) -> str:                           # для логов
        where = f" -> {self.path.name}" if self.path is not None else ""
        return f"{self.md5}/{self.sha}{where}"


def _norm_path(path: Path | None) -> Path | None:
    """Путь записи в сравнимом виде (`resolve` без падения на несуществующем файле)."""
    if path is None:
        return None
    try:
        return Path(path).resolve()
    except OSError:                                     # pragma: no cover — Windows-специфика
        return Path(path).absolute()


def supersedes(write: CacheWrite, entry: EmbedsEntry, policy: SupersedePolicy) -> bool:
    """Делает ли `write` запись `entry` устаревшей по политике `policy`?

    Правила: текущая пара и текущий путь не вытесняются никогда; вариант
    `(enc, dtype, device)` обязан совпасть; `crop` не участвует; `legacy`-записи (без `sha`)
    в живую политику не входят.
    """
    if policy == "none":
        return False
    if entry.legacy or not entry.sha:
        return False
    if (entry.md5, entry.sha) == write.pair:            # текущая пара — не трогаем
        return False
    if write.path is not None and _norm_path(entry.path) == _norm_path(write.path):
        return False
    if (entry.enc, entry.dtype, entry.device) != write.variant:
        return False
    same_prompt = entry.sha == write.sha and entry.md5 != write.md5
    same_image = entry.md5 == write.md5 and entry.sha != write.sha
    if policy == "prompt":
        return same_prompt
    if policy == "image":
        return same_image
    return same_prompt or same_image                    # policy == "pair"


class DeferredCleanup:
    """Копит `CacheWrite`; план считает в момент вызова, по фактическому содержимому `cache/embeds/`.

    Регистрация — сразу после успешной записи `.pt` (`pipeline._conditioning`), исполнение —
    в самом конце прогона. Ни `register()`, ни `plan()` ничего не удаляют. Легаси-корень `cache`
    разрешается в `cache/embeds` (`cache.embeds_root`), поэтому старые вызывающие не меняются.
    """

    def __init__(self, root: Path | str | None = None) -> None:
        self.root = embeds_root(root) if root is not None else Path(CACHE_DIR)
        self._writes: list[CacheWrite] = []

    # -- состояние --------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._writes)

    @property
    def writes(self) -> list[CacheWrite]:
        return list(self._writes)

    @property
    def variants(self) -> list[Variant]:
        """Текущие варианты `(enc, dtype, device)` — их вытеснение не касается."""
        return sorted({w.variant for w in self._writes})

    def register(self, write: CacheWrite) -> None:
        """Запомнить запись текущего прогона. Ничего не удаляет и не пересчитывает."""
        self._writes.append(write)

    # -- политика ---------------------------------------------------------- #

    def plan(self, policy: SupersedePolicy | str, root: Path | str | None = None,
             ) -> list[EmbedsEntry]:
        """Что устарело **сейчас**: разбор `cache/embeds/` в момент вызова (не снимок на register)."""
        policy = normalize_policy(policy)
        if policy == "none" or not self._writes:
            return []
        entries = scan_embeds(Path(root) if root is not None else self.root)
        targets: dict[Path, EmbedsEntry] = {}
        for write in self._writes:
            for entry in entries:
                if supersedes(write, entry, policy):
                    targets[entry.path] = entry
        return sorted(targets.values(), key=lambda e: str(e.path))

    def flush(self, policy: SupersedePolicy | str, *, apply: bool = True,
              root: Path | str | None = None) -> tuple[int, int]:
        """Исполнить политику: `(удалено записей, освобождено байт)`.

        `apply=False` (`--dry-run`) печатает план и не трогает ни один файл. Удаление — только
        `cache.delete_embeds`: она сама проверяет, что путь лежит прямо в `cache/embeds/` и имя —
        `embeds_*.pt`, поэтому `cache/models/fp8_transformer*`, веса и чужие файлы не удаляются никогда.
        """
        policy = normalize_policy(policy)
        cleanup_root = embeds_root(root) if root is not None else self.root
        if policy == "none" or not self._writes:
            return (0, 0)

        targets = self.plan(policy, cleanup_root)
        kept = len(self.variants)
        if not targets:
            print(f"[cache] вытеснение ({policy}): удалять нечего; "
                  f"текущих вариантов сохранено {kept}", flush=True)
            return (0, 0)

        planned = sum(int(e.size) for e in targets)
        if not apply:
            for entry in targets:
                print(f"  [plan] удалить {entry.path.name} "
                      f"({human_bytes(entry.size)}, {entry.created})", flush=True)
            print(f"[cache] вытеснение ({policy}): план — {len(targets)} записей, "
                  f"{human_bytes(planned)}; текущих вариантов сохранено {kept}; "
                  f"ничего не удалено (dry-run)", flush=True)
            return (0, 0)

        print(f"[cache] вытеснение ({policy}): {len(targets)} записей "
              f"({human_bytes(planned)}), текущих вариантов сохранено {kept}", flush=True)
        deleted, freed = delete_embeds(targets, cleanup_root)
        print(f"[cache] вытеснение ({policy}): удалено {deleted} записей, "
              f"освобождено {human_bytes(freed)}; текущих вариантов сохранено {kept}",
              flush=True)
        return (deleted, freed)

    # -- диагностика ------------------------------------------------------- #

    def summary(self) -> str:
        """Одна строка для логов: сколько задач, пар и вариантов накоплено."""
        if not self._writes:
            return "DeferredCleanup: задач нет"
        pairs = sorted({w.pair for w in self._writes})
        variants = self.variants
        shown = ", ".join("/".join(v) for v in variants)
        return (f"DeferredCleanup: задач {len(self._writes)}, пар {len(pairs)}, "
                f"вариантов {len(variants)} ({shown})")

    def __str__(self) -> str:                           # для логов
        return self.summary()


def register_write(cleanup: DeferredCleanup | None, write: CacheWrite | None) -> None:
    """Регистрация через `None`-безопасную обёртку: `None` = политика выключена вызывающим."""
    if cleanup is not None and write is not None:
        cleanup.register(write)


def plan_names(entries: Iterable[EmbedsEntry]) -> list[str]:
    """Имена записей плана — для машинно-читаемого вывода `--dry-run`."""
    return [e.path.name for e in entries]


__all__ = [
    "CacheWrite",
    "DeferredCleanup",
    "SupersedePolicy",
    "Variant",
    "normalize_policy",
    "plan_names",
    "register_write",
    "supersedes",
]
