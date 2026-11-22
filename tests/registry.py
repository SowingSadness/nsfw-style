"""Карта «что изменил → что проверять» (план `cube-architecture`, шаг 9).

Зачем: до шага 9 все проверки были e2e (CUDA + веса + кэш, минуты на файл), поэтому правка одной
строки в кубике гоняла полнабора или не гоняла ничего. Здесь — данные, а не код: по каждому правилу
видно, какие тесты отвечают за какой исходник, и `tests/run.py --changed <path>` берёт только их.

Правила (`Rule`):

* `paths` — posix-шаблоны от корня репозитория (`fnmatch`: `qwenstyle/cubes/*.py`);
* `tests` — конкретные файлы от `tests/` (`unit/test_strategies.py`) — то, что реально ловит регресс;
* `tags` — какие категории добавить целиком (`unit`, `contract`, `cube`, `e2e`). Нужно там, где
  правка меняет общий контракт и точечным файлом не ограничивается;
* `why` — одна строка для человека: почему именно эти тесты.

Приоритет: если путь попал хотя бы в одно правило, категории из `tags` **добавляются** к его
`tests`. Если не попал ни в одно — `tests_for` возвращает `wide=True` и прогон `unit + contract`
(дешёвая страховка вместо тишины).
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Rule", "RULES", "TAGS", "TESTS_DIR", "REPO_ROOT", "describe", "tests_for", "tests_in_tag"]

#: Корень репозитория и каталог тестов.
TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent

#: Категории тестов и их смысл (порядок — от дешёвых к дорогим).
TAGS: dict[str, str] = {
    "unit": "без torch и без CUDA, <2 с на файл: логика, ключи кэша, парсинг, геометрия",
    "contract": "декларация кубика == реализация (порты, параметры, self_test), без весов",
    "cube": "кубик на заглушках портов: нужен torch-CPU, веса и CUDA не нужны",
    "e2e": "реальный прогон: CUDA + веса (без CUDA печатают SKIP и не падают)",
}


@dataclass(frozen=True)
class Rule:
    """Одно правило карты: шаблоны исходников → тесты и категории."""

    paths: tuple[str, ...]
    tests: tuple[str, ...] = ()
    tags: frozenset[str] = field(default_factory=frozenset)
    why: str = ""


#: Сами правила. Порядок значения не имеет: совпадения объединяются.
RULES: tuple[Rule, ...] = (
    # -- контракт и композитор: общий механизм, от него зависят все кубики --------- #
    Rule(
        paths=("qwenstyle/cube.py", "qwenstyle/ports.py", "qwenstyle/compose.py"),
        tests=("unit/test_cube_contract.py", "cube/test_full_chain.py"),
        tags=frozenset({"unit", "contract"}),
        why="порты/реестр/композитор — общий контракт всех кубиков и полная цепочка",
    ),
    # -- кубики: у каждого свой точечный набор ------------------------------- #
    Rule(
        paths=("qwenstyle/cubes/decoder_vae.py", "qwenstyle/cubes/postprocess_image.py"),
        tests=("unit/test_decoder_vae.py", "cube/test_postprocess_image.py"),
        tags=frozenset({"contract"}),
        why="тайлинг/слайсинг декодера, класс Qwen21Decoder, frames -> PIL в постпроцессе",
    ),
    Rule(
        paths=("qwenstyle/cubes/encoder_qwen3vl.py", "qwenstyle/backends/qwen_image21.py"),
        tests=("cube/test_encoder_cube.py",),
        tags=frozenset({"contract"}),
        why="ключ кэша эмбеддингов обязан совпадать с RunConfig.embeds_key",
    ),
    Rule(
        paths=("qwenstyle/cubes/strategy_precision.py", "qwenstyle/cubes/strategy_kv.py",
               "qwenstyle/cubes/strategy_window_attention.py", "qwenstyle/backends/qwen_kv.py",
               "qwenstyle/backends/qwen_sliding.py"),
        tests=("unit/test_strategies.py",),
        tags=frozenset({"contract"}),
        why="стратегии (fp8/kv/window): домены режимов и геометрия окна",
    ),
    Rule(
        paths=("qwenstyle/cubes/rewrite_pe.py",),
        tests=("unit/test_rewrite_pe.py",),
        tags=frozenset({"contract"}),
        why="PE на torch: парсинг ответа, система-промт, ключ кэша ответов",
    ),
    Rule(
        paths=("qwenstyle/cubes/rewrite_pe_llamacpp.py", "qwenstyle/quant/adapters/gguf.py"),
        tests=("unit/test_gguf_adapter.py",),
        tags=frozenset({"contract"}),
        why="PE на llama.cpp: аргументы сервера, data-URL картинок, ошибки бюджета",
    ),
    # -- стадии шага 6: источник, трансформер, постобработка ------------------ #
    Rule(
        paths=("qwenstyle/cubes/source_frame.py", "qwenstyle/cubes/source_txtimage.py",
               "qwenstyle/io.py"),
        tests=("cube/test_source_cubes.py",),
        tags=frozenset({"contract"}),
        why="источники входов: промт+Image1…N, геометрия кадра (frame), ключи md5/sha",
    ),
    Rule(
        paths=("qwenstyle/cubes/transformer_qwen21.py",),
        tags=frozenset({"contract"}),
        why="трансформер: сверка параметров кубика с бекендом (verify)",
    ),
    Rule(
        paths=("qwenstyle/cubes/post_esrgan.py", "qwenstyle/cubes/post_colorfix.py",
               "qwenstyle/cubes/post_seedvr2.py", "qwenstyle/colorfix.py"),
        tests=("cube/test_post.py",),
        tags=frozenset({"contract"}),
        why="постобработка: множитель/пропуск апскейла, цветокоррекция, целевой размер SeedVR2",
    ),
    # -- хранилище квантованных весов ---------------------------------------- #
    Rule(
        paths=("qwenstyle/quant/store.py", "qwenstyle/quant/reader.py", "qwenstyle/quant/__init__.py",
               "qwenstyle/precision.py"),
        tests=("cube/test_quant_store.py", "e2e/test_precision.py"),
        why="fp8-кэш: схема сайдкара, digest, шарды, де-квант",
    ),
    Rule(
        paths=("qwenstyle/quant/adapters/transformers.py",),
        tests=("cube/test_transformers_adapter.py",),
        why="fp8-патч Linear для LM (PE)",
    ),
    Rule(
        paths=("qwenstyle/quant/adapters/diffusers.py",),
        tests=("e2e/test_precision.py",),
        why="fp8-кэш диффузионного трансформера",
    ),
    # -- ядро пайплайна и CLI ------------------------------------------------- #
    Rule(
        paths=("qwenstyle/config.py",),
        tests=("e2e/test_parameters.py", "e2e/test_t2i_preflight.py"),
        why="RunConfig/--set/валидация параметров",
    ),
    Rule(
        paths=("qwenstyle/pipeline.py",),
        tests=("unit/test_run_uses_composition.py", "e2e/test_generation_logs.py",
               "e2e/test_gen_folder.py"),
        why="оркестратор: проверка композиции до load, батчи, имена кадров, лог/meta",
    ),
    Rule(
        paths=("qwenstyle/presets/*.yaml", "qwenstyle/profiles.py"),
        tests=("e2e/test_presets_vram.py", "e2e/test_profiles.py",
               "unit/test_composition_from_config.py"),
        why="пресеты и профили: бюджет VRAM, публичные имена, проекция в композицию",
    ),
    Rule(
        paths=("qwenstyle/composition.py", "qwenstyle/cmd/config.py"),
        tests=("unit/test_composition_from_config.py",),
        tags=frozenset({"contract"}),
        why="проекция RunConfig -> кубики и печать `config show --cubes`",
    ),
    Rule(
        paths=("qwenstyle/workflow.py", "qwenstyle/runspec.py", "qwenstyle/preset_paths.py"),
        tests=("unit/test_workflow.py",),
        tags=frozenset({"contract"}),
        why="файл воркфлоу: версия, запрет наследования, --set по путям кубиков, RunSpec",
    ),
    Rule(
        paths=("qwenstyle/cmd/workflow.py",),
        tests=("unit/test_workflow_cli.py",),
        tags=frozenset({"contract"}),
        why="CLI воркфлоу: check/show/new/move без весов",
    ),
    Rule(
        paths=("qwenstyle/templates/*/*.yaml", "qwenstyle/profiles.py"),
        tests=("unit/test_templates.py",),
        tags=frozenset({"contract"}),
        why="шаблоны без наследования: инварианты тарифа и совпадение значений с пресетами",
    ),
    Rule(
        paths=("qwenstyle/cmd/cubes.py", "qwenstyle/cmd/compose.py"),
        tests=("unit/test_cubes_cli.py",),
        tags=frozenset({"contract"}),
        why="API кубиков для UI: list/show/test/schema и проверка композиции из файла",
    ),
    Rule(
        paths=("qwenstyle/cmd/*.py", "qwenstyle/cli.py"),
        tests=("e2e/test_gen_project.py", "e2e/test_gen_seeds.py"),
        why="подкоманды: разбор флагов и раскладка сессий",
    ),
    Rule(
        paths=("qwenstyle/cleanup.py", "qwenstyle/cmd/cache.py", "qwenstyle/cache.py"),
        tests=("e2e/test_cleanup.py", "e2e/test_cache_project.py",
               "e2e/test_cache_variants.py"),
        why="политика вытеснения кэша и `cache`",
    ),
    Rule(
        paths=("qwenstyle/streaming.py", "qwenstyle/shards.py"),
        tests=("e2e/test_streaming.py", "e2e/test_streaming_prefetch.py",
               "e2e/test_shard_reader.py", "e2e/test_pinned_fallback.py"),
        why="диск-стриминг энкодера и чтение шардов",
    ),
    Rule(
        paths=("qwenstyle/backends/minimax_h3_image.py", "qwenstyle/presets/h3-image.yaml",
               "qwenstyle/presets/max-h3.yaml"),
        tests=("e2e/test_h3_image_caps.py", "e2e/test_h3_encoder.py"),
        why="H3: caps/budget и фаза энкодера отдельным процессом",
    ),
    Rule(
        paths=("qwenstyle/backends/esrgan.py", "qwenstyle/backends/seedvr2.py"),
        tests=("e2e/test_upscale_skip.py",),
        why="постобработка: пропуск апскейла и имена файлов",
    ),
    Rule(
        paths=("qwenstyle/runlog.py",),
        tests=("e2e/test_generation_logs.py",),
        why="лог прогона и meta рядом с кадром",
    ),
)


def _norm(path: str) -> str:
    """Путь → posix-вид от корня репозитория (принимаем и абсолютный, и относительный)."""
    p = Path(path)
    if p.is_absolute():
        try:
            p = p.resolve().relative_to(REPO_ROOT)
        except ValueError:
            return p.as_posix()
    return p.as_posix().lstrip("./")


def tests_in_tag(tag: str) -> list[str]:
    """Все тесты категории (`tests/<tag>/test_*.py`), отсортированные по имени."""
    folder = TESTS_DIR / tag
    if not folder.is_dir():
        return []
    return [f"{tag}/{p.name}" for p in sorted(folder.glob("test_*.py"))]


def tests_for(changed: list[str]) -> tuple[list[str], bool]:
    """Тесты, которые надо прогнать для изменённых путей.

    Возвращает `(пути от tests/, wide)`; `wide=True` — ни одно правило не совпало, взяты
    `unit + contract` целиком (дешёвая страховка).
    """
    selected: set[str] = set()
    matched = False
    for raw in changed:
        path = _norm(raw)
        for rule in RULES:
            if any(fnmatch.fnmatchcase(path, pattern) for pattern in rule.paths):
                matched = True
                selected.update(rule.tests)
                for tag in rule.tags:
                    selected.update(tests_in_tag(tag))
    if not matched:
        for tag in ("unit", "contract"):
            selected.update(tests_in_tag(tag))
        return sorted(selected), True
    # Таг-набор мог добавить каталог, которого нет на диске — фильтруем по факту.
    return sorted(p for p in selected if (TESTS_DIR / p).is_file()), False


def describe() -> str:
    """Человекочитаемая карта: правило → тесты (для `run.py --map` и ревью)."""
    lines: list[str] = []
    for rule in RULES:
        tests = ", ".join(rule.tests) or ("—" if not rule.tags else "")
        tags = "+".join(sorted(rule.tags))
        lines.append(f"{' | '.join(rule.paths)}\n    -> {tests}{'  [' + tags + ']' if tags else ''}"
                     f"\n    {rule.why}")
    return "\n".join(lines)


if __name__ == "__main__":                                           # pragma: no cover
    print(describe())
