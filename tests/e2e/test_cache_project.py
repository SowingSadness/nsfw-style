"""Кэш эмбеддингов в проекте: `prompts/<project>/cache` вместо общего `cache/embeds`.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_cache_project.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем: `gen --project` кладёт записи в каталог проекта, чтобы политика `cache_supersede` и
`cache prune` одного проекта не вытесняли кэш другого. Ключ и имя файла от места не зависят,
поэтому перенос каталога не инвалидирует записи (R1: изменился путь, а не математика).

Кейсы:

1. **`RunConfig.cache_dir` -> `cfg_cache`:** относительный путь разрешается от корня репозитория,
   `prompts/<p>/cache` разворачивается в `.../cache/embeds` (как и легаси-корень `cache`),
   а `None` даёт глобальный `cache/embeds`.
2. **`project_cache_dir` / `repo_relative`:** путь проекта и стабильная posix-строка для логов.
3. **Сайдкар пишет сетку conditioning, а не кадр:** с `ref_px` они разные, а сверка
   `embeds_expected` идёт по сетке — иначе кэш не попадает никогда.
4. **`cache list|move` видят корень проекта:** `cache_root(--project)` и план переноса выбирают
   записи по `image_sha` референсов проекта, а без `--apply` ничего не двигают.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                   # тест не должен оставлять __pycache__ в пакете

from qwenstyle.backends.base import REPO_ROOT as PACKAGE_ROOT      # noqa: E402
from qwenstyle.cache import Blob, embeds_root, project_cache_dir, repo_relative   # noqa: E402
from qwenstyle.cmd import cache as cache_cmd                      # noqa: E402
from qwenstyle.config import EMBEDS_DIR, RunConfig                # noqa: E402
from qwenstyle.pipeline import Plan, _embeds_meta, blob_cache, cfg_cache   # noqa: E402


def case_cache_dir() -> None:
    """`cache_dir` в конфиге решает, где лежат записи; `None` — глобальный каталог."""
    cfg0 = RunConfig.from_profile("base")
    plain = cfg0
    assert cfg_cache(plain).root == blob_cache().root, "без cache_dir путь должен быть глобальным"
    assert cfg_cache(plain).root == (PACKAGE_ROOT / "cache" / "embeds").resolve(), \
        f"глобальный корень обязан быть cache/embeds, получено {cfg_cache(plain).root}"

    project = cfg0.merged(cache_dir="prompts/demo/cache")
    root = cfg_cache(project).root
    assert root == (PACKAGE_ROOT / "prompts" / "demo" / "cache" / "embeds").resolve(), \
        f"кэш проекта обязан быть .../prompts/demo/cache/embeds, получено {root}"

    absolute = RunConfig.from_profile("base").merged(
        cache_dir=str(PACKAGE_ROOT / "prompts" / "abs" / "cache"))
    assert cfg_cache(absolute).root == (PACKAGE_ROOT / "prompts" / "abs" / "cache" / "embeds")


def case_paths() -> None:
    """Хелперы пути: `project_cache_dir` и стабильная posix-строка."""
    assert project_cache_dir("prompts/demo") == Path("prompts/demo/cache")
    assert repo_relative("prompts/demo/cache") == "prompts/demo/cache"
    inside = PACKAGE_ROOT / "prompts" / "demo" / "cache"
    assert repo_relative(inside) == "prompts/demo/cache", repo_relative(inside)
    outside = Path(tempfile.gettempdir()) / "elsewhere"
    assert repo_relative(outside) == outside.as_posix()
    assert embeds_root("prompts/demo/cache") == Path("prompts/demo/cache/embeds")


def case_sidecar_grid() -> None:
    """Сайдкар хранит сетку conditioning (`cond`), а не кадр (`crop`): иначе промах навсегда."""
    cfg = RunConfig.from_profile("base").merged(cache_dir="prompts/demo/cache")
    plan = Plan(
        images=[], in_size=(1728, 1152),
        crop=(2912, 2176), cond=(1024, 768),        # ref_px: кадр больше сетки референсов
        resolution=887, prompt="p", prompt_md5="0" * 12, img_sha="1" * 12,
        blob=Blob(stage="embeds", key="x"), cache_path=Path("x.pt"), cached=None,
        cache_meta=None,
        need_encoder=True, factor=0.0, out_path="out.png",
    )
    meta = _embeds_meta(cfg, plan, {})
    assert meta["crop"] == "1024x768", f"crop обязан быть сеткой conditioning, получено {meta['crop']}"
    assert meta["canvas"] == "2912x2176", f"кадр обязан попасть в canvas, получено {meta.get('canvas')}"


def case_cli_root() -> None:
    """`cache list/move` берут корень проекта, а `move` без `--apply` ничего не двигает."""
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "demo"
        refs = base / "refs"
        refs.mkdir(parents=True)
        image = refs / "01.png"
        image.write_bytes(b"not-a-real-png-but-sha-is-what-matters")
        sha = hashlib.sha256(image.read_bytes()).hexdigest()[:12]

        args = argparse.Namespace(project=str(base), cache_dir=None)
        root = cache_cmd.cache_root(args)
        assert root == embeds_root(project_cache_dir(base)).resolve(), f"корень проекта: {root}"

        args_rel = argparse.Namespace(project=None, cache_dir="prompts/demo/cache")
        assert cache_cmd.cache_root(args_rel) == \
            embeds_root(PACKAGE_ROOT / "prompts" / "demo" / "cache"), "флаг --cache-dir главнее"

        # глобальный кэш: одна запись по референсу проекта, одна чужая
        name = f"embeds_{'a' * 12}_{sha}_te-heretic-dev-cuda_1024x768.pt"
        foreign = f"embeds_{'a' * 12}_{'f' * 12}_te-heretic-dev-cuda_1024x768.pt"
        for item in (name, foreign):
            (EMBEDS_DIR / item).write_bytes(b"\0" * 32)
            (EMBEDS_DIR / item).with_suffix(".json").write_text(json.dumps({"crop": "1024x768"}),
                                                               encoding="utf-8")
        try:
            cache_cmd.cmd_move(argparse.Namespace(project=str(base), apply=False))
            target = project_cache_dir(base) / "embeds"
            assert not target.exists() or not list(target.glob("embeds_*.pt")), \
                "без --apply переносить нечего"
            cache_cmd.cmd_move(argparse.Namespace(project=str(base), apply=True))
            moved = sorted(p.name for p in target.glob("embeds_*.pt"))
            assert moved == [name], f"перенестись должна только запись проекта, получено {moved}"
            assert not (EMBEDS_DIR / name).exists(), "источник обязан исчезнуть"
            assert (EMBEDS_DIR / foreign).exists(), "чужая запись обязана остаться"
        finally:
            for item in (name, foreign):
                for path in (EMBEDS_DIR / item, (EMBEDS_DIR / item).with_suffix(".json")):
                    if path.exists():
                        path.unlink()


CASES: list[tuple[str, Callable[[], None]]] = [
    ("RunConfig.cache_dir -> cfg_cache", case_cache_dir),
    ("project_cache_dir / repo_relative", case_paths),
    ("сайдкар по сетке conditioning", case_sidecar_grid),
    ("cache list|move и корень проекта", case_cli_root),
]


def main() -> int:
    failed = 0
    for name, fn in CASES:
        try:
            fn()
        except Exception as exc:                                     # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
            traceback.print_exc()
        else:
            print(f"PASS {name}")
    if failed:
        print(f"\n{failed} из {len(CASES)} кейсов провалено")
        return 1
    print("\nALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
