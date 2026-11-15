"""`python -m qwenstyle check-parity` — паритет кадров новым каркасом против эталонов.

Зачем: переезд с `run_style.py`/`tools/batch_seeds.py` удаляет монолиты, поэтому единственная
надёжная проверка «ничего не сломалось» — побайтовое совпадение кадров. Эталоны сняты **старым**
кодом до удаления (`tools/parity_refs.py --write`) и лежат в `tests/reference.json`; здесь те же
сценарии воспроизводятся новым каркасом (`pipeline.run`/`run_batch`) и сравниваются по `sha256`.

Правила сравнения (зафиксированы в самом `tests/reference.json`):

* сравнивается **sha256 файла**, а не имя: старый код тегирует дробный фактор как `-x4.0`, новый
  каркас — `-x4`;
* у записи `kind="batch"` сверяются только `s{N}-gen.png` из `files`: финальный `s{N}.png`
  зависит от способа расчёта ширины (старый `--out-res 1024` даёт 1376x768, новый считает высоту
  768p по готовому кадру -> 1382x768);
* кэш эмбеддингов обязан уже существовать: промах означает 55–255 с CPU-энкода, поэтому
  `check-parity` падает с кодом 2 и просит сначала сделать одиночный `run`, а не молча платит;
* прогоны идут с `cache_supersede=none` (`config_from_params`): регресс-тест — не обслуживание
  кэша, он не вытесняет (и не удаляет) ни одной чужой записи;
* у записи может быть `"requires": [...]` — список репозиториев Хаба и/или путей, без которых эталон
  физически не воспроизвести (например, веса FLUX.2). Такую запись `check-parity` **пропускает**:
  печатает `[skip] <entry>: <причина>`, считает её отдельно от PASS/FAIL и не роняет прогон
  (`missing_requirements`). Пропуск никогда не выдаётся за PASS: причина печатается, `skipped N`
  выводится в итоговой строке.

Запуск:
    .venv/Scripts/python.exe -m qwenstyle check-parity
    .venv/Scripts/python.exe -m qwenstyle check-parity --only base-640-8
    .venv/Scripts/python.exe -m qwenstyle check-parity --entry base-640-8   # алиас --only
    .venv/Scripts/python.exe -m qwenstyle check-parity --file tests/reference.json

`run --check-parity [ENTRY|auto]` использует те же функции (`check_result`): после прогона
сверяет один кадр с эталоном и возвращает 0/1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from ..backends.base import REPO_ROOT
from ..config import ConfigError, RunConfig

NAME = "check-parity"
HELP = "воспроизвести эталоны tests/reference.json новым каркасом и сверить sha256 кадров"
DEFAULT_REFERENCE = "tests/reference.json"
#: Куда писать кадры проверки (вне `out/`-раскладки пользователя).
DEFAULT_OUT_ROOT = "out/_parity/qwenstyle"


# --------------------------------------------------------------------------- #
# Эталоны и снимки
# --------------------------------------------------------------------------- #

def sha256_file(path: str | Path) -> str:
    """sha256 файла — без torch и без чтения всего в память (кадры бывают по 6 МБ)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_reference(path: str | Path = DEFAULT_REFERENCE) -> dict[str, Any]:
    """`tests/reference.json` целиком (записи в `entries`)."""
    raw = Path(path)
    if not raw.is_file():
        raise ConfigError(
            f"нет файла эталонов {raw}: он снимается старым кодом "
            f"(`tools/parity_refs.py --write`) до удаления монолитов"
        )
    try:
        data = json.loads(raw.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ConfigError(f"{raw}: некорректный JSON ({exc})") from exc
    if not isinstance(data, dict) or not isinstance(data.get("entries"), Mapping):
        raise ConfigError(f"{raw}: ожидался объект с секцией entries")
    return data


def expected_shas(entry: Mapping[str, Any]) -> dict[str, str]:
    """Ожидаемые хеши записи: `{'': sha}` для run, `{s0-gen.png: sha, …}` для batch."""
    files = entry.get("files")
    if isinstance(files, Mapping) and files:
        return {str(name): str(meta.get("sha256", "")) for name, meta in files.items()}
    return {"": str(entry.get("sha256", ""))}


# --------------------------------------------------------------------------- #
# `requires`: чего не хватает, чтобы воспроизвести запись
# --------------------------------------------------------------------------- #

def _hub_dirs(hf_cache: Path | None, root: Path) -> list[Path]:
    """Каталоги hub-кэша: явный `hf_cache` -> HF_HUB_CACHE -> HF_HOME/hub -> <root>/hf-cache/hub.

    Тот же порядок, что у `cmd/models.py::cache_dirs`/`backends/*._hf_cache_dirs` (переиспользуем
    правило, но без импорта `cmd.models`: он тянет реестр бекендов).
    """
    if hf_cache is not None:
        return [Path(hf_cache).expanduser() / "hub"]
    candidates: list[Path] = []
    raw = os.environ.get("HF_HUB_CACHE")
    if raw:
        candidates.append(Path(raw).expanduser())          # это уже сам hub-каталог
    home = os.environ.get("HF_HOME")
    if home:
        candidates.append(Path(home).expanduser() / "hub")
    candidates.append(Path(root) / "hf-cache" / "hub")
    unique: list[Path] = []
    for path in candidates:
        if all(str(path) != str(other) for other in unique):
            unique.append(path)
    return unique


def _candidate(text: str, root: Path) -> Path:
    """Путь из `requires` -> абсолютный: относительные считаются от корня репозитория."""
    path = Path(text).expanduser()
    return path if path.is_absolute() else Path(root) / path


def _looks_like_path(text: str, root: Path) -> bool:
    """Явный путь или имя репозитория Хаба.

    Путь — если начинается с `/`, `.` или `~`, содержит обратный слэш (Windows-форма) либо
    существует на диске (в том числе относительно корня репозитория, как
    `cache/fp8_transformer_flux2`). Иначе — id репозитория Хаба.
    """
    if text.startswith(("/", "\\", ".", "~")) or "\\" in text:
        return True
    return _candidate(text, root).exists()


def _repo_present(repo: str, hub_dirs: list[Path]) -> bool:
    """Есть ли у репозитория непустой snapshot в каком-нибудь из hub-кэшей.

    Обход раскладки `models--org--name/snapshots/<commit>/` — без Hub API и без сети (R3).
    """
    folder = "models--" + repo.replace("/", "--")
    for hub_dir in hub_dirs:
        snapshots = Path(hub_dir) / folder / "snapshots"
        if not snapshots.is_dir():
            continue
        try:
            for item in snapshots.iterdir():
                if item.is_dir() and any(item.iterdir()):
                    return True
        except OSError:                                    # нечитаемый кэш — считаем «нет»
            continue
    return False


def missing_requirements(entry: Mapping[str, Any], *, hf_cache: Path | None = None,
                         root: Path | None = None) -> str | None:
    """Чего не хватает для воспроизведения записи эталона (`requires`), или `None`.

    `requires` — необязательный список строк: репозиториев Хаба (`Qwen/Qwen-Image-2.1`,
    ищется как `<hf_cache>/hub/models--org--name` с непустым `snapshots/`) и/или путей в
    репозитории (`cache/fp8_transformer_flux2` — проверяется существование). Отсутствие поля —
    пустой список, то есть `None`. Возвращается человекочитаемая причина с тем же текстом, который
    записан в `requires`, — её печатает `[skip]`.

    `hf_cache` по умолчанию — `HF_HUB_CACHE`/`HF_HOME` либо `hf-cache` в корне репозитория;
    `root` — корень репозитория (`REPO_ROOT`), от него считаются относительные пути.
    """
    requires = entry.get("requires") or ()
    if isinstance(requires, str):                          # одиночная строка — тоже валидно
        requires = [requires]
    if not isinstance(requires, (list, tuple)):
        raise ConfigError(f"requires должен быть списком строк, а не {type(requires).__name__}")
    base = Path(root) if root is not None else REPO_ROOT
    for raw in requires:
        text = str(raw).strip()
        if not text:
            continue
        if _looks_like_path(text, base):
            path = _candidate(text, base)
            if not path.exists():
                return f"нет {text}"
            continue
        hub_dirs = _hub_dirs(hf_cache, base)
        if not _repo_present(text, hub_dirs):
            folder = "models--" + text.replace("/", "--")
            where = next((Path(d) for d in hub_dirs if (Path(d) / folder).is_dir()), hub_dirs[0])
            return f"нет {text} (нет в hf-cache: {where})"
    return None


# --------------------------------------------------------------------------- #
# Параметры записи -> RunConfig
# --------------------------------------------------------------------------- #

def _preset(params: Mapping[str, Any]) -> str:
    """`model` из эталона -> имя пресета (base/preview-qwen/turbo/…).

    Эталоны сняты на Qwen-Image-2.1, а `preview` теперь — черновик FLUX.2 (у него `lora: null`),
    поэтому для Qwen-черновика используется `preview-qwen`.
    """
    model = str(params.get("model") or "base")
    if params.get("extra", {}).get("preview") and model == "base":
        return "preview-qwen"                # 480p + 8 шагов на base (lora сохраняется)
    return model


def config_from_params(params: Mapping[str, Any], *, out: str, out_dir: str | None = None) -> RunConfig:
    """Воспроизвести параметры эталона (`tests/reference.json`) как `RunConfig`.

    Маппинг старых флагов: `resolution`/`gen_res` -> `resolution` (не `size_px`: у эталона
    зафиксирован именно resolution профиля), `out_res` -> `upscale.target_height`,
    `upscale` -> `upscale.factor`, `lora_scale` -> `generation.lora.scale`.
    """
    overrides: dict[str, Any] = {}
    if params.get("input"):
        overrides["input"] = str(params["input"])
    if params.get("prompt_file"):
        overrides["prompt_file"] = str(params["prompt_file"])
    resolution = params.get("resolution", params.get("gen_res"))
    if resolution is not None:
        overrides["resolution"] = int(resolution)
    steps = params.get("steps")
    if steps is not None:
        overrides["generation.steps"] = int(steps)
    if params.get("seed") is not None:
        overrides["seed"] = int(params["seed"])
    if params.get("lora_scale") is not None:
        overrides["generation.lora.scale"] = float(params["lora_scale"])
    # Регресс-тест не обслуживает кэш: даже на промахе (теоретически возможном) его прогон
    # не должен вытеснять чужие записи той же картинки — паритет только читает кэш и пишет кадры.
    overrides["cache_supersede"] = "none"
    # Эталоны сняты с CPU-энкодом (`embeds_<md5>_<sha>_<W>x<H>.pt`, без `-dev-`): GPU даёт другие
    # округления и другой кадр (R1), поэтому паритет — про численность каркаса, а не про
    # умолчание устройства. Иначе после `encoder.device=auto` ключ стал бы `-dev-cuda` и каждая
    # запись падала бы с «нет кэша эмбеддингов» — сверять было бы нечего.
    overrides["encoder.device"] = "cpu"

    extra = params.get("extra") or {}
    if extra.get("preview"):
        overrides["preview"] = True
    if extra.get("upscale_tile") is not None:
        overrides["upscale.tile"] = int(extra["upscale_tile"])
    if extra.get("upscale_pad") is not None:
        overrides["upscale.pad"] = int(extra["upscale_pad"])
    for key in ("color_fix", "color_strength", "color_sigma"):
        if extra.get(key) is not None:
            overrides[f"upscale.{key}"] = extra[key]

    if params.get("upscale") is not None:
        overrides["upscale.factor"] = float(params["upscale"])
    if params.get("upscale_backend"):
        overrides["upscale.backend"] = str(params["upscale_backend"])
    if params.get("out_res") is not None and out_dir is not None:
        # старый --out-res 1024 (resolution профиля) -> высота кадра; ширина, как и раньше,
        # считается по пропорциям уже готового кадра (см. notes эталона: 1382 vs 1376)
        from .batch import out_res_height, _input_size

        overrides["upscale.target_height"] = out_res_height(
            int(params["out_res"]), _input_size(str(params.get("input") or "")))
    if out_dir is not None:
        overrides["batch"] = {
            "seeds": str(params.get("seeds") or "0-1"),
            "out_dir": str(out_dir),
        }
        overrides["batch.resume"] = False   # эталон всегда генерируется заново
        overrides["batch.keep_gen"] = True
    overrides["out"] = str(out)
    # Энкодер эталонов: все записи в `tests/reference.json` сняты со штатным Qwen3VL-8B
    # (`embeds_<md5>_<sha>_<W>x<H>.pt`, без `_te-`). После того как дефолтом в base.yaml стал
    # heretic (D7), паритет обязан пинить `encoder.name=base`, иначе conditioning другой, кадр
    # другой и каждая запись — FAIL. FLUX.2-записи не трогаем: там свой энкодер (`qwen3`).
    preset = _preset(params)
    if str(RunConfig.from_profile(preset).generation.backend) == "qwen_image21":
        overrides["encoder.name"] = "base"
    return RunConfig.from_profile(preset, **overrides)


# --------------------------------------------------------------------------- #
# Одна запись: прогон + сравнение
# --------------------------------------------------------------------------- #

def _norm(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def run_entry(name: str, entry: Mapping[str, Any], *,
              out_root: str = DEFAULT_OUT_ROOT) -> list[dict[str, Any]]:
    """Воспроизвести одну запись эталона и вернуть список проверок.

    Каждый элемент: `{"label", "expected", "actual", "path", "ok", "reason"}`. Генерация идёт
    через `pipeline.run`/`pipeline.run_batch` — тот же путь, что у обычного прогона.
    """
    params = dict(entry.get("params") or {})
    kind = str(entry.get("kind") or "run")
    out_path = f"{out_root}/{name}.png"
    out_dir = f"{out_root}/{name}" if kind == "batch" else None
    cfg = config_from_params(params, out=out_path, out_dir=out_dir)

    from .. import pipeline

    expected = expected_shas(entry)
    checks: list[dict[str, Any]] = []
    if kind == "batch":
        if cfg.batch is None:
            raise ConfigError(f"{name}: запись kind=batch, но в параметрах нет seeds")
        if pipeline.dry_run(cfg).get("need_encoder"):
            raise ConfigError(f"{name}: нет кэша эмбеддингов для этой пары — сначала один run")
        pipeline.run_batch(cfg)
        from ..io import batch_paths

        seeds = pipeline.parse_seeds(cfg.batch.seeds)
        for file_name, sha in expected.items():
            seed = int("".join(ch for ch in Path(file_name).stem if ch.isdigit()) or 0)
            gen_path, _up = batch_paths(cfg.batch.out_dir, seed)
            checks.append(_compare(file_name, sha, gen_path))
        if len(seeds) != len(expected):
            checks.append({"label": "seeds", "expected": len(expected), "actual": len(seeds),
                           "path": "", "ok": len(seeds) == len(expected),
                           "reason": "число seed'ов не совпало с эталоном"})
        return checks

    if pipeline.dry_run(cfg).get("need_encoder"):
        raise ConfigError(f"{name}: нет кэша эмбеддингов для этой пары — сначала один run")
    result = pipeline.run(cfg)
    actual_path = _norm(result.out_path) if result is not None else ""
    sha = expected.get("", "")
    checks.append(_compare(name, sha, actual_path))
    if result is not None and entry.get("image_size"):
        want = tuple(int(v) for v in entry["image_size"])
        same = tuple(result.out_size) == want
        checks.append({"label": f"{name} size", "expected": list(want), "actual": list(result.out_size),
                       "path": actual_path, "ok": same,
                       "reason": "" if same else "размер кадра отличается от эталона"})
    return checks


def _compare(label: str, expected: str, path: str) -> dict[str, Any]:
    """Сравнить sha256 файла с ожидаемым (нет файла/хеша -> FAIL с причиной)."""
    if not path or not Path(path).is_file():
        return {"label": label, "expected": expected, "actual": "", "path": path, "ok": False,
                "reason": "файла нет (прогон не выполнялся?)"}
    actual = sha256_file(path)
    return {"label": label, "expected": expected, "actual": actual, "path": path,
            "ok": actual == expected,
            "reason": "" if actual == expected else "sha256 не совпал"}


# --------------------------------------------------------------------------- #
# Сверка одного кадра (`run --check-parity`)
# --------------------------------------------------------------------------- #

def _match_entry(reference: Mapping[str, Any], cfg: RunConfig) -> tuple[str, Mapping[str, Any]] | None:
    """Подобрать запись эталона по параметрам прогона (model/steps/seed/resolution/upscale)."""
    from .run import model_label

    label = model_label(cfg)
    for name, entry in (reference.get("entries") or {}).items():
        if str(entry.get("kind") or "run") != "run":
            continue
        params = entry.get("params") or {}
        if str(params.get("model") or "base") != label:
            continue
        if params.get("seed") is not None and int(params["seed"]) != int(cfg.seed):
            continue
        if params.get("steps") is not None and int(params["steps"]) != int(cfg.steps):
            continue
        if params.get("resolution") is not None and cfg.resolution is not None \
                and int(params["resolution"]) != int(cfg.resolution):
            continue
        if float(params.get("upscale") or 0.0) != float(cfg.upscale.factor or 0.0):
            continue
        if float(cfg.upscale.factor or 0.0) > 0 and str(params.get("upscale_backend") or "esrgan") \
                != str(cfg.upscale.backend):
            continue
        return str(name), entry
    return None


def check_result(cfg: RunConfig, result: Any, *, entry: str = "auto",
                 file: str | Path = DEFAULT_REFERENCE) -> int:
    """Сверить только что сохранённый кадр с эталоном; 0 — PASS, 1 — FAIL, 2 — нет записи."""
    reference = load_reference(file)
    entries = reference.get("entries") or {}
    name, record = (str(entry), entries.get(str(entry))) if entry not in ("auto", "1", "yes", "") \
        else (_match_entry(reference, cfg) or (None, None))
    if not record:
        available = ", ".join(sorted(entries)) or "—"
        print(f"[parity] нет подходящей записи в {file} (доступны: {available})", file=sys.stderr)
        return 2
    expected = expected_shas(record).get("", "")
    check = _compare(str(name), expected, _norm(result.out_path))
    _print_check(name, check)
    return 0 if check["ok"] else 1


# --------------------------------------------------------------------------- #
# Печать
# --------------------------------------------------------------------------- #

def _print_check(name: str, check: Mapping[str, Any]) -> None:
    status = "PASS" if check["ok"] else "FAIL"
    print(f"[{status}] {name}: {check['label']}", flush=True)
    print(f"        path     {check['path'] or '—'}", flush=True)
    print(f"        expected {str(check['expected'])[:32] or '—'}", flush=True)
    print(f"        actual   {str(check['actual'])[:32] or '—'}"
          + (f"   ({check['reason']})" if check.get("reason") else ""), flush=True)


def _run_all(reference: Mapping[str, Any], names: list[str], out_root: str) -> int:
    """Прогнать записи; `requires` которых не выполнены, печатаются как `[skip]` и не считаются FAIL."""
    failures = 0
    total = 0
    skipped = 0
    for name in names:
        entry = reference["entries"][name]
        print(f"\n[entry] {name} (kind={entry.get('kind') or 'run'})", flush=True)
        reason = missing_requirements(entry)
        if reason:
            print(f"[skip] {name}: {reason}", flush=True)
            skipped += 1
            continue
        try:
            checks = run_entry(name, entry, out_root=out_root)
        except ConfigError as exc:
            print(f"[FAIL] {name}: {exc}", flush=True)
            failures += 1
            total += 1
            continue
        for check in checks:
            total += 1
            if not check["ok"]:
                failures += 1
            _print_check(name, check)
    summary = (f"\n[check-parity] {total - failures}/{total} проверок PASS, FAIL {failures}"
               f", skipped {skipped}")
    print(summary, flush=True)
    return 0 if failures == 0 else 1


# --------------------------------------------------------------------------- #
# CLI-контракт подкоманды
# --------------------------------------------------------------------------- #

def _cmd_parity(args: argparse.Namespace) -> int:
    try:
        reference = load_reference(args.file)
    except ConfigError as exc:
        print(f"[check-parity] {exc}", file=sys.stderr, flush=True)
        return 2
    entries = reference.get("entries") or {}
    names = list(entries)
    if args.only:
        wanted = [item for raw in args.only for item in str(raw).split(",") if item.strip()]
        missing = [name for name in wanted if name not in entries]
        if missing:
            print(f"[check-parity] нет записей {missing} в {args.file}; "
                  f"доступны: {', '.join(names) or '—'}", file=sys.stderr, flush=True)
            return 2
        names = [name for name in names if name in wanted]
    if not names:
        print(f"[check-parity] в {args.file} нет записей", file=sys.stderr, flush=True)
        return 2
    print(f"[check-parity] эталоны {args.file} (сняты старым кодом), кадры -> {args.out_root}",
          flush=True)
    print("[check-parity] вытеснение кэша отключено на время паритета "
          "(cache_supersede=none): ни одна запись в cache/ не удаляется", flush=True)
    return _run_all(reference, names, args.out_root)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--file", default=DEFAULT_REFERENCE, metavar="PATH",
                        help=f"файл эталонов (по умолчанию {DEFAULT_REFERENCE})")
    parser.add_argument("--only", "--entry", dest="only", action="append", default=[],
                        metavar="NAME[,NAME]",
                        help="проверить только эти записи (можно повторять; --entry — алиас)")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, metavar="DIR",
                        help=f"куда писать кадры проверки (по умолчанию {DEFAULT_OUT_ROOT})")


def run(args: argparse.Namespace) -> int:
    return _cmd_parity(args)


__all__ = ["DEFAULT_OUT_ROOT", "DEFAULT_REFERENCE", "HELP", "NAME", "add_args",
           "check_result", "config_from_params", "expected_shas", "load_reference",
           "missing_requirements", "run", "run_entry", "sha256_file"]
