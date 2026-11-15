"""Эталонные sha256 кадров, снятые **старым** кодом (`run_style.py`, `tools/batch_seeds.py`).

ВАЖНО: повторная запись эталонов (`--write`) требует кода **до** рефакторинга — `run_style.py`
и `tools/batch_seeds.py` удалены на шаге 11 переезда, поэтому этот скрипт оставлен только как
документ того, как снимались хеши в `tests/reference.json`. Проверка паритета после переезда
живёт в пакете: `python -m qwenstyle check-parity` (`qwenstyle/cmd/parity.py`).

Зачем: шаг 11 переезда на пакет `qwenstyle` удаляет монолиты. Единственная надёжная проверка,
что рефакторинг ничего не сломал, — побайтовое совпадение кадров до/после. Поэтому эталоны
снимаются ДО удаления и лежат в `tests/reference.json`; шаг 11 прогоняет те же сценарии через
`python -m qwenstyle …` и сравнивает `sha256`.

Состав эталонов (AGENTS.md, инвариант 2 «детерминизм»):

| имя | что фиксируется |
|---|---|
| `base-640-8` | одиночный кадр, base + R80, `--preview` (640p, 8 шагов), seed 20261113 |
| `turbo-640` | одиночный кадр на turbo-профиле (свой transformer + E6 R128 LoRA) |
| `esrgan-x4` | тот же кадр + Real-ESRGAN anime6B x4 (проверяет порядок «release → апскейл») |
| `batch-2seeds` | `s0-gen.png` / `s1-gen.png` из батча: одна загрузка пайплайна, два seed'а |

У батча фиксируются **только** `s{N}-gen.png`: финальный `s{N}.png` зависит от способа расчёта
ширины (старый `--out-res` даёт 1376px, новый каркас считает высоту → 1382px), и сравнивать
финальные кадры бессмысленно. Чистая генерация от этого не зависит.

Кэш эмбеддингов (AGENTS.md, R1) обязан уже существовать: скрипт считает ключ
`md5(prompt_file.strip()) + sha256(input)[:12]` и ищет `cache/embeds_*.pt` **до** запуска
генерации. Иначе промах кэша стоил бы 55–255 с CPU-энкода Qwen3VL-8B; в этом случае скрипт
печатает, какие пары в кэше есть, и выходит с кодом 2, ничего не генерируя.

Режимы:

    # снять эталоны (по умолчанию пишет tests/reference.json)
    .venv/Scripts/python.exe tools/parity_refs.py --write

    # посмотреть, что уже записано
    .venv/Scripts/python.exe tools/parity_refs.py

    # повторный снимок: печатает «совпало / ИЗМЕНИЛОСЬ» по каждой записи (проверка детерминизма)
    .venv/Scripts/python.exe tools/parity_refs.py --write

Только stdlib + PIL: torch/diffusers не импортируются (роут preflight идёт по сайдкару `.json`
рядом с `embeds_*.pt`, а не через `calculate_dimensions`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
PY = ROOT / ".venv" / "Scripts" / "python.exe"

CACHE_DIR = ROOT / "cache"
DEFAULT_INPUT = "out/canyon-edit-640-lora1.0-s777.png"
DEFAULT_PROMPT = "promts/exp3.txt"
DEFAULT_SEED = 20261113
DEFAULT_RESOLUTION = 640
DEFAULT_OUT_ROOT = "out/_parity"
DEFAULT_REFERENCE = "tests/reference.json"

#: Строки, которых в выводе быть не должно: они означают, что платился CPU-энкод.
#: (обычная строка «    ok, 6s» после генерации — не маркер, поэтому ищем только специфичные.)
MISS_MARKERS = (
    "[4/5] кодирование промта на CPU",
    "эмбеддинги сохранены",
    "-> embeds_",                # «ok, 62s (1, 637, 4096) -> embeds_*.pt» после энкода
)

NOTES = (
    "Эталоны сняты СТАРЫМ кодом (run_style.py / tools/batch_seeds.py) до переезда на пакет "
    "qwenstyle; хеши валидны для стека из 'env'. Пара (input, prompt_file) выбрана закэшированной: "
    "cache/embeds_<md5>_<sha>_<W>x<H>.pt существовал до снимка, поэтому CPU-энкод не платился. "
    "Правила сравнения для шага 11: (1) сравнивать sha256 файла, а не имя — старый код тегирует "
    "дробный фактор как '-x4.0', новый каркас как '-x4'; (2) у записи 'batch-2seeds' сверять "
    "только '* -gen.png' (в files), финальный 's{N}.png' отличается шириной: старый --out-res 1024 "
    "даёт 1376x768, новый считает высоту 768p -> 1382x768; (3) batch_seeds берёт эмбеддинги по "
    "картинке+сетке (промт не входит в его поиск), поэтому эталон валиден только пока самый свежий "
    "кеш для этого входа — кеш именно этого промта (preflight это проверяет)."
)


# --------------------------------------------------------------------------- #
# Мелочи
# --------------------------------------------------------------------------- #

def sha256_file(path: str | Path, limit: int | None = None) -> str:
    """sha256 файла (или его первых `limit` байт) — БЕЗ импорта torch."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            if limit is not None and fh.tell() >= limit:
                break
    return digest.hexdigest()


def prompt_md5(text: str) -> str:
    """Первые 12 hex md5 промта — как в `run_style.py` (там md5 считается от `strip`-текста)."""
    return hashlib.md5(text.encode()).hexdigest()[:12]


def rel(path: str | Path) -> str:
    """Путь от корня репозитория с прямыми '/' (в JSON и логах читается одинаково)."""
    raw = Path(path)
    try:
        raw = raw.resolve().relative_to(ROOT)
    except (ValueError, OSError):
        pass
    return Path(raw).as_posix()


def read_sidecar(pt: Path) -> dict[str, Any] | None:
    side = pt.with_suffix(".json")
    if not side.is_file():
        return None
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def child_env() -> dict[str, str]:
    """Окружение дочерних прогонов: оффлайн, utf-8 в логах, HF_HOME в репозитории (AGENTS.md R3)."""
    env = dict(os.environ)
    env.update({
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
        "HF_HOME": str(ROOT / "hf-cache"),
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    })
    return env


def portable_cmd(cmd: list[str]) -> str:
    """Команда одной строкой с путём питона от корня репо (JSON читается на любой машине)."""
    prefix = str(PY)
    return " ".join([rel(PY) if part == prefix else part for part in cmd])


def run_stream(cmd: list[str]) -> tuple[int, list[str], float]:
    """Запустить команду, показать её вывод по строке и вернуть (код, строки, секунды)."""
    print("  $ " + " ".join(cmd), flush=True)
    started = time.time()
    lines: list[str] = []
    proc = subprocess.Popen(                                    # noqa: S603 — команда наша
        cmd, cwd=str(ROOT), env=child_env(),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\r\n")
        lines.append(line)
        print("  " + line, flush=True)
    return proc.wait(), lines, time.time() - started


# --------------------------------------------------------------------------- #
# Preflight кэша эмбеддингов (до любой генерации)
# --------------------------------------------------------------------------- #

def report_cache() -> None:
    """Список того, что в `cache/` уже есть — чтобы было видно, где взять закэшированную пару."""
    files = sorted(CACHE_DIR.glob("embeds_*.pt"))
    print(f"[cache] записей в {rel(CACHE_DIR)}: {len(files)}", flush=True)
    for path in files:
        meta = read_sidecar(path) or {}
        print(f"        {path.name}  prompt={meta.get('prompt_file') or '?'} "
              f"crop={meta.get('crop') or '?'} res={meta.get('resolution') or '?'}", flush=True)


def preflight(inp: str, prompt_file: str, resolution: int) -> dict[str, Any]:
    """Есть ли готовый кэш эмбеддингов для пары (промт, вход) на этой резолюции.

    Возвращает `{prompt_md5, image_sha, crop, embeds, meta}`; при промахе печатает состав кэша
    и бросает `SystemExit(2)` — до загрузки модели, чтобы не заплатить CPU-энкод.
    """
    inp_path = ROOT / inp
    prompt_path = ROOT / prompt_file
    if not inp_path.is_file():
        print(f"[abort] нет входного файла {inp}", flush=True)
        raise SystemExit(2)
    if not prompt_path.is_file():
        print(f"[abort] нет файла промта {prompt_file}", flush=True)
        raise SystemExit(2)

    text = prompt_path.read_text(encoding="utf-8").strip()
    pmd5 = prompt_md5(text)
    img_sha = sha256_file(inp_path, limit=None)[:12]
    print(f"[preflight] input {inp} (sha {img_sha}) | prompt {prompt_file} (md5 {pmd5}) | "
          f"resolution {resolution}", flush=True)

    cands = sorted(CACHE_DIR.glob(f"embeds_{pmd5}_{img_sha}*.pt"))
    hits: list[tuple[Path, dict[str, Any]]] = []
    for path in cands:
        meta = read_sidecar(path)
        if meta is None:
            continue
        if int(meta.get("resolution") or 0) != int(resolution):
            continue
        hits.append((path, meta))

    if not hits:
        print(f"[abort] нет кэша эмбеддингов для этой пары @ resolution {resolution} — "
              f"сначала один прогон run_style.py с теми же --input/--prompt-file, иначе "
              f"снимок эталонов заплатит 55-255 с CPU-энкода", flush=True)
        if cands:
            print(f"[abort] для этой пары есть записи на других резолюциях: "
                  f"{', '.join(p.name for p in cands)}", flush=True)
        report_cache()
        raise SystemExit(2)

    path, meta = max(hits, key=lambda item: item[0].stat().st_mtime)
    print(f"[preflight] кэш на месте: {path.name} (энкод не понадобится)", flush=True)
    return {"prompt_md5": pmd5, "image_sha": img_sha, "crop": str(meta.get("crop") or ""),
            "embeds": path.name, "meta": meta, "prompt_text_len": len(text)}


def check_batch_embeds(ctx: dict[str, Any]) -> None:
    """`tools/batch_seeds.py` ищет эмбеддинги по картинке+сетке, БЕЗ промта (`find_embeds`).

    Если самый свежий кеш для этого входа — чужой промт, батч сгенерирует не то, и эталон
    батча станет ложным (новый каркас считает ключ по промту). Поэтому проверяем явно.
    """
    crop = ctx["crop"]
    if not crop:
        print("[abort] в сайдкаре кэша нет поля crop — не могу проверить выбор батча", flush=True)
        raise SystemExit(2)
    cands = [p for p in CACHE_DIR.glob(f"embeds_*_{ctx['image_sha']}_{crop}.pt")
             if "_te-" not in p.name]
    if not cands:
        print(f"[abort] нет записей embeds_*_{ctx['image_sha']}_{crop}.pt", flush=True)
        raise SystemExit(2)
    newest = max(cands, key=lambda p: p.stat().st_mtime)
    if not newest.name.startswith(f"embeds_{ctx['prompt_md5']}_"):
        print(f"[abort] batch_seeds возьмёт чужой кэш {newest.name} (нужен "
              f"embeds_{ctx['prompt_md5']}_*) — эталон батча был бы ложным. "
              f"Сделай один прогон run_style.py с этим промтом или подчисти более свежую запись.",
              flush=True)
        raise SystemExit(2)
    print(f"[preflight] батч возьмёт {newest.name}", flush=True)


# --------------------------------------------------------------------------- #
# Сценарии снимка
# --------------------------------------------------------------------------- #

def scenarios(a: argparse.Namespace) -> list[dict[str, Any]]:
    """Описание всех эталонных прогонов: имя, команда, параметры, как найти результат."""
    common = ["--input", a.input, "--prompt-file", a.prompt_file]
    seed = ["--seed", str(a.seed), "--lora-scale", "0.7"]
    out_root = a.out_root.rstrip("/")
    batch_dir = f"{out_root}/legacy-batch"

    base_dir = dict(input=a.input, prompt_file=a.prompt_file, size="640p", resolution=a.resolution,
                    steps=8, seed=a.seed, lora_scale=0.7, model="base", upscale=0.0,
                    upscale_backend=None, extra={"preview": True})
    specs: list[dict[str, Any]] = [
        dict(
            name="base-640-8", kind="run",
            cmd=[str(PY), "-u", "run_style.py", *common, "--preview", "--steps", "8", *seed,
                 "--out", f"{out_root}/legacy-base-640-8.png"],
            params=dict(base_dir),
        ),
        dict(
            name="turbo-640", kind="run",
            cmd=[str(PY), "-u", "run_style.py", "--model", "turbo", *common, "--preview", *seed,
                 "--out", f"{out_root}/legacy-turbo-640.png"],
            params=dict(base_dir, model="turbo",
                        extra={"preview": True, "steps": "8 (фиксировано sample_sigmas)"}),
        ),
        dict(
            name="esrgan-x4", kind="run",
            cmd=[str(PY), "-u", "run_style.py", *common, "--preview", "--steps", "8", *seed,
                 "--upscale", "4", "--upscale-backend", "esrgan",
                 "--out", f"{out_root}/legacy-esrgan4.png"],
            params=dict(base_dir, upscale=4.0, upscale_backend="esrgan",
                        extra={"preview": True, "upscale_tile": 256, "upscale_pad": 32,
                               "color_fix": "wavelet", "color_strength": 0.75,
                               "color_sigma": 12.0}),
        ),
        dict(
            name="batch-2seeds", kind="batch",
            cmd=[str(PY), "-u", "tools/batch_seeds.py", *common, "--gen-res", str(a.resolution),
                 "--out-res", "1024", "--seeds", "0-1", "--lora-scale", "0.7", "--steps", "8",
                 "--out-dir", batch_dir],
            params=dict(input=a.input, prompt_file=a.prompt_file, gen_res=a.resolution,
                        out_res=1024, resolution=a.resolution, steps=8, seeds="0-1",
                        lora_scale=0.7, model="base", out_dir=batch_dir,
                        upscale=2, notes="фиксируются только s0-gen.png и s1-gen.png"),
            files=[f"{batch_dir}/s0-gen.png", f"{batch_dir}/s1-gen.png"],
        ),
    ]
    if a.only:
        wanted = {name.strip() for name in a.only}
        unknown = wanted - {spec["name"] for spec in specs}
        if unknown:
            print(f"[abort] неизвестные записи: {', '.join(sorted(unknown))}", flush=True)
            raise SystemExit(2)
        specs = [spec for spec in specs if spec["name"] in wanted]
    return specs


def done_path(lines: list[str]) -> str:
    """Путь из строки `[done] <path> gen …` (так печатает run_style.py)."""
    for line in reversed(lines):
        if line.startswith("[done] "):
            raw = line[len("[done] "):].split()[0]
            return rel(ROOT / raw)
    raise ValueError("в выводе нет строки [done] — не знаю, какой кадр сохранился")


def file_facts(path: str) -> dict[str, Any]:
    full = ROOT / path
    if not full.is_file():
        raise FileNotFoundError(f"нет файла эталона {path}")
    from PIL import Image                            # только размеры; torch не нужен

    with Image.open(full) as image:
        size = [int(image.size[0]), int(image.size[1])]
        mode = image.mode
    return {"path": path, "sha256": sha256_file(full), "size": full.stat().st_size,
            "image_size": size, "mode": mode}


def snapshot(spec: dict[str, Any]) -> dict[str, Any]:
    code, lines, elapsed = run_stream(spec["cmd"])
    missed = [mark for mark in MISS_MARKERS if any(mark in line for line in lines)]
    if code != 0:
        print(f"[abort] {spec['name']}: команда вернула {code}", flush=True)
        raise SystemExit(2)
    if missed or any("[abort]" in line for line in lines):
        print(f"[abort] {spec['name']}: сработал маркер промаха кэша/отказа "
              f"{missed or ['[abort]']} — эталон не записан", flush=True)
        raise SystemExit(2)

    entry: dict[str, Any] = dict(
        kind=spec["kind"],
        cmd=portable_cmd(spec["cmd"]),
        params=spec["params"],
        timing_s=round(elapsed, 1),
    )
    if spec["kind"] == "batch":
        files = {path.rsplit("/", 1)[-1]: file_facts(path) for path in spec["files"]}
        first = files[spec["files"][0].rsplit("/", 1)[-1]]
        entry.update(files=files, path=first["path"], sha256=first["sha256"],
                     size=first["size"], image_size=first["image_size"], mode=first["mode"])
    else:
        entry.update(file_facts(done_path(lines)))
    return entry


# --------------------------------------------------------------------------- #
# JSON: запись и печать
# --------------------------------------------------------------------------- #

def env_versions() -> dict[str, Any]:
    """Версии стека: хеши валидны только для них (AGENTS.md §4)."""
    code = ("import torch, diffusers, transformers;"
            "print(torch.__version__, diffusers.__version__, transformers.__version__)")
    try:
        out = subprocess.run([str(PY), "-c", code], cwd=str(ROOT), env=child_env(),
                             capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=300, check=True)
        torch_v, diffusers_v, transformers_v = out.stdout.split()
    except (subprocess.SubprocessError, ValueError, OSError):
        torch_v = diffusers_v = transformers_v = "?"
    return {"python": sys.version.split()[0], "torch": torch_v,
            "diffusers": diffusers_v, "transformers": transformers_v,
            "gpu": "NVIDIA RTX 4080 16 GiB"}


def previous(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    entries = data.get("entries")
    return entries if isinstance(entries, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def print_table(payload: dict[str, Any]) -> None:
    entries = payload.get("entries") or {}
    print(f"[reference] {payload.get('generated_at', '?')} | записей: {len(entries)}", flush=True)
    for name, entry in entries.items():
        extra = ""
        if "files" in entry:
            extra = "  files: " + ", ".join(
                f"{key} {value['sha256'][:12]}" for key, value in entry["files"].items())
        print(f"  {name:<14} {entry.get('sha256', '?')[:12]}  {entry.get('size', 0):>9} B  "
              f"{'x'.join(str(x) for x in entry.get('image_size', []))}{extra}", flush=True)
    print(f"  в генерации платился CPU-энкод: "
          f"{'да' if payload.get('encoder_paid') else 'нет'}", flush=True)


def cmd_write(a: argparse.Namespace) -> int:
    target = Path(a.write)
    ctx = preflight(a.input, a.prompt_file, a.resolution)
    specs = scenarios(a)
    if any(spec["kind"] == "batch" for spec in specs):
        check_batch_embeds(ctx)
    if a.verbose_cache:
        report_cache()

    old = previous(target)
    entries: dict[str, Any] = {}
    for spec in specs:
        print(f"\n[parity] {spec['name']}", flush=True)
        entry = snapshot(spec)
        prev = old.get(spec["name"])
        if prev and prev.get("sha256") == entry["sha256"]:
            entry["deterministic"] = "совпало с предыдущим снимком"
        elif prev:
            entry["deterministic"] = f"ИЗМЕНИЛОСЬ: было {prev.get('sha256')}"
        print(f"[parity] {spec['name']}: sha256={entry['sha256']} size={entry['size']} "
              f"{'x'.join(str(x) for x in entry['image_size'])} {entry['timing_s']}s"
              + (f"  ({entry['deterministic']})" if entry.get("deterministic") else ""), flush=True)
        entries[spec["name"]] = entry

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "generator": {"script": "tools/parity_refs.py", "legacy_code":
                      ["run_style.py", "tools/batch_seeds.py"],
                      "cmd": " ".join(sys.argv)},
        "env": env_versions(),
        "inputs": {"input": a.input, "input_sha12": ctx["image_sha"],
                   "prompt_file": a.prompt_file, "prompt_md5": ctx["prompt_md5"],
                   "prompt_chars": ctx["prompt_text_len"], "embeds": ctx["embeds"],
                   "crop": ctx["crop"], "resolution": a.resolution, "seed": a.seed},
        "encoder_paid": False,
        "entries": entries,
        "notes": NOTES,
    }
    write_json(target, payload)
    print(f"\n[parity] записано {len(entries)} записей в {rel(target)}", flush=True)
    print_table(payload)
    return 0


def cmd_show(a: argparse.Namespace) -> int:
    target = Path(a.write or a.reference)
    if not target.is_file():
        print(f"[reference] {rel(target)} ещё нет — сними эталоны: "
              f"tools/parity_refs.py --write", flush=True)
        return 0
    data = json.loads(target.read_text(encoding="utf-8"))
    print(f"[reference] {rel(target)}", flush=True)
    for key in ("generated_at", "env", "inputs"):
        value = data.get(key)
        if isinstance(value, dict):
            print(f"  {key}: " + ", ".join(f"{k}={v}" for k, v in value.items()), flush=True)
        else:
            print(f"  {key}: {value}", flush=True)
    print_table(data)
    print(f"  notes: {data.get('notes', '')}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Эталонные sha256 кадров старого кода (run_style.py / tools/batch_seeds.py). "
                    "Без --write просто печатает сохранённый tests/reference.json.")
    ap.add_argument("--write", nargs="?", const=DEFAULT_REFERENCE, default=None,
                    metavar="PATH",
                    help=f"снять эталоны старым кодом и записать JSON (по умолчанию {DEFAULT_REFERENCE})")
    ap.add_argument("--reference", default=DEFAULT_REFERENCE,
                    help=f"какой JSON читать без --write (по умолчанию {DEFAULT_REFERENCE})")
    ap.add_argument("--input", default=DEFAULT_INPUT, help=f"вход (по умолчанию {DEFAULT_INPUT})")
    ap.add_argument("--prompt-file", default=DEFAULT_PROMPT,
                    help=f"промт (по умолчанию {DEFAULT_PROMPT})")
    ap.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                    help=f"resolution профиля, должна быть закэширована (по умолчанию {DEFAULT_RESOLUTION})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"seed (по умолчанию {DEFAULT_SEED})")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                    help=f"куда писать кадры эталонов (по умолчанию {DEFAULT_OUT_ROOT})")
    ap.add_argument("--only", action="append", metavar="NAME",
                    help="снять только указанные записи (можно повторять)")
    ap.add_argument("--verbose-cache", action="store_true", help="напечатать весь состав cache/embeds_*.pt")
    a = ap.parse_args(argv)
    if a.write is None and a.only:
        print("[abort] --only имеет смысл только вместе с --write", flush=True)
        return 2
    return cmd_show(a) if a.write is None else cmd_write(a)


if __name__ == "__main__":
    raise SystemExit(main())
