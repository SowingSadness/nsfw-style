"""`python -m qwenstyle bench vram` — замер пика VRAM/RSS по фазам (перенос `tools/measure_vram.py`).

Зачем: пик ~15.5 GiB @1024 складывается из весов и «активаций», но неизвестно, что именно в них
доминирует (attention, MLP, декод в fp32, резерв аллокатора). Команда снимает
`allocated`/`reserved`/`peak` вокруг каждой фазы, медиану forward'а трансформера, время VAE-энкода
и декода, пик RSS и пишет json в `out/`.

Важное отличие от старого скрипта: **логики загрузки здесь нет**. Пайплайн собирается тем же
`backend.prepare()`, что и обычный прогон (fp8-кэш, LoRA, размещение, offload), генерация — тем же
`backend.generate`. Поэтому замер не разъедется с реальным путём: если поменять бекенд, bench
померит уже новую раскладку.

Эмбеддинги берутся **только из кэша** (CPU-энкод не платится): запись ищется по `sha256(вход)` и
сетке conditioning, без знания текста промта (как `find_embeds` в старом скрипте). Нет записи ->
код 2 и подсказка, какой `run` нужен.

Второе действие — `bench encoder`: замер диск-стриминга текст-энкодера (`encoder.stream`, см.
`qwenstyle/streaming.py`). Логики загрузки весов здесь тоже нет: `--stream 0|1` уходит в
`streaming.load_text_encoder`, а команда только снимает память, печатает hidden states и умеет
сохранять/сверять их (`--save`/`--compare`). Главная метрика — **uss** (приватная память): страницы
mmap из `safe_open` попадают в rss и дают ложное «не помогло». Плечи конвейера
(`--device/--prefetch/--no-pin/--lookahead`) печатаются строкой `StreamStats.summary()`
(`[stream] read=… copy=… compute=… wait=… total=… ratio_wait=…`), пик VRAM — `peak_vram`.

Третье действие — `bench io`: микробенчмарки плеч конвейера (NVMe -> RAM и RAM -> VRAM), чтобы
вердикт «упёрлись в диск или в compute» опирался на числа, а не на оценки. NVMe читается
`streaming.ShardedWeights` (mmap) два прохода подряд — первый (вероятно холодный) и второй (тёплый
page cache); H2D копируется на GPU блоками `--chunk-mib` pinned vs pageable, время — по
`torch.cuda.Event`. Своей логики загрузки весов здесь тоже нет.

Запуск:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m qwenstyle bench vram --resolution 640
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m qwenstyle bench vram --resolution 1024 \\
        --input out/canyon-edit-640-lora1.0-s777.png --vae-dtype bf16
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256 .venv/Scripts/python.exe \\
        -m qwenstyle bench vram --resolution 1024 --tag maxsplit
    .venv/Scripts/python.exe -m qwenstyle bench encoder \\
        --repo darknight9121/FLUX.2-klein-base-9B-bucket-uncensored --subfolder text_encoder \\
        --tokens 8 --stream 0 --save cache/models/_bench/te_full.pt
    .venv/Scripts/python.exe -m qwenstyle bench encoder \\
        --repo darknight9121/FLUX.2-klein-base-9B-bucket-uncensored --subfolder text_encoder \\
        --tokens 8 --stream 1 --compare cache/models/_bench/te_full.pt
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m qwenstyle bench encoder \\
        --repo darknight9121/FLUX.2-klein-base-9B-bucket-uncensored --subfolder text_encoder \\
        --tokens 8 --stream 1 --device cuda --prefetch 4 --lookahead 1
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m qwenstyle bench io \\
        --read-gib 2 --h2d-gib 0.5 --chunk-mib 128
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from ..config import DEFAULT_STEPS, EMBEDS_DIR, PREVIEW_STEPS, RunConfig, available_presets
from . import parse_sets

NAME = "bench"
HELP = ("замеры: vram — пик VRAM/RSS по фазам на реальном прогоне; "
        "encoder — стриминг текст-энкодера (rss/uss/VRAM, плечи конвейера, диффы по слоям); "
        "io — микробенчмарки плеч NVMe->RAM и RAM->VRAM")
DEFAULT_CONFIG = "base"
DEFAULT_SEED = 20261113
DEFAULT_LORA_SCALE = 0.7
DEFAULT_RESOLUTION = 1024
#: Разрешения, на которых проект замерялся (640 = 480p, 1024 = 768p, 1152 = 864p, 1280 = 960p).
RESOLUTIONS = (640, 896, 1024, 1152, 1280)
GIB = 1024 ** 3
#: `bench encoder`: слои по умолчанию (у `Flux2KleinPipeline` — 9/18/27, у остальных 10/20/30).
DEFAULT_OUT_LAYERS = "9,18,27"
DEFAULT_TOKENS = 64
DEFAULT_PROMPT = "a test prompt"
#: Интервал фонового сэмплера памяти (с).
MEM_INTERVAL = 0.1
#: `bench io`: объёмы по умолчанию — команда должна занимать секунды, а не минуты.
DEFAULT_READ_GIB = 2.0
DEFAULT_H2D_GIB = 0.5
#: Блоки H2D-копии (МиБ): три характерных размера для замера pinned/pageable.
H2D_CHUNKS_MIB = (64, 128, 256)
DEFAULT_CHUNK_MIB = 128
#: Веса по умолчанию для `bench io`: уже лежат в `hf-cache` (R3: сеть не нужна).
DEFAULT_IO_REPO = "darknight9121/FLUX.2-klein-base-9B-bucket-uncensored"
DEFAULT_IO_SUBFOLDER = "text_encoder"
#: Десятичный GB: объёмы в выводе — GiB, скорости — GB/s (так «GB/s» читается буквально).
GB = 1e9


# --------------------------------------------------------------------------- #
# Замеры
# --------------------------------------------------------------------------- #

def _gib(value: float) -> float:
    return value / GIB


def _fmt_gib(value: float | None) -> str:
    """GiB-число для логов (`?`, если psutil недоступен и значение не снято)."""
    return "?" if value is None else f"{value:.2f}"


def _gbps(nbytes: int, seconds: float) -> float:
    """Скорость в десятичных GB/s (`0.0`, если время не набрано)."""
    return nbytes / GB / seconds if seconds > 0 else 0.0


def _rss_gib() -> float | None:
    """RSS процесса в GiB (psutil; без него — None, замер VRAM всё равно валиден)."""
    try:
        import psutil

        return round(_gib(psutil.Process().memory_info().rss), 3)
    except Exception:                              # noqa: BLE001 — диагностика не обязательна
        return None


def _memory_now() -> dict[str, float | None]:
    """`rss`/`uss` процесса в GiB прямо сейчас (`memory_full_info`; без psutil — None).

    `uss` (unique set size ≈ приватная память) — главная метрика диск-стриминга: страницы
    mmap-файлов попадают в rss, но не в uss, поэтому «rss не упал» ≠ «стриминг не помогает».
    """
    try:
        import psutil

        proc = psutil.Process()
        try:
            info = proc.memory_full_info()          # на Windows даёт uss (private working set)
        except Exception:                           # noqa: BLE001 — деградация до rss
            info = proc.memory_info()
        rss = getattr(info, "rss", None)
        uss = getattr(info, "uss", None)
        return {
            "rss_GiB": round(_gib(float(rss)), 3) if rss is not None else None,
            "uss_GiB": round(_gib(float(uss)), 3) if uss is not None else None,
        }
    except Exception:                              # noqa: BLE001 — диагностика не обязательна
        return {"rss_GiB": None, "uss_GiB": None}


class _MemSampler:
    """Фоновый сэмплер памяти: поток каждые `interval` с, только пики и последний снимок.

    Зачем поток, а не два замера до/после: пик у стриминга живёт внутри forward (модуль
    материализован + активации), между фазами его уже не видно. Ошибки psutil глушим:
    диагностика не должна ронять замер времени/диффов.
    """

    def __init__(self, interval: float = MEM_INTERVAL) -> None:
        self.interval = float(interval)
        self.peak_rss: float | None = None
        self.peak_uss: float | None = None
        self.last: dict[str, float | None] = {"rss_GiB": None, "uss_GiB": None}
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        now = _memory_now()
        self.samples += 1
        self.last = now
        for key, peak_name in (("rss_GiB", "peak_rss"), ("uss_GiB", "peak_uss")):
            value = now.get(key)
            if value is None:
                continue
            current = getattr(self, peak_name)
            if current is None or value > current:
                setattr(self, peak_name, value)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self._sample()
            except Exception:                       # noqa: BLE001 — фон не должен падать
                pass

    def start(self) -> "_MemSampler":
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="bench-mem", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> "_MemSampler":
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._sample()
        return self

    def peak_line(self) -> str:
        def _fmt(value: float | None) -> str:
            return "?" if value is None else f"{value:.2f}"

        return (f"peak_rss={_fmt(self.peak_rss)} peak_uss={_fmt(self.peak_uss)} GiB "
                f"(сейчас rss={_fmt(self.last.get('rss_GiB'))} "
                f"uss={_fmt(self.last.get('uss_GiB'))}, сэмплов {self.samples})")


def snapshot() -> dict[str, Any]:
    """Снимок памяти: аллокатор torch + по процессу + RSS."""
    import torch

    free, total = torch.cuda.mem_get_info()
    data = {
        "allocated_GiB": round(_gib(torch.cuda.memory_allocated()), 3),
        "reserved_GiB": round(_gib(torch.cuda.memory_reserved()), 3),
        "peak_allocated_GiB": round(_gib(torch.cuda.max_memory_allocated()), 3),
        "peak_reserved_GiB": round(_gib(torch.cuda.max_memory_reserved()), 3),
        "used_total_GiB": round(_gib(total - free), 3),
        "device_total_GiB": round(_gib(total), 3),
        "rss_GiB": _rss_gib(),
    }
    return data


def _install_hooks(pipe: Any, meta: dict[str, Any]) -> None:
    """Обернуть decode/scheduler.step/forward/vae.encode таймерами (только у torch-пайплайна).

    Пики снимаются `max_memory_allocated()`: `reset_peak_memory_stats()` вызывается перед
    денойзингом и перед декодом, поэтому пик декода не смешивается с пиком денойзинга.
    """
    import torch

    decode, steps, forward, encode = {}, {}, {}, {}
    meta.update({"decode": decode, "scheduler": steps, "forward": forward, "encode": encode})

    vae = getattr(pipe, "vae", None)
    if vae is not None and hasattr(vae, "decode"):
        original_decode = vae.decode

        def timed_decode(*args: Any, **kwargs: Any) -> Any:
            torch.cuda.synchronize()
            decode["denoise_peak_allocated_GiB"] = round(_gib(torch.cuda.max_memory_allocated()), 3)
            decode["denoise_peak_reserved_GiB"] = round(_gib(torch.cuda.max_memory_reserved()), 3)
            torch.cuda.reset_peak_memory_stats()
            started = time.time()
            out = original_decode(*args, **kwargs)
            torch.cuda.synchronize()
            decode["s"] = round(time.time() - started, 2)
            decode["peak_allocated_GiB"] = round(_gib(torch.cuda.max_memory_allocated()), 3)
            decode["peak_reserved_GiB"] = round(_gib(torch.cuda.max_memory_reserved()), 3)
            return out

        vae.decode = timed_decode

    if vae is not None and hasattr(vae, "encode"):
        original_encode = vae.encode

        def timed_encode(*args: Any, **kwargs: Any) -> Any:
            started = time.time()
            out = original_encode(*args, **kwargs)
            torch.cuda.synchronize()
            encode.setdefault("durs", []).append(round(time.time() - started, 2))
            return out

        vae.encode = timed_encode

    scheduler = getattr(pipe, "scheduler", None)
    if scheduler is not None and hasattr(scheduler, "step"):
        original_step = scheduler.step

        def timed_step(*args: Any, **kwargs: Any) -> Any:
            started = time.time()
            out = original_step(*args, **kwargs)
            torch.cuda.synchronize()
            steps.setdefault("durs", []).append(round(time.time() - started, 3))
            return out

        scheduler.step = timed_step

    transformer = getattr(pipe, "transformer", None)
    if transformer is not None and hasattr(transformer, "forward"):
        original_forward = transformer.forward

        def timed_forward(*args: Any, **kwargs: Any) -> Any:
            started = time.time()
            out = original_forward(*args, **kwargs)
            torch.cuda.synchronize()
            forward.setdefault("durs", []).append(round(time.time() - started, 3))
            forward["peak_alloc_GiB"] = max(
                float(forward.get("peak_alloc_GiB") or 0.0),
                round(_gib(torch.cuda.max_memory_allocated()), 3),
            )
            return out

        transformer.forward = timed_forward


# --------------------------------------------------------------------------- #
# Кэш эмбеддингов: поиск по картинке и сетке (как find_embeds в measure_vram.py)
# --------------------------------------------------------------------------- #

def find_embeds(image_sha: str, crop: tuple[int, int], root: str | Path = EMBEDS_DIR) -> Any:
    """Самая свежая запись `cache/embeds/embeds_*<sha>_<W>x<H>.pt` без тега энкодера.

    Возвращает `(Blob, payload)` или `(None, None)`: промт в поиске не участвует (bench не
    принимает `--prompt-file`), поэтому берётся базовый энкодер и именно эта сетка.
    """
    from ..cache import Blob, BlobCache, scan_embeds
    from ..pipeline import blob_cache

    cache = blob_cache(root)
    wanted = f"{int(crop[0])}x{int(crop[1])}"
    candidates = [entry for entry in scan_embeds(cache.root)
                  if entry.sha == image_sha and entry.crop == wanted and entry.enc == "base"]
    if not candidates:
        return None, None
    newest = max(candidates, key=lambda entry: entry.mtime)
    key = newest.path.name[len("embeds_"):-len(".pt")]
    blob = Blob(stage="embeds", key=key)
    return blob, BlobCache(cache.root).get(blob)


# --------------------------------------------------------------------------- #
# Сборка конфига и прогон
# --------------------------------------------------------------------------- #

def build_config(args: argparse.Namespace) -> RunConfig:
    """`bench vram` -> `RunConfig`: resolution/шаги/seed/lora/vae из флагов."""
    preset = str(getattr(args, "model", None) or getattr(args, "config", None) or DEFAULT_CONFIG)
    overrides: dict[str, Any] = dict(parse_sets(getattr(args, "sets", None)))
    overrides.update({
        "input": str(args.input),
        "resolution": int(args.resolution),
        "generation.steps": int(args.steps if args.steps is not None
                                else (PREVIEW_STEPS if args.resolution == 640 else DEFAULT_STEPS)),
        "seed": int(args.seed),
        "generation.lora.scale": float(args.lora_scale),
    })
    # Замер не обслуживает кэш: даже если бы прогон писал эмбеддинги, вытеснять чужие записи
    # он не должен (как и `check-parity`). `bench encoder` RunConfig вообще не строит и ничего
    # не пишет в `cache/embeds/embeds_*`, так что у него вытеснять нечего by construction.
    overrides["cache_supersede"] = "none"
    if getattr(args, "offload", None):
        overrides["generation.offload"] = str(args.offload)
    return RunConfig.from_profile(preset, **overrides)


def run_vram(args: argparse.Namespace) -> int:
    """Реальный прогон с замерами: prepare -> generate (+ хуки по фазам) -> json."""
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"[bench] нет входного кадра {input_path} (--input)", file=sys.stderr, flush=True)
        return 2
    try:
        cfg = build_config(args)
    except (ValueError, OSError) as exc:
        print(f"[bench] {exc}", file=sys.stderr, flush=True)
        return 2

    from .. import pipeline

    info = pipeline.dry_run(cfg)
    crop = tuple(int(v) for v in str(info["crop"]).split("x"))
    blob, payload = find_embeds(str(info["image_sha"]), crop)
    if payload is None:
        print(f"[abort] нет кэша эмбеддингов для image {info['image_sha']} @ {info['crop']}",
              flush=True)
        print(f"        сначала один прогон: python -m qwenstyle run --config {cfg.preset} "
              f"--resolution {cfg.resolution} --input {cfg.input} --prompt-file promts/exp3.txt "
              f"--out out/_bench.png", flush=True)
        print("        (иначе замер заплатит 55–255 с CPU-энкода)", flush=True)
        return 2

    import torch
    from PIL import Image

    from ..registry import get_generation

    print(f"[env] torch {torch.__version__} | gpu={torch.cuda.get_device_name(0)} "
          f"| alloc_conf={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')!r}", flush=True)
    print(f"[cfg] res={args.resolution} crop={info['crop']} steps={cfg.steps} "
          f"vae={args.vae_dtype} preset={cfg.preset} offload={cfg.generation.offload}", flush=True)
    print(f"[in]  image {info['image_sha']} | embeds {blob.name}", flush=True)

    image = Image.open(cfg.input).convert("RGBA")
    backend_cls = get_generation(cfg.generation.backend)
    backend = backend_cls()
    result: dict[str, Any] = {
        "config": {"preset": cfg.preset, "resolution": args.resolution, "crop": info["crop"],
                   "steps": cfg.steps, "vae_dtype": args.vae_dtype, "tag": args.tag,
                   "offload": cfg.generation.offload, "seed": args.seed, "lora_scale": args.lora_scale,
                   "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
                   "image_sha12": info["image_sha"], "embeds": blob.name,
                   "torch": torch.__version__},
    }
    handle = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        handle = backend.prepare(cfg, need_encoder=False)
        torch.cuda.synchronize()
        result["load_s"] = round(time.time() - started, 2)
        result["lora_tensors_bf16"] = None      # печатает бекенд; здесь только факт загрузки

        pipe = getattr(handle, "pipe", None)
        vae_before = None
        if pipe is not None:
            meta: dict[str, Any] = {}
            vae = getattr(pipe, "vae", None)
            if vae is not None:
                vae_before = str(getattr(vae, "dtype", "?"))
                if args.vae_dtype == "bf16":
                    vae.to(torch.bfloat16)
                result["vae_dtype_load"] = str(vae.dtype)
            _install_hooks(pipe, meta)
        else:
            meta = {}
            print("[bench] бекенд не отдаёт pipe — общий замер без хуков по фазам", flush=True)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        fases = result["fases"] = {"after_load": snapshot()}
        print(f"[1/5] загрузка {result['load_s']}s | vae {vae_before} -> "
              f"{result.get('vae_dtype_load', '?')} | занято "
              f"{fases['after_load']['used_total_GiB']} GiB "
              f"(rss {fases['after_load']['rss_GiB']})", flush=True)

        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        generated = backend.generate(cfg, handle, payload, image, int(cfg.seed))
        torch.cuda.synchronize()
        total_s = round(time.time() - started, 2)
        fases["after_run"] = snapshot()
        result["total_s"] = total_s
        result["vae_dtype_run"] = str(getattr(getattr(pipe, "vae", None), "dtype", "?"))

        decode = meta.get("decode") or {}
        durs = (meta.get("scheduler") or {}).get("durs") or []
        fdurs = (meta.get("forward") or {}).get("durs") or []
        enc = (meta.get("encode") or {}).get("durs") or []
        fases["denoise"] = {"peak_allocated_GiB": decode.get("denoise_peak_allocated_GiB"),
                            "peak_reserved_GiB": decode.get("denoise_peak_reserved_GiB")}
        fases["decode"] = {"peak_allocated_GiB": decode.get("peak_allocated_GiB"),
                           "peak_reserved_GiB": decode.get("peak_reserved_GiB"),
                           "s": decode.get("s")}
        result["steps_s"] = {
            "scheduler_count": len(durs),
            "scheduler_median": round(sorted(durs)[len(durs) // 2], 3) if durs else None,
            "transformer_count": len(fdurs),
            "transformer_median": round(sorted(fdurs)[len(fdurs) // 2], 3) if fdurs else None,
            "transformer_total": round(sum(fdurs), 2),
            "transformer_peak_alloc_GiB": (meta.get("forward") or {}).get("peak_alloc_GiB"),
            "vae_encode_durs": enc,
            "overhead_s": round(total_s - sum(fdurs) - float(decode.get("s") or 0.0) - sum(enc), 2),
        }
        rss_values = [value for value in (fases.get(key, {}).get("rss_GiB")
                                          for key in ("after_load", "after_run")) if value]
        result["peak_overall_GiB"] = fases["after_run"]["used_total_GiB"]
        result["rss_peak_GiB"] = max(rss_values) if rss_values else None
        result["image_size"] = [int(generated.size[0]), int(generated.size[1])]
    finally:
        if handle is not None:
            try:
                backend.release(handle)
            except Exception as exc:               # noqa: BLE001 — освобождение не должно падать
                print(f"[bench] release: {type(exc).__name__}: {exc}", flush=True)

    tag = f"-{args.tag}" if args.tag else ""
    out_json = Path("out") / f"measure-vram-{args.resolution}-{args.vae_dtype}{tag}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if int(args.save or 0):
        from ..io import save_atomic

        saved = save_atomic(generated, Path("out") / f"measure-{args.resolution}-{args.vae_dtype}{tag}.png")
        result["image"] = str(saved)

    fases = result.get("fases", {})
    steps_s = result.get("steps_s", {})
    print(f"[2/5] шаги: {steps_s.get('transformer_count')} forward, медиана "
          f"{steps_s.get('transformer_median')} s/шаг, сумма {steps_s.get('transformer_total')} s; "
          f"VAE-энкод {steps_s.get('vae_encode_durs')}; прочее (внутри pipe) "
          f"{steps_s.get('overhead_s')} s", flush=True)
    print(f"[3/5] пик денойзинга: alloc {(fases.get('denoise') or {}).get('peak_allocated_GiB')} / "
          f"reserved {(fases.get('denoise') or {}).get('peak_reserved_GiB')} GiB", flush=True)
    decode = fases.get("decode") or {}
    print(f"[4/5] VAE-декод: {decode.get('s')} s, пик alloc {decode.get('peak_allocated_GiB')} / "
          f"reserved {decode.get('peak_reserved_GiB')} GiB (dtype на прогоне "
          f"{result.get('vae_dtype_run')})", flush=True)
    after = fases.get("after_run") or {}
    print(f"[5/5] итого {result.get('total_s')} s | пик по процессу {result.get('peak_overall_GiB')} GiB "
          f"(allocated {after.get('peak_allocated_GiB')} / reserved {after.get('peak_reserved_GiB')}, "
          f"rss {result.get('rss_peak_GiB')})", flush=True)
    print(f"       json: {out_json}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# `bench encoder`: диск-стриминг текст-энкодера (`encoder.stream`)
# --------------------------------------------------------------------------- #

def _read_json(path: Path) -> dict[str, Any] | None:
    """`json` файла или None (битый/отсутствующий — не ошибка: ниже есть дефолты)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _encoder_local_dir(target: str, subfolder: str | None) -> Path:
    """Каталог с `config.json` + шардами энкодера: резолв только средствами `streaming`.

    Сеть здесь запрещена (`local_files_only=True`): bench никогда ничего не скачивает, а ищет
    готовый snapshot в `hf-cache` или локальный каталог. Нет весов -> `RuntimeError` из
    `streaming` с командой `snapshot_download`.
    """
    from .. import streaming

    source = streaming.resolve_stream_source(target, subfolder=subfolder, local_files_only=True)
    return Path(source.shards[0]).parent


def _resolve_subfolder(local_dir: str | None, subfolder: str | None) -> str | None:
    """Подкаталог энкодера: если в `local_dir` его нет, но `config.json` лежит в корне — корень.

    Ничего не скачивает и веса не читает (два `is_file()`): нужно, чтобы `--local-dir PATH`
    работал и когда PATH — каталог энкодера, и когда — корень snapshot'а.
    """
    if not local_dir or not subfolder:
        return subfolder
    explicit = Path(str(local_dir))
    if (explicit / str(subfolder) / "config.json").is_file() \
            or not (explicit / "config.json").is_file():
        return subfolder
    print(f"[te] в {explicit} нет подкаталога {subfolder!r}, но есть config.json — "
          f"считаю его каталогом энкодера", flush=True)
    return None


def _encoder_class_for(encoder_dir: Path) -> tuple[Any, str]:
    """Класс энкодера: `model_index.json` -> `config.json:architectures` -> дефолт `PIPELINE_SPECS`.

    Импорт класса — существующим хелпером `backends/flux2.py::_import_encoder` (он же ищет
    в `transformers`), т.е. своих таблиц «имя -> класс» здесь не заводится.
    """
    from ..backends import flux2

    name = ""
    origin = ""
    index = _read_json(encoder_dir.parent / "model_index.json")
    value = (index or {}).get("text_encoder")
    if isinstance(value, (list, tuple)) and len(value) == 2:
        name, origin = str(value[1]), "model_index.json"
    if not name:
        config = _read_json(encoder_dir / "config.json") or {}
        architectures = config.get("architectures") or []
        if architectures:
            name, origin = str(architectures[0]), "config.json"
    if not name:
        default = flux2.PIPELINE_SPECS[flux2.DEFAULT_PIPELINE]["encoder_class"]
        name, origin = str(default), "дефолт PIPELINE_SPECS"
    return flux2._import_encoder(name), f"{name} ({origin})"


def _tokenizer_dir(encoder_dir: Path) -> Path | None:
    """Каталог токенизатора: рядом с энкодером или в корне snapshot'а (`tokenizer/` у FLUX.2)."""
    for candidate in (encoder_dir / "tokenizer", encoder_dir.parent / "tokenizer", encoder_dir):
        if (candidate / "tokenizer_config.json").is_file() \
                or (candidate / "tokenizer.json").is_file():
            return candidate
    return None


def _parse_layers(text: str) -> list[int]:
    """`"9,18,27"` -> `[9, 18, 27]`; мусор -> `ValueError` (а не молчаливый пустой список)."""
    layers: list[int] = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        try:
            layers.append(int(part))
        except ValueError as exc:
            raise ValueError(f"--out-layers ждёт числа через запятую, получено {text!r}") from exc
    return layers


def _encode_prompt(tokenizer: Any, prompt: str, tokens: int) -> tuple[Any, Any]:
    """Промт -> `(input_ids, attention_mask)` фиксированной длины `tokens` (trunc/pad).

    Фиксированная длина нужна для сравнения прогонов: память и время forward зависят от
    длины последовательности, а не от содержания промта. Шаблон чата применяется, если он есть
    (энкодеры FLUX.2 в diffusers работают через него); ошибка шаблона -> обычный текст.
    """
    import torch

    text = prompt
    if getattr(tokenizer, "chat_template", None):
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
            )
        except Exception:                          # noqa: BLE001 — шаблон есть, но не для dict
            text = prompt
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=int(tokens),
                        padding="max_length")
    input_ids = encoded["input_ids"].to(torch.long)
    attention = encoded.get("attention_mask")
    if attention is None:
        attention = torch.ones_like(input_ids)
    return input_ids, attention.to(torch.long)


def run_encoder(args: argparse.Namespace) -> int:
    """Диск-стриминг энкодера на реальных весах: время фаз, rss/uss, hidden states, диффы.

    Логики загрузки весов здесь нет: оба режима идут через `streaming.load_text_encoder`
    (`--stream 0` = `from_pretrained`, `--stream 1` = meta + подгрузка модулей), поэтому
    `--compare` сравнивает ровно то, что сравнивает проект, а не две разные обвязки.
    """
    target = str(getattr(args, "local_dir", None) or getattr(args, "repo", None) or "")
    if not target:
        print("[bench] нужен --repo <id> или --local-dir <path>", file=sys.stderr, flush=True)
        return 2
    try:
        layers = _parse_layers(args.out_layers)
    except ValueError as exc:
        print(f"[bench] {exc}", file=sys.stderr, flush=True)
        return 2
    if not layers:
        print("[bench] --out-layers пуст: нужен хотя бы один слой", file=sys.stderr, flush=True)
        return 2
    tokens = int(args.tokens)
    if tokens <= 0:
        print("[bench] --tokens должен быть > 0", file=sys.stderr, flush=True)
        return 2
    stream = bool(int(args.stream))
    offline = str(os.environ.get("HF_HUB_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}
    device = str(getattr(args, "device", "cpu") or "cpu")
    device_kind = device.split(":", 1)[0].strip().lower()
    if device_kind not in ("cpu", "cuda"):
        print(f"[bench] --device поддерживает cpu|cuda, получено {device!r}",
              file=sys.stderr, flush=True)
        return 2
    prefetch = int(getattr(args, "prefetch", 0) or 0)
    lookahead_arg = getattr(args, "lookahead", None)
    lookahead = int(1 if lookahead_arg is None else lookahead_arg)
    if prefetch < 0 or lookahead < 1:
        # те же границы, что у `EncoderConfig` (config.py): prefetch >= 0, lookahead >= 1
        print("[bench] --prefetch должен быть >= 0 (0 = без конвейера), "
              "--lookahead >= 1 (сколько копий H2D выдавать заранее)",
              file=sys.stderr, flush=True)
        return 2
    pin_memory = not bool(getattr(args, "no_pin", False))

    subfolder = _resolve_subfolder(getattr(args, "local_dir", None),
                                   getattr(args, "subfolder", None))

    try:
        encoder_dir = _encoder_local_dir(target, subfolder)
        model_cls, class_note = _encoder_class_for(encoder_dir)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"[bench] {exc}", file=sys.stderr, flush=True)
        return 2

    tokenizer_dir = _tokenizer_dir(encoder_dir)
    if tokenizer_dir is None:
        print(f"[bench] нет токенизатора рядом с {encoder_dir} (искал tokenizer_config.json "
              f"в <энкодер>/tokenizer, <snapshot>/tokenizer, <энкодер>)", file=sys.stderr, flush=True)
        return 2

    import torch
    from transformers import AutoTokenizer

    from .. import streaming

    print(f"[te] {class_note} | веса {encoder_dir} | tokenizer {tokenizer_dir.name} "
          f"| stream={int(stream)} device={device} prefetch={prefetch} pin={int(pin_memory)} "
          f"lookahead={lookahead} dtype={args.dtype} tokens={tokens} offline={int(offline)}",
          flush=True)
    if prefetch > 0 and (not stream or device_kind != "cuda"):
        print(f"[te] конвейер (--prefetch={prefetch}) работает только при --stream 1 --device cuda: "
              f"сейчас stream={int(stream)} device={device}, prefetch игнорируется", flush=True)
    if not stream and device_kind == "cuda":
        print(f"[te] --stream 0: модель грузится целиком и целиком же переедет на {device} "
              f"(для 15 GiB весов это может не влезть в VRAM; для постраничного размещения — "
              f"--stream 1)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    input_ids, attention = _encode_prompt(tokenizer, str(args.prompt), tokens)
    print(f"[te] input_ids {tuple(input_ids.shape)} | prompt {str(args.prompt)!r}", flush=True)

    model: Any = None
    result_code = 0
    peak_vram: float | None = None
    stats = streaming.StreamStats()
    if device_kind == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    sampler = _MemSampler().start()
    try:
        started = time.time()
        model = streaming.load_text_encoder(
            model_cls, str(encoder_dir), dtype=str(args.dtype), local_files_only=True,
            device=device, stream=stream, prefetch=prefetch, pin_memory=pin_memory,
            lookahead=lookahead, stats=stats,
        )
        load_s = time.time() - started
        model.eval()                               # stream-путь уже eval, но режим один
        if not stream and device_kind != "cpu":
            # stream=False раскладку делает вызывающий (как в пайплайне); для stream=True это no-op.
            try:
                streaming.move_to(model, device)
            except RuntimeError as exc:              # OOM на 15 GiB весах — понятная подсказка
                print(f"[bench] не удалось перенести модель на {device}: {exc}; для постраничного "
                      f"размещения нужен --stream 1", file=sys.stderr, flush=True)
                return 2
        print(f"[te] загрузка {load_s:.1f}s | rss={sampler.last.get('rss_GiB')} "
              f"uss={sampler.last.get('uss_GiB')}", flush=True)

        run_device = next(model.parameters()).device
        if run_device.type != "cpu":
            input_ids = input_ids.to(run_device)
            attention = attention.to(run_device)
        with torch.no_grad():
            started = time.time()
            out = model(input_ids=input_ids, attention_mask=attention,
                        output_hidden_states=True, use_cache=False)
            if device_kind == "cuda":
                torch.cuda.synchronize()           # иначе forward_s — только выдача ядер
            forward_s = time.time() - started
        hidden = tuple(getattr(out, "hidden_states", ()) or ())
        if not hidden:
            print("[bench] модель не вернула hidden_states (output_hidden_states=True)",
                  file=sys.stderr, flush=True)
            return 2
        too_deep = [layer for layer in layers if layer >= len(hidden) or layer < 0]
        if too_deep:
            print(f"[bench] слои {too_deep} вне диапазона: hidden_states {len(hidden)} шт. "
                  f"(0..{len(hidden) - 1}); проверь --out-layers", file=sys.stderr, flush=True)
            return 2
        shape = tuple(hidden[layers[0]].shape)
        if device_kind == "cuda":
            peak_vram = _gib(torch.cuda.max_memory_allocated())
        print(f"[te] forward {forward_s:.1f}s | peak uss={sampler.peak_uss} "
              f"rss={sampler.peak_rss} vram={_fmt_gib(peak_vram)}", flush=True)
        print(f"[te] hidden states: layers={layers} shape={shape}", flush=True)

        selected = {int(layer): hidden[int(layer)].detach().to("cpu").clone() for layer in layers}
        del hidden, out

        if args.save:
            save_path = Path(str(args.save))
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"hidden": selected, "input_ids": input_ids.detach().to("cpu")}, save_path)
            print(f"[save] {save_path} ({save_path.stat().st_size / 1024 ** 2:.1f} MiB, "
                  f"слои {sorted(selected)})", flush=True)

        if args.compare:
            compare_path = Path(str(args.compare))
            if not compare_path.is_file():
                print(f"[bench] нет файла для сверки {compare_path}", file=sys.stderr, flush=True)
                return 2
            try:                                   # safe-режим: файл — только тензоры, не pickle-код
                expected = torch.load(compare_path, map_location="cpu", weights_only=True)
            except Exception as exc:               # noqa: BLE001 — старый/битый файл, нужна подсказка
                print(f"[bench] {compare_path} не читается в safe-режиме torch.load "
                      f"(weights_only=True): {type(exc).__name__}: {exc}. Пересоздай эталон на этой "
                      f"версии: тот же запуск с --save {compare_path}", file=sys.stderr, flush=True)
                return 2
            expected_hidden = (expected or {}).get("hidden") or {}
            expected_ids = (expected or {}).get("input_ids")
            if expected_ids is not None:
                same_ids = torch.equal(expected_ids.to("cpu"), input_ids.detach().to("cpu"))
                print(f"[compare] input_ids совпадают={same_ids} ({tuple(input_ids.shape)})",
                      flush=True)
            if not expected_hidden:
                print(f"[bench] в {compare_path} нет словаря 'hidden': сверять нечего",
                      file=sys.stderr, flush=True)
                return 2
            for layer in layers:
                reference = expected_hidden.get(int(layer))
                if reference is None:
                    reference = expected_hidden.get(str(layer))
                current = selected[int(layer)]
                if reference is None:
                    print(f"[diff] layer={layer} max=? (нет в {compare_path})", flush=True)
                    result_code = 1
                    continue
                reference = reference.to("cpu")
                if tuple(reference.shape) != tuple(current.shape):
                    print(f"[diff] layer={layer} max=? shape {tuple(current.shape)} != "
                          f"{tuple(reference.shape)}", flush=True)
                    result_code = 1
                    continue
                max_diff = (current - reference).abs().max().item()
                print(f"[diff] layer={layer} max={max_diff}", flush=True)
                if float(max_diff) > 0.0:
                    result_code = 1
        del selected
    finally:
        sampler.stop()
        weights = getattr(model, "_stream_weights", None)
        if weights is not None:
            try:
                weights.close()                    # закрыть mmap-шарды; файлы на диске не трогаем
            except Exception:                      # noqa: BLE001 — закрытие не должно падать
                pass
        del model
        gc.collect()
        if device_kind == "cuda":
            torch.cuda.empty_cache()

    if device_kind == "cuda":
        # пик аллокатора не сбрасывается empty_cache(): читаем его и после освобождения модели
        peak_vram = _gib(torch.cuda.max_memory_allocated())
    suffix = "" if stream else "  (стриминга не было: --stream 0)"
    print(stats.summary() + suffix, flush=True)
    if stream and stats.t_total > 0:
        # `ratio_wait` из summary видит только `Event.wait()` слоя; блокировка consumer'а на очереди
        # продюсера (самый частый вид ожидания при I/O-bound) в него не попадает. Поэтому печатаем
        # ещё и бюджет total: compute + H2D + всё остальное (ожидание данных/собственное чтение) —
        # именно последняя доля и есть индикатор I/O-bound (вместе с `read / total`), а не `ratio_wait`.
        spent = stats.t_compute + stats.t_copy
        starved = max(0.0, stats.t_total - spent)
        print(f"[stream] budget: total={stats.t_total:.2f}s = compute {stats.t_compute:.2f}s "
              f"({stats.t_compute / stats.t_total:.2f}) + copy {stats.t_copy:.2f}s "
              f"({stats.t_copy / stats.t_total:.2f}) + ожидание данных {starved:.2f}s "
              f"({starved / stats.t_total:.2f}); ratio_wait в summary считает только Event.wait, "
              f"блокировку на очереди продюсера не видит", flush=True)
    print(f"[bench] peak_uss={_fmt_gib(sampler.peak_uss)} peak_vram={_fmt_gib(peak_vram)} "
          f"peak_rss={_fmt_gib(sampler.peak_rss)} GiB (сейчас rss="
          f"{_fmt_gib(sampler.last.get('rss_GiB'))} uss={_fmt_gib(sampler.last.get('uss_GiB'))}, "
          f"сэмплов {sampler.samples})", flush=True)
    print(f"[done] stream={int(stream)} device={device} prefetch={prefetch} "
          f"peak_uss={_fmt_gib(sampler.peak_uss)} peak_vram={_fmt_gib(peak_vram)} "
          f"peak_rss={_fmt_gib(sampler.peak_rss)} GiB | код {result_code}", flush=True)
    return result_code


# --------------------------------------------------------------------------- #
# `bench io`: плечи конвейера (NVMe -> RAM, RAM -> VRAM)
# --------------------------------------------------------------------------- #

def _read_pass(weights: Any, names: list[str], target_bytes: int) -> tuple[int, float]:
    """Один проход чтения: тензоры в порядке файла, пока не набрано `target_bytes`.

    `ShardedWeights.__getitem__` — это mmap + cast/clone (ровно то, что делает реальный стриминг),
    поэтому замер включает и приведение dtype, а не только чтение страниц. Тензор сразу
    отпускается: RAM держит один тензор, page cache растёт сам собой (сбросить его средствами
    Python на Windows нельзя — поэтому `run_io` печатает оба прохода).
    """
    started = time.perf_counter()
    done = 0
    for name in names:
        tensor = weights[name]
        done += tensor.numel() * tensor.element_size()
        del tensor
        if done >= target_bytes:
            break
    return done, time.perf_counter() - started


def _h2d_transfer(host: Any, gpu: Any, chunk: int) -> float:
    """Скопировать `host` в `gpu` блоками `chunk` -> секунды по `torch.cuda.Event`.

    Именно `Event`, а не `time.time()`: H2D идёт асинхронно, а выдача ядер возвращается сразу;
    `end.synchronize()` гарантирует, что замер — всё копирование, а не только постановка в очередь.
    """
    import torch

    total = int(host.numel())
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for offset in range(0, total, chunk):
        size = min(chunk, total - offset)
        gpu.narrow(0, 0, size).copy_(host.narrow(0, offset, size), non_blocking=True)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0


def _h2d_mode(total: int, chunk: int, *, pin_memory: bool) -> float:
    """Один режим H2D (`pinned`/`pageable`): warmup + замер -> секунды.

    Буферы освобождаются сразу после замера: pinned 0.5 GiB и pageable 0.5 GiB не должны
    одновременно висеть в RAM (а GPU держит только один блок `chunk`).
    """
    import torch

    kwargs: dict[str, Any] = {"pin_memory": True} if pin_memory else {}
    host = torch.empty(total, dtype=torch.uint8, **kwargs)
    gpu = torch.empty(min(chunk, total), dtype=torch.uint8, device="cuda")
    try:
        host.fill_(1)                          # тач всех страниц — в замере только копия
        _h2d_transfer(host.narrow(0, 0, min(chunk, total)), gpu, min(chunk, total))
        return _h2d_transfer(host, gpu, chunk)
    finally:
        del host, gpu
        gc.collect()
        torch.cuda.empty_cache()


def _run_h2d(total_bytes: int, chunk: int) -> int:
    """Pinned vs pageable H2D: печатает `h2d_pinned_GBs`/`h2d_pageable_GBs` и выигрыш."""
    import torch

    if not torch.cuda.is_available():
        print("h2d: нет CUDA — плечо RAM -> VRAM не измеряется (NVMe-часть выше уже снята)", flush=True)
        return 0
    if total_bytes <= 0:
        print("[io] h2d: пропущено (--h2d-gib <= 0)", flush=True)
        return 0
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:                              # noqa: BLE001 — диагностика не обязательна
        name = "cuda"
    gib = total_bytes / GIB
    print(f"[h2d] device={name} | объём {gib:.2f} GiB | блок {chunk / 1024 ** 2:.0f} MiB | "
          f"замер torch.cuda.Event | non_blocking=True для обоих режимов", flush=True)
    pinned_s = _h2d_mode(total_bytes, chunk, pin_memory=True)
    pageable_s = _h2d_mode(total_bytes, chunk, pin_memory=False)
    pinned = _gbps(total_bytes, pinned_s)
    pageable = _gbps(total_bytes, pageable_s)
    speedup = pageable_s / pinned_s if pinned_s > 0 else 0.0
    print(f"[h2d] pinned: {gib:.2f} GiB за {pinned_s:.3f} s -> h2d_pinned_GBs={pinned:.2f} GB/s",
          flush=True)
    print(f"[h2d] pageable: {gib:.2f} GiB за {pageable_s:.3f} s -> "
          f"h2d_pageable_GBs={pageable:.2f} GB/s", flush=True)
    print(f"[h2d] h2d_pinned_GBs={pinned:.2f} h2d_pageable_GBs={pageable:.2f} "
          f"выигрыш pinned/pageable={speedup:.2f}x", flush=True)
    return 0


def run_io(args: argparse.Namespace) -> int:
    """Микробенчмарки плеч: NVMe -> RAM (два прохода) и RAM -> VRAM (pinned vs pageable).

    Своей логики загрузки весов здесь нет: файл открывает `streaming.ShardedWeights` (mmap),
    а копию на GPU делает torch. Нет CUDA — NVMe-часть всё равно измеряется, `h2d` печатает
    "нет CUDA" и команда завершается кодом 0.
    """
    target = str(getattr(args, "local_dir", None) or getattr(args, "repo", None) or "")
    subfolder = getattr(args, "subfolder", None)
    if not target:
        target = DEFAULT_IO_REPO
        subfolder = subfolder or DEFAULT_IO_SUBFOLDER
    subfolder = _resolve_subfolder(getattr(args, "local_dir", None), subfolder)

    read_gib = float(args.read_gib)
    h2d_gib = float(args.h2d_gib)
    chunk = int(args.chunk_mib) * 1024 ** 2
    if read_gib <= 0 and h2d_gib <= 0:
        print("[bench] нечего мерить: --read-gib и --h2d-gib оба <= 0", file=sys.stderr, flush=True)
        return 2

    try:
        from .. import streaming

        source = streaming.resolve_stream_source(target, subfolder=subfolder,
                                                local_files_only=True)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"[bench] {exc}", file=sys.stderr, flush=True)
        return 2

    files_gib = sum(p.stat().st_size for p in source.shards) / GIB
    print(f"[io] веса {target}{'/' + str(subfolder) if subfolder else ''} | {source.summary()} "
          f"| шардов {len(source.shards)}, {files_gib:.2f} GiB | h2d блок {chunk / 1024 ** 2:.0f} MiB",
          flush=True)

    if read_gib > 0:
        target_bytes = int(read_gib * GIB)
        weights = streaming.ShardedWeights(source)
        try:
            names = list(weights)
            cold_bytes, cold_s = _read_pass(weights, names, target_bytes)
            warm_bytes, warm_s = _read_pass(weights, names, target_bytes)
        finally:
            weights.close()                    # mmap-шарды закрываются; файлы не трогаем
        cold = _gbps(cold_bytes, cold_s)
        warm = _gbps(warm_bytes, warm_s)
        print(f"[nvme] проход 1 (вероятно холодный): {cold_bytes / GIB:.2f} GiB за {cold_s:.2f} s "
              f"-> nvme_cold_GBs={cold:.2f} GB/s", flush=True)
        print(f"[nvme] проход 2 (тёплый, page cache): {warm_bytes / GIB:.2f} GiB за {warm_s:.2f} s "
              f"-> nvme_warm_GBs={warm:.2f} GB/s", flush=True)
        print(f"[io] nvme_cold_GBs={cold:.2f} nvme_warm_GBs={warm:.2f} "
              f"({len(names)} тензоров в наборе; объём — целыми тензорами, поэтому последний "
              f"может перевалить за --read-gib; GB/s десятичные) | честно: первый проход может "
              f"быть уже тёплым — page cache ОС здесь не сбрасывается, поэтому печатаются оба "
              f"числа", flush=True)
    else:
        print("[io] nvme: пропущено (--read-gib <= 0)", flush=True)

    return _run_h2d(int(h2d_gib * GIB), chunk)


# --------------------------------------------------------------------------- #
# CLI-контракт подкоманды
# --------------------------------------------------------------------------- #

def add_args(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="bench_action", required=True, metavar="<action>")
    vram = actions.add_parser(
        "vram",
        help="пик VRAM/RSS по фазам на реальном прогоне (эмбеддинги только из кэша)",
        description="Пайплайн грузится тем же backend.prepare(), что и обычный run; "
                    "эмбеддинги берутся из cache/embeds/ (CPU-энкод не платится).",
        prog=f"{parser.prog} vram",
    )
    vram.add_argument("--input", required=True, metavar="PATH",
                      help="входной кадр (по его sha256 ищется кэш эмбеддингов)")
    vram.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION, choices=RESOLUTIONS,
                      metavar="N",
                      help=f"resolution профиля: 640=480p, 1024=768p (по умолчанию {DEFAULT_RESOLUTION}); "
                           f"960p+ только с --offload group")
    vram.add_argument("--steps", type=int, default=None, metavar="N",
                      help=f"шаги; по умолчанию {PREVIEW_STEPS} при 640, иначе {DEFAULT_STEPS}")
    vram.add_argument("--seed", type=int, default=DEFAULT_SEED, metavar="N",
                      help=f"seed (по умолчанию {DEFAULT_SEED})")
    vram.add_argument("--lora-scale", type=float, default=DEFAULT_LORA_SCALE, metavar="F",
                      help=f"сила LoRA (по умолчанию {DEFAULT_LORA_SCALE})")
    vram.add_argument("--vae-dtype", dest="vae_dtype", choices=("keep", "bf16"), default="keep",
                      help="keep = как в run (VAE во fp32), bf16 = декод в bf16")
    vram.add_argument("--tag", default="", metavar="NAME",
                      help="суффикс имени json/png (например maxsplit)")
    vram.add_argument("--save", type=int, default=0, choices=(0, 1), metavar="0|1",
                      help="1 = сохранить кадр в out/measure-*.png")
    vram.add_argument("--config", default=None, metavar="PRESET",
                      help=f"пресет ({', '.join(available_presets())}); по умолчанию {DEFAULT_CONFIG}")
    vram.add_argument("--model", default=None, choices=("base", "turbo"), metavar="PROFILE",
                      help="профиль генерации (base/turbo); синоним --config")
    vram.add_argument("--set", dest="sets", action="append", default=[], metavar="KEY=VALUE",
                      help="точечный параметр RunConfig (повторяемый)")
    vram.add_argument("--offload", choices=("none", "group"), default=None,
                      help="group = блоки трансформера стримятся с CPU (нужно для 960p+)")
    vram.set_defaults(_bench_action="vram")

    encoder = actions.add_parser(
        "encoder",
        help="диск-стриминг текст-энкодера: время фаз, rss/uss, hidden states, диффы (--compare)",
        description="Грузит энкодер только через streaming.load_text_encoder: --stream 0 — "
                    "обычное from_pretrained, --stream 1 — meta-параметры + подгрузка весов "
                    "модуля с диска. Пик памяти снимает фоновый сэмплер (rss/uss); главная "
                    "метрика — uss (страницы mmap в неё не попадают).",
        prog=f"{parser.prog} encoder",
    )
    encoder.add_argument("--repo", default=None, metavar="ID",
                      help="id репозитория Хаба (только локальный hf-cache, без сети)")
    encoder.add_argument("--local-dir", dest="local_dir", default=None, metavar="PATH",
                      help="локальный каталог с config.json + *.safetensors")
    encoder.add_argument("--subfolder", default="text_encoder", metavar="NAME",
                      help="подкаталог репозитория с энкодером (по умолчанию text_encoder)")
    encoder.add_argument("--dtype", default="bfloat16", metavar="DTYPE",
                      help="dtype весов (по умолчанию bfloat16; bf16 — как грузит проект)")
    encoder.add_argument("--stream", type=int, choices=(0, 1), default=0, metavar="0|1",
                      help="1 = веса читаются по слоям с диска (encoder.stream), 0 = обычная загрузка")
    encoder.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                      help="где живут веса при --stream 1: cpu (RAM, умолчание) или cuda (VRAM, только "
                           "текущий модуль); с --prefetch включается конвейер")
    encoder.add_argument("--prefetch", type=int, default=0, metavar="N",
                      help="глубина конвейера (слоёв в очереди RAM); 0 = синхронный путь "
                           "(по умолчанию), >0 — только с --stream 1 --device cuda")
    encoder.add_argument("--no-pin", dest="no_pin", action="store_true",
                      help="не пиннить staging-буферы RAM (вклад pin_memory в конвейере)")
    encoder.add_argument("--lookahead", type=int, default=1, metavar="N",
                      help="сколько слоёв вперёд выдавать H2D-копии (>=1, по умолчанию 1)")
    encoder.add_argument("--tokens", type=int, default=DEFAULT_TOKENS, metavar="N",
                      help=f"длина input_ids: trunc/pad до N (по умолчанию {DEFAULT_TOKENS})")
    encoder.add_argument("--prompt", default=DEFAULT_PROMPT, metavar="TEXT",
                      help=f"промт для forward (по умолчанию {DEFAULT_PROMPT!r})")
    encoder.add_argument("--out-layers", dest="out_layers", default=DEFAULT_OUT_LAYERS,
                      metavar="N,N,N",
                      help=f"какие hidden states печатать/сравнивать (по умолчанию {DEFAULT_OUT_LAYERS})")
    encoder.add_argument("--save", default=None, metavar="PATH.pt",
                      help="сохранить {'hidden': {layer: tensor}, 'input_ids': tensor} в .pt")
    encoder.add_argument("--compare", default=None, metavar="PATH.pt",
                      help="сверить с сохранённым .pt: [diff] layer=N max=…; max>0 -> код 1")
    encoder.set_defaults(_bench_action="encoder")

    io = actions.add_parser(
        "io",
        help="микробенчмарки плеч: NVMe -> RAM (два прохода) и RAM -> VRAM (pinned vs pageable)",
        description="Читает N GiB весов с диска через streaming.ShardedWeights (mmap) два прохода "
                    "подряд (первый — вероятно холодный, второй — тёплый page cache) и копирует "
                    "M GiB на GPU блоками --chunk-mib, замеряя pinned/pageable по "
                    "torch.cuda.Event. Без CUDA NVMe-часть всё равно измеряется, а h2d "
                    "печатает «нет CUDA»; код возврата 0.",
        prog=f"{parser.prog} io",
    )
    io.add_argument("--read-gib", dest="read_gib", type=float, default=DEFAULT_READ_GIB, metavar="N",
                    help=f"сколько GiB читать с диска (0 = пропустить; по умолчанию "
                         f"{DEFAULT_READ_GIB:g})")
    io.add_argument("--h2d-gib", dest="h2d_gib", type=float, default=DEFAULT_H2D_GIB, metavar="N",
                    help=f"сколько GiB копировать на GPU (0 = пропустить; по умолчанию "
                         f"{DEFAULT_H2D_GIB:g})")
    io.add_argument("--chunk-mib", dest="chunk_mib", type=int, default=DEFAULT_CHUNK_MIB,
                    choices=H2D_CHUNKS_MIB, metavar="64|128|256",
                    help=f"размер блока H2D-копии в МиБ (по умолчанию {DEFAULT_CHUNK_MIB})")
    io.add_argument("--repo", default=None, metavar="ID",
                    help=f"id репозитория Хаба (только локальный hf-cache; по умолчанию "
                         f"{DEFAULT_IO_REPO})")
    io.add_argument("--local-dir", dest="local_dir", default=None, metavar="PATH",
                    help="локальный каталог с config.json + *.safetensors")
    io.add_argument("--subfolder", default=None, metavar="NAME",
                    help=f"подкаталог с энкодером (для репозитория по умолчанию — "
                         f"{DEFAULT_IO_SUBFOLDER}; для явного --repo/--local-dir — без подкаталога)")
    io.set_defaults(_bench_action="io")


def run(args: argparse.Namespace) -> int:
    action = getattr(args, "_bench_action", None) or getattr(args, "bench_action", None)
    if action == "encoder":
        return run_encoder(args)
    if action == "io":
        return run_io(args)
    if action != "vram":                           # argparse с required=True сюда не пустит
        print(f"[bench] неизвестное действие {action!r}")
        return 2
    return run_vram(args)


__all__ = ["DEFAULT_CHUNK_MIB", "DEFAULT_CONFIG", "DEFAULT_H2D_GIB", "DEFAULT_IO_REPO",
           "DEFAULT_IO_SUBFOLDER", "DEFAULT_OUT_LAYERS", "DEFAULT_PROMPT", "DEFAULT_READ_GIB",
           "DEFAULT_RESOLUTION", "DEFAULT_TOKENS", "H2D_CHUNKS_MIB", "HELP", "NAME",
           "RESOLUTIONS", "_MemSampler", "add_args", "build_config", "find_embeds", "run",
           "run_encoder", "run_io", "run_vram", "snapshot"]
