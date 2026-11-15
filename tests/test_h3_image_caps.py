"""Скелет бекенда `minimax_h3_image`: регистрация, caps/budget, пресет h3-image, dry-run план.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_h3_image_caps.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем (план `minimax-h3-image`, шаг 2): бекенд входит адаптером (R6) — ядро знает его только по
имени. Тест фиксирует ровно тот контракт, от которого зависят ядро и CLI:

(а) `get_generation("minimax_h3_image")` зарегистрирован, `caps` совпадают с ожидаемыми
    (`conditioning=internal`, `image_inputs=multi`, `requires_image=False`, `fixed_steps=6`).
(б) `models list`-строка (`Discovered.to_dict()`/`Capabilities.summary()`) строится и содержит
    нужные поля.
(в) `RunConfig.from_profile("h3-image")` резолвится: `encoder.cache_tag`, `generation.lora is None`,
    fp8 + group offload + 6 шагов, `transformer` не унаследован от base.
(г) без `--input` preflight молчит (`pipeline.input_problem(...) is None`), `--aspect 1:1` при
    `size_px=1024` даёт crop 1024x1024, а ключ кэша содержит `_te-h3qwen32-heretic`.
(д) заглушек шага 4 в бекенде нет (`NotImplementedError` не встречается), методы реализованы,
    preflight энкодера падает до загрузки весов, а `probe` читает hf-cache без torch.
(ж) импорт модуля бекенда НЕ тянет torch (в подпроцессе), и `models list` видит бекенд через CLI.

Ничего не грузит: ни весов, ни сети, ни GPU; файлов не пишет (torch проверяется подпроцессом).
"""
from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                   # тест не должен оставлять __pycache__ в пакете

from qwenstyle import pipeline                    # noqa: E402 — после правки sys.path
from qwenstyle.config import RunConfig, prompt_md5  # noqa: E402
from qwenstyle.registry import RegistryError, discover, get_generation, get_encoder  # noqa: E402

BACKEND = "minimax_h3_image"
PRESET = "h3-image"
PROMPT = "fusal style. [TASK: TEXT_TO_IMAGE] test prompt for h3 caps"
#: Снапшот модели в hf-cache (probe читает его, если он есть; без него — «нет данных»).
H3_SNAPSHOT = (REPO_ROOT / "hf-cache" / "hub"
               / "models--MiniMaxAI--MiniMax-H3-Image" / "snapshots")

EXPECTED_CAPS = {
    "conditioning": "internal",
    "image_inputs": "multi",
    "requires_image": False,
    "encoder": "required",
    "fixed_steps": 6,
    "supports_lora": False,
    "precision": ("bf16", "fp8"),
    "offload": ("none", "group"),
}


# --------------------------------------------------------------------------- #
# Кейсы
# --------------------------------------------------------------------------- #

def case_registry_and_caps() -> None:
    """(а) бекенд зарегистрирован и объявляет ровно те возможности, на которые рассчитывает ядро."""
    cls = get_generation(BACKEND)
    assert cls.name == BACKEND, f"имя бекенда {cls.name!r}"
    caps = cls.caps
    for field_name, expected in EXPECTED_CAPS.items():
        actual = getattr(caps, field_name)
        assert actual == expected, f"caps.{field_name}: ожидалось {expected!r}, дано {actual!r}"
    assert caps.requires_image is False, "T2I обязан быть разрешён (requires_image=False)"
    assert caps.image_inputs == "multi", "edit принимает несколько референсов"
    assert cls.cache_stage == "text_embeds", f"этап кэша {cls.cache_stage!r}"
    assert "MiniMaxAI/MiniMax-H3-Image" in tuple(cls.repos), f"repos={cls.repos!r}"
    for method in ("prepare", "encode", "generate", "release", "probe"):
        assert callable(getattr(cls, method, None)), f"нет метода {method}"


def case_models_list_row() -> None:
    """(б) строка `models list` строится без torch и содержит conditioning/image_inputs/steps."""
    found = discover()
    assert BACKEND in found.generation, f"discover() не видит {BACKEND}: {sorted(found.generation)}"
    assert BACKEND not in found.errors, f"ошибка импорта модуля: {found.errors.get(BACKEND)}"
    payload = found.to_dict()["generation"][BACKEND]
    assert payload["caps"]["conditioning"] == "internal", payload["caps"]
    assert payload["caps"]["image_inputs"] == "multi", payload["caps"]
    assert payload["caps"]["fixed_steps"] == 6, payload["caps"]
    assert payload["budget"]["weights_bf16_gib"] > 0, payload["budget"]
    summary = found.generation[BACKEND].caps.summary()
    for token in ("conditioning=internal", "image_inputs=multi", "fixed_steps=6"):
        assert token in summary, f"в строке caps нет {token!r}: {summary}"
    budget_summary = found.generation[BACKEND].budget.summary()
    assert "fp8" in budget_summary, f"бюджет не объявляет fp8: {budget_summary}"


def case_preset() -> None:
    """(в) пресет h3-image резолвится: энкодер 32B-heretic, без LoRA, fp8 + group offload."""
    cfg = RunConfig.from_profile(PRESET)
    assert cfg.preset == PRESET, f"preset={cfg.preset!r}"
    assert cfg.generation.backend == BACKEND, cfg.generation.backend
    assert cfg.generation.repo == "MiniMaxAI/MiniMax-H3-Image", cfg.generation.repo
    assert cfg.generation.lora is None, f"LoRA не должна наследоваться от base: {cfg.generation.lora}"
    assert cfg.generation.transformer is None, \
        f"transformer не должен наследоваться от base: {cfg.generation.transformer}"
    assert cfg.generation.precision == "fp8", cfg.generation.precision
    assert cfg.generation.precision_cache == "cache/fp8_transformer_h3_image", \
        cfg.generation.precision_cache
    assert cfg.generation.offload == "group", cfg.generation.offload
    assert cfg.generation.sampler_kwargs.get("offload_blocks") == 1, cfg.generation.sampler_kwargs
    assert cfg.steps == 6, f"steps={cfg.steps}"
    assert cfg.size_px == 1024, f"size_px={cfg.size_px}"
    assert cfg.encoder.cache_tag == "h3qwen32-heretic", cfg.encoder.cache_tag
    assert cfg.encoder.name == "h3qwen32-heretic", cfg.encoder.name
    assert cfg.encoder.repo == "llmfan46/Qwen3-VL-32B-Instruct-ultra-uncensored-heretic", \
        cfg.encoder.repo
    assert cfg.encoder.local_dir is None, \
        f"local_dir не должен наследоваться от base (heretic-te 8B): {cfg.encoder.local_dir}"
    assert cfg.encoder.stream is True, "веса 62 ГиБ обязаны читаться стримингом"
    assert cfg.encoder.prefetch == 4, cfg.encoder.prefetch
    assert cfg.upscale.target_height is None, \
        f"апскейл до фиксированной высоты не наследуется: {cfg.upscale.target_height}"


def case_t2i_preflight_and_crop() -> None:
    """(г) без `--input` preflight молчит; `--aspect 1:1` -> crop 1024x1024 и тег энкодера в ключе."""
    cls = get_generation(BACKEND)
    cfg = RunConfig.from_profile(PRESET, prompt=PROMPT, input="", aspect="1:1")
    reason = pipeline.input_problem(cfg, cls.caps)
    assert reason is None, f"T2I без --input обязан быть валиден, а получено: {reason}"
    crop = cfg.crop(0, 0)
    assert crop == (1024, 1024), f"crop 1:1 при size_px=1024: ожидалось (1024, 1024), дано {crop}"
    key = cfg.embeds_key(crop, prompt_md5(PROMPT), "noref")
    assert "_te-h3qwen32-heretic" in key, f"в ключе кэша нет тега энкодера H3: {key}"
    assert "noref" in key, f"в ключе нет sentinel'а noref (записи T2I смешаются с edit): {key}"
    # 16:9 без входа тоже валиден: пропорция берётся из --aspect
    wide = RunConfig.from_profile(PRESET, prompt=PROMPT, input="", aspect="16:9")
    assert pipeline.input_problem(wide, cls.caps) is None, "16:9 без --input обязан быть валиден"
    width, height = wide.crop(0, 0)
    assert width > height and width % 16 == 0 and height % 16 == 0, \
        f"сетка 16:9 не кратна 16 или сплющена: {width}x{height}"


def case_implementation_and_probe() -> None:
    """(д) реализация на месте (не заглушки), preflight падает до весов, probe читает hf-cache.

    Веса не грузятся: `prepare` целиком не вызывается. Проверяется, что шаг 4 действительно
    реализован (`NotImplementedError` в файле нет, методы собирают модульный пайплайн), а
    отсутствующий энкодер (шаг 5) отсекается preflight'ом до каста 30.9 GiB.
    """
    cls = get_generation(BACKEND)
    backend = cls()
    cfg = RunConfig.from_profile(PRESET)

    module = importlib.import_module(cls.__module__)
    source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
    assert "NotImplementedError" not in source, \
        "в бекенде остались заглушки: реализация шага 4 не доделана"
    assert "MiniMaxH3ImageModularPipeline" in source, "бекенд не собирает модульный пайплайн"

    markers = {
        "prepare": ("_encoder_preflight", "_modular_pipeline", "_load_transformer"),
        "encode": ("encode_pipe", "prompt_embeds"),
        "generate": ("generator", "text_token_tags"),
        "release": ("gc.collect", "empty_cache"),
    }
    for name, tokens in markers.items():
        func = getattr(cls, name, None)
        assert callable(func), f"нет метода {name}"
        body = inspect.getsource(func)
        for token in tokens:
            assert token in body, f"{name}: в теле нет {token!r} — это заглушка, а не реализация"

    # preflight энкодера обязан стоять до загрузки снапшота/каста (иначе падение через 20 минут)
    prepare_body = inspect.getsource(cls.prepare)
    assert prepare_body.index("_encoder_preflight(cfg)") < prepare_body.index("_snapshot_root(cfg)"), \
        "preflight энкодера обязан вызываться до загрузки весов"
    try:
        get_encoder(cfg.encoder.name)
    except RegistryError:
        try:
            module._encoder_preflight(cfg)
        except RuntimeError as exc:
            text = str(exc)
            for token in (cfg.encoder.name, cfg.encoder.repo, cfg.encoder.cache_tag):
                assert token in text, f"в тексте preflight нет {token!r}: {text}"
            assert "шаг 5" in text, f"preflight не называет недостающий шаг: {text}"
        else:
            raise AssertionError(
                f"preflight обязан падать до весов: энкодер {cfg.encoder.name!r} не зарегистрирован"
            )
    else:
        print(f"    (энкодер {cfg.encoder.name} зарегистрирован — preflight проверен структурно)",
              flush=True)

    info = backend.probe(cfg)
    assert info["backend"] == BACKEND, info.get("backend")
    assert info["caps"]["requires_image"] is False, info["caps"]
    assert info["text_hidden_dim"] == 5120, info["text_hidden_dim"]
    assert "components" in info and "precision_cache" in info, sorted(info)
    assert Path(info["precision_cache"]["path"]) == Path("cache/fp8_transformer_h3_image"), \
        info["precision_cache"]
    has_snapshot = any(H3_SNAPSHOT.glob("*/transformer_diffusers")) if H3_SNAPSHOT.is_dir() else False
    if has_snapshot:
        assert info["components"]["transformer_diffusers"]["present"], info["components"]
        assert info["components"]["vae_diffusers"]["present"], info["components"]
        assert info["components"]["pdd_fused_heads.safetensors"]["present"], info["components"]
        assert info["pdd"]["steps"] == 6, info["pdd"]
        assert len(info["pdd"]["sigmas"]) == 7, info["pdd"]
        assert info["missing"] == [], f"не хватает компонентов: {info['missing']}"
    else:
        assert info["missing"], "без снапшота probe обязан перечислить недостающее"


def case_import_without_torch() -> None:
    """(ж) импорт модуля бекенда не тянет torch, а `models list` видит бекенд через CLI."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env.setdefault("HF_HUB_OFFLINE", "1")
    code = ("import sys; import qwenstyle.backends.minimax_h3_image as m; "
            "print('torch' in sys.modules, 'diffusers' in sys.modules, m.MiniMaxH3ImageBackend.name)")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT), env=env,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, f"импорт бекенда упал: {proc.stderr.strip()}"
    out = proc.stdout.strip()
    assert out.startswith("False False"), f"импорт тянет тяжёлые библиотеки: {out!r}"
    assert out.endswith(BACKEND), f"имя класса не совпало: {out!r}"

    cli = subprocess.run([sys.executable, "-m", "qwenstyle", "models", "list"],
                         cwd=str(REPO_ROOT), env=env,
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert cli.returncode == 0, f"models list exit {cli.returncode}: {cli.stderr.strip()}"
    rows = [line for line in cli.stdout.splitlines() if line.strip().startswith(BACKEND)]
    assert rows, f"в `models list` нет строки {BACKEND}:\n{cli.stdout}"
    row = rows[0]
    for token in ("internal", "multi", "6"):
        assert token in row, f"в строке {BACKEND} нет {token!r}: {row}"


CASES = (
    ("а: регистрация и caps", case_registry_and_caps),
    ("б: строка models list", case_models_list_row),
    ("в: пресет h3-image", case_preset),
    ("г: T2I preflight и crop", case_t2i_preflight_and_crop),
    ("д: реализация и probe", case_implementation_and_probe),
    ("ж: импорт без torch", case_import_without_torch),
)


def main() -> int:
    failed = 0
    for title, func in CASES:
        try:
            func()
        except Exception as exc:                    # noqa: BLE001 — тест печатает всё сам
            failed += 1
            print(f"FAIL {title}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
        else:
            print(f"PASS {title}", flush=True)
    if failed:
        print(f"FAIL: {failed} из {len(CASES)}", flush=True)
        return 1
    print("ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
