"""Параметры из адаптеров моделей: `parameters()` -> `--set`, `config show --explain-params`.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_parameters.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем: до шага 9 имена настраиваемых параметров жили в ядре, и `--set` неизвестного ключа молча
принимался (`sampler_kwargs` — свободный `dict[str, Any]`) и просто игнорировался бекендом: прогон
стоил минуты, а «настройка» не применялась. Теперь имена объявляют адаптеры
(`@register_generation`/`@register_upscale` + `parameters()` -> `ParameterSpec`), а ядро валидирует
по ним. Тест фиксирует контракт **без весов, GPU и сети**:

1. у всех публичных бекендов (`qwen_image21`, `minimax_h3_image`, `flux2`, `esrgan`) `parameters()`
   непустой, имена и пути уникальны, тип/scope/док заполнены;
2. числа согласованы: `default` попадает в `minimum..maximum`/`choices`, у enum есть варианты,
   `readonly`-спеки называются (H3 `steps=6` — расписание PDD, а не подсказка);
3. `config show --explain-params` строится из этих же спеков (таблица, пути `--set`, readonly);
4. валидация `--set`: неизвестный `generation.sampler_kwargs.*` отбивается `ConfigError`
   со списком доступных ключей модели; известный — проходит; бекенд без спеков — не ломается;
5. `readonly_params` ловит `--set` readonly-ключа (H3 `steps`) и молчит про обычные.
"""
from __future__ import annotations

import contextlib
import io
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                   # тест не должен оставлять __pycache__ в пакете

from qwenstyle.cmd import config as config_cmd    # noqa: E402 — после правки sys.path
from qwenstyle.config import (                    # noqa: E402
    ConfigError,
    RunConfig,
    backend_parameters,
    check_sampler_kwargs,
    readonly_params,
)

#: Публичные бекенды и их минимальный ожидаемый состав параметров (шаг 9 плана публикации).
EXPECTED = {
    ("generation", "qwen_image21"): 9,
    ("generation", "minimax_h3_image"): 10,
    ("generation", "flux2"): 12,
    ("upscale", "esrgan"): 5,
}
#: Допустимые значения полей спека (структура `ParameterSpec`).
TYPES = ("int", "float", "bool", "str", "enum")
SCOPES = ("generation", "sampler", "encoder", "upscale")
#: Пресеты для проверки таблицы/валидации (публичный минимум и H3 с readonly-шагами).
PRESET_QWEN = "strong-release"
PRESET_H3 = "h3-image"


def case_specs_present() -> None:
    """1. `parameters()` непустой у всех бекендов, структура и уникальность полей соблюдены."""
    for (kind, backend), minimum in EXPECTED.items():
        specs = backend_parameters(kind, backend)
        if not specs:
            raise AssertionError(f"{kind}.{backend}: parameters() пуст — ядру нечего валидировать")
        if len(specs) < minimum:
            raise AssertionError(f"{kind}.{backend}: параметров {len(specs)}, ожидалось "
                                 f"не меньше {minimum}")
        names = [spec.name for spec in specs]
        paths = [spec.path for spec in specs]
        if len(set(names)) != len(names):
            raise AssertionError(f"{kind}.{backend}: повтор имён {names}")
        if len(set(paths)) != len(paths):
            raise AssertionError(f"{kind}.{backend}: повтор путей --set {paths}")
        for spec in specs:
            if spec.type not in TYPES:
                raise AssertionError(f"{backend}.{spec.name}: тип {spec.type!r} вне {TYPES}")
            if spec.scope not in SCOPES:
                raise AssertionError(f"{backend}.{spec.name}: scope {spec.scope!r} вне {SCOPES}")
            if not str(spec.doc or "").strip():
                raise AssertionError(f"{backend}.{spec.name}: пустой doc (таблица --explain-params "
                                     f"останется без объяснения)")
            if not spec.path.startswith(f"{spec.scope}.") and spec.scope != "sampler":
                raise AssertionError(f"{backend}.{spec.name}: путь {spec.path!r} не начинается "
                                     f"с {spec.scope!r}")
            if spec.scope == "sampler" and spec.path != f"generation.sampler_kwargs.{spec.name}":
                raise AssertionError(f"{backend}.{spec.name}: путь {spec.path!r} не "
                                     f"generation.sampler_kwargs.{spec.name}")
    print(f"  ok: {len(EXPECTED)} бекендов объявляют "
          f"{sum(len(backend_parameters(*key)) for key in EXPECTED)} параметров, без повторов")


def case_values() -> None:
    """2. Числа согласованы: default в диапазоне/choices, enum не пустой, readonly объяснён."""
    for kind, backend in EXPECTED:
        for spec in backend_parameters(kind, backend):
            if spec.type == "enum":
                if not spec.choices:
                    raise AssertionError(f"{backend}.{spec.name}: enum без choices")
                if str(spec.default) not in tuple(str(choice) for choice in spec.choices):
                    raise AssertionError(f"{backend}.{spec.name}: default {spec.default!r} "
                                         f"не из {spec.choices}")
                continue
            if spec.type in ("int", "float") and isinstance(spec.default, (int, float)):
                if spec.minimum is not None and float(spec.default) < float(spec.minimum):
                    raise AssertionError(f"{backend}.{spec.name}: default {spec.default} "
                                         f"< минимума {spec.minimum}")
                if spec.maximum is not None and float(spec.default) > float(spec.maximum):
                    raise AssertionError(f"{backend}.{spec.name}: default {spec.default} "
                                         f"> максимума {spec.maximum}")
            if spec.minimum is not None and spec.maximum is not None:
                if float(spec.minimum) > float(spec.maximum):
                    raise AssertionError(f"{backend}.{spec.name}: диапазон перевёрнут "
                                         f"({spec.minimum} > {spec.maximum})")
            if spec.readonly:
                # readonly обязан сам объяснять, почему значение фиксировано
                note = spec.readonly_note()
                if not note or spec.format_default() == "-":
                    raise AssertionError(f"{backend}.{spec.name}: readonly без умолчания/пояснения")
    print("  ok: default/минимумы/choices согласованы, readonly-спеки с пояснением")


def case_h3_steps_readonly() -> None:
    """3. У MiniMax-H3 `steps=6` readonly: расписание PDD (`--steps` игнорируется)."""
    specs = {spec.name: spec for spec in backend_parameters("generation", "minimax_h3_image")}
    steps = specs.get("steps")
    if steps is None:
        raise AssertionError("у minimax_h3_image нет спека steps")
    if not steps.readonly:
        raise AssertionError("steps у H3 обязан быть readonly (6 шагов PDD)")
    if steps.default != 6 or steps.minimum != 6 or steps.maximum != 6:
        raise AssertionError(f"steps у H3: default={steps.default}, "
                             f"{steps.minimum}..{steps.maximum} — ожидалось 6/6/6")
    if "6" not in steps.readonly_note():
        raise AssertionError(f"readonly_note не называет 6 шагов: {steps.readonly_note()!r}")
    # LoRA у H3 не поддерживается (.caps), но параметр `lora` не объявляется как настраиваемый
    if "lora" in specs:
        raise AssertionError("H3 не поддерживает LoRA: спек lora не должен объявляться")
    # qwen, наоборот, шаги настраиваемые
    qwen_steps = {spec.name: spec for spec in backend_parameters("generation", "qwen_image21")}["steps"]
    if qwen_steps.readonly:
        raise AssertionError("steps у qwen_image21 не readonly")
    print(f"  ok: H3 steps=6 readonly ('{steps.readonly_note()}'), у qwen steps настраиваемые")


def case_explain_params() -> None:
    """4. `config show --explain-params` строится из этих спеков (чистая функция, без CLI)."""
    for preset in (PRESET_QWEN, PRESET_H3):
        cfg = RunConfig.from_profile(preset)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            config_cmd.explain_params(cfg)
        text = buffer.getvalue()
        for header in ("scope", "name", "type", "default", "range/choices", "readonly", "doc"):
            if header not in text:
                raise AssertionError(f"{preset}: в таблице нет колонки {header!r}: {text[:300]!r}")
        for kind, backend in (("generation", cfg.generation.backend),
                              ("upscale", cfg.upscale.backend)):
            specs = backend_parameters(kind, backend)
            for spec in specs:
                if spec.name not in text:
                    raise AssertionError(f"{preset}: в таблице нет параметра {spec.name!r}")
                if spec.path not in text:
                    raise AssertionError(f"{preset}: нет готового пути --set {spec.path!r}")
        readonly = [spec for kind, backend in (("generation", cfg.generation.backend),
                                               ("upscale", cfg.upscale.backend))
                    for spec in backend_parameters(kind, backend) if spec.readonly]
        if readonly and "fix" not in text.lower() and "readonly" not in text.lower():
            raise AssertionError(f"{preset}: таблица не помечает readonly: {text[-300:]!r}")
    print(f"  ok: таблица --explain-params содержит все спеки и пути --set ({PRESET_QWEN}, {PRESET_H3})")


def case_set_validation() -> None:
    """5. Неизвестный `--set` отбивается со списком ключей модели; известный проходит."""
    try:
        check_sampler_kwargs("qwen_image21", {"max_sequence_length": 512})
    except ConfigError as exc:
        text = str(exc)
        for token in ("qwen_image21", "max_sequence_length", "offload_blocks", "explain-params"):
            if token not in text:
                raise AssertionError(f"в ошибке `--set` нет {token!r}: {text}")
    else:
        raise AssertionError("max_sequence_length у qwen_image21 не отбит (ключа нет в спеке)")
    # ключи, которые бекенд читает, проходят молча
    for backend, kwargs in (("qwen_image21", {"offload_blocks": 4, "offload_stream": False}),
                            ("minimax_h3_image", {"offload_low_cpu_mem": True}),
                            ("flux2", {"max_sequence_length": 512})):
        check_sampler_kwargs(backend, kwargs)
    # ошибиться в имени бекенда нельзя: без спеков ключи принимаются как раньше
    check_sampler_kwargs("no_such_backend", {"whatever": 1})
    check_sampler_kwargs("qwen_image21", {})
    print("  ok: неизвестный ключ -> ConfigError со списком доступных; известные проходят")


def case_readonly_params() -> None:
    """6. `readonly_params` ловит `--set` readonly-ключа и молчит про обычные."""
    h3 = RunConfig.from_profile(PRESET_H3)
    specs = readonly_params(h3, ["generation.steps"])
    if not specs or specs[0].name != "steps":
        raise AssertionError(f"readonly_params(H3, steps) -> {specs}")
    if not specs[0].readonly:
        raise AssertionError("readonly_params вернул не readonly-спек")
    if readonly_params(h3, ["generation.offload", "generation.precision"]):
        raise AssertionError("readonly_params пометил настраиваемые ключи")
    if readonly_params(h3, []):
        raise AssertionError("readonly_params без путей обязан вернуть пустой список")
    qwen = RunConfig.from_profile(PRESET_QWEN)
    if readonly_params(qwen, ["generation.steps"]):
        raise AssertionError("у qwen steps настраиваемый — предупреждения быть не должно")
    if not readonly_params(qwen, ["generation.max_images"]):
        raise AssertionError("max_images фиксирован моделью (Capabilities) — должен ловиться")
    print("  ok: H3 steps ловится, qwen steps нет; max_images readonly у обеих моделей")


def case_unknown_backend() -> None:
    """7. Бекенд без `parameters()` не ломает ни таблицу, ни валидацию (обратная совместимость)."""
    if backend_parameters("generation", "no_such_backend"):
        raise AssertionError("незарегистрированный бекенд вернул спеки")
    if backend_parameters("upscale", "seedvr2") and not any(
            spec.name for spec in backend_parameters("upscale", "seedvr2")):
        raise AssertionError("пустые спеки seedvr2 вели себя неожиданно")
    try:
        backend_parameters("nope", "qwen_image21")
    except ConfigError:
        pass
    else:
        raise AssertionError("неизвестный вид бекенда не отбит")
    print("  ok: бекенд без спеков -> пустой кортеж, неизвестный вид -> ConfigError")


CASES = (
    ("1. parameters() у всех бекендов", case_specs_present),
    ("2. согласованность значений спеков", case_values),
    ("3. H3 steps=6 readonly", case_h3_steps_readonly),
    ("4. config show --explain-params", case_explain_params),
    ("5. валидация --set", case_set_validation),
    ("6. readonly_params", case_readonly_params),
    ("7. бекенд без parameters()", case_unknown_backend),
)


def main() -> int:
    failed = 0
    for label, case in CASES:
        try:
            case()
            print(f"PASS {label}", flush=True)
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {label}: {exc}", flush=True)
        except Exception:                                   # noqa: BLE001 — тест не должен врать
            failed += 1
            print(f"FAIL {label}: исключение\n{traceback.format_exc()}", flush=True)
    if failed:
        print(f"{failed} FAIL", flush=True)
        return 1
    print("ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
