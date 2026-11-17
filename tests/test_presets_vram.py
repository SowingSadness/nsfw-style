"""Пресеты названы по VRAM: один словарь имён в `qwenstyle/profiles.py`, пресетах и CLI.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_presets_vram.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем: имя пресета — обещание тарифа. `VRAM8-*` значит «влезает в 8 ГБ VRAM», `VRAM12-*` — «в 12»,
`VRAM16-*` — «в 16», и это обещание держится тремя полями, а не комментарием: `offload: group`
у черновиков `VRAM12` (без него 768p даёт 15.5 ГиБ — то есть 16 ГБ, а не 12), `kv_cache: offload`
у тарифов 12/16 (KV префикса — в RAM) и `upscale: null` у `-release` (кадр отдаётся native, а не
добирается ESRGAN). Тест фиксирует раскладку **без весов, без CUDA и без сети**:

1. **один словарь**: каждый файл `qwenstyle/presets/*.yaml` — либо тариф (ровно в одном профиле
   `PROFILES`), либо известный dev-пресет (`base`/`turbo`/`h3-image`/`max-h3`) вне профилей; обратное тоже
   верно — ни одного имени профиля без файла, ни одного осиротевшего файла;
2. **имена семейств**: только `VRAM8`/`VRAM12`/`VRAM16` у тарифов (никаких `weak-*`/`strong-*`/
   `flagship-*`), у суффиксов — известный набор;
3. **тариф по KV**: `VRAM12-*`/`VRAM16-*` -> `kv_cache: offload`, `VRAM8-*` -> дефолт
   `auto-offload-ref2` (там KV мал: 480/640p);
4. **апскейл**: у `VRAM12-release*`/`VRAM16-release*` — `upscale: null` (`target_height is None`,
   `factor == 0`), у `VRAM8-release*` — 1080 (640p + ESRGAN), у `VRAM16-4k-fast`/`VRAM16-r3-*` — 2160;
5. **черновик 12 ГБ**: у `VRAM12-preview*` — `offload: group`;
6. **`--upscale=null`**: `parse_upscale` (`null|none|off` -> `None`, конечное число >= 0 ->
   `float`, а пустая строка/`nan`/`inf`/отрицательное/мусор -> `ValueError`) и превращение `None`
   в пустой блок `upscale` (апскейл выключен целиком);
7. **обратное включение**: `--set upscale.target_height=1440` на `VRAM12-release` снова даёт
   апскейл (960 -> 1440, фактор 1.5) — приоритет высоты над `factor` не поехал;
8. **регистр в `--preset`**: `vram12` -> `VRAM12-release` (та же функция `resolve_preset`, что
   у `gen`), путь к YAML возвращается как есть, два имени с разным регистром — `ConfigError`.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                   # тест не должен оставлять __pycache__ в пакете

from qwenstyle.cmd._args import parse_upscale      # noqa: E402 — после правки sys.path
from qwenstyle.cmd.gen import casefold_index, resolve_preset   # noqa: E402 — та же функция, что у gen
from qwenstyle.config import ConfigError, PRESETS_DIR, RunConfig, available_presets   # noqa: E402
from qwenstyle.profiles import PROFILE_ORDER, PROFILES                  # noqa: E402

#: Тарифы железа: имя файла = `<семейство>-<вариант>`.
FAMILIES = ("VRAM8", "VRAM12", "VRAM16")
#: Варианты внутри семейства (`-release`, `-preview-anime`, `-4k-fast`, `-r3-169`, …).
SUFFIXES = ("preview", "release", "preview-anime", "release-anime", "anime",
            "large", "4k-fast", "4k-slide", "r3-169", "r3-43")
#: Пресеты-не-тарифы: конкретная модель/база, а не «сколько VRAM». В профили не входят.
NON_TARIFF = ("base", "turbo", "h3-image", "max-h3")
#: KV-кеш: у тарифов 12/16 префикс живёт в RAM, у 8 — по числу референсов (дефолт конфига).
KV_TARIFF = "offload"
KV_SMALL = "auto-offload-ref2"


# --------------------------------------------------------------------------- #
# Вспомогательное
# --------------------------------------------------------------------------- #

def preset_files() -> list[str]:
    """Имена всех пресетов на диске (`qwenstyle/presets/*.yaml`), без расширения."""
    return sorted(path.stem for path in Path(PRESETS_DIR).glob("*.yaml"))


def profile_of(name: str) -> list[str]:
    """Профили, в которые входит пресет (для тарифа — ровно один, для dev-пресета — ни одного)."""
    return [profile_name for profile_name in PROFILE_ORDER
            if name in PROFILES[profile_name].presets]


def _family(name: str) -> str:
    """Семейство имени: `VRAM12-release` -> `VRAM12`; `base` -> `base` (без дефиса)."""
    return name.split("-", 1)[0]


# --------------------------------------------------------------------------- #
# Кейсы
# --------------------------------------------------------------------------- #

def case_single_dictionary() -> None:
    """1. Каждый файл — ровно в одном профиле, и ни одного имени профиля без файла."""
    files = preset_files()
    if len(files) != 21:
        raise AssertionError(f"в qwenstyle/presets {len(files)} файлов, ожидалось 21: {files}")
    registered = set(available_presets())
    unknown = [name for name in files if name not in registered]
    if unknown:
        raise AssertionError(f"файлы не резолвятся как пресеты: {unknown}")
    for name in files:
        owners = profile_of(name)
        if name in NON_TARIFF:
            if owners:
                raise AssertionError(f"{name} — не тариф, а попал в профили {owners}")
            continue
        if len(owners) != 1:
            raise AssertionError(f"{name}: профилей {owners or 'ни одного'}, ожидался ровно один")
    # Обратное направление: ни одного обещанного имени без файла (иначе `--preset` падает).
    for profile_name in PROFILE_ORDER:
        for preset in PROFILES[profile_name].presets:
            if preset not in files:
                raise AssertionError(f"{preset} объявлен в профиле {profile_name}, "
                                     f"но файла qwenstyle/presets/{preset}.yaml нет")
    total = sum(len(PROFILES[profile_name].presets) for profile_name in PROFILE_ORDER)
    if total != len(files) - len(NON_TARIFF):
        raise AssertionError(f"тарифов в профилях {total}, файлов вне dev-пресетов "
                             f"{len(files) - len(NON_TARIFF)} — расхождение словаря")
    print(f"  ok: {len(files)} файлов = {total} тарифов в 3 профилях + {len(NON_TARIFF)} dev-пресета")


def case_family_names() -> None:
    """2. Семейства имён: только `VRAM8|VRAM12|VRAM16` (+ dev-пресеты вне тарифов)."""
    files = preset_files()
    for name in files:
        family = _family(name)
        if name in NON_TARIFF:
            continue
        if family not in FAMILIES:
            raise AssertionError(f"{name}: семейство {family!r} вне {FAMILIES} "
                                 f"(старые имена weak/strong/flagship не поддерживаются)")
        suffix = name[len(family) + 1:] if name != family else ""
        if suffix not in SUFFIXES:
            raise AssertionError(f"{name}: суффикс {suffix!r} вне известных {SUFFIXES}")
    legacy = [name for name in files if _family(name) in ("weak", "strong", "flagship")]
    if legacy:
        raise AssertionError(f"в наборе остались старые имена: {legacy}")
    if tuple(FAMILIES) != tuple(PROFILE_ORDER):
        raise AssertionError(f"PROFILE_ORDER={PROFILE_ORDER}, семейства {FAMILIES} — "
                             f"шорткат `--preset VRAM12` строится из PROFILE_ORDER")
    print(f"  ok: семейства {', '.join(FAMILIES)}; dev-пресеты вне тарифов: "
          f"{', '.join(NON_TARIFF)}")


def case_kv_cache_tariff() -> None:
    """3. KV-кеш: у тарифов 12/16 — `offload` (в RAM), у 8 — дефолт по числу референсов."""
    for name in preset_files():
        if name in NON_TARIFF:
            continue
        family = _family(name)
        cfg = RunConfig.from_profile(name)
        want = KV_SMALL if family == "VRAM8" else KV_TARIFF
        got = str(cfg.generation.kv_cache)
        if got != want:
            raise AssertionError(f"{name}: kv_cache={got!r}, ожидалось {want!r} "
                                 f"(иначе тариф врёт про VRAM)")
        if family in ("VRAM12", "VRAM16") and "kv_cache" not in Path(
                Path(PRESETS_DIR) / f"{name}.yaml").read_text(encoding="utf-8"):
            raise AssertionError(f"{name}: kv_cache пришёл из дефолта, а не из пресета тарифа")
    print(f"  ok: VRAM12/VRAM16 -> kv_cache={KV_TARIFF}, VRAM8 -> {KV_SMALL}")


def case_upscale_native() -> None:
    """4. `upscale: null` у `-release` VRAM12/VRAM16; 1080 у VRAM8; 2160 у 4k-fast/r3-*."""
    for name in preset_files():
        if name in NON_TARIFF:
            continue
        cfg = RunConfig.from_profile(name)
        target = cfg.upscale.target_height
        factor = float(cfg.upscale.factor or 0.0)
        if name.startswith("VRAM12-release") or name.startswith("VRAM16-release"):
            if target is not None or factor != 0.0:
                raise AssertionError(f"{name}: target_height={target}, factor={factor} — "
                                     f"у release без апскейла должно быть пусто (upscale: null)")
            native = int(cfg.size_px or 0)
            if cfg.upscale_factor(native) != 0.0:
                raise AssertionError(f"{name}: upscale_factor({native}) != 0 — апскейл не выключен")
            continue
        expected = 1080 if name.startswith("VRAM8-release") else (
            2160 if name.startswith(("VRAM16-4k-fast", "VRAM16-r3-")) else 0)
        if int(target or 0) != expected:
            raise AssertionError(f"{name}: target_height={target}, ожидалось {expected or 'нет'}")
        if factor != 0.0:
            raise AssertionError(f"{name}: factor={factor} — множитель обязан считаться из высоты")
    # `--upscale=null` в CLI = `upscale: null` в конфиге: пустой блок выключает апскейл целиком,
    # даже если у пресета была цель (иначе `factor: 0` цель бы не отменил).
    cleared = RunConfig.from_profile("VRAM8-release").merged(upscale=None)
    if cleared.upscale.target_height is not None or cleared.upscale_factor(640) != 0.0:
        raise AssertionError(f"merged(upscale=None) не выключил апскейл: "
                             f"target_height={cleared.upscale.target_height}, "
                             f"factor={cleared.upscale_factor(640)}")
    print("  ok: VRAM12/VRAM16-release — native (upscale null), VRAM8-release 1080, "
          "4k-fast/r3-* 2160; upscale=None выключает целиком")


def case_offload_preview_12() -> None:
    """5. `VRAM12-preview*` — `offload: group`: без него 768p даёт 15.5 ГиБ, то есть 16 ГБ."""
    previews = [name for name in preset_files()
                if name == "VRAM12-preview" or name.startswith("VRAM12-preview-")]
    if len(previews) != 2:
        raise AssertionError(f"черновиков VRAM12 ожидалось 2, найдено {previews}")
    for name in previews:
        cfg = RunConfig.from_profile(name)
        if cfg.generation.offload != "group":
            raise AssertionError(f"{name}: offload={cfg.generation.offload!r}, "
                                 f"ожидалось 'group' (иначе пик 15.5 ГиБ > 12 ГБ)")
        if int(cfg.size_px or 0) != 768:
            raise AssertionError(f"{name}: size_px={cfg.size_px}, ожидалось 768")
    print(f"  ok: {', '.join(previews)} -> offload=group, 768p")


def case_parse_upscale() -> None:
    """6. `parse_upscale`: `null|none|off` -> `None`, число >= 0 -> float, всё остальное — `ValueError`."""
    # `None` = «флаг не задан» (вызывающий код проверяет `args.upscale is not None` до вызова).
    for text in ("null", "NULL", "none", "None", "off", "OFF", " null ", None):
        if parse_upscale(text) is not None:
            raise AssertionError(f"parse_upscale({text!r}) должен вернуть None (апскейл выключен)")
    # `0` — «фактор 0», допустим: целиком апскейл выключает только `null`.
    for text, want in (("2", 2.0), ("2.5", 2.5), (" 1.5 ", 1.5), ("0", 0.0), ("0.0", 0.0), ("4", 4.0)):
        got = parse_upscale(text)
        if got != want or not isinstance(got, float):
            raise AssertionError(f"parse_upscale({text!r}) = {got!r}, ожидалось {want}")
    # Пусто/пробелы, не-конечные и отрицательные — не множитель: молча выключать апскейл нельзя.
    for text in ("", "   ", "\t", "nan", "NaN", "inf", "-inf", "infinity", "-1", "-0.5",
                 "abc", "nullx", "--", "2,5x"):
        try:
            parse_upscale(text)
        except ValueError as exc:
            message = str(exc)
            for part in ("--upscale", "число > 0", "null|none|off"):
                if part not in message:
                    raise AssertionError(f"{text!r}: текст ошибки без {part!r}: {message}") from None
            continue
        raise AssertionError(f"parse_upscale({text!r}) не поднял ValueError")
    print("  ok: null|none|off -> None, число >= 0 -> float, пусто/nan/inf/-1/мусор -> ValueError")


def case_upscale_reenable() -> None:
    """7. `--set upscale.target_height=1440` на `VRAM12-release` снова включает апскейл (×1.5)."""
    cfg = RunConfig.from_profile("VRAM12-release").merged(**{"upscale.target_height": "1440"})
    native = int(cfg.size_px or 0)
    if native != 960:
        raise AssertionError(f"VRAM12-release: size_px={native}, ожидалось 960")
    factor = float(cfg.upscale_factor(native))
    if abs(factor - 1.5) > 1e-6:
        raise AssertionError(f"upscale_factor({native}) = {factor}, ожидалось 1.5 "
                             f"(target_height главнее factor)")
    if cfg.upscale.factor:
        raise AssertionError(f"upscale.factor={cfg.upscale.factor} — множитель не должен "
                             f"перебивать target_height")
    print(f"  ok: VRAM12-release + --set upscale.target_height=1440 -> {native} * {factor:g} = "
          f"{int(round(native * factor))}")


def case_preset_case_insensitive() -> None:
    """8. `--preset` не зависит от регистра; путь к YAML не трогается; неоднозначность — ошибка."""
    for raw, want in (("vram12", "VRAM12-release"), ("VRAM12", "VRAM12-release"),
                      ("Vram12", "VRAM12-release"), (" vram8-preview ", "VRAM8-preview"),
                      ("vram12-release", "VRAM12-release"),
                      ("vram16-4k-fast", "VRAM16-4k-fast"),
                      ("vram16-anime", "VRAM16-release-anime"),
                      ("base", "base"), ("Turbo", "turbo")):
        got = resolve_preset(raw)
        if got != want:
            raise AssertionError(f"resolve_preset({raw!r}) = {got!r}, ожидалось {want!r}")
    # Путь к файлу резолвит `config.preset_path`: регистр и текст не трогаем.
    for path in ("qwenstyle/presets/VRAM12-release.yaml",
                 "qwenstyle/presets/LowerCase.yaml"):
        if resolve_preset(path) != path:
            raise AssertionError(f"resolve_preset({path!r}) должен вернуть путь как есть")
    try:
        casefold_index(("VRAM12-release", "Vram12-release"), "тест")
    except ConfigError as exc:
        if "регистр" not in str(exc):
            raise AssertionError(f"текст ошибки без объяснения: {exc}") from None
    else:
        raise AssertionError("два имени, отличающиеся только регистром, должны давать ConfigError")
    print("  ok: vram12 -> VRAM12-release, путь к YAML как есть, неоднозначность -> ConfigError")


CASES = (
    ("1. один словарь: файлы <-> профили", case_single_dictionary),
    ("2. имена семейств VRAM8/VRAM12/VRAM16", case_family_names),
    ("3. kv_cache по тарифу", case_kv_cache_tariff),
    ("4. upscale: native у release, 1080/2160 у остальных", case_upscale_native),
    ("5. VRAM12-preview*: offload=group", case_offload_preview_12),
    ("6. parse_upscale", case_parse_upscale),
    ("7. апскейл включается обратно через target_height", case_upscale_reenable),
    ("8. --preset без учёта регистра", case_preset_case_insensitive),
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
