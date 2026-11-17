"""Манифесты профилей машин: `VRAM8` / `VRAM12` / `VRAM16` — данные и чистые функции.

Профиль отвечает на три вопроса без единого тензора: **какие репозитории** нужны на такой машине,
**что из этого качается** (`qwenstyle/cmd/models.py download --profile`) и **какие пресеты**
этой машине адресованы (`qwenstyle/presets/<имя>.yaml`, шаг 3).

Импортов тяжелее stdlib здесь нет и быть не должно: модуль читают `models download`, `selftest`,
`config show` и установочный скрипт (`bootstrap.*`), в том числе на машине, где torch ещё не стоит.

Раскладка на диске — **финальная** (после миграции кешей): веса энкодеров, которые не лежат в
`hf-cache`, качаются в `cache/models/weights/<имя>/` (`ProfileRepo.local_dir`). До миграции тот же
набор лежит в `cache/weights/<имя>/`, поэтому `local_dir_candidates()` отдаёт оба пути — проверка
«веса есть» работает и до, и после переезда.

Оценки размеров (`REPO_SIZE_GIB`) — из метаданных Хаба на 2026-11-15; они нужны только там, где
метаданные недоступны (оффлайн на чистой машине), а `Profile.disk_gib` — полный след профиля на
диске, включая производные кэши (`cache/models/fp8_transformer*`, ESRGAN). То есть «скачать» и
«занять на диске» — разные числа, и оба печатаются.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Профили в порядке возрастания железа: этот порядок печатают подсказки об ошибке.
PROFILE_ORDER: tuple[str, ...] = ("VRAM8", "VRAM12", "VRAM16")


@dataclass(frozen=True)
class ProfileRepo:
    """Репозиторий Хаба, нужный профилю.

    `include` — подкаталоги и/или glob-паттерны для `snapshot_download(allow_patterns=…)`
    (`transformer`, `fusal-*.safetensors`); пусто = весь репозиторий.
    `local_dir` — качать не в `hf-cache`, а в этот каталог (веса, которые наш загрузчик читает
    напрямую: энкодеры heretic-32B/8B).
    `gated` — репозиторий требует принятия лицензии на Хабе и `HF_TOKEN` (без них `--probe`
    и `download` вернут код 2 с подсказкой).
    """

    repo: str
    include: tuple[str, ...] = ()
    local_dir: str | None = None
    license: str = ""
    gated: bool = False
    note: str = ""


@dataclass(frozen=True)
class Profile:
    """Машина + её модели. `presets` — публичные пресеты этой машины (шаг 3)."""

    name: str                 # VRAM8 | VRAM12 | VRAM16
    hardware: str             # "8 GB VRAM / 16 GB RAM"
    repos: tuple[ProfileRepo, ...]
    presets: tuple[str, ...]
    disk_gib: float           # полный след на диске: загрузка + fp8-кэши + апскейлер
    #: Высота кадра, которую профиль даёт **из коробки**: у `-release` она либо native
    #: (`upscale: null`), либо добирается ESRGAN; 2160 — только отдельными пресетами
    #: (`VRAM16-4k-fast`, `VRAM16-r3-*`), поэтому в поле её нет.
    target_height: int        # 1080 | 960 | 1440


# --------------------------------------------------------------------------- #
# Репозитории
# --------------------------------------------------------------------------- #

#: Qwen-Image-2.1 целиком: transformer + text-encoder + VAE + tokenizer (≈30.9 ГиБ bf16).
BASE_REPO = ProfileRepo(
    "Qwen/Qwen-Image-2.1",
    license="Apache-2.0",
    note="базовая модель: transformer, text-encoder, VAE",
)
#: Из Turbo-репозитория нужен только transformer; энкодер и VAE — базовые (они байт-в-байт те же).
TURBO_REPO = ProfileRepo(
    "Qwen/Qwen-Image-2.1-Turbo",
    include=("transformer",),
    license="Apache-2.0",
    note="distill-трансформер (8 шагов), энкодер/VAE берутся из base",
)
#: FUSAL-LoRA: оба адаптера лежат в корне репозитория (`fusal-*.safetensors`, 0.39 + 0.63 ГиБ).
#: `include` намеренно не `*.safetensors`: тот же шаблон затянул бы дубль энкодера
#: `text_encoders/qwen3-vl-8b-heretic-1.3.0_bf16.safetensors` (+16.33 ГиБ), который у профиля
#: уже есть отдельным репозиторием (DreamFast) в `cache/models/weights/heretic-te`.
FUSAL_REPO = ProfileRepo(
    "MicksHF/Qwen-Image-2.1-Anime-Fusal",
    include=("fusal-*.safetensors",),
    license="other",
    note="адаптеры FUSAL R80 (base) и R128 (turbo); лицензия — в карточке репозитория",
)
#: heretic-энкодер 8B (в репо также gguf/comfyui-кванты — берём только 4 bf16-шарда + конфиги).
HERETIC_REPO = ProfileRepo(
    "DreamFast/Qwen3-VL-8B-Heretic-1.3.0",
    include=("model-*.safetensors", "*.index.json", "config.json", "generation_config.json"),
    local_dir="cache/models/weights/heretic-te",
    license="Apache-2.0",
    gated=True,
    note="текст-энкодер heretic 8B bf16; gated — нужен HF_TOKEN и принятие лицензии",
)
#: MiniMax-H3-Image: трансформер (bf16, диффузерная раскладка), VAE, процессор/токенизатор, PDD-головы.
H3_REPO = ProfileRepo(
    "MiniMaxAI/MiniMax-H3-Image",
    include=("transformer_diffusers", "vae_diffusers", "processor", "tokenizer", "scheduler",
             "pdd_fused_heads.safetensors", "pdd_config.json"),
    license="MiniMax H3 Community License",
    note="33.13B bf16, 6 шагов PDD; T2I и edit одним пайплайном",
)
#: heretic-энкодер 32B для H3-Image (62.1 ГиБ bf16). Наш загрузчик пересобирает веса в 3 шарда
#: <40 ГиБ (обход бага safetensors на Windows, D12) и читает их из `local_dir`, а не из `hf-cache`.
H3_TE_REPO = ProfileRepo(
    "llmfan46/Qwen3-VL-32B-Instruct-ultra-uncensored-heretic",
    local_dir="cache/models/weights/h3qwen32-heretic",
    license="Apache-2.0",
    note="32B-энкодер H3-Image; веса пересобираются в 3 шарда < 40 ГиБ (D12)",
)

#: Оценки размеров скачивания по репозиториям (ГиБ, метаданные Хаба 2026-11-15).
#: Нужны только когда метаданные недоступны (оффлайн/нет сети) — для таблиц и проверок.
REPO_SIZE_GIB: dict[str, float] = {
    BASE_REPO.repo: 30.86,
    TURBO_REPO.repo: 13.25,       # только transformer
    FUSAL_REPO.repo: 1.02,        # оба корневых адаптера (0.39 + 0.63)
    HERETIC_REPO.repo: 16.33,     # 4 шарда model-*
    H3_REPO.repo: 63.94,
    H3_TE_REPO.repo: 62.14,
}

#: Производные кэши, которые не качаются, но занимают диск (печатаются в плане профиля).
#: Заполняется по факту на машине; порядок — от общего к частному.
_MODELS_CACHE_GIB: dict[str, float] = {
    "base fp8-трансформер": 6.7,
    "turbo fp8-трансформер": 6.7,
    "h3-image fp8-трансформер": 30.9,
    "ESRGAN x4plus_anime_6B": 0.02,
}


def _models_caches(*names: str) -> float:
    return round(sum(_MODELS_CACHE_GIB[name] for name in names), 1)


# --------------------------------------------------------------------------- #
# Профили
# --------------------------------------------------------------------------- #

VRAM8 = Profile(
    name="VRAM8",
    hardware="8 GB VRAM / 16 GB RAM",
    repos=(BASE_REPO, TURBO_REPO, FUSAL_REPO, HERETIC_REPO),
    presets=("VRAM8-preview", "VRAM8-release", "VRAM8-preview-anime", "VRAM8-release-anime"),
    # загрузка 61.5 + два fp8-кэша 13.4 + ESRGAN
    disk_gib=round(30.86 + 13.25 + 1.02 + 16.33
                   + _models_caches("base fp8-трансформер", "turbo fp8-трансформер",
                                    "ESRGAN x4plus_anime_6B"), 1),
    target_height=1080,
)

VRAM12 = Profile(
    name="VRAM12",
    hardware="12 GB VRAM / 32 GB RAM",
    repos=(BASE_REPO, FUSAL_REPO, HERETIC_REPO),
    presets=("VRAM12-preview", "VRAM12-release", "VRAM12-preview-anime", "VRAM12-release-anime"),
    # загрузка 48.2 + fp8-кэш base 6.7 + ESRGAN
    disk_gib=round(30.86 + 1.02 + 16.33
                   + _models_caches("base fp8-трансформер", "ESRGAN x4plus_anime_6B"), 1),
    # release отдаёт native 960p (`upscale: null`); 1440 — это `--upscale 1.5`
    target_height=960,
)

VRAM16 = Profile(
    name="VRAM16",
    hardware="16 GB VRAM / 64 GB RAM",
    repos=(BASE_REPO, FUSAL_REPO, HERETIC_REPO, H3_REPO, H3_TE_REPO),
    presets=("VRAM16-preview", "VRAM16-preview-anime", "VRAM16-release",
             "VRAM16-release-anime", "VRAM16-large", "VRAM16-4k-fast",
             "VRAM16-r3-169", "VRAM16-r3-43", "4K-slide"),
    # VRAM12 + H3-Image 63.94 + 32B-энкодер 62.14 (+ fp8-кэш H3 строится на месте)
    disk_gib=round(30.86 + 1.02 + 16.33 + 63.94 + 62.14
                   + _models_caches("base fp8-трансформер", "ESRGAN x4plus_anime_6B"), 1),
    # release отдаёт native 1440p (`upscale: null`); 2160 — `VRAM16-4k-fast`/`VRAM16-r3-*`
    # или `--upscale 1.5` у release
    target_height=1440,
)

#: Все профили: `VRAM8` (8 ГБ VRAM) -> `VRAM12` (12 ГБ) -> `VRAM16` (16 ГБ + 64 ГБ RAM).
PROFILES: dict[str, Profile] = {item.name: item for item in (VRAM8, VRAM12, VRAM16)}

#: Поиск по имени без учёта регистра (`VRAM12`, `vram12` и `Vram12` — один профиль).
_PROFILE_KEYS: dict[str, Profile] = {name.lower(): item for name, item in PROFILES.items()}


# --------------------------------------------------------------------------- #
# Чистые функции
# --------------------------------------------------------------------------- #

def names() -> tuple[str, ...]:
    """Имена профилей в порядке возрастания железа (`VRAM8`, `VRAM12`, `VRAM16`)."""
    return PROFILE_ORDER


def profile(name: str) -> Profile:
    """Профиль по имени (регистр не важен); неизвестное имя — `ValueError` со списком доступных."""
    key = str(name or "").strip().lower()
    found = _PROFILE_KEYS.get(key)
    if found is None:
        raise ValueError(f"неизвестный профиль {name!r}; доступны: {', '.join(PROFILE_ORDER)}")
    return found


def disk_estimate(name: str) -> float:
    """Полный след профиля на диске (ГиБ): загрузка + производные кэши + апскейлер."""
    return float(profile(name).disk_gib)


def download_estimate(name: str) -> float:
    """Сколько качается по профилю (ГиБ): сумма `REPO_SIZE_GIB` его репозиториев.

    Расходится с `disk_estimate` на производные кэши (fp8-трансформеры строятся на месте из
    скачанных bf16-весов, но занимают диск) — поэтому в плане печатаются оба числа.
    """
    return round(sum(repo_size_gib(entry.repo) or 0.0 for entry in profile(name).repos), 1)


def repo_size_gib(repo: str) -> float | None:
    """Оценка размера скачивания репозитория (ГиБ) или `None`, если её нет в манифесте."""
    return REPO_SIZE_GIB.get(str(repo or "").strip())


def local_dir_candidates(local_dir: str | None) -> tuple[Path, ...]:
    """Где искать локальные веса: финальный путь и до-миграционный (`cache/models/x` -> `cache/x`).

    Миграция кешей (шаг 4) переносит `cache/weights/<имя>` в `cache/models/weights/<имя>`;
    проверка «веса уже есть» должна работать в обоих состояниях, иначе `selftest` и
    `download --probe --offline` врали бы на одной из версий раскладки.
    """
    if not local_dir:
        return ()
    path = Path(str(local_dir)).expanduser()
    candidates = [path]
    parts = path.parts
    for idx, part in enumerate(parts[:-1]):
        if part != "cache" or parts[idx + 1] != "models":
            continue
        legacy = Path(*parts[:idx + 1]) / Path(*parts[idx + 2:])
        if legacy != path:
            candidates.append(legacy)
        break
    return tuple(candidates)


__all__ = [
    "BASE_REPO", "FUSAL_REPO", "H3_REPO", "H3_TE_REPO", "HERETIC_REPO",
    "PROFILES", "PROFILE_ORDER", "REPO_SIZE_GIB", "TURBO_REPO", "VRAM8", "VRAM12", "VRAM16",
    "Profile", "ProfileRepo", "disk_estimate", "download_estimate", "local_dir_candidates",
    "names", "profile", "repo_size_gib",
]
