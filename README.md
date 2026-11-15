# Qwen-Image-2.1 + FUSAL Anime LoRA — style-transfer на 16 ГБ VRAM

Безопасная стилизация (composition-preserving img2img) в стилистике аниме **Prison School**
(яп. 監獄学園 〜プリズンスクール〜 / *Kangoku Gakuen ~Prison School~*, манга Акиры Хирамото, аниме J.C.Staff, 2015).
NSFW-генерация не выполняется и в промт не входит.

## Результат

`out/prison-school-style.png` — 1376×768, вход `A8850259-B9D1-42FD-B3D4-67A70661E9C8.png` (1672×941).
Композиция, позы, одежда и фон сохранены, изменён только рендер: жирный чёрный контур,
двухтоновые тени, приглушённая палитра. 25 шагов ≈ 37 с (после прогрева), полный прогон 357 с.

## Железо и раскладка памяти

Правила для агентов/разработчиков (кэширование параметров, обязательный CLI, проверенные
ограничения и цифры замеров) — в [`AGENTS.md`](AGENTS.md).

RTX 4080 16 ГБ, 63.6 ГБ RAM. Модель 33.7 ГБ bf16 целиком не влезает:

| Компонент | Размер | Где |
|---|---|---|
| transformer (7B DiT) | 14.2 ГБ bf16 → **~7 ГБ fp8** | GPU, `enable_layerwise_casting` |
| text_encoder (Qwen3VL-8B) | 17.5 ГБ bf16 | GPU стримингом (пик VRAM 1–3.5 ГиБ), только при промахе кэша |
| vae | 1.35 ГБ | GPU, `enable_slicing` + `enable_tiling` |
| LoRA R80 | 420 МБ bf16 | GPU |

Пик VRAM — 15.5 ГБ при разрешении 1024 (768p). Резолюция 1280 (960p) в 16 ГБ не влезает целиком:
нужен `--offload group` — замерено 66 с и пик 10.8 ГБ (см. `AGENTS.md` §4).

## Установка

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129
.venv/Scripts/python.exe -m pip install "transformers>=5.17" accelerate pillow safetensors peft
.venv/Scripts/python.exe -m pip install "git+https://github.com/huggingface/diffusers"
```

Важно: обычный `pip install torch` на Windows ставит **CPU-сборку** — нужен `--index-url`
на CUDA-индекс. `torchvision` обязателен: без него `transformers` падает на
`Qwen3VLVideoProcessor`. Веса ~34 ГБ качаются в `hf-cache/` в папке проекта
(`HF_HOME` в скрипте), embeddings кэшируются в `cache/`.

## Запуск

Точка входа — пакет `qwenstyle` (монолитный `run_style.py` удалён на шаге 11 переезда):

```bash
# черновик: 480p/640p, 8 шагов — ~11 с после прогрева кэшей (fp8-трансформер + эмбеддинги)
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m qwenstyle run --preview \
    --input out/camp/canyon-edit-640-lora1.0-s777.png --prompt-file promts/exp3.txt --out out/preview.png

# финал: 768p (--resolution 1024), 25 шагов — ~45 с, если кэш эмбеддингов уже есть
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m qwenstyle run --size 768p --steps 25 \
    --input out/camp/canyon-edit-640-lora1.0-s777.png --prompt-file promts/exp3.txt --out out/final.png

# turbo: distill-трансформер, 8 шагов (фиксированное расписание) — 21 с на 1024
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m qwenstyle run --model turbo --resolution 1024 \
    --input out/camp/canyon-edit-640-lora1.0-s777.png --prompt-file promts/exp3.txt --out out/turbo.png

# план без загрузки весов: сетка, ключ кэша, нужен ли энкод, шаги, апскейл
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m qwenstyle run --dry-run --config preview \
    --steps 8 --input out/camp/canyon-edit-640-lora1.0-s777.png --prompt-file promts/exp3.txt --out out/p.png
```

Финальный кадр (`full-qwen`): **1080p** (1440×1088, 6120 токенов) — **2.4 s/шаг** (плато 2.41–2.43,
4.1 s/шаг — среднее по 2 шагам диагностики), 25 шагов **66–67 s**, `[done] total 75 s` без поста;
**нативный 1440p** (1920×1440, 10800 токенов) — **16–17 s/шаг** (спилл активаций, 25 шагов ≈ 7 мин),
а при 34 ГиБ занятой RAM снаружи — **70 s/шаг**. Поэтому дефолт `full-qwen` — рендер 1080p
плюс ESRGAN до высоты 1440.

`--aspect 16:9|4:3|1:1` задаёт пропорцию кадра вместо пропорций входа (по умолчанию — как у
входного изображения). Пропорция влияет на сетку conditioning и попадает в ключ кэша
(`embeds_…_2560x1440.pt`), поэтому смена аспекта = новый энкод (6–35 с на GPU). Референс при этом
подгоняется scale-to-cover с центральным кропом — без сплющивания под чужую пропорцию.

Все флаги старого `run_style.py` сохранены 1:1: `--input --out --lora-scale --model --preview --steps
--size --resolution --aspect --seed --fp8 --fp8-cast-cache --rebuild-fp8-cache --offload --offload-blocks
--offload-stream --prompt-file --upscale --upscale-backend --upscale-weights --upscale-tile
--upscale-color-fix --upscale-color-strength --upscale-color-sigma --text-encoder`. Новое:
`--config <пресет|путь>` (base/preview-qwen/turbo/flux2/preview/demo/full-flux2/full-qwen/h3-image/anime/qwen), `--set a.b=c` (точечный параметр `RunConfig`),
`--dry-run` (план без модели) и `--check-parity` (сверка sha256 с эталоном).

Остальные подкоманды (та же точка входа):

```bash
python -m qwenstyle batch ...           # N seed'ов за одну загрузку + пакетный апскейл (бывший tools/batch_seeds.py)
python -m qwenstyle env                 # версии, GPU/VRAM/RAM, offline, счётчики кэшей, свободный диск
python -m qwenstyle cache list|show|prune   # обслуживание кэша эмбеддингов (read-only по умолчанию)
python -m qwenstyle models list|probe <repo> [--offline]   # бекенды и зондирование модели без скачивания
python -m qwenstyle models download <repo> [--dry-run] [--yes]   # план файлов/размера; скачивание — только с --yes
python -m qwenstyle compare-upscale --input <png> --height 960   # метрики апскейлеров на одном кадре
python -m qwenstyle bench vram --resolution 640 --input <png>   # пик VRAM/RSS по фазам (бывший tools/measure_vram.py; --input обязателен)
python -m qwenstyle check-parity        # паритет sha256 с tests/reference.json
python -m qwenstyle config show base --set generation.steps=8   # резолвленный RunConfig
```

Параметры: `--model base|turbo` (turbo — только явно), `--preview` (640p + 8 шагов), `--size 768p`
(высота кадра; `--resolution` считается по пропорциям входа, 1280 = 960p — только с `--offload group`),
`--steps` (turbo — всегда 8 по расписанию), `--lora-scale` (по умолчанию из пресета: 0.7), `--seed`
(по умолчанию из пресета: 20261113), `--fp8 0/1`, `--prompt-file`, `--out`, `--fp8-cast-cache`,
`--rebuild-fp8-cache`, `--text-encoder base|heretic`, `--upscale 0|2|3|4|<дробный>`,
`--upscale-backend esrgan|seedvr2`, `--upscale-weights`, `--upscale-tile`, `--no-logs`.

### Генерация без референса (`--aspect` вместо `--input`)

`--input` больше не обязателен: нужен ли модели референс, объявляет сам бекенд
(`Capabilities.requires_image`, по умолчанию `True` — прежнее поведение), а ядро решает это в
`pipeline.input_problem()` **до** загрузки весов. `qwen_image21` и `flux2` без картинки падают
одной строкой `модель qwen_image21 требует референс: передайте --input <png>` (exit 2, без
трейсбека), а `h3-image` работает и без неё — режим (`text2image` или `image_conditioned`)
модульный пайплайн выбирает сам по наличию `image`:

```bash
# T2I: пропорцию кадра задаёт --aspect (без него — 1:1), --size задаёт высоту
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m qwenstyle run --config h3-image \
    --prompt-file promts/exp3.txt --aspect 1:1 --size 512p --seed 20261113 --out out/h3i-t2i.png

# edit: тот же пресет, но с референсом
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m qwenstyle run --config h3-image \
    --prompt-file promts/exp3.txt --size 512p --input imgs/nita-tmp.png --out out/h3i-edit.png

# план без весов: сетка, ключ кэша, нужен ли энкодер
... -m qwenstyle run --config h3-image --dry-run --prompt-file promts/exp3.txt --aspect 1:1 --out out/_h3i.png
```

В ключ кэша вместо sha256 картинки попадает сегмент `noref`:
`text_embeds_…_noref_te-h3qwen32-heretic-dev-cuda_1024x1024.pt` — T2I-записи не смешиваются с
записями по картинкам, старые ключи Qwen/FLUX.2 не изменились. Замеры (512×512, seed 20261113,
4080): холодный edit — **201 s** (энкод 32B 82.8 s + генерация 86.2 s), тёплый — **34 s**; T2I при
готовом кэше `text_embeds` — **`total 96–99 s`** (генерация 83.0–83.1 s), тёплый — **29 s**;
пик VRAM 6.2/7.1 ГиБ. **На дефолтном размере пресета (1088×1024, edit, seed 84):** энкод 32B
**127.5 s** (2311 токенов), генерация **101.8 s** на первом кадре и **~39.8 s** на последующих,
peak-ish VRAM **7.5 GiB**, `[done] total 264 s`; батч из 8 seed'ов — **327 s** (поэтому финалом
можно сразу брать `size_px: 1024`, а 512p держать для черновиков).

### Правила написания промтов

Как писать запросы к Qwen-Image 2.1 + FUSAL (два режима: структурные блоки и инструкционная
правка, скелеты, что в каждом блоке, длина, антипаттерны, чек-лист, цена правки) —
[`PROMPT_RULES.md`](PROMPT_RULES.md). Краткая обязательная версия и эталоны — `AGENTS.md` §5,
`promts/exp1.txt`…`exp4.txt`; скилл `fusal-prompt` — `.pi/skills/fusal-prompt/SKILL.md`.

### Логи прогона: `<кадр>/run.log` и `--no-logs`

Каждый прогон кладёт свой отладочный лог **рядом со своим кадром**, а не в общий файл:

```
out/rainforest-s20261113.png          # кадр (путь не изменился)
out/rainforest-s20261113/run.log      # весь вывод прогона: [profile], [size], [in], ключ кэша,
                                      #   [1/4]…[4/4], время фаз, [post], [done]
out/rainforest-s20261113/meta.json    # параметры прогона (RunConfig) + argv + время
```

* stdout и stderr идут одновременно в консоль и в лог (прогресс виден, traceback падения
  остаётся в файле); ранние строки (`[profile]` печатается до открытия файла) попадают в лог
  из буфера — порядок строк как в консоли;
* батч — одна папка на сессию: `<out-dir>/_logs/<YYYYmmdd-HHMMSS>/run.log`;
* `--no-logs` не создаёт ни папки, ни файла (вывод только в консоль); в пресетах это поле
  `logs: false` / `--set logs=0`;
* лог — наблюдение: сбой записи печатает `[log] …` и не роняет генерацию; ключ кэша эмбеддингов
  и сам кадр от него не зависят;
* `check-parity` и тесты идут без логов (`pipeline.run(..., log=None)`) — регресс-прогон не
  оставляет папок рядом с кадрами паритета.


### Пресеты

`qwenstyle/presets/*.yaml` — только данные (`extends:` = deep-merge), читаются `RunConfig.from_profile()`:

| пресет | что это | модель / шаги / кэш точности |
|---|---|---|
| `base` | профиль по умолчанию: FUSAL R80 v2 | `Qwen/Qwen-Image-2.1`, 25 шагов, fp8, `cache/fp8_transformer` |
| `preview-qwen` | черновик Qwen для итераций по промту | `extends: base`, `size_px: 480` (864×480 при 16:9), 8 шагов |
| `turbo` | distill-профиль | transformer из `Qwen/Qwen-Image-2.1-Turbo` + Turbo E6 R128 LoRA, 8 шагов (расписание `sample_sigmas`), `cache/fp8_transformer_turbo` |
| `flux2` | базовый FLUX.2 klein | `Flux2KleinPipeline` + `Qwen3ForCausalLM`, `conditioning=internal`, `image_inputs=multi`, fp8 |
| `preview` | черновик FLUX.2 | `extends: flux2`, `size_px: 768` (1376×768), 8 шагов, без offload — 33 с |
| `demo` | быстрая демонстрация | `extends: flux2`, `size_px: 960` + ESRGAN до `target_height: 1440` → 2592×1440 за ~61 с |
| `full-flux2` | финал FLUX.2 | `extends: flux2`, `size_px: 1440` + `offload: group`, `offload_blocks: 2` → 121 с, пик VRAM 15.7 GiB |
| `h3-image` | MiniMax-H3-Image: T2I и edit одним пайплайном | `MiniMaxAI/MiniMax-H3-Image` (33.13B), **6 фиксированных шагов** PDD, fp8 30.9 ГиБ + `offload: group`, `cache/fp8_transformer_h3_image`; LoRA неприменима, энкодер — Qwen3-VL-32B (`h3qwen32-heretic`), `--input` необязателен |
| `anime` | мягкий перенос стиля | `extends: base`, `size_px: 1080`, 25 шагов, `offload: group`, **`lora.scale: 0.3`**, `upscale.target_height: 1440`; энкодер **пинит `base`** (не heretic) |
| `full-qwen` | финал Qwen: 1080p + апскейл | `extends: base`, `size_px: 1080`, 25 шагов, `offload: group`, `offload_blocks: 1`, `upscale.target_height: 1440` → 1440×1088, **2.4 s/шаг** |
| `qwen` | алиас `base` | `extends: base` (меняется base.yaml — меняется и qwen) |

Апскейлер по умолчанию во всех пресетах — **Real-ESRGAN anime6B** (`upscale.backend: esrgan`,
≈1.5 с и <1 ГиБ VRAM); SeedVR2 включается флагом `--upscale-backend seedvr2` (см. «Апскейлеры»).

Энкодер по умолчанию — **Heretic-1.3.0** (`presets/base.yaml` → `encoder.name: heretic`): именно на
его conditioning обучалась LoRA. Откат к штатному — `--text-encoder base` (другое имя кэша: без
`_te-heretic`). Пресет `anime` пинит `base` явно.

### Проверка паритета после переезда

`python -m qwenstyle check-parity` воспроизводит записи `tests/reference.json` (`base-640-8`,
`turbo-640`, `esrgan-x4`, `batch-2seeds`, `flux2-preview-8`) **новым** каркасом и сравнивает `sha256`
кадров (у батча — только `s{N}-gen.png`), печатает PASS/FAIL по каждой записи и выходит с кодом 1
при любом FAIL. Эталоны сняты старым кодом до его удаления (`tools/parity_refs.py` — документ, как
именно), кроме `flux2-preview-8`: эта запись снята уже новым каркасом (`--config preview`, FLUX.2,
1376×768, 8 шагов, seed 20261113, sha256 `233b527c…`) и проверяется наравне с остальными.

Необязательное поле `requires` в записи (репозиторий Хаба вроде
`darknight9121/FLUX.2-klein-base-9B-bucket-uncensored` или путь вроде `./cache/fp8_transformer_flux2`)
делает эталон переносимым: если чего-то нет, печатается `[skip] <entry>: <причина>`, запись не
считается ни PASS, ни FAIL, а итог показывает `skipped N` (код возврата 0 при FAIL 0). На этой
машине все веса на месте: `10/10 проверок PASS, FAIL 0, skipped 0`.

`run --check-parity auto` подбирает запись по **метке профиля** (`cmd/run.py::model_label`): `base` —
всё семейство Qwen-Image-2.1 + FUSAL R80 (`base`, `qwen`, `preview-qwen`, `full-qwen`), `turbo` —
свой, а `preview` — FLUX.2 и остаётся `preview`. Раньше `preview` считался «тем же base, только
черновым» и авто-сверка подбирала чужой эталон `base-640-8` (тест `tests/test_profiles.py`).

### Как добавить модель генерации / апскейла / энкодера

1. **Модуль бекенда** `qwenstyle/backends/<name>.py` с `@register_generation` (`@register_upscale`,
   `@register_encoder`), `caps: Capabilities` и `budget: ResourceBudget` — контракты в
   `qwenstyle/backends/base.py`. Conditioning объявляется честно: `precomputed` (сами считаем
   эмбеддинги и кэшируем `.pt`) или `internal` (промт + референсы уходят в пайплайн, как у FLUX.2).
2. **Пресет** `qwenstyle/presets/<name>.yaml` — только данные: `extends: base`, `generation.backend:
   <name>`, `encoder.name`, `upscale.backend`, шаги/точность/`precision_cache`.
3. **Проверка без скачивания**: `python -m qwenstyle models list` (caps/budget в строке) и
   `python -m qwenstyle models probe <repo> --offline` — компоненты и вердикт по 16 ГБ VRAM,
   63.6 ГБ RAM и свободному диску.
4. **Прогон**: `python -m qwenstyle run --config <name> --dry-run …`, затем без `--dry-run`.

`qwenstyle/pipeline.py` и `qwenstyle/cli.py` при этом не меняются: ядро знает бекенды только по имени.

### Текстовый энкодер: Heretic по умолчанию, `--text-encoder base` для штатного (D7)

Оба энкодера — Qwen3-VL-8B (`hidden_size 4096` — ровно то, что требует
`transformer/config.json → context_in_dim`), различаются только весами MLP `down_proj`:

| `encoder.name` | что это | веса | тег в ключе кэша |
|---|---|---|---|
| **`heretic`** (по умолчанию) | `DreamFast/Qwen3-VL-8B-Heretic-1.3.0` @ `28dc0129…` — abliterated-версия, conditioning как при тренировке LoRA | `cache/weights/heretic-te` (16.72 ГиБ bf16, 4 шарда) | `_te-heretic` |
| `base` | штатный Qwen3VL-8B из `Qwen/Qwen-Image-2.1` (`text_encoder/`) | `hf-cache/…` | без сегмента |

Пресет `anime` пинит `base`; `base`, `qwen`, `preview-qwen`, `full-qwen`, `turbo` берут heretic из
`presets/base.yaml`. Переключение флагом: `--text-encoder base|heretic`. processor/tokenizer в обоих
случаях базовые — так предписывает карточка LoRA.

```bash
PYTHONIOENCODING=utf-8 HF_HUB_OFFLINE=0 .venv/Scripts/python.exe -c "from huggingface_hub import snapshot_download as s; \
  s('DreamFast/Qwen3-VL-8B-Heretic-1.3.0', revision='28dc0129b4c7c16304bc2ed3697c9437ae8ac2f3', \
  allow_patterns=['model-*.safetensors','*.index.json','config.json','generation_config.json'], \
  local_dir='cache/weights/heretic-te')"

# дальше как обычно; по умолчанию энкодер — heretic, откат к штатному:
... -m qwenstyle run --text-encoder base --out out/base-encoder.png
```

В ключ кэша эмбеддингов попадает тег `_te-heretic`, так что записи двух энкодеров не путаются,
а базовые остаются валидными. Каждая пара (промт, картинка, резолюция) с heretic — новый энкод:
190 с @864×480 (замер CPU-пути). По умолчанию (`encoder.device: auto`) энкодер стримится на GPU
(пик VRAM 1–3.5 ГиБ) и после кодирования освобождается; на CPU — только `--set encoder.device=cpu`.

### Энкод промта на GPU по умолчанию

`encoder.device: auto` (пресет `base`) разрешается в `cuda` при доступной CUDA и сам включает
стриминг весов (`stream=True`, `prefetch=4`): Qwen3VL-8B (16.33 ГиБ, вместе с vision-башней) идёт
с диска через VRAM по одному модулю, поэтому ни 16.3 ГиБ RAM, ни 16 ГиБ VRAM целиком не нужны.
После кодирования VRAM освобождается (`text_encoder=None`), в шагах сэмплера энкодер не участвует,
флагов не требуется:

```bash
... -m qwenstyle run --size 1080p --input … --prompt-file … --out out/final.png
# [4/5] кодирование промта на GPU (Qwen3VL-8B, стриминг)... ok, 6s (1, 2513, 4096)
```

Замеры: **35 с** cold (640×480, 1283 токена) · **6 с** warm (1440×1088, 2513 токенов) · **0 с**
при попадании в кэш эмбеддингов. Откат на CPU-путь — `--set encoder.device=cpu`
(55 с @864×480 … 255 с @1376×768, RSS 17.6 ГиБ; такая запись кэша идёт без `-dev-cuda`).

### Энкодер больше RAM (`--set encoder.stream=true`)

Если энкодер не влезает в RAM целиком (32B в bf16 — 62 ГиБ), модель можно собирать на meta-параметрах
и читать веса каждого модуля с диска перед его forward (`qwenstyle/streaming.py`). Результат
бит-в-бит, ключ кэша эмбеддингов не меняется; поле `stream` по умолчанию выключено, но на GPU-пути
(`encoder.device: auto` + CUDA) включается автоматически — см. раздел выше. Замер на FLUX.2-энкодере
(15.26 ГиБ, 16 токенов, CPU):

```bash
... -m qwenstyle bench encoder --repo <repo> --subfolder text_encoder --tokens 16 --stream 0 --save cache/_bench/te_full.pt
... -m qwenstyle bench encoder --repo <repo> --subfolder text_encoder --tokens 16 --stream 1 --compare cache/_bench/te_full.pt
```

`peak_uss` 15.91 → **1.75 ГиБ** (`peak_rss` 15.93 → 17.10 — mmap попадает в rss, метрика режима —
`uss`), `max diff 0` на слоях 9/18/27; forward медленнее: 5.8 → 8.4 с. На реальных 4-шардовых
весах heretic-te (`cache/weights/heretic-te`, 16.33 ГиБ, 750 тензоров) то же самое:
`peak_uss` 16.93 → **1.75 ГиБ** (−15.19 ГиБ, ~9.7×), `max diff 0`, forward 4.1 → 5.6 с.
В текущих пресетах режим нужен только для FLUX.2-энкодера и будущих внешних (`encoder.local_dir`),
а на GPU-пути Qwen (`encoder.device: auto` + CUDA) он включается автоматически; `stream=True` работает
и на `device='cpu'`, и на `cuda` (см. раздел ниже).

### Энкодер на GPU: RAM-стейджинг и конвейер

Тот же `qwenstyle/streaming.py`, но веса модулей читаются с диска в pinned-буфер RAM фоновым
потоком и асинхронно копируются в VRAM: пока GPU считает слой N, читается слой N+1. В VRAM живёт
один модуль (пик VRAM 1.2–3.5 ГиБ), поэтому схема работает и для весов, не влезающих в VRAM.

```bash
# плечи: NVMe→RAM (cold/warm) и RAM→VRAM (pinned/pageable) — секунды
... -m qwenstyle bench io --read-gib 4 --h2d-gib 0.5

# GPU-энкод одного прогона: качество сверяется с CPU через --save/--compare
... -m qwenstyle bench encoder --repo <ID> --stream 1 --device cuda --prefetch 4 --tokens 512

# то же в реальном прогоне (prefetch/пиннинг/lookahead — только при stream=true + device=cuda)
... -m qwenstyle run ... --set encoder.stream=true --set encoder.device=cuda --set encoder.prefetch=4
```

Замерено (`out/bench/te-gpu-stream.md`, FLUX.2-энкодер 8.19B / 15.26 ГиБ bf16): на 512 токенах CPU
**97.6 s** → GPU **10.4 s** (9.4×); время GPU почти не зависит от длины промта (10.3 / 10.4 / 10.5 s
на 16 / 512 / 4128 токенах) — режим **I/O-bound** (ожидание данных 0.92–0.97 бюджета), а не
compute-bound; плечи — NVMe 4.39 (cold) / 6.81 (warm) ГБ/с, H2D pinned 26.70 ГБ/с; `max|diff|`
prefetch 0 vs 4 = 0.0, CPU vs GPU = 128.0 (поэтому `device`/`dtype` — в ключе кэша: сегменты
`-fp16` / `-dev-cuda` в имени файла). Это же — умолчание на машине с CUDA (`encoder.device: auto`
→ `cuda` + `prefetch=4`); откат на CPU — `--set encoder.device=cpu`.

**Pinned staging и `--offload group`:** на Windows pinned-память CUDA — это тот же ресурс, что и
VRAM (блокировка страниц + маппинг в адресное пространство GPU), а групповой offload уже держит в
ней 6.6–8.5 ГиБ весов fp8-трансформера. Поэтому при `generation.offload=group`
(`full`, `full-qwen`) пиннинг staging-буферов энкодера **выключается автоматически**
(`RunConfig.encoder_pin_memory`; `--set encoder.pin_memory=1` его не включает), а если pinned-буфер
всё равно не выделился — `qwenstyle/streaming.py` деградирует до обычной RAM и печатает `[stream]
pin_memory выключен`. Цена — H2D 20.5 вместо 26.7 ГБ/с (0.05–0.31 s из 8–10 s энкода), выигрыш —
прогон не падает с `CUDA error: out of memory` при свободной VRAM. Замер: `full-qwen` 1088p с нуля —
энкод 16.33 ГиБ стримится с `pin_memory=0`, полный прогон **76 s**.

### Энкодер для `h3-image`: Qwen3-VL-32B (`h3qwen32-heretic`)

У MiniMax-H3-Image свой энкодер, и ни один из 8B проекта сюда не подходит: H3 читает hidden state
**после 50-го decoder layer** и ждёт `text_dim 5120`, а у всех 8B (Qwen-Image-2.1, heretic-8B,
FLUX.2-энкодер) `hidden 4096` (плюс у FLUX.2 нет vision). Поэтому пресет `h3-image` пинит
`encoder.name: h3qwen32-heretic` — `llmfan46/Qwen3-VL-32B-Instruct-ultra-uncensored-heretic`
@ `c44b949b30d11166…`, **62.13 GiB bf16**, тег `_te-h3qwen32-heretic` в ключе кэша.

| что | замер |
|---|---|
| веса | `cache/weights/h3qwen32-heretic` — 62.13 GiB, **3 шарда** < 40 GiB (23.79/23.86/14.49); в hf-cache шард 46.5 GiB, а `safetensors` 0.8.0 на Windows падает на файлах ≥ ~40 GiB, поэтому веса пересобраны (`out/bench/h3i-te-reshard.py`) |
| stream 1 + cuda + prefetch 4 (512 токенов) | загрузка 0.4 s, forward **205.2 s**, `peak_uss` **5.25 GiB**, `peak_vram` 1.93 GiB |
| stream 0 + cpu (те же 512 токенов) | загрузка 175.1 s, forward 524.8 s, `peak_uss` **48.98 GiB** — вариант не работает вместе с fp8-трансформером |
| в прогоне (512×512) | T2I: **118.9 s** (218 токенов, `(1,218,5120)`); edit: **82.8 s** (1249 токенов, синхронный стриминг быстрее конвейера) |

Фаза энкодера идёт **отдельным процессом** (на этой машине mmap списывается в commit целиком,
а fp8-кэш держит ~2× своего размера — см. `AGENTS.md` §4), fp8-трансформер грузится позже, в
`generate`. На холодную за энкод платится 118.9–205.2 s, дальше — 0 s из кэша `text_embeds`.
Холодная фаза пока хрупкая (фоллбек — одна повторная попытка синхронным стримингом, долг D10).

Устаревшие эмбеддинги убирает политика вытеснения `cache_supersede` (`qwenstyle/cleanup.py`,
умолчание `pair`: сменил промт или картинку → старые записи этой пары удаляются отложенной задачей
в конце прогона; разрешение/аспект не вытесняются никогда). Если параллельно идут два `run` с
разными промтами на одной картинке — `--set cache_supersede=none`, иначе они могут вытеснить кэш
друг друга. Регресс-тест `check-parity` и `bench` (vram/encoder) всегда идут с этой политикой
выключенной: они не вытесняют и не удаляют ни одной записи.

Имена файлов: к `--out` автоматически добавляется `-s<seed>`, а при `--upscale` ещё и `-x4`/`-x2` —
так генерёный и апскейленный кадры не перезаписывают друг друга, а seed всегда видно в имени.
При смене `--steps`/`--lora-scale` задавай свой `--out`.

### Апскейлеры: список и настройки

Апскейл — отдельный бекенд (`qwenstyle/backends/{seedvr2,esrgan}.py`), запускается после того, как
пайплайн генерации освобождён. **По умолчанию в пресетах — `esrgan`** (быстро и дёшево по VRAM);
`seedvr2` включается флагом `--upscale-backend seedvr2`. Выключить совсем:
`--upscale 0 --set upscale.target_height=0`.

| `upscale.backend` | что делает | веса / диск | скорость (4080) | пик VRAM |
|---|---|---|---|---|
| **`esrgan`** (по умолчанию) | Real-ESRGAN x4plus_anime_6B (своя RRDBNet без `basicsr`), тайловый инференс | `cache/weights/RealESRGAN_x4plus_anime_6B.pth` — 17.9 МиБ, качаются сами | 864×480 → 3456×1920: 0.4 с; 1376×768 → 5504×3072: 1.5 с | < 1 ГиБ |
| `seedvr2` (опция) | SeedVR2-3B (ByteDance) — реставрация с переносом текстуры; отдельное окружение `seedvr2/` (torch 2.3 cu121) | `seedvr2/SeedVR/ckpts/`: 12.6 ГиБ c fp8-хранением + 0.93 ГиБ VAE, **17 ГиБ** на диске | 864×480 → 1728×960: **71 с** в пайплайне; 1440×1088 → 1924×1440: **62 с** (+19 с с `color_fix=wavelet`); x3 (2592×1440) работает, **x4 — OOM** | 8.5 ГиБ |

Настройки (`UpscaleConfig` — пресет, `--set`, CLI-флаг):

| поле | флаг | по умолчанию | к чему относится |
|---|---|---|---|
| `backend` | `--upscale-backend esrgan\|seedvr2` | `esrgan` | выбор бекенда |
| `factor` | `--upscale F` | `0.0` (=выкл) | во сколько раз увеличить; дробный считается от готового кадра |
| `target_height` | `--upscale-to NNNp` | `null` | целевая высота; если задана, `factor` вычисляется из неё (1440/1088 = ×1.32353) |
| `weights` | `--upscale-weights PATH` | `null` | путь к `.pth` — **только `esrgan`** (seedvr2 берёт свои ckpts) |
| `tile` | `--upscale-tile N` | `256` | **только `esrgan`**: тайл инференса (тайлы дают max diff 7/255 к целому кадру) |
| `pad` | `--set upscale.pad=N` | `32` | **только `esrgan`**: перекрытие тайлов |
| `color_fix` | `--upscale-color-fix none\|wavelet\|reinhard` | `wavelet` | **только `seedvr2`**: цветокоррекция результата по исходному кадру |
| `color_strength` | `--upscale-color-strength F` | `0.75` | **только `seedvr2` + `wavelet`**: сила коррекции |
| `color_sigma` | `--upscale-color-sigma F` | `12.0` | **только `seedvr2` + `wavelet`**: радиус вейвлет-разложения |

```bash
# апскейлер из пресета (esrgan) + target_height 1440
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m qwenstyle run --config full-qwen \
    --seed 84 --input imgs/nita-iren-only.png --prompt-file promts/nita-bedroom-film.txt --out out/f.png

# SeedVR2 вместо ESRGAN, его цветокоррекция и выключение апскейла совсем
... -m qwenstyle run --config full-qwen --upscale-backend seedvr2 --out out/f-sv2.png
... -m qwenstyle run --config full-qwen --upscale-backend seedvr2 --upscale-color-fix none --out out/f-sv2-nocf.png
... -m qwenstyle run --config full-qwen --set upscale.target_height=0 --out out/f-raw.png
```

**Real-ESRGAN anime6B — дефолтный бекенд**: веса `cache/weights/RealESRGAN_x4plus_anime_6B.pth`
(17.9 МиБ, качаются автоматически), своя реализация RRDBNet без `basicsr`, тайлы 256/pad 32.
На текущем стеке тайлы дают **max diff 7** (из 255) к прогону целым кадром — это не ошибка каркаса
(старый `upscale.py` давал то же), сверка old-vs-new теперь в `python -m qwenstyle check-parity`
(запись `esrgan-x4`). Пик VRAM < 1 ГиБ, 1376×768 → 5504×3072 за 1.5 с.

Апскейл **пропускается автоматически**, если кадр уже не ниже `upscale.target_height` (или фактор ≈ 1):
бекенд не загружается, печатается `[post] upscale пропущен: …`, `-x1` в имени не появляется.
При этом `--upscale 0` не отменяет `upscale.target_height` из пресета — чистый кадр без поста:
`--set upscale.target_height=0`.

**SeedVR2-3B (по флагу `--upscale-backend seedvr2`)** — установлен, работает в отдельном окружении `seedvr2/` (torch 2.3 cu121 +
официальный код ByteDance-Seed/SeedVR; его требования несовместимы с torch 2.8 / diffusers 0.43).
Веса bf16 12.6 ГиБ + VAE 0.93 ГиБ; на 16 ГБ работает только с **fp8-хранением весов (по умолчанию
включено)** и с патчами из `seedvr2/patches.py` (flash_attn-шим на SDPA, apex → RMSNorm,
RoPE-таблица под фактический кадр вместо фиксированных 7.88 ГиБ). Замер: 864×480 → 1728×960
за 71 с, пик 8.5 ГиБ. x4 не проверялся (вероятен OOM). Установка на новой машине:
`python seedvr2/setup.py`.

`--model turbo` использует только transformer из `Qwen/Qwen-Image-2.1-Turbo` (14 GiB) + Turbo LoRA
(640 MiB); text-encoder и VAE — базовые, и это не компромисс: проверено по файлам, что в Turbo-репо
лежат ровно те же VAE (F32 vs BF16 хранения) и text-encoder (все 750 тензоров), что в базовом.
Адаптер помечен авторами как preview (still under training).

## Кэши

| Кэш | Где | Что экономит |
|---|---|---|
| Веса | `hf-cache/` (32 GiB, канонически: `blobs/` + симлинки) | повторные загрузки с HF |
| fp8-трансформер | `cache/fp8_transformer/` и `cache/fp8_transformer_turbo/` (по 6.7 GiB) | 26 с каста + чтение 14.2 ГБ вместо 6.7 на каждом старте |
| Эмбеддинги | `cache/embeds_<md5(промт)>_<sha256(картинки)>_<W>x<H>.pt` + `.json` | **6–35 с** GPU-энкода (35 с cold @640×480, 6 с warm @1440×1088), 0 с из кэша; 55 с @864×480 / 255 с @1376×768 — только на CPU (`--set encoder.device=cpu`) |

Порядок работы: один раз зафиксировать картинку и промт → перебирать `--seed` / `--lora-scale` /
`--steps` на `--preview` (бесплатно) → один финальный прогон на 1024. Любая правка текста промта,
другой входной файл или другая резолюция = новый энкод (6–35 с на GPU, 0 с из кэша).
Подробно — `AGENTS.md` §6.

Батч по seed'ам (замена `tools/batch_seeds.py`): `python -m qwenstyle batch --config preview-qwen
--seed-range 0-9 --upscale-to 768p --out-dir out/batch --input … --prompt-file …`. Пайплайн грузится
один раз, апскейл идёт после его выгрузки, файлы — `s{N}-gen.png`/`s{N}.png`, `--resume` по умолчанию.
Ширина финального кадра считается по пропорциям **готового** кадра: старый `--out-res 1024` давал
1376×768, новый `--upscale-to 768p` — 1382×768 (высота совпадает). Чистая генерация `s{N}-gen.png`
от этого не зависит.

Правила написания промтов (структурированные блоки, тиры, режимы правки картинки) — скилл
`.pi/skills/fusal-prompt/SKILL.md`, эталоны — `promts/exp1..4.txt`.

## Отступления от примера из переписки

В присланном коде было четыре проблемы:

1. `dtype=` вместо `torch_dtype=` — правильное имя параметра в diffusers 0.43.
2. `device_map="cuda"` — 33.7 ГБ не влезают в 16 ГБ, нужен fp8 + стриминговый text-encoder
   (по модулю в VRAM, см. «Энкод промта на GPU по умолчанию»).
3. `pipe(...)` в `DiffusionPipeline` без `QwenImage21Pipeline` — у модели своя обёртка
   (`_class_name: QwenImage21Pipeline` в `model_index.json`).
4. результат не сохранялся (нет `image.save`).

Дополнительно: `true_cfg_scale` у 2.1 по умолчанию 1.0, поэтому `negative_prompt`
**игнорируется** — CFG выключен, он не нужен. `peft` создаёт LoRA-тензоры в dtype
базового слоя (fp8), а `addmm` для fp8 не реализован — в скрипте они кастуются в bf16.

## Приватность этой сборки

* Инференс локальный: изображение никуда не уходит, у `diffusers` нет телеметрии.
* Наружу уходят только скачивания весов с HuggingFace (анонимно, без токена).
* Веса LoRA — `.safetensors` (не pickle), исполняемого кода при загрузке нет.
* Выходной PNG сохраняется через `PIL.save` без `pnginfo`: промта и параметров внутри нет.
* Отладочный лог (`<кадр>/run.log`) и `meta.json` рядом с кадром **содержат промт и пути** —
  это локальные файлы для отладки; не нужны — запускать с `--no-logs` (или удалить папку кадра).
* У исходного файла EXIF отсутствует (проверено) — GPS/даты/устройства в нём нет.
* «Heretic» (расцензуренный) text encoder из репозитория LoRA **не используется** —
  взят официальный энкодер `Qwen/Qwen-Image-2.1`.
