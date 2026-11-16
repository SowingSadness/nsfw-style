# DEVELOPMENT.md — внутреннее устройство и dev-опции

Документ для разработчика пакета. Пользовательские инструкции (установка, профили, `gen`,
статистика по пресетам) — в [`README.md`](../README.md); обязательные правила для агентов и
проверенные ограничения — в [`AGENTS.md`](../AGENTS.md).

## 1. Публичное и внутреннее

Публичный контракт пакета — не «всё, что лежит в репозитории», а ровно:

| публичное | внутреннее (может меняться без предупреждения) |
|---|---|
| 15 публичных пресетов (`VRAM8-*` 4, `VRAM12-*` 4, `VRAM16-*` 7) и профили машин (`qwenstyle/profiles.py`) | остальные пресеты (`base`, `turbo`, `h3-image`) |
| подкоманды `gen`, `run`, `selftest`, `env`, `models`, `cache`, `bench`, `config`, `compare-upscale` | внутренние функции `pipeline.run_batch`, `io.batch_paths`, `runlog`, `streaming`, `precision`, `cleanup` |
| файлы `prompts/<имя>/prompt.txt` + `image*`, папка сессии `out/<имя>/gen-<штамп>/` | layout `cache/embeds` (`*.pt` + `.json`), `cache/models`, имена `fp8_transformer*` |
| `RunConfig.from_profile(<имя>)`, флаги `gen`/`run`, `--set` по `ParameterSpec` | dev-бекенды (`flux2`, `seedvr2`) и их параметры |

Правило: публичные шаги (README, `bootstrap.*`, скрипты пользователя) **не должны** ссылаться на
dev-пресеты и dev-бекенды. Всё, что адресуется пользователю, идёт через профиль
(`qwenstyle/profiles.py`), а не через имя пресета-черновика.

Пакет — NSFW-style: FUSAL R80 — **один из** стилевых адаптеров; публичные пресеты без суффикса
`-anime` его не включают; замена стиля = другой LoRA/`--lora-scale`, ядро при этом не меняется.

## 2. Dev-пресеты

Оставлены для экспериментов и замеров; в `profiles.py` они не входят и публичной поддержки у них нет.

| пресет | что это | зачем живёт |
|---|---|---|
| `base` | `Qwen/Qwen-Image-2.1` + FUSAL R80, 25 шагов, fp8, энкодер `heretic` | корневой `extends` для всех остальных пресетов |
| `turbo` | transformer из `Qwen/Qwen-Image-2.1-Turbo` + Turbo E6 R128, 8 шагов (расписание `sample_sigmas`) | быстрый distill-профиль; адаптер помечен авторами как preview; база `VRAM8-*` |
| `h3-image` | `MiniMaxAI/MiniMax-H3-Image` (33.13B), 6 шагов PDD, fp8 30.9 ГиБ + `offload: group`, энкодер Qwen3-VL-32B | T2I + edit одним пайплайном |

Три референса на 16 ГиБ (`VRAM16-r3-169`, `VRAM16-r3-43`) — публичные пресеты профиля `VRAM16`
(`profiles.py`), не dev: `size_px: 912` + `aspect: 16:9` → crop 1632×896, ESRGAN ×2.41 → 3934×2160
и `size_px: 1056` + `aspect: 4:3` → crop 1408×1056, ×2.045 → 2880×2160. Это максимум, который
проходит энкодер при трёх референсах (`928p`/`1088p`/`1440p` — OOM на кодировании промта).

Удалены (пресеты есть только в истории): `preview-qwen`, `anime`, `full-qwen`, `qwen-2k`, `qwen`,
`flux2`, `preview`, `demo`, `full-flux2` — например, FLUX.2-пресеты (`flux2`, `preview`, `demo`,
`full-flux2`) удалены вместе с остальными dev-надстройками. Бекенд `qwenstyle/backends/flux2.py`
при этом остался — без пресетов он доступен только через `--set generation.backend=flux2`
и прямую сборку `RunConfig`.

## 3. SeedVR2 — dev-опция

Апскейлер SeedVR2-3B (ByteDance) установлен и работает, но **не входит в публичные пресеты**
(в них только `esrgan`) и в профили машин: у него отдельное окружение `seedvr2/`
(torch 2.3 cu121 / diffusers 0.29.1), свой `setup.py` с патчами и нестабильный x4 (OOM по
активациям). Запускается явным флагом `--upscale-backend seedvr2`; см. `AGENTS.md` §4 за замерами.
Прежде чем добавлять его в публичный пресет, нужны: x4 на 16 ГБ, воспроизводимый `setup` и
проверка, что веса не тянут за собой лишние зависимости в основное окружение.

## 4. Политика кешей

Две ветки с разной природой — разный жизненный цикл.

```text
cache/
  embeds/                       # conditioning: мелкий, часто пересоздаётся
    embeds_<md5(prompt)>_<sha(images)>[_te-<тег>][-<dtype>][-dev-<device>]_<W>x<H>.pt (+ .json)
    text_embeds_<…>_<W>x<H>.pt (+ .json)     # FLUX.2 / MiniMax-H3
    _stale/                                  # старые записи, ожидающие prune
  models/                       # веса и квантование: десятки ГиБ, пересоздаются раз в жизни
    fp8_transformer/  fp8_transformer_turbo/  fp8_transformer_flux2/  fp8_transformer_h3_image/
    weights/          # heretic-te (8B), h3qwen32-heretic (32B), RealESRGAN_x4plus_anime_6B.pth
```

- **`cache/embeds` вытесняется**: `cache prune` (без `--apply` — только план) и автоматическая
  политика `cache_supersede` (`none|prompt|image|pair`, умолчание `pair`) отложенной задачей в конце
  `run`/`gen`. Правила: одна запись на сетку; смена `--size`/`--resolution`/`--aspect` **ничего не
  удаляет** (иначе каждое переключение резолюции стоило бы нового энкода); `-fp16`/`-dev-cuda`
  варианты друг друга не вытесняют; запись текущего прогона не удаляется никогда; `bench` и
  `selftest` идут с `cache_supersede=none`.
- **`cache/models` не вытесняется никогда**: `cache list|show|prune` в эту ветку не заглядывают,
  пересборка квантования — только `--rebuild-fp8-cache` (26 с каста + 6.7 ГиБ записи для base,
  202.8 с / 30.9 ГиБ для H3).
- **`hf-cache/`** (`HF_HOME`) — исходные веса в канонической раскладке `blobs/` + относительные
  симлинки `snapshots/<commit>/`; автоматической чистки нет.
- **Ключ кэша эмбеддингов от путей не зависит**: переезд `cache/*.pt` → `cache/embeds/` — это путь,
  а не другая математика, поэтому ранее созданные записи остаются валидными.
- Запись атомарная (`*.tmp` → `os.replace`), рядом с `.pt` всегда лежит `.json`-сайдкар: при
  попадании он сверяется по белому списку полей (`encoder.repo`/`revision`, `schema`), расхождение
  печатает `[cache] miss: изменился <поле>`.

## 5. Правила замеров (R4 из `AGENTS.md`)

Любая правка `--fp8`, `.to("cuda")`, `enable_layerwise_casting`, `enable_model_cpu_offload`,
префетча/стриминга сопровождается замером **до и после**: время фазы, пик VRAM (`torch.cuda`
`max_memory_allocated` и `nvidia-smi`), RSS/USS (`psutil.memory_full_info()` при стриминге — важна
`uss`, mmap попадает в `rss`).

- Время кадра мерить одной командой и одним процессом на пресет:
  `python -m qwenstyle gen --preset <имя> --prompt <имя> --seeds 1` (в логе `[done] … total`).
- Публичные числа — только из `docs/bench/presets.md` (условия: RTX 4080 16 ГБ / 63.6 ГБ RAM,
  квадратный референс 768×768, `HF_HUB_OFFLINE=1`). На 16:9 числа выше примерно в 1.78× по площади.
- Новый замер, меняющий выводы, обновляет `docs/bench/presets.md` **и** §4 в `AGENTS.md`, а не
  остаётся в переписке.

### Память энкодера: `pin_memory` и `offload=group`

`pin_memory` включён по умолчанию **и при `generation.offload=group`**: `RunConfig.encoder_pin_memory`
читает только поле `encoder.pin_memory`, поэтому выключить его можно исключительно явно
(`--set encoder.pin_memory=0`). При неудаче pinned-аллокации `_PinnedPool` деградирует до pageable RAM
(результат **бит-в-бит**, медленнее только H2D) и печатает причину — это страховка, а не штатный режим.

## 6. История миграции

| шаг | что сделано |
|---|---|
| переезд на `G:\Work\nsfw-style` | проект переехал с `D:\…\qwen-imate-2.1-anime` как замороженный снимок; монолиты `run_style.py`, `upscale.py`, `colorfix.py`, `tools/*` удалены — вся логика в пакете `qwenstyle` (git тогда не было, восстановить нельзя, поэтому возможности добавляются только в пакет) |
| `v0.1.0`, шаг 1 | git пересоздан с нуля, `.pi/` вне репозитория, `.gitattributes` (`* text=auto eol=lf`) |
| шаг 2–3 | `qwenstyle/profiles.py` (VRAM8/VRAM12/VRAM16), `models download --profile/--probe`, `selftest` |
| шаг 4 | кеши разделены: `cache/embeds` ↔ `cache/models` (переименование в пределах тома, без пересчёта) |
| шаг 5 | команда сверки кадров (parity) и файл эталонов удалены: сверка нового каркаса со старыми монолитами была завершена до удаления; проверка установки теперь `selftest` |
| шаг 6 | установка одной командой (`bootstrap.ps1/.sh/.cmd`), `pyproject.toml`, MIT `LICENSE` |
| шаг 7–9 | multi-ref (до `max_images`), команда `gen` вместо `batch`, параметры из адаптеров (`parameters()`, `--set` валидируется по спекам) |
| шаг 11–12 | интеграционный прогон пресетов (`docs/bench/presets.md`), публичная документация |
| шаг 13–14 | пресеты и профили названы по VRAM (`VRAM8`/`VRAM12`/`VRAM16`), `VRAM12` влезает в 12 ГБ (`offload: group` + `kv_cache: offload`), у `VRAM12-release*`/`VRAM16-release*` `upscale: null`; тарифы проверены замером (`docs/bench/vram-tiers.md`) |

## 7. Как добавить бэкенд (модель генерации, апскейла или энкодера)

Ядро (`pipeline.py`, `cli.py`) не правится — только новый модуль и данные.

1. **Модуль** `qwenstyle/backends/<name>.py` с декоратором `@register_generation`
   (`@register_upscale`, `@register_encoder`), `caps: Capabilities` и `budget: ResourceBudget`
   (контракты — `qwenstyle/backends/base.py`).
   - `Capabilities` объявляет честно: `conditioning` (`precomputed` — сами считаем `.pt` и кэшируем,
     `internal` — промт и референсы уходят в пайплайн), `image_inputs` (`none|single|multi`),
     `max_images`, `requires_image`, `encoder`, `fixed_steps`, `supports_lora`, `precision`,
     `offload`, `max_pixels`.
   - `parameters() -> tuple[ParameterSpec, ...]` — настраиваемые параметры модели
     (`name`, `scope` ∈ `generation|sampler|encoder|upscale`, `type`, `default`, `choices`,
     `minimum`, `maximum`, `readonly`, `doc`). Из них печатается `config show --explain-params`,
     `models list --json` и валидируется `--set`; захардкоженных имён параметров в ядре быть не должно.
     Фиксированное моделью значение (`readonly=True`, например 6 шагов PDD) CLI предупреждает, а не
     молча принимает.
2. **Пресет** `qwenstyle/presets/<name>.yaml` — только данные: `extends: base` (deep-merge),
   `generation.backend`, `encoder.name`/`cache_tag`, `upscale.backend`, шаги/точность,
   `generation.precision_cache: cache/models/fp8_transformer_<name>`.
3. **Профиль** `qwenstyle/profiles.py` — если пресет публичный: добавить его имя в `Profile.presets`
   нужного профиля, а репозитории — в `Profile.repos` (+ `REPO_SIZE_GIB`); при необходимости
   уточнить `disk_gib`. Без этого пресет считается dev-only и в README не попадает.
4. **Проверка без скачивания**: `python -m qwenstyle models list` (строка `caps`/`budget`),
   `python -m qwenstyle models download --profile <имя> --dry-run`, плюс `gen --preset <имя>
   --dry-run` — план покажет сетку, ключ кэша и нужен ли энкодер.
5. **Прогон**: `python -m qwenstyle gen --preset <имя> --prompt <имя> --count 1`, затем замер по §5.
