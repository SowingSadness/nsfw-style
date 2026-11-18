# renderkit — калькулятор настроек рендера

Целевая картинка + железо → самодостаточный YAML-конфиг для `qwenstyle`, оценки VRAM/RAM/времени
и объяснение каждого выбранного значения. Это **производный слой над данными проекта**: он читает
возможности бэкендов и замеры, но не меняет Python-пакет и не требует сети, весов или CUDA.
Ноль зависимостей: `dependencies: {}`, `devDependencies: {}`, тесты — `node --test`, сервер — `node:http`.

## Быстрый старт

```bash
cd renderkit
npm test                                  # 206 тестов: геометрия, оценки, сочетания, YAML, UI, снапшоты
npm run inventory -- --check              # данные из Python-пакета не разошлись (exit 1 при расхождении)
npm run fit -- --check                    # модель оценок соответствует замерам
npm run plan -- --width 2560 --height 1440 --quality max \
                --vram 16 --ram 64 --disk nvme --model qwen21-base --out plan.yaml
npm start                                 # браузерный прототип: http://127.0.0.1:8123
npm run snapshot                          # матрица запросов → test/__snapshots__/plans.mjs
npm run snapshot -- --check               # пересчитать и сравнить (exit 1 при расхождении)
```

`node tools/plan.mjs …` без `--out` печатает YAML в stdout. Коды выхода: `0` — план есть,
`1` — вердикт `impossible` (YAML не собирается), `2` — ошибка запроса.

Полученный файл:

```bash
python -m qwenstyle run --config ./plan.yaml --input refs/frame.png --out out/
python -m qwenstyle gen --preset ./plan.yaml --prompt my-scene/prompt-qwen --count 4
python -m qwenstyle config show ./plan.yaml --json      # авторитетная проверка без загрузки весов
```

## Архитектура

```
src/**      изоморфное ESM-ядро без `node:`, DOM и зависимостей:
            computePlan(request) → Plan — детерминированная чистая функция
src/model/  geometry → memory → cost → candidates/rank → resolve → emit
src/data/   measurements.mjs, help.mjs   — руками;
            inventory.mjs, model.mjs     — СГЕНЕРИРОВАНЫ из Python/замеров
web/**      тонкий слой: форма, YAML-панель, подсказки (DOM) + статический сервер (`node:http`)
tools/**    генерирует данные и артефакты (inventory, fit, plan, snapshot, audit)
test/**     node --test, включая CLI-снапшоты и (при наличии venv) round-trip с Python
```

Один и тот же `src/index.mjs` работает в браузере, в Node, в CLI и в будущем сервисе (`POST /api/plan`).
JSDoc + `jsconfig.json` (`checkJs`) дают типы без сборки. `deps.test.mjs` падает на непустых
зависимостях и на `import "node:…"` внутри `src/**`.

## Контракт `Plan`

`computePlan(request)` возвращает один объект; поля со значениями, влияющими на кадр, память и
время, плюс провенанс решения. Сокращённо (16 ГиБ / 64 ГиБ / NVMe / base / 2560×1440 / `max`):

```js
{
  schemaVersion: 1,
  rulesetVersion: "r1+9248b4eb8288-39a882621db0+emit1",
  verdict: "ok",              // ok | degraded | impossible | unsupported
  confidence: "measured",     // measured | interpolated | extrapolated
  image: {
    requested: { width: 2560, height: 1440 },
    native: { width: 2560, height: 1440, tokens: 14400, sizePx: 1440 },
    aspect: "16:9",           // null = пропорция референса
    snap: { dx: 0, dy: 0, reachable: [[2592,1440],[2560,1408]] },
    bitDepth: 8, format: "png", exact: true
  },
  render: {
    steps: 25, offload: "group", offloadBlocks: 1, precision: "fp8",
    encoder: { name: "heretic", device: "cuda", stream: true, prefetch: 4 },
    loraScale: 0,
    memory: {
      kvCache: { mode: "offload", placement: "ram", prefixTokens: 7860,
                 gib: 3.84, perLayerMib: 122.7, costPct: 13, reason: "…" },
      vae: { tilePx: 512, stridePx: 384, gridArtifact: 1.69, peakGib: 7.4, oom: false },
      attention: { mode: "dense", windowPx: 0, marginPx: 0, coverage: 1, bands: 1, … },
      ref: { px: 960, tokensPerRef: 6360, totalTokens: 6360, reason: "…" }
    }
  },
  stages: [
    { stage: "render", enabled: true, estimates: {…}, quality: 1920 },
    { stage: "postprocess", enabled: false, reason: "модель постобработки не передана",
      estimates: null, paramsSchema: null }
  ],
  yaml: "size_px: 1440\n…",
  cli: { save, run, gen, checks },
  estimates: { vram: {…}, ram: {…}, time: {…}, disk: {…} },
  combinations: [ {…} ],       // сочетания с q/timeS/peakVram, pareto и best
  degradations: [], suggestions: [], warnings: [], trace: []
}
```

`estimates` считается по рендер-стадии; `stages` не смешивает бюджеты стадий (в Phase A
`postprocess.estimates === null`). `yaml`/`cli` — `null` только при `verdict: "impossible"`.
Каждое решение несёт `reason` — почему выбрано именно это значение (`kv-vram-does-not-fit`,
`window-not-needed`, `ref-px-budget`, `vae-tile-default` и т.д.).

## YAML-контракт

- **Без `extends` и без имён пресетов.** Калькулятор не знает ни `VRAM8-*`, ни `VRAM12-*`, ни
  `VRAM16-*`; в файл попадает только то, что объявляют бэкенды и `profiles.py`. Проверяют
  `test/emit.test.mjs` и `grep` по `src/**`.
- **Имя файла произвольное.** `run --config ./any-name.yaml` и `gen --preset ./any-name.yaml`
  работают: `RunConfig.from_profile` принимает имя в `presets/` **или** путь к YAML.
- **Самодостаточность доказана round-trip'ом**: `config show <файл> --json` даёт ровно `Plan.render`
  (0 расхождений, `test/roundtrip.test.mjs`; без venv — SKIP). Поэтому `--set`-форма и YAML не
  расходятся: оба строятся из одного `presetFor(plan)`.
- **Всё, что влияет на кадр, память и качество**, выписано явно: `size_px`, `aspect`, весь
  `generation` (включая `kv_cache`, `offload`, `steps`, `precision`, `lora`), `sampler_kwargs`
  (`offload_blocks`, `offload_stream`, `vae_tile_px`, `ref_px`, `window_attn_px` +
  `window_attn_margin_px` только когда окно включено), весь `encoder.*`, `upscale: null`, `logs: true`.
- **`upscale` выключен явно**: апскейла в этой стадии нет (см. «Честные ограничения»).
- **16 бит**: YAML остаётся валидным (никаких несуществующих ключей), но шапка честно говорит,
  что пайплайн пишет 8-битный PNG.

Пример (фрагмент реального вывода для запроса выше):

```yaml
size_px: 1440
aspect: "16:9"
generation:
  backend: qwen_image21
  repo: Qwen/Qwen-Image-2.1
  transformer: [Qwen/Qwen-Image-2.1, transformer]
  lora: null
  steps: 25
  guidance: 1.0
  precision: fp8
  precision_cache: cache/models/fp8_transformer
  offload: group
  kv_cache: offload
  sampler_kwargs:
    offload_blocks: 1
    offload_stream: true
    vae_tile_px: 512
    ref_px: 960
encoder:
  name: heretic
  cache_tag: heretic
  device: cuda
  stream: true
  prefetch: 4
upscale: null
logs: true
```

## Раскладка памяти

`Plan.render.memory` — первоклассный выход, а не побочный эффект: каждое поле показывает, откуда
взято число и что оно стоит.

| поле | значения | откуда число |
|---|---|---|
| `kvCache.mode` / `placement` | `true`→`vram`, `offload`→`ram`, `false`→`none` | 512 КиБ на префиксный токен (`32 слоя × 2 × 4096 × 2 Б`); `true` только если `peak + kv ≤ 0.90·budget`, иначе `offload` (+12–14 % времени, бит-в-бит) |
| `kvCache.prefixTokens` / `gib` / `perLayerMib` | напр. `7860` / `3.84` / `122.7` | текст (`textTokens`, по умолчанию 1500) + `refs·(refGridW/16)·(refGridH/16)`; слой = `gib/32` |
| `vae.tilePx` / `stridePx` | `512/384`, `0`→`256/192`, `1024/768`, `-1` | умолчание 512 (сетка 16 px ×1.69, пик декода 7.4 ГиБ); `256/192` — ×2.71; `1024` — ×1.51, пик 15 ГиБ, только при VRAM ≥ 24; `-1` запрещён от 1.5 Мп (измеренный OOM) |
| `vae.peakGib` / `oom` | `7.4` / `false` | замеры `AGENTS.md` §4, «Тайлинг декодера VAE» |
| `attention.mode/windowPx/marginPx/coverage/bands` | `dense` или `window` + 384/384 | окно включается при канвасе ≥ 1600 px и покрытии `(px + 2·margin)/H ≥ 0.60`; в «серой» зоне (> 5.73 Мп) — 1088/448 |
| `attention.costS` / `denseCostS` | `314` против `376` на 1792p | замер: окно покупает −17 % времени ценой −2.6 % резкости |
| `ref.px` / `tokensPerRef` / `reason` | `0`, `960`, `768`, `640` | политика: `0` при ≤1088p и ≤2 рефах, `960` от 1440p; ладдер уменьшает до потолка ≤13 000 токенов референсов |

**Почему окно не меняет VRAM.** Полосы режут только внимание, а не активации: 1792p идёт
**10.3 ГиБ и с окном (314 с), и без него (376 с)**. Поэтому в модели окно стоит единственным
множителем времени (`bWindow = ln(314/376)`), а вклад в пик — ровно ноль.

**Почему `ref_px` клампится кадром.** `config.py::embedding_crop` считает
`ref_h = max(256, min(ref_px, crop_h))`, `ref_w = min(round(ref_h·ratio), crop_w)` с округлением до 32
(требование `image_processor`). Сетка референса **не бывает больше сетки кадра**: `ref_px: 0` —
это максимум точности, а `960` на 1440p — сознательно более грубый референс в обмен на KV
3.70 вместо 7.14 ГиБ. Каждая смена `ref_px` меняет ключ кэша эмбеддингов.

## Честные ограничения

- **Апскейла нет и в этой стадии не будет.** В YAML `upscale: null`, в `Q` нет «texture factor».
  Постобработка — отдельная стадия `postprocess` со своей моделью и своим бюджетом; в Phase A
  она выключена (`enabled: false`, `estimates: null`, причина `postprocess-not-connected`).
  Когда модель передадут, её параметры валидируются по **её** `ParameterSpec`-подобному набору,
  а не по полям ядра. Калькулятор никогда не «добирает» размер постобработкой молча.
- **16 бит — `unsupported`**: VAE декодирует в bf16, но сохранение идёт 8-битным PNG. YAML валиден,
  оценка памяти/времени корректна, размер файла ×2, в `warnings` — `bit-depth-16-unsupported`.
- **Потолок — реальный, а не консервативный**: до 3200×1792 (5.73 Мп, измерено 314 с / 10.3 ГиБ) —
  `ok`; 5.73–8.35 Мп (3840×2176, 513 с / 13.8 ГиБ) — `degraded` + `comfort-limit` (фактура уходит
  в рябь); выше — `impossible`.
- **Сетка канваса кратна 32**, поэтому «ровно 1920×1080» нативно не бывает: будет 1920×1088
  (+8 px) или 1920×1056; `snap.reachable` предлагает оба варианта.
- **`--aspect` знает только `16:9 | 4:3 | 1:1 | 2:3`**. Произвольная пропорция возможна только
  с референсом такой пропорции, иначе `aspect-unsupported-t2i` + ближайшие разрешённые.
- **`Q` — эвристика**: `√pixels · stepsFactor · attentionFactor · comfortFactor`, где
  `stepsFactor` (8→0.60 … 25→1.00) и часть `comfortFactor` помечены `assumption`. UI/`warnings`
  говорят об этом (`quality-heuristic`); A/B через `metrics.py` — этап C.
- **Оценки врут на чужом железе**: за конвертом замеров `confidence: "extrapolated"` и
  предупреждение `extrapolated-estimate`.
- **H3 на 32 ГиБ RAM / 12 ГиБ VRAM не советуется**: 30.9 ГиБ fp8-трансформера в RAM + 62 ГиБ
  энкодер не влезают — вердикт `impossible` (`h3-ram-32gb`, `h3-needs-16gib-vram`).
- **Место на диске по тарифам** — 74.9 / 54.9 / 181.0 ГиБ (`inventory.tiers`); `freeDiskGib`
  даёт `disk-not-enough`.
- **`frames` не принимается**: Phase A считает один кадр, набор умножает сервис.

## Калибровка

Цепочка одна: **`data/measurements.mjs` → `tools/fit.mjs` → `data/model.mjs` → `cost.mjs`/`quality.mjs`**.

- `src/data/measurements.mjs` — **руками**: 40 строк реальных прогонов (38 `stage: "render"`,
  в фит 34, в аудит 35), у каждой `source` — `<файл>#<якорь>`, `config` — резолвленные поля,
  `measured` — только снятые числа. Строки `stage: "postprocess"` (SeedVR2) складываются, но в фит
  рендера не попадают никогда. `suspect`-строки (противоречат двойнику того же конфига) в фит и
  агрегат не идут, а печатаются с причиной в `audit-measurements.mjs`.
- `tools/fit.mjs` — обычные нормальные уравнения + метод Гаусса, без зависимостей. Пики приводятся
  к метрике `nvidia-smi` измеренной поправкой **2.05 ГиБ** («свой процесс» → полное `memory.used`
  карты). `provenance` у каждой константы: `kind: measured | fitted | assumption`, `source`, `note`;
  `provenance.test.mjs` падает на константе без источника.
- `src/data/model.mjs` — сгенерированные коэффициенты, конверт и статистика. Сейчас: медиана
  `|err|` 4.1 % по 93 строкам аудита; VRAM — медиана 4.27 %, максимум 19.45 %; время — максимум
  21.72 %. Тесты держат медиану ≤ 15 % и максимум ≤ 30 % по VRAM и времени.
- `textTokens` по умолчанию **1500**: реальный промт проекта живёт между измеренными 232
  (`docs/bench/vram-tiers.md`) и ~1230 (`kv-cache-offload.md`) токенами; больше префикс — KV больше,
  то есть оценка консервативнее. Переопределяется через `render.textTokens`.
- Активации — **насыщающаяся** кривая `a0 + a1·x/(1 + a2·x)`, `x = tokens/1000`; время шага —
  `ln tStep` с изгибом по токенам, `[group]`, `[kv offload]`, `[h3]`, `[turbo]`, `ln(steps/fixedSteps)`
  и измеренным окном. `bWindow` — **измерен**: `ln(314/376)`, а не подобран (свободный подбор уводил
  его в плюс, компенсируя изгиб эластичности).

Перекалибровать:

```bash
# 1) добавить/поправить строку в src/data/measurements.mjs (source обязателен)
cd renderkit
npm run fit -- --check          # покажет, что модель устарела
npm run fit                     # перезаписать src/data/model.mjs
npm run audit:measurements      # est/measured/err по каждой строке
npm test                        # cost/provenance/snapshot
npm run snapshot                # обновить снапшоты (дифф читаемый)
```

После правок `qwenstyle/presets/*.yaml`, `qwenstyle/backends/*.py` или `docs/bench/*` прогоняются
`npm run inventory -- --check && npm run fit -- --check && npm test`.

**Добавить стадию или параметр**: стадия — запись в `Plan.stages` с `enabled`, `estimates` и
`paramsSchema` (как `postprocess.mjs`); параметр — поле в `schema.mjs`, `help.mjs`, при необходимости
`inventory`/эмиттер, плюс тесты (`schema`, `help`, `emit`, `snapshot`). Число в модель попадает
только через `measurements.mjs` + `fit.mjs`, не руками в `cost.mjs`.

## Подсказки

`src/data/help.mjs` — 22 записи на параметры формы и `Plan` (`width`, `height`, `quality`, `bitDepth`,
`model`, `vramGib`, `ramGib`, `disk`, `refs`, `anime`, `freeDiskGib`, `steps`, `offload`,
`offloadBlocks`, `encoderDevice`, `precision`, `loraScale`, `kvCache`, `vaeTilePx`, `window`, `refPx`,
`postprocess`). У каждой — три обязательных аспекта (**качество**, **производительность**,
**система**), `sources` (существующие строки замеров/документов) и `live` — id для what-if.

`help.test.mjs` проверяет: у каждого параметра все три аспекта непустые и ≤ 320 символов, `sources`
существуют, `live` — корректный id или `null`, в текстах нет имён пресетов. Живые цифры на
hover/фокус считает `web/viewmodel.mjs::hintLive(id, request)` одним вызовом `computePlan` с
изменённым параметром (дельта `+12 с`, `+2.13 ГиБ`, `Q 1920 → 1632`), а не ручные числа в вёрстке.
Кнопка `?` — `button` + `aria-describedby` + `role="tooltip"`; никакого `title=`.

## Анти-скоуп прототипа

Апскейл/постобработка как расчёт, 16-битный райтер, рендер в браузере, загрузка моделей,
БД/ORM, авторизация, бандлер, TypeScript-сборка, фреймворк, i18n, WebGPU-детект железа.
Сервис (`POST /api/plan`, `/api/validate`, `/api/rules`, `/api/options`) — этап B, после прототипа.
