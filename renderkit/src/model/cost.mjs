// @ts-check
/**
 * Оценки стоимости рендера: VRAM, RAM, время, диск + гейты (шаг 4 плана, ADR A4/A6).
 *
 * Формулы — те же, что откалиброваны в `tools/fit.mjs` (шаг 3), но перенесены в рантайм:
 * **все коэффициенты читаются из `src/data/model.mjs`**, здесь нет ни одного продублированного
 * числа фита. Единственные константы модуля — `COST_ASSUMPTIONS` (нулевой вклад окна в пик,
 * RAM-пул стриминга, хостовые активации, байт на пиксель PNG): у фита еще нет строк с пиком RAM
 * и размером файла, а «окно VRAM не меняет» — отдельный измеренный факт; записи — той же формы, что
 * `MODEL.provenance` (`value`/`unit`/`kind`/`source`/`note`), и переезжают в модель, когда такие
 * замеры/коэффициенты там появятся.
 *
 * ```
 * tokens      = (W/16)·(H/16)                                  # units.tokensFor, факт
 * prefix      = textTokens + Σ refs·tokensPerRef(refPx)         # 512 КиБ/токен; tokensPerRef — сетка
 *                                                              # refGrid (та же, что memory.ref)
 * peakVram    = resident(model, offload) + a0 + a1·x/(1 + a2·x)   # x = T/1000, активации насыщаются
 *             + (KV в VRAM ? kvGibPerToken·prefix : 0) + overhead
 * peakRam     = (offload group ? fp8 веса : 0) + (KV в RAM ? prefix·512 КиБ : 0)
 *             + (энкодер на CPU ? bf16 энкодера : pinned-пул) + хостовые активации
 * ln tStep    = b0 + b1·ln(T/4000) + b2·ln²(T/4000) + b3·[group] + b4·[KV в RAM] + b5·[h3]
 *             + b6·[turbo] + b7·ln(steps/fixedSteps) + b8·[окно] + b9·[h3]·ln(T/4000)
 *             + b10·[group]·ln(T/4000)
 * tGen        = steps·tStep
 * loadS       = coldS(model) + fp8Gib·(1/bw(disk) − 1/bw(nvme))
 * encodeColdS = base(устройство, модель) + encoderGib·(1/bw(disk) − 1/bw(nvme))
 * pngMb       = 1.5e-6·W·H·(bitDepth === 16 ? 2 : 1)
 * ```
 *
 * **Окно внимания в VRAM не входит** (`vram.windowEffectGib` = 0, измерено): 1792p — 10.3 ГиБ и с
 * окном (314 с), и без (376 с), то есть окно покупает время, а не память (`AGENTS.md` §4,
 * «Скользящее окно внимания»: «VRAM он не экономит»). Единственное место окна в модели — флаг
 * `[окно]` во времени шага.
 *
 * Гейты (плановые): `peak > 0.97·budget` — кандидат невалиден (`spills`), `> 0.90·budget` —
 * `degraded`; `confidence` — по конверту `MODEL.envelope` (`measured|interpolated|extrapolated`).
 *
 * Модуль изоморфный: без `node:`, `process` и DOM.
 */

import { MODEL } from '../data/model.mjs';
import { encoderInfo, modelCaps } from './inventory.mjs';
import { refGrid } from '../units.mjs';

/**
 * Границы гейтов по бюджету (доли): выше `invalid` кандидат отбрасывается, выше `degraded`
 * помечается «рискованным» (`degraded` — рендерится, но запаса нет).
 */
export const GATE = Object.freeze({ invalid: 0.97, degraded: 0.9 });

/**
 * Допущение о длине текстового префикса, токены.
 *
 * Префикс KV-кеша = текст промта + все референсы, и текст в замерах разный: у строк
 * `docs/bench/vram-tiers.md` это **232** токена (`MODEL.coefficients.textTokens` — измеренный
 * минимум), у строк `docs/bench/kv-cache-offload.md` — около **1230** (длинный промт). Реальный
 * промт проекта живёт между этими значениями (1–3 предложения на раздел, §5 AGENTS.md), поэтому
 * по умолчанию берётся **1500** токенов — середина диапазона 1000…4000 из «Модели оценок» плана:
 * это запас в безопасную сторону (больше префикс ⇒ больше KV ⇒ оценка консервативнее).
 * Запрос может переопределить значение (`render.textTokens`), тогда оценка сдвигается линейно.
 */
export const DEFAULT_TEXT_TOKENS = 1500;

/**
 * Константы cost-модели, которых нет в `src/data/model.mjs` (провенанс — как у `MODEL.provenance`).
 *
 * @type {Readonly<Record<string, {value: number, unit: string, kind: 'measured'|'fitted'|'assumption', source: string, note: string}>>}
 */
export const COST_ASSUMPTIONS = Object.freeze({
  'ram.pinnedPoolGib': {
    value: 2.4,
    unit: 'GiB',
    kind: 'assumption',
    source: 'AGENTS.md#§4 «GPU-стриминг энкодера»: pinned-плечо даёт ≈2.4 GiB peak_uss при 1.30× на H2D',
    note: 'pinned-staging энкодера ограничен пулом; при encoder.device=cuda модель поднимает в RAM '
      + 'не весь энкодер (62.1/16.7 ГиБ bf16), а только этот пул',
  },
  'ram.hostActivationsGib': {
    value: 2.4,
    unit: 'GiB',
    kind: 'assumption',
    source: 'AGENTS.md#§4 «GPU-стриминг энкодера»: peak_uss 4.72–4.83 GiB при pinned-пуле ≈2.4 GiB',
    note: 'torch + активации host-части сверх pinned-пула; из суммы peak_uss минус пул. '
      + 'При encoder.device=cpu не складывается с bf16-энкодером (там веса уже в RAM, но пик '
      + 'процесса растёт на ту же величину), поэтому учитывается в обеих ветках',
  },
  'disk.pngBytesPerPixel': {
    value: 1.5,
    unit: 'B/px',
    kind: 'assumption',
    source: '.pi/plans/render-settings-kit.md#«Факты»: PNG 1.39–1.70 Б/пиксель (2560×1440 → 5.1 МБ)',
    note: 'середина измеренного диапазона 1.39–1.70 Б/пиксель: пиксель 8 бит, файл сжимается '
      + 'примерно в 2.7 раза; 16 бит даёт ×2 к размеру того же кадра',
  },
  'kv.autoRefsOffload': {
    value: 2,
    unit: 'референса',
    kind: 'measured',
    source: 'docs/bench/kv-cache-offload.md#«Что это значит»: 2–3 рефа на 768p с KV в VRAM — OOM на шаге 0',
    note: 'правило авто-режима бэкенда `auto-offload-ref2`: 1 референс → K/V в VRAM, 2+ → в RAM. '
      + 'Это не эвристика, а измеренный отказ, поэтому бюджетная проверка его не отменяет',
  },
  'vram.windowEffectGib': {
    value: 0,
    unit: 'GiB',
    kind: 'measured',
    source: 'AGENTS.md#§4 «Скользящее окно внимания»: 1792p — 314 с против 376 с плотного, '
      + '«VRAM он не экономит в рабочей зоне: 1792p — 10.3 ГиБ и с ним, и без»',
    note: 'вклад окна внимания в пик VRAM — ноль: полосы режут только внимание, а не активации, '
      + 'поэтому плотное и оконное сочетание на одном канвасе имеют одинаковый пик (разница только '
      + 'во времени: −17 % у окна). Коэффициент вынесен сюда, чтобы разбор пика показывал эту статью '
      + 'явным нулём, а фит её не подбирал',
  },
});

/** Ярлыки к допущениям RAM/диска/окна (чтобы формулы читались). */
const PINNED_POOL_GIB = COST_ASSUMPTIONS['ram.pinnedPoolGib'].value;
const HOST_ACTIVATIONS_GIB = COST_ASSUMPTIONS['ram.hostActivationsGib'].value;
const PNG_BYTES_PER_PIXEL = COST_ASSUMPTIONS['disk.pngBytesPerPixel'].value;
const WINDOW_VRAM_EFFECT_GIB = COST_ASSUMPTIONS['vram.windowEffectGib'].value;

/**
 * С какого числа референсов авто-режим уводит K/V в RAM (`auto-offload-ref2`: 1 → VRAM, 2+ → RAM).
 * Измерено: 2–3 рефа на 768p с KV в VRAM — OOM на шаге 0 (`docs/bench/kv-cache-offload.md`).
 */
export const KV_AUTO_REFS_OFFLOAD = COST_ASSUMPTIONS['kv.autoRefsOffload'].value;

/**
 * @typedef {Object} Render
 * @property {string} model `qwen21-base` | `qwen21-turbo` | `h3-image`
 * @property {{width: number, height: number}} canvas native-канвас (`geometry.resolveGeometry`)
 * @property {number} tokens латентные токены канваса (`(W/16)·(H/16)`)
 * @property {number} steps шаги сэмплера (`levels.resolveLevels`)
 * @property {'none'|'group'} offload режим `generation.offload`
 * @property {boolean|'true'|'offload'|'false'|'auto-offload-ref2'|null} kvCache
 *   режим `generation.kv_cache`; `auto-offload-ref2`/`null` раскрывается здесь же
 *   (1 референс → VRAM, 2+ → RAM), чтобы оценка работала и до `memory.resolveKvCache`
 * @property {number} windowAttnPx ширина окна внимания, px (`0` = плотное внимание)
 * @property {number} [marginPx] запас окна, px (в оценку VRAM входит только режим окна)
 * @property {number|null} [vaeTilePx] тайл декодера VAE (`512|0|1024|-1`, `null` = не применимо)
 * @property {number} refs число референсов (0…8)
 * @property {number} refPx сетка референсов, px (`0` = сетка канваса)
 * @property {'cuda'|'cpu'} encoderDevice устройство энкодера (`auto` → `cuda` разрешает вызывающий)
 * @property {number} vramGib бюджет VRAM, ГиБ
 * @property {number} ramGib бюджет RAM, ГиБ
 * @property {'hdd'|'ssd'|'nvme'} disk класс накопителя
 * @property {number} [bitDepth] битность выхода (по умолчанию 8)
 * @property {number} [textTokens] длина текстового префикса (по умолчанию `DEFAULT_TEXT_TOKENS`)
 */

/**
 * @typedef {Object} Gate
 * @property {number} peakGib оценка пика, ГиБ
 * @property {number} budgetGib бюджет, ГиБ
 * @property {number} headroomGib свободный остаток бюджета, ГиБ
 * @property {number} headroomPct тот же остаток в процентах бюджета
 * @property {boolean} fits влезает (`peak ≤ 0.97·budget`)
 * @property {boolean} spills кандидат невалиден (`peak > 0.97·budget`)
 * @property {boolean} degraded запаса нет (`0.90·budget < peak ≤ 0.97·budget`)
 * @property {'ok'|'degraded'|'spill'} zone зона гейта
 */

/** @typedef {'qwen21-base'|'qwen21-turbo'|'h3-image'} ModelId */

/**
 * Режим KV-кеша по правилу бэкенда `auto-offload-ref2` (1 референс → VRAM, 2+ → RAM).
 * Это **режим**, а не бюджетное решение: влезает ли K/V в VRAM — решает `memory.resolveKvCache`
 * (и `autoKvMode` ниже для `estimateAll`).
 *
 * @param {Render} render
 * @returns {'true'|'offload'|'false'}
 */
function backendKvMode(render) {
  const raw = render.kvCache;
  if (raw === true || raw === 'true') return 'true';
  if (raw === 'offload') return 'offload';
  if (raw === false || raw === 'false') return 'false';
  if (raw === undefined || raw === null || raw === 'auto-offload-ref2') {
    return (render.refs ?? 0) >= KV_AUTO_REFS_OFFLOAD ? 'offload' : 'true';
  }
  throw new RangeError(`kvCache: неизвестный режим ${JSON.stringify(raw)}`);
}

/**
 * Бюджетно-осознанный авто-режим KV: 2+ референса → `offload` (измеренный OOM авто-режима),
 * иначе `true` только если `peak + kv ≤ 0.90·budget`, и `offload` в противном случае.
 * Та же политика, что в `memory.resolveKvCache`, — чтобы `estimateAll` не показывал спилл,
 * вызванный неразрешённым авто-режимом.
 *
 * @param {Render} render
 * @param {number} [prefix] токены префикса
 * @returns {'true'|'offload'}
 */
export function autoKvMode(render, prefix = prefixTokens(render)) {
  const explicit = backendKvMode(render);
  if (explicit !== 'true' && explicit !== 'offload') return explicit;
  if ((render.refs ?? 0) >= KV_AUTO_REFS_OFFLOAD) return 'offload';
  const kv = MODEL.coefficients.kvGibPerToken * prefix;
  const peak = estimateVram({ ...render, kvCache: 'offload' }, prefix) + kv;
  return peak <= GATE.degraded * render.vramGib ? 'true' : 'offload';
}

/**
 * Токены префикса KV-кеша: текст промта + все референсы.
 *
 * Стоимость референса — его латентная сетка из `memory.refGrid` (та же функция, что решает
 * `memory.ref.tokensPerRef` и `ref_px`): `tokensPerRef = (gridW/16)·(gridH/16)`. Так `ref_px 960`
 * на 16:9 — это сетка 1696×960 = 6360 токенов (KV 3.10 ГиБ), а не `(960/16)² = 3600` (2.49 ГиБ):
 * прежняя формула занижала префикс на треть и расходилась с `memory.ref` в одном и том же плане.
 * `refPx: 0` — сетка самого кадра, то есть `tokens` канваса.
 *
 * @param {Render} render
 * @returns {number} токены префикса
 */
export function prefixTokens(render) {
  const textTokens = render.textTokens ?? DEFAULT_TEXT_TOKENS;
  const refs = render.refs ?? 0;
  if (!Number.isFinite(textTokens) || textTokens < 0) {
    throw new RangeError(`prefixTokens: textTokens должен быть ≥ 0, получено ${textTokens}`);
  }
  if (!Number.isInteger(refs) || refs < 0) {
    throw new RangeError(`prefixTokens: refs должен быть целым ≥ 0, получено ${refs}`);
  }
  if (refs === 0) return textTokens;
  const canvas = render.canvas;
  if (!canvas || !Number.isFinite(canvas.width) || !Number.isFinite(canvas.height)) {
    throw new RangeError('prefixTokens: нужен canvas {width, height} — сетка референса '
      + 'считается по канвасу (memory.refGrid)');
  }
  const perRef = refGrid(render.refPx ?? 0, canvas).tokensPerRef;
  return textTokens + refs * perRef;
}

/**
 * Резидентные fp8-веса модели в VRAM при данном режиме offload.
 * Таблица — `MODEL.coefficients.vram.resident`; `null` = режим моделью не поддерживается.
 *
 * @param {string} model
 * @param {'none'|'group'} offload
 * @returns {number} ГиБ
 */
export function residentWeights(model, offload) {
  const table = /** @type {Record<string, Record<string, number|null>>} */ (
    MODEL.coefficients.vram.resident);
  const byMode = table[model];
  if (!byMode) {
    throw new RangeError(`нет резидентных весов для модели ${JSON.stringify(model)}; `
      + `доступны: ${Object.keys(table).join(', ')}`);
  }
  const value = byMode[offload];
  if (value === null || value === undefined) {
    throw new RangeError(`модель ${model} не умеет offload=${JSON.stringify(offload)} `
      + `(доступно: ${Object.entries(byMode).filter(([, v]) => v !== null).map(([k]) => k).join(', ')})`);
  }
  return value;
}

/**
 * @typedef {Object} VramBreakdown
 * @property {number} resident веса (fp8) в VRAM
 * @property {number} activations активации канваса (`a0 + a1·T/1000`)
 * @property {number} window вклад окна внимания в пик — **0** (измерено: окно VRAM не меняет)
 * @property {number} kv префикс KV в VRAM (0 при `offload`/`false`)
 * @property {number} overhead контекст CUDA + резерв аллокатора + рабочий стол
 * @property {number} peak сумма статей, ГиБ
 */

/**
 * Разбор пика VRAM по статьям — чтобы UI мог показать, «из чего пик» (и чтобы тесты видели KV).
 *
 * @param {Render} render
 * @param {number} [prefix] токены префикса (по умолчанию `prefixTokens(render)`)
 * @returns {VramBreakdown} ГиБ
 */
export function vramBreakdown(render, prefix = prefixTokens(render)) {
  const vram = MODEL.coefficients.vram;
  const tokens = Number(render.tokens);
  if (!Number.isFinite(tokens) || tokens <= 0) {
    throw new RangeError(`vramBreakdown: tokens должен быть > 0, получено ${render.tokens}`);
  }
  const thousands = tokens / 1000;
  const resident = residentWeights(render.model, render.offload);
  // Активации — насыщающаяся кривая `a1·x/(1 + a2·x)`: линейна на малых канвасах и выполаживается
  // на больших (измерено: 12.85 ГиБ при 14400 токенах, 12.35 при 22400, 15.85 при 32640).
  const activations = vram.activationBaseGib
    + (vram.activationGibPerMtoken * thousands) / (1 + vram.activationSaturationPerMtoken * thousands);
  // Окно внимания в пик VRAM не входит: замер даёт один и тот же пик с окном и без
  // (1792p — 10.3 ГиБ; одиночное отличие — время 314 с против 376 с). Статья оставлена в
  // разборе пика явным нулём (`vram.windowEffectGib`), чтобы UI показывал, что она проверена.
  const window = render.windowAttnPx > 0 ? WINDOW_VRAM_EFFECT_GIB : 0;
  const kv = backendKvMode(render) === 'true' ? MODEL.coefficients.kvGibPerToken * prefix : 0;
  const overhead = vram.overheadGib;
  return {
    resident,
    activations,
    window,
    kv,
    overhead,
    peak: resident + activations + window + kv + overhead,
  };
}

/**
 * Пик VRAM, ГиБ. Метрика — **полное `memory.used` карты** (`nvidia-smi`), то есть ответ на вопрос
 * «влезет ли в N ГиБ»; от `steps` не зависит (активации не накапливаются от шага к шагу).
 *
 * @param {Render} render
 * @param {number} [prefix] токены префикса (по умолчанию `prefixTokens(render)`)
 * @returns {number} ГиБ
 */
export function estimateVram(render, prefix = prefixTokens(render)) {
  return vramBreakdown(render, prefix).peak;
}

/**
 * Пик RAM, ГиБ: fp8-веса при групповом offload, KV-префикс при подкачке, энкодер
 * (bf16 целиком на CPU либо только pinned-пул на GPU) и хостовые активации.
 *
 * @param {Render} render
 * @param {number} [prefix] токены префикса (по умолчанию `prefixTokens(render)`)
 * @returns {number} ГиБ
 */
export function estimateRam(render, prefix = prefixTokens(render)) {
  const caps = modelCaps(render.model);
  const fp8InRam = render.offload === 'group' ? caps.weightGibFp8 : 0;
  const kvInRam = backendKvMode(render) === 'offload'
    ? MODEL.coefficients.kvGibPerToken * prefix
    : 0;
  const encoder = encoderInfo(caps.encoder);
  const encoderRam = render.encoderDevice === 'cpu' ? encoder.memory_gib : PINNED_POOL_GIB;
  return fp8InRam + kvInRam + encoderRam + HOST_ACTIVATIONS_GIB;
}

/**
 * Гейт по бюджету: `peak > 0.97·budget` — невалиден (`spills`), `> 0.90·budget` — `degraded`.
 *
 * @param {number} peakGib
 * @param {number} budgetGib
 * @returns {Gate}
 */
export function gateBudget(peakGib, budgetGib) {
  if (!Number.isFinite(peakGib) || peakGib < 0) {
    throw new RangeError(`gateBudget: peakGib должен быть ≥ 0, получено ${peakGib}`);
  }
  if (!Number.isFinite(budgetGib) || budgetGib <= 0) {
    throw new RangeError(`gateBudget: budgetGib должен быть > 0, получено ${budgetGib}`);
  }
  const spills = peakGib > GATE.invalid * budgetGib;
  const degraded = !spills && peakGib > GATE.degraded * budgetGib;
  const headroomGib = budgetGib - peakGib;
  return {
    peakGib,
    budgetGib,
    headroomGib,
    headroomPct: (headroomGib / budgetGib) * 100,
    fits: !spills,
    spills,
    degraded,
    zone: spills ? 'spill' : (degraded ? 'degraded' : 'ok'),
  };
}

/**
 * Ширина полосы чтения класса накопителя, ГиБ/с (`MODEL.coefficients.disk`).
 *
 * @param {string} disk
 * @returns {number}
 */
function bandwidthFor(disk) {
  const table = /** @type {Record<string, number>} */ (MODEL.coefficients.disk.bandwidthGibPerS);
  const value = table[disk];
  if (value === undefined) {
    throw new RangeError(`неизвестный класс накопителя ${JSON.stringify(disk)}; `
      + `доступны: ${Object.keys(table).join(', ')}`);
  }
  return value;
}

/**
 * Базовая цена холодного энкода на эталонном NVMe (до поправки на диск):
 * GPU-путь и 32B у H3 — I/O-bound константы, CPU-путь растёт линейно по токенам.
 *
 * @param {Render} render
 * @returns {number} с
 */
function encodeBaseS(render) {
  const enc = MODEL.coefficients.encode;
  if (render.encoderDevice === 'cpu') {
    return enc.cpuBaseS + enc.cpuPerMtokenS * (render.tokens / 1000);
  }
  if (render.model === 'h3-image') return enc.h3ColdS;
  return enc.coldGpuS;
}

/**
 * Время одного шага сэмплера по фиту, с. Первый шаг дороже (prefill) — это уже учтено
 * коэффициентом `bSteps` через нормировку на полное расписание модели.
 *
 * @param {Render} render
 * @returns {number} с/шаг
 */
export function estimateStepS(render) {
  const t = MODEL.coefficients.time;
  const caps = modelCaps(render.model);
  const lnTokens = Math.log(render.tokens / 4000);
  const group = render.offload === 'group' ? 1 : 0;
  const kvOffload = backendKvMode(render) === 'offload' ? 1 : 0;
  const h3 = render.model === 'h3-image' ? 1 : 0;
  const turbo = render.model === 'qwen21-turbo' ? 1 : 0;
  const window = render.windowAttnPx > 0 ? 1 : 0;
  const stepsNorm = Math.log(render.steps / caps.fixedSteps);
  return Math.exp(
    t.b0
    + t.bTokens * lnTokens
    + t.bTokens2 * lnTokens * lnTokens
    + t.bGroup * group
    + t.bKvOffload * kvOffload
    + t.bH3 * h3
    + t.bTurbo * turbo
    + t.bSteps * stepsNorm
    + t.bWindow * window
    + t.bH3Tokens * h3 * lnTokens
    + t.bGroupTokens * group * lnTokens,
  );
}

/**
 * @typedef {Object} TimeEstimate
 * @property {number} loadS загрузка модели (импорт, fp8-кэш, LoRA, размещение)
 * @property {number} encodeColdS энкод промта при промахе кэша эмбеддингов
 * @property {number} encodeWarmS энкод при попадании в кэш эмбеддингов (`0` — измерено)
 * @property {number} genPerFrameS генерация одного кадра (`steps · tStep`)
 * @property {number} firstStepS **первый кадр серии** целиком: загрузка + холодный энкод + генерация
 * @property {number} warmFrameS **прогретый кадр серии**: энкод из кэша + генерация
 */

/**
 * Время по фазам. Загрузка и холодный энкод линейно зависят от класса накопителя:
 * `base + прочитанные ГиБ·(1/bw(disk) − 1/bw(nvme))`, то есть на NVMe оценка совпадает
 * с калибровкой бит-в-бит, а на SSD/HDD хуже ровно на цену чтения.
 *
 * @param {Render} render
 * @param {'hdd'|'ssd'|'nvme'} [disk] класс накопителя (по умолчанию `render.disk`)
 * @param {number} [prefix] токены префикса — параметр оставлен для симметрии с `estimateVram`:
 *   в текущем фите время шага зависит от **режима** KV, а не от длины префикса
 *   (замеры не показали зависимости; `bKvOffload` — флаг режима)
 * @returns {TimeEstimate}
 */
export function estimateTime(render, disk = render.disk ?? 'nvme', prefix = prefixTokens(render)) {
  void prefix; // см. JSDoc: длина префикса в модель времени не входит
  const coefficients = MODEL.coefficients;
  const caps = modelCaps(render.model);
  const bandwidth = bandwidthFor(disk);
  const reference = bandwidthFor('nvme');
  const readPenalty = 1 / bandwidth - 1 / reference;

  const loadBase = /** @type {Record<string, number>} */ (coefficients.load.coldS)[render.model];
  if (loadBase === undefined) {
    throw new RangeError(`нет оценки загрузки для модели ${JSON.stringify(render.model)}`);
  }
  const weightsGib = caps.weightGibFp8;

  const encoder = encoderInfo(caps.encoder);
  const loadS = loadBase + weightsGib * readPenalty;
  const encodeColdS = encodeBaseS(render) + encoder.memory_gib * readPenalty;
  const encodeWarmS = coefficients.encode.warmS;

  const genPerFrameS = render.steps * estimateStepS(render);
  return {
    loadS,
    encodeColdS,
    encodeWarmS,
    genPerFrameS,
    firstStepS: loadS + encodeColdS + genPerFrameS,
    warmFrameS: encodeWarmS + genPerFrameS,
  };
}

/**
 * @typedef {Object} DiskEstimate
 * @property {number} pngPerFrameMb размер PNG одного кадра, МБ (десятичные, как в плане)
 * @property {number} bitDepthFactor множитель битности (`8 → 1`, `16 → 2`)
 */

/**
 * Оценка диска: размер выходного файла одного кадра.
 *
 * @param {Render} render
 * @param {number} [bitDepth] битность выхода (по умолчанию `render.bitDepth ?? 8`)
 * @returns {DiskEstimate}
 */
export function estimateDisk(render, bitDepth = render.bitDepth ?? 8) {
  if (bitDepth !== 8 && bitDepth !== 16) {
    throw new RangeError(`estimateDisk: bitDepth должен быть 8 или 16, получено ${bitDepth}`);
  }
  const { width, height } = render.canvas;
  const bitDepthFactor = bitDepth === 16 ? 2 : 1;
  return {
    pngPerFrameMb: (PNG_BYTES_PER_PIXEL * width * height * bitDepthFactor) / 1e6,
    bitDepthFactor,
  };
}

/**
 * @typedef {Object} Confidence
 * @property {'measured'|'interpolated'|'extrapolated'} level
 * @property {string[]} reasons что именно вывело оценку из конверта/замеров
 */

/**
 * Уверенность оценки по конверту `MODEL.envelope`:
 * - `extrapolated` — хоть одна ось вне конверта (модель, токены, шаги, референсы, VRAM, RAM,
 *   окно, тайл VAE);
 * - `interpolated` — всё внутри конверта, но расписание шагов не из наблюдённых
 *   (`fixedSteps` модели и границы `envelope.steps`) — так работают явные `image.steps`;
 * - `measured` — точка внутри конверта на наблюдённом расписании.
 *
 * @param {Render} render
 * @returns {Confidence}
 */
export function estimateConfidence(render) {
  const env = MODEL.envelope;
  const caps = modelCaps(render.model);
  const refs = render.refs ?? 0;
  /** @type {string[]} */
  const reasons = [];
  const check = (/** @type {boolean} */ ok, /** @type {string} */ reason) => {
    if (!ok) reasons.push(reason);
  };

  check(env.models.includes(render.model), `модель ${render.model} вне конверта (${env.models.join(', ')})`);
  check(render.tokens >= env.tokens.min && render.tokens <= env.tokens.max,
    `токены ${render.tokens} вне ${env.tokens.min}…${env.tokens.max}`);
  check(render.steps >= env.steps.min && render.steps <= env.steps.max,
    `шаги ${render.steps} вне ${env.steps.min}…${env.steps.max}`);
  check(refs >= env.refs.min && refs <= env.refs.max, `референсы ${refs} вне ${env.refs.min}…${env.refs.max}`);
  check(render.vramGib >= env.vramGib.min && render.vramGib <= env.vramGib.max,
    `VRAM ${render.vramGib} ГиБ вне ${env.vramGib.min}…${env.vramGib.max}`);
  check(render.ramGib >= env.ramGib.min && render.ramGib <= env.ramGib.max,
    `RAM ${render.ramGib} ГиБ вне ${env.ramGib.min}…${env.ramGib.max}`);
  if (render.windowAttnPx > 0) {
    check(render.windowAttnPx >= env.windowAttnPx.min && render.windowAttnPx <= env.windowAttnPx.max,
      `окно ${render.windowAttnPx} px вне ${env.windowAttnPx.min}…${env.windowAttnPx.max}`);
  }
  if (render.vaeTilePx !== null && render.vaeTilePx !== undefined) {
    check(env.vaeTilePx.includes(render.vaeTilePx),
      `тайл VAE ${render.vaeTilePx} не мерился (в замерах ${env.vaeTilePx.join(', ')})`);
  }
  if (reasons.length > 0) return { level: 'extrapolated', reasons };

  const measuredSteps = new Set([caps.fixedSteps, env.steps.min, env.steps.max]);
  if (!measuredSteps.has(render.steps)) {
    reasons.push(`шаги ${render.steps} не из наблюдённых расписаний `
      + `(${[...measuredSteps].join(', ')}): время шага интерполируется по bSteps`);
    return { level: 'interpolated', reasons };
  }
  return { level: 'measured', reasons };
}

/**
 * @typedef {Object} Estimates
 * @property {Gate} vram пик/бюджет/запас VRAM + гейты
 * @property {Gate} ram пик/бюджет/запас RAM + гейты
 * @property {TimeEstimate} time фазы
 * @property {DiskEstimate} disk размер файла кадра
 * @property {'measured'|'interpolated'|'extrapolated'} confidence уверенность оценки
 * @property {string[]} confidenceReasons что вывело оценку из конверта
 * @property {number} prefixTokens токены префикса KV, на которых считалась оценка
 * @property {'true'|'offload'|'false'} kvMode режим KV, с которым посчитан пик VRAM
 */

/**
 * Все оценки сразу: VRAM/RAM с гейтами, время по фазам, диск и уверенность.
 *
 * Авто-режим KV (`kvCache: null`/`auto-offload-ref2`) разрешается **бюджетно**: 1 референс → VRAM,
 * если `peak + kv ≤ 0.90·budget`, иначе RAM (2+ референса → RAM всегда). Готовое решение
 * `memory.resolveKvCache` можно передать явно (`options.kvCache`) — тогда оно и используется.
 *
 * @param {Render} render
 * @param {{prefix?: number, disk?: 'hdd'|'ssd'|'nvme', bitDepth?: number, kvCache?: 'true'|'offload'|'false'}} [options]
 * @returns {Estimates}
 */
export function estimateAll(render, options = {}) {
  const prefix = options.prefix ?? prefixTokens(render);
  const disk = options.disk ?? render.disk ?? 'nvme';
  const bitDepth = options.bitDepth ?? render.bitDepth ?? 8;
  const kvMode = options.kvCache ?? autoKvMode(render, prefix);
  const withKv = { ...render, kvCache: kvMode };
  const confidence = estimateConfidence(withKv);
  return {
    vram: gateBudget(estimateVram(withKv, prefix), render.vramGib),
    ram: gateBudget(estimateRam(withKv, prefix), render.ramGib),
    time: estimateTime(withKv, disk, prefix),
    disk: estimateDisk(withKv, bitDepth),
    confidence: confidence.level,
    confidenceReasons: confidence.reasons,
    prefixTokens: prefix,
    kvMode,
  };
}
