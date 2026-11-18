// @ts-check
/**
 * Объяснение плана (шаг 5, ADR A6/A7/A10b): `trace` по ключевым полям, `degradations` (ладдер),
 * `suggestions` (что не применили и почему) и `warnings` (коды плана).
 *
 * Объяснение — часть продукта, а не отладка: каждое решение по раскладке памяти видно с ценой
 * (`ГиБ`, секунды, `%`), а у каждого отклонённого варианта есть причина. Числа берутся из
 * `data/measurements.mjs` там, где замер существует (`measuredS`), иначе из оценок `cost.mjs` —
 * поэтому `measuredS` заполняется **только** при точном совпадении конфига с замером.
 *
 * Модуль изоморфный: без `node:`/DOM.
 */

import { MODEL } from '../data/model.mjs';
import { MEASUREMENTS } from '../data/measurements.mjs';
import { GATE, estimateAll, estimateVram, prefixTokens, residentWeights } from './cost.mjs';
import { stepsAxis } from './candidates.mjs';
import { diskFootprint, tierFor } from './inventory.mjs';
import { VAE_TILES } from './memory.mjs';

/** @typedef {import('./geometry.mjs').Geometry} Geometry */
/** @typedef {import('./levels.mjs').Levels} Levels */
/** @typedef {import('./candidates.mjs').Candidate} Candidate */
/** @typedef {import('./candidates.mjs').CandidatesResult} CandidatesResult */
/** @typedef {import('./candidates.mjs').Canonical} Canonical */
/** @typedef {import('../schema.mjs').NormalizedRequest} NormalizedRequest */
/** @typedef {import('./postprocess.mjs').PostprocessStage} PostprocessStage */

/** H3: минимальный проверенный бюджет VRAM, ГиБ (`AGENTS.md#§0`: профиль `VRAM16`). */
export const H3_MIN_VRAM_GIB = 16;

/** H3: минимальный проверенный бюджет RAM, ГиБ (`AGENTS.md#§0`: на 32 ГиБ — спилл 30.9 ГиБ fp8). */
export const H3_MIN_RAM_GIB = 64;

/**
 * Коды предупреждений плана (полный словарь из плана; часть выдаётся геометрией, часть — здесь).
 *
 * @type {Readonly<Record<string, string>>}
 */
export const WARNING_CODES = Object.freeze({
  NATIVE_SIZE_SNAPPED: 'native-size-snapped',
  ASPECT_UNSUPPORTED_T2I: 'aspect-unsupported-t2i',
  REF_RATIO_MUST_MATCH: 'ref-ratio-must-match',
  COMFORT_LIMIT: 'comfort-limit',
  NO_NATIVE_PATH_FOR_SIZE: 'no-native-path-for-size',
  BIT_DEPTH_16_UNSUPPORTED: 'bit-depth-16-unsupported',
  POSTPROCESS_NOT_CONNECTED: 'postprocess-not-connected',
  KV_VRAM_VS_RAM: 'kv-vram-vs-ram',
  VAE_TILE_GRID: 'vae-tile-grid',
  VAE_TILE_OOM_RISK: 'vae-tile-oom-risk',
  WINDOW_COVERAGE_LOW: 'window-coverage-low',
  WINDOW_INERT: 'window-inert',
  REF_PX_CLAMPED: 'ref-px-clamped',
  REFS_TOKEN_CAP: 'refs-token-cap',
  H3_RAM_32GB: 'h3-ram-32gb',
  H3_NEEDS_16GIB_VRAM: 'h3-needs-16gib-vram',
  VRAM_SPILL_RISK: 'vram-spill-risk',
  RAM_PAGECACHE_THRASH: 'ram-pagecache-thrash',
  HDD_FP8_LOAD: 'hdd-fp8-load',
  HDD_ENCODER_STREAM: 'hdd-encoder-stream',
  ENCODER_CPU_SLOW: 'encoder-cpu-slow',
  DISK_NOT_ENOUGH: 'disk-not-enough',
  QUALITY_HEURISTIC: 'quality-heuristic',
  EXTRAPOLATED_ESTIMATE: 'extrapolated-estimate',
  ANIME_LORA_UNSUPPORTED: 'anime-lora-unsupported',
});

/**
 * @typedef {Object} Warning
 * @property {string} code код из словаря плана
 * @property {string} field поле плана, к которому относится предупреждение
 * @property {string} detail пояснение с числами
 */

/**
 * @typedef {Object} Suggestion
 * @property {string} code `smaller-canvas` | `other-hardware` | `use-turbo` | `quality-max` |
 *   `enable-postprocess`
 * @property {string} field поле запроса, которое предлагается изменить
 * @property {unknown} value предлагаемое значение
 * @property {string} detail почему и что это даёт
 */

/**
 * @typedef {Object} Degradation
 * @property {string} code id ступени ладдера (`kv-offload`, `offload-group`, `vae-tile-512`, …)
 * @property {string} ruleId правило, которым ступень обоснована
 * @property {string} field что именно ослаблено
 * @property {unknown} from исходное значение запроса
 * @property {unknown} to значение, на которое спустились
 * @property {string} reason причина с числами
 */

/**
 * @typedef {Object} TraceRejected
 * @property {unknown} value отклонённое значение
 * @property {string} why причина
 * @property {number} [measuredS] измеренная генерация этого варианта (только при точном совпадении с замером)
 */

/**
 * @typedef {Object} TraceEntry
 * @property {string} field путь поля в плане
 * @property {unknown} value выбранное значение
 * @property {string} ruleId правило плана (`A4.canvas`, `A7.kv`, …)
 * @property {string} source источник числа/правила
 * @property {TraceRejected[]} rejected отклонённые варианты с причиной
 */

/**
 * @typedef {Object} ExplainInput
 * @property {NormalizedRequest} request
 * @property {Geometry} geometry
 * @property {Levels} levels
 * @property {CandidatesResult} result
 * @property {Candidate|null} best победитель (`null` — валидных сочетаний нет)
 * @property {Array<Degradation>} ladder применённые ступени ладдера (в порядке плана)
 * @property {string} verdict
 * @property {'measured'|'interpolated'|'extrapolated'} confidence
 * @property {PostprocessStage} postprocess
 * @property {{smallerCanvas: null|{width: number, height: number},
 *   biggerHardware: null|{field: string, value: number}, greyZoneBlocked: boolean}} probes
 *   что удалось найти «сбоку» для подсказок
 */

/** @param {number} value @param {number} [digits] */
function round(value, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

/**
 * Измеренная генерация кадра для конфига — **только** при точном совпадении с замером
 * (`data/measurements.mjs`, стадия `render`). Иначе `undefined`: оценка в `measuredS` не подставляется.
 *
 * @param {Record<string, any>} render
 * @returns {number|undefined}
 */
export function measuredGenS(render) {
  for (const row of MEASUREMENTS) {
    if (row.stage !== 'render') continue;
    const config = /** @type {any} */ (row.config);
    if (config.model !== render.model) continue;
    if (config.tokens !== render.tokens) continue;
    if (config.steps !== render.steps) continue;
    if ((config.offload ?? null) !== (render.offload ?? null)) continue;
    if ((config.windowAttnPx ?? 0) !== (render.windowAttnPx ?? 0)) continue;
    if ((config.vaeTilePx ?? null) !== (render.vaeTilePx ?? null)) continue;
    // Режим KV в замере меняет только размещение K/V, но если он записан — сверяем и его,
    // иначе `measuredS` приписал бы замер другому конфигу.
    if (config.kvCache !== null && config.kvCache !== undefined
      && config.kvCache !== String(render.kvCache)) continue;
    if (row.measured.genS === undefined) continue;
    return row.measured.genS;
  }
  return undefined;
}

/**
 * Предупреждения плана: сначала геометрия (A4), затем битность, стадия postprocess, раскладка
 * памяти (A7) и железо. Коды не дублируются; порядок детерминирован.
 *
 * @param {ExplainInput} input
 * @returns {Warning[]}
 */
export function buildWarnings(input) {
  const {
    request, geometry, result, best, verdict, confidence,
  } = input;
  const chosen = best ?? result.canonical;
  const budget = request.hardware.vramGib;
  /** @type {Warning[]} */
  const warnings = [];
  /** @param {string} code @param {string} detail @param {string} [field] */
  const push = (code, detail, field = 'image') => {
    if (warnings.some((warning) => warning.code === code)) return;
    warnings.push({ code, field, detail });
  };

  for (const warning of geometry.warnings) push(warning.code, warning.detail);

  if (request.image.bitDepth === 16) {
    push(WARNING_CODES.BIT_DEPTH_16_UNSUPPORTED,
      'пайплайн сохраняет 8-битный PNG: 16 бит не реализованы. Оценки памяти и времени остаются '
      + 'верными, размер файла в `estimates.disk` учтён ×2', 'image.bitDepth');
  }
  const postprocessRequested = request.postprocess !== null && request.postprocess !== undefined;
  push(WARNING_CODES.POSTPROCESS_NOT_CONNECTED,
    postprocessRequested
      ? `стадия \`postprocess\` запрошена (модель «${request.postprocess.model}»), но модели `
        + 'постобработки нет в inventory: план посчитан только по рендер-стадии, estimates/paramsSchema '
        + '= null — апскейлом размер не «добирается»'
      : 'стадия `postprocess` не считается: модель постобработки не передана, апскейлом размер не «добирается»',
    'postprocess');
  push(WARNING_CODES.QUALITY_HEURISTIC,
    'Q — эвристика A6 (√пикселей × шаги × внимание × комфорт); факторы помечены `assumption` до A/B (`metrics.py`)',
    'quality');

  // Когда валидного сочетания нет, раскладка памяти не выбиралась: предупреждения о ней
  // описывали бы несуществующий план. Вместо них — один честный код про пик; предупреждения
  // про железо (диск, энкодер, H3) остаются: они объясняют, в каких условиях считали.
  if (verdict === 'impossible') {
    const peak = result.canonical.estimates.vram.peakGib;
    push(WARNING_CODES.VRAM_SPILL_RISK,
      `ни одно сочетание не влезает: пик ${round(peak)} ГиБ против порога `
      + `${round(result.limitGib)} ГиБ (0.97·${round(budget)}${result.greyZone ? ', «серая» зона — 1.00' : ''}); `
      + 'ладдер исчерпан', 'estimates.vram');
  } else {
    const kv = chosen.layout.kvCache;
    if (kv.applicable && kv.gib > 0) {
      if (kv.mode === 'offload' && kv.peakWithKvGib <= GATE.invalid * budget) {
        push(WARNING_CODES.KV_VRAM_VS_RAM,
          `K/V префикса (${round(kv.gib)} ГиБ) влезли бы в VRAM (пик ${round(kv.peakWithKvGib)} ГиБ), `
          + 'но запас < 10 %: выбрана подкачка из RAM '
          + `(+${kv.costPct} % времени, картинка бит-в-бит)`, 'render.memory.kvCache');
      } else if (kv.mode === 'true' && kv.peakWithKvGib > GATE.degraded * budget) {
        push(WARNING_CODES.KV_VRAM_VS_RAM,
          `K/V в VRAM (${round(kv.gib)} ГиБ) оставляют ${round(budget - kv.peakWithKvGib)} ГиБ запаса `
          + `(пик ${round(kv.peakWithKvGib)} из ${round(budget)}): замер 768p — 14.11 из 16 ГиБ; `
          + 'подкачка из RAM даёт +13 % времени, но освобождает VRAM', 'render.memory.kvCache');
      }
    }

    const vae = chosen.layout.vae;
    if (vae.tilePx === 0) {
      push(WARNING_CODES.VAE_TILE_GRID,
        `тайл 256/192: сетка 16 px на ровных поверхностях ×${VAE_TILES[0].gridArtifact} `
        + `(против ×${VAE_TILES[512].gridArtifact} у 512/384)`, 'render.memory.vae.tilePx');
    }
    if (vae.oom === true || vae.rejected !== null) {
      push(WARNING_CODES.VAE_TILE_OOM_RISK,
        vae.reason, 'render.memory.vae.tilePx');
    }

    const attention = chosen.layout.attention;
    if (attention.reason.startsWith('window-coverage-low')) {
      push(WARNING_CODES.WINDOW_COVERAGE_LOW, attention.reason, 'render.memory.attention.mode');
    }
    if (attention.equivalentToDense) {
      push(WARNING_CODES.WINDOW_INERT, attention.reason, 'render.memory.attention.mode');
    }

    const ref = chosen.layout.ref;
    if (ref.clamped) push(WARNING_CODES.REF_PX_CLAMPED, ref.reason, 'render.memory.ref.px');
    if (ref.reduced) push(WARNING_CODES.REFS_TOKEN_CAP, ref.reason, 'render.memory.ref.px');

    if (result.greyZone && chosen.estimates.vram.peakGib > GATE.invalid * budget) {
      push(WARNING_CODES.VRAM_SPILL_RISK,
        `«серая» зона: пик ${round(chosen.estimates.vram.peakGib)} из ${round(budget)} ГиБ `
        + '(гейт 0.97 смягчён до 1.00) — прогон на грани, активации могут уйти в спилл',
        'estimates.vram');
    }
    if (chosen.estimates.ram.zone !== 'ok') {
      push(WARNING_CODES.RAM_PAGECACHE_THRASH,
        `пик RAM ${round(chosen.estimates.ram.peakGib)} из ${round(request.hardware.ramGib)} ГиБ: `
        + 'page cache fp8-весов вытесняется — шаги замедляются в разы', 'estimates.ram');
    }
  }

  // Железо: эти предупреждения не про выбранную раскладку, а про условия прогона, поэтому
  // выдаются и при `impossible`.
  if (request.hardware.disk === 'hdd') {
    push(WARNING_CODES.HDD_FP8_LOAD,
      'HDD: чтение fp8-кэша трансформера (6.6 ГиБ) — ~50 с на старт против ~2 с на NVMe',
      'hardware.disk');
    if (chosen.render.encoderDevice === 'cuda') {
      push(WARNING_CODES.HDD_ENCODER_STREAM,
        'HDD: GPU-стриминг энкодера читает веса с диска каждый forward — режим I/O-bound, '
        + '3.0 ГиБ/с NVMe против 0.13 ГиБ/с на HDD', 'hardware.disk');
    }
  }
  if (chosen.render.encoderDevice === 'cpu') {
    push(WARNING_CODES.ENCODER_CPU_SLOW,
      'энкодер считается на CPU (VRAM ≤ 8 ГиБ): холодный энкод 55–255 с на пару '
      + '(промт, картинка, резолюция)', 'render.encoder.device');
  }
  if (request.hardware.model === 'h3-image') {
    if (request.hardware.ramGib < H3_MIN_RAM_GIB) {
      push(WARNING_CODES.H3_RAM_32GB,
        `H3-Image: групповой offload держит 30.9 ГиБ fp8-трансформера в RAM, а энкодер — 62.1 ГиБ; `
        + `${request.hardware.ramGib} ГиБ RAM даёт спилл`, 'hardware.ramGib');
    }
    if (request.hardware.vramGib < H3_MIN_VRAM_GIB) {
      push(WARNING_CODES.H3_NEEDS_16GIB_VRAM,
        `H3-Image: измерен только на ${H3_MIN_VRAM_GIB} ГиБ VRAM (fp8 30.9 ГиБ + энкодер Qwen3-VL-32B)`,
        'hardware.vramGib');
    }
  }
  if (request.anime && chosen.render.loraScale === 0) {
    push(WARNING_CODES.ANIME_LORA_UNSUPPORTED,
      `у модели ${request.hardware.model} нет стилевого адаптера: \`anime: true\` не даёт LoRA`,
      'anime');
  }
  if (request.freeDiskGib !== null && request.freeDiskGib !== undefined) {
    const needed = diskFootprint(tierFor(budget));
    if (needed > request.freeDiskGib) {
      push(WARNING_CODES.DISK_NOT_ENOUGH,
        `полный след весов и кэшей ближайшего тарифа — ${round(needed, 1)} ГиБ, свободно `
        + `${round(request.freeDiskGib, 1)} ГиБ`, 'freeDiskGib');
    }
  }
  if (confidence === 'extrapolated' && verdict !== 'impossible') {
    push(WARNING_CODES.EXTRAPOLATED_ESTIMATE,
      'запрос вне конверта калибровки: оценки помечены `extrapolated` (числа могут врать на чужом железе)',
      'estimates');
  }
  return warnings;
}

/**
 * Подсказки «что ещё можно сделать» — только **не применённые** варианты: turbo для черновика,
 * меньший кадр, другое железо, уровень `max` для «серой» зоны, стадия `postprocess`.
 *
 * @param {ExplainInput} input
 * @returns {Suggestion[]}
 */
export function buildSuggestions(input) {
  const {
    request, levels, verdict, probes,
  } = input;
  /** @type {Suggestion[]} */
  const suggestions = [];

  if (levels.level === 'draft' && request.hardware.model === 'qwen21-base') {
    suggestions.push({
      code: 'use-turbo',
      field: 'hardware.model',
      value: 'qwen21-turbo',
      detail: 'черновик на turbo дешевле: 8 шагов — расписание модели (640p/8 ≈ 5 с против ≈ 15 с у base), '
        + 'кэш эмбеддингов тот же',
    });
  }
  if (probes.smallerCanvas !== null) {
    const { width, height } = probes.smallerCanvas;
    suggestions.push({
      code: 'smaller-canvas',
      field: 'image',
      value: `${width}×${height}`,
      detail: `самый большой native-кадр этой пропорции, который влезает в ${round(request.hardware.vramGib)} ГиБ `
        + `(${width}×${height}); потолок между 5.73 и 8.35 Мп — «серая» зона`,
    });
  }
  if (probes.biggerHardware !== null) {
    suggestions.push({
      code: 'other-hardware',
      field: probes.biggerHardware.field,
      value: probes.biggerHardware.value,
      detail: `тот же кадр становится выполнимым при ${probes.biggerHardware.field} = ${probes.biggerHardware.value}`,
    });
  }
  if (probes.greyZoneBlocked) {
    suggestions.push({
      code: 'quality-max',
      field: 'image.quality',
      value: 'max',
      detail: 'размер в «серой» зоне (> 5.73 Мп): уровень standard/draft её не покупает, max разрешает '
        + '(рендерится, но фактура уходит в рябь — `comfort-limit`)',
    });
  }
  if (verdict === 'impossible' || verdict === 'degraded') {
    const requestedModel = request.postprocess === null || request.postprocess === undefined
      ? null
      : request.postprocess.model;
    suggestions.push({
      code: 'enable-postprocess',
      field: 'postprocess',
      value: null,
      detail: requestedModel === null
        ? 'нативный путь ограничен; «добрать» размер можно только отдельной стадией `postprocess` — '
          + 'модель не передана, поэтому стадия выключена'
        : `стадия \`postprocess\` уже запрошена («${requestedModel}»), но модели нет в inventory: `
          + 'её расчёт и параметры появятся, когда её подключат — сейчас план считает только рендер',
    });
  }
  return suggestions;
}

/**
 * Деградации — применённые ступени ладдера в фиксированном порядке плана.
 *
 * @param {ExplainInput} input
 * @returns {Degradation[]}
 */
export function buildDegradations(input) {
  return input.ladder.map((step) => ({
    code: step.code,
    ruleId: step.ruleId,
    field: step.field,
    from: step.from,
    to: step.to,
    reason: step.reason,
  }));
}

/**
 * Причина, почему валидный вариант проиграл: числа, а не «хуже».
 *
 * @param {Candidate} alt
 * @param {{metrics: {q: number, timeS: number, headroomGib: number}}} chosen
 * @param {Levels} levels
 * @returns {string}
 */
function whyAlternative(alt, chosen, levels) {
  return `проходит гейты, но проигрывает по весам ${levels.level}: `
    + `Q ${Math.round(alt.metrics.q)} против ${Math.round(chosen.metrics.q)}, `
    + `время ${round(alt.metrics.timeS, 1)} с против ${round(chosen.metrics.timeS, 1)} с, `
    + `запас VRAM ${round(alt.metrics.headroomGib)} против ${round(chosen.metrics.headroomGib)} ГиБ`;
}

/**
 * `trace` по восьми ключевым полям плана: канвас, шаги, KV, тайл VAE, внимание, `ref_px`,
 * offload и устройство энкодера. У каждого — правило, источник и отклонённые варианты.
 *
 * @param {ExplainInput} input
 * @returns {TraceEntry[]}
 */
export function buildTrace(input) {
  const {
    request, geometry, levels, result, best,
  } = input;
  const chosen = /** @type {Candidate|Canonical} */ (best ?? result.canonical);
  const render = chosen.render;
  const layout = chosen.layout;
  const candidates = result.candidates;
  /** @type {TraceEntry[]} */
  const trace = [];

  /** @param {(candidate: Candidate) => boolean} predicate */
  const findAlternative = (predicate) => candidates.find((candidate) => predicate(candidate)) ?? null;

  // 1. Канвас: `sizePx = H` запроса, сетка 32, снап и соседние точки.
  /** @type {TraceRejected[]} */
  const canvasRejected = [];
  if (geometry.snap.dx !== 0 || geometry.snap.dy !== 0) {
    canvasRejected.push({
      value: `${request.image.width}×${request.image.height}`,
      why: 'запрошено, но не на сетке 32: пайплайн округляет `calculate_dimensions` '
        + `(dx ${geometry.snap.dx > 0 ? '+' : ''}${geometry.snap.dx}, dy ${geometry.snap.dy > 0 ? '+' : ''}${geometry.snap.dy})`,
    });
  }
  for (const [width, height] of geometry.snap.reachable) {
    canvasRejected.push({
      value: `${width}×${height}`,
      why: 'соседняя точка сетки 32 той же пропорции: другая высота кадра, другой энкод',
    });
  }
  trace.push({
    field: 'image.canvas',
    value: {
      width: geometry.canvas.width,
      height: geometry.canvas.height,
      tokens: geometry.tokens,
      snap: { dx: geometry.snap.dx, dy: geometry.snap.dy },
    },
    ruleId: 'A4.canvas',
    source: 'units.canvasFor — копия calculate_dimensions(sizePx = H, ratio), сетка 32 (pyRound)',
    rejected: canvasRejected,
  });

  // 2. Шаги: расписание уровня, не рычаг памяти (решение 5).
  /** @type {TraceRejected[]} */
  const stepsRejected = [];
  for (const value of stepsAxis(levels)) {
    if (value === render.steps) continue;
    const alt = findAlternative((candidate) => candidate.render.steps === value);
    const measured = measuredGenS({ ...render, steps: value });
    stepsRejected.push({
      value,
      why: alt
        ? whyAlternative(alt, chosen, levels)
        : 'не проходит гейты памяти/уровня для этого запроса',
      ...(measured === undefined ? {} : { measuredS: measured }),
    });
  }
  trace.push({
    field: 'render.steps',
    value: render.steps,
    ruleId: 'A3.level',
    source: levels.reason,
    rejected: stepsRejected,
  });

  // 3. KV-кеш: A7 — `true` только при `peak + kv ≤ 0.90·budget`.
  const kv = layout.kvCache;
  /** @type {TraceRejected[]} */
  const kvRejected = [];
  if (kv.applicable && kv.mode !== null) {
    const alternative = kv.mode === 'true' ? 'offload' : 'true';
    const alt = findAlternative((candidate) => String(candidate.render.kvCache) === alternative);
    const measured = measuredGenS({ ...render, kvCache: alternative });
    kvRejected.push({
      value: alternative,
      why: alt
        ? whyAlternative(alt, chosen, levels)
        : `${alternative === 'true' ? 'K/V в VRAM' : 'K/V в RAM'}: пик ${round(kv.peakWithKvGib)} ГиБ `
          + `против порога ${round(kv.limitGib)} ГиБ (0.90·${round(request.hardware.vramGib)})`,
      ...(measured === undefined ? {} : { measuredS: measured }),
    });
  }
  trace.push({
    field: 'render.memory.kvCache.mode',
    value: kv.mode,
    ruleId: 'A7.kv',
    source: 'docs/bench/kv-cache-offload.md (512 КиБ/токен, +12–17 % времени при RAM)',
    rejected: kvRejected,
  });

  // 4. Тайл декодера VAE: сетка 16 px против пика декода.
  const vae = layout.vae;
  /** @type {TraceRejected[]} */
  const vaeRejected = [];
  if (vae.applicable) {
    for (const tilePx of /** @type {number[]} */ ([1024, 512, 0, -1])) {
      if (tilePx === vae.tilePx) continue;
      const info = VAE_TILES[tilePx];
      const grid = info.gridArtifact === null ? 'без тайлинга' : `сетка 16 px ×${info.gridArtifact}`;
      const peak = info.peakGib === null ? 'пик не мерился' : `пик декода ${info.peakGib} ГиБ`;
      vaeRejected.push({
        value: tilePx,
        why: tilePx === -1
          ? 'без тайлинга декод не влезает в 16 ГиБ уже с ~1.5 Мп (измеренный OOM)'
          : `${grid}, ${peak}`,
      });
    }
  }
  trace.push({
    field: 'render.memory.vae.tilePx',
    value: vae.tilePx,
    ruleId: 'A7.vae-tile',
    source: 'AGENTS.md#§4 «Тайлинг декодера VAE» (512/384 ×1.69, 256/192 ×2.71, 1024/768 ×1.51)',
    rejected: vaeRejected,
  });

  // 5. Окно внимания: покрытие ≥ 0.60, VRAM не экономит.
  const attention = layout.attention;
  /** @type {TraceRejected[]} */
  const attentionRejected = [];
  if (attention.applicable) {
    const alternative = attention.mode === 'window' ? 'dense' : 'window';
    const alt = findAlternative((candidate) => candidate.layout.attention.mode === alternative);
    const measured = measuredGenS({
      ...render,
      windowAttnPx: alternative === 'dense' ? 0 : render.windowAttnPx,
      marginPx: alternative === 'dense' ? 0 : render.marginPx,
    });
    const densePeak = alternative === 'dense'
      ? estimateVram({ ...render, windowAttnPx: 0, marginPx: 0 })
      : null;
    attentionRejected.push({
      value: alternative,
      why: alt
        ? whyAlternative(alt, chosen, levels)
        : alternative === 'dense' && densePeak !== null
          ? `плотное внимание на 2.6 % резче, но пик ${round(densePeak)} ГиБ > порога `
            + `${round(result.limitGib)} ГиБ — без окна кадр не влезает`
          : attention.reason,
      ...(measured === undefined ? {} : { measuredS: measured }),
    });
  }
  trace.push({
    field: 'render.memory.attention.mode',
    value: attention.mode,
    ruleId: 'A7.window',
    source: 'AGENTS.md#§4 «Скользящее окно внимания» (1792p: 314 с против 376 с, резкость 14.8 против 15.2)',
    rejected: attentionRejected,
  });

  // 6. ref_px: точность референса против бюджета KV; больше канваса не бывает.
  const ref = layout.ref;
  /** @type {TraceRejected[]} */
  const refRejected = [];
  if (ref.applicable) {
    const alternatives = ref.refPx === 0 ? [960, 768] : [0, ...(ref.refPx > 960 ? [] : [960, 768, 640])];
    for (const px of alternatives) {
      if (px === ref.refPx) continue;
      const prefix = prefixTokens({ ...render, refPx: px });
      const gib = MODEL.coefficients.kvGibPerToken * prefix;
      const why = px === 0
        ? `сетка канваса: префикс ${prefix} токенов → KV ${round(gib)} ГиБ (максимум точности, дороже)`
        : `сетка референса ${px} px: префикс ${prefix} токенов → KV ${round(gib)} ГиБ`;
      refRejected.push({ value: px, why });
    }
  }
  trace.push({
    field: 'render.memory.ref.px',
    value: ref.refPx,
    ruleId: 'A7.ref-px',
    source: 'config.py::embedding_crop (кламп по канвасу) + AGENTS.md#§4 «ref_px 0 дал бы 22400 ток./реф и OOM энкодера»',
    rejected: refRejected,
  });

  // 7. Offload: fp8-блоки в RAM против пика VRAM.
  /** @type {TraceRejected[]} */
  const offloadRejected = [];
  const otherOffload = render.offload === 'group' ? 'none' : 'group';
  const altOffload = findAlternative((candidate) => candidate.render.offload === otherOffload);
  if (altOffload) {
    offloadRejected.push({
      value: otherOffload,
      why: whyAlternative(altOffload, chosen, levels),
      ...(measuredGenS({ ...render, offload: otherOffload, kvCache: 'offload' }) === undefined
        ? {}
        : { measuredS: measuredGenS({ ...render, offload: otherOffload, kvCache: 'offload' }) }),
    });
  } else {
    let resident = null;
    try {
      resident = residentWeights(render.model, /** @type {'none'} */ (otherOffload));
    } catch {
      resident = null;
    }
    offloadRejected.push({
      value: otherOffload,
      why: resident === null
        ? `модель не умеет offload=${otherOffload}`
        : `резидентные веса ${resident} ГиБ → пик `
          + `${round(estimateVram({ ...render, offload: otherOffload, kvCache: 'offload' }))} ГиБ `
          + `против порога ${round(result.limitGib)} ГиБ`,
    });
  }
  trace.push({
    field: 'render.offload',
    value: render.offload,
    ruleId: 'A7.offload',
    source: 'AGENTS.md#§4 «Режим --offload group» (−6.6 ГиБ VRAM, ×1.5–2.6 по времени, кадр бит-в-бит)',
    rejected: offloadRejected,
  });

  // 8. Устройство энкодера: CPU на 8 ГиБ VRAM, иначе GPU-стриминг.
  const device = render.encoderDevice;
  const otherDevice = device === 'cuda' ? 'cpu' : 'cuda';
  const altDevice = estimateAll({ ...render, encoderDevice: otherDevice });
  trace.push({
    field: 'render.encoder.device',
    value: device,
    ruleId: 'A7.encoder',
    source: 'AGENTS.md#§4 «GPU-стриминг энкодера» (10.5 с GPU против 55–255 с CPU; 8 ГиБ VRAM → CPU)',
    rejected: [{
      value: otherDevice,
      why: `энкод ${round(altDevice.time.encodeColdS, 1)} с против ${round(chosen.estimates.time.encodeColdS, 1)} с `
        + `(${otherDevice === 'cpu' ? 'CPU-путь: веса энкодера в RAM' : 'GPU-путь: стриминг в VRAM'})`,
    }],
  });

  return trace;
}

/**
 * Всё объяснение плана одним вызовом.
 *
 * @param {ExplainInput} input
 * @returns {{warnings: Warning[], suggestions: Suggestion[], degradations: Degradation[], trace: TraceEntry[]}}
 */
export function explainResolution(input) {
  return {
    warnings: buildWarnings(input),
    suggestions: buildSuggestions(input),
    degradations: buildDegradations(input),
    trace: buildTrace(input),
  };
}

