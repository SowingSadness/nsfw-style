// @ts-check
/**
 * Резолв плана (шаг 5 плана): жёсткие гейты → сочетания (`candidates.mjs`) → ладдер → вердикт.
 *
 * Порядок ровно как в плане:
 *
 * ```
 * impossible (геометрия, H3 на слабом железе, refs > maxImages, политика уровня)
 *   > unsupported (16 бит: пайплайн пишет 8-битный PNG)
 *   > degraded  («серая» зона, зона 0.90–0.97 бюджета, применённый ладдер)
 *   > ok
 * ```
 *
 * Ладдер **без шагов** (решение 5: шаги на память не влияют) и в фиксированном порядке:
 *
 * ```
 * kv true→offload → offload none→group → vae_tile 1024→512→0 → ref_px 0→960→768→640
 *   → урезать refs → impossible
 * ```
 *
 * Каждая применённая ступень попадает в `degradations[]` со своим `ruleId` и причиной. Если
 * валидного сочетания не нашлось ни после ладдера, план честно отвечает `impossible` и отдаёт
 * подсказки (меньший кадр, другое железо, стадия `postprocess`) — «добрать» размер апскейлом
 * калькулятор не может по построению (A5).
 *
 * Модуль изоморфный: без `node:`/DOM.
 */

import { estimateAll } from './cost.mjs';
import { resolveGeometry } from './geometry.mjs';
import { resolveLevels } from './levels.mjs';
import { limits } from './inventory.mjs';
import {
  OFFLOAD_BLOCKS,
  buildCandidates,
  encoderProfile,
  loraScaleFor,
  modelPrecision,
} from './candidates.mjs';
import { LEVEL_WEIGHTS, paretoFront, pickBest, rankCandidates } from './rank.mjs';
import { H3_MIN_RAM_GIB, H3_MIN_VRAM_GIB, explainResolution } from './explain.mjs';
import { resolvePostprocess } from './postprocess.mjs';

/** @typedef {import('./cost.mjs').Estimates} Estimates */
/** @typedef {import('./candidates.mjs').Candidate} Candidate */
/** @typedef {import('./candidates.mjs').CandidatesResult} CandidatesResult */
/** @typedef {import('./geometry.mjs').Geometry} Geometry */
/** @typedef {import('./levels.mjs').Levels} Levels */
/** @typedef {import('../schema.mjs').NormalizedRequest} NormalizedRequest */
/** @typedef {import('./explain.mjs').Warning} Warning */
/** @typedef {import('./explain.mjs').Suggestion} Suggestion */
/** @typedef {import('./explain.mjs').Degradation} Degradation */
/** @typedef {import('./explain.mjs').TraceEntry} TraceEntry */
/** @typedef {import('./postprocess.mjs').PostprocessStage} PostprocessStage */

/** Вердикты плана в порядке приоритета (A4/A9). @typedef {'ok'|'degraded'|'unsupported'|'impossible'} Verdict */

/**
 * @typedef {Object} LadderRung
 * @property {string} id id ступени (`kv-offload`, `offload-group`, `vae-tile-512`, `ref-px-960`, …)
 * @property {string} ruleId правило плана, которым ступень обоснована (`A7.kv`, `A7.offload`, …)
 * @property {string} field какое поле плана меняется
 * @property {'kvCache'|'offload'|'vaeTilePx'|'refPx'|'refs'} axis ось раскладки
 * @property {unknown} from исходное значение (сравнивается с разрешённым)
 * @property {unknown} to значение, на которое спускаемся
 */

/**
 * Ладдер плана: порядок фиксирован, шагов в нём нет (они не влияют на память).
 *
 * @type {ReadonlyArray<LadderRung>}
 */
export const LADDER = Object.freeze([
  Object.freeze({
    id: 'kv-offload', ruleId: 'A7.kv', field: 'render.memory.kvCache.mode', axis: /** @type {const} */ ('kvCache'), from: 'true', to: 'offload',
  }),
  Object.freeze({
    id: 'offload-group', ruleId: 'A7.offload', field: 'render.offload', axis: /** @type {const} */ ('offload'), from: 'none', to: 'group',
  }),
  Object.freeze({
    id: 'vae-tile-512', ruleId: 'A7.vae-tile', field: 'render.memory.vae.tilePx', axis: /** @type {const} */ ('vaeTilePx'), from: 1024, to: 512,
  }),
  Object.freeze({
    id: 'vae-tile-0', ruleId: 'A7.vae-tile', field: 'render.memory.vae.tilePx', axis: /** @type {const} */ ('vaeTilePx'), from: 512, to: 0,
  }),
  Object.freeze({
    id: 'ref-px-960', ruleId: 'A7.ref-px', field: 'render.memory.ref.px', axis: /** @type {const} */ ('refPx'), from: 0, to: 960,
  }),
  Object.freeze({
    id: 'ref-px-768', ruleId: 'A7.ref-px', field: 'render.memory.ref.px', axis: /** @type {const} */ ('refPx'), from: 960, to: 768,
  }),
  Object.freeze({
    id: 'ref-px-640', ruleId: 'A7.ref-px', field: 'render.memory.ref.px', axis: /** @type {const} */ ('refPx'), from: 768, to: 640,
  }),
  Object.freeze({
    id: 'refs-trim', ruleId: 'A7.ref-px', field: 'refs', axis: /** @type {const} */ ('refs'), from: null, to: null,
  }),
]);

/** Сколько референсов оставить как минимум при урезании (0 — T2I). */
export const REFS_FLOOR = 0;

/**
 * Кандидаты размеров для подсказки «меньший кадр»: сетка 16:9/4:3/1:1, от реального потолка вниз.
 * Значение — высота канваса (`size_px`), пропорция берётся из запроса.
 *
 * @type {ReadonlyArray<number>}
 */
export const SMALLER_CANVAS_LADDER = Object.freeze([1792, 1600, 1440, 1280, 1152, 1088, 960, 864, 768, 640, 480]);

/**
 * Пробные конфигурации железа для подсказки «другое железо»: `[vramGib, ramGib]`.
 *
 * @type {ReadonlyArray<readonly [number, number]>}
 */
export const HARDWARE_LADDER = Object.freeze([
  Object.freeze([12, 32]),
  Object.freeze([16, 64]),
  Object.freeze([24, 64]),
  Object.freeze([32, 128]),
  Object.freeze([48, 128]),
]);

/**
 * @typedef {Object} Resolution
 * @property {Geometry} geometry
 * @property {Levels} levels
 * @property {Verdict} verdict
 * @property {'measured'|'interpolated'|'extrapolated'} confidence
 * @property {Candidate|null} best победитель (`null` — только при `impossible`)
 * @property {CandidatesResult} result сочетания и каноническая раскладка
 * @property {Array<Candidate & {score: number, qNorm: number, timeNorm: number, headroomNorm: number,
 *   pareto: boolean}>} ranked сочетания по убыванию веса, с флагом Парето
 * @property {Estimates|null} estimates оценки победителя (`null` при `impossible`)
 * @property {any} render `Plan.render` (`null` при `impossible`)
 * @property {PostprocessStage} postprocess
 * @property {Degradation[]} degradations
 * @property {Warning[]} warnings
 * @property {Suggestion[]} suggestions
 * @property {TraceEntry[]} trace
 * @property {Array<{code: string, detail: string}>} blockers причины `impossible`
 */

/**
 * Полный резолв плана: гейты, сочетания, ладдер, вердикт, объяснение.
 *
 * @param {NormalizedRequest} request нормализованный запрос (`schema.normalizeRequest`)
 * @returns {Resolution}
 */
export function resolvePlan(request) {
  const geometry = resolveGeometry(request);
  const levels = resolveLevels(request);
  const blockers = hardGates(request, geometry, levels);
  const context = { geometry, levels };

  let current = attempt(request, context, {});
  /** @type {Degradation[]} */
  const ladder = [];
  if (blockers.length === 0) {
    let overrides = {};
    for (const rung of LADDER) {
      if (current.result.candidates.length > 0) break;
      const applied = applyRung(rung, current.result, overrides, request, geometry);
      if (applied === null) continue;
      overrides = applied.overrides;
      current = attempt(request, context, overrides);
      ladder.push({
        code: rung.id,
        ruleId: rung.ruleId,
        field: rung.field,
        from: applied.from,
        to: applied.to,
        reason: applied.reason,
      });
    }
  }

  if (blockers.length === 0 && current.result.candidates.length === 0) {
    const peak = current.result.canonical.estimates.vram.peakGib;
    blockers.push({
      code: 'no-native-candidate',
      detail: `ни одно сочетание не влезает: пик ${round(peak)} ГиБ против порога `
        + `${round(current.result.limitGib)} ГиБ, ладдер (kv → offload → vae → ref_px → refs) исчерпан`,
    });
  }
  const impossible = blockers.length > 0 || current.result.candidates.length === 0;
  const best = impossible ? null : pickBest(current.result.candidates, LEVEL_WEIGHTS[levels.level]);
  const paretoIds = new Set(paretoFront(current.result.candidates).map((candidate) => candidate.id));
  // При `impossible` таблица сочетаний пуста: плана нет, а список «прошедших гейт» сочетаний для
  // снэпнутого канваса читался бы как рекомендация. Сочетания остаются в `trace` (объяснение
  // отклонённых вариантов) — там они и нужны.
  const ranked = impossible ? [] : rankCandidates(current.result.candidates, LEVEL_WEIGHTS[levels.level])
    .map((candidate) => ({ ...candidate, pareto: paretoIds.has(candidate.id) }));

  /** @type {Verdict} */
  let verdict;
  if (impossible) verdict = 'impossible';
  else if (request.image.bitDepth === 16) verdict = 'unsupported';
  else if (geometry.verdict === 'degraded' || ladder.length > 0 || (best && best.estimates.vram.zone !== 'ok')) verdict = 'degraded';
  else verdict = 'ok';

  const confidence = best ? best.estimates.confidence : 'extrapolated';
  const postprocess = resolvePostprocess(request, best ? best.render : null, request.hardware);

  const probes = {
    smallerCanvas: verdict === 'impossible' || verdict === 'degraded'
      ? probeSmallerCanvas(request, levels)
      : null,
    biggerHardware: verdict === 'impossible' ? probeBiggerHardware(request, geometry, levels) : null,
    greyZoneBlocked: geometry.verdict === 'degraded' && !levels.allowDegradedZone,
  };

  const explanation = explainResolution({
    request,
    geometry,
    levels,
    result: current.result,
    best,
    ladder,
    verdict,
    confidence,
    postprocess,
    probes,
  });

  return {
    geometry,
    levels,
    verdict,
    confidence,
    best,
    result: current.result,
    ranked,
    estimates: best ? best.estimates : null,
    render: best ? buildRender(request, best) : null,
    postprocess,
    degradations: impossible ? [] : explanation.degradations,
    warnings: explanation.warnings,
    suggestions: explanation.suggestions,
    trace: explanation.trace,
    blockers,
  };
}

/**
 * Жёсткие гейты до перебора сочетаний: если хоть один сработал — `impossible`, ладдер не нужен.
 *
 * @param {NormalizedRequest} request
 * @param {Geometry} geometry
 * @param {{allowDegradedZone: boolean}} [levels] уровень качества: `standard`/`draft` не
 *   допускают «серую» зону (A3), поэтому для них она — блокер, а не `degraded`
 * @returns {Array<{code: string, detail: string}>}
 */
export function hardGates(request, geometry, levels) {
  /** @type {Array<{code: string, detail: string}>} */
  const blockers = [];
  if (geometry.verdict === 'impossible') {
    for (const warning of geometry.warnings) blockers.push({ code: warning.code, detail: warning.detail });
  }
  if (geometry.verdict === 'degraded' && levels && !levels.allowDegradedZone) {
    // Политика уровня главнее геометрии: `standard`/`draft` обещают только проверенные размеры
    // (≤ 5.73 Мп). Иначе на 24 ГиБ вернулся бы план «деградированного» кадра, которого уровень
    // не обещает, и пользователь узнал бы об этом только из бейджа.
    blockers.push({
      code: 'grey-zone-not-allowed-by-level',
      detail: `размер ${geometry.canvas.width}×${geometry.canvas.height} лежит в «серой» зоне `
        + `(${round(geometry.pixels / 1e6)} Мп > 5.73 Мп): качество ${request.image.quality} `
        + `её не допускает — нужен меньший кадр или quality: max`,
    });
  }
  const maxImages = /** @type {number} */ (limits().maxImages);
  if (request.refs > maxImages) {
    blockers.push({
      code: 'refs-exceed-model',
      detail: `бэкенд объявляет maxImages = ${maxImages}, в запросе ${request.refs}`,
    });
  }
  if (request.hardware.model === 'h3-image') {
    if (request.hardware.vramGib < H3_MIN_VRAM_GIB) {
      blockers.push({
        code: 'h3-needs-16gib-vram',
        detail: `H3-Image измерен только на ${H3_MIN_VRAM_GIB} ГиБ VRAM: 30.9 ГиБ fp8 в RAM при групповом offload `
          + `и энкодер Qwen3-VL-32B (62.1 ГиБ), в ${request.hardware.vramGib} ГиБ не влезает`,
      });
    }
    if (request.hardware.ramGib < H3_MIN_RAM_GIB) {
      blockers.push({
        code: 'h3-ram-32gb',
        detail: `H3-Image требует ${H3_MIN_RAM_GIB} ГиБ RAM (fp8-трансформер 30.9 ГиБ + энкодер 62.1 ГиБ): `
          + `${request.hardware.ramGib} ГиБ упирается в спилл`,
      });
    }
  }
  return dedupeBlockers(blockers);
}

/**
 * @param {Array<{code: string, detail: string}>} blockers
 * @returns {Array<{code: string, detail: string}>}
 */
function dedupeBlockers(blockers) {
  const seen = new Set();
  return blockers.filter((blocker) => {
    if (seen.has(blocker.code)) return false;
    seen.add(blocker.code);
    return true;
  });
}

/**
 * @typedef {{overrides: Record<string, unknown>, from: unknown, to: unknown, reason: string}} RungApplication
 */

/**
 * Применить ступень ладдера к текущему состоянию. Ступень применяется только если текущее
 * разрешённое значение совпадает с её `from` (иначе она не про этот запрос).
 *
 * @param {LadderRung} rung
 * @param {CandidatesResult} result текущее разрешение (до ступени)
 * @param {Record<string, unknown>} overrides уже применённые переопределения
 * @param {NormalizedRequest} request
 * @param {Geometry} geometry геометрия запроса (T2I без `--aspect` не бывает)
 * @returns {RungApplication|null}
 */
function applyRung(rung, result, overrides, request, geometry) {
  const canonical = result.canonical;
  const limit = result.limitGib;
  const peak = canonical.estimates.vram.peakGib;
  if (rung.axis === 'kvCache') {
    if (canonical.layout.kvCache.mode !== rung.from) return null;
    return {
      overrides: { ...overrides, kvCache: rung.to },
      from: rung.from,
      to: rung.to,
      reason: `K/V в VRAM не влезает: пик ${round(peak)} ГиБ + KV ${round(canonical.layout.kvCache.gib)} ГиБ`
        + ` > порога ${round(limit)} ГиБ (0.90·${round(request.hardware.vramGib)}) — K/V уходят в RAM`
        + ` (+${canonical.layout.kvCache.costPct} % времени, кадр бит-в-бит)`,
    };
  }
  if (rung.axis === 'offload') {
    const current = canonical.render.offload;
    if (current !== rung.from) return null;
    return {
      overrides: { ...overrides, offload: rung.to },
      from: rung.from,
      to: rung.to,
      reason: `при offload=${rung.from} пик ${round(peak)} ГиБ > порога ${round(limit)} ГиБ — `
        + 'fp8-блоки выгружаются в RAM (пик −6.6 ГиБ, время ×1.5–2.6, кадр бит-в-бит)',
    };
  }
  if (rung.axis === 'vaeTilePx') {
    if (canonical.layout.vae.tilePx !== rung.from) return null;
    return {
      overrides: { ...overrides, vaeTilePx: rung.to },
      from: rung.from,
      to: rung.to,
      reason: `декод VAE с тайлом ${rung.from} не влезает в ${round(request.hardware.vramGib)} ГиБ: `
        + `пик декода ${canonical.layout.vae.peakGib} ГиБ — тайл ${rung.to}`,
    };
  }
  if (rung.axis === 'refPx') {
    if (canonical.layout.ref.refPx !== rung.from) return null;
    return {
      overrides: { ...overrides, refPx: rung.to },
      from: rung.from,
      to: rung.to,
      reason: `сетка референсов ${rung.from} px даёт префикс ${canonical.layout.kvCache.prefixTokens} токенов `
        + `(KV ${round(canonical.layout.kvCache.gib)} ГиБ) — спускаемся на ${rung.to} px`,
    };
  }
  // refs-trim: только вниз, до пола `REFS_FLOOR`, и только если это изменит префикс.
  const refs = overrides.refs === undefined ? request.refs : /** @type {number} */ (overrides.refs);
  if (refs <= REFS_FLOOR) return null;
  const next = refs - 1;
  // До нуля референсы можно урезать только у кадра с разрешённой пропорцией: T2I без `--aspect`
  // невозможен (A4), а пропорцию запроса задаёт первый референс.
  if (next === 0 && geometry.aspect === null) return null;
  return {
    overrides: { ...overrides, refs: next },
    from: refs,
    to: next,
    reason: `референсы не влезают: урезаем ${refs} → ${next} (префикс KV `
      + `${canonical.layout.kvCache.prefixTokens} токенов) — меняется conditioning, нужен новый энкод`,
  };
}

/**
 * Одна попытка разрешения: сочетания для запроса с применёнными переопределениями ладдера.
 *
 * @param {NormalizedRequest} request
 * @param {{geometry: Geometry, levels: Levels}} context
 * @param {{kvCache?: unknown, offload?: unknown, vaeTilePx?: unknown, refPx?: unknown, refs?: unknown}} overrides
 * @returns {{request: NormalizedRequest, result: CandidatesResult}}
 */
function attempt(request, context, overrides) {
  const effective = withOverrides(request, overrides);
  const result = buildCandidates(effective, {
    geometry: context.geometry,
    levels: context.levels,
    offload: /** @type {'none'|'group'|null} */ (overrides.offload ?? null),
    refPx: /** @type {number|null} */ (overrides.refPx ?? null),
  });
  return { request: effective, result };
}

/**
 * Запрос с переопределениями ладдера (память запроса + число референсов).
 *
 * @param {NormalizedRequest} request
 * @param {{kvCache?: unknown, vaeTilePx?: unknown, refs?: unknown}} overrides
 * @returns {NormalizedRequest}
 */
function withOverrides(request, overrides) {
  const keys = Object.keys(overrides);
  if (keys.length === 0) return request;
  const memory = { ...request.memory };
  if (overrides.kvCache !== undefined) {
    memory.kvCache = /** @type {any} */ (overrides.kvCache);
  }
  if (overrides.vaeTilePx !== undefined) {
    memory.vaeTilePx = /** @type {any} */ (overrides.vaeTilePx);
  }
  return {
    ...request,
    refs: overrides.refs === undefined ? request.refs : /** @type {number} */ (overrides.refs),
    memory,
  };
}

/**
 * `Plan.render` из победителя: шаги, offload, точность, энкодер, LoRA и раскладка памяти с ценами.
 *
 * @param {NormalizedRequest} request
 * @param {Candidate} best
 * @returns {any}
 */
export function buildRender(request, best) {
  const model = request.hardware.model;
  const layout = best.layout;
  const denseEstimates = best.layout.attention.mode === 'window'
    ? estimateAll({ ...best.render, windowAttnPx: 0, marginPx: 0 })
    : best.estimates;
  const costS = Math.round(best.metrics.timeS);
  const denseCostS = Math.round(denseEstimates.time.genPerFrameS);
  // Окно в модели времени — измеренный коэффициент (`bWindow = ln(314/376)`), поэтому цена режима
  // видна прямо в причине: сколько секунд оно снимает и сколько резкости стоит.
  const attentionReason = best.layout.attention.mode === 'window'
    ? `${best.layout.attention.reason}; в модели времени это ${costS} с против ${denseCostS} с плотного '
      + '(замер 1792p: 314 против 376 с, −17 %)`
    : best.layout.attention.reason;
  return {
    steps: best.render.steps,
    offload: best.render.offload,
    offloadBlocks: OFFLOAD_BLOCKS,
    precision: modelPrecision(model),
    encoder: encoderProfile(model, request.hardware.vramGib),
    loraScale: loraScaleFor(request, model),
    memory: {
      kvCache: {
        mode: layout.kvCache.mode,
        placement: layout.kvCache.placement,
        prefixTokens: layout.kvCache.prefixTokens,
        gib: round(layout.kvCache.gib),
        perLayerMib: layout.kvCache.perLayerMib === null ? null : round(layout.kvCache.perLayerMib, 1),
        costPct: layout.kvCache.costPct,
        reason: layout.kvCache.reason,
      },
      vae: {
        tilePx: layout.vae.tilePx,
        stridePx: layout.vae.stridePx,
        gridArtifact: layout.vae.gridArtifact,
        peakGib: layout.vae.peakGib,
        oom: layout.vae.oom,
        reason: layout.vae.reason,
      },
      attention: {
        mode: layout.attention.mode,
        windowPx: layout.attention.windowPx,
        marginPx: layout.attention.marginPx,
        coverage: round(layout.attention.coverage, 3),
        bands: layout.attention.bands,
        costS,
        denseCostS,
        reason: attentionReason,
      },
      ref: {
        px: layout.ref.refPx,
        tokensPerRef: layout.ref.tokensPerRef,
        totalTokens: layout.ref.refTokens,
        reason: layout.ref.reason,
      },
    },
  };
}

/**
 * Подсказка «меньший кадр»: самый большой `sizePx` этой пропорции, у которого есть валидные
 * сочетания. Идёт сверху вниз по `SMALLER_CANVAS_LADDER`; возвращает первый успех.
 *
 * @param {NormalizedRequest} request
 * @param {Levels} levels
 * @returns {{width: number, height: number}|null}
 */
export function probeSmallerCanvas(request, levels) {
  const ratio = request.image.width / request.image.height;
  for (const sizePx of SMALLER_CANVAS_LADDER) {
    const height = sizePx;
    const width = Math.round(sizePx * ratio);
    if (height >= request.image.height) continue;
    /** @type {NormalizedRequest} */
    const probe = {
      ...request,
      image: { ...request.image, width, height },
    };
    const geometry = resolveGeometry(probe);
    if (geometry.verdict === 'impossible') continue;
    if (geometry.verdict === 'degraded' && !levels.allowDegradedZone) continue;
    const result = buildCandidates(probe, { geometry, levels });
    if (result.candidates.length === 0) continue;
    return { width: geometry.canvas.width, height: geometry.canvas.height };
  }
  return null;
}

/**
 * Подсказка «другое железо»: минимальный бюджет из `HARDWARE_LADDER`, при котором запрос
 * становится выполнимым.
 *
 * @param {NormalizedRequest} request
 * @param {Geometry} geometry
 * @param {Levels} levels
 * @returns {{field: string, value: number}|null}
 */
export function probeBiggerHardware(request, geometry, levels) {
  for (const [vramGib, ramGib] of HARDWARE_LADDER) {
    if (vramGib <= request.hardware.vramGib && ramGib <= request.hardware.ramGib) continue;
    /** @type {NormalizedRequest} */
    const probe = {
      ...request,
      hardware: { ...request.hardware, vramGib, ramGib },
    };
    const blockers = hardGates(probe, geometry, levels);
    if (blockers.length > 0) continue;
    const result = buildCandidates(probe, { geometry, levels });
    if (result.candidates.length === 0) continue;
    return vramGib > request.hardware.vramGib
      ? { field: 'hardware.vramGib', value: vramGib }
      : { field: 'hardware.ramGib', value: ramGib };
  }
  return null;
}

/**
 * @param {number} value
 * @param {number} [digits]
 * @returns {number}
 */
function round(value, digits = 2) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}
