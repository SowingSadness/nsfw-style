// @ts-check
/**
 * Публичный вход `renderkit` — изоморфное ESM-ядро (ADR A1).
 *
 * Один и тот же модуль работает в браузере, в Node (тесты/CI), в будущем сервисе и в CLI-режиме.
 * Никаких `node:`, `process`, DOM и зависимостей: только чистые функции.
 *
 * `computePlan(request)` — детерминированная сборка `Plan` (шаги 5–6 плана):
 * `request → normalizeRequest → resolvePlan (гейты/сочетания/ладдер/вердикт) →
 * emitPreset (YAML + команды) → Plan`.
 *
 * `yaml`/`cli` заполняются эмиттером (`./model/emit.mjs`) только для собранного плана: у
 * `verdict: 'impossible'` рендер-стадии нет, а значит нет и конфига — там честные `null`
 * (бросок был бы хуже: UI обязан показать причину, а не исключение).
 */

import {
  ASPECT_TOLERANCE,
  ASPECTS,
  SIZE_LIMITS,
} from './model/geometry.mjs';
import {
  BIT_DEPTHS,
  DEFAULTS,
  DISKS,
  KV_CACHE_MODES,
  LIMITS,
  MODELS,
  QUALITIES,
  SCHEMA_VERSION,
  VAE_TILES,
  normalizeRequest,
} from './schema.mjs';
import { inventoryDigest, tierFor } from './model/inventory.mjs';
import { OFFLOAD_BLOCKS, loraScaleFor } from './model/candidates.mjs';
import { emitPreset } from './model/emit.mjs';
import { resolvePlan } from './model/resolve.mjs';

export { normalizeRequest };
export { PlanError, normalizeWindow, SCHEMA_VERSION } from './schema.mjs';
export { canvasFor, kvGib, pyRound, tokensFor } from './units.mjs';
export {
  ASPECTS,
  GEOMETRY_WARNINGS,
  SIZE_LIMITS,
  matchAspect,
  resolveGeometry,
} from './model/geometry.mjs';
export { LADDER, resolvePlan } from './model/resolve.mjs';
export { paretoFront, pickBest } from './model/rank.mjs';
export { resolvePostprocess, validatePostprocessParams } from './model/postprocess.mjs';

/**
 * Версия правил калькулятора: база + дайджест входных данных генератора (`inventory.mjs`:
 * резолвленные конфиги пресетов и манифесты тарифов). Растёт при перекалибровке
 * (`tools/fit.mjs`) или изменении инвентаря — по ней видно, какими правилами собран план.
 *
 * Формат `r1+<configs12>-<profiles12>+emit1`: дайджест инвентаря плюс **ревизия эмиттера**
 * (`+emit1`), потому что артефакт (`Plan.yaml`) зависит не только от данных, но и от того, как
 * их форматирует `model/emit.mjs`. Когда `tools/fit.mjs` начнёт писать `fitSha`, сюда добавится
 * и он (сейчас `data/model.mjs` шахты не содержит — считать дайджест нечем, а хеш-функции
 * вроде `node:crypto` в `src/**` запрещены: ядро изоморфно, ADR A1).
 */
const DIGEST = inventoryDigest();
const EMIT_VERSION = 'emit1';
export const RULESET_VERSION = `r1+${DIGEST.configs}-${DIGEST.profiles}+${EMIT_VERSION}`;

/**
 * Статический справочник опций для UI/сервиса (`GET /api/options`).
 * Без имён пресетов: только значения, подписи и границы.
 *
 * @returns {{
 *   schemaVersion: number,
 *   rulesetVersion: string,
 *   models: Array<{value: string, label: string}>,
 *   qualities: Array<{value: string, label: string}>,
 *   bitDepths: Array<{value: number, label: string}>,
 *   disks: Array<{value: string, label: string}>,
 *   aspects: Array<{value: string, ratio: number}>,
 *   limits: typeof LIMITS,
 *   defaults: typeof DEFAULTS & {qualitySteps: Record<string, number|null>},
 *   memory: {kvCache: Array<{value: unknown, label: string}>, vaeTilePx: Array<{value: number, label: string}>},
 *   aspectTolerance: number,
 *   sizeLimits: {okMaxPx: number, hardMaxPx: number}
 * }}
 */
export function listOptions() {
  return {
    schemaVersion: SCHEMA_VERSION,
    rulesetVersion: RULESET_VERSION,
    models: [
      { value: 'qwen21-base', label: 'Qwen-Image-2.1 (base, 25 шагов)' },
      { value: 'qwen21-turbo', label: 'Qwen-Image-2.1-Turbo (distill, 8 шагов)' },
      { value: 'h3-image', label: 'MiniMax-H3-Image (6 шагов PDD)' },
    ],
    qualities: [
      { value: 'draft', label: 'draft — дешёвый кадр для композиции' },
      { value: 'standard', label: 'standard — рабочий кадр, только ok-размеры' },
      { value: 'max', label: 'max — всё, что даёт модель, включая «серую» зону' },
    ],
    bitDepths: [
      { value: 8, label: '8 бит — PNG (поддержано)' },
      { value: 16, label: '16 бит — пока не поддержано пайплайном' },
    ],
    disks: [
      { value: 'hdd', label: 'HDD / SMB' },
      { value: 'ssd', label: 'SATA SSD' },
      { value: 'nvme', label: 'NVMe' },
    ],
    aspects: ASPECTS.map((aspect) => ({ value: aspect.key, ratio: aspect.ratio })),
    limits: LIMITS,
    defaults: {
      ...DEFAULTS,
      // Шаги по уровням: h3 фиксирован 6 (расписание PDD), `draft` — 8 ради дешёвых кадров.
      // Точный выбор делает `levels.mjs`; здесь — справочник для подсказок.
      qualitySteps: { draft: 8, standard: 25, max: 25 },
    },
    memory: {
      kvCache: [
        { value: true, label: 'VRAM — быстрее, но префикс занимает VRAM' },
        { value: 'offload', label: 'RAM — 212 МиБ/слой, +12–14 % времени' },
        { value: false, label: 'без кеша — префикс считается каждый шаг' },
        { value: 'auto-offload-ref2', label: 'авто: 1 реф → VRAM, 2+ → RAM' },
      ],
      vaeTilePx: [
        { value: 512, label: '512/384 — умолчание, сетка 16 px ×1.69' },
        { value: 0, label: '256/192 — сетка ×2.71, только малый кадр' },
        { value: 1024, label: '1024/768 — сетка ×1.51, пик 15 ГиБ' },
        { value: -1, label: 'без тайлинга — OOM от ~1.5 Мп' },
      ],
    },
    aspectTolerance: ASPECT_TOLERANCE,
    sizeLimits: { okMaxPx: SIZE_LIMITS.okMaxPx, hardMaxPx: SIZE_LIMITS.hardMaxPx },
  };
}

/**
 * Расчёт плана рендера по запросу: полная сборка `Plan` (шаг 5).
 *
 * Форма ответа — раздел «Выход (`Plan`)» плана. `estimates` — по рендер-стадии; `stages` не
 * смешивает бюджеты стадий (постобработка в Phase A выключена и `estimates: null`).
 * `yaml` — самодостаточный YAML-конфиг (ADR A2: без `extends` и имён пресетов), `cli` —
 * `{save, run, gen, checks, sets}` того же плана (`model/emit.mjs` → `model/cli.mjs`). Оба поля
 * `null` только когда плана нет (`verdict: 'impossible'`, `render === null`).
 *
 * @param {unknown} input нормализованный или сырой `RenderRequest`
 * @returns {any} `Plan`
 * @throws {import('./schema.mjs').PlanError} если запрос не проходит схему
 */
export function computePlan(input) {
  const request = normalizeRequest(input);
  const resolution = resolvePlan(request);
  const estimates = resolution.estimates ? planEstimates(resolution.estimates) : null;
  const render = resolution.render;
  const bestId = resolution.best ? resolution.best.id : null;
  const loraScale = loraScaleFor(request, request.hardware.model);
  const blocked = resolution.blockers.map((blocker) => `${blocker.code}: ${blocker.detail}`).join('; ');

  /** @type {Record<string, any>} */
  const plan = {
    schemaVersion: SCHEMA_VERSION,
    rulesetVersion: RULESET_VERSION,
    request: { ...request, tier: tierFor(request.hardware.vramGib) },
    verdict: resolution.verdict,
    confidence: resolution.confidence,
    image: {
      requested: { ...resolution.geometry.requested },
      native: {
        width: resolution.geometry.canvas.width,
        height: resolution.geometry.canvas.height,
        tokens: resolution.geometry.tokens,
        sizePx: resolution.geometry.sizePx,
      },
      aspect: resolution.geometry.aspect,
      snap: {
        dx: resolution.geometry.snap.dx,
        dy: resolution.geometry.snap.dy,
        reachable: resolution.geometry.snap.reachable.map(([width, height]) => [width, height]),
      },
      bitDepth: request.image.bitDepth,
      format: 'png',
      exact: resolution.geometry.snap.dx === 0 && resolution.geometry.snap.dy === 0,
    },
    render,
    stages: [
      {
        stage: 'render',
        enabled: render !== null,
        reason: render === null
          ? `нативного пути нет: ${blocked}`
          : 'нативный рендер без апскейла (A5)',
        estimates,
        quality: resolution.best && resolution.best.quality.q !== null
          ? Math.round(/** @type {number} */ (resolution.best.quality.q))
          : null,
      },
      resolution.postprocess,
    ],
    // Заполняется ниже эмиттером: `null` только для плана без рендер-стадии.
    yaml: null,
    cli: null,
    estimates,
    combinations: resolution.ranked.map((candidate) => combinationRow(candidate, bestId, loraScale)),
    degradations: resolution.degradations,
    suggestions: resolution.suggestions,
    warnings: resolution.warnings,
    trace: resolution.trace,
  };

  // Шаг 6: артефакт из того же плана, который уходит клиенту (один источник значений —
  // `presetFor`, поэтому YAML и `--set`-форма не расходятся по определению).
  if (plan.render !== null) {
    const emitted = emitPreset(plan);
    plan.yaml = emitted.yaml;
    plan.cli = emitted.cli;
  }
  return plan;
}

/**
 * Строка таблицы сочетаний для UI (Парето-подсветка, дельты к победителю).
 *
 * @param {any} candidate сочетание из `resolvePlan().ranked`
 * @param {string|null} bestId id победителя
 * @param {number} loraScale сила LoRA (одинакова у всех сочетаний — не ось)
 * @returns {any}
 */
function combinationRow(candidate, bestId, loraScale) {
  const layout = candidate.layout;
  return {
    id: candidate.id,
    steps: candidate.render.steps,
    offload: candidate.render.offload,
    offloadBlocks: OFFLOAD_BLOCKS,
    kvCache: layout.kvCache.mode,
    vaeTilePx: candidate.render.vaeTilePx,
    attention: layout.attention.mode,
    windowPx: layout.attention.windowPx,
    marginPx: layout.attention.marginPx,
    refPx: candidate.render.refPx,
    encoderDevice: candidate.render.encoderDevice,
    loraScale,
    q: Math.round(candidate.metrics.q),
    timeS: round1(candidate.metrics.timeS),
    peakVramGib: round2(candidate.estimates.vram.peakGib),
    headroomGib: round2(candidate.metrics.headroomGib),
    zone: candidate.gates.zone,
    score: round(candidate.score, 4),
    pareto: candidate.pareto === true,
    best: candidate.id === bestId,
  };
}

/**
 * Оценки для `Plan.estimates`: число знаков фиксировано, чтобы снапшоты (шаг 8) были читаемыми.
 *
 * @param {import('./model/cost.mjs').Estimates} estimates
 * @returns {any}
 */
function planEstimates(estimates) {
  return {
    vram: {
      ...estimates.vram,
      peakGib: round2(estimates.vram.peakGib),
      headroomGib: round2(estimates.vram.headroomGib),
      headroomPct: round1(estimates.vram.headroomPct),
    },
    ram: {
      ...estimates.ram,
      peakGib: round2(estimates.ram.peakGib),
      headroomGib: round2(estimates.ram.headroomGib),
      headroomPct: round1(estimates.ram.headroomPct),
    },
    time: {
      loadS: round1(estimates.time.loadS),
      encodeColdS: round1(estimates.time.encodeColdS),
      encodeWarmS: round1(estimates.time.encodeWarmS),
      genPerFrameS: round1(estimates.time.genPerFrameS),
      firstFrameS: round1(estimates.time.firstStepS),
      warmFrameS: round1(estimates.time.warmFrameS),
    },
    disk: {
      pngPerFrameMb: round2(estimates.disk.pngPerFrameMb),
      bitDepthFactor: estimates.disk.bitDepthFactor,
    },
    confidence: estimates.confidence,
    confidenceReasons: [...estimates.confidenceReasons],
    prefixTokens: estimates.prefixTokens,
    kvMode: estimates.kvMode,
  };
}

/** @param {number} value */
function round2(value) {
  return Math.round(value * 100) / 100;
}

/** @param {number} value */
function round1(value) {
  return Math.round(value * 10) / 10;
}

/**
 * @param {number} value
 * @param {number} digits
 */
function round(value, digits) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

/** Пере-экспорт кодовых списков для UI (единый источник значений). */
export const OPTION_VALUES = Object.freeze({
  models: MODELS,
  qualities: QUALITIES,
  bitDepths: BIT_DEPTHS,
  disks: DISKS,
  kvCacheModes: KV_CACHE_MODES,
  vaeTiles: VAE_TILES,
});
