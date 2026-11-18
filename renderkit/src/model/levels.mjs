// @ts-check
/**
 * Уровни качества `draft|standard|max` → шаги и политика риска (ADR A3, шаг 4).
 *
 * Ключевое решение плана (A3, п.5): **шаги не являются рычагом памяти**. Активации создаются
 * внутри forward и не накапливаются от шага к шагу, поэтому пик VRAM от `steps` не зависит;
 * шаги покупают качество и время, но никогда «влезание». Отсюда:
 *
 * - `draft` → 8 шагов (дешёвый набор кадров для композиции и стиля промта);
 * - `standard` → полное расписание модели (25 у qwen21-base, 8 у turbo);
 * - `max` → то же расписание, но политика разрешает «серую» зону и окно внимания;
 * - модели с объявленным фиксированным расписанием (`limits.fixedSteps`, у нас H3 = 6 PDD)
 *   игнорируют `image.steps` — это расписание, а не настройка.
 *
 * Различие уровней — **политика риска и потолка**, а не шаги:
 *
 * | уровень | degraded-зона | окно внимания | запас VRAM |
 * |---|---|---|---|
 * | `draft` | нет | нет | 0 % (кадр всё равно черновой) |
 * | `standard` | нет | нет | ≥ 10 % |
 * | `max` | да (до 8.35 Мп) | да | 0 % |
 *
 * Поэтому `steps` из запроса меняет качество/длительность, но **не** результат `fits`
 * (`cost.estimateVram` шагов не принимает вовсе).
 *
 * Модуль изоморфный: без `node:`/DOM.
 */

import { modelCaps, limits } from './inventory.mjs';
import { QUALITIES } from '../schema.mjs';

/**
 * Политика уровней. `steps` у `standard`/`max` — `null`: берётся полное расписание модели
 * (`modelCaps(model).fixedSteps`), чтобы значение не жило в двух местах.
 *
 * @type {Readonly<Record<'draft'|'standard'|'max', {
 *   steps: number|null, allowDegradedZone: boolean, allowWindow: boolean, minHeadroom: number,
 *   note: string
 * }>>}
 */
export const LEVEL_POLICY = Object.freeze({
  draft: Object.freeze({
    steps: 8,
    allowDegradedZone: false,
    allowWindow: false,
    minHeadroom: 0,
    note: 'черновик: композиция и стиль промта. Короткое расписание (8) — единственное отличие '
      + 'от standard; потолок и окно те же, что у standard (черновик не покупает «серую» зону), '
      + 'но запас VRAM не требуется — кадр всё равно не финальный',
  }),
  standard: Object.freeze({
    steps: null,
    allowDegradedZone: false,
    allowWindow: false,
    minHeadroom: 0.1,
    note: 'предсказуемый рабочий кадр: только ok-размеры (≤ 5.73 Мп), без окна внимания, '
      + 'запас VRAM ≥ 10 % — чтобы прогон не зависел от рабочего стола и соседних процессов',
  }),
  max: Object.freeze({
    steps: null,
    allowDegradedZone: true,
    allowWindow: true,
    minHeadroom: 0,
    note: 'всё, что даёт модель: полное расписание, «серая» зона до 8.35 Мп (фактура уходит '
      + 'в рябь — это вердикт comfort-limit, а не отказ) и окно внимания там, где оно экономит время',
  }),
});

/**
 * @typedef {Object} Levels
 * @property {'draft'|'standard'|'max'} level уровень из запроса
 * @property {number} steps итоговые шаги сэмплера
 * @property {number} fullSteps полное расписание модели (`modelCaps().fixedSteps`)
 * @property {boolean} fixed расписание фиксировано моделью (`limits.fixedSteps`)
 * @property {number|null} override `image.steps` из запроса как есть
 * @property {number|null} ignoredSteps шаги, которые пришлось игнорировать (у фиксированной модели)
 * @property {boolean} allowDegradedZone разрешена «серая» зона (> 5.73 Мп)
 * @property {boolean} allowWindow разрешено скользящее окно внимания
 * @property {number} minHeadroom требуемый запас VRAM (доля бюджета, 0…1)
 * @property {string} reason почему такие шаги
 */

/**
 * Шаги и политика риска по уровню качества.
 *
 * @param {{hardware: {model: string}, image: {quality: string, steps: number|null}}} request
 *   нормализованный запрос (`schema.normalizeRequest`)
 * @returns {Levels}
 */
export function resolveLevels(request) {
  const quality = request.image.quality;
  if (!QUALITIES.includes(quality)) {
    throw new RangeError(`resolveLevels: неизвестный уровень качества ${JSON.stringify(quality)}; `
      + `доступны: ${QUALITIES.join(', ')}`);
  }
  const policy = LEVEL_POLICY[/** @type {'draft'|'standard'|'max'} */ (quality)];
  const model = request.hardware.model;
  const caps = modelCaps(model);
  const fullSteps = caps.fixedSteps;
  const declaredFixed = /** @type {Record<string, number>} */ (limits().fixedSteps);
  const fixedSteps = declaredFixed[model] ?? null;
  const fixed = fixedSteps !== null;
  const override = request.image.steps ?? null;
  let ignoredSteps = null;

  /** @type {number} */
  let steps;
  /** @type {string} */
  let reason;

  if (fixed) {
    steps = /** @type {number} */ (fixedSteps);
    if (override !== null && override !== steps) ignoredSteps = override;
    reason = `fixed-schedule: ${model} считает ${steps} шагов по расписанию модели `
      + '(PDD) — steps не настройка'
      + (ignoredSteps === null ? '' : `; явные ${ignoredSteps} шагов проигнорированы`);
  } else if (override !== null) {
    steps = override;
    reason = `steps-override: явный image.steps=${steps} переопределяет уровень `
      + `${quality} (полное расписание модели ${fullSteps}); память от шагов не зависит`;
  } else if (policy.steps !== null) {
    steps = Math.min(policy.steps, fullSteps);
    reason = `level-${quality}: ${steps} шагов (${policy.note})`;
  } else {
    steps = fullSteps;
    reason = `level-${quality}: полное расписание модели — ${steps} шагов (${policy.note})`;
  }

  return {
    level: /** @type {'draft'|'standard'|'max'} */ (quality),
    steps,
    fullSteps,
    fixed,
    override,
    ignoredSteps,
    allowDegradedZone: policy.allowDegradedZone,
    allowWindow: policy.allowWindow,
    minHeadroom: policy.minHeadroom,
    reason,
  };
}
