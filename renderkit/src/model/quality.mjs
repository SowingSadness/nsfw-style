// @ts-check
/**
 * Q — рейтинг «качества кадра» для ранжирования сочетаний (ADR A6, шаг 4):
 *
 * ```
 * Q = √nativePixels · stepsFactor · attentionFactor · comfortFactor
 * ```
 *
 * Апскейла в Q нет и не будет (A5): вся фактура кадра модельная, а постобработка — отдельная
 * стадия со своим расчётом. Поэтому множителей ровно три, и каждый помечен как допущение
 * (`assumption`): таблица шагов, цена окна внимания и комфорт по пикселям.
 *
 * `comfortFactor` — единственный фактор, который может обнулить Q (`null`): выше «серой» зоны
 * (`SIZE_LIMITS.hardMaxPx`) нативного пути нет, и это вердикт `impossible`, а не «плохое качество».
 *
 * Модуль изоморфный: без `node:`/DOM, только арифметика и `geometry.SIZE_LIMITS`.
 */

import { SIZE_LIMITS } from './geometry.mjs';

/**
 * Таблица факторов Q с `assumption`-пометками.
 *
 * Числа — из ADR A6 и наших замеров:
 * - шаги: 8 → 0.60, 12 → 0.75, 16 → 0.85, 25 → 1.00 (12/16 встречаются только через явный
 *   `image.steps`; 25 — полное расписание модели);
 * - внимание: плотное 1.00, окно 0.97 (резкость 15.2 против 14.8 на 1792p — −2.6 %);
 * - комфорт: 1.00 до `okMaxPx` (5 734 400 px = 3200×1792, замер 314 с / 10.3 ГиБ), линейный спад
 *   до 0.75 к `hardMaxPx` (8 355 840 px = 3840×2176, фактура уходит в рябь), выше — `null`.
 *
 * @type {Readonly<{
 *   steps: {kind: 'assumption', source: string, note: string,
 *     values: ReadonlyArray<{steps: number, factor: number}>},
 *   attention: {kind: 'assumption', source: string, note: string,
 *     values: Readonly<{dense: number, window: number}>},
 *   comfort: {kind: 'assumption', source: string, note: string,
 *     fullPx: number, floorPx: number, floorFactor: number}
 * }>}
 */
export const QUALITY_FACTORS = Object.freeze({
  steps: Object.freeze({
    kind: 'assumption',
    source: '.pi/plans/render-settings-kit.md#A6 «Q и ранжирование»: stepsFactor 8→0.60, 12→0.75, 16→0.85, 25→1.00',
    note: 'между точками фактор интерполируется линейно, ниже 8 и выше 25 — прижимается к краю: '
      + 'точки 12/16 не мерились отдельно (это эвристика уровня «полное расписание / черновик»), '
      + 'а 25 — расписание модели',
    values: Object.freeze([
      Object.freeze({ steps: 8, factor: 0.6 }),
      Object.freeze({ steps: 12, factor: 0.75 }),
      Object.freeze({ steps: 16, factor: 0.85 }),
      Object.freeze({ steps: 25, factor: 1 }),
    ]),
  }),
  attention: Object.freeze({
    kind: 'assumption',
    source: 'AGENTS.md#§4 «Скользящее окно внимания»: резкость 14.8 (окно) против 15.2 (плотное)',
    note: 'окно экономит время на крупных канвасах, но стоит ~2.6 % резкости; VRAM оно не меняет, '
      + 'поэтому платит именно качеством',
    values: Object.freeze({ dense: 1, window: 0.97 }),
  }),
  comfort: Object.freeze({
    kind: 'assumption',
    source: '.pi/plans/render-settings-kit.md#A4/A6: 5 734 400 px — замер 314 с / 10.3 ГиБ, 8 355 840 px — «рябь», 513 с / 13.8 ГиБ',
    note: 'спад линейный по пикселям: до okMaxPx фактура модели держится, к hardMaxPx уходит в рябь '
      + '(это уже стоимость 8.3 Мп, а не режима), выше hardMaxPx нативного пути нет',
    fullPx: SIZE_LIMITS.okMaxPx,
    floorPx: SIZE_LIMITS.hardMaxPx,
    floorFactor: 0.75,
  }),
});

/**
 * Фактор шагов: линейная интерполяция по таблице A6, прижатие к краям за её пределами.
 *
 * @param {number} steps шаги сэмплера (≥ 1)
 * @returns {number} множитель 0.60…1.00
 */
export function stepsFactor(steps) {
  if (!Number.isInteger(steps) || steps < 1) {
    throw new RangeError(`stepsFactor: ожидаются целые шаги ≥ 1, получено ${steps}`);
  }
  const values = QUALITY_FACTORS.steps.values;
  const first = values[0];
  const last = values[values.length - 1];
  if (steps <= first.steps) return first.factor;
  if (steps >= last.steps) return last.factor;
  for (let index = 1; index < values.length; index += 1) {
    const left = values[index - 1];
    const right = values[index];
    if (steps <= right.steps) {
      const share = (steps - left.steps) / (right.steps - left.steps);
      return left.factor + (right.factor - left.factor) * share;
    }
  }
  /* c8 ignore next */
  return last.factor;
}

/**
 * Фактор внимания: плотное `1.00`, окно `0.97` (плата за −2.6 % резкости).
 *
 * @param {'dense'|'window'} mode
 * @returns {number}
 */
export function attentionFactor(mode) {
  const values = QUALITY_FACTORS.attention.values;
  if (mode === 'dense') return values.dense;
  if (mode === 'window') return values.window;
  throw new RangeError(`attentionFactor: ожидается 'dense'|'window', получено ${JSON.stringify(mode)}`);
}

/**
 * Фактор комфорта: `1.00` до `fullPx`, линейный спад до `floorFactor` на `floorPx`,
 * выше — `null` (нативного пути нет).
 *
 * @param {number} pixels пиксели канваса (native, без апскейла)
 * @returns {number|null} 1.00…0.75 либо `null` за потолком
 */
export function comfortFactor(pixels) {
  if (!Number.isFinite(pixels) || pixels <= 0) {
    throw new RangeError(`comfortFactor: пиксели должны быть > 0, получено ${pixels}`);
  }
  const { fullPx, floorPx, floorFactor } = QUALITY_FACTORS.comfort;
  if (pixels <= fullPx) return 1;
  if (pixels > floorPx) return null;
  const share = (pixels - fullPx) / (floorPx - fullPx);
  return 1 - (1 - floorFactor) * share;
}

/**
 * @typedef {Object} QualityScore
 * @property {number|null} q рейтинг (`null` — выше «серой» зоны, нативного пути нет)
 * @property {number} nativePixels пиксели native-канваса
 * @property {number} stepsFactor фактор шагов
 * @property {number} attentionFactor фактор внимания
 * @property {number|null} comfortFactor фактор комфорта (`null` за потолком)
 * @property {boolean} beyondComfort `true`, когда Q обнулён потолком (вердикт `impossible`)
 */

/**
 * Q по A6: `√pixels · steps · attention · comfort`. Апскейла/постобработки здесь нет.
 *
 * @param {{pixels: number, steps: number, attention: 'dense'|'window'}} input
 * @returns {QualityScore}
 */
export function qualityScore(input) {
  const { pixels, steps, attention } = input;
  const comfort = comfortFactor(pixels);
  const score = {
    q: null,
    nativePixels: pixels,
    stepsFactor: stepsFactor(steps),
    attentionFactor: attentionFactor(attention),
    comfortFactor: comfort,
    beyondComfort: comfort === null,
  };
  if (comfort === null) return score;
  return {
    ...score,
    q: Math.sqrt(pixels) * score.stepsFactor * score.attentionFactor * comfort,
  };
}
