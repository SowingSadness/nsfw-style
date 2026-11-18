// @ts-check
/**
 * Ранжирование сочетаний (шаг 5 плана, ADR A6): Парето-фронт по `(Q↑, time↓, headroom↑)` и
 * взвешенный победитель по весам уровня качества.
 *
 * Q — эвристика A6 (`quality.mjs`), поэтому победитель не «единственно верный»: план показывает
 * весь фронт, а выбор объясняется числами (Q, время кадра, запас VRAM). Веса уровней — это и есть
 * различие `draft`/`standard`/`max` в части выбора сочетания (шаги уровня и политика гейтов
 * приходят из `levels.mjs`):
 *
 * | уровень | Q | время | запас VRAM | зачем |
 * |---|---|---|---|---|
 * | `draft` | 0.25 | 0.65 | 0.10 | черновик: важна скорость набора кадров |
 * | `standard` | 0.45 | 0.35 | 0.20 | рабочий кадр: баланс качества и предсказуемости |
 * | `max` | 0.70 | 0.20 | 0.10 | максимум качества; запас важен лишь как tie-break |
 *
 * Метрики нормируются min-max по набору (Q и запас — «больше лучше», время — «меньше лучше»),
 * поэтому веса сравнимы между собой. Tie-break стабилен и не зависит от порядка входа:
 * Q → меньше времени → больше запаса → `id`.
 *
 * Модуль изоморфный: без `node:`/DOM.
 */

/**
 * Веса уровней качества (A3 × A6). План фиксирует цель уровня, а не число — веса подобраны так,
 * чтобы на одинаковых гейтах получались ожидаемые планом ответы: `max` — полное расписание,
 * `draft` — 8 шагов, `standard` — то же, что `max`, когда потолок `ok` и окно не разрешено.
 *
 * @type {Readonly<Record<'draft'|'standard'|'max', Readonly<{q: number, time: number, headroom: number}>>}
 */
export const LEVEL_WEIGHTS = Object.freeze({
  draft: Object.freeze({ q: 0.25, time: 0.65, headroom: 0.1 }),
  standard: Object.freeze({ q: 0.45, time: 0.35, headroom: 0.2 }),
  max: Object.freeze({ q: 0.7, time: 0.2, headroom: 0.1 }),
});

/** @typedef {{q: number, timeS: number, headroomGib: number}} Metrics */

/**
 * Метрики сочетания. Принимает и `candidates.mjs`-объект (`metrics`), и «сырой»
 * `{quality, estimates}` — чтобы ранжирование можно было тестировать отдельно от генератора.
 *
 * @param {{metrics?: Metrics, quality?: {q: number|null}, estimates?: any}} item
 * @returns {Metrics}
 */
export function metricsOf(item) {
  if (item.metrics) return item.metrics;
  const quality = item.quality ?? { q: 0 };
  const estimates = item.estimates ?? {};
  return {
    q: quality.q ?? 0,
    timeS: estimates.time?.genPerFrameS ?? 0,
    headroomGib: estimates.vram?.headroomGib ?? 0,
  };
}

/**
 * Доминирует ли `a` точку `b`: не хуже по всем трём осям и лучше хотя бы по одной.
 *
 * @param {Metrics} a
 * @param {Metrics} b
 * @returns {boolean}
 */
function dominates(a, b) {
  const notWorse = a.q >= b.q && a.timeS <= b.timeS && a.headroomGib >= b.headroomGib;
  const better = a.q > b.q || a.timeS < b.timeS || a.headroomGib > b.headroomGib;
  return notWorse && better;
}

/**
 * Парето-фронт: точки, которые никто не доминирует. Порядок входа сохраняется (детерминизм).
 *
 * @template {{id?: string}} T
 * @param {T[]} items
 * @returns {T[]}
 */
export function paretoFront(items) {
  return items.filter((item, index) => {
    const mine = metricsOf(/** @type {any} */ (item));
    return !items.some((other, otherIndex) => otherIndex !== index
      && dominates(metricsOf(/** @type {any} */ (other)), mine));
  });
}

/**
 * @typedef {Object} Scored
 * @property {number} score итоговый вес 0…1 (после нормировки весов)
 * @property {number} qNorm нормированный Q
 * @property {number} timeNorm нормированное время («меньше — лучше»)
 * @property {number} headroomNorm нормированный запас VRAM
 */

/**
 * Min-max нормировка одной оси. Вырожденный набор (все значения равны) даёт `1` по этой оси:
 * ось не различает сочетания и не должна сдвигать выбор.
 *
 * @param {number[]} values
 * @param {number} value
 * @param {boolean} higherIsBetter
 * @returns {number}
 */
function normalize(values, value, higherIsBetter) {
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (max - min <= 1e-12) return 1;
  const norm = (value - min) / (max - min);
  return higherIsBetter ? norm : 1 - norm;
}

/**
 * Оценки сочетаний по весам уровня: `score = Σ wᵢ·normᵢ`, веса нормируются на сумму.
 *
 * @param {any[]} items
 * @param {{q: number, time: number, headroom: number}} weights
 * @returns {Map<any, Scored>}
 */
export function scoreOf(items, weights) {
  const total = weights.q + weights.time + weights.headroom;
  if (!(total > 0)) {
    throw new RangeError('scoreOf: сумма весов должна быть > 0');
  }
  const metrics = items.map((item) => metricsOf(item));
  const qs = metrics.map((m) => m.q);
  const times = metrics.map((m) => m.timeS);
  const headrooms = metrics.map((m) => m.headroomGib);
  /** @type {Map<any, Scored>} */
  const scores = new Map();
  items.forEach((item, index) => {
    const qNorm = normalize(qs, qs[index], true);
    const timeNorm = normalize(times, times[index], false);
    const headroomNorm = normalize(headrooms, headrooms[index], true);
    scores.set(item, {
      score: (weights.q * qNorm + weights.time * timeNorm + weights.headroom * headroomNorm) / total,
      qNorm,
      timeNorm,
      headroomNorm,
    });
  });
  return scores;
}

/**
 * Победитель по весам. Tie-break: Q → меньше времени → больше запаса VRAM → `id`.
 * Функция чистая и не зависит от порядка входа (сравнение по значениям, а не по индексу).
 *
 * @param {any[]} items
 * @param {{q: number, time: number, headroom: number}} weights
 * @returns {any|null} лучший элемент или `null` на пустом наборе
 */
export function pickBest(items, weights) {
  if (items.length === 0) return null;
  const scores = scoreOf(items, weights);
  const comparator = compareForBest(scores);
  return [...items].sort(comparator)[0];
}

/**
 * Компаратор «лучший первым»: `score` (до 1e-9) → Q → время → запас → `id`.
 *
 * @param {Map<any, Scored>} scores
 * @returns {(a: any, b: any) => number}
 */
function compareForBest(scores) {
  return (a, b) => {
    const sa = /** @type {Scored} */ (scores.get(a)).score;
    const sb = /** @type {Scored} */ (scores.get(b)).score;
    const rounded = Math.round(sb * 1e9) - Math.round(sa * 1e9);
    if (rounded !== 0) return rounded;
    const ma = metricsOf(a);
    const mb = metricsOf(b);
    if (mb.q !== ma.q) return mb.q - ma.q;
    if (ma.timeS !== mb.timeS) return ma.timeS - mb.timeS;
    if (mb.headroomGib !== ma.headroomGib) return mb.headroomGib - ma.headroomGib;
    return String(a.id ?? '').localeCompare(String(b.id ?? ''));
  };
}

/**
 * Полный ранжированный список: сочетания лучшими вперёд с полем `score`. Нужен `Plan.combinations`
 * (таблица в UI) и тестам: `pickBest(...).id === rankCandidates(...)[0].id`.
 *
 * @param {any[]} items
 * @param {{q: number, time: number, headroom: number}} weights
 * @returns {Array<any & {score: number, qNorm: number, timeNorm: number, headroomNorm: number}>}
 */
export function rankCandidates(items, weights) {
  if (items.length === 0) return [];
  const scores = scoreOf(items, weights);
  const comparator = compareForBest(scores);
  return [...items].sort(comparator).map((item) => ({ ...item, .../** @type {Scored} */ (scores.get(item)) }));
}
