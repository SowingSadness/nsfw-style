// @ts-check
/**
 * Геометрия кадра (ADR A4 плана): от запрошенных `W×H` к достижимому native-канвасу.
 *
 * Порядок:
 * 1. `ratio = W/H`; совпадение с одной из разрешённых пропорций (`16:9|4:3|1:1|2:3`) в пределах ±0.5 %
 *    → `aspect` = этот ключ (работает и T2I, и правка по референсу);
 * 2. иначе при `refs ≥ 1` → `aspect: null` (кадр берёт пропорцию референса) + warning
 *    `ref-ratio-must-match`;
 * 3. иначе (T2I, экзотическая пропорция) → `impossible` + `aspect-unsupported-t2i`;
 * 4. `sizePx = H`; `canvas = canvasFor(sizePx, ratio)` — точная копия `calculate_dimensions` системы;
 * 5. отличие канваса от запроса → `snap` + `reachable` (две ближайшие точки сетки 32) + warning
 *    `native-size-snapped`;
 * 6. потолок: `Wc·Hc ≤ 3200×1792` → `ok`; до `3840×2176` → `degraded` + `comfort-limit`;
 *    выше → `impossible` + `no-native-path-for-size`.
 *
 * Апскейла здесь нет и не будет: «добрать» запрошенный размер постобработкой молча нельзя (A5).
 */

import { CANVAS_GRID, canvasFor, tokensFor } from '../units.mjs';

/** @typedef {{key: string, ratio: number, delta: number}} AspectMatch */
/** @typedef {{code: string, detail: string}} Warning */
/** @typedef {[number, number]} GridPoint */

/** Разрешённые системой пропорции (`--aspect` знает только эти четыре). */
export const ASPECTS = Object.freeze([
  Object.freeze({ key: '16:9', ratio: 16 / 9 }),
  Object.freeze({ key: '4:3', ratio: 4 / 3 }),
  Object.freeze({ key: '1:1', ratio: 1 }),
  Object.freeze({ key: '2:3', ratio: 2 / 3 }),
]);

/** Допуск совпадения пропорции (доля, ±0.5 %). */
export const ASPECT_TOLERANCE = 0.005;

/** Измеренные потолки (наши замеры, 16 ГиБ VRAM). */
export const SIZE_LIMITS = Object.freeze({
  /** 3200×1792 = 5.7344 Мп — реальный max: 314 с / 10.3 ГиБ, окно 384+384. */
  okMaxPx: 3200 * 1792,
  /** 3840×2176 = 8.3558 Мп — «серая» зона: 513 с / 13.8 ГиБ, фактура уходит в рябь. */
  hardMaxPx: 3840 * 2176,
});

/** Коды предупреждений геометрии (общий словарь — `explain.mjs` на шагах 5+). */
export const GEOMETRY_WARNINGS = Object.freeze({
  NATIVE_SIZE_SNAPPED: 'native-size-snapped',
  REF_RATIO_MUST_MATCH: 'ref-ratio-must-match',
  ASPECT_UNSUPPORTED_T2I: 'aspect-unsupported-t2i',
  COMFORT_LIMIT: 'comfort-limit',
  NO_NATIVE_PATH_FOR_SIZE: 'no-native-path-for-size',
});

/**
 * Вердикты геометрии. `degraded` — рендерится, но фактура уходит в рябь (зона 4K);
 * `impossible` — нативно недостижимо.
 *
 * @typedef {'ok'|'degraded'|'impossible'} GeometryVerdict
 */

/**
 * @typedef {Object} Geometry
 * @property {{width: number, height: number}} requested
 * @property {number} ratio запрошенная пропорция `W/H`
 * @property {string|null} aspect ключ `--aspect` или `null` (пропорция референса)
 * @property {number} sizePx высота канваса = `H` запроса
 * @property {{width: number, height: number}} canvas native-канвас на сетке 32
 * @property {number} pixels `Wc·Hc`
 * @property {number} tokens латентные токены `(Wc/16)·(Hc/16)`
 * @property {{dx: number, dy: number, reachable: GridPoint[]}} snap сдвиг канваса и ближайшие точки сетки
 * @property {GeometryVerdict} verdict
 * @property {number} limitPx пиксельный потолок, в котором оказался вердикт
 * @property {AspectMatch[]} nearestAspects две ближайшие разрешённые пропорции (для `suggestions`)
 * @property {Warning[]} warnings
 */

/**
 * Ближайшая разрешённая пропорция в пределах допуска.
 *
 * @param {number} ratio `W/H`
 * @returns {AspectMatch|null}
 */
export function matchAspect(ratio) {
  let best = null;
  for (const aspect of ASPECTS) {
    const delta = Math.abs(ratio - aspect.ratio) / aspect.ratio;
    if (delta <= ASPECT_TOLERANCE && (best === null || delta < best.delta)) {
      best = { key: aspect.key, ratio: aspect.ratio, delta };
    }
  }
  return best;
}

/**
 * Две ближайшие разрешённые пропорции — по возрастанию отклонения.
 * Нужны для `suggestions` в случае `aspect-unsupported-t2i`.
 *
 * @param {number} ratio
 * @param {number} [limit]
 * @returns {AspectMatch[]}
 */
export function nearestAspects(ratio, limit = 2) {
  return ASPECTS.map((aspect) => ({
    key: aspect.key,
    ratio: aspect.ratio,
    delta: Math.abs(ratio - aspect.ratio) / aspect.ratio,
  }))
    .sort((a, b) => a.delta - b.delta || a.key.localeCompare(b.key))
    .slice(0, limit);
}

/**
 * Соседние значения сетки 32 по одной оси: пол и потолок; если значение уже на сетке,
 * берутся ±32 (иначе «пол» и «потолок» совпали бы и соседей не осталось).
 *
 * @param {number} value
 * @param {boolean} expandOnGrid
 * @returns {number[]}
 */
function gridNeighbors(value, expandOnGrid) {
  const floor = Math.floor(value / CANVAS_GRID) * CANVAS_GRID;
  const ceil = Math.ceil(value / CANVAS_GRID) * CANVAS_GRID;
  const values = new Set([floor, ceil]);
  if (expandOnGrid && floor === value) {
    values.add(value - CANVAS_GRID);
    values.add(value + CANVAS_GRID);
  }
  return [...values].filter((v) => v > 0).sort((a, b) => a - b);
}

/**
 * Углы сетки 32 вокруг запроса (без самого запроса), ближайшие первыми.
 *
 * @param {number} width
 * @param {number} height
 * @param {boolean} expandOnGrid
 * @returns {Array<{x: number, y: number, d: number, score: number, ady: number, adx: number, dx: number, dy: number}>}
 */
function gridCorners(width, height, expandOnGrid) {
  const xs = gridNeighbors(width, expandOnGrid);
  const ys = gridNeighbors(height, expandOnGrid);
  /** @type {Array<{x: number, y: number, d: number, score: number, ady: number, adx: number, dx: number, dy: number}>} */
  const points = [];
  for (const x of xs) {
    for (const y of ys) {
      const dx = x - width;
      const dy = y - height;
      const d = Math.hypot(dx, dy);
      if (d === 0) continue; // сам запрос уже на сетке — он не «сосед»
      // Меньше «штрафа» — тем естественнее точка: рост вправо/вниз предпочтительнее.
      const score = (dx < 0 ? 1 : 0) + (dy > 0 ? 1 : 0);
      points.push({ x, y, d, score, ady: Math.abs(dy), adx: Math.abs(dx), dx, dy });
    }
  }
  return points;
}

/**
 * Две ближайшие к запросу точки сетки 32 (углы вокруг запроса, без самой запрошенной точки).
 * Порядок: расстояние → «естественность» (`+dx`/`−dy` лучше) → `|dy|` → `|dx|`.
 * Так для 1920×1080 получаются `1920×1088` и `1920×1056` (как в примере плана),
 * а для 2560×1440 — `2592×1440` и `2560×1408`.
 *
 * @param {number} width
 * @param {number} height
 * @returns {GridPoint[]}
 */
export function nearestGridPoints(width, height) {
  const compare = (
    /** @type {{d: number, score: number, ady: number, adx: number, dx: number, dy: number, x: number, y: number}} */ a,
    /** @type {{d: number, score: number, ady: number, adx: number, dx: number, dy: number, x: number, y: number}} */ b,
  ) => a.d - b.d
    || a.score - b.score
    || a.ady - b.ady
    || a.adx - b.adx
    || b.dx - a.dx
    || a.dy - b.dy
    || a.x - b.x
    || a.y - b.y;

  // Обычный случай — углы вокруг запроса.
  let points = gridCorners(width, height, false).sort(compare);
  if (points.length < 2) {
    // Запрос уже на сетке: углы выродились в одну точку — расширяем оси на ±32.
    points = gridCorners(width, height, true).sort(compare);
  }
  return points.slice(0, 2).map((p) => /** @type {GridPoint} */ ([p.x, p.y]));
}

/**
 * Разбор геометрии запроса. Вход уже нормализован схемой, но функция защищена от мусора,
 * чтобы быть пригодной и для прямых вызовов в тестах/UI.
 *
 * @param {{image: {width: number, height: number}, refs: number}} request
 * @returns {Geometry}
 */
export function resolveGeometry(request) {
  const width = request.image.width;
  const height = request.image.height;
  const refs = request.refs;
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new RangeError(`resolveGeometry: ожидаются целые W×H, получено ${width}×${height}`);
  }
  if (!Number.isInteger(refs) || refs < 0) {
    throw new RangeError(`resolveGeometry: refs должен быть целым ≥ 0, получено ${refs}`);
  }

  const ratio = width / height;
  const matched = matchAspect(ratio);
  /** @type {Warning[]} */
  const warnings = [];
  /** @type {string|null} */
  let aspect = null;
  let aspectImpossible = false;

  if (matched) {
    aspect = matched.key;
  } else if (refs >= 1) {
    // Пропорцию задаст первый референс: `--aspect` не ставится, в ключ кэша идёт crop референса.
    warnings.push({
      code: GEOMETRY_WARNINGS.REF_RATIO_MUST_MATCH,
      detail: `пропорция ${formatRatio(width, height)} не из списка ${ASPECTS.map((a) => a.key).join('|')}: `
        + 'кадр возьмёт пропорцию первого референса — референс обязан быть этой пропорции',
    });
  } else {
    aspectImpossible = true;
    warnings.push({
      code: GEOMETRY_WARNINGS.ASPECT_UNSUPPORTED_T2I,
      detail: `T2I (refs=0) умеет только ${ASPECTS.map((a) => a.key).join('|')}; `
        + `ближайшие: ${nearestAspects(ratio).map((a) => a.key).join(', ')}`,
    });
  }

  const sizePx = height;
  const canvas = canvasFor(sizePx, ratio);
  const pixels = canvas.width * canvas.height;
  const tokens = tokensFor(canvas.width, canvas.height);
  const dx = canvas.width - width;
  const dy = canvas.height - height;
  const snap = { dx, dy, reachable: nearestGridPoints(width, height) };

  if (dx !== 0 || dy !== 0) {
    warnings.push({
      code: GEOMETRY_WARNINGS.NATIVE_SIZE_SNAPPED,
      detail: `${width}×${height} недостижим нативно (сетка ${CANVAS_GRID}) → ${canvas.width}×${canvas.height} `
        + `(dx ${dx > 0 ? '+' : ''}${dx}, dy ${dy > 0 ? '+' : ''}${dy}); рядом: ${snap.reachable.map((p) => p.join('×')).join(', ')}`,
    });
  }

  /** @type {GeometryVerdict} */
  let verdict;
  /** @type {number} */
  let limitPx;

  if (aspectImpossible) {
    verdict = 'impossible';
    limitPx = SIZE_LIMITS.okMaxPx;
  } else if (pixels <= SIZE_LIMITS.okMaxPx) {
    verdict = 'ok';
    limitPx = SIZE_LIMITS.okMaxPx;
  } else if (pixels <= SIZE_LIMITS.hardMaxPx && dx === 0 && dy === 0) {
    // «Серая» зона — только для размеров, которые существуют на сетке: она уже на пределе,
    // округлять в неё вверх нельзя (иначе каждый запрос «чуть больше» проходил бы как рендер).
    verdict = 'degraded';
    limitPx = SIZE_LIMITS.hardMaxPx;
    warnings.push({
      code: GEOMETRY_WARNINGS.COMFORT_LIMIT,
      detail: `${formatPixels(pixels)} выше комфортных ${formatPixels(SIZE_LIMITS.okMaxPx)}: рендерится, `
        + 'но мелкая фактура уходит в рябь (замер 513 с / 13.8 ГиБ, окно 1088/448)',
    });
  } else {
    verdict = 'impossible';
    limitPx = SIZE_LIMITS.hardMaxPx;
    warnings.push({
      code: GEOMETRY_WARNINGS.NO_NATIVE_PATH_FOR_SIZE,
      detail: `${formatPixels(pixels)} выше потолка ${formatPixels(SIZE_LIMITS.hardMaxPx)} `
        + `(${canvas.width}×${canvas.height}); ближайшие сеточные: ${snap.reachable.map((p) => p.join('×')).join(', ')}`,
    });
  }

  return {
    requested: { width, height },
    ratio,
    aspect,
    sizePx,
    canvas,
    pixels,
    tokens,
    snap,
    verdict,
    limitPx,
    nearestAspects: nearestAspects(ratio),
    warnings,
  };
}

/**
 * @param {number} width
 * @param {number} height
 * @returns {string}
 */
function formatRatio(width, height) {
  return `${width}:${height} (${(width / height).toFixed(3)})`;
}

/**
 * Мегапиксели по-человечески: `5.73 Мп`.
 *
 * @param {number} pixels
 * @returns {string}
 */
function formatPixels(pixels) {
  return `${(pixels / 1e6).toFixed(2)} Мп`;
}
