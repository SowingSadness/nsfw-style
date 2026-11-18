// @ts-check
/**
 * Числовые примитивы `renderkit`: питоновское округление, сетка канваса, токены, KV-кеш
 * и форматтеры для человека.
 *
 * Модуль изоморфный: никаких `node:`, `process`, DOM — только арифметика.
 * Значения констант — из наших замеров (см. `.pi/plans/render-settings-kit.md`).
 */

/** @typedef {{ width: number, height: number }} Canvas */

/** Сетка канваса: каждая сторона кратна 32 (латент 8× + патч 2×). */
export const CANVAS_GRID = 32;

/** Пикселей на латентный токен по каждой оси: `tokens = (W/16)·(H/16)`. */
export const TOKEN_PX = 16;

/**
 * Цена одного префиксного токена в KV-кеше, ГиБ/токен.
 * `32 слоя × 2 (K и V) × 4096 × 2 Б = 512 КиБ` — ровно 0.000488 ГиБ.
 */
export const KV_GIB_PER_TOKEN = 0.000488;

/**
 * Питоновский `round()` для чисел с плавающей точкой: половина округляется **к чётному**
 * (half-to-even), а не от нуля, как у `Math.round`.
 *
 * Нужен потому, что `diffusers.pipelines.qwenimage21.pipeline_qwenimage21.calculate_dimensions`
 * вызывает `round(width / 32) * 32`, и полутон там решается именно так:
 * `round(28.5 / 1) === 28`, из-за чего 912p при 16:9 даёт 1632×**896**, а не 1632×928.
 *
 * @param {number} value конечное число
 * @returns {number} целое
 */
export function pyRound(value) {
  if (!Number.isFinite(value)) {
    throw new RangeError(`pyRound: ожидается конечное число, получено ${value}`);
  }
  const floor = Math.floor(value);
  const rest = value - floor;
  if (rest > 0.5) return floor + 1;
  if (rest < 0.5) return floor;
  // Ровно половина: к чётному (для отрицательных floor уже меньше по модулю, логика та же).
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * Округление к сетке 32 питоновским `round` (half-to-even), как в `calculate_dimensions`.
 *
 * @param {number} value
 * @returns {number}
 */
export function snapToGrid(value) {
  return pyRound(value / CANVAS_GRID) * CANVAS_GRID;
}

/**
 * Точная копия `calculate_dimensions(target_area, ratio)` из
 * `diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py` (строки 149–157):
 *
 * ```python
 * width = math.sqrt(target_area * ratio)
 * height = width / ratio
 * width = round(width / 32) * 32
 * height = round(height / 32) * 32
 * ```
 *
 * В пайплайне `target_area = resolution ** 2`, а `resolution ≈ H·√ratio`, поэтому
 * `w = √(H·ratio·H·ratio) = H·ratio`, `h = H`. Здесь это записано буквально, как в Python,
 * чтобы совпадение с системой держалось на копии формулы, а не на её упрощении.
 *
 * @param {number} sizePx целевая высота кадра (`--size NNNp`), > 0
 * @param {number} ratio пропорция кадра `W/H`, > 0
 * @returns {Canvas} канвас на сетке 32
 */
export function canvasFor(sizePx, ratio) {
  if (!Number.isFinite(sizePx) || sizePx <= 0) {
    throw new RangeError(`canvasFor: sizePx должен быть > 0, получено ${sizePx}`);
  }
  if (!Number.isFinite(ratio) || ratio <= 0) {
    throw new RangeError(`canvasFor: ratio должен быть > 0, получено ${ratio}`);
  }
  const width = Math.sqrt(sizePx * ratio * sizePx * ratio);
  const height = width / ratio;
  return { width: snapToGrid(width), height: snapToGrid(height) };
}

/**
 * Латентные токены канваса: `(W/16)·(H/16)`.
 * Канвас на сетке 32 делится на 16 без остатка, поэтому результат всегда целый.
 *
 * @param {number} width
 * @param {number} height
 * @returns {number}
 */
export function tokensFor(width, height) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new RangeError(`tokensFor: ожидаются целые > 0, получено ${width}×${height}`);
  }
  if (width % TOKEN_PX !== 0 || height % TOKEN_PX !== 0) {
    // Кратность 16 — инвариант канваса (сетка 32); нечётная сторона = ошибка вызова.
    throw new RangeError(`tokensFor: стороны должны быть кратны ${TOKEN_PX}, получено ${width}×${height}`);
  }
  return (width / TOKEN_PX) * (height / TOKEN_PX);
}

/**
 * Сетка референса для значения `ref_px` — точная копия `config.py::embedding_crop`:
 * `ref_h = max(256, min(ref_px, canvasH))`, `ref_w = max(256, min(round(ref_h·ratio), canvasW))`,
 * затем пол до кратного 32 (требование `image_processor` пайплайна).
 * `ref_px: 0` — сетка самого кадра (референс виден в том масштабе, в котором его рисуют).
 *
 * **Единственная** функция сетки референса: её используют и `memory.resolveRefPx` (решение
 * `ref_px` и `memory.ref.tokensPerRef`), и `cost.prefixTokens` (токены префикса KV), и
 * `tools/fit.mjs`. Раньше префикс считался как `(refPx/16)²` (3600 токенов при `ref_px 960`),
 * а `memory.ref` рапортовал сетку 1696×960 = 6360 — две формулы на одну величину занижали KV на треть.
 *
 * @param {number} refPx высота сетки референса (`0` — сетка кадра)
 * @param {Canvas} canvas канвас кадра
 * @returns {{refWidth: number, refHeight: number, tokensPerRef: number}}
 */
export function refGrid(refPx, canvas) {
  if (refPx <= 0) {
    return {
      refWidth: canvas.width,
      refHeight: canvas.height,
      tokensPerRef: tokensFor(canvas.width, canvas.height),
    };
  }
  const ratio = canvas.width / Math.max(1, canvas.height);
  const refHeight = Math.max(256, Math.min(refPx, Math.floor(canvas.height)));
  const refWidth = Math.max(256, Math.min(pyRound(refHeight * ratio), Math.floor(canvas.width)));
  const snappedWidth = Math.max(256, Math.floor(refWidth / CANVAS_GRID) * CANVAS_GRID);
  const snappedHeight = Math.max(256, Math.floor(refHeight / CANVAS_GRID) * CANVAS_GRID);
  return {
    refWidth: snappedWidth,
    refHeight: snappedHeight,
    tokensPerRef: tokensFor(snappedWidth, snappedHeight),
  };
}

/**
 * Цена префикса в KV-кеше, ГиБ (512 КиБ на токен, см. `KV_GIB_PER_TOKEN`).
 *
 * @param {number} prefixTokens токены префикса (текст + все референсы)
 * @returns {number} ГиБ
 */
export function kvGib(prefixTokens) {
  if (!Number.isFinite(prefixTokens) || prefixTokens < 0) {
    throw new RangeError(`kvGib: ожидается неотрицательное число токенов, получено ${prefixTokens}`);
  }
  return KV_GIB_PER_TOKEN * prefixTokens;
}

/**
 * `12.85 ГиБ`
 *
 * @param {number} value ГиБ
 * @param {number} [digits] знаков после запятой (по умолчанию 2)
 * @returns {string}
 */
export function formatGib(value, digits = 2) {
  return `${Number(value).toFixed(digits)} ГиБ`;
}

/**
 * `228.6 МиБ`
 *
 * @param {number} value МиБ
 * @param {number} [digits] знаков после запятой (по умолчанию 1)
 * @returns {string}
 */
export function formatMib(value, digits = 1) {
  return `${Number(value).toFixed(digits)} МиБ`;
}

/**
 * Длительность по-человечески: `8 с`, `205 с`, `3 мин 25 с`, `1 ч 2 мин`.
 * Единица измерения — всегда секунды, округление до целого.
 *
 * @param {number} value секунды
 * @returns {string}
 */
export function formatSeconds(value) {
  if (!Number.isFinite(value) || value < 0) {
    throw new RangeError(`formatSeconds: ожидается неотрицательное число секунд, получено ${value}`);
  }
  const total = Math.round(value);
  if (total < 60) return `${total} с`;
  if (total < 3600) {
    const minutes = Math.floor(total / 60);
    const seconds = total % 60;
    return seconds === 0 ? `${minutes} мин` : `${minutes} мин ${seconds} с`;
  }
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  return minutes === 0 ? `${hours} ч` : `${hours} ч ${minutes} мин`;
}
