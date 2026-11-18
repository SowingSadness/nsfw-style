#!/usr/bin/env node
// @ts-check
/**
 * Калибровка модели оценок по замерам: `src/data/measurements.mjs` → `src/data/model.mjs`.
 *
 * Никаких зависимостей: нормальные уравнения + метод Гаусса с выбором ведущего элемента.
 * Модель (см. `.pi/plans/render-settings-kit.md`, «Модель оценок» и A8):
 *
 * ```
 * tokens      = (W/16)·(H/16)                                  # факт, не фит
 * prefix      = textTokens + refs·tokensPerRef(refPx)            # 512 КиБ/токен (units.mjs),
 *             tokensPerRef = (gridW/16)·(gridH/16) из `refGrid`  # та же функция, что у memory.mjs
 * peakVram    = resident(model, offload)                        # none: 7.67 фикс.; group: фит; h3: фит
 *             + a0 + a1·x/(1 + a2·x),  x = tokens/1000        # активации насыщаются; окно в VRAM
 *                                                              # не входит (измерено: 1792p —
 *                                                              # 10.3 ГиБ и с окном, и без)
 *             + (kvCache === 'true' ? 0.000488·prefix : 0)      # K/V префикса в VRAM
 *             + overhead                                        # контекст CUDA + резерв + рабочий стол
 * ln tStep    = b0 + b1·ln(tokens/4000) + b2·ln²(tokens/4000) + b3·[group]
 *             + b4·[kv offload] + b5·[h3] + b6·[turbo] + b7·ln(steps/fixedSteps)
 *             + b8·[window] + b9·[h3]·ln(tokens/4000)   # H3 упирается в стриминг 30.9 ГиБ, а не в токены
 *             + b10·[group]·ln(tokens/4000)             # в group шаг тоже упирается в стриминг блоков
 *                                        # b8 — измерено ln(314/376), в МНК не входит;
 *                                        # b2 — изгиб эластичности (≈2 у малых канвасов, ≈1.2 у больших)
 * genS        = steps·exp(ln tStep)
 * loadS       = константа по модели (импорт + размещение, fp8-кэш на диске)
 * encodeS     = cold: константа (GPU 8B / h3 32B) или линейно по токенам (CPU); hit: 0
 * bandwidth   = {nvme: 3.0, ssd: 0.45, hdd: 0.13} ГиБ/с
 * ```
 *
 * **Окно внимания в VRAM-фит не входит.** Замер (`AGENTS.md` §4, «Скользящее окно внимания»):
 * 1792p плотным вниманием — 376 с при **том же** пике 10.3 ГиБ, что с окном (314 с). Окно покупает
 * время (−17 %) ценой резкости (−2.6 %), а не память; поэтому единственное место окна в модели —
 * коэффициент `bWindow` во времени, и он **измерен**: ln(314/376) = −0.18, а не подобран (при
 * свободном подборе уходил в плюс, компенсируя изгиб эластичности по токенам, который теперь явно
 * выражен членом `bTokens2` во времени и насыщением `a2` в VRAM).
 * Раньше окно стояло регрессором и в VRAM, из-за чего плотное внимание на 1792p оценивалось
 * в 15.95 ГиБ (ложный `impossible`).
 *
 * VRAM-фит идёт по **одной метрике** — полное `memory.used` карты (`nvidia-smi`): так мерили
 * `docs/bench/vram-tiers.md` и `docs/bench/kv-cache-offload.md`, и так отвечает на вопрос «влезет
 * ли в 16 ГБ». Строки с другой метрикой (`presets.md` — пик аллокатора torch; `AGENTS.md` §4 —
 * пик процесса) приводятся к этой метрике **измеренной** поправкой `measurementDesktopReserveGib`
 * (2.05 ГиБ: в vram-tiers `полное` − `свой процесс` = 2.05–2.32 ГиБ на четырёх строках) и идут
 * в фит наравне; аудит печатает их со звёздочкой. Время, загрузка и энкод метрикой не страдают —
 * они меряются секундами в любом документе.
 *
 * Запуск:
 * ```
 * node tools/fit.mjs            # пересчитать и записать src/data/model.mjs
 * node tools/fit.mjs --check    # пересчитать и сравнить с файлом (exit 1 при расхождении)
 * ```
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import {
  MEASUREMENTS, auditedRows, fitRows, peakMetricOf, renderRows,
} from '../src/data/measurements.mjs';
import { KV_GIB_PER_TOKEN, refGrid } from '../src/units.mjs';
import { modelCaps } from '../src/model/inventory.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const MODEL_PATH = join(HERE, '..', 'src', 'data', 'model.mjs');

// ───────────────────────────────────────────────────────────────────────────────────────────────
// Константы, которые не подбираются, а берутся из замеров (у каждой — провенанс в MODEL).
// ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Токены текстового префикса (промт + служебные) в наших замерах. Выводится из четырёх строк
 * `[kv]`: 4360−4128, 6592−6360, 8392−8160, 14632−14400 — во всех ровно **232**.
 */
const TEXT_TOKENS = 232;

/**
 * Резидентные fp8-веса базового трансформера при `offload: none` — **измерено**:
 * `AGENTS.md` §4, «веса после загрузки (allocated) 7.67 GiB»; при 4128 токенах
 * 7.67 + 3.63 активаций + 1.55 резерва + 2.62 вне аллокатора = 15.5 ГиБ (пик процесса).
 */
const RESIDENT_NONE_GIB = 7.67;

/**
 * Поправка «пик процесса → полное `memory.used` карты»: рабочий стол, который держал
 * 1797–2371 МиБ VRAM. Измерено как `полное − свой процесс` в `docs/bench/vram-tiers.md`:
 * 7.17−5.11, 8.26−6.21, 11.98−9.93, 12.85−10.80 = 2.06/2.05/2.05/2.05, 7.68−5.36 = 2.32.
 */
const DESKTOP_RESERVE_GIB = 2.05;

/**
 * База активаций вне зависимости от числа токенов: из двух измеренных точек `AGENTS.md` §4
 * (1.59 ГиБ при 1728 токенах, 3.63 при 4128) линейная часть даёт 0.85 ГиБ/1000 токенов,
 * остаток на нуле — 0.12 ГиБ. Наклон `a1` подбирается фитом (у нас разные режимы/KV).
 */
const ACTIVATION_BASE_GIB = 0.12;

/** Ширина полосы чтения с диска, ГиБ/с (`AGENTS.md` §4: NVMe 4.39–6.81 измерено, 1.7–1.9 — mmap). */
const DISK_BANDWIDTH = /** @type {const} */ ({ nvme: 3.0, ssd: 0.45, hdd: 0.13 });

/** Сколько шагов у модели по расписанию (turbo 8, h3 6 — фиксированы, не «настройка»). */
const fixedSteps = (model) => modelCaps(model).fixedSteps;

// ───────────────────────────────────────────────────────────────────────────────────────────────
// Линейная алгебра: нормальные уравнения + Гаусс (без зависимостей).
// ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Решение СЛАУ `A·c = y` методом наименьших квадратов с ridge-регуляризацией:
 * `(AᵀA + λI)·c = Aᵀy`. Реализация — обычный Гаусс с выбором ведущего элемента.
 *
 * @param {number[][]} a матрица `n × k`
 * @param {number[]} y вектор `n`
 * @param {number} [lambda] ridge (по умолчанию 0 — точный МНК)
 * @returns {number[]} коэффициенты длины `k`
 */
export function leastSquares(a, y, lambda = 0) {
  const n = a.length;
  if (n === 0) throw new RangeError('leastSquares: пустая выборка');
  const k = a[0].length;
  if (y.length !== n) throw new RangeError('leastSquares: длина y не совпадает с числом строк');
  /** @type {number[][]} */
  const m = Array.from({ length: k }, () => new Array(k).fill(0));
  /** @type {number[]} */
  const rhs = new Array(k).fill(0);
  for (let r = 0; r < n; r += 1) {
    if (a[r].length !== k) throw new RangeError('leastSquares: строки матрицы разной длины');
    for (let i = 0; i < k; i += 1) {
      rhs[i] += a[r][i] * y[r];
      for (let j = 0; j < k; j += 1) m[i][j] += a[r][i] * a[r][j];
    }
  }
  for (let i = 0; i < k; i += 1) m[i][i] += lambda;

  // Прямой ход.
  for (let col = 0; col < k; col += 1) {
    let pivot = col;
    for (let row = col + 1; row < k; row += 1) {
      if (Math.abs(m[row][col]) > Math.abs(m[pivot][col])) pivot = row;
    }
    if (Math.abs(m[pivot][col]) < 1e-12) {
      throw new RangeError('leastSquares: матрица вырождена (проверьте набор признаков)');
    }
    [m[col], m[pivot]] = [m[pivot], m[col]];
    [rhs[col], rhs[pivot]] = [rhs[pivot], rhs[col]];
    for (let row = col + 1; row < k; row += 1) {
      const factor = m[row][col] / m[col][col];
      if (factor === 0) continue;
      for (let j = col; j < k; j += 1) m[row][j] -= factor * m[col][j];
      rhs[row] -= factor * rhs[col];
    }
  }
  // Обратный ход.
  /** @type {number[]} */
  const c = new Array(k).fill(0);
  for (let row = k - 1; row >= 0; row -= 1) {
    let sum = rhs[row];
    for (let j = row + 1; j < k; j += 1) sum -= m[row][j] * c[j];
    c[row] = sum / m[row][row];
  }
  return c;
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
// Признаки и предсказания.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Токены одного референса в префиксе: сетка `ref_px` из `refGrid` (`units.mjs` — та же функция,
 * которой пользуется `memory.resolveRefPx` и `cost.prefixTokens`), то есть
 * `(gridW/16)·(gridH/16)`; `refPx: 0` — сетка самого кадра.
 *
 * @param {{tokens: number, refs: number, refPx: number | null, width?: number, height?: number}} config
 * @returns {number}
 */
export function refTokensPerRef(config) {
  if (config.width === undefined || config.height === undefined) {
    throw new RangeError('refTokensPerRef: нужен канвас config.width×config.height — '
      + 'сетка референса считается по нему (refGrid)');
  }
  return refGrid(config.refPx ?? 0, { width: config.width, height: config.height }).tokensPerRef;
}

/**
 * Токены префикса (текст + все референсы): 512 КиБ на токен.
 * `refPx: 0` (или `null`) = референс виден в сетке кадра, то есть `tokens` на референс.
 *
 * @param {{tokens: number, refs: number, refPx: number | null, width?: number, height?: number}} config
 * @returns {number}
 */
export function prefixTokens(config) {
  const refs = Number(config.refs ?? 0);
  if (refs === 0) return TEXT_TOKENS;
  return TEXT_TOKENS + refs * refTokensPerRef(config);
}

/**
 * Резидентная часть модели: сколько весов реально живёт в VRAM при данном режиме offload.
 * Ключи — из `MODEL.coefficients.vram.resident`, чтобы предсказание и данные не расходились.
 *
 * @param {typeof import('../src/data/model.mjs').MODEL['coefficients']['vram']} vram
 * @param {string} model
 * @param {'none' | 'group' | null} offload
 * @returns {number}
 */
function residentFor(vram, model, offload) {
  const table = /** @type {Record<string, Record<string, number | null>>} */ (vram.resident);
  const byMode = table[model];
  if (!byMode) throw new RangeError(`нет резидентных весов для модели ${JSON.stringify(model)}`);
  const value = byMode[offload ?? 'group'];
  if (value === null || value === undefined) {
    throw new RangeError(`модель ${model} не умеет offload=${JSON.stringify(offload)}`);
  }
  return value;
}

/**
 * Оценка пика VRAM, ГиБ. Метрика — **полное `memory.used` карты** (`nvidia-smi`), то есть вместе
 * с рабочим столом: так отвечает на вопрос «влезет ли в N ГБ». Окно внимания в эту оценку
 * **не входит** (измерено: 1792p — 10.3 ГиБ и с окном, и без; AGENTS §4).
 *
 * Активации — насыщающаяся кривая `a1·x/(1 + a2·x)` (`x = tokens/1000`): в конверте 4 000…14 400
 * токенов пик растёт почти линейно, а на 1792p/4K — медленнее (измерено: 12.85 → 12.35 → 15.85 ГиБ).
 * Кривая монотонна и ограничена сверху (`a1/a2`), поэтому за конвертом не ломается.
 *
 * @param {EstimateConfig} config
 * @param {typeof import('../src/data/model.mjs').MODEL['coefficients']} [coefficients]
 * @returns {number}
 */
export function predictVram(config, coefficients) {
  const c = coefficients ?? ALL_COEFFICIENTS;
  const vram = c.vram;
  const tk = config.tokens / 1000;
  const kv = config.kvCache === 'true' ? KV_GIB_PER_TOKEN * prefixTokens(config) : 0;
  return residentFor(vram, config.model, config.offload)
    + ACTIVATION_BASE_GIB
    + activationGib(vram, tk)
    + kv
    + vram.overheadGib;
}

/**
 * Активации канваса, ГиБ: `a1·x/(1 + a2·x)`.
 *
 * @param {typeof import('../src/data/model.mjs').MODEL['coefficients']['vram']} vram
 * @param {number} tk токены канваса в тысячах
 * @returns {number}
 */
export function activationGib(vram, tk) {
  return (vram.activationGibPerMtoken * tk) / (1 + vram.activationSaturationPerMtoken * tk);
}

/**
 * Время одного шага сэмплера, с (`genS / steps`).
 *
 * @param {EstimateConfig} config
 * @param {typeof import('../src/data/model.mjs').MODEL['coefficients']} [coefficients]
 * @returns {number}
 */
export function predictStepS(config, coefficients) {
  const t = (coefficients ?? ALL_COEFFICIENTS).time;
  const lnTokens = Math.log(config.tokens / 4000);
  const group = config.offload === 'group' ? 1 : 0;
  const kvOffload = config.kvCache === 'offload' ? 1 : 0;
  const h3 = config.model === 'h3-image' ? 1 : 0;
  const turbo = config.model === 'qwen21-turbo' ? 1 : 0;
  const window = config.windowAttnPx > 0 ? 1 : 0;
  const stepsNorm = Math.log(config.steps / fixedSteps(config.model));
  const lnStep = t.b0
    + t.bTokens * lnTokens
    + t.bTokens2 * lnTokens * lnTokens
    + t.bGroup * group
    + t.bKvOffload * kvOffload
    + t.bH3 * h3
    + t.bTurbo * turbo
    + t.bSteps * stepsNorm
    + t.bWindow * window
    + t.bH3Tokens * h3 * lnTokens
    + t.bGroupTokens * group * lnTokens;
  return Math.exp(lnStep);
}

/**
 * Полное время генерации, с: `steps · tStep`.
 *
 * @param {EstimateConfig} config
 * @param {typeof import('../src/data/model.mjs').MODEL['coefficients']} [coefficients]
 * @returns {number}
 */
export function predictGenS(config, coefficients) {
  return config.steps * predictStepS(config, coefficients);
}

/**
 * Загрузка модели (импорт torch/diffusers, fp8-кэш с диска, LoRA, размещение), с.
 *
 * @param {EstimateConfig} config
 * @param {typeof import('../src/data/model.mjs').MODEL['coefficients']} [coefficients]
 * @returns {number}
 */
export function predictLoadS(config, coefficients) {
  const load = (coefficients ?? ALL_COEFFICIENTS).load;
  const table = /** @type {Record<string, number>} */ (load.coldS);
  const value = table[config.model];
  if (value === undefined) throw new RangeError(`нет оценки загрузки для модели ${JSON.stringify(config.model)}`);
  return value;
}

/**
 * Энкод промта (холодный, то есть с промахом кэша эмбеддингов), с.
 * GPU-путь и 32B у H3 — I/O-bound константа, CPU-путь растёт линейно по токенам.
 *
 * @param {EstimateConfig} config
 * @param {typeof import('../src/data/model.mjs').MODEL['coefficients']} [coefficients]
 * @returns {number}
 */
export function predictEncodeS(config, coefficients) {
  const enc = (coefficients ?? ALL_COEFFICIENTS).encode;
  if (config.encoderDevice === 'cpu') return enc.cpuBaseS + enc.cpuPerMtokenS * (config.tokens / 1000);
  if (config.model === 'h3-image') return enc.h3ColdS;
  return enc.coldGpuS;
}

/**
 * Все три фазы вместе: загрузка + холодный энкод + генерация (постобработки в Phase A нет).
 *
 * @param {EstimateConfig} config
 * @param {typeof import('../src/data/model.mjs').MODEL['coefficients']} [coefficients]
 * @returns {number}
 */
export function predictTotalS(config, coefficients) {
  return predictLoadS(config, coefficients)
    + predictEncodeS(config, coefficients)
    + predictGenS(config, coefficients);
}

/**
 * Замер пика, приведённый к метрике модели (`nvidia-smi`): пики процесса/аллокатора torch
 * поднимаются на измеренную поправку `measurementDesktopReserveGib` (рабочий стол).
 * Возвращает `null`, если пика в строке нет.
 *
 * @param {import('../src/data/measurements.mjs').Measurement} row
 * @param {typeof import('../src/data/model.mjs').MODEL['coefficients']} coefficients
 * @returns {number | null}
 */
export function measuredVramTarget(row, coefficients) {
  const peak = row.measured.peakVramGib;
  if (peak === undefined) return null;
  const metric = peakMetricOf(row);
  const reserve = coefficients.vram.measurementDesktopReserveGib;
  return metric === 'nvidia-smi' ? peak : peak + reserve;
}

/**
 * @typedef {{
 *   model: string,
 *   vramGib?: number,
 *   ramGib?: number,
 *   disk?: 'hdd' | 'ssd' | 'nvme',
 *   refs?: number,
 *   aspect?: string | null,
 *   width?: number,
 *   height?: number,
 *   tokens: number,
 *   steps: number,
 *   offload?: 'none' | 'group' | null,
 *   offloadBlocks?: number | null,
 *   kvCache?: 'true' | 'offload' | 'false' | null,
 *   encoderDevice?: 'cuda' | 'cpu' | null,
 *   windowAttnPx?: number,
 *   vaeTilePx?: number | null,
 *   refPx?: number | null
 * }} EstimateConfig
 */

/** @type {any} */
let ALL_COEFFICIENTS = null;

// ───────────────────────────────────────────────────────────────────────────────────────────────
// Фиты.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Границы сетки насыщения `a2` (1/1000 токенов). Сетка — потому что `a2` входит в модель
 * нелинейно, а остальные четыре коэффициента при фиксированном `a2` — обычный МНК.
 */
const SATURATION_MAX = 0.08;
const SATURATION_STEPS = 320;

/**
 * Фит VRAM: насыщающиеся активации `a1·x/(1 + a2·x)` + `[overheadGib, resident.group, resident.h3]`.
 * `resident.none` фиксирован (7.67 ГиБ, `AGENTS.md` §4), база активаций — тоже (0.12 ГиБ).
 *
 * Окна внимания среди признаков **нет** и быть не должно: замер (`AGENTS.md` §4) даёт один и тот же
 * пик с окном и без (1792p — 10.3 ГиБ; разница только во времени 314 против 376 с). Пока окно
 * стояло регрессором, оно забирало часть активаций на себя и плотное внимание на 1792p
 * оценивалось в 15.95 ГиБ, хотя мерилось 10.3.
 *
 * Насыщение (`a2`) нужно потому, что пик растёт не строго линейно по токенам: 12.85 ГиБ при
 * 14 400 токенах (2560×1440), 12.35 при 22 400 (1792p) и 15.85 при 32 640 (4K) — прирост падает.
 * Прямая линия вынуждена была бы выбрать одно из двух (или переоценивать 1792p, или занижать
 * 1440p); насыщающаяся кривая огибает весь конверт.
 *
 * @param {ReadonlyArray<import('../src/data/measurements.mjs').Measurement>} rows
 * @returns {{coefficients: any, n: number}}
 */
export function fitVram(rows) {
  const usable = rows.filter((row) => row.measured.peakVramGib !== undefined
    && peakMetricOf(row) !== null);
  const prepared = usable.map((row) => {
    const config = row.config;
    const converted = peakMetricOf(row) === 'nvidia-smi' ? 0 : DESKTOP_RESERVE_GIB;
    const kv = config.kvCache === 'true' ? KV_GIB_PER_TOKEN * prefixTokens(config) : 0;
    const residentNone = config.offload === 'none' ? RESIDENT_NONE_GIB : 0;
    return {
      row,
      tk: config.tokens / 1000,
      target: (row.measured.peakVramGib ?? 0) + converted - kv - residentNone - ACTIVATION_BASE_GIB,
      group: config.model === 'h3-image' ? 0 : (config.offload === 'group' ? 1 : 0),
      h3: config.model === 'h3-image' ? 1 : 0,
    };
  });

  /** @type {{a1: number, a2: number, overhead: number, group: number, h3: number, sse: number}|null} */
  let best = null;
  for (let step = 0; step <= SATURATION_STEPS; step += 1) {
    const a2 = (SATURATION_MAX * step) / SATURATION_STEPS;
    /** @type {number[][]} */
    const a = prepared.map((entry) => [entry.tk / (1 + a2 * entry.tk), 1, entry.group, entry.h3]);
    const y = prepared.map((entry) => entry.target);
    const [a1, overhead, group, h3] = leastSquares(a, y);
    const sse = prepared.reduce((sum, entry, index) => {
      const predicted = a[index][0] * a1 + overhead + group * entry.group + h3 * entry.h3;
      const residual = predicted - entry.target;
      return sum + residual * residual;
    }, 0);
    if (best === null || sse < best.sse) {
      best = { a1, a2, overhead, group, h3, sse };
    }
  }
  if (best === null) throw new RangeError('fitVram: нет строк с пиком VRAM');

  const coefficients = {
    activationGibPerMtoken: best.a1,
    activationSaturationPerMtoken: best.a2,
    overheadGib: best.overhead,
    resident: {
      'qwen21-base': { none: RESIDENT_NONE_GIB, group: best.group },
      'qwen21-turbo': { none: RESIDENT_NONE_GIB, group: best.group },
      'h3-image': { none: null, group: best.h3 },
    },
  };
  return { coefficients, n: prepared.length };
}

/**
 * Ускорение от скользящего окна внимания — **измерено**, а не подобрано:
 * 1792p — 314 с с окном против 376 с плотного при одинаковых токенах (314/376 = 0.835, −17 %).
 * Поэтому `bWindow` — такая же измеренная константа, как `textTokens` или `kvGibPerToken`:
 * в МНК она не входит (иначе компенсирует шум эластичности по токенам и легко становится
 * положительной — «окно дороже плотного», чего в замерах нет).
 */
const WINDOW_SPEEDUP = 314 / 376;

/**
 * Фит времени: `ln tStep` по признакам плана (ridge 1e-3). Окно — фиксированный измеренный
 * коэффициент (`WINDOW_SPEEDUP`): из `y` он вычитается, а не подбирается.
 *
 * @param {ReadonlyArray<import('../src/data/measurements.mjs').Measurement>} rows
 * @returns {{coefficients: any, n: number}}
 */
export function fitTime(rows) {
  const usable = rows.filter((row) => row.measured.genS !== undefined && row.measured.genS > 0);
  const bWindow = Math.log(WINDOW_SPEEDUP);
  /** @type {number[][]} */
  const a = [];
  /** @type {number[]} */
  const y = [];
  /** @type {any[]} */
  const used = [];
  for (const row of usable) {
    const config = row.config;
    const lnTokens = Math.log(config.tokens / 4000);
    const h3 = config.model === 'h3-image' ? 1 : 0;
    const window = config.windowAttnPx > 0 ? 1 : 0;
    a.push([
      1,
      lnTokens,
      lnTokens * lnTokens,
      config.offload === 'group' ? 1 : 0,
      config.kvCache === 'offload' ? 1 : 0,
      h3,
      config.model === 'qwen21-turbo' ? 1 : 0,
      Math.log(config.steps / fixedSteps(config.model)),
      h3 * lnTokens,
      (config.offload === 'group' ? 1 : 0) * lnTokens,
    ]);
    y.push(Math.log((row.measured.genS ?? 0) / config.steps) - bWindow * window);
    used.push(row);
  }
  const [b0, bTokens, bTokens2, bGroup, bKvOffload, bH3, bTurbo, bSteps, bH3Tokens, bGroupTokens]
    = leastSquares(a, y, 1e-3);
  const coefficients = {
    b0, bTokens, bTokens2, bGroup, bKvOffload, bH3, bTurbo, bSteps, bWindow, bH3Tokens, bGroupTokens,
  };
  return { coefficients, n: used.length };
}

/**
 * Фит загрузки/энкода — константы по фазам (I/O-bound): среднее по строкам с измеренной фазой.
 *
 * @param {ReadonlyArray<import('../src/data/measurements.mjs').Measurement>} rows
 */
export function fitPhases(rows) {
  const loadRows = rows.filter((row) => (row.measured.loadS ?? 0) > 0);
  const loadMean = (model) => mean(loadRows
    .filter((row) => row.config.model === model)
    .map((row) => row.measured.loadS ?? 0));
  const loadColdS = {
    'qwen21-base': loadMean('qwen21-base'),
    'qwen21-turbo': loadMean('qwen21-turbo'),
    'h3-image': loadMean('h3-image'),
  };
  const encodeGpu = rows.filter((row) => row.config.encoderDevice === 'cuda'
    && (row.measured.encodeS ?? 0) > 0 && row.config.model !== 'h3-image');
  const encodeCpu = rows.filter((row) => row.config.encoderDevice === 'cpu'
    && (row.measured.encodeS ?? 0) > 0);
  const encodeH3 = rows.filter((row) => row.config.model === 'h3-image'
    && (row.measured.encodeS ?? 0) > 0);
  const cpuPoints = encodeCpu.map((row) => ({
    x: row.config.tokens / 1000,
    y: row.measured.encodeS ?? 0,
  }));
  const [cpuBaseS, cpuPerMtokenS] = leastSquares(
    cpuPoints.map((point) => [1, point.x]),
    cpuPoints.map((point) => point.y),
  );
  const coefficients = {
    load: { coldS: loadColdS },
    encode: {
      coldGpuS: mean(encodeGpu.map((row) => row.measured.encodeS ?? 0)),
      cpuBaseS,
      cpuPerMtokenS,
      h3ColdS: mean(encodeH3.map((row) => row.measured.encodeS ?? 0)),
      warmS: 0,
    },
  };
  return { coefficients };
}

/**
 * @param {number[]} values
 * @returns {number}
 */
function mean(values) {
  if (values.length === 0) throw new RangeError('mean: пустая выборка');
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

/**
 * Сводка ошибок: `n`, медиана и максимум `|err|`, список точек (для отчёта).
 *
 * @param {Array<{id: string, est: number, measured: number, errPct: number}>} perRow
 */
function summarise(perRow) {
  const abs = perRow.map((row) => Math.abs(row.errPct)).sort((left, right) => left - right);
  if (abs.length === 0) return { n: 0, medianAbsErrPct: 0, maxAbsErrPct: 0 };
  const middle = Math.floor(abs.length / 2);
  const median = abs.length % 2 === 0 ? (abs[middle - 1] + abs[middle]) / 2 : abs[middle];
  return {
    n: abs.length,
    medianAbsErrPct: round(median, 2),
    maxAbsErrPct: round(abs[abs.length - 1], 2),
  };
}

/**
 * @param {number} value
 * @param {number} digits
 * @returns {number}
 */
function round(value, digits) {
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
// Сборка MODEL.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Полная калибровка: коэффициенты, конверт, статистика, провенанс.
 *
 * @param {ReadonlyArray<import('../src/data/measurements.mjs').Measurement>} measurements
 */
export function calibrate(measurements = MEASUREMENTS) {
  const trainable = fitRows(measurements);
  // Фазы загрузки/энкода — константы режима, а не времени шага: берём все рендер-строки без «≈»
  // (включая `fitExcluded`: аномалия в генерации не делает аномальным энкод того же прогона).
  const phaseRows = renderRows(measurements).filter((row) => !row.estimated);
  const vram = fitVram(trainable);
  const time = fitTime(trainable);
  const phases = fitPhases(phaseRows);

  const coefficients = {
    /** Сколько токенов префикса добавляет текст (измерено по четырём строкам `[kv]`). */
    textTokens: TEXT_TOKENS,
    /** 512 КиБ на префиксный токен: 32 слоя × 2 × 4096 × 2 Б (units.mjs). */
    kvGibPerToken: KV_GIB_PER_TOKEN,
    vram: {
      activationBaseGib: ACTIVATION_BASE_GIB,
      activationGibPerMtoken: round(vram.coefficients.activationGibPerMtoken, 6),
      activationSaturationPerMtoken: round(vram.coefficients.activationSaturationPerMtoken, 6),
      overheadGib: round(vram.coefficients.overheadGib, 4),
      measurementDesktopReserveGib: DESKTOP_RESERVE_GIB,
      resident: {
        'qwen21-base': {
          none: round(vram.coefficients.resident['qwen21-base'].none, 4),
          group: round(vram.coefficients.resident['qwen21-base'].group, 4),
        },
        'qwen21-turbo': {
          none: round(vram.coefficients.resident['qwen21-turbo'].none, 4),
          group: round(vram.coefficients.resident['qwen21-turbo'].group, 4),
        },
        'h3-image': { none: null, group: round(vram.coefficients.resident['h3-image'].group, 4) },
      },
    },
    time: {
      b0: round(time.coefficients.b0, 6),
      bTokens: round(time.coefficients.bTokens, 6),
      bTokens2: round(time.coefficients.bTokens2, 6),
      bGroup: round(time.coefficients.bGroup, 6),
      bKvOffload: round(time.coefficients.bKvOffload, 6),
      bH3: round(time.coefficients.bH3, 6),
      bTurbo: round(time.coefficients.bTurbo, 6),
      bSteps: round(time.coefficients.bSteps, 6),
      bWindow: round(time.coefficients.bWindow, 6),
      bH3Tokens: round(time.coefficients.bH3Tokens, 6),
      bGroupTokens: round(time.coefficients.bGroupTokens, 6),
    },
    load: {
      coldS: {
        'qwen21-base': round(phases.coefficients.load.coldS['qwen21-base'], 4),
        'qwen21-turbo': round(phases.coefficients.load.coldS['qwen21-turbo'], 4),
        'h3-image': round(phases.coefficients.load.coldS['h3-image'], 4),
      },
    },
    encode: {
      coldGpuS: round(phases.coefficients.encode.coldGpuS, 4),
      cpuBaseS: round(phases.coefficients.encode.cpuBaseS, 4),
      cpuPerMtokenS: round(phases.coefficients.encode.cpuPerMtokenS, 4),
      h3ColdS: round(phases.coefficients.encode.h3ColdS, 4),
      warmS: phases.coefficients.encode.warmS,
    },
    disk: {
      bandwidthGibPerS: { ...DISK_BANDWIDTH },
    },
  };

  ALL_COEFFICIENTS = coefficients;

  const render = renderRows(measurements);
  const audited = auditedRows(measurements);
  const withPeak = audited.filter((row) => row.measured.peakVramGib !== undefined);
  const withGen = audited.filter((row) => row.measured.genS !== undefined && row.measured.genS > 0);
  const withTotal = audited.filter((row) => row.measured.totalS !== undefined);

  /** @param {any} row */
  const predict = (row) => ({
    vram: predictVram(row.config, coefficients),
    gen: predictGenS(row.config, coefficients),
    load: predictLoadS(row.config, coefficients),
    encode: (row.measured.encodeS ?? 0) > 0
      ? predictEncodeS(row.config, coefficients)
      : 0,
  });

  const vramRows = withPeak.map((row) => {
    const target = /** @type {number} */ (measuredVramTarget(row, coefficients));
    return {
      id: row.id,
      est: round(predict(row).vram, 3),
      measured: row.measured.peakVramGib,
      target: round(target, 3),
      metric: peakMetricOf(row),
      errPct: round((predict(row).vram / target - 1) * 100, 2),
    };
  });
  const timeRows = withGen.map((row) => ({
    id: row.id,
    est: round(predict(row).gen, 2),
    measured: row.measured.genS,
    errPct: round((predict(row).gen / (row.measured.genS ?? 1) - 1) * 100, 2),
  }));
  const totalRows = withTotal.map((row) => {
    const est = predict(row).load + predict(row).encode + predict(row).gen;
    return {
      id: row.id,
      est: round(est, 2),
      measured: row.measured.totalS,
      errPct: round((est / (row.measured.totalS ?? 1) - 1) * 100, 2),
    };
  });

  const envelope = {
    measurementCount: measurements.length,
    renderCount: render.length,
    fitCount: trainable.length,
    auditedCount: audited.length,
    tokens: { min: min(render.map((row) => row.config.tokens)), max: max(render.map((row) => row.config.tokens)) },
    steps: { min: min(render.map((row) => row.config.steps)), max: max(render.map((row) => row.config.steps)) },
    refs: { min: min(render.map((row) => row.config.refs)), max: max(render.map((row) => row.config.refs)) },
    vramGib: { min: min(render.map((row) => row.config.vramGib)), max: max(render.map((row) => row.config.vramGib)) },
    ramGib: { min: min(render.map((row) => row.config.ramGib)), max: max(render.map((row) => row.config.ramGib)) },
    models: /** @type {string[]} */ ([...new Set(render.map((row) => row.config.model))].sort()),
    windowAttnPx: {
      min: min(render.map((row) => row.config.windowAttnPx)),
      max: max(render.map((row) => row.config.windowAttnPx)),
    },
    vaeTilePx: /** @type {number[]} */ ([...new Set(render.map((row) => row.config.vaeTilePx).filter((value) => value !== null))].sort((left, right) => left - right)),
  };

  /** @type {Record<string, any>} */
  const provenance = {
    'textTokens': {
      value: TEXT_TOKENS,
      unit: 'tokens',
      kind: 'measured',
      source: 'docs/bench/vram-tiers.md#строки [kv]: 4360−4128, 6592−6360, 8392−8160, 14632−14400',
      note: 'во всех четырёх строках префикс ровно на 232 токена больше канваса. У строк '
        + 'kv-cache-offload.md промт длиннее (1230 токенов текста): модель берёт 232 как '
        + 'минимум, поэтому на длинном промте KV-терм занижен на ~0.5 ГиБ (это запас в fitted-константах; '
        + 'расхождение с измеренными prefix-строками до ±25 % — сознательное, не подгоняется)',
    },
    'kvGibPerToken': {
      value: KV_GIB_PER_TOKEN,
      unit: 'GiB/token',
      kind: 'measured',
      source: 'docs/bench/kv-cache-offload.md#«Строки [kv]»: 5358 токена → 2.62 ГиБ, 9390 → 4.58, 13402 → 6.54',
      note: '512 КиБ на слой×токен: 32 слоя × 2 (K и V) × 4096 × 2 Б; сходится на трёх строках',
    },
    'vram.activationBaseGib': {
      value: ACTIVATION_BASE_GIB,
      unit: 'GiB',
      kind: 'measured',
      source: 'AGENTS.md#§4 «Из чего состоит пик VRAM»: активации 1.59 ГиБ при 1728 токенах, 3.63 при 4128',
      note: 'две точки: линейная часть 0.85 ГиБ/1000 токенов, остаток на нуле — 0.12 ГиБ; наклон уточнён фитом',
    },
    'vram.activationGibPerMtoken': {
      value: coefficients.vram.activationGibPerMtoken,
      unit: 'GiB/1000 tokens',
      kind: 'fitted',
      source: 'renderkit/tools/fit.mjs#fitVram: МНК по всем строкам с пиком (vram-tiers + kv-cache-offload + AGENTS §4), без регрессора окна',
      note: 'начальный наклон насыщающейся кривой `a1·x/(1+a2·x)`: на малых канвасах активации '
        + 'растут почти линейно. От шагов не зависит (активации не накапливаются), окно внимания '
        + 'тоже не входит: 1792p — 10.3 ГиБ и с ним, и без',
    },
    'vram.activationSaturationPerMtoken': {
      value: coefficients.vram.activationSaturationPerMtoken,
      unit: '1/(1000 tokens)',
      kind: 'fitted',
      source: 'AGENTS.md#§4 «Скользящее окно внимания»: пик 12.85 ГиБ (1440p, vram-tiers) → 12.35 '
        + '(1792p) → 15.85 (4K) — прирост падает',
      note: 'насыщение: на больших канвасах пик растёт медленнее токенов. Прямая линия не могла '
        + 'одновременно уложить 1440p и 1792p/4K, а роль «изгиба» раньше играло окно внимания '
        + '(регрессор уходил в минус и обнулял плотное внимание), что противоречит замеру '
        + '«VRAM окно не экономит». Предел роста активаций — a1/a2 ГиБ',
    },
    'vram.overheadGib': {
      value: coefficients.vram.overheadGib,
      unit: 'GiB',
      kind: 'fitted',
      source: 'renderkit/tools/fit.mjs#fitVram; метрика vram-tiers.md «полное memory.used карты»',
      note: 'контекст CUDA + резерв аллокатора + рабочий стол ≈2 ГиБ; поэтому метрика оценки — nvidia-smi, а не свой процесс',
    },
    'vram.measurementDesktopReserveGib': {
      value: DESKTOP_RESERVE_GIB,
      unit: 'GiB',
      kind: 'measured',
      source: 'docs/bench/vram-tiers.md#колонки «пик VRAM, ГиБ» и «свой процесс, ГиБ»: 7.17−5.11, 8.26−6.21, 11.98−9.93, 12.85−10.80, 7.68−5.36',
      note: 'поправка метрики: пик процесса/аллокатора torch → полное memory.used карты (рабочий стол 1797–2371 МиБ); в предсказание не входит, нужна аудиту',
    },
    'vram.resident.qwen21-base.none': {
      value: RESIDENT_NONE_GIB,
      unit: 'GiB',
      kind: 'measured',
      source: 'AGENTS.md#§4 «Из чего состоит пик VRAM»: веса после загрузки (allocated) 7.67 GiB',
      note: 'fp8-трансформер 6.6 ГиБ + LoRA/буферы; при offload: group в VRAM остаётся одна группа блоков',
    },
    'vram.resident.qwen21-base.group': {
      value: coefficients.vram.resident['qwen21-base'].group,
      unit: 'GiB',
      kind: 'fitted',
      source: 'renderkit/tools/fit.mjs#fitVram по строкам с offload: group (vram-tiers + kv-cache-offload)',
      note: 'остаток от 7.67: в VRAM живёт одна группа блоков (~0.2 ГиБ) и то, что не выгружается',
    },
    'vram.resident.qwen21-turbo.none': {
      value: RESIDENT_NONE_GIB,
      unit: 'GiB',
      kind: 'measured',
      source: 'AGENTS.md#§4 «Профили --model (замеры)»: пики 15.7 (turbo) против 15.5 ГиБ (base) при 4128 токенах',
      note: 'разница fp8-весов 6.16 против 6.6 ГиБ в пределах шума пика — принято одно значение',
    },
    'vram.resident.qwen21-turbo.group': {
      value: coefficients.vram.resident['qwen21-turbo'].group,
      unit: 'GiB',
      kind: 'fitted',
      source: 'renderkit/tools/fit.mjs#fitVram (общий коэффициент с base)',
      note: 'turbo в групповом режиме не мерился отдельно; вес турбо-трансформера меньше на 0.44 ГиБ',
    },
    'vram.resident.h3-image.group': {
      value: coefficients.vram.resident['h3-image'].group,
      unit: 'GiB',
      kind: 'fitted',
      source: 'renderkit/tools/fit.mjs#fitVram по строкам h3-512-t2i / h3-512-edit / h3-1024-edit',
      note: 'H3 всегда с offload: group (30.9 ГиБ fp8 в RAM); плюс свой VAE 2.2 ГиБ — входит в остаток',
    },
    'time.b0': {
      value: coefficients.time.b0,
      unit: 'ln s',
      kind: 'fitted',
      source: 'renderkit/tools/fit.mjs#fitTime: ridge 1e-3 по строкам с измеренным genS',
      note: 'базовая скорость шага при 4000 токенах, без offload/KV-подкачки',
    },
    'time.bTokens': {
      value: coefficients.time.bTokens,
      unit: '1',
      kind: 'fitted',
      source: 'renderkit/tools/fit.mjs#fitTime',
      note: 'линейная часть эластичности времени шага по токенам канваса (вместе с bTokens2)',
    },
    'time.bTokens2': {
      value: coefficients.time.bTokens2,
      unit: '1',
      kind: 'fitted',
      source: 'AGENTS.md#§4: 14400 ток. → 205 с (vram-tiers), 22400 → 376 с, 32640 → 513 с '
        + '(15–20 с/шаг на 1792p/4K против 8.2 на 1440p)',
      note: 'квадратичный член по ln(tokens): эластичность падает с ростом канваса (≈2 на малых '
        + 'канвасах и ≈1.2 на больших). Без него окно приходилось держать свободным коэффициентом, '
        + 'и он уходил в плюс, «компенсируя» именно этот изгиб',
    },
    'time.bGroup': {
      value: coefficients.time.bGroup,
      unit: 'ln (×)',
      kind: 'fitted',
      source: 'docs/bench/presets.md#Замечания: 768p 3.9 ГиБ/10.3 с против 10.3 ГиБ/8.2 с (group против none)',
      note: 'лог-надбавка за стриминг блоков с диска/из RAM на каждом шаге',
    },
    'time.bKvOffload': {
      value: coefficients.time.bKvOffload,
      unit: 'ln (×)',
      kind: 'fitted',
      source: 'docs/bench/kv-cache-offload.md#«Что это значит»: +12–17 % времени при KV в RAM (C1 93 с → C2 105 с)',
      note: 'подкачка K/V по слоям каждый шаг; картинка при этом бит-в-бит',
    },
    'time.bH3': {
      value: coefficients.time.bH3,
      unit: 'ln (×)',
      kind: 'fitted',
      source: 'renderkit/tools/fit.mjs#fitTime по строкам h3-image (6 шагов PDD, 33B)',
      note: 'H3 считает 30.9 ГиБ fp8 по группам, поэтому шаг дороже qwen21 при том же канвасе',
    },
    'time.bTurbo': {
      value: coefficients.time.bTurbo,
      unit: 'ln (×)',
      kind: 'fitted',
      source: 'AGENTS.md#§4 «Профили --model»: turbo 1024 — 21 с против base 38 с; 640 — 5 против 6 с',
      note: 'turbo — distill-расписание из 8 шагов, считается той же формулой (нормировка по fixedSteps)',
    },
    'time.bSteps': {
      value: coefficients.time.bSteps,
      unit: '1',
      kind: 'fitted',
      source: 'docs/bench/vram-tiers.md#768p: 8 шагов — 17 с (2.1 с/шаг), 25 — 97 с (3.9 с/шаг)',
      note: 'первый шаг дороже остальных (prefill), поэтому время шага падает с ростом steps',
    },
    'time.bWindow': {
      value: coefficients.time.bWindow,
      unit: 'ln (×)',
      kind: 'measured',
      source: 'AGENTS.md#§4 «Скользящее окно внимания»: 1792p — 314 с окном против 376 с плотного '
        + '(одинаковые токены, ref_px 960) — ln(314/376)',
      note: 'окно покупает время (≈ −17 %) ценой резкости (−2.6 %), VRAM не меняет. Коэффициент '
        + '**измерен**, а не подобран: в МНК не входит (при свободном подборе уходил в плюс, '
        + 'компенсируя шум эластичности по токенам, — а окно быстрее плотного в любом замере)',
    },
    'time.bGroupTokens': {
      value: coefficients.time.bGroupTokens,
      unit: '1',
      kind: 'fitted',
      source: 'renderkit/tools/fit.mjs#fitTime: взаимодействие [group]·ln(tokens/4000)',
      note: 'в group шаг упирается в стриминг блоков fp8 из RAM, поэтому эластичность по токенам ниже плотной',
    },
    'time.bH3Tokens': {
      value: coefficients.time.bH3Tokens,
      unit: '1',
      kind: 'fitted',
      source: 'renderkit/tools/fit.mjs#fitTime: взаимодействие [h3]·ln(tokens/4000)',
      note: 'у H3 шаг упирается в стриминг весов, а не в токены: эластичность ≈0.5 против ≈1.2 у qwen21',
    },
    'load.coldS.qwen21-base': {
      value: coefficients.load.coldS['qwen21-base'],
      unit: 's',
      kind: 'fitted',
      source: 'renderkit/tools/fit.mjs#fitPhases: среднее по измеренным «загрузка модели» (docs/bench/presets.md)',
      note: 'импорт torch/diffusers + чтение 6.7 ГиБ fp8-кэша + LoRA + размещение на устройстве',
    },
    'load.coldS.qwen21-turbo': {
      value: coefficients.load.coldS['qwen21-turbo'],
      unit: 's',
      kind: 'fitted',
      source: 'renderkit/tools/fit.mjs#fitPhases (общая оценка с base: замеры 16/17 с против 13–18 с)',
      note: 'fp8-кэш turbo 6.16 ГиБ — на 0.44 ГиБ меньше базового',
    },
    'load.coldS.h3-image': {
      value: 6.4,
      unit: 's',
      kind: 'measured',
      source: 'AGENTS.md#§4 «MiniMax-H3-Image»: prepare (fp8-кэш + компоненты) 6.4 с cold / ~2 с warm',
      note: 'fp8-кэш 30.93 ГиБ читается шардами, поэтому загрузка не пропорциональна размеру',
    },
    'encode.coldGpuS': {
      value: coefficients.encode.coldGpuS,
      unit: 's',
      kind: 'fitted',
      source: 'renderkit/tools/fit.mjs#fitPhases по строкам с энкодом на GPU (15–16 с, от длины промта не зависит)',
      note: 'GPU-энкод I/O-bound: 10.5 с на 4128 токенах, но с обвязкой прогона — 15–16 с',
    },
    'encode.cpuBaseS': {
      value: coefficients.encode.cpuBaseS,
      unit: 's',
      kind: 'fitted',
      source: 'renderkit/tools/fit.mjs#fitPhases по двум CPU-строкам (90 с при 900 токенах, 126 с при 1600)',
      note: 'линейная подгонка по двум точкам: на 4128 токенах даёт ~254 с (AGENTS §4: 255 с)',
    },
    'encode.cpuPerMtokenS': {
      value: coefficients.encode.cpuPerMtokenS,
      unit: 's/1000 tokens',
      kind: 'fitted',
      source: 'renderkit/tools/fit.mjs#fitPhases по двум CPU-строкам (90 с при 900 токенах, 126 с при 1600)',
      note: 'CPU-энкод Qwen3VL-8B растёт с числом токенов, в отличие от GPU-пути',
    },
    'encode.h3ColdS': {
      value: coefficients.encode.h3ColdS,
      unit: 's',
      kind: 'fitted',
      source: 'AGENTS.md#§4 «MiniMax-H3-Image»: 118.9 с (T2I 218 токенов), 82.8 с (edit 512²), 127.5 с (edit 1088×1024)',
      note: 'три замера одной константой: 82.8 с — синхронный стриминг edit-пути, 118.9/127.5 — конвейер T2I',
    },
    'encode.warmS': {
      value: 0,
      unit: 's',
      kind: 'measured',
      source: 'docs/bench/presets.md#Таблица: «0 (cache)» у семи строк, docs/bench/vram-tiers.md — «0 (из кэша)»',
      note: 'попадание в кэш эмбеддингов не платит энкод вовсе',
    },
    'disk.bandwidthGibPerS.nvme': {
      value: DISK_BANDWIDTH.nvme,
      unit: 'GiB/s',
      kind: 'assumption',
      source: 'AGENTS.md#§4 «GPU-стриминг энкодера», плечи I/O: NVMe 4.39 GB/s (проход 1) и 6.81 (тёплый)',
      note: 'взято 3.0 ГиБ/с — нижняя граница измеренного: page cache не сбрасывался, «честный холод» не мерился',
    },
    'disk.bandwidthGibPerS.ssd': {
      value: DISK_BANDWIDTH.ssd,
      unit: 'GiB/s',
      kind: 'assumption',
      source: 'AGENTS.md#§4 «GPU-стриминг энкодера»: оценка порядка для SATA/NVMe-классов',
      note: 'на нашей машине SSD не мерился: 0.45 ГиБ/с — типичный SATA SSD, замеров нет (явное допущение)',
    },
    'disk.bandwidthGibPerS.hdd': {
      value: DISK_BANDWIDTH.hdd,
      unit: 'GiB/s',
      kind: 'assumption',
      source: 'AGENTS.md#§4 «GPU-стриминг энкодера»: оценка порядка для HDD',
      note: 'HDD не мерился: 0.13 ГиБ/с — типичный 7200 rpm последовательно; на HDD fp8-кэш 6.7 ГиБ читается ~50 с',
    },
  };

  return {
    schemaVersion: 1,
    generatedBy: 'tools/fit.mjs',
    coefficients,
    envelope,
    stats: {
      // Сводка по всем строкам аудита (VRAM + генерация + итог) — одна цифра «точность модели».
      ...summarise([...vramRows, ...timeRows, ...totalRows]),
      vram: { ...summarise(vramRows), rows: vramRows },
      time: { ...summarise(timeRows), rows: timeRows },
      total: { ...summarise(totalRows), rows: totalRows },
      fit: { vram: vram.n, time: time.n },
      // Противоречивые замеры: в фит и в агрегат не идут (полный note — в measurements.mjs).
      excluded: render.filter((row) => row.suspect)
        .map((row) => ({ id: row.id, reason: (row.note ?? '').split('. ')[0] })),
    },
    provenance,
  };
}

/**
 * @param {number[]} values
 * @returns {number}
 */
function min(values) {
  return values.reduce((best, value) => (value < best ? value : best), values[0]);
}

/**
 * @param {number[]} values
 * @returns {number}
 */
function max(values) {
  return values.reduce((best, value) => (value > best ? value : best), values[0]);
}

/**
 * Сериализация `MODEL` в стабильный текст файла (порядок ключей — как в `calibrate`).
 *
 * @param {ReturnType<typeof calibrate>} model
 * @returns {string}
 */
export function renderModelFile(model) {
  const header = [
    '// @ts-check',
    '// generated by tools/fit.mjs — do not edit; regenerate with `npm run fit`',
    `// source: src/data/measurements.mjs (${model.envelope.measurementCount} замеров, `
      + `в фите ${model.envelope.fitCount} рендер-строк)`,
    '// метрика VRAM — полное memory.used карты (nvidia-smi), см. шапку tools/fit.mjs',
    '',
    '/**',
    ' * Коэффициенты, конверт и провенанс модели оценок.',
    ' *',
    ' * `coefficients` — единственные числа, на которых считает cost/quality; каждое описано в',
    ' * `provenance` (`kind: measured | fitted | assumption`, `source` — файл+строка, `note` — почему).',
    ' * `envelope` — диапазон замеров: за его пределами оценка помечается `extrapolated`.',
    ' * `stats` — общая сводка (`n`, медиана/максимум `|err|` по всем строкам аудита) и та же',
    ' * сводка по метрикам (`vram`/`time`/`total`) с построчными значениями; допуски и их',
    ' * расширения описаны в `src/data/measurements.mjs` (`tolerance` + `note`).',
    ' */',
    '',
    'export const MODEL = Object.freeze(',
  ];
  const body = JSON.stringify(model, null, 2)
    .split('\n')
    .map((line) => `  ${line}`)
    .join('\n');
  return `${header.join('\n')}\n${body}\n);\n`;
}

// ───────────────────────────────────────────────────────────────────────────────────────────────
// CLI.
// ───────────────────────────────────────────────────────────────────────────────────────────────

function main() {
  const args = process.argv.slice(2);
  const check = args.includes('--check');
  const model = calibrate();
  const text = renderModelFile(model);
  if (check) {
    let current = '';
    try {
      current = readFileSync(MODEL_PATH, 'utf8');
    } catch {
      current = '';
    }
    if (current !== text) {
      const expected = current.split('\n');
      const actual = text.split('\n');
      const limit = Math.max(expected.length, actual.length);
      /** @type {string[]} */
      const diff = [];
      for (let index = 0; index < limit; index += 1) {
        if (expected[index] !== actual[index]) {
          diff.push(`строка ${index + 1}:`);
          diff.push(`  файл:     ${expected[index] ?? '<нет>'}`);
          diff.push(`  ожидается: ${actual[index] ?? '<нет>'}`);
          if (diff.length > 30) break;
        }
      }
      console.error(`[fit] src/data/model.mjs устарел — запустите \`npm run fit\`\n${diff.join('\n')}`);
      process.exit(1);
    }
    console.log(`[fit] src/data/model.mjs совпадает (vram n=${model.stats.fit.vram}, `
      + `time n=${model.stats.fit.time}, медиана |err| ${model.stats.vram.medianAbsErrPct}/${model.stats.time.medianAbsErrPct} %)`);
    return;
  }
  writeFileSync(MODEL_PATH, text, 'utf8');
  console.log(`[fit] записан src/data/model.mjs: ${model.envelope.measurementCount} замеров, `
    + `в фите vram ${model.stats.fit.vram} / time ${model.stats.fit.time} строк`);
  console.log(`[fit] vram: медиана ${model.stats.vram.medianAbsErrPct} %, максимум ${model.stats.vram.maxAbsErrPct} %`);
  console.log(`[fit] время: медиана ${model.stats.time.medianAbsErrPct} %, максимум ${model.stats.time.maxAbsErrPct} %`);
  console.log(`[fit] итог: медиана ${model.stats.total.medianAbsErrPct} %, максимум ${model.stats.total.maxAbsErrPct} %`);
}

const invokedDirectly = process.argv[1] !== undefined
  && import.meta.url === new URL(`file:///${process.argv[1].replaceAll('\\', '/')}`).href;

if (invokedDirectly) main();
