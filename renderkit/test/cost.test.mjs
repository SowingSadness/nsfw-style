// @ts-check
/**
 * Оценки против замеров: 6 контрольных точек плана (16 ГиБ/1440p 12.85/205/231, 16 ГиБ/768p
 * 11.98/15/30, 12 ГиБ/768p 7.17/17/42, 16 ГиБ/1792p 10.3/314, 16 ГиБ/4K 13.8/513,
 * 16 ГиБ/1088p 7.68/96) и сводный аудит `tools/audit-measurements.mjs`.
 *
 * Допуски — те же, что в аудите: максимум `|err| ≤ 30 %` на строку, медиана `≤ 15 %`.
 */

import { execFileSync } from 'node:child_process';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import test from 'node:test';

import { MODEL } from '../src/data/model.mjs';
import {
  predictEncodeS, predictGenS, predictLoadS, predictTotalS, predictVram,
} from '../tools/fit.mjs';
import {
  estimateAll, estimateTime, estimateVram, gateBudget, prefixTokens,
} from '../src/model/cost.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = join(HERE, '..');

/** Допуск на строку и на медиану — как в `tools/audit-measurements.mjs`. */
const MAX_ERR_PCT = 30;
const MEDIAN_ERR_PCT = 15;

/** Конфиг базовой модели с одним референсом в сетке кадра (`ref_px: 0`). */
function baseConfig({
  model = 'qwen21-base', vramGib = 16, ramGib = 64, width, height, steps, offload = 'group',
  kvCache = 'offload', windowAttnPx = 0, refs = 1, refPx = 0, encoderDevice = 'cuda',
}) {
  return {
    model,
    vramGib,
    ramGib,
    disk: /** @type {const} */ ('nvme'),
    refs,
    aspect: '16:9',
    width,
    height,
    tokens: (width / 16) * (height / 16),
    steps,
    offload,
    offloadBlocks: offload === 'group' ? 1 : null,
    kvCache,
    encoderDevice,
    windowAttnPx,
    vaeTilePx: 512,
    refPx,
  };
}

/**
 * Контрольные точки плана: `expect` — измеренные числа (ГиБ и секунды). У 1792p и 4K пик в документе —
 * пик процесса (без рабочего стола): оценка идёт в метрике `nvidia-smi`, поэтому сравниваем
 * с замером, приведённым той же поправкой (`MODEL.coefficients.vram.measurementDesktopReserveGib`).
 */
const DESKTOP_RESERVE = MODEL.coefficients.vram.measurementDesktopReserveGib;
const CONTROL_POINTS = [
  {
    id: '16 ГБ / 1440p',
    config: baseConfig({ width: 2560, height: 1440, steps: 25 }),
    encodeCold: true,
    expect: { vramGib: 12.85, genS: 205, totalS: 231 },
  },
  {
    id: '16 ГБ / 768p',
    config: baseConfig({ width: 1376, height: 768, steps: 8, offload: 'none' }),
    encodeCold: false,
    expect: { vramGib: 11.98, genS: 15, totalS: 30 },
  },
  {
    id: '12 ГБ / 768p',
    config: baseConfig({ vramGib: 12, ramGib: 32, width: 1376, height: 768, steps: 8 }),
    encodeCold: true,
    expect: { vramGib: 7.17, genS: 17, totalS: 42 },
  },
  {
    id: '16 ГБ / 1792p (окно)',
    config: baseConfig({
      width: 3200, height: 1792, steps: 25, windowAttnPx: 384, refs: 2, refPx: 960,
    }),
    expect: { vramGib: 10.3 + DESKTOP_RESERVE, genS: 314 },
  },
  {
    id: '16 ГБ / 1792p (плотное)',
    config: baseConfig({
      width: 3200, height: 1792, steps: 25, windowAttnPx: 0, refs: 2, refPx: 960,
    }),
    expect: { vramGib: 10.3 + DESKTOP_RESERVE, genS: 376 },
  },
  {
    id: '16 ГБ / 4K (окно)',
    config: baseConfig({
      width: 3840, height: 2176, steps: 25, windowAttnPx: 1088, refs: 3, refPx: 768,
    }),
    expect: { vramGib: 13.8 + DESKTOP_RESERVE, genS: 513 },
  },
  {
    id: '16 ГБ / 1088p',
    config: baseConfig({ width: 1920, height: 1088, steps: 25 }),
    encodeCold: true,
    expect: { vramGib: 7.68, genS: 96, totalS: 126 },
  },
];

/** @param {number} est @param {number} measured */
const errPct = (est, measured) => ((est / measured) - 1) * 100;

test('7 контрольных точек плана: VRAM, генерация и итог в допуске 30 %', () => {
  assert.equal(CONTROL_POINTS.length, 7);
  /** @type {string[]} */
  const problems = [];
  for (const point of CONTROL_POINTS) {
    const vramEst = predictVram(point.config, MODEL.coefficients);
    const genEst = predictGenS(point.config, MODEL.coefficients);
    const checks = [['VRAM', vramEst, point.expect.vramGib]];
    if (point.expect.genS !== undefined) checks.push(['генерация', genEst, point.expect.genS]);
    if (point.expect.totalS !== undefined) {
      // В замеренной строке энкод либо платился (cold), либо брался из кэша (0 с) — как в аудите.
      const total = predictLoadS(point.config, MODEL.coefficients)
        + (point.encodeCold === true ? predictEncodeS(point.config, MODEL.coefficients) : 0)
        + genEst;
      checks.push(['итого', total, point.expect.totalS]);
    }
    for (const [metric, est, measured] of checks) {
      const error = errPct(est, measured);
      if (Math.abs(error) > MAX_ERR_PCT) {
        problems.push(`${point.id} ${metric}: оценка ${est.toFixed(2)} против `
          + `${measured} (${error.toFixed(1)} %)`);
      }
    }
  }
  assert.deepEqual(problems, []);
});

test('медиана ошибки по контрольным точкам ≤ 15 %', () => {
  const errors = CONTROL_POINTS.map((point) => Math.abs(errPct(
    predictVram(point.config, MODEL.coefficients),
    point.expect.vramGib,
  ))).sort((left, right) => left - right);
  const middle = Math.floor(errors.length / 2);
  const median = errors.length % 2 === 0
    ? (errors[middle - 1] + errors[middle]) / 2
    : errors[middle];
  assert.ok(median <= MEDIAN_ERR_PCT, `медиана VRAM по контрольным точкам ${median.toFixed(1)} %`);
});

test('статистика модели в файле: медиана ≤ 15 %, максимум ≤ 30 %', () => {
  for (const key of /** @type {const} */ (['vram', 'time'])) {
    const stats = MODEL.stats[key];
    assert.ok(stats.n > 20, `${key}: ожидается ≥20 строк, получено ${stats.n}`);
    assert.ok(stats.medianAbsErrPct <= MEDIAN_ERR_PCT,
      `${key}: медиана ${stats.medianAbsErrPct} % > ${MEDIAN_ERR_PCT} %`);
    assert.ok(stats.maxAbsErrPct <= MAX_ERR_PCT,
      `${key}: максимум ${stats.maxAbsErrPct} % > ${MAX_ERR_PCT} %`);
  }
});

test('аудит замеров проходит: `tools/audit-measurements.mjs` → exit 0', () => {
  const output = execFileSync(process.execPath, [join(PACKAGE_ROOT, 'tools', 'audit-measurements.mjs')], {
    cwd: PACKAGE_ROOT,
    encoding: 'utf8',
  });
  assert.match(output, /=== VRAM, пик/u);
  assert.match(output, /=== время генерации/u);
  assert.match(output, /=== итого \(загрузка \+ энкод \+ генерация\)/u);
  assert.match(output, /\[audit\] все строки в допуске/u);
  assert.doesNotMatch(output, /ВНЕ ДОПУСКА/u);
});

test('пик VRAM от шагов не зависит (следствие A4/решения 5)', () => {
  const draft = baseConfig({ width: 2560, height: 1440, steps: 8 });
  const full = baseConfig({ width: 2560, height: 1440, steps: 25 });
  assert.equal(
    predictVram(draft, MODEL.coefficients).toFixed(6),
    predictVram(full, MODEL.coefficients).toFixed(6),
  );
});

test('окно внимания пик VRAM не меняет — его нет в VRAM-формуле', () => {
  // Замер (AGENTS.md §4): 1792p — 314 с с окном и 376 с плотным при одном и том же пике 10.3 ГиБ.
  // Пока окно стояло регрессором, плотное внимание на 1792p оценивалось в 15.95 ГиБ (ложный impossible).
  const dense = baseConfig({ width: 3200, height: 1792, steps: 25, windowAttnPx: 0, refs: 2, refPx: 960 });
  const window = baseConfig({ width: 3200, height: 1792, steps: 25, windowAttnPx: 384, refs: 2, refPx: 960 });
  const densePeak = predictVram(dense, MODEL.coefficients);
  const windowPeak = predictVram(window, MODEL.coefficients);
  assert.equal(densePeak, windowPeak, 'плотное и оконное сочетание на одном канвасе — один пик');
  // Контроль плана: 10.3 ГиБ процесса + 2.05 рабочего стола = 12.35 ГиБ (±15 %).
  const measured = 10.3 + MODEL.coefficients.vram.measurementDesktopReserveGib;
  assert.ok(Math.abs(densePeak - measured) / measured <= 0.15,
    `плотное 1792p: оценка ${densePeak.toFixed(2)} против ${measured.toFixed(2)}`);
  // Время при этом различается — и окно быстрее (bWindow < 0, ln(314/376)).
  assert.ok(MODEL.coefficients.time.bWindow < 0, `bWindow = ${MODEL.coefficients.time.bWindow}`);
  const denseGen = predictGenS(dense, MODEL.coefficients);
  const windowGen = predictGenS(window, MODEL.coefficients);
  assert.ok(windowGen < denseGen, `окно ${windowGen.toFixed(1)} с должно быть быстрее ${denseGen.toFixed(1)} с`);
  for (const [cfg, measuredS] of [[dense, 376], [window, 314]]) {
    const est = predictGenS(cfg, MODEL.coefficients);
    assert.ok(Math.abs(est - measuredS) / measuredS <= 0.15,
      `${cfg.windowAttnPx > 0 ? 'окно' : 'плотное'}: ${est.toFixed(1)} с против ${measuredS} с`);
  }
});

test('префикс KV в рантайме считается сеткой референса (как memory.ref.tokensPerRef)', () => {
  // `ref_px 960` на 16:9 — сетка 1696×960 = 6360 токенов; прежняя формула `(960/16)²`
  // давала 3600 и занижала KV на треть (2.49 вместо 3.70 ГиБ в плане 1440p).
  const wide = runtimeRender({ width: 2560, height: 1440, refs: 1, refPx: 960, textTokens: 1230 });
  const prefix = prefixTokens(wide);
  assert.equal(prefix, 1230 + 6360);
  assert.ok(Math.abs(prefix * MODEL.coefficients.kvGibPerToken - 3.7) < 0.05, `KV ${prefix * MODEL.coefficients.kvGibPerToken}`);
  // `ref_px: 0` — сетка кадра (14400 токенов на 2560×1440).
  assert.equal(prefixTokens({ ...wide, refPx: 0 }), 1230 + 14400);
  // 768p с 1 рефом в сетке кадра: 232 текста + 4128 канваса — строка [kv] из vram-tiers.
  const narrow = runtimeRender({ width: 1376, height: 768, refs: 1, refPx: 0, textTokens: 232 });
  assert.equal(prefixTokens(narrow), 4360);
});

test('KV в VRAM стоит ровно 0.000488 ГиБ на префиксный токен', () => {
  const offload = baseConfig({ width: 1920, height: 1088, steps: 25 });
  const inVram = { ...offload, kvCache: /** @type {const} */ ('true') };
  const delta = predictVram(inVram, MODEL.coefficients) - predictVram(offload, MODEL.coefficients);
  // префикс модели = 232 текста + 8160 канваса (ref_px: 0)
  const expected = MODEL.coefficients.kvGibPerToken * (MODEL.coefficients.textTokens + 8160);
  assert.ok(Math.abs(delta - expected) < 1e-9, `${delta} против ${expected}`);
  // Константа проверяется на измеренной строке [kv] из kv-cache-offload.md (C1: 9390 → 4.58 ГиБ).
  assert.ok(Math.abs(MODEL.coefficients.kvGibPerToken * 9390 - 4.58) < 0.01);
});

test('Turbo: 8 фиксированных шагов дороже базового шага, но быстрее полного расписания', () => {
  const base = baseConfig({ width: 1376, height: 768, steps: 25, offload: 'none' });
  const turbo = { ...base, model: /** @type {const} */ ('qwen21-turbo'), steps: 8 };
  assert.ok(predictGenS(turbo, MODEL.coefficients) < predictGenS(base, MODEL.coefficients));
});

test('неизвестная модель и невозможный режим offload — исключение, а не NaN', () => {
  const config = baseConfig({ width: 1376, height: 768, steps: 8 });
  assert.throws(() => predictVram({ ...config, model: 'no-such-model' }, MODEL.coefficients), RangeError);
  assert.throws(
    () => predictVram({ ...config, model: 'h3-image', offload: 'none' }, MODEL.coefficients),
    /offload/u,
  );
  assert.ok(Number.isFinite(predictTotalS(config, MODEL.coefficients)));
});

// ───────────────────────────────────────────────────────────────────────────────────────────────
// Шаг 4: рантайм-оценки (`src/model/cost.mjs`) на тех же осях, что и калибровка.
// ───────────────────────────────────────────────────────────────────────────────────────────────

/**
 * `Render` для рантайм-оценок: полный набор полей, которые читают `cost.mjs`.
 *
 * @param {Record<string, any>} [overrides]
 * @returns {import('../src/model/cost.mjs').Render}
 */
function runtimeRender(overrides = {}) {
  const width = overrides.width ?? 1376;
  const height = overrides.height ?? 768;
  return /** @type {any} */ ({
    model: 'qwen21-base',
    canvas: { width, height },
    tokens: (width / 16) * (height / 16),
    steps: 25,
    offload: 'group',
    kvCache: 'offload',
    windowAttnPx: 0,
    marginPx: 0,
    vaeTilePx: 512,
    refs: 1,
    refPx: 0,
    encoderDevice: 'cuda',
    vramGib: 16,
    ramGib: 64,
    disk: 'nvme',
    ...overrides,
  });
}

test('estimateVram не зависит от steps (активации не накапливаются от шага к шагу)', () => {
  const eight = runtimeRender({ width: 2560, height: 1440, steps: 8 });
  const full = runtimeRender({ width: 2560, height: 1440, steps: 25 });
  assert.equal(estimateVram(eight), estimateVram(full));
  assert.equal(estimateAll(eight).vram.peakGib, estimateAll(full).vram.peakGib);
  assert.equal(estimateAll(eight).vram.fits, estimateAll(full).vram.fits);
});

test('спилл только при `peak > 0.97·budget`: 1792p на 12 ГиБ — да, 1088p на 16 ГиБ — нет', () => {
  const wide = estimateAll(runtimeRender({ width: 3200, height: 1792, vramGib: 12 }));
  assert.ok(wide.vram.peakGib > 0.97 * 12, `пик ${wide.vram.peakGib}`);
  assert.equal(wide.vram.spills, true);
  assert.equal(wide.vram.fits, false);
  assert.equal(wide.vram.zone, 'spill');

  const narrow = estimateAll(runtimeRender({ width: 1920, height: 1088 }));
  assert.ok(narrow.vram.peakGib < 0.97 * 16, `пик ${narrow.vram.peakGib}`);
  assert.equal(narrow.vram.spills, false);
  assert.equal(narrow.vram.fits, true);

  // Граница гейта: `>` а не `≥` — ровно 0.97 бюджета ещё влезает.
  assert.equal(gateBudget(0.97 * 16, 16).spills, false);
  assert.equal(gateBudget(0.97 * 16 + 1e-9, 16).spills, true);
  // Зона `degraded` — между 0.90 и 0.97 бюджета.
  assert.equal(gateBudget(0.95 * 16, 16).zone, 'degraded');
  assert.equal(gateBudget(0.95 * 16, 16).degraded, true);
  assert.equal(gateBudget(0.5 * 16, 16).zone, 'ok');
});

test('диск монотонен: nvme < ssd < hdd по loadS и encodeColdS', () => {
  /** @type {const} */ const order = ['nvme', 'ssd', 'hdd'];
  for (const encoderDevice of /** @type {const} */ (['cuda', 'cpu'])) {
    const times = order.map((disk) => estimateTime(runtimeRender({ disk, encoderDevice }), disk));
    for (let index = 1; index < order.length; index += 1) {
      assert.ok(times[index].loadS > times[index - 1].loadS,
        `${encoderDevice}: loadS ${order[index]} ${times[index].loadS} ≤ ${times[index - 1].loadS}`);
      assert.ok(times[index].encodeColdS > times[index - 1].encodeColdS,
        `${encoderDevice}: encodeColdS ${order[index]} ${times[index].encodeColdS} ≤ ${times[index - 1].encodeColdS}`);
    }
  }
  // На эталонном NVMe оценка загрузки совпадает с калиброванной константой бит-в-бит.
  const nvme = estimateTime(runtimeRender(), 'nvme');
  assert.equal(nvme.loadS, MODEL.coefficients.load.coldS['qwen21-base']);
  assert.equal(nvme.encodeColdS, MODEL.coefficients.encode.coldGpuS);
  // Энкод из кэша эмбеддингов бесплатен (измерено).
  assert.equal(nvme.encodeWarmS, MODEL.coefficients.encode.warmS);
});
