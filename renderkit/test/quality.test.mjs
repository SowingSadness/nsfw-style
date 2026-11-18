// @ts-check
/**
 * Q (шаг 4, ADR A6): `√pixels · steps · attention · comfort`.
 * Проверяется монотонность факторов, границы `comfortFactor` (1.00 на 5.73 Мп, 0.75 на 8.35 Мп,
 * `null` выше) и главное свойство — **апскейла в Q нет вовсе**.
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { SIZE_LIMITS } from '../src/model/geometry.mjs';
import {
  QUALITY_FACTORS,
  attentionFactor,
  comfortFactor,
  qualityScore,
  stepsFactor,
} from '../src/model/quality.mjs';

test('stepsFactor монотонен по таблице A6 и прижат к краям', () => {
  const table = QUALITY_FACTORS.steps.values;
  assert.deepEqual(table.map((row) => [row.steps, row.factor]),
    [[8, 0.6], [12, 0.75], [16, 0.85], [25, 1]]);
  let previous = 0;
  for (let steps = 8; steps <= 25; steps += 1) {
    const value = stepsFactor(steps);
    assert.ok(value >= previous - 1e-12, `шаги ${steps}: ${value} < ${previous}`);
    previous = value;
  }
  assert.equal(stepsFactor(25), 1);
  assert.equal(stepsFactor(8), 0.6);
  // Ниже 8 (H3 считает 6 шагов PDD) и выше 25 — край таблицы.
  assert.equal(stepsFactor(6), 0.6);
  assert.equal(stepsFactor(30), 1);
  // Между точками — линейная интерполяция.
  assert.ok(stepsFactor(20) > 0.85 && stepsFactor(20) < 1);
  assert.throws(() => stepsFactor(0), RangeError);
});

test('attentionFactor: плотное 1.00, окно 0.97 (резкость 15.2 против 14.8)', () => {
  assert.equal(attentionFactor('dense'), 1);
  assert.equal(attentionFactor('window'), 0.97);
  assert.ok(Math.abs(attentionFactor('window') / attentionFactor('dense') - 0.974) < 0.01);
  // @ts-expect-error неправильный режим обязан падать, а не молча давать 1
  assert.throws(() => attentionFactor('bands'), RangeError);
});

test('comfortFactor: 1.00 до 5.73 Мп, 0.75 на 8.35 Мп, null выше', () => {
  assert.equal(comfortFactor(SIZE_LIMITS.okMaxPx), 1, '3200×1792 — комфортный максимум');
  assert.equal(comfortFactor(2560 * 1440), 1);
  assert.equal(comfortFactor(SIZE_LIMITS.hardMaxPx), 0.75, '3840×2176 — «серая» зона');
  assert.equal(comfortFactor(4000 * 2200), null, 'выше потолка нативного пути нет');
  // Спад линейный: середина диапазона — середина фактора.
  const middle = (SIZE_LIMITS.okMaxPx + SIZE_LIMITS.hardMaxPx) / 2;
  assert.ok(Math.abs(comfortFactor(middle) - 0.875) < 1e-12);
  let previous = 1;
  for (let pixels = SIZE_LIMITS.okMaxPx; pixels < SIZE_LIMITS.hardMaxPx; pixels += 100_000) {
    const value = /** @type {number} */ (comfortFactor(pixels));
    assert.ok(value <= previous, `спад не монотонен на ${pixels}`);
    previous = value;
  }
  assert.throws(() => comfortFactor(0), RangeError);
});

test('Q монотонен по пикселям (в комфортной зоне) и по шагам', () => {
  const at = (width, height, steps = 25, attention = 'dense') => {
    const score = qualityScore({ pixels: width * height, steps, attention });
    assert.ok(score.q !== null);
    return /** @type {number} */ (score.q);
  };
  const sizes = [[1376, 768], [1920, 1088], [2560, 1440], [3200, 1792]];
  let previous = 0;
  for (const [width, height] of sizes) {
    const q = at(width, height);
    assert.ok(q > previous, `${width}×${height}: Q ${q} не вырос`);
    previous = q;
  }
  const steps = [8, 12, 16, 25].map((value) => at(1376, 768, value));
  for (let index = 1; index < steps.length; index += 1) {
    assert.ok(steps[index] > steps[index - 1], `шаги: ${steps[index]} не больше ${steps[index - 1]}`);
  }
});

test('Q совпадает с числами плана: 3200×1792 с окном ≈ 2323, 1376×768 draft ≈ 617', () => {
  const max = qualityScore({ pixels: 3200 * 1792, steps: 25, attention: 'window' });
  assert.ok(Math.abs(/** @type {number} */ (max.q) - 2323) < 1, `Q = ${max.q}`);
  const draft = qualityScore({ pixels: 1376 * 768, steps: 8, attention: 'dense' });
  assert.ok(Math.abs(/** @type {number} */ (draft.q) - 617) < 1, `Q = ${draft.q}`);
  const wide = qualityScore({ pixels: 1920 * 1088, steps: 25, attention: 'dense' });
  assert.ok(Math.abs(/** @type {number} */ (wide.q) - 1445) < 1, `Q = ${wide.q}`);
  const hd = qualityScore({ pixels: 2560 * 1440, steps: 25, attention: 'dense' });
  assert.ok(Math.abs(/** @type {number} */ (hd.q) - 1920) < 1, `Q = ${hd.q}`);
});

test('выше «серой» зоны Q обнуляется, а не занижается', () => {
  const beyond = qualityScore({ pixels: 4000 * 2200, steps: 25, attention: 'dense' });
  assert.equal(beyond.q, null);
  assert.equal(beyond.beyondComfort, true);
  assert.equal(beyond.comfortFactor, null);
  // Остальные факторы при этом посчитаны — их видно в trace.
  assert.equal(beyond.stepsFactor, 1);
  assert.equal(beyond.attentionFactor, 1);
});

test('апскейла в Q нет: ровно три множителя и ни одного поля про постобработку', () => {
  const score = qualityScore({ pixels: 2560 * 1440, steps: 25, attention: 'dense' });
  assert.deepEqual(Object.keys(score).sort(),
    ['attentionFactor', 'beyondComfort', 'comfortFactor', 'nativePixels', 'q', 'stepsFactor']);
  const raw = JSON.stringify(score).toLowerCase();
  assert.equal(raw.includes('upscale'), false);
  assert.equal(raw.includes('texture'), false);
  assert.equal(score.nativePixels, 2560 * 1440, 'Q считается по native-пикселям');
  assert.deepEqual(Object.keys(QUALITY_FACTORS).sort(), ['attention', 'comfort', 'steps']);
  for (const factor of Object.values(QUALITY_FACTORS)) {
    const typed = /** @type {{kind: string, source: string, note: string}} */ (factor);
    assert.equal(typed.kind, 'assumption', 'каждый фактор — эвристика, а не замер Q');
    assert.ok(typed.source.length > 8 && typed.note.length > 20, 'нужны source и note');
  }
});
