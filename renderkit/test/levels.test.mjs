// @ts-check
/**
 * Уровни качества (шаг 4, ADR A3): `draft` → 8 шагов, `standard`/`max` → полное расписание,
 * H3 → 6 фиксированных PDD-шагов, `image.steps` — override, который **не меняет `fits`**
 * (пик VRAM от шагов не зависит вовсе).
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { estimateAll, estimateVram } from '../src/model/cost.mjs';
import { LEVEL_POLICY, resolveLevels } from '../src/model/levels.mjs';

/**
 * Минимальный нормализованный запрос (поля, которые читает `levels.mjs`).
 *
 * @param {'draft'|'standard'|'max'} quality
 * @param {{model?: string, steps?: number|null}} [options]
 */
const request = (quality, { model = 'qwen21-base', steps = null } = {}) => ({
  hardware: { model, vramGib: 16, ramGib: 64, disk: 'nvme' },
  image: { width: 2560, height: 1440, quality, bitDepth: 8, steps },
  refs: 1,
  anime: false,
  freeDiskGib: null,
  memory: { kvCache: null, vaeTilePx: null, window: null },
  postprocess: null,
});

test('уровни: draft 8 / standard 25 / max 25 (полное расписание qwen21-base)', () => {
  assert.equal(resolveLevels(request('draft')).steps, 8);
  assert.equal(resolveLevels(request('standard')).steps, 25);
  assert.equal(resolveLevels(request('max')).steps, 25);
  // `draft` короче не ради памяти, а чтобы набор кадров был дешёвым.
  assert.match(resolveLevels(request('draft')).reason, /level-draft/u);
  assert.match(resolveLevels(request('standard')).reason, /level-standard/u);
  assert.match(resolveLevels(request('max')).reason, /level-max/u);
  // Полное расписание модели берётся из инвентаря, а не из константы уровня.
  for (const quality of /** @type {const} */ (['draft', 'standard', 'max'])) {
    const levels = resolveLevels(request(quality));
    assert.equal(levels.fullSteps, 25);
    assert.equal(levels.fixed, false);
    assert.equal(levels.override, null);
  }
});

test('h3: 6 шагов PDD фиксированы, любой `image.steps` игнорируется с причиной', () => {
  for (const quality of /** @type {const} */ (['draft', 'standard', 'max'])) {
    const levels = resolveLevels(request(quality, { model: 'h3-image' }));
    assert.equal(levels.steps, 6, `${quality}: H3 считает 6 шагов`);
    assert.equal(levels.fixed, true);
    assert.equal(levels.ignoredSteps, null);
    assert.match(levels.reason, /fixed-schedule/u);
  }
  const forced = resolveLevels(request('max', { model: 'h3-image', steps: 12 }));
  assert.equal(forced.steps, 6, 'расписание модели главнее');
  assert.equal(forced.ignoredSteps, 12);
  assert.match(forced.reason, /12 шагов проигнорированы/u);
  // Явное совпадение с расписанием — не «игнор».
  assert.equal(resolveLevels(request('max', { model: 'h3-image', steps: 6 })).ignoredSteps, null);
});

test('`image.steps` переопределяет уровень у моделей без фиксированного расписания', () => {
  const levels = resolveLevels(request('max', { steps: 12 }));
  assert.equal(levels.steps, 12);
  assert.equal(levels.override, 12);
  assert.equal(levels.fullSteps, 25);
  assert.match(levels.reason, /steps-override/u);
  for (const quality of /** @type {const} */ (['draft', 'standard', 'max'])) {
    assert.equal(resolveLevels(request(quality, { steps: 16 })).steps, 16);
  }
});

test('`steps` не меняет `fits`: пик VRAM от шагов не зависит (решение 5 плана)', () => {
  /** @type {import('../src/model/cost.mjs').Render} */
  const base = {
    model: 'qwen21-base',
    canvas: { width: 2560, height: 1440 },
    tokens: (2560 / 16) * (1440 / 16),
    steps: 25,
    offload: 'group',
    kvCache: 'offload',
    windowAttnPx: 0,
    marginPx: 0,
    vaeTilePx: 512,
    refs: 1,
    refPx: 960,
    encoderDevice: 'cuda',
    vramGib: 12,
    ramGib: 64,
    disk: 'nvme',
  };
  const draft = resolveLevels(request('draft', { model: 'qwen21-base' }));
  const max = resolveLevels(request('max', { model: 'qwen21-base' }));
  const draftEstimates = estimateAll({ ...base, steps: draft.steps });
  const maxEstimates = estimateAll({ ...base, steps: max.steps });
  assert.equal(draft.steps, 8);
  assert.equal(max.steps, 25);
  assert.equal(estimateVram({ ...base, steps: 8 }), estimateVram({ ...base, steps: 25 }));
  assert.equal(draftEstimates.vram.peakGib, maxEstimates.vram.peakGib);
  assert.equal(draftEstimates.vram.fits, maxEstimates.vram.fits);
  // От шагов зависит только время.
  assert.ok(draftEstimates.time.genPerFrameS < maxEstimates.time.genPerFrameS);
});

test('политика риска: standard — ok-размеры и запас ≥10 %, max — «серая» зона и окно', () => {
  const standard = resolveLevels(request('standard'));
  assert.equal(standard.allowDegradedZone, false);
  assert.equal(standard.allowWindow, false);
  assert.equal(standard.minHeadroom, LEVEL_POLICY.standard.minHeadroom);
  assert.ok(standard.minHeadroom >= 0.1, 'standard требует запас VRAM ≥ 10 %');

  const max = resolveLevels(request('max'));
  assert.equal(max.allowDegradedZone, true);
  assert.equal(max.allowWindow, true);
  assert.equal(max.minHeadroom, 0);

  const draft = resolveLevels(request('draft'));
  assert.equal(draft.allowDegradedZone, false, 'черновик не покупает «серую» зону');
  assert.equal(draft.allowWindow, false);
  assert.equal(draft.minHeadroom, 0);

  // Шаги — единственная ось, которая у уровней различается кроме политики.
  assert.equal(LEVEL_POLICY.standard.steps, null);
  assert.equal(LEVEL_POLICY.max.steps, null);
  assert.equal(LEVEL_POLICY.draft.steps, 8);
  for (const policy of Object.values(LEVEL_POLICY)) {
    assert.ok(policy.note.length > 20, 'у каждой политики объяснение');
  }
});

test('turbo: полное расписание — 8 шагов (уровень и расписание совпадают)', () => {
  const levels = resolveLevels(request('standard', { model: 'qwen21-turbo' }));
  assert.equal(levels.fullSteps, 8);
  assert.equal(levels.steps, 8);
  assert.equal(resolveLevels(request('draft', { model: 'qwen21-turbo' })).steps, 8);
});

test('неизвестный уровень качества — исключение, а не молчаливый дефолт', () => {
  assert.throws(() => resolveLevels(request(/** @type {any} */ ('high'))), RangeError);
});
