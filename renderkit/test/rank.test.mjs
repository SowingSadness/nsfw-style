// @ts-check
/**
 * Ранжирование сочетаний (шаг 5, A6): Парето-фронт по `(Q↑, time↓, headroom↑)` и победитель по
 * весам уровня.
 *
 * Держим три свойства:
 * - на фронте нет ни одной доминируемой точки (и наоборот, все недоминируемые — на фронте);
 * - победитель не зависит от порядка входа и устойчив к ±10 % весов (Q — эвристика, выбор не
 *   должен разваливаться от правки весов);
 * - tie-break стабилен: Q → меньше времени → больше запаса → `id`.
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { normalizeRequest } from '../src/schema.mjs';
import { resolveGeometry } from '../src/model/geometry.mjs';
import { resolveLevels } from '../src/model/levels.mjs';
import { buildCandidates } from '../src/model/candidates.mjs';
import {
  LEVEL_WEIGHTS,
  metricsOf,
  paretoFront,
  pickBest,
  rankCandidates,
  scoreOf,
} from '../src/model/rank.mjs';

/**
 * Синтетическая точка ранжирования.
 *
 * @param {string} id
 * @param {number} q
 * @param {number} timeS
 * @param {number} headroomGib
 */
function point(id, q, timeS, headroomGib) {
  return { id, metrics: { q, timeS, headroomGib } };
}

/**
 * Сочетания реального запроса (`candidates.mjs`).
 *
 * @param {{width?: number, height?: number, quality?: 'draft'|'standard'|'max'}} [options]
 */
function candidates(options = {}) {
  const request = normalizeRequest({
    hardware: {
      model: 'qwen21-base', vramGib: 16, ramGib: 64, disk: 'nvme',
    },
    image: {
      width: options.width ?? 2560, height: options.height ?? 1440, quality: options.quality ?? 'max',
    },
  });
  const geometry = resolveGeometry(request);
  const levels = resolveLevels(request);
  return buildCandidates(request, { geometry, levels }).candidates;
}

/** @param {any} a @param {any} b */
function dominates(a, b) {
  const ma = metricsOf(a);
  const mb = metricsOf(b);
  return ma.q >= mb.q && ma.timeS <= mb.timeS && ma.headroomGib >= mb.headroomGib
    && (ma.q > mb.q || ma.timeS < mb.timeS || ma.headroomGib > mb.headroomGib);
}

test('Парето: доминируемые точки отброшены, недоминируемые — все на фронте', () => {
  const points = [
    point('a', 100, 10, 1), // доминируема точкой b
    point('b', 120, 10, 2),
    point('c', 100, 5, 0.5), // не доминируема: быстрее всех
    point('d', 90, 20, 5), // не доминируема: самый большой запас
    point('e', 115, 8, 0.6), // не доминируема: между b и c
  ];
  const front = paretoFront(points).map((item) => item.id);
  assert.deepEqual(front.sort(), ['b', 'c', 'd', 'e']);
  // Ни одна точка фронта не доминируема другой точкой фронта.
  for (const a of paretoFront(points)) {
    for (const b of paretoFront(points)) {
      if (a === b) continue;
      assert.equal(dominates(a, b), false, `${a.id} доминирует ${b.id}`);
    }
  }
});

test('Парето на реальных сочетаниях 2560×1440 max: фронт без доминируемых', () => {
  const items = candidates();
  const front = paretoFront(items);
  assert.ok(front.length > 0 && front.length <= items.length);
  for (const item of items) {
    const dominated = front.some((other) => other !== item && dominates(other, item));
    assert.equal(dominated, false, `${item.id} на фронте, но доминируем`);
  }
  // Каждая точка вне фронта доминируема хоть одной точкой фронта.
  for (const item of items) {
    if (front.includes(item)) continue;
    assert.ok(front.some((other) => dominates(other, item)), `${item.id} вне фронта и не доминируем`);
  }
});

test('победитель max на 2560×1440 — полное расписание, плотное внимание, KV в RAM', () => {
  const items = candidates();
  const best = pickBest(items, LEVEL_WEIGHTS.max);
  assert.ok(best !== null);
  assert.equal(best.render.steps, 25, 'Q 1920 против 1152 у 8 шагов');
  assert.equal(best.render.windowAttnPx, 0, 'на 1440p окно не нужно');
  assert.equal(String(best.render.kvCache), 'offload', 'запас VRAM важнее +0.5 % времени');
  assert.equal(Math.round(best.quality.q), 1920);
  assert.equal(best.estimates.vram.zone, 'ok');
  // `rankCandidates` даёт тот же первый элемент, что `pickBest`.
  const ranked = rankCandidates(items, LEVEL_WEIGHTS.max);
  assert.equal(ranked[0].id, best.id);
  for (let index = 1; index < ranked.length; index += 1) {
    assert.ok(ranked[index - 1].score >= ranked[index].score, 'список отсортирован по весу');
  }
});

test('победитель не зависит от порядка входа и от масштаба весов', () => {
  const items = candidates();
  const best = pickBest(items, LEVEL_WEIGHTS.max);
  assert.equal(pickBest([...items].reverse(), LEVEL_WEIGHTS.max).id, best.id);
  // ±10 % на каждую ось по отдельности — тот же победитель.
  const weights = LEVEL_WEIGHTS.max;
  for (const axis of /** @type {const} */ (['q', 'time', 'headroom'])) {
    for (const factor of [0.9, 1.1]) {
      const perturbed = { ...weights, [axis]: weights[axis] * factor };
      assert.equal(pickBest(items, perturbed).id, best.id, `${axis} × ${factor} поменял победителя`);
    }
  }
  // Общий масштаб весов ничего не меняет (веса нормируются на сумму).
  const scaled = { q: weights.q * 3, time: weights.time * 3, headroom: weights.headroom * 3 };
  assert.equal(pickBest(items, scaled).id, best.id);
});

test('tie-break: Q → меньше времени → больше запаса → id', () => {
  // Равные метрики — решает id.
  const identical = [point('c02', 100, 10, 5), point('c01', 100, 10, 5)];
  assert.equal(pickBest(identical, LEVEL_WEIGHTS.max).id, 'c01');
  // Равные Q и время — больше запаса.
  const headroom = [point('c01', 100, 10, 1), point('c02', 100, 10, 4)];
  assert.equal(pickBest(headroom, LEVEL_WEIGHTS.max).id, 'c02');
  // Равные Q и запас — меньше времени.
  const time = [point('c01', 100, 20, 5), point('c02', 100, 10, 5)];
  assert.equal(pickBest(time, LEVEL_WEIGHTS.max).id, 'c02');
  // Разный Q при равном времени и запасе — решает Q даже при «быстрых» весах.
  const quality = [point('c01', 90, 10, 5), point('c02', 100, 10, 5)];
  assert.equal(pickBest(quality, LEVEL_WEIGHTS.draft).id, 'c02');
});

test('веса уровней: три уровня, непустые, draft — про скорость, max — про качество', () => {
  assert.deepEqual(Object.keys(LEVEL_WEIGHTS).sort(), ['draft', 'max', 'standard']);
  for (const [level, weights] of Object.entries(LEVEL_WEIGHTS)) {
    const sum = weights.q + weights.time + weights.headroom;
    assert.ok(sum > 0, `${level}: сумма весов > 0`);
  }
  assert.ok(LEVEL_WEIGHTS.draft.time > LEVEL_WEIGHTS.draft.q);
  assert.ok(LEVEL_WEIGHTS.max.q > LEVEL_WEIGHTS.max.time);
  assert.ok(LEVEL_WEIGHTS.max.q > LEVEL_WEIGHTS.standard.q);
});

test('draft на 2560×1440 выбирает 8 шагов, standard и max — полное расписание', () => {
  const draft = pickBest(candidates({ quality: 'draft' }), LEVEL_WEIGHTS.draft);
  assert.equal(draft.render.steps, 8);
  assert.ok(Math.round(draft.quality.q) === 1152, `Q draft ${draft.quality.q}`);
  const standard = pickBest(candidates({ quality: 'standard' }), LEVEL_WEIGHTS.standard);
  assert.equal(standard.render.steps, 25);
});

test('пустой набор: pickBest → null, rankCandidates → [], scoreOf требует веса > 0', () => {
  assert.equal(pickBest([], LEVEL_WEIGHTS.max), null);
  assert.deepEqual(rankCandidates([], LEVEL_WEIGHTS.max), []);
  assert.throws(() => scoreOf([point('c01', 1, 1, 1)], { q: 0, time: 0, headroom: 0 }), /сумма весов/u);
});
