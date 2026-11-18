// @ts-check
/**
 * Сочетания настроек (шаг 5, ADR A6/A7): оси, авто-значения, гейты, дедуп и стабильные id.
 *
 * Тесты держат четыре свойства, на которых стоит резолв:
 * - набор сочетаний детерминирован: два вызова дают один JSON, id идут подряд `c01…`;
 * - все сочетания проходят плановые гейты (`0.97·budget`, декод VAE, запас уровня, окно уровня),
 *   а в «серой» зоне порог смягчён до `1.00·budget` — иначе решение 8 плана невыразимо;
 * - оси начинаются с авто-значений `memory.resolve*`, но явный `memory.*` их сужает (жёсткий
 *   override): подменять запрос «более удобным» тайлом/режимом KV калькулятор не должен;
 * - контрольные точки плана воспроизводятся моделью: 3200×1792 — окно 384+384, Q ≈ 2323,
 *   оценка времени ≈ 314 с и пик по метрике `nvidia-smi` ≈ 10.3 + 2.05 ГиБ.
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { MODEL } from '../src/data/model.mjs';
import { normalizeRequest } from '../src/schema.mjs';
import { resolveGeometry } from '../src/model/geometry.mjs';
import { resolveLevels } from '../src/model/levels.mjs';
import {
  CANDIDATE_ASSUMPTIONS,
  OFFLOAD_BLOCKS,
  autoOffload,
  buildCandidates,
  candidateKey,
  encoderDeviceFor,
  encoderProfile,
  gateLimit,
  loraScaleFor,
  modelPrecision,
  stepsAxis,
  supportsOffload,
} from '../src/model/candidates.mjs';

/**
 * Запрос + геометрия + уровни + сочетания одним вызовом.
 *
 * @param {{width?: number, height?: number, quality?: 'draft'|'standard'|'max',
 *   hardware?: Record<string, unknown>, image?: Record<string, unknown>, refs?: number,
 *   anime?: boolean, memory?: Record<string, unknown>}} [options]
 */
function build(options = {}) {
  const request = normalizeRequest({
    hardware: {
      model: 'qwen21-base', vramGib: 16, ramGib: 64, disk: 'nvme', ...options.hardware,
    },
    image: {
      width: options.width ?? 2560,
      height: options.height ?? 1440,
      quality: options.quality ?? 'max',
      ...options.image,
    },
    refs: options.refs,
    anime: options.anime,
    memory: options.memory,
  });
  const geometry = resolveGeometry(request);
  const levels = resolveLevels(request);
  return {
    request,
    geometry,
    levels,
    result: buildCandidates(request, { geometry, levels }),
  };
}

test('16 ГиБ / 64 ГиБ / NVMe / base, 2560×1440 max: ≥ 8 сочетаний, все проходят гейты', () => {
  const { result } = build();
  assert.ok(result.candidates.length >= 8, `сочетаний ${result.candidates.length}, ожидалось ≥ 8`);
  assert.equal(result.candidates.length, result.candidates
    .map((candidate) => candidate.key).filter((key, index, keys) => keys.indexOf(key) === index).length,
  'ключи сочетаний уникальны (дедуп)');
  for (const [index, candidate] of result.candidates.entries()) {
    assert.equal(candidate.id, `c${String(index + 1).padStart(2, '0')}`, 'id идут подряд и стабильно');
    assert.ok(candidate.gates.fits, `${candidate.id}: пик в гейте`);
    assert.ok(candidate.estimates.vram.peakGib <= candidate.gates.limitGib, `${candidate.id}: пик ≤ порога`);
    assert.equal(candidate.gates.vaeFit, true);
    assert.equal(candidate.gates.headroomOk, true);
    assert.equal(candidate.gates.windowOk, true);
    assert.ok(candidate.metrics.timeS > 0 && candidate.metrics.q > 0);
  }
  // Детерминизм: повторный вызов — тот же JSON (порядок и числа).
  const again = build();
  assert.equal(JSON.stringify(again.result), JSON.stringify(result));
  // Первое сочетание — каноническое: все оси как решила политика уровня.
  const canonical = result.canonical;
  assert.equal(result.candidates[0].render.steps, canonical.render.steps);
  assert.equal(result.candidates[0].render.offload, canonical.render.offload);
  assert.equal(result.candidates[0].layout.kvCache.mode, canonical.layout.kvCache.mode);
});

test('3200×1792 max: окно 384+384 выигрывает по времени, Q ≈ 2323, ≈ 314 с и ≈ 12.35 ГиБ', () => {
  const { result, geometry } = build({ width: 3200, height: 1792 });
  assert.equal(geometry.verdict, 'ok');
  assert.ok(result.candidates.length > 0);
  // Окно внимания VRAM не меняет (замер), поэтому плотное внимание на 22400 токенах тоже проходит
  // гейт — оба режима в списке и с одинаковым пиком; окно выигрывает временем (314 против 376 с).
  const modes = new Set(result.candidates.map((candidate) => candidate.layout.attention.mode));
  assert.deepEqual([...modes].sort(), ['dense', 'window']);
  const peaks = new Set(result.candidates.map((candidate) => candidate.estimates.vram.peakGib.toFixed(6)));
  assert.equal(peaks.size, 1, 'окно пик VRAM не меняет');
  const canonical = result.canonical;
  assert.equal(canonical.layout.attention.mode, 'window');
  assert.equal(canonical.layout.attention.windowPx, 384);
  assert.equal(canonical.layout.attention.marginPx, 384);
  assert.ok(Math.abs(canonical.layout.attention.coverage - 0.643) < 0.01);
  assert.equal(canonical.layout.attention.bands, 5);
  assert.equal(canonical.render.steps, 25);
  // A7: на 1792p сетка референсов грубее выхода (960 против 1792) — иначе префикс KV
  // стоит как канвас (22400 токенов на референс).
  assert.equal(canonical.layout.ref.refPx, 960, 'ref_px 960: префикс KV дешевле выхода на 1792p');

  // Время: замер 314 с (окно) и 376 с (плотное), модель в пределах ±15 %.
  const genS = canonical.estimates.time.genPerFrameS;
  assert.ok(Math.abs(genS - 314) / 314 <= 0.15, `время ${genS} против замера 314`);
  const dense = result.candidates.find((candidate) => candidate.layout.attention.mode === 'dense');
  assert.ok(dense, 'плотное сочетание тоже в списке');
  assert.ok(Math.abs(dense.estimates.time.genPerFrameS - 376) / 376 <= 0.15,
    `плотное ${dense.estimates.time.genPerFrameS} против 376`);

  // Пик: замер §4 — 10.3 ГиБ «своего процесса», метрика модели — `nvidia-smi`
  // (тот же кадр = 10.3 + пересчёт рабочего стола 2.05 ГиБ по `MODEL`).
  const measuredPeak = 10.3 + MODEL.coefficients.vram.measurementDesktopReserveGib;
  const peak = canonical.estimates.vram.peakGib;
  assert.ok(Math.abs(peak - measuredPeak) / measuredPeak <= 0.15, `пик ${peak} против ${measuredPeak}`);

  // Q из плана: √5 734 400 · 1.00 (25 шагов) · 0.97 (окно) · 1.00 (комфорт) ≈ 2323.
  assert.ok(Math.abs(canonical.quality.q - 2323) < 5, `Q = ${canonical.quality.q}`);
});

test('оси: шаги уровня + draft, оба режима KV, тайлы 512/0/1024, только поддержанный offload', () => {
  const max = build();
  assert.deepEqual(stepsAxis(max.levels), [25, 8], 'полное расписание первым, draft — альтернатива');
  assert.deepEqual(stepsAxis(build({ quality: 'draft' }).levels), [8], 'у draft альтернативы нет');
  const overridden = build({ quality: 'max', image: { steps: 12 } });
  assert.deepEqual(stepsAxis(overridden.levels), [12], 'явный steps — одна точка');
  const h3 = build({ hardware: { model: 'h3-image', ramGib: 64 }, width: 1024, height: 1024 });
  assert.deepEqual(stepsAxis(h3.levels), [6], 'H3 — фиксированное расписание PDD');

  const kvModes = new Set(max.result.candidates.map((candidate) => candidate.render.kvCache));
  assert.deepEqual([...kvModes].sort(), ['offload', 'true'], 'KV: политика и альтернатива');
  const tiles = new Set(max.result.candidates.map((candidate) => candidate.render.vaeTilePx));
  assert.deepEqual([...tiles].sort((a, b) => /** @type {number} */ (a) - /** @type {number} */ (b)), [0, 512, 1024]);
  const offloads = new Set(max.result.candidates.map((candidate) => candidate.render.offload));
  assert.deepEqual([...offloads], ['group'], 'на 14400 токенах offload=none не влезает — ось вырождена');

  // H3 не объявляет раскладку памяти: ось вырождена в `null`.
  assert.deepEqual([...new Set(h3.result.candidates.map((candidate) => candidate.render.kvCache))], [false]);
  assert.deepEqual([...new Set(h3.result.candidates.map((candidate) => candidate.render.vaeTilePx))], [null]);
});

test('явный memory.* сужает ось (жёсткий override), а не подменяется «удобным» значением', () => {
  const kvTrue = build({ width: 1920, height: 1088, memory: { kvCache: true } });
  assert.deepEqual([...new Set(kvTrue.result.candidates.map((c) => String(c.render.kvCache)))], ['true']);
  for (const candidate of kvTrue.result.candidates) {
    assert.equal(candidate.layout.kvCache.mode, 'true', 'решение по KV следует запросу');
    assert.equal(candidate.layout.kvCache.placement, 'vram');
  }

  const tile0 = build({ memory: { vaeTilePx: 0 } });
  assert.deepEqual([...new Set(tile0.result.candidates.map((c) => c.render.vaeTilePx))], [0]);

  const denseOnly = build({ width: 3200, height: 1792, memory: { window: false } });
  assert.equal(denseOnly.result.canonical.render.windowAttnPx, 0, 'окно выключено запросом');
  assert.ok(denseOnly.result.candidates.length > 0,
    'плотное внимание на 1792p влезает: окно VRAM не экономит');
  for (const candidate of denseOnly.result.candidates) {
    assert.equal(candidate.layout.attention.mode, 'dense', `${candidate.id}: только плотное`);
  }
});

test('гейты: 0.97·budget отбрасывает, «серая» зона (1.00·budget) оставляет и деградирует', () => {
  const normal = build();
  assert.equal(normal.result.limitGib, 16 * 0.97);
  assert.equal(normal.result.greyZone, false);

  const grey = build({ width: 3840, height: 2176, refs: 3 });
  assert.equal(grey.geometry.verdict, 'degraded');
  assert.equal(grey.result.greyZone, true);
  assert.equal(grey.result.limitGib, 16, 'в «серой» зоне порог смягчён до 1.00·budget');
  assert.ok(grey.result.candidates.length > 0, '4K рендерится (решение 8 плана)');
  for (const candidate of grey.result.candidates) {
    assert.ok(candidate.estimates.vram.peakGib <= 16);
    assert.equal(candidate.estimates.vram.spills, false, 'смягчённый гейт: не спилл, а degraded');
    assert.equal(candidate.estimates.vram.zone, 'degraded');
  }

  // standard не покупает «серую» зону (levels.allowDegradedZone = false) — порог остаётся 0.97.
  const standard = build({ width: 3840, height: 2176, quality: 'standard' });
  assert.equal(standard.result.greyZone, false);
  assert.equal(standard.result.limitGib, 16 * 0.97);

  const small = build({ width: 1920, height: 1080 });
  assert.equal(gateLimit(small.request, small.geometry, small.levels).greyZone, false);
});

test('A7: энкодер идёт на CPU при vram ≤ 8, на GPU — стримингом', () => {
  assert.equal(encoderDeviceFor(8), 'cpu');
  assert.equal(encoderDeviceFor(12), 'cuda');
  assert.equal(encoderDeviceFor(16), 'cuda');
  assert.deepEqual(encoderProfile('qwen21-base', 16), {
    name: 'heretic', device: 'cuda', stream: true, prefetch: 4,
  });
  assert.deepEqual(encoderProfile('qwen21-base', 8), {
    name: 'heretic', device: 'cpu', stream: true, prefetch: 4,
  });
  // CPU-путь тоже стримит: 16.33 ГиБ весов в RAM целиком не поднимаются (AGENTS §0).
  assert.equal(encoderProfile('h3-image', 8).stream, true);
});

test('прочие авто-значения: offload, LoRA, точность, offloadBlocks', () => {
  const { request, geometry, levels } = build();
  const render = { ...build().result.canonical.render };
  assert.equal(autoOffload(render, gateLimit(request, geometry, levels).limitGib), 'group');
  assert.equal(supportsOffload('qwen21-base', 'group'), true);
  assert.equal(supportsOffload('h3-image', 'none'), false, 'у H3 нет режима offload=none');
  assert.equal(loraScaleFor({ anime: false }, 'qwen21-base'), 0);
  assert.equal(loraScaleFor({ anime: true }, 'qwen21-base'), 0.7, 'сила адаптера из inventory');
  assert.equal(loraScaleFor({ anime: true }, 'h3-image'), 0, 'у H3 адаптера нет');
  assert.equal(modelPrecision('qwen21-base'), 'fp8');
  assert.equal(OFFLOAD_BLOCKS, 1);
  // Провенанс допущений генератора сочетаний — как у констант фита.
  for (const [key, entry] of Object.entries(CANDIDATE_ASSUMPTIONS)) {
    assert.ok(entry.source.length > 0, `${key}: source`);
    assert.ok(entry.note.length > 0, `${key}: note`);
    assert.ok(['measured', 'assumption'].includes(entry.kind), `${key}: kind`);
  }
});

test('ключ дедупа включает конфиг и канвас: смена резолюции — другое сочетание', () => {
  const a = build().result.canonical.render;
  const b = build().result.canonical.render;
  assert.equal(candidateKey(a), candidateKey(b));
  const taller = build({ height: 1792, width: 3200 }).result.canonical.render;
  assert.notEqual(candidateKey(a), candidateKey(taller), 'другой канвас — другой ключ');
  const otherSteps = candidateKey({ ...a, steps: 8 });
  assert.notEqual(candidateKey(a), otherSteps, 'другие шаги — другой ключ');
});
