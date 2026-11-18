// @ts-check
/**
 * Тесты геометрии и числовых примитивов (шаг 1 плана).
 *
 * Держат три вещи, на которых стоит вся модель:
 * - `pyRound` повторяет питоновский `round()` (half-to-even) — иначе полутоны сетки 32 разъезжаются
 *   с пайплайном (`912p + 16:9 → 1632×896`, а не `1632×928`);
 * - канвас считается **копией** `calculate_dimensions`, поэтому 18+ фикстур сверяют результат с
 *   тем, что реально отрендерит система;
 * - потолок A4: `ok` до 3200×1792, `degraded` до 3840×2176, выше — `impossible`.
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { resolveGeometry, GEOMETRY_WARNINGS, SIZE_LIMITS } from '../src/model/geometry.mjs';
import {
  canvasFor,
  formatGib,
  formatMib,
  formatSeconds,
  kvGib,
  pyRound,
  tokensFor,
} from '../src/units.mjs';

/**
 * Фикстуры канвасов: `[W, H, refs, canvasW, canvasH, aspect, verdict, snapDx, snapDy]`.
 * `aspect === null` — пропорция не из списка, кадр берёт пропорцию референса.
 *
 * @type {Array<[number, number, number, number, number, string|null, string, number, number]>}
 */
const CANVAS_FIXTURES = [
  // 1. «ровно 1080» нативно не бывает: сетка 32 даёт 1088
  [1920, 1080, 1, 1920, 1088, '16:9', 'ok', 0, 8],
  // 2. 1440p native — точное попадание в сетку
  [2560, 1440, 1, 2560, 1440, '16:9', 'ok', 0, 0],
  // 3. реальный max: 3200×1792 = 5.7344 Мп, ok (314 с / 10.3 ГиБ)
  [3200, 1792, 1, 3200, 1792, '16:9', 'ok', 0, 0],
  // 4. 4K-slide: 3840×2176 = 8.3558 Мп — «серая» зона (513 с / 13.8 ГиБ)
  [3840, 2176, 1, 3840, 2176, null, 'degraded', 0, 0],
  // 5. 4K UHD округляется вверх в «серую» зону — не разрешаем (невозможно нативно)
  [3840, 2160, 1, 3840, 2176, '16:9', 'impossible', 0, 16],
  // 6. полутон half-to-even: 912/32 = 28.5 -> 28 -> 896
  [1621, 912, 1, 1632, 896, '16:9', 'ok', 11, -16],
  // 7. 4:3 на сетке
  [1408, 1056, 1, 1408, 1056, '4:3', 'ok', 0, 0],
  // 8. 2:3
  [1024, 1536, 1, 1024, 1536, '2:3', 'ok', 0, 0],
  // 9. 1:1
  [2048, 2048, 1, 2048, 2048, '1:1', 'ok', 0, 0],
  // 10. 960p 16:9 пресета: 1696×960 = 1.7667, отклонение 0.63 % > 0.5 % — не «16:9»
  [1696, 960, 1, 1696, 960, null, 'ok', 0, 0],
  // 11. 768p пресета: 1376×768 = 1.7917, отклонение 0.78 % — тоже не «16:9»
  [1376, 768, 1, 1376, 768, null, 'ok', 0, 0],
  // 12. 1088p native
  [1920, 1088, 1, 1920, 1088, null, 'ok', 0, 0],
  // 13. 600/32 = 18.75 -> 19: канвас 800×608
  [800, 600, 1, 800, 608, '4:3', 'ok', 0, 8],
  // 14. 720p: 720/32 = 22.5 -> 22 (half-to-even) -> 704
  [1280, 720, 1, 1280, 704, '16:9', 'ok', 0, -16],
  // 15. 1440×1080 -> 1440×1088
  [1440, 1080, 1, 1440, 1088, '4:3', 'ok', 0, 8],
  // 16. минимальный размер
  [256, 256, 1, 256, 256, '1:1', 'ok', 0, 0],
  // 17. 3:2 попадает в «серую» зону только после округления вверх -> impossible
  [3000, 2000, 1, 3008, 1984, null, 'impossible', 8, -16],
  // 18. 3456×2048 = 7.08 Мп — сеточный размер внутри «серой» зоны
  [3456, 2048, 1, 3456, 2048, null, 'degraded', 0, 0],
  // 19. T2I: пропорция из списка, рефов нет — работает
  [1920, 1080, 0, 1920, 1088, '16:9', 'ok', 0, 8],
  // 20. T2I экзотической пропорции невозможен
  [1280, 800, 0, 1280, 800, null, 'impossible', 0, 0],
  // 21. та же 16:10, но с референсом — берёт пропорцию референса
  [1280, 800, 2, 1280, 800, null, 'ok', 0, 0],
  // 22. 4096² = 16.78 Мп — выше потолка
  [4096, 4096, 1, 4096, 4096, '1:1', 'impossible', 0, 0],
  // 23. максимум схемы (8192) — тем более невозможно
  [8192, 8192, 1, 8192, 8192, '1:1', 'impossible', 0, 0],
];

test('pyRound повторяет питоновский round (half-to-even)', () => {
  // Фикстура приёмки шага 1.
  assert.equal(pyRound(28.5), 28);
  // Полутон чётный вверх: 67.5 -> 68 (это и даёт 2160 -> 2176).
  assert.equal(pyRound(67.5), 68);
  assert.equal(pyRound(2.5), 2);
  assert.equal(pyRound(3.5), 4);
  assert.equal(pyRound(-2.5), -2);
  assert.equal(pyRound(-3.5), -4);
  // Не полутон — обычное округление.
  assert.equal(pyRound(18.75), 19);
  assert.equal(pyRound(22.5), 22);
  assert.equal(pyRound(0), 0);
  assert.throws(() => pyRound(Number.NaN), RangeError);
});

test('canvasFor — копия calculate_dimensions', () => {
  assert.deepEqual(canvasFor(912, 16 / 9), { width: 1632, height: 896 });
  assert.deepEqual(canvasFor(1056, 4 / 3), { width: 1408, height: 1056 });
  assert.deepEqual(canvasFor(1440, 16 / 9), { width: 2560, height: 1440 });
  assert.deepEqual(canvasFor(1080, 16 / 9), { width: 1920, height: 1088 });
  assert.throws(() => canvasFor(0, 16 / 9), RangeError);
  assert.throws(() => canvasFor(1080, 0), RangeError);
});

test('tokensFor: (W/16)·(H/16) на замеренных канвасах', () => {
  assert.equal(tokensFor(1376, 768), 4128);
  assert.equal(tokensFor(1696, 960), 6360);
  assert.equal(tokensFor(1920, 1088), 8160);
  assert.equal(tokensFor(2560, 1440), 14400);
  assert.equal(tokensFor(3200, 1792), 22400);
  assert.throws(() => tokensFor(1376, 770), RangeError);
});

test('kvGib: 512 КиБ на префиксный токен', () => {
  assert.equal(kvGib(1), 0.000488);
  assert.equal(formatGib(kvGib(4360)), '2.13 ГиБ');
  assert.equal(formatGib(kvGib(8392)), '4.10 ГиБ');
  assert.equal(formatGib(kvGib(14632)), '7.14 ГиБ');
  // 228.6 МиБ на слой при 14632 токенах: 7.14 ГиБ / 32 слоя = 0.2231 ГиБ = 228.5 МиБ
  assert.equal(formatMib((kvGib(14632) * 1024) / 32), '228.5 МиБ');
  assert.throws(() => kvGib(-1), RangeError);
});

test('форматтеры для человека', () => {
  assert.equal(formatGib(12.85), '12.85 ГиБ');
  assert.equal(formatMib(228.6), '228.6 МиБ');
  assert.equal(formatSeconds(8), '8 с');
  assert.equal(formatSeconds(205), '3 мин 25 с');
  assert.equal(formatSeconds(513), '8 мин 33 с');
  assert.equal(formatSeconds(3725), '1 ч 2 мин');
  assert.throws(() => formatSeconds(-1), RangeError);
});

test('18+ фикстур канвасов: сетка, пропорция, вердикт, snap', () => {
  assert.ok(CANVAS_FIXTURES.length >= 18, `ожидается ≥18 фикстур, в наборе ${CANVAS_FIXTURES.length}`);
  for (const [width, height, refs, canvasWidth, canvasHeight, aspect, verdict, dx, dy] of CANVAS_FIXTURES) {
    const geometry = resolveGeometry({ image: { width, height }, refs });
    const label = `${width}×${height} (refs=${refs})`;
    assert.deepEqual(geometry.canvas, { width: canvasWidth, height: canvasHeight }, `${label}: канвас`);
    assert.equal(geometry.aspect, aspect, `${label}: aspect`);
    assert.equal(geometry.verdict, verdict, `${label}: verdict`);
    assert.equal(geometry.snap.dx, dx, `${label}: snap.dx`);
    assert.equal(geometry.snap.dy, dy, `${label}: snap.dy`);
    assert.equal(geometry.sizePx, height, `${label}: sizePx = H`);
    assert.equal(geometry.pixels, canvasWidth * canvasHeight, `${label}: пиксели`);
    assert.equal(geometry.tokens, (canvasWidth / 16) * (canvasHeight / 16), `${label}: токены`);
  }
});

test('912p + 16:9 даёт 1632×896 (полутон сетки)', () => {
  const geometry = resolveGeometry({ image: { width: 1621, height: 912 }, refs: 1 });
  assert.deepEqual(geometry.canvas, { width: 1632, height: 896 });
  assert.equal(geometry.aspect, '16:9');
  assert.equal(geometry.tokens, 5712);
});

test('reachable: две ближайшие точки сетки 32', () => {
  const snapped = resolveGeometry({ image: { width: 1920, height: 1080 }, refs: 1 });
  assert.deepEqual(snapped.snap.reachable, [[1920, 1088], [1920, 1056]]);

  const exact = resolveGeometry({ image: { width: 2560, height: 1440 }, refs: 1 });
  assert.deepEqual(exact.snap.reachable, [[2592, 1440], [2560, 1408]]);
});

test('warnings: коды и их отсутствие', () => {
  const snapped = resolveGeometry({ image: { width: 1920, height: 1080 }, refs: 1 });
  assert.deepEqual(snapped.warnings.map((w) => w.code), [GEOMETRY_WARNINGS.NATIVE_SIZE_SNAPPED]);

  const exact = resolveGeometry({ image: { width: 2560, height: 1440 }, refs: 1 });
  assert.deepEqual(exact.warnings, []);

  const refRatio = resolveGeometry({ image: { width: 1696, height: 960 }, refs: 1 });
  assert.deepEqual(refRatio.warnings.map((w) => w.code), [GEOMETRY_WARNINGS.REF_RATIO_MUST_MATCH]);

  const degraded = resolveGeometry({ image: { width: 3840, height: 2176 }, refs: 1 });
  assert.deepEqual(degraded.warnings.map((w) => w.code), [
    GEOMETRY_WARNINGS.REF_RATIO_MUST_MATCH,
    GEOMETRY_WARNINGS.COMFORT_LIMIT,
  ]);

  const impossible = resolveGeometry({ image: { width: 3840, height: 2160 }, refs: 1 });
  assert.deepEqual(impossible.warnings.map((w) => w.code), [
    GEOMETRY_WARNINGS.NATIVE_SIZE_SNAPPED,
    GEOMETRY_WARNINGS.NO_NATIVE_PATH_FOR_SIZE,
  ]);

  const t2i = resolveGeometry({ image: { width: 1280, height: 800 }, refs: 0 });
  assert.deepEqual(t2i.warnings.map((w) => w.code), [GEOMETRY_WARNINGS.ASPECT_UNSUPPORTED_T2I]);
  assert.deepEqual(t2i.nearestAspects.map((a) => a.key), ['16:9', '4:3']);
});

test('потолки A4 совпадают с измеренными канвасами', () => {
  assert.equal(SIZE_LIMITS.okMaxPx, 3200 * 1792);
  assert.equal(SIZE_LIMITS.hardMaxPx, 3840 * 2176);
  assert.equal(resolveGeometry({ image: { width: 3200, height: 1792 }, refs: 1 }).verdict, 'ok');
  assert.equal(resolveGeometry({ image: { width: 3840, height: 2176 }, refs: 1 }).verdict, 'degraded');
  assert.equal(resolveGeometry({ image: { width: 4000, height: 2304 }, refs: 1 }).verdict, 'impossible');
});

test('resolveGeometry защищён от мусора', () => {
  assert.throws(() => resolveGeometry({ image: { width: 0, height: 1080 }, refs: 1 }), RangeError);
  assert.throws(() => resolveGeometry({ image: { width: 1920, height: 1080 }, refs: -1 }), RangeError);
});
