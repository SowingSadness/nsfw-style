// @ts-check
/**
 * Раскладка памяти (шаг 4, ADR A7): цена KV-префикса, правило `true`/`offload`, рефы на 768p,
 * тайлинг VAE (`stride = 0.75·tile`, запрет `-1` на ≥1.5 Мп), окно внимания (порог, покрытие,
 * `equivalentToDense`) и политика `ref_px` (кламп по канвасу, ≤13000 токенов, инвариант сетки).
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { MODEL } from '../src/data/model.mjs';
import { kvGib } from '../src/units.mjs';
import { estimateVram, prefixTokens } from '../src/model/cost.mjs';
import {
  MEMORY_ASSUMPTIONS,
  REF_LADDER,
  VAE_TILES,
  refGrid,
  resolveAttention,
  resolveKvCache,
  resolveMemory,
  resolveRefPx,
  resolveVaeTile,
} from '../src/model/memory.mjs';
/**
 * Полный `Render` для тестов (все поля, которые читают cost/memory).
 *
 * @param {Record<string, any>} [overrides]
 * @returns {import('../src/model/cost.mjs').Render}
 */
function render(overrides = {}) {
  const width = overrides.width ?? 1376;
  const height = overrides.height ?? 768;
  return /** @type {any} */ ({
    model: 'qwen21-base',
    canvas: { width, height },
    tokens: (width / 16) * (height / 16),
    steps: 25,
    offload: 'group',
    kvCache: null,
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

/** @param {number} width @param {number} height @param {Record<string, any>} [overrides] */
const canvasRender = (width, height, overrides = {}) => render({ width, height, ...overrides });

test('KV: 512 КиБ на префиксный токен на четырёх измеренных точках', () => {
  const points = /** @type {const} */ ([[4360, 2.13], [6592, 3.22], [8392, 4.1], [14632, 7.14]]);
  for (const [tokens, expected] of points) {
    assert.ok(Math.abs(kvGib(tokens) - expected) < 0.01,
      `kvGib(${tokens}) = ${kvGib(tokens)} против измеренного ${expected}`);
    assert.ok(Math.abs(MODEL.coefficients.kvGibPerToken * tokens - expected) < 0.01,
      `коэффициент модели на ${tokens} токенах не сходится с замером ${expected}`);
  }
  // Префикс из 232 текстовых токенов + канвас 4128 = 4360 (строка `[kv]` vram-tiers.md).
  const decision = resolveKvCache(canvasRender(1376, 768, { textTokens: 232 }));
  assert.equal(decision.prefixTokens, 4360);
  assert.ok(Math.abs(decision.gib - 2.13) < 0.01, `gib = ${decision.gib}`);
  // 7.14 ГиБ на 32 слоя — ≈228.6 МиБ на слой (A7: `perLayerMib`; разница в 0.1 МиБ — округление).
  const large = resolveKvCache(canvasRender(2560, 1440, { textTokens: 232 }));
  assert.equal(large.prefixTokens, 14632);
  assert.ok(Math.abs(/** @type {number} */ (large.perLayerMib) - 228.6) < 0.2,
    `perLayerMib = ${large.perLayerMib}`);
});

test('KV: `true` только при `peak + kvGib ≤ 0.90·budget`, иначе `offload`', () => {
  /** @type {Array<[number, number, string]>} */
  const cases = [
    [1376, 768, '768p'],
    [1920, 1088, '1088p'],
    [1376, 768, '768p-12gib'],
    [864, 480, '480p'],
    [2560, 1440, '1440p'],
  ];
  for (const [width, height, label] of cases) {
    const base = canvasRender(width, height, { vramGib: label.endsWith('12gib') ? 12 : 16 });
    const decision = resolveKvCache(base);
    const peak = estimateVram({ ...base, kvCache: 'offload' }, decision.prefixTokens);
    const fits = peak + decision.gib <= 0.9 * base.vramGib;
    assert.equal(decision.mode, fits ? 'true' : 'offload', `${label}: ${decision.reason}`);
    assert.equal(decision.fitsForVram, fits, `${label}: fitsForVram`);
    assert.equal(decision.placement, fits ? 'vram' : 'ram', `${label}: placement`);
    assert.equal(decision.costPct, fits ? 0 : MEMORY_ASSUMPTIONS['kv.offloadCostPct'].value,
      `${label}: costPct`);
    assert.ok(decision.peakWithKvGib > decision.peakGib, `${label}: KV должен что-то стоить`);
  }
  // 1440p на 16 ГиБ: 12.07 + 7.76 = 19.83 > 14.40 — в VRAM не влезает.
  const wide = resolveKvCache(canvasRender(2560, 1440));
  assert.equal(wide.mode, 'offload');
  assert.match(wide.reason, /kv-vram-does-not-fit/u);
});

test('KV: 2+ референса на 768p — `offload` (измеренный OOM авто-режима)', () => {
  for (const refs of [2, 3]) {
    const decision = resolveKvCache(canvasRender(1376, 768, { refs }));
    assert.equal(decision.mode, 'offload', `refs=${refs}: ${decision.reason}`);
    assert.match(decision.reason, /kv-ref2-rule|kv-vram/u);
    assert.ok(decision.gib > resolveKvCache(canvasRender(1376, 768)).gib, 'префикс должен расти');
  }
});

test('KV: `false` — только по явному запросу (иначе авто никогда его не берёт)', () => {
  const muted = resolveKvCache(canvasRender(1376, 768), false);
  assert.equal(muted.mode, 'false');
  assert.equal(muted.placement, 'none');
  assert.equal(muted.costPct, MEMORY_ASSUMPTIONS['kv.noCacheCostPct'].value);
  assert.match(muted.reason, /muted-by-request/u);
  for (const [width, height] of /** @type {const} */ ([[864, 480], [1376, 768], [1920, 1088], [2560, 1440], [3200, 1792]])) {
    const auto = resolveKvCache(canvasRender(width, height));
    assert.notEqual(auto.mode, 'false', `${width}×${height}: авто не выключает кеш`);
  }
  // Явный `true` при нехватке бюджета превращается в `offload` с предупреждением.
  const forced = resolveKvCache(canvasRender(2560, 1440), true);
  assert.equal(forced.mode, 'offload');
  assert.match(forced.reason, /kv-vram-vs-ram/u);
});

test('KV: префикс считается той же сеткой референса, что `memory.ref.tokensPerRef`', () => {
  // Одна формула на обе величины: `ref_px 960` на 16:9 — сетка 1696×960 = 6360 токенов
  // (KV 3.10 ГиБ), а не `(960/16)² = 3600` (2.49 ГиБ), как считал `cost.prefixTokens` раньше.
  const cases = /** @type {const} */ ([
    [2560, 1440, 960],
    [3200, 1792, 960],
    [3840, 2176, 768],
    [1376, 768, 0],
  ]);
  for (const [width, height, refPx] of cases) {
    const base = canvasRender(width, height, { refs: 2, refPx, textTokens: 232 });
    const decision = resolveRefPx(base);
    // Как в `candidates.buildCandidates`: политика сначала решает `ref_px` (с клампом по канвасу и
    // потолком токенов), и только потом считается префикс — иначе KV и `memory.ref` разъезжаются.
    const resolved = { ...base, refPx: decision.refPx };
    const kv = resolveKvCache(resolved);
    assert.equal(decision.tokensPerRef, refGrid(decision.refPx, base.canvas).tokensPerRef,
      `${width}×${height}: tokensPerRef и сетка`);
    assert.equal(kv.prefixTokens, 232 + 2 * decision.tokensPerRef,
      `${width}×${height}: префикс = text + refs·tokensPerRef`);
    assert.equal(prefixTokens(resolved), kv.prefixTokens, `${width}×${height}: prefixTokens и KV`);
  }
  // Измеренные значения A7: 768p/1 реф — сетка кадра (4128), 1440p/960 — 6360, ref_px 0 на 1440p — 14400.
  assert.equal(resolveKvCache(canvasRender(1376, 768, { refs: 1, refPx: 0, textTokens: 232 })).prefixTokens, 232 + 4128);
  assert.equal(resolveRefPx(canvasRender(2560, 1440)).tokensPerRef, 6360);
  // Явный `0` при одном рефе — сетка кадра (14400). При двух рефах она превышает потолок
  // 13000 токенов, и `refs-token-cap` опускает сетку до 960 (иначе OOM энкодера на [4/5]).
  assert.equal(resolveRefPx(canvasRender(2560, 1440, { refs: 1 }), 0).tokensPerRef, 14400);
  const capped = resolveRefPx(canvasRender(2560, 1440, { refs: 2 }), 0);
  assert.equal(capped.tokensPerRef, 6360);
  assert.equal(capped.reduced, true);
  assert.match(capped.reason, /refs-token-cap/);
  assert.ok(capped.refTokens <= 13000, `потолок соблюдён: ${capped.refTokens}`);
  // `ref_px` режется по канвасу — префикс тоже (сетка не бывает больше кадра).
  const clamped = canvasRender(1376, 768, { refs: 1, refPx: 960, textTokens: 232 });
  assert.equal(prefixTokens(clamped), 232 + refGrid(768, clamped.canvas).tokensPerRef);
});

test('VAE: `stride = 0.75·tile` и ожидаемая сетка 16 px у всех тайлов', () => {
  for (const [tile, expected] of /** @type {const} */ ([[512, 384], [0, 192], [1024, 768]])) {
    const decision = resolveVaeTile(canvasRender(1376, 768), tile);
    assert.equal(decision.tilePx, tile === 0 ? 0 : tile);
    assert.equal(decision.stridePx, expected, `tile ${tile}: шаг`);
    assert.equal(decision.stridePx, 0.75 * VAE_TILES[tile].tilePx, `tile ${tile}: правило 0.75`);
    assert.ok(decision.gridArtifact !== null);
  }
  assert.equal(resolveVaeTile(render(), null).tilePx, 512, 'по умолчанию 512');
  assert.equal(resolveVaeTile(render({ vramGib: 24 }), null).tilePx, 1024, '≥24 ГиБ — 1024');
  assert.equal(resolveVaeTile(render({ vramGib: 12 }), null).tilePx, 512, '12 ГиБ — 512');
});

test('VAE: `-1` запрещён при ≥1.5 Мп (измеренный OOM), ниже — разрешён', () => {
  const big = resolveVaeTile(canvasRender(2560, 1440), -1);
  assert.equal(big.tilePx, 512, 'без тайлинга на 3.69 Мп — OOM, остаётся 512');
  assert.deepEqual(big.rejected, { value: -1, why: 'vae-tile-oom-risk' });
  assert.match(big.reason, /vae-tile-oom-risk/u);
  assert.equal(big.peakGib, VAE_TILES[512].peakGib, 'у выбранного тайла есть пик декода');

  const small = resolveVaeTile(canvasRender(864, 480), -1);
  assert.equal(small.tilePx, -1, 'на 0.41 Мп без тайлинга допустим');
  assert.equal(small.stridePx, null);
  assert.equal(small.rejected, null);
});

test('Окно: включается только при canvasH ≥ 1600 и покрытии ≥ 0.60', () => {
  // Порог намеренно ниже измеренного рабочего покрытия 0.64 (AGENTS.md §4: 36 % ломается, 64 % корректно).
  assert.equal(MEMORY_ASSUMPTIONS['window.minCoverage'].value, 0.6);

  assert.equal(resolveAttention(canvasRender(1376, 768)).mode, 'dense');
  assert.match(resolveAttention(canvasRender(1376, 768)).reason, /window-not-needed/u);

  // 1600×900: ширина большая, но высота ниже `window.minCanvasH` — окно не включается.
  const short = resolveAttention(canvasRender(1600, 900));
  assert.equal(short.mode, 'dense', short.reason);
  assert.match(short.reason, /window-not-needed/u);
  assert.equal(short.bands, 1);
  assert.equal(short.coverage, 1, 'у плотного внимания покрытие полное');

  // 3200×1792 — измеренный рабочий конфиг: окно 384, запас 384, покрытие 64 %, ладдер не добирает.
  const plan = resolveAttention(canvasRender(3200, 1792));
  assert.equal(plan.mode, 'window', plan.reason);
  assert.equal(plan.windowPx, 384, 'ширина полосы из замера 1792p');
  assert.equal(plan.marginPx, 384, 'проверенный запас не раздувается добивкой');
  assert.ok(Math.abs(plan.coverage - 0.643) < 0.001, `покрытие ${plan.coverage}`);
  assert.ok(plan.coverage >= MEMORY_ASSUMPTIONS['window.minCoverage'].value, `покрытие ${plan.coverage}`);
  assert.equal(plan.bands, 5, 'полос — ceil(1792 / 384)');
  assert.equal(plan.bands, Math.ceil(1792 / 384));
  assert.equal(plan.equivalentToDense, false);

  assert.equal(resolveAttention(canvasRender(3200, 1792), { enabled: false, windowPx: null, marginPx: null }).mode,
    'dense', 'явное выключение уважается');
});

test('Окно: 3840×2176 — крупный канвас 1088/448, покрытие ≈ 0.91, две полосы', () => {
  const wide = resolveAttention(canvasRender(3840, 2176));
  assert.equal(wide.mode, 'window', wide.reason);
  assert.equal(wide.windowPx, 1088, 'в «серой» зоне — 1088 из замера 4K');
  assert.equal(wide.marginPx, 448);
  // Точное значение — (1088 + 2·448) / 2176 = 1984 / 2176 ≈ 0.912.
  assert.ok(Math.abs(wide.coverage - 0.913) < 0.002, `покрытие ${wide.coverage}`);
  assert.ok(wide.coverage >= MEMORY_ASSUMPTIONS['window.minCoverage'].value, `покрытие ${wide.coverage}`);
  assert.equal(wide.bands, 2, 'полос — ceil(2176 / 1088)');
  assert.equal(wide.equivalentToDense, false);
});

test('Окно: явный запрос на низком канвасе (1280×720) — `windowPx ≥ canvasH` даёт `equivalentToDense`', () => {
  // Явный флаг обходит только `window.minCanvasH`, но не проверку вырождения окна.
  const oneBand = resolveAttention(canvasRender(1280, 720), { enabled: true, windowPx: 384, marginPx: null });
  assert.equal(oneBand.mode, 'window', oneBand.reason);
  assert.equal(oneBand.windowPx, 384);
  assert.equal(oneBand.bands, 2, 'при 384 px окне на 720 px канвасе — две полосы');
  assert.equal(oneBand.equivalentToDense, false);

  const equivalent = resolveAttention(canvasRender(1280, 720), { enabled: true, windowPx: 768, marginPx: null });
  assert.equal(equivalent.mode, 'window', equivalent.reason);
  assert.equal(equivalent.equivalentToDense, true, 'canvasH 720 ≤ окна 768 — одна полоса = плотное');
  assert.equal(equivalent.bands, 1);
  assert.equal(equivalent.coverage, 1);
});

test('Окно: `canvasH ≤ windowPx` помечается `equivalentToDense` и не дублирует плотное', () => {
  const same = resolveAttention(canvasRender(3200, 1792), { enabled: true, windowPx: 1792, marginPx: 0 });
  assert.equal(same.mode, 'window');
  assert.equal(same.equivalentToDense, true);
  assert.equal(same.bands, 1);
  assert.equal(same.coverage, 1);
  assert.match(same.reason, /window-equivalent-to-dense/u);
});

test('ref_px: кламп по канвасу и округление вниз до сетки 32 (как embedding_crop)', () => {
  const clamped = resolveRefPx(canvasRender(1376, 768), 960);
  assert.equal(clamped.refPx, 768, 'ref_px не бывает больше канваса');
  assert.equal(clamped.clamped, true);
  assert.equal(clamped.reason.includes('ref-px-clamped'), true);
  assert.ok(clamped.refWidth <= 1376 && clamped.refHeight <= 768);

  const snapped = resolveRefPx(canvasRender(2560, 1440), 1000);
  assert.equal(snapped.refPx, 992, '1000 → 992 (пол до сетки 32)');
  assert.equal(snapped.clamped, true);

  // Ни одна сетка референса не больше сетки кадра — инвариант плана.
  for (const value of [0, 256, 384, 640, 768, 960, 1440, 4096]) {
    const decision = resolveRefPx(canvasRender(2560, 1440), value);
    assert.ok(decision.refWidth <= 2560 && decision.refHeight <= 1440,
      `ref_px ${value} → ${decision.refWidth}×${decision.refHeight}`);
  }
});

test('ref_px: политика 768p→0, 1088p→0, 1440p→960, ≥1600p→960', () => {
  const cases = /** @type {const} */ ([
    [1376, 768, 0],
    [1920, 1088, 0],
    [2560, 1440, 960],
    [3200, 1792, 960],
    [3840, 2176, 960],
  ]);
  for (const [width, height, expected] of cases) {
    const decision = resolveRefPx(canvasRender(width, height));
    assert.equal(decision.refPx, expected, `${width}×${height}: ${decision.reason}`);
  }
  // `0` = сетка кадра: референс стоит столько же токенов, сколько канвас.
  const canvasGrid = resolveRefPx(canvasRender(1376, 768));
  assert.equal(canvasGrid.tokensPerRef, 4128);
  assert.equal(canvasGrid.refWidth, 1376);
  // 960 на 16:9 — 1696×960 = 6360 токенов (как в плане).
  const policy = resolveRefPx(canvasRender(2560, 1440));
  assert.equal(policy.tokensPerRef, 6360);
});

test('ref_px: 3 референса упираются в потолок ≈13000 токенов', () => {
  const three = resolveRefPx(canvasRender(2560, 1440, { refs: 3 }));
  assert.ok(three.refTokens <= MEMORY_ASSUMPTIONS['ref.tokenCap'].value,
    `${three.refTokens} > ${MEMORY_ASSUMPTIONS['ref.tokenCap'].value}`);
  assert.equal(three.reduced, true);
  assert.equal(three.refPx, 768, 'ладдер спускает 960 → 768');
  assert.match(three.reason, /refs-token-cap/u);

  // 768p и 3 рефа: сетка кадра ещё влезает в потолок (3 × 4128 = 12384 ≤ 13000).
  const small = resolveRefPx(canvasRender(1376, 768, { refs: 3 }));
  assert.equal(small.refPx, 768, 'policy 960 клампится по канвасу (768 px)');
  assert.equal(small.reduced, false, 'уменьшать нечего — потолок не нарушен');
  assert.ok(small.refTokens <= 13000);

  // Ладдер только уменьшает значение и заканчивается на 256 px (пол embedding_crop).
  assert.deepEqual([...REF_LADDER].sort((a, b) => b - a), [...REF_LADDER]);
  assert.equal(REF_LADDER[REF_LADDER.length - 1], 256);
});

test('ref_px: `0` никогда не дешевле `960` по префиксу (инвариант «не быстрее»)', () => {
  // Больше префиксных токенов ⇒ не меньше времени (при KV в RAM) и не меньше VRAM (при `true`).
  const big = canvasRender(2560, 1440);
  const atCanvas = prefixTokens({ ...big, refPx: 0 });
  const at960 = prefixTokens({ ...big, refPx: 960 });
  assert.ok(atCanvas > at960, `${atCanvas} против ${at960}`);
  assert.ok(kvGib(atCanvas) > kvGib(at960));
  // И сетка референса не больше сетки кадра — проверяется на самом решении.
  const decision = resolveRefPx(big);
  assert.ok(decision.tokensPerRef <= big.tokens);
});

test('H3 не объявляет раскладку памяти: KV/тайл/окно/ref_px — `applicable: false`', () => {
  const h3 = render({ model: 'h3-image', canvas: { width: 1088, height: 1024 }, tokens: (1088 / 16) * (1024 / 16), steps: 6 });
  const kv = resolveKvCache(h3);
  assert.equal(kv.mode, null);
  assert.equal(kv.applicable, false);
  assert.equal(kv.gib, 0);
  assert.match(kv.reason, /model-has-no-kv-cache/u);

  const vae = resolveVaeTile(h3);
  assert.equal(vae.tilePx, null);
  assert.equal(vae.applicable, false);

  const attention = resolveAttention(h3);
  assert.equal(attention.mode, 'dense');
  assert.equal(attention.applicable, false);

  const ref = resolveRefPx(h3);
  assert.equal(ref.applicable, false);
  assert.equal(ref.refPx, 0);
});

test('`resolveMemory` собирает все четыре оси в фиксированном порядке', () => {
  const memory = resolveMemory(canvasRender(2560, 1440, { refs: 2 }));
  assert.deepEqual(Object.keys(memory), ['kvCache', 'vae', 'attention', 'ref']);
  assert.equal(memory.kvCache.mode, 'offload', '2 рефа — RAM');
  assert.equal(memory.vae.tilePx, 512);
  assert.equal(memory.attention.mode, 'dense');
  assert.equal(memory.ref.refPx, 960);
  // Явные переопределения запроса доходят до решений.
  const forced = resolveMemory(canvasRender(1376, 768), { kvCache: false, vaeTilePx: 0, window: { enabled: true, windowPx: null, marginPx: null } });
  assert.equal(forced.kvCache.mode, 'false');
  assert.equal(forced.vae.tilePx, 0);
  assert.equal(forced.attention.mode, 'window', 'окно включено явно, даже на 768 px');
});

test('`refGrid` — копия embedding_crop: пол 256 px и округление вниз до сетки 32', () => {
  const grid = refGrid(256, { width: 2560, height: 1440 });
  assert.equal(grid.refHeight, 256, 'пол 256 px');
  assert.equal(grid.refWidth, 448, 'round(256·1.7778)=455 → 448 (пол до 32)');
  assert.equal(grid.tokensPerRef, (448 / 16) * (256 / 16));
  // `ref_px: 0` = сетка кадра целиком.
  assert.deepEqual(refGrid(0, { width: 864, height: 480 }), {
    refWidth: 864, refHeight: 480, tokensPerRef: 1620,
  });
});
