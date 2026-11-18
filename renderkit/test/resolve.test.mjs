// @ts-check
/**
 * Резолв плана (шаг 5): вердикт, ладдер, оценки, стадии и контракт постобработки.
 *
 * Инварианты, которые проверяются:
 * - форма `Plan` (разделы из «Выход (`Plan`)» плана) и детерминизм: два вызова — один JSON;
 * - порядок вердикта `impossible > unsupported > degraded > ok` и его причины
 *   (3840×2160, 8 ГиБ на 1920×1080, H3 на 32 ГиБ RAM, 16 бит, «серая» зона);
 * - ладдер **без шагов** и в фиксированном порядке, применённые ступени — в `degradations[]`;
 * - `standard` и `max` дают одинаковый `render`, когда потолок `ok` (шаги не рычаг памяти);
 * - `stages.postprocess` в Phase A всегда выключена (`estimates: null`), а запрос стадии
 *   принимается (verdict не меняется) — без `PhaseNotSupported`; параметры чужой
 *   модели отвергаются `PlanError` с `allowed`.
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { computePlan, RULESET_VERSION } from '../src/index.mjs';
import { PlanError, normalizeRequest } from '../src/schema.mjs';
import { LADDER, resolvePlan } from '../src/model/resolve.mjs';
import { resolvePostprocess, validatePostprocessParams } from '../src/model/postprocess.mjs';

/**
 * Запрос плана.
 *
 * @param {{width?: number, height?: number, quality?: 'draft'|'standard'|'max', bitDepth?: 8|16,
 *   vramGib?: number, ramGib?: number, disk?: string, model?: string, refs?: number,
 *   anime?: boolean, memory?: Record<string, unknown>, freeDiskGib?: number}} [options]
 */
function request(options = {}) {
  return {
    hardware: {
      model: options.model ?? 'qwen21-base',
      vramGib: options.vramGib ?? 16,
      ramGib: options.ramGib ?? 64,
      disk: options.disk ?? 'nvme',
    },
    image: {
      width: options.width ?? 2560,
      height: options.height ?? 1440,
      quality: options.quality ?? 'max',
      bitDepth: options.bitDepth ?? 8,
    },
    refs: options.refs,
    anime: options.anime,
    memory: options.memory,
    freeDiskGib: options.freeDiskGib,
  };
}

/** @param {Record<string, any>} [options] */
const plan = (options = {}) => computePlan(request(options));

/** @param {any} planResult */
const codes = (planResult) => planResult.warnings.map((warning) => warning.code);

test('Plan: форма по схеме, детерминизм и 8 полей трейса', () => {
  const planResult = plan();
  assert.equal(planResult.schemaVersion, 1);
  assert.equal(planResult.rulesetVersion, RULESET_VERSION);
  assert.match(planResult.rulesetVersion, /^r1\+[0-9a-f]{12}-[0-9a-f]{12}\+emit1$/u);
  assert.equal(planResult.request.tier, 'VRAM16');
  assert.equal(planResult.verdict, 'ok');
  assert.equal(planResult.confidence, 'measured');
  // Шаг 6: артефакт собирается из этого же плана — строка YAML и команды запуска.
  assert.equal(typeof planResult.yaml, 'string', 'YAML собирается для собранного плана');
  assert.ok(planResult.yaml.length > 0, 'YAML не пустой');
  assert.equal(typeof planResult.cli.run, 'string', 'CLI-команды — объект {save, run, gen, checks, sets}');

  assert.deepEqual(planResult.image.requested, { width: 2560, height: 1440 });
  assert.deepEqual(planResult.image.native, {
    width: 2560, height: 1440, tokens: 14400, sizePx: 1440,
  });
  assert.equal(planResult.image.aspect, '16:9');
  assert.equal(planResult.image.exact, true);
  assert.equal(planResult.image.format, 'png');
  assert.deepEqual(planResult.image.snap.reachable, [[2592, 1440], [2560, 1408]]);

  assert.equal(planResult.render.steps, 25);
  assert.equal(planResult.render.offload, 'group');
  assert.equal(planResult.render.offloadBlocks, 1);
  assert.equal(planResult.render.precision, 'fp8');
  assert.deepEqual(planResult.render.encoder, {
    name: 'heretic', device: 'cuda', stream: true, prefetch: 4,
  });
  assert.equal(planResult.render.loraScale, 0);
  assert.equal(planResult.render.memory.kvCache.mode, 'offload');
  assert.equal(planResult.render.memory.kvCache.placement, 'ram');
  assert.ok(planResult.render.memory.kvCache.gib > 0);
  assert.equal(planResult.render.memory.vae.tilePx, 512);
  assert.equal(planResult.render.memory.vae.stridePx, 384);
  assert.equal(planResult.render.memory.attention.mode, 'dense');
  assert.equal(planResult.render.memory.ref.px, 960);

  assert.equal(planResult.stages[0].stage, 'render');
  assert.equal(planResult.stages[0].enabled, true);
  assert.equal(planResult.stages[0].quality, 1920);
  assert.equal(planResult.stages[1].stage, 'postprocess');

  assert.ok(planResult.estimates.vram.peakGib > 0 && planResult.estimates.vram.fits);
  assert.ok(planResult.estimates.time.genPerFrameS > 0);
  assert.ok(planResult.estimates.disk.pngPerFrameMb > 0);

  assert.ok(planResult.combinations.length >= 8);
  assert.equal(planResult.combinations.filter((row) => row.best).length, 1, 'ровно один победитель');

  assert.equal(planResult.trace.length, 8, 'трейс по 8 ключевым полям');
  const fields = planResult.trace.map((entry) => entry.field);
  assert.deepEqual(fields, [
    'image.canvas',
    'render.steps',
    'render.memory.kvCache.mode',
    'render.memory.vae.tilePx',
    'render.memory.attention.mode',
    'render.memory.ref.px',
    'render.offload',
    'render.encoder.device',
  ]);
  for (const entry of planResult.trace) {
    assert.ok(entry.ruleId.length > 0, `${entry.field}: ruleId`);
    assert.ok(entry.source.length > 0, `${entry.field}: source`);
    assert.ok(Array.isArray(entry.rejected), `${entry.field}: rejected`);
    for (const rejected of entry.rejected) {
      assert.ok(rejected.why.length > 0, `${entry.field}: причина отказа`);
    }
  }

  // Детерминизм: тот же запрос — тот же JSON (включая порядок сочетаний).
  assert.equal(JSON.stringify(plan()), JSON.stringify(planResult));
});

test('standard и max при 2560×1440 дают одинаковый render (шаги — не рычаг памяти)', () => {
  const standard = plan({ quality: 'standard' });
  const max = plan({ quality: 'max' });
  assert.deepEqual(standard.render, max.render, 'раскладка и кадр совпадают');
  assert.equal(standard.verdict, 'ok');
  assert.equal(max.verdict, 'ok');
  assert.equal(standard.stages[0].quality, max.stages[0].quality);
  // Политика отличается: у standard нет окна и «серой» зоны, поэтому сочетаний меньше.
  assert.ok(standard.combinations.length <= max.combinations.length);
});

test('1920×1080: нативно недостижим — snap 1920×1088 + native-size-snapped', () => {
  const planResult = plan({ width: 1920, height: 1080 });
  assert.deepEqual(planResult.image.native, {
    width: 1920, height: 1088, tokens: 8160, sizePx: 1080,
  });
  assert.equal(planResult.image.snap.dy, 8);
  assert.equal(planResult.image.snap.dx, 0);
  assert.equal(planResult.image.exact, false);
  assert.ok(codes(planResult).includes('native-size-snapped'));
  assert.notEqual(planResult.verdict, 'impossible');
});

test('3840×2160 → impossible: no-native-path-for-size и ближайшие размеры', () => {
  const planResult = plan({ width: 3840, height: 2160 });
  assert.equal(planResult.verdict, 'impossible');
  assert.equal(planResult.render, null);
  assert.equal(planResult.estimates, null);
  assert.deepEqual(planResult.combinations, []);
  assert.ok(codes(planResult).includes('no-native-path-for-size'));
  assert.equal(planResult.stages[0].enabled, false);
  assert.ok(planResult.suggestions.some((suggestion) => suggestion.code === 'smaller-canvas'));
  assert.ok(planResult.suggestions.some((suggestion) => suggestion.code === 'enable-postprocess'));
  const smaller = planResult.suggestions.find((suggestion) => suggestion.code === 'smaller-canvas');
  assert.equal(smaller.value, '3200×1792', 'реальный потолок плана');
});

test('8 ГиБ VRAM + 1920×1080 → impossible + suggestions (меньший кадр, другое железо)', () => {
  const planResult = plan({ width: 1920, height: 1080, vramGib: 8, ramGib: 16 });
  assert.equal(planResult.verdict, 'impossible');
  assert.equal(planResult.render, null);
  assert.ok(planResult.suggestions.some((suggestion) => suggestion.code === 'smaller-canvas'));
  assert.ok(planResult.suggestions.some((suggestion) => suggestion.code === 'other-hardware'));
  assert.ok(codes(planResult).includes('encoder-cpu-slow'), 'на 8 ГиБ энкодер идёт на CPU');
  // Причина невозможности видна в стадии: сочетаний нет и ладдер не помог.
  assert.match(planResult.stages[0].reason, /no-native-candidate|влезает/u);
});

test('16 бит → unsupported: оценки заполнены, размер файла ×2', () => {
  const eight = plan();
  const sixteen = plan({ bitDepth: 16 });
  assert.equal(sixteen.verdict, 'unsupported');
  assert.ok(codes(sixteen).includes('bit-depth-16-unsupported'));
  assert.ok(sixteen.render !== null, 'YAML останется валидным');
  assert.ok(sixteen.estimates !== null && sixteen.estimates.vram.peakGib > 0);
  assert.equal(
    Math.round(sixteen.estimates.disk.pngPerFrameMb * 100),
    Math.round(eight.estimates.disk.pngPerFrameMb * 2 * 100),
    '16 бит — ×2 к размеру PNG',
  );
  assert.equal(sixteen.estimates.disk.bitDepthFactor, 2);
});

test('H3 на 32 ГиБ RAM → impossible; на 64 ГиБ — считается', () => {
  const blocked = plan({
    model: 'h3-image', width: 1024, height: 1024, ramGib: 32, vramGib: 16,
  });
  assert.equal(blocked.verdict, 'impossible');
  assert.ok(codes(blocked).includes('h3-ram-32gb'));
  assert.ok(blocked.suggestions.some((suggestion) => suggestion.code === 'other-hardware'));

  const ok = plan({
    model: 'h3-image', width: 1024, height: 1024, ramGib: 64, vramGib: 16,
  });
  assert.equal(ok.verdict, 'ok');
  assert.equal(ok.render.steps, 6, 'расписание PDD фиксировано');
  assert.equal(ok.render.memory.kvCache.mode, null, 'H3 не объявляет kv_cache');
  assert.equal(ok.render.memory.vae.tilePx, null);
  assert.equal(ok.render.encoder.name, 'h3qwen32-heretic');
});

test('ладдер: фиксированный порядок без шагов, применённая ступень — в degradations', () => {
  assert.deepEqual(LADDER.map((rung) => rung.axis), [
    'kvCache', 'offload', 'vaeTilePx', 'vaeTilePx', 'refPx', 'refPx', 'refPx', 'refs',
  ]);
  assert.deepEqual(LADDER.map((rung) => rung.id), [
    'kv-offload', 'offload-group', 'vae-tile-512', 'vae-tile-0',
    'ref-px-960', 'ref-px-768', 'ref-px-640', 'refs-trim',
  ]);
  for (const rung of LADDER) {
    assert.doesNotMatch(`${rung.id} ${rung.field}`, /steps/u, 'шагов в ладдере нет (решение 5)');
  }

  // Явный тайл 1024 на 12 ГиБ не влезает в декод → ладдер спускает его до 512.
  const degraded = plan({
    width: 1920, height: 1088, vramGib: 12, ramGib: 32, memory: { vaeTilePx: 1024 },
  });
  assert.equal(degraded.verdict, 'degraded');
  assert.equal(degraded.render.memory.vae.tilePx, 512);
  assert.deepEqual(degraded.degradations.map((step) => step.code), ['vae-tile-512']);
  const step = degraded.degradations[0];
  assert.equal(step.ruleId, 'A7.vae-tile');
  assert.equal(step.from, 1024);
  assert.equal(step.to, 512);
  assert.ok(step.reason.length > 0);
  // Ладдер не трогает шаги: изменилась только раскладка памяти.
  assert.equal(degraded.render.steps, 25);

  // Повторный вызов — та же деградация (план идемпотентен).
  const again = plan({
    width: 1920, height: 1088, vramGib: 12, ramGib: 32, memory: { vaeTilePx: 1024 },
  });
  assert.equal(JSON.stringify(again.degradations), JSON.stringify(degraded.degradations));
});

test('«серая» зона 3840×2176 max → degraded + comfort-limit, пик на пределе', () => {
  const planResult = plan({ width: 3840, height: 2176, refs: 3 });
  assert.equal(planResult.verdict, 'degraded');
  assert.ok(codes(planResult).includes('comfort-limit'));
  // Пик в «серой» зоне сидит на самом пределе (15.50 из 16 ГиБ): до гейта 0.97·16 = 15.52 остаётся
  // 0.02 ГиБ, поэтому `vram-spill-risk` не выдаётся — сочетание проходит без смягчения.
  assert.ok(planResult.estimates.vram.peakGib > 15.4, `пик ${planResult.estimates.vram.peakGib}`);
  assert.equal(planResult.render.memory.attention.mode, 'window');
  assert.equal(planResult.render.memory.attention.windowPx, 1088);
  assert.equal(planResult.render.memory.attention.marginPx, 448);
  assert.ok(planResult.stages[0].quality < 2300, 'Q падает в «серой» зоне');

  // standard «серую» зону не покупает — ответ impossible с явной подсказкой.
  const standard = plan({ width: 3840, height: 2176, quality: 'standard' });
  assert.equal(standard.verdict, 'impossible');
  assert.ok(standard.suggestions.some((suggestion) => suggestion.code === 'quality-max'));
});

test('KV в VRAM по запросу: деградация видна предупреждением, а не тишиной', () => {
  const planResult = plan({ width: 2560, height: 1440, memory: { kvCache: true } });
  assert.equal(planResult.render.memory.kvCache.mode, 'true');
  assert.equal(planResult.verdict, 'degraded', 'пик уходит в зону 0.90–0.97 бюджета');
  assert.ok(codes(planResult).includes('kv-vram-vs-ram'));
});

test('стадия postprocess в Phase A выключена при любом запросе', () => {
  const requests = [
    {},
    { width: 1920, height: 1080, quality: 'draft' },
    { bitDepth: 16 },
    { model: 'h3-image', width: 1024, height: 1024 },
    { width: 3840, height: 2160 },
    { width: 1920, height: 1080, vramGib: 8, ramGib: 16 },
  ];
  for (const options of requests) {
    const planResult = plan(options);
    const stage = planResult.stages.find((entry) => entry.stage === 'postprocess');
    assert.ok(stage, JSON.stringify(options));
    assert.equal(stage.enabled, false);
    assert.equal(stage.estimates, null, 'бюджеты стадий не смешиваются');
    assert.equal(stage.paramsSchema, null);
    assert.equal(stage.reason, 'модель постобработки не передана');
    assert.ok(codes(planResult).includes('postprocess-not-connected'));
  }
});

test('postprocess в запросе: стадия выключена, warning с пометкой, verdict не меняется', () => {
  const requested = {
    ...request(),
    postprocess: { model: 'seedvr2-3b', params: { tile: 256 } },
  };
  const planResult = computePlan(requested);
  // Запрос принимается: план собирается так же, как без стадии (verdict от неё не зависит).
  assert.equal(planResult.verdict, plan().verdict);
  assert.equal(planResult.render.memory.vae.tilePx, 512, 'postprocess не трогает рендер-стадию');

  const stage = planResult.stages[1];
  assert.deepEqual(stage, {
    stage: 'postprocess',
    enabled: false,
    reason: 'модель постобработки не подключена',
    estimates: null,
    paramsSchema: null,
    model: 'seedvr2-3b',
  });

  const warning = planResult.warnings.find((entry) => entry.code === 'postprocess-not-connected');
  assert.ok(warning, 'warning о несоединённой стадии есть');
  assert.match(warning.detail, /запрошена/u, 'видно, что запрос был');
  assert.match(warning.detail, /seedvr2-3b/u, 'названа запрошенная модель');
  assert.match(warning.detail, /нет в inventory/u, 'одна связная фраза, без остатков конкатенации');
  assert.doesNotMatch(warning.detail, /'\s*\+\s*'/u, 'в тексте нет литеральных склеек');
  assert.equal(warning.field, 'postprocess');
  assert.equal(planResult.request.postprocess.model, 'seedvr2-3b', 'запрос сохранён в плане');
});

test('postprocess запрошен, а размер недостижим — обычный impossible + suggestion', () => {
  const planResult = computePlan({ ...request({ width: 3840, height: 2160 }), postprocess: { model: 'seedvr2-3b' } });
  assert.equal(planResult.verdict, 'impossible');
  const suggestion = planResult.suggestions.find((entry) => entry.code === 'enable-postprocess');
  assert.ok(suggestion, 'подсказка про стадию есть');
  assert.match(suggestion.detail, /уже запрошена/u);
  assert.match(suggestion.detail, /seedvr2-3b/u);
  assert.match(suggestion.detail, /inventory/u);
  assert.doesNotMatch(suggestion.detail, /'\s*\+\s*'/u, 'в тексте нет литеральных склеек');
  assert.ok(planResult.warnings.some((entry) => entry.code === 'postprocess-not-connected'));
});

test('resolvePostprocess: контракт стадии, а не заглушка «ноль»', () => {
  const stage = resolvePostprocess(normalizeRequest(request()), null, null);
  assert.deepEqual(stage, {
    stage: 'postprocess',
    enabled: false,
    reason: 'модель постобработки не передана',
    estimates: null,
    paramsSchema: null,
    model: null,
  });
});

test('validatePostprocessParams: чужая модель приносит свои параметры и свой allowed', () => {
  const descriptor = {
    model: 'demo-postprocess',
    params: [
      {
        scope: 'postprocess', name: 'precision', type: /** @type {const} */ ('enum'), default: 'fp8', choices: ['fp8', 'bf16'],
      },
      {
        scope: 'postprocess', name: 'tile', type: /** @type {const} */ ('int'), default: 256, choices: null,
      },
      {
        scope: 'postprocess', name: 'model_id', type: /** @type {const} */ ('string'), default: 'x', readonly: true,
      },
    ],
  };
  // Значения по умолчанию и порядок объявления.
  assert.deepEqual(validatePostprocessParams(null, descriptor), {
    model: 'demo-postprocess',
    values: { precision: 'fp8', tile: 256, model_id: 'x' },
  });
  assert.deepEqual(validatePostprocessParams({ tile: 512 }, descriptor).values, {
    precision: 'fp8', tile: 512, model_id: 'x',
  });

  // Чужой параметр — PlanError с `allowed` из набора модели.
  const error = /** @type {PlanError} */ (
    (() => {
      try {
        validatePostprocessParams({ target: 'x' }, descriptor);
        return null;
      } catch (caught) {
        return caught;
      }
    })()
  );
  assert.ok(error instanceof PlanError);
  assert.equal(error.code, 'unknown-postprocess-param');
  assert.equal(error.field, 'postprocess.params.target');
  assert.deepEqual(error.allowed, ['precision', 'tile', 'model_id']);

  // Тип/перечисление и readonly — тоже ошибки, а не молчаливое приведение.
  assert.throws(
    () => validatePostprocessParams({ precision: 'fp8x' }, descriptor),
    (caught) => caught instanceof PlanError && caught.code === 'invalid-postprocess-param'
      && caught.allowed.includes('fp8'),
  );
  assert.throws(
    () => validatePostprocessParams({ tile: 1.5 }, descriptor),
    (caught) => caught instanceof PlanError && caught.code === 'invalid-postprocess-param',
  );
  assert.throws(
    () => validatePostprocessParams({ model_id: 'y' }, descriptor),
    (caught) => caught instanceof PlanError && caught.code === 'readonly-postprocess-param',
  );
  assert.throws(
    () => validatePostprocessParams({ precision: 'fp8' }, null),
    (caught) => caught instanceof PlanError && caught.code === 'postprocess-model-not-provided',
  );
});

test('3200×1792 max через публичный computePlan: окно 384/384, ref_px 960, Q 2323', () => {
  const planResult = plan({ width: 3200, height: 1792 });
  assert.equal(planResult.verdict, 'ok');
  assert.ok(planResult.combinations.length >= 8, `сочетаний ${planResult.combinations.length}`);
  // Окно VRAM не экономит (замер), поэтому плотное внимание тоже в списке — с тем же пиком;
  // победитель всё равно оконный: −17 % времени против −2.6 % резкости.
  const attentionModes = new Set(planResult.combinations.map((row) => row.attention));
  assert.deepEqual([...attentionModes].sort(), ['dense', 'window']);
  const { attention, ref } = planResult.render.memory;
  assert.equal(attention.mode, 'window');
  assert.equal(attention.windowPx, 384);
  assert.equal(attention.marginPx, 384);
  assert.ok(Math.abs(attention.coverage - 0.643) < 0.01);
  assert.equal(attention.bands, 5);
  assert.ok(attention.costS < attention.denseCostS, 'окно быстрее плотного — bWindow < 0');
  assert.equal(ref.px, 960, 'ref_px 960 — политика A7 для 1792p');
  assert.equal(planResult.stages[0].quality, 2323, 'Q победителя из плана');
  assert.ok(planResult.estimates.vram.peakGib > 9, 'оценка пика заполнена');
  assert.ok(Math.abs(planResult.estimates.time.genPerFrameS - 314) / 314 <= 0.15);
  // `standard` окно не покупает, но плотное внимание теперь проходит — кадр достижим.
  const standard = plan({ width: 3200, height: 1792, quality: 'standard' });
  assert.equal(standard.verdict, 'ok');
  assert.equal(standard.render.memory.attention.mode, 'dense');
  assert.equal(standard.estimates.vram.peakGib, planResult.estimates.vram.peakGib,
    'окно и плотное внимание занимают одинаково VRAM');
});

test('resolvePlan: блокеры невозможности описаны кодом и причиной (не пустые)', () => {
  const requestObject = normalizeRequest(request({ width: 1920, height: 1080, vramGib: 8, ramGib: 16 }));
  const resolution = resolvePlan(requestObject);
  assert.equal(resolution.verdict, 'impossible');
  assert.ok(resolution.blockers.length > 0);
  for (const blocker of resolution.blockers) {
    assert.ok(blocker.code.length > 0 && blocker.detail.length > 0);
  }
  assert.equal(resolution.estimates, null);
  assert.equal(resolution.render, null);
  // Объяснение при этом есть: трейс считается по канонической раскладке.
  assert.equal(resolution.trace.length, 8);
});
