// @ts-check
/**
 * Вью-модель UI (шаг 7 плана): состояние формы ↔ хеш, `buildView`, `scanYaml`.
 *
 * Тесты идут через **публичный** `computePlan`, поэтому падают не только на вёрстке, но и на
 * расхождении ядра с тем, что страница обещает пользователю (вердикт, дельта, «сейчас»).
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import { computePlan } from '../src/index.mjs';
import { HELP } from '../src/data/help.mjs';
import {
  DEFAULT_STATE,
  buildView,
  hashFromState,
  hintLive,
  requestFromState,
  scanYaml,
  stateFromHash,
} from '../web/viewmodel.mjs';

/**
 * @param {Record<string, unknown>} [patch]
 * @param {Record<string, unknown>} [overrides]
 * @returns {any} `Plan`
 */
function planFor(patch = {}, overrides = {}) {
  return computePlan(requestFromState({ ...DEFAULT_STATE, ...patch }, overrides));
}

/** 16 ГиБ / 64 ГБ / NVMe / base / 2560×1440 / max — «лучшее сочетание» плана. */
const BASE_QUERY = { image: { width: 2560, height: 1440, quality: 'max' }, hardware: { model: 'qwen21-base', vramGib: 16, ramGib: 64, disk: 'nvme' } };

test('stateFromHash: полный хеш, пустой хеш и мусор', () => {
  const full = stateFromHash('#w=1920&h=1080&q=standard&bits=16&model=h3-image&vram=12&ram=32&disk=hdd&refs=3&anime=1');
  assert.deepEqual(full, {
    w: 1920, h: 1080, q: 'standard', bits: 16, model: 'h3-image',
    vram: 12, ram: 32, disk: 'hdd', refs: 3, anime: true,
  });
  assert.deepEqual(stateFromHash(''), { ...DEFAULT_STATE });
  assert.deepEqual(stateFromHash('#'), { ...DEFAULT_STATE });
  const junk = stateFromHash('#w=abc&h=99999&q=high&bits=12&model=nope&vram=0&disk=usb&refs=-1&anime=maybe');
  assert.deepEqual(junk, { ...DEFAULT_STATE }, 'мусор в хеше не роняет форму: каждое поле — умолчание');
  assert.equal(stateFromHash('#w=1280&h=720&anime=1').anime, true);
  assert.equal(stateFromHash('#w=1280&h=720&anime=0').anime, false);
});

test('hashFromState: стабильный порядок и round-trip', () => {
  const hash = hashFromState({ ...DEFAULT_STATE, w: 1920, h: 1080, q: 'draft', anime: true });
  assert.equal(hash, '#w=1920&h=1080&q=draft&bits=8&model=qwen21-base&vram=16&ram=64&disk=nvme&refs=1&anime=1');
  assert.deepEqual(stateFromHash(hash), { ...DEFAULT_STATE, w: 1920, h: 1080, q: 'draft', anime: true });
  assert.equal(hashFromState(stateFromHash(hash)), hash, 'хеш → состояние → хеш идемпотентно');
});

test('requestFromState: форма запроса ядра и переопределения', () => {
  const request = requestFromState({ ...DEFAULT_STATE, w: 1920, h: 1080, q: 'draft', bits: 16, anime: true }, { kvCache: true, freeDiskGib: 200 });
  assert.deepEqual(request, {
    image: { width: 1920, height: 1080, quality: 'draft', bitDepth: 16 },
    hardware: { model: 'qwen21-base', vramGib: 16, ramGib: 64, disk: 'nvme' },
    refs: 1,
    anime: true,
    memory: { kvCache: true },
    freeDiskGib: 200,
  });
  assert.equal(computePlan(request).verdict, 'unsupported', 'посчитанный запрос доезжает до ядра');
  assert.equal('memory' in requestFromState(DEFAULT_STATE), false, 'без переопределений memory не выписывается');
  assert.equal(requestFromState(DEFAULT_STATE, { steps: 8 }).image.steps, 8);
});

test('buildView: ok на 2560×1440 — вердикт, победитель, YAML, сочетания', () => {
  const plan = planFor();
  const view = buildView(plan);
  assert.equal(view.verdict, 'ok');
  assert.deepEqual(view.badge, { label: 'ok', tone: 'ok' });
  assert.match(view.headline, /2560×1440/u);
  assert.match(view.headline, /25 шагов/u);
  assert.match(view.headline, /Q 1920/u);

  assert.equal(view.form.tier, 'VRAM16');
  assert.equal(view.form.native, '2560×1440');
  assert.deepEqual(view.form.state, { ...DEFAULT_STATE });

  assert.equal(view.winner.canvas.native, '2560×1440');
  assert.equal(view.winner.canvas.tokens, 14400);
  assert.equal(view.winner.canvas.snap, 'точно');
  assert.equal(view.winner.steps.count, 25);
  assert.equal(view.winner.steps.fixed, false);
  assert.equal(view.winner.output.bitDepth, 8);
  assert.equal(view.winner.memory.kvPlacement, 'ram');
  assert.equal(view.winner.estimates.confidence, 'measured');
  assert.ok(view.winner.estimates.vramPeak <= view.winner.estimates.vramBudget, 'пик VRAM внутри бюджета');

  assert.equal(typeof view.yaml, 'string');
  assert.match(view.yaml, /upscale: null/u);
  assert.match(view.yaml, /kv_cache:/u);
  assert.match(view.yaml, /vae_tile_px:/u);
  assert.equal(view.yaml, plan.yaml, 'вью-модель не пересобирает YAML');
  assert.ok(view.cli && view.cli.run.includes('--config'));

  assert.ok(view.combinations.length >= 8, `сочетаний ${view.combinations.length}, ждали ≥8`);
  const best = view.combinations.filter((row) => row.best);
  assert.equal(best.length, 1, 'победитель ровно один');
  assert.equal(best[0].delta, '—');
  for (const row of view.combinations) {
    assert.equal(row.canvas, '2560×1440');
    assert.equal(typeof row.delta, 'string');
    // Сочетание может отличаться только тайлом декодера — тогда дельта честно `—`.
  }
  assert.ok(
    view.combinations.some((row) => !row.best && row.delta !== '—'),
    'хотя бы у одного сочетания видна дельта к победителю',
  );
  assert.ok(view.combinations.some((row) => row.pareto), 'Парето-фронт не пуст');

  for (const key of ['kv', 'vae', 'window', 'ref', 'offload']) {
    const row = view.memoryRows.find((item) => item.key === key);
    assert.ok(row, `в раскладке памяти нет строки ${key}`);
    assert.ok(HELP[row.helpId], `${key}: helpId ${row.helpId} отсутствует в HELP`);
    assert.ok(row.reason.length > 0, `${key}: причина не пуста`);
    assert.ok(row.cost.length > 0, `${key}: цена не пуста`);
  }
});

test('buildView: draft на 16 ГиБ — degraded (пик 14.6 из 16), standard — ok', () => {
  const draft = buildView(planFor({ q: 'draft' }));
  assert.equal(draft.verdict, 'degraded');
  assert.deepEqual(draft.badge, { label: 'degraded', tone: 'warn' });
  assert.equal(draft.winner.steps.count, 8);
  assert.equal(draft.winner.memory.kvPlacement, 'vram', 'на draft KV влезает в VRAM ценой запаса');

  const standard = buildView(planFor({ q: 'standard' }));
  assert.equal(standard.verdict, 'ok');
  assert.equal(standard.winner.steps.count, 25);
  assert.equal(standard.winner.memory.kvPlacement, 'ram');
});

test('buildView: impossible 3840×2160 — без победителя и без YAML, но с подсказками', () => {
  const view = buildView(planFor({ w: 3840, h: 2160 }));
  assert.equal(view.verdict, 'impossible');
  assert.deepEqual(view.badge, { label: 'impossible', tone: 'bad' });
  assert.equal(view.winner, null);
  assert.equal(view.yaml, null);
  assert.equal(view.cli, null);
  assert.deepEqual(view.combinations, []);
  assert.deepEqual(view.memoryRows, []);
  assert.match(view.headline, /недостижим/u);
  assert.match(view.headline, /8\.\d\d Мп/u);
  assert.ok(view.suggestions.length >= 1, 'подсказки «что делать» обязательны');
  assert.ok(view.warnings.some((warning) => warning.code === 'no-native-path-for-size'));
});

test('buildView: 8 ГиБ VRAM — impossible с конкретным «что делать»', () => {
  const view = buildView(planFor({ vram: 8, ram: 16 }));
  assert.equal(view.verdict, 'impossible');
  assert.ok(view.suggestions.some((item) => item.code === 'smaller-canvas'));
  assert.ok(view.suggestions.some((item) => item.code === 'other-hardware'));
});

test('buildView: HDD и turbo — предупреждения о диске и фиксированные 8 шагов', () => {
  const view = buildView(planFor({ w: 1920, h: 1080, model: 'qwen21-turbo', disk: 'hdd' }));
  const codes = view.warnings.map((warning) => warning.code);
  assert.ok(codes.includes('hdd-fp8-load'), 'HDD: загрузка fp8-кэша');
  assert.ok(codes.includes('hdd-encoder-stream'), 'HDD: стриминг энкодера');
  assert.equal(view.winner.steps.count, 8);
  assert.ok(view.combinations.length >= 8);
});

test('buildView: h3-image — 6 шагов фиксировано, KV/VAE/окно не используются', () => {
  const view = buildView(planFor({ w: 1024, h: 1024, model: 'h3-image' }));
  assert.equal(view.verdict, 'ok');
  assert.equal(view.winner.steps.count, 6);
  assert.equal(view.winner.steps.fixed, true);
  assert.match(view.winner.steps.note, /фиксировано/u);
  assert.equal(view.winner.memory.kvMode, null);
  const kv = view.memoryRows.find((row) => row.key === 'kv');
  assert.match(kv.value, /не используется/u);
  const post = view.stages.find((stage) => stage.stage === 'postprocess');
  assert.equal(post.enabled, false);
  assert.equal(post.hasEstimates, false);
});

test('buildView: 16 бит — unsupported, YAML остаётся валидным, диск ×2', () => {
  const view = buildView(planFor({ bits: 16 }));
  assert.equal(view.verdict, 'unsupported');
  assert.deepEqual(view.badge, { label: 'unsupported', tone: 'warn' });
  assert.equal(view.winner.output.bitDepth, 16);
  assert.equal(view.winner.output.factor, 2);
  assert.match(view.winner.output.note, /16/u);
  assert.match(view.headline, /16 бит/u);
  assert.equal(typeof view.yaml, 'string', 'YAML не содержит несуществующих ключей');
  assert.equal(view.yaml.includes('bit_depth'), false);
});

test('buildView: snapped 1920×1080 → 1920×1088, дельта в карточке кадра', () => {
  const view = buildView(planFor({ w: 1920, h: 1080 }));
  assert.equal(view.winner.canvas.native, '1920×1088');
  assert.equal(view.winner.canvas.snap, 'высота +8 px');
  assert.equal(view.winner.canvas.exact, false);
  assert.ok(view.warnings.some((warning) => warning.code === 'native-size-snapped'));
});

test('buildView: KV VRAM⇄RAM и тайл 0 — переключаются запросом', () => {
  const ram = buildView(planFor());
  assert.equal(ram.memoryRows[0].placement, 'ram');

  const vram = buildView(planFor({}, { kvCache: true }));
  assert.equal(vram.memoryRows[0].placement, 'vram');
  assert.match(vram.memoryRows[0].value, /VRAM/u);
  assert.ok(vram.winner.estimates.vramPeak > ram.winner.estimates.vramPeak, 'KV в VRAM поднимает пик');

  const smallTile = buildView(planFor({}, { vaeTilePx: 0 }));
  const vae = smallTile.memoryRows.find((row) => row.key === 'vae');
  assert.equal(vae.value, '256 / 192');
  assert.match(vae.reason, /vae-tile-requested/u);

  const noCache = buildView(planFor({}, { kvCache: false }));
  assert.match(noCache.memoryRows[0].value, /без кеша/u);
});

test('buildView: окно внимания на 1792p и сетка референсов', () => {
  const view = buildView(planFor({ w: 3200, h: 1792 }));
  const window = view.memoryRows.find((row) => row.key === 'window');
  assert.match(window.value, /384 \+ 384/u);
  assert.match(window.value, /5 полос/u);
  const ref = view.memoryRows.find((row) => row.key === 'ref');
  assert.match(ref.value, /960 px/u);
  assert.ok(view.combinations.some((row) => row.attention === 'window'));
});

test('buildView: trace и stages переведены в строки для вёрстки', () => {
  const view = buildView(planFor());
  assert.ok(view.trace.length > 0);
  const canvas = view.trace[0];
  assert.equal(canvas.field, 'image.canvas');
  assert.equal(canvas.value, '2560×1440');
  assert.ok(canvas.rejected.length >= 2);
  assert.ok(canvas.rejected.every((item) => typeof item.value === 'string' && typeof item.why === 'string'));
  assert.ok(view.trace.some((row) => row.field === 'render.steps'));
  assert.equal(view.stages[0].stage, 'render');
  assert.equal(view.stages[0].enabled, true);
});

test('buildView: сериализуема и детерминирована', () => {
  const first = buildView(planFor());
  const second = buildView(planFor());
  assert.deepEqual(first, second, 'два вызова на одинаковом запросе дают одинаковую вью-модель');
  assert.deepEqual(JSON.parse(JSON.stringify(first)), first, 'во вью-модели нет функций/undefined');
});

test('buildView: scan прикладывается отдельно (по умолчанию null)', () => {
  assert.equal(buildView(planFor()).scan, null);
  const scan = scanYaml(planFor().yaml);
  assert.equal(buildView(planFor(), { scan }).scan.ok, true);
});

test('scanYaml: свой YAML проходит, неизвестный ключ и мусор — нет', () => {
  const plan = planFor();
  const ok = scanYaml(plan.yaml);
  assert.equal(ok.ok, true);
  assert.equal(ok.problem, null);
  assert.ok(ok.keys.length > 20);
  assert.ok(ok.keys.includes('generation.steps'));
  assert.ok(ok.keys.includes('generation.sampler_kwargs.vae_tile_px'));
  assert.equal(ok.values['generation.steps'], 25);
  assert.equal(ok.values['upscale'], null);
  assert.match(ok.authority, /config show/u);

  const unknown = scanYaml('size_px: 1440\ngeneration:\n  steps: 25\n  bogus_key: 1\n');
  assert.equal(unknown.ok, false);
  assert.deepEqual(unknown.unknown, ['generation.bogus_key']);
  assert.match(unknown.problem, /bogus_key/u);

  const duplicate = scanYaml(`${plan.yaml}\ngeneration:\n  steps: 8\n`);
  assert.equal(duplicate.ok, false, 'дублирующийся ключ — ошибка разбора, а не «последний выиграл»');
  assert.match(duplicate.problem, /не разобрано/u);

  const broken = scanYaml('generation:\n  steps: [1\n');
  assert.equal(broken.ok, false);
  assert.match(broken.problem, /не разобрано/u);

  const empty = scanYaml('   \n');
  assert.equal(empty.ok, false);
  assert.match(empty.problem, /пустой/u);
});

test('hintLive: живые дельты в контексте плана', () => {
  assert.match(hintLive('steps', BASE_QUERY) ?? '', /25 → 8 шагов/u);
  assert.match(hintLive('kvCache', BASE_QUERY) ?? '', /KV RAM → VRAM/u);
  assert.match(hintLive('vaeTilePx', BASE_QUERY) ?? '', /тайл 512 → 1024/u);
  assert.equal(hintLive('vramGib', BASE_QUERY), null);
});
