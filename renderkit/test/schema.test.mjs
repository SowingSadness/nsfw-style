// @ts-check
/**
 * Тесты схемы входа (шаг 1 плана): 12 негативных кейсов с точными `code`/`field`,
 * дефолты и форма `PlanError`.
 *
 * Каждый негативный кейс проверяет и код, и путь поля: по ним ветвятся UI и сервис,
 * поэтому «просто упало» здесь недостаточно.
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { PlanError, SCHEMA_VERSION, normalizeRequest } from '../src/schema.mjs';
import { RULESET_VERSION, computePlan, listOptions, normalizeRequest as normalizeFromIndex } from '../src/index.mjs';

/** Валидный запрос-основа: каждый негативный кейс портит ровно одно поле. */
function validRequest() {
  return {
    hardware: { model: 'qwen21-base', vramGib: 16, ramGib: 64, disk: 'nvme' },
    image: { width: 2560, height: 1440, quality: 'max' },
  };
}

/**
 * 12 негативных кейсов: `[описание, мутация запроса, code, field]`.
 *
 * @type {Array<[string, (request: any) => void, string, string]>}
 */
const NEGATIVE_CASES = [
  ['нет hardware', (r) => { delete r.hardware; }, 'MissingField', 'hardware'],
  ['ширина ниже минимума', (r) => { r.image.width = 100; }, 'OutOfRange', 'image.width'],
  ['ширина не целая', (r) => { r.image.width = 1920.5; }, 'NotInteger', 'image.width'],
  ['высота строкой', (r) => { r.image.height = '1080'; }, 'NotInteger', 'image.height'],
  ['уровня качества high больше нет', (r) => { r.image.quality = 'high'; }, 'InvalidEnum', 'image.quality'],
  ['битность 12 не существует', (r) => { r.image.bitDepth = 12; }, 'InvalidEnum', 'image.bitDepth'],
  ['слишком мало шагов', (r) => { r.image.steps = 2; }, 'OutOfRange', 'image.steps'],
  ['неизвестная модель', (r) => { r.hardware.model = 'flux2'; }, 'InvalidEnum', 'hardware.model'],
  ['VRAM ниже минимума', (r) => { r.hardware.vramGib = 2; }, 'OutOfRange', 'hardware.vramGib'],
  ['RAM строкой', (r) => { r.hardware.ramGib = '64'; }, 'NotNumber', 'hardware.ramGib'],
  ['неизвестный класс диска', (r) => { r.hardware.disk = 'floppy'; }, 'InvalidEnum', 'hardware.disk'],
  // Верхняя граница совпадает с `Capabilities.maxImages = 10` бэкенда (иначе гейт
  // `refs-exceed-model` в resolve.mjs был бы недостижим).
  ['референсов больше десяти', (r) => { r.refs = 11; }, 'OutOfRange', 'refs'],
];

test('12 негативных кейсов: точные code и field', () => {
  assert.equal(NEGATIVE_CASES.length, 12);
  for (const [label, mutate, code, field] of NEGATIVE_CASES) {
    const request = validRequest();
    mutate(request);
    assert.throws(
      () => normalizeRequest(request),
      (error) => {
        assert.ok(error instanceof PlanError, `${label}: ожидался PlanError, получено ${String(error)}`);
        assert.equal(error.code, code, `${label}: code`);
        assert.equal(error.field, field, `${label}: field`);
        return true;
      },
      label,
    );
  }
});

test('дополнительные отказы схемы', () => {
  const cases = [
    ['нет image', (r) => { delete r.image; }, 'MissingField', 'image'],
    ['anime строкой', (r) => { r.anime = 'yes'; }, 'NotBoolean', 'anime'],
    ['frames не принимается', (r) => { r.frames = 4; }, 'UnknownField', 'frames'],
    ['неизвестное поле в image', (r) => { r.image.upscale = 4; }, 'UnknownField', 'image.upscale'],
    ['неизвестный режим KV', (r) => { r.memory = { kvCache: 'maybe' }; }, 'InvalidEnum', 'memory.kvCache'],
    ['неизвестный тайл VAE', (r) => { r.memory = { vaeTilePx: 100 }; }, 'InvalidEnum', 'memory.vaeTilePx'],
    ['окно не того типа', (r) => { r.memory = { window: 'yes' }; }, 'InvalidType', 'memory.window'],
    ['окно нулевой ширины', (r) => { r.memory = { window: 0 }; }, 'OutOfRange', 'memory.window'],
    ['свободный диск строкой', (r) => { r.freeDiskGib = 'много'; }, 'NotNumber', 'freeDiskGib'],
    ['postprocess без модели', (r) => { r.postprocess = { params: {} }; }, 'MissingField', 'postprocess.model'],
    ['неизвестное поле в postprocess', (r) => { r.postprocess = { model: 'seedvr2-3b', upscale: 2 }; }, 'UnknownField', 'postprocess.upscale'],
    ['параметры постобработки не объект', (r) => { r.postprocess = { model: 'seedvr2-3b', params: 5 }; }, 'NotObject', 'postprocess.params'],
    ['postprocess строкой', (r) => { r.postprocess = 'seedvr2-3b'; }, 'NotObject', 'postprocess'],
    ['запрос не объект', null, 'NotObject', ''],
  ];
  for (const [label, mutate, code, field] of cases) {
    assert.throws(
      () => normalizeRequest(mutate === null ? null : (() => { const r = validRequest(); mutate(r); return r; })()),
      (error) => {
        assert.ok(error instanceof PlanError, `${label}: ожидался PlanError`);
        assert.equal(error.code, code, `${label}: code`);
        assert.equal(error.field, field, `${label}: field`);
        return true;
      },
      label,
    );
  }
});

test('дефолты: bitDepth 8, refs 1, anime false, memory/postprocess auto', () => {  const normalized = normalizeRequest(validRequest());
  assert.equal(normalized.image.bitDepth, 8);
  assert.equal(normalized.image.steps, null);
  assert.equal(normalized.refs, 1);
  assert.equal(normalized.anime, false);
  assert.equal(normalized.freeDiskGib, null);
  assert.deepEqual(normalized.memory, { kvCache: null, vaeTilePx: null, window: null });
  assert.equal(normalized.postprocess, null);
  assert.deepEqual(normalized.hardware, { model: 'qwen21-base', vramGib: 16, ramGib: 64, disk: 'nvme' });

  const explicit = normalizeRequest({
    ...validRequest(),
    image: { width: 1920, height: 1080, quality: 'draft', bitDepth: 16, steps: 12 },
    refs: 0,
    anime: true,
    freeDiskGib: 120,
    memory: { kvCache: 'offload', vaeTilePx: 0, window: { windowPx: 384, marginPx: 384 } },
  });
  assert.equal(explicit.image.bitDepth, 16);
  assert.equal(explicit.image.steps, 12);
  assert.equal(explicit.refs, 0);
  assert.equal(explicit.anime, true);
  assert.equal(explicit.freeDiskGib, 120);
  assert.deepEqual(explicit.memory, {
    kvCache: 'offload',
    vaeTilePx: 0,
    window: { enabled: true, windowPx: 384, marginPx: 384 },
  });
});

test('postprocess: запрос принимается и нормализуется (модель приносит свои параметры)', () => {
  const base = validRequest();
  // Модель без параметров — только умолчания модели.
  assert.deepEqual(normalizeRequest({ ...base, postprocess: { model: 'seedvr2-3b' } }).postprocess,
    { model: 'seedvr2-3b', params: null });
  // С параметрами — как пришли (значения проверит набор самой модели, а не схема ядра).
  assert.deepEqual(normalizeRequest({ ...base, postprocess: { model: 'seedvr2-3b', params: { tile: 256 } } }).postprocess,
    { model: 'seedvr2-3b', params: { tile: 256 } });
  assert.equal(normalizeRequest({ ...base, postprocess: null }).postprocess, null);
  // Вложенные подобъекты не переиспользуются (запрос не мутирует вызывающего).
  const params = { tile: 256 };
  const normalized = normalizeRequest({ ...base, postprocess: { model: 'x', params } });
  params.tile = 512;
  assert.deepEqual(normalized.postprocess?.params, { tile: 256 });
});

test('memory.window нормализуется из boolean/числа/объекта', () => {
  const withWindow = (/** @type {unknown} */ window) =>
    normalizeRequest({ ...validRequest(), memory: { window } }).memory.window;

  assert.deepEqual(withWindow(true), { enabled: true, windowPx: null, marginPx: null });
  assert.deepEqual(withWindow(false), { enabled: false, windowPx: null, marginPx: null });
  assert.deepEqual(withWindow(384), { enabled: true, windowPx: 384, marginPx: null });
  assert.deepEqual(withWindow({ marginPx: 448 }), { enabled: true, windowPx: null, marginPx: 448 });
  assert.equal(withWindow(null), null);
});

test('PlanError машиночитаема: code/field/allowed и toJSON', () => {
  assert.throws(
    () => normalizeRequest({ ...validRequest(), image: { width: 1920, height: 1080, quality: 'high' } }),
    (error) => {
      assert.ok(error instanceof PlanError);
      assert.equal(error.name, 'PlanError');
      assert.equal(error.code, 'InvalidEnum');
      assert.equal(error.field, 'image.quality');
      assert.deepEqual(error.allowed, ['draft', 'standard', 'max']);
      assert.match(error.message, /^image\.quality: InvalidEnum:/);
      const json = error.toJSON();
      assert.equal(json.code, 'InvalidEnum');
      assert.equal(json.field, 'image.quality');
      assert.deepEqual(json.allowed, ['draft', 'standard', 'max']);
      return true;
    },
  );
});

test('уровни качества — ровно draft/standard/max', () => {
  assert.equal(SCHEMA_VERSION, 1);
  for (const quality of ['draft', 'standard', 'max']) {
    const normalized = normalizeRequest({ ...validRequest(), image: { width: 1920, height: 1080, quality } });
    assert.equal(normalized.image.quality, quality);
  }
});

test('публичный вход: RULESET_VERSION, listOptions, computePlan собирает Plan', () => {
  // Версия правил = база + дайджесты входных данных генератора (`inventory.mjs`) + ревизия эмиттера.
  assert.match(RULESET_VERSION, /^r1\+[0-9a-f]{12}-[0-9a-f]{12}\+emit1$/u);
  assert.equal(normalizeFromIndex, normalizeRequest);

  const options = listOptions();
  assert.equal(options.rulesetVersion, RULESET_VERSION);
  assert.equal(options.schemaVersion, SCHEMA_VERSION);
  assert.deepEqual(options.models.map((model) => model.value), ['qwen21-base', 'qwen21-turbo', 'h3-image']);
  assert.deepEqual(options.qualities.map((quality) => quality.value), ['draft', 'standard', 'max']);
  assert.deepEqual(options.bitDepths.map((bits) => bits.value), [8, 16]);
  assert.deepEqual(options.disks.map((disk) => disk.value), ['hdd', 'ssd', 'nvme']);
  assert.deepEqual(options.aspects.map((aspect) => aspect.value), ['16:9', '4:3', '1:1', '2:3']);
  assert.ok(options.memory.kvCache.length > 0 && options.memory.vaeTilePx.length > 0);
  // Имён пресетов в справочнике нет: только значения и подписи.
  assert.doesNotMatch(JSON.stringify(options), /VRAM(?:8|12|16)|-preview|-release/u);

  const plan = computePlan(validRequest());
  assert.equal(plan.schemaVersion, SCHEMA_VERSION);
  assert.equal(plan.rulesetVersion, RULESET_VERSION);
  assert.equal(plan.verdict, 'ok');
  assert.equal(typeof plan.yaml, 'string', 'YAML-эмиттер подключён к computePlan');
  assert.equal(plan.stages.at(-1).stage, 'postprocess');
});
