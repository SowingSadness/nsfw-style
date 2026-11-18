// @ts-check
/**
 * Тесты инвентаря (шаг 2 плана): сгенерированные данные совпадают с Python, а API
 * (`src/model/inventory.mjs`) отдаёт то, на что опираются cost/candidates/emit.
 *
 * Первый тест запускает `node tools/gen-inventory.mjs --check` — это паритет с `qwenstyle`
 * (правка `presets/*.yaml`, `backends/*.py` или `profiles.py` без перегенерации → падение).
 * Нет Python/venv (`exit 2`) — тест печатает `SKIP` и проходит: конвенция репозитория
 * (без CUDA/Python скрипты печатают `SKIP`, а не падают).
 */

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { INVENTORY } from '../src/data/inventory.mjs';
import {
  ENCODER_IDS,
  MODEL_IDS,
  TIER_NAMES,
  diskFootprint,
  downloadGib,
  encoderInfo,
  inventoryDigest,
  limits,
  modelCaps,
  models,
  paramSpecs,
  settableKeys,
  tierFor,
} from '../src/model/inventory.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = join(HERE, '..');
const SRC_ROOT = join(PACKAGE_ROOT, 'src');
const GENERATOR = join(PACKAGE_ROOT, 'tools', 'gen-inventory.mjs');

/**
 * Рекурсивный список файлов с заданным расширением (для grep-проверки `src/**`).
 *
 * @param {string} dir
 * @param {string} extension
 * @returns {string[]}
 */
function listFiles(dir, extension) {
  /** @type {string[]} */
  const result = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) result.push(...listFiles(full, extension));
    else if (entry.endsWith(extension)) result.push(full);
  }
  return result;
}

test('паритет: `gen-inventory.mjs --check` совпадает с Python, повтор идемпотентен', (context) => {
  const run = () => spawnSync(process.execPath, [GENERATOR, '--check'], {
    cwd: PACKAGE_ROOT,
    encoding: 'utf8',
    windowsHide: true,
  });

  const first = run();
  if (first.status === 2) {
    context.skip(`SKIP: ${(first.stdout || first.stderr || 'нет Python/venv').trim()}`);
    return;
  }
  assert.equal(first.status, 0, `--check упал:\n${first.stdout}\n${first.stderr}`);

  const second = run();
  assert.equal(second.status, 0, `повторный --check не идемпотентен:\n${second.stdout}\n${second.stderr}`);
  assert.match(second.stdout, /совпадает/u);
});

test('models: 3 модели, backend/repo/шаги/веса/энкодер', () => {
  assert.deepEqual([...MODEL_IDS], ['qwen21-base', 'qwen21-turbo', 'h3-image']);
  assert.equal(models().length, 3);

  const base = INVENTORY.models['qwen21-base'];
  assert.equal(base.backend, 'qwen_image21');
  assert.equal(base.repo, 'Qwen/Qwen-Image-2.1');
  assert.deepEqual(base.transformer, ['Qwen/Qwen-Image-2.1', 'transformer']);
  assert.equal(base.precision, 'fp8');
  assert.equal(base.precision_cache, 'cache/models/fp8_transformer');
  assert.equal(base.fixedSteps, 25);
  assert.equal(base.encoderId, 'heretic');
  assert.equal(base.lora?.scale, 0.7);
  assert.equal(base.lora?.strip_alpha, false);

  const turbo = INVENTORY.models['qwen21-turbo'];
  assert.deepEqual(turbo.transformer, ['Qwen/Qwen-Image-2.1-Turbo', 'transformer']);
  assert.equal(turbo.fixedSteps, 8);
  assert.equal(turbo.precision_cache, 'cache/models/fp8_transformer_turbo');
  assert.equal(turbo.lora?.scale, null, 'турбо-адаптер объявлен, но ни один пресет его не включает');

  const h3 = INVENTORY.models['h3-image'];
  assert.equal(h3.backend, 'minimax_h3_image');
  assert.equal(h3.repo, 'MiniMaxAI/MiniMax-H3-Image');
  assert.equal(h3.transformer, null);
  assert.equal(h3.fixedSteps, 6);
  assert.equal(h3.weightGibBf16, 61.7);
  assert.equal(h3.weightGibFp8, 30.9);
  assert.equal(h3.encoderId, 'h3qwen32-heretic');
  assert.equal(h3.lora, null, 'FUSAL к H3 неприменим (caps.supports_lora=false)');
});

test('encoders: 3 декларации, поля берутся из бэкенда, рантайм — из конфигов', () => {
  assert.ok(ENCODER_IDS.length >= 2, 'ожидается минимум два энкодера');
  assert.deepEqual([...ENCODER_IDS], ['base', 'h3qwen32-heretic', 'heretic']);

  const heretic = encoderInfo('heretic');
  assert.equal(heretic.cache_tag, 'heretic');
  assert.equal(heretic.repo, 'DreamFast/Qwen3-VL-8B-Heretic-1.3.0');
  assert.match(String(heretic.revision), /^[0-9a-f]{40}$/u);
  assert.equal(heretic.local_dir, 'cache/models/weights/heretic-te');
  assert.equal(heretic.requires_vision, true);
  assert.equal(heretic.dtype, 'bfloat16');
  assert.equal(heretic.stream, true);
  assert.equal(heretic.prefetch, 4);
  assert.equal(heretic.pin_memory, true);
  assert.equal(heretic.lookahead, 1);

  const h3 = encoderInfo('h3qwen32-heretic');
  assert.equal(h3.local_dir, 'cache/models/weights/h3qwen32-heretic');
  assert.equal(h3.memory_gib, 62.1);
  assert.equal(encoderInfo('base').cache_tag, 'base');

  assert.throws(() => encoderInfo('nope'), RangeError);

  // Каждая модель ссылается на существующий энкодер.
  for (const model of models()) assert.ok(encoderInfo(model.encoderId));
});

test('limits: maxImages, фиксированные шаги, точности/offload, пропорции', () => {
  const caps = limits();
  assert.equal(caps.maxImages, 10);
  assert.deepEqual(caps.fixedSteps, { 'h3-image': 6 });
  assert.deepEqual([...caps.precisions], ['bf16', 'fp8']);
  assert.deepEqual([...caps.offloadModes], ['none', 'group']);
  assert.deepEqual([...caps.aspectRatios], ['16:9', '4:3', '1:1', '2:3']);
  assert.deepEqual(caps, INVENTORY.limits);
});

test('tiers: следы на диске и загрузка из profiles.py', () => {
  assert.deepEqual([...TIER_NAMES], ['VRAM8', 'VRAM12', 'VRAM16']);
  assert.equal(diskFootprint('VRAM8'), 74.9);
  assert.equal(diskFootprint('VRAM12'), 54.9);
  assert.equal(diskFootprint('VRAM16'), 181);
  assert.equal(downloadGib('VRAM8'), 61.5);
  assert.equal(downloadGib('VRAM12'), 48.2);
  assert.equal(downloadGib('VRAM16'), 174.3);
  assert.equal(INVENTORY.tiers.VRAM8.targetHeight, 1080);
  assert.equal(INVENTORY.tiers.VRAM12.targetHeight, 960);
  assert.equal(INVENTORY.tiers.VRAM16.targetHeight, 1440);
  assert.equal(downloadGib('vram12'), 48.2, 'имя тарифа нечувствительно к регистру');

  assert.throws(() => diskFootprint('VRAM24'), RangeError);
  assert.throws(() => downloadGib('nope'), RangeError);
});

test('tierFor: ближайший тариф по VRAM (инфо-поле, не выбор)', () => {
  assert.equal(tierFor(8), 'VRAM8');
  assert.equal(tierFor(12), 'VRAM12');
  assert.equal(tierFor(16), 'VRAM16');
  assert.equal(tierFor(11), 'VRAM12');
  assert.equal(tierFor(4), 'VRAM8');
  assert.equal(tierFor(64), 'VRAM16');
  assert.equal(tierFor(10), 'VRAM8', 'при равном расстоянии — консервативнее (меньше VRAM)');
  assert.throws(() => tierFor(Number.NaN), TypeError);
});

test('modelCaps: лимиты из limits, веса и адаптер из модели', () => {
  const base = modelCaps('qwen21-base');
  assert.equal(base.fixedSteps, 25);
  assert.equal(base.maxImages, 10);
  assert.deepEqual([...base.precisions], ['bf16', 'fp8']);
  assert.deepEqual([...base.offloadModes], ['none', 'group']);
  assert.equal(base.weightGibBf16, 14.2);
  assert.equal(base.weightGibFp8, 6.6);
  assert.equal(base.encoder, 'heretic');
  assert.equal(base.lora?.filename, 'fusal-qwen-image-2.1-v2-exact-r80.safetensors');

  const h3 = modelCaps('h3-image');
  assert.equal(h3.fixedSteps, 6);
  assert.equal(h3.encoder, 'h3qwen32-heretic');
  assert.equal(h3.lora, null);
  assert.equal(h3.weightGibFp8, 30.9);

  assert.throws(() => modelCaps('nope'), RangeError);
});

test('settable/params: whitelist `--set` и полные дескрипторы ParameterSpec', () => {
  const keys = settableKeys('qwen_image21');
  assert.ok(keys.includes('generation.steps'));
  assert.ok(keys.includes('generation.kv_cache'));
  assert.ok(keys.includes('generation.sampler_kwargs.vae_tile_px'));
  assert.ok(keys.includes('generation.sampler_kwargs.ref_px'));
  assert.ok(!keys.includes('generation.max_images'), 'readonly-ключи в whitelist не попадают');
  assert.ok(settableKeys('esrgan').includes('upscale.tile'));
  assert.ok(!settableKeys('minimax_h3_image').includes('generation.steps'), '6 шагов H3 фиксированы');

  const specs = paramSpecs('qwen_image21');
  const steps = specs.find((spec) => spec.path === 'generation.steps');
  assert.ok(steps);
  assert.equal(steps.type, 'int');
  assert.equal(steps.default, 25);
  assert.equal(steps.readonly, false);
  const maxImages = specs.find((spec) => spec.path === 'generation.max_images');
  assert.ok(maxImages);
  assert.equal(maxImages.readonly, true);
  assert.equal(maxImages.default, 10);
  assert.ok(paramSpecs('minimax_h3_image').some((spec) => spec.readonly && spec.path === 'generation.steps'));

  assert.throws(() => settableKeys('nope'), RangeError);
  assert.throws(() => paramSpecs('nope'), RangeError);
  // Копия, а не сам массив данных: правка результата не портит INVENTORY.
  keys.push('generation.mutated');
  assert.ok(!INVENTORY.settable.qwen_image21.includes('generation.mutated'));
});

test('данные сгенерированы: шапка «do not edit» и дайджест входов', () => {
  const text = readFileSync(join(SRC_ROOT, 'data', 'inventory.mjs'), 'utf8');
  assert.match(text, /^\/\/ generated by tools\/gen-inventory\.mjs — do not edit/u);
  assert.match(text, /export const INVENTORY = Object\.freeze\(/u);
  const digest = inventoryDigest();
  assert.match(digest.configs, /^[0-9a-f]{12}$/u);
  assert.match(digest.profiles, /^[0-9a-f]{12}$/u);
  assert.ok(INVENTORY.source.presets >= 15, 'должны быть как минимум 15 публичных пресетов');
});

test('в src/** нет имён пресетов: только модели/бэкенды/энкодеры/тарифы', () => {
  // Все 15 публичных имён начинаются с `<тариф>-`, поэтому проверки по префиксу достаточно.
  // Подстроку `-preview`/`-release` целиком запретить нельзя: имя LoRA-артефакта
  // `fusal-…-turbo-e6-preview-r128.safetensors` — это файл модели, а не пресет; поэтому
  // ловятся именно пресетные токены (`preview`/`release` как значение, а не внутри имени файла).
  const forbidden = [
    /VRAM8-|VRAM12-|VRAM16-/u,
    /["'`](?:preview|release)(?:-anime)?["'`]/u,
  ];
  /** @type {string[]} */
  const violations = [];
  for (const file of listFiles(SRC_ROOT, '.mjs')) {
    const text = readFileSync(file, 'utf8');
    const name = relative(PACKAGE_ROOT, file).replaceAll('\\', '/');
    for (const [index, line] of text.split('\n').entries()) {
      for (const pattern of forbidden) {
        if (pattern.test(line)) violations.push(`${name}:${index + 1}: ${line.trim()}`);
      }
    }
  }
  assert.deepEqual(violations, []);
});
