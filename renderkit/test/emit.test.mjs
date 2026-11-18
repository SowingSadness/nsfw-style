// @ts-check
/**
 * YAML-артефакт (шаг 6, ADR A2): состав файла, отсутствие имён пресетов, whitelist ключей,
 * окно внимания, 16 бит и команды запуска.
 *
 * Проверяется то, что нельзя увидеть глазами в одном примере:
 * - в YAML **нет** `extends` и имён пресетов — ни в ключах, ни в шапке, ни в имени файла;
 * - все обязательные ключи на месте (иначе файл несамодостаточен);
 * - **ни одного ключа вне whitelist** (`inventory.settable` + поля `RunConfig` + объявленные
 *   поля конфига), и при этом `sampler_kwargs` не содержит того, чего бэкенд не объявляет
 *   (h3 без `vae_tile_px`/`ref_px`/окна);
 * - `upscale: null` и `logs: true` всегда; `window_attn_px`/`margin` — только в оконном режиме;
 * - 16-битный вариант валиден (без несуществующих ключей) и отличается только шапкой;
 * - `--set`-форма (`cli.sets`) согласована со значениями YAML и целиком лежит в whitelist;
 * - `computePlan` сам отдаёт готовый артефакт (`Plan.yaml`/`Plan.cli`), а у `impossible` — `null`
 *   без исключения.
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { computePlan } from '../src/index.mjs';
import { PlanError } from '../src/schema.mjs';
import {
  emitPreset,
  fileNameFor,
  headerLines,
  kvCacheValue,
  presetFor,
} from '../src/model/emit.mjs';
import {
  CORE_SET_KEYS,
  PLAN_FILE,
  allowedSetKeys,
  assertSetKey,
  commandsFor,
  setArgs,
  setsFor,
  validateCount,
} from '../src/model/cli.mjs';
import { INVENTORY, settableKeys } from '../src/model/inventory.mjs';
import { YamlRaw, parseYaml } from '../src/yaml.mjs';

/**
 * Запрос плана (те же умолчания, что у `test/resolve.test.mjs`).
 *
 * @param {{width?: number, height?: number, quality?: 'draft'|'standard'|'max', bitDepth?: 8|16,
 *   vramGib?: number, ramGib?: number, disk?: string, model?: string, refs?: number,
 *   anime?: boolean}} [options]
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
  };
}

/** @param {Record<string, any>} [options] */
const planFor = (options = {}) => computePlan(request(options));
/** @param {Record<string, any>} [options] */
const emitFor = (options = {}) => emitPreset(planFor(options));

/** Сценарии, на которых держатся проверки (пары «запрос → ожидаемый режим»). */
const SCENARIOS = Object.freeze({
  base1440: { width: 2560, height: 1440, quality: 'max', model: 'qwen21-base' },
  window1792: { width: 3200, height: 1792, quality: 'max', model: 'qwen21-base' },
  draftTurbo: { width: 1376, height: 768, quality: 'draft', model: 'qwen21-turbo' },
  h3: { width: 1024, height: 1024, quality: 'max', model: 'h3-image' },
});

/**
 * Все листовые пути объекта (`a.b.c`; массивы и `null` — листья).
 *
 * @param {unknown} value
 * @param {string} [prefix]
 * @returns {string[]}
 */
function leafPaths(value, prefix = '') {
  if (Array.isArray(value) || value === null || typeof value !== 'object') return [prefix];
  /** @type {string[]} */
  const paths = [];
  for (const [key, item] of Object.entries(/** @type {Record<string, unknown>} */ (value))) {
    paths.push(...leafPaths(item, prefix ? `${prefix}.${key}` : key));
  }
  return paths;
}

/**
 * Значение по пути `a.b.c` (для сверки `--set` с YAML).
 *
 * @param {any} value
 * @param {string} path
 * @returns {unknown}
 */
function getPath(value, path) {
  return path.split('.').reduce((node, key) => (node === null || node === undefined ? undefined : node[key]), value);
}

/**
 * Ключи, которые эмиттер имеет право писать: объявленные бэкендом `--set`-пути
 * (`inventory.settable`, то есть `ParameterSpec`) плюс поля `RunConfig` из ADR A2.
 *
 * @param {string} model
 * @returns {Set<string>}
 */
function allowedPaths(model) {
  const backend = INVENTORY.models[model].backend;
  const allowed = new Set([
    'size_px',
    'aspect',
    'generation.backend',
    'generation.repo',
    'generation.transformer',
    'generation.lora',
    'generation.lora.repo',
    'generation.lora.filename',
    'generation.lora.scale',
    'generation.lora.strip_alpha',
    'generation.steps',
    'generation.guidance',
    'generation.precision',
    'generation.precision_cache',
    'generation.offload',
    'encoder.name',
    'encoder.cache_tag',
    'encoder.repo',
    'encoder.revision',
    'encoder.local_dir',
    'encoder.requires_vision',
    'encoder.dtype',
    'encoder.device',
    'encoder.stream',
    'encoder.prefetch',
    'encoder.pin_memory',
    'encoder.lookahead',
    'upscale',
    'logs',
  ]);
  for (const key of settableKeys(backend)) allowed.add(key);
  return allowed;
}

/**
 * Значения `YamlRaw` заменяются на числа — иначе сравнение с разобранным YAML бессмысленно.
 *
 * @param {unknown} value
 * @returns {unknown}
 */
function plain(value) {
  if (value instanceof YamlRaw) return Number(value.text);
  if (Array.isArray(value)) return value.map(plain);
  if (value !== null && typeof value === 'object') {
    /** @type {Record<string, unknown>} */
    const out = {};
    for (const [key, item] of Object.entries(/** @type {Record<string, any>} */ (value))) out[key] = plain(item);
    return out;
  }
  return value;
}

test('YAML: без `extends` и без имён пресетов — в тексте, шапке и имени файла', () => {
  for (const scenario of Object.values(SCENARIOS)) {
    const emitted = emitFor(scenario);
    const combined = `${emitted.yaml}\n${emitted.fileName}`;
    assert.ok(!/extends/u.test(combined), `${scenario.model}: в файле не должно быть extends`);
    assert.ok(!/\bpreset:/u.test(emitted.yaml), `${scenario.model}: в файле не должно быть ключа preset`);
    assert.ok(
      !/VRAM(?:8|12|16)-|-(?:preview|release|release-anime|preview-anime)\b/u.test(combined),
      `${scenario.model}: имена пресетов просочились в артефакт:\n${emitted.yaml}`,
    );
    // Шапка описывает запрос, но это не имя пресета (ADR A2).
    const first = emitted.yaml.split('\n')[0];
    assert.match(first, /^# renderkit r1\+[0-9a-f]{12}-[0-9a-f]{12}\+emit1 \| /u);
    assert.match(first, /GiB VRAM \/ .* GiB RAM \/ (?:hdd|ssd|nvme) \| /u);
    assert.match(first, /quality=(?:draft|standard|max) \| (?:8|16) bit$/u);
  }
});

test('YAML: все обязательные ключи на месте, `upscale: null`, `logs: true`', () => {
  const emitted = emitFor(SCENARIOS.base1440);
  const parsed = parseYaml(emitted.yaml);
  const required = [
    'size_px',
    'aspect',
    'generation.backend',
    'generation.repo',
    'generation.transformer',
    'generation.lora',
    'generation.steps',
    'generation.guidance',
    'generation.precision',
    'generation.precision_cache',
    'generation.offload',
    'generation.kv_cache',
    'generation.sampler_kwargs.offload_blocks',
    'generation.sampler_kwargs.offload_stream',
    'generation.sampler_kwargs.vae_tile_px',
    'generation.sampler_kwargs.ref_px',
    'encoder.name',
    'encoder.cache_tag',
    'encoder.repo',
    'encoder.revision',
    'encoder.local_dir',
    'encoder.requires_vision',
    'encoder.dtype',
    'encoder.device',
    'encoder.stream',
    'encoder.prefetch',
    'encoder.pin_memory',
    'encoder.lookahead',
    'upscale',
    'logs',
  ];
  for (const path of required) {
    assert.notEqual(getPath(parsed, path), undefined, `нет обязательного ключа ${path}`);
  }
  assert.equal(parsed.upscale, null);
  assert.equal(parsed.logs, true);
  assert.equal(parsed.size_px, 1440);
  assert.equal(parsed.aspect, '16:9');
  assert.equal(parsed.generation.steps, 25);
  assert.equal(parsed.generation.offload, 'group');
  assert.equal(parsed.generation.kv_cache, 'offload');
  assert.equal(parsed.generation.sampler_kwargs.offload_blocks, 1);
  assert.equal(parsed.generation.sampler_kwargs.vae_tile_px, 512);
  assert.equal(parsed.generation.sampler_kwargs.ref_px, 960);
  assert.deepEqual(parsed.generation.transformer, ['Qwen/Qwen-Image-2.1', 'transformer']);
  assert.equal(parsed.encoder.name, 'heretic');
  assert.equal(parsed.encoder.cache_tag, 'heretic');
  assert.match(String(parsed.encoder.revision), /^[0-9a-f]{40}$/u);
  // `1.0` в файле — именно float-запись (YamlRaw), а не `1`.
  assert.match(emitted.yaml, /^ {2}guidance: 1\.0$/mu);
});

test('YAML: все ключи из whitelist, `sampler_kwargs` — только объявленные бэкендом', () => {
  for (const [name, scenario] of Object.entries(SCENARIOS)) {
    const emitted = emitFor(scenario);
    const parsed = parseYaml(emitted.yaml);
    const allowed = allowedPaths(scenario.model);
    const backend = INVENTORY.models[scenario.model].backend;
    const settable = new Set(settableKeys(backend));
    for (const path of leafPaths(parsed)) {
      assert.ok(allowed.has(path), `${name}: ключ ${path} вне whitelist`);
      if (path.startsWith('generation.sampler_kwargs.')) {
        assert.ok(settable.has(path), `${name}: ${path} не объявлен бэкендом ${backend}`);
      }
    }
  }
});

test('окно внимания: ключи есть только в оконном режиме', () => {
  const windowed = emitFor(SCENARIOS.window1792);
  const parsedWindowed = parseYaml(windowed.yaml);
  assert.equal(parsedWindowed.size_px, 1792);
  assert.equal(parsedWindowed.generation.sampler_kwargs.window_attn_px, 384);
  assert.equal(parsedWindowed.generation.sampler_kwargs.window_attn_margin_px, 384);

  const dense = emitFor(SCENARIOS.base1440);
  const denseText = dense.yaml;
  assert.ok(!denseText.includes('window_attn_px'), 'плотное внимание не должно писать окно');
  assert.ok(!denseText.includes('window_attn_margin_px'), 'плотное внимание не должно писать запас окна');
});

test('h3: без kv_cache/vae_tile_px/ref_px/окна, `transformer: null`, энкодер 32B', () => {
  const emitted = emitFor(SCENARIOS.h3);
  const parsed = parseYaml(emitted.yaml);
  assert.equal(parsed.generation.backend, 'minimax_h3_image');
  assert.equal(parsed.generation.transformer, null);
  assert.equal(parsed.generation.steps, 6);
  assert.equal(parsed.encoder.name, 'h3qwen32-heretic');
  assert.equal(parsed.encoder.local_dir, 'cache/models/weights/h3qwen32-heretic');
  for (const forbidden of [
    'kv_cache',
    'vae_tile_px',
    'ref_px',
    'window_attn_px',
    'window_attn_margin_px',
  ]) {
    assert.ok(!emitted.yaml.includes(forbidden), `h3 не объявляет ${forbidden} — ключа быть не должно`);
  }
});

test('16 бит: файл валиден, отличается только шапкой и не содержит несуществующих ключей', () => {
  const eight = emitFor(SCENARIOS.base1440);
  const sixteen = emitFor({ ...SCENARIOS.base1440, bitDepth: 16 });
  const plan = planFor({ ...SCENARIOS.base1440, bitDepth: 16 });
  assert.equal(plan.verdict, 'unsupported');
  assert.match(sixteen.yaml, /^# запрошено 16 bit: пайплайн пока сохраняет 8-bit PNG/mu);
  assert.match(sixteen.yaml, /8-bit PNG/u);
  assert.deepEqual(parseYaml(sixteen.yaml), parseYaml(eight.yaml));
  assert.ok(!/bit_?depth/iu.test(sixteen.yaml), 'ключа битности в конфиге нет');
  assert.match(sixteen.fileName, /-16bit\.yaml$/u);
  assert.equal(headerLines(plan).length, 4);
});

test('шапка, имя файла и `presetFor` — производные от плана', () => {
  const plan = planFor(SCENARIOS.base1440);
  assert.equal(fileNameFor(plan), 'renderkit-qwen21-base-2560x1440-max-8bit.yaml');
  assert.deepEqual(parseYaml(emitPreset(plan).yaml), plain(presetFor(plan)));
  const header = headerLines(plan);
  assert.match(header[0], /#? ?renderkit r1\+/u);
  assert.match(header[0], /qwen21-base \| 2560x1440 \| quality=max \| 8 bit$/u);
});

test('kvCacheValue: режимы плана → значения YAML', () => {
  assert.equal(kvCacheValue('true'), true);
  assert.equal(kvCacheValue('false'), false);
  assert.equal(kvCacheValue('offload'), 'offload');
  assert.equal(kvCacheValue(null), null);
  assert.throws(() => kvCacheValue(/** @type {any} */ ('auto')), RangeError);
});

test('cli: команды запуска (run/gen/checks/save) и запрет набора кадров', () => {
  const withRefs = commandsFor(planFor(SCENARIOS.base1440));
  assert.equal(withRefs.run, `python -m qwenstyle run --config ${PLAN_FILE} --input <png> --out out/`);
  assert.equal(withRefs.gen, `python -m qwenstyle gen --preset ${PLAN_FILE} --prompt <project>/<stem> --count 1`);
  assert.deepEqual(withRefs.checks, [`python -m qwenstyle config show ${PLAN_FILE} --json`]);
  assert.match(withRefs.save, /^npm run plan -- --width 2560 --height 1440 --quality max --bits 8 /u);
  assert.match(withRefs.save, /--out plan\.yaml$/u);
  assert.ok(!withRefs.gen.includes('--count 2'));

  const t2i = commandsFor(planFor({ ...SCENARIOS.base1440, refs: 0 }));
  assert.ok(!t2i.run.includes('--input'), 'T2I: `--input` не передаётся вовсе');

  const anime = commandsFor(planFor({ ...SCENARIOS.base1440, anime: true }));
  assert.match(anime.save, / --anime /u);

  assert.equal(validateCount(1), 1);
  assert.throws(() => validateCount(2), (error) => error instanceof PlanError
    && error.code === 'frames-not-supported'
    && JSON.stringify(error.allowed) === '[1]');
  assert.throws(() => validateCount(0), (error) => error instanceof PlanError && error.code === 'invalid-count');
  assert.throws(() => commandsFor(planFor(SCENARIOS.base1440), { count: 3 }), /frames-not-supported/u);
});

test('cli: `--set`-форма только из whitelist, согласована с YAML, чужой ключ отвергнут', () => {
  for (const [name, scenario] of Object.entries(SCENARIOS)) {
    const plan = planFor(scenario);
    const parsed = parseYaml(emitPreset(plan).yaml);
    const backend = INVENTORY.models[scenario.model].backend;
    const allowed = allowedSetKeys(backend);
    const entries = setsFor(plan);
    assert.ok(entries.length >= 4, `${name}: ожидались настройки в --set-форме`);
    for (const { key, value } of entries) {
      assert.ok(allowed.includes(key), `${name}: ${key} вне whitelist ${backend}`);
      const yamlValue = getPath(parsed, key);
      const expected = key === 'generation.kv_cache' ? kvCacheValue(/** @type {any} */ (value)) : value;
      assert.deepEqual(yamlValue, expected, `${name}: ${key} разошёлся между YAML и --set`);
    }
    // `aspect: null` — не настройка: `--set aspect=null` в командной строке бессмысленен.
    if (plan.image.aspect === null) {
      assert.ok(!entries.some((entry) => entry.key === 'aspect'), `${name}: aspect=null не выписывается`);
    }
    const args = setArgs(plan);
    if (scenario.model !== 'h3-image') {
      assert.ok(args.some((arg) => arg.startsWith('generation.steps=')), `${name}: шаги в --set`);
    }
    assert.ok(args.every((arg) => arg.includes('=')), `${name}: формат key=value`);
  }

  const plan = planFor(SCENARIOS.h3);
  const entries = setsFor(plan);
  assert.ok(!entries.some((entry) => entry.key === 'generation.steps'), 'h3: шаги readonly, в --set не попадают');
  assert.throws(
    () => setsFor(plan, { 'generation.nope': 1 }),
    (/** @type {any} */ error) => error instanceof PlanError
      && error.code === 'unknown-set-key'
      && error.field === 'generation.nope'
      && Array.isArray(error.allowed)
      && error.allowed.includes('generation.offload'),
  );
  const withOverride = setsFor(planFor(SCENARIOS.base1440), { 'upscale.target_height': 0 });
  assert.ok(withOverride.some((entry) => entry.key === 'upscale.target_height' && entry.value === 0));
  assert.ok(CORE_SET_KEYS.includes('upscale.target_height'));
  assert.equal(assertSetKey('qwen_image21', 'generation.kv_cache'), 'generation.kv_cache');
});

test('anime: блок `lora` — из inventory с силой плана, комментарий объясняет включение', () => {
  const declared = INVENTORY.models['qwen21-base'].lora;
  assert.ok(declared, 'у qwen21-base объявлен адаптер');
  const emitted = emitFor({ ...SCENARIOS.base1440, anime: true });
  const parsed = parseYaml(emitted.yaml);
  assert.deepEqual(parsed.generation.lora, {
    repo: declared.repo,
    filename: declared.filename,
    scale: 0.7,
    strip_alpha: declared.strip_alpha,
  });
  assert.match(emitted.yaml, /^ {2}# anime: адаптер FUSAL /mu);
  // H3 без адаптера: `lora: null` и при `anime` (модель его не поддерживает).
  const h3 = parseYaml(emitFor({ ...SCENARIOS.h3, anime: true }).yaml);
  assert.equal(h3.generation.lora, null);
});

test('computePlan: `yaml` — готовый конфиг, `cli` — команды того же плана', () => {
  const plan = planFor(SCENARIOS.base1440);
  assert.equal(plan.verdict, 'ok');
  // Артефакт — непустая строка: без `extends` и имён пресетов, разбирается своим парсером.
  assert.equal(typeof plan.yaml, 'string');
  assert.ok(plan.yaml.trim().length > 0, 'yaml не должен быть пустым');
  assert.ok(!/extends/u.test(plan.yaml), 'в конфиге не должно быть extends');
  assert.ok(
    !/VRAM(?:8|12|16)-|-(?:preview|release|release-anime|preview-anime)\b/u.test(plan.yaml),
    'имена пресетов просочились в `Plan.yaml`',
  );
  const parsed = parseYaml(plan.yaml);
  assert.equal(parsed.upscale, null);
  assert.equal(parsed.size_px, 1440);
  // `cli` — объект эмиттера, `run` ссылается на файл-плейсхолдер `./…yaml`.
  assert.deepEqual(Object.keys(plan.cli).sort(), ['checks', 'gen', 'run', 'save', 'sets']);
  assert.match(plan.cli.run, /--config \.\/\S+\.yaml /u);
  assert.ok(plan.cli.run.includes('./plan.yaml'), 'команда `run` ссылается на имя файла');
  // `computePlan` и прямой эмиттер дают один и тот же файл (один источник значений).
  assert.equal(plan.yaml, emitPreset(plan).yaml);
});

test('план без рендер-стадии (`impossible`) не превращается в YAML', () => {
  const plan = planFor({ width: 3840, height: 2160, model: 'qwen21-base' });
  assert.equal(plan.verdict, 'impossible');
  assert.equal(plan.render, null);
  // Артефакта нет, но `computePlan` не бросает: `yaml`/`cli` — честные `null`.
  assert.equal(plan.yaml, null);
  assert.equal(plan.cli, null);
  for (const call of [() => emitPreset(plan), () => presetFor(plan), () => commandsFor(plan), () => setsFor(plan)]) {
    assert.throws(call, (error) => error instanceof PlanError && error.code === 'no-render-for-plan');
  }
});
