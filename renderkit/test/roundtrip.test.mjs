// @ts-check
/**
 * Round-trip «наш YAML = конфиг системы» (шаг 6, A2): `config show <файл> --json` должен дать
 * **ровно** `Plan.render` (0 расхождений), а `run --config <файл> --dry-run` — работать без весов.
 *
 * Файл генерируется в временный каталог (`os.tmpdir()`) и называется произвольно: этим
 * проверяется и «YAML без имён пресетов», и то, что `--config`/`--preset` принимают путь.
 *
 * Без Python/venv тест печатает `SKIP` и проходит (конвенция репозитория: успех = отсутствие
 * падений, а не наличие PyTorch).
 */

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { computePlan } from '../src/index.mjs';
import { emitPreset, kvCacheValue } from '../src/model/emit.mjs';
import { INVENTORY, settableKeys } from '../src/model/inventory.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
/** Корень пакета `renderkit/`. */
const PACKAGE_ROOT = resolve(HERE, '..');
/** Корень репозитория (там `.venv/` и пакет `qwenstyle`). */
const REPO_ROOT = resolve(PACKAGE_ROOT, '..');
/** Промт для `--prompt-file` (по конвенции AGENTS.md тесты берут его из `tests/texts/`). */
const PROMPT_FILE = join(REPO_ROOT, 'tests', 'texts', 'exp1.txt');

/**
 * @typedef {Object} Scenario
 * @property {string} name
 * @property {any} request
 * @property {string} note чем сценарий отличается от остальных
 */

/** Сценарии приёмки: финал, окно внимания, draft на turbo, H3, 8 ГиБ/CPU, портрет, T2I с тайлом 0. */
const SCENARIOS = /** @type {Scenario[]} */ ([
  {
    name: 'base-1440',
    request: {
      hardware: { model: 'qwen21-base', vramGib: 16, ramGib: 64, disk: 'nvme' },
      image: { width: 2560, height: 1440, quality: 'max', bitDepth: 8 },
      refs: 1,
    },
    note: '2560×1440 / max / base',
  },
  {
    name: 'window-1792',
    request: {
      hardware: { model: 'qwen21-base', vramGib: 16, ramGib: 64, disk: 'nvme' },
      image: { width: 3200, height: 1792, quality: 'max', bitDepth: 8 },
      refs: 1,
    },
    note: '3200×1792 / max / base (окно внимания)',
  },
  {
    name: 'draft-turbo',
    request: {
      hardware: { model: 'qwen21-turbo', vramGib: 16, ramGib: 64, disk: 'nvme' },
      image: { width: 1376, height: 768, quality: 'draft', bitDepth: 8 },
      refs: 1,
    },
    note: '1376×768 / draft / turbo',
  },
  {
    name: 'h3-1024',
    request: {
      hardware: { model: 'h3-image', vramGib: 16, ramGib: 64, disk: 'nvme' },
      image: { width: 1024, height: 1024, quality: 'max', bitDepth: 8 },
      refs: 1,
    },
    note: '1024×1024 / max / h3-image',
  },
  {
    name: 'small-8gib-cpu',
    request: {
      hardware: { model: 'qwen21-base', vramGib: 8, ramGib: 16, disk: 'ssd' },
      image: { width: 1152, height: 640, quality: 'max', bitDepth: 8 },
      refs: 1,
    },
    note: '1152×640 / max / base на 8 ГиБ: энкодер на CPU, stream остаётся true',
  },
  {
    name: 'portrait-ref-ratio',
    request: {
      hardware: { model: 'qwen21-base', vramGib: 12, ramGib: 32, disk: 'ssd' },
      image: { width: 960, height: 1440, quality: 'standard', bitDepth: 8 },
      refs: 1,
    },
    note: 'портрет 2:3 с референсом: aspect = null (пропорция от референса)',
  },
  {
    name: 'small-tile-zero',
    request: {
      hardware: { model: 'qwen21-base', vramGib: 8, ramGib: 16, disk: 'ssd' },
      image: { width: 1024, height: 576, quality: 'draft', bitDepth: 8 },
      refs: 0,
      memory: { vaeTilePx: 0 },
    },
    note: '1024×576 (16:9) / draft / T2I: vae_tile_px 0 (историческое 256/192)',
  },
]);

/**
 * Кандидаты на интерпретатор: сначала venv репозитория, затем системный `python`.
 *
 * @returns {string[]}
 */
function pythonCandidates() {
  const venv = process.platform === 'win32'
    ? join(REPO_ROOT, '.venv', 'Scripts', 'python.exe')
    : join(REPO_ROOT, '.venv', 'bin', 'python');
  /** @type {string[]} */
  const candidates = [];
  if (existsSync(venv)) candidates.push(venv);
  candidates.push(...(process.platform === 'win32' ? ['python'] : ['python3', 'python']));
  return candidates;
}

/**
 * @param {string} command
 * @param {string[]} args
 * @param {{cwd?: string, env?: Record<string, string>}} [options]
 * @returns {{status: number|null, stdout: string, stderr: string}}
 */
function runPython(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: options.cwd ?? REPO_ROOT,
    encoding: 'utf8',
    windowsHide: true,
    maxBuffer: 64 * 1024 * 1024,
    env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1', ...options.env },
  });
  return { status: result.status, stdout: result.stdout ?? '', stderr: result.stderr ?? '' };
}

/** @type {string|null|undefined} */
let cachedPython;

/**
 * Интерпретатор, который импортирует `qwenstyle`, или `null` (тогда тесты печатают SKIP).
 *
 * @returns {string|null}
 */
function findPython() {
  if (cachedPython !== undefined) return cachedPython;
  cachedPython = null;
  for (const command of pythonCandidates()) {
    const probe = runPython(command, ['-c', 'import qwenstyle']);
    if (probe.status === 0) {
      cachedPython = command;
      break;
    }
  }
  return cachedPython;
}

/**
 * Прогнать `computePlan` на сценарии и записать YAML во временный каталог.
 *
 * @param {Scenario} scenario
 * @returns {{plan: any, yaml: string, dir: string, file: string}}
 */
function writePlanFile(scenario) {
  const plan = computePlan(scenario.request);
  assert.equal(plan.verdict, 'ok', `${scenario.name}: сценарий должен быть валидным планом`);
  const emitted = emitPreset(plan);
  const dir = mkdtempSync(join(tmpdir(), 'renderkit-rt-'));
  // Имя произвольное: система не должна опираться на имя файла (A2).
  const file = join(dir, 'my-plan.yaml');
  writeFileSync(file, emitted.yaml, 'utf8');
  return { plan, yaml: emitted.yaml, dir, file };
}

/**
 * Строки сверки «YAML → конфиг системы»: только то, что план действительно задаёт
 * (не объявленное бэкендом — не выписывается в файл и не сравнивается).
 *
 * @param {any} plan
 * @param {any} config разобранный `config show --json`
 * @returns {Array<{path: string, expected: unknown, actual: unknown}>}
 */
function comparisonRows(plan, config) {
  const render = plan.render;
  const model = INVENTORY.models[plan.request.hardware.model];
  const settable = new Set(settableKeys(model.backend));
  const kvMode = (/** @type {unknown} */ value) => (value === true ? 'true' : (value === false ? 'false' : value));
  /** @type {Array<{path: string, expected: unknown, actual: unknown}>} */
  const rows = [
    { path: 'generation.backend', expected: model.backend, actual: config.generation.backend },
    { path: 'generation.repo', expected: model.repo, actual: config.generation.repo },
    { path: 'generation.transformer', expected: model.transformer, actual: config.generation.transformer },
    { path: 'generation.precision_cache', expected: model.precision_cache, actual: config.generation.precision_cache },
    { path: 'generation.steps', expected: render.steps, actual: config.generation.steps },
    { path: 'generation.offload', expected: render.offload, actual: config.generation.offload },
    { path: 'generation.precision', expected: render.precision, actual: config.generation.precision },
    {
      path: 'generation.sampler_kwargs.offload_blocks',
      expected: render.offloadBlocks,
      actual: config.generation.sampler_kwargs.offload_blocks,
    },
    { path: 'generation.lora.scale', expected: render.loraScale, actual: config.generation.lora ? config.generation.lora.scale : 0 },
    { path: 'size_px', expected: plan.image.native.sizePx, actual: config.size_px },
    { path: 'aspect', expected: plan.image.aspect, actual: config.aspect },
    { path: 'encoder.name', expected: render.encoder.name, actual: config.encoder.name },
    { path: 'encoder.device', expected: render.encoder.device, actual: config.encoder.device },
    { path: 'encoder.stream', expected: render.encoder.stream, actual: config.encoder.stream },
    { path: 'encoder.prefetch', expected: render.encoder.prefetch, actual: config.encoder.prefetch },
    { path: 'upscale.factor', expected: 0, actual: config.upscale.factor },
    { path: 'upscale.target_height', expected: null, actual: config.upscale.target_height },
    { path: 'logs', expected: true, actual: config.logs },
  ];
  if (settable.has('generation.kv_cache')) {
    rows.push({
      path: 'generation.kv_cache',
      expected: render.memory.kvCache.mode,
      actual: kvMode(config.generation.kv_cache),
    });
  }
  if (settable.has('generation.sampler_kwargs.offload_stream')) {
    rows.push({
      path: 'generation.sampler_kwargs.offload_stream',
      expected: true,
      actual: config.generation.sampler_kwargs.offload_stream,
    });
  }
  if (settable.has('generation.sampler_kwargs.vae_tile_px')) {
    rows.push({
      path: 'generation.sampler_kwargs.vae_tile_px',
      expected: render.memory.vae.tilePx,
      actual: config.generation.sampler_kwargs.vae_tile_px,
    });
  }
  if (settable.has('generation.sampler_kwargs.ref_px')) {
    rows.push({
      path: 'generation.sampler_kwargs.ref_px',
      expected: render.memory.ref.px,
      actual: config.generation.sampler_kwargs.ref_px,
    });
  }
  if (render.memory.attention.mode === 'window') {
    rows.push(
      {
        path: 'generation.sampler_kwargs.window_attn_px',
        expected: render.memory.attention.windowPx,
        actual: config.generation.sampler_kwargs.window_attn_px,
      },
      {
        path: 'generation.sampler_kwargs.window_attn_margin_px',
        expected: render.memory.attention.marginPx,
        actual: config.generation.sampler_kwargs.window_attn_margin_px,
      },
    );
  }
  return rows;
}

test('round-trip: `config show <yaml> --json` совпадает с `Plan.render` (7 сценариев)', (context) => {
  const python = findPython();
  if (python === null) {
    context.skip('SKIP: не найден Python с установленным qwenstyle');
    return;
  }
  /** @type {Array<{path: string, expected: unknown, actual: unknown}>} */
  const mismatches = [];
  const dirs = [];
  try {
    for (const scenario of SCENARIOS) {
      const { plan, dir, file } = writePlanFile(scenario);
      dirs.push(dir);
      const run = runPython(python, ['-m', 'qwenstyle', 'config', 'show', file, '--json']);
      assert.equal(run.status, 0, `${scenario.name}: config show упал:\n${run.stdout}\n${run.stderr}`);
      const config = JSON.parse(run.stdout);
      // Имя файла — произвольное, и система не требует имени пресета.
      assert.equal(config.preset, 'my-plan', `${scenario.name}: система берёт preset из имени файла`);
      for (const row of comparisonRows(plan, config)) {
        if (JSON.stringify(row.expected) !== JSON.stringify(row.actual)) {
          mismatches.push({ ...row, scenario: scenario.name });
        }
      }
    }
  } finally {
    for (const dir of dirs) rmSync(dir, { recursive: true, force: true });
  }
  assert.deepEqual(mismatches, [], 'YAML разошёлся с Plan.render');
});

test('round-trip: `run --config <yaml> --dry-run` — exit 0 без весов', (context) => {
  const python = findPython();
  if (python === null) {
    context.skip('SKIP: не найден Python с установленным qwenstyle');
    return;
  }
  const scenario = SCENARIOS[0];
  const { dir, file } = writePlanFile(scenario);
  try {
    const run = runPython(python, [
      '-m', 'qwenstyle', 'run',
      '--config', file,
      '--dry-run',
      '--prompt-file', PROMPT_FILE,
      '--out', join(dir, 'x.png'),
    ]);
    assert.equal(run.status, 0, `run --dry-run упал:\n${run.stdout}\n${run.stderr}`);
    assert.match(run.stdout, /\[plan\]/u, 'ожидался план прогона в stdout');
    assert.match(run.stdout, /dry-run/u);
    assert.ok(!run.stderr.includes('Traceback'), `в stderr трейсбек:\n${run.stderr}`);
    assert.ok(!existsSync(join(dir, 'x-20261113-25.png')), 'dry-run не должен писать кадр');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('kvCacheValue: значения YAML совпадают с режимами, которые знает система', (context) => {
  // Чистая проверка формы (без Python): `true`/`false` — булевы, `offload` — строка.
  assert.equal(kvCacheValue('true'), true);
  assert.equal(kvCacheValue('offload'), 'offload');
  const python = findPython();
  if (python !== null) {
    const probe = runPython(python, ['-c', 'import qwenstyle.config as c; print(c.GenerationConfig().kv_cache)']);
    assert.equal(probe.status, 0, `не удалось прочитать GenerationConfig:\n${probe.stderr}`);
    assert.match(probe.stdout.trim(), /auto-offload-ref2/u);
  }
  context.diagnostic('kv_cache: true | offload | false — как в `qwenstyle/backends/qwen_kv.py`');
});
