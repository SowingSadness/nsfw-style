// @ts-check
/**
 * CLI-режим `gen` (шаг 6, A2/A10): сгенерированный YAML работает как `--preset` под любым именем.
 *
 * `gen --prompt` резолвит промт **внутри `prompts/`** (не как путь к файлу), поэтому тест создаёт
 * изолированный «репозиторий» во временном каталоге: `prompts/tests/texts/exp1.txt` + `plan.yaml`,
 * а запускает `python -m qwenstyle` из этого каталога (`PYTHONPATH` указывает на настоящий пакет).
 *
 * Проверяется:
 * - `gen --preset <yaml> --prompt … --dry-run` → exit 0 без весов, с произвольным именем файла;
 * - dry-run не создаёт кадров/логов (ни в кадре, ни в кэше проекта);
 * - `gen`-команда из `cli` всегда несёт `--count 1`, а `--count > 1` отвергается (набора кадров
 *   в калькуляторе нет).
 *
 * Без Python/venv тест печатает `SKIP`.
 */

import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

import { computePlan } from '../src/index.mjs';
import { emitPreset } from '../src/model/emit.mjs';
import { commandsFor, validateCount } from '../src/model/cli.mjs';
import { PlanError } from '../src/schema.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
/** Корень пакета `renderkit/`. */
const PACKAGE_ROOT = resolve(HERE, '..');
/** Корень репозитория (`.venv/`, пакет `qwenstyle`, `tests/texts/`). */
const REPO_ROOT = resolve(PACKAGE_ROOT, '..');
const PROMPT_SOURCE = join(REPO_ROOT, 'tests', 'texts', 'exp1.txt');

/** Запрос сценария: обычный финал на base (aspect 16:9 берётся из YAML, референсов нет → T2I). */
const REQUEST = {
  hardware: { model: 'qwen21-base', vramGib: 16, ramGib: 64, disk: 'nvme' },
  image: { width: 2560, height: 1440, quality: 'max', bitDepth: 8 },
  refs: 1,
};

/**
 * @returns {string[]} кандидаты на интерпретатор (venv, затем системный)
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
 * @param {{cwd: string, env?: Record<string, string>}} options
 * @returns {{status: number|null, stdout: string, stderr: string}}
 */
function runPython(command, args, options) {
  const result = spawnSync(command, args, {
    cwd: options.cwd,
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
 * @returns {string|null}
 */
function findPython() {
  if (cachedPython !== undefined) return cachedPython;
  cachedPython = null;
  for (const command of pythonCandidates()) {
    const probe = runPython(command, ['-c', 'import qwenstyle'], { cwd: REPO_ROOT });
    if (probe.status === 0) {
      cachedPython = command;
      break;
    }
  }
  return cachedPython;
}

/**
 * Все файлы во временном каталоге (для проверки «dry-run ничего не создал»).
 *
 * @param {string} dir
 * @returns {string[]}
 */
function listFiles(dir) {
  /** @type {string[]} */
  const files = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) files.push(...listFiles(full));
    else files.push(full);
  }
  return files;
}

test('gen --preset <произвольный yaml> --dry-run: exit 0 и никаких артефактов', (context) => {
  const python = findPython();
  if (python === null) {
    context.skip('SKIP: не найден Python с установленным qwenstyle');
    return;
  }
  const plan = computePlan(REQUEST);
  const emitted = emitPreset(plan);
  const root = mkdtempSync(join(tmpdir(), 'renderkit-gen-'));
  try {
    const planFile = join(root, 'my-arbitrary-plan.yaml');
    writeFileSync(planFile, emitted.yaml, 'utf8');
    // Изолированный «репозиторий»: `--prompt` резолвится от `prompts/` текущего каталога.
    const promptDir = join(root, 'prompts', 'tests', 'texts');
    mkdirSync(promptDir, { recursive: true });
    writeFileSync(join(promptDir, 'exp1.txt'), readFileSync(PROMPT_SOURCE, 'utf8'), 'utf8');

    const run = runPython(python, [
      '-m', 'qwenstyle', 'gen',
      '--preset', planFile,
      '--prompt', 'tests/texts/exp1.txt',
      '--dry-run',
    ], { cwd: root, env: { PYTHONPATH: REPO_ROOT } });

    assert.equal(run.status, 0, `gen --dry-run упал:\n${run.stdout}\n${run.stderr}`);
    assert.match(run.stdout, /\[gen\]/u, 'ожидались строки `[gen]` в stdout');
    assert.match(run.stdout, /dry-run/u, 'ожидалась пометка dry-run');
    assert.ok(!run.stderr.includes('Traceback'), `в stderr трейсбек:\n${run.stderr}`);
    // Пресет взят из файла под произвольным именем, а не из `presets/`.
    assert.match(run.stdout, /my-arbitrary-plan/u, 'в логе должно быть видно имя файла-пресета');

    const created = listFiles(root).filter((file) => !file.startsWith(promptDir) && file !== planFile);
    assert.deepEqual(created, [], `dry-run не должен создавать файлов, найдено: ${created.join(', ')}`);
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});

test('команда gen: всегда `--count 1`, набор кадров не поддерживается', () => {
  const plan = computePlan(REQUEST);
  const commands = commandsFor(plan);
  assert.match(commands.gen, /--count 1$/u);
  assert.ok(!commands.gen.includes('--count 2'));
  assert.throws(
    () => commandsFor(plan, { count: 2 }),
    (error) => error instanceof PlanError
      && error.code === 'frames-not-supported'
      && JSON.stringify(error.allowed) === '[1]',
  );
  assert.equal(validateCount(1), 1);
  // Файл сохранён под любым именем: в команде — плейсхолдер, а не имя пресета.
  assert.match(commands.gen, /--preset \.\/plan\.yaml /u);
});
