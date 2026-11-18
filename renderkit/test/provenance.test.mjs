// @ts-check
/**
 * Провенанс калибровки: у каждой константы модели есть источник и вид, допущения объяснены,
 * каждый `source` замера ссылается на существующий файл, а строки `stage: 'postprocess'`
 * не влияют на фит рендера (A5, A8).
 */

import { existsSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';
import test from 'node:test';

import { MODEL } from '../src/data/model.mjs';
import {
  MEASUREMENTS, fitRows, renderRows, sourceFile,
} from '../src/data/measurements.mjs';
import { fitTime, fitVram } from '../tools/fit.mjs';
import { COST_ASSUMPTIONS } from '../src/model/cost.mjs';
import { MEMORY_ASSUMPTIONS } from '../src/model/memory.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = join(HERE, '..');
/** Корень репозитория: `source` замеров и провенанса записан относительно него. */
const REPO_ROOT = join(PACKAGE_ROOT, '..');

const KINDS = /** @type {const} */ (['measured', 'fitted', 'assumption']);

/**
 * Плоский список числовых листьев объекта: `{'vram.resident.qwen21-base.none': 7.67, …}`.
 * `null` (неприменимо, например `h3` + `offload: none`) пропускается — это не константа.
 *
 * @param {Record<string, any>} node
 * @param {string} [prefix]
 * @returns {Map<string, number>}
 */
function numericLeaves(node, prefix = '') {
  /** @type {Map<string, number>} */
  const out = new Map();
  for (const [key, value] of Object.entries(node)) {
    const path = prefix === '' ? key : `${prefix}.${key}`;
    if (typeof value === 'number') out.set(path, value);
    else if (value !== null && typeof value === 'object') {
      for (const [inner, innerValue] of numericLeaves(value, path)) out.set(inner, innerValue);
    }
  }
  return out;
}

/**
 * Границы допуска строки замера (или `null`, если допуск не расширяли).
 *
 * @param {import('../src/data/measurements.mjs').Measurement} row
 * @returns {{vramPct?: number, timePct?: number, totalPct?: number} | null}
 */
const toleranceOf = (row) => row.tolerance ?? null;

test('каждый числовой коэффициент модели описан в provenance тем же значением', () => {
  const leaves = numericLeaves(MODEL.coefficients);
  assert.ok(leaves.size >= 20, `ожидается ≥20 констант, найдено ${leaves.size}`);
  /** @type {string[]} */
  const missing = [];
  for (const [path, value] of leaves) {
    const entry = MODEL.provenance[path];
    if (!entry) {
      missing.push(`${path} = ${value}: нет записи в provenance`);
      continue;
    }
    if (entry.value !== value) {
      missing.push(`${path}: в provenance ${entry.value}, в коэффициентах ${value}`);
    }
  }
  assert.deepEqual(missing, []);
});

test('в provenance нет записей-сирот: каждый ключ — реальный лист коэффициентов', () => {
  const leaves = numericLeaves(MODEL.coefficients);
  /** @type {string[]} */
  const orphans = [];
  for (const path of Object.keys(MODEL.provenance)) {
    if (!leaves.has(path)) orphans.push(path);
  }
  assert.deepEqual(orphans, []);
});

test('у каждой константы непустые source и kind; assumption — только с объяснением в note', () => {
  for (const [path, entry] of Object.entries(MODEL.provenance)) {
    assert.equal(typeof entry.source, 'string', `${path}: source не строка`);
    assert.ok(entry.source.trim().length > 8, `${path}: пустой source`);
    assert.match(entry.source, /^(docs\/|renderkit\/|AGENTS\.md|README\.md)/u,
      `${path}: source должен указывать на файл репозитория, получено ${entry.source}`);
    assert.ok(KINDS.includes(entry.kind), `${path}: kind=${entry.kind} не из ${KINDS.join('/')}`);
    assert.equal(typeof entry.unit, 'string', `${path}: нет unit`);
    if (entry.kind === 'assumption') {
      assert.equal(typeof entry.note, 'string', `${path}: assumption без note`);
      assert.ok(entry.note.trim().length > 20, `${path}: assumption с слишком коротким note`);
    }
  }
});

test('каждый файл из provenance существует в репозитории', () => {
  /** @type {string[]} */
  const missing = [];
  for (const [path, entry] of Object.entries(MODEL.provenance)) {
    const file = sourceFile(entry.source);
    if (!existsSync(join(REPO_ROOT, file))) missing.push(`${path}: ${file}`);
  }
  assert.deepEqual(missing, []);
});

test('каждый замер: source с существующим файлом и якорем, stage и config на месте', () => {
  /** @type {string[]} */
  const problems = [];
  const ids = new Set();
  for (const row of MEASUREMENTS) {
    if (ids.has(row.id)) problems.push(`${row.id}: повторяющийся id`);
    ids.add(row.id);
    assert.ok(row.stage === 'render' || row.stage === 'postprocess', `${row.id}: stage=${row.stage}`);
    assert.equal(typeof row.machine, 'string', `${row.id}: нет machine`);
    assert.ok(row.machine.length > 0, `${row.id}: пустой machine`);

    const file = sourceFile(row.source);
    if (file === row.source) problems.push(`${row.id}: в source нет якоря «#…»`);
    if (!existsSync(join(REPO_ROOT, file))) problems.push(`${row.id}: файла ${file} нет в репозитории`);

    const config = row.config;
    for (const field of /** @type {const} */ ([
      'model', 'vramGib', 'ramGib', 'disk', 'refs', 'width', 'height', 'tokens', 'steps',
      'offload', 'kvCache', 'encoderDevice', 'windowAttnPx',
    ])) {
      if (!(field in config)) problems.push(`${row.id}: в config нет поля ${field}`);
    }
    for (const field of /** @type {const} */ (['width', 'height', 'tokens'])) {
      if (config[field] === null || config[field] === undefined) {
        problems.push(`${row.id}: пустое поле config.${field}`);
      }
    }
    if (config.width % 16 !== 0 || config.height % 16 !== 0) {
      problems.push(`${row.id}: канвас ${config.width}×${config.height} не кратен 16`);
    }
    if (config.tokens !== (config.width / 16) * (config.height / 16)) {
      problems.push(`${row.id}: tokens ${config.tokens} ≠ (W/16)·(H/16)`);
    }
    if (Object.keys(row.measured).length === 0) problems.push(`${row.id}: нет ни одного измерения`);
    if (row.tolerance && !row.note) problems.push(`${row.id}: расширенный допуск без note`);
    if (row.suspect && !row.note) problems.push(`${row.id}: suspect без note`);
  }
  assert.deepEqual(problems, []);
});

test('строки stage: postprocess не попадают в фит рендера', () => {
  const postprocess = MEASUREMENTS.filter((row) => row.stage === 'postprocess');
  assert.ok(postprocess.length >= 1, 'ожидается хотя бы одна строка postprocess (контракт A5)');
  for (const row of postprocess) {
    assert.ok(!fitRows(MEASUREMENTS).some((trainable) => trainable.id === row.id),
      `${row.id}: postprocess-строка попала в фит`);
  }

  // Подмена замера на заведомо чужой: коэффициенты рендера не должны шевельнуться.
  const tampered = MEASUREMENTS.map((row) => (row.stage === 'postprocess'
    ? { ...row, measured: { ...row.measured, genS: 99999, peakVramGib: 42 } }
    : row));
  assert.deepEqual(fitTime(fitRows(tampered)).coefficients, fitTime(fitRows(MEASUREMENTS)).coefficients);
  assert.deepEqual(fitVram(fitRows(tampered)).coefficients, fitVram(fitRows(MEASUREMENTS)).coefficients);

  // И в статистику модели они тоже не входят.
  const statIds = new Set([
    ...MODEL.stats.vram.rows.map((row) => row.id),
    ...MODEL.stats.time.rows.map((row) => row.id),
    ...MODEL.stats.total.rows.map((row) => row.id),
  ]);
  for (const row of postprocess) assert.ok(!statIds.has(row.id), `${row.id}: postprocess в статистике`);
  assert.equal(MODEL.envelope.renderCount, renderRows(MEASUREMENTS).length);
});

test('оценка != NaN на всех замерах: фит покрывает каждую строку', () => {
  for (const row of renderRows(MEASUREMENTS)) {
    const vram = MODEL.stats.vram.rows.find((entry) => entry.id === row.id);
    const time = MODEL.stats.time.rows.find((entry) => entry.id === row.id);
    if (row.measured.peakVramGib !== undefined && !row.suspect) {
      assert.ok(vram, `${row.id}: нет оценки VRAM в статистике`);
      assert.ok(Number.isFinite(vram.est), `${row.id}: VRAM не число`);
    }
    if ((row.measured.genS ?? 0) > 0 && !row.suspect) {
      assert.ok(time, `${row.id}: нет оценки времени в статистике`);
      assert.ok(Number.isFinite(time.est), `${row.id}: время не число`);
    }
  }
  for (const row of renderRows(MEASUREMENTS).filter((entry) => entry.suspect)) {
    assert.ok(MODEL.stats.excluded.some((entry) => entry.id === row.id),
      `${row.id}: противоречивый замер не отмечен в stats.excluded`);
    assert.ok(!MODEL.stats.time.rows.some((entry) => entry.id === row.id),
      `${row.id}: противоречивый замер попал в агрегат`);
  }
});

test('конверт модели описывает диапазон замеров и модели', () => {
  const envelope = MODEL.envelope;
  assert.ok(envelope.tokens.min < envelope.tokens.max);
  assert.ok(envelope.steps.min <= 8 && envelope.steps.max >= 25);
  assert.deepEqual(envelope.models, ['h3-image', 'qwen21-base', 'qwen21-turbo']);
  assert.ok(envelope.vramGib.min <= 8 && envelope.vramGib.max >= 16);
  assert.ok(envelope.windowAttnPx.max >= 384, 'окно внимания должно быть в конверте');
  assert.equal(envelope.measurementCount, MEASUREMENTS.length);
  assert.equal(envelope.fitCount, fitRows(MEASUREMENTS).length);
});

test('файл модели помечен как сгенерированный и не правится руками', () => {
  const text = readFileSync(join(PACKAGE_ROOT, 'src', 'data', 'model.mjs'), 'utf8');
  const generatedMarker = ['// @ts-check', '// generated by tools/fit.mjs — do not edit; '
    + 'regenerate with `npm run fit`'].join(String.fromCharCode(10));
  assert.ok(text.startsWith(generatedMarker), 'в файле нет шапки сгенерированного файла');
  assert.match(text, /export const MODEL = Object\.freeze\(/u);
});

test('расширенный допуск — исключение, а не правило (≤ 15 % строк)', () => {
  const rows = renderRows(MEASUREMENTS).filter((row) => toleranceOf(row) !== null && !row.suspect);
  const all = renderRows(MEASUREMENTS).filter((row) => !row.suspect);
  assert.ok(rows.length > 0, 'механизм расширения допуска должен быть задействован');
  assert.ok(rows.length <= Math.ceil(all.length * 0.15),
    `расширенный допуск у ${rows.length} из ${all.length} строк — это уже не исключение`);
});

test('строки вне 30 % обязаны иметь расширенный допуск и объяснение', () => {
  const byId = new Map(MEASUREMENTS.map((row) => [row.id, row]));
  const outliers = MODEL.stats.total.rows.filter((row) => Math.abs(row.errPct) > 30);
  assert.ok(outliers.length > 0, 'ожидается хотя бы одна строка вне 30 % (иначе тест ничего не охраняет)');
  const problems = [];
  for (const row of outliers) {
    const source = byId.get(row.id);
    if (!source) {
      problems.push(`${row.id}: строки нет в measurements.mjs`);
      continue;
    }
    const tolerance = toleranceOf(source);
    if (!tolerance || typeof tolerance.totalPct !== 'number') {
      problems.push(`${row.id}: ошибка ${row.errPct.toFixed(2)} % без tolerance.totalPct`);
      continue;
    }
    if (tolerance.totalPct < Math.abs(row.errPct)) {
      problems.push(`${row.id}: допуск ${tolerance.totalPct} % меньше фактической ошибки ${row.errPct.toFixed(2)} %`);
    }
    if (!source.note || source.note.length < 40) {
      problems.push(`${row.id}: расширенный допуск без внятного note`);
    }
  }
  assert.deepEqual(problems, []);
});

test('провенанс локальных допущений cost/memory: source, kind и note у assumptions', () => {
  const problems = [];
  for (const [label, table] of /** @type {const} */ ([
    ['COST_ASSUMPTIONS', COST_ASSUMPTIONS],
    ['MEMORY_ASSUMPTIONS', MEMORY_ASSUMPTIONS],
  ])) {
    for (const [key, entry] of Object.entries(table)) {
      const where = `${label}[${key}]`;
      if (typeof entry.value !== 'number') problems.push(`${where}: value не число`);
      if (!KINDS.includes(entry.kind)) problems.push(`${where}: kind ${JSON.stringify(entry.kind)} вне ${KINDS.join('|')}`);
      if (typeof entry.source !== 'string' || entry.source.trim() === '') problems.push(`${where}: пустой source`);
      if (entry.kind === 'assumption' && (typeof entry.note !== 'string' || entry.note.trim() === '')) {
        problems.push(`${where}: assumption без note`);
      }
    }
  }
  assert.deepEqual(problems, []);
});
