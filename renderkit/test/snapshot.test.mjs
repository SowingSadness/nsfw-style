// @ts-check
/**
 * Проверка снапшотов (`npm run snapshot` → `test/__snapshots__/plans.mjs`).
 *
 * 1. `node tools/snapshot.mjs --check` — пересчёт матрицы (3 качества × 3 модели × 4 VRAM ×
 *    4 RAM × 3 диска × 3 размера = 1296 запросов) и байтовое сравнение с файлом: расхождение
 *    означает, что модель оценок или политика памяти изменились без обновления снапшота.
 * 2. Выборочные ассерты по данным файла: контрольная точка, отсутствие апскейла и закрытый
 *    набор вердиктов — то, что снапшот обязан показывать даже при крупной перекалибровке.
 */

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath } from 'node:url';

import { SNAPSHOTS } from './__snapshots__/plans.mjs';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const TOOL = resolve(ROOT, 'tools/snapshot.mjs');

const VERDICTS = Object.freeze(['ok', 'degraded', 'impossible', 'unsupported']);

test('snapshot: матрица детерминирована, `--check` совпадает с файлом', () => {
  const stdout = execFileSync(process.execPath, [TOOL, '--check'], {
    cwd: ROOT,
    encoding: 'utf8',
  });
  assert.match(stdout, /\[snapshot\] ok: 1296 записей совпали/);
});

test('snapshot: контрольная точка 16/64/nvme/base/2560x1440/max', () => {
  const row = SNAPSHOTS['q-max|qwen21-base|v16|r64|nvme|2560x1440'];
  assert.ok(row, 'записи нет в снапшоте');
  assert.equal(row.verdict, 'ok');
  assert.equal(row.canvas, '2560x1440');
  assert.equal(row.steps, 25);
  assert.equal(row.offload, 'group');
  assert.equal(row.kv, 'offload');
  assert.equal(row.kvPlacement, 'ram');
  assert.equal(row.vaeTile, 512);
  assert.equal(row.window, 'dense');
  assert.equal(row.refPx, 960);
  assert.equal(row.q, 1920);
  assert.equal(typeof row.peakVram, 'number');
  assert.equal(typeof row.timeS, 'number');
});

test('snapshot: апскейла нет ни в одной записи', () => {
  const keys = Object.keys(SNAPSHOTS);
  assert.equal(keys.length, 1296);
  for (const key of keys) {
    assert.equal(SNAPSHOTS[key].upscale, null, `${key}: upscale должен быть null (A5)`);
  }
});

test('snapshot: вердикты только из четырёх значений, ключи стабильного формата', () => {
  for (const [key, row] of Object.entries(SNAPSHOTS)) {
    assert.ok(VERDICTS.includes(row.verdict), `${key}: неизвестный verdict ${row.verdict}`);
    assert.match(
      key,
      /^q-(draft|standard|max)\|(qwen21-base|qwen21-turbo|h3-image)\|v(8|12|16|24)\|r(16|32|64|128)\|(hdd|ssd|nvme)\|\d+x\d+$/,
    );
    if (row.verdict === 'impossible') {
      assert.equal(row.canvas, null, `${key}: у impossible не должно быть канваса`);
      assert.equal(row.q, null, `${key}: у impossible не должно быть Q`);
    } else {
      assert.match(row.canvas, /^\d+x\d+$/, `${key}: канвас обязателен`);
    }
  }
});
