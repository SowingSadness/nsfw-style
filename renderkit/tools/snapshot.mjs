#!/usr/bin/env node
// @ts-check
/**
 * `tools/snapshot.mjs` — матрица запросов → `test/__snapshots__/plans.mjs`.
 *
 * ```
 * node tools/snapshot.mjs           # пересчитать и записать снапшот
 * node tools/snapshot.mjs --check   # пересчитать и сравнить; exit 1 при расхождении
 * ```
 *
 * Матрица: `quality` × `model` × `vram` × `ram` × `disk` × размер кадра — всего 1296 запросов.
 * Порядок обхода фиксирован, поэтому файл детерминирован: повторный запуск не меняет его.
 * Пишется только `verdict`, `canvas`, `steps`, `offload`, `kv`, `kvPlacement`, `vaeTile`, `window`,
 * `refPx`, `upscale` (всегда `null`: апскейла в калькуляторе нет), `q`, `peakVram`, `timeS` —
 * этого достаточно, чтобы увидеть регресс модели оценок или политики памяти, не раздувая файл
 * полным `Plan`.
 *
 * Без зависимостей: `node:fs`/`node:path` + `src/index.mjs` (чистое ядро).
 *
 * Коды выхода: `0` — снапшот записан или совпал; `1` — расхождение в `--check`.
 */

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { computePlan, RULESET_VERSION } from '../src/index.mjs';

const TOOL_DIR = dirname(fileURLToPath(import.meta.url));
const OUT_PATH = resolve(TOOL_DIR, '../test/__snapshots__/plans.mjs');

/** Порядок обхода фиксирован: снапшот обязан совпадать байт-в-байт при повторном запуске. */
export const QUALITIES = Object.freeze(['draft', 'standard', 'max']);
export const MODELS = Object.freeze(['qwen21-base', 'qwen21-turbo', 'h3-image']);
export const VRAM_GIB = Object.freeze([8, 12, 16, 24]);
export const RAM_GIB = Object.freeze([16, 32, 64, 128]);
export const DISKS = Object.freeze(['hdd', 'ssd', 'nvme']);
export const SIZES = Object.freeze([
  Object.freeze([1280, 720]),
  Object.freeze([1920, 1080]),
  Object.freeze([2560, 1440]),
]);

/**
 * Ключ записи: `q-max|qwen21-base|v16|r64|nvme|2560x1440`.
 *
 * @param {{quality: string, model: string, vramGib: number, ramGib: number, disk: string,
 *          width: number, height: number}} cell
 * @returns {string}
 */
export function snapshotKey(cell) {
  return [
    `q-${cell.quality}`,
    cell.model,
    `v${cell.vramGib}`,
    `r${cell.ramGib}`,
    cell.disk,
    `${cell.width}x${cell.height}`,
  ].join('|');
}

/**
 * Сжатая запись снапшота из `Plan` (только устойчивые к форматированию поля).
 *
 * @param {any} plan
 * @returns {any}
 */
function snapshotRow(plan) {
  const memory = plan.render === null ? null : plan.render.memory;
  const kvCache = memory === null ? null : memory.kvCache;
  const vae = memory === null ? null : memory.vae;
  const attention = memory === null ? null : memory.attention;
  const ref = memory === null ? null : memory.ref;
  return {
    verdict: plan.verdict,
    canvas: plan.render === null
      ? null
      : `${plan.image.native.width}x${plan.image.native.height}`,
    steps: plan.render === null ? null : plan.render.steps,
    offload: plan.render === null ? null : plan.render.offload,
    kv: kvCache === null ? null : kvCache.mode,
    kvPlacement: kvCache === null ? null : kvCache.placement,
    vaeTile: vae === null ? null : vae.tilePx,
    window: attention === null
      ? null
      : (attention.mode === 'window' ? `${attention.windowPx}+${attention.marginPx}` : 'dense'),
    refPx: ref === null ? null : ref.px,
    // Апскейла в калькуляторе нет и не будет (A5): стадия `postprocess` считается отдельно.
    upscale: null,
    q: plan.stages[0].quality,
    peakVram: plan.estimates === null ? null : plan.estimates.vram.peakGib,
    timeS: plan.estimates === null ? null : plan.estimates.time.genPerFrameS,
  };
}

/**
 * Значение в синтаксисе JS-объекта: `null`, строка в одинарных кавычках, число как есть.
 * Строки в снапшоте — только `verdict`/`canvas`/`offload`/`kv`/`kvPlacement`/`window`.
 *
 * @param {string|number|boolean|null} value
 * @returns {string}
 */
function literal(value) {
  if (value === null) return 'null';
  if (typeof value === 'string') return `'${value.replaceAll('\\', '\\\\').replaceAll("'", "\\'")}'`;
  return String(value);
}

/**
 * Собрать весь файл снапшота (детерминированно).
 *
 * @returns {{entries: Map<string, any>, text: string, count: number}}
 */
export function buildSnapshot() {
  /** @type {Map<string, any>} */
  const entries = new Map();
  for (const quality of QUALITIES) {
    for (const model of MODELS) {
      for (const vramGib of VRAM_GIB) {
        for (const ramGib of RAM_GIB) {
          for (const disk of DISKS) {
            for (const [width, height] of SIZES) {
              const key = snapshotKey({ quality, model, vramGib, ramGib, disk, width, height });
              const plan = computePlan({
                hardware: { model, vramGib, ramGib, disk },
                image: { width, height, quality, bitDepth: 8 },
                refs: 1,
              });
              entries.set(key, snapshotRow(plan));
            }
          }
        }
      }
    }
  }

  const lines = [];
  lines.push('// СГЕНЕРИРОВАНО `node tools/snapshot.mjs` — не править руками.');
  lines.push(`// ruleset: ${RULESET_VERSION}`);
  lines.push(`// матрица: quality × model × vram × ram × disk × размер, ${entries.size} записей`);
  lines.push('export const SNAPSHOTS = Object.freeze({');
  for (const [key, row] of entries) {
    const fields = Object.entries(row).map(([name, value]) => `${name}: ${literal(value)}`);
    lines.push(`  '${key}': {${fields.join(', ')}},`);
  }
  lines.push('});');
  lines.push('');
  return { entries, text: lines.join('\n'), count: entries.size };
}

/**
 * Точка входа CLI.
 *
 * @param {string[]} argv
 * @returns {number} код выхода
 */
function main(argv) {
  const check = argv.includes('--check');
  const unknown = argv.filter((arg) => arg !== '--check' && arg !== '--help' && arg !== '-h');
  if (unknown.length > 0) {
    console.error(`[snapshot] неизвестный аргумент: ${unknown.join(' ')}`);
    console.error('использование: node tools/snapshot.mjs [--check]');
    return 2;
  }
  if (argv.includes('--help') || argv.includes('-h')) {
    console.log('node tools/snapshot.mjs [--check]  # матрица запросов → test/__snapshots__/plans.mjs');
    return 0;
  }

  const { text, count } = buildSnapshot();
  if (check) {
    /** @type {string|null} */
    let current = null;
    try {
      current = readFileSync(OUT_PATH, 'utf8');
    } catch {
      console.error(`[snapshot] файла нет: ${OUT_PATH}`);
      console.error('[snapshot] выполните `npm run snapshot` и закоммитьте результат');
      return 1;
    }
    if (current !== text) {
      console.error('[snapshot] снапшот разошёлся с моделью оценок');
      console.error(`[snapshot] файл: ${OUT_PATH} (${count} записей в пересчёте)`);
      console.error('[snapshot] выполните `npm run snapshot` и проверьте дифф');
      return 1;
    }
    console.log(`[snapshot] ok: ${count} записей совпали (${OUT_PATH})`);
    return 0;
  }

  mkdirSync(dirname(OUT_PATH), { recursive: true });
  writeFileSync(OUT_PATH, text, 'utf8');
  console.log(`[snapshot] записано ${count} записей → ${OUT_PATH}`);
  return 0;
}

const invokedDirectly = process.argv[1] !== undefined
  && import.meta.url === pathToFileURL(process.argv[1]).href;

if (invokedDirectly) process.exitCode = main(process.argv.slice(2));
