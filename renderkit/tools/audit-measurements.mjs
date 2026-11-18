#!/usr/bin/env node
// @ts-check
/**
 * Аудит калибровки: печатает `id | est | measured | err%` по VRAM, времени генерации и итогу,
 * сводку (медиана/максимум) и падает (exit 1), если модель вышла из допуска.
 *
 * Допуск по умолчанию: **медиана |err| ≤ 15 %, максимум ≤ 30 %**. Строка может расширить свой
 * допуск (`tolerance` + `note` в `src/data/measurements.mjs`) — тогда она проверяется по своей
 * графе, а сводка считается по строкам без расширения (и печатает, сколько строк расширено).
 *
 * Пики VRAM в наших документах сняты тремя метриками (`nvidia-smi` — полное `memory.used` карты;
 * аллокатор torch своего процесса; пик процесса). Оценка живёт в метрике `nvidia-smi`, поэтому
 * строки с другой метрикой приводятся к ней измеренной поправкой
 * `MODEL.coefficients.vram.measurementDesktopReserveGib` и помечаются в таблице `*`.
 *
 * Запуск: `npm run audit:measurements` (или `node tools/audit-measurements.mjs`).
 */

import {
  MEASUREMENTS, auditedRows, peakMetricOf, renderRows,
} from '../src/data/measurements.mjs';
import { MODEL } from '../src/data/model.mjs';
import {
  measuredVramTarget, predictEncodeS, predictGenS, predictLoadS, predictVram,
} from './fit.mjs';

/** Допуск по умолчанию: медиана и максимум `|err|` в процентах. */
const DEFAULT_TOLERANCE = /** @type {const} */ ({ medianPct: 15, maxPct: 30 });

/** @typedef {{id: string, est: number, measured: number, note: string, errPct: number,
 *   tolerance: {pct: number} | null}} AuditRow */

const coefficients = MODEL.coefficients;

/**
 * Строка аудита: оценка против замера (для VRAM — против приведённого к метрике значения).
 *
 * @param {import('../src/data/measurements.mjs').Measurement} row
 * @returns {AuditRow | null}
 */
function vramRow(row) {
  const target = measuredVramTarget(row, coefficients);
  if (target === null) return null;
  const est = predictVram(row.config, coefficients);
  const metric = peakMetricOf(row);
  return {
    id: row.id,
    est,
    measured: target,
    note: metric === 'nvidia-smi' ? '' : `приведено к nvidia-smi из «${metric}» (+${coefficients.vram.measurementDesktopReserveGib} ГиБ)`,
    errPct: (est / target - 1) * 100,
    tolerance: row.tolerance?.vramPct ? { pct: row.tolerance.vramPct } : null,
  };
}

/**
 * @param {import('../src/data/measurements.mjs').Measurement} row
 * @returns {AuditRow | null}
 */
function timeRow(row) {
  const measured = row.measured.genS;
  if (measured === undefined || measured <= 0) return null;
  const est = predictGenS(row.config, coefficients);
  return {
    id: row.id,
    est,
    measured,
    note: '',
    errPct: (est / measured - 1) * 100,
    tolerance: row.tolerance?.timePct ? { pct: row.tolerance.timePct } : null,
  };
}

/**
 * Итог = загрузка + энкод (0 при попадании в кэш эмбеддингов) + генерация.
 *
 * @param {import('../src/data/measurements.mjs').Measurement} row
 * @returns {AuditRow | null}
 */
function totalRow(row) {
  const measured = row.measured.totalS;
  if (measured === undefined) return null;
  const cacheHit = (row.measured.encodeS ?? 0) === 0;
  const est = predictLoadS(row.config, coefficients)
    + (cacheHit ? 0 : predictEncodeS(row.config, coefficients))
    + predictGenS(row.config, coefficients);
  return {
    id: row.id,
    est,
    measured,
    note: cacheHit ? 'энкод из кэша (0 с)' : '',
    errPct: (est / measured - 1) * 100,
    tolerance: row.tolerance?.totalPct ? { pct: row.tolerance.totalPct } : null,
  };
}

/**
 * Сводка: медиана и максимум `|err|` по строкам без расширенного допуска + проверка расширенных.
 *
 * @param {AuditRow[]} rows
 */
function summarise(rows) {
  const plain = rows.filter((row) => row.tolerance === null);
  const expanded = rows.filter((row) => row.tolerance !== null);
  const abs = plain.map((row) => Math.abs(row.errPct)).sort((left, right) => left - right);
  const middle = Math.floor(abs.length / 2);
  const median = abs.length === 0
    ? 0
    : (abs.length % 2 === 0 ? (abs[middle - 1] + abs[middle]) / 2 : abs[middle]);
  const max = abs.length === 0 ? 0 : abs[abs.length - 1];
  const failures = [];
  if (median > DEFAULT_TOLERANCE.medianPct) {
    failures.push(`медиана ${median.toFixed(2)} % > ${DEFAULT_TOLERANCE.medianPct} %`);
  }
  if (max > DEFAULT_TOLERANCE.maxPct) {
    failures.push(`максимум ${max.toFixed(2)} % > ${DEFAULT_TOLERANCE.maxPct} %`);
  }
  for (const row of expanded) {
    const limit = /** @type {{pct: number}} */ (row.tolerance).pct;
    if (Math.abs(row.errPct) > limit) {
      failures.push(`${row.id}: ${row.errPct.toFixed(1)} % > допуск ${limit} %`);
    }
  }
  return {
    n: rows.length,
    plain: plain.length,
    expanded,
    median,
    max,
    failures,
  };
}

/**
 * @param {string} title
 * @param {AuditRow[]} rows
 * @param {string} unit
 */
function printTable(title, rows, unit) {
  console.log(`\n=== ${title} (${unit}) ===`);
  console.log(`${'id'.padEnd(34)} ${'est'.padStart(8)} ${'measured'.padStart(9)} ${'err%'.padStart(8)}`);
  for (const row of rows) {
    const mark = row.note.startsWith('приведено') ? '*' : ' ';
    const err = `${row.errPct >= 0 ? '+' : ''}${row.errPct.toFixed(1)}`;
    console.log(`${row.id.padEnd(34)} ${row.est.toFixed(2).padStart(8)}`
      + ` ${row.measured.toFixed(2).padStart(9)} ${err.padStart(8)} ${mark}`);
  }
  const summary = summarise(rows);
  const status = summary.failures.length === 0 ? 'ок' : 'ВНЕ ДОПУСКА';
  console.log(`\nсводка: n=${summary.n} (в медиане/максимуме ${summary.plain}, `
    + `с расширенным допуском ${summary.expanded.length}), `
    + `медиана ${summary.median.toFixed(2)} %, максимум ${summary.max.toFixed(2)} % — ${status}`);
  if (summary.expanded.length > 0) {
    console.log('расширенный допуск (см. note в замере):');
    for (const row of summary.expanded) {
      console.log(`  ${row.id.padEnd(32)} ${(row.errPct >= 0 ? '+' : '') + row.errPct.toFixed(1)} %`
        + ` при допуске ${/** @type {{pct: number}} */ (row.tolerance).pct} %`);
    }
  }
  for (const failure of summary.failures) console.log(`  ✗ ${failure}`);
  return summary;
}

/**
 * Перенос длинного текста по словам (для пояснений в отчёте).
 *
 * @param {string} text
 * @param {number} width
 * @param {string} indent
 * @returns {string}
 */
function wrap(text, width, indent) {
  /** @type {string[]} */
  const lines = [];
  let current = '';
  for (const word of text.split(' ')) {
    if (current === '') current = word;
    else if (`${current} ${word}`.length <= width) current = `${current} ${word}`;
    else {
      lines.push(current);
      current = word;
    }
  }
  if (current !== '') lines.push(current);
  return lines.map((line) => `${indent}${line}`).join('\n');
}

function main() {
  const audited = auditedRows(MEASUREMENTS);
  const excluded = renderRows(MEASUREMENTS).filter((row) => row.suspect);
  const postprocess = MEASUREMENTS.filter((row) => row.stage === 'postprocess');

  console.log('renderkit · аудит замеров (tools/fit.mjs → src/data/model.mjs)');
  console.log(`замеров всего ${MEASUREMENTS.length}: рендер ${renderRows(MEASUREMENTS).length}`
    + ` (в агрегате ${audited.length}, противоречивых ${excluded.length}), postprocess ${postprocess.length}`);
  console.log(`допуск: медиана ≤ ${DEFAULT_TOLERANCE.medianPct} %, максимум ≤ `
    + `${DEFAULT_TOLERANCE.maxPct} %; «*» — пик приведён к метрике nvidia-smi`);

  const vramRows = audited.map(vramRow).filter(/** @returns {row is AuditRow} */ (row) => row !== null);
  const timeRows = audited.map(timeRow).filter(/** @returns {row is AuditRow} */ (row) => row !== null);
  const totalRows = audited.map(totalRow).filter(/** @returns {row is AuditRow} */ (row) => row !== null);

  const vram = printTable('VRAM, пик', vramRows, 'ГиБ');
  const time = printTable('время генерации', timeRows, 'с');
  const total = printTable('итого (загрузка + энкод + генерация)', totalRows, 'с');

  if (excluded.length > 0) {
    console.log('\nне в агрегате — противоречивые замеры (см. note):');
    for (const row of excluded) {
      const measured = row.measured.genS ?? row.measured.peakVramGib ?? 0;
      const est = row.measured.genS !== undefined
        ? predictGenS(row.config, coefficients)
        : predictVram(row.config, coefficients);
      console.log(`  ${row.id.padEnd(32)} est ${est.toFixed(1)} · measured ${measured}`);
      console.log(wrap(row.note ?? '', 96, '      '));
    }
  }
  if (postprocess.length > 0) {
    console.log(`\nстадия postprocess: ${postprocess.length} (в фит и в аудит рендера не входят, A5)`);
  }

  const failures = [...vram.failures, ...time.failures, ...total.failures];
  if (failures.length > 0) {
    console.error(`\n[audit] ${failures.length} нарушений допуска`);
    process.exit(1);
  }
  console.log('\n[audit] все строки в допуске');
}

main();
