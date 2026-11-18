#!/usr/bin/env node
// @ts-check
/**
 * `tools/plan.mjs` — CLI калькулятора: запрос → `plan.yaml` (+ краткий резолв).
 *
 * ```
 * node tools/plan.mjs --width 2560 --height 1440 --quality max --bits 8 \
 *                     --vram 16 --ram 64 --disk nvme --model qwen21-base --refs 1 \
 *                     [--anime] [--steps N] [--out plan.yaml]
 * ```
 *
 * Без `--out` YAML печатается в stdout (удобно для `> file.yaml`), с `--out` — пишется в файл и
 * печатается резолв: вердикт, канвас, выход, шаги, память, время, путь. Ошибки ввода — `PlanError`
 * из `src/schema.mjs` (тот же валидатор, что у сервиса и UI), поэтому CLI не может принять запрос,
 * который не принял бы `computePlan`.
 *
 * Коды выхода: `0` — план есть и (при `--out`) файл записан; `1` — вердикт `impossible`, YAML не
 * собирается; `2` — ошибка запроса/аргументов.
 *
 * Артефакты пишутся ровно туда, куда сказал `--out`; ничего другого скрипт не создаёт.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

import { computePlan } from '../src/index.mjs';
import { emitPreset } from '../src/model/emit.mjs';
import { PlanError } from '../src/schema.mjs';
import { formatGib, formatSeconds } from '../src/units.mjs';

const USAGE = `node tools/plan.mjs --width W --height H [--quality draft|standard|max] [--bits 8|16]
                     [--vram GIB] [--ram GIB] [--disk hdd|ssd|nvme]
                     [--model qwen21-base|qwen21-turbo|h3-image] [--refs N] [--anime]
                     [--steps N] [--free-disk GIB] [--out PATH]

  --width, --height  целевой кадр в пикселях (обязательные)
  --quality          политика риска и потолка: draft | standard | max (по умолчанию max)
  --bits             битность выходного файла: 8 | 16 (по умолчанию 8; 16 — unsupported, A9)
  --vram / --ram     бюджет ГиБ (по умолчанию 16 / 64)
  --disk             класс накопителя: hdd | ssd | nvme (по умолчанию nvme)
  --model            трансформер: qwen21-base | qwen21-turbo | h3-image (по умолчанию qwen21-base)
  --refs             число референсов, 0 = T2I (по умолчанию 1)
  --anime            включить стилевой адаптер FUSAL (если модель его поддерживает)
  --steps            явный override шагов (на память не влияет; h3/turbo фиксированы расписанием)
  --free-disk        свободно на диске, ГиБ (проверка места)
  --out              путь к YAML; без флага YAML печатается в stdout
  --help             эта справка`;

/** @typedef {{[key: string]: string|boolean}} RawArgs */

/**
 * Разобрать аргументы: `--key value`, `--key=value` и флаги `--anime`.
 *
 * @param {string[]} argv
 * @returns {RawArgs}
 */
function parseArgs(argv) {
  /** @type {RawArgs} */
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (token === '--help' || token === '-h') {
      args.help = true;
      continue;
    }
    if (!token.startsWith('--')) {
      throw new PlanError('unknown-argument', '', `лишний аргумент ${JSON.stringify(token)}`, null);
    }
    const eq = token.indexOf('=');
    if (eq !== -1) {
      args[token.slice(2, eq)] = token.slice(eq + 1);
      continue;
    }
    const name = token.slice(2);
    const next = argv[index + 1];
    if (name === 'anime') {
      args.anime = true;
      continue;
    }
    if (next === undefined || next.startsWith('--')) {
      throw new PlanError('missing-value', name, `флаг --${name} требует значение`, null);
    }
    args[name] = next;
    index += 1;
  }
  return args;
}

/**
 * Числовой аргумент (строка → число; мусор → `PlanError`).
 *
 * @param {RawArgs} args
 * @param {string} name
 * @param {number} fallback
 * @returns {number}
 */
function numeric(args, name, fallback) {
  const raw = args[name];
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value)) {
    throw new PlanError('NotNumber', name, `ожидается число, получено ${JSON.stringify(raw)}`, null);
  }
  return value;
}

/**
 * Собрать запрос (`RenderRequest`) из аргументов.
 *
 * @param {RawArgs} args
 * @returns {any}
 */
function requestFromArgs(args) {
  if (args.width === undefined || args.height === undefined) {
    throw new PlanError('MissingField', 'image.width', 'нужны --width и --height целевого кадра', null);
  }
  return {
    hardware: {
      model: typeof args.model === 'string' ? args.model : 'qwen21-base',
      vramGib: numeric(args, 'vram', 16),
      ramGib: numeric(args, 'ram', 64),
      disk: typeof args.disk === 'string' ? args.disk : 'nvme',
    },
    image: {
      width: numeric(args, 'width', 0),
      height: numeric(args, 'height', 0),
      quality: typeof args.quality === 'string' ? args.quality : 'max',
      bitDepth: numeric(args, 'bits', 8),
      ...(args.steps === undefined ? {} : { steps: numeric(args, 'steps', 0) }),
    },
    refs: numeric(args, 'refs', 1),
    anime: args.anime === true,
    ...(args['free-disk'] === undefined ? {} : { freeDiskGib: numeric(args, 'free-disk', 0) }),
  };
}

/**
 * Краткий резолв для человека (по одной строке на аспект).
 *
 * @param {any} plan
 * @param {{fileName: string, yaml: string, cli: {checks: string[]}}} emitted
 * @param {string} outPath
 * @returns {string[]}
 */
function summaryLines(plan, emitted, outPath) {
  const render = plan.render;
  const memory = render.memory;
  const estimates = plan.estimates;
  const { native } = plan.image;
  const payload = estimates.disk.pngPerFrameMb * estimates.disk.bitDepthFactor;
  const canvas = `${native.width}x${native.height}`;
  const aspect = plan.image.aspect === null ? 'по пропорции референса' : plan.image.aspect;
  const kv = memory.kvCache.mode === null
    ? 'n/a (модель без KV-кеша)'
    : `${memory.kvCache.mode} (${formatGib(memory.kvCache.gib)} `
      + `${memory.kvCache.placement === 'ram' ? 'в RAM' : 'в VRAM'}, +${memory.kvCache.costPct} % времени)`;
  const vae = memory.vae.tilePx === null ? 'n/a' : `${memory.vae.tilePx}/${memory.vae.stridePx}`;
  const attention = memory.attention.mode === 'window'
    ? `window ${memory.attention.windowPx}+${memory.attention.marginPx} (${memory.attention.bands} полос, `
      + `покрытие ${(memory.attention.coverage * 100).toFixed(0)} %)`
    : 'dense';
  const lora = render.loraScale > 0 ? `FUSAL ${render.loraScale}` : 'off';
  return [
    `[plan] verdict   ${plan.verdict} · confidence ${plan.confidence}`,
    `[plan] canvas    ${canvas} (${native.tokens} токенов) · aspect ${aspect} · `
      + `выход ${canvas} (png, ${plan.image.bitDepth} bit, ~${payload.toFixed(2)} МБ)`,
    `[plan] шаги      ${render.steps} · offload ${render.offload} · precision ${render.precision} · LoRA ${lora}`,
    `[plan] память    KV ${kv} · тайл VAE ${vae} · внимание ${attention} · ref_px ${memory.ref.px}`,
    `[plan] VRAM      ${formatGib(estimates.vram.peakGib)} / ${formatGib(estimates.vram.budgetGib)} `
      + `(запас ${formatGib(estimates.vram.headroomGib)}) · RAM ${formatGib(estimates.ram.peakGib)} / `
      + `${formatGib(estimates.ram.budgetGib)}`,
    `[plan] время     загрузка ${formatSeconds(estimates.time.loadS)} · энкод ${formatSeconds(estimates.time.encodeColdS)} · `
      + `кадр ${formatSeconds(estimates.time.genPerFrameS)} · первый кадр ${formatSeconds(estimates.time.firstFrameS)}`,
    `[plan] файл      ${outPath} (имя по запросу: ${emitted.fileName}, ${emitted.yaml.length} байт)`,
    `[plan] проверка  ${emitted.cli.checks[0]}`,
  ];
}

/**
 * Точка входа.
 *
 * @param {string[]} argv
 * @returns {number} код выхода
 */
function main(argv) {
  /** @type {RawArgs} */
  let args;
  try {
    args = parseArgs(argv);
  } catch (error) {
    return fail(error);
  }
  if (args.help) {
    console.log(USAGE);
    return 0;
  }
  /** @type {any} */
  let plan;
  try {
    plan = computePlan(requestFromArgs(args));
  } catch (error) {
    return fail(error);
  }

  if (!plan.render) {
    const reasons = plan.degradations.map((entry) => `${entry.code}: ${entry.reason}`).join('\n  ');
    console.error('[plan] impossible: нативного пути нет — YAML не собирается');
    const stageReason = plan.stages?.[0]?.reason;
    if (stageReason) console.error(`  ${stageReason}`);
    if (reasons) console.error(`  ${reasons}`);
    for (const suggestion of plan.suggestions) {
      console.error(`[plan] подсказка: ${suggestion.code}: ${suggestion.detail}`);
    }
    return 1;
  }

  const emitted = emitPreset(plan);
  const outPath = typeof args.out === 'string' ? resolve(args.out) : null;
  if (outPath === null) {
    process.stdout.write(emitted.yaml);
    return 0;
  }
  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, emitted.yaml, 'utf8');
  for (const line of summaryLines(plan, emitted, outPath)) console.log(line);
  const codes = plan.warnings.map((warning) => warning.code);
  if (codes.length > 0) console.log(`[plan] warnings  ${codes.join(', ')}`);
  return 0;
}

/**
 * Ошибка запроса/аргументов по-человечески (с `allowed`, если он есть).
 *
 * @param {unknown} error
 * @returns {number} код выхода 2
 */
function fail(error) {
  if (error instanceof PlanError) {
    console.error(`[plan] ${error.code}: ${error.field || '<request>'}: ${error.detail}`);
    if (error.allowed) console.error(`[plan] допустимо: ${error.allowed.join(', ')}`);
    return 2;
  }
  throw error;
}

process.exitCode = main(process.argv.slice(2));
