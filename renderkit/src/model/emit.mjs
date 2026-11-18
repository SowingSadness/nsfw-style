// @ts-check
/**
 * `Plan` → самодостаточный YAML-конфиг (ADR A2): **без `extends` и без имён пресетов**.
 *
 * Файл можно сохранить под любым именем и передать системе как
 * `run --config ./any-name.yaml` или `gen --preset ./any-name.yaml` — она резолвит YAML-путь
 * (`RunConfig.from_profile` принимает имя в `presets/` **или** путь к файлу). Полноту доказывает
 * round-trip: `python -m qwenstyle config show <файл> --json` даёт ровно `Plan.render`
 * (`test/roundtrip.test.mjs`, 0 расхождений).
 *
 * Что попадает в файл, а что нет:
 *
 * - **попадает всё, что влияет на кадр, память и качество** — `size_px`, `aspect`, весь блок
 *   `generation` (включая `kv_cache` и `sampler_kwargs`), весь блок `encoder`, `upscale: null`,
 *   `logs: true`. Значения, которые бэкенд объявляет сам (`repo`, `transformer`, `precision_cache`,
 *   `lora`, `encoder.*`), берутся из `inventory`, а не из головы;
 * - **не попадает ничего, что ссылается на пресет**: ни `extends`, ни `preset:`, ни имени тарифа
 *   со стадией (`тариф-стадия`) — калькулятор таких имён не знает вовсе (проверяет
 *   `test/inventory-parity.test.mjs`);
 * - `upscale` выключен **явно** (`null`), даже если базовая конфигурация системы его включает:
 *   апскейла в этой стадии нет, постобработка — отдельная стадия со своим бюджетом (A5);
 * - при `bitDepth: 16` файл остаётся валидным (никаких несуществующих ключей), а шапка честно
 *   говорит, что пайплайн пока пишет 8-битный PNG (A9).
 *
 * Модуль чистый (без `node:`/DOM/IO): эмиттер годится и браузеру (скачать `plan.yaml`), и CLI.
 */

import { INVENTORY, encoderInfo, settableKeys } from './inventory.mjs';
import { PlanError } from '../schema.mjs';
import { YamlRaw, stringifyYaml } from '../yaml.mjs';
import { formatGib } from '../units.mjs';
import { commandsFor } from './cli.mjs';

/** @typedef {import('../schema.mjs').NormalizedRequest} NormalizedRequest */

/**
 * @typedef {Object} PlanLike
 * @property {number} schemaVersion
 * @property {string} rulesetVersion
 * @property {NormalizedRequest & {tier?: string}} request
 * @property {string} verdict
 * @property {{requested: {width: number, height: number}, native: {width: number, height: number,
 *   tokens: number, sizePx: number}, aspect: string|null, bitDepth: number}} image
 * @property {any} render рендер-стадия (`null`, если плана нет — тогда YAML тоже нет)
 */

/**
 * `generation.kv_cache` в YAML: режим плана — строкой (`'true'|'offload'|'false'`), в файле
 * булевы режимы пишутся булевыми (`true`/`false`), как их принимает система.
 *
 * @param {'true'|'offload'|'false'|null} mode
 * @returns {boolean|string|null}
 */
export function kvCacheValue(mode) {
  if (mode === null || mode === undefined) return null;
  if (mode === 'true') return true;
  if (mode === 'false') return false;
  if (mode === 'offload') return 'offload';
  throw new RangeError(`kvCacheValue: неизвестный режим ${JSON.stringify(mode)}`);
}

/**
 * Конфиг плана как данные (до форматирования в YAML и до `--set`-аргументов).
 *
 * Единственный источник значений и для `emitPreset`, и для `setsFor` (`cli.mjs`): YAML и
 * `--set`-форма не могут разойтись по определению.
 *
 * @param {PlanLike} plan
 * @returns {Record<string, any>}
 */
export function presetFor(plan) {
  const render = plan.render;
  if (!render) {
    const blocked = plan.verdict === 'impossible' ? 'вердикт impossible: нативного пути нет' : 'рендер-стадия выключена';
    throw new PlanError('no-render-for-plan', 'render', `YAML не собирается: ${blocked}`);
  }
  const modelId = plan.request.hardware.model;
  const model = INVENTORY.models[modelId];
  if (!model) throw new RangeError(`emit: неизвестная модель ${JSON.stringify(modelId)}`);
  const backend = model.backend;
  const settable = new Set(settableKeys(backend));
  const encoder = encoderInfo(render.encoder.name);
  const memory = render.memory;
  const kvYaml = settable.has('generation.kv_cache') ? kvCacheValue(memory.kvCache.mode) : null;

  /** @type {Record<string, unknown>} */
  const generation = {
    backend,
    repo: model.repo,
    transformer: model.transformer,
    lora: loraBlock(model, render.loraScale),
    steps: render.steps,
    // В Qwen-Image-2.1 `true_cfg_scale` = 1.0 и CFG-ветки нет (AGENTS.md §4): значение фиксировано.
    guidance: new YamlRaw('1.0'),
    precision: render.precision,
    precision_cache: model.precision_cache,
    offload: render.offload,
  };
  if (kvYaml !== null) generation.kv_cache = kvYaml;
  generation.sampler_kwargs = samplerKwargs(render, settable);

  return {
    size_px: plan.image.native.sizePx,
    aspect: plan.image.aspect,
    generation,
    encoder: {
      name: encoder.name,
      cache_tag: encoder.cache_tag,
      repo: encoder.repo,
      revision: encoder.revision,
      local_dir: encoder.local_dir,
      requires_vision: encoder.requires_vision,
      dtype: encoder.dtype,
      // `device`/`stream`/`prefetch` — решение плана по бюджету VRAM, а не объявление бэкенда:
      // на 8 ГиБ энкодер считается на CPU (`stream: false`, `prefetch: 0`), и YAML обязан
      // воспроизводить именно это, иначе round-trip разойдётся с `Plan.render`.
      device: render.encoder.device,
      stream: render.encoder.stream,
      prefetch: render.encoder.prefetch,
      pin_memory: encoder.pin_memory,
      lookahead: encoder.lookahead,
    },
    upscale: null,
    logs: true,
  };
}

/**
 * Адаптер модели: `null` при силе 0 (нет адаптера или выключен), иначе объявление из `inventory`
 * с силой из плана.
 *
 * @param {typeof INVENTORY.models[keyof typeof INVENTORY.models]} model
 * @param {number} loraScale
 * @returns {{repo: string, filename: string, scale: number, strip_alpha: boolean} | null}
 */
function loraBlock(model, loraScale) {
  if (!model.lora || !(loraScale > 0)) return null;
  return {
    repo: model.lora.repo,
    filename: model.lora.filename,
    scale: loraScale,
    strip_alpha: model.lora.strip_alpha,
  };
}

/**
 * `generation.sampler_kwargs` — только те ключи, которые бэкенд объявил (`inventory.settable`),
 *
/**
 * `generation.sampler_kwargs` — только те ключи, которые бэкенд объявил (`inventory.settable`),
 * и только с осмысленным значением (`null` = «модель такого не умеет», например h3 без
 * `vae_tile_px`/`ref_px`/окна).
 *
 * `offload_stream` — умолчание системы (`--offload-stream 1`), выписывается явно, чтобы файл был
 * самодостаточным; `offload_blocks` берётся из плана (ось не варьируется, ADR A7).
 *
 * @param {any} render рендер-стадия `Plan.render`
 * @param {Set<string>} settable
 * @returns {Record<string, unknown>}
 */
function samplerKwargs(render, settable) {
  const memory = render.memory;
  const windowMode = memory.attention.mode === 'window';
  /** @type {Array<[string, unknown]>} */
  const candidates = [
    ['offload_blocks', render.offloadBlocks],
    ['offload_stream', true],
    ['vae_tile_px', memory.vae.tilePx],
    ['ref_px', memory.ref.px],
    ['window_attn_px', windowMode ? memory.attention.windowPx : null],
    ['window_attn_margin_px', windowMode ? memory.attention.marginPx : null],
  ];
  /** @type {Record<string, unknown>} */
  const result = {};
  for (const [key, value] of candidates) {
    if (value === null || value === undefined) continue;
    if (!settable.has(`generation.sampler_kwargs.${key}`)) continue;
    result[key] = value;
  }
  return result;
}

/**
 * `emitPreset(plan)` — YAML-текст, имя файла и команды запуска.
 *
 * @param {PlanLike} plan
 * @returns {{
 *   yaml: string,
 *   fileName: string,
 *   cli: {save: string, run: string, gen: string, checks: string[], sets: string[]},
 *   preset: Record<string, any>
 * }}
 */
export function emitPreset(plan) {
  const preset = presetFor(plan);
  const header = headerLines(plan);
  const yaml = stringifyYaml(preset, { header, comments: commentsFor(plan, preset) });
  return {
    yaml,
    fileName: fileNameFor(plan),
    cli: commandsFor(plan),
    preset,
  };
}

/**
 * Шапка-описание запроса: это **не имя пресета**, а строка «кем и для чего собран файл»
 * (`rulesetVersion` растёт при перекалибровке, поэтому по файлу видно, какими правилами он собран).
 *
 * @param {PlanLike} plan
 * @returns {string[]}
 */
export function headerLines(plan) {
  const { hardware } = plan.request;
  const { native, bitDepth } = plan.image;
  const lines = [
    `renderkit ${plan.rulesetVersion} | ${trimNumber(hardware.vramGib)} GiB VRAM / `
      + `${trimNumber(hardware.ramGib)} GiB RAM / ${hardware.disk} | ${hardware.model} | `
      + `${native.width}x${native.height} | quality=${plan.request.image.quality} | ${bitDepth} bit`,
    'самодостаточный конфиг: выписано всё, что влияет на кадр, память и качество, без ссылок на другие файлы',
    'проверка: python -m qwenstyle config show <этот файл> --json | '
      + 'запуск: run --config <этот файл> / gen --preset <этот файл>',
  ];
  if (bitDepth === 16) {
    lines.push('запрошено 16 bit: пайплайн пока сохраняет 8-bit PNG (A9) — файл валиден, райтер появится отдельно');
  }
  return lines;
}

/**
 * Комментарии — только к явно перечисленным путям (иначе YAML разрастается).
 *
 * @param {PlanLike} plan
 * @param {Record<string, any>} preset
 * @returns {Record<string, string>}
 */
function commentsFor(plan, preset) {
  /** @type {Record<string, string>} */
  const comments = {};
  const lora = preset.generation.lora;
  comments['generation.lora'] = lora
    ? `anime: адаптер FUSAL ${lora.filename} · scale ${lora.scale} (${lora.repo})`
    : 'anime: off — стилевой адаптер не подключён';
  const render = /** @type {any} */ (plan.render);
  const kv = render.memory.kvCache;
  if (kv.mode !== null) {
    comments['generation.kv_cache'] = kvComment(kv);
  }
  comments.upscale = 'апскейла нет: рендер native, постобработка — отдельная стадия со своим бюджетом (A5)';
  return comments;
}

/**
 * Короткое пояснение к `generation.kv_cache` из плана (раскладка + цена).
 *
 * @param {any} kv `Plan.render.memory.kvCache`
 * @returns {string}
 */
function kvComment(kv) {
  const gib = formatGib(kv.gib);
  if (kv.placement === 'vram') return `KV-префикс ${gib} в VRAM (быстрее всего; следи за запасом VRAM)`;
  if (kv.placement === 'ram') return `KV-префикс ${gib} в RAM: послойная подкачка, +${kv.costPct} % времени`;
  return 'KV-кеша нет: префикс пересчитывается каждый шаг (примерно вдвое дольше)';
}

/**
 * Имя файла артефакта — производное от запроса, без имён пресетов.
 *
 * @param {PlanLike} plan
 * @returns {string}
 */
export function fileNameFor(plan) {
  const { native } = plan.image;
  const { hardware } = plan.request;
  return `renderkit-${hardware.model}-${native.width}x${native.height}-`
    + `${plan.request.image.quality}-${plan.image.bitDepth}bit.yaml`;
}

/**
 * `16` вместо `16.0`, `12.5` как есть.
 *
 * @param {number} value
 * @returns {string}
 */
function trimNumber(value) {
  return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(2)));
}
