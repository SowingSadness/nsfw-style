// @ts-check
/**
 * Команды запуска для YAML-артефакта и `--set`-форма тех же настроек (ADR A2/A10).
 *
 * Здесь два независимых, но согласованных артефакта одного плана:
 *
 * 1. **команды** (`run` / `gen` / `config show`) — то, что пользователь копирует из UI или печатает
 *    `tools/plan.mjs`. Имя файла в командах — плейсхолдер `./plan.yaml`: файл можно сохранить под
 *    любым именем (проверено: `--preset` с `.yaml` — это путь, а не имя пресета);
 * 2. **`--set`-форма** (`setsFor`/`setArgs`) — те же настройки в виде
 *    `--set generation.steps=25 …`. Ключи проверяются по whitelist бэкенда
 *    (`inventory.settableKeys`, то есть `config show --explain-params`) плюс поля `RunConfig`,
 *    которые система принимает без объявления в бэкенде (`size_px`, `aspect`,
 *    `upscale.target_height`). Неизвестный ключ — `PlanError{code:'unknown-set-key'}` со списком
 *    `allowed`: опечатка не должна молча превратиться в «настройка не применилась».
 *
 * Набор кадров в калькуляторе не считается (Phase A: один кадр, набор умножает сервис), поэтому
 * `gen`-команда всегда несёт `--count 1`, а `--count > 1` отвергается `PlanError`
 * (`frames-not-supported`).
 *
 * Модуль чистый: без `node:`/DOM/IO.
 */

import { PlanError } from '../schema.mjs';
import { INVENTORY, settableKeys } from './inventory.mjs';
import { YamlRaw } from '../yaml.mjs';

/**
 * Плейсхолдер имени файла в командах: конфиг можно сохранить под любым именем.
 * @type {string}
 */
export const PLAN_FILE = './plan.yaml';

/**
 * Поля `RunConfig`, которые не объявляются бэкендом, но задаются точечно (`--set size_px=1440`).
 * `upscale.target_height` — потому что апскейл в калькуляторе выключается, а не настраивается.
 *
 * @type {ReadonlyArray<string>}
 */
export const CORE_SET_KEYS = Object.freeze(['size_px', 'aspect', 'upscale.target_height']);

/**
 * Допустимые `--set`-пути бэкенда: объявленные `ParameterSpec` (не-`readonly`) + поля `RunConfig`.
 *
 * @param {string} backend
 * @returns {string[]}
 */
export function allowedSetKeys(backend) {
  return [...settableKeys(backend), ...CORE_SET_KEYS];
}

/**
 * Проверить ключ `--set` и вернуть его (для цепочки).
 *
 * @param {string} backend
 * @param {string} key
 * @returns {string}
 * @throws {PlanError} `unknown-set-key` со списком `allowed`
 */
export function assertSetKey(backend, key) {
  const allowed = allowedSetKeys(backend);
  if (!allowed.includes(key)) {
    throw new PlanError(
      'unknown-set-key',
      key,
      `бэкенд ${backend} не объявляет такой параметр (см. \`config show --explain-params\`)`,
      allowed,
    );
  }
  return key;
}

/**
 * Можно ли задать ключ этому бэкенду (не бросает: нужно для перебора осей плана).
 *
 * Ключ может быть объявлен другим бэкендом и при этом быть `readonly` у этого — например
 * `generation.steps` у `minimax_h3_image` (6 шагов PDD фиксированы): такой ключ просто не
 * попадает в `--set`-форму, но и ошибкой не считается.
 *
 * @param {string} backend
 * @param {string} key
 * @returns {boolean}
 */
export function isSetKey(backend, key) {
  return allowedSetKeys(backend).includes(key);
}

/**
 * Настройки плана в виде `--set`-пар (`{key, value}`), все ключи — из whitelist.
 *
 * Источник значений — `Plan` (`render` + `image`), а не YAML-текст: обе формы собираются из одного
 * плана, и их согласованность проверяет `test/emit.test.mjs` (значения `--set` равны значениям
 * соответствующих путей YAML).
 *
 * @param {any} plan `Plan` из `computePlan`
 * @param {Record<string, unknown> | null} [overrides] дополнительные пары «ключ → значение»;
 *   каждый ключ проходит `assertSetKey` (опечатка → `PlanError{code:'unknown-set-key'}`)
 * @returns {Array<{key: string, value: unknown}>}
 */
export function setsFor(plan, overrides = null) {
  const render = plan.render;
  if (!render) throw new PlanError('no-render-for-plan', 'render', 'настроек рендера нет: план без рендер-стадии');
  const backend = backendOf(plan);
  const settable = new Set(settableKeys(backend));
  /** @type {Array<{key: string, value: unknown}>} */
  const entries = [];
  /**
   * @param {string} key
   * @param {unknown} value
   */
  const push = (key, value) => {
    if (value === null || value === undefined) return;
    // Ключ, который бэкенд не объявляет (h3: шаги/guidance/kv_cache), — не ошибка, а «не выписывается».
    if (!isSetKey(backend, key)) return;
    assertSetKey(backend, key);
    entries.push({ key, value });
  };

  push('size_px', plan.image.native.sizePx);
  push('aspect', plan.image.aspect);
  push('generation.steps', render.steps);
  push('generation.guidance', 1);
  push('generation.offload', render.offload);
  push('generation.precision', render.precision);
  push('generation.kv_cache', kvSetValue(render.memory.kvCache.mode));
  push('generation.sampler_kwargs.offload_blocks', render.offloadBlocks);
  if (settable.has('generation.sampler_kwargs.offload_stream')) {
    push('generation.sampler_kwargs.offload_stream', true);
  }
  const windowMode = render.memory.attention.mode === 'window';
  /** @type {Array<[string, unknown]>} */
  const knobs = [
    ['vae_tile_px', render.memory.vae.tilePx],
    ['ref_px', render.memory.ref.px],
    ['window_attn_px', windowMode ? render.memory.attention.windowPx : null],
    ['window_attn_margin_px', windowMode ? render.memory.attention.marginPx : null],
  ];
  for (const [name, value] of knobs) {
    const key = `generation.sampler_kwargs.${name}`;
    if (!settable.has(key)) continue;
    push(key, value);
  }
  for (const [key, value] of Object.entries(overrides ?? {})) {
    assertSetKey(backend, key);
    entries.push({ key, value });
  }
  return entries;
}

/**
 * `--set`-форма: `['generation.steps=25', 'size_px=1440', …]`.
 *
 * @param {any} plan
 * @returns {string[]}
 */
export function setArgs(plan) {
  return setsFor(plan).map(({ key, value }) => `${key}=${formatSetValue(value)}`);
}

/**
 * Значение в виде, который принимает `--set key=value` (`true`/`false`/`null`/`1.0`).
 *
 * @param {unknown} value
 * @returns {string}
 */
export function formatSetValue(value) {
  if (value instanceof YamlRaw) return value.text;
  if (value === null) return 'null';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  return String(value);
}

/**
 * Команды для `Plan.cli`: `save` (как пересобрать файл), `run` (`--config`), `gen` (`--preset`),
 * `checks` (авторитетная проверка без весов) и `sets` (`--set`-форма тех же настроек).
 *
 * @param {any} plan `Plan` из `computePlan`
 * @param {{count?: number}} [options] `count` — число кадров; калькулятор считает один
 * @returns {{save: string, run: string, gen: string, checks: string[], sets: string[]}}
 */
export function commandsFor(plan, options = {}) {
  const render = plan.render;
  if (!render) throw new PlanError('no-render-for-plan', 'render', 'команд запуска нет: план без рендер-стадии');
  const count = validateCount(options.count ?? 1);
  const { hardware, image } = plan.request;
  const refs = plan.request.refs;

  const run = ['python -m qwenstyle run --config', PLAN_FILE];
  // Без референсов это T2I: `--input` не передаётся вовсе (модель не требует картинку, A5).
  if (refs > 0) run.push('--input <png>');
  run.push('--out out/');

  const saveParts = [
    'npm run plan --',
    `--width ${image.width}`,
    `--height ${image.height}`,
    `--quality ${image.quality}`,
    `--bits ${plan.image.bitDepth}`,
    `--vram ${hardware.vramGib}`,
    `--ram ${hardware.ramGib}`,
    `--disk ${hardware.disk}`,
    `--model ${hardware.model}`,
    `--refs ${refs}`,
  ];
  if (plan.request.anime) saveParts.push('--anime');
  saveParts.push('--out plan.yaml');

  return {
    save: saveParts.join(' '),
    run: run.join(' '),
    gen: `python -m qwenstyle gen --preset ${PLAN_FILE} --prompt <project>/<stem> --count ${count}`,
    checks: [`python -m qwenstyle config show ${PLAN_FILE} --json`],
    sets: setArgs(plan),
  };
}

/**
 * Число кадров в калькуляторе — ровно один (`frames` в запросе не принимается, ADR A3).
 *
 * @param {number} count
 * @returns {number}
 * @throws {PlanError} `frames-not-supported` при `count > 1`, `invalid-count` при мусоре
 */
export function validateCount(count) {
  if (!Number.isInteger(count) || count < 1) {
    throw new PlanError('invalid-count', 'count', 'число кадров должно быть целым ≥ 1', [1]);
  }
  if (count > 1) {
    throw new PlanError(
      'frames-not-supported',
      'count',
      'калькулятор считает один кадр: набор кадров умножает сервис, а не конфиг (Phase A)',
      [1],
    );
  }
  return count;
}

/**
 * `generation.kv_cache` для `--set`: булевы режимы — словами (`true`/`false`), `offload` — как есть.
 *
 * @param {'true'|'offload'|'false'|null} mode
 * @returns {string|null}
 */
function kvSetValue(mode) {
  if (mode === null || mode === undefined) return null;
  return mode;
}

/**
 * Бэкенд модели из плана (id модели → `backend` в `inventory`): `Plan` не дублирует это поле.
 *
 * @param {any} plan
 * @returns {string}
 */
function backendOf(plan) {
  const modelId = plan.request.hardware.model;
  const model = INVENTORY.models[modelId];
  if (!model) {
    throw new PlanError('unknown-model', 'hardware.model', `в инвентаре нет модели ${JSON.stringify(modelId)}`);
  }
  return model.backend;
}
