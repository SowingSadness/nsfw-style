// @ts-check
/**
 * Контракт стадии постобработки (шаг 5 плана, ADR A5).
 *
 * Постобработка — **отдельная стадия и отдельная модель**, а не апскейл внутри рендера. В Phase A
 * модель не передана, поэтому стадия честно выключена: `estimates: null`, `paramsSchema: null`,
 * `reason: «модель постобработки не передана»`. Бюджеты стадий не смешиваются: пик VRAM считается
 * по стадиям (пайплайн выгружается между ними — измерено: общий пик прогона с SeedVR2 8.5 ГиБ,
 * меньше суммы стадий).
 *
 * Когда модель появится, она приносит **свой** набор параметров (`ParameterSpec`-подобные
 * дескрипторы: `scope|name|type|default|choices|readonly` — та же схема, что у бэкендов
 * генерации/апскейла), и `validatePostprocessParams` проверяет `postprocess.params` **по её набору**,
 * а не по полям ядра. Имени модели в коде калькулятора нет ни в одной ветке: только данные.
 *
 * Модуль изоморфный: без `node:`/DOM.
 */

import { PlanError } from '../schema.mjs';

/**
 * @typedef {Object} PostprocessParamSpec объявленный моделью параметр (подмножество `ParameterSpec`)
 * @property {string} scope `postprocess`
 * @property {string} name имя параметра (`precision`, `tile`, …)
 * @property {'int'|'float'|'bool'|'enum'|'string'} type тип
 * @property {unknown} default значение по умолчанию
 * @property {ReadonlyArray<unknown>|null} [choices] допустимые значения для `enum`
 * @property {boolean} [readonly] параметр нельзя переопределять
 * @property {string} [doc] пояснение для UI
 */

/**
 * @typedef {Object} PostprocessModelDescriptor описание модели постобработки
 * @property {string} model id модели (`seedvr2-3b` и т.п.)
 * @property {PostprocessParamSpec[]} params объявленный набор параметров
 */

/**
 * @typedef {Object} PostprocessStage
 * @property {'postprocess'} stage
 * @property {boolean} enabled считается ли стадия в этом плане
 * @property {string} reason почему стадия в таком состоянии
 * @property {null} estimates оценки стадии (`null` — стадия не считается; бюджеты стадий не смешиваются)
 * @property {ReadonlyArray<PostprocessParamSpec>|null} paramsSchema набор параметров модели (`null` — модели нет)
 * @property {string|null} model id переданной модели (`null` в Phase A)
 */

/** Причина выключенной стадии в Phase A (одна строка, её показывает UI). */
export const POSTPROCESS_DISABLED_REASON = 'модель постобработки не передана';

/**
 * Стадия постобработки для плана. В Phase A всегда выключена: модели постобработки нет в инвентаре,
 * поэтому `estimates` и `paramsSchema` честно `null`, а не «нули». Запрос стадии при этом **принимается**
 * (`schema.normalizePostprocess`): план собирается, в `warnings` появляется `postprocess-not-connected`
 * с пометкой, что запрос был, а в `stages[1]` видно, какая именно модель запрошена.
 *
 * @param {{postprocess?: null|{model?: string}|undefined}} request нормализованный запрос
 * @param {unknown} [render] рендер-стадия (не используется: бюджеты стадий независимы)
 * @param {unknown} [hardware] железо (не используется: бюджет стадии считает её модель)
 * @returns {PostprocessStage}
 */
export function resolvePostprocess(request, render, hardware) {
  void render;
  void hardware;
  const requested = request?.postprocess ?? null;
  if (requested === null) {
    return {
      stage: 'postprocess',
      enabled: false,
      reason: POSTPROCESS_DISABLED_REASON,
      estimates: null,
      paramsSchema: null,
      model: null,
    };
  }
  return {
    stage: 'postprocess',
    enabled: false,
    reason: 'модель постобработки не подключена',
    estimates: null,
    paramsSchema: null,
    model: typeof requested.model === 'string' ? requested.model : null,
  };
}

/**
 * Валидация `postprocess.params` по объявленному набору модели (`A5`: модель приносит свои
 * параметры). Неизвестный параметр — `PlanError` с `code: 'unknown-postprocess-param'` и
 * `allowed` — списком имён из `modelDescriptor.params`.
 *
 * @param {unknown} params присланные значения (`null`/`undefined` — только умолчания)
 * @param {PostprocessModelDescriptor|null|undefined} modelDescriptor описание модели
 * @returns {{model: string, values: Record<string, unknown>}} значения в порядке объявления
 * @throws {PlanError} `postprocess-model-not-provided` | `invalid-postprocess-params` |
 *   `unknown-postprocess-param` | `readonly-postprocess-param` | `invalid-postprocess-param`
 */
export function validatePostprocessParams(params, modelDescriptor) {
  if (!modelDescriptor || typeof modelDescriptor !== 'object' || !Array.isArray(modelDescriptor.params)) {
    throw new PlanError(
      'postprocess-model-not-provided',
      'postprocess.model',
      'параметры постобработки валидируются по набору объявленной модели, а модель не передана',
      [null],
    );
  }
  const specs = modelDescriptor.params;
  const allowed = specs.map((spec) => spec.name);
  const provided = params === null || params === undefined ? {} : params;
  if (typeof provided !== 'object' || Array.isArray(provided)) {
    throw new PlanError(
      'invalid-postprocess-params',
      'postprocess.params',
      `ожидается объект { ${allowed.join(', ')} }, получено ${JSON.stringify(provided)}`,
      allowed,
    );
  }
  for (const key of Object.keys(provided)) {
    if (!allowed.includes(key)) {
      throw new PlanError(
        'unknown-postprocess-param',
        `postprocess.params.${key}`,
        `параметр не объявлен моделью ${modelDescriptor.model}; допустимые: ${allowed.join(', ')}`,
        allowed,
      );
    }
  }
  /** @type {Record<string, unknown>} */
  const values = {};
  for (const spec of specs) {
    const sent = /** @type {Record<string, unknown>} */ (provided)[spec.name];
    if (sent === undefined) {
      values[spec.name] = spec.default;
      continue;
    }
    if (spec.readonly === true) {
      throw new PlanError(
        'readonly-postprocess-param',
        `postprocess.params.${spec.name}`,
        `параметр модели ${modelDescriptor.model} только для чтения`,
        [spec.default],
      );
    }
    values[spec.name] = checkParamValue(spec, sent, modelDescriptor.model);
  }
  return { model: modelDescriptor.model, values };
}

/**
 * Проверка одного значения по объявленному типу/`choices`.
 *
 * @param {PostprocessParamSpec} spec
 * @param {unknown} value
 * @param {string} model
 * @returns {unknown}
 */
function checkParamValue(spec, value, model) {
  const field = `postprocess.params.${spec.name}`;
  if (spec.type === 'enum') {
    const choices = spec.choices ?? [];
    if (!choices.includes(value)) {
      throw new PlanError(
        'invalid-postprocess-param',
        field,
        `модель ${model} объявляет ${spec.name} из ${choices.map((v) => JSON.stringify(v)).join(', ')}`,
        choices,
      );
    }
    return value;
  }
  if (spec.type === 'int' || spec.type === 'float') {
    if (typeof value !== 'number' || !Number.isFinite(value) || (spec.type === 'int' && !Number.isInteger(value))) {
      throw new PlanError(
        'invalid-postprocess-param',
        field,
        `модель ${model} объявляет ${spec.name} как ${spec.type}, получено ${JSON.stringify(value)}`,
        [spec.default],
      );
    }
    return value;
  }
  if (spec.type === 'bool') {
    if (typeof value !== 'boolean') {
      throw new PlanError(
        'invalid-postprocess-param',
        field,
        `модель ${model} объявляет ${spec.name} как bool, получено ${JSON.stringify(value)}`,
        [spec.default],
      );
    }
    return value;
  }
  if (typeof value !== 'string') {
    throw new PlanError(
      'invalid-postprocess-param',
      field,
      `модель ${model} объявляет ${spec.name} как string, получено ${JSON.stringify(value)}`,
      [spec.default],
    );
  }
  return value;
}
