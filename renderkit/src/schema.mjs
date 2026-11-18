// @ts-check
/**
 * Схема входа (`RenderRequest`): нормализация и валидация «image-first» запроса.
 *
 * Всё, что приходит снаружи (UI, CLI, будущий сервис), проходит через `normalizeRequest()`
 * и превращается в **нормализованный запрос** с проставленными умолчаниями. Любое расхождение
 * с таблицей «Вход (`RenderRequest`)» плана — `PlanError` с машиночитаемыми `code`/`field`/`allowed`.
 *
 * Модуль чистый: никаких зависимостей, никакого ввода-вывода.
 */

/** Код схемы: растёт при несовместимом изменении контракта запроса. */
export const SCHEMA_VERSION = 1;

/** Уровни качества: политика риска и потолка, а не шаги (шаги — `levels.mjs`). */
export const QUALITIES = Object.freeze(['draft', 'standard', 'max']);

/** Модели генерации, которые калькулятор умеет считать. */
export const MODELS = Object.freeze(['qwen21-base', 'qwen21-turbo', 'h3-image']);

/** Классы накопителя: влияют только на время чтения весов. */
export const DISKS = Object.freeze(['hdd', 'ssd', 'nvme']);

/** Битность выходного файла; 16 бит пайплайном пока не поддерживается (verdict `unsupported`). */
export const BIT_DEPTHS = Object.freeze([8, 16]);

/** Режимы KV-кеша, которые понимает пайплайн (`generation.kv_cache`). */
export const KV_CACHE_MODES = Object.freeze([true, false, 'offload', 'auto-offload-ref2']);

/** Тайлы декодера VAE, поддерживаемые системой (`512/384`, `256/192`, `1024/768`, без тайлинга). */
export const VAE_TILES = Object.freeze([-1, 0, 512, 1024]);

/** Границы числовых полей (из таблицы входа плана). */
export const LIMITS = Object.freeze({
  width: Object.freeze({ min: 256, max: 8192 }),
  height: Object.freeze({ min: 256, max: 8192 }),
  vramGib: Object.freeze({ min: 4, max: 96 }),
  ramGib: Object.freeze({ min: 8, max: 512 }),
  refs: Object.freeze({ min: 0, max: 10 }),
  steps: Object.freeze({ min: 4, max: 50 }),
});

/** Умолчания, которые проставляются при отсутствии поля. */
export const DEFAULTS = Object.freeze({
  bitDepth: 8,
  refs: 1,
  anime: false,
  freeDiskGib: null,
});

/**
 * Разрешённые ключи по уровням: всё остальное — `UnknownField`.
 * `frames` в списке нет осознанно: Phase A считает **один** кадр, набор умножает сервис.
 */
const ALLOWED_KEYS = Object.freeze({
  root: Object.freeze(['hardware', 'image', 'refs', 'anime', 'freeDiskGib', 'memory', 'postprocess']),
  hardware: Object.freeze(['model', 'vramGib', 'ramGib', 'disk']),
  image: Object.freeze(['width', 'height', 'quality', 'bitDepth', 'steps']),
  memory: Object.freeze(['kvCache', 'vaeTilePx', 'window']),
  window: Object.freeze(['windowPx', 'marginPx']),
  postprocess: Object.freeze(['model', 'params']),
});

/** Нижняя граница для окна внимания (одна латентная клетка сетки). */
const CANVAS_GRID_MIN = 32;

/**
 * Ошибка запроса: машиночитаемая (можно ветвиться по `code`/`field`) и одновременно
 * показываемая человеку (`message`).
 */
export class PlanError extends Error {
  /**
   * @param {string} code машинный код (`NotInteger`, `InvalidEnum`, …)
   * @param {string} field путь поля в запросе (`image.width`); `''` — корень
   * @param {string} detail пояснение по-русски
   * @param {ReadonlyArray<unknown>|null} [allowed] допустимые значения, если код про перечисление
   */
  constructor(code, field, detail, allowed = null) {
    super(`${field || '<request>'}: ${code}: ${detail}`);
    this.name = 'PlanError';
    this.code = code;
    this.field = field;
    this.detail = detail;
    this.allowed = allowed;
  }

  /** @returns {{code: string, field: string, allowed: ReadonlyArray<unknown>|null, message: string}} */
  toJSON() {
    return { code: this.code, field: this.field, allowed: this.allowed, message: this.message };
  }
}

/**
 * @typedef {Object} NormalizedMemory
 * @property {boolean|'offload'|'auto-offload-ref2'|null} kvCache null = авто
 * @property {number|null} vaeTilePx null = авто (`512`)
 * @property {null|{enabled: boolean, windowPx: number|null, marginPx: number|null}} window null = авто
 */

/**
 * Запрос стадии постобработки (ADR A5): **модель приносит свои параметры**.
 * В Phase A моделей постобработки в инвентаре нет, поэтому запрос принимается и доезжает до плана
 * (стадия остаётся выключенной с явной причиной и `postprocess-not-connected`), а не падает
 * `PhaseNotSupported`: калькулятор не должен терять весь план из-за того, что модель ещё не подключена.
 * `params` валидируются позже по объявленному набору модели (`validatePostprocessParams`).
 *
 * @typedef {Object} NormalizedPostprocess
 * @property {string} model id модели постобработки
 * @property {Record<string, unknown>|null} params её параметры (`null` — только умолчания модели)
 */

/**
 * @typedef {Object} NormalizedRequest
 * @property {{model: string, vramGib: number, ramGib: number, disk: string}} hardware
 * @property {{width: number, height: number, quality: string, bitDepth: number, steps: number|null}} image
 * @property {number} refs
 * @property {boolean} anime
 * @property {number|null} freeDiskGib
 * @property {NormalizedMemory} memory
 * @property {NormalizedPostprocess|null} postprocess `null` — стадия не запрошена
 */

/** @param {unknown} value @returns {value is Record<string, any>} */
function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Проверка на неизвестные ключи внутри объекта.
 *
 * @param {Record<string, any>} value
 * @param {ReadonlyArray<string>} allowed
 * @param {string} prefix
 * @returns {void}
 */
function rejectUnknownKeys(value, allowed, prefix) {
  for (const key of Object.keys(value)) {
    if (!allowed.includes(key)) {
      const field = prefix ? `${prefix}.${key}` : key;
      throw new PlanError('UnknownField', field, `поле не принимается; допустимые: ${allowed.join(', ')}`, allowed);
    }
  }
}

/**
 * @param {unknown} value
 * @param {string} field
 * @returns {number}
 */
function requireInteger(value, field) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new PlanError('NotInteger', field, `ожидается целое число, получено ${JSON.stringify(value)}`);
  }
  if (!Number.isInteger(value)) {
    throw new PlanError('NotInteger', field, `ожидается целое число, получено ${value}`);
  }
  return value;
}

/**
 * @param {unknown} value
 * @param {string} field
 * @returns {number}
 */
function requireNumber(value, field) {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new PlanError('NotNumber', field, `ожидается число, получено ${JSON.stringify(value)}`);
  }
  return value;
}

/**
 * @param {number} value
 * @param {{min: number, max: number}} range
 * @param {string} field
 * @returns {number}
 */
function requireRange(value, range, field) {
  if (value < range.min || value > range.max) {
    throw new PlanError(
      'OutOfRange',
      field,
      `вне диапазона ${range.min}…${range.max}: ${value}`,
      /** @type {ReadonlyArray<unknown>} */ ([range.min, range.max]),
    );
  }
  return value;
}

/**
 * @param {unknown} value
 * @param {ReadonlyArray<unknown>} allowed
 * @param {string} field
 * @returns {any}
 */
function requireEnum(value, allowed, field) {
  if (!allowed.includes(value)) {
    throw new PlanError(
      'InvalidEnum',
      field,
      `недопустимое значение ${JSON.stringify(value)}; допустимые: ${allowed.map((v) => JSON.stringify(v)).join(', ')}`,
      allowed,
    );
  }
  return value;
}

/**
 * Нормализация и валидация запроса.
 *
 * Дефолты: `image.bitDepth = 8`, `refs = 1`, `anime = false`, `freeDiskGib = null`,
 * `memory.*` = `null` («авто»), `postprocess = null` (стадия не запрошена).
 *
 * @param {unknown} input произвольный вход (JSON из UI/сервиса, объект из CLI)
 * @returns {NormalizedRequest}
 * @throws {PlanError} при первом же нарушении контракта (порядок полей фиксирован)
 */
export function normalizeRequest(input) {
  if (!isPlainObject(input)) {
    throw new PlanError('NotObject', '', `ожидается объект запроса, получено ${JSON.stringify(input)}`);
  }
  rejectUnknownKeys(input, ALLOWED_KEYS.root, '');

  const rawImage = input.image;
  if (!isPlainObject(rawImage)) {
    throw new PlanError('MissingField', 'image', 'обязательный объект { width, height, quality }', ['image.width', 'image.height', 'image.quality']);
  }
  const rawHardware = input.hardware;
  if (!isPlainObject(rawHardware)) {
    throw new PlanError('MissingField', 'hardware', 'обязательный объект { model, vramGib, ramGib, disk }', ['hardware.model', 'hardware.vramGib', 'hardware.ramGib', 'hardware.disk']);
  }
  rejectUnknownKeys(rawImage, ALLOWED_KEYS.image, 'image');
  rejectUnknownKeys(rawHardware, ALLOWED_KEYS.hardware, 'hardware');

  // --- image -------------------------------------------------------------
  if (rawImage.width === undefined) {
    throw new PlanError('MissingField', 'image.width', `обязательное целое ${LIMITS.width.min}…${LIMITS.width.max}`);
  }
  if (rawImage.height === undefined) {
    throw new PlanError('MissingField', 'image.height', `обязательное целое ${LIMITS.height.min}…${LIMITS.height.max}`);
  }
  const width = requireRange(requireInteger(rawImage.width, 'image.width'), LIMITS.width, 'image.width');
  const height = requireRange(requireInteger(rawImage.height, 'image.height'), LIMITS.height, 'image.height');

  if (rawImage.quality === undefined) {
    throw new PlanError('MissingField', 'image.quality', `обязательный уровень качества; допустимые: ${QUALITIES.join(', ')}`, QUALITIES);
  }
  const quality = requireEnum(rawImage.quality, QUALITIES, 'image.quality');

  const bitDepth = requireEnum(
    rawImage.bitDepth === undefined ? DEFAULTS.bitDepth : rawImage.bitDepth,
    BIT_DEPTHS,
    'image.bitDepth',
  );

  let steps = null;
  if (rawImage.steps !== undefined) {
    steps = requireRange(requireInteger(rawImage.steps, 'image.steps'), LIMITS.steps, 'image.steps');
  }

  // --- hardware ----------------------------------------------------------
  if (rawHardware.model === undefined) {
    throw new PlanError('MissingField', 'hardware.model', `обязательная модель; допустимые: ${MODELS.join(', ')}`, MODELS);
  }
  const model = requireEnum(rawHardware.model, MODELS, 'hardware.model');
  if (rawHardware.vramGib === undefined) {
    throw new PlanError('MissingField', 'hardware.vramGib', `обязательный объём VRAM, ${LIMITS.vramGib.min}…${LIMITS.vramGib.max} ГиБ`);
  }
  const vramGib = requireRange(requireNumber(rawHardware.vramGib, 'hardware.vramGib'), LIMITS.vramGib, 'hardware.vramGib');
  if (rawHardware.ramGib === undefined) {
    throw new PlanError('MissingField', 'hardware.ramGib', `обязательный объём RAM, ${LIMITS.ramGib.min}…${LIMITS.ramGib.max} ГиБ`);
  }
  const ramGib = requireRange(requireNumber(rawHardware.ramGib, 'hardware.ramGib'), LIMITS.ramGib, 'hardware.ramGib');
  if (rawHardware.disk === undefined) {
    throw new PlanError('MissingField', 'hardware.disk', `обязательный класс накопителя; допустимые: ${DISKS.join(', ')}`, DISKS);
  }
  const disk = requireEnum(rawHardware.disk, DISKS, 'hardware.disk');

  // --- refs / anime / свободный диск -------------------------------------
  const refs = requireRange(
    requireInteger(input.refs === undefined ? DEFAULTS.refs : input.refs, 'refs'),
    LIMITS.refs,
    'refs',
  );

  const anime = input.anime === undefined ? DEFAULTS.anime : input.anime;
  if (typeof anime !== 'boolean') {
    throw new PlanError('NotBoolean', 'anime', `ожидается true|false, получено ${JSON.stringify(anime)}`);
  }

  let freeDiskGib = DEFAULTS.freeDiskGib;
  if (input.freeDiskGib !== undefined && input.freeDiskGib !== null) {
    freeDiskGib = requireNumber(input.freeDiskGib, 'freeDiskGib');
    if (freeDiskGib < 0) {
      throw new PlanError('OutOfRange', 'freeDiskGib', `не может быть отрицательным: ${freeDiskGib}`, [0, Number.POSITIVE_INFINITY]);
    }
  }

  // --- memory (жёсткий override раскладки) -------------------------------
  /** @type {NormalizedMemory} */
  const memory = { kvCache: null, vaeTilePx: null, window: null };
  if (input.memory !== undefined && input.memory !== null) {
    if (!isPlainObject(input.memory)) {
      throw new PlanError('NotObject', 'memory', `ожидается объект { kvCache?, vaeTilePx?, window? }`);
    }
    rejectUnknownKeys(input.memory, ALLOWED_KEYS.memory, 'memory');
    if (input.memory.kvCache !== undefined) {
      memory.kvCache = requireEnum(input.memory.kvCache, KV_CACHE_MODES, 'memory.kvCache');
    }
    if (input.memory.vaeTilePx !== undefined) {
      memory.vaeTilePx = requireEnum(
        requireInteger(input.memory.vaeTilePx, 'memory.vaeTilePx'),
        VAE_TILES,
        'memory.vaeTilePx',
      );
    }
    if (input.memory.window !== undefined) {
      memory.window = normalizeWindow(input.memory.window);
    }
  }

  // --- postprocess: отдельная стадия (ADR A5) — запрос принимается, модель решает потом ---
  const postprocess = normalizePostprocess(input.postprocess);

  return {
    hardware: { model, vramGib, ramGib, disk },
    image: { width, height, quality, bitDepth, steps },
    refs,
    anime,
    freeDiskGib,
    memory,
    postprocess,
  };
}

/**
 * Нормализация `postprocess`: `null` (стадия не запрошена) или `{ model, params? }`.
 *
 * Модели постобработки в Phase A нет в инвентаре, поэтому здесь проверяется только форма запроса
 * (`model` — непустая строка, `params` — плоский объект), а значения параметров — позже и по
 * объявленному набору самой модели (`postprocess.validatePostprocessParams`).
 *
 * @param {unknown} value
 * @returns {NormalizedPostprocess|null}
 */
export function normalizePostprocess(value) {
  if (value === undefined || value === null) return null;
  if (!isPlainObject(value)) {
    throw new PlanError('NotObject', 'postprocess', `ожидается null или объект { model, params? }, получено ${JSON.stringify(value)}`);
  }
  rejectUnknownKeys(value, ALLOWED_KEYS.postprocess, 'postprocess');
  const model = value.model;
  if (typeof model !== 'string' || model.trim() === '') {
    throw new PlanError(
      'MissingField',
      'postprocess.model',
      'обязательное имя модели постобработки (непустая строка)',
      ['postprocess.model'],
    );
  }
  /** @type {Record<string, unknown>|null} */
  let params = null;
  if (value.params !== undefined && value.params !== null) {
    if (!isPlainObject(value.params)) {
      throw new PlanError('NotObject', 'postprocess.params', `ожидается объект параметров модели { … }, получено ${JSON.stringify(value.params)}`);
    }
    params = { ...value.params };
  }
  return { model: model.trim(), params };
}

/**
 * Нормализация `memory.window`:
 * `null` — авто, `true`/`false` — включить/выключить с параметрами по умолчанию,
 * число — включить с этим `windowPx`, объект — включить с явными `windowPx`/`marginPx`.
 *
 * @param {unknown} value
 * @returns {null|{enabled: boolean, windowPx: number|null, marginPx: number|null}}
 */
export function normalizeWindow(value) {
  if (value === null) return null;
  if (typeof value === 'boolean') return { enabled: value, windowPx: null, marginPx: null };
  if (typeof value === 'number') {
    return { enabled: true, windowPx: requireWindowPx(value, 'memory.window'), marginPx: null };
  }
  if (!isPlainObject(value)) {
    throw new PlanError('InvalidType', 'memory.window', `ожидается null|boolean|число|{ windowPx, marginPx }, получено ${JSON.stringify(value)}`);
  }
  rejectUnknownKeys(value, ALLOWED_KEYS.window, 'memory.window');
  const windowPx = value.windowPx === undefined ? null : requireWindowPx(value.windowPx, 'memory.window.windowPx');
  const marginPx = value.marginPx === undefined ? null : requireWindowPx(value.marginPx, 'memory.window.marginPx');
  return { enabled: true, windowPx, marginPx };
}

/**
 * @param {unknown} value
 * @param {string} field
 * @returns {number}
 */
function requireWindowPx(value, field) {
  const px = requireInteger(value, field);
  if (px <= 0) {
    throw new PlanError('OutOfRange', field, `ширина окна/запас должны быть > 0: ${px}`, [CANVAS_GRID_MIN, Number.POSITIVE_INFINITY]);
  }
  return px;
}
