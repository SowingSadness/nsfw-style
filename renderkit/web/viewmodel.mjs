// @ts-check
/**
 * Вью-модель UI: **чистый** слой без DOM (шаг 7 плана).
 *
 * Здесь три вещи, которых нет в ядре:
 *
 * 1. **состояние формы ↔ хеш** (`stateFromHash`/`hashFromState`/`requestFromState`) — состояние
 *    живёт в `location.hash`, поэтому ссылку на расчёт можно переслать;
 * 2. **`buildView(plan)`** — сериализуемая вью-модель для вёрстки: вердикт, победитель, таблица
 *    сочетаний с дельтами к победителю, строки раскладки памяти, предупреждения/`trace`, YAML и
 *    команды. Никаких функций и DOM — тест сравнивает её как данные;
 * 3. **what-if для подсказок** (`hintLive`) и **сканирование YAML** (`scanYaml`).
 *
 * `hintLive(id, request)` делает **один** `computePlan` с изменённым параметром и отдаёт строку
 * дельты (`25 → 8 шагов: 201 → 92 с (−109 с)`); ответы кэшируются по ключу запроса, ошибка или
 * «параметр не входной» → `null`. Это дешёвая операция (ядро считает план за 1–3 мс), но в панели
 * подсказки она не должна влиять на форму: функция ничего не мутирует.
 *
 * `scanYaml` разбирает вставленный YAML и сверяет ключи с whitelist `inventory` — **без Python**.
 * Авторитетная проверка всё равно за `python -m qwenstyle config show <файл> --json`: браузер не
 * видит ни `RunConfig`, ни бэкендов, поэтому UI пишет это рядом с результатом сканирования.
 *
 * Модуль изоморфный: `node:`/DOM не импортируются.
 */

import { computePlan } from '../src/index.mjs';
import { INVENTORY } from '../src/data/inventory.mjs';
import { HELP } from '../src/data/help.mjs';
import { parseYaml } from '../src/yaml.mjs';

/** Разрешённые значения качества (порядок = порядок уровней). */
export const QUALITIES = Object.freeze(['draft', 'standard', 'max']);
/** Разрешённые классы накопителя. */
export const DISKS = Object.freeze(['hdd', 'ssd', 'nvme']);
/** Поля хеша: ровно в этом порядке `hashFromState` печатает строку. */
export const HASH_FIELDS = Object.freeze(['w', 'h', 'q', 'bits', 'model', 'vram', 'ram', 'disk', 'refs', 'anime']);
/** Умолчания формы (совпадают с «лучшим сочетанием» плана для 16 ГиБ / 64 ГБ / NVMe). */
export const DEFAULT_STATE = Object.freeze({
  w: 2560,
  h: 1440,
  q: 'max',
  bits: 8,
  model: 'qwen21-base',
  vram: 16,
  ram: 64,
  disk: 'nvme',
  refs: 1,
  anime: false,
});

/** Границы числовых полей формы: повторяют схему ядра (сообщение формы — до вызова `computePlan`). */
export const FIELD_LIMITS = Object.freeze({
  w: Object.freeze({ min: 256, max: 8192 }),
  h: Object.freeze({ min: 256, max: 8192 }),
  vram: Object.freeze({ min: 4, max: 96 }),
  ram: Object.freeze({ min: 8, max: 512 }),
  refs: Object.freeze({ min: 0, max: 10 }),
});

const MODEL_IDS = Object.freeze(Object.keys(INVENTORY.models));

/**
 * @typedef {Object} AppState
 * @property {number} w ширина кадра
 * @property {number} h высота кадра
 * @property {'draft'|'standard'|'max'} q уровень качества
 * @property {8|16} bits глубина цвета
 * @property {string} model id модели
 * @property {number} vram VRAM, ГиБ
 * @property {number} ram RAM, ГиБ
 * @property {'hdd'|'ssd'|'nvme'} disk класс накопителя
 * @property {number} refs число референсов
 * @property {boolean} anime стилевой адаптер
 */

/**
 * Переопределения поверх формы (в хеш не входят: это разовые переключатели панели памяти).
 *
 * @typedef {Object} StateOverrides
 * @property {unknown} [kvCache] `true|'offload'|false` — переключатель KV в панели памяти
 * @property {number|null} [vaeTilePx] тайл декодера (`512|0|1024|-1`)
 * @property {unknown} [window] режим окна внимания (`true|false|число|{windowPx, marginPx}`)
 * @property {number|null} [freeDiskGib] проверка свободного места
 * @property {number|null} [steps] явный override шагов
 */

/**
 * Прочитать состояние из `location.hash` (`#w=2560&h=1440&q=max…`).
 *
 * Мусор в хеше не роняет форму: неизвестное значение поля заменяется умолчанием поля, остальные
 * поля читаются как есть (ссылка не должна открываться ошибкой из-за одной опечатки).
 *
 * @param {string} hash строка вида `#w=2560&h=1440` (или пустая)
 * @returns {AppState}
 */
export function stateFromHash(hash) {
  const params = new URLSearchParams(String(hash ?? '').replace(/^#/u, ''));
  /** @type {AppState} */
  const state = { ...DEFAULT_STATE };
  state.w = intField(params, 'w', FIELD_LIMITS.w, state.w);
  state.h = intField(params, 'h', FIELD_LIMITS.h, state.h);
  state.q = enumField(params, 'q', QUALITIES, state.q);
  state.bits = /** @type {8|16} */ (enumField(params, 'bits', ['8', '16'], String(state.bits)) === '16' ? 16 : 8);
  state.model = enumField(params, 'model', MODEL_IDS, state.model);
  state.vram = intField(params, 'vram', FIELD_LIMITS.vram, state.vram);
  state.ram = intField(params, 'ram', FIELD_LIMITS.ram, state.ram);
  state.disk = enumField(params, 'disk', DISKS, state.disk);
  state.refs = intField(params, 'refs', FIELD_LIMITS.refs, state.refs);
  const anime = params.get('anime');
  if (anime === '1' || anime === 'true') state.anime = true;
  else if (anime === '0' || anime === 'false') state.anime = false;
  return state;
}

/**
 * Состояние → строка хеша. Порядок полей фиксирован (`HASH_FIELDS`), поэтому одинаковое
 * состояние всегда даёт одинаковую ссылку (это же свойство проверяет тест).
 *
 * @param {Partial<AppState>} state
 * @returns {string} `#w=2560&h=1440&…&anime=0`
 */
export function hashFromState(state) {
  const s = normalizeState(state);
  return `#${HASH_FIELDS.map((key) => `${key}=${hashValue(key, s)}`).join('&')}`;
}

/**
 * `RenderRequest` из состояния формы (`image`/`hardware`/`refs`/`anime`).
 *
 * `overrides.kvCache`/`vaeTilePx`/`window` попадают в `memory` — так работает переключатель
 * VRAM⇄RAM в панели «Раскладка памяти» и любой другой жёсткий override раскладки;
 * `overrides.freeDiskGib`/`steps` — необязательные поля запроса.
 *
 * @param {Partial<AppState>} state
 * @param {StateOverrides} [overrides]
 * @returns {Record<string, unknown>} сырой `RenderRequest` для `computePlan`
 */
export function requestFromState(state, overrides = {}) {
  const s = normalizeState(state);
  /** @type {Record<string, unknown>} */
  const request = {
    image: {
      width: s.w,
      height: s.h,
      quality: s.q,
      bitDepth: s.bits,
    },
    hardware: {
      model: s.model,
      vramGib: s.vram,
      ramGib: s.ram,
      disk: s.disk,
    },
    refs: s.refs,
    anime: s.anime,
  };
  /** @type {Record<string, unknown>} */
  const memory = {};
  if (overrides.kvCache !== undefined && overrides.kvCache !== null) memory.kvCache = overrides.kvCache;
  if (overrides.vaeTilePx !== undefined && overrides.vaeTilePx !== null) memory.vaeTilePx = overrides.vaeTilePx;
  if (overrides.window !== undefined && overrides.window !== null) memory.window = overrides.window;
  if (Object.keys(memory).length > 0) request.memory = memory;
  if (overrides.freeDiskGib !== undefined && overrides.freeDiskGib !== null) {
    request.freeDiskGib = overrides.freeDiskGib;
  }
  if (overrides.steps !== undefined && overrides.steps !== null) {
    /** @type {Record<string, any>} */ (request.image).steps = overrides.steps;
  }
  return request;
}

/**
 * Вью-модель плана: всё, что показывает страница, и ничего больше.
 *
 * @param {any} plan `Plan` из `computePlan`
 * @param {{scan?: any}} [options] `scan` — результат `scanYaml` для панели импорта
 * @returns {{
 *   verdict: string,
 *   badge: {label: string, tone: 'ok'|'warn'|'bad'},
 *   headline: string,
 *   form: any,
 *   winner: any|null,
 *   combinations: any[],
 *   memoryRows: any[],
 *   warnings: any[],
 *   suggestions: any[],
 *   degradations: any[],
 *   trace: any[],
 *   stages: any[],
 *   yaml: string|null,
 *   cli: any|null,
 *   confidence: string,
 *   scan: any
 * }}
 */
export function buildView(plan, options = {}) {
  const render = plan.render;
  const state = stateFromPlan(plan);
  const badge = badgeFor(plan);
  return {
    verdict: plan.verdict,
    badge,
    headline: headlineFor(plan),
    form: {
      state,
      fields: ['width', 'height', 'quality', 'bitDepth', 'model', 'vramGib', 'ramGib', 'disk', 'refs', 'anime', 'freeDiskGib'],
      tier: plan.request && plan.request.tier ? plan.request.tier : null,
      aspect: plan.image.aspect,
      native: canvasText(plan.image.native.width, plan.image.native.height),
      tokens: plan.image.native.tokens,
      model: state.model,
    },
    winner: render ? winnerFor(plan) : null,
    combinations: render ? plan.combinations.map((row) => combinationRow(row, plan)) : [],
    memoryRows: render ? memoryRowsFor(plan) : [],
    warnings: (plan.warnings ?? []).map(cloneNote),
    suggestions: (plan.suggestions ?? []).map(cloneNote),
    degradations: (plan.degradations ?? []).map(cloneNote),
    trace: (plan.trace ?? []).map(traceRow),
    stages: (plan.stages ?? []).map(stageRow),
    yaml: plan.yaml ?? null,
    cli: plan.cli ?? null,
    confidence: plan.confidence,
    scan: options.scan ?? null,
  };
}

// ── состояние формы ─────────────────────────────────────────────────────────────────────────

/**
 * Дополнить частичное состояние умолчаниями (и не отдавать наружу посторонние поля).
 *
 * @param {Partial<AppState>} state
 * @returns {AppState}
 */
export function normalizeState(state) {
  /** @type {AppState} */
  const s = { ...DEFAULT_STATE };
  if (!state || typeof state !== 'object') return s;
  for (const key of HASH_FIELDS) {
    if (state[key] === undefined || state[key] === null) continue;
    /** @type {any} */ (s)[key] = /** @type {any} */ (state)[key];
  }
  return s;
}

/**
 * @param {URLSearchParams} params
 * @param {string} key
 * @param {{min: number, max: number}} range
 * @param {number} fallback
 * @returns {number}
 */
function intField(params, key, range, fallback) {
  const raw = params.get(key);
  if (raw === null || raw.trim() === '') return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || !Number.isInteger(value)) return fallback;
  if (value < range.min || value > range.max) return fallback;
  return value;
}

/**
 * @param {URLSearchParams} params
 * @param {string} key
 * @param {ReadonlyArray<string>} allowed
 * @param {any} fallback
 * @returns {any}
 */
function enumField(params, key, allowed, fallback) {
  const raw = params.get(key);
  return raw !== null && allowed.includes(raw) ? raw : fallback;
}

/**
 * @param {string} key
 * @param {AppState} state
 * @returns {string|number}
 */
function hashValue(key, state) {
  if (key === 'anime') return state.anime ? '1' : '0';
  return /** @type {any} */ (state)[key];
}

/**
 * Состояние из нормализованного запроса плана (обратная операция к `requestFromState`).
 *
 * @param {any} plan
 * @returns {AppState}
 */
function stateFromPlan(plan) {
  const request = plan.request;
  return normalizeState({
    w: request.image.width,
    h: request.image.height,
    q: request.image.quality,
    bits: request.image.bitDepth,
    model: request.hardware.model,
    vram: request.hardware.vramGib,
    ram: request.hardware.ramGib,
    disk: request.hardware.disk,
    refs: request.refs,
    anime: request.anime,
  });
}

// ── вердикт и заголовок ────────────────────────────────────────────────────────────────────

/**
 * @param {any} plan
 * @returns {{label: string, tone: 'ok'|'warn'|'bad'}}
 */
function badgeFor(plan) {
  if (plan.verdict === 'ok') return { label: 'ok', tone: 'ok' };
  if (plan.verdict === 'degraded') return { label: 'degraded', tone: 'warn' };
  if (plan.verdict === 'unsupported') return { label: 'unsupported', tone: 'warn' };
  return { label: 'impossible', tone: 'bad' };
}

/**
 * Одна строка-итог: кадр, шаги, время, пик VRAM, Q — или причина отказа.
 *
 * @param {any} plan
 * @returns {string}
 */
function headlineFor(plan) {
  const requested = canvasText(plan.image.requested.width, plan.image.requested.height);
  const native = canvasText(plan.image.native.width, plan.image.native.height);
  if (plan.render === null) {
    const reason = firstReason(plan);
    const snapped = plan.image.exact ? '' : ` (сетка 32 даёт ${native})`;
    const size = megapixels(plan.image.native.width, plan.image.native.height);
    return `${requested} недостижим${snapped}: ${size}${reason ? ` · ${reason}` : ''}`;
  }
  const render = plan.render;
  const estimates = plan.estimates;
  const parts = [
    native,
    `${render.steps} шагов`,
    `${sec(estimates.time.genPerFrameS)}/кадр`,
    `пик ${gib1(estimates.vram.peakGib)}`,
    `Q ${plan.stages[0].quality ?? '—'}`,
  ];
  if (plan.verdict === 'unsupported') parts.unshift('16 бит не поддержаны — оценка для 8-битного PNG');
  return parts.join(' · ');
}

/**
 * Первая причина отказа одной фразой для заголовка: сначала «какой кадр достижим», потом «сколько
 * нужно железа», и только затем текст предупреждения плана (он длиннее и рассчитан на панель).
 *
 * @param {any} plan
 * @returns {string}
 */
function firstReason(plan) {
  const suggestions = plan.suggestions ?? [];
  const smaller = suggestions.find((item) => item.code === 'smaller-canvas');
  if (smaller && smaller.value) return `ближайший рабочий кадр — ${smaller.value}`;
  const hardware = suggestions.find((item) => item.code === 'other-hardware');
  if (hardware && hardware.value !== null && hardware.value !== undefined) {
    return `нужно ${hardware.value} ГиБ VRAM на этой модели`;
  }
  const warning = (plan.warnings ?? []).find((item) => item.code === 'vram-spill-risk');
  if (warning && warning.detail) return warning.detail;
  return suggestions[0] && suggestions[0].detail ? suggestions[0].detail : '';
}

// ── победитель ────────────────────────────────────────────────────────────────────────────

/**
 * Карточка «резолв»: кадр, выход, шаги, раскладка памяти и оценки.
 *
 * @param {any} plan
 * @returns {any}
 */
function winnerFor(plan) {
  const render = plan.render;
  const estimates = plan.estimates;
  const image = plan.image;
  const memory = render.memory;
  const snap = image.snap;
  const snapText = snapTextFor(snap);
  const model = INVENTORY.models[plan.request.hardware.model];
  const fixedSteps = INVENTORY.limits.fixedSteps[plan.request.hardware.model] ?? null;
  const fixed = fixedSteps !== null && fixedSteps === render.steps;
  return {
    canvas: {
      requested: canvasText(image.requested.width, image.requested.height),
      native: canvasText(image.native.width, image.native.height),
      tokens: image.native.tokens,
      aspect: image.aspect ?? 'пропорция референса',
      snap: snapText,
      exact: image.exact,
    },
    output: {
      size: canvasText(image.native.width, image.native.height),
      format: image.format,
      bitDepth: image.bitDepth,
      perFrameMb: round2(estimates.disk.pngPerFrameMb),
      factor: estimates.disk.bitDepthFactor,
      note: image.bitDepth === 16 ? '16 бит: файл пока 8-битный, размер учтён ×2' : '',
    },
    steps: {
      count: render.steps,
      level: plan.request.image.quality,
      fixed,
      note: fixed ? 'фиксировано моделью' : (plan.request.image.steps !== null ? 'override запроса' : ''),
    },
    memory: {
      kv: memory.kvCache.mode === null
        ? 'не используется'
        : `${placementLabel(memory.kvCache.placement)} · ${gib2(memory.kvCache.gib)}`,
      kvMode: memory.kvCache.mode,
      kvPlacement: memory.kvCache.placement,
      vae: vaeText(memory.vae),
      window: memory.attention.mode === 'window'
        ? `${memory.attention.windowPx} + ${memory.attention.marginPx}`
        : 'плотное',
      ref: refText(memory.ref),
    },
    estimates: {
      vram: `${gib1(estimates.vram.peakGib)} / ${gib1(estimates.vram.budgetGib)} (запас ${gib1(estimates.vram.headroomGib)})`,
      ram: `${gib1(estimates.ram.peakGib)} / ${gib1(estimates.ram.budgetGib)}`,
      time: `${sec(estimates.time.genPerFrameS)}/кадр, первый ${sec(estimates.time.firstFrameS)} (энкод ${sec(estimates.time.encodeColdS)})`,
      disk: `${round2(estimates.disk.pngPerFrameMb)} МБ/кадр`,
      confidence: estimates.confidence,
      confidenceReasons: (estimates.confidenceReasons ?? []).slice(),
      // Числовые дубли для полос бюджета: строки выше — для текста, эти — для ширины полосы.
      vramPeak: estimates.vram.peakGib,
      vramBudget: estimates.vram.budgetGib,
      ramPeak: estimates.ram.peakGib,
      ramBudget: estimates.ram.budgetGib,
    },
  };
}

/**
 * Строка таблицы сочетаний: раскладка, оценки, Парето/победитель и дельта к победителю.
 *
 * @param {any} row строка `Plan.combinations`
 * @param {any} plan
 * @returns {any}
 */
function combinationRow(row, plan) {
  const canvas = canvasText(plan.image.native.width, plan.image.native.height);
  const best = plan.combinations.find((item) => item.best) ?? plan.combinations[0] ?? row;
  return {
    id: row.id,
    canvas,
    steps: row.steps,
    offload: row.offload,
    offloadBlocks: row.offloadBlocks,
    kv: kvLabel(row.kvCache),
    kvCache: row.kvCache,
    vae: vaeLabel(row.vaeTilePx),
    vaeTilePx: row.vaeTilePx,
    window: row.attention === 'window' ? `${row.windowPx} + ${row.marginPx}` : 'плотное',
    attention: row.attention,
    refPx: row.refPx,
    encoderDevice: row.encoderDevice,
    loraScale: row.loraScale,
    timeS: row.timeS,
    peakVram: row.peakVramGib,
    headroomGib: row.headroomGib,
    q: row.q,
    zone: row.zone,
    pareto: row.pareto === true,
    best: row.best === true,
    delta: row.best ? '—' : deltaText(row, best),
  };
}

/**
 * Дельта строки к победителю: время, пик VRAM, Q (что изменится, если взять это сочетание).
 *
 * @param {any} row
 * @param {any} best
 * @returns {string}
 */
function deltaText(row, best) {
  /** @type {string[]} */
  const parts = [];
  const dTime = round1(row.timeS - best.timeS);
  if (Math.abs(dTime) >= 1) parts.push(`${signed(dTime)} с`);
  const dVram = round2(row.peakVramGib - best.peakVramGib);
  if (Math.abs(dVram) >= 0.05) parts.push(`${signed(dVram)} ГиБ`);
  const dQ = row.q - best.q;
  if (dQ !== 0) parts.push(`Q ${signed(dQ)}`);
  return parts.length > 0 ? parts.join(' · ') : '—';
}

// ── раскладка памяти ──────────────────────────────────────────────────────────────────────

/**
 * Строки панели «Раскладка памяти»: что где лежит, чем это стоит и почему выбрано.
 *
 * @param {any} plan
 * @returns {Array<{key: string, label: string, value: string, placement: string, cost: string, reason: string, helpId: string}>}
 */
function memoryRowsFor(plan) {
  const render = plan.render;
  const memory = render.memory;
  const model = INVENTORY.models[plan.request.hardware.model];
  const kv = memory.kvCache;
  const vae = memory.vae;
  const attention = memory.attention;
  const ref = memory.ref;
  return [
    {
      key: 'kv',
      label: 'KV-кеш префикса',
      value: kvValue(kv),
      placement: kv.placement ?? 'none',
      cost: kv.mode === null ? '0 ГиБ' : `${kv.prefixTokens} ток. · ${kv.costPct >= 0 ? '+' : ''}${kv.costPct} % времени`,
      reason: kv.reason,
      helpId: 'kvCache',
    },
    {
      key: 'vae',
      label: 'Тайл декодера VAE',
      value: vaeText(vae),
      placement: 'vram',
      cost: vae.peakGib === null ? '—' : `пик декода ${gib1(vae.peakGib)}`,
      reason: vae.reason,
      helpId: 'vaeTilePx',
    },
    {
      key: 'window',
      label: 'Окно внимания',
      value: attention.mode === 'window'
        ? `${attention.windowPx} + ${attention.marginPx} · ${attention.bands} полос`
        : 'плотное',
      placement: 'vram',
      cost: `${sec(attention.costS)} против ${sec(attention.denseCostS)} плотного`,
      reason: attention.reason,
      helpId: 'window',
    },
    {
      key: 'ref',
      label: 'Сетка референсов',
      value: refText(ref),
      placement: 'prefix',
      cost: `${ref.tokensPerRef} ток./реф · ${ref.totalTokens} всего`,
      reason: ref.reason,
      helpId: 'refPx',
    },
    {
      key: 'offload',
      label: 'Веса трансформера',
      value: render.offload === 'group' ? 'fp8-блоки в RAM (group)' : 'fp8 в VRAM',
      placement: render.offload === 'group' ? 'ram' : 'vram',
      cost: model ? `${gib1(model.weightGibFp8)} fp8` : '—',
      reason: render.offload === 'group'
        ? 'group снимает 6.6 ГиБ VRAM и стримит блоки на GPU по группе (в 1.5–2.6 раза медленнее)'
        : 'без офлоада блоки живут в VRAM: быстрее, но выше пик',
      helpId: 'offload',
    },
  ];
}

/**
 * Сетка референсов: `сетка кадра` (`0` — максимум точности) или `960 px`; у модели без `ref_px` —
 * честное «референсы как есть».
 *
 * @param {any} ref `Plan.render.memory.ref`
 * @returns {string}
 */
function refText(ref) {
  if (String(ref.reason).startsWith('model-has-no-ref-px')) return 'референсы как есть';
  return ref.px === 0 ? 'сетка кадра' : `${ref.px} px`;
}

/**
 * @param {string} placement
 * @returns {string}
 */
function placementLabel(placement) {
  if (placement === 'vram') return 'VRAM';
  if (placement === 'ram') return 'RAM';
  if (placement === 'none') return 'нет';
  return String(placement);
}

/**
 * Сдвиг нативного кадра к запросу одной строкой: `точно`, `высота +8 px`, `ширина −32 px, высота +8 px`.
 *
 * @param {{dx: number, dy: number}} snap
 * @returns {string}
 */
function snapTextFor(snap) {
  /** @type {string[]} */
  const parts = [];
  if (snap.dx !== 0) parts.push(`ширина ${signed(snap.dx)} px`);
  if (snap.dy !== 0) parts.push(`высота ${signed(snap.dy)} px`);
  return parts.length === 0 ? 'точно' : parts.join(', ');
}

/**
 * Значение строки KV в панели памяти: режим, место и объём (или честное «без кеша»).
 *
 * @param {any} kv `Plan.render.memory.kvCache`
 * @returns {string}
 */
function kvValue(kv) {
  if (kv.mode === null) return 'не используется';
  if (kv.mode === 'false') return `без кеша · префикс ${gib2(kv.gib)} считался бы`;
  return `${placementLabel(kv.placement)} · ${gib2(kv.gib)}`;
}

/**
 * @param {unknown} mode `Plan.combinations[].kvCache`
 * @returns {string}
 */
function kvLabel(mode) {
  if (mode === true || mode === 'true') return 'VRAM';
  if (mode === 'offload') return 'RAM';
  if (mode === false || mode === 'false') return 'без кеша';
  return '—';
}

/**
 * Тайл декодера в человеческом виде: `512 / 384`, `256 / 192` (`tilePx: 0` — это 256 px!), `1024 / 768`.
 *
 * @param {any} vae `Plan.render.memory.vae`
 * @returns {string}
 */
function vaeText(vae) {
  if (vae.tilePx === null || vae.tilePx === undefined) return 'декодер модели';
  if (vae.tilePx === -1) return 'без тайлинга';
  return `${vae.tilePx === 0 ? 256 : vae.tilePx} / ${vae.stridePx}`;
}

/**
 * @param {number|null} tilePx
 * @returns {string}
 */
function vaeLabel(tilePx) {
  if (tilePx === 512) return '512 / 384';
  if (tilePx === 0) return '256 / 192';
  if (tilePx === 1024) return '1024 / 768';
  if (tilePx === -1) return 'без тайлинга';
  return tilePx === null ? 'модель' : String(tilePx);
}

// ── what-if для подсказок ────────────────────────────────────────────────────────────────

/** Кэш ответов `hintLive`: ключ — id + сериализованный запрос. @type {Map<string, string|null>} */
const liveCache = new Map();
const LIVE_CACHE_LIMIT = 200;

/**
 * Живая дельта подсказки: один `computePlan` с изменённым параметром.
 *
 * Возвращает строку вида `25 → 8 шагов: 201 → 92 с (−109 с)` — её показывает строка «сейчас:».
 * `null` — если параметр не входной (`live: null`), запрос невалиден или плана нет.
 *
 * @param {string} id id подсказки (`HELP[id].live`)
 * @param {unknown} request `RenderRequest` текущей формы
 * @returns {string|null}
 */
export function hintLive(id, request) {
  const entry = Object.prototype.hasOwnProperty.call(HELP, id) ? HELP[id] : null;
  const live = entry ? entry.live : null;
  if (!live) return null;
  const key = `${id}|${safeJson(request)}`;
  if (liveCache.has(key)) return /** @type {string|null} */ (liveCache.get(key));
  /** @type {string|null} */
  let result = null;
  try {
    const base = computePlan(request);
    if (base.render === null) {
      result = null;
    } else {
      const probe = probeFor(live, request, base);
      if (probe === null) {
        result = null;
      } else if (typeof probe === 'string') {
        result = probe;
      } else if (probe.request === null) {
        result = `${probe.label}: —`;
      } else {
        const alt = computePlan(probe.request);
        result = deltaLine(probe.label, base, alt, probe.skipSteps === true);
      }
    }
  } catch {
    result = null;
  }
  if (liveCache.size >= LIVE_CACHE_LIMIT) liveCache.clear();
  liveCache.set(key, result);
  return result;
}

/**
 * Пробный запрос для оси what-if.
 *
 * @param {string} live
 * @param {any} request
 * @param {any} base уже посчитанный план
 * @returns {{label: string, request: any, skipSteps?: boolean}|string|null}
 *   строка = «пересчитывать нечего, вот почему»; `null` = дельты нет
 */
function probeFor(live, request, base) {
  const model = request.hardware.model;
  const fixed = INVENTORY.models[model] ? INVENTORY.models[model].fixedSteps : null;
  const memory = base.render.memory;
  switch (live) {
    case 'steps': {
      const current = base.render.steps;
      if (fixed !== null && fixed === current && model === 'h3-image') {
        return `фиксировано моделью: ${current} шагов (расписание PDD), параметр игнорируется`;
      }
      const target = current > 12 ? 8 : 25;
      return {
        label: `${current} → ${target} шагов`,
        request: withImage(request, { steps: target }),
        skipSteps: true,
      };
    }
    case 'width':
    case 'height': {
      const key = live === 'width' ? 'width' : 'height';
      const current = request.image[key];
      const target = Math.max(256, Math.round((current * 0.75) / 32) * 32);
      if (target === current) return null;
      return {
        label: `0.75× ${live === 'width' ? 'ширины' : 'высоты'} (${target} px)`,
        request: withImage(request, { [key]: target }),
      };
    }
    case 'quality': {
      const current = request.image.quality;
      const target = current === 'draft' ? 'standard' : 'draft';
      return {
        label: `${current} → ${target}`,
        request: withImage(request, { quality: target }),
      };
    }
    case 'kvCache': {
      if (memory.kvCache.mode === null) {
        return 'модель без KV-кеша префикса: переключать нечего';
      }
      const target = memory.kvCache.mode === 'offload' ? true : 'offload';
      return {
        label: `KV ${placementLabel(memory.kvCache.placement)} → ${target === true ? 'VRAM' : 'RAM'}`,
        request: withMemory(request, { kvCache: target }),
      };
    }
    case 'vaeTilePx': {
      if (memory.vae.tilePx === null) return 'модель без тайлинга декодера: переключать нечего';
      const target = memory.vae.tilePx === 1024 ? 0 : 1024;
      return {
        label: `тайл ${memory.vae.tilePx} → ${target}`,
        request: withMemory(request, { vaeTilePx: target }),
      };
    }
    case 'window': {
      const windowPx = memory.attention.windowPx;
      if (String(memory.attention.reason).startsWith('model-has-no-window')) {
        return 'модель без окна внимания: внимание бэкенда не режется на полосы';
      }
      if (memory.attention.mode === 'dense' && (windowPx === 0 || windowPx === null)
        && base.image.native.height < 1600) {
        return 'окно измерено только с 1600 px: ниже плотное внимание быстрее проверенного и не теряет резкость';
      }
      const enabled = memory.attention.mode !== 'window';
      return {
        label: enabled ? 'плотное → окно' : 'окно → плотное',
        request: withMemory(request, { window: enabled }),
      };
    }
    case 'anime': {
      return {
        label: request.anime ? 'anime выкл' : 'anime вкл',
        request: { ...request, anime: !request.anime },
      };
    }
    default:
      return null;
  }
}

/**
 * @param {any} request
 * @param {Record<string, unknown>} patch
 * @returns {any}
 */
function withImage(request, patch) {
  return { ...request, image: { ...request.image, ...patch } };
}

/**
 * @param {any} request
 * @param {Record<string, unknown>} patch
 * @returns {any}
 */
function withMemory(request, patch) {
  return { ...request, memory: { ...(request.memory ?? {}), ...patch } };
}

/**
 * Строка дельты для панели подсказки: время, пик VRAM, KV, декод, Q.
 *
 * @param {string} label что именно изменили
 * @param {any} base план текущего запроса
 * @param {any} alt план изменённого запроса
 * @param {boolean} skipSteps не повторять шаги (они уже в `label`)
 * @returns {string}
 */
function deltaLine(label, base, alt, skipSteps) {
  if (alt.render === null) return `${label}: при таком варианте план невозможен`;
  /** @type {string[]} */
  const parts = [];
  if (!skipSteps && alt.render.steps !== base.render.steps) {
    parts.push(`шаги ${base.render.steps} → ${alt.render.steps}`);
  }
  const t0 = base.estimates.time.genPerFrameS;
  const t1 = alt.estimates.time.genPerFrameS;
  if (Math.abs(t1 - t0) >= 1) {
    parts.push(`${sec(t0)} → ${sec(t1)} (${signed(round1(t1 - t0))} с)`);
  }
  const v0 = base.estimates.vram.peakGib;
  const v1 = alt.estimates.vram.peakGib;
  if (Math.abs(v1 - v0) >= 0.05) {
    parts.push(`пик ${gib1(v0)} → ${gib1(v1)} (${signed(round2(v1 - v0))} ГиБ)`);
  }
  const k0 = base.render.memory.kvCache.gib;
  const k1 = alt.render.memory.kvCache.gib;
  if (Math.abs(k1 - k0) >= 0.05) {
    parts.push(`KV ${gib2(k0)} → ${gib2(k1)}`);
  }
  const p0 = base.render.memory.vae.peakGib;
  const p1 = alt.render.memory.vae.peakGib;
  if (p0 !== null && p1 !== null && Math.abs(p1 - p0) >= 0.05) {
    parts.push(`декод ${gib1(p0)} → ${gib1(p1)}`);
  }
  const q0 = base.stages[0].quality;
  const q1 = alt.stages[0].quality;
  if (q0 !== null && q1 !== null && q0 !== q1) {
    parts.push(`Q ${q0} → ${q1}`);
  }
  return parts.length > 0 ? `${label}: ${parts.join(' · ')}` : `${label}: — оценки не меняются`;
}

/**
 * Стабильный ключ для кэша what-if: порядок ключей объекта не должен влиять.
 *
 * @param {unknown} value
 * @returns {string}
 */
function safeJson(value) {
  /** @param {unknown} item @returns {unknown} */
  const sort = (item) => {
    if (Array.isArray(item)) return item.map(sort);
    if (item && typeof item === 'object') {
      /** @type {Record<string, unknown>} */
      const out = {};
      for (const key of Object.keys(item).sort()) out[key] = sort(/** @type {any} */ (item)[key]);
      return out;
    }
    return item;
  };
  try {
    return JSON.stringify(sort(value));
  } catch {
    return String(value);
  }
}

/**
 * Сбросить кэш what-if (нужен тестам и кнопке «пересчитать»).
 *
 * @returns {void}
 */
export function clearHintCache() {
  liveCache.clear();
}

// ── сканирование YAML ────────────────────────────────────────────────────────────────────

/** Whitelist `--set` всех бэкендов: полные пути, которые система точно знает. */
const SETTABLE_PATHS = new Set(Object.values(INVENTORY.settable).flat());

/** Поля `RunConfig`/бэкендов, которые пишет эмиттер и понимает `config show` (по контексту). */
const YAML_KEYS = Object.freeze({
  '': Object.freeze(['size_px', 'aspect', 'resolution', 'upscale', 'logs', 'seed', 'steps', 'model', 'output_dir', 'preset']),
  generation: Object.freeze(['backend', 'repo', 'transformer', 'lora', 'steps', 'guidance', 'precision', 'precision_cache', 'offload', 'kv_cache', 'sampler_kwargs']),
  'generation.lora': Object.freeze(['repo', 'filename', 'scale', 'strip_alpha']),
  // Ключи `sampler_kwargs` проверяются по полному пути в `SETTABLE_PATHS` (их объявляет бэкенд).
  'generation.sampler_kwargs': Object.freeze([]),
  encoder: Object.freeze(['name', 'cache_tag', 'repo', 'revision', 'local_dir', 'requires_vision', 'dtype', 'device', 'stream', 'prefetch', 'pin_memory', 'lookahead']),
  upscale: Object.freeze(['factor', 'target_height', 'tile', 'pad', 'backend']),
});

/**
 * Разобрать вставленный YAML и сверить ключи с `inventory` (**без Python**).
 *
 * Возвращает `{ok, keys, values, problem, unknown, authority}`:
 *
 * - `keys` — все встреченные пути (и контейнеры, и листья), в порядке файла;
 * - `values` — плоская карта «путь → значение» по листьям;
 * - `problem` — первая проблема (ошибка разбора, пустой файл, неизвестный ключ) или `null`;
 * - `authority` — напоминание, что авторитетная проверка — `config show` в CLI.
 *
 * @param {string} text YAML-текст (свой диалект, см. `src/yaml.mjs`)
 * @returns {{ok: boolean, keys: string[], values: Record<string, unknown>, problem: string|null,
 *   unknown: string[], authority: string}}
 */
export function scanYaml(text) {
  const authority = 'python -m qwenstyle config show <файл> --json';
  const source = String(text ?? '');
  if (source.trim() === '') {
    return { ok: false, keys: [], values: {}, problem: 'пустой YAML: вставьте конфиг', unknown: [], authority };
  }
  /** @type {Record<string, unknown>} */
  let parsed;
  try {
    parsed = /** @type {Record<string, unknown>} */ (parseYaml(source));
  } catch (error) {
    const raw = error && typeof error === 'object' && 'message' in error ? String(error.message) : String(error);
    return { ok: false, keys: [], values: {}, problem: `не разобрано: ${raw}`, unknown: [], authority };
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return { ok: false, keys: [], values: {}, problem: 'корень файла — не маппинг', unknown: [], authority };
  }
  /** @type {string[]} */
  const keys = [];
  /** @type {Record<string, unknown>} */
  const values = {};
  /** @type {string[]} */
  const unknown = [];
  walkYaml(parsed, '', keys, values, unknown);
  const problem = unknown.length > 0
    ? `неизвестные ключи (нет в whitelist inventory): ${unknown.join(', ')}`
    : (Object.keys(values).length === 0 ? 'в файле нет ни одного параметра' : null);
  return { ok: problem === null, keys, values, problem, unknown, authority };
}

/**
 * Обход разобранного YAML: контейнеры разворачиваются, листья проверяются по whitelist.
 *
 * @param {Record<string, any>} node
 * @param {string} prefix
 * @param {string[]} keys
 * @param {Record<string, unknown>} values
 * @param {string[]} unknown
 * @returns {void}
 */
function walkYaml(node, prefix, keys, values, unknown) {
  /** @type {ReadonlyArray<string>} */
  const allowed = YAML_KEYS[/** @type {keyof typeof YAML_KEYS} */ (prefix)] ?? [];
  for (const [key, value] of Object.entries(node)) {
    const path = prefix ? `${prefix}.${key}` : key;
    keys.push(path);
    const isMap = value !== null && typeof value === 'object' && !Array.isArray(value);
    if (isMap && Object.prototype.hasOwnProperty.call(YAML_KEYS, path)) {
      walkYaml(/** @type {Record<string, any>} */ (value), path, keys, values, unknown);
      continue;
    }
    if (isMap) {
      // Маппинг под неизвестным контейнером: ключ не проверяем, но и не принимаем.
      unknown.push(path);
      continue;
    }
    values[path] = value;
    if (allowed.includes(key) || SETTABLE_PATHS.has(path)) continue;
    unknown.push(path);
  }
}

// ── форматирование ───────────────────────────────────────────────────────────────────────

/**
 * @param {number} width
 * @param {number} height
 * @returns {string} `2560×1440`
 */
export function canvasText(width, height) {
  return `${width}×${height}`;
}

/**
 * Мегапиксели кадра, `8.36` (2 знака).
 *
 * @param {number} width
 * @param {number} height
 * @returns {string} `8.36 Мп`
 */
export function megapixels(width, height) {
  return `${(width * height / 1e6).toFixed(2)} Мп`;
}

/** @param {number} value @returns {string} */
function sec(value) {
  return `${Math.round(value)} с`;
}

/** @param {number} value @returns {string} */
function gib1(value) {
  return `${Number(value).toFixed(1)} ГиБ`;
}

/** @param {number} value @returns {string} */
function gib2(value) {
  return `${Number(value).toFixed(2)} ГиБ`;
}

/** @param {number} value @returns {string} */
function round1(value) {
  return Math.round(value * 10) / 10;
}

/** @param {number} value @returns {string} */
function round2(value) {
  return Math.round(value * 100) / 100;
}

/**
 * Знак числа для дельт: `+12`, `−4.5` (типографский минус — так читаемее в таблице).
 *
 * @param {number} value
 * @returns {string}
 */
function signed(value) {
  const fixed = Number.isInteger(value) ? String(value) : String(Number(value.toFixed(2)));
  return value > 0 ? `+${fixed}` : fixed.replace(/^-/u, '−');
}

/**
 * @param {any} note
 * @returns {any}
 */
function cloneNote(note) {
  return { ...note };
}

// ── строки trace/stages ──────────────────────────────────────────────────────────────────

/**
 * Строка `trace` для `<details>`: что решили, почему и что отвергли.
 *
 * @param {any} entry
 * @returns {any}
 */
function traceRow(entry) {
  return {
    field: entry.field,
    ruleId: entry.ruleId,
    source: entry.source,
    value: traceValue(entry.value),
    rejected: (entry.rejected ?? []).map((item) => ({ value: traceValue(item.value), why: item.why })),
  };
}

/**
 * @param {unknown} value
 * @returns {string}
 */
function traceValue(value) {
  if (value === null || value === undefined) return '—';
  if (typeof value === 'object') {
    const anyValue = /** @type {any} */ (value);
    if (typeof anyValue.width === 'number' && typeof anyValue.height === 'number') {
      return canvasText(anyValue.width, anyValue.height);
    }
    if (Array.isArray(value)) return value.map(traceValue).join(', ');
    if (typeof anyValue.px === 'number') return `${anyValue.px} px`;
    return safeJson(value);
  }
  return String(value);
}

/**
 * Строка стадии: включена ли, почему и что стоит.
 *
 * @param {any} stage
 * @returns {any}
 */
function stageRow(stage) {
  return {
    stage: stage.stage,
    enabled: stage.enabled === true,
    reason: stage.reason ?? '',
    quality: stage.quality ?? null,
    hasEstimates: stage.estimates !== null && stage.estimates !== undefined,
  };
}
