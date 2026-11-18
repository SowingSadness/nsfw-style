// @ts-check
/**
 * Раскладка памяти — первоклассный выход плана (ADR A7): `kv_cache`, `vae_tile_px`,
 * скользящее окно внимания и `ref_px`.
 *
 * Каждая функция решает **одну** ось и честно возвращает не только значение, но и причину
 * (`reason`), место (`placement`), цену (`costPct`) и риск (`oom`/`degraded`). Решения принимаются
 * по оценкам `cost.mjs` (никаких собственных формул VRAM) и по правилам A7:
 *
 * - **KV**: `true` (в VRAM) только если `peak + kvGib ≤ 0.90·budget` **и** референсов меньше
 *   `KV_AUTO_REFS_OFFLOAD` (2+ рефа дают измеренный OOM на шаге 0 — то же правило, что
 *   `auto-offload-ref2` у бэкенда); иначе `offload` (KV в RAM, послойная подкачка, картинка
 *   бит-в-бит, +13 % времени); `false` — только по явному запросу (префикс пересчитывается
 *   каждый шаг, ≈×2 времени).
 * - **Тайл VAE**: `512` по умолчанию, `0` (256/192) — явным запросом при малом кадре,
 *   `1024` — на широкой VRAM (≥24 ГиБ), `-1` (без тайлинга) запрещён при ≥1.5 Мп: декод даёт OOM.
 * - **Окно**: включается при `canvasH ≥ 1600` px и покрытии `(px + 2·margin)/canvasH ≥ 0.60`;
 *   при `canvasH ≤ px` окно вырождается в плотное внимание (`equivalentToDense`, не дублируется).
 * - **`ref_px`**: кламп по канвасу (`min(ref_px, canvasH)`, вниз до сетки 32, пол 256 — как
 *   `config.py::embedding_crop`); политика `0` при ≤1088p и ≤2 рефах, `960` от 1440p; при
 *   превышении потолка ≈13000 токенов референсов значение спускается по ладдеру `960→768→640→…`.
 *
 * Числа политики — не фит (`MODEL` их не содержит), поэтому они собраны в `MEMORY_ASSUMPTIONS`
 * с провенансом той же формы, что `MODEL.provenance`. Модуль изоморфный: без `node:`/DOM.
 */

import { MODEL } from '../data/model.mjs';
import { INVENTORY, settableKeys } from './inventory.mjs';
import {
  GATE, KV_AUTO_REFS_OFFLOAD, estimateVram, prefixTokens,
} from './cost.mjs';
import { SIZE_LIMITS } from './geometry.mjs';
import { CANVAS_GRID, refGrid } from '../units.mjs';

/**
 * Сетка референса — единственная реализация в `units.mjs`; здесь она ре-экспортируется, чтобы
 * `memory.refGrid` оставался публичным именем (им пользуются `resolveRefPx` и тесты).
 */
export { refGrid };

/** @typedef {import('./cost.mjs').Render} Render */

/**
 * Допущения политики раскладки памяти (провенанс — как у `MODEL.provenance`).
 *
 * @type {Readonly<Record<string, {value: number, unit: string, kind: 'measured'|'assumption', source: string, note: string}>>}
 */
export const MEMORY_ASSUMPTIONS = Object.freeze({
  'kv.offloadCostPct': {
    value: 13,
    unit: '%',
    kind: 'measured',
    source: 'docs/bench/kv-cache-offload.md#«Что это значит»: +12–17 % времени при KV в RAM (93 с → 105 с)',
    note: 'надбавка за послойную подкачку K/V из RAM; картинка при этом бит-в-бит, поэтому цена '
      + 'только во времени. В фит времени она вошла крошечным `bKvOffload` (флаг режима), '
      + 'поэтому здесь взята измеренная надбавка, а не коэффициент',
  },
  'kv.noCacheCostPct': {
    value: 100,
    unit: '%',
    kind: 'measured',
    source: 'docs/bench/kv-cache-offload.md#«false»: ~2× медленнее — 51 с против 24 с на 768p/3 рефа',
    note: 'без кеша префикс пересчитывается на каждом шаге, то есть время генерации примерно ×2',
  },
  'vae.omitTileOomPx': {
    value: 1.5e6,
    unit: 'px',
    kind: 'measured',
    source: 'AGENTS.md#§4 «Тайлинг декодера VAE»: без тайлинга — OOM на декоде (2912×1632), «отключить нельзя»',
    note: 'порог, выше которого декод без тайлинга гарантированно не влезает в 16 ГиБ',
  },
  'vae.wideVramGib': {
    value: 24,
    unit: 'GiB',
    kind: 'assumption',
    source: '.pi/plans/render-settings-kit.md#A7 «Тайл VAE»: 1024 только при VRAM ≥ 24',
    note: 'тайл 1024 держит пик декода 15.0 ГиБ, поэтому автоматически берётся только на 24 ГиБ+',
  },
  'window.minCanvasH': {
    value: 1600,
    unit: 'px',
    kind: 'measured',
    source: 'AGENTS.md#§4 «Скользящее окно внимания»: выигрыш измерен с 1792p (314 с против 376 с)',
    note: 'ниже ~1600p окно покупает время только за счёт резкости (14.8 против 15.2), смысла нет',
  },
  'window.minCoverage': {
    value: 0.60,
    unit: 'доля',
    kind: 'measured',
    source: 'AGENTS.md#§4 «Скользящее окно внимания»: «запас 128 px (36 %) → сцена размножилась '
      + 'по вертикали; запас 384 px (64 %) → корректно» (1792p, окно 384)',
    note: 'порог намеренно ниже измеренного рабочего покрытия 0.64 (384/384 на 1792p), чтобы ладдер '
      + 'не увеличивал запас сверх проверенного конфига; ниже 0.60 запас всё равно добирается '
      + 'кратным 32 до порога, а если и это не помогает — окно выключается (window-coverage-low)',
  },
  'window.defaultPx': {
    value: 384,
    unit: 'px',
    kind: 'measured',
    source: 'AGENTS.md#§4 «Скользящее окно внимания»: 1792p — окно 384 px с запасом 384 px (10.3 ГиБ, 314 с)',
    note: 'ширина полосы по умолчанию для обычных канвасов',
  },
  'window.defaultMarginPx': {
    value: 384,
    unit: 'px',
    kind: 'measured',
    source: 'AGENTS.md#§4 «Скользящее окно внимания»: покрытие 64 % на 1792p достигнуто запасом 384 px',
    note: 'запас по умолчанию; если он не даёт покрытия 0.60 — добирается кратным 32',
  },
  'window.largePx': {
    value: 1088,
    unit: 'px',
    kind: 'measured',
    source: 'AGENTS.md#§4 «Скользящее окно внимания»: 4K 3840×2176 — окно 1088 px с запасом 448 px (513 с, 13.8 ГиБ)',
    note: 'ширина полосы в «серой» зоне (выше 5.73 Мп): там полос меньше, но каждая шире',
  },
  'window.largeMarginPx': {
    value: 448,
    unit: 'px',
    kind: 'measured',
    source: 'AGENTS.md#§4 «Скользящее окно внимания»: 4K — запас 448 px, покрытие 91 % на 2176 px',
    note: 'запас для крупного канваса (покрытие 0.91 на 2176 px)',
  },
  'ref.tokenCap': {
    value: 14600,
    unit: 'tokens',
    kind: 'measured',
    source: 'AGENTS.md#§4: 3×4128 = 12384 ток. работают; 1440p с сеткой кадра (14400) измерен'
      + ' (префикс 14632, KV 7.14 ГиБ — docs/bench/vram-tiers.md); выше — OOM энкодера на [4/5]',
    note: 'потолок энкодера Qwen3VL-8B: 14600 — верхняя измеренная граница (14400 работает),'
      + ' 16320 (2 рефа × сетка 1088p) — уже нет; поэтому 13000 из первой редакции поднят до фактической'
      + ' измеренной границы, а не наоборот',
  },
  'ref.policyPx': {
    value: 960,
    unit: 'px',
    kind: 'measured',
    source: 'AGENTS.md#§4 / docs/bench/kv-cache-offload.md: 1440p с `ref_px 960` — KV 3.70 вместо 7.14 ГиБ',
    note: 'сетка референсов по умолчанию от 1440p: сознательно грубее выхода в обмен на префикс',
  },
});

const KV_OFFLOAD_COST_PCT = MEMORY_ASSUMPTIONS['kv.offloadCostPct'].value;
const KV_NO_CACHE_COST_PCT = MEMORY_ASSUMPTIONS['kv.noCacheCostPct'].value;
const VAE_OOM_PX = MEMORY_ASSUMPTIONS['vae.omitTileOomPx'].value;
const VAE_WIDE_VRAM_GIB = MEMORY_ASSUMPTIONS['vae.wideVramGib'].value;
const WINDOW_MIN_CANVAS_H = MEMORY_ASSUMPTIONS['window.minCanvasH'].value;
const WINDOW_MIN_COVERAGE = MEMORY_ASSUMPTIONS['window.minCoverage'].value;
const WINDOW_DEFAULT_PX = MEMORY_ASSUMPTIONS['window.defaultPx'].value;
const WINDOW_DEFAULT_MARGIN_PX = MEMORY_ASSUMPTIONS['window.defaultMarginPx'].value;
const WINDOW_LARGE_PX = MEMORY_ASSUMPTIONS['window.largePx'].value;
const WINDOW_LARGE_MARGIN_PX = MEMORY_ASSUMPTIONS['window.largeMarginPx'].value;
const REF_TOKEN_CAP = MEMORY_ASSUMPTIONS['ref.tokenCap'].value;
const REF_POLICY_PX = MEMORY_ASSUMPTIONS['ref.policyPx'].value;

/**
 * Слоёв с префиксным KV-кешем у qwen21 — 32.
 *
 * `MODEL.coefficients.kvGibPerToken = 512 КиБ/токен = 32 слоя × 2 (K и V) × 4096 × 2 Б`
 * (`docs/bench/kv-cache-offload.md`), поэтому цена слоя = `gib / 32`. У H3 своего префиксного
 * KV-кеша нет вовсе (это параметр бэкенда `qwen_image21`, а не модельный).
 *
 * @type {Readonly<Record<string, number>>}
 */
export const KV_LAYERS = Object.freeze({ 'qwen21-base': 32, 'qwen21-turbo': 32 });

/**
 * Тайлы декодера VAE: шаг и ожидаемая сетка 16 px на ровных поверхностях + пик VRAM декода.
 * Все числа — из замера `AGENTS.md` §4 «Тайлинг декодера VAE» (2912×1632, fp8, LoRA off).
 *
 * `stridePx = 0.75·tilePx` (512/384, 1024/768); `0` — историческое 256/192 у diffusers.
 * `-1` — без тайлинга: пик не мерился, потому что на ≥1.5 Мп это OOM.
 *
 * @type {Readonly<Record<number, {tilePx: number, stridePx: number|null, gridArtifact: number|null, peakGib: number|null}>>}
 */
export const VAE_TILES = Object.freeze({
  512: Object.freeze({ tilePx: 512, stridePx: 384, gridArtifact: 1.69, peakGib: 7.4 }),
  0: Object.freeze({ tilePx: 256, stridePx: 192, gridArtifact: 2.71, peakGib: 7.2 }),
  1024: Object.freeze({ tilePx: 1024, stridePx: 768, gridArtifact: 1.51, peakGib: 15 }),
  [-1]: Object.freeze({ tilePx: -1, stridePx: null, gridArtifact: null, peakGib: null }),
});

/**
 * Ладдер уменьшения `ref_px` (только вниз): план даёт `0→960→768→640`, нижняя граница —
 * 256 px из `config.py::embedding_crop` (`max(256, …)`), промежуточные 512/384/320 нужны,
 * чтобы правило «≤13000 токенов референсов» выполнялось всегда, а не «обычно».
 *
 * @type {ReadonlyArray<number>}
 */
export const REF_LADDER = Object.freeze([960, 768, 640, 512, 384, 320, 256]);

/** @type {Map<string, Set<string>>} */
const SUPPORTED_KEYS = new Map();

/**
 * Объявляет ли бэкенд модели такой параметр (`settableKeys` из инвентаря, R6).
 * У H3 нет `kv_cache`/`vae_tile_px`/окна/`ref_px` — их раскладку решает сам бэкенд.
 *
 * @param {string} model
 * @param {string} path `generation.kv_cache` и т.п.
 * @returns {boolean}
 */
function supports(model, path) {
  let keys = SUPPORTED_KEYS.get(model);
  if (!keys) {
    const declared = INVENTORY.models[model];
    if (!declared) {
      throw new RangeError(`неизвестная модель ${JSON.stringify(model)}; `
        + `доступны: ${Object.keys(INVENTORY.models).join(', ')}`);
    }
    keys = new Set(settableKeys(declared.backend));
    SUPPORTED_KEYS.set(model, keys);
  }
  return keys.has(path);
}

/**
 * @typedef {Object} KvDecision
 * @property {'true'|'offload'|'false'|null} mode режим `generation.kv_cache` (`null` — не применимо)
 * @property {'vram'|'ram'|'none'} placement где живёт K/V префикса
 * @property {boolean} applicable модель объявляет `kv_cache`
 * @property {boolean|'true'|'offload'|'false'|'auto-offload-ref2'|null} requested что просил запрос
 * @property {number} prefixTokens токены префикса (текст + референсы)
 * @property {number} gib цена префикса, ГиБ
 * @property {number|null} perLayerMib цена одного слоя, МиБ
 * @property {number} costPct надбавка ко времени генерации, % (`0` для VRAM)
 * @property {number} peakGib пик без KV в VRAM, ГиБ
 * @property {number} peakWithKvGib пик, если K/V положить в VRAM, ГиБ
 * @property {number} limitGib порог правила (`0.90·budget`), ГиБ
 * @property {boolean} fitsForVram влезает ли KV в VRAM по правилу
 * @property {string} reason почему выбрано именно это
 */

/**
 * Решение по KV-кешу (A7): `true` только если пик с K/V в VRAM остаётся в зоне `≤0.90·budget`,
 * иначе `offload`; `false` — только по явному запросу.
 *
 * @param {Render} render
 * @param {boolean|'true'|'offload'|'false'|'auto-offload-ref2'|null} [requested]
 *   явный запрос (`request.memory.kvCache`); `null`/`auto-offload-ref2` — авто-политика
 * @returns {KvDecision}
 */
export function resolveKvCache(render, requested = null) {
  const prefix = prefixTokens(render);
  const gib = MODEL.coefficients.kvGibPerToken * prefix;
  if (!supports(render.model, 'generation.kv_cache')) {
    return {
      mode: null,
      placement: 'none',
      applicable: false,
      requested,
      prefixTokens: prefix,
      gib: 0,
      perLayerMib: null,
      costPct: 0,
      peakGib: estimateVram({ ...render, kvCache: 'offload' }, prefix),
      peakWithKvGib: estimateVram({ ...render, kvCache: 'offload' }, prefix),
      limitGib: GATE.degraded * render.vramGib,
      fitsForVram: true,
      reason: `model-has-no-kv-cache: ${render.model} не объявляет generation.kv_cache `
        + '(префикс не кэшируется послойно, раскладку решает бэкенд)',
    };
  }

  const layers = KV_LAYERS[render.model];
  const peakGib = estimateVram({ ...render, kvCache: 'offload' }, prefix);
  const peakWithKvGib = peakGib + gib;
  const limitGib = GATE.degraded * render.vramGib;
  const fitsForVram = peakWithKvGib <= limitGib;
  const explicitFalse = requested === false || requested === 'false';
  const explicitOffload = requested === 'offload';
  const explicitVram = requested === true || requested === 'true';

  /** @type {'true'|'offload'|'false'} */
  let mode;
  /** @type {string} */
  let reason;
  if (explicitFalse) {
    mode = 'false';
    reason = `muted-by-request: KV отключён явно (${KV_NO_CACHE_COST_PCT} % времени — префикс `
      + `считается каждый шаг; ${prefix} токенов, ${gib.toFixed(2)} ГиБ кеша не нужны)`;
  } else if (explicitOffload) {
    mode = 'offload';
    reason = `forced-by-request: KV в RAM (${gib.toFixed(2)} ГиБ, +${KV_OFFLOAD_COST_PCT} % времени)`;
  } else if (explicitVram && fitsForVram) {
    mode = 'true';
    reason = `forced-by-request: KV в VRAM влезает — пик ${peakWithKvGib.toFixed(2)} ГиБ `
      + `≤ ${limitGib.toFixed(2)} (0.90·${render.vramGib})`;
  } else if (!explicitVram && (render.refs ?? 0) >= KV_AUTO_REFS_OFFLOAD) {
    mode = 'offload';
    reason = `kv-ref2-rule: референсов ${render.refs} ≥ ${KV_AUTO_REFS_OFFLOAD} — авто-режим бэкенда `
      + '(`auto-offload-ref2`) уводит K/V в RAM: с 2 рефами на 768p KV в VRAM даёт OOM на шаге 0 '
      + `(замер); ${gib.toFixed(2)} ГиБ в RAM, +${KV_OFFLOAD_COST_PCT} % времени`;
  } else if (fitsForVram) {
    mode = 'true';
    reason = `kv-vram-fits: пик ${peakGib.toFixed(2)} + KV ${gib.toFixed(2)} = `
      + `${peakWithKvGib.toFixed(2)} ГиБ ≤ ${limitGib.toFixed(2)} (0.90·${render.vramGib}) — K/V в VRAM, `
      + `0 % времени; ${prefix} токенов × 512 КиБ`;
  } else {
    mode = 'offload';
    reason = `${explicitVram ? 'kv-vram-vs-ram' : 'kv-vram-does-not-fit'}: пик ${peakGib.toFixed(2)} + `
      + `KV ${gib.toFixed(2)} = ${peakWithKvGib.toFixed(2)} ГиБ > ${limitGib.toFixed(2)} `
      + `(0.90·${render.vramGib}) — K/V в RAM, +${KV_OFFLOAD_COST_PCT} % времени, картинка бит-в-бит`;
  }

  return {
    mode,
    placement: mode === 'true' ? 'vram' : (mode === 'offload' ? 'ram' : 'none'),
    applicable: true,
    requested,
    prefixTokens: prefix,
    gib,
    perLayerMib: layers ? (gib * 1024) / layers : null,
    costPct: mode === 'true' ? 0 : (mode === 'offload' ? KV_OFFLOAD_COST_PCT : KV_NO_CACHE_COST_PCT),
    peakGib,
    peakWithKvGib,
    limitGib,
    fitsForVram,
    reason,
  };
}

/**
 * @typedef {Object} VaeTileDecision
 * @property {number|null} tilePx выбранный `generation.sampler_kwargs.vae_tile_px`
 * @property {number|null} stridePx шаг тайла (`0.75·tile`)
 * @property {number|null} gridArtifact ожидаемая сетка 16 px на ровных поверхностях (`×`)
 * @property {number|null} peakGib пик VRAM фазы декода, ГиБ
 * @property {boolean} applicable модель объявляет тайлинг VAE
 * @property {number|null} requested что просил запрос
 * @property {boolean} degraded пик декода выше `0.90·budget`
 * @property {boolean} oom выбранный тайл не влезает в бюджет
 * @property {{value: number, why: string}|null} rejected отклонённый запрос с причиной
 * @property {string} reason почему выбрано именно это
 */

/**
 * Решение по тайлингу декодера VAE (A7): `512` по умолчанию, `1024` на широкой VRAM,
 * `0` только явным запросом, `-1` запрещён при ≥1.5 Мп (измеренный OOM).
 *
 * @param {Render} render
 * @param {number|null} [requested] явный `request.memory.vaeTilePx`
 * @returns {VaeTileDecision}
 */
export function resolveVaeTile(render, requested = null) {
  const pixels = render.canvas.width * render.canvas.height;
  if (!supports(render.model, 'generation.sampler_kwargs.vae_tile_px')) {
    return {
      tilePx: null,
      stridePx: null,
      gridArtifact: null,
      peakGib: null,
      applicable: false,
      requested,
      degraded: false,
      oom: false,
      rejected: null,
      reason: `model-has-no-vae-tiling: ${render.model} не объявляет vae_tile_px `
        + '(декодер VAE у модели свой)',
    };
  }

  /** @type {number} */
  let tile = 512;
  /** @type {string} */
  let reason;
  /** @type {{value: number, why: string}|null} */
  let rejected = null;

  if (requested === null || requested === undefined) {
    if (render.vramGib >= VAE_WIDE_VRAM_GIB) {
      tile = 1024;
      reason = `vae-tile-wide-vram: ${render.vramGib} ГиБ ≥ ${VAE_WIDE_VRAM_GIB} — тайл 1024/768, `
        + 'сетка 16 px ×1.51 (минимум артефакта)';
    } else {
      tile = 512;
      reason = `vae-tile-default: ${render.vramGib} ГиБ — тайл 512/384, сетка 16 px ×1.69 `
        + `при пике декода ${VAE_TILES[512].peakGib} ГиБ`;
    }
  } else if (requested === -1 && pixels >= VAE_OOM_PX) {
    tile = 512;
    rejected = { value: -1, why: 'vae-tile-oom-risk' };
    reason = `vae-tile-oom-risk: без тайлинга декод на ${(pixels / 1e6).toFixed(2)} Мп ≥ `
      + `${(VAE_OOM_PX / 1e6).toFixed(2)} Мп падает в OOM (замер 2912×1632), оставлен 512/384`;
  } else if (VAE_TILES[requested] === undefined) {
    throw new RangeError(`resolveVaeTile: неизвестный тайл ${JSON.stringify(requested)}; `
      + `доступны: ${Object.keys(VAE_TILES).join(', ')}`);
  } else {
    tile = requested;
    reason = `vae-tile-requested: явно ${requested}`
      + (requested === 0 ? ' (256/192, сетка ×2.71 — только для малого кадра)' : '')
      + (requested === -1 ? ' (без тайлинга; пик декода не мерился)' : '');
  }

  const info = VAE_TILES[tile];
  const peakGib = info.peakGib;
  const degraded = peakGib !== null && peakGib > GATE.degraded * render.vramGib;
  const oom = peakGib !== null && peakGib > render.vramGib;
  return {
    tilePx: tile,
    stridePx: info.stridePx,
    gridArtifact: info.gridArtifact,
    peakGib,
    applicable: true,
    requested,
    degraded,
    oom,
    rejected,
    reason,
  };
}

/**
 * @typedef {Object} AttentionDecision
 * @property {'dense'|'window'} mode режим внимания
 * @property {number} windowPx ширина полосы (`0` у плотного внимания)
 * @property {number} marginPx запас сверху/снизу, px
 * @property {number} coverage доля канваса, которую видит полоса
 * @property {number} bands число полос
 * @property {boolean} equivalentToDense окно выродилось в одну полосу = плотное внимание
 * @property {boolean} applicable модель объявляет окно
 * @property {null|{enabled: boolean, windowPx: number|null, marginPx: number|null}} requested
 * @property {string} reason почему выбрано именно это
 */

/**
 * Запас окна, добирающий покрытие до требования: кратное 32, но не больше половины канваса.
 *
 * @param {number} windowPx
 * @param {number} startMarginPx
 * @param {number} canvasH
 * @returns {{marginPx: number, coverage: number}}
 */
function marginForCoverage(windowPx, startMarginPx, canvasH) {
  let margin = Math.max(0, startMarginPx);
  /** @type {number} */
  let coverage = Math.min(1, (windowPx + 2 * margin) / canvasH);
  const maxMargin = Math.ceil(canvasH / 2 / CANVAS_GRID) * CANVAS_GRID;
  while (coverage < WINDOW_MIN_COVERAGE && margin + CANVAS_GRID <= maxMargin) {
    margin += CANVAS_GRID;
    coverage = Math.min(1, (windowPx + 2 * margin) / canvasH);
  }
  return { marginPx: margin, coverage };
}

/**
 * Решение по скользящему окну внимания (A7): включается при `canvasH ≥ 1600` и покрытии
 * `(px + 2·margin)/canvasH ≥ 0.60`; при `canvasH ≤ px` помечается `equivalentToDense`.
 *
 * @param {Render} render
 * @param {null|{enabled: boolean, windowPx: number|null, marginPx: number|null}} [requested]
 *   `request.memory.window` (`null` — авто)
 * @returns {AttentionDecision}
 */
export function resolveAttention(render, requested = null) {
  const canvasH = render.canvas.height;
  if (!supports(render.model, 'generation.sampler_kwargs.window_attn_px')) {
    return {
      mode: 'dense',
      windowPx: 0,
      marginPx: 0,
      coverage: 1,
      bands: 1,
      equivalentToDense: false,
      applicable: false,
      requested,
      reason: `model-has-no-window: ${render.model} не объявляет window_attn_px `
        + '(внимание бэкенда не режется на полосы)',
    };
  }

  const wide = render.canvas.width * canvasH > SIZE_LIMITS.okMaxPx;
  const defaultPx = wide ? WINDOW_LARGE_PX : WINDOW_DEFAULT_PX;
  const defaultMargin = wide ? WINDOW_LARGE_MARGIN_PX : WINDOW_DEFAULT_MARGIN_PX;

  if (requested !== null && requested.enabled === false) {
    return {
      mode: 'dense',
      windowPx: 0,
      marginPx: 0,
      coverage: 1,
      bands: 1,
      equivalentToDense: false,
      applicable: true,
      requested,
      reason: 'window-dense: плотное внимание — резче на 2.6 %, но на крупных канвасах дороже '
        + 'по времени (окно снимает ≈ 17 %); VRAM оба режима занимают одинаково',
    };
  }

  const explicit = requested !== null && requested.enabled === true;
  if (!explicit && canvasH < WINDOW_MIN_CANVAS_H) {
    return {
      mode: 'dense',
      windowPx: 0,
      marginPx: 0,
      coverage: 1,
      bands: 1,
      equivalentToDense: false,
      applicable: true,
      requested,
      reason: `window-not-needed: канвас ${canvasH} px < ${WINDOW_MIN_CANVAS_H} px — выигрыш окна `
        + 'по времени измерен только с 1792p, а резкость оно теряет всегда',
    };
  }

  const windowPx = (explicit && requested.windowPx !== null) ? requested.windowPx : defaultPx;
  const startMargin = (explicit && requested.marginPx !== null) ? requested.marginPx : defaultMargin;
  const { marginPx, coverage } = marginForCoverage(windowPx, startMargin, canvasH);
  const bands = Math.max(1, Math.ceil(canvasH / windowPx));
  const equivalentToDense = canvasH <= windowPx;

  if (equivalentToDense) {
    return {
      mode: 'window',
      windowPx,
      marginPx,
      coverage: 1,
      bands: 1,
      equivalentToDense,
      applicable: true,
      requested,
      reason: `window-equivalent-to-dense: канвас ${canvasH} px ≤ окна ${windowPx} px — одна полоса, `
        + 'результат обязан совпасть с плотным вниманием (побитовый гейт корректности); '
        + 'как отдельный кандидат не дублируется',
    };
  }

  if (coverage < WINDOW_MIN_COVERAGE) {
    return {
      mode: 'dense',
      windowPx: 0,
      marginPx: 0,
      coverage: 1,
      bands: 1,
      equivalentToDense: false,
      applicable: true,
      requested,
      reason: `window-coverage-low: покрытие ${(coverage * 100).toFixed(1)} % < `
        + `${(WINDOW_MIN_COVERAGE * 100).toFixed(0)} % даже с запасом ${marginPx} px — `
        + 'полоса не видит и 60 % канваса, сцена размножается; остаётся плотное внимание',
    };
  }

  return {
    mode: 'window',
    windowPx,
    marginPx,
    coverage,
    bands,
    equivalentToDense: false,
    applicable: true,
    requested,
    reason: explicit
      ? `window-forced: окно запрошено явно — ${bands} полос по ${windowPx} px с запасом `
        + `±${marginPx} px, покрытие ${(coverage * 100).toFixed(1)} % (VRAM не меняет, `
        + 'резкость −2.6 %)'
      : `window-on: канвас ${canvasH} px ≥ ${WINDOW_MIN_CANVAS_H} и покрытие `
        + `${(coverage * 100).toFixed(1)} % ≥ ${(WINDOW_MIN_COVERAGE * 100).toFixed(0)} % — `
        + `${bands} полос по ${windowPx} px с запасом ±${marginPx} px`,
  };
}

/**
 * @typedef {Object} RefPxDecision
 * @property {number} refPx выбранный `generation.sampler_kwargs.ref_px` (`0` = сетка канваса)
 * @property {number} refWidth ширина сетки референса после клампа/снапа, px
 * @property {number} refHeight высота сетки референса, px
 * @property {number} tokensPerRef токены одного референса `(W/16)·(H/16)`
 * @property {number} refTokens токены всех референсов
 * @property {number} capTokens потолок токенов референсов
 * @property {boolean} applicable модель объявляет `ref_px`
 * @property {number|null} requested что просил запрос
 * @property {boolean} clamped запрос был больше канваса/не на сетке 32
 * @property {boolean} reduced значение уменьшено правилом ≤`capTokens`
 * @property {string} reason почему выбрано именно это
 */

/**
 * Решение по `ref_px` (A7): это точность референса против бюджета KV, а не рычаг скорости.
 * Политика: `0` при канвасе ≤1088 px и ≤2 рефах, `960` от 1440p; больше канваса не бывает
 * (кламп + `ref-px-clamped`); при превышении ≈13000 токенов референсов значение спускается
 * по ладдеру. Инвариант: сетка референса никогда не больше сетки кадра.
 *
 * @param {Render} render
 * @param {number|null} [requested] явный `ref_px` (из YAML/CLI; схема запроса его пока не несёт)
 * @returns {RefPxDecision}
 */
export function resolveRefPx(render, requested = null) {
  const canvas = render.canvas;
  const canvasH = canvas.height;
  const refs = render.refs ?? 0;
  if (!supports(render.model, 'generation.sampler_kwargs.ref_px')) {
    const grid = refGrid(0, canvas);
    return {
      refPx: 0,
      ...grid,
      refTokens: refs * grid.tokensPerRef,
      capTokens: REF_TOKEN_CAP,
      applicable: false,
      requested,
      clamped: false,
      reduced: false,
      reason: `model-has-no-ref-px: ${render.model} не объявляет ref_px `
        + '(референсы идут в энкодер как есть)',
    };
  }
  if (!Number.isInteger(refs) || refs < 0) {
    throw new RangeError(`resolveRefPx: refs должен быть целым ≥ 0, получено ${refs}`);
  }

  /** @type {number} */
  let refPx;
  /** @type {string} */
  let reason;
  let clamped = false;
  let reduced = false;

  if (requested === null || requested === undefined) {
    // Политика по умолчанию: ≤1088p и ≤2 рефа — сетка кадра (максимум точности, префикс влезает),
    // от 1440p — 960 px (префикс 3.70 вместо 7.14 ГиБ при KV в VRAM).
    refPx = (canvasH <= 1088 && refs <= 2) ? 0 : REF_POLICY_PX;
    reason = refPx === 0
      ? `ref-px-canvas: канвас ${canvasH} px ≤ 1088 и референсов ${refs} ≤ 2 — сетка кадра `
        + '(максимум точности референса, префикс влезает)'
      : `ref-px-budget: канвас ${canvasH} px — сетка референсов ${REF_POLICY_PX} px вместо `
        + `${canvasH} (префикс KV дешевле, референс сознательно грубее выхода)`;
  } else {
    if (!Number.isInteger(requested) || requested < 0) {
      throw new RangeError(`resolveRefPx: ref_px должен быть целым ≥ 0, получено ${requested}`);
    }
    refPx = requested;
  }

  const requestedValue = refPx;
  if (refPx > 0) {
    const heightClamped = Math.max(256, Math.min(refPx, canvasH));
    const snapped = Math.max(256, Math.floor(heightClamped / CANVAS_GRID) * CANVAS_GRID);
    if (refPx > canvasH || snapped !== refPx) clamped = true;
    refPx = snapped;
    if (requested !== null && requested !== undefined) {
      reason = clamped
        ? `ref-px-clamped: ref_px ${requestedValue} > канваса ${canvasH} px — кламп до ${refPx} px `
          + '(сетка референса не бывает больше сетки кадра)'
        : `ref-px-requested: явно ${refPx} px`;
    }
  }

  // Потолок токенов референсов работает и для сетки кадра (`ref_px: 0`): при 2+ рефах на ≥1088p
  // кадр-сетка даёт 2×8160 = 16320 токенов > 13000, и энкодер падает в OOM на [4/5]. Поэтому
  // ладдер спускается ниже кадра, а не молча оставляет "0".
  let grid = refGrid(refPx, canvas);
  const capLimit = refPx > 0 ? refPx : canvasH;
  if (refs * grid.tokensPerRef > REF_TOKEN_CAP) {
    const ladder = REF_LADDER.filter((value) => value < capLimit);
    for (const candidate of ladder) {
      const next = refGrid(candidate, canvas);
      if (refs * next.tokensPerRef <= REF_TOKEN_CAP) {
        reason = `refs-token-cap: ${refs} реф. × ${grid.tokensPerRef} токенов > `
          + `${REF_TOKEN_CAP} — сетка референса уменьшена до ${candidate} px `
          + `(${refs} × ${next.tokensPerRef} = ${refs * next.tokensPerRef} ≤ ${REF_TOKEN_CAP})`;
        refPx = candidate;
        grid = next;
        reduced = true;
        break;
      }
    }
    if (!reduced) {
      const floorValue = REF_LADDER[REF_LADDER.length - 1];
      refPx = floorValue;
      grid = refGrid(floorValue, canvas);
      reduced = true;
      reason = `refs-token-cap: даже ${floorValue} px даёт ${refs * grid.tokensPerRef} токенов > `
        + `${REF_TOKEN_CAP} — значение оставлено минимальным, референсы придётся урезать`;
    }
  }

  return {
    refPx,
    ...grid,
    refTokens: refs * grid.tokensPerRef,
    capTokens: REF_TOKEN_CAP,
    applicable: true,
    requested,
    clamped,
    reduced,
    reason,
  };
}

/**
 * Вся раскладка памяти одним объектом — вход `Plan.memory` (шаг 5) и панель памяти в UI.
 * Порядок осей фиксирован: KV → VAE → окно → референсы (так же, как в ладдере A7).
 *
 * @param {Render} render
 * @param {{
 *   kvCache?: boolean|'true'|'offload'|'false'|'auto-offload-ref2'|null,
 *   vaeTilePx?: number|null,
 *   window?: null|{enabled: boolean, windowPx: number|null, marginPx: number|null},
 *   refPx?: number|null
 * }} [requested] `request.memory`
 * @returns {{kvCache: KvDecision, vae: VaeTileDecision, attention: AttentionDecision, ref: RefPxDecision}}
 */
export function resolveMemory(render, requested = {}) {
  return {
    kvCache: resolveKvCache(render, requested.kvCache ?? null),
    vae: resolveVaeTile(render, requested.vaeTilePx ?? null),
    attention: resolveAttention(render, requested.window ?? null),
    ref: resolveRefPx(render, requested.refPx ?? null),
  };
}

/**
 * Модель без своих параметров раскладки (H3): удобно для гейтов шага 5.
 *
 * @param {string} model
 * @returns {boolean}
 */
export function hasMemoryLayout(model) {
  return supports(model, 'generation.kv_cache');
}
