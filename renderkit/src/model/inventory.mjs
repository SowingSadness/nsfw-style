// @ts-check
/**
 * Возможности моделей/энкодеров/лимиты — API поверх **сгенерированных** данных
 * (`src/data/inventory.mjs`, источник — Python-пакет `qwenstyle`, см. `tools/gen-inventory.mjs`).
 *
 * Здесь нет ни одного числа: всё приходит из `INVENTORY`. Имён пресетов в данных тоже нет —
 * калькулятор знает только модели (`qwen21-base` / `qwen21-turbo` / `h3-image`), бэкенды,
 * энкодеры, тарифы по VRAM и объявленные `ParameterSpec`-дескрипторы.
 *
 * Функции чистые (никакого I/O), поэтому модуль годится и браузеру, и тестам, и сервису.
 */

import { INVENTORY } from '../data/inventory.mjs';

export { INVENTORY };

/** Id моделей в стабильном порядке (порядок задаёт генератор данных). */
export const MODEL_IDS = Object.freeze(Object.keys(INVENTORY.models));
/** Id энкодеров в стабильном порядке (алфавитный — так их пишет генератор). */
export const ENCODER_IDS = Object.freeze(Object.keys(INVENTORY.encoders));
/** Имена тарифов по VRAM в порядке возрастания железа (`VRAM8` → `VRAM16`). */
export const TIER_NAMES = Object.freeze(Object.keys(INVENTORY.tiers));

/**
 * Модель по id или исключение со списком доступных (опечатка не должна молча давать `undefined`).
 *
 * @param {string} id
 * @returns {typeof INVENTORY.models[keyof typeof INVENTORY.models]}
 */
function modelById(id) {
  const model = INVENTORY.models[id];
  if (!model) {
    throw new RangeError(`неизвестная модель ${JSON.stringify(id)}; доступны: ${MODEL_IDS.join(', ')}`);
  }
  return model;
}

/**
 * Тариф по имени (регистр не важен); неизвестное имя — исключение со списком доступных.
 *
 * @param {string} tier
 * @returns {string}
 */
function tierName(tier) {
  const key = String(tier ?? '').trim().toUpperCase();
  const found = TIER_NAMES.find((name) => name.toUpperCase() === key);
  if (!found) {
    throw new RangeError(`неизвестный тариф ${JSON.stringify(tier)}; доступны: ${TIER_NAMES.join(', ')}`);
  }
  return found;
}

/**
 * Объём VRAM тарифа в ГиБ — из объявленного `hardware` (`"8 GB VRAM / 16 GB RAM"`),
 * с запасным разбором самого имени (`VRAM8`).
 *
 * @param {string} tier
 * @returns {number}
 */
function tierVramGib(tier) {
  const name = tierName(tier);
  const hardware = INVENTORY.tiers[name].hardware;
  const fromHardware = /(\d+(?:\.\d+)?)\s*GB\s*VRAM/iu.exec(hardware);
  if (fromHardware) return Number(fromHardware[1]);
  const fromName = /(\d+(?:\.\d+)?)/u.exec(name);
  if (fromName) return Number(fromName[1]);
  throw new RangeError(`тариф ${name}: не удалось определить VRAM из hardware ${JSON.stringify(hardware)}`);
}

/**
 * Все модели (объекты в стабильном порядке). Поля: `id`, `backend`, `repo`, `transformer`,
 * `precision`, `precision_cache`, `fixedSteps`, `weightGibBf16`, `weightGibFp8`, `encoderId`, `lora`.
 *
 * @returns {Array<typeof INVENTORY.models[keyof typeof INVENTORY.models]>}
 */
export function models() {
  return MODEL_IDS.map((id) => INVENTORY.models[id]);
}

/**
 * Возможности модели в форме, которой пользуются cost/candidates: лимиты берутся из общих
 * `limits` (они объявлены бэкендами), веса и адаптер — из самой модели.
 *
 * `encoder` — id энкодера (разворачивается через `encoderInfo`), `lora` — адаптер модели
 * (`null` = модель не поддерживает LoRA; `lora.scale: null` = сила не задекларирована пресетами).
 *
 * @param {string} id
 * @returns {{
 *   fixedSteps: number,
 *   maxImages: number | null,
 *   precisions: readonly string[],
 *   offloadModes: readonly string[],
 *   weightGibBf16: number,
 *   weightGibFp8: number,
 *   encoder: string,
 *   lora: {repo: string, filename: string, scale: number | null, strip_alpha: boolean} | null
 * }}
 */
export function modelCaps(id) {
  const model = modelById(id);
  return {
    fixedSteps: model.fixedSteps,
    maxImages: INVENTORY.limits.maxImages,
    precisions: INVENTORY.limits.precisions,
    offloadModes: INVENTORY.limits.offloadModes,
    weightGibBf16: model.weightGibBf16,
    weightGibFp8: model.weightGibFp8,
    encoder: model.encoderId,
    lora: model.lora,
  };
}

/**
 * Декларация энкодера: `name`, `cache_tag`, `repo`, `revision`, `local_dir`, `requires_vision`,
 * `dtype`, `device`, `stream`, `prefetch`, `pin_memory`, `lookahead`, `memory_gib`.
 *
 * @param {string} id
 * @returns {typeof INVENTORY.encoders[keyof typeof INVENTORY.encoders]}
 */
export function encoderInfo(id) {
  const encoder = INVENTORY.encoders[id];
  if (!encoder) {
    throw new RangeError(`неизвестный энкодер ${JSON.stringify(id)}; доступны: ${ENCODER_IDS.join(', ')}`);
  }
  return encoder;
}

/**
 * Whitelist ключей `--set` для бэкенда (только не-`readonly` спеки, в порядке объявления).
 *
 * @param {string} backend
 * @returns {string[]}
 */
export function settableKeys(backend) {
  const keys = INVENTORY.settable[backend];
  if (!keys) {
    throw new RangeError(`неизвестный бэкенд ${JSON.stringify(backend)}; `
      + `доступны: ${Object.keys(INVENTORY.settable).join(', ')}`);
  }
  return [...keys];
}

/**
 * Полные дескрипторы параметров бэкенда (`ParameterSpec`: scope/name/type/default/choices/
 * minimum/maximum/readonly/path/doc) — для валидации YAML и подсказок в UI.
 *
 * @param {string} backend
 * @returns {Array<{name: string, scope: string, path: string, type: string, default: unknown,
 *   choices: string[] | null, minimum: number | null, maximum: number | null,
 *   readonly: boolean, doc: string}>}
 */
export function paramSpecs(backend) {
  const specs = INVENTORY.params[backend];
  if (!specs) {
    throw new RangeError(`неизвестный бэкенд ${JSON.stringify(backend)}; `
      + `доступны: ${Object.keys(INVENTORY.params).join(', ')}`);
  }
  return [...specs];
}

/**
 * Ближайший тариф по VRAM — **только для информационного поля** запроса (`request.tier`),
 * выбор раскладки памяти делает cost/memory по фактическому бюджету.
 *
 * При равном расстоянии берётся тариф с меньшей VRAM (консервативнее).
 *
 * @param {number} vramGib
 * @returns {string} имя тарифа
 */
export function tierFor(vramGib) {
  const budget = Number(vramGib);
  if (!Number.isFinite(budget)) {
    throw new TypeError(`tierFor(vramGib=${JSON.stringify(vramGib)}): ожидается число ГиБ`);
  }
  /** @type {{name: string, vram: number, distance: number} | null} */
  let best = null;
  for (const name of TIER_NAMES) {
    const vram = tierVramGib(name);
    const distance = Math.abs(vram - budget);
    if (best === null || distance < best.distance || (distance === best.distance && vram < best.vram)) {
      best = { name, vram, distance };
    }
  }
  if (best === null) throw new RangeError('tierFor: в inventory нет ни одного тарифа');
  return best.name;
}

/**
 * Полный след тарифа на диске (ГиБ): загрузка + производные кэши (`profiles.disk_gib`).
 *
 * @param {string} tier
 * @returns {number}
 */
export function diskFootprint(tier) {
  return INVENTORY.tiers[tierName(tier)].diskGib;
}

/**
 * Сколько качается по тарифу (ГиБ) — без производных кэшей (`profiles.download_estimate`).
 *
 * @param {string} tier
 * @returns {number}
 */
export function downloadGib(tier) {
  return INVENTORY.tiers[tierName(tier)].downloadGib;
}

/**
 * Лимиты, объявленные бэкендами: `maxImages`, `fixedSteps` (по моделям с фиксированным
 * расписанием), `precisions`, `offloadModes`, `aspectRatios`.
 *
 * @returns {typeof INVENTORY.limits}
 */
export function limits() {
  return INVENTORY.limits;
}

/**
 * Дайджест входных данных генератора (`configs` — резолвленные пресеты, `profiles` — манифесты).
 * Из него шаг 6 собирает `rulesetVersion: "r1+<sha12>"`.
 *
 * @returns {{configs: string, profiles: string}}
 */
export function inventoryDigest() {
  return INVENTORY.source.digest;
}
