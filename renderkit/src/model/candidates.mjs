// @ts-check
/**
 * Сочетания настроек рендера (шаг 5 плана, ADR A3/A6/A7).
 *
 * Сочетание — это полный набор осей, которые реально меняют кадр, время или раскладку памяти:
 *
 * ```
 * steps × offload × kvCache × vaeTilePx × window × encoderDevice × loraScale
 * ```
 *
 * Канвас — не сочетание, а запрос: он приходит из `geometry.resolveGeometry` (A4) и у всех
 * сочетаний одного плана один и тот же. Шаги — ось **скорости/качества**, не памяти (решение 5
 * плана): `cost.estimateVram` их не принимает вовсе.
 *
 * Каждая ось начинается с **авто-значения** (`memory.resolveKvCache/VaeTile/Attention/RefPx` и
 * политика уровня), поэтому первое сочетание в списке — «всё как решила политика», а остальные —
 * альтернативы, между которыми выбирает `rank.mjs`. Ось `encoder` не варьируется (значение следует
 * из бюджета VRAM, A7), ось `lora` тоже (`anime` — булев флаг запроса). Дедуп — по ключу конфига и
 * канвасу: два сочетания, дающие один и тот же кадр и одинаковую раскладку, не дублируются
 * (выродившееся окно `equivalentToDense` тоже).
 *
 * Гейты (плановые): `peak > 0.97·budget` — сочетание отбрасывается, `> 0.90·budget` — остаётся
 * (`degraded`); плюс `vae.oom` (декод VAE не влезает), запас VRAM уровня (`levels.minHeadroom`),
 * политика окна (`levels.allowWindow`) и пик RAM. В «серой» зоне (>5.73 Мп), где уровень уже
 * деградирован, жёсткий порог — `1.00·budget` (см. `CANDIDATE_ASSUMPTIONS['gate.greyZoneLimit']`).
 *
 * Модуль изоморфный: без `node:`/DOM. Числа — из `data/model.mjs`, `data/inventory.mjs` и
 * допущений с провенансом.
 */

import { INVENTORY, encoderInfo, limits, modelCaps } from './inventory.mjs';
import { GATE, estimateAll, estimateVram, residentWeights } from './cost.mjs';
import { LEVEL_POLICY } from './levels.mjs';
import {
  resolveAttention,
  resolveKvCache,
  resolveRefPx,
  resolveVaeTile,
} from './memory.mjs';
import { qualityScore } from './quality.mjs';

/** @typedef {import('./cost.mjs').Render} Render */
/** @typedef {import('./cost.mjs').Estimates} Estimates */
/** @typedef {import('./memory.mjs').KvDecision} KvDecision */
/** @typedef {import('./memory.mjs').VaeTileDecision} VaeTileDecision */
/** @typedef {import('./memory.mjs').AttentionDecision} AttentionDecision */
/** @typedef {import('./memory.mjs').RefPxDecision} RefPxDecision */
/** @typedef {import('./quality.mjs').QualityScore} QualityScore */
/** @typedef {import('./geometry.mjs').Geometry} Geometry */
/** @typedef {import('./levels.mjs').Levels} Levels */
/** @typedef {import('../schema.mjs').NormalizedRequest} NormalizedRequest */

/**
 * Допущения генератора сочетаний (провенанс — как у `MODEL.provenance`).
 *
 * @type {Readonly<Record<string, {value: number, unit: string, kind: 'measured'|'assumption', source: string, note: string}>>}
 */
export const CANDIDATE_ASSUMPTIONS = Object.freeze({
  'encoder.cpuMaxVramGib': {
    value: 8,
    unit: 'GiB',
    kind: 'measured',
    source: 'AGENTS.md#§0 «8 ГБ VRAM — только тариф на 8 ГиБ»: там энкодер идёт на CPU '
      + '(`encoder.device: cpu` + диск-стриминг), на 16 ГиБ — GPU-стриминг (пик VRAM ~1 ГиБ)',
    note: 'правило A7: `cpu` при `vram ≤ 8`. На 8 ГиБ GPU-энкод выбивает тариф, поэтому ось '
      + '`encoderDevice` не варьируется — значение следует из бюджета VRAM',
  },
  'gate.greyZoneLimit': {
    value: 1,
    unit: 'доля бюджета',
    kind: 'assumption',
    source: 'src/data/measurements.mjs#agents-4k-window: 3840×2176 (13.8 ГиБ процесса = 15.85 GiB '
      + 'nvidia-smi) отработал за 513 с; жёсткий гейт 0.97·16 = 15.52 этот замер отбрасывал бы',
    note: 'в «серой» зоне (> 5.73 Мп) политика уже деградирована (`comfort-limit`), поэтому порог '
      + '0.97·budget смягчается до 1.00·budget: сочетание остаётся, а план получает '
      + '`vram-spill-risk`. Без этого решение 8 плана («4K рендерится, но фактура уходит в рябь») '
      + 'невыразимо: оценка 4K — 15.79 из 16 ГиБ',
  },
  'offload.blocks': {
    value: 1,
    unit: 'группа',
    kind: 'measured',
    source: 'AGENTS.md#§4 «Режим --offload group»: `--offload-blocks` 1 против 8 — ≤4 % по времени, '
      + 'пик VRAM тот же (активации доминируют)',
    note: 'мельче группа не хуже крупной, поэтому 1 — умолчание: ось `offloadBlocks` не варьируется',
  },
});

const ENCODER_CPU_MAX_VRAM_GIB = CANDIDATE_ASSUMPTIONS['encoder.cpuMaxVramGib'].value;
const GREY_ZONE_LIMIT = CANDIDATE_ASSUMPTIONS['gate.greyZoneLimit'].value;

/** Группа offload-блоков плана (`generation.sampler_kwargs.offload_blocks`). */
export const OFFLOAD_BLOCKS = CANDIDATE_ASSUMPTIONS['offload.blocks'].value;

/**
 * Устройство энкодера по A7: `cpu` на 8 ГиБ VRAM и меньше, иначе `cuda` (GPU-стриминг).
 *
 * @param {number} vramGib
 * @returns {'cuda'|'cpu'}
 */
export function encoderDeviceFor(vramGib) {
  return vramGib <= ENCODER_CPU_MAX_VRAM_GIB ? 'cpu' : 'cuda';
}

/**
 * Поля энкодера для `Plan.render.encoder`: имя/устройство/стриминг/префетч.
 *
 * Стриминг включён **и на CPU**: 16.33 ГиБ весов heretic-8B в RAM целиком не поднимаются
 * (`cache/models/weights/heretic-te` читается позиционно), а конфигурация для 8 ГиБ держит
 * `device: cpu` вместе с `stream: true` и `prefetch: 4` — AGENTS §0/§4. Поэтому единственное,
 * что меняется с устройством, — это `device` (префетч берётся из декларации энкодера).
 *
 * @param {string} model
 * @param {number} vramGib
 * @returns {{name: string, device: 'cuda'|'cpu', stream: boolean, prefetch: number}}
 */
export function encoderProfile(model, vramGib) {
  const caps = modelCaps(model);
  const declared = encoderInfo(caps.encoder);
  const device = encoderDeviceFor(vramGib);
  return {
    name: declared.name,
    device,
    stream: true,
    prefetch: declared.prefetch,
  };
}

/**
 * Точность трансформера, объявленная моделью (`fp8` у всех трёх).
 *
 * @param {string} model
 * @returns {string}
 */
export function modelPrecision(model) {
  const declared = INVENTORY.models[model];
  if (!declared) {
    throw new RangeError(`modelPrecision: неизвестная модель ${JSON.stringify(model)}; `
      + `доступны: ${Object.keys(INVENTORY.models).join(', ')}`);
  }
  return declared.precision;
}

/**
 * Сила LoRA: `0` без стилевого адаптера, сила адаптера модели (0.7 у FUSAL) при `anime: true`.
 * У модели без адаптера (H3) сила остаётся `0` — это и означает «адаптера нет».
 *
 * @param {{anime: boolean}} request
 * @param {string} model
 * @returns {number}
 */
export function loraScaleFor(request, model) {
  if (!request.anime) return 0;
  const caps = modelCaps(model);
  if (!caps.lora) return 0;
  return caps.lora.scale ?? 0;
}

/**
 * Порог жёсткого гейта VRAM: `0.97·budget`, в «серой» зоне (если уровень её разрешает) — `1.00·budget`.
 *
 * @param {{hardware: {vramGib: number}}} request
 * @param {Geometry} geometry
 * @param {Levels} levels
 * @returns {{limitGib: number, greyZone: boolean}}
 */
export function gateLimit(request, geometry, levels) {
  const budget = request.hardware.vramGib;
  const greyZone = geometry.verdict === 'degraded' && levels.allowDegradedZone;
  return { limitGib: (greyZone ? GREY_ZONE_LIMIT : GATE.invalid) * budget, greyZone };
}

/**
 * Ось шагов: значение уровня плюс короткое расписание `draft` (8) у моделей без фиксированного
 * расписания. Явный `image.steps` и фиксированное расписание (H3 = 6 PDD) дают одну точку.
 *
 * @param {Levels} levels
 * @returns {number[]}
 */
export function stepsAxis(levels) {
  if (levels.fixed || levels.override !== null) return [levels.steps];
  const draftSteps = LEVEL_POLICY.draft.steps;
  const values = new Set([levels.steps]);
  if (draftSteps !== null && draftSteps < levels.fullSteps) values.add(draftSteps);
  return [...values];
}

/**
 * Объявляет ли модель режим offload (таблица резидентных весов — `null` = режим не поддержан).
 *
 * @param {string} model
 * @param {'none'|'group'} mode
 * @returns {boolean}
 */
export function supportsOffload(model, mode) {
  if (!/** @type {string[]} */ (limits().offloadModes).includes(mode)) return false;
  try {
    residentWeights(model, mode);
    return true;
  } catch {
    return false;
  }
}

/**
 * Авто-режим offload: `none` (быстрее), если пик без выгрузки влезает в гейт, иначе `group`.
 *
 * @param {Render} render
 * @param {number} limitGib
 * @returns {'none'|'group'}
 */
export function autoOffload(render, limitGib) {
  if (supportsOffload(render.model, 'none')) {
    const peak = estimateVram({ ...render, offload: 'none', kvCache: 'offload' });
    if (peak <= limitGib) return 'none';
  }
  if (supportsOffload(render.model, 'group')) return 'group';
  throw new RangeError(`autoOffload: модель ${render.model} не умеет ни none, ни group`);
}

/**
 * @typedef {Object} CandidateGates
 * @property {boolean} fits пик VRAM в гейте (`peak ≤ limitGib`)
 * @property {boolean} vaeFit декод VAE не даёт OOM
 * @property {boolean} ramFit пик RAM в бюджете (информационно: RAM не отбрасывает сочетание, а даёт `ram-pagecache-thrash`)
 * @property {boolean} headroomOk запас VRAM не меньше политики уровня
 * @property {boolean} windowOk окно разрешено уровнем (или запрошено явно)
 * @property {'ok'|'degraded'|'spill'} zone зона VRAM из `cost.gateBudget`
 * @property {number} limitGib порог гейта, ГиБ
 */

/**
 * @typedef {Object} CandidateLayout
 * @property {KvDecision} kvCache
 * @property {VaeTileDecision} vae
 * @property {AttentionDecision} attention
 * @property {RefPxDecision} ref
 */

/**
 * @typedef {Object} Candidate
 * @property {string} id стабильный id (`c01`, `c02`, …) — по порядку генерации
 * @property {string} key ключ дедупа: конфиг + канвас
 * @property {Render} render полный вход `cost.mjs`
 * @property {CandidateLayout} layout решения по осям памяти
 * @property {Estimates} estimates оценки VRAM/RAM/времени/диска
 * @property {QualityScore} quality Q по A6
 * @property {{q: number, timeS: number, headroomGib: number}} metrics метрики Парето/ранга
 * @property {CandidateGates} gates гейты сочетания
 */

/**
 * Каноническая раскладка (все оси — авто), даже если она не проходит гейты: нужна ладдеру
 * (`resolve.mjs`) и `trace` (`explain.mjs`), чтобы объяснить, с чего начинали.
 *
 * @typedef {Object} Canonical
 * @property {Render} render
 * @property {CandidateLayout} layout
 * @property {Estimates} estimates
 * @property {QualityScore} quality
 * @property {{q: number, timeS: number, headroomGib: number}} metrics те же метрики ранга, что у сочетаний
 * @property {CandidateGates} gates
 */

/**
 * @typedef {Object} CandidatesResult
 * @property {Candidate[]} candidates сочетания, прошедшие гейты (в порядке генерации)
 * @property {Canonical} canonical каноническая раскладка (все оси — авто)
 * @property {number} total сколько сочетаний сгенерировано до гейтов
 * @property {number} limitGib порог гейта, ГиБ
 * @property {boolean} greyZone порог смягчён «серой» зоной
 */

/**
 * Все сочетания запроса, прошедшие гейты, в детерминированном порядке (первое — каноническое).
 *
 * @param {NormalizedRequest} request нормализованный запрос
 * @param {{geometry: Geometry, levels: Levels, offload?: 'none'|'group'|null,
 *   refPx?: number|null}} context `offload`/`refPx` — жёсткие override ладдера
 * @returns {CandidatesResult}
 */
export function buildCandidates(request, context) {
  const { geometry, levels } = context;
  const model = request.hardware.model;
  const { limitGib, greyZone } = gateLimit(request, geometry, levels);

  /** @type {Render} */
  const base = {
    model,
    canvas: { width: geometry.canvas.width, height: geometry.canvas.height },
    tokens: geometry.tokens,
    steps: levels.steps,
    offload: 'none',
    kvCache: null,
    windowAttnPx: 0,
    marginPx: 0,
    vaeTilePx: null,
    refs: request.refs,
    refPx: 0,
    encoderDevice: encoderDeviceFor(request.hardware.vramGib),
    vramGib: request.hardware.vramGib,
    ramGib: request.hardware.ramGib,
    disk: request.hardware.disk,
    bitDepth: request.image.bitDepth,
  };

  // Референсы первыми: их сетка входит в префикс KV, а префикс — в решение по KV.
  const ref = resolveRefPx(base, context.refPx ?? null);
  base.refPx = ref.refPx;

  const auto = context.offload ?? autoOffload(base, limitGib);
  base.offload = auto;
  const kv = resolveKvCache(base, request.memory.kvCache ?? null);
  const vae = resolveVaeTile(base, request.memory.vaeTilePx ?? null);
  const attention = resolveAttention(base, request.memory.window ?? null);
  base.kvCache = kv.mode ?? false;
  base.vaeTilePx = vae.tilePx;
  base.windowAttnPx = attention.windowPx;
  base.marginPx = attention.marginPx;

  const offloadList = offloadAxis(model, auto);
  const kvList = kvAxis(kv, request.memory.kvCache);
  const tileList = tileAxis(vae, request.memory.vaeTilePx);
  const windowList = windowAxis(base, attention, request.memory.window);
  const stepsList = stepsAxis(levels);

  /** @type {Candidate[]} */
  const candidates = [];
  /** @type {Canonical|null} */
  let canonical = null;
  const seen = new Set();
  let total = 0;

  for (const steps of stepsList) {
    for (const offload of offloadList) {
      for (const kvMode of kvList) {
        for (const tile of tileList) {
          for (const win of windowList) {
            total += 1;
            /** @type {Render} */
            const render = {
              ...base,
              steps,
              offload,
              kvCache: kvMode === null ? false : kvMode,
              vaeTilePx: tile,
              windowAttnPx: win.decision.windowPx,
              marginPx: win.decision.marginPx,
            };
            const kvc = kvDecisionFor(render, kvMode, request.memory.kvCache ?? null);
            const vaeDecision = vaeDecisionFor(render, tile, request.memory.vaeTilePx ?? null);
            // Эффективные значения осей: политика могла понизить запрошенное (например `-1` тайла
            // на ≥1.5 Мп — измеренный OOM), и оценка/гейты обязаны считать именно их.
            render.kvCache = kvc.mode ?? false;
            render.vaeTilePx = vaeDecision.tilePx;
            const estimates = applyGreyZone(estimateAll(render, { kvCache: kvModeOf(render) }), limitGib, greyZone);
            const quality = qualityScore({
              pixels: geometry.pixels,
              steps,
              attention: win.decision.mode,
            });
            const layout = { kvCache: kvc, vae: vaeDecision, attention: win.decision, ref };
            const gates = gateCandidate({
              estimates, vae: vaeDecision, attention: win.decision, levels, request, limitGib,
            });
            const metrics = {
              q: quality.q ?? 0,
              timeS: estimates.time.genPerFrameS,
              headroomGib: estimates.vram.headroomGib,
            };
            /** @type {Candidate} */
            const candidate = {
              id: '',
              key: candidateKey(render),
              render,
              layout,
              estimates,
              quality,
              metrics,
              gates,
            };
            if (canonical === null && isCanonical(candidate, {
              steps: levels.steps, auto, kv, vae, attention,
            })) {
              canonical = {
                render, layout, estimates, quality, metrics, gates,
              };
            }
            if (!candidatePasses(gates)) continue;
            if (seen.has(candidate.key)) continue;
            seen.add(candidate.key);
            candidates.push(candidate);
          }
        }
      }
    }
  }

  if (canonical === null) {
    // Авто-значение какой-то оси не попало в набор (например, `-1` тайла VAE): собираем
    // каноническую раскладку отдельно — ладдеру и трейсу она нужна всегда.
    const estimates = applyGreyZone(estimateAll(base, { kvCache: kvModeOf(base) }), limitGib, greyZone);
    const canonicalQuality = qualityScore({
      pixels: geometry.pixels, steps: base.steps, attention: attention.mode,
    });
    canonical = {
      render: base,
      layout: { kvCache: kv, vae, attention, ref },
      estimates,
      quality: canonicalQuality,
      metrics: {
        q: canonicalQuality.q ?? 0,
        timeS: estimates.time.genPerFrameS,
        headroomGib: estimates.vram.headroomGib,
      },
      gates: gateCandidate({
        estimates, vae, attention, levels, request, limitGib,
      }),
    };
  }

  return {
    candidates: candidates.map((candidate, index) => ({
      ...candidate,
      id: `c${String(index + 1).padStart(2, '0')}`,
    })),
    canonical,
    total,
    limitGib,
    greyZone,
  };
}

/**
 * Совпадает ли сочетание с канонической раскладкой (все оси — авто-значения уровня).
 *
 * @param {Candidate} candidate
 * @param {{steps: number, auto: 'none'|'group', kv: KvDecision, vae: VaeTileDecision,
 *   attention: AttentionDecision}} axes
 * @returns {boolean}
 */
function isCanonical(candidate, axes) {
  const { render, layout } = candidate;
  return render.steps === axes.steps
    && render.offload === axes.auto
    && render.kvCache === (axes.kv.mode ?? false)
    && render.vaeTilePx === axes.vae.tilePx
    && render.windowAttnPx === axes.attention.windowPx
    && layout.kvCache.mode === axes.kv.mode;
}

/**
 * Ключ дедупа: полный конфиг сочетания + канвас. Два сочетания с одним ключом дают один и тот же
 * кадр и одинаковую раскладку памяти — второе не нужно.
 *
 * @param {Render} render
 * @returns {string}
 */
export function candidateKey(render) {
  return [
    render.model,
    `${render.canvas.width}x${render.canvas.height}`,
    render.steps,
    render.offload,
    String(render.kvCache),
    String(render.vaeTilePx),
    `${render.windowAttnPx}/${render.marginPx}`,
    render.encoderDevice,
    render.refPx,
    render.refs,
  ].join('|');
}

/**
 * Ось offload: сначала авто-режим, затем альтернатива (если модель её поддерживает).
 *
 * @param {string} model
 * @param {'none'|'group'} preferred
 * @returns {Array<'none'|'group'>}
 */
function offloadAxis(model, preferred) {
  const values = /** @type {Array<'none'|'group'>} */ (['none', 'group'])
    .filter((mode) => supportsOffload(model, mode));
  if (values.length === 0) return [preferred];
  return values.includes(preferred)
    ? [preferred, ...values.filter((value) => value !== preferred)]
    : values;
}

/**
 * Ось KV: разрешённый режим первым, затем `true`/`offload` как альтернатива. Явный
 * `memory.kvCache` — жёсткий override (схема называет его именно так), поэтому ось сужается до
 * запрошенного значения: политика `resolveKvCache` может его понизить (`kv-vram-vs-ram`), но
 * подменять запрос «более удобным» вариантом калькулятор не должен. `false` (префикс считается
 * каждый шаг) — только по явному запросу: это диагностика, не вариант.
 *
 * @param {KvDecision} kv
 * @param {boolean|'offload'|'auto-offload-ref2'|null} requested `request.memory.kvCache`
 * @returns {Array<'true'|'offload'|'false'|null>}
 */
function kvAxis(kv, requested) {
  if (!kv.applicable) return [null];
  if (requested !== null && requested !== undefined && requested !== 'auto-offload-ref2') {
    if (requested === true) return ['true'];
    if (requested === false) return ['false'];
    return [/** @type {'offload'} */ (requested)];
  }
  if (kv.mode === 'false') return ['false'];
  const preferred = /** @type {'true'|'offload'} */ (kv.mode ?? 'offload');
  const other = preferred === 'true' ? 'offload' : 'true';
  return [preferred, other];
}

/**
 * Ось тайла VAE: разрешённое значение первым, затем `512`, `0`, `1024` (`-1` не предлагается —
 * измеренный OOM на ≥1.5 Мп). Явный `memory.vaeTilePx` сужает ось до запрошенного значения
 * (жёсткий override): если он не влезает, его починит ладдер, а не подмена «удобным» тайлом.
 * У модели без тайлинга (H3) ось вырождена в `null`.
 *
 * @param {VaeTileDecision} vae
 * @param {number|null} requested `request.memory.vaeTilePx`
 * @returns {Array<number|null>}
 */
function tileAxis(vae, requested) {
  if (!vae.applicable) return [null];
  if (requested !== null && requested !== undefined) return [requested];
  /** @type {Array<number|null>} */
  const unique = [];
  for (const value of /** @type {Array<number|null>} */ ([vae.tilePx, 512, 0, 1024])) {
    if (!unique.includes(value)) unique.push(value);
  }
  return unique;
}

/**
 * Ось окна внимания: разрешённый режим первым плюс плотное внимание, если разрешённый режим —
 * окно и оно не выродилось в одну полосу (`equivalentToDense` не дублируется). Явный
 * `memory.window` (включить/выключить) сужает ось до своего решения.
 *
 * @param {Render} base
 * @param {AttentionDecision} attention
 * @param {null|{enabled: boolean, windowPx: number|null, marginPx: number|null}} requested
 * @returns {Array<{decision: AttentionDecision}>}
 */
function windowAxis(base, attention, requested) {
  if (requested !== null && requested !== undefined) return [{ decision: attention }];
  if (attention.mode !== 'window' || attention.equivalentToDense) {
    return [{ decision: attention }];
  }
  const dense = resolveAttention(base, { enabled: false, windowPx: null, marginPx: null });
  return [{ decision: attention }, { decision: dense }];
}

/**
 * Режим KV сочетания в форме, которую ждёт `cost.estimateAll`.
 *
 * @param {Render} render
 * @returns {'true'|'offload'|'false'}
 */
function kvModeOf(render) {
  if (render.kvCache === true || render.kvCache === 'true') return 'true';
  if (render.kvCache === 'offload') return 'offload';
  return 'false';
}

/**
 * Гейт «серой» зоны: `cost.gateBudget` работает с порогом `0.97`, а в деградированной зоне
 * порог смягчён до `1.00` (`CANDIDATE_ASSUMPTIONS['gate.greyZoneLimit']`). Поэтому пик между
 * `0.97·budget` и `1.00·budget` в этой зоне — не спилл, а `degraded` (план получает
 * `vram-spill-risk`).
 *
 * @param {Estimates} estimates
 * @param {number} limitGib
 * @param {boolean} greyZone
 * @returns {Estimates}
 */
function applyGreyZone(estimates, limitGib, greyZone) {
  const vram = estimates.vram;
  if (!greyZone || !vram.spills || vram.peakGib > limitGib) return estimates;
  return {
    ...estimates,
    vram: { ...vram, fits: true, spills: false, degraded: true, zone: 'degraded' },
  };
}

/**
 * Решение по KV для конкретного режима сочетания: числа (префикс, ГиБ, слои) — из политики,
 * `mode`/`placement`/`costPct` — то, что реально стоит в конфиге этого сочетания.
 *
 * @param {Render} render
 * @param {'true'|'offload'|'false'|null} mode
 * @param {boolean|'offload'|'auto-offload-ref2'|null} requested
 * @returns {KvDecision}
 */
function kvDecisionFor(render, mode, requested) {
  if (mode === null) return resolveKvCache(render, null);
  const policy = resolveKvCache(render, requested);
  if (policy.mode === mode) return policy;
  if (mode === 'offload') {
    const ram = resolveKvCache(render, 'offload');
    return { ...ram, reason: `candidate-kv-ram: K/V в RAM как альтернатива — ${ram.reason}` };
  }
  if (mode === 'false') {
    const off = resolveKvCache(render, false);
    return { ...off, mode: 'false', placement: 'none' };
  }
  const vram = resolveKvCache(render, true);
  return {
    ...vram,
    mode: 'true',
    placement: 'vram',
    costPct: 0,
    reason: 'candidate-kv-vram: K/V кладём в VRAM — альтернатива политике '
      + `(пик ${vram.peakWithKvGib.toFixed(2)} ГиБ против порога ${vram.limitGib.toFixed(2)} = `
      + `0.90·${Math.round(vram.limitGib / GATE.degraded)} ГиБ; правило A7 выбрало бы RAM)`,
  };
}

/**
 * Решение по тайлу VAE для выбранного значения оси: если оно совпадает с авто-политикой —
 * возвращается её причина, иначе значение помечено как альтернатива.
 *
 * @param {Render} render
 * @param {number|null} tile
 * @param {number|null} requested
 * @returns {VaeTileDecision}
 */
function vaeDecisionFor(render, tile, requested) {
  const auto = resolveVaeTile(render, requested);
  if (tile === null || auto.tilePx === tile) return auto;
  const forced = resolveVaeTile(render, tile);
  return { ...forced, reason: `candidate-vae-tile: тайл ${tile} как альтернатива — ${forced.reason}` };
}

/**
 * Гейты сочетания по плановым правилам.
 *
 * @param {{estimates: Estimates, vae: VaeTileDecision, attention: AttentionDecision,
 *   levels: Levels, request: NormalizedRequest, limitGib: number}} input
 * @returns {CandidateGates}
 */
function gateCandidate({ estimates, vae, attention, levels, request, limitGib }) {
  const windowOk = levels.allowWindow || request.memory.window !== null || attention.mode === 'dense';
  return {
    fits: estimates.vram.peakGib <= limitGib,
    vaeFit: vae.oom !== true,
    ramFit: !estimates.ram.spills,
    headroomOk: estimates.vram.headroomGib >= levels.minHeadroom * request.hardware.vramGib,
    windowOk,
    zone: estimates.vram.zone,
    limitGib,
  };
}

/**
 * @param {CandidateGates} gates
 * @returns {boolean}
 */
function candidatePasses(gates) {
  // Пик RAM не отбрасывает сочетание (план гейтит только VRAM; RAM даёт `ram-pagecache-thrash`):
  // модель RAM — самая слабая в фите, жёсткий гейт по ней давал бы ложные `impossible`.
  return gates.fits && gates.vaeFit && gates.headroomOk && gates.windowOk;
}
