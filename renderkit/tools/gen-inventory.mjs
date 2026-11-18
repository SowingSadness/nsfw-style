#!/usr/bin/env node
// @ts-check
/**
 * `tools/gen-inventory.mjs` — инвентарь возможностей из Python (шаг 2 плана `render-settings-kit`).
 *
 * Источник истины — Python-пакет `qwenstyle`, а не копия чисел в JS:
 *
 * 1. `qwenstyle.profiles` — список **публичных** пресетов (поле `Profile.presets`), тарифы
 *    (`disk_estimate`/`download_estimate`/`target_height`) и размеры репозиториев;
 * 2. реестр бэкендов (`qwenstyle.registry`) — `caps`/`budget` моделей генерации,
 *    декларации энкодеров (`repo`/`revision`/`local_dir`/`cache_tag`/`requires_vision`);
 * 3. `qwenstyle.config.backend_parameters()` — дескрипторы `ParameterSpec` (whitelist `--set`);
 * 4. `python -m qwenstyle config show <пресет> --json` (= `RunConfig.to_mapping()`) — достоверные
 *    значения полей (`repo`/`transformer`/`lora`/`precision`/`precision_cache`/`offload`/
 *    `kv_cache`/`sampler_kwargs`/`encoder.*`) для каждой модели и разрешённой силы LoRA.
 *
 * На выходе — `src/data/inventory.mjs` **без имён пресетов**: только `models`, `encoders`, `limits`,
 * `settable`, `params`, `tiers`, `repos` (+ `source` с дайджестом входов). Дайджест нужен, чтобы
 * правка `qwenstyle/presets/*.yaml` (даже не влияющая на поля модели) меняла файл: `--check` в CI
 * тогда падает на рассинхроне, а не молчит.
 *
 * Запуск: `npm run inventory` (запись, только если изменилось) · `npm run inventory -- --check`
 * (пересчитать в память и сравнить с файлом; exit 1 при расхождении).
 *
 * Коды выхода: `0` — успех/совпадение, `1` — расхождение (`--check`), `2` — нет Python/venv
 * (тесты трактуют это как `SKIP`, конвенция репозитория).
 */

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
/** Корень пакета `renderkit/`. */
const PACKAGE_ROOT = resolve(HERE, '..');
/** Корень репозитория (там живут `qwenstyle/` и `.venv/`). */
const REPO_ROOT = resolve(PACKAGE_ROOT, '..');
const OUT_PATH = join(PACKAGE_ROOT, 'src', 'data', 'inventory.mjs');

/** Нет Python/venv: не ошибка, а «нечего сверять» (тест печатает SKIP). */
const SKIP_EXIT = 2;

/**
 * Наши доменные id моделей (план A3) и то, чем каждая модель опознаётся в Python.
 * Имена пресетов здесь не участвуют: `backend` — имя бэкенда из реестра,
 * `transformerRepo` — первый элемент `generation.transformer` (`null` = трансформер из `repo`).
 *
 * @type {ReadonlyArray<{id: string, backend: string, transformerRepo: string | null}>}
 */
const MODELS = Object.freeze([
  Object.freeze({ id: 'qwen21-base', backend: 'qwen_image21', transformerRepo: 'Qwen/Qwen-Image-2.1' }),
  Object.freeze({ id: 'qwen21-turbo', backend: 'qwen_image21', transformerRepo: 'Qwen/Qwen-Image-2.1-Turbo' }),
  Object.freeze({ id: 'h3-image', backend: 'minimax_h3_image', transformerRepo: null }),
]);

/** Поля `encoder.*`, которые берутся из резолвленного конфига (остальные объявлены бэкендом). */
const RUNTIME_ENCODER_FIELDS = Object.freeze([
  'dtype', 'device', 'stream', 'prefetch', 'pin_memory', 'lookahead',
]);

/**
 * Python-снимок: всё, что можно взять без `config show` (профили, реестр, `ParameterSpec`,
 * размеры репозиториев, список доступных пресетов). Печатает один JSON в stdout.
 */
const SNAPSHOT_PY = `
import json
from qwenstyle import profiles
from qwenstyle.config import ASPECT_RATIOS, available_presets, backend_parameters
from qwenstyle.registry import discover

registry = discover()


def spec(s):
    return {
        "name": s.name,
        "scope": s.scope,
        "path": s.path,
        "type": s.type,
        "default": s.default,
        "choices": list(s.choices) if s.choices else None,
        "minimum": s.minimum,
        "maximum": s.maximum,
        "readonly": bool(s.readonly),
        "doc": s.doc or "",
    }


def parameter_list(kind, name):
    return [spec(item) for item in backend_parameters(kind, name)]


def caps_of(cls):
    caps = getattr(cls, "caps", None)
    return caps.to_dict() if caps is not None else None


def budget_of(cls):
    budget = getattr(cls, "budget", None)
    return budget.to_dict() if budget is not None else None


generation = registry.table("generation")
encoders = {}
for enc_name, enc_cls in registry.table("encoder").items():
    encoders[enc_name] = {
        "name": enc_name,
        "cache_tag": getattr(enc_cls, "cache_tag", None),
        "repo": getattr(enc_cls, "repo", None),
        "revision": getattr(enc_cls, "revision", None),
        "local_dir": getattr(enc_cls, "local_dir", None),
        "requires_vision": bool(getattr(enc_cls, "requires_vision", False)),
        "memory_gib": getattr(enc_cls, "memory_gib", None),
    }

payload = {
    "profiles": [
        {
            "name": profile.name,
            "hardware": profile.hardware,
            "disk_gib": profiles.disk_estimate(profile.name),
            "download_gib": profiles.download_estimate(profile.name),
            "target_height": profile.target_height,
            "repos": [entry.repo for entry in profile.repos],
            "presets": list(profile.presets),
        }
        for profile in (profiles.profile(item) for item in profiles.names())
    ],
    "repo_sizes": dict(profiles.REPO_SIZE_GIB),
    "backends": {
        backend: {
            "caps": caps_of(generation.get(backend)),
            "budget": budget_of(generation.get(backend)),
            "parameters": parameter_list("generation", backend),
        }
        for backend in ("qwen_image21", "minimax_h3_image")
    },
    "upscales": {"esrgan": {"parameters": parameter_list("upscale", "esrgan")}},
    "encoders": encoders,
    "aspect_ratios": list(ASPECT_RATIOS),
    "available_presets": list(available_presets()),
}
print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
`;

/** @typedef {{name: string, scope: string, path: string, type: string, default: unknown,
 *            choices: string[] | null, minimum: number | null, maximum: number | null,
 *            readonly: boolean, doc: string}} ParamSpecJSON */

/**
 * Кандидаты на интерпретатор Python: сначала venv репозитория, затем системный `python`.
 *
 * @returns {string[]}
 */
function pythonCandidates() {
  /** @type {string[]} */
  const candidates = [];
  const venv = process.platform === 'win32'
    ? join(REPO_ROOT, '.venv', 'Scripts', 'python.exe')
    : join(REPO_ROOT, '.venv', 'bin', 'python');
  if (existsSync(venv)) candidates.push(venv);
  candidates.push(...(process.platform === 'win32' ? ['python'] : ['python3', 'python']));
  return candidates;
}

/**
 * Запустить Python и вернуть stdout. Ошибки (нет интерпретатора, ненулевой код) — исключение.
 *
 * @param {string} command
 * @param {string[]} args
 * @returns {string}
 */
function runPython(command, args) {
  return execFileSync(command, args, {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe'],
    maxBuffer: 128 * 1024 * 1024,
    env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
  });
}

/**
 * Первый интерпретатор, который импортирует `qwenstyle`, или `null`.
 *
 * @returns {{command: string, run: (args: string[]) => string} | null}
 */
function findPython() {
  for (const command of pythonCandidates()) {
    try {
      runPython(command, ['-c', 'import qwenstyle']);
      return { command, run: (args) => runPython(command, args) };
    } catch {
      // следующий кандидат (или SKIP, если ни один не подошёл)
    }
  }
  return null;
}

/**
 * Округление до 1 знака (ГиБ) — как в `qwenstyle.profiles` и `ResourceBudget`.
 *
 * @param {number} value
 * @returns {number}
 */
function round1(value) {
  return Math.round(value * 10) / 10;
}

/**
 * Округление до 2 знаков (ГиБ весов).
 *
 * @param {number} value
 * @returns {number}
 */
function round2(value) {
  return Math.round(value * 100) / 100;
}

/**
 * Короткий sha256 (`<12>` символов) от стабильного JSON.
 *
 * @param {unknown} value
 * @returns {string}
 */
function sha12(value) {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex').slice(0, 12);
}

/**
 * Значение, на котором сошлись все источники; расхождение — ошибка генератора (а не тихий выбор).
 *
 * @template T
 * @param {T[]} values
 * @param {string} what
 * @returns {T}
 */
function agree(values, what) {
  if (values.length === 0) throw new Error(`gen-inventory: ${what} — нет ни одного источника`);
  const first = values[0];
  const firstKey = JSON.stringify(first);
  for (const value of values) {
    if (JSON.stringify(value) !== firstKey) {
      throw new Error(`gen-inventory: ${what} — источники расходятся: ${firstKey} vs ${JSON.stringify(value)}`);
    }
  }
  return first;
}

/**
 * Самое частое значение (при равенстве — лексикографически меньшее): так рантайм-поля энкодера
 * берутся из большинства пресетов, а не из случайного первого.
 *
 * @template T
 * @param {T[]} values
 * @returns {T | null}
 */
function mode(values) {
  /** @type {Map<string, {value: T, count: number}>} */
  const counts = new Map();
  for (const value of values) {
    const key = JSON.stringify(value);
    const entry = counts.get(key);
    if (entry) entry.count += 1;
    else counts.set(key, { value, count: 1 });
  }
  const sorted = [...counts.entries()].sort((a, b) => b[1].count - a[1].count || (a[0] < b[0] ? -1 : 1));
  return sorted.length ? sorted[0][1].value : null;
}

/**
 * Дескриптор `ParameterSpec` в том виде, в котором он нужен калькулятору.
 *
 * @param {ParamSpecJSON} spec
 * @returns {ParamSpecJSON}
 */
function toParamSpec(spec) {
  return {
    name: spec.name,
    scope: spec.scope,
    path: spec.path,
    type: spec.type,
    default: spec.default,
    choices: spec.choices ?? null,
    minimum: spec.minimum ?? null,
    maximum: spec.maximum ?? null,
    readonly: Boolean(spec.readonly),
    doc: spec.doc ?? '',
  };
}

/**
 * Печать расхождения для `--check` (первая различающаяся строка + контекст).
 *
 * @param {string} expected что должно быть в файле
 * @param {string} actual что лежит сейчас
 * @returns {string}
 */
function describeDiff(expected, actual) {
  const want = expected.split('\n');
  const have = actual.split('\n');
  const lines = [];
  let index = 0;
  while (index < Math.min(want.length, have.length) && want[index] === have[index]) index += 1;
  lines.push(`  первая разница — строка ${index + 1}:`);
  for (let i = Math.max(index - 2, 0); i < Math.min(index + 4, Math.max(want.length, have.length)); i += 1) {
    if (i === index) {
      lines.push(`  ! - ${have[i] ?? '<нет строки>'}`);
      lines.push(`  ! + ${want[i] ?? '<нет строки>'}`);
    } else {
      lines.push(`    ${have[i] ?? '<нет строки>'}`);
    }
  }
  if (want.length !== have.length) lines.push(`  строк: файл ${have.length}, ожидается ${want.length}`);
  return lines.join('\n');
}

/**
 * Собрать `INVENTORY` из снимка Python и резолвленных конфигов пресетов.
 *
 * @param {any} snapshot результат `SNAPSHOT_PY`
 * @param {Array<string>} configNames порядок: публичные пресеты профилей, затем остальные
 * @param {(name: string) => any} loadConfig резолвленный конфиг пресета (`config show --json`)
 * @param {(encoderName: string) => any} loadEncoderConfig конфиг для энкодера, который не использует
 *        ни один пресет (нужен только `base`)
 * @returns {Record<string, any>}
 */
function buildInventory(snapshot, configNames, loadConfig, loadEncoderConfig) {
  /** @type {Map<string, any>} */
  const configs = new Map();
  for (const name of configNames) configs.set(name, loadConfig(name));

  // -- модели: группируем конфиги по (backend, transformerRepo) ------------------------------- //
  /** @type {Map<string, any[]>} */
  const byModel = new Map(MODELS.map((model) => [model.id, []]));
  for (const [preset, config] of configs) {
    const backend = config.generation?.backend;
    const transformerRepo = config.generation?.transformer ? config.generation.transformer[0] : null;
    for (const model of MODELS) {
      if (backend !== model.backend || transformerRepo !== model.transformerRepo) continue;
      byModel.get(model.id).push({ preset, config });
    }
  }

  /** @type {Record<string, any>} */
  const models = {};
  for (const model of MODELS) {
    const entries = byModel.get(model.id) ?? [];
    const where = `модель ${model.id} (${model.backend}/${model.transformerRepo ?? 'repo'})`;
    if (entries.length === 0) {
      throw new Error(`gen-inventory: ${where} — ни один резолвленный конфиг не подошёл `
        + `(проверьте пресеты и список моделей в tools/gen-inventory.mjs)`);
    }
    const cfgs = entries.map((entry) => entry.config);
    const backendInfo = snapshot.backends[model.backend];
    if (!backendInfo) throw new Error(`gen-inventory: бэкенд ${model.backend} не найден в реестре`);
    const caps = backendInfo.caps ?? {};
    const budget = backendInfo.budget ?? {};
    /** @type {ParamSpecJSON[]} */
    const backendParams = backendInfo.parameters ?? [];

    const repo = agree(cfgs.map((cfg) => cfg.generation.repo), `${where}: generation.repo`);
    const transformer = agree(
      cfgs.map((cfg) => (cfg.generation.transformer ? [...cfg.generation.transformer] : null)),
      `${where}: generation.transformer`,
    );
    const precision = agree(cfgs.map((cfg) => cfg.generation.precision), `${where}: generation.precision`);
    const precisionCache = agree(
      cfgs.map((cfg) => cfg.generation.precision_cache ?? null),
      `${where}: generation.precision_cache`,
    );
    const encoderId = agree(cfgs.map((cfg) => cfg.encoder?.name ?? null), `${where}: encoder.name`);

    // Шаги: фиксированные моделью (caps/readonly-спек) или полное расписание (максимум по пресетам).
    const stepsSpec = backendParams.find((spec) => spec.path === 'generation.steps');
    const declaredFixed = caps.fixed_steps ?? (stepsSpec?.readonly ? Number(stepsSpec.default) : null);
    const fixedSteps = declaredFixed ?? Math.max(...cfgs.map((cfg) => Number(cfg.generation.steps)));

    // Веса: bf16 из бюджета бэкенда, но если трансформер берётся из другого репозитория
    // (turbo), бюджет бэкенда про него не знает — берём размер репозитория из profiles.
    const transformerRepo = Array.isArray(transformer) ? transformer[0] : null;
    let weightGibBf16 = Number(budget.weights_bf16_gib ?? 0);
    if (transformerRepo && transformerRepo !== repo) {
      const size = snapshot.repo_sizes[transformerRepo];
      if (size == null) {
        throw new Error(`gen-inventory: размер ${transformerRepo} не объявлен в profiles.REPO_SIZE_GIB`);
      }
      weightGibBf16 = Number(size);
    }
    const fp8Budget = budget.weights_fp8_gib == null ? null : Number(budget.weights_fp8_gib);
    const fp8Ratio = fp8Budget != null && Number(budget.weights_bf16_gib) > 0
      ? fp8Budget / Number(budget.weights_bf16_gib)
      : 1;

    models[model.id] = {
      id: model.id,
      backend: model.backend,
      repo,
      transformer,
      precision,
      precision_cache: precisionCache,
      fixedSteps,
      weightGibBf16: round2(weightGibBf16),
      weightGibFp8: round2(weightGibBf16 * fp8Ratio),
      encoderId,
      lora: caps.supports_lora === false ? null : buildLora(cfgs, where),
    };
  }

  // -- энкодеры: декларации бэкенда + рантайм-поля из резолвленных конфигов ------------------ //
  /** @type {Record<string, any>} */
  const encoders = {};
  const encoderNames = Object.keys(snapshot.encoders).sort();
  for (const name of encoderNames) {
    const declared = snapshot.encoders[name];
    const used = [...configs.values()].filter((cfg) => cfg.encoder?.name === name);
    const sources = used.length > 0 ? used : [loadEncoderConfig(name)];
    /** @type {Record<string, any>} */
    const runtime = {};
    for (const field of RUNTIME_ENCODER_FIELDS) {
      runtime[field] = mode(sources.map((cfg) => cfg.encoder?.[field] ?? null));
    }
    if (runtime.dtype == null || runtime.device == null) {
      throw new Error(`gen-inventory: у энкодера ${name} не нашлось dtype/device в резолвленных конфигах`);
    }
    encoders[name] = {
      name,
      cache_tag: declared.cache_tag,
      repo: declared.repo,
      revision: declared.revision ?? null,
      local_dir: declared.local_dir ?? null,
      requires_vision: Boolean(declared.requires_vision),
      dtype: runtime.dtype,
      device: runtime.device,
      stream: Boolean(runtime.stream),
      prefetch: Number(runtime.prefetch ?? 0),
      pin_memory: Boolean(runtime.pin_memory),
      lookahead: Number(runtime.lookahead ?? 1),
      memory_gib: declared.memory_gib == null ? null : Number(declared.memory_gib),
    };
  }

  // -- лимиты: только из объявленного (caps/спеки/ASPECT_RATIOS) ----------------------------- //
  /** @type {Record<string, number>} */
  const fixedSteps = {};
  for (const model of MODELS) {
    const backendParams = snapshot.backends[model.backend]?.parameters ?? [];
    const stepsSpec = backendParams.find((spec) => spec.path === 'generation.steps');
    if (stepsSpec?.readonly) fixedSteps[model.id] = models[model.id].fixedSteps;
  }
  const capsList = MODELS.map((model) => snapshot.backends[model.backend]?.caps ?? {});
  /** @param {'precision' | 'offload'} field */
  const unionCaps = (field) => {
    /** @type {string[]} */
    const out = [];
    for (const caps of capsList) {
      for (const value of caps[field] ?? []) if (!out.includes(value)) out.push(value);
    }
    return out;
  };
  const maxImagesList = capsList.map((caps) => caps.max_images).filter((value) => value != null);
  const limits = {
    maxImages: maxImagesList.length ? Math.max(...maxImagesList) : null,
    fixedSteps,
    precisions: unionCaps('precision'),
    offloadModes: unionCaps('offload'),
    aspectRatios: [...snapshot.aspect_ratios],
  };

  // -- params/settable по бэкендам (whitelist `--set`) --------------------------------------- //
  /** @type {Record<string, ParamSpecJSON[]>} */
  const paramMap = {};
  for (const backend of Object.keys(snapshot.backends)) {
    paramMap[backend] = (snapshot.backends[backend].parameters ?? []).map(toParamSpec);
  }
  for (const backend of Object.keys(snapshot.upscales)) {
    paramMap[backend] = (snapshot.upscales[backend].parameters ?? []).map(toParamSpec);
  }
  /** @type {Record<string, ParamSpecJSON[]>} */
  const params = {};
  /** @type {Record<string, string[]>} */
  const settable = {};
  for (const backend of Object.keys(paramMap).sort()) {
    params[backend] = paramMap[backend];
    settable[backend] = paramMap[backend].filter((spec) => !spec.readonly).map((spec) => spec.path);
  }

  // -- тарифы и размеры репозиториев --------------------------------------------------------- //
  /** @type {Record<string, any>} */
  const tiers = {};
  for (const profile of snapshot.profiles) {
    tiers[profile.name] = {
      hardware: profile.hardware,
      diskGib: round1(Number(profile.disk_gib)),
      downloadGib: round1(Number(profile.download_gib)),
      targetHeight: Number(profile.target_height),
    };
  }
  /** @type {Record<string, number>} */
  const repos = {};
  for (const repo of Object.keys(snapshot.repo_sizes).sort()) {
    repos[repo] = Number(snapshot.repo_sizes[repo]);
  }

  // -- дайджест входов: правка пресета/бэкенда обязана менять файл ---------------------------- //
  const configDigestInput = [...configs.entries()].sort((a, b) => (a[0] < b[0] ? -1 : 1)).map(([name, cfg]) => [name, cfg]);
  const source = {
    python: 'qwenstyle.profiles + qwenstyle.registry + config.backend_parameters + config show --json',
    presets: configNames.length,
    digest: {
      configs: sha12(configDigestInput),
      profiles: sha12({
        profiles: snapshot.profiles,
        repo_sizes: snapshot.repo_sizes,
        backends: snapshot.backends,
        upscales: snapshot.upscales,
        encoders: snapshot.encoders,
        aspect_ratios: snapshot.aspect_ratios,
      }),
    },
  };

  return { schemaVersion: 1, source, models, encoders, limits, settable, params, tiers, repos };
}

/**
 * LoRA-адаптер модели: репозиторий/файл из пресетов, сила — объявленная максимумом по пресетам
 * (`null` = адаптер есть, но ни один пресет не включает его, силу задаёт запрос).
 *
 * @param {any[]} cfgs
 * @param {string} where
 * @returns {{repo: string, filename: string, scale: number | null, strip_alpha: boolean} | null}
 */
function buildLora(cfgs, where) {
  const loras = cfgs.map((cfg) => cfg.generation.lora).filter(Boolean);
  if (loras.length === 0) return null;
  const repo = agree(loras.map((lora) => lora.repo), `${where}: lora.repo`);
  const filename = agree(loras.map((lora) => lora.filename), `${where}: lora.filename`);
  const stripAlpha = agree(loras.map((lora) => Boolean(lora.strip_alpha)), `${where}: lora.strip_alpha`);
  const maxScale = Math.max(...loras.map((lora) => Number(lora.scale) || 0));
  return { repo, filename, scale: maxScale > 0 ? maxScale : null, strip_alpha: stripAlpha };
}

/**
 * Полный текст `src/data/inventory.mjs` (детерминированный: без путей, времени и версий машины).
 *
 * @param {Record<string, any>} inventory
 * @returns {string}
 */
function renderModule(inventory) {
  const header = [
    '// generated by tools/gen-inventory.mjs — do not edit; regenerate with `npm run inventory`',
    `// source: ${inventory.source.python}`,
    `// presets: ${inventory.source.presets} resolved configs · digest configs ${inventory.source.digest.configs},`
      + ` profiles ${inventory.source.digest.profiles}`,
  ].join('\n');
  const body = JSON.stringify(inventory, null, 2);
  return `${header}\nexport const INVENTORY = Object.freeze(${body});\n`;
}

/**
 * Точка входа.
 *
 * @returns {number} код выхода
 */
function main() {
  const check = process.argv.includes('--check');
  const python = findPython();
  if (!python) {
    console.log('[inventory] SKIP: не найден Python с установленным `qwenstyle` '
      + `(искал ${pythonCandidates().join(', ')}; venv ожидается в ${relative(process.cwd(), REPO_ROOT)}/.venv)`);
    return SKIP_EXIT;
  }

  const snapshot = JSON.parse(python.run(['-c', SNAPSHOT_PY]));
  const publicPresets = snapshot.profiles.flatMap((profile) => profile.presets);
  const otherPresets = snapshot.available_presets.filter((name) => !publicPresets.includes(name));
  const configNames = [...publicPresets, ...otherPresets];

  /** @type {Map<string, any>} */
  const loaded = new Map();
  const loadConfig = (name) => {
    if (!loaded.has(name)) {
      loaded.set(name, JSON.parse(python.run(['-m', 'qwenstyle', 'config', 'show', name, '--json'])));
    }
    return loaded.get(name);
  };
  const probe = configNames.includes('base') ? 'base' : configNames[0];
  const loadEncoderConfig = (encoderName) => JSON.parse(
    python.run(['-m', 'qwenstyle', 'config', 'show', probe, '--json', '--set', `encoder.name=${encoderName}`]),
  );

  const inventory = buildInventory(snapshot, configNames, loadConfig, loadEncoderConfig);
  const text = renderModule(inventory);

  if (check) {
    if (!existsSync(OUT_PATH)) {
      console.error(`[inventory] --check: ${relative(process.cwd(), OUT_PATH)} отсутствует — запустите \`npm run inventory\``);
      return 1;
    }
    const actual = readFileSync(OUT_PATH, 'utf8');
    if (actual === text) {
      console.log(`[inventory] --check: совпадает (${configNames.length} конфигов, digest ${inventory.source.digest.configs})`);
      return 0;
    }
    console.error(`[inventory] --check: расхождение с ${relative(process.cwd(), OUT_PATH)} — данные Python изменились`);
    console.error(describeDiff(text, actual));
    return 1;
  }

  const current = existsSync(OUT_PATH) ? readFileSync(OUT_PATH, 'utf8') : null;
  if (current === text) {
    console.log(`[inventory] без изменений: ${relative(process.cwd(), OUT_PATH)}`);
    return 0;
  }
  mkdirSync(dirname(OUT_PATH), { recursive: true });
  writeFileSync(OUT_PATH, text, 'utf8');
  console.log(`[inventory] записано: ${relative(process.cwd(), OUT_PATH)} (${text.split('\n').length - 1} строк, `
    + `${configNames.length} конфигов, digest ${inventory.source.digest.configs})`);
  return 0;
}

process.exitCode = main();
