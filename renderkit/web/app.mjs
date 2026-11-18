// @ts-check
/**
 * Страница `renderkit`: форма → `computePlan` → `buildView` → панели результата.
 *
 * Принципы:
 *
 * - **никакого `innerHTML` из данных**: текст только через `textContent`/`createTextNode`, узлы —
 *   через `createElement`. Разметка панелей собирается кодом, поэтому данные ядра не могут
 *   «просочиться» в HTML;
 * - состояние живёт в `location.hash` (`#w=2560&h=1440&q=max…`), поэтому ссылку можно переслать;
 *   `hashchange`/`popstate` синхронизируют форму обратно;
 * - пересчёт дебаунсится 120 мс: набор цифр в поле не должен гонять ядро на каждое нажатие;
 * - ошибки — текстом **рядом с полем** (плюс общий `role="alert"`, если поле не определить).
 *
 * Модуль — точка входа страницы: ядро (`../src/**`) и вью-модель (`./viewmodel.mjs`) остаются
 * изоморфными и без DOM, здесь только браузерные API.
 */

import { computePlan, listOptions } from '../src/index.mjs';
import { HELP } from '../src/data/help.mjs';
import { attachHints } from './hint.mjs';
import {
  DEFAULT_STATE,
  FIELD_LIMITS,
  buildView,
  canvasText,
  hashFromState,
  hintLive,
  requestFromState,
  scanYaml,
  stateFromHash,
} from './viewmodel.mjs';

const DEBOUNCE_MS = 120;

const options = listOptions();

/** Идентификаторы контролов формы по полю состояния. */
const INPUT_IDS = Object.freeze({ w: 'w', h: 'h', vram: 'vram', ram: 'ram', refs: 'refs' });

/** Поле `PlanError.field` → id контрола или контейнера поля (для ошибки рядом с полем). */
const ERROR_FIELDS = Object.freeze({
  'image.width': 'w',
  'image.height': 'h',
  'image.quality': 'q-max',
  'image.bitDepth': 'bits-8',
  'image.steps': 'w',
  'hardware.model': 'model',
  'hardware.vramGib': 'vram',
  'hardware.ramGib': 'ram',
  'hardware.disk': 'disk-nvme',
  refs: 'refs',
  anime: 'anime',
  freeDiskGib: 'free-disk',
});

/** @type {import('./viewmodel.mjs').AppState} */
let state = stateFromHash(location.hash);
/** Разовые переопределения панели памяти (в хеш не входят). @type {{kvCache?: unknown, freeDiskGib?: number|null}} */
let overrides = {};
/** @type {ReturnType<typeof buildView>|null} */
let view = null;
/** @type {ReturnType<typeof scanYaml>|null} */
let lastScan = null;
/** @type {number|null} */
let timer = null;

bootstrap();

/** Инициализация: значения формы, подсказки, обработчики, первый расчёт. */
function bootstrap() {
  fillForm(state);
  bindForm();
  bindButtons();
  bindHash();
  bindScan();
  attachHints(document, HELP, liveFor);
  const meta = `правила ${options.rulesetVersion}`;
  setText('ruleset-line', meta);
  setText('foot-ruleset', meta);
  const pp = document.querySelector('#result');
  if (pp) pp.setAttribute('aria-busy', 'false');
  renderAll();
}

// ── форма ────────────────────────────────────────────────────────────────────────────────

/** Проставить значения формы из состояния (программная запись событий не вызывает). */
function fillForm(next) {
  setValue('w', next.w);
  setValue('h', next.h);
  setValue('vram', next.vram);
  setValue('ram', next.ram);
  setValue('refs', next.refs);
  setValue('vram-range', next.vram);
  setValue('ram-range', next.ram);
  setValue('model', next.model);
  setValue('free-disk', overrides.freeDiskGib ?? '');
  checkRadio('quality', next.q);
  checkRadio('bits', String(next.bits));
  checkRadio('disk', next.disk);
  const anime = /** @type {HTMLInputElement|null} */ (document.getElementById('anime'));
  if (anime) anime.checked = Boolean(next.anime);
}

/** Первый расчёт и подписки формы. */
function bindForm() {
  const form = document.getElementById('form');
  if (!form) return;
  form.addEventListener('submit', (event) => event.preventDefault());
  form.addEventListener('input', () => schedule());
  form.addEventListener('change', () => schedule());
  for (const pair of [['vram', 'vram-range'], ['ram', 'ram-range']]) {
    const [numberId, rangeId] = pair;
    const number = /** @type {HTMLInputElement|null} */ (document.getElementById(numberId));
    const range = /** @type {HTMLInputElement|null} */ (document.getElementById(rangeId));
    if (!number || !range) continue;
    number.addEventListener('input', () => {
      if (number.value !== '') range.value = number.value;
    });
    range.addEventListener('input', () => {
      number.value = range.value;
      schedule();
    });
  }
}

/** Кнопки быстрых размеров, профилей железа и переключатели панели памяти. */
function bindButtons() {
  for (const button of Array.from(document.querySelectorAll('[data-size]'))) {
    button.addEventListener('click', () => {
      const [w, h] = String(button.getAttribute('data-size')).split('x').map(Number);
      state = { ...state, w, h };
      fillForm(state);
      schedule(true);
    });
  }
  for (const button of Array.from(document.querySelectorAll('[data-hw]'))) {
    button.addEventListener('click', () => {
      const [vram, ram] = String(button.getAttribute('data-hw')).split('x').map(Number);
      state = { ...state, vram, ram };
      fillForm(state);
      schedule(true);
    });
  }
  const kv = document.getElementById('kv-toggle');
  if (kv) kv.addEventListener('click', toggleKv);
  const copy = document.getElementById('copy-yaml');
  if (copy) copy.addEventListener('click', copyYaml);
  const download = document.getElementById('download-yaml');
  if (download) download.addEventListener('click', downloadYaml);
}

/** Пересчёт с дебаунсом (`immediate` — для кнопок: ответ должен быть мгновенным). */
function schedule(immediate = false) {
  if (timer !== null) {
    clearTimeout(timer);
    timer = null;
  }
  if (immediate) {
    renderAll();
    return;
  }
  timer = window.setTimeout(() => {
    timer = null;
    renderAll();
  }, DEBOUNCE_MS);
}

/**
 * Прочитать состояние формы, проверив числа.
 *
 * @returns {{state: import('./viewmodel.mjs').AppState, problems: Array<{id: string, message: string}>}}
 */
function readState() {
  /** @type {Array<{id: string, message: string}>} */
  const problems = [];
  /** @param {string} key @param {keyof FIELD_LIMITS} limitKey @param {number} fallback */
  const num = (key, limitKey, fallback) => {
    const input = /** @type {HTMLInputElement|null} */ (document.getElementById(INPUT_IDS[key] ?? key));
    if (!input) return fallback;
    const raw = input.value.trim();
    if (raw === '') {
      problems.push({ id: input.id, message: 'укажите число' });
      return fallback;
    }
    const value = Number(raw);
    const limit = FIELD_LIMITS[limitKey];
    if (!Number.isFinite(value) || !Number.isInteger(value)) {
      problems.push({ id: input.id, message: 'нужно целое число' });
      return fallback;
    }
    if (value < limit.min || value > limit.max) {
      problems.push({ id: input.id, message: `вне диапазона ${limit.min}…${limit.max}` });
      return fallback;
    }
    return value;
  };
  const next = {
    w: num('w', 'w', state.w),
    h: num('h', 'h', state.h),
    vram: num('vram', 'vram', state.vram),
    ram: num('ram', 'ram', state.ram),
    refs: num('refs', 'refs', state.refs),
    q: radioValue('quality', state.q),
    bits: Number(radioValue('bits', String(state.bits))) === 16 ? 16 : 8,
    model: selectValue('model', state.model),
    disk: radioValue('disk', state.disk),
    anime: Boolean(/** @type {HTMLInputElement|null} */ (document.getElementById('anime'))?.checked),
  };
  const free = /** @type {HTMLInputElement|null} */ (document.getElementById('free-disk'));
  if (free) {
    const raw = free.value.trim();
    if (raw === '') overrides.freeDiskGib = null;
    else {
      const value = Number(raw);
      if (Number.isFinite(value) && value >= 0) overrides.freeDiskGib = value;
      else problems.push({ id: free.id, message: 'нужно неотрицательное число' });
    }
  }
  return { state: /** @type {import('./viewmodel.mjs').AppState} */ (next), problems };
}

// ── расчёт и отрисовка ───────────────────────────────────────────────────────────────────

/** Пересчитать план и перерисовать панели. */
function renderAll() {
  clearErrors();
  const read = readState();
  if (read.problems.length > 0) {
    for (const problem of read.problems) showErrorAt(problem.id, problem.message);
    return;
  }
  state = read.state;
  /** @type {any} */
  let plan;
  try {
    plan = computePlan(requestFromState(state, overrides));
  } catch (error) {
    showPlanError(error);
    return;
  }
  view = buildView(plan, { scan: lastScan });
  paint(view);
  syncHash();
}

/**
 * Запрос для `hintLive`: текущее состояние плюс те же переопределения.
 *
 * @returns {Record<string, unknown>}
 */
function currentRequest() {
  return requestFromState(state, overrides);
}

/**
 * @param {string} id
 * @returns {string|null}
 */
function liveFor(id) {
  try {
    return hintLive(id, currentRequest());
  } catch {
    return null;
  }
}

/**
 * Перерисовать все панели.
 *
 * @param {ReturnType<typeof buildView>} model
 * @returns {void}
 */
function paint(model) {
  paintVerdict(model);
  paintResolve(model);
  paintMemory(model);
  paintCombinations(model);
  paintYaml(model);
  paintCli(model);
  paintNotes(model);
  paintStages(model);
  paintScan();
  attachHints(document.getElementById('resolve') ?? document, HELP, liveFor);
  attachHints(document.getElementById('memory-panel') ?? document, HELP, liveFor);
}

/** @param {ReturnType<typeof buildView>} model */
function paintVerdict(model) {
  const badge = document.getElementById('badge');
  if (badge) {
    badge.className = `badge badge-${model.badge.tone}`;
    badge.textContent = model.badge.label;
  }
  setText('headline', model.headline);
  const reasons = model.winner ? model.winner.estimates.confidenceReasons : [];
  setText('confidence', `уверенность модели: ${model.confidence}${reasons.length > 0 ? ` (${reasons.join('; ')})` : ''}`);
  setText('ruleset-line', `правила ${options.rulesetVersion} · ${model.form.tier ?? 'тариф не определён'}`);
}

/** @param {ReturnType<typeof buildView>} model */
function paintResolve(model) {
  const root = document.getElementById('resolve');
  if (!root) return;
  root.replaceChildren();
  if (!model.winner) {
    const note = el('p', 'panel-note', 'Рендер-стадии нет: план не собирается. Ниже — почему и что можно сделать.');
    root.append(note);
    return;
  }
  const winner = model.winner;
  const grid = el('dl', 'resolve-grid');
  const cells = [
    ['Кадр', winner.canvas.native, `${winner.canvas.aspect} · ${winner.canvas.tokens} токенов · ${winner.canvas.snap}`, 'width'],
    ['Выход', `${winner.output.size} · ${winner.output.format.toUpperCase()} ${winner.output.bitDepth} бит`, `${winner.output.perFrameMb} МБ/кадр · фактор диска ${winner.output.factor}${winner.output.note ? ` · ${winner.output.note}` : ''}`, 'bitDepth'],
    ['Шаги', `${winner.steps.count}`, `уровень ${winner.steps.level}${winner.steps.note ? ` · ${winner.steps.note}` : ''}`, 'steps'],
    ['Память', winner.estimates.vram, `RAM ${winner.estimates.ram}`, 'vramGib'],
    ['Время', winner.estimates.time, `диск ${winner.estimates.disk}`, 'encoderDevice'],
  ];
  for (const [label, value, sub, helpId] of cells) {
    const cell = el('div', 'resolve-cell');
    const dt = el('dt', '', label);
    dt.setAttribute('data-help', helpId);
    const dd = el('dd', '', value);
    dd.append(el('span', 'sub', sub));
    cell.append(dt, dd);
    grid.append(cell);
  }
  root.append(grid);

  const bars = el('div', 'bars');
  bars.append(bar('VRAM', winner.estimates.vramPeak, winner.estimates.vramBudget));
  bars.append(bar('RAM', winner.estimates.ramPeak, winner.estimates.ramBudget));
  root.append(bars);

  const memory = el('dl', 'resolve-grid');
  const items = [
    ['KV-кеш', winner.memory.kv, 'kvCache'],
    ['Тайл VAE', winner.memory.vae, 'vaeTilePx'],
    ['Внимание', winner.memory.window, 'window'],
    ['Сетка реф.', winner.memory.ref, 'refPx'],
  ];
  for (const [label, value, helpId] of items) {
    const cell = el('div', 'resolve-cell');
    const dt = el('dt', '', label);
    dt.setAttribute('data-help', helpId);
    cell.append(dt, el('dd', '', value));
    memory.append(cell);
  }
  root.append(memory);
}

/**
 * Полоса бюджета: заполнение = пик/бюджет, цвет по зоне.
 *
 * @param {string} label
 * @param {number} peak
 * @param {number} budget
 * @returns {HTMLElement}
 */
function bar(label, peak, budget) {
  const row = el('div', 'bar-row');
  const ratio = budget > 0 ? Math.min(1, peak / budget) : 0;
  const track = el('div', `bar${ratio > 0.9 ? (ratio > 0.97 ? ' bad' : ' warn') : ''}`);
  const fill = el('span');
  fill.style.width = `${Math.round(ratio * 100)}%`;
  track.append(fill);
  row.append(el('span', '', label), track, el('span', '', `${peak.toFixed(2)} / ${budget} ГиБ`));
  return row;
}

/** @param {ReturnType<typeof buildView>} model */
function paintMemory(model) {
  const root = document.getElementById('memory-panel');
  if (!root) return;
  root.replaceChildren();
  const toggle = /** @type {HTMLButtonElement|null} */ (document.getElementById('kv-toggle'));
  const kv = model.winner ? model.winner.memory : null;
  if (toggle) {
    if (!kv || !model.winner) {
      toggle.disabled = true;
      toggle.textContent = 'KV: —';
    } else if (kv.kvMode === null) {
      toggle.disabled = true;
      toggle.textContent = 'KV: не используется';
    } else {
      toggle.disabled = false;
      const next = kv.kvPlacement === 'ram' ? 'VRAM' : 'RAM';
      toggle.textContent = `KV: ${kv.kvPlacement === 'ram' ? 'RAM' : 'VRAM'} → ${next}`;
      toggle.setAttribute('aria-pressed', 'false');
    }
  }
  if (model.memoryRows.length === 0) {
    root.append(el('p', 'panel-note', 'Раскладки нет: рендер-стадия не собрана.'));
    return;
  }
  for (const row of model.memoryRows) {
    const card = el('div', 'memory-row');
    const top = el('div', 'memory-top');
    const label = el('span', 'memory-label', row.label);
    label.setAttribute('data-help', row.helpId);
    top.append(label, el('span', 'memory-value', row.value), el('span', 'placement', row.placement));
    card.append(top, el('p', 'memory-cost', row.cost), el('p', 'memory-reason', row.reason));
    root.append(card);
  }
}

/** @param {ReturnType<typeof buildView>} model */
function paintCombinations(model) {
  const root = document.getElementById('combinations');
  if (!root) return;
  root.replaceChildren();
  if (model.combinations.length === 0) {
    root.append(el('p', 'panel-note', 'Сочетаний нет: на этом железе и кадре рендер не собирается.'));
    return;
  }
  const wrap = el('div', 'table-wrap');
  const table = el('table');
  const caption = el('caption', '', `Сочетания для ${model.winner ? model.winner.canvas.native : model.form.native}: `
    + `${model.combinations.length} вариантов, Парето ${model.combinations.filter((row) => row.pareto).length}.`);
  table.append(caption);
  const head = el('thead');
  const headRow = el('tr');
  for (const title of ['#', 'шаги', 'офлоад', 'KV', 'тайл', 'окно', 'пик, ГиБ', 'время, с', 'Q', 'Δ к победителю']) {
    headRow.append(el('th', '', title));
  }
  head.append(headRow);
  table.append(head);
  const body = el('tbody');
  for (const row of model.combinations) {
    const tr = el('tr');
    if (row.pareto) tr.classList.add('pareto');
    if (row.best) tr.classList.add('best');
    const idCell = el('td', '', row.id);
    idCell.append(el('span', 'tag', row.best ? 'лучший' : (row.pareto ? 'парето' : row.zone)));
    tr.append(
      idCell,
      el('td', 'num', String(row.steps)),
      el('td', '', row.offload),
      el('td', '', row.kv),
      el('td', '', row.vae),
      el('td', '', row.window),
      el('td', 'num', row.peakVram.toFixed(2)),
      el('td', 'num', row.timeS.toFixed(1)),
      el('td', 'num', String(row.q)),
      el('td', '', row.delta),
    );
    body.append(tr);
  }
  table.append(body);
  wrap.append(table);
  root.append(wrap);
}

/** @param {ReturnType<typeof buildView>} model */
function paintYaml(model) {
  const pre = document.getElementById('yaml-out');
  const status = document.getElementById('yaml-status');
  const copy = /** @type {HTMLButtonElement|null} */ (document.getElementById('copy-yaml'));
  const download = /** @type {HTMLButtonElement|null} */ (document.getElementById('download-yaml'));
  const hasYaml = typeof model.yaml === 'string' && model.yaml.length > 0;
  if (pre) pre.textContent = hasYaml ? model.yaml : '— план без рендер-стадии: YAML не собирается';
  if (copy) copy.disabled = !hasYaml;
  if (download) download.disabled = !hasYaml;
  if (status) {
    if (!hasYaml) {
      status.textContent = 'Файла нет: сначала должен собраться рендер (вердикт impossible — это ответ, а не ошибка).';
    } else {
      status.textContent = `${yamlFileName(model.cli)} · ${model.yaml.split('\n').length} строк · `
        + 'самодостаточный конфиг без имён пресетов и без extends';
    }
  }
}

/** @param {ReturnType<typeof buildView>} model */
function paintCli(model) {
  const root = document.getElementById('cli-panel');
  if (!root) return;
  root.replaceChildren();
  if (!model.cli) {
    root.append(el('p', 'panel-note', 'Команд нет: план не собран.'));
    return;
  }
  const rows = [
    ['проверка (без весов)', model.cli.checks.join(' && ')],
    ['одиночный кадр', model.cli.run],
    ['серия через проект', model.cli.gen],
    ['пересобрать конфиг', model.cli.save],
    ['--set-форма', model.cli.sets.join(' ')],
  ];
  for (const [label, value] of rows) {
    const row = el('div', 'cli-row');
    row.append(el('span', '', label), el('code', '', value));
    root.append(row);
  }
}

/** @param {ReturnType<typeof buildView>} model */
function paintNotes(model) {
  paintWarnings(model.warnings);
  paintSuggestions(model.suggestions);
  paintTrace(model.trace);
}

/** @param {Array<any>} warnings */
function paintWarnings(warnings) {
  const root = document.getElementById('warnings-panel');
  if (!root) return;
  root.replaceChildren();
  const head = el('h3', '', `Предупреждения (${warnings.length})`);
  root.append(head);
  if (warnings.length === 0) {
    root.append(el('p', 'panel-note', 'Предупреждений нет.'));
    return;
  }
  const list = el('div', 'notes');
  for (const warning of warnings) {
    const neutral = warning.code === 'postprocess-not-connected' || warning.code === 'quality-heuristic';
    const item = el('div', `note${neutral ? '' : ' warn'}`);
    item.append(el('code', '', warning.code));
    item.append(el('p', '', warning.detail ?? ''));
    list.append(item);
  }
  root.append(list);
}

/** @param {Array<any>} suggestions */
function paintSuggestions(suggestions) {
  const root = document.getElementById('suggestions-panel');
  if (!root) return;
  root.replaceChildren();
  if (suggestions.length === 0) return;
  root.append(el('h3', '', `Что делать (${suggestions.length})`));
  const list = el('div', 'notes');
  for (const suggestion of suggestions) {
    const item = el('div', 'note ok');
    item.append(el('code', '', suggestion.code));
    item.append(el('p', '', suggestion.detail ?? ''));
    list.append(item);
  }
  root.append(list);
}

/** @param {Array<any>} trace */
function paintTrace(trace) {
  const root = document.getElementById('trace-panel');
  if (!root) return;
  root.replaceChildren();
  if (trace.length === 0) return;
  const details = el('details', 'trace-entry');
  details.append(el('summary', '', `trace (${trace.length}): почему выбрано именно это`));
  for (const entry of trace) {
    const block = el('div');
    block.append(el('p', '', `${entry.field} = ${entry.value}`));
    block.append(el('p', '', `${entry.ruleId} · ${entry.source}`));
    if (entry.rejected.length > 0) {
      const list = el('ul');
      for (const rejected of entry.rejected) {
        list.append(el('li', '', `${rejected.value}: ${rejected.why}`));
      }
      block.append(list);
    }
    details.append(block);
  }
  root.append(details);
}

/** @param {ReturnType<typeof buildView>} model */
function paintStages(model) {
  const post = model.stages.find((stage) => stage.stage === 'postprocess');
  const reason = document.getElementById('pp-reason');
  if (reason && post) {
    reason.textContent = `причина: ${post.reason}${post.hasEstimates ? '' : ' · оценок нет (стадия выключена)'}`;
  }
  const render = model.stages.find((stage) => stage.stage === 'render');
  const result = document.getElementById('result');
  if (result && render) result.setAttribute('data-stage', render.enabled ? 'render' : 'none');
}

/** Панель сканирования YAML. */
function paintScan() {
  const root = document.getElementById('scan-panel');
  if (!root) return;
  root.replaceChildren();
  if (!lastScan) {
    root.append(el('p', 'panel-note', 'Вставьте конфиг и нажмите «проверить».'));
    return;
  }
  const scan = lastScan;
  const line = el('p', scan.ok ? 'scan-ok' : 'scan-bad',
    scan.ok
      ? `ок · ${scan.keys.length} ключей · сверено с whitelist бэкендов`
      : `проблема: ${scan.problem}`);
  root.append(line);
  if (scan.keys.length > 0) {
    const details = el('details', 'trace-entry');
    details.append(el('summary', '', `ключи (${scan.keys.length})`));
    details.append(el('p', '', scan.keys.join(', ')));
    root.append(details);
  }
  root.append(el('p', 'panel-note', `авторитетная проверка: ${scan.authority}`));
}

// ── действия ─────────────────────────────────────────────────────────────────────────────

/** Переключатель KV VRAM⇄RAM: ставит `memory.kvCache` в запрос и пересчитывает. */
function toggleKv() {
  if (!view || !view.winner) return;
  const memory = view.winner.memory;
  if (memory.kvMode === null || memory.kvMode === undefined) return;
  overrides.kvCache = memory.kvPlacement === 'ram' ? true : 'offload';
  renderAll();
}

/** Копирование YAML: `navigator.clipboard`, фоллбек — скрытая `textarea` + `execCommand`. */
async function copyYaml() {
  if (!view || !view.yaml) return;
  const status = document.getElementById('yaml-status');
  let ok = false;
  try {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
      await navigator.clipboard.writeText(view.yaml);
      ok = true;
    }
  } catch {
    ok = false;
  }
  if (!ok) ok = copyFallback(view.yaml);
  if (status) {
    const text = status.textContent ?? '';
    status.textContent = ok
      ? `${yamlFileName(view.cli)} скопирован в буфер`
      : `не удалось скопировать: выделите текст в блоке ниже (${text.split(' · ')[0]})`;
    window.setTimeout(() => paintYaml(view ?? { yaml: null, cli: null }), 2500);
  }
}

/**
 * @param {string} text
 * @returns {boolean}
 */
function copyFallback(text) {
  const area = document.createElement('textarea');
  area.value = text;
  area.setAttribute('readonly', 'readonly');
  area.style.position = 'fixed';
  area.style.top = '-1000px';
  document.body.append(area);
  area.select();
  let ok = false;
  try {
    ok = document.execCommand('copy');
  } catch {
    ok = false;
  }
  area.remove();
  return ok;
}

/** Скачать YAML файлом: имя — из `plan.cli` (`--out plan.yaml`), иначе `plan.yaml`. */
function downloadYaml() {
  if (!view || !view.yaml) return;
  const name = yamlFileName(view.cli);
  const blob = new Blob([view.yaml], { type: 'text/yaml;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  link.rel = 'noopener';
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Имя файла из команд плана: `--out <name>` в `cli.save`.
 *
 * @param {any} cli
 * @returns {string}
 */
export function yamlFileName(cli) {
  const match = /--out\s+(\S+)/u.exec(String(cli && cli.save ? cli.save : ''));
  if (match && /\.ya?ml$/iu.test(match[1])) return match[1].replace(/^.*[/\\]/u, '');
  return 'plan.yaml';
}

/** Синхронизировать хеш с состоянием (без записи истории: расчёт — это не навигация). */
function syncHash() {
  const hash = hashFromState(state);
  if (location.hash !== hash) {
    history.replaceState(null, '', `${location.pathname}${location.search}${hash}`);
  }
}

/** Подписка на `hashchange`/`popstate`: ссылка должна восстанавливать состояние формы. */
function bindHash() {
  const apply = () => {
    const next = stateFromHash(location.hash);
    if (hashFromState(next) === hashFromState(state) && view !== null) return;
    state = next;
    overrides = {};
    fillForm(state);
    renderAll();
  };
  window.addEventListener('hashchange', apply);
  window.addEventListener('popstate', apply);
}

/** Сканирование YAML: разбор + сверка ключей (авторитет — CLI). */
function bindScan() {
  const run = document.getElementById('scan-run');
  const input = /** @type {HTMLTextAreaElement|null} */ (document.getElementById('scan-input'));
  const take = document.getElementById('scan-current');
  if (run && input) {
    run.addEventListener('click', () => {
      lastScan = scanYaml(input.value);
      paintScan();
    });
  }
  if (take && input) {
    take.addEventListener('click', () => {
      input.value = view && view.yaml ? view.yaml : '';
      lastScan = scanYaml(input.value);
      paintScan();
    });
  }
}

// ── ошибки ──────────────────────────────────────────────────────────────────────────────

/**
 * Ошибка запроса: текст рядом с полем (`PlanError.field` → контрол), иначе — общий alert.
 *
 * @param {any} error
 * @returns {void}
 */
function showPlanError(error) {
  const field = error && typeof error === 'object' ? String(error.field ?? '') : '';
  const message = error && typeof error === 'object' && error.detail
    ? `${error.code}: ${error.detail}`
    : String(error && typeof error === 'object' && 'message' in error ? error.message : error);
  const id = ERROR_FIELDS[field];
  if (id && showErrorAt(id, message)) return;
  const global = document.getElementById('form-error');
  if (global) {
    global.textContent = `запрос не прошёл схему — ${message}`;
    global.hidden = false;
  }
}

/**
 * Показать текст ошибки у поля.
 *
 * @param {string} id id контрола или контейнера
 * @param {string} message
 * @returns {boolean} удалось ли найти место
 */
function showErrorAt(id, message) {
  const control = document.getElementById(id);
  if (!control) return false;
  control.setAttribute('aria-invalid', 'true');
  const container = control.closest('.field') ?? control.parentElement;
  if (!container) return false;
  const note = el('p', 'field-error', message);
  container.append(note);
  return true;
}

/** Убрать все тексты ошибок и `aria-invalid`. */
function clearErrors() {
  for (const node of Array.from(document.querySelectorAll('.field-error'))) node.remove();
  for (const node of Array.from(document.querySelectorAll('[aria-invalid="true"]'))) {
    node.removeAttribute('aria-invalid');
  }
  const global = document.getElementById('form-error');
  if (global) {
    global.textContent = '';
    global.hidden = true;
  }
}

// ── мелкие помощники ─────────────────────────────────────────────────────────────────────

/**
 * @param {string} tag
 * @param {string} [className]
 * @param {string} [text]
 * @returns {HTMLElement}
 */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * @param {string} id
 * @param {string} text
 * @returns {void}
 */
function setText(id, text) {
  const node = document.getElementById(id);
  if (node) node.textContent = text;
}

/**
 * @param {string} id
 * @param {string|number} value
 * @returns {void}
 */
function setValue(id, value) {
  const node = /** @type {HTMLInputElement|HTMLSelectElement|null} */ (document.getElementById(id));
  if (node) node.value = String(value);
}

/**
 * @param {string} name
 * @param {string} value
 * @returns {string}
 */
function radioValue(name, value) {
  const checked = document.querySelector(`input[name="${name}"]:checked`);
  return checked instanceof HTMLInputElement ? checked.value : value;
}

/**
 * @param {string} id
 * @param {string} value
 * @returns {string}
 */
function selectValue(id, value) {
  const node = /** @type {HTMLSelectElement|null} */ (document.getElementById(id));
  return node ? node.value : value;
}

/**
 * @param {string} name
 * @param {string} value
 * @returns {void}
 */
function checkRadio(name, value) {
  const node = document.querySelector(`input[name="${name}"][value="${value}"]`);
  if (node instanceof HTMLInputElement) node.checked = true;
}

// `canvasText` ре-экспортируется из вью-модели: странице он нужен только в подписях.
export { canvasText, DEFAULT_STATE };
