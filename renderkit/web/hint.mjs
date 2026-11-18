// @ts-check
/**
 * Подсказки `?` в кружке у каждого параметра (ADR A10b) — **без библиотек** и без `title=`.
 *
 * Почему не `title`:
 *
 * - нет форматирования (три аспекта в одну строку не влезают);
 * - не открывается с клавиатуры (`focusin`) и на тач-устройстве (`click`);
 * - не связан с полем через `aria-describedby`.
 *
 * Поэтому у кнопки есть `aria-describedby="hint-<id>"` и `aria-expanded`, а панель —
 * `role="tooltip"`; открытие/закрытие делается JS (а не CSS-селекторами), чтобы работали задержка
 * 150 мс, `Escape` и клик вне.
 *
 * Разметка панели вынесена в `hintPanelMarkup(id)` — **строку** без DOM: `test/html.test.mjs`
 * сверяет по ней, что у каждого `aria-describedby` в `index.html` действительно есть цель
 * (`role="tooltip"`), не поднимая jsdom.
 *
 * Панели живут в `document.body` с `position: fixed`: их не режут `overflow: hidden` карточки и
 * пересчёт формы, а позиция считается от `getBoundingClientRect()` кнопки с переворотом у краёв.
 */

/** Разделы панели в фиксированном порядке: подпись + ключ аспекта. */
const ASPECTS = Object.freeze([
  Object.freeze({ key: 'quality', label: 'Качество' }),
  Object.freeze({ key: 'perf', label: 'Производительность' }),
  Object.freeze({ key: 'system', label: 'Система' }),
]);

/** Задержка открытия/закрытия, мс (спека A10b). */
export const HINT_DELAY_MS = 150;

/** Отступ панели от края экрана при позиционировании, px. */
const EDGE_MARGIN = 8;

/**
 * Шаблон панели подсказки: `id="hint-<id>"`, `role="tooltip"`, `hidden`.
 *
 * Текст в шаблоне пустой: `attachHints` заполняет его через `textContent` (никакого `innerHTML`
 * из данных), а тест читает саму строку — ему нужны только id и `role`.
 *
 * @param {string} id id подсказки (`HELP[id]`)
 * @returns {string} HTML-строка панели
 */
export function hintPanelMarkup(id) {
  const rows = ASPECTS
    .map((aspect) => `<dt class="aspect-${aspect.key}">${aspect.label}</dt><dd data-aspect="${aspect.key}"></dd>`)
    .join('');
  return `<div class="hint-panel" id="hint-${id}" role="tooltip" hidden>`
    + `<dl class="hint-aspects">${rows}</dl>`
    + '<p class="hint-note" data-hint-note hidden></p>'
    + '<p class="hint-live" data-hint-live hidden></p>'
    + '<p class="hint-src" data-hint-src></p>'
    + '</div>';
}

/** Глобальные слушатели `Escape`/клика вне регистрируются один раз. */
let globalsBound = false;

/** Все открытые панели: нужны `Escape` и клику вне. @type {Set<{panel: HTMLElement, button: HTMLElement, close: () => void}>} */
const openPanels = new Set();

/**
 * Подключить подсказки ко всем `[data-help]` внутри `root`.
 *
 * Контейнер поля помечается `data-help="<id>"`, внутри него уже есть `<label>`/`<legend>` —
 * кнопка `?` дописывается туда, панель — в `document.body`.
 *
 * Функция идемпотентна на контейнере: повторный вызов не плодит вторую кнопку (нужно, потому что
 * динамические панели — карточка плана и раскладка памяти — перерисовываются на каждый пересчёт).
 *
 * @param {Document|Element} root где искать контейнеры полей
 * @param {Record<string, import('../src/data/help.mjs').HelpEntry>} helpMap `HELP`
 * @param {(id: string) => (string|null)} [getLive] живая дельта для строки «сейчас»
 * @returns {void}
 */
export function attachHints(root, helpMap, getLive) {
  if (!root || typeof root.querySelectorAll !== 'function') return;
  bindGlobals();
  for (const container of Array.from(root.querySelectorAll('[data-help]'))) {
    const id = container.getAttribute('data-help');
    if (!id) continue;
    const entry = helpMap[id];
    if (!entry) continue;
    if (container.querySelector(':scope > .hint, :scope > label > .hint, :scope > legend > .hint')) continue;
    const button = makeButton(id, entry.label);
    // Панель на параметр одна: `?` в карточке плана и в раскладке памяти ссылаются на ту же
    // панель, что и поле формы (`aria-describedby="hint-<id>"`), иначе в документе были бы
    // дубли id. Живую дельту в неё пишет любой из привязанных кнопок (`getLive(id)`).
    const existing = document.getElementById(`hint-${id}`);
    const panel = existing
      ? { element: existing, live: (value) => setLiveText(existing, value) }
      : makePanel(id, entry);
    const anchor = container.querySelector(':scope > label, :scope > legend') ?? container;
    anchor.append(button);
    if (!existing) document.body.append(panel.element);
    wire(id, button, panel, getLive);
  }
}

/**
 * @param {string} id
 * @param {string} label
 * @returns {HTMLButtonElement}
 */
function makeButton(id, label) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'hint';
  button.setAttribute('aria-describedby', `hint-${id}`);
  button.setAttribute('aria-expanded', 'false');
  button.setAttribute('aria-label', `Пояснение: ${String(label).toLowerCase()}`);
  button.append(document.createTextNode('?'));
  const sr = document.createElement('span');
  sr.className = 'sr-only';
  sr.textContent = String(label);
  button.append(sr);
  return button;
}

/**
 * Панель из шаблона `hintPanelMarkup` + текст аспектов/оговорки/живой дельты/источника.
 *
 * Шаблон — наша же константа (не пользовательские данные), поэтому он вставляется через
 * `<template>`: так разметка в тесте и в браузере не может разойтись.
 *
 * @param {string} id
 * @param {import('../src/data/help.mjs').HelpEntry} entry
 * @returns {{element: HTMLElement, live: (value: string|null) => void}}
 */
function makePanel(id, entry) {
  const template = document.createElement('template');
  template.innerHTML = hintPanelMarkup(id);
  const element = /** @type {HTMLElement} */ (template.content.firstElementChild);
  for (const aspect of ASPECTS) {
    const cell = element.querySelector(`[data-aspect="${aspect.key}"]`);
    if (cell) cell.textContent = entry.aspects[aspect.key] ?? '';
  }
  const note = element.querySelector('[data-hint-note]');
  if (note && entry.note) {
    note.textContent = `оговорка: ${entry.note}`;
    /** @type {HTMLElement} */ (note).hidden = false;
  }
  const source = element.querySelector('[data-hint-src]');
  if (source) source.textContent = `источник: ${entry.sources.join(' · ')}`;
  return { element, live: (value) => setLiveText(element, value) };
}

/**
 * Строка «сейчас: …» в панели: `null` — строку спрятать (живой дельты нет).
 *
 * @param {HTMLElement} panel
 * @param {string|null} value
 * @returns {void}
 */
function setLiveText(panel, value) {
  const liveCell = /** @type {HTMLElement|null} */ (panel.querySelector('[data-hint-live]'));
  if (!liveCell) return;
  if (!value) {
    liveCell.hidden = true;
    liveCell.textContent = '';
    return;
  }
  liveCell.textContent = `сейчас: ${value}`;
  liveCell.hidden = false;
}

/**
 * Навесить события на кнопку и панель.
 *
 * Открытие: `mouseenter`/`focusin` с задержкой 150 мс. Клик/тап **фиксирует** панель
 * (`aria-expanded="true"`), повторный клик закрывает; `mouseleave`/`focusout` закрывают только
 * незафиксированную панель, `Escape` и клик вне — любую.
 *
 * @param {string} id
 * @param {HTMLButtonElement} button
 * @param {{element: HTMLElement, live: (value: string|null) => void}} panel
 * @param {((id: string) => (string|null))|undefined} getLive живая дельта для строки «сейчас»
 * @returns {void}
 */
function wire(id, button, panel, getLive) {
  /** @type {number|null} */
  let timer = null;
  let pinned = false;

  const clear = () => {
    if (timer !== null) {
      clearTimeout(timer);
      timer = null;
    }
  };

  /** @param {number} delay @param {() => void} action */
  const later = (delay, action) => {
    clear();
    timer = window.setTimeout(() => {
      timer = null;
      action();
    }, delay);
  };

  /** @type {() => void} */
  const open = () => {
    clear();
    panel.element.hidden = false;
    button.setAttribute('aria-expanded', 'true');
    openPanels.add(handle);
    place(button, panel.element);
  };

  /** @type {() => void} */
  const closeNow = () => {
    clear();
    pinned = false;
    panel.element.hidden = true;
    button.setAttribute('aria-expanded', 'false');
    openPanels.delete(handle);
  };

  /** @type {() => void} */
  const closeSoon = () => later(HINT_DELAY_MS, () => {
    if (!pinned) closeNow();
  });

  const handle = { panel: panel.element, button, close: closeNow };

  const refresh = () => {
    if (typeof getLive !== 'function') return;
    let value = null;
    try {
      value = getLive(id);
    } catch {
      value = null;
    }
    panel.live(value);
  };

  button.addEventListener('mouseenter', () => {
    refresh();
    later(HINT_DELAY_MS, open);
  });
  button.addEventListener('mouseleave', () => {
    later(HINT_DELAY_MS, () => {
      if (!pinned && !panel.element.matches(':hover')) closeNow();
    });
  });
  button.addEventListener('focusin', () => {
    refresh();
    later(HINT_DELAY_MS, open);
  });
  button.addEventListener('focusout', () => closeSoon());
  button.addEventListener('click', (event) => {
    event.preventDefault();
    event.stopPropagation();
    refresh();
    if (pinned || !panel.element.hidden) {
      closeNow();
      return;
    }
    pinned = true;
    open();
  });

  panel.element.addEventListener('mouseenter', clear);
  panel.element.addEventListener('mouseleave', closeSoon);
  panel.element.addEventListener('click', (event) => event.stopPropagation());
}

/**
 * Поставить панель рядом с кнопкой: `position: fixed`, с переворотом по вертикали и
 * горизонтали и ограничением по `innerWidth`/`innerHeight`.
 *
 * @param {HTMLElement} button
 * @param {HTMLElement} panel
 * @returns {void}
 */
export function place(button, panel) {
  const rect = button.getBoundingClientRect();
  panel.style.visibility = 'hidden';
  panel.style.left = '0px';
  panel.style.top = '0px';
  const width = panel.offsetWidth;
  const height = panel.offsetHeight;
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  const maxLeft = Math.max(EDGE_MARGIN, viewportWidth - width - EDGE_MARGIN);
  const maxTop = Math.max(EDGE_MARGIN, viewportHeight - height - EDGE_MARGIN);

  let left = rect.left + rect.width / 2 - width / 2;
  if (left + width > viewportWidth - EDGE_MARGIN) left = maxLeft;
  if (left < EDGE_MARGIN) left = EDGE_MARGIN;

  // Сначала пробуем под кнопкой, затем над ней, и в любом случае не выходим за экран: кнопка
  // может быть ниже вьюпорта (страница прокручена вверх), и «переворот» сам по себе не спасает.
  let top = rect.bottom + EDGE_MARGIN;
  if (top + height > viewportHeight - EDGE_MARGIN) {
    const above = rect.top - height - EDGE_MARGIN;
    top = above >= EDGE_MARGIN ? above : maxTop;
  }
  if (top + height > viewportHeight - EDGE_MARGIN) top = maxTop;
  if (top < EDGE_MARGIN) top = EDGE_MARGIN;

  panel.style.left = `${Math.round(Math.min(left, maxLeft))}px`;
  panel.style.top = `${Math.round(top)}px`;
  panel.style.visibility = 'visible';
}

/**
 * `Escape` закрывает все панели, клик/тап вне — тоже; прокрутка закрывает (панель привязана к
 * кнопке и иначе «уезжает»).
 *
 * @returns {void}
 */
function bindGlobals() {
  if (globalsBound) return;
  globalsBound = true;
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' && event.key !== 'Esc') return;
    for (const handle of Array.from(openPanels)) handle.close();
  });
  document.addEventListener('pointerdown', (event) => {
    const target = /** @type {Node|null} */ (event.target);
    for (const handle of Array.from(openPanels)) {
      if (!target) continue;
      if (handle.panel.contains(target) || handle.button.contains(target)) continue;
      handle.close();
    }
  });
  window.addEventListener('scroll', () => {
    for (const handle of Array.from(openPanels)) handle.close();
  }, true);
  window.addEventListener('resize', () => {
    for (const handle of Array.from(openPanels)) handle.close();
  });
}

/**
 * Закрыть все открытые панели (нужно тестам/перерисовке формы).
 *
 * @returns {void}
 */
export function closeAllHints() {
  for (const handle of Array.from(openPanels)) handle.close();
}
