// @ts-check
/**
 * Статические проверки страницы: разметка, доступность подсказок, MIME сервера, границы слоёв.
 *
 * Тест не поднимает DOM (в пакете нет зависимостей и jsdom не появится): разметка читается как
 * текст, а шаблон панели подсказки берётся из `hint.mjs` — там он вынесен в строку именно ради
 * этой проверки.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { HELP } from '../src/data/help.mjs';
import { HINT_DELAY_MS, hintPanelMarkup } from '../web/hint.mjs';

const WEB = join(dirname(fileURLToPath(import.meta.url)), '..', 'web');
const HTML = readFileSync(join(WEB, 'index.html'), 'utf8');
const APP = readFileSync(join(WEB, 'app.mjs'), 'utf8');
const HINT = readFileSync(join(WEB, 'hint.mjs'), 'utf8');
const VIEWMODEL = readFileSync(join(WEB, 'viewmodel.mjs'), 'utf8');
const CSS = readFileSync(join(WEB, 'styles.css'), 'utf8');
const SERVE = readFileSync(join(WEB, 'serve.mjs'), 'utf8');

/** Поля формы, для которых подсказка обязательна. */
const FORM_HELP_IDS = Object.freeze([
  'width', 'height', 'quality', 'bitDepth', 'model', 'vramGib', 'ramGib', 'disk', 'refs', 'anime', 'freeDiskGib',
]);

/**
 * @param {string} tag
 * @param {string} name
 * @returns {string|null} значение атрибута (без кавычек)
 */
function attribute(tag, name) {
  const pattern = new RegExp(`\\s${name}\\s*=\\s*"([^"]*)"`, 'iu');
  const match = pattern.exec(tag);
  return match ? match[1] : null;
}

/** @returns {Array<{tag: string, id: string|null, describedBy: string|null}>} */
function formControls() {
  const tags = HTML.match(/<(?:input|select|textarea)\b[^>]*>/giu) ?? [];
  return tags.map((tag) => ({
    tag,
    type: attribute(tag, 'type'),
    id: attribute(tag, 'id'),
    describedBy: attribute(tag, 'aria-describedby'),
  }));
}

/** @returns {string[]} все `id` в разметке */
function staticIds() {
  return (HTML.match(/\sid\s*=\s*"([^"]+)"/giu) ?? []).map((match) => match.replace(/^.*"([^"]+)"$/u, '$1'));
}

test('index.html: модульный скрипт, никаких инлайн-скриптов и title=', () => {
  assert.match(HTML, /<script\s+type="module"\s+src="app\.mjs"><\/script>/u);
  const scripts = HTML.match(/<script\b[^>]*>/giu) ?? [];
  assert.equal(scripts.length, 1, 'должен быть ровно один тег script (модуль app.mjs)');
  for (const tag of scripts) assert.ok(attribute(tag, 'src'), 'скрипт обязан быть внешним файлом');
  assert.equal(/\stitle\s*=/iu.test(HTML), false, 'title= не используется: подсказки только через role="tooltip"');
  assert.equal(/\son(?:click|input|change|load|mouseover)\s*=/iu.test(HTML), false, 'никаких инлайн-обработчиков');
});

test('index.html: у каждого контрола есть id и aria-describedby на существующую цель', () => {
  const ids = new Set(staticIds());
  const controls = formControls();
  assert.ok(controls.length >= 14, `контролов ${controls.length}, ждали форму целиком`);
  for (const control of controls) {
    assert.ok(control.id, `у контрола нет id: ${control.tag.slice(0, 70)}`);
    assert.ok(control.describedBy, `${control.id}: нет aria-describedby`);
    const target = /** @type {string} */ (control.describedBy);
    if (target.startsWith('hint-')) {
      const helpId = target.slice('hint-'.length);
      assert.ok(HELP[helpId], `${control.id}: подсказки ${helpId} нет в HELP`);
      const panel = hintPanelMarkup(helpId);
      assert.match(panel, new RegExp(`id="${target}"`, 'u'), `${control.id}: панель ${target} не найдена в шаблоне`);
      assert.match(panel, /role="tooltip"/u, `${target}: панель без role="tooltip"`);
    } else {
      assert.ok(ids.has(target), `${control.id}: цель ${target} не существует в разметке`);
    }
  }
});

test('index.html: подписи и подсказки накрывают все поля формы', () => {
  const helpIds = new Set((HTML.match(/data-help="([^"]+)"/gu) ?? []).map((match) => match.replace(/^.*"([^"]+)"$/u, '$1')));
  assert.deepEqual([...helpIds].sort(), [...FORM_HELP_IDS].sort());
  for (const id of FORM_HELP_IDS) assert.ok(HELP[id], `подсказка ${id} не описана`);

  for (const id of ['w', 'h', 'model', 'vram', 'ram', 'refs', 'free-disk']) {
    assert.match(HTML, new RegExp(`for="${id}"`, 'u'), `у поля ${id} нет label`);
  }
  // Радиокнопки и чекбокс живут внутри <label>, а не через for=.
  for (const id of ['q-draft', 'q-standard', 'q-max', 'bits-8', 'bits-16', 'disk-hdd', 'disk-ssd', 'disk-nvme', 'anime']) {
    assert.ok(staticIds().includes(id), `нет контрола ${id}`);
  }
});

test('index.html: быстрые размеры, профили железа, панели результата', () => {
  for (const size of ['1280x720', '1920x1080', '2560x1440', '3200x1792']) {
    assert.ok(HTML.includes(`data-size="${size}"`), `нет кнопки размера ${size}`);
  }
  for (const hw of ['8x16', '12x32', '16x64']) assert.ok(HTML.includes(`data-hw="${hw}"`), `нет профиля ${hw}`);
  for (const id of ['badge', 'headline', 'resolve', 'memory-panel', 'combinations', 'yaml-out', 'cli-panel',
    'warnings-panel', 'suggestions-panel', 'trace-panel', 'scan-input', 'scan-panel', 'pp-reason']) {
    assert.ok(staticIds().includes(id), `нет контейнера ${id}`);
  }
  assert.match(HTML, /id="memory-panel-card"[\s\S]*?id="kv-toggle"/u, 'в панели памяти нет переключателя KV');
  assert.match(HTML, /id="copy-yaml"/u);
  assert.match(HTML, /id="download-yaml"/u);
  assert.match(HTML, /role="alert"/u, 'ошибки формы должны объявляться как alert');
  assert.match(HTML, /aria-live="polite"/u, 'результат обновляется без перезагрузки');
});

test('index.html: без внешних ресурсов (прототип работает офлайн)', () => {
  assert.equal(/(?:src|href)\s*=\s*"https?:\/\//iu.test(HTML), false);
  assert.match(HTML, /href="styles\.css"/u);
});

test('hint.mjs: шаблон панели с тремя аспектами и role="tooltip"', () => {
  for (const id of FORM_HELP_IDS) {
    const panel = hintPanelMarkup(id);
    assert.match(panel, new RegExp(`id="hint-${id}"`, 'u'));
    assert.match(panel, /class="hint-panel"/u);
    assert.match(panel, /role="tooltip"/u);
    assert.match(panel, /\shidden/u, 'панель по умолчанию скрыта');
    assert.match(panel, /<dl class="hint-aspects">/u);
    assert.match(panel, /data-aspect="quality"/u);
    assert.match(panel, /data-aspect="perf"/u);
    assert.match(panel, /data-aspect="system"/u);
    assert.match(panel, /data-hint-live/u, 'строка «сейчас»');
    assert.match(panel, /data-hint-src/u, 'строка «источник»');
  }
  const panel = hintPanelMarkup('steps');
  for (const label of ['Качество', 'Производительность', 'Система']) assert.ok(panel.includes(label), `нет раздела ${label}`);
});

test('hint.mjs: кнопка `?` доступна и события навешены', () => {
  assert.match(HINT, /export function attachHints/u);
  assert.match(HINT, /button\.className = 'hint'/u);
  assert.match(HINT, /aria-describedby/u);
  assert.match(HINT, /aria-expanded/u);
  assert.match(HINT, /aria-label/u);
  assert.match(HINT, /sr-only/u);
  assert.match(HINT, /'Escape'/u, 'Escape обязан закрывать панель');
  assert.match(HINT, /pointerdown/u, 'клик вне закрывает');
  assert.match(HINT, /getBoundingClientRect\(\)/u, 'позиционирование от кнопки');
  assert.match(HINT, /innerWidth/u);
  assert.match(HINT, /innerHeight/u);
  assert.match(HINT, /maxTop/u, 'панель обязана клампиться по вертикали, а не только переворачиваться');
  assert.equal(HINT_DELAY_MS, 150);
  // innerHTML допускается только для собственного шаблона (=<template>), не для данных.
  const innerHtmlUses = HINT.match(/\.innerHTML\s*=/gu) ?? [];
  assert.equal(innerHtmlUses.length, 1, 'innerHTML только в makePanel — для собственного шаблона');
  assert.match(HINT, /template\.innerHTML = hintPanelMarkup\(id\)/u);
});

test('app.mjs: без innerHTML из данных, с дебаунсом и модулями ядра', () => {
  assert.equal(/\.innerHTML\s*=|insertAdjacentHTML|outerHTML/u.test(APP), false, 'страница собирает DOM текстом, а не строками HTML');
  assert.equal(/document\.write/iu.test(APP), false);
  assert.match(APP, /import \{ computePlan, listOptions \} from '\.\.\/src\/index\.mjs'/u);
  assert.match(APP, /from '\.\/viewmodel\.mjs'/u);
  assert.match(APP, /const DEBOUNCE_MS = 120/u);
  assert.match(APP, /addEventListener\('hashchange'/u);
  assert.match(APP, /addEventListener\('popstate'/u);
  assert.match(APP, /navigator\.clipboard/u, 'кнопка «копировать» с фоллбеком');
  assert.match(APP, /execCommand\('copy'\)/u);
  assert.match(APP, /URL\.createObjectURL/u, 'кнопка «скачать»');
  assert.match(APP, /data-hw|data-size/u, 'быстрые кнопки подключены');
});

test('viewmodel.mjs: остаётся чистым (без DOM)', () => {
  assert.equal(/document\./u.test(VIEWMODEL), false, 'вью-модель не трогает DOM');
  assert.equal(/window\./u.test(VIEWMODEL), false, 'вью-модель не трогает window');
  assert.equal(/\.innerHTML\s*=|insertAdjacentHTML|outerHTML/u.test(VIEWMODEL), false);
  assert.match(VIEWMODEL, /export function buildView/u);
  assert.match(VIEWMODEL, /export function hintLive/u);
  assert.match(VIEWMODEL, /export function scanYaml/u);
  assert.match(VIEWMODEL, /export function stateFromHash/u);
  assert.match(VIEWMODEL, /export function hashFromState/u);
  assert.match(VIEWMODEL, /export function requestFromState/u);
});

test('styles.css: кружок 16 px, панель 340 px, цвета аспектов, адаптивность', () => {
  assert.match(CSS, /\.hint\s*\{[^}]*width:\s*16px/u);
  assert.match(CSS, /\.hint\s*\{[^}]*height:\s*16px/u);
  assert.match(CSS, /\.hint-panel\s*\{[^}]*width:\s*340px/u);
  assert.match(CSS, /\.aspect-quality\s*\{[^}]*var\(--quality\)/u);
  assert.match(CSS, /\.aspect-perf\s*\{[^}]*var\(--perf\)/u);
  assert.match(CSS, /\.aspect-system\s*\{[^}]*var\(--system\)/u);
  assert.match(CSS, /\.sr-only\s*\{/u);
  assert.match(CSS, /@media \(max-width: 640px\)/u);
  assert.match(CSS, /@media \(max-width: 400px\)/u, 'страница должна собираться на 360 px');
  assert.match(CSS, /prefers-reduced-motion/u);
  assert.match(CSS, /\.badge-ok/u);
  assert.match(CSS, /\.badge-bad/u);
  assert.match(CSS, /\.bar\b/u);
});

test('serve.mjs: MIME для .mjs, no-store, 404 и запрет выхода за web/', () => {
  assert.match(SERVE, /from 'node:http'/u);
  assert.match(SERVE, /'\.mjs': 'text\/javascript; charset=utf-8'/u);
  assert.match(SERVE, /'\.html': 'text\/html; charset=utf-8'/u);
  assert.match(SERVE, /'\.css': 'text\/css; charset=utf-8'/u);
  assert.match(SERVE, /Cache-Control': 'no-store'/u);
  assert.match(SERVE, /404/u);
  assert.match(SERVE, /8123/u, 'порт по умолчанию');
  assert.match(SERVE, /process\.argv\[2\]/u, 'порт из аргумента');
  assert.match(SERVE, /export function resolvePath/u);
  assert.match(SERVE, /'\.\.'/u, 'проверка сегментов ..');
  // Ядро монтируется одной точкой `/src/`: без неё `import '../src/index.mjs'` в браузере — 404.
  assert.match(SERVE, /PKG_SRC/u);
  assert.match(SERVE, /\/src\//u);
});

test('подсказка: метка аспекта не налезает на текст (CSS-инвариант)', () => {
  const block = (selector) => {
    const at = CSS.indexOf(`${selector} {`);
    assert.ok(at >= 0, `в styles.css нет блока ${selector}`);
    return CSS.slice(at, CSS.indexOf('}', at));
  };
  const aspects = block('.hint-aspects');
  assert.match(aspects, /grid-template-columns:\s*max-content\s+minmax\(0,\s*1fr\)/u,
    'метка должна занимать свою ширину, текст — остаток с minmax(0, 1fr)');
  const dt = block('.hint-aspects dt');
  assert.match(dt, /overflow-wrap:\s*break-word/u, 'метке нужен перенос длинного слова');
  assert.match(dt, /hyphens:\s*auto/u, 'и перенос по слогам (lang="ru" на <html>)');
  const dd = block('.hint-aspects dd');
  assert.match(dd, /min-width:\s*0/u, 'dd не должен раздувать грид');
  const panel = block('.hint-panel');
  assert.match(panel, /overflow-wrap:\s*anywhere/u, 'длинные ключи не должны вылезать из панели');
  assert.match(HTML, /<html lang="ru">/u);
});
