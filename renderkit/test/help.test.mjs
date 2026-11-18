// @ts-check
/**
 * Подсказки (`src/data/help.mjs`) как данные: покрытие, длина, источники, живой пересчёт.
 *
 * Тест намеренно ничего не знает о вёрстке: он проверяет контракт A10b «по записи на параметр»,
 * а `live` — тем, что `hintLive` действительно отдаёт строку для текущего запроса.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { HELP, HELP_SECTIONS, LIVE_IDS } from '../src/data/help.mjs';
import { DEFAULT_STATE, hintLive, requestFromState } from '../web/viewmodel.mjs';

/** Корень репозитория: `renderkit/test/help.test.mjs` → `../..`. */
const REPO_ROOT = join(dirname(fileURLToPath(import.meta.url)), '..', '..');

/** Все параметры, для которых подсказка обязательна (A10b: форма + `Plan.render`/`memory` + стадия). */
const REQUIRED_IDS = Object.freeze([
  // поля входа
  'width', 'height', 'quality', 'bitDepth', 'model', 'vramGib', 'ramGib', 'disk', 'refs', 'anime', 'freeDiskGib',
  // `Plan.render` / `Plan.render.memory`
  'steps', 'offload', 'offloadBlocks', 'encoderDevice', 'precision', 'loraScale',
  'kvCache', 'vaeTilePx', 'window', 'refPx',
  // отдельная стадия
  'postprocess',
]);

/** Имена пресетов/тарифов, которых в подсказках быть не должно (калькулятор их не знает). */
const PRESET_TOKENS = Object.freeze([
  /VRAM8-/u,
  /VRAM12-/u,
  /VRAM16-/u,
  /["'`(](?:preview|release)(?:-anime)?["'`)]/u,
]);

/** Значения `live`, которые разрешено указывать в записи. */
const ALLOWED_LIVE = new Set([...LIVE_IDS, null]);

/** 16:9 на 2560×1440 — рабочий запрос для what-if (кэш эмбеддингов не нужен, ядро считает план). */
const QUERY = { image: { width: 2560, height: 1440, quality: 'max' }, hardware: { model: 'qwen21-base', vramGib: 16, ramGib: 64, disk: 'nvme' } };
const H3_QUERY = { image: { width: 1024, height: 1024, quality: 'max' }, hardware: { model: 'h3-image', vramGib: 16, ramGib: 64, disk: 'nvme' } };

/**
 * @param {string} source
 * @returns {string} путь к файлу (часть до `#`)
 */
function sourceFile(source) {
  return source.split('#')[0];
}

test('HELP: запись на каждый параметр и никаких лишних', () => {
  assert.deepEqual(Object.keys(HELP).sort(), [...REQUIRED_IDS].sort());
  for (const id of REQUIRED_IDS) {
    const entry = HELP[id];
    assert.equal(entry.id, id, `${id}: id записи должен совпадать с ключом`);
    assert.ok(entry.label && entry.label.trim().length > 0, `${id}: пустой label`);
    assert.ok(HELP_SECTIONS.includes(entry.section), `${id}: неизвестный раздел ${entry.section}`);
    assert.ok(ALLOWED_LIVE.has(entry.live ?? null), `${id}: live=${JSON.stringify(entry.live)} не из списка`);
  }
});

test('HELP: три непустых аспекта, каждый не длиннее 320 символов', () => {
  for (const [id, entry] of Object.entries(HELP)) {
    for (const key of ['quality', 'perf', 'system']) {
      const text = entry.aspects[key];
      assert.equal(typeof text, 'string', `${id}.${key}: ожидается строка`);
      assert.ok(text.trim().length >= 10, `${id}.${key}: слишком короткий текст`);
      assert.ok([...text].length <= 320, `${id}.${key}: ${[...text].length} символов > 320`);
    }
  }
});

test('HELP: разделы покрывают форму, план и стадию постобработки', () => {
  const sections = new Set(Object.values(HELP).map((entry) => entry.section));
  for (const section of HELP_SECTIONS) assert.ok(sections.has(section), `раздел ${section} не покрыт`);
  assert.equal(HELP.postprocess.section, 'postprocess');
  for (const id of ['kvCache', 'vaeTilePx', 'window', 'refPx']) assert.equal(HELP[id].section, 'memory');
  for (const id of ['steps', 'offload', 'offloadBlocks', 'encoderDevice', 'precision', 'loraScale']) {
    assert.equal(HELP[id].section, 'render');
  }
});

test('HELP: в текстах нет имён пресетов', () => {
  for (const [id, entry] of Object.entries(HELP)) {
    const texts = [entry.label, entry.note ?? '', ...Object.values(entry.aspects)].join(' \n ');
    for (const pattern of PRESET_TOKENS) {
      assert.equal(pattern.test(texts), false, `${id}: найдено имя пресета (${pattern})`);
    }
  }
});

test('HELP: sources ссылаются на существующие файлы репозитория', () => {
  for (const [id, entry] of Object.entries(HELP)) {
    assert.ok(Array.isArray(entry.sources) && entry.sources.length > 0, `${id}: пустой sources`);
    for (const source of entry.sources) {
      const file = sourceFile(source);
      const path = join(REPO_ROOT, file);
      assert.ok(existsSync(path), `${id}: источника ${source} нет (${path})`);
      assert.ok(statSync(path).isFile(), `${id}: ${file} — не файл`);
    }
  }
});

test('HELP: числа подсказок — из наших замеров (A10b)', () => {
  const text = (id) => `${HELP[id].aspects.quality} ${HELP[id].aspects.perf} ${HELP[id].aspects.system}`;
  assert.match(text('steps'), /8\.2/u, 'steps: 8.2 с/шаг на 1440p');
  assert.match(text('kvCache'), /2\.13/u, 'kvCache: нижняя граница префикса');
  assert.match(text('kvCache'), /7\.14/u, 'kvCache: верхняя граница префикса');
  assert.match(text('vaeTilePx'), /1\.69/u, 'vaeTilePx: сетка 512/384');
  assert.match(text('vaeTilePx'), /2\.71/u, 'vaeTilePx: сетка 256/192');
  assert.match(text('vaeTilePx'), /1\.51/u, 'vaeTilePx: сетка 1024/768');
  assert.match(text('vaeTilePx'), /15\.0/u, 'vaeTilePx: пик 1024/768');
  assert.match(text('window'), /17 %/u, 'window: −17 % времени');
  assert.match(text('window'), /2\.6 %/u, 'window: −2.6 % резкости');
  assert.match(text('encoderDevice'), /10\.5/u, 'encoderDevice: GPU 10.5 с');
  assert.match(text('encoderDevice'), /55–255/u, 'encoderDevice: CPU 55–255 с');
  assert.match(text('disk'), /3\.0/u, 'disk: NVMe 3.0 ГиБ/с');
  assert.match(text('disk'), /0\.13/u, 'disk: HDD 0.13 ГиБ/с');
  assert.match(text('disk'), /74\.9/u, 'disk: футпринт 8 ГиБ VRAM');
  assert.match(text('disk'), /54\.9/u, 'disk: футпринт 12 ГиБ VRAM');
  assert.match(text('disk'), /181\.0/u, 'disk: футпринт 16 ГиБ VRAM');
  assert.match(text('precision'), /14\.2/u, 'precision: bf16 14.2 ГиБ');
  assert.match(text('refs'), /10\.5/u, 'refs: +10.5 с энкод');
  assert.match(text('refs'), /2\.13/u, 'refs: KV +2.13 ГиБ на 768p');
  assert.match(text('refPx'), /2\.62/u, 'refPx: 768-сетка');
  assert.match(text('refPx'), /7\.14/u, 'refPx: 1440-сетка');
  assert.match(text('width'), /5\.73/u, 'width: потолок 5.73 Мп');
  assert.match(text('width'), /12\.85/u, 'width: активации 1440p');
  assert.match(text('vramGib'), /0\.90/u, 'vramGib: гейт 0.90');
  assert.match(text('vramGib'), /0\.97/u, 'vramGib: гейт 0.97');
  assert.match(text('model'), /30\.9/u, 'model: h3 fp8');
  assert.match(text('model'), /62/u, 'model: h3 энкодер');
});

test('HELP: фиксированные и не входные параметры объяснены в note', () => {
  assert.match(HELP.steps.note ?? '', /фиксирован/u);
  assert.match(HELP.refPx.note ?? '', /Не входной|не входной/u);
  assert.match(HELP.postprocess.note ?? '', /выключен/u);
  assert.match(HELP.bitDepth.note ?? '', /16/u);
});

test('hintLive: строка дельты для входных осей', () => {
  const steps = hintLive('steps', QUERY);
  assert.equal(typeof steps, 'string');
  assert.match(steps, /шагов/u);
  assert.match(steps, /с\)/u, 'дельта времени в секундах');

  const kv = hintLive('kvCache', QUERY);
  assert.equal(typeof kv, 'string');
  assert.match(kv, /KV/u);
  assert.match(kv, /ГиБ/u);

  const vae = hintLive('vaeTilePx', QUERY);
  assert.equal(typeof vae, 'string');
  assert.match(vae, /декод/u);

  const window = hintLive('window', QUERY);
  assert.equal(typeof window, 'string', 'на 1440p окно не применяется — это объясняется текстом');

  const width = hintLive('width', QUERY);
  assert.match(width ?? '', /Q \d+ → \d+/u);

  const anime = hintLive('anime', QUERY);
  assert.match(anime ?? '', /оценки не меняются/u);
});

test('hintLive: null-safe для параметров без живой дельты и для мусора', () => {
  assert.equal(hintLive('postprocess', QUERY), null);
  assert.equal(hintLive('vramGib', QUERY), null);
  assert.equal(hintLive('refPx', QUERY), null, 'refPx не входит в запрос: его выбирает политика');
  assert.equal(hintLive('нет-такого', QUERY), null);
  assert.equal(hintLive('steps', { image: { width: 1, height: 1, quality: 'max' } }), null);
});

test('hintLive: фиксированные шаги и невозможный план', () => {
  assert.match(hintLive('steps', H3_QUERY) ?? '', /фиксирован/u);
  const impossible = { image: { width: 3840, height: 2160, quality: 'max' }, hardware: { model: 'qwen21-base', vramGib: 16, ramGib: 64, disk: 'nvme' } };
  assert.equal(hintLive('steps', impossible), null);
});

test('hintLive: кэш по ключу запроса (второй вызов — то же значение)', () => {
  const first = hintLive('steps', QUERY);
  const second = hintLive('steps', QUERY);
  assert.equal(first, second);
});

test('hintLive: работает и на запросе из состояния формы', () => {
  const request = requestFromState(DEFAULT_STATE);
  assert.match(hintLive('steps', request) ?? '', /шагов/u);
});
