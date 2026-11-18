// @ts-check
/**
 * YAML-эмиттер и парсер подмножества (шаг 6 плана, ADR A2/A10 п.6).
 *
 * Проверяется:
 * - round-trip: `parseYaml(stringifyYaml(value))` — то же значение (кроме `YamlRaw`, который
 *   существует именно для того, чтобы `1.0` не превратилось в `1`);
 * - стабильный порядок ключей, кавычки только когда нужны, `null`, inline-массивы;
 * - комментарии: шапка и комментарии к явно перечисленным путям; на разбор не влияют;
 * - парсер собственного подмножества: блочные массивы, массивы маппингов, кавычки, литералы;
 * - ошибки — `YamlError` **с номером строки** (по нему UI подсвечивает место).
 */

import assert from 'node:assert/strict';
import test from 'node:test';

import { YamlError, YamlRaw, parseYaml, stringifyYaml } from '../src/yaml.mjs';

test('round-trip: скаляры, null, вложенность, inline-массивы', () => {
  const value = {
    size_px: 1440,
    aspect: '16:9',
    empty: null,
    flag: true,
    off: false,
    zero: 0,
    negative: -1,
    float: 0.75,
    name: 'Qwen/Qwen-Image-2.1',
    trick: 'a: b # not a comment',
    apostrophe: "it's",
    nested: { a: 1, b: { c: [1, 2, 3] } },
    list: [1, 'two', null, true],
    deep: [[1, 2], [3]],
  };
  const text = stringifyYaml(value);
  assert.deepEqual(parseYaml(text), value);
  // Массивы скаляров — inline, как требует A2 (`transformer: [repo, subfolder]`).
  assert.match(text, /^list: \[1, two, null, true\]$/mu);
});

test('порядок ключей — как в объекте (детерминированный текст)', () => {
  const text = stringifyYaml({ b: 1, a: 2, c: { z: 3, y: 4 } });
  assert.equal(text, 'b: 1\na: 2\nc:\n  z: 3\n  y: 4\n');
});

test('кавычки только когда нужны', () => {
  const cases = /** @type {Array<[string, string]>} */ ([
    ['Qwen/Qwen-Image-2.1', 'value: Qwen/Qwen-Image-2.1\n'],
    ['16:9', 'value: "16:9"\n'],
    ['null', 'value: "null"\n'],
    ['true', 'value: "true"\n'],
    ['1.5', 'value: "1.5"\n'],
    ['-1', 'value: "-1"\n'],
    ['', 'value: ""\n'],
    [' x', 'value: " x"\n'],
    ['a # b', 'value: "a # b"\n'],
    ['say "hi"', 'value: "say \\"hi\\""\n'],
    ['line\nbreak', 'value: "line\\nbreak"\n'],
  ]);
  for (const [input, expected] of cases) {
    assert.equal(stringifyYaml({ value: input }), expected, JSON.stringify(input));
  }
});

test('null, boolean, числа и пустые коллекции', () => {
  assert.equal(stringifyYaml({ a: null, b: true, c: false, d: 0, e: 1.25 }), 'a: null\nb: true\nc: false\nd: 0\ne: 1.25\n');
  assert.equal(stringifyYaml({ a: [], b: {} }), 'a: []\nb: {}\n');
  assert.throws(() => stringifyYaml({ a: Number.NaN }), RangeError);
});

test('YamlRaw: `1.0` остаётся `1.0`, а не `1`', () => {
  const text = stringifyYaml({ guidance: new YamlRaw('1.0') });
  assert.equal(text, 'guidance: 1.0\n');
  assert.equal(parseYaml(text).guidance, 1);
  assert.throws(() => new YamlRaw('a\nb'), TypeError);
});

test('шапка и комментарии: печатаются, на разбор не влияют, якоря явные', () => {
  const value = { generation: { lora: null, steps: 25 }, upscale: null };
  const text = stringifyYaml(value, {
    header: ['renderkit r1+x | 16 GiB VRAM', 'вторая строка'],
    comments: { 'generation.lora': 'anime: off', upscale: 'апскейла нет' },
  });
  assert.match(text, /^# renderkit r1\+x \| 16 GiB VRAM$/mu);
  assert.match(text, /^ {2}# anime: off\n {2}lora: null$/mu);
  assert.match(text, /^# апскейла нет\nupscale: null$/mu);
  assert.deepEqual(parseYaml(text), value);
  // Комментариев, которых нет в списке, эмиттер не придумывает: 2 строки шапки + 2 явных.
  assert.equal(text.split('\n').filter((line) => line.includes('#')).length, 4);
});

test('парсер: блочные массивы, inline-вложенность и комментарии', () => {
  const text = [
    '# шапка',
    'refs:',
    '  - a',
    "  - 'b b'",
    'inline: [x, [y, z], "q, r"]',
    'scalar: 1 # хвостовой комментарий',
    'hash: "a # b"',
    'marker: ~',
    'empty:',
    '---',
    'tail: null',
    '...',
  ].join('\n');
  assert.deepEqual(parseYaml(text), {
    refs: ['a', 'b b'],
    inline: ['x', ['y', 'z'], 'q, r'],
    scalar: 1,
    hash: 'a # b',
    marker: null,
    empty: null,
    tail: null,
  });
});

test('парсер: блочный массив маппингов и смешанные скаляры', () => {
  const text = [
    'items:',
    '  - name: a',
    '    value: 1',
    '  - name: b',
    '    value: [x, y]',
    '  -',
    '    nested: true',
    'numbers: [-1, 0, 1.5, 2e3]',
    'flags: [true, false, null]',
  ].join('\n');
  assert.deepEqual(parseYaml(text), {
    items: [
      { name: 'a', value: 1 },
      { name: 'b', value: ['x', 'y'] },
      { nested: true },
    ],
    numbers: [-1, 0, 1.5, 2000],
    flags: [true, false, null],
  });
});

test('парсер: пустой документ, кавычки и escape-последовательности', () => {
  assert.equal(parseYaml(''), null);
  assert.equal(parseYaml('# только комментарий\n'), null);
  assert.deepEqual(parseYaml('a: "x\\ty\\nz"\nb: \'it\'\'s\'\nc: "\\u0041"'.replace('\\u0041', 'A')), {
    a: 'x\ty\nz',
    b: "it's",
    c: 'A',
  });
});

test('ошибки парсера: YamlError с номером строки', () => {
  const cases = /** @type {Array<[string, number, RegExp]>} */ ([
    ['a: 1\n\tb: 2', 2, /табы в отступе/u],
    ['a: 1\nb: "хвост', 2, /незакрытая строка/u],
    ['a: 1\n  b: 2', 2, /неожиданный отступ/u],
    ['a: {x: 1}', 1, /inline-маппинги/u],
    ['a: 1\nb: &anchor 1', 2, /якоря\/алиасы/u],
    ['a: 1\na: 2', 2, /дублирующийся ключ/u],
    ['просто текст', 1, /ожидался ключ/u],
    ['a: 1\nb: [1, 2', 2, /незакрытый массив/u],
    ['a: 1\nb:\n  c: 2\n d: 3', 4, /неожиданный отступ/u],
    ['a: "x" tail', 1, /лишние символы после строки/u],
    ['a: 1\nb: |\n  text', 2, /блочные скаляры/u],
  ]);
  for (const [text, line, pattern] of cases) {
    assert.throws(
      () => parseYaml(text),
      (/** @type {any} */ error) => {
        assert.ok(error instanceof YamlError, `${JSON.stringify(text)}: ожидался YamlError, получено ${error}`);
        assert.equal(error.line, line, `${JSON.stringify(text)}: номер строки`);
        assert.match(error.message, pattern);
        return true;
      },
      JSON.stringify(text),
    );
  }
});
