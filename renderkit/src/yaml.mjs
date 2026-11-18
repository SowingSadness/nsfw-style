// @ts-check
/**
 * YAML для `renderkit`: эмиттер (стабильный порядок, комментарии из явного списка) и парсер
 * **собственного подмножества** (для UI-«сканирования» вставленного YAML, A10 п.6).
 *
 * Без зависимостей и без `node:`: модуль изоморфный (браузер, Node, сервис).
 *
 * Поддерживается ровно то, что печатает эмиттер, плюс то, что люди пишут руками в таких же
 * файлах:
 *
 * - вложенные маппинги (блочно, отступ 2 пробела);
 * - скаляры: `null`/`~`, `true`/`false`, целые и дробные числа, строки в кавычках и без;
 * - массивы inline (`[a, b]`) и блочные (`- a`);
 * - блочные массивы маппингов (`- name: a` + продолжение с отступом);
 * - комментарии (`# …` до конца строки, внутри кавычек — часть строки);
 * - маркеры документа `---`/`...` (пропускаются).
 *
 * Не поддерживается осознанно (это не наш диалект) и даёт `YamlError` с номером строки:
 * табы в отступе, якоря/алиасы (`&`/`*`), теги (`!`), блочные скаляры (`|`/`>`), inline-маппинги
 * (`{a: 1}`), мультидокументность. Явная ошибка лучше молчаливо неверного резолва: авторитетная
 * проверка всё равно остаётся за `python -m qwenstyle config show <файл> --json`.
 */

/**
 * Ошибка разбора YAML: сообщение + 1-based номер строки (по нему UI подсвечивает место).
 */
export class YamlError extends Error {
  /**
   * @param {string} message что именно не так
   * @param {number} line 1-based номер строки исходного текста
   */
  constructor(message, line) {
    super(`строка ${line}: ${message}`);
    this.name = 'YamlError';
    this.line = line;
    this.detail = message;
  }
}

/**
 * Скаляр, который печатается **как есть**.
 *
 * Нужен там, где JS не различает `1` и `1.0` (у числа один тип), а YAML различает: например
 * `generation.guidance` в конфиге пайплайна объявлено `float`, и в файле честнее `1.0`.
 *
 * @example stringifyYaml({guidance: new YamlRaw('1.0')}) // 'guidance: 1.0\n'
 */
export class YamlRaw {
  /** @param {string} text готовый YAML-скаляр (без переводов строк) */
  constructor(text) {
    const value = String(text);
    if (/[\n\r]/u.test(value)) throw new TypeError('YamlRaw: скаляр не может содержать перевод строки');
    this.text = value;
  }

  toString() {
    return this.text;
  }
}

/** Отступ вложенности — 2 пробела (так печатает эмиттер, так ждёт парсер). */
const INDENT = 2;

/**
 * Строка требует кавычек, если начинается/кончается пробелом, пуста, содержит символы со
 * специальным смыслом в YAML или совпадает с литералом (`null`, `true`, число).
 */
const PLAIN_FORBIDDEN = /[:[\]{},#&*!|>'"%@`\n\r\t]/u;
/** Литералы, которые YAML прочитает не как строку, — их надо закавычить. */
const LITERALS = /^(?:~|null|true|false|yes|no|on|off)$/iu;
/** Число (включая экспоненту) — тоже литерал, а не строка. */
const NUMBER_LIKE = /^[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?$/u;

/**
 * Нужны ли кавычки строковому скаляру.
 *
 * @param {string} text
 * @returns {boolean}
 */
function needsQuotes(text) {
  if (text === '') return true;
  if (text !== text.trim()) return true;
  if (PLAIN_FORBIDDEN.test(text)) return true;
  if (LITERALS.test(text) || NUMBER_LIKE.test(text)) return true;
  // Начало с `-`/`?`/`:` в начале строки — признак блока/ключа, а не скаляра.
  if (/^[-?:]\s|^[-?:]$/u.test(text)) return true;
  return false;
}

/**
 * Закавычить строку в двойные кавычки с экранированием (единственный используемый стиль —
 * он читается однозначно и не требует remember-правил как одинарные кавычки).
 *
 * @param {string} text
 * @returns {string}
 */
function quote(text) {
  const escaped = text
    .replace(/\\/gu, '\\\\')
    .replace(/"/gu, '\\"')
    .replace(/\t/gu, '\\t')
    .replace(/\r/gu, '\\r')
    .replace(/\n/gu, '\\n');
  return `"${escaped}"`;
}

/**
 * Значение-скаляр в YAML-представлении.
 *
 * @param {unknown} value
 * @returns {string}
 */
function scalarToYaml(value) {
  if (value === null || value === undefined) return 'null';
  if (value instanceof YamlRaw) return value.text;
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new RangeError(`stringifyYaml: недопустимое число ${String(value)} (NaN/Infinity не пишутся)`);
    }
    return String(value);
  }
  if (typeof value === 'string') return needsQuotes(value) ? quote(value) : value;
  throw new TypeError(`stringifyYaml: неподдерживаемый скаляр ${typeof value}`);
}

/** @param {unknown} value */
function isScalar(value) {
  return value === null || value === undefined
    || value instanceof YamlRaw
    || typeof value === 'string'
    || typeof value === 'number'
    || typeof value === 'boolean';
}

/**
 * Массив печатается inline (`[a, b]`), если все его элементы — скаляры; иначе блочно `- …`.
 *
 * @param {unknown[]} items
 * @returns {string}
 */
function inlineArrayToYaml(items) {
  return `[${items.map((item) => scalarToYaml(item)).join(', ')}]`;
}

/**
 * Inline-форма значения (`[a, b]`, `{}`, `[]`) или `null`, если значение печатается блоком.
 *
 * @param {unknown} value
 * @returns {string|null}
 */
function inlineCollectionToYaml(value) {
  if (Array.isArray(value)) {
    if (value.length === 0) return '[]';
    return value.every(isScalar) ? inlineArrayToYaml(value) : null;
  }
  if (typeof value === 'object' && value !== null) {
    return Object.keys(value).length === 0 ? '{}' : null;
  }
  return null;
}

/**
 * `stringifyYaml(value, options)` — YAML-текст с **порядком ключей как в объекте**.
 *
 * @param {unknown} value корневое значение (обычно маппинг)
 * @param {{
 *   header?: string | string[] | null,
 *   comments?: Record<string, string> | null
 * }} [options]
 *   `header` — строки шапки (`# …`), `comments` — комментарии, привязанные к путям
 *   (`'generation.lora'`), печатаются **над** ключом. Никаких других комментариев эмиттер
 *   не расставляет: список явный, поэтому текст файла детерминирован.
 * @returns {string} текст с завершающим переводом строки
 */
export function stringifyYaml(value, options = {}) {
  /** @type {string[]} */
  const lines = [];
  const header = normalizeHeader(options.header);
  const comments = options.comments ?? {};
  for (const line of header) {
    for (const part of String(line).split('\n')) lines.push(`# ${part}`.trimEnd());
  }
  writeValue(value, 0, '', lines, comments);
  return `${lines.join('\n')}\n`;
}

/**
 * @param {string | string[] | null | undefined} header
 * @returns {string[]}
 */
function normalizeHeader(header) {
  if (header === null || header === undefined) return [];
  return Array.isArray(header) ? header.map(String) : [String(header)];
}

/**
 * Записать значение в строки (маппинг/массив — блочно, скаляры — на месте).
 *
 * @param {unknown} value
 * @param {number} depth
 * @param {string} path путь до значения (для комментариев)
 * @param {string[]} lines
 * @param {Record<string, string>} comments
 */
function writeValue(value, depth, path, lines, comments) {
  const pad = ' '.repeat(depth * INDENT);
  if (isScalar(value)) {
    lines.push(`${pad}${scalarToYaml(value)}`);
    return;
  }
  if (Array.isArray(value)) {
    const inline = inlineCollectionToYaml(value);
    if (inline !== null) {
      lines.push(`${pad}${inline}`);
      return;
    }
    for (const item of value) {
      if (isScalar(item) || Array.isArray(item)) {
        const childPad = ' '.repeat((depth + 1) * INDENT);
        const before = lines.length;
        writeValue(item, depth + 1, path, lines, comments);
        lines[before] = `${pad}- ${lines[before].slice(childPad.length)}`;
      } else {
        lines.push(`${pad}-`);
        writeValue(item, depth + 1, path, lines, comments);
      }
    }
    return;
  }
  if (typeof value === 'object') {
    const entries = Object.entries(/** @type {Record<string, unknown>} */ (value));
    if (entries.length === 0) {
      lines.push(`${pad}{}`);
      return;
    }
    for (const [key, item] of entries) {
      const childPath = path ? `${path}.${key}` : key;
      const comment = comments[childPath];
      if (comment) {
        for (const part of String(comment).split('\n')) lines.push(`${pad}# ${part}`.trimEnd());
      }
      const inline = isScalar(item) ? scalarToYaml(item) : inlineCollectionToYaml(item);
      if (inline !== null) {
        lines.push(`${pad}${key}: ${inline}`);
      } else {
        lines.push(`${pad}${key}:`);
        writeValue(item, depth + 1, childPath, lines, comments);
      }
    }
    return;
  }
  throw new TypeError(`stringifyYaml: неподдерживаемое значение ${typeof value}`);
}

// --------------------------------------------------------------------------- //
// Парсер подмножества
// --------------------------------------------------------------------------- //

/**
 * @typedef {Object} Line
 * @property {number} indent отступ в пробелах
 * @property {string} text текст без отступа (и без комментария)
 * @property {number} line 1-based номер строки в исходном тексте
 */

/**
 * `parseYaml(text)` — разобрать подмножество (см. шапку модуля).
 *
 * @param {string} text
 * @returns {unknown} значение (маппинг/массив/скаляр); пустой текст — `null`
 * @throws {YamlError} неподдерживаемый синтаксис — с номером строки
 */
export function parseYaml(text) {
  if (typeof text !== 'string') throw new TypeError('parseYaml: ожидается строка');
  const lines = preprocess(text);
  if (lines.length === 0) return null;
  const { value, consumed } = parseBlock(lines, lines[0].indent);
  if (consumed < lines.length) {
    const extra = lines[consumed];
    throw new YamlError('лишние данные: отступ не совпадает с началом документа', extra.line);
  }
  return value;
}

/**
 * Разбить текст на значимые строки: отступ, текст без комментария, номер строки.
 *
 * @param {string} text
 * @returns {Line[]}
 */
function preprocess(text) {
  /** @type {Line[]} */
  const lines = [];
  const raw = text.replace(/^\uFEFF/u, '').split(/\r?\n/u);
  for (let index = 0; index < raw.length; index += 1) {
    const lineNo = index + 1;
    const source = raw[index];
    if (/^\s*$/u.test(source)) continue;
    const withoutComment = stripComment(source, lineNo);
    if (/^\s*$/u.test(withoutComment)) continue;
    const marker = withoutComment.trim();
    if (marker === '---' || marker === '...') continue;
    const indentMatch = /^[ \t]*/u.exec(withoutComment);
    const indentText = indentMatch ? indentMatch[0] : '';
    if (indentText.includes('\t')) {
      throw new YamlError('табы в отступе не поддерживаются (только пробелы)', lineNo);
    }
    const content = withoutComment.slice(indentText.length).replace(/\s+$/u, '');
    if (content === '') continue;
    lines.push({ indent: indentText.length, text: content, line: lineNo });
  }
  return lines;
}

/**
 * Отрезать комментарий: `#` вне кавычек и в начале слова/строки.
 *
 * @param {string} source
 * @param {number} lineNo
 * @returns {string}
 */
function stripComment(source, lineNo) {
  let quoteChar = '';
  for (let index = 0; index < source.length; index += 1) {
    const char = source[index];
    if (quoteChar) {
      if (quoteChar === '"' && char === '\\') {
        index += 1;
        continue;
      }
      if (char === quoteChar) {
        if (quoteChar === "'" && source[index + 1] === "'") {
          index += 1;
          continue;
        }
        quoteChar = '';
      }
      continue;
    }
    if (char === '"' || char === "'") {
      quoteChar = char;
      continue;
    }
    if (char === '#' && (index === 0 || /\s/u.test(source[index - 1]))) {
      return source.slice(0, index);
    }
  }
  if (quoteChar) {
    throw new YamlError(`незакрытая строка (${quoteChar})`, lineNo);
  }
  return source;
}

/**
 * Разобрать блок с заданным отступом; вернуть значение и число съеденных строк.
 *
 * @param {Line[]} lines
 * @param {number} indent
 * @returns {{value: unknown, consumed: number}}
 */
function parseBlock(lines, indent) {
  const first = lines[0];
  if (!first) return { value: null, consumed: 0 };
  if (first.indent !== indent) {
    throw new YamlError(`ожидался отступ ${indent}, получено ${first.indent}`, first.line);
  }
  if (isSequenceLine(first.text)) return parseSequence(lines, indent);
  return parseMapping(lines, indent);
}

/**
 * @param {string} text
 * @returns {boolean}
 */
function isSequenceLine(text) {
  return text === '-' || text.startsWith('- ');
}

/**
 * Блочный маппинг: `key: value` или `key:` + вложенный блок.
 *
 * @param {Line[]} lines
 * @param {number} indent
 * @returns {{value: Record<string, unknown>, consumed: number}}
 */
function parseMapping(lines, indent) {
  /** @type {Record<string, unknown>} */
  const result = {};
  let index = 0;
  while (index < lines.length) {
    const current = lines[index];
    if (current.indent < indent) break;
    if (current.indent > indent) {
      throw new YamlError(`неожиданный отступ ${current.indent} (ожидался ${indent})`, current.line);
    }
    if (isSequenceLine(current.text)) break;
    const split = splitEntry(current.text, current.line);
    if (Object.hasOwn(result, split.key)) {
      throw new YamlError(`дублирующийся ключ ${JSON.stringify(split.key)}`, current.line);
    }
    index += 1;
    if (split.rest !== '') {
      result[split.key] = parseInlineValue(split.rest, current.line);
      continue;
    }
    const next = lines[index];
    if (next && next.indent > indent) {
      const nested = parseBlock(lines.slice(index), next.indent);
      result[split.key] = nested.value;
      index += nested.consumed;
    } else if (next && next.indent === indent && isSequenceLine(next.text)) {
      const nested = parseSequence(lines.slice(index), indent);
      result[split.key] = nested.value;
      index += nested.consumed;
    } else {
      result[split.key] = null;
    }
  }
  return { value: result, consumed: index };
}

/**
 * Блочный массив: `- scalar`, `- key: value` (+ продолжение с отступом) или `-` + вложенный блок.
 *
 * @param {Line[]} lines
 * @param {number} indent
 * @returns {{value: unknown[], consumed: number}}
 */
function parseSequence(lines, indent) {
  /** @type {unknown[]} */
  const result = [];
  let index = 0;
  while (index < lines.length) {
    const current = lines[index];
    if (current.indent !== indent || !isSequenceLine(current.text)) {
      if (current.indent > indent) {
        throw new YamlError(`неожиданный отступ ${current.indent} в массиве`, current.line);
      }
      break;
    }
    const restAll = current.text === '-' ? '' : current.text.slice(1);
    const rest = restAll.trim();
    index += 1;
    if (rest === '') {
      const next = lines[index];
      if (next && next.indent > indent) {
        const nested = parseBlock(lines.slice(index), next.indent);
        result.push(nested.value);
        index += nested.consumed;
      } else {
        result.push(null);
      }
      continue;
    }
    if (looksLikeMapping(rest)) {
      // `- key: value` — компактная запись маппинга: содержимое начинается с колонки после дефиса.
      const itemIndent = indent + (current.text.length - restAll.trimStart().length);
      const itemLines = [{ indent: itemIndent, text: rest, line: current.line }, ...lines.slice(index)];
      const nested = parseBlock(itemLines, itemIndent);
      if (!nested.value || typeof nested.value !== 'object' || Array.isArray(nested.value)) {
        throw new YamlError('элемент массива с `key: value` должен быть маппингом', current.line);
      }
      result.push(nested.value);
      index += nested.consumed - 1;
      continue;
    }
    result.push(parseInlineValue(rest, current.line));
    const next = lines[index];
    if (next && next.indent > indent) {
      throw new YamlError('многострочные скаляры не поддерживаются', next.line);
    }
  }
  return { value: result, consumed: index };
}

/**
 * Разбить строку `key: value` (/`key:`) на ключ и остаток.
 *
 * @param {string} text
 * @param {number} lineNo
 * @returns {{key: string, rest: string}}
 */
function splitEntry(text, lineNo) {
  const split = findEntrySeparator(text);
  if (split === -1) {
    throw new YamlError(`ожидался ключ вида "имя: значение", получено ${JSON.stringify(text)}`, lineNo);
  }
  const rawKey = text.slice(0, split).trim();
  if (rawKey === '') throw new YamlError('пустой ключ', lineNo);
  const key = rawKey.startsWith('"') || rawKey.startsWith("'")
    ? parseQuoted(rawKey, lineNo)
    : rawKey;
  return { key: String(key), rest: text.slice(split + 1).trim() };
}

/**
 * Позиция `:`-разделителя ключа (вне кавычек; `:` должно быть в конце или перед пробелом).
 *
 * @param {string} text
 * @returns {number} индекс двоеточия или `-1`
 */
function findEntrySeparator(text) {
  let quoteChar = '';
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoteChar) {
      if (quoteChar === '"' && char === '\\') index += 1;
      else if (char === quoteChar) quoteChar = '';
      continue;
    }
    if (char === '"' || char === "'") {
      quoteChar = char;
      continue;
    }
    if (char === ':' && (index === text.length - 1 || text[index + 1] === ' ')) return index;
  }
  return -1;
}

/**
 * Похоже ли на компактный маппинг (для `- key: value`).
 *
 * @param {string} text
 * @returns {boolean}
 */
function looksLikeMapping(text) {
  if (text.startsWith('[') || text.startsWith('"') || text.startsWith("'")) return false;
  return findEntrySeparator(text) !== -1;
}

/**
 * Значение из строки: скаляр, inline-массив или — как ошибка — неподдерживаемая форма.
 *
 * @param {string} text
 * @param {number} lineNo
 * @returns {unknown}
 */
function parseInlineValue(text, lineNo) {
  const trimmed = text.trim();
  if (trimmed.startsWith('{')) {
    throw new YamlError('inline-маппинги `{…}` не поддерживаются', lineNo);
  }
  if (trimmed.startsWith('&') || trimmed.startsWith('*')) {
    throw new YamlError('якоря/алиасы (`&`/`*`) не поддерживаются', lineNo);
  }
  if (trimmed.startsWith('!')) {
    throw new YamlError('теги (`!`) не поддерживаются', lineNo);
  }
  if (trimmed.startsWith('|') || trimmed.startsWith('>')) {
    throw new YamlError('блочные скаляры (`|`/`>`) не поддерживаются', lineNo);
  }
  if (trimmed.startsWith('[')) {
    const parsed = parseFlow(trimmed, 0, lineNo);
    if (parsed.next !== trimmed.length) {
      throw new YamlError(`лишние символы после массива: ${JSON.stringify(trimmed.slice(parsed.next))}`, lineNo);
    }
    return parsed.value;
  }
  return parseScalar(trimmed, lineNo);
}

/**
 * Рекурсивный разбор inline-значения (массив или скаляр) с позиции `start`.
 *
 * @param {string} text
 * @param {number} start
 * @param {number} lineNo
 * @returns {{value: unknown, next: number}}
 */
function parseFlow(text, start, lineNo) {
  if (text[start] === '[') {
    /** @type {unknown[]} */
    const items = [];
    let index = start + 1;
    let expectValue = true;
    while (index < text.length) {
      const char = text[index];
      if (/\s/u.test(char)) {
        index += 1;
        continue;
      }
      if (char === ']') {
        if (expectValue && items.length > 0) throw new YamlError('висячая запятая в массиве', lineNo);
        return { value: items, next: index + 1 };
      }
      if (char === ',') {
        if (expectValue) throw new YamlError('пустой элемент массива', lineNo);
        expectValue = true;
        index += 1;
        continue;
      }
      if (!expectValue) throw new YamlError('ожидалась `,` или `]` в массиве', lineNo);
      if (char === '[') {
        const nested = parseFlow(text, index, lineNo);
        items.push(nested.value);
        index = nested.next;
      } else if (char === '"' || char === "'") {
        const quoted = scanQuoted(text, index, lineNo);
        items.push(quoted.value);
        index = quoted.next;
      } else {
        let end = index;
        while (end < text.length && text[end] !== ',' && text[end] !== ']') end += 1;
        items.push(parseScalar(text.slice(index, end).trim(), lineNo));
        index = end;
      }
      expectValue = false;
    }
    throw new YamlError('незакрытый массив `[`', lineNo);
  }
  if (text[start] === '"' || text[start] === "'") return scanQuoted(text, start, lineNo);
  let end = start;
  while (end < text.length && text[end] !== ',' && text[end] !== ']') end += 1;
  return { value: parseScalar(text.slice(start, end).trim(), lineNo), next: end };
}

/**
 * Прочитать скаляр в кавычках с позиции `start`.
 *
 * @param {string} text
 * @param {number} start
 * @param {number} lineNo
 * @returns {{value: string, next: number}}
 */
function scanQuoted(text, start, lineNo) {
  const quoteChar = text[start];
  let index = start + 1;
  let result = '';
  while (index < text.length) {
    const char = text[index];
    if (quoteChar === '"' && char === '\\') {
      const escaped = text[index + 1];
      result += unescapeChar(escaped, lineNo);
      index += 2;
      continue;
    }
    if (char === quoteChar) {
      if (quoteChar === "'" && text[index + 1] === "'") {
        result += "'";
        index += 2;
        continue;
      }
      return { value: result, next: index + 1 };
    }
    result += char;
    index += 1;
  }
  throw new YamlError(`незакрытая строка (${quoteChar})`, lineNo);
}

/**
 * @param {string} char
 * @param {number} lineNo
 * @returns {string}
 */
function unescapeChar(char, lineNo) {
  switch (char) {
    case 'n': return '\n';
    case 't': return '\t';
    case 'r': return '\r';
    case '"': return '"';
    case '\\': return '\\';
    case '0': return '\0';
    default:
      throw new YamlError(`неизвестная escape-последовательность "\\${String(char)}"`, lineNo);
  }
}

/**
 * Строка целиком в кавычках (`"…"`/`'…'`) — с проверкой, что после закрывающей кавычки пусто.
 *
 * @param {string} text
 * @param {number} lineNo
 * @returns {string}
 */
function parseQuoted(text, lineNo) {
  const scanned = scanQuoted(text, 0, lineNo);
  if (scanned.next !== text.length) {
    throw new YamlError(`лишние символы после строки: ${JSON.stringify(text.slice(scanned.next))}`, lineNo);
  }
  return scanned.value;
}

/**
 * Скаляр без кавычек: литералы `null`/`true`/`false`, числа, иначе строка.
 *
 * @param {string} text
 * @param {number} lineNo
 * @returns {unknown}
 */
function parseScalar(text, lineNo) {
  if (text === '') return null;
  if (text.startsWith('"') || text.startsWith("'")) return parseQuoted(text, lineNo);
  if (text.startsWith('[')) {
    const parsed = parseFlow(text, 0, lineNo);
    if (parsed.next !== text.length) {
      throw new YamlError(`лишние символы после массива: ${JSON.stringify(text.slice(parsed.next))}`, lineNo);
    }
    return parsed.value;
  }
  if (text === '~' || /^null$/iu.test(text)) return null;
  if (/^true$/iu.test(text)) return true;
  if (/^false$/iu.test(text)) return false;
  if (/^[-+]?\d+$/u.test(text)) return Number(text);
  if (NUMBER_LIKE.test(text)) return Number(text);
  return text;
}
