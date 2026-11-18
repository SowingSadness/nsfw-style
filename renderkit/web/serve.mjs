#!/usr/bin/env node
/**
 * Статический сервер прототипа: `node web/serve.mjs [port]` (или `npm start`).
 *
 * Зачем свой: пакет без зависимостей (ADR A11), а браузеру нужен правильный MIME для `.mjs` —
 * иначе модуль не исполнится. Больше сервер ничего не умеет и не должен: API появится отдельным
 * этапом (`POST /api/plan`), ядро уже изоморфно.
 *
 * Правила:
 *
 * - путь берётся **сырым** из запроса (`request.url`), а не из `new URL(...)`: WHATWG-парсер
 *   нормализует `%2e%2e` в dot-segment ещё до нашей проверки, и обход корня выглядел бы как
 *   обычный путь (`/%2e%2e/src/index.mjs` → `/src/index.mjs`);
 * - после `decodeURIComponent` путь нормализуется через `node:path` и склеивается `path.resolve`;
 *   файл отдаётся, только если итог начинается с `<корень> + sep` — иначе **403** `text/plain`;
 *   сегменты `..`, обратный слэш и выход за корень — 403; NUL и невалидное кодирование — 400;
 * - каталог → `index.html` внутри него, всё остальное (нет файла, нет `index.html`) — 404;
 * - `Cache-Control: no-store`: прототип правится на ходу, кэш браузера только мешает;
 * - GET и HEAD; порт — `process.argv[2] || 8123` (строка порта в окружении — только для тестов/CI).
 *
 * Экспорты (`createServer`, `start`, `resolvePath`, `parsePort`) существуют ради
 * `test/serve.test.mjs`: тест поднимает сервер на `listen(0)` и проверяет границы корня.
 */

import { createServer as createHttpServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

/**
 * Две точки монтирования, обе с одинаковой строгой проверкой пути:
 *
 * - `/` → `web/` — сама страница (`index.html`, `app.mjs`, `styles.css`); выйти за него нельзя;
 * - `/src/` → `src/` пакета — **изоморфное ядро**: `app.mjs` делает `import '../src/index.mjs'`, и
 *   без этой точки браузер получил бы 404. Это не «раздача всего диска», а один узкий read-only
 *   префикс: всё остальное (включая `..`) остаётся за корнем.
 */
const ROOT = resolve(fileURLToPath(new URL('.', import.meta.url)));
/** Корень ядра (`renderkit/src/`), смонтированный как `/src/`. */
const PKG_SRC = resolve(fileURLToPath(new URL('../src/', import.meta.url)));

/** Порт по умолчанию — 8123 (как в плане и в `npm start`). */
const DEFAULT_PORT = 8123;

/** MIME по расширению: `.mjs` обязан быть `text/javascript`, иначе браузер откажется. */
const MIME = Object.freeze({
  '.html': 'text/html; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.map': 'application/json; charset=utf-8',
});

/**
 * Отказ: код + текст для `text/plain`.
 *
 * @param {number} code
 * @param {string} message
 * @returns {{ok: false, code: number, message: string}}
 */
function deny(code, message) {
  return { ok: false, code, message };
}

/**
 * Разобрать аргумент порта (`0` — «любой свободный», нужен тестам).
 *
 * @param {string|number|undefined} raw
 * @returns {number}
 */
export function parsePort(raw) {
  if (raw === undefined || raw === '') return DEFAULT_PORT;
  const port = Number(raw);
  if (!Number.isInteger(port) || port < 0 || port > 65535) {
    throw new RangeError(`порт должен быть целым 0…65535, получено ${JSON.stringify(raw)}`);
  }
  return port;
}

/**
 * Безопасно превратить URL-путь в абсолютный путь: `/src/...` → ядро пакета, всё остальное → `web/`.
 *
 * Проверка идёт по **сырому** пути (query/fragment отрезаются, но URL-нормализации нет), поэтому
 * `%2e%2e` виден как `..` и отбивается, а не схлопывается в `/src/...`.
 *
 * @param {string} urlPath путь из URL (сырой `request.url` или `location.pathname`)
 * @param {string} [root] корень раздачи (по умолчанию каталог `web/`)
 * @returns {{ok: true, file: string}|{ok: false, code: number, message: string}}
 */
export function resolvePath(urlPath, root = ROOT) {
  const base = resolve(root);
  const raw = String(urlPath ?? '/').split('?')[0].split('#')[0];
  /** @type {string} */
  let decoded;
  try {
    decoded = decodeURIComponent(raw);
  } catch {
    return deny(400, 'битый URL');
  }
  if (decoded.indexOf(String.fromCharCode(0)) >= 0) {
    return deny(400, 'битый URL');
  }
  const parts = decoded.split('/');
  // `..` проверяется до normalize/resolve: иначе схлопывание у корня выдало бы обход за обычный путь.
  if (parts.indexOf('..') >= 0) {
    return deny(403, 'путь вне корня не отдаётся');
  }
  // Обратный слэш в URL-пути — либо ошибка клиента, либо попытка обойти границу на Windows.
  if (decoded.indexOf(String.fromCharCode(92)) >= 0) {
    return deny(403, 'путь вне корня не отдаётся');
  }
  const segments = parts.filter((part) => part !== '' && part !== '.');
  const isSrcMount = segments[0] === 'src' && segments.length > 1;
  const mount = isSrcMount ? PKG_SRC : base;
  const rest = isSrcMount ? segments.slice(1) : segments;
  const file = resolve(mount, normalize(rest.join('/')));
  if (file !== mount && !file.startsWith(mount + sep)) {
    return deny(403, 'путь вне корня не отдаётся');
  }
  return { ok: true, file };
}

/**
 * Файл по пути или `index.html`, если это каталог.
 *
 * @param {string} file абсолютный путь внутри уже проверенного корня
 * @returns {Promise<string>}
 */
async function fileOrIndex(file) {
  const info = await stat(file);
  if (info.isDirectory()) {
    const index = join(file, 'index.html');
    const nested = await stat(index);
    if (!nested.isFile()) throw new Error('нет index.html');
    return index;
  }
  if (!info.isFile()) throw new Error('не файл');
  return file;
}

/**
 * Обработать один запрос (GET/HEAD, раздача одного файла из `root`).
 *
 * @param {import('node:http').IncomingMessage} request
 * @param {import('node:http').ServerResponse} response
 * @param {string} root
 * @returns {Promise<void>}
 */
async function handle(request, response, root) {
  const method = request.method ?? 'GET';
  if (method !== 'GET' && method !== 'HEAD') {
    response.writeHead(405, { Allow: 'GET, HEAD', 'Content-Type': 'text/plain; charset=utf-8' });
    response.end('только GET и HEAD\n');
    return;
  }
  const resolved = resolvePath(request.url ?? '/', root);
  if (resolved.ok === false) {
    response.writeHead(resolved.code, {
      'Content-Type': 'text/plain; charset=utf-8',
      'Cache-Control': 'no-store',
    });
    response.end(`${resolved.message}\n`);
    return;
  }
  try {
    const file = await fileOrIndex(resolved.file);
    const body = await readFile(file);
    response.writeHead(200, {
      'Content-Type': MIME[extname(file).toLowerCase()] ?? 'application/octet-stream',
      'Content-Length': String(body.byteLength),
      'Cache-Control': 'no-store',
      'X-Content-Type-Options': 'nosniff',
    });
    response.end(method === 'HEAD' ? undefined : body);
  } catch {
    response.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8', 'Cache-Control': 'no-store' });
    response.end(`404: ${request.url ?? '/'} не найден\n`);
  }
}

/**
 * Создать сервер (без `listen` — тесту нужен `port: 0`).
 *
 * @param {string} [root] каталог, отдаваемый с `/` (по умолчанию `web/`)
 * @returns {import('node:http').Server}
 */
export function createServer(root = ROOT) {
  const webRoot = resolve(root);
  return createHttpServer((request, response) => {
    void handle(request, response, webRoot);
  });
}

/**
 * Поднять сервер и дождаться `listening`.
 *
 * @param {string|number} [port] `0` — любой свободный порт
 * @param {string} [root]
 * @returns {Promise<import('node:http').Server>}
 */
export async function start(port = DEFAULT_PORT, root = ROOT) {
  const server = createServer(root);
  const actualPort = parsePort(port);
  await new Promise((listening, failed) => {
    const onError = (error) => failed(error);
    server.once('error', onError);
    server.listen(actualPort, '127.0.0.1', () => {
      server.off('error', onError);
      listening(undefined);
    });
  });
  return server;
}

/** Прежнее имя фабрики — оставлено для совместимости. */
export function createStaticServer() {
  return createServer(ROOT);
}

/** Запуск из командной строки (`npm start`). */
async function main() {
  const server = await start(process.argv[2]);
  const address = server.address();
  const actual = typeof address === 'object' && address !== null ? address.port : DEFAULT_PORT;
  process.stdout.write(`renderkit web: http://127.0.0.1:${actual}/\n`);
  const stop = () => {
    server.close(() => process.exit(0));
  };
  process.on('SIGINT', stop);
  process.on('SIGTERM', stop);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
    process.exit(1);
  });
}
