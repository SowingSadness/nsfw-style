// @ts-check
/**
 * Границы статического сервера (`web/serve.mjs`): что отдаётся, что отбивается.
 *
 * Запросы идут через сырой `node:http`, а не `fetch`: undici (как и `new URL`) нормализует `%2e%2e`
 * в dot-segment ещё до отправки, и обход корня превратился бы в безобидный `/src/index.mjs` —
 * именно на этой нормализации и держался прежний дефект (200 вместо 403).
 */

import { after, before, test } from 'node:test';
import assert from 'node:assert/strict';
import { request } from 'node:http';

import { start } from '../web/serve.mjs';

/** @type {import('node:http').Server|undefined} */
let server;
/** @type {number} */
let port = 0;

before(async () => {
  server = await start(0);
  const address = server.address();
  assert.ok(address !== null && typeof address === 'object', 'сервер должен слушать TCP-порт');
  port = address.port;
});

after(async () => {
  if (!server) return;
  await new Promise((done) => server.close(() => done(undefined)));
});

/**
 * Сырой GET/HEAD: путь отправляется ровно как задан, без клиентской нормализации.
 *
 * @param {string} path
 * @param {'GET'|'HEAD'} [method]
 * @returns {Promise<{status: number, contentType: string, body: string}>}
 */
function get(path, method = 'GET') {
  return new Promise((done, failed) => {
    const req = request({ host: '127.0.0.1', port, path, method }, (res) => {
      /** @type {Buffer[]} */
      const chunks = [];
      res.on('data', (chunk) => chunks.push(/** @type {Buffer} */ (chunk)));
      res.on('end', () =>
        done({
          status: res.statusCode ?? 0,
          contentType: String(res.headers['content-type'] ?? ''),
          body: Buffer.concat(chunks).toString('utf8'),
        }),
      );
    });
    req.on('error', failed);
    req.end();
  });
}

test('GET / — index.html c text/html', async () => {
  const res = await get('/');
  assert.equal(res.status, 200);
  assert.match(res.contentType, /^text\/html/u);
  assert.match(res.body, /<script type="module" src="app\.mjs">/u);
});

test('GET /app.mjs — text/javascript (иначе браузер не исполнит модуль)', async () => {
  const res = await get('/app.mjs');
  assert.equal(res.status, 200);
  assert.match(res.contentType, /^text\/javascript/u);
  assert.match(res.body, /import \{ computePlan/u);
});

test('GET /src/index.mjs — ядро отдаётся своим маунтом', async () => {
  const res = await get('/src/index.mjs');
  assert.equal(res.status, 200);
  assert.match(res.contentType, /^text\/javascript/u);
  assert.match(res.body, /export function computePlan/u);
});

test('GET /%2e%2e/src/index.mjs — 403: dot-segment не должен выходить за корень', async () => {
  const res = await get('/%2e%2e/src/index.mjs');
  assert.equal(res.status, 403);
  assert.match(res.contentType, /^text\/plain/u);
  assert.match(res.body, /вне корня/u);
});

test('GET /../package.json — 403 или 404, но не 200', async () => {
  const res = await get('/../package.json');
  assert.ok(res.status === 403 || res.status === 404, `ожидалось 403/404, получено ${res.status}`);
  assert.equal(res.body.includes('"name": "qwenstyle-renderkit"'), false, 'package.json отдан');
});

test('GET /%2e%2e/package.json — 403 (выход к package.json пакета)', async () => {
  const res = await get('/%2e%2e/package.json');
  assert.equal(res.status, 403);
  assert.match(res.contentType, /^text\/plain/u);
});

test('GET /package.json — 404: файл лежит вне web/', async () => {
  const res = await get('/package.json');
  assert.equal(res.status, 404);
  assert.equal(res.body.includes('"name": "qwenstyle-renderkit"'), false);
});

test('GET /nope.txt — 404 текстом', async () => {
  const res = await get('/nope.txt');
  assert.equal(res.status, 404);
  assert.match(res.contentType, /^text\/plain/u);
});

test('GET /assets/ — 404: каталога нет (директория → index.html, но index.html нет)', async () => {
  const res = await get('/assets/');
  assert.equal(res.status, 404);
});

test('GET /%zz — 400: невалидное процентное кодирование', async () => {
  const res = await get('/%zz');
  assert.equal(res.status, 400);
  assert.match(res.contentType, /^text\/plain/u);
  assert.match(res.body, /битый URL/u);
});

test('GET /%00 — 400: NUL в пути', async () => {
  const res = await get('/%00');
  assert.equal(res.status, 400);
  assert.match(res.body, /битый URL/u);
});

test('HEAD /app.mjs — 200 без тела', async () => {
  const res = await get('/app.mjs', 'HEAD');
  assert.equal(res.status, 200);
  assert.match(res.contentType, /^text\/javascript/u);
  assert.equal(res.body, '');
});
