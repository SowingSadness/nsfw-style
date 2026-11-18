// @ts-check
/**
 * Тест «ноль зависимостей» (ADR A11 плана) — проверяемое свойство, а не обещание.
 *
 * Падает, если:
 * - `package.json` объявляет любые `dependencies`/`devDependencies`;
 * - в `renderkit/` появился `node_modules/`;
 * - в `src/**` есть `import "node:…"`, `require(` или динамический `import('node:…')`
 *   (ядро обязано быть изоморфным: браузер, Node, сервис — один и тот же код).
 */

import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const PACKAGE_ROOT = join(HERE, '..');
const SRC_ROOT = join(PACKAGE_ROOT, 'src');

const packageJson = JSON.parse(readFileSync(join(PACKAGE_ROOT, 'package.json'), 'utf8'));

/**
 * Рекурсивный список файлов с заданным расширением.
 *
 * @param {string} dir
 * @param {string} extension
 * @returns {string[]}
 */
function listFiles(dir, extension) {
  /** @type {string[]} */
  const result = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      result.push(...listFiles(full, extension));
    } else if (entry.endsWith(extension)) {
      result.push(full);
    }
  }
  return result;
}

test('package.json: type=module, private, engines.node>=20, скрипты', () => {
  assert.equal(packageJson.name, 'qwenstyle-renderkit');
  assert.equal(packageJson.type, 'module');
  assert.equal(packageJson.private, true);
  assert.equal(packageJson.engines.node, '>=20');
  assert.equal(packageJson.scripts.test, 'node --test "test/**/*.test.mjs"');
});

test('package.json: пустые dependencies и devDependencies', () => {
  assert.deepEqual(packageJson.dependencies, {});
  assert.deepEqual(packageJson.devDependencies, {});
  // Никаких optional/peer-зависимостей тоже: калькулятор ставится копированием каталога.
  for (const field of ['optionalDependencies', 'peerDependencies', 'bundledDependencies', 'overrides']) {
    assert.equal(packageJson[field], undefined, `поле ${field} не должно объявляться`);
  }
});

test('нет node_modules/', () => {
  const entries = readdirSync(PACKAGE_ROOT);
  assert.ok(!entries.includes('node_modules'), 'renderkit/node_modules присутствует: зависимости запрещены');
});

test('src/** не тянет node: и не использует require', () => {
  const files = listFiles(SRC_ROOT, '.mjs');
  assert.ok(files.length > 0, 'в src/ нет ни одного .mjs');
  /** @type {string[]} */
  const violations = [];
  for (const file of files) {
    const text = readFileSync(file, 'utf8');
    const name = relative(PACKAGE_ROOT, file);
    if (/(?:from|import)\s*\(?\s*['"]node:/u.test(text)) violations.push(`${name}: import "node:…"`);
    if (/\brequire\s*\(/u.test(text)) violations.push(`${name}: require(`);
  }
  assert.deepEqual(violations, []);
});
