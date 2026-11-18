// @ts-check
/**
 * Подсказки к параметрам: **единственное место** текстов, которые UI показывает в панели `?`
 * (ADR A10b). Здесь же — ссылки на замеры (`sources`) и id живого пересчёта (`live`).
 *
 * Форма записи:
 *
 * ```js
 * { id, label, section: 'input'|'render'|'memory'|'postprocess',
 *   aspects: { quality, perf, system }, sources: ['docs/bench/vram-tiers.md#…'],
 *   live: 'steps'|'refPx'|'kvCache'|'vaeTilePx'|'window'|'width'|'height'|'quality'|'anime'|null,
 *   note?: string }
 * ```
 *
 * Правила (проверяются `test/help.test.mjs`):
 *
 * - покрыты **все** поля входа и все поля `Plan.render`/`Plan.memory`;
 * - три аспекта непустые и не длиннее 320 символов: в панели 320 px длинный текст нечитаем;
 * - числа — только наши замеры (`docs/bench/*.md`, `AGENTS.md` §4), никаких обещаний;
 * - имён пресетов нет вовсе (калькулятор их не знает, `INVENTORY` тоже);
 * - `sources` — пути от корня репозитория (`docs/...`, `AGENTS.md`), файл обязан существовать;
 * - `live` — id what-if для `web/viewmodel.mjs::hintLive` или `null`, если живой дельты нет
 *   (параметр не входной: раскладку выбирает политика плана).
 *
 * Стиль текстов — короткие фразы без вводных слов: «↑ детали и контраст; 25 — полное расписание».
 */

/** Разрешённые id живого пересчёта (`hintLive`). @type {ReadonlyArray<string>} */
export const LIVE_IDS = Object.freeze([
  'steps',
  'refPx',
  'kvCache',
  'vaeTilePx',
  'window',
  'width',
  'height',
  'quality',
  'anime',
]);

/** Разделы формы/плана, к которым относится параметр. @type {ReadonlyArray<string>} */
export const HELP_SECTIONS = Object.freeze(['input', 'render', 'memory', 'postprocess']);

/**
 * @typedef {Object} HelpEntry
 * @property {string} id id параметра (= путь в `Plan`, где это возможно)
 * @property {string} label подпись для `aria-label` кнопки `?`
 * @property {'input'|'render'|'memory'|'postprocess'} section раздел формы или плана
 * @property {{quality: string, perf: string, system: string}} aspects три обязательных раздела панели
 * @property {string[]} sources ссылки на замеры (путь от корня репозитория, `#якорь` необязателен)
 * @property {string|null} live id what-if (`hintLive`) или `null`
 * @property {string} [note] оговорка: фиксировано моделью / не входной параметр / не поддержано
 */

const VRAM_TIERS = 'docs/bench/vram-tiers.md#тарифы';
const KV_DOC = 'docs/bench/kv-cache-offload.md#режимы';
const PRESETS_DOC = 'docs/bench/presets.md';
const AGENTS_4 = 'AGENTS.md#4-проверенные-ограничения-и-цифры';
const PIN_DOC = 'docs/bench/pin-memory.md';

/** @type {Record<string, HelpEntry>} */
export const HELP = Object.freeze({
  // ── поля входа ────────────────────────────────────────────────────────────────────────────
  width: {
    id: 'width',
    label: 'ширина кадра',
    section: 'input',
    aspects: {
      quality: 'Больше пикселей — больше реальной фактуры. Выше 5.73 Мп модель уходит в рябь, выше 8.35 Мп нативного пути нет.',
      perf: 'Время ∝ токенам (W/16·H/16): 768p — 15 с, 1440p — 205 с на 25 шагов, 1792p — 314 с.',
      system: 'Активации 11.98 ГиБ на 768p → 12.85 ГиБ на 1440p; KV-префикс растёт вместе с канвасом.',
    },
    sources: [VRAM_TIERS, AGENTS_4],
    live: 'width',
  },
  height: {
    id: 'height',
    label: 'высота кадра',
    section: 'input',
    aspects: {
      quality: 'Высота задаёт пропорцию и «p» кадра: 1440p — это 2560×1440 при 16:9. Нативность требует сетки 32.',
      perf: 'Время ∝ числу токенов: 1088p — 96 с, 1440p — 205 с, 1792p — 314 с на 25 шагов.',
      system: 'Пик VRAM 7.68 ГиБ на 1088p → 12.85 ГиБ на 1440p; окно внимания включается с 1600 px.',
    },
    sources: [VRAM_TIERS, AGENTS_4],
    live: 'height',
  },
  quality: {
    id: 'quality',
    label: 'уровень качества',
    section: 'input',
    aspects: {
      quality: 'Это политика, а не шаги: draft теряет мелкую фактуру, standard держит только ok-размеры, max разрешает «серую» зону и окно внимания.',
      perf: 'draft примерно ×3 быстрее: 8 шагов на 1440p — 92 с, 25 шагов — 205 с.',
      system: 'Сам по себе не влияет; max разрешает окно и KV в VRAM, поэтому пик может подрасти.',
    },
    sources: [VRAM_TIERS, AGENTS_4],
    live: 'quality',
  },
  bitDepth: {
    id: 'bitDepth',
    label: 'глубина цвета',
    section: 'input',
    aspects: {
      quality: '16 бит — запас по градиентам, но пайплайн пока сохраняет 8-битный PNG: вердикт unsupported.',
      perf: 'Не влияет на время генерации.',
      system: 'Диск ×2 на кадр: 2560×1440 — 5.1 МБ против ~10.2 МБ.',
    },
    sources: ['AGENTS.md', VRAM_TIERS],
    live: null,
    note: '16 бит не реализованы в пайплайне: память и время считаются как для 8 бит, размер файла — ×2.',
  },
  model: {
    id: 'model',
    label: 'модель генерации',
    section: 'input',
    aspects: {
      quality: 'base — 25 шагов, turbo — 8 шагов дистиллята, h3-image — 6 шагов PDD; фактура и стиль у моделей разные.',
      perf: 'turbo ~2× быстрее base на том же кадре: 640p — 5 с против 6 с, 1024 — 21 с против 38 с.',
      system: 'h3: fp8-трансформер 30.9 ГиБ и энкодер 62 ГиБ — нужны 16 ГиБ VRAM и 64 ГиБ RAM.',
    },
    sources: [AGENTS_4, PRESETS_DOC],
    live: null,
  },
  vramGib: {
    id: 'vramGib',
    label: 'объём VRAM',
    section: 'input',
    aspects: {
      quality: 'Кадр не меняет: бюджет памяти влияет на раскладку, а не на картинку.',
      perf: 'Мало VRAM — офлоад и CPU-энкод: спилл активаций на RAM удваивает время шага.',
      system: 'Это и есть бюджет: гейты 0.90 и 0.97 от него; на 8 ГиБ энкодер уходит на CPU (90–126 с).',
    },
    sources: [VRAM_TIERS, AGENTS_4],
    live: null,
  },
  ramGib: {
    id: 'ramGib',
    label: 'объём RAM',
    section: 'input',
    aspects: {
      quality: 'Кадр не меняет.',
      perf: 'Мало RAM — вытеснение page cache fp8-весов, шаги замедляются в разы.',
      system: 'В RAM живут KV-префикс (2.13–7.14 ГиБ) и fp8-блоки при офлоаде (6.6 ГиБ); h3 требует 64 ГиБ.',
    },
    sources: [KV_DOC, AGENTS_4],
    live: null,
  },
  disk: {
    id: 'disk',
    label: 'класс накопителя',
    section: 'input',
    aspects: {
      quality: 'Кадр не меняет.',
      perf: 'NVMe 3.0 ГиБ/с против HDD 0.13 ГиБ/с: кэш fp8 6.7 ГиБ — 2 с против ~50 с на старт.',
      system: 'Место под веса и кэши: 74.9 / 54.9 / 181.0 ГиБ для 8 / 12 / 16 ГиБ VRAM.',
    },
    sources: ['docs/bench/vram-tiers.md', AGENTS_4],
    live: null,
  },
  refs: {
    id: 'refs',
    label: 'число референсов',
    section: 'input',
    aspects: {
      quality: 'Больше референсов — больше контроля композиции, но выше риск конфликта кадра и референса. 0 — только текстовый кадр.',
      perf: '+10.5 с энкод на каждую новую пару (GPU); из кэша эмбеддингов — 0 с.',
      system: 'KV +2.13 ГиБ на референс на 768p; суммарно ≤~13 тыс. токенов референсов, иначе OOM энкодера.',
    },
    sources: [KV_DOC, AGENTS_4],
    live: null,
  },
  anime: {
    id: 'anime',
    label: 'стилевой адаптер',
    section: 'input',
    aspects: {
      quality: 'Включает адаптер FUSAL силой 0.7: линия, шейдинг и палитра аниме. Без него — чистая база.',
      perf: 'Бесплатно: адаптер применяется на тех же шагах.',
      system: 'Не влияет: адаптер 420 МБ уже в весах, VRAM и RAM не растут.',
    },
    sources: [AGENTS_4],
    live: 'anime',
  },
  freeDiskGib: {
    id: 'freeDiskGib',
    label: 'свободно на диске',
    section: 'input',
    aspects: {
      quality: 'Кадр не меняет.',
      perf: 'Не влияет на время.',
      system: 'Проверка места: меньше футпринта модели — предупреждение disk-not-enough.',
    },
    sources: ['AGENTS.md'],
    live: null,
    note: 'Пустое поле = не проверять.',
  },

  // ── план: рендер ─────────────────────────────────────────────────────────────────────────
  steps: {
    id: 'steps',
    label: 'шаги сэмплера',
    section: 'render',
    aspects: {
      quality: '↑ детали и контраст; 25 — полное расписание модели, 8 — быстрый черновик композиции.',
      perf: 'Линейно: 8.2 с/шаг на 1440p → +12 с на 1–2 шага (25 шагов = 205 с).',
      system: 'Не влияет: пик VRAM от шагов не зависит — 8 и 25 шагов дают один и тот же пик.',
    },
    sources: [VRAM_TIERS, AGENTS_4],
    live: 'steps',
    note: 'h3-image: 6 шагов фиксированы расписанием PDD, turbo — 8 шагов дистиллята.',
  },
  offload: {
    id: 'offload',
    label: 'офлоад весов',
    section: 'render',
    aspects: {
      quality: 'Не влияет: картинка бит-в-бит совпадает с прогоном без офлоада.',
      perf: 'group в 1.5–2.6 раза медленнее: 768p — 83–97 с против 36–44 с.',
      system: '−6.6 ГиБ VRAM: fp8-блоки живут в RAM и стримятся на GPU по группе.',
    },
    sources: [AGENTS_4],
    live: null,
    note: 'Выбирается по бюджету VRAM; от 960p групповой офлоад обязателен.',
  },
  offloadBlocks: {
    id: 'offloadBlocks',
    label: 'блоков в группе офлоада',
    section: 'render',
    aspects: {
      quality: 'Не влияет: результат бит-в-бит.',
      perf: '1 против 8 — ≤4 % (19.9 против 20.7 с/шаг), рычаг не здесь.',
      system: 'Мельче группа — меньше пик VRAM; на 960p разницы нет, активации доминируют.',
    },
    sources: [AGENTS_4],
    live: null,
  },
  encoderDevice: {
    id: 'encoderDevice',
    label: 'устройство энкодера',
    section: 'render',
    aspects: {
      quality: 'Не влияет: результат бит-в-бит, расхождение только в округлениях разных устройств.',
      perf: 'GPU 10.5 с на 4128 токенов против CPU 55–255 с; режим I/O-bound, от длины промта почти не зависит.',
      system: 'GPU-стриминг: 1–3.5 ГиБ VRAM и ~4.8 ГиБ RAM; CPU: 17.6 ГиБ RAM резидентно.',
    },
    sources: [PIN_DOC, AGENTS_4],
    live: null,
    note: 'Выбирается по бюджету VRAM: на 8 ГиБ энкодер считается на CPU.',
  },
  precision: {
    id: 'precision',
    label: 'точность весов',
    section: 'render',
    aspects: {
      quality: 'fp8 против bf16 — разницы в кадре не замечено.',
      perf: 'fp8 стартует быстрее: кэш 6.7 ГиБ читается за ~4 с вместо каста 26 с.',
      system: 'bf16 — 14.2 ГиБ VRAM: на 16 ГиБ не влезает вместе с VAE.',
    },
    sources: [AGENTS_4, PRESETS_DOC],
    live: null,
  },
  loraScale: {
    id: 'loraScale',
    label: 'сила стилевого адаптера',
    section: 'render',
    aspects: {
      quality: '0.65–0.7 для правки кадра, 0.85–1.0 для генерации с нуля; 0 — чистая база, >1 — экстраполяция.',
      perf: 'Бесплатно: те же шаги, та же модель.',
      system: 'Не влияет: адаптер уже в весах, VRAM и RAM не растут.',
    },
    sources: ['AGENTS.md#56-сила-lora---lora-scale'],
    live: null,
    note: 'Задаётся полем anime; по-слойной настройки у адаптера нет, масштаб глобальный.',
  },

  // ── план: раскладка памяти ───────────────────────────────────────────────────────────────
  kvCache: {
    id: 'kvCache',
    label: 'KV-кеш префикса',
    section: 'memory',
    aspects: {
      quality: 'Не влияет: режим бит-в-бит (проверено побайтным sha256 кадра).',
      perf: 'true в VRAM — −13 % времени; offload — +13 % на подкачку слоёв.',
      system: 'Префикс 512 КиБ/токен: 2.13–7.14 ГиБ. В VRAM при запасе ≥10 %, иначе в RAM.',
    },
    sources: [KV_DOC, AGENTS_4],
    live: 'kvCache',
  },
  vaeTilePx: {
    id: 'vaeTilePx',
    label: 'тайл декодера VAE',
    section: 'memory',
    aspects: {
      quality: '512/384 — сетка 16 px ×1.69; 256/192 — ×2.71; 1024/768 — ×1.51.',
      perf: 'Декод 0.99 с — в пределах шума, на время генерации не влияет.',
      system: 'Пик декода 7.4 ГиБ (512), 15.0 ГиБ (1024), без тайлинга — OOM от 1.5 Мп.',
    },
    sources: ['AGENTS.md', VRAM_TIERS],
    live: 'vaeTilePx',
  },
  window: {
    id: 'window',
    label: 'окно внимания',
    section: 'memory',
    aspects: {
      quality: 'Плотное внимание на 2.6 % резче (14.8 против 15.2). Окну нужно покрытие ≥60 % канваса.',
      perf: '−17 % времени на 1792p: 314 с против 376 с у плотного.',
      system: 'VRAM не меняет; полосы 384+384, 5 полос, покрытие 64 %.',
    },
    sources: ['AGENTS.md#скользящее-окно-внимания', AGENTS_4],
    live: 'window',
  },
  refPx: {
    id: 'refPx',
    label: 'сетка референсов',
    section: 'memory',
    aspects: {
      quality: '0 — максимум (сетка кадра); меньше — референс грубее выхода, композиция держится хуже.',
      perf: 'Не влияет или чуть быстрее: префикс короче.',
      system: 'KV 2.62 ГиБ (768-сетка) против 7.14 ГиБ (1440-сетка); суммарно ≤~13k токенов.',
    },
    sources: [KV_DOC, AGENTS_4],
    live: null,
    note: 'Не входной параметр: 0 при ≤1088p, 960 от 1440p, 768 при трёх референсах.',
  },

  // ── стадия постобработки ─────────────────────────────────────────────────────────────────
  postprocess: {
    id: 'postprocess',
    label: 'стадия постобработки',
    section: 'postprocess',
    aspects: {
      quality: 'Отдельная модель — своё качество. Калькулятор её не считает и размер ею не «добирает».',
      perf: 'Отдельный расчёт: свои секунды, своя загрузка весов.',
      system: 'Отдельная стадия со своим бюджетом VRAM/RAM/диска; с рендером не складывается.',
    },
    sources: ['AGENTS.md', 'docs/bench/presets.md'],
    live: null,
    note: 'Стадия выключена: модель постобработки не передана.',
  },
});

/**
 * Запись подсказки по id (или `null`): UI не должен падать на отсутствующем ключе.
 *
 * @param {string} id
 * @returns {HelpEntry|null}
 */
export function helpFor(id) {
  return Object.prototype.hasOwnProperty.call(HELP, id) ? /** @type {HelpEntry} */ (HELP[id]) : null;
}
