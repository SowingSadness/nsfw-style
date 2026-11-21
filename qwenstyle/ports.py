"""Порты пайплайна — типизированные данные, которыми обмениваются кубики (план `cube-architecture`, шаг 1).

Зачем отдельный модуль: сегодня «conditioning» — это `dict` с ключами `embeds`/`mask`/`img_mask`,
а «латенты» — голый тензор. Из-за этого несовместимую цепочку кубиков ловит Python в момент
`forward`, а не композитор. Порт объявляет **имя типа** (`port`) и **состав**; кубик в спеке
(`qwenstyle/cube.py`) перечисляет, какие порты он потребляет и какой производит, а
`qwenstyle/compose.py` проверяет стыковку до загрузки весов.

Правила:

* порт — `frozen` dataclass с `ClassVar[str] port`, тензоры внутри не копируются;
* кубик не создаёт порты «на лету»: только те, что объявлены здесь (`PORT_TYPES`);
* сервисы (кэш, логгер) портами не являются — они передаются через `CubeContext`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

__all__ = [
    "PORT_TYPES",
    "Conditioning",
    "Frames",
    "Latents",
    "Pixels",
    "Prompt",
    "RefImages",
    "port_of",
]


@dataclass(frozen=True)
class Prompt:
    """Текст запроса. `md5` — ключ кэша, `origin` — откуда взят (файл, проект, PE-кубик)."""

    port: ClassVar[str] = "prompt"

    text: str
    md5: str
    origin: str = ""


@dataclass(frozen=True)
class RefImages:
    """Референсы входа. `grid` — сетка, на которой их увидел энкодер (W, H); `sha` — ключ кэша."""

    port: ClassVar[str] = "refs"

    images: tuple[Any, ...] = ()
    grid: tuple[int, int] = (0, 0)
    sha: str = ""

    def __bool__(self) -> bool:
        return bool(self.images)


@dataclass(frozen=True)
class Conditioning:
    """Эмбеддинги промта и референсов — выход энкодера, вход трансформера.

    `ref_grid` — сетка, на которой считались референсы (`ref_px`): она нужна трансформеру, чтобы
    понять, что генерация идёт по канвасу, а референсы — мельче (иначе OOM на энкодере).
    """

    port: ClassVar[str] = "conditioning"

    embeds: Any = None
    mask: Any = None
    img_mask: Any = None
    ref_grid: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class Latents:
    """Шум/латент канваса. `canvas` — (W, H) в пикселях (кратно 32), `seed` — для трассировки."""

    port: ClassVar[str] = "latents"

    x: Any = None
    canvas: tuple[int, int] = (0, 0)
    seed: int = 0


@dataclass(frozen=True)
class Frames:
    """Тензор кадра после `vae.decode`, ещё **не** картинка.

    Зачем отдельный порт (а не сразу `Pixels`): декодер VAE умеет отдавать не только изображение,
    но и последовательность (временная ось `T`); преобразование тензора в `PIL.Image`/файл — это
    отдельная стадия (`decoder.postprocess.image`, в будущем `decoder.postprocess.video`), и она
    отвечает за контейнер и формат, а не декодер.

    `x` — значения в диапазоне ~[-1, 1] (то, что postprocess нормализует в 0..255),
    `size` — (W, H) в пикселях, `temporal` — длина по времени (1 = один кадр).
    """

    port: ClassVar[str] = "frames"

    x: Any = None
    size: tuple[int, int] = (0, 0)
    temporal: int = 1
    source: str = ""


@dataclass(frozen=True)
class Pixels:
    """Готовый кадр в пикселях. `trace` — какие кубики его сделали (для лога и метаданных)."""

    port: ClassVar[str] = "pixels"

    image: Any = None
    stage_trace: tuple[str, ...] = field(default_factory=tuple)


#: Все объявленные порты. Композитор сверяет `consumes`/`produces`/`reads` с этим набором.
PORT_TYPES: dict[str, type] = {
    cls.port: cls for cls in (Prompt, RefImages, Conditioning, Latents, Frames, Pixels)
}


def port_of(value: Any) -> str:
    """Имя порта у значения (или ошибка, если это не порт).

    Нужен ассертам в кубиках и тестам: `port_of(conditioning) == "conditioning"`.
    """
    name = getattr(type(value), "port", None)
    if not isinstance(name, str) or name not in PORT_TYPES:
        raise TypeError(f"{type(value).__name__} не является портом пайплайна "
                        f"(допустимые: {', '.join(sorted(PORT_TYPES))})")
    return name
