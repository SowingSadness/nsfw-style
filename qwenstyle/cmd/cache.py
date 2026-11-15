"""`python -m qwenstyle cache list|show|prune` — обслуживание кэша (AGENTS.md, R2).

Политика перенесена из `tools/cli.py` без изменения поведения:

  * `list`/`show` — только чтение, ничего не подтверждают и не меняют;
  * `prune` без `--apply` — план, а не удаление;
  * по умолчанию на каждое разрешение внутри группы (промт + картинка + энкодер + dtype/device)
    остаётся один самый свежий кеш; `--group-by image` строже (один кеш на разрешение
    по картинке), `--keep-per-resolution N`, `--older-than 30d`, `--legacy`, `--all`;
  * удаляются только `cache/embeds_*.pt` и их `.json`; `cache/fp8_transformer*`
    и `cache/weights/*` не трогаются никогда.
"""
from __future__ import annotations

import argparse
import json
import sys

from ..cache import (
    CACHE_DIR,
    EmbedsEntry,
    delete_embeds,
    human_bytes,
    parse_age,
    plan_prune,
    scan_embeds,
)

NAME = "cache"
HELP = "обслуживание кэшей: list / show / prune (read-only по умолчанию)"


# --------------------------------------------------------------------------- list

def _entry_json(e: EmbedsEntry) -> dict:
    return {
        "path": str(e.path),
        "size": e.size,
        "mtime": e.mtime,
        "created": e.created,
        "md5": e.md5,
        "sha": e.sha,
        "enc": e.enc,
        "encoder_dtype": e.dtype,
        "encoder_device": e.device,
        "crop": e.crop,
        "resolution": e.resolution,
        "legacy": e.legacy,
    }


def cmd_list(args: argparse.Namespace) -> int:
    entries = scan_embeds(CACHE_DIR)
    if args.resolution:
        entries = [e for e in entries if e.resolution == args.resolution]
    if args.legacy is True:
        entries = [e for e in entries if e.legacy]
    elif args.legacy is False:
        entries = [e for e in entries if not e.legacy]
    entries.sort(key=lambda e: (e.md5, e.sha or "", e.enc, e.dtype, e.device,
                                e.resolution, -e.mtime))

    if args.json:
        print(json.dumps([_entry_json(e) for e in entries],
                         ensure_ascii=False, indent=2, default=str))
        return 0
    if not entries:
        print("[cache] пусто")
        return 0
    print(f"{'файл':60} {'разм.':>9} {'возр.':>7}  {'сетка':>9}  "
          f"{'промт':12} {'картинка':12} {'энкодер':18} dtype/device")
    for e in entries:
        mark = " (legacy)" if e.legacy else ""
        print(f"{e.path.name:60} {human_bytes(e.size):>9} {e.age_days:6.1f}d  {e.crop:>9}  "
              f"{e.md5:12} {e.sha or '-':12} {e.enc:18} {e.dtype}/{e.device}{mark}")
    print(f"\nитого: {len(entries)} записей, {human_bytes(sum(e.size for e in entries))}")
    return 0


# --------------------------------------------------------------------------- show

def _match(entries: list[EmbedsEntry], name: str) -> list[EmbedsEntry]:
    raw = str(name)
    if raw.endswith(".json"):
        raw = raw[:-5]
    exact = [e for e in entries if e.path.name == raw or e.path.stem == raw]
    if exact:
        return exact
    return [e for e in entries
            if e.path.name.startswith(raw) or e.path.stem.startswith(f"embeds_{raw}")]


def cmd_show(args: argparse.Namespace) -> int:
    match = _match(scan_embeds(CACHE_DIR), args.name)
    if not match:
        print(f"[cache] не найдено: {args.name}")
        return 2
    if len(match) > 1:
        print(f"[cache] префикс неоднозначен, найдено {len(match)}:")
        for e in match:
            print(f"  {e.path.name}")
        return 2
    e = match[0]
    print(f"файл     {e.path}")
    print(f"размер   {human_bytes(e.size)}")
    print(f"создан   {e.created} ({e.age_days:.1f} дн.)")
    print(f"сетка    {e.crop} (разрешение профиля ≈ {e.resolution})")
    print(f"промт    md5 {e.md5}")
    print(f"картинка sha256[:12] {e.sha or '— (старая схема ключа, мёртвая запись)'}")
    print(f"энкодер  {e.enc}")
    print(f"dtype    {e.dtype}")
    print(f"device   {e.device}")
    side = e.sidecar
    if side.is_file():
        print("сайдкар:")
        try:
            data = json.loads(side.read_text(encoding="utf-8"))
            for k, v in data.items():
                if k == "prompt":
                    print(f"  prompt: {str(v)[:200].replace(chr(10), ' ⏎ ')}…")
                else:
                    print(f"  {k}: {v}")
        except Exception as exc:                          # noqa: BLE001
            print(f"  <не читается: {exc}>")
    else:
        print("сайдкар: отсутствует (чем сгенерировано — неизвестно)")
    return 0


# -------------------------------------------------------------------------- prune

def cmd_prune(args: argparse.Namespace) -> int:
    entries = scan_embeds(CACHE_DIR)
    if not entries:
        print("[cache] чистить нечего")
        return 0
    older_than: float | None = None
    if args.older_than:
        try:
            older_than = parse_age(args.older_than)
        except ValueError as exc:
            print(f"[cache] --older-than: {exc}", file=sys.stderr)
            return 2
    targets = plan_prune(entries,
                         keep_per_resolution=args.keep_per_resolution,
                         group_by=args.group_by,
                         older_than=older_than,
                         legacy=args.legacy,
                         all=args.all)
    if not targets:
        print("[cache] удалять нечего — политика соблюдена "
              f"(keep-per-resolution={args.keep_per_resolution})")
        return 0

    freed = sum(e.size + (e.sidecar.stat().st_size if e.sidecar.is_file() else 0)
                for e in targets)
    print(f"[cache] план: удалить {len(targets)} записей, освободить {human_bytes(freed)}")
    for e in targets:
        tag = "legacy" if e.legacy else f"{e.crop} {e.age_days:.1f}d"
        print(f"  - {e.path.name}  ({human_bytes(e.size)}, {tag})")
    if not args.apply:
        print("\nэто план: ничего не удалено. Повтори с --apply, если согласен.")
        return 0

    deleted, freed_actual = delete_embeds(targets, CACHE_DIR)
    print(f"[cache] удалено {deleted} записей, освобождено {human_bytes(freed_actual)}")
    return 0


# --------------------------------------------------------------------------- CLI

def add_args(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="cache_action", required=True, metavar="<action>")

    p_list = actions.add_parser(
        "list", help="список записей кэша эмбеддингов",
        description="Печатает найденные `cache/embeds_*.pt`; ничего не меняет.",
        prog=f"{parser.prog} list",
    )
    p_list.add_argument("--resolution", type=int, default=None,
                        help="только это разрешение профиля (≈ sqrt(W*H) сетки)")
    p_list.add_argument("--legacy", dest="legacy", action="store_true", default=None,
                        help="только записи старой схемы ключа (без хеша картинки)")
    p_list.add_argument("--no-legacy", dest="legacy", action="store_false",
                        help="без записей старой схемы")
    p_list.add_argument("--json", action="store_true", help="JSON вместо таблицы")
    p_list.set_defaults(_cache_action="list")

    p_show = actions.add_parser(
        "show", help="метаданные одной записи (имя или префикс)",
        description="Печатает разобранный ключ и `.json`-сайдкар записи; read-only.",
        prog=f"{parser.prog} show",
    )
    p_show.add_argument("name", help="полное имя файла, имя без .pt или префикс (>=8 символов удобно)")
    p_show.set_defaults(_cache_action="show")

    p_prune = actions.add_parser(
        "prune", help="план/удаление; по умолчанию 1 кеш на разрешение",
        description="Без --apply только печатает план. Удаляются лишь embeds_*.pt и их .json.",
        prog=f"{parser.prog} prune",
    )
    p_prune.add_argument("--keep-per-resolution", type=int, default=1,
                         help="сколько самых свежих записей оставлять на каждое разрешение "
                              "(по умолчанию 1)")
    p_prune.add_argument("--group-by", choices=("prompt", "image"), default="prompt",
                         help="prompt (по умолчанию): чистить дубли внутри одного промта; "
                              "image: оставить ровно один кеш на разрешение, вытесняя старые промты")
    p_prune.add_argument("--older-than", default=None, metavar="30d|12h|90m",
                         help="удалить всё старше указанного возраста")
    p_prune.add_argument("--legacy", action="store_true",
                         help="дополнительно удалить записи старой схемы (без хеша картинки)")
    p_prune.add_argument("--all", action="store_true", help="удалить все записи кэша эмбеддингов")
    p_prune.add_argument("--apply", action="store_true",
                         help="выполнить удаление (без него — только план)")
    p_prune.set_defaults(_cache_action="prune")


def run(args: argparse.Namespace) -> int:
    action = getattr(args, "_cache_action", None) or getattr(args, "cache_action", None)
    handlers = {"list": cmd_list, "show": cmd_show, "prune": cmd_prune}
    handler = handlers.get(str(action))
    if handler is None:
        print(f"[cache] неизвестное действие {action!r}", file=sys.stderr)
        return 2
    return handler(args)


__all__ = ["NAME", "HELP", "add_args", "run", "cmd_list", "cmd_show", "cmd_prune"]
