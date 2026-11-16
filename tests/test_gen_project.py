"""`project.yaml` и приоритет настроек: preset -> yaml -> CLI.

Запуск: `PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tests/test_gen_project.py`
Успех: `PASS` по каждому кейсу, `ALL PASS` и exit 0; провал: `FAIL <кейс>: ...` и exit 1.

Зачем: `project.yaml` — единственное место, где пользователь хранит настройки запуска вместе
с промтом. Ошибка в приоритете не падает: она тихо берёт preset не из того места, игнорирует
`--set` или подхватывает yaml там, где его не просили. Тест фиксирует:

1. `preset` именем + поля `RunConfig` поверх него;
2. inline-пресет `{extends: base, size_px: 960, ...}` — тот же deep-merge, что у файлов;
3. CLI (`--preset`, `--set`) перекрывает yaml; `--prompt` вместе с `--project` меняет только промт;
4. `--prompt` без `--project` **не читает** лежащий рядом `project.yaml`;
5. относительные `prompt`/`refs`/`out` считаются от каталога yaml; имя yaml может быть любым;
6. ошибки: битый yaml, пустой inline-пресет, неизвестный ключ, `seeds`+`count`, нет промта;
7. `seeds/count` из yaml и приоритет CLI.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True                   # тест не должен оставлять __pycache__ в пакете

from qwenstyle.cmd import _project as project    # noqa: E402
from qwenstyle.cmd import gen as gen_cmd         # noqa: E402
from qwenstyle.config import ConfigError, RunConfig   # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
TEST_PROMPT = TESTS_DIR / "texts" / "exp1.txt"


def png_bytes(width: int = 64, height: int = 32) -> bytes:
    """Минимальный настоящий PNG (PIL открывает по содержимому, а не по расширению)."""
    import zlib

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return (len(data).to_bytes(4, "big") + body
                + zlib.crc32(body).to_bytes(4, "big"))

    raw = b"".join(b"\x00" + bytes([40, 80, 120]) * width for _ in range(height))
    header = (b"\x89PNG\r\n\x1a\n"
              + chunk(b"IHDR", width.to_bytes(4, "big") + height.to_bytes(4, "big")
                      + b"\x08\x02\x00\x00\x00"))
    return header + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


@contextlib.contextmanager
def temp_root():
    """`prompts/` во временном каталоге (генерации нет — только план)."""
    with tempfile.TemporaryDirectory(prefix="qwenstyle-gen-project-") as tmp:
        root = Path(tmp) / "prompts"
        root.mkdir()
        original = project.PROMPT_ROOT
        project.PROMPT_ROOT = root.as_posix()
        try:
            yield root
        finally:
            project.PROMPT_ROOT = original


def make_project(root: Path, name: str = "p", *, yaml_text: str | None = None,
                 prompts: tuple[str, ...] = ("prompt-qwen.txt",),
                 refs: tuple[str, ...] = ("01-a.png",)) -> Path:
    """Проект с refs/, промтами и (опционально) project.yaml."""
    folder = root / name
    folder.mkdir(parents=True)
    if refs:
        (folder / "refs").mkdir()
        for index, filename in enumerate(refs, start=1):
            (folder / "refs" / filename).write_bytes(png_bytes(width=32 + index))
    for filename in prompts:
        shutil.copyfile(TEST_PROMPT, folder / filename)
    if yaml_text is not None:
        (folder / project.DEFAULT_PROJECT_FILE).write_text(yaml_text, encoding="utf-8")
    return folder


def gen_args(**overrides) -> argparse.Namespace:
    values = dict(preset=None, prompt=None, project=None, out=None, aspect=None, sets=[],
                  upscale=None, seeds="1", count=None, dry_run=True, no_logs=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def call_gen(args: argparse.Namespace) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = gen_cmd.run(args)
    return code, out.getvalue(), err.getvalue()


# --------------------------------------------------------------------------- #
# Кейсы
# --------------------------------------------------------------------------- #

def case_preset_and_fields() -> None:
    """1. `preset` именем + поля RunConfig поверх пресета."""
    spec = project.ProjectSpec(preset="base", config={"size_px": 960, "aspect": "4:3"})
    cfg = project.build_run_config(spec, None)
    if (cfg.size_px, cfg.aspect, cfg.steps) != (960, "4:3", 25):
        raise AssertionError(f"cfg size_px={cfg.size_px} aspect={cfg.aspect} steps={cfg.steps}")
    if cfg.preset != "base":
        raise AssertionError(f"preset-штамп {cfg.preset!r}, ожидался 'base'")
    print("  ok: base + size_px/aspect из yaml, steps из пресета")


def case_inline_preset() -> None:
    """2. Inline-пресет с `extends` даёт те же поля, что и файл пресета."""
    spec = project.ProjectSpec(preset={
        "extends": "base",
        "size_px": 960,
        "generation": {"kv_cache": "offload", "steps": 8},
    }, config={"aspect": "1:1"})
    cfg = project.build_run_config(spec, None)
    if (cfg.size_px, cfg.steps, cfg.generation.kv_cache, cfg.aspect) != (960, 8, "offload", "1:1"):
        raise AssertionError(
            f"inline: size_px={cfg.size_px} steps={cfg.steps} kv={cfg.generation.kv_cache} "
            f"aspect={cfg.aspect}")
    if cfg.generation.backend != "qwen_image21":
        raise AssertionError(f"extends не сработал: backend={cfg.generation.backend}")
    if not str(cfg.preset).endswith("#preset") and "base" not in str(cfg.preset):
        raise AssertionError(f"preset-штамп inline: {cfg.preset!r}")
    print(f"  ok: inline {{extends: base}} -> size_px 960, steps 8, kv offload, preset={cfg.preset}")


def case_cli_overrides() -> None:
    """3. CLI перекрывает yaml: пресет и точечные параметры."""
    spec = project.ProjectSpec(preset="base", config={"size_px": 960, "generation": {"steps": 25}})
    cfg = project.build_run_config(spec, "VRAM12-preview", {"size_px": 480, "generation.steps": 4})
    if (cfg.size_px, cfg.steps) != (480, 4):
        raise AssertionError(f"CLI не перекрыл: size_px={cfg.size_px}, steps={cfg.steps}")
    if cfg.preset != "VRAM12-preview":
        raise AssertionError(f"preset из CLI не применился: {cfg.preset!r}")
    print("  ok: --preset/--set перекрывают yaml (480p, 4 шага, VRAM12-preview)")


def case_yaml_paths() -> None:
    """5. Относительные prompt/refs/out — от каталога yaml; имя yaml любое."""
    with temp_root() as root:
        folder = make_project(root, "p", prompts=("prompt-qwen.txt", "prompt-h3.txt"))
        (folder / "pictures").mkdir()
        (folder / "pictures" / "01-a.png").write_bytes(png_bytes(width=40))
        (folder / "settings.yaml").write_text(
            "preset: base\n"
            "prompt: prompt-h3\n"
            "refs: pictures\n"
            "out: renders\n",
            encoding="utf-8",
        )
        path = project.resolve_project((folder / "settings.yaml").as_posix())
        spec = project.load_project(path)
        if spec.prompt is None or spec.prompt.prompt_file.name != "prompt-h3.txt":
            raise AssertionError(f"prompt из yaml: {spec.prompt}")
        if spec.prompt.tag != "h3":
            raise AssertionError(f"tag: {spec.prompt.tag}")
        if spec.refs_dir != folder / "pictures":
            raise AssertionError(f"refs_dir={spec.refs_dir}, ожидался {folder / 'pictures'}")
        if spec.out_dir != folder / "renders":
            raise AssertionError(f"out_dir={spec.out_dir}, ожидался {folder / 'renders'}")
        # каталог и имя проекта разрешаются только для project.yaml (дефолтное имя)
        project_yaml = folder / project.DEFAULT_PROJECT_FILE
        project_yaml.write_text("preset: base\n", encoding="utf-8")
        if project.resolve_project(folder.as_posix()) != project_yaml:
            raise AssertionError("resolve_project(каталог) не нашёл project.yaml")
        if project.resolve_project("p") != project_yaml:
            raise AssertionError("resolve_project(имя проекта) не нашёл project.yaml")
        print("  ok: пути от каталога yaml, custom-имя, каталог/имя -> project.yaml")


def case_project_vs_prompt() -> None:
    """4/3b. `--project + --prompt` меняет только промт; `--prompt` без проекта не читает yaml."""
    with temp_root() as root:
        folder = make_project(
            root, "p", prompts=("prompt-qwen.txt", "prompt-h3.txt"),
            yaml_text=("preset: VRAM12-preview\n"
                       "prompt: prompt-qwen\n"
                       "generation: {steps: 4}\n"),
        )
        code, out, err = call_gen(gen_args(project=folder.as_posix(), prompt="p/prompt-h3"))
        if code != 0:
            raise AssertionError(f"--project+--prompt: code={code}, err={err!r}")
        if "prompt-h3.txt" not in out:
            raise AssertionError(f"промт из CLI не применился: {out[:400]!r}")
        if "VRAM12-preview" not in out or "4 шагов" not in out:
            raise AssertionError(f"preset/steps из yaml потерялись: {out[:500]!r}")
        # без --project yaml не читается: preset и steps только из CLI, шапка «только CLI»
        code, out, err = call_gen(gen_args(preset="VRAM12-preview", prompt="p/prompt-h3"))
        if code != 0:
            raise AssertionError(f"--prompt без --project: code={code}, err={err!r}")
        if "только CLI" not in out:
            raise AssertionError(f"нет пометки «только CLI»: {out[:400]!r}")
        if "4 шагов" in out:
            raise AssertionError(f"yaml подхватился без --project: {out[:500]!r}")
        print("  ok: --project+--prompt -> только промт; --prompt без --project -> только CLI")


def case_seeds_from_yaml() -> None:
    """7. `seeds/count` из yaml и приоритет CLI."""
    spec = project.ProjectSpec(seeds="3,5")
    seeds = gen_cmd.resolve_seeds(None, None, spec)
    if seeds != [3, 5]:
        raise AssertionError(f"seeds из yaml -> {seeds}")
    if gen_cmd.resolve_seeds(None, 2, spec) != [3, 5] and len(gen_cmd.resolve_seeds(None, 2)) != 2:
        raise AssertionError("--count не перекрыл seeds из yaml")
    spec = project.ProjectSpec(count=2)
    if len(gen_cmd.resolve_seeds(None, None, spec)) != 2:
        raise AssertionError("count из yaml не применился")
    print("  ok: seeds/count из yaml; --seeds/--count главнее")


def case_errors() -> None:
    """6. Ошибки: битый/пустой/неизвестный yaml, seeds+count, нет промта."""
    with temp_root() as root:
        folder = make_project(root, "bad", yaml_text="preset: [unclosed\n")
        try:
            project.load_project(folder / project.DEFAULT_PROJECT_FILE)
        except ConfigError as exc:
            if "некорректный YAML" not in str(exc):
                raise AssertionError(f"битый yaml: {exc}") from None
        else:
            raise AssertionError("битый yaml не отбит")

        folder = make_project(root, "empty", yaml_text="prompt: prompt-qwen\n")
        try:
            project.build_run_config(project.load_project(
                folder / project.DEFAULT_PROJECT_FILE), None)
        except ConfigError:
            pass
        else:
            raise AssertionError("yaml без preset не отбит")

        folder = make_project(root, "unknown", yaml_text="preset: base\nbogus: 1\n")
        try:
            project.build_run_config(project.load_project(
                folder / project.DEFAULT_PROJECT_FILE), None)
        except ConfigError as exc:
            if "неизвестные параметры" not in str(exc):
                raise AssertionError(f"неизвестный ключ: {exc}") from None
        else:
            raise AssertionError("неизвестный ключ не отбит")

        folder = make_project(root, "both", yaml_text="preset: base\nseeds: '1'\ncount: 2\n")
        try:
            project.load_project(folder / project.DEFAULT_PROJECT_FILE)
        except ConfigError as exc:
            if "seeds" not in str(exc):
                raise AssertionError(f"seeds+count: {exc}") from None
        else:
            raise AssertionError("seeds+count не отбиты")

        folder = make_project(root, "noprompt", yaml_text="preset: VRAM12-preview\n")
        code, _, err = call_gen(gen_args(project=folder.as_posix()))
        if code != 2 or "--prompt" not in err:
            raise AssertionError(f"yaml без prompt: code={code}, err={err!r}")
        print("  ok: битый YAML, пустой preset, неизвестный ключ, seeds+count, нет prompt")


CASES = (
    ("1. preset + поля yaml", case_preset_and_fields),
    ("2. inline preset (extends)", case_inline_preset),
    ("3. CLI перекрывает yaml", case_cli_overrides),
    ("4. пути от каталога yaml", case_yaml_paths),
    ("5. --project+--prompt и --prompt без yaml", case_project_vs_prompt),
    ("6. seeds/count из yaml", case_seeds_from_yaml),
    ("7. ошибки yaml", case_errors),
)


def main() -> int:
    failed = 0
    for label, case in CASES:
        try:
            case()
            print(f"PASS {label}", flush=True)
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {label}: {exc}", flush=True)
        except Exception:                                   # noqa: BLE001 — тест не должен врать
            failed += 1
            print(f"FAIL {label}: исключение\n{traceback.format_exc()}", flush=True)
    if failed:
        print(f"{failed} FAIL", flush=True)
        return 1
    print("ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
