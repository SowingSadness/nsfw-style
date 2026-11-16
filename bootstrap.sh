#!/usr/bin/env bash
# qwenstyle bootstrap (Linux / macOS): one command to a working install.
#
# Creates .venv, installs this package with CUDA 12.9 torch wheels (Linux), prints the model plan
# of a machine profile, downloads those repositories from HuggingFace and runs a self-test.
# Idempotent: an existing .venv and already installed packages are reused, nothing is deleted.
#
# Profile repository lists live in qwenstyle/profiles.py only - this script never duplicates them.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONIOENCODING=utf-8

TOTAL=9
STEP=0
PROFILE="VRAM12"
DRY_RUN=0
CHECK=0
SKIP_MODELS=0
NO_VENV=0

usage() {
    cat <<'EOF'
qwenstyle bootstrap - one command to a working install (Linux / macOS)

usage: ./bootstrap.sh [VRAM8|VRAM12|VRAM16] [--profile VRAM8|VRAM12|VRAM16]
                      [--dry-run] [--check] [--skip-models] [--no-venv]

  VRAM8|VRAM12|VRAM16  machine profile as the first positional argument
                        (same as --profile NAME)
  --profile NAME  machine profile: VRAM8 (8 GB VRAM), VRAM12 (12 GB, default),
                  VRAM16 (16 GB VRAM + 64 GB RAM)
  --dry-run       print every step and the model plan; install and download nothing
  --check         setup + Hub access probe (one small file per repository), no full download
  --skip-models   do not download weights (already present / offline machine)
  --no-venv       use the system Python 3.12 instead of creating .venv
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        VRAM8|VRAM12|VRAM16)
            PROFILE="$1"; shift ;;
        --profile)
            if [ $# -lt 2 ]; then usage >&2; exit 2; fi
            PROFILE="$2"; shift 2 ;;
        --profile=*) PROFILE="${1#*=}"; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --check) CHECK=1; shift ;;
        --skip-models) SKIP_MODELS=1; shift ;;
        --no-venv) NO_VENV=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf '[error] unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$PROFILE" in
    VRAM8|VRAM12|VRAM16) ;;
    *) printf "[error] unknown profile '%s'; available: VRAM8, VRAM12, VRAM16\n" "$PROFILE" >&2; exit 2 ;;
esac

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

step() {
    STEP=$((STEP + 1))
    printf '\n[%d/%d] %s\n' "$STEP" "$TOTAL" "$1"
}

note() { printf '    %s\n' "$1"; }
warn() { printf '    ! %s\n' "$1"; }

die() {
    printf '\n[error] %s\n' "$1" >&2
    exit "$2"
}

trap 'code=$?; [ "$code" -eq 0 ] || printf "\n[error] aborted with exit code %s\n" "$code" >&2' EXIT

# Runs a command, echoes it and fails the script (set -e) on a non-zero exit.
run() {
    note "> $*"
    "$@"
}

find_python312() {
    for candidate in python3.12 python3 python; do
        command -v "$candidate" >/dev/null 2>&1 || continue
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)' >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

# torch/diffusers/transformers/qwenstyle importable -> the pip step can be skipped.
env_ready() {
    [ -n "$PY" ] || return 1
    "$PY" -c "import importlib.util as u, sys; names = ('torch', 'diffusers', 'transformers', 'qwenstyle'); sys.exit(0 if all(u.find_spec(n) for n in names) else 1)" >/dev/null 2>&1
}

# Loads the manifest by path: profiles.py imports stdlib only, so this works before pip install.
MANIFEST="import importlib.util as u, sys; s = u.spec_from_file_location('_qp', 'qwenstyle/profiles.py'); m = u.module_from_spec(s); sys.modules['_qp'] = m; s.loader.exec_module(m); p = m.profile(sys.argv[1]); "

profile_line() {
    [ -n "$PY" ] || return 0
    "$PY" -c "${MANIFEST}print(p.hardware + ', ~%.1f GiB on disk' % p.disk_gib)" "$PROFILE" 2>/dev/null || true
}

profile_presets() {
    [ -n "$PY" ] || return 0
    "$PY" -c "${MANIFEST}print(', '.join(p.presets))" "$PROFILE" 2>/dev/null || true
}

profile_plan_fallback() {
    [ -n "$PY" ] || return 0
    "$PY" -c "${MANIFEST}print('  download %.1f GiB, on disk %.1f GiB' % (m.download_estimate(p.name), p.disk_gib)); [print('  %-52s %-34s %-6s %s' % (r.repo, r.license or '-', 'gated' if r.gated else '-', r.note)) for r in p.repos]" "$PROFILE" || true
}

# --------------------------------------------------------------------------- #
# Banner
# --------------------------------------------------------------------------- #

echo
echo 'qwenstyle bootstrap - local NSFW style-transfer and generation'
echo '18+ / NSFW: adult content; model licences belong to their HuggingFace repositories.'
note "root:    $ROOT"
note "profile: $PROFILE"
if [ "$DRY_RUN" -eq 1 ]; then
    note 'mode:    --dry-run (prints the plan, installs nothing, downloads nothing)'
elif [ "$CHECK" -eq 1 ]; then
    note 'mode:    --check (setup + Hub access probe, no full download)'
else
    note 'mode:    full setup'
fi

# --------------------------------------------------------------------------- #
# 1/9 interpreter and virtual environment
# --------------------------------------------------------------------------- #

step 'Python 3.12 and virtual environment'

BASE_PY="$(find_python312 || true)"
VENV_PY="$ROOT/.venv/bin/python"
PY=""

if [ "$NO_VENV" -eq 1 ]; then
    if [ -z "$BASE_PY" ]; then
        if [ "$DRY_RUN" -eq 1 ]; then
            warn 'python 3.12 not found (preview only: setup steps are printed, not run)'
        else
            die 'python 3.12 not found: install Python 3.12 or drop --no-venv' 2
        fi
    else
        PY="$BASE_PY"
        note "--no-venv: using $PY"
    fi
elif [ -x "$VENV_PY" ]; then
    PY="$VENV_PY"
    note "using existing .venv: $PY"
elif [ -z "$BASE_PY" ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
        warn 'python 3.12 not found (preview only: setup steps are printed, not run)'
    else
        die 'python 3.12 not found: install Python 3.12, then re-run ./bootstrap.sh' 2
    fi
elif [ "$DRY_RUN" -eq 1 ]; then
    note "would create .venv with: $BASE_PY -m venv .venv"
    PY="$BASE_PY"
else
    note "creating .venv with: $BASE_PY -m venv .venv"
    run "$BASE_PY" -m venv .venv
    [ -x "$VENV_PY" ] || die '.venv/bin/python was not created' 2
    PY="$VENV_PY"
fi

if [ -n "$PY" ]; then
    line="$(profile_line)"
    [ -z "$line" ] || note "profile hardware: $line"
fi

# --------------------------------------------------------------------------- #
# 2/9 packages (torch with CUDA wheels first, then this package)
# --------------------------------------------------------------------------- #

step 'Python packages: torch (CUDA 12.9 wheels on Linux), then this package'

if [ -z "$PY" ]; then
    note 'skipped: no Python 3.12 interpreter'
elif [ "$DRY_RUN" -eq 1 ]; then
    note 'would run (torch first: plain pip could pull a CPU-only build):'
    note "  $PY -m pip install --upgrade pip"
    note "  $PY -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129"
    note "  $PY -m pip install -e ."
elif env_ready; then
    note 'already installed (torch/diffusers/transformers/qwenstyle importable), skipping pip'
else
    run "$PY" -m pip install --upgrade pip
    case "$(uname -s)" in
        Linux)
            run "$PY" -m pip install torch torchvision \
                --index-url https://download.pytorch.org/whl/cu129 ;;
        *)
            note 'not Linux: installing the default torch build (no CUDA index)'
            run "$PY" -m pip install torch torchvision ;;
    esac
    run "$PY" -m pip install -e .
fi

# --------------------------------------------------------------------------- #
# 3/9 model plan (always printed, never downloads)
# --------------------------------------------------------------------------- #

step "Model plan for profile $PROFILE (no download; gated repositories are marked)"
if [ -z "$PY" ]; then
    note 'skipped: no Python 3.12 interpreter'
else
    code=0
    "$PY" -m qwenstyle models download --profile "$PROFILE" --dry-run || code=$?
    if [ "$code" -ne 0 ]; then
        warn "plan command failed with exit code $code; showing the manifest directly"
        profile_plan_fallback
    fi
fi

# --------------------------------------------------------------------------- #
# 4/9 --dry-run stops here
# --------------------------------------------------------------------------- #

if [ "$DRY_RUN" -eq 1 ]; then
    step 'Preview finished (--dry-run): nothing installed, nothing downloaded'
    note 'run without --dry-run to install; add --check to stop after the Hub access probe'
    echo
    exit 0
fi

# --------------------------------------------------------------------------- #
# 5/9 --check: Hub access probe, one small file per repository
# --------------------------------------------------------------------------- #

if [ "$CHECK" -eq 1 ]; then
    step "Hub access probe for profile $PROFILE (one small file per repository, no weights)"
    code=0
    HF_HUB_OFFLINE=0 "$PY" -m qwenstyle models download --profile "$PROFILE" \
        --probe --probe-mib 1 --cleanup || code=$?
    if [ "$code" -eq 0 ]; then
        printf '\n    check passed: every repository of this profile is reachable\n'
    else
        warn "check failed with exit code $code - see the FAIL rows above (gated repositories need HF_TOKEN)"
    fi
    exit "$code"
fi

# --------------------------------------------------------------------------- #
# 6/9 download weights
# --------------------------------------------------------------------------- #

step "Download model weights for profile $PROFILE"
if [ "$SKIP_MODELS" -eq 1 ]; then
    note 'skipped: --skip-models'
else
    note 'only the repositories of this profile are fetched, under their own licences'
    code=0
    HF_HUB_OFFLINE=0 "$PY" -m qwenstyle models download --profile "$PROFILE" --yes || code=$?
    [ "$code" -eq 0 ] || die "models download failed with exit code $code" "$code"
fi

# --------------------------------------------------------------------------- #
# 7/9 environment report
# --------------------------------------------------------------------------- #

step 'Environment report'
run "$PY" -m qwenstyle env

# --------------------------------------------------------------------------- #
# 8/9 self-test (one 480p render on a synthetic image)
# --------------------------------------------------------------------------- #

step 'Self-test (one 480p render on a synthetic image)'
run "$PY" -m qwenstyle selftest --profile "$PROFILE"

# --------------------------------------------------------------------------- #
# 9/9 next steps
# --------------------------------------------------------------------------- #

step 'Ready'

presets="$(profile_presets)"
echo
echo '    Put a scene folder together:'
note 'prompts/my-scene/prompt.txt   (FUSAL prompt, see PROMPT_RULES.md)'
note 'prompts/my-scene/image1.png    (optional references, up to 10)'
echo
echo '    Generate:'
note "$PY -m qwenstyle gen --preset $PROFILE --prompt my-scene --count 1"
echo
echo '    Presets of this profile:'
if [ -n "$presets" ]; then note "$presets"; else note 'python -m qwenstyle models list'; fi
echo
echo '    Docs: README.md (quick start), PROMPT_RULES.md (how to write a prompt)'
echo

exit 0
