#!/usr/bin/env bash
# stealth — idempotent Linux setup for the TRAIN venv (Python 3.11, exact lane pins).
#
#   bash scripts/setup.sh            # from anywhere; the script cd's to the repo root
#
# What it does (safe to re-run):
#   1. finds/installs Python 3.11: prefers `uv` (installs it to ~/.local/bin if missing), falls back to python3.11
#   2. creates .venv with Python 3.11 (reused if it already exists and is 3.11)
#   3. installs requirements-lock.txt + the package in editable mode (torch is deliberately NOT installed)
#   4. git config core.hooksPath scripts/hooks   (secret-scanning pre-commit hook)
#   5. runs  $PY -m stealth.selfcheck            (exit 1 on pin mismatch)
#   6. prints the absolute $PY to use in every handoff command
# Never touches .env, never prints secrets. Needs network for pip/uv only.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
VENV="$REPO/.venv"
PY="$VENV/bin/python"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

say() { printf '\n==> %s\n' "$*"; }

# ---- 1. python 3.11 -----------------------------------------------------------------------------
have_uv=0
if command -v uv >/dev/null 2>&1; then
  have_uv=1
elif command -v curl >/dev/null 2>&1; then
  say "uv not found; installing to ~/.local/bin (https://astral.sh/uv)"
  if curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 && command -v uv >/dev/null 2>&1; then
    have_uv=1
  else
    echo "uv install failed; falling back to python3.11 -m venv"
  fi
fi

venv_ok=0
if [ -x "$PY" ] && "$PY" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)' 2>/dev/null; then
  venv_ok=1
  say "reusing existing .venv ($("$PY" --version))"
fi

# ---- 2. venv --------------------------------------------------------------------------------------
if [ "$venv_ok" -eq 0 ]; then
  if [ -d "$VENV" ]; then
    say "existing .venv is not Python 3.11; moving it to .venv.old"
    rm -rf "$VENV.old"; mv "$VENV" "$VENV.old"
  fi
  if [ "$have_uv" -eq 1 ]; then
    say "uv python install 3.11 && uv venv .venv --python 3.11 --seed"
    uv python install 3.11
    uv venv "$VENV" --python 3.11 --seed      # --seed installs pip into the venv so "$PY -m pip ..." repair steps work
  elif command -v python3.11 >/dev/null 2>&1; then
    say "python3.11 -m venv .venv"
    python3.11 -m venv "$VENV"
  else
    echo "ERROR: neither uv nor python3.11 is available. Install one of:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh      (then re-run this script)"
    echo "  sudo apt install python3.11 python3.11-venv        (Debian/Ubuntu with deadsnakes if needed)"
    exit 1
  fi
fi

# ---- 3. pins ---------------------------------------------------------------------------------------
say "installing requirements-lock.txt + stealth (editable) into $VENV"
if [ "$have_uv" -eq 1 ]; then
  uv pip install --python "$PY" -r requirements-lock.txt -e .
else
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install --quiet -r requirements-lock.txt -e .
fi

# ---- 3b. pip present? (uv venv without --seed has none; the handoffs' repair commands use "$PY -m pip") ---------
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  say "venv has no pip; adding it (ensurepip, else uv)"
  "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || { [ "$have_uv" -eq 1 ] && uv pip install --python "$PY" pip >/dev/null 2>&1; } || true
fi

# ---- 4. hook ----------------------------------------------------------------------------------------
say "git config core.hooksPath scripts/hooks"
chmod +x scripts/hooks/pre-commit scripts/daily_task.sh 2>/dev/null || true
git config core.hooksPath scripts/hooks
if ! git config --get user.email >/dev/null 2>&1 || ! git config --get user.name >/dev/null 2>&1; then
  echo "GIT IDENTITY MISSING -> HUMAN STEP: git config user.name \"<name>\" && git config user.email \"<email>\"  (git commit will fail until then)"
fi

# ---- 5. selfcheck -------------------------------------------------------------------------------------
say "selfcheck"
set +e
"$PY" -m stealth.selfcheck
rc=$?
set -e

# ---- 6. result ------------------------------------------------------------------------------------------
echo ""
if [ "$rc" -ne 0 ]; then
  echo "SETUP INCOMPLETE: selfcheck reported a pin mismatch (exit $rc). Re-run: bash scripts/setup.sh"
  echo "  (manual equivalent: uv pip install --python $PY -r requirements-lock.txt -e .   or   $PY -m pip install -r requirements-lock.txt -e .)"
  exit "$rc"
fi
if [ ! -s "$REPO/.env" ]; then
  echo "NOTE: .env is missing. HUMAN STEP: cp .env.example .env and paste EIQ_API_KEY (offline steps work without it)."
fi
echo "SETUP OK. Use this interpreter in every command:"
echo "PY=$PY"
