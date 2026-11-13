#!/usr/bin/env bash
# stealth -- OPTIONAL GPU venv for the MLP member only (.venv-gpu). docs/INTERFACES.md section 19.
#
#   bash scripts/setup_gpu.sh        # from anywhere; the script cd's to the repo root; safe to re-run
#
# What it does:
#   1. creates .venv-gpu with uv (Python 3.11, falling back to 3.12), else python3.11/python3.12 -m venv
#   2. installs torch from the CUDA 12.8 wheel index (override: TORCH_INDEX=..., TORCH_SPEC="torch==x.y")
#   3. installs requirements-gpu.txt (lane-pinned numpy/pandas/pyarrow/PyYAML/scipy)
#   4. pip install --no-deps -e .   (only stealth.data / stealth.knobs / stealth._runtime are used there)
#   5. GPU smoke: torch.cuda.is_available(), compute capability, a 4096x4096 matmul checked against the CPU
#   6. prints  GPU_PY=<abs path>  and the final status line  GPU OK  |  GPU FAIL -> skip MLP
#
# Exit codes: 0 = GPU OK; 3 = GPU FAIL (the venv works but has no usable GPU -> the handoff skips the MLP member);
#             1 = setup error (venv / pip failed); 64 = bad usage (it takes no arguments).
#             torch NEVER goes into .venv (the train venv) or the pickle.
set -euo pipefail

case "${1:-}" in
  -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "") ;;
  *) echo "usage: bash scripts/setup_gpu.sh   (no arguments; --help shows what it does)" >&2; exit 64 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
VENV="$REPO/.venv-gpu"
GPU_PY="$VENV/bin/python"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCH_SPEC="${TORCH_SPEC:-torch}"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

say() { printf '\n==> %s\n' "$*"; }
fail() { echo "SETUP ERROR: $*"; echo "GPU_PY=$GPU_PY"; echo "GPU FAIL -> skip MLP"; exit 1; }

have_uv=0
command -v uv >/dev/null 2>&1 && have_uv=1

# ---- 1. venv ----------------------------------------------------------------------------------------
venv_ok=0
if [ -x "$GPU_PY" ] && "$GPU_PY" -c 'import sys; sys.exit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)' 2>/dev/null; then
  venv_ok=1
  say "reusing existing .venv-gpu ($("$GPU_PY" --version))"
fi
if [ "$venv_ok" -eq 0 ]; then
  rm -rf "$VENV"
  for ver in 3.11 3.12; do
    if [ "$have_uv" -eq 1 ]; then
      say "uv venv .venv-gpu --python $ver --seed"
      if uv python install "$ver" >/dev/null 2>&1 && uv venv "$VENV" --python "$ver" --seed >/dev/null 2>&1; then venv_ok=1; break; fi
    fi
    if command -v "python$ver" >/dev/null 2>&1; then
      say "python$ver -m venv .venv-gpu"
      if "python$ver" -m venv "$VENV"; then venv_ok=1; break; fi
    fi
    rm -rf "$VENV"
  done
  [ "$venv_ok" -eq 1 ] || fail "could not create .venv-gpu with Python 3.11 or 3.12 (install uv: curl -LsSf https://astral.sh/uv/install.sh | sh)"
fi
if ! "$GPU_PY" -m pip --version >/dev/null 2>&1; then
  "$GPU_PY" -m ensurepip --upgrade >/dev/null 2>&1 || { [ "$have_uv" -eq 1 ] && uv pip install --python "$GPU_PY" pip >/dev/null 2>&1; } \
    || fail "no pip in .venv-gpu"
fi
echo "python: $("$GPU_PY" --version)"

# ---- 2. torch (CUDA 12.8 wheels) ----------------------------------------------------------------------
# torch's pure-Python dependencies come from PyPI FIRST: the CUDA wheel index serves some of them only as sdists
# (typing_extensions 4.16.0 as of 2026-11) whose build backend (flit_core) that index does not carry, so a plain
# `pip install torch --index-url .../cu128` dies with "No matching distribution found for flit_core" (seen on the
# sandbox). Dependencies that are already satisfied are not fetched from the torch index again.
TORCH_DEPS=(typing-extensions filelock sympy networkx jinja2 fsspec)
say "installing torch's pure-Python dependencies from PyPI: ${TORCH_DEPS[*]}"
if [ "$have_uv" -eq 1 ]; then
  uv pip install --python "$GPU_PY" "${TORCH_DEPS[@]}" || fail "torch dependency install failed"
else
  "$GPU_PY" -m pip install --quiet "${TORCH_DEPS[@]}" || fail "torch dependency install failed"
fi
say "installing $TORCH_SPEC from $TORCH_INDEX (large download the first time)"
if [ "$have_uv" -eq 1 ]; then
  uv pip install --python "$GPU_PY" "$TORCH_SPEC" --index-url "$TORCH_INDEX" || fail "torch install failed"
else
  "$GPU_PY" -m pip install --quiet "$TORCH_SPEC" --index-url "$TORCH_INDEX" || fail "torch install failed"
fi

# ---- 3. lane-pinned libraries ---------------------------------------------------------------------------
say "installing requirements-gpu.txt"
if [ "$have_uv" -eq 1 ]; then
  uv pip install --python "$GPU_PY" -r requirements-gpu.txt || fail "requirements-gpu.txt install failed"
else
  "$GPU_PY" -m pip install --quiet -r requirements-gpu.txt || fail "requirements-gpu.txt install failed"
fi

# ---- 4. the package itself, no dependencies (pyproject pins python 3.11; 3.12 is fine for the MLP script) -------
say "pip install --no-deps -e ."
if ! "$GPU_PY" -m pip install --quiet --no-deps --ignore-requires-python -e . ; then
  site="$("$GPU_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  echo "editable install failed; adding $REPO to $site/stealth_repo.pth instead"
  echo "$REPO" > "$site/stealth_repo.pth"
fi
"$GPU_PY" -c "import stealth.data, stealth.knobs, stealth._runtime; print('stealth importable from', stealth.__file__)" \
  || fail "stealth is not importable in .venv-gpu"

# ---- 5. GPU smoke ---------------------------------------------------------------------------------------
say "GPU smoke"
set +e
"$GPU_PY" - <<'PY'
import sys, time
import torch
print(f"torch {torch.__version__} (CUDA build {torch.version.cuda})")
if not torch.cuda.is_available():
    print("torch.cuda.is_available() is False (driver missing/too old, or a CPU-only wheel)")
    sys.exit(3)
try:
    cap = torch.cuda.get_device_capability(0)
    print(f"device: {torch.cuda.get_device_name(0)}  compute capability sm_{cap[0]}{cap[1]}  arch list {torch.cuda.get_arch_list()}")
    torch.backends.cuda.matmul.allow_tf32 = False
    g = torch.Generator().manual_seed(0)
    a = torch.randn(4096, 4096, generator=g)
    t0 = time.perf_counter()
    c = (a.cuda() @ a.cuda())
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ref = a[:256] @ a
    err = float((c[:256].cpu() - ref).abs().max() / ref.abs().max())
    print(f"4096^2 matmul {dt:.3f}s (incl. transfer + first-call init), rel max err vs CPU {err:.2e}")
    if not torch.isfinite(c).all() or err > 1e-3:
        print("matmul result is wrong")
        sys.exit(3)
except Exception as exc:  # e.g. "no kernel image is available for execution on the device" (wheel lacks sm_120)
    print(f"CUDA smoke failed: {type(exc).__name__}: {exc}")
    sys.exit(3)
sys.exit(0)
PY
rc=$?
set -e

echo
echo "GPU_PY=$GPU_PY"
if [ "$rc" -eq 0 ]; then
  echo "GPU OK"
  exit 0
fi
echo "GPU FAIL -> skip MLP"
exit 3
