#!/usr/bin/env bash
# WIP (2026-11-12): build the shared cluster venv. Run INSIDE the worker pod:
#   scripts/remote.sh --bg install-venv "setsid bash scripts/install_venv.sh"
#   scripts/remote.sh --bg install-venv "setsid bash scripts/install_venv.sh --venv /home/<user>/cvar-venv"
#   ... add --lock to install the exact pins from requirements.lock.txt
#
# Only the venv (and the Python 3.10 it links to, placed next to it) touches a quota:
#   - uv itself and its cache live in the worker's node-local /tmp (ephemeral);
#     UV_LINK_MODE=copy because the cache and the venv are on different filesystems;
#   - the log goes to /tmp first and is copied to $RUNS_DIR/_bg/ at exit, so a full quota can't
#     swallow the error message (that's how the first two attempts died silently).
# VENV_DIR: --venv DIR > $VENV_DIR from the caller > env.sh default ($CVL_STORE/.venvs/cvar-lewm).
# If you move it, also export the same VENV_DIR for every remote.sh call and GPU job.
# torch from the cu128 index: CUDA 12.8 runs on drivers >= 525 (12.x minor-version compatibility)
# and has sm_120 kernels for the Blackwell cards medium jobs may land on.
set -euo pipefail

LOCK=0
while [ $# -gt 0 ]; do
  case "$1" in
    --venv) export VENV_DIR="${2:?--venv needs a directory}"; shift 2 ;;
    --lock) LOCK=1; shift ;;
    *) echo "usage: install_venv.sh [--venv DIR] [--lock]" >&2; exit 2 ;;
  esac
done

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh            # keeps a pre-set VENV_DIR

TMPW=/tmp/cvl-install
mkdir -p "$TMPW"
LOG="$TMPW/install-venv.log"
exec >"$LOG" 2>&1

finish() {
  rc=$?
  echo "== install_venv.sh exit rc=$rc at $(date -Is)"
  mkdir -p "$RUNS_DIR/_bg"
  cp "$LOG" "$RUNS_DIR/_bg/install-venv.full.log" || echo "could not copy log to $RUNS_DIR/_bg (quota?); it stays at $LOG" >&2
  [ -f "$TMPW/requirements.lock.txt" ] && cp "$TMPW/requirements.lock.txt" "$RUNS_DIR/_bg/requirements.lock.txt" || true
}
trap finish EXIT
for s in TERM HUP INT; do trap "echo '== got SIG$s at \$(date -Is)'; exit 143" "$s"; done

export UV_CACHE_DIR=/tmp/uv-cache
export UV_LINK_MODE=copy
export UV_CONCURRENT_DOWNLOADS=4
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR_OVERRIDE:-$(dirname "$VENV_DIR")/.uv-python}"
UV_VERSION=0.13.7
TORCH_BACKEND=cu128
REQ=requirements.in
[ "$LOCK" = 1 ] && REQ=requirements.lock.txt
echo "== $(date -Is) VENV_DIR=$VENV_DIR UV_CACHE_DIR=$UV_CACHE_DIR UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR REQ=$REQ"
df -h /tmp "$(dirname "$VENV_DIR")" 2>/dev/null || true

UV=/tmp/uv-bin/uv
if [ ! -x "$UV" ] || [ "$("$UV" --version | cut -d' ' -f2)" != "$UV_VERSION" ]; then
  echo "== fetching uv $UV_VERSION into /tmp/uv-bin"
  mkdir -p /tmp/uv-bin
  curl -fsSL -m 120 "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" \
    | tar xz -C /tmp/uv-bin --strip-components=1
fi
"$UV" --version

echo "== python 3.10"
"$UV" python install 3.10
[ -x "$VENV_DIR/bin/python" ] || "$UV" venv "$VENV_DIR" --python 3.10
"$VENV_DIR/bin/python" --version

echo "== install from $REQ (torch backend $TORCH_BACKEND) $(date -Is)"
"$UV" pip install --python "$VENV_DIR/bin/python" --torch-backend "$TORCH_BACKEND" -r "$REQ"
echo "== install done $(date -Is)"

"$VENV_DIR/bin/python" - <<'PY'
import sys, torch, torchvision, stable_worldmodel, stable_pretraining, h5py, pymunk, pygame
print("python", sys.version.split()[0], "| torch", torch.__version__, "cuda build", torch.version.cuda,
      "| torchvision", torchvision.__version__, "| pymunk", pymunk.version)
sys.path.insert(0, ".")
import jepa, module  # upstream LeWM files import cleanly
print("imports OK")
PY

echo "== freeze"
"$UV" pip freeze --python "$VENV_DIR/bin/python" > "$TMPW/requirements.lock.txt"
echo "$(wc -l < "$TMPW/requirements.lock.txt") packages frozen"
echo "== sizes"
du -sh "$VENV_DIR" "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR" 2>/dev/null
echo "INSTALL_OK"
