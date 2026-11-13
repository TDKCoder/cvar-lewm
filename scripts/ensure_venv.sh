#!/usr/bin/env bash
# Ephemeral project venv in the pod's node-local /tmp (decision 2026-11-13: nothing on the quota).
# Idempotent: a no-op (< 1 s) when $VENV_DIR exists and was built from the current
# requirements.lock.txt; otherwise (re)builds it (~minutes, measured in runs/_bg/venv-builds.tsv).
#
#   bash scripts/ensure_venv.sh            # in a pod (remote.sh command, GPU job script)
#   scripts/withvenv.sh python ...         # ensure + activate + run (what remote.sh commands use)
#   bash scripts/ensure_venv.sh --from-in  # resolve requirements.in afresh and write a new lock
#
# Inputs: requirements.lock.txt (exact pins). If it is missing, or with --from-in, resolves
# requirements.in and writes the frozen result to $RUNS_DIR/_bg/requirements.lock.txt (copy it into
# the repo and commit it). torch comes from the cu128 index (driver >= 525; sm_120 kernels).
# Everything lives in /tmp: uv (/tmp/uv-bin), its cache (/tmp/uv-cache), Python 3.10
# (/tmp/uv-python), the venv ($VENV_DIR, default /tmp/cvl-venv) and the build log. Concurrent
# callers serialize on a flock; a build that dies leaves no stamp, so the next call rebuilds.
# Progress goes to stderr, so it never mixes into the caller's stdout.
set -euo pipefail

# The repo on the cluster is an NFS mirror that remote.sh re-syncs; bash reads scripts lazily, so a
# sync during a long run kills it ("Stale file handle", 2026-11-13). Run from a private /tmp copy.
if [ -z "${CVL_SELF_COPY:-}" ]; then
  export CVL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  _copy="$(mktemp /tmp/cvl-script.XXXXXX)"; cp "${BASH_SOURCE[0]}" "$_copy"
  CVL_SELF_COPY="$_copy" exec bash "$_copy" "$@"
fi
_SELF="$CVL_SELF_COPY"; unset CVL_SELF_COPY   # don't leak into children (they get their own copy)
FROM_IN=0
[ "${1:-}" = "--from-in" ] && FROM_IN=1

cd "$CVL_REPO"
source scripts/env.sh >/dev/null

UV_VERSION=0.13.7
TORCH_BACKEND=cu128
export UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python
export UV_LINK_MODE=copy UV_CONCURRENT_DOWNLOADS=4 UV_NO_PROGRESS=1
UV=/tmp/uv-bin/uv

REQ=requirements.lock.txt
{ [ "$FROM_IN" = 1 ] || [ ! -f "$REQ" ]; } && REQ=requirements.in
STAMP="$(sha256sum "$REQ" | cut -c1-16) $REQ uv=$UV_VERSION torch=$TORCH_BACKEND"

ok() { [ -f "$VENV_DIR/.cvl-stamp" ] && [ "$(cat "$VENV_DIR/.cvl-stamp")" = "$STAMP" ]; }
if [ "$FROM_IN" = 0 ] && ok; then
  echo "ensure_venv: $VENV_DIR up to date ($REQ)" >&2
  rm -f "$_SELF"; exit 0
fi

exec 9>/tmp/cvl-venv.flock
flock 9                                   # wait for a concurrent build, then re-check
if [ "$FROM_IN" = 0 ] && ok; then
  echo "ensure_venv: $VENV_DIR built by a concurrent caller" >&2
  rm -f "$_SELF"; exit 0
fi

LOG=/tmp/cvl-venv-build.log
: > "$LOG"
say() { echo "ensure_venv: $*" | tee -a "$LOG" >&2; }
run() { "$@" >>"$LOG" 2>&1 || { rc=$?; tail -n 40 "$LOG" >&2; say "FAILED ($rc): $*"; exit "$rc"; }; }
copy_log() { mkdir -p "$RUNS_DIR/_bg" && cp "$LOG" "$RUNS_DIR/_bg/venv-build-$(hostname).log" 2>/dev/null || true; rm -f "$_SELF"; }
trap copy_log EXIT

t0=$(date +%s)
cache_state=$([ -n "$(ls -A "$UV_CACHE_DIR" 2>/dev/null)" ] && echo warm || echo cold)  # env.sh mkdirs it
say "building $VENV_DIR from $REQ on $(hostname) (uv cache $cache_state) at $(date -Is)"
df -h /tmp | tail -1 >>"$LOG"

if [ ! -x "$UV" ] || [ "$("$UV" --version | cut -d' ' -f2)" != "$UV_VERSION" ]; then
  mkdir -p /tmp/uv-bin
  curl -fsSL -m 120 --retry 3 "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" \
    | tar xz -C /tmp/uv-bin --strip-components=1
fi
t_uv=$(date +%s)

rm -rf "$VENV_DIR"
run "$UV" python install 3.10
run "$UV" venv "$VENV_DIR" --python 3.10
t_py=$(date +%s)

if [ "$REQ" = requirements.lock.txt ]; then
  run "$UV" pip sync --python "$VENV_DIR/bin/python" --torch-backend "$TORCH_BACKEND" "$REQ"
else
  run "$UV" pip install --python "$VENV_DIR/bin/python" --torch-backend "$TORCH_BACKEND" -r "$REQ"
fi
t_pkg=$(date +%s)

run "$VENV_DIR/bin/python" - <<'PY'
import sys
sys.path.insert(0, ".")
import torch, torchvision, stable_worldmodel, stable_pretraining, h5py, hdf5plugin, pymunk, pygame, sklearn
import datasets, transformers
from stable_pretraining.backbone.utils import vit_hf  # the HF checkpoint's encoder target
import jepa, module  # upstream LeWM imports cleanly
print("python", sys.version.split()[0], "| torch", torch.__version__, "cuda build", torch.version.cuda,
      "| transformers", transformers.__version__, "| pymunk", pymunk.version, "| imports OK")
PY
tail -n 1 "$LOG" >&2

if [ "$REQ" = requirements.in ]; then
  mkdir -p "$RUNS_DIR/_bg"
  "$UV" pip freeze --python "$VENV_DIR/bin/python" > "$RUNS_DIR/_bg/requirements.lock.txt"
  say "froze $(wc -l < "$RUNS_DIR/_bg/requirements.lock.txt") packages to \$RUNS_DIR/_bg/requirements.lock.txt"
  # Stamp it as built from the lock too, so the next call doesn't rebuild once it's committed.
  STAMP="$(sha256sum "$RUNS_DIR/_bg/requirements.lock.txt" | cut -c1-16) requirements.lock.txt uv=$UV_VERSION torch=$TORCH_BACKEND"
fi
echo "$STAMP" > "$VENV_DIR/.cvl-stamp"

t1=$(date +%s)
sizes=$(du -sh "$VENV_DIR" /tmp/uv-python "$UV_CACHE_DIR" 2>/dev/null | awk '{printf "%s=%s ", $2, $1}')
say "BUILD_OK total=$((t1 - t0))s (uv $((t_uv - t0))s, python+venv $((t_py - t_uv))s, packages $((t_pkg - t_py))s, check $((t1 - t_pkg))s) $sizes"
mkdir -p "$RUNS_DIR/_bg"
[ -f "$RUNS_DIR/_bg/venv-builds.tsv" ] || printf 'date\thost\treq\tcache\ttotal_s\tuv_s\tpython_s\tpackages_s\tcheck_s\n' > "$RUNS_DIR/_bg/venv-builds.tsv"
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$(hostname)" "$REQ" "$cache_state" "$((t1 - t0))" \
  "$((t_uv - t0))" "$((t_py - t_uv))" "$((t_pkg - t_py))" "$((t1 - t_pkg))" >> "$RUNS_DIR/_bg/venv-builds.tsv"
