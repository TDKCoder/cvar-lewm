#!/usr/bin/env bash
# Transient download of the LeWM PushT expert dataset into the worker's node-local /tmp (NOT the
# ~/private quota), extract scaler stats + eval start/goal pairs into a small .npz under
# $STABLEWM_HOME, then delete everything in /tmp. Run INSIDE the worker pod:
#   scripts/remote.sh --bg fetch-expert "bash scripts/fetch_expert_stats.sh"            # extract
#   scripts/remote.sh --bg verify-expert "bash scripts/fetch_expert_stats.sh --verify"  # re-check npz
# --verify leaves the npz alone and runs cvar/data/verify_eval_states.py against the real h5
# (sklearn scalers, eval.py's literal sampling, stored rows re-read) -> .../verify.json.
#   scripts/remote.sh --bg expert-columns "bash scripts/fetch_expert_stats.sh --columns"
# --columns keeps every non-pixel column (+ excluded eval episodes, + pixels of the 50 upstream-eval
# episodes) in $STABLEWM_HOME/pusht/expert_columns.npz (cvar/data/extract_expert_columns.py).
# Sizes (2026-11-12): 13,136,247,974 B compressed; zstd frame content size 46,300,921,856 B.
set -euo pipefail
# The repo on the cluster is an NFS mirror that remote.sh re-syncs; bash reads scripts lazily, so a
# sync during a long run kills it ("Stale file handle", 2026-11-13). Run from a private /tmp copy.
if [ -z "${CVL_SELF_COPY:-}" ]; then
  export CVL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  _copy="$(mktemp /tmp/cvl-script.XXXXXX)"; cp "${BASH_SOURCE[0]}" "$_copy"
  CVL_SELF_COPY="$_copy" exec bash "$_copy" "$@"
fi
_SELF="$CVL_SELF_COPY"; unset CVL_SELF_COPY   # don't leak into children (they get their own copy)
MODE=extract
[ "${1:-}" = "--verify" ] && MODE=verify
[ "${1:-}" = "--columns" ] && MODE=columns
cd "$CVL_REPO"
source scripts/env.sh

URL="https://huggingface.co/datasets/quentinll/lewm-pusht/resolve/main/pusht_expert_train.h5.zst"
ZST_BYTES=13136247974
H5_BYTES=46300921856
W=/tmp/cvl-data
OUT="$STABLEWM_HOME/pusht/expert_eval_states.npz"
SUMMARY="$RUNS_DIR/phase0-expert-extract/summary.json"

cleanup() { rc=$?; rm -rf "$W" /tmp/cvl-probe "$_SELF"; echo "== cleanup: removed $W and /tmp/cvl-probe; exit rc=$rc at $(date -Is)"; }
trap cleanup EXIT
mkdir -p "$W"

need=$(( (ZST_BYTES + H5_BYTES) / 1024 + 5 * 1024 * 1024 ))   # KiB, +5 GiB margin
avail=$(df -Pk /tmp | awk 'NR==2 {print $4}')
echo "== /tmp avail ${avail} KiB, need ${need} KiB"
[ "$avail" -gt "$need" ] || { echo "not enough room in /tmp"; exit 1; }

echo "== download $(date -Is)"
curl -fL --retry 5 --retry-delay 10 -C - -sS -o "$W/expert.h5.zst" "$URL"
got=$(stat -c %s "$W/expert.h5.zst"); echo "downloaded $got B $(date -Is)"
[ "$got" = "$ZST_BYTES" ] || { echo "size mismatch (want $ZST_BYTES)"; exit 1; }

echo "== decompress $(date -Is)"
zstd -d --rm -q -o "$W/pusht_expert_train.h5" "$W/expert.h5.zst"   # --rm drops the .zst on success
got=$(stat -c %s "$W/pusht_expert_train.h5"); echo "decompressed $got B $(date -Is)"
[ "$got" = "$H5_BYTES" ] || { echo "size mismatch (want $H5_BYTES)"; exit 1; }

# The project venv (if env.sh activated it) has h5py + hdf5plugin + sklearn. The image's python has
# h5py but not hdf5plugin; then add it inside /tmp only if a filter is missing.
python3 -c "import hdf5plugin, sklearn" 2>/dev/null && echo "using $(command -v python3) (has hdf5plugin, sklearn)"
if ! python3 -c "import h5py,sys
try: import hdf5plugin
except ImportError: pass
f=h5py.File(sys.argv[1]); [f[k][:1] for k in f]" "$W/pusht_expert_train.h5" 2>"$W/h5err"; then
  echo "plain h5py can't read every column ($(tail -n1 "$W/h5err")); installing hdf5plugin into $W/pylib"
  # --no-deps: pulling h5py/numpy 2 into PYTHONPATH shadowed the image's numpy 1.26 and broke the
  # sklearn cross-check on 2026-11-12. Only `pixels` uses a filter (32001 = Blosc).
  python3 -m pip install -q --no-cache-dir --no-deps --target "$W/pylib" hdf5plugin
  export PYTHONPATH="$W/pylib:$PYTHONPATH"
fi

if [ "$MODE" = verify ]; then
  echo "== verify $(date -Is)"
  python3 -m cvar.data.verify_eval_states --h5 "$W/pusht_expert_train.h5" --npz "$OUT" \
    --out "$(dirname "$SUMMARY")/verify.json"
  echo "FETCH_VERIFY_OK"
  exit 0
fi
if [ "$MODE" = columns ]; then
  echo "== columns $(date -Is)"
  COLS="$STABLEWM_HOME/pusht/expert_columns.npz"
  python3 -m cvar.data.extract_expert_columns --h5 "$W/pusht_expert_train.h5" --eval-npz "$OUT" \
    --out "$COLS" --summary "$RUNS_DIR/phase0-expert-columns/summary.json"
  ls -l "$COLS"; du -sh "$STABLEWM_HOME"
  echo "FETCH_COLUMNS_OK"
  exit 0
fi
echo "== extract $(date -Is)"
python3 -m cvar.data.extract_eval_states --h5 "$W/pusht_expert_train.h5" --out "$OUT" --summary "$SUMMARY"
ls -l "$OUT"
echo "FETCH_OK"
