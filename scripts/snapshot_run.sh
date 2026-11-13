#!/usr/bin/env bash
# Run a long worker job from a frozen /tmp copy of the repo, so a later remote.sh sync of the NFS
# mirror can't change code under it (decision 2026-11-13). GPU jobs get the same from submit_job.sh's
# snapshot; this is the worker-pod equivalent:
#
#   scripts/remote.sh --bg <name> "bash scripts/snapshot_run.sh <name> python -m cvar.... --args"
#
# The snapshot is /tmp/cvl-snap/<name>-<YYYYmmdd-HHMMSS>/ (the mirror minus runs/ and caches, ~7 MB),
# with SNAPSHOT.txt (source, git SHA, dirty-file count, command). The command runs there through
# scripts/withvenv.sh, with PYTHONPATH = the snapshot only. Outputs still go to $RUNS_DIR (the
# mirror's runs/), which env.sh sets independently of the code location.
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: snapshot_run.sh <name> <command...>" >&2; exit 2; }
NAME="$1"; shift

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SNAP="/tmp/cvl-snap/$NAME-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$SNAP"
tar -C "$SRC" --exclude=./runs --exclude=./jobs/logs --exclude='__pycache__' --exclude='.ipynb_checkpoints' \
  -cf - . | tar -C "$SNAP" -xf -

{
  echo "snapshot: $SNAP"
  echo "source:   $SRC at $(date -Is) on $(hostname)"
  echo "git:      $(git -C "$SNAP" rev-parse HEAD 2>/dev/null || echo none) dirty=$(git -C "$SNAP" status --porcelain 2>/dev/null | wc -l)"
  echo "command:  $*"
} | tee "$SNAP/SNAPSHOT.txt" >&2

cd "$SNAP"
unset PYTHONPATH PROJECT_ROOT   # drop the mirror; env.sh (via withvenv.sh) points both at the snapshot
exec bash scripts/withvenv.sh "$@"
