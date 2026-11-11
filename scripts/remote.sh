#!/usr/bin/env bash
# Run a command in the CPU worker pod, from the repo mirror, with scripts/env.sh sourced.
#
#   scripts/remote.sh "python -m pytest -q tests/"
#   scripts/remote.sh --bg collect-p015 "python -m cvar.data.collect --p 0.15"   # detached
#   scripts/remote.sh --no-sync "tail -n 30 runs/_bg/collect-p015.log"
#
# Pushes the laptop repo first (unless --no-sync), so the pod always runs what's on the laptop.
# Exit code is the command's. Detached jobs log to runs/_bg/NAME.log and survive this call,
# but not the worker pod's 12 h limit.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SYNC=1
BG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --no-sync) SYNC=0; shift ;;
    --bg) BG="${2:?--bg needs a NAME}"; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: remote.sh [--no-sync] [--bg NAME] \"<command>\"" >&2; exit 2; }
CMD="$*"

require_connection

phase="$(worker_phase)"
if [ "$phase" != "Running" ]; then
  echo "Worker pod '$WORKER_NAME' is ${phase:-not present}. Start it with: scripts/worker.sh start" >&2
  exit 3
fi

if [ "$SYNC" = 1 ]; then
  "$REPO_ROOT/scripts/sync.sh" push
fi

if [ -n "$BG" ]; then
  [[ "$BG" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "bad --bg name '$BG'" >&2; exit 2; }
  CMD="mkdir -p runs/_bg && nohup bash -c $(printf '%q' "$CMD") > runs/_bg/$BG.log 2>&1 < /dev/null & echo \"started $BG (pid \$!), log: runs/_bg/$BG.log\""
fi

# Ship the command base64-encoded so no quoting survives three shells (laptop, login, pod).
B64="$(printf '%s' "$CMD" | base64 | tr -d '\n')"
POD_SCRIPT="cd $POD_ROOT && source scripts/env.sh && eval \"\$(echo $B64 | base64 -d)\""
dsmlp_ssh "kubectl exec $WORKER_NAME -- bash -c $(printf '%q' "$POD_SCRIPT")"
