# shellcheck shell=bash
# Shared setup for the laptop-side scripts. Sourced, not run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONF="$REPO_ROOT/scripts/dsmlp.conf"
if [ ! -f "$CONF" ]; then
  echo "Missing $CONF (copy scripts/dsmlp.conf.example and fill it in)" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONF"

: "${SSH_HOST:?set SSH_HOST in dsmlp.conf}"
: "${REMOTE_ROOT:?set REMOTE_ROOT in dsmlp.conf}"
: "${POD_ROOT:?set POD_ROOT in dsmlp.conf}"
WORKER_NAME="${WORKER_NAME:-cvl-worker}"
LAUNCH="${LAUNCH:-/opt/launch-sh/bin/launch.sh}"

# BatchMode: never prompt for a password or Duo. If the shared connection is down, fail fast.
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20)

# Run one command string on the login node. DSMLP only sets up kubectl/workspace credentials in
# login shells (a plain `ssh dsmlp kubectl ...` hits localhost:8080), so wrap it in `bash -lc`.
# rsync doesn't need this and calls ssh directly.
dsmlp_ssh() {
  ssh "${SSH_OPTS[@]}" "$SSH_HOST" "bash -lc $(printf '%q' "$*")"
}

require_connection() {
  if ! ssh -O check "$SSH_HOST" >/dev/null 2>&1; then
    {
      echo "No open SSH connection to '$SSH_HOST'."
      echo "Rohan: in your own terminal run   ssh $SSH_HOST true   (password + Duo once; lasts 8 h)."
    } >&2
    exit 255
  fi
}

# Answers come back on marker lines, so anything a login profile prints is ignored.
worker_phase() {
  { dsmlp_ssh "echo CVL_PHASE=\$(kubectl get pod $WORKER_NAME -o jsonpath='{.status.phase}' 2>/dev/null)" || true; } \
    | sed -n 's/^CVL_PHASE=//p' | tail -n 1
}

pod_exists() {
  { dsmlp_ssh "kubectl get pod $1 >/dev/null 2>&1 && echo CVL_EXISTS" || true; } | grep -q '^CVL_EXISTS$'
}

# Quote an array of arguments for safe use inside a remote shell command string.
quote_args() {
  local out="" a
  for a in "$@"; do out+="$(printf '%q' "$a") "; done
  printf '%s' "${out% }"
}
