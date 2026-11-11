#!/usr/bin/env bash
# Move files between the laptop repo (source of truth) and the cluster mirror.
#
#   scripts/sync.sh push                 mirror code + .git to the cluster (run before remote work;
#                                        remote.sh and submit_job.sh do it for you)
#   scripts/sync.sh pull [RUN_GLOB]      copy small result files from cluster runs/ back to runs/ here,
#                                        plus jobs/logs/  (default glob: '*')
#   scripts/sync.sh put LOCAL [DEST]     copy a file/dir into the cluster store (STABLEWM_HOME),
#                                        e.g. a checkpoint downloaded on the laptop
#
# push never deletes cluster-only outputs: runs/, jobs/logs/ and the venv are excluded.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_connection

RSH="ssh ${SSH_OPTS[*]}"
cmd="${1:-push}"

case "$cmd" in
  push)
    rsync -az --delete -e "$RSH" \
      --exclude '/runs/' --exclude '/jobs/logs/' --exclude '/.venv*/' \
      --exclude '/scripts/dsmlp.conf' --exclude '__pycache__/' --exclude '.ipynb_checkpoints/' \
      --exclude '*.ckpt' --exclude '*.pt' --exclude '*.h5' --exclude '*.zst' \
      "$REPO_ROOT/" "$SSH_HOST:$REMOTE_ROOT/"
    ;;
  pull)
    glob="${2:-*}"
    [[ "$glob" =~ ^[A-Za-z0-9._*?-]+$ ]] || { echo "bad run glob '$glob'" >&2; exit 2; }
    mkdir -p "$REPO_ROOT/runs" "$REPO_ROOT/jobs/logs"
    # Small files only: logs, per-episode/metric jsonl, summaries, figures. No checkpoints.
    rsync -az -e "$RSH" --prune-empty-dirs \
      --include '*/' --include 'episodes.jsonl' --include 'metrics.jsonl' --include 'env.txt' \
      --include 'log.txt' --include '*.json' --include '*.csv' --include '*.png' --include '*.pdf' \
      --exclude '*' \
      "$SSH_HOST:$REMOTE_ROOT/runs/$glob" "$REPO_ROOT/runs/" || {
        echo "nothing matched runs/$glob on the cluster (or transfer failed)" >&2; exit 1; }
    rsync -az -e "$RSH" "$SSH_HOST:$REMOTE_ROOT/jobs/logs/" "$REPO_ROOT/jobs/logs/" 2>/dev/null || true
    echo "pulled runs/$glob and jobs/logs/"
    ;;
  put)
    src="${2:?usage: sync.sh put LOCAL [DEST under the store]}"
    dest="${3:-}"
    size="$(du -sh "$src" | cut -f1)"
    echo "copying $src ($size) -> cluster:$STORE_REMOTE/$dest"
    dsmlp_ssh "mkdir -p $(quote_args "$STORE_REMOTE/$dest")"
    rsync -az --info=progress2 -e "$RSH" "$src" "$SSH_HOST:$STORE_REMOTE/$dest"
    ;;
  *)
    echo "usage: sync.sh push | pull [RUN_GLOB] | put LOCAL [DEST]" >&2; exit 2 ;;
esac
