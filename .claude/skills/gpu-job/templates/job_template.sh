#!/usr/bin/env bash
# JOB-META class=medium est_hours=1.0 desc="REPLACE: one-line purpose (hypothesis / grid cell)"
#
# Prepared by Claude Code on REPLACE-DATE at commit REPLACE-SHA.
# CPU smoke test:  REPLACE-COMMAND --smoke   -> REPLACE-RESULT
# Resume test:     killed with SIGTERM at REPLACE, rerun resumed at REPLACE
#
# Rohan submits it from the laptop with: scripts/submit_job.sh jobs/queue/<this file>
# The GPU pod runs the copy inside a code snapshot (cvl-snapshots/<name>/jobs/submitted/),
# so cd to that snapshot's root. stdout/stderr go to the mirror's jobs/logs/<name>.out.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh     # RUNS_DIR points at the main mirror's runs/, shared across snapshots

# Fixed per job so a resubmission resumes the same run directory.
RUN_ID="REPLACE-job-name"
RUN_DIR="$RUNS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

NVSMI="$(command -v nvidia-smi || echo /usr/local/nvidia/bin/nvidia-smi)"
{
  echo "=== $(date -Is) start on $(hostname)"
  "$NVSMI" || true
  echo "git: $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"
  python -c 'import torch,sys; print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
} >> "$RUN_DIR/env.txt" 2>&1

# Refuse to burn GPU time if CUDA isn't visible.
python -c 'import torch; assert torch.cuda.is_available(), "CUDA not available in this pod"'

# Main command. --resume auto must make a rerun continue where it stopped.
python -u -m cvar.REPLACE_ENTRYPOINT \
  --run-dir "$RUN_DIR" \
  --resume auto \
  REPLACE_ARGS \
  2>&1 | tee -a "$RUN_DIR/log.txt"

echo "=== $(date -Is) done" >> "$RUN_DIR/env.txt"
