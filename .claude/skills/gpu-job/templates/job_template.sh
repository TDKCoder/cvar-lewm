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

# Ephemeral venv in this pod's /tmp from the snapshot's requirements.lock.txt (decision 2026-11-13).
# Cold build measured at 143 s on a CPU worker; the real time per job goes into env.txt below.
t_venv=$(date +%s)
bash scripts/ensure_venv.sh
venv_s=$(( $(date +%s) - t_venv ))
source scripts/env.sh     # now activates $VENV_DIR

# Fixed per job so a resubmission resumes the same run directory.
RUN_ID="REPLACE-job-name"
RUN_DIR="$RUNS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

NVSMI="$(command -v nvidia-smi || echo /usr/local/nvidia/bin/nvidia-smi)"
{
  echo "=== $(date -Is) start on $(hostname)"
  # Signal path (decision 2026-11-13): an eviction's SIGTERM goes to PID 1 (tini), which signals its
  # child; the job script must be that child (pid 1's child, ppid 1) for its TERM trap to run.
  echo "pid1: $(tr '\0' ' ' < /proc/1/cmdline 2>/dev/null || echo unreadable)"
  echo "self: pid=$$ ppid=$PPID pgid=$(awk '{print $5}' /proc/$$/stat) sid=$(awk '{print $6}' /proc/$$/stat)"
  echo "venv: ensure_venv.sh took ${venv_s}s ($(cat "$VENV_DIR/.cvl-stamp"))"
  "$NVSMI" || true
  echo "git: $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"
  python -c 'import torch,sys; print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
} >> "$RUN_DIR/env.txt" 2>&1

# Refuse to burn GPU time if CUDA isn't visible.
python -c 'import torch; assert torch.cuda.is_available(), "CUDA not available in this pod"'

# bash does not pass SIGTERM (eviction) to a foreground child: run python in the background, forward
# TERM to it, and wait for its real exit code. tee ignores TERM so the log pipe survives a group-wide
# TERM (otherwise python dies on a broken pipe with exit 1). Tested in jobs/queue/smoke-eval-pusht.sh.
child=""
trap 'echo "=== $(date -Is) SIGTERM: forwarding to pid ${child:-none}" >> "$RUN_DIR/env.txt"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null || true' TERM

# Main command. --resume auto must make a rerun continue where it stopped.
set +e
python -u -m cvar.REPLACE_ENTRYPOINT \
  --run-dir "$RUN_DIR" \
  --resume auto \
  REPLACE_ARGS \
  > >(trap '' TERM; exec tee -a "$RUN_DIR/log.txt") 2>&1 &
child=$!
wait "$child"; rc=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
set -e

echo "=== $(date -Is) done, exit $rc" >> "$RUN_DIR/env.txt"
[ "$rc" = 143 ] && rc=75   # killed by TERM before its handler ran: still resumable
exit "$rc"
