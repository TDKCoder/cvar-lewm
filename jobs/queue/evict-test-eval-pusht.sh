#!/usr/bin/env bash
# JOB-META class=medium est_hours=1.0 desc="Eviction test: kick check on our pairs 50-99 (250 ep); Rohan deletes the pod mid-run, then resubmits"
#
# Prepared by Claude Code on 2026-11-13; code at commit 331b09c (this script is committed right after).
# Purpose: test the real eviction path before the first multi-hour GPU job (docs/EXPERIMENTS.md,
# Phase 0 open item 3), and while doing it, collect useful data: the same 5 kick-check conditions on
# our seed-0 pairs 50-99 (extra paired data; reported separately from pairs 0-49 unless Rohan says pool).
#
# Procedure:
#   1. Rohan submits:  scripts/submit_job.sh jobs/queue/evict-test-eval-pusht.sh
#   2. Claude watches runs/phase0-evict-test/kickcheck/episodes.jsonl from the worker; once >= 10
#      episodes are logged, Rohan deletes the pod:  kubectl delete pod cvl-evict-test-eval-pusht
#   3. Pass if env.txt shows "pid1: ...tini..." with this script as pid 1's child (ppid=1),
#      "SIGTERM: forwarding", "phase kickcheck exit 75", "interrupted"; log.txt shows
#      "[resilience] SIGTERM received" and "stopping between episodes".
#   4. git mv jobs/submitted/evict-test-eval-pusht.sh jobs/queue/ ; Rohan resubmits. Pass if the run
#      logs "N already logged", finishes the rest, and report-results' duplicate check passes.
# CPU smoke test:  SMOKE=1 bash jobs/queue/evict-test-eval-pusht.sh   (worker pod, CEM 30x3, pairs 50-51)
#                  -> exit 0; 10 episodes; drift exact on CPU; env.txt has the pid1/self lines; 51/51 pytest.
#                  The SIGTERM/resume path is the same code as smoke-eval-pusht's (tested there).
#
# SU estimate (medium: 1 SU/GPU-h peak, 0.5 off-peak 1 AM-4 PM Pacific); ASSUMED 2 s per GPU plan
# call (unmeasured until smoke-eval-pusht reports; CPU: 23.9 s):
#   250 episodes x <= 2 plan calls x 2 s = 16.7 min; env + render 0.4 min; model load 0.3 min;
#   per pod: venv cold build ~2.5 min + pod start ~2 min, and there are 2 pods (kill + resubmit) = 9 min
#   total ~26.4 min x 1.25 = 33 min ~ 0.55 h over both submissions -> ~0.6 SU peak / ~0.3 off-peak
#   expected; est_hours 1.0 per submission in JOB-META for headroom (billing is for time used).
#
# The GPU pod runs the copy inside a code snapshot (cvl-snapshots/<name>/jobs/submitted/).
# SMOKE=1 runs it on CPU with tiny settings (worker-pod test), in runs/smoke-job-evict-test.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh     # RUNS_DIR points at the main mirror's runs/, shared across snapshots

SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = 1 ]; then
  RUN_ID="smoke-job-evict-test"; DEVICE=cpu; FLAGS=(--smoke)
else
  RUN_ID="phase0-evict-test"; DEVICE=cuda; FLAGS=()   # fixed: a resubmission resumes this dir
fi
RUN_DIR="$RUNS_DIR/$RUN_ID"
mkdir -p "$RUN_DIR"

# Ephemeral venv in this pod's /tmp from the snapshot's requirements.lock.txt (decision 2026-11-13).
t_venv=$(date +%s)
bash scripts/ensure_venv.sh
venv_s=$(( $(date +%s) - t_venv ))
source scripts/env.sh     # now activates $VENV_DIR

NVSMI="$(command -v nvidia-smi || echo /usr/local/nvidia/bin/nvidia-smi)"
{
  echo "=== $(date -Is) start on $(hostname) (SMOKE=$SMOKE)"
  # Signal path (decision 2026-11-13): an eviction's SIGTERM goes to PID 1 (tini), which signals its
  # child; the job script must be that child (pid 1's child, ppid 1) for its TERM trap to run.
  echo "pid1: $(tr '\0' ' ' < /proc/1/cmdline 2>/dev/null || echo unreadable)"
  echo "self: pid=$$ ppid=$PPID pgid=$(awk '{print $5}' /proc/$$/stat) sid=$(awk '{print $6}' /proc/$$/stat)"
  echo "venv: ensure_venv.sh took ${venv_s}s ($(cat "$VENV_DIR/.cvl-stamp"))"
  echo "gpu: $("$NVSMI" --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>&1 || true)"
  "$NVSMI" 2>&1 || true
  echo "git: $(git rev-parse HEAD 2>/dev/null || echo none) dirty=$(git status --porcelain 2>/dev/null | wc -l)"
  echo "CVL_NCPU=$CVL_NCPU"
  python -c 'import torch,sys; print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)'
} >> "$RUN_DIR/env.txt" 2>&1

if [ "$SMOKE" != 1 ]; then
  # Refuse to burn GPU time if CUDA isn't visible, and fail fast on an arch mismatch (e.g. a GPU
  # newer than the wheel's kernels): a real kernel launch, not just is_available().
  python -c '
import torch
assert torch.cuda.is_available(), "CUDA not available in this pod"
a = torch.randn(256, 256, device="cuda"); b = (a @ a).sum().item()
print("cuda matmul ok:", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0), "sum", round(b, 3))
' 2>&1 | tee -a "$RUN_DIR/env.txt"
fi

# bash does not pass SIGTERM to a foreground child: forward it, so python's InterruptGuard stops
# cleanly between episodes (exit 75) within the 600 s eviction grace period.
child=""
trap 'echo "=== $(date -Is) SIGTERM: forwarding to pid ${child:-none}" >> "$RUN_DIR/env.txt"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null || true' TERM

failed=()
run_phase() {
  local name="$1"; shift
  echo "=== $(date -Is) phase $name: $*" | tee -a "$RUN_DIR/log.txt"
  set +e
  # tee ignores TERM (inherited across exec), so the log pipe outlives a group-wide TERM and python's
  # handler can still print; otherwise python dies on a broken pipe with exit 1 (seen in the resume test).
  "$@" > >(trap '' TERM; exec tee -a "$RUN_DIR/log.txt") 2>&1 &
  child=$!
  wait "$child"; local rc=$?
  while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done   # wait returned early on a trapped TERM
  set -e
  child=""
  echo "=== $(date -Is) phase $name exit $rc" | tee -a "$RUN_DIR/log.txt" "$RUN_DIR/env.txt"
  if [ "$rc" = 75 ] || [ "$rc" = 143 ]; then
    echo "=== interrupted in phase $name; resubmit to resume" | tee -a "$RUN_DIR/env.txt"
    exit 75
  fi
  [ "$rc" = 0 ] || failed+=("$name:$rc")
}

run_phase kickcheck python -u -m cvar.eval.closed_loop --phase kickcheck --run-dir "$RUN_DIR" --device "$DEVICE" --resume auto \
  --pairs 50 --pair-offset 50 ${FLAGS[@]+"${FLAGS[@]}"}

echo "=== $(date -Is) done; failed phases: ${failed[*]:-none}" | tee -a "$RUN_DIR/env.txt"
[ ${#failed[@]} -eq 0 ] || exit 1
