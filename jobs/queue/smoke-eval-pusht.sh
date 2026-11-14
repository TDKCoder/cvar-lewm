#!/usr/bin/env bash
# JOB-META class=medium est_hours=0.6 desc="Phase 0 GPU smoke: get_cost/encode/collect timings, 10-ep upstream eval, closed-loop kick check (250 ep)"
#
# Prepared by Claude Code on 2026-11-13; code at commit 27e799e (this script is committed right after).
# CPU smoke test:  SMOKE=1 bash jobs/queue/smoke-eval-pusht.sh   (worker pod, CEM 30x3, 2 pairs)
#                  -> all 3 phases exit 0 in 55 s; upstream 2 ep, kickcheck 10 ep, replan every 25
#                  env steps; drift.json on CPU: first-plan actions identical across all 5
#                  conditions, pair-0 repeat identical (max |diff| 0.0); 51/51 pytest pass.
# Resume test:     SMOKE=1 timeout [--foreground] -s TERM 45 bash ... (TERM to the script only, and
#                  to the whole process group): kickcheck stopped between episodes with 4/10 logged,
#                  phase exit 75; rerun skipped timing + upstream, ran the 6 missing episodes, 10 total,
#                  no duplicates.
#
# Phases, in order; each one resumes on resubmission, and a failed phase does not stop the next:
#   1. timing    cvar/eval/gpu_timing.py  -> timing/timing.json
#                get_cost at candidate batch 300/1200/2400/4800 (S = 1/4/8/16), median of 10 after
#                warm-up; encoder frames/s; collect-and-encode in one pass (env on CPU, encoder on
#                GPU); env-only rate with CVL_NCPU processes; fp16 latent rounding check.
#   2. upstream  cvar/eval/closed_loop.py --phase upstream: pretrained LeWM, upstream's eval config,
#                p = 0, upstream's seed-42 pairs 0-9 with the dataset's own frames.
#   3. kickcheck cvar/eval/closed_loop.py --phase kickcheck: 50 of our seed-0 pairs x {p0,
#                p0.05/p0.15 x medium/large kicks} = 250 episodes, same kick + CEM seeds per pair in
#                every condition; then drift.json (planner CRN check on GPU: reported, not enforced).
# Exit codes: 0 all phases ok; 75 interrupted (SIGTERM / termination warning), resubmit to resume;
# 1 a phase failed (the others still ran; see log.txt).
#
# SU estimate (medium: 1 SU/GPU-h peak, 0.5 off-peak 1 AM-4 PM Pacific):
#   CPU reference: one default plan call (300 x 30 CEM) = 23.9 s on the 4-CPU worker
#   (results/phase0/pretrained_cpu.json). GPU plan time is not measured yet (phase 1 measures it);
#   ASSUMED 2 s per plan call (12x the CPU; deliberately pessimistic for a 24 GB GPU and a 18 M-
#   parameter model).
#   episodes: 10 + 250 + 1 drift repeat = 261, at most 2 plan calls each (replan every 25 env steps,
#   50-step budget) -> <= 522 plan calls x 2 s = 17.4 min
#   env stepping + 224 px render: 261 x 50 steps / ~400 steps/s = 0.5 min; 3 model loads ~1 min
#   phase 1 (timing): ~2 min;  venv cold build in the fresh pod: ~2.5 min (measured 143-146 s)
#   pod start / image pull: ~2 min (unmeasured allowance)
#   total ~25.4 min x 1.25 buffer = 31.8 min ~ 0.53 h -> est_hours 0.6
#   => ~0.6 SU peak / ~0.3 SU off-peak. If plans take 1 s: ~0.4 h -> 0.4 / 0.2 SU.
#
# Rohan submits it from the laptop with: scripts/submit_job.sh jobs/queue/smoke-eval-pusht.sh
# The GPU pod runs the copy inside a code snapshot (cvl-snapshots/<name>/jobs/submitted/),
# so cd to that snapshot's root. stdout/stderr go to the mirror's jobs/logs/<name>.out.
# SMOKE=1 runs the same phases on CPU with tiny settings (worker-pod test), in runs/smoke-job-eval-pusht.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source scripts/env.sh     # RUNS_DIR points at the main mirror's runs/, shared across snapshots

SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = 1 ]; then
  RUN_ID="smoke-job-eval-pusht"; DEVICE=cpu; FLAGS=(--smoke)
else
  RUN_ID="phase0-gpu-eval-pusht"; DEVICE=cuda; FLAGS=()   # fixed: a resubmission resumes this dir
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

run_phase timing    python -u -m cvar.eval.gpu_timing  --out "$RUN_DIR/timing" --device "$DEVICE" ${FLAGS[@]+"${FLAGS[@]}"}
run_phase upstream  python -u -m cvar.eval.closed_loop --phase upstream  --run-dir "$RUN_DIR" --device "$DEVICE" --resume auto ${FLAGS[@]+"${FLAGS[@]}"}
run_phase kickcheck python -u -m cvar.eval.closed_loop --phase kickcheck --run-dir "$RUN_DIR" --device "$DEVICE" --resume auto ${FLAGS[@]+"${FLAGS[@]}"}

echo "=== $(date -Is) done; failed phases: ${failed[*]:-none}" | tee -a "$RUN_DIR/env.txt"
[ ${#failed[@]} -eq 0 ] || exit 1
