# shellcheck shell=bash
# Sourced INSIDE DSMLP pods (remote.sh and GPU job scripts do this for you).
# Keeps everything big in ~/private (personal pool, same in every workspace), not in the
# small course-workspace home.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # mirror or a job snapshot
export PROJECT_ROOT
export CVL_STORE="${CVL_STORE:-$HOME/private}"

export STABLEWM_HOME="${STABLEWM_HOME:-$CVL_STORE/stablewm}"       # LeWM data + checkpoints
# Runs always land in the main mirror's runs/, even when a job runs from a snapshot, so a
# resubmitted job resumes from the same directory and sync.sh pull finds everything.
export RUNS_DIR="${RUNS_DIR:-$CVL_STORE/cvar-lewm/runs}"
export PODINFO="${PODINFO:-/etc/podinfo}"                          # termination-warning files

# The venv is ephemeral, in the pod's node-local /tmp (decision 2026-11-13): scripts/ensure_venv.sh
# builds it from requirements.lock.txt when a pod doesn't have it (scripts/withvenv.sh wraps that).
# uv, its cache and Python 3.10 live in /tmp too, so none of this touches the ~/private quota.
export VENV_DIR="${VENV_DIR:-/tmp/cvl-venv}"
export UV_CACHE_DIR="/tmp/uv-cache"
export UV_PYTHON_INSTALL_DIR="/tmp/uv-python"
export PIP_CACHE_DIR="$CVL_STORE/.cache/pip"
export HF_HOME="$CVL_STORE/.cache/huggingface"
export TORCH_HOME="$CVL_STORE/.cache/torch"
export MPLCONFIGDIR="$CVL_STORE/.cache/matplotlib"

# CPU threads. os.cpu_count() (and nproc, unless OMP_NUM_THREADS is set) report every core on
# the node (128-256 on DSMLP), not the pod's limit, so BLAS/OpenMP/torch would oversubscribe
# the 4 CPUs we actually get. Read the cgroup CPU quota instead: v2 cpu.max
# ("<quota> <period>" or "max <period>"), then v1 cfs_quota_us/cfs_period_us (-1 = unlimited),
# else 4. Rounded up. Pre-set values win.
_cvl_ncpu() {
  local root="${1:-/sys/fs/cgroup}" quota="" period=""
  if [ -r "$root/cpu.max" ]; then
    read -r quota period < "$root/cpu.max" || true
  fi
  if ! [[ "$quota" =~ ^[0-9]+$ ]] && [ -r "$root/cpu/cpu.cfs_quota_us" ] && [ -r "$root/cpu/cpu.cfs_period_us" ]; then
    quota="$(cat "$root/cpu/cpu.cfs_quota_us")"
    period="$(cat "$root/cpu/cpu.cfs_period_us")"
  fi
  if [[ "$quota" =~ ^[0-9]+$ && "$period" =~ ^[0-9]+$ ]] && [ "$quota" -gt 0 ] && [ "$period" -gt 0 ]; then
    echo $(( (quota + period - 1) / period ))
  else
    echo 4
  fi
}
export CVL_NCPU="${CVL_NCPU:-$(_cvl_ncpu)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$CVL_NCPU}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-$CVL_NCPU}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$CVL_NCPU}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-$CVL_NCPU}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
export TZ="America/Los_Angeles"

# Tools installed without root (e.g. uv), and the project root on PYTHONPATH so
# `python -m cvar...` and upstream modules import from any cwd.
export PATH="$HOME/.local/bin:/tmp/uv-bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$STABLEWM_HOME" "$RUNS_DIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"

# Only a finished build (ensure_venv.sh writes the stamp last); never a half-built venv.
if [ -f "$VENV_DIR/bin/activate" ] && [ -f "$VENV_DIR/.cvl-stamp" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi
