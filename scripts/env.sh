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

# The venv lives outside the repo so job snapshots share it.
export VENV_DIR="${VENV_DIR:-$CVL_STORE/.venvs/cvar-lewm}"
export UV_CACHE_DIR="$CVL_STORE/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$CVL_STORE/.uv-python"
export PIP_CACHE_DIR="$CVL_STORE/.cache/pip"
export HF_HOME="$CVL_STORE/.cache/huggingface"
export TORCH_HOME="$CVL_STORE/.cache/torch"
export MPLCONFIGDIR="$CVL_STORE/.cache/matplotlib"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
export TZ="America/Los_Angeles"

# Tools installed without root (e.g. uv), and the project root on PYTHONPATH so
# `python -m cvar...` and upstream modules import from any cwd.
export PATH="$HOME/.local/bin:$CVL_STORE/tools/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$STABLEWM_HOME" "$RUNS_DIR" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" "$MPLCONFIGDIR"

if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
fi
