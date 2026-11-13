#!/usr/bin/env bash
# Run a command inside the project venv, building it first if this pod doesn't have it yet:
#   scripts/remote.sh "scripts/withvenv.sh python -m pytest -q tests/"
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$here/ensure_venv.sh"
source "$here/env.sh"   # activates $VENV_DIR now that it exists
exec "$@"
