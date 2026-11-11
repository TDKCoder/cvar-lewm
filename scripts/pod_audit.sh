#!/usr/bin/env bash
# Runs INSIDE the worker pod (via: scripts/remote.sh "bash scripts/pod_audit.sh").
# Prints what the pod has and can reach. Changes nothing.
set -uo pipefail

section() { echo; echo "== $*"; }
section "identity";  echo "user=$(whoami) host=$(hostname) date=$(date -Is) home=$HOME"
section "cpu/ram";   echo "nproc=$(nproc)"; free -g 2>/dev/null || true
section "gpu";       (command -v nvidia-smi >/dev/null && nvidia-smi -L) || echo "no GPU visible (expected in the worker)"
section "python";    command -v python3; python3 --version 2>&1
python3 - <<'EOF' 2>&1
try:
    import torch; print("image torch", torch.__version__, "cuda build", torch.version.cuda)
except Exception as e:
    print("image torch: not importable:", e)
try:
    import numpy; print("image numpy", numpy.__version__)
except Exception as e:
    print("image numpy: not importable:", e)
EOF
section "tools"
for t in git uv pip conda mamba curl wget rsync; do printf '%-6s %s\n' "$t" "$(command -v "$t" || echo MISSING)"; done
section "proxy env"
env | grep -i -E '^(https?|no)_proxy=' || echo "(none)"
section "disk"
df -h "$HOME" "$HOME/private" 2>/dev/null || true
echo "home usage (excluding private):"; du -sh --exclude="$HOME/private" "$HOME" 2>/dev/null || true
echo "private usage:"; du -sh "$HOME/private" 2>/dev/null || true
section "network (HTTP status; 000 = unreachable)"
for u in https://pypi.org/simple/ https://files.pythonhosted.org https://huggingface.co https://cdn-lfs.huggingface.co \
         https://github.com https://objects.githubusercontent.com https://astral.sh https://download.pytorch.org \
         https://conda.anaconda.org https://api.anthropic.com; do
  printf '%-40s %s\n' "$u" "$(curl -s -o /dev/null -m 10 -w '%{http_code}' "$u")"
done
section "podinfo"
ls -la /etc/podinfo 2>/dev/null || echo "no /etc/podinfo"
