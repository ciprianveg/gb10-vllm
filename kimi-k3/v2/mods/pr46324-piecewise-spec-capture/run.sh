#!/usr/bin/env bash
# pr46324-piecewise-spec-capture — vLLM PR #46324: align spec-decode CUDA
# graph capture sizes in PIECEWISE mode (previously only FULL mode applied
# adjust_cudagraph_sizes_for_spec_decode, so PIECEWISE silently misaligned
# spec-decode steps and they fell back to eager).
#
# Applies the upstream PR diff via git apply. Idempotent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d /opt/kimi-k3/vllm/.git ]; then
    VLLM_ROOT=/opt/kimi-k3/vllm
elif [ -d /usr/local/lib/python3.12/dist-packages/vllm ]; then
    VLLM_ROOT=/usr/local/lib/python3.12/dist-packages
else
    echo "[pr46324] ERROR: cannot locate vllm tree" >&2
    exit 1
fi

cd "$VLLM_ROOT"

if git apply --reverse --check --include='vllm/*' "$SCRIPT_DIR/patch.diff" >/dev/null 2>&1; then
    echo "[pr46324] already applied, skipping"
elif git apply --check --include='vllm/*' "$SCRIPT_DIR/patch.diff" >/dev/null 2>&1; then
    git apply --include='vllm/*' "$SCRIPT_DIR/patch.diff"
    echo "[pr46324] patch applied"
else
    echo "[pr46324] ERROR: patch does not apply cleanly to this tree" >&2
    exit 1
fi

python3 - "$VLLM_ROOT" <<'PYEOF'
import py_compile, sys
from pathlib import Path
p = Path(sys.argv[1]) / "vllm/config/compilation.py"
if p.exists():
    py_compile.compile(str(p), doraise=True)
print("[pr46324] syntax check OK")
PYEOF

echo "=== pr46324-piecewise-spec-capture complete ==="
