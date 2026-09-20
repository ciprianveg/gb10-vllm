#!/usr/bin/env bash
# pr51508-stale-zero-accept — vLLM PR #51508: skip GDN/KDA recurrent state
# updates for stale (zero-accept) spec-decode rows.
#
# When a speculative request accepts 0 tokens, kernels computed offset
# num_accepted_tokens - 1 = -1 (underflow) → garbage recurrent-state
# reads/writes. The patch clamps the offset to >= 0 in the Triton kernels
# and nulls stale rows' state indices to NULL_BLOCK_ID on the Python side.
#
# Applies the upstream PR diff (vllm/ files only) via git apply.
# Idempotent: skips when already applied.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Locate the vllm source tree (new sm121 image layout vs old dist-packages)
if [ -d /opt/kimi-k3/vllm/.git ]; then
    VLLM_ROOT=/opt/kimi-k3/vllm
elif [ -d /usr/local/lib/python3.12/dist-packages/vllm ]; then
    VLLM_ROOT=/usr/local/lib/python3.12/dist-packages
else
    echo "[pr51508] ERROR: cannot locate vllm tree" >&2
    exit 1
fi

cd "$VLLM_ROOT"

if git apply --reverse --check --include='vllm/*' "$SCRIPT_DIR/patch.diff" >/dev/null 2>&1; then
    echo "[pr51508] already applied, skipping"
elif git apply --check --include='vllm/*' "$SCRIPT_DIR/patch.diff" >/dev/null 2>&1; then
    git apply --include='vllm/*' "$SCRIPT_DIR/patch.diff"
    echo "[pr51508] patch applied"
else
    echo "[pr51508] ERROR: patch does not apply cleanly to this tree" >&2
    exit 1
fi

# Byte-compile touched files as a syntax sanity check
python3 - "$VLLM_ROOT" <<'PYEOF'
import py_compile, sys
from pathlib import Path
root = Path(sys.argv[1])
files = [
    "vllm/model_executor/layers/mamba/mamba_utils.py",
    "vllm/model_executor/layers/mamba/ops/causal_conv1d.py",
    "vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py",
    "vllm/models/kimi_k3/nvidia/kda_metadata.py",
    "vllm/models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py",
    "vllm/third_party/flash_linear_attention/ops/fused_recurrent.py",
    "vllm/third_party/flash_linear_attention/ops/fused_sigmoid_gating.py",
    "vllm/v1/attention/backends/gdn_attn.py",
    "vllm/v1/spec_decode/utils.py",
]
for f in files:
    p = root / f
    if p.exists():
        py_compile.compile(str(p), doraise=True)
print("[pr51508] syntax check OK")
PYEOF

echo "=== pr51508-stale-zero-accept complete ==="
