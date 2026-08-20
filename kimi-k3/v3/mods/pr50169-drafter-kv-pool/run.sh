#!/bin/bash
# PR #50169: dedicated KV cache groups for sliding-window drafters +
# amortized local-attention pool sizing (GB10: KV pool 415k -> 871k tokens).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve the vLLM install root:
#   1. VLLM_MOD_TARGET_ROOT env override (used for testing against a copy)
#   2. /opt/kimi-k3/vllm (source checkout in the kimi3 image)
#   3. /usr/local/lib/python3.12/dist-packages (pip-installed fallback)
if [ -n "${VLLM_MOD_TARGET_ROOT:-}" ]; then
    VLLM_ROOT="$VLLM_MOD_TARGET_ROOT"
elif [ -d /opt/kimi-k3/vllm/vllm ]; then
    VLLM_ROOT="/opt/kimi-k3/vllm"
else
    VLLM_ROOT="/usr/local/lib/python3.12/dist-packages"
fi

echo "--- Applying PR #50169 drafter KV pool patch to: $VLLM_ROOT"

python3 "$SCRIPT_DIR/patch_drafter_kv_pool.py" "$VLLM_ROOT"

# Byte-compile the touched files as a final syntax check.
FILES=(
    "vllm/v1/core/kv_cache_utils.py"
    "vllm/v1/core/single_type_kv_cache_manager.py"
    "vllm/v1/kv_cache_interface.py"
    "vllm/model_executor/models/laguna_dflash.py"
)
for rel in "${FILES[@]}"; do
    f="$VLLM_ROOT/$rel"
    if [ -f "$f" ]; then
        python3 -m py_compile "$f"
        echo "py_compile OK: $rel"
    else
        echo "WARNING: not found, skipping py_compile: $rel"
    fi
done

echo "=== pr50169-drafter-kv-pool mod complete ==="
