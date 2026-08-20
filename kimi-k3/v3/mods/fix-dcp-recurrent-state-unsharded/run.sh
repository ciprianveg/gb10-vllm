#!/usr/bin/env bash
# fix-dcp-recurrent-state-unsharded — PR #418
# Keep recurrent KDA token-position state unsharded under DCP.
# Without this, MambaManager block tables are too narrow under DCP>1,
# causing out-of-bounds block IDs → nonfinite logits at 63,744+ tokens.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /opt/kimi-k3/vllm

patch -p1 --dry-run < "$SCRIPT_DIR/patch.diff" 2>/dev/null && {
    patch -p1 < "$SCRIPT_DIR/patch.diff"
    echo "[fix-dcp-recurrent-state-unsharded] applied"
} || {
    grep -q "Recurrent state has one token-position shard" vllm/v1/kv_cache_interface.py 2>/dev/null && \
        echo "[fix-dcp-recurrent-state-unsharded] already present, skipping" || \
        { echo "[fix-dcp-recurrent-state-unsharded] FAILED to apply"; exit 1; }
}
