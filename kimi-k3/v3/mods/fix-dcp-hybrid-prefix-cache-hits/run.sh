#!/usr/bin/env bash
# fix-dcp-hybrid-prefix-cache-hits — allow hash-aligned DCP hybrid prefix hits.
#
# Port of local-inference-lab/vllm PR #401 by myshytf (same author as #395),
# "fix(prefix-cache): allow hash-aligned DCP hybrid hits". Production-only
# change to vllm/v1/core/kv_cache_coordinator.py (tests intentionally excluded).
#
# Bug: HybridKVCacheCoordinator enabled fine-grained prefix hash hits only when
# dcp_world_size == 1. Under DCP>1 it blanket-coarsened local APC hits to the
# DCP-expanded full-attention block, even when the aligned Mamba (KDA) recurrent
# manager already materializes a complete state at every hash boundary. That
# wastes prefill work — a hit degrades from the hash-block granularity (e.g.
# 1536) up to the DCP×hash attention block (12288 at DCP8, 24576 at DCP16), so
# the regression is WORSE on DCP16.
#
# Fix: enable fine-grained hits under ANY DCP size when every aligned Mamba
# manager's block_size == hash_block_size (so no partial recurrent-state
# hand-off across DCP ranks is needed). The logic is fully dcp_world_size-
# generic — no DCP8 hardcoding; DCP16 is covered by the same condition.
#
# Activation on our K3 config requires: an aligned KDA group whose block_size
# equals the prefix-cache hash block size, plus a fine-grained group whose
# block_size > hash block (the full-attention group). Harmless (no-op) if those
# do not hold. Idempotent git-apply patch. Applies to /opt/kimi-k3/vllm or
# site-packages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -d /opt/kimi-k3/vllm/.git ]; then
    VLLM_ROOT=/opt/kimi-k3/vllm
elif [ -d /usr/local/lib/python3.12/dist-packages/vllm ]; then
    VLLM_ROOT=/usr/local/lib/python3.12/dist-packages
else
    echo "[fix-dcp-hybrid-prefix-cache-hits] ERROR: cannot locate vllm tree" >&2
    exit 1
fi

cd "$VLLM_ROOT"

if git apply --reverse --check --include='vllm/v1/core/*' "$SCRIPT_DIR/patch.diff" >/dev/null 2>&1; then
    echo "[fix-dcp-hybrid-prefix-cache-hits] already applied, skipping"
elif git apply --check --include='vllm/v1/core/*' "$SCRIPT_DIR/patch.diff" >/dev/null 2>&1; then
    git apply --include='vllm/v1/core/*' "$SCRIPT_DIR/patch.diff"
    echo "[fix-dcp-hybrid-prefix-cache-hits] patch applied"
else
    echo "[fix-dcp-hybrid-prefix-cache-hits] ERROR: patch does not apply cleanly to this tree" >&2
    exit 1
fi

python3 - "$VLLM_ROOT" <<'PYEOF'
import py_compile, sys
from pathlib import Path
p = Path(sys.argv[1]) / "vllm/v1/core/kv_cache_coordinator.py"
if p.exists():
    py_compile.compile(str(p), doraise=True)
print("[fix-dcp-hybrid-prefix-cache-hits] syntax check OK")
PYEOF

echo "=== fix-dcp-hybrid-prefix-cache-hits complete ==="
