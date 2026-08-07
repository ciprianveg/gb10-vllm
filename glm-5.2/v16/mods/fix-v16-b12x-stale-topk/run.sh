#!/bin/bash
# Fix stale topk_indices_buffer in B12X_MLA_SPARSE attention backend.
#
# Root cause: B12xMLASparseImpl caches topk_indices_buffer in __init__ from
# the indexer passed at construction time. Under spec decode + async
# scheduling, the indexer's buffer can be refreshed (reallocated) between
# steps, but the cached reference still points to the old buffer. This
# causes stale topk indices → incorrect attention scores → invalid draft
# tokens → "Failed to advance FSM" errors during tool calling.
#
# Fix: Store the indexer reference and read topk_indices_buffer live from
# it at each forward pass, falling back to the cached buffer only when
# no indexer is available (backbone skip layers).
#
# This is patch 06 from patches/v16-final/ — missing from the v16 prebuilt
# wheel but present in v17's image. v17 works without this mod because the
# fix is built in.
#
# Source: v16-final patch set (06-b12x-stale-topk-buffer.patch)
set -eux

# Detect v18 (venv) vs v16 (dist-packages) path
if [ -d "/opt/venv/lib/python3.12/site-packages/vllm" ]; then
    VLLM_DIR="/opt/venv/lib/python3.12/site-packages"
else
    VLLM_DIR="/usr/local/lib/python3.12/dist-packages"
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/apply_fix.py"

find "$VLLM_DIR/vllm" -name '*.pyc' -delete
find "$VLLM_DIR/vllm" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

echo "✓ B12X stale topk_indices_buffer fix applied"
