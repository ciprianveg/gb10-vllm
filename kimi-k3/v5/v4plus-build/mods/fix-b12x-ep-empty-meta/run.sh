#!/usr/bin/env bash
# fix-b12x-ep-empty-meta — Tolerate empty expert metadata in B12X EP
# workspace planning (warmup/empty microbatches report 0 local experts
# while weights are prepared for the full local set).
#
# Root cause: workspace_shapes() in b12x_ep_moe.py raises ValueError when
# expert metadata says 0 local experts but 56 experts' weights are prepared.
# Seen at TP16+DCP8+EP boot (warmup forward), killing init. The marlin
# backend serves the identical EP config fine, so placement itself is
# consistent — the B12X EP check demands live occupancy where it should
# plan for capacity (the plan is built from `prepared`, not the metadata).
#
# Fix: only raise on REAL mismatches (local > 0 and != prepared). When
# local == 0, warn once and proceed planning with prepared capacity.
# Real misplacements still raise; quality gate validates output.
set -euo pipefail

echo "=== fix-b12x-ep-empty-meta ==="

VLLM_PKG_DIR="$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))" 2>/dev/null || true)"
if [ -z "$VLLM_PKG_DIR" ]; then
    echo "[fix-b12x-ep-empty-meta] ERROR: cannot locate vllm package" >&2
    exit 1
fi

TARGET="$VLLM_PKG_DIR/model_executor/layers/fused_moe/b12x_ep_moe.py"
MARKER="FIX-B12X-EP-EMPTY-META"

# Idempotency: check if already patched
if grep -q "$MARKER" "$TARGET" 2>/dev/null; then
    echo "[fix-b12x-ep-empty-meta] already applied (marker present), skipping"
    exit 0
fi

MOD_DIR="$(cd "$(dirname "$0")" && pwd)"
PATCH_SCRIPT="$MOD_DIR/patch.py"

if [ ! -f "$PATCH_SCRIPT" ]; then
    echo "[fix-b12x-ep-empty-meta] ERROR: patch script missing: $PATCH_SCRIPT" >&2
    exit 1
fi

python3 "$PATCH_SCRIPT" "$TARGET"

# Verify — fail loud: booting unpatched while the recipe expects the fix
# would hit the same ValueError.
if grep -q "$MARKER" "$TARGET" 2>/dev/null; then
    echo "[fix-b12x-ep-empty-meta] verified: marker present"
else
    echo "[fix-b12x-ep-empty-meta] ERROR: marker not found after patch — aborting boot" >&2
    exit 1
fi
