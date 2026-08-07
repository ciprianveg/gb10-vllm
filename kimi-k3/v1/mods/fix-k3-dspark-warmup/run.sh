#!/bin/bash
# Apply fix-k3-dspark-warmup mod to vLLM in the running container.
# Patches vllm/v1/worker/gpu/warmup.py to skip the mixed spec/non-spec
# warmup decode step that triggers a CUDA indexSelectSmallIndex assert
# in Kimi-K3's KDA metadata builder (kda_metadata.py:342).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/patch_warmup.py"

# Guard: confirm marker is present after patch.
VLLM_DIR="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
WARMUP="$VLLM_DIR/v1/worker/gpu/warmup.py"
if ! grep -q 'fix-k3-dspark-warmup' "$WARMUP"; then
    echo "ERROR: fix-k3-dspark-warmup marker not found in $WARMUP" >&2
    exit 1
fi
echo "fix-k3-dspark-warmup verified in $WARMUP"
