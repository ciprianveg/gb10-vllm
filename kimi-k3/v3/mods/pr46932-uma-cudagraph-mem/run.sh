#!/usr/bin/env bash
# pr46932-uma-cudagraph-mem -- port of upstream vLLM PR #46932 (fixes #44740).
#
# On unified-memory GPUs (GB10 / DGX Spark, sm121) cudaMemGetInfo
# underreports free memory, and the MTP/spec-decode CUDA-graph memory
# estimate can go NEGATIVE (e.g. -35.69 GiB). That negative size is
# subtracted in determine_available_memory, INFLATING the KV cache budget
# and causing a silent OOM at high --gpu-memory-utilization (we run 0.9
# with cudagraphs on UMA -- this is a live OOM risk).
#
# The mod adds a psutil-based get_device_memory_info() helper to
# vllm/utils/mem_utils.py, routes every cudagraph memory-estimation reading
# in both model runners through it, and clamps negative capture deltas to
# >= 0. Anchor-based patcher, idempotent via the "PR_46932" marker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -n "${VLLM_MOD_TARGET_ROOT:-}" ]; then
    VLLM_ROOT="$VLLM_MOD_TARGET_ROOT"
elif [ -d /opt/kimi-k3/vllm ]; then
    VLLM_ROOT=/opt/kimi-k3/vllm
elif [ -d /usr/local/lib/python3.12/dist-packages/vllm ]; then
    VLLM_ROOT=/usr/local/lib/python3.12/dist-packages
else
    echo "[pr46932] ERROR: cannot locate vllm tree" >&2
    exit 1
fi

echo "[pr46932] patching vLLM tree at $VLLM_ROOT"
python3 "$SCRIPT_DIR/patch_uma_cudagraph_mem.py" "$VLLM_ROOT"

# Belt-and-suspenders syntax check of every touched file.
python3 - "$VLLM_ROOT" <<'PYEOF'
import py_compile, sys
from pathlib import Path
root = Path(sys.argv[1])
for f in [
    "vllm/utils/mem_utils.py",
    "vllm/v1/worker/gpu/model_runner.py",
    "vllm/v1/worker/gpu_model_runner.py",
]:
    p = root / f
    if p.exists():
        py_compile.compile(str(p), doraise=True)
print("[pr46932] syntax check OK")
PYEOF

echo "=== pr46932-uma-cudagraph-mem complete ==="
