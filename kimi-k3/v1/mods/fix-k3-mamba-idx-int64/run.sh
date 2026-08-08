#!/bin/bash
# fix-k3-mamba-idx-int64 — cast idx_mapping to int64 in mamba_hybrid.py postprocess_state
#
# The running fork (local-inference-lab/vllm) allocates input_batch.idx_mapping
# as int32 (perf optimization for Triton kernels). mamba_hybrid.py:299 calls
# `index_fill_(0, idx_mapping, ...)`, which strictly requires int64 indices,
# so it raises "IndexError: index_fill_(): Expected dtype int64 for index."
# on any step where num_sampled is a Python int (chunked prefill continuation).
#
# Affects TP16 and PP2 equally; draft token count (5 vs 7) is irrelevant.
# One-line cast; zero measurable decode overhead (max_num_reqs-sized tensor).
#
# Idempotent — guarded by marker comment; re-application is a no-op.
set -euo pipefail

echo "--- Applying fix-k3-mamba-idx-int64 (mamba_hybrid.py index_fill_ dtype)..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/patch_mamba_idx_int64.py"

TARGET="/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
python3 -c "import py_compile; py_compile.compile('$TARGET', doraise=True)" \
    && echo "Syntax OK" || {
    echo "ERROR: Syntax check failed" >&2
    exit 1
}

echo "=== fix-k3-mamba-idx-int64 mod complete ==="
