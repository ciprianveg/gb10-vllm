#!/bin/bash
# Fix: Reduce shared memory usage in Triton MLA decode attention kernel for GB10 (SM121).
#
# Problem:
#   The _fwd_grouped_kernel_stage1 Triton kernel in vLLM's MLA decode attention path
#   requires 102,400 bytes of shared memory (BLOCK=32, num_stages=2), but GB10 (SM121)
#   hardware limit is 101,376 bytes — only 1,024 bytes over.
#
# Fix:
#   Reduce BLOCK from 32 to 16 and increase num_stages from 2 to 3.
#   BLOCK must be a power of 2 (required by tl.arange in the kernel).
#   This brings shared memory to 96,000 bytes (within the 101,376 limit) while
#   using deeper pipelining (num_stages=3) to compensate for the smaller block.
#
#   Shared memory calculation:
#     smem = num_stages × BLOCK × (BLOCK_DMODEL + BLOCK_DV) × dtype_size
#     Original: 2 × 32 × (512 + 288) × 2 = 102,400  (over limit)
#     Fixed:    3 × 16 × (512 + 288) × 2 =  96,000  (fits)
#
# Target file:
#   /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_decode_attention.py
set -e

SITE_PACKAGES="/usr/local/lib/python3.12/dist-packages"
TRITON_FILE="$SITE_PACKAGES/vllm/v1/attention/ops/triton_decode_attention.py"

if [[ ! -f "$TRITON_FILE" ]]; then
    echo "[fix-gb10-mla-smem] WARNING: Target file not found: $TRITON_FILE"
    exit 0
fi

echo "=== fix-gb10-mla-smem mod ==="

# Check if already patched
if grep -q "fix-gb10-mla-smem" "$TRITON_FILE" 2>/dev/null; then
    echo "[fix-gb10-mla-smem] Already patched, skipping."
    exit 0
fi

# Patch BLOCK from 32 to 16 in _decode_grouped_att_m_fwd
if grep -q "BLOCK = 32" "$TRITON_FILE"; then
    sed -i 's/BLOCK = 32/BLOCK = 16  # fix-gb10-mla-smem: reduced from 32 for GB10 smem limit/' "$TRITON_FILE"
    echo "[fix-gb10-mla-smem] Patched BLOCK 32 → 16"
else
    echo "[fix-gb10-mla-smem] ERROR: 'BLOCK = 32' not found"
    exit 1
fi

# Patch num_stages from 2 to 3 in _decode_grouped_att_m_fwd
# The line is: num_stages = 2  (variable assignment, around line 493)
if grep -q "num_stages = 2" "$TRITON_FILE"; then
    sed -i 's/num_stages = 2/num_stages = 3  # fix-gb10-mla-smem: increased from 2 for deeper pipeline/' "$TRITON_FILE"
    echo "[fix-gb10-mla-smem] Patched num_stages 2 → 3"
else
    echo "[fix-gb10-mla-smem] WARNING: 'num_stages = 2' not found (may already be 1)"
fi

# Clear Python cache
find "$SITE_PACKAGES/vllm/v1/attention" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Verify syntax
if python3 -c "import py_compile; py_compile.compile('$TRITON_FILE', doraise=True)" 2>/dev/null; then
    echo "[fix-gb10-mla-smem] Syntax check passed."
else
    echo "[fix-gb10-mla-smem] ERROR: Syntax error in patched file"
    exit 1
fi

echo "=== fix-gb10-mla-smem mod complete ==="
