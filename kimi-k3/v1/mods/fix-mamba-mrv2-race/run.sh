#!/bin/bash
# fix-mamba-mrv2-race — Patch vllm/v1/worker/mamba_utils.py: fix MRv2 cross-block race on num_accepted.
#
# Upstream: vllm-project/vllm#50432 (merged 2026-08-03, post Jul-30 image).
#
# Bug
# ---
# In MRv2 hybrid postprocess (`run_fused_postprocess_align`), the Triton kernel
# `postprocess_mamba_fused_kernel` is launched with grid (num_reqs, num_layers *
# num_state_types). For each req_idx, programs read `num_accepted_tokens_ptr`
# early in the kernel, and `state_idx == 0` programs write 1 to it (in place)
# when the accepted position stays in the running block.
#
# Different `state_idx` programs for the same `req_idx` can land in different
# waves on the GPU. Later-wave programs observe the (1) write and compute
# accept_token_bias == 0 (early return, no state copy). Earlier-wave programs
# already committed to a state copy with the original `num_accepted > 1`.
# Result: Mamba state is inconsistent across layers for the same request.
#
# The race is severity-dependent on GPU launch timing / number of SMs / number
# of hybrid layers. K3-full (93 layers, TP=16) triggers it more than REAP-320
# (pruned, fewer layers -> fewer waves -> race less likely). The inconsistent
# Mamba state feeds wrong hidden states into the DSpark drafter's aux layers
# [2,23,47,71,89], which is the most plausible cause of near-zero acceptance
# (0.03-0.19 position-0) vs REAP's 0.33-0.81.
#
# Fix
# ---
# Apply the MRv1 pattern: the kernel reads from a snapshot buffer and writes
# to a distinct output buffer. The caller copies the snapshot back into
# `num_accepted_tokens_gpu` after the kernel finishes (same stream, strictly
# ordered).
#
# Three changes in vllm/v1/worker/mamba_utils.py:
# 1. Kernel (~line 251): remove the `if HAS_IDX_MAPPING` branch that wrote
#    in-place. Always write to `num_accepted_tokens_out_ptr`.
# 2. Caller (~line 862): snapshot `num_accepted_tokens_gpu` into
#    `self.num_accepted_tokens_out` before the kernel launch.
# 3. Caller (~line 884): pass the snapshot as the read source and
#    `num_accepted_tokens_gpu` as the write target.
#
# Applies to: vllm-node-kimi3 image (vLLM 0.26.1rc1.dev160+g2ac91211d, Jul 30).
# Effect: eliminates Mamba state inconsistency in MRv2 hybrid, which should
# improve DSpark drafter acceptance from 1.6-5.3% toward REAP's 33-81%.
# Memory: no change (same buffers, just different write target).
#
# Idempotent: guarded by marker comment `fix-mamba-mrv2-race`.
set -e

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VLLM_DIR="$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))" 2>/dev/null || true)"
if [ -z "$VLLM_DIR" ]; then
    echo "[fix-mamba-mrv2-race] ERROR: cannot locate vllm package" >&2
    exit 1
fi

TARGET_FILE="$VLLM_DIR/v1/worker/mamba_utils.py"
if [ ! -f "$TARGET_FILE" ]; then
    echo "[fix-mamba-mrv2-race] ERROR: $TARGET_FILE not found" >&2
    exit 1
fi

echo "[fix-mamba-mrv2-race] Patching $TARGET_FILE"
python3 "$MOD_DIR/patch_mamba.py" "$TARGET_FILE"

python3 -c "
import ast
with open('$TARGET_FILE') as f:
    ast.parse(f.read())
print('[fix-mamba-mrv2-race] Syntax OK')
" || { echo "[fix-mamba-mrv2-race] ERROR: patched file has syntax errors" >&2; exit 1; }

MARKER_COUNT=$(grep -c "fix-mamba-mrv2-race" "$TARGET_FILE" || true)
if [ "$MARKER_COUNT" -lt 3 ]; then
    echo "[fix-mamba-mrv2-race] ERROR: expected 3 markers, found $MARKER_COUNT" >&2
    exit 1
fi

echo "[fix-mamba-mrv2-race] Mod applied successfully ($MARKER_COUNT markers)."
