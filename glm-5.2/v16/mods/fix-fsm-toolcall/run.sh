#!/bin/bash
# Fix FSM advancement failure during tool calling + MTP speculative decoding.
#
# The v16 fork (local-inference-lab/vllm @fathomless-firmament-v16-unified)
# already includes PR #44297 (trim_reasoning_for_advance) and PR #46149
# (structural-tag reasoning=True).  However it is MISSING PR #44993 which
# fixes the broken delta-window calculation in should_advance():
#
#   delta_from = num_computed_tokens - num_output_placeholders
#
# Under async scheduling + MTP spec decode, when some draft tokens are
# rejected, num_output_placeholders stays > 0 and the computed delta
# window starts PAST the reasoning-end marker.  is_reasoning_end_streaming
# never fires, reasoning_ended never flips, the grammar is never enforced,
# and the model generates tokens the FSM rejects → "Failed to advance FSM"
# + HTTP 500.
#
# This mod applies the two key changes from PR #44993:
#   1. should_advance() accepts new_token_ids and uses it directly as the
#      delta window (bypassing the broken placeholder math).
#   2. The scheduler passes new_token_ids to should_advance().
#   3. Removes the structural-tag-only restriction — all backend types get
#      same-step advance at the reasoning boundary.
#
# Refs:
#   https://github.com/vllm-project/vllm/pull/44993
#   https://github.com/vllm-project/vllm/issues/44006
#   https://github.com/vllm-project/vllm/issues/43388
set -eux

VLLM_DIR="/usr/local/lib/python3.12/dist-packages"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$VLLM_DIR"

# Strip test files from the diff — only patch source
awk '/^diff --git a\/tests\//{skip=1} /^diff --git a\/vllm\//{skip=0} skip==0' \
    "$SCRIPT_DIR/pr44993.diff" > /tmp/pr44993-src.diff

# Apply with --forward to skip already-applied hunks gracefully
patch -p1 --forward < /tmp/pr44993-src.diff || {
    # If patch fails, check if it's already applied
    if patch -p1 --dry-run --reverse < /tmp/pr44993-src.diff >/dev/null 2>&1; then
        echo "✓ PR #44993 already applied"
    else
        echo "✗ PR #44993 patch failed"
        exit 1
    fi
}

# Clear bytecode caches so the patched .py files take effect
find "$VLLM_DIR/vllm" -name '*.pyc' -delete
find "$VLLM_DIR/vllm" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

echo "✓ PR #44993 applied: should_advance new_token_ids + boundary advance for all backends"
