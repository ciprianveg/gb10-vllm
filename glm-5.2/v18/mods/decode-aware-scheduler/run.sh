#!/bin/bash
# Decode-Aware Custom Scheduler for GLM-5.2 on GB10
#
# Prevents multiple long-prefill requests from blocking decode streams.
# When decode is active, prefill is limited to a shared token budget (default 1024).
# When idle, prefill can use the full batched token budget (default 16384).
# At most one long-prefill per step, with round-robin/aging across requests.
#
# Source: penguinchang @ NVIDIA Developer Forums (2026-07-15)
# Patch baseline: local-inference-lab/vllm@a663653d (spark4-overlay)
# Adapted for v16 fork: fathomless-firmament-v16-unified-20260712
#
# New CLI flags added by this mod:
#   --enable-decode-aware-prefill
#   --decode-prefill-token-budget 1024
#   --idle-prefill-token-budget 16384
#   --max-long-prefills-per-step 1
#
# Requires: --enable-chunked-prefill and --long-prefill-token-threshold > 0
#
# Rollback: remove --enable-decode-aware-prefill from the serve command
# (or set it to 0). The patched code path is not entered when disabled.
set -eux

# Detect v18 (venv) vs v16 (dist-packages) path
if [ -d "/opt/venv/lib/python3.12/site-packages/vllm" ]; then
    VLLM_DIR="/opt/venv/lib/python3.12/site-packages"
else
    VLLM_DIR="/usr/local/lib/python3.12/dist-packages"
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$VLLM_DIR"

# ── Part 1: Patch config/scheduler.py and engine/arg_utils.py ───────────────
# These add SchedulerConfig fields and CLI argument definitions.
# Uses patch with --forward for idempotency.

if patch -p1 --dry-run --reverse < "$SCRIPT_DIR/config_args.patch" >/dev/null 2>&1; then
    echo "✓ config_args.patch already applied"
else
    patch -p1 --forward < "$SCRIPT_DIR/config_args.patch" || {
        if patch -p1 --dry-run --reverse < "$SCRIPT_DIR/config_args.patch" >/dev/null 2>&1; then
            echo "✓ config_args.patch already applied"
        else
            echo "✗ config_args.patch failed"
            exit 1
        fi
    }
    echo "✓ config_args.patch applied"
fi

# ── Part 2: Patch vllm/v1/core/sched/scheduler.py ───────────────────────────
# The v16 fork has a spec-decode padding block not in the original patch,
# so we use a Python script with targeted string replacements instead of
# a unified diff. This is more robust against line-number shifts.

python3 "$SCRIPT_DIR/apply_scheduler_patch.py"

# ── Clear bytecode caches ───────────────────────────────────────────────────
find "$VLLM_DIR/vllm" -name '*.pyc' -delete
find "$VLLM_DIR/vllm" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

echo "✓ Decode-aware prefill scheduler patch applied"
echo "  Enable with: --enable-decode-aware-prefill --long-prefill-token-threshold 2048"
