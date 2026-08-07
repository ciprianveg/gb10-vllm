#!/bin/bash
# fix-k3-flash-kda-prefill — PR #51311: Flash KDA out kernel for prefill.
#
# Preallocates FlashKDA workspace/output/final_state tensors via the
# workspace manager instead of allocating them on every kernel call.
# 1.1-1.4x kernel-level speedup for KDA prefill on KIMI-K3 REAP-320.
#
# Applies to: vllm-node-kimi3 image (fork commit 3846d740, Aug 4 2026).
# The fork's kda.py is heavily diverged from upstream, so this mod uses
# targeted string replacements adapted from PR #51311.
#
# No vLLM rebuild needed — pure Python patch.
set -euo pipefail

echo "--- Applying Flash KDA prefill optimization (PR #51311, 1.1-1.4x KDA prefill kernel)..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$SCRIPT_DIR/patch_kda.py"

# Verify syntax
TARGET="/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/nvidia/kda.py"
python3 -c "import py_compile; py_compile.compile('$TARGET', doraise=True)" \
    && echo "Syntax OK" || {
    echo "ERROR: Syntax check failed" >&2
    exit 1
}

echo "=== fix-k3-flash-kda-prefill mod complete ==="
