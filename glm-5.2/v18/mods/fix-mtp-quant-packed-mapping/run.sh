#!/bin/bash
# Fix MTP draft model quantized weight loading for fused modules.
#
# Root cause: The v16 image's vLLM wheel is missing patch 03
# (03-draft-quant-packed-mapping.patch) which adds packed_modules_mapping
# entries for fused_qkv_a_proj and gate_up_proj to the draft quant config.
# Without this, the MTP draft model's fused_qkv_a_proj module is not
# quantized (built as BF16), but the checkpoint has weight_packed weights,
# causing KeyError at load time.
#
# This mod applies the same fix as patch 03 at runtime.
# Auto-detects v18 (/opt/venv) vs v16 (/usr/local) install path.
set -eux

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "$SCRIPT_DIR/apply_fix.py"

# Auto-detect vllm dir for bytecode cache cleanup
if [ -d "/opt/venv/lib/python3.12/site-packages/vllm" ]; then
    VLLM_DIR="/opt/venv/lib/python3.12/site-packages"
elif [ -d "/usr/local/lib/python3.12/dist-packages/vllm" ]; then
    VLLM_DIR="/usr/local/lib/python3.12/dist-packages"
else
    echo "WARNING: Could not find vllm dir for cache cleanup"
    exit 0
fi

# Clear bytecode caches
find "$VLLM_DIR/vllm" -name '*.pyc' -delete
find "$VLLM_DIR/vllm" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

echo "✓ MTP quant packed mapping fix applied"
