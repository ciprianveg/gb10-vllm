#!/bin/bash
# fix-k3-fp8-triton-mla — Enable fp8 KV cache for KIMI-K3 with TRITON_MLA on sm121.
#
# K3's model code (kimi_k3/nvidia/mla.py) asserts impl.supports_quant_query_input
# for fp8 KV decode/prefill. TRITON_MLA (the only sm121-capable dense MLA backend)
# sets supports_quant_query_input=False (Mode 1: dequantizes fp8 KV to bf16
# internally, wants bf16 queries). The K3 fused op couples fp8-query quant with
# fp8-cache insert and cannot produce a bf16 query.
#
# This mod adds a Mode-1 branch: when the backend doesn't support quantized query
# input, bypass the fused fp8 op and use torch.cat (bf16 query) +
# concat_and_cache_mla (fp8 cache insert), mirroring the generic MLAAttention
# Mode-1 flow. No vLLM rebuild needed — concat_and_cache_mla is pre-compiled.
#
# Applies to: vllm-node-kimi3 image (vLLM 0.26.1rc1.dev160+g2ac91211d).
# Effect: --kv-cache-dtype fp8 works with TRITON_MLA on sm121 (GB10).
set -e

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find vLLM site-packages (handles both /usr/local and /opt/venv layouts)
VLLM_DIR="$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))" 2>/dev/null || true)"
if [ -z "$VLLM_DIR" ]; then
    echo "[fix-k3-fp8-triton-mla] ERROR: cannot locate vllm package" >&2
    exit 1
fi

MLA_FILE="$VLLM_DIR/models/kimi_k3/nvidia/mla.py"
if [ ! -f "$MLA_FILE" ]; then
    echo "[fix-k3-fp8-triton-mla] ERROR: $MLA_FILE not found (K3 model not in this image?)" >&2
    exit 1
fi

echo "[fix-k3-fp8-triton-mla] Patching $MLA_FILE"
python3 "$MOD_DIR/patch_mla.py" "$MLA_FILE"

# Verify the patch compiled cleanly
python3 -c "
import ast, sys
with open('$MLA_FILE') as f:
    ast.parse(f.read())
print('[fix-k3-fp8-triton-mla] Syntax OK')
" || { echo "[fix-k3-fp8-triton-mla] ERROR: patched file has syntax errors" >&2; exit 1; }

echo "[fix-k3-fp8-triton-mla] Mod applied successfully."
