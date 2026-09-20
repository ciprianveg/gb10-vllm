#!/usr/bin/env bash
# fix-mxfp4-triton-sm121 — unlock OAI Triton MoE experts on SM12x (GB10).
# Upstream gate `_triton_kernel_moe_supports_current_device` keeps CUDA at
# (9,0)<=cap<(11,0), excluding SM120/121 although the Triton MXFP4 kernels
# run there (upstream PR #41028, open). Widen the CUDA window to <(13,0).
# Escape hatch: VLLM_OAI_TRITON_MOE_FORCE_OFF=1 handled by reverting this mod.
set -euo pipefail
VLLM_ROOT="${VLLM_ROOT:-/opt/kimi-k3/vllm/vllm}"
F="$VLLM_ROOT/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py"
[ -f "$F" ] || { echo "[fix-mxfp4-triton-sm121] ERROR: $F not found"; exit 1; }
grep -q "SM12X-UNLOCK" "$F" && { echo "[fix-mxfp4-triton-sm121] SKIP (already present)"; exit 0; }
grep -q "return cap is not None and (9, 0) <= (cap.major, cap.minor) < (11, 0)" "$F" || { echo "[fix-mxfp4-triton-sm121] ERROR: anchor not found"; exit 1; }
sed -i 's|return cap is not None and (9, 0) <= (cap.major, cap.minor) < (11, 0)|return cap is not None and (9, 0) <= (cap.major, cap.minor) < (13, 0)  # SM12X-UNLOCK (upstream #41028 style)|' "$F"
python3 -m py_compile "$F" && echo "[fix-mxfp4-triton-sm121] APPLIED + py_compile OK" || { echo "[fix-mxfp4-triton-sm121] py_compile FAILED"; exit 1; }
