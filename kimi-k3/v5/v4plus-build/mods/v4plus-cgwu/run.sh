#!/usr/bin/env bash
# v4plus-cgwu: skip post-capture warmup_kernels behind VLLM_K3_SKIP_WARMUP (payload overwrite)
set -eu
VLLM_ROOT="${VLLM_ROOT:-/opt/kimi-k3/vllm/vllm}"
TARGET="$VLLM_ROOT/v1/worker/gpu_worker.py"
if grep -q "VLLM_K3_SKIP_WARMUP" "$TARGET"; then
  echo "[v4plus-cgwu] SKIP: warmup-skip already present"
else
  cp "$(dirname "$0")/gpu_worker.py" "$TARGET"
  echo "[v4plus-cgwu] APPLY: gpu_worker.py warmup-skip (VLLM_K3_SKIP_WARMUP)"
fi
python3 -m py_compile "$TARGET" && echo "[v4plus-cgwu] py_compile OK"
