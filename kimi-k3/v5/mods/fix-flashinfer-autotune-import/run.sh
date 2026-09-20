#!/bin/bash
# fix-flashinfer-autotune-import — runtime mod (Python-only, no rebuild).
# Guards the `set_autotune_process_group` import in the K3 fork's
# flashinfer_autotune() kernel-warmup: the installed flashinfer (0.7.0rc1)
# does not export that symbol. With the mxfp8 online overlay active,
# FlashInfer compute kernels pass _uses_flashinfer_compute_kernels() and
# the autotune warmup path runs, crashing every rank at boot with
#   ImportError: cannot import name 'set_autotune_process_group'
# Fallback is a no-op shim: ranks autotune independently (per-rank GEMM
# tactic choice) instead of averaging timings over the world CPU group.
# Idempotent.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_NAME="fix-flashinfer-autotune-import"

VLLM_ROOT=""
for cand in /opt/kimi-k3/vllm; do
  if [ -f "$cand/vllm/model_executor/warmup/kernel_warmup.py" ]; then
    VLLM_ROOT="$cand"
    break
  fi
done
if [ -z "$VLLM_ROOT" ]; then
  pkg_dir="$(python3 -c 'import vllm, os; print(os.path.dirname(vllm.__file__))' 2>/dev/null || true)"
  if [ -n "$pkg_dir" ] && [ -f "$pkg_dir/model_executor/warmup/kernel_warmup.py" ]; then
    VLLM_ROOT="$(cd "$pkg_dir/.." && pwd)"
  fi
fi
if [ -z "$VLLM_ROOT" ]; then
  echo "=====> [$MOD_NAME] PREREQUISITE MISSING: no vLLM source root found"
  exit 1
fi

if [ "${1:-}" = "--simulate" ]; then
  echo "=====> [$MOD_NAME] SIMULATE mode (no writes)"
  python3 "$SCRIPT_DIR/patch_autotune_import.py" --simulate "$VLLM_ROOT"
  exit $?
fi

echo "=====> [$MOD_NAME] VLLM_ROOT=$VLLM_ROOT"
python3 "$SCRIPT_DIR/patch_autotune_import.py" "$VLLM_ROOT"
