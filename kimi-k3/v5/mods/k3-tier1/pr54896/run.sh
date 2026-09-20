#!/usr/bin/env bash
# pr54896 — upstream vLLM PR #54896 (MLA decode concat/cache epilogue).
# CSRc-ONLY PR (fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu): 3-warp row
# split for decode batches <= 64 tokens + deferred grid-dependency wait.
# NOT portable as a Python mod — requires rebuilding the stable-libtorch
# extension. This run.sh documents the skip; it makes no changes.
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -ne 0 ]; then
  echo "usage: run.sh (no args; resolves VLLM_ROOT automatically)"
  exit 2
fi

VLLM_ROOT="$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)"
if [ -z "$VLLM_ROOT" ] || [ ! -f "$VLLM_ROOT/envs.py" ]; then
  for cand in /opt/kimi-k3/vllm/vllm /opt/vllm/vllm; do
    if [ -f "$cand/envs.py" ]; then VLLM_ROOT="$cand"; break; fi
  done
fi
if [ ! -f "$VLLM_ROOT/envs.py" ]; then
  echo "[pr54896] PREREQUISITE FAILED: VLLM_ROOT not resolvable (no envs.py)"
  exit 1
fi
echo "[pr54896] VLLM_ROOT=$VLLM_ROOT"

python3 "$SCRIPT_DIR/patch_mla_epilogue.py" "$VLLM_ROOT"
rc=$?

echo "[pr54896] dry-run checklist:"
echo "  1. SKIP + NOTE lines are EXPECTED: csrc-only PR, nothing to patch."
echo "  2. To engage the optimization, rebuild the image with the #54896"
echo "     csrc patch (writeLatent576 lane/lane_stride + launchPdlSlots);"
echo "     no Python-side changes are required."
exit $rc
