#!/usr/bin/env bash
# pr53524 — backport of upstream vLLM PR #53524.
# Prefetch ll_bf16 router weights (M=1) into registers before the PDL wait;
# ll_bf16_gemm() auto-selects a prefetch-specialized kernel for M==1.
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -ne 0 ]; then
  echo "usage: run.sh (no args; resolves VLLM_ROOT automatically)"
  exit 2
fi

# Resolve VLLM_ROOT the way batch1 does: installed vllm package location.
VLLM_ROOT="$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)"
if [ -z "$VLLM_ROOT" ] || [ ! -f "$VLLM_ROOT/envs.py" ]; then
  for cand in /opt/kimi-k3/vllm/vllm /opt/vllm/vllm; do
    if [ -f "$cand/envs.py" ]; then VLLM_ROOT="$cand"; break; fi
  done
fi
if [ ! -f "$VLLM_ROOT/envs.py" ]; then
  echo "[pr53524] PREREQUISITE FAILED: VLLM_ROOT not resolvable (no envs.py)"
  exit 1
fi
echo "[pr53524] VLLM_ROOT=$VLLM_ROOT"

python3 "$SCRIPT_DIR/patch_ll_bf16_prefetch.py" "$VLLM_ROOT"
rc=$?

echo "[pr53524] dry-run checklist:"
echo "  1. No 'PREREQUISITE FAILED' above."
echo "  2. APPLY lines for _ll_bf16_dotprod.py and ll_bf16.py, or SKIP"
echo "     (already present). NOTE lines = hunk skipped — investigate."
echo "  3. Serving: M=1 decode router GEMM (ll_bf16_gemm) now compiles a"
echo "     weight-prefetch PDL variant on first use; expect no behavior"
echo "     change beyond latency (first-compile cost at warmup)."
exit $rc
