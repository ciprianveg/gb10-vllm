#!/usr/bin/env bash
# pr54168 — backport of upstream vLLM PR #54168.
# Low-M fused latent-MoE tail: 7-CTA/64-thread collective geometry with a
# bf16 top-16 finalize fast path (TP8 3584/7168), parity-alternating DSM
# slots, NVLS multicast stores, compact ReduceScatter roles, vectorized
# skinny up-projection, and a fused lamport copy.
# PREREQUISITE: pr53152 must be applied first (hard dependency).
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
  echo "[pr54168] PREREQUISITE FAILED: VLLM_ROOT not resolvable (no envs.py)"
  exit 1
fi
echo "[pr54168] VLLM_ROOT=$VLLM_ROOT"

python3 "$SCRIPT_DIR/patch_lowm_tail.py" "$VLLM_ROOT"
rc=$?

echo "[pr54168] dry-run checklist:"
echo "  1. No 'PREREQUISITE FAILED' above (pr53152 must be applied first)."
echo "  2. APPLY lines for primitives.py, the allreduce collective, the"
echo "     skinny up-projection GEMM, lamport_copy.py, latent_moe_tail.py,"
echo "     or SKIP (already present)."
echo "  3. Serving (TP8, top-16, bf16, SM100): M<=4 tail compiles the 7-CTA"
echo "     specializations at warmup; watch tail latency at M=1..4."
echo "  4. TP16: the 7-CTA geometry is TP8-shape-gated and stays off; the"
echo "     generic-path FP32 accumulation, lamport-copy and skinny-GEMM"
echo "     improvements still apply."
exit $rc
