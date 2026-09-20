#!/usr/bin/env bash
# pr54697 — backport of upstream vLLM PR #54697.
# Overlap low-M KDA projections: Q/K/V/G on the main stream, F_A/beta + F_B
# on the auxiliary stream (TP8-shape-gated; capture-only). Adds the KDA
# skinny GEMMs, the overlap plan in low_latency_gemm, kda.py plumbing, and
# the FlashInfer QKVG autotune hook. TP16 stays stock (see PORT-REPORT.md).
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
  echo "[pr54697] PREREQUISITE FAILED: VLLM_ROOT not resolvable (no envs.py)"
  exit 1
fi
echo "[pr54697] VLLM_ROOT=$VLLM_ROOT"

python3 "$SCRIPT_DIR/patch_kda_overlap.py" "$VLLM_ROOT"
rc=$?

echo "[pr54697] dry-run checklist:"
echo "  1. No 'PREREQUISITE FAILED' above."
echo "  2. APPLY for the new kda_skinny_gemm.py + low_latency_gemm.py +"
echo "     kda.py + model.py + kernel_warmup.py, or SKIP (already present)."
echo "  3. NOTE about TP16 is EXPECTED: overlap is TP8-shape-gated."
echo "  4. Serving (TP8, bf16, unquantized KDA path): after FlashInfer"
echo "     autotune, KDA decode projections (M<=14) overlap across streams"
echo "     during CUDA graph capture; expect a small decode-step win."
exit $rc
