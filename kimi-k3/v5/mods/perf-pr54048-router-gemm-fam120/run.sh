#!/usr/bin/env bash
# perf-pr54048-router-gemm-fam120 — port of upstream vllm-project/vllm#54048.
#
# Un-gates the cuBLAS bf16→fp32 router GEMM (torch.mm out_dtype epilogue)
# for family-120 Blackwell (GB10 / DGX Spark, sm121). On GB10,
# is_device_capability_family(100) excludes family 120, so
# allow_specialized_router_gemm is False and the router falls back to
# F.linear: bf16-rounded logits + a separate bf16→fp32 copy kernel before
# grouped_topk, every MoE layer, every decode step. The plain cuBLAS
# out_dtype epilogue has no SM90+ requirement, so tier 5 is re-gated on
# (not bias and current_platform.is_cuda()).
#
# Only the cuBLAS tier gate changes; allow_specialized_router_gemm and the
# cuteDSL ll_bf16 path (SM90+) are untouched. No env gates. Idempotent via
# marker. Missing anchors NO-OP with a loud NOTE (never block boot).
#
# usage: run.sh [apply|simulate]
#   apply    (default) patch the installed vllm tree in place
#   simulate never write; patch a temp copy, py_compile it, print the diff
#            (falls back to the /var/tmp/v6_drytest copy if vllm is not
#            importable — useful on the head node outside the container)
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-apply}"

if [ "$MODE" != "apply" ] && [ "$MODE" != "simulate" ]; then
  echo "usage: run.sh [apply|simulate]"
  exit 2
fi

TARGET_REL="model_executor/layers/fused_moe/router/gate_linear.py"

# Resolve VLLM_ROOT the way the other mods do: installed vllm package location.
VLLM_ROOT="$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)"
if [ -z "$VLLM_ROOT" ] || [ ! -f "$VLLM_ROOT/envs.py" ]; then
  for cand in /opt/kimi-k3/vllm/vllm /opt/vllm/vllm; do
    if [ -f "$cand/envs.py" ]; then VLLM_ROOT="$cand"; break; fi
  done
fi

if [ "$MODE" = "simulate" ]; then
  SIM_TARGET="$VLLM_ROOT/$TARGET_REL"
  if [ ! -f "$SIM_TARGET" ] && [ -f "/var/tmp/v6_drytest/$TARGET_REL" ]; then
    SIM_TARGET="/var/tmp/v6_drytest/$TARGET_REL"
  fi
  echo "=====> [perf-pr54048-router-gemm-fam120] SIMULATE mode (no writes)"
  python3 "$SCRIPT_DIR/patch_router_gemm_fam120.py" --simulate "$SIM_TARGET"
  exit $?
fi

if [ ! -f "$VLLM_ROOT/envs.py" ]; then
  echo "=====> [perf-pr54048-router-gemm-fam120] NOTE: PREREQUISITE MISSING — VLLM_ROOT not resolvable (no envs.py). NO-OP, not blocking boot."
  exit 0
fi
echo "=====> [perf-pr54048-router-gemm-fam120] VLLM_ROOT=$VLLM_ROOT"

python3 "$SCRIPT_DIR/patch_router_gemm_fam120.py" "$VLLM_ROOT"
rc=$?

echo "=====> [perf-pr54048-router-gemm-fam120] dry-run checklist:"
echo "  1. APPLY line for gate_linear.py (or SKIP: already applied), no anchor NOTEs."
echo "  2. Pre-flight: ./run.sh simulate — patched temp copy must py_compile and"
echo "     the diff must touch only the cuBLAS tier gate (init + set_out_dtype)."
echo "  3. Serving on GB10: router GEMM hits Tier 5 (torch.mm out_dtype=fp32);"
echo "     the separate bf16→fp32 copy kernel before grouped_topk disappears."
echo "  4. allow_specialized_router_gemm / cuteDSL ll_bf16 path unchanged (SM90+ only)."
exit $rc
