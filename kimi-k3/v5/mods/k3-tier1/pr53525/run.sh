#!/usr/bin/env bash
# pr53525 — backport of upstream vLLM PR #53525 (Python side).
# C=1 KDA PDL pipeline: early dependent-launch trigger in the cute skinny
# GEMM after the mainloop; K3 plan requests it for M=1 projections. Native
# (dsv3 csrc) parts are schema-gated: they engage once the .so is rebuilt.
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
  echo "[pr53525] PREREQUISITE FAILED: VLLM_ROOT not resolvable (no envs.py)"
  exit 1
fi
echo "[pr53525] VLLM_ROOT=$VLLM_ROOT"

python3 "$SCRIPT_DIR/patch_c1_kda_pdl.py" "$VLLM_ROOT"
rc=$?

echo "[pr53525] dry-run checklist:"
echo "  1. No 'PREREQUISITE FAILED' above."
echo "  2. APPLY lines for _skinny_gemm.py, skinny_gemm.py, _custom_ops.py,"
echo "     low_latency_gemm.py, or SKIP (already present)."
echo "  3. NOTE lines about csrc hunks are EXPECTED (native half needs a"
echo "     C++ rebuild; gated on the early_pdl_trigger op schema)."
echo "  4. Serving: M=1 in_proj_qkvgfab / o_proj cute GEMMs compile the"
echo "     early-trigger variant at warmup. dsv3 f_b_proj / fused_qkv_a_proj"
echo "     early trigger + single-row stride acceptance stay off until the"
echo "     image is rebuilt with the #53525 csrc."
exit $rc
