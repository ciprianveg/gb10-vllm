#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# V4PLUS CGFIX2 — disable the DFlash context-KV CUDA graph capture phase,
# keeping the main DSpark FULL graphs.
#
# Fixes the b5-image boot crash: "Capturing dspark CUDA graphs (FULL):
# 2/2" completes, then CUDA illegal memory access at "Capturing DFlash
# context-KV CUDA graphs (FULL): 0%". That phase is fork-PR-#251-specific;
# the fork's newer lineages and upstream run the context-KV precompute
# EAGERLY outside any graph, and this tree already carries the native
# eager fallback (dispatch -> NONE mode -> eager
# model.precompute_and_store_context_kv).
#
# Env: VLLM_K3_DISABLE_CONTEXT_GRAPHS (read via os.getenv in
# speculator.py — no envs.py change):
#   "1" (DEFAULT) = context-KV graphs DISABLED, eager precompute
#                    (fork-converged design; the mod is live by default
#                    because the capture phase crashes the boot)
#   "0"/"false"   = restore the #251 context-KV graph capture (A/B only)
# A loud INFO at speculator init states the active mode.
#
# Prerequisites: the DFlash/DSpark speculator lineage markers (fail-loud
# inside the patch script).  Builds on the b5 image state (b4 +
# v4plus-cgfix); this file was not touched by any earlier v4plus mod, so
# no further batch markers apply.
#
# Usage: run inside the v4-prd container (or with VLLM_ROOT pointing at
# the image tree).  Idempotent.
#
# Date: 2026-09-04

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- vllm tree ------------------------------------------------------------
if [ -n "${VLLM_ROOT:-}" ]; then
    :
elif [ -d /opt/kimi-k3/vllm/vllm ]; then
    VLLM_ROOT=/opt/kimi-k3/vllm/vllm
else
    echo "ERROR: vllm tree not found (set VLLM_ROOT)" >&2
    exit 1
fi
export VLLM_ROOT
echo "VLLM_ROOT=${VLLM_ROOT}"

echo ""
echo "=== CGFIX2: disable DFlash context-KV graph capture ==="
python3 "${here}/patch_disable_context_graphs.py"

echo ""
echo "=== v4plus-cgfix2 complete ==="
echo ""
echo "Serving flags (unchanged from cgfix):"
echo "  --compilation-config '{\"cudagraph_mode\":\"FULL\",\"max_cudagraph_capture_size\":30}'"
echo "  or the small-size test:"
echo "  --compilation-config '{\"cudagraph_mode\":\"FULL\",\"cudagraph_capture_sizes\":[1,2,4,8]}'"
echo "Context-KV mode env:"
echo "  VLLM_K3_DISABLE_CONTEXT_GRAPHS=1  (default, eager precompute)"
echo "  VLLM_K3_DISABLE_CONTEXT_GRAPHS=0  (restore #251 context graphs, A/B only)"
echo ""
echo "Dry-run checks for the orchestrator:"
echo "  1. The patch script must NOT print 'PREREQUISITE FAILED'."
echo "  2. At speculator init expect the new INFO line:"
echo "     'DSpark context-KV precompute runs EAGERLY outside CUDA graphs"
echo "      (VLLM_K3_DISABLE_CONTEXT_GRAPHS=1).'"
echo "  3. Boot must proceed past 'Capturing dspark CUDA graphs (FULL)'"
echo "     WITHOUT any 'Capturing DFlash context-KV CUDA graphs' phase"
echo "     appearing at all (the phase is skipped, not failed)."
echo "  4. Serving: the context-KV precompute now runs eagerly per step —"
echo "     watch decode throughput at nst=6; if the eager precompute costs"
echo "     measurably, A/B VLLM_K3_DISABLE_CONTEXT_GRAPHS=0 is NOT advised"
echo "     on this image (it crashes); instead report the delta and we"
echo "     will look at batching the eager path."
echo "  5. The 64K quality gate must pass (eager precompute is the"
echo "     numerically identical path — same kernel, same inputs, just"
echo "     outside a graph)."
