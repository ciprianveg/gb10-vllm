#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# V4PLUS CGFIX — CUDA-graph capture fixes (upstream fork PRs #617 + #628,
# Sep 3 2026) for the v4-plus-b4 image.
#
# Fixes the capture-time cudaErrorIllegalAddress when enabling CUDA graphs
# (removing --enforce-eager) with DSpark spec + B12X_MLA + marlin MXFP4,
# TP8 RoCE, cudagraph_capture_sizes [1,2,4,8].
#
#   CG1 (#617, patch_cudagraph_sync.py):
#     - torch.accelerator.synchronize() after every uncaptured warmup
#       forward, BEFORE capture begins (verbatim upstream hunk);
#     - the same synchronize after our lineage's B12X prewarm forward
#       (lineage-adapted — that uncaptured forward is specific to our
#       tree);
#     - monitor.is_cudagraph_capturing_enabled() exposed (verbatim; the
#       API aux-stream overlap code uses to disable overlap during graph
#       preparation).
#
#   CG2 (#628 functional, patch_planned_token_counts.py):
#     - CudaGraphManager.planned_token_counts(): every model-row count
#       staged for capture, INCLUDING the spec verifier shapes
#       batch x (nst+1) that the scheduler capture-size list never
#       registers;
#     - capture() logs the planned row-count set at boot with the fork
#       #451 contract.  Our lineage has no centralized b12x_warmup.py to
#       union into — the per-descriptor warmup/prewarm forwards ARE the
#       registration, backed by guard_b12x_kernel_resolution and the
#       B12X_MLA uncompiled-layout fail-loud guard.
#
#   #619 (KDA gate side-stream overlap, +343 lines): NOT PORTED.  Our
#   tree serves Kimi K3 (MLA + marlin MoE) and carries no GLM5Next KDA
#   gate side-stream.  The one aux-stream feature we do carry — the
#   offloader prefetch copy-stream — is already capture-synchronized at
#   exactly the boundaries #619 worries about (sync_prev_onload before
#   capture and before replay, join_after_forward inside the capture
#   scope), and #617's synchronize now covers the warmup forwards.
#
# Prerequisites: the kimi.k3.aligned lineage markers in the target files
# (fail-loud inside the patch scripts).  The v4plus batch state (batch1/2/4)
# lives in OTHER files this mod does not touch; run.sh reports its presence
# as a soft check (the CGFIX patches are orthogonal to the batch mods).
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

# --- soft check: v4plus batch state (b4 post-state) ------------------------
# The CGFIX mod does not touch these files; this is a boot-context report,
# not a gate.
echo ""
echo "=== v4plus batch-state report (soft check) ==="
for marker_file in "envs.py:VLLM_K3_FUSED_TILE" \
                   "v1/attention/backends/mla/b12x_mla.py:_dense_mla_verify8_plans" \
                   "model_executor/models/utils.py:_ONLINE_SHORTHANDS"; do
    f="${VLLM_ROOT}/${marker_file%%:*}"
    marker="${marker_file#*:}"
    if [ -f "$f" ] && grep -q "$marker" "$f"; then
        echo "  [batch] OK    ${marker_file%%:*} carries ${marker}"
    else
        echo "  [batch] MISS  ${marker_file%%:*} lacks ${marker} (batch mod not applied?)"
    fi
done

echo ""
echo "=== CG1: fork #617 — synchronize auxiliary warmup streams ==="
python3 "${here}/patch_cudagraph_sync.py"

echo ""
echo "=== CG2: fork #628 — planned token counts (B12X warmup contract) ==="
python3 "${here}/patch_planned_token_counts.py"

echo ""
echo "=== v4plus-cgfix complete ==="
echo ""
echo "Recommended serving flags (proven graph config on this hardware):"
echo "  --compilation-config '{\"cudagraph_mode\":\"FULL\",\"max_cudagraph_capture_size\":30}'"
echo "  (GLM v19 production recipe, same GB10/RoCE stack; no custom sizes list.)"
echo "Your test starts with small sizes instead:"
echo "  --compilation-config '{\"cudagraph_mode\":\"FULL\",\"cudagraph_capture_sizes\":[1,2,4,8]}'"
echo "Both are supported by this mod."
echo ""
echo "Dry-run checks for the orchestrator:"
echo "  1. Both scripts must NOT print 'PREREQUISITE FAILED'."
echo "  2. At capture start expect the new log line:"
echo "     'CUDA graph capture preparing model-row counts [...]' — verify the"
echo "     list includes the spec verifier shapes batch x (nst+1) for every"
echo "     capture size (e.g. nst=6, sizes [1,2,4,8] -> 7, 14, 28, 56 present)"
echo "     AND the plain decode sizes."
echo "  3. CG1 verification: capture must proceed past the first descriptor"
echo "     without cudaErrorIllegalAddress (the #617 signature was a crash"
echo "     AT capture-begin; three consecutive clean FULL-graph startups was"
echo "     the upstream validation bar)."
echo "  4. If capture instead fails with 'B12X_MLA encountered an uncompiled"
echo "     layout during CUDA graph capture', that is the #628 class surfacing"
echo "     through our lineage's fail-loud guard (a warmup did not cover a"
echo "     staged row count) — capture the log line from check 2 and the"
echo "     descriptor; the fix would be extending the B12X prewarm, which"
echo "     needs vllm/compilation/b12x_capture.py extracted from the image."
echo "  5. #619 NOT ported (no KDA side-stream in this tree; the offloader"
echo "     copy-stream is already capture-synchronized). If a future config"
echo "     enables any VLLM_USE_B12X_* kernel family with side-stream overlap,"
echo "     re-evaluate."
echo "  6. A/B: --enforce-eager vs FULL graphs on the s6 bench; expect the"
echo "     graph win to show at sustained decode."
