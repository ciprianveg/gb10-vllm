#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# V4PLUS Batch 2 — next speed items for the v4-prd fork image.
#
# Target: v4-prd (fork vLLM @881ac39a4 at /opt/kimi-k3/vllm, b12x
# #124/#138/#139-era at /opt/kimi-k3/b12x) WITH v4plus-batch1 applied.
# Serving: TP8, no DCP, fp8 KV, dspark (nst recipe-controlled), B12X_MLA,
# marlin.  Baseline after Batch 1: 13.3 tok/s.
#
# Items (in order):
#   B1  b12x #271 fused 4-query verify kernel (dense_mla)  — patch_b12x_fused_verify.py
#   B1  fork #565 fused verify plans (b12x_mla.py + envs) — patch_vllm_fused_verify.py
#       (env-gated: VLLM_K3_FUSED_VERIFY, default from a b12x probe; only
#        active at nst=3 — inert at nst=6; decode-plan bucketing gated on
#        VLLM_K3_BUCKETED_DECODE, default OFF — b2 regressed -8% at nst=6
#        with it always-on; opt back in with =1)
#   B2  #570 fp8 draft weights + fp8 draft head            — patch_fp8_draft.py
#       (env-gated: VLLM_K3_FP8_DRAFT, default ON, self-disabling fallbacks)
#   B3  complete batch1's inert M4b workspace reserve      — patch_workspace_reserve.py
#   B4  f71e0deaa dynamic draft depth                      — patch_dynamic_depth.py
#       (DOCUMENTED SKIP: applicability check only)
#
# Usage: run inside the v4-prd container (or with VLLM_ROOT/B12X_ROOT
# pointing at the image trees).  Idempotent: re-running skips applied hunks.
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

# --- b12x package (resolved at runtime, like batch1) ----------------------
if [ -z "${B12X_ROOT:-}" ]; then
    B12X_ROOT="$(python3 -c 'import b12x, os; print(os.path.dirname(b12x.__file__))')" \
        || B12X_ROOT=""
    if [ -z "${B12X_ROOT:-}" ] && [ -d /opt/kimi-k3/b12x/b12x ]; then
        B12X_ROOT=/opt/kimi-k3/b12x/b12x
    fi
fi
if [ -z "${B12X_ROOT:-}" ] || [ ! -d "${B12X_ROOT}" ]; then
    echo "ERROR: b12x package not resolvable (set B12X_ROOT)" >&2
    exit 1
fi
export B12X_ROOT
echo "B12X_ROOT=${B12X_ROOT}"

echo ""
echo "=== B1a: b12x #271 fused 4-query verify kernel (dense_mla) ==="
python3 "${here}/patch_b12x_fused_verify.py"

echo ""
echo "=== B1b: fork #565 fused verify plans (b12x_mla.py + envs) ==="
python3 "${here}/patch_vllm_fused_verify.py"

echo ""
echo "=== B2: #570 fp8 draft weights + fp8 draft head ==="
python3 "${here}/patch_fp8_draft.py"

echo ""
echo "=== B3: complete batch1's M4b workspace reserve ==="
python3 "${here}/patch_workspace_reserve.py"

echo ""
echo "=== B4: f71e0deaa dynamic draft depth (documented skip) ==="
python3 "${here}/patch_dynamic_depth.py"

echo ""
echo "=== v4plus-batch2 complete ==="
echo "Dry-run checks for the orchestrator:"
echo "  1. B1: 'VLLM_K3_FUSED_VERIFY default ON/OFF' probe line — ON only when"
echo "     the b12x #271 surface is present (B1a applied first)."
echo "  2. B1: at builder init with nst=3 + fp8 KV, expect the 'Kimi-K3 fused"
echo "     4-query DSpark verification is ENABLED' warning; at nst=6 expect the"
echo "     'configured ON but inactive' info line and NO behavior change."
echo "  3. B1 RE-QUALIFICATION: cp=1 was not the fork's qualified config for"
echo "     the fused plans — first boot + 64K quality gate must pass."
echo "  4. B2: expect 'DSpark draft linear layers use online fp8_per_channel'"
echo "     (only when the draft would otherwise be BF16) and the rowwise-fp8"
echo "     head info line; VLLM_K3_FP8_DRAFT=0 A/Bs the BF16 draft."
echo "  5. B2 FIX: 'DSpark draft linear layers use online fp8_per_channel' now"
echo "     requires BOTH quantization fields — if boot instead logs the"
echo "     'online fp8 draft resolution failed ... falling back to BF16'"
echo "     warning, capture it (the fallback keeps serving safe)."
echo "  6. B2 FIX (A/B): VLLM_K3_BUCKETED_DECODE=1 vs default OFF at nst=6 —"
echo "     the default must restore the pre-B1b single decode plan (expect"
echo "     control-level 13.2+ tok/s; =1 re-qualifies the bucketing)."
echo "  7. B3: the 'Kimi-K3 retained %.2f MiB/rank' line must appear at"
echo "     startup (proves the load_weights-path reserve runs)."
