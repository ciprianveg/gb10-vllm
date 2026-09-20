#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# V4PLUS Batch 4 — 8-row fused DSpark verify tile (nst=4..7, practically
# nst=6) with auto-selection by nst.
#
# PREREQUISITE: v4plus-batch2 MUST be applied first (both sides):
#   - b12x: the #271 fused-verify surface (uses_query_cache_seqlens) and
#     compile-spec version 5  -> patch_b12x_verify_tile.py fails loud
#     without them;
#   - vllm: B1b's fused verify plans + the VLLM_K3_BUCKETED_DECODE fix
#     -> patch_vllm_verify_tile.py fails loud without them.
#
# What it does: extends fused verification from the 4-row request (nst=3)
# to an 8-row request span. A literal 8-row CTA tile is physically
# impossible (1056 threads > 1024; ~126 KiB smem > the device opt-in
# budget), so an 8-row request runs as TWO of the proven 4-row query
# tiles (new b12x kernel param `tiles_per_request`): a request's KV
# chunks are read twice per verify sweep instead of once per row (nst=6
# flat: 7x). Padded tail rows live outside the request's cu_seqlens span
# and are never computed, written, or read.
#
# Env gates:
#   VLLM_K3_FUSED_VERIFY   master gate (unchanged; probe-baked default)
#   VLLM_K3_FUSED_TILE     8 (default) = 8-row family for nst=4..7 AND
#                          4-row family for nst=3; 4 = nst=3 only;
#                          0 = fused verify disabled entirely
#   VLLM_K3_BUCKETED_DECODE  untouched (batch2 fix)
#
# Usage: run inside the v4-prd container (or with VLLM_ROOT/B12X_ROOT
# pointing at the image trees), AFTER v4plus-batch2.  Idempotent.
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

# --- b12x package (resolved at runtime, like batch2) ----------------------
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
echo "=== B4a: b12x 8-row fused verify requests (tiles_per_request) ==="
python3 "${here}/patch_b12x_verify_tile.py"

echo ""
echo "=== B4b: vllm 8-row fused verify plans + nst auto-select ==="
python3 "${here}/patch_vllm_verify_tile.py"

echo ""
echo "=== v4plus-batch4 complete ==="
echo "Dry-run checks for the orchestrator:"
echo "  1. PREREQ: both scripts must NOT print 'PREREQUISITE FAILED' — if"
echo "     they do, apply mods/v4plus-batch2 first."
echo "  2. JIT: the compile-spec version bump (5 -> 6) re-JITs EVERY"
echo "     dense-MLA forward kernel on first boot (including the proven"
echo "     4-row ones) — expect a one-time startup latency increase."
echo "  3. nst=6 + fp8 KV + dcp=1 + VLLM_K3_FUSED_TILE=8 (default): expect"
echo "     the 'Kimi-K3 fused 8-row (two-tile) DSpark verification is"
echo "     ENABLED' warning at builder init, then verify the fused path"
echo "     actually engages (no fallback-to-flat info line)."
echo "  4. QUALITY GATE: this path is NEW and never qualified in the fork."
echo "     First boot + 64K quality gate must pass. The padded-row"
echo "     semantics (rows outside the cu_seqlens span -> query_valid=0 ->"
echo "     output AND lse stores skipped) were verified structurally in"
echo "     the b12x source, NOT numerically — the acceptance logic must"
echo "     read only the first nst+1 rows of each 8-row span (the vllm"
echo "     gather enforces this; a mismatch would surface as corrupted"
echo "     verify outputs at nst != 7)."
echo "  5. A/B: nst=6 fused-8 (this mod) vs b3-s6 flat (14.09 tok/s"
echo "     champion). Also A/B VLLM_K3_FUSED_TILE=4 (nst=6 falls back to"
echo "     flat) and =0 (fused fully off) to isolate the effect."
echo "  6. nst=3 regression check: the 4-row path must be byte-identical"
echo "     (const_expr tiles_per_request == 1 keeps the old request"
echo "     mapping); b3-k3 numbers (13.26 tok/s, acceptance 3.70-3.73/4)"
echo "     must not move."
echo "  7. If boot fails with 'tiled verify plans require ...' from"
echo "     b12x Caps validation, capture it — it means an 8-row plan was"
echo "     created with a shape this mod's validation rejects (bug in the"
echo "     family creation, not a serving-path issue)."
