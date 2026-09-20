#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# V4PLUS DCPFIX — un-gate the 8-row fused verify family at DCP>1
# (DCP8 serving).
#
# Changes (patch_ungate_verify8_dcp.py, b12x_mla.py only):
#   1. The 8-row verify plan family is created at ANY dcp_size (drop the
#      batch4 dcp==1 condition).
#   2. The strided scatter/gather path in forward_mqa accepts dcp>1
#      (drop the RuntimeError); the DCP composition is the same machinery
#      the 4-row family already uses in production at DCP8: per-row LOCAL
#      visible lens from the DCP branch of _materialize_query_cache_seq_lens,
#      the DCP head gather upstream of the scatter, and the per-rank LSE
#      reduce downstream of the compact gather.
#
# NOT included: the #565 mla.py DCP query-replication hunks. BLOCKED ON
# ARTIFACTS — /tmp/opencode/patches/v4plus/ (the #565 raw diffs) and the
# image's mla.py were wiped from /tmp before this task, and no surviving
# session read those hunks. Per the audit, the B12X fused-verify path
# does not consume them (they carry the query-replication machinery for
# the NON-B12X MLA backend: DCPGroupColumnParallelLinear plumbing and the
# qrep-driven head-check relocation). If that backend ever runs under
# DCP, re-extract and port them separately.
#
# GROUND-TRUTH CAVEAT: the v4img extracts were also wiped. The patch
# anchors are transcribed byte-exact from this conversation's own batch4
# patch text and post-state reads; the reported simulation ran against a
# labeled RECONSTRUCTION of the b4 post-state. RE-EXTRACT
# vllm/v1/attention/backends/mla/b12x_mla.py from the image and re-run
# this script (idempotent) before baking.
#
# Prerequisites: the batch2+batch4 post-state (fail-loud in the script).
#
# Usage: run inside the v4-prd container (or with VLLM_ROOT pointing at
# the image tree).  Idempotent.
#
# Date: 2026-09-05

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
echo "=== DCPFIX: un-gate 8-row fused verify at DCP>1 ==="
python3 "${here}/patch_ungate_verify8_dcp.py"

echo ""
echo "=== v4plus-dcpfix complete ==="
echo ""
echo "Dry-run checks for the orchestrator:"
echo "  1. The patch script must NOT print 'PREREQUISITE FAILED' (requires"
echo "     the batch2+batch4 post-state)."
echo "  2. At DCP8 + nst=6 + fp8 KV, expect the 'Kimi-K3 fused 8-row"
echo "     (two-tile) DSpark verification is ENABLED' warning at builder"
echo "     init — it must now fire at DCP8 (it previously only fired at"
echo "     dcp=1)."
echo "  3. QUALITY GATE FIRST: this is the first time the 8-row strided"
echo "     path runs under DCP — run the 64K quality gate BEFORE any"
echo "     throughput bench. The 4-row family's DCP8 pass is the template,"
echo "     but the strided scatter/gather under the DCP head gather is"
echo "     newly exercised."
echo "  4. Watch decode at DCP8 nst=6: the tiled path must engage (no"
echo "     fallback-to-flat info line) and throughput should be compared"
echo "     against the DCP8 flat baseline."
echo "  5. If boot raises 'B12X_MLA strided fused verify ...' (the old"
echo "     dcp=1 RuntimeError), the image is not running this mod's"
echo "     post-state."
echo "  6. RE-EXTRACT b12x_mla.py from the image and re-run this script"
echo "     (idempotent) to confirm the anchors against the real file."
echo "  7. MISSING ARTIFACTS for the #565 mla.py port (task 1):"
echo "     a. /tmp/opencode/patches/v4plus/ — the #565 raw diffs"
echo "        (raw_cadc3ed3f.patch / vllm_565_cadc3ed3f.patch and"
echo "        vllm_565_d461572be.patch),"
echo "     b. the image's mla.py — the exact path is named by the diff's"
echo "        'diff --git' headers (likely vllm/v1/attention/backends/mla/"
echo "        mla.py)."
echo "     Only needed if the non-B12X MLA backend runs under DCP."
