#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# V4PLUS CGFIX3 — run the DSpark draft speculator EAGERLY, keeping the
# TARGET model's FULL CUDA graphs.
#
# Fixes the v6 boot crash: all capture phases complete cleanly, then the
# first post-capture warmup_kernels decode step faults — fork issue #396
# (replay of the speculator's own captured draft-decode FULL graph:
# _multi_step_decode -> run_fullgraph -> illegal access; open/unfixed
# upstream; the fork's K3 production recipe still runs --enforce-eager).
#
# Design: force wants_full off in init_cudagraph_manager when
# VLLM_K3_EAGER_DRAFT=1 (the default). This selects the exact
# configuration the existing "draft attention does not support full CUDA
# graphs" path already produces in production: the query graph manager is
# constructed with mode NONE, stages no descriptors, captures nothing,
# and dispatch always returns NONE — the draft runs eagerly. The TARGET
# model's FULL graphs (separate manager, owned by the model runner) are
# untouched. The cgfix2 context-KV manager gate also reads wants_full, so
# an eager draft is graph-free end to end.
#
# Env: VLLM_K3_EAGER_DRAFT (direct os.getenv in speculator.py):
#   "1" (DEFAULT) = eager draft speculator; target keeps FULL graphs
#   "0"/"false"   = restore the full-graph speculator (A/B only; known
#                   to crash the first decode step on this image via #396)
#
# Prerequisites: the v6 post-state (b5 + mods/v4plus-cgfix2) — the script
# fails loud without the cgfix2 markers.
#
# GROUND-TRUTH CAVEAT: the /tmp/opencode/k3spec extracts were wiped
# before this mod was authored. The patch anchor is transcribed
# byte-exact from the cgfix2-session read of the real b5 file (a region
# cgfix2 does not modify), and the reported simulation ran against a
# labeled RECONSTRUCTION of the v6 post-state. RE-EXTRACT
# vllm/v1/worker/gpu/spec_decode/dflash/speculator.py from the v6 image
# and re-run this script against it (idempotent) before baking.
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
echo "=== CGFIX3: eager DSpark draft, target keeps FULL graphs ==="
python3 "${here}/patch_eager_draft.py"

echo ""
echo "=== v4plus-cgfix3 complete ==="
echo ""
echo "Serving flags (unchanged):"
echo "  --compilation-config '{\"cudagraph_mode\":\"FULL\",\"max_cudagraph_capture_size\":30}'"
echo "  or the small-size test:"
echo "  --compilation-config '{\"cudagraph_mode\":\"FULL\",\"cudagraph_capture_sizes\":[1,2,4,8]}'"
echo "Draft mode env:"
echo "  VLLM_K3_EAGER_DRAFT=1  (default, eager draft; target FULL graphs)"
echo "  VLLM_K3_EAGER_DRAFT=0  (full-graph speculator, A/B only — #396 crash)"
echo ""
echo "Dry-run checks for the orchestrator:"
echo "  1. The patch script must NOT print 'PREREQUISITE FAILED' (requires"
echo "     the cgfix2 post-state)."
echo "  2. At speculator init expect the new INFO line:"
echo "     'DSpark draft speculator runs EAGERLY (VLLM_K3_EAGER_DRAFT=1);"
echo "      target model keeps FULL CUDA graphs.'"
echo "  3. The 'Capturing dspark CUDA graphs' phase must DISAPPEAR from the"
echo "     boot log (needs_capture() is False), while the target's FULL"
echo "     capture phases remain and complete."
echo "  4. The post-capture warmup_kernels decode step must now pass — that"
echo "     is the #396 crash site. If it still faults with the speculator"
echo "     fully eager, the fault is NOT the draft graph replay; capture"
echo "     the exact phase and stack."
echo "  5. Throughput: the draft now runs eagerly per step (launch-overhead"
echo "     only; the draft is a small share of step time at nst<=6). A/B"
echo "     against --enforce-eager: expect most of the target-graph win to"
echo "     be retained. Do NOT A/B VLLM_K3_EAGER_DRAFT=0 on this image."
echo "  6. RE-EXTRACT speculator.py from the v6 image and re-run this"
echo "     script (idempotent) to confirm the anchor against the real file"
echo "     — the authored simulation used a labeled reconstruction because"
echo "     the k3spec extracts were wiped."
