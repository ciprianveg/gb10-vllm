#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# V4PLUS ROCANTE-DCP — RoCEnante (b12x.comm.roce RDMA one-shot) for the
# DCP-group collectives (DCP8 serving).
#
# What it does (patch_rocante_dcp.py, cuda_communicator.py only):
#   - The DCP group's CudaCommunicator gets its own B12xRoceAllReduce
#     instance (the batch1 M2 shim, group-parameterized ctor) — the same
#     adapter pattern as the TP hook, which is untouched.
#   - Wired collectives (ride the per-instance dispatch): the decode-path
#     query head-gather fallback (cp_group.all_gather dim=1), the generic
#     MLA query gather, and any group all-reduce legs (e.g. the ag_rs LSE
#     variant's AR legs) — under the shim's existing size caps
#     (VLLM_ROCE_ALLREDUCE_MAX_SIZE / VLLM_ROCE_ALLGATHER_MAX_SIZE).
#   - NOT wired (no b12x.comm.roce surface — #295 is AR+AG only; stays
#     exact NCCL): the a2a LSE reduce (dist.all_to_all_single), any
#     reduce-scatter legs, and the direct torch.distributed KV shard
#     gather (context path, above caps anyway).
#   - The fork's CUDA-IPC DCP transports (b12x.comm.pcie) remain
#     preferred when they can initialize (same-node); on cross-node DCP8
#     they self-disable and this mod's RDMA path serves the fallback.
#
# Env: VLLM_ENABLE_ROCE_DCP (direct os.getenv in cuda_communicator.py).
#   Default baked at patch time from a healthy-probe of the image's
#   B12xRoceAllReduce shim ("1" healthy / "0" not); runtime overrides the
#   bake. A/B with VLLM_ENABLE_ROCE_DCP=0.
#
# Prerequisites: the batch1 M2 post-state in cuda_communicator.py
# (fail-loud in the script).
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
echo "=== ROCANTE-DCP: RoCEnante for the DCP group ==="
python3 "${here}/patch_rocante_dcp.py"

echo ""
echo "=== v4plus-rocante-dcp complete ==="
echo ""
echo "Dry-run checks for the orchestrator:"
echo "  1. The patch script must NOT print 'PREREQUISITE FAILED' (requires"
echo "     the batch1 M2 post-state)."
echo "  2. The shim-probe line must say HEALTHY -> default '1'. If UNHEALTHY,"
echo "     the wiring is baked OFF (NCCL preserved); fix the shim first."
echo "  3. Boot log: expect the new INFO line"
echo "     'RoCEnante DCP collectives enabled for group <name>' — the printed"
echo "     name resolves the DCP group's unique_name (the gate matches any"
echo "     non-TP name containing 'cp'). If NO line appears, the DCP group's"
echo "     name differs — report it and the gate will be adjusted."
echo "  4. Boot log: the per-communicator backend-selection lines should now"
echo "     list B12X_ROCENANTE TWICE (TP group + DCP group)."
echo "  5. QUALITY GATE: run the 64K gate — the wired collectives are exact"
echo "     replacements (same semantics, one-shot RDMA), but this is the"
echo "     first cross-group use of the roce pool; two concurrent comm"
echo "     instances (TP + DCP) is the key thing the gate exercises."
echo "  6. A/B at DCP8 short-ctx decode: VLLM_ENABLE_ROCE_DCP=1 (default)"
echo "     vs =0, against the 13.85 baseline (cp=1 reference: 15.86)."
echo "  7. If the shim's construction fails for the DCP group at boot, the"
echo "     instance comes up disabled and dispatch falls through to exact"
echo "     NCCL — capture any 'B12X PCIe DCP collective initialization"
echo "     failed'-style warning or b12x roce errors and report them."
echo "  8. The a2a LSE reduce and reduce-scatter legs stay NCCL by design"
echo "     (no roce surface) — do not expect those call sites to change."
