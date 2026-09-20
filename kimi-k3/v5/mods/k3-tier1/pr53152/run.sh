#!/usr/bin/env bash
# pr53152 — backport of upstream vLLM PRs #53152 + #53327 (kernel side).
# Fuse the top-k finalization into the K3 latent-MoE tail collective:
# KimiK3LatentMoETailOp.initialize(..., experts_per_token=K) configures the
# collective to consume UnfinalizedMoEOutput directly. Production defer
# plumbing (trtllm experts + FusedMoE config) is NOT ported — the fork
# lacks that infrastructure; see PORT-REPORT.md.
# NOTE: pr54168 (low-M tail optimization) must be applied AFTER this mod.
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
  echo "[pr53152] PREREQUISITE FAILED: VLLM_ROOT not resolvable (no envs.py)"
  exit 1
fi
echo "[pr53152] VLLM_ROOT=$VLLM_ROOT"

python3 "$SCRIPT_DIR/patch_topk_fuse.py" "$VLLM_ROOT"
rc=$?

echo "[pr53152] dry-run checklist:"
echo "  1. No 'PREREQUISITE FAILED' above."
echo "  2. APPLY for the new moe_output.py + latent_moe_tail.py + the"
echo "     allreduce collective, or SKIP (already present)."
echo "  3. NOTE lines about defer plumbing are EXPECTED (fork lacks the"
echo "     deferred-finalize MoE stack; kernel-side fusion only)."
echo "  4. Tail-fusion capacity is raised to 128 tokens — watch tier-0"
echo "     (TAIL_FUSION) engagement up to M=128 after warmup."
echo "  5. Apply pr54168 AFTER this mod (it builds on the top_k block)."
exit $rc
