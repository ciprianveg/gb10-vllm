#!/usr/bin/env bash
# pr56159 — backport of upstream vLLM PR #56159.
# Avoid the KDA mixed-batch gather/scatter: when spec and non-spec tokens
# are contiguous in the scheduled batch, slice inputs and write attention
# outputs straight into core_attn_out (no index_select / index_copy_).
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
  echo "[pr56159] PREREQUISITE FAILED: VLLM_ROOT not resolvable (no envs.py)"
  exit 1
fi
echo "[pr56159] VLLM_ROOT=$VLLM_ROOT"

python3 "$SCRIPT_DIR/patch_kda_mixed_batch.py" "$VLLM_ROOT"
rc=$?

echo "[pr56159] dry-run checklist:"
echo "  1. No 'PREREQUISITE FAILED' above."
echo "  2. APPLY for kda_metadata.py, kda.py, chunk.py, fused_recurrent.py,"
echo "     or SKIP (already present)."
echo "  3. Serving (spec decode + mixed regular/spec batches): contiguous"
echo "     batches skip the index_select gathers and index_copy_ scatter;"
echo "     expect a small mixed-batch decode win."
exit $rc
