#!/usr/bin/env bash
# fix-k3-kda-spec-token-init — initialize spec partition offsets on the
# no-spec path of the KDA metadata builder (UnboundLocalError: spec_token_start).
# Pure bug fix, no env gates. See patch_kda_spectoken.py for the root cause.
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -ne 0 ]; then
  echo "usage: run.sh (no args; resolves VLLM_ROOT automatically)"
  exit 2
fi

# Resolve VLLM_ROOT the way batch1 does: installed vllm package location.
VLLM_ROOT="$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)"
if [ -z "$VLLM_ROOT" ] || [ ! -f "$VLLM_ROOT/envs.py" ]; then
  for cand in /opt/kimi-k3/vllm/vllm /opt/vllm/vllm; do
    if [ -f "$cand/envs.py" ]; then VLLM_ROOT="$cand"; break; fi
  done
fi
if [ ! -f "$VLLM_ROOT/envs.py" ]; then
  echo "[fix-k3-kda-spec-token-init] PREREQUISITE FAILED: VLLM_ROOT not resolvable (no envs.py)"
  exit 1
fi
echo "[fix-k3-kda-spec-token-init] VLLM_ROOT=$VLLM_ROOT"

python3 "$SCRIPT_DIR/patch_kda_spectoken.py" "$VLLM_ROOT"
rc=$?

echo "[fix-k3-kda-spec-token-init] dry-run checklist:"
echo "  1. No 'PREREQUISITE FAILED' above."
echo "  2. APPLY line for models/kimi_k3/nvidia/kda_metadata.py, or SKIP (already present)."
echo "  3. Serving: the cudagraph-memory profiling capture (determine_available_memory)"
echo "     with zero spec sequences must no longer raise UnboundLocalError: spec_token_start."
exit $rc
