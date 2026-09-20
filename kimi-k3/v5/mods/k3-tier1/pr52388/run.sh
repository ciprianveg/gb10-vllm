#!/usr/bin/env bash
# pr52388 — upstream vLLM PR #52388 (Mamba metadata prep optimization).
# ALREADY PRESENT in the b4f fork baseline: this mod is a verification-only
# no-op that fails loudly if the feature ever goes missing from the tree.
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
  echo "[pr52388] PREREQUISITE FAILED: VLLM_ROOT not resolvable (no envs.py)"
  exit 1
fi
echo "[pr52388] VLLM_ROOT=$VLLM_ROOT"

python3 "$SCRIPT_DIR/patch_mamba_metadata.py" "$VLLM_ROOT"
rc=$?

echo "[pr52388] dry-run checklist:"
echo "  1. SKIP lines for all 5 required snippets = feature already present."
echo "  2. Any NOTE line means the fork baseline lost upstream #52388 —"
echo "     investigate before applying the rest of k3-tier1."
exit $rc
