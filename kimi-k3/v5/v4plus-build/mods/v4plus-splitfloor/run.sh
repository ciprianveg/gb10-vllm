#!/usr/bin/env bash
# V4PLUS-SPLITFLOOR — env-gated minimum-split floor for b12x dense-MLA.
# Fills ~48 SMs (GB10) at small batch. Default OFF (VLLM_K3_MIN_SPLITS=0).
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
  echo "[v4plus-splitfloor] PREREQUISITE FAILED: VLLM_ROOT not resolvable (no envs.py)"
  exit 1
fi
echo "[v4plus-splitfloor] VLLM_ROOT=$VLLM_ROOT"

python3 "$SCRIPT_DIR/patch_splitfloor.py" "$VLLM_ROOT"
rc=$?

echo "[v4plus-splitfloor] dry-run checklist:"
echo "  1. No 'PREREQUISITE FAILED' above."
echo "  2. APPLY lines for envs.py (field+lambda) and b12x_mla.py (floor), or SKIP (already present)."
echo "  3. Serving: VLLM_K3_MIN_SPLITS unset/0 = current behavior. Start A/B with \"2\" at batch 1-4."
echo "  4. Confirm the 'B12X_MLA split floor active' info line appears when the floor engages."
echo "  5. Quality gate (keyword test) + bench vs floor-off on the same recipe."
exit $rc
