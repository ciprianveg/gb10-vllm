#!/usr/bin/env bash
# V4PLUS-MARLIN-NOPAD — env-gated skip of the KimiMoE 192->256 pad.
# Default OFF (VLLM_K3_MARLIN_NOPAD=0) = stock padded-4096 behavior.
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
  echo "[fix-k3-marlin-nopad] PREREQUISITE FAILED: VLLM_ROOT not resolvable (no envs.py)"
  exit 1
fi
echo "[fix-k3-marlin-nopad] VLLM_ROOT=$VLLM_ROOT"

python3 "$SCRIPT_DIR/patch_marlin_nopad.py" "$VLLM_ROOT"
rc=$?

# Part 2: expert-layer marlin roundup (round_up 192->256) — must be skipped
# together with the model-level pad, or create_weights re-inflates to the
# same ~118.9 GiB/rank.
python3 "$SCRIPT_DIR/patch_marlin_roundup.py" "$VLLM_ROOT"
rc2=$?
[ "$rc" -eq 0 ] && rc=$rc2

echo "[fix-k3-marlin-nopad] dry-run checklist:"
echo "  1. No 'PREREQUISITE FAILED' above."
echo "  2. APPLY lines for model.py AND oracle/mxfp4.py (or SKIP each)."
echo "  3. Serving: VLLM_K3_MARLIN_NOPAD unset/0 = stock padded path. Test with \"1\" stacked with the in-place repack mod."
echo "  4. Boot gate (marlin TP-only, no EP) must clear load WITHOUT the 118.9 GiB footprint."
exit $rc
