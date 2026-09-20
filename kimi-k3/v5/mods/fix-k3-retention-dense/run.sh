#!/usr/bin/env bash
# fix-k3-retention-dense — backport of upstream #55760/#55861.
# Sparse retention (0) + hybrid model + EAGLE-style spec decode -> dense.
# Escape hatch: VLLM_K3_RETENTION_ALLOW_SPARSE=1. Prefix caching untouched.
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
  echo "[fix-k3-retention-dense] PREREQUISITE FAILED: VLLM_ROOT not resolvable (no envs.py)"
  exit 1
fi
echo "[fix-k3-retention-dense] VLLM_ROOT=$VLLM_ROOT"

python3 "$SCRIPT_DIR/patch_retention_dense.py" "$VLLM_ROOT"
rc=$?

echo "[fix-k3-retention-dense] dry-run checklist:"
echo "  1. No 'PREREQUISITE FAILED' above."
echo "  2. APPLY line for v1/core/kv_cache_coordinator.py, or SKIP (already present)."
echo "  3. Serving: with VLLM_PREFIX_CACHE_RETENTION_INTERVAL=0 on K3+spec, expect the"
echo "     'overriding to dense retention' warning at boot. VLLM_K3_RETENTION_ALLOW_SPARSE=1 restores sparse."
echo "  4. Long-ctx gate: rerun the >200K context bench — sawtooth should not recur;"
echo "     prefix-cache hit rate should become nonzero at replay boundaries."
exit $rc
