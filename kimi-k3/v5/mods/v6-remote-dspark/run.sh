#!/usr/bin/env bash
# v6-remote-dspark — port of myshytf/vllm@a653e74 (remote DSpark draft).
# Adds a ZMQ-based RemoteK3DSparkSpeculator + standalone K3 draft server so
# the Kimi-K3 draft model can run on a dedicated GPU (e.g. RTX 3090).
# Engagement: VLLM_K3_DRAFT_REMOTE_ADDRESS / VLLM_K3_DSPARK_REMOTE_ADDRESS.
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
  echo "[v6-remote-dspark] PREREQUISITE FAILED: VLLM_ROOT not resolvable (no envs.py)"
  exit 1
fi
echo "[v6-remote-dspark] VLLM_ROOT=$VLLM_ROOT"

python3 "$SCRIPT_DIR/patch_remote_dspark.py" "$VLLM_ROOT"
rc=$?

echo "[v6-remote-dspark] dry-run checklist:"
echo "  1. No 'PREREQUISITE FAILED' above."
echo "  2. APPLY lines for the 3 new vllm files + 4 modified files, or SKIP"
echo "     (already present). NOTE lines = hunk/file skipped — investigate."
echo "  3. Verifier side: boot with VLLM_K3_DRAFT_REMOTE_ADDRESS=tcp://<host>:8092"
echo "     (dspark also accepts VLLM_K3_DSPARK_REMOTE_ADDRESS); expect the"
echo "     'Remote K3 ... proxy initialized' log line. Unset = stock behavior."
echo "  4. Draft side: python -m vllm.entrypoints.k3_dspark_standalone --help"
echo "     must list the CLI (draft-model/target-weights/target-config...)."
echo "  5. Draft server /healthz returns ready=true; verifier PING/PONG at boot."
exit $rc
