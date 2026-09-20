#!/usr/bin/env bash
# pr53942 — backport of upstream vLLM PR #53942.
# MTP eh_proj: nn.Linear -> ReplicatedLinear + measured (7168, 14336)
# low-latency GEMM spec so the draft block's eh_proj uses the tuned
# static-K skinny kernel at M=1/2.
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
  echo "[pr53942] PREREQUISITE FAILED: VLLM_ROOT not resolvable (no envs.py)"
  exit 1
fi
echo "[pr53942] VLLM_ROOT=$VLLM_ROOT"

python3 "$SCRIPT_DIR/patch_eh_proj.py" "$VLLM_ROOT"
rc=$?

echo "[pr53942] dry-run checklist:"
echo "  1. No 'PREREQUISITE FAILED' above."
echo "  2. APPLY lines for mtp.py and low_latency_gemm.py, or SKIP."
echo "  3. Serving (dspark/MTP draft active): eh_proj GEMM compiles the"
echo "     static-K (14336) skinny config at warmup; draft step latency"
echo "     should drop slightly at M=1."
exit $rc
