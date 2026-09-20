#!/usr/bin/env bash
# v4plus-batch1 — speed improvements as mods on the STABLE v4-prd image
# (ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v4-prd: sm121 native base + fork
# vLLM @881ac39a4 + b12x @#124/#138/#139 lineage).
#
# Serving config: TP8 (no DCP), fp8 KV, dspark nst=6, B12X_MLA decode,
# marlin MoE, CUDA graphs. Baseline to beat: 12.2 tok/s.
# Batch 1 is SPEED-ONLY: no nst change, no recipe behavior change.
#
# Items (in execution order):
#  M1  patch_b12x_roce_kernel.py   — b12x PR #295 (RoCEnante kernel side):
#      adds b12x/comm/roce/ (8 new files, verbatim) + minimal registration.
#      One-shot RDMA all-reduce/all-gather for multi-node DGX Spark TP over
#      ConnectX-7; runtime-JIT CuTe DSL + host-cc C proxy — no .so rebuild.
#  M2  patch_vllm_rocenante.py     — fork #597 adapter, adapted to v4 (no
#      b12x AR slot in v4's cuda_communicator — hand-wired): shim payload
#      (verbatim, one import adaptation), envs (VLLM_ENABLE_ROCE_ALLREDUCE
#      default baked from a b12x.comm.roce probe; OFF + loud note when
#      absent/broken), cuda_communicator construction/dispatch/logging,
#      gpu_worker fail-stop health check.
#  M3  patch_scheduler_budgets.py  — fork #605 scheduler input budgets:
#      DOCUMENTED SKIP — v4's scheduler predates upstream #52996's
#      draft_slots machinery that #605 builds on (v4 uses fork-native
#      num_tokens_with_spec accounting; the fixed problem cannot occur).
#      The script is a presence check that flags if this ever goes stale.
#  M4  patch_568_riders.py         — fork #568 subset:
#      (a) ee429fb01 bf16 kv_b_proj cast BUGFIX — presence check (already
#          in v4's impl-side _compute_prefill_context);
#      (b) 8a89a1d2d retained MLA context-projection workspace (−144 MiB
#          hot allocs/chunk), adapted to v4's impl-side chunk loop.
#  M5  patch_b12x_f5394625c.py     — b12x f5394625c "gate speculative
#      chunk copies + split-merge rewrite" (63 verbatim hunks over the 5
#      _shared/mla files, per-file ATOMIC; byte-validated against git apply
#      on both v4-era b12x checkouts). Tests + 296cb8647 skipped.
#
# Sources: b12x PR #295; fork vLLM PRs #597, #605, #568 (ee429fb01,
# 8a89a1d2d); b12x commit f5394625c. Date: 2026-09-03.
#
# Paths on v4-prd (DIFFERENT from the v5 images):
#   * vLLM tree: /opt/kimi-k3/vllm/vllm  (NOT dist-packages)
#   * b12x:      resolved at runtime via `import b12x` (likely
#                /opt/kimi-k3/b12x/b12x) and exported as B12X_ROOT.
#
# Policy: anchor-based hunks (count==1) with marker idempotency; a missing
# anchor is NOT a failure — the hunk skips with a printed NOTE so the dry-run
# review sees exactly what applied. Scripts exit non-zero ONLY on missing
# files or py_compile failure. Review APPLIED/NOT-APPLIED notes before baking.
set -euo pipefail

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VLLM_ROOT="${VLLM_ROOT:-/opt/kimi-k3/vllm/vllm}"
if [ ! -f "$VLLM_ROOT/envs.py" ]; then
    echo "[v4plus-batch1] ERROR: vLLM tree not found at $VLLM_ROOT (set VLLM_ROOT)" >&2
    exit 1
fi
export VLLM_ROOT

B12X_ROOT="${B12X_ROOT:-$(python3 -c "import b12x,os;print(os.path.dirname(b12x.__file__))")}"
if [ ! -f "$B12X_ROOT/__init__.py" ]; then
    echo "[v4plus-batch1] ERROR: b12x package not resolvable at $B12X_ROOT (set B12X_ROOT)" >&2
    exit 1
fi
export B12X_ROOT
echo "[v4plus-batch1] vLLM tree: $VLLM_ROOT"
echo "[v4plus-batch1] b12x package: $B12X_ROOT"

echo "[v4plus-batch1] applying Batch 1 (M1-M5)"

python3 "$MOD_DIR/patch_b12x_roce_kernel.py"    && echo "[v4plus-batch1] M1/5 b12x comm/roce (RoCEnante kernel side): OK"
python3 "$MOD_DIR/patch_vllm_rocenante.py"      && echo "[v4plus-batch1] M2/5 vLLM RoCEnante adapter (#597 adapted): OK"
python3 "$MOD_DIR/patch_scheduler_budgets.py"   && echo "[v4plus-batch1] M3/5 #605 scheduler budgets (applicability check): OK"
python3 "$MOD_DIR/patch_568_riders.py"          && echo "[v4plus-batch1] M4/5 #568 riders (bf16 cast check + workspace): OK"
python3 "$MOD_DIR/patch_b12x_f5394625c.py"      && echo "[v4plus-batch1] M5/5 b12x f5394625c split-merge rewrite: OK"

echo "[v4plus-batch1] done — review APPLIED/NOT-APPLIED notes above before baking"
