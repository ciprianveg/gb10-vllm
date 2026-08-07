#!/bin/bash
# fix-k3-dspark-perf-unified — Cherry-pick DSpark perf optimizations from
# codex/hh-kimi-k3-unified-20260805 (local-inference-lab/vllm fork).
#
# Five pure-Python commits targeting the decode / draft-verify loop:
#
#   49c1d79eed6e  Fused router for DSpark verification (M<=8 tokens)
#   03f9c5719a78  Fused DSpark TP16 greedy sampling (B12X vocab argmax)
#   aacd3aab5500  Composed reduction (opt-in fused AR + RMSNorm)
#   142d21a06326  Capture sharded Markov tail (opt-in CUDAGraph capture)
#   27399cda9b52  Shared-expert prelaunch experiment
#
# All optimizations are behind env vars that default to OFF, so this mod is
# a behavioral no-op unless the operator sets the flags. The env vars are:
#   VLLM_KIMI_K3_B12X_DSPARK_ARGMAX=1            (commit 2)
#   VLLM_DSPARK_PREFER_B12X_ALLREDUCE_RMS=1      (commit 3)
#   VLLM_DSPARK_CAPTURE_SHARDED_MARKOV=1         (commit 4; also needs
#       VLLM_DSPARK_SHARD_MARKOV_HEAD=1 and a DSparkMarkovHead that exposes
#       `replicate_w1` — see KNOWN LIMITATIONS below)
#   VLLM_KIMI_PRELAUNCH_SHARED_EXPERTS=1         (commit 5)
#   VLLM_KIMI_USE_B12X_BATCHED_PROJECTION_TOPK=1 (commit 1 warmup path)
# Commit 1's main code path (M<=8 fused router) is active whenever
#   VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER=1 is set (existing flag).
#
# This mod runs LAST in the mod list, after:
#   fix-dspark-fused-kv          (replaces dspring_mla.py)
#   fix-dspark-pp2-aux-forward   (patches model.py, model_runner.py, pp_utils,
#                                 utils.py, eagle3_utils.py, interfaces.py)
#   fix-k3-ep-intermediate-padding (patches model.py KimiMoE.__init__)
#   fix-k3-flash-kda-prefill     (patches kda.py)
#
# STRATEGY
#   The unified branch diverged from our base (dspark-dcp16). For files NOT
#   touched by prior mods we install the unified-branch file verbatim (it is
#   exactly base + the relevant commit(s), verified by diff). For files the
#   prior mods already patched (model.py) we apply idempotent string
#   replacements. envs.py gets additive env-var declarations.
#
# KNOWN LIMITATIONS
#   - VLLM_DSPARK_CAPTURE_SHARDED_MARKOV=1 requires DSparkMarkovHead.replicate_w1
#     which the base qwen3_dspark.py does not expose. That opt-in will raise
#     AttributeError unless qwen3_dspark.py is also patched; it defaults off.
#   - dspring_mla.py is REPLACED by this mod's unified version, which uses the
#     per-layer attn.fused_qkv_a_proj fused-KV path (compatible with the
#     running mla.py). This supersedes fix-dspark-fused-kv's standalone
#     context_kv_proj variant.
set -e

echo "--- Applying DSpark perf-unified (5 commits from codex/hh-kimi-k3-unified-20260805)..."

VLLM="/usr/local/lib/python3.12/dist-packages/vllm"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/patches"

# ─── 1. Full file replacements (files not touched by prior mods) ─────────
# Each pair: <patch-file> <container-relative-target>
replace() {
    local src="$PATCH_DIR/$1"
    local dst="$VLLM/$2"
    if [ ! -f "$src" ]; then
        echo "ERROR: replacement source missing: $src" >&2
        exit 1
    fi
    cp "$src" "$dst"
    python3 -c "import py_compile; py_compile.compile('$dst', doraise=True)" \
        || { echo "ERROR: syntax check failed for $dst" >&2; exit 1; }
    echo "[replace] $2"
}

# Commit 1: tp_projection.py (M<=8 max_batch_size), dcp_alltoall.py (M<=8 path)
replace "tp_projection.py"        "models/kimi_k3/nvidia/tp_projection.py"
replace "dcp_alltoall.py"         "v1/attention/ops/dcp_alltoall.py"

# Commits 2 + 3: dspring_mla.py (B12X argmax + prefer_b12x). Supersedes the
# fix-dspark-fused-kv replacement with the unified fused-KV + argmax version.
replace "dspark_mla.py"           "models/kimi_k3/nvidia/dspark_mla.py"

# Re-apply the get_total_num_hidden_layers() fix from fix-dspark-pp2-aux-forward.
# That mod patches dspring_mla.py to compute start_layer_id from the TOTAL target
# layer count (across all PP stages) rather than the local rank's count. Our
# replacement above restores the upstream get_num_layers() call, so re-apply the
# fix here so PP2 deployments keep working. Idempotent.
python3 - <<'PYEOF'
from pathlib import Path
p = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/nvidia/dspark_mla.py")
src = p.read_text()
if "get_total_num_hidden_layers" in src:
    print("[run.sh] dspring_mla get_total_num_hidden_layers: already patched")
else:
    old = """        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )"""
    if old in src:
        src = src.replace(old, "        target_layer_num = vllm_config.model_config.get_total_num_hidden_layers()", 1)
        p.write_text(src)
        print("[run.sh] dspring_mla get_total_num_hidden_layers: patched")
    else:
        print("[run.sh] WARNING: dspring_mla get_num_layers pattern not found")
PYEOF
python3 -c "import py_compile; py_compile.compile('$VLLM/models/kimi_k3/nvidia/dspark_mla.py', doraise=True)" \
    || { echo "ERROR: syntax check failed for dspring_mla.py after get_total_num_hidden_layers patch" >&2; exit 1; }

# Commits 2 + 4: speculator.py (B12X sampler reorder + sharded-Markov capture)
replace "speculator.py"           "v1/worker/gpu/spec_decode/dspark/speculator.py"

# Commit 3: custom_all_reduce.py (composed AR), fused_allreduce_rms_norm.py (prefer_b12x)
replace "custom_all_reduce.py"    "distributed/device_communicators/custom_all_reduce.py"
replace "fused_allreduce_rms_norm.py" "models/common/ops/fused_allreduce_rms_norm.py"

# Commit 5: moe_runner.py (prelaunch_shared_experts), shared_experts.py (prelaunch)
replace "moe_runner.py"           "model_executor/layers/fused_moe/runner/moe_runner.py"
replace "shared_experts.py"       "model_executor/layers/fused_moe/runner/shared_experts.py"

# ─── 2. Additive env-var declarations ────────────────────────────────────
python3 "$PATCH_DIR/patch_envs.py"
python3 -c "import py_compile; py_compile.compile('$VLLM/envs.py', doraise=True)" \
    || { echo "ERROR: syntax check failed for envs.py" >&2; exit 1; }

# ─── 3. Idempotent model.py patch (commits 1 + 5) ────────────────────────
python3 "$PATCH_DIR/patch_model.py"
python3 -c "import py_compile; py_compile.compile('$VLLM/models/kimi_k3/nvidia/model.py', doraise=True)" \
    || { echo "ERROR: syntax check failed for model.py" >&2; exit 1; }

# ─── 4. Verify the new env vars resolve ──────────────────────────────────
python3 - <<'PYEOF'
import vllm.envs as envs
for v in [
    "VLLM_KIMI_K3_B12X_DSPARK_ARGMAX",
    "VLLM_DSPARK_PREFER_B12X_ALLREDUCE_RMS",
    "VLLM_DSPARK_CAPTURE_SHARDED_MARKOV",
    "VLLM_KIMI_PRELAUNCH_SHARED_EXPERTS",
    "VLLM_KIMI_USE_B12X_BATCHED_PROJECTION_TOPK",
    "VLLM_DSPARK_REPLICATE_MARKOV_W1",
]:
    getattr(envs, v)  # raises AttributeError if missing
    print(f"[envs] {v} OK")
PYEOF

echo "=== fix-k3-dspark-perf-unified mod complete ==="
