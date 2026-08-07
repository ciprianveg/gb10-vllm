#!/bin/bash
set -e
echo "--- Applying DSpark PP2 aux-hidden-state forwarding (PR #50514)..."

VLLM="/usr/local/lib/python3.12/dist-packages/vllm"
PATCH_DIR="$(dirname "$0")/patches"

# ─── 1. Full replacements ────────────────────────────────────────────────

# pp_utils.py — PendingRecv.draft_tokens, broadcast_drafts/receive_drafts,
# padded broadcast, num_speculative_steps.
cp "$PATCH_DIR/pp_utils.py" "$VLLM/v1/worker/gpu/pp_utils.py"

# dspark/utils.py — remove PP guard, add PPMissingLayer embed loading.
cp "$PATCH_DIR/dspark_utils.py" "$VLLM/v1/worker/gpu/spec_decode/dspark/utils.py"

# eagle3_utils.py — supports_aux_hidden_states_over_pp + reserve slots.
cp "$PATCH_DIR/eagle3_utils.py" "$VLLM/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py"

# ─── 2. Additive Python patches ──────────────────────────────────────────

# interfaces.py — EagleModelMixin class fields + 6 new methods.
python3 "$PATCH_DIR/interfaces_patch.py"

# kimi_k3/nvidia/model.py — KimiLinearModel aux-over-PP forwarding.
python3 "$PATCH_DIR/model_patch.py"

# deepseek_v4/nvidia/model.py — DeepseekV4Model aux-over-PP forwarding.
python3 "$PATCH_DIR/deepseek_v4_model_patch.py"

# weight_utils.py — use TP group instead of WORLD for weight loading (Issue #50959).
python3 "$PATCH_DIR/weight_utils_patch.py"

# v1/worker/gpu/model_runner.py — import, guard move, draft broadcast/recv.
python3 "$PATCH_DIR/model_runner_patch.py"

# ─── 3. sed: config files ────────────────────────────────────────────────

# speculative.py: draft parallel config always PP=1 (drafter on last stage).
sed -i 's|pipeline_parallel_size=target_parallel_config.pipeline_parallel_size,|pipeline_parallel_size=1,  # drafter runs on last PP stage only|' \
    "$VLLM/config/speculative.py"

# vllm.py: remove the "EAGLE3 with pipeline parallelism" unsupported block.
python3 - <<'PYEOF'
from pathlib import Path
p = Path("/usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py")
src = p.read_text()
block = '''
            if (
                speculative_config.method == "eagle3"
                and self.parallel_config.pipeline_parallel_size > 1
            ):
                unsupported.append("EAGLE3 with pipeline parallelism")
'''
if "EAGLE3 with pipeline parallelism" in src:
    src = src.replace(block, "", 1)
    p.write_text(src)
    print("[run.sh] removed EAGLE3+PP unsupported block from vllm.py")
else:
    print("[run.sh] EAGLE3+PP block already absent from vllm.py")
PYEOF

# ─── 4. sed: dspring_mla.py — get_total_num_hidden_layers ───────────────

sed -i 's|vllm_config.model_config.get_num_layers(\n            vllm_config.parallel_config\n        )|vllm_config.model_config.get_total_num_hidden_layers()|' \
    "$VLLM/models/kimi_k3/nvidia/dspark_mla.py" 2>/dev/null || true

# sed doesn't handle multi-line well; use python for the 3-line replacement.
python3 - <<'PYEOF'
from pathlib import Path
p = Path("/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/nvidia/dspark_mla.py")
src = p.read_text()
old = """        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )"""
new = "        target_layer_num = vllm_config.model_config.get_total_num_hidden_layers()"
if "get_total_num_hidden_layers" in src:
    print("[run.sh] dspring_mla already patched")
elif old in src:
    src = src.replace(old, new, 1)
    p.write_text(src)
    print("[run.sh] dspring_mla patched")
else:
    print("[run.sh] WARNING: dspring_mla pattern not found")
PYEOF

# ─── 5. sed: draft model files — get_total_num_hidden_layers ─────────────

for f in \
    model_executor/models/deepseek_eagle3.py \
    model_executor/models/llama_eagle3.py \
    model_executor/models/qwen3_eagle3.py \
    model_executor/models/qwen3_dflash.py \
    model_executor/models/qwen3_dspark.py \
    model_executor/models/laguna_dflash.py
do
    python3 - "$f" <<'PYEOF'
import sys
from pathlib import Path
fname = sys.argv[1]
p = Path("/usr/local/lib/python3.12/dist-packages/vllm") / fname
src = p.read_text()
old = """vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )"""
new = "vllm_config.model_config.get_total_num_hidden_layers()"
if "get_total_num_hidden_layers" in src:
    print(f"[run.sh] {fname}: already patched")
elif old in src:
    src = src.replace(old, new, 1)
    p.write_text(src)
    print(f"[run.sh] {fname}: patched")
else:
    # Try alternate indentation
    old2 = """vllm_config.model_config.get_num_layers(
                vllm_config.parallel_config
            )"""
    if old2 in src:
        src = src.replace(old2, new, 1)
        p.write_text(src)
        print(f"[run.sh] {fname}: patched (alt indent)")
    else:
        print(f"[run.sh] WARNING: {fname}: pattern not found")
PYEOF
done

# ─── 6. Verify all modified files compile ─────────────────────────────────

echo "--- Verifying syntax..."
for f in \
    v1/worker/gpu/pp_utils.py \
    v1/worker/gpu/spec_decode/dspark/utils.py \
    v1/worker/gpu/spec_decode/eagle/eagle3_utils.py \
    model_executor/models/interfaces.py \
    models/kimi_k3/nvidia/model.py \
    models/deepseek_v4/nvidia/model.py \
    models/kimi_k3/nvidia/dspark_mla.py \
    v1/worker/gpu/model_runner.py \
    config/speculative.py \
    config/vllm.py \
    model_executor/models/deepseek_eagle3.py \
    model_executor/models/llama_eagle3.py \
    model_executor/models/qwen3_eagle3.py \
    model_executor/models/qwen3_dflash.py \
    model_executor/models/qwen3_dspark.py \
    model_executor/models/laguna_dflash.py
do
    python3 -c "import py_compile; py_compile.compile('$VLLM/$f', doraise=True)" \
        || { echo "FAIL: $f"; exit 1; }
done

echo "=== fix-dspark-pp2-aux-forward mod complete ==="
