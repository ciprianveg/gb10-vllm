#!/bin/bash
# fix-pp2-sm121 — PP2 + DSpark spec-decode for Kimi-K3 on vllm-node-kimi3-sm121
set -e
VLLM="/opt/kimi-k3/vllm/vllm"
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)/patches"
echo "--- fix-pp2-sm121: applying PP2 + DSpark patches to $VLLM"

# 1. Full-file replacements
cp "$PATCH_DIR/pp_utils.py" "$VLLM/v1/worker/gpu/pp_utils.py"
cp "$PATCH_DIR/dspark_utils.py" "$VLLM/v1/worker/gpu/spec_decode/dspark/utils.py"
cp "$PATCH_DIR/eagle3_utils.py" "$VLLM/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py"

# 2. Additive python patches
python3 "$PATCH_DIR/interfaces_patch.py"
python3 "$PATCH_DIR/model_patch.py"
python3 "$PATCH_DIR/weight_utils_patch.py"
python3 "$PATCH_DIR/model_runner_patch.py"
python3 "$PATCH_DIR/embed_sharing_patch.py"
python3 "$PATCH_DIR/drafter_guard_patch.py"

# 3. speculative.py — draft PP=1
python3 - <<'PYEOF'
from pathlib import Path
p = Path("/opt/kimi-k3/vllm/vllm/config/speculative.py")
src = p.read_text()
old = "pipeline_parallel_size=target_parallel_config.pipeline_parallel_size,"
new = "pipeline_parallel_size=1,  # drafter runs on last PP stage only"
if new in src:
    print("[run.sh] speculative.py draft PP=1 already applied")
elif old in src:
    p.write_text(src.replace(old, new, 1))
    print("[run.sh] speculative.py draft PP=1 applied")
else:
    print("[run.sh] WARNING: speculative.py draft PP anchor not found")
PYEOF

# 4. config/vllm.py — remove EAGLE3+PP unsupported block
python3 - <<'PYEOF'
from pathlib import Path
p = Path("/opt/kimi-k3/vllm/vllm/config/vllm.py")
src = p.read_text()
block = '''
            if (
                speculative_config.method == "eagle3"
                and self.parallel_config.pipeline_parallel_size > 1
            ):
                unsupported.append("EAGLE3 with pipeline parallelism")
'''
if "EAGLE3 with pipeline parallelism" in src:
    p.write_text(src.replace(block, "", 1))
    print("[run.sh] removed EAGLE3+PP block from vllm.py")
else:
    print("[run.sh] EAGLE3+PP block already absent from vllm.py")
PYEOF

# 5. get_num_layers → get_total_num_hidden_layers
python3 - <<'PYEOF'
from pathlib import Path
VLLM = Path("/opt/kimi-k3/vllm/vllm")
files = [
    "models/kimi_k3/nvidia/dspark_mla.py",
    "model_executor/models/deepseek_eagle3.py",
    "model_executor/models/llama_eagle3.py",
    "model_executor/models/qwen3_eagle3.py",
    "model_executor/models/qwen3_dflash.py",
    "model_executor/models/qwen3_dspark.py",
    "model_executor/models/laguna_dflash.py",
]
for rel in files:
    p = VLLM / rel
    if not p.exists(): continue
    src = p.read_text()
    if "get_total_num_hidden_layers" in src: continue
    for old in (
        "vllm_config.model_config.get_num_layers(\n            vllm_config.parallel_config\n        )",
        "vllm_config.model_config.get_num_layers(\n                vllm_config.parallel_config\n            )",
    ):
        if old in src:
            p.write_text(src.replace(old, "vllm_config.model_config.get_total_num_hidden_layers()", 1))
            print(f"[run.sh] {rel}: patched")
            break
PYEOF

# 6. Verify syntax
echo "--- Verifying syntax..."
for f in v1/worker/gpu/pp_utils.py v1/worker/gpu/spec_decode/dspark/utils.py v1/worker/gpu/spec_decode/eagle/eagle3_utils.py model_executor/models/interfaces.py models/kimi_k3/nvidia/model.py models/kimi_k3/nvidia/dspark_mla.py v1/worker/gpu/model_runner.py v1/worker/gpu_model_runner.py v1/spec_decode/llm_base_proposer.py model_executor/model_loader/weight_utils.py config/speculative.py config/vllm.py; do
    [ -f "$VLLM/$f" ] || continue
    python3 -c "import py_compile; py_compile.compile('$VLLM/$f', doraise=True)" || { echo "FAIL: $f"; exit 1; }
done
find "$VLLM" -name '*.pyc' -delete 2>/dev/null || true
find "$VLLM" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
echo "=== fix-pp2-sm121 complete ==="
