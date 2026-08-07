#!/bin/bash
# fix-k3-ep-intermediate-padding — Make Kimi K3 MoE intermediate padding EP-aware.
#
# Bug: KimiMoE.__init__ pads moe_intermediate_size from 3072 to 4096 when
# tp_size > 1 and the per-partition value (3072/16 = 192) is below the
# min_moe_intermediate_per_partition default of 256. The padding uses
# `self.tp_size = get_tensor_model_parallel_world_size()` (global TP), which
# is EP-unaware. Under --enable-expert-parallel (ep_size=16), FusedMoE gives
# each rank the FULL intermediate per local expert (56 experts/rank), so the
# padded 4096 is replicated per local expert and the EP memory savings are
# annihilated: 56 * 4096 = 229,376 inter-units/rank = same as 896 * 256 no-EP
# -> ~118.4 GiB of MoE weights per rank, exceeding the 121 GiB GB10 UMA pool
# and triggering NVRM NV_ERR_NO_MEMORY during weight load (driver wedge,
# node reboot).
#
# Fix: skip the min-256 padding when enable_expert_parallel is True, so
# padded_moe_intermediate_size stays at 3072. Under EP: 56 * 3072 = 172,032
# units -> ~89.1 GiB/rank -> fits with ~25-30 GiB headroom for KV + drafter.
# The downstream `if padded != moe_intermediate_size` zero-fill block does not
# execute, so FusedMoEConfig.__post_init__'s default
# intermediate_size_per_partition_unpadded (= intermediate_size_per_partition
# = 3072 under EP tp=1) is used -- correct.
#
# Applies to: vllm-node-kimi3 image (vLLM 0.26.1rc1.dev160+g2ac91211d).
# Effect: --enable-expert-parallel actually reduces per-rank MoE memory from
# 118.4 GiB to 89.1 GiB. No vLLM rebuild needed.
#
# Requires: --enable-expert-parallel in the vllm serve command (the recipe
# already passes it). If EP is off, this mod is a no-op (the guard keeps the
# original padding behavior).
set -e

MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find vLLM site-packages (handles both /usr/local and /opt/venv layouts)
VLLM_DIR="$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))" 2>/dev/null || true)"
if [ -z "$VLLM_DIR" ]; then
    echo "[fix-k3-ep-intermediate-padding] ERROR: cannot locate vllm package" >&2
    exit 1
fi

MODEL_FILE="$VLLM_DIR/models/kimi_k3/nvidia/model.py"
if [ ! -f "$MODEL_FILE" ]; then
    echo "[fix-k3-ep-intermediate-padding] ERROR: $MODEL_FILE not found (K3 model not in this image?)" >&2
    exit 1
fi

echo "[fix-k3-ep-intermediate-padding] Patching $MODEL_FILE"
python3 "$MOD_DIR/patch_model.py" "$MODEL_FILE"

# Verify the patch compiled cleanly
python3 -c "
import ast
with open('$MODEL_FILE') as f:
    ast.parse(f.read())
print('[fix-k3-ep-intermediate-padding] Syntax OK')
" || { echo "[fix-k3-ep-intermediate-padding] ERROR: patched file has syntax errors" >&2; exit 1; }

# Verify the guard is in place
if ! grep -q 'fix-k3-ep-intermediate-padding' "$MODEL_FILE"; then
    echo "[fix-k3-ep-intermediate-padding] ERROR: marker not found after patch" >&2
    exit 1
fi

echo "[fix-k3-ep-intermediate-padding] Mod applied successfully."
