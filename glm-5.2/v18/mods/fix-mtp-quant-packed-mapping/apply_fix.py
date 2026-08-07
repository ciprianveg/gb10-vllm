#!/usr/bin/env python3
"""Add packed_modules_mapping for fused modules to the MTP draft quant config.

This is the runtime equivalent of patches/v16-final/03-draft-quant-packed-mapping.patch.
It adds packed_modules_mapping entries to DeepSeekMultiTokenPredictorLayer.__init__
so that fused_qkv_a_proj and gate_up_proj are recognized as packed modules
by the compressed-tensors quant config, allowing the MTP draft model's
quantized checkpoint weights to load correctly.

Idempotent: detects already-applied changes and exits cleanly.
"""
import sys
import os

# Auto-detect vllm install path (v18 uses /opt/venv, v16 uses /usr/local)
VLLM_CANDIDATES = [
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_mtp.py",
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/deepseek_mtp.py",
]
FILE_PATH = next((p for p in VLLM_CANDIDATES if os.path.isfile(p)), None)
if FILE_PATH is None:
    print("ERROR: Could not find deepseek_mtp.py in any known location")
    sys.exit(1)
print(f"  Using: {FILE_PATH}")

with open(FILE_PATH, "r") as f:
    content = f.read()

if "packed_modules_mapping.setdefault" in content and "fused_qkv_a_proj" in content:
    print("  fix-mtp-quant-packed-mapping: already applied")
    sys.exit(0)

# Find the quant_config assignment in DeepSeekMultiTokenPredictorLayer.__init__
# The pattern: quant_config = _maybe_disable_unserialized_modelopt_fp4_nextn(...)
OLD_BLOCK = """        quant_config = _maybe_disable_unserialized_modelopt_fp4_nextn(
            config, vllm_config, get_draft_quant_config(vllm_config)
        )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)"""

NEW_BLOCK = """        quant_config = _maybe_disable_unserialized_modelopt_fp4_nextn(
            config, vllm_config, get_draft_quant_config(vllm_config)
        )
        if quant_config is not None:
            # Draft quant configs are built fresh (get_draft_quant_config) and
            # never pass through configure_quant_config, so fused-module shard
            # expansion is empty; quant schemes whose targets name the
            # constituent projections (q_a_proj / kv_a_proj_with_mqa) then
            # fail to match the fused module and the block silently builds
            # unquantized, breaking checkpoints with quantized NextN layers.
            quant_config.packed_modules_mapping.setdefault(
                "fused_qkv_a_proj", ["q_a_proj", "kv_a_proj_with_mqa"]
            )
            quant_config.packed_modules_mapping.setdefault(
                "gate_up_proj", ["gate_proj", "up_proj"]
            )

        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)"""

if OLD_BLOCK not in content:
    print("ERROR: Could not find quant_config block in DeepSeekMultiTokenPredictorLayer.__init__")
    # Try to find it with relaxed matching
    if "quant_config = _maybe_disable_unserialized_modelopt_fp4_nextn" in content:
        print("  Found quant_config assignment but surrounding context doesn't match exactly")
    sys.exit(1)

content = content.replace(OLD_BLOCK, NEW_BLOCK, 1)

with open(FILE_PATH, "w") as f:
    f.write(content)

print("  fix-mtp-quant-packed-mapping: applied packed_modules_mapping for fused_qkv_a_proj + gate_up_proj")
