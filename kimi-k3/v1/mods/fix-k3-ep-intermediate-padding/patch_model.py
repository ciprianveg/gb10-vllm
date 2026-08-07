#!/usr/bin/env python3
"""Patch kimi_k3/nvidia/model.py: make MoE intermediate padding EP-aware.

Works with both stock vLLM and the local-inference-lab fork (which has
additional use_native_b12x_intermediate checks).
"""
import sys

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/usr/local/lib/python3.12/dist-packages/vllm/models/kimi_k3/nvidia/model.py"

MARKER = "# fix-k3-ep-intermediate-padding:"

with open(MODEL_PATH, "r") as f:
    src = f.read()

if MARKER in src:
    print("[fix-k3-ep-intermediate-padding] Already patched, skipping")
    sys.exit(0)

# Fork pattern: has "and not use_native_b12x_intermediate" in the condition
OLD_FORK = """        if self.tp_size > 1 and not use_native_b12x_intermediate:
            moe_intermediate_per_partition = moe_intermediate_size // self.tp_size
            if moe_intermediate_per_partition < min_moe_intermediate_per_partition:
                self.padded_moe_intermediate_size = (
                    min_moe_intermediate_per_partition * self.tp_size
                )"""

NEW_FORK = """        # fix-k3-ep-intermediate-padding: skip padding under EP
        if (
            self.tp_size > 1
            and not use_native_b12x_intermediate
            and not vllm_config.parallel_config.enable_expert_parallel
        ):
            moe_intermediate_per_partition = moe_intermediate_size // self.tp_size
            if moe_intermediate_per_partition < min_moe_intermediate_per_partition:
                self.padded_moe_intermediate_size = (
                    min_moe_intermediate_per_partition * self.tp_size
                )"""

# Stock pattern: simple "if self.tp_size > 1:"
OLD_STOCK = """        if self.tp_size > 1:
            moe_intermediate_per_partition = moe_intermediate_size // self.tp_size
            if moe_intermediate_per_partition < min_moe_intermediate_per_partition:
                self.padded_moe_intermediate_size = (
                    min_moe_intermediate_per_partition * self.tp_size
                )"""

NEW_STOCK = """        # fix-k3-ep-intermediate-padding: skip padding under EP
        if (
            self.tp_size > 1
            and not vllm_config.parallel_config.enable_expert_parallel
        ):
            moe_intermediate_per_partition = moe_intermediate_size // self.tp_size
            if moe_intermediate_per_partition < min_moe_intermediate_per_partition:
                self.padded_moe_intermediate_size = (
                    min_moe_intermediate_per_partition * self.tp_size
                )"""

if OLD_FORK in src:
    src = src.replace(OLD_FORK, NEW_FORK, 1)
    print("[fix-k3-ep-intermediate-padding] Patched (fork pattern)")
elif OLD_STOCK in src:
    src = src.replace(OLD_STOCK, NEW_STOCK, 1)
    print("[fix-k3-ep-intermediate-padding] Patched (stock pattern)")
else:
    print(
        "[fix-k3-ep-intermediate-padding] ERROR: padding block anchor not found "
        "in {}".format(MODEL_PATH),
        file=sys.stderr,
    )
    sys.exit(1)

with open(MODEL_PATH, "w") as f:
    f.write(src)
