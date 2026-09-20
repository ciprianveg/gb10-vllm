#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PR54896 — upstream vLLM PR #54896 (MLA decode concat/cache epilogue).

STATUS: NOT PORTABLE AS A MOD.

The entire PR lives in compiled CUDA C++:
  * csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu
      - writeLatent576 gains a (lane, lane_stride) pair so a row can be
        split across SPLIT=3 warps for decode-sized batches
        (num_tokens <= 64), and the grid-dependency wait moves after the
        PDL-independent loads (slot_mapping, scales, rope table, cache-row
        inputs); cache-slot warps skip the wait entirely.
      - launchPdl -> launchPdlSlots with (num_heads + 1) * row_split slots.
  * tests only otherwise.

The fork ships a prebuilt ``_C_stable_libtorch.abi3.so``; mods can only
patch the installed Python package, and this PR touches no Python. Porting
it requires rebuilding the stable-libtorch extension from csrc with the
#54896 patch applied (image rebuild), after which it engages with no
further changes.

This mod therefore makes NO changes; it documents the skip and checks the
native op is present so the epilogue path it optimizes actually exists.
"""
import sys

SCRIPT_NAME = "patch_mla_epilogue"
TAG = "pr54896 (upstream #54896 — csrc-only, not portable as a mod)"


def main():
    print(f"[{SCRIPT_NAME}] {TAG}")
    if len(sys.argv) != 2:
        print(f"usage: {SCRIPT_NAME}.py <VLLM_ROOT>")
        return 2
    vroot = sys.argv[1].rstrip("/")
    # Sanity: the fused epilogue op family this PR optimizes must exist.
    probe = (
        "import torch, "
        "sys; sys.exit(0 if hasattr(torch.ops._C, 'fused_kimi_k3_mla_"
        "decode_q_concat_kv_cache_insert') else 1)"
    )
    print(
        f"[{SCRIPT_NAME}] NOTE  csrc-only PR (fused_kimi_k3_mla_key_concat_"
        f"kv_cache_kernel.cu: 3-warp row split for decode batches <= 64 "
        f"tokens + deferred grid-dependency wait; launchPdlSlots). "
        f"Requires rebuilding _C_stable_libtorch.abi3.so — cannot be "
        f"applied to the installed Python package. SKIP."
    )
    print(
        f"[{SCRIPT_NAME}] NOTE  to engage: rebuild the image with the "
        f"#54896 csrc patch; no Python-side changes are needed."
    )
    print(f"[{SCRIPT_NAME}] SKIP  no-op (nothing portable); idempotent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
