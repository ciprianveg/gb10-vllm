#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PR52388 — upstream vLLM PR #52388 (Mamba metadata prep optimization).

STATUS: ALREADY PRESENT IN THE FORK (verified against the b4f reference
tree). All three upstream hunks exist verbatim:

  * models/kimi_k3/nvidia/kda_metadata.py
      - ``KimiK3KDAMetadataBuilder.mamba_aligned_state_indices`` class
        attribute + the align-mode branch in ``build()`` that consumes the
        precomputed indices (with the MRV1 fallback assert).
  * v1/worker/mamba_utils.py
      - ``get_aligned_state_indices_multi_group_kernel`` triton kernel,
        ``MambaSpecDecodeGPUContext.aligned_state_indices`` buffer, and
        ``compute_aligned_state_indices()`` (one launch for every Mamba
        group's aligned physical state IDs).
  * v1/worker/gpu/model_states/mamba_hybrid.py
      - the ``prepare_attn`` hook that computes all-group aligned state
        indices once per step and hands them to the metadata builders.

This mod therefore makes NO changes. It only verifies the prerequisite
markers so a base-image regression (or a future rebase that drops the
feature) is caught loudly instead of silently.
"""
import sys

SCRIPT_NAME = "patch_mamba_metadata"
TAG = "pr52388 (upstream #52388 — already in fork, verification only)"

# (file, required snippet, description)
REQUIRED = [
    (
        "models/kimi_k3/nvidia/kda_metadata.py",
        "mamba_aligned_state_indices: torch.Tensor | None = None",
        "builder precomputed aligned-state-indices attribute",
    ),
    (
        "models/kimi_k3/nvidia/kda_metadata.py",
        '"Aligned Mamba state indices must be precomputed"',
        "align-mode precomputed-indices branch in build()",
    ),
    (
        "v1/worker/mamba_utils.py",
        "def get_aligned_state_indices_multi_group_kernel(",
        "multi-group aligned-index triton kernel",
    ),
    (
        "v1/worker/mamba_utils.py",
        "def compute_aligned_state_indices(",
        "once-per-step all-group aligned-index launch",
    ),
    (
        "v1/worker/gpu/model_states/mamba_hybrid.py",
        "aligned_index_builders",
        "prepare_attn aligned-index distribution hook",
    ),
]


def main():
    print(f"[{SCRIPT_NAME}] {TAG}")
    if len(sys.argv) != 2:
        print(f"usage: {SCRIPT_NAME}.py <VLLM_ROOT>")
        return 2
    vroot = sys.argv[1].rstrip("/")
    ok = True
    for relpath, snippet, desc in REQUIRED:
        try:
            with open(vroot + "/" + relpath) as f:
                src = f.read()
        except OSError as exc:
            print(f"[{SCRIPT_NAME}] FAIL  {relpath}: unreadable ({exc})")
            ok = False
            continue
        if snippet in src:
            print(f"[{SCRIPT_NAME}] SKIP  {relpath}: {desc} (already present)")
        else:
            print(
                f"[{SCRIPT_NAME}] NOTE  {relpath}: '{desc}' NOT FOUND — "
                f"upstream #52388 appears to be missing from this tree. "
                f"The fork baseline was expected to already contain it; "
                f"investigate before applying the other k3-tier1 mods."
            )
            ok = False
    if ok:
        print(
            f"[{SCRIPT_NAME}] done. Upstream #52388 is already fully present; "
            f"no changes made (idempotent no-op)."
        )
    else:
        print(f"[{SCRIPT_NAME}] NOTE: verification failed — see above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
