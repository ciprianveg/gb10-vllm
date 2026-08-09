#!/usr/bin/env python3
"""Patch vllm/v1/worker/mamba_utils.py: fix MRv2 cross-block race on num_accepted.

Upstream PR: vllm-project/vllm#50432 (merged 2026-08-03, post Jul-30 image).
Bug
---
In MRv2 hybrid postprocess (`run_fused_postprocess_align`), the Triton kernel
`postprocess_mamba_fused_kernel` is launched with grid (num_reqs, num_layers *
num_state_types). For each req_idx, programs read `num_accepted_tokens_ptr`
early in the kernel, and `state_idx == 0` programs write 1 to it (in place)
when the accepted position stays in the running block.

Different `state_idx` programs for the same `req_idx` can land in different
waves on the GPU. Later-wave programs observe the (1) write and compute
accept_token_bias == 0 (early return, no state copy). Earlier-wave programs
already committed to a state copy with the original `num_accepted > 1`.
Result: Mamba state is inconsistent across layers for the same request.

The race is severity-dependent on GPU launch timing / number of SMs / number
of hybrid layers — K3-full (93 layers, TP=16) triggers it more than REAP-320
(pruned, fewer layers → fewer waves → race less likely). The inconsistent
Mamba state feeds wrong hidden states into the DSpark drafter's aux layers
[2,23,47,71,89], which is the most plausible cause of near-zero acceptance
(0.03-0.19 position-0) vs REAP's 0.33-0.81.

Fix
---
Apply the MRv1 pattern: the kernel reads from a snapshot buffer and writes
to a distinct output buffer. The caller copies the snapshot back into
`num_accepted_tokens_gpu` after the kernel finishes (same stream, strictly
ordered).

Two changes in vllm/v1/worker/mamba_utils.py:

1. Kernel (`postprocess_mamba_fused_kernel`, ~line 270): remove the
   `if HAS_IDX_MAPPING` branch that wrote in-place. Always write to
   `num_accepted_tokens_out_ptr`.

2. Caller (`run_fused_postprocess_align`, ~line 886): snapshot
   `num_accepted_tokens_gpu` into `self.num_accepted_tokens_out` before the
   kernel launch, then pass the snapshot as the read source and
   `num_accepted_tokens_gpu` as the write target.

Idempotent: both changes are guarded by a marker comment
`# fix-mamba-mrv2-race` so re-application is a no-op.
"""

import re
import sys
from pathlib import Path


MARKER = "# fix-mamba-mrv2-race"


def patch(text: str) -> tuple[str, bool]:
    """Apply the two patches. Returns (new_text, changed)."""
    changed = False

    # Patch 1: kernel — always write to num_accepted_tokens_out_ptr
    kernel_old = (
        "    if src_block_idx == dest_block_idx and state_idx == 0:\n"
        "        if HAS_IDX_MAPPING:\n"
        "            tl.store(num_accepted_tokens_ptr + req_idx, 1)\n"
        "        else:\n"
        "            tl.store(num_accepted_tokens_out_ptr + req_idx, 1)\n"
    )
    kernel_new = (
        f"    if src_block_idx == dest_block_idx and state_idx == 0: {MARKER}: always write to _out\n"
        "        tl.store(num_accepted_tokens_out_ptr + req_idx, 1)\n"
    )
    if kernel_old in text:
        text = text.replace(kernel_old, kernel_new, 1)
        changed = True

    # Patch 2: caller — snapshot num_accepted before kernel launch,
    # pass snapshot as read source, num_accepted_tokens_gpu as write target.
    fn_old = (
        "        if num_reqs == 0 or not self.is_initialized:\n"
        "            return\n"
        "        total_states = self.num_layers * self.num_state_types\n"
        "        grid = (num_reqs, total_states)\n"
        "        postprocess_mamba_fused_kernel[grid](\n"
        "            num_accepted_tokens_gpu,\n"
        "            state_idx_gpu,\n"
    )
    fn_new = (
        "        if num_reqs == 0 or not self.is_initialized:\n"
        "            return\n"
        f"        {MARKER}: snapshot num_accepted into self.num_accepted_tokens_out\n"
        "        # before kernel launch to avoid cross-program races (PR #50432).\n"
        "        num_accepted_tokens_snapshot = self.num_accepted_tokens_out\n"
        "        num_accepted_tokens_snapshot.copy_(num_accepted_tokens_gpu)\n"
        "        total_states = self.num_layers * self.num_state_types\n"
        "        grid = (num_reqs, total_states)\n"
        "        postprocess_mamba_fused_kernel[grid](\n"
        "            num_accepted_tokens_snapshot,\n"
        "            state_idx_gpu,\n"
    )
    if fn_old in text:
        text = text.replace(fn_old, fn_new, 1)
        changed = True

    # Patch 3: caller — pass num_accepted_tokens_gpu as num_accepted_out write target
    write_old = (
        "            self.state_dim_row_stride,\n"
        "            None,  # num_accepted_out: V2 updates num_accepted in place\n"
        "            idx_mapping,\n"
        "            num_reqs,\n"
        "            block_size=self.block_size,\n"
        "            COPY_BLOCK_SIZE=1024,\n"
        "            CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),\n"
        "            HAS_IDX_MAPPING=True,\n"
        "            PRECOMPUTED_NEW_COMPUTED=True,\n"
        "        )\n"
    )
    write_new = (
        "            self.state_dim_row_stride,\n"
        f"            num_accepted_tokens_gpu,  {MARKER}: write target (was None)\n"
        "            idx_mapping,\n"
        "            num_reqs,\n"
        "            block_size=self.block_size,\n"
        "            COPY_BLOCK_SIZE=1024,\n"
        "            CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),\n"
        "            HAS_IDX_MAPPING=True,\n"
        "            PRECOMPUTED_NEW_COMPUTED=True,\n"
        "        )\n"
    )
    if write_old in text:
        text = text.replace(write_old, write_new, 1)
        changed = True

    return text, changed


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path-to-mamba_utils.py>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    text = path.read_text()
    if MARKER in text:
        print(f"[fix-mamba-mrv2-race] Already patched (marker present), skipping.")
        return 0
    new_text, changed = patch(text)
    if not changed:
        print(
            f"[fix-mamba-mrv2-race] ERROR: no patches matched in {path}. "
            "File may have drifted from upstream main.",
            file=sys.stderr,
        )
        return 1
    path.write_text(new_text)
    print(f"[fix-mamba-mrv2-race] Patched {path} (3 hunks: kernel + caller snapshot + caller write target)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
