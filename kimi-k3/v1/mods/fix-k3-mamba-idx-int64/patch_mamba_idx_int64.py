#!/usr/bin/env python3
"""Patch vllm/v1/worker/gpu/model_states/mamba_hybrid.py to cast idx_mapping to
int64 before index_fill_.

Bug
---
In the Mamba-hybrid model state's `postprocess_state`, the non-spec scalar
fill branch calls:

    self.num_accepted_tokens_gpu.index_fill_(
        0, idx_mapping, max(num_sampled, 1)
    )

`torch.Tensor.index_fill_()` strictly requires the index tensor to be
`int64`. This fork allocates `input_batch.idx_mapping` as `int32` (the
fork's pattern, also seen in `make_dummy`), so the call raises:

    IndexError: index_fill_(): Expected dtype int64 for index.

The crash fires on any step where `num_sampled` is a Python int (chunked
prefill continuation, no spec sample on the frame) — TP16 or PP2; draft
count is irrelevant. Cast `idx_mapping.to(torch.int64)` once per call.

Fix
---
A one-line cast in `mamba_hybrid.py`. Marker `# fix-k3-mamba-idx-int64`
guards the patch for idempotent re-application.
"""

import re
import sys
from pathlib import Path

TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
)


def main() -> None:
    src = TARGET.read_text()

    marker = "# fix-k3-mamba-idx-int64"
    if marker in src:
        print("[fix-k3-mamba-idx-int64] already applied (marker found); skipping")
        return

    old = """        else:
            # Fill with single value.
            self.num_accepted_tokens_gpu.index_fill_(
                0, idx_mapping, max(num_sampled, 1)
            )
"""
    new = """        else:
            # Fill with single value.
            self.num_accepted_tokens_gpu.index_fill_(
                0, idx_mapping.to(torch.int64), max(num_sampled, 1)  # fix-k3-mamba-idx-int64
            )
"""

    if old not in src:
        print("[fix-k3-mamba-idx-int64] ERROR: anchor block not found in "
              f"{TARGET}", file=sys.stderr)
        sys.exit(1)

    src = src.replace(old, new, 1)
    TARGET.write_text(src)
    print("[fix-k3-mamba-idx-int64] patched (idx_mapping.to(torch.int64) before index_fill_)")


if __name__ == "__main__":
    main()
