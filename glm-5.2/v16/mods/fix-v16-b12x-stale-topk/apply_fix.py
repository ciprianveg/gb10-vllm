#!/usr/bin/env python3
"""Fix stale topk_indices_buffer in B12xMLASparseImpl.

Stores the indexer reference in __init__ and reads topk_indices_buffer
live from it at forward time, instead of caching a potentially-stale
reference.

Idempotent: detects already-applied changes and exits cleanly.
"""
import sys

import os
VLLM_BASE = "/opt/venv/lib/python3.12/site-packages" if os.path.isdir("/opt/venv/lib/python3.12/site-packages/vllm") else "/usr/local/lib/python3.12/dist-packages"
FILE_PATH = VLLM_BASE + "/vllm/v1/attention/backends/mla/b12x_mla_sparse.py"

with open(FILE_PATH, "r") as f:
    content = f.read()

changed = False

# ── Fix 1: Store indexer reference in __init__ ──────────────────────────────

INIT_OLD = """        # The indexer carries the shared buffer for normal layers and tests;
        # the explicitly-passed buffer covers backbone skip layers, whose
        # indexer is not constructed (see deepseek_v2.py).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )"""

INIT_NEW = """        # The indexer carries the shared buffer for normal layers and tests;
        # the explicitly-passed buffer covers backbone skip layers, whose
        # indexer is not constructed (see deepseek_v2.py).
        self._indexer = indexer
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer if indexer is not None else topk_indices_buffer
        )"""

if "self._indexer = indexer" not in content:
    if INIT_OLD not in content:
        print("ERROR: Could not find __init__ topk_indices_buffer block")
        sys.exit(1)
    content = content.replace(INIT_OLD, INIT_NEW, 1)
    changed = True
    print("  Fix 1: stored indexer reference in __init__")
else:
    print("  Fix 1: already applied")

# ── Fix 2: Read topk_indices_buffer live from indexer at forward time ───────

FWD_OLD = """        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]"""

FWD_NEW = """        buf = (
            self._indexer.topk_indices_buffer
            if self._indexer is not None
            else self.topk_indices_buffer
        )
        assert buf is not None, "topk_indices_buffer required for sparse MLA"
        topk_indices = buf[:num_actual_toks]"""

if "self._indexer.topk_indices_buffer" not in content or "if self._indexer is not None" not in content:
    if FWD_OLD not in content:
        print("ERROR: Could not find forward topk_indices_buffer block")
        sys.exit(1)
    content = content.replace(FWD_OLD, FWD_NEW, 1)
    changed = True
    print("  Fix 2: read topk_indices_buffer live from indexer at forward time")
else:
    print("  Fix 2: already applied")

# ── Write if changed ────────────────────────────────────────────────────────

if changed:
    with open(FILE_PATH, "w") as f:
        f.write(content)
    print("  b12x_mla_sparse.py patched successfully")
else:
    print("  No changes needed")
