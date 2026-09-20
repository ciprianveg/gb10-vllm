#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V4PLUS Batch 2 / B4 — dynamic draft depth (f71e0deaa): DOCUMENTED SKIP.

Applicability check only — no changes are made.

f71e0deaa ("feat(spec-decode): adapt draft depth from acceptance", Aug 29,
1555 lines / 21 source files) reworks the acceptance-length machinery into
per-batch-size-band controllers, threads scheduler-selected depth (incl.
zero-depth scheduling) through the GPU model runners, and reworks CUDA
graph capture for the reachable depth shapes.

Why this is SKIPPED for v4-plus Batch 2 (verified against the clones):
  1. The v4 tree ALREADY CARRIES an acceptance-length controller:
     vllm/v1/spec_decode/dynamic/acceptance_length.py (with
     AcceptanceLengthController.observe_batch) wired through the scheduler
     (acceptance_length_controller at scheduler.py, feeding
     num_spec_tokens_to_schedule) and the adaptive window
     (num_speculative_tokens_per_batch_size at scheduler init).  The patch
     REPLACES this working machinery with its per-band rework.
  2. Every source anchor fails on the 881ac39a4 tree (git apply --check:
     speculative.py, scheduler.py, async_scheduler.py, output.py,
     metrics.py, cudagraph_utils.py, model_runner.py, both speculators all
     reject; the acceptance_length.py file "already exists").  The patch
     targets the post-Sep upstream-rebased lineage, not the fork-native
     v4 scheduler/proposer pair.
  3. Porting semantically = re-plumbing the scheduler depth selection, the
     runner's graph capture, and the proposer pair across 15+ drifted
     files, replacing a mechanism that already exists in an earlier form —
     a serving-behavior change by definition, which Batch 2's A/B
     (fused-verify + fp8 draft) must stay clean of.

If dynamic depth is wanted later, it needs its own batch with its own A/B,
porting the per-band controller onto v4's existing
AcceptanceLengthController wiring (not the other way around).

This script re-checks the signatures and reports; if the tree ever gains
the f71e0deaa surface (per-band controllers), it says so loudly.
"""

from __future__ import annotations

import os
import sys

SCRIPT_NAME = "patch_dynamic_depth"
TAG = "# V4PLUS-B2 (B4: f71e0deaa dynamic depth — documented skip)"

VLLM_ROOT = os.environ.get("VLLM_ROOT", "/opt/kimi-k3/vllm/vllm")
ACCEPTANCE = os.path.join(
    VLLM_ROOT, "v1", "spec_decode", "dynamic", "acceptance_length.py"
)
SCHEDULER = os.path.join(VLLM_ROOT, "v1", "core", "sched", "scheduler.py")

# Signatures of the f71e0deaa (per-band) surface.
NEW_SIGNATURES = [
    "AcceptanceLengthBand",  # per-band controller class name in the rework
    "batch_size_bands",
    "num_speculative_tokens_per_batch_size=",  # per-band spec config field
]
# Signatures of v4's existing (single-window) controller wiring.
V4_SIGNATURES = [
    "class AcceptanceLengthController",
    "def observe_batch",
    "acceptance_length_controller",
    "num_speculative_tokens_per_batch_size",
]


def main() -> int:
    print(f"[{SCRIPT_NAME}] {TAG}")
    new_found = False
    for path, sigs in ((ACCEPTANCE, NEW_SIGNATURES[:2]), (SCHEDULER, NEW_SIGNATURES[2:])):
        try:
            with open(path) as f:
                src = f.read()
        except FileNotFoundError:
            print(f"[{SCRIPT_NAME}] NOTE: {path} not found; check skipped")
            continue
        for sig in sigs:
            if sig in src:
                print(
                    f"[{SCRIPT_NAME}] *** LOUD NOTE: found '{sig}' in {os.path.basename(path)} — "
                    "the f71e0deaa per-band surface may already be present. "
                    "This SKIP may be stale: re-evaluate the dynamic-depth "
                    "port. ***"
                )
                new_found = True
    if not new_found:
        v4_found = []
        for path, sigs in ((ACCEPTANCE, V4_SIGNATURES[:2]), (SCHEDULER, V4_SIGNATURES[2:])):
            try:
                with open(path) as f:
                    src = f.read()
            except FileNotFoundError:
                continue
            v4_found.extend(sig for sig in sigs if sig in src)
        print(
            f"[{SCRIPT_NAME}] SKIP (not applicable as a Batch 2 port): "
            f"f71e0deaa's per-band dynamic-depth rework targets the "
            f"post-Sep lineage — every source anchor fails on 881ac39a4 "
            f"(verified via git apply --check) — and the v4 tree already "
            f"carries the earlier single-window acceptance-length machinery"
            f"{f' (found: {chr(44).join(v4_found)})' if v4_found else ' (signature files not readable in this tree — check manually)'}. "
            f"Porting would "
            f"replace working machinery across 15+ drifted files and change "
            f"serving behavior, which Batch 2's A/B must stay clean of. "
            f"Revisit as its own env-gated batch (VLLM_K3_DYNAMIC_DEPTH) "
            f"if wanted. No changes made."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
