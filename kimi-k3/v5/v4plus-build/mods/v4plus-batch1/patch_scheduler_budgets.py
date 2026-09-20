#!/usr/bin/env python3
# V4PLUS M3 — fork #605 separate DSpark/DFlash scheduler input budgets.
#
# VERDICT: NOT PORTABLE to the v4 tree — documented skip (this script is a
# presence check + explanation, always exits 0).
#
# Why: the #605 diff (vllm_605_scheduler_budgets.diff) builds on upstream
# #52996's draft_slots reservation machinery:
#     draft_slots = spec.max_num_new_slots_for_drafting
#     input_budget = max_num_batched_tokens - (num_new_tokens + draft_slots)
# and separates the draft rows' admission budget so target prefill chunks
# keep the full token budget. The v4 tree at 881ac39a4 PREDATES #52996:
#   * vllm/v1/core/sched/scheduler.py has NO draft_slots / input_budget
#     machinery at all (verified against the clone);
#   * vllm/config/speculative.py has no use_dflash()/use_dspark() helpers
#     and no max_num_new_slots_for_drafting;
#   * v4 schedules spec tokens through its own fork-native accounting
#     (request.num_tokens_with_spec + num_output_placeholders folded into
#     each request's num_new_tokens, admitted against a single
#     token_budget), so the problem #605 fixes — draft-slot reservation
#     starving target prefill chunks — structurally cannot occur here.
# Porting would mean first dragging in upstream #52996 (a scheduling
# BEHAVIOR change), which Batch 1 excludes ("speed-only, no behavior
# change").
#
# If a future batch wants this: port #52996 onto the v4 scheduler, then
# re-evaluate the #605 hunks (they should then apply with small fuzz).

import os
import sys

SCRIPT_NAME = "patch_scheduler_budgets"

VLLM_ROOT = os.environ.get("VLLM_ROOT", "/opt/kimi-k3/vllm/vllm")
SCHED_PATH = os.path.join(VLLM_ROOT, "v1", "core", "sched", "scheduler.py")
SPEC_CFG_PATH = os.path.join(VLLM_ROOT, "config", "speculative.py")

# Signatures that matter: SCHEDULER-side machinery from upstream #52996
# (its presence would mean the #605 port should be re-evaluated). The
# speculative-config helpers (use_dflash/use_dspark/
# max_num_new_slots_for_drafting) DO exist in v4 (informational only —
# they arrived with the fork's in-tree spec machinery and are used
# elsewhere, not by the scheduler's budgeting).
SCHED_SIGNATURES = [
    "draft_slots = spec.max_num_new_slots_for_drafting",
    "input_budget = self.scheduler_config.max_num_batched_tokens",
    "draft_input_budget",
    "if input_budget <= draft_slots",
]
SPEC_SIGNATURES = [
    "def use_dflash(",
    "def use_dspark(",
    "max_num_new_slots_for_drafting",
]


def main() -> int:
    print(f"[{SCRIPT_NAME}] fork #605 scheduler input budgets — applicability check")
    sched_found = False
    try:
        with open(SCHED_PATH) as f:
            sched = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] NOTE: {SCHED_PATH} not found; check skipped")
        sched = ""
    for sig in SCHED_SIGNATURES:
        if sig in sched:
            print(
                f"  *** LOUD NOTE: found '{sig}' in scheduler.py — the v4 "
                "tree has (parts of) upstream #52996's draft_slots "
                "reservation. This skip is STALE: re-evaluate the #605 "
                "port (its hunks may now apply with small fuzz). ***"
            )
            sched_found = True
    try:
        with open(SPEC_CFG_PATH) as f:
            spec_cfg = f.read()
        helpers = all(s in spec_cfg for s in SPEC_SIGNATURES)
        if helpers:
            print(
                "  NOTE (informational): the speculative-config helpers "
                "(use_dflash/use_dspark/max_num_new_slots_for_drafting) are "
                "present in v4 — they arrived with the fork's in-tree spec "
                "machinery and are NOT used by the scheduler's budgeting; "
                "this does not change the verdict."
            )
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] NOTE: {SPEC_CFG_PATH} not found; helper check skipped")
    if not sched_found:
        print(
            "  SKIP (not applicable): the v4 scheduler has NO draft-slot "
            "reservation machinery (no draft_slots / input_budget paths — "
            "verified). v4 schedules spec tokens through fork-native "
            "num_tokens_with_spec + num_output_placeholders accounting "
            "folded into each request's own token count, so the cross-"
            "request draft-slot starvation #605 fixes cannot occur. Porting "
            "would require first dragging in upstream #52996 (a scheduling "
            "behavior change), which Batch 1 excludes. No changes made."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
