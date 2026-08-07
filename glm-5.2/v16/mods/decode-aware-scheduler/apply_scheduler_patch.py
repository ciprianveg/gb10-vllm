#!/usr/bin/env python3
"""Apply decode-aware prefill scheduler changes to vllm/v1/core/sched/scheduler.py.

This script handles the scheduler.py changes from the decode-aware prefill
scheduler patch, adapted for the v16 fork (local-inference-lab/vllm
@fathomless-firmament-v16-unified) which has a spec-decode padding block
not present in the original patch baseline.

Idempotent: detects already-applied changes and exits cleanly.
"""
import re
import sys

import os
VLLM_BASE = "/opt/venv/lib/python3.12/site-packages" if os.path.isdir("/opt/venv/lib/python3.12/site-packages/vllm") else "/usr/local/lib/python3.12/dist-packages"
FILE_PATH = VLLM_BASE + "/vllm/v1/core/sched/scheduler.py"

with open(FILE_PATH, "r") as f:
    content = f.read()

# ── Hunk 1: Add instance variables in __init__ ──────────────────────────────

INIT_MARKER = "        self.max_model_len = vllm_config.model_config.max_model_len\n"
INIT_ADDITION = """        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_decode_aware_prefill = (
            self.scheduler_config.enable_decode_aware_prefill
        )
        self.decode_prefill_token_budget = (
            self.scheduler_config.decode_prefill_token_budget
        )
        self.idle_prefill_token_budget = self.scheduler_config.idle_prefill_token_budget
        self.max_long_prefills_per_step = (
            self.scheduler_config.max_long_prefills_per_step
        )
        # request_id -> most recent scheduler step that issued long-prefill work.
        # Selecting the oldest entry provides round-robin behavior while arrival
        # time breaks ties for requests that have never run.
        self._long_prefill_last_scheduled: dict[str, int] = {}
"""

if "self._long_prefill_last_scheduled" not in content:
    if INIT_MARKER not in content:
        print("ERROR: Could not find __init__ marker for Hunk 1")
        sys.exit(1)
    content = content.replace(INIT_MARKER, INIT_ADDITION, 1)
    print("  Hunk 1: added instance variables to __init__")
else:
    print("  Hunk 1: already applied")

# ── Hunk 2: Add _is_long_prefill_request and _select_long_prefill_request_ids ──

METHODS_MARKER = "    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:"
METHODS_ADDITION = """    def _is_long_prefill_request(self, request: Request) -> bool:
        if (
            request.num_prompt_tokens
            <= self.scheduler_config.long_prefill_token_threshold
        ):
            return False
        if request.status == RequestStatus.RUNNING:
            return request.is_prefill_chunk
        return request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED)

    def _select_long_prefill_request_ids(self) -> set[str]:
        if not self.enable_decode_aware_prefill:
            return set()

        candidates: dict[str, Request] = {}
        for request in itertools.chain(
            self.running, self.waiting, self.skipped_waiting
        ):
            if self._is_long_prefill_request(request):
                candidates[request.request_id] = request

        active_request_ids = self.requests.keys()
        self._long_prefill_last_scheduled = {
            request_id: step
            for request_id, step in self._long_prefill_last_scheduled.items()
            if request_id in active_request_ids
        }
        ordered = sorted(
            candidates.values(),
            key=lambda request: (
                self._long_prefill_last_scheduled.get(request.request_id, -1),
                request.arrival_time,
            ),
        )
        return {
            request.request_id for request in ordered[: self.max_long_prefills_per_step]
        }

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:"""

if "_is_long_prefill_request" not in content:
    if METHODS_MARKER not in content:
        print("ERROR: Could not find schedule() method marker for Hunk 2")
        sys.exit(1)
    content = content.replace(METHODS_MARKER, METHODS_ADDITION, 1)
    print("  Hunk 2: added _is_long_prefill_request and _select_long_prefill_request_ids")
else:
    print("  Hunk 2: already applied")

# ── Hunk 3: Add decode-aware budget logic at start of schedule() ────────────

BUDGET_MARKER = "            # Do not schedule any requests when paused.\n            token_budget = 0\n"
BUDGET_ADDITION = """            # Do not schedule any requests when paused.
            token_budget = 0

        has_active_decode = any(
            not request.is_prefill_chunk for request in self.running
        )
        prefill_token_budget = token_budget
        if self.enable_decode_aware_prefill:
            configured_prefill_budget = (
                self.decode_prefill_token_budget
                if has_active_decode
                else self.idle_prefill_token_budget
            )
            prefill_token_budget = min(
                configured_prefill_budget, self.max_num_scheduled_tokens
            )
        prefill_tokens_scheduled = 0
        selected_long_prefill_ids = self._select_long_prefill_request_ids()

"""

if "selected_long_prefill_ids" not in content:
    if BUDGET_MARKER not in content:
        print("ERROR: Could not find token_budget marker for Hunk 3")
        sys.exit(1)
    content = content.replace(BUDGET_MARKER, BUDGET_ADDITION, 1)
    print("  Hunk 3: added decode-aware budget logic to schedule()")
else:
    print("  Hunk 3: already applied")

# ── Hunk 4: Add prefill filtering in running queue loop ─────────────────────

# 4a: Insert is_prefill/is_long_prefill checks before num_new_tokens calculation
RUNNING_LOOP_OLD = """                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)
"""

RUNNING_LOOP_NEW = """                req_index += 1
                continue

            is_prefill = request.is_prefill_chunk
            is_long_prefill = self._is_long_prefill_request(request)
            if (
                self.enable_decode_aware_prefill
                and is_long_prefill
                and request.request_id not in selected_long_prefill_ids
            ):
                req_index += 1
                continue

            remaining_prefill_budget = prefill_token_budget - prefill_tokens_scheduled
            if (
                self.enable_decode_aware_prefill
                and is_prefill
                and remaining_prefill_budget <= 0
            ):
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if (
                not (self.enable_decode_aware_prefill and is_prefill)
                and 0
                < self.scheduler_config.long_prefill_token_threshold
                < num_new_tokens
            ):
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)
            if self.enable_decode_aware_prefill and is_prefill:
                num_new_tokens = min(num_new_tokens, remaining_prefill_budget)
"""

if "is_long_prefill = self._is_long_prefill_request(request)" not in content.split("def schedule")[1].split("# Encoder-related")[0] if "def schedule" in content else True:
    # More precise check: look for the running-loop specific marker
    NEEDS_HUNK4 = "is_prefill = request.is_prefill_chunk" not in content
else:
    NEEDS_HUNK4 = False

if NEEDS_HUNK4:
    if RUNNING_LOOP_OLD not in content:
        print("ERROR: Could not find running loop marker for Hunk 4")
        sys.exit(1)
    content = content.replace(RUNNING_LOOP_OLD, RUNNING_LOOP_NEW, 1)
    print("  Hunk 4: added prefill filtering to running queue loop")
else:
    print("  Hunk 4: already applied")

# ── Hunk 5: Track prefill tokens in running queue ───────────────────────────

TRACK_RUNNING_OLD = """            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
"""

TRACK_RUNNING_NEW = """            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            if self.enable_decode_aware_prefill and is_prefill:
                prefill_tokens_scheduled += num_new_tokens
                if is_long_prefill:
                    self._long_prefill_last_scheduled[request_id] = self.current_step
            req_index += 1

            # Speculative decode related.
"""

if "prefill_tokens_scheduled += num_new_tokens" not in content:
    if TRACK_RUNNING_OLD not in content:
        print("ERROR: Could not find running queue tracking marker for Hunk 5")
        sys.exit(1)
    content = content.replace(TRACK_RUNNING_OLD, TRACK_RUNNING_NEW, 1)
    print("  Hunk 5: added prefill token tracking to running queue")
else:
    print("  Hunk 5: already applied")

# ── Hunk 6: Decode-aware logic in waiting queue (v16-specific) ──────────────
# The v16 fork has a spec-decode padding block between num_new_tokens and threshold.
# We insert the decode-aware logic AFTER the spec-decode padding, replacing the
# threshold block.

WAITING_OLD = """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked"""

WAITING_NEW = """                    is_long_prefill = self._is_long_prefill_request(request)
                    if self.enable_decode_aware_prefill:
                        if (
                            is_long_prefill
                            and request_id not in selected_long_prefill_ids
                        ):
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        remaining_prefill_budget = (
                            prefill_token_budget - prefill_tokens_scheduled
                        )
                        if remaining_prefill_budget <= 0:
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        num_new_tokens = min(num_new_tokens, remaining_prefill_budget)
                    else:
                        threshold = self.scheduler_config.long_prefill_token_threshold
                        if 0 < threshold < num_new_tokens:
                            num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked"""

# Check if already applied (the "is_long_prefill" in the waiting section)
# We need to be careful not to match the running section's is_long_prefill
# Look for the specific pattern in the waiting section
WAITING_CHECK = "is_long_prefill = self._is_long_prefill_request(request)\n                    if self.enable_decode_aware_prefill:\n                        if (\n                            is_long_prefill\n                            and request_id not in selected_long_prefill_ids\n                        ):\n                            request_queue.pop_request()"

if WAITING_CHECK not in content:
    if WAITING_OLD not in content:
        print("ERROR: Could not find waiting queue threshold marker for Hunk 6")
        sys.exit(1)
    content = content.replace(WAITING_OLD, WAITING_NEW, 1)
    print("  Hunk 6: added decode-aware logic to waiting queue")
else:
    print("  Hunk 6: already applied")

# ── Hunk 7: Track prefill tokens in waiting queue ───────────────────────────

TRACK_WAITING_OLD = """                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
"""

TRACK_WAITING_NEW = """                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                if self.enable_decode_aware_prefill:
                    prefill_tokens_scheduled += num_new_tokens
                    if is_long_prefill:
                        self._long_prefill_last_scheduled[request_id] = (
                            self.current_step
                        )
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
"""

# Check if already applied by looking for the tracking before request.status
WAITING_TRACK_CHECK = "if self.enable_decode_aware_prefill:\n                    prefill_tokens_scheduled += num_new_tokens\n                    if is_long_prefill:\n                        self._long_prefill_last_scheduled[request_id] = (\n                            self.current_step\n                        )\n                request.status = RequestStatus.RUNNING"

if WAITING_TRACK_CHECK not in content:
    # This marker appears in both running and waiting sections.
    # We need the SECOND occurrence (waiting section). The first was already
    # handled by Hunk 5 which uses a different surrounding context.
    idx1 = content.find(TRACK_WAITING_OLD)
    if idx1 == -1:
        print("ERROR: Could not find waiting queue tracking marker for Hunk 7")
        sys.exit(1)
    idx2 = content.find(TRACK_WAITING_OLD, idx1 + 1)
    if idx2 == -1:
        # Only one occurrence — might be because Hunk 5 already changed the first
        content = content.replace(TRACK_WAITING_OLD, TRACK_WAITING_NEW, 1)
    else:
        content = content[:idx2] + TRACK_WAITING_NEW + content[idx2 + len(TRACK_WAITING_OLD):]
    print("  Hunk 7: added prefill token tracking to waiting queue")
else:
    print("  Hunk 7: already applied")

# ── Hunk 8: Add assertion at end of schedule() ──────────────────────────────

ASSERT_OLD = "        assert token_budget >= 0\n        assert len(self.running) <= self.max_num_running_reqs\n"
ASSERT_NEW = "        assert token_budget >= 0\n        assert prefill_tokens_scheduled <= prefill_token_budget\n        assert len(self.running) <= self.max_num_running_reqs\n"

if "assert prefill_tokens_scheduled <= prefill_token_budget" not in content:
    if ASSERT_OLD not in content:
        print("ERROR: Could not find assertion marker for Hunk 8")
        sys.exit(1)
    content = content.replace(ASSERT_OLD, ASSERT_NEW, 1)
    print("  Hunk 8: added prefill budget assertion")
else:
    print("  Hunk 8: already applied")

# ── Write the patched file ──────────────────────────────────────────────────

with open(FILE_PATH, "w") as f:
    f.write(content)

print("  scheduler.py patched successfully")
