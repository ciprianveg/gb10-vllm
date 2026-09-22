#!/usr/bin/env bash
# fix-mamba-align-state-free — free the correct KDA align state block
# (two steps ago, not one step ago). Fixes the 7.6× memory inflation
# on APC-on long prefill where KDA state blocks accumulate because
# remove_skipped_blocks tracks the previous step's running-state block
# instead of the step before that.
#
# Root cause: in MambaManager.allocate_new_blocks (align mode),
# last_state_block_idx is updated to the CURRENT step's running-state
# block. remove_skipped_blocks then tries to free that same block
# (which is still needed for the current step's CoW copy). The block
# that should be freed is the one from TWO steps ago — the value
# last_state_block_idx held BEFORE the previous update. We now save
# that old value in _two_steps_ago_block_idx and free it instead.
#
# Read-only aside from the fix; no behavior change for non-align modes.

set -euo pipefail

FILE="/opt/kimi-k3/vllm/vllm/v1/core/single_type_kv_cache_manager.py"

if grep -q "fix-mamba-align-state-free" "$FILE" 2>/dev/null; then
    echo "[fix-mamba-align-state-free] already applied"
    exit 0
fi

python3 - "$FILE" <<'PYEOF'
import sys
import py_compile
p = sys.argv[1]
s = open(p).read()

# --- 1. Add _two_steps_ago_block_idx dict in __init__ (align mode) ---
old_init = '''        if self.mamba_cache_mode == "align":
            # Mapping from request ID to the index of the block
            # allocated in the previous step
            self.last_state_block_idx: dict[str, int] = {}
            # The set of the requests that have been allocated blocks
            self._allocated_block_reqs: set[str] = set()
            # Requests that registered their own last-prompt-boundary partial
            # tail (producers). On the next step's CoW the boundary state moves
            # into a private cow_block; we record that block for connector
            # offload (see _pending_partial_tail_offloads).
            self._producer_partial_tail_reqs: dict[str, int] = {}'''
new_init = '''        if self.mamba_cache_mode == "align":
            # Mapping from request ID to the index of the block
            # allocated in the previous step
            self.last_state_block_idx: dict[str, int] = {}
            # Mapping from request ID to the index of the block
            # allocated TWO steps ago (the one that can be freed).
            self._two_steps_ago_block_idx: dict[str, int] = {}
            # The set of the requests that have been allocated blocks
            self._allocated_block_reqs: set[str] = set()
            # Requests that registered their own last-prompt-boundary partial
            # tail (producers). On the next step's CoW the boundary state moves
            # into a private cow_block; we record that block for connector
            # offload (see _pending_partial_tail_offloads).
            self._producer_partial_tail_reqs: dict[str, int] = {}'''
assert old_init in s, "init anchor not found"
s = s.replace(old_init, new_init, 1)

# --- 2. In allocate_new_blocks (align), save old last_state_block_idx before updating ---
old_alloc = '''                if blocks_allocated:
                    # We always save the running state at the last
                    # (1 + num_speculative_blocks) block
                    self.last_state_block_idx[request_id] = (
                        prev_block_len - 1 - self.num_speculative_blocks
                    )'''
new_alloc = '''                if blocks_allocated:
                    # Save the previous step's running-state block index
                    # so we can free it two steps later (fix-mamba-align-state-free).
                    old_last = self.last_state_block_idx.get(request_id)
                    if old_last is not None:
                        self._two_steps_ago_block_idx[request_id] = old_last
                    # We always save the running state at the last
                    # (1 + num_speculative_blocks) block
                    self.last_state_block_idx[request_id] = (
                        prev_block_len - 1 - self.num_speculative_blocks
                    )'''
assert old_alloc in s, "allocate_new_blocks anchor not found"
s = s.replace(old_alloc, new_alloc, 1)

# --- 3. In remove_skipped_blocks (align), free the two-steps-ago block ---
old_remove = '''        if self.mamba_cache_mode == "align":
            # `last_state_block_idx` refers to the block index allocated two steps ago.
            # The block allocated in the previous step is used to copy Mamba states
            # into the block allocated in the current step; the earlier block is
            # no longer needed and should be freed here.
            last_state_block_idx = self.last_state_block_idx.get(request_id)
            # Blocks allocated during prefill may be non-contiguous. Use
            # `last_state_block_idx` to free the appropriate block and replace it
            # with a null block.
            if (
                last_state_block_idx is not None
                and last_state_block_idx
                < cdiv(processed_computed_tokens, self.block_size) - 1
            ):
                blocks = self.req_to_blocks[request_id]
                if blocks[last_state_block_idx] != self._null_block:
                    self.block_pool.free_blocks([blocks[last_state_block_idx]])
                    blocks[last_state_block_idx] = self._null_block'''
new_remove = '''        if self.mamba_cache_mode == "align":
            # fix-mamba-align-state-free: free the block from TWO steps ago,
            # which we saved in _two_steps_ago_block_idx during the previous
            # allocate_new_blocks call. The previous step's running-state block
            # (last_state_block_idx) is still needed for the current step's CoW.
            two_steps_ago_idx = self._two_steps_ago_block_idx.get(request_id)
            if two_steps_ago_idx is not None:
                # Blocks allocated during prefill may be non-contiguous. Use
                # the saved index to free the appropriate block and replace it
                # with a null block.
                blocks = self.req_to_blocks[request_id]
                if two_steps_ago_idx < len(blocks):
                    blk = blocks[two_steps_ago_idx]
                    if blk != self._null_block:
                        self.block_pool.free_blocks([blk])
                        blocks[two_steps_ago_idx] = self._null_block
                # Clean up the saved index
                self._two_steps_ago_block_idx.pop(request_id, None)'''
assert old_remove in s, "remove_skipped_blocks anchor not found"
s = s.replace(old_remove, new_remove, 1)

# --- 4. Clean up _two_steps_ago_block_idx on request free ---
old_free = '''    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        if self.mamba_cache_mode == "align":
            self._allocated_block_reqs.discard(request_id)
            self.last_state_block_idx.pop(request_id, None)
            self._producer_partial_tail_reqs.pop(request_id, None)
            # A hand-off whose request died in this same scheduling pass must
            # not reach the connector: its unpin hook (free) has already run.
            self._pending_partial_tail_offloads = [
                entry
                for entry in self._pending_partial_tail_offloads
                if entry[0] != request_id
            ]
        return super().pop_blocks_for_free(request_id)'''
new_free = '''    def pop_blocks_for_free(self, request_id: str) -> list[KVCacheBlock]:
        if self.mamba_cache_mode == "align":
            self._allocated_block_reqs.discard(request_id)
            self.last_state_block_idx.pop(request_id, None)
            self._two_steps_ago_block_idx.pop(request_id, None)
            self._producer_partial_tail_reqs.pop(request_id, None)
            # A hand-off whose request died in this same scheduling pass must
            # not reach the connector: its unpin hook (free) has already run.
            self._pending_partial_tail_offloads = [
                entry
                for entry in self._pending_partial_tail_offloads
                if entry[0] != request_id
            ]
        return super().pop_blocks_for_free(request_id)'''
assert old_free in s, "pop_blocks_for_free anchor not found"
s = s.replace(old_free, new_free, 1)

open(p, "w").write(s)
py_compile.compile(p, doraise=True)
print("[fix-mamba-align-state-free] patched + py_compile OK")
PYEOF