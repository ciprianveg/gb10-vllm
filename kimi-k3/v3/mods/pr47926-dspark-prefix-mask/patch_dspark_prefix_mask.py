#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PR_47926 — mask prefix-cache-restored tokens out of the DFlash/DSpark draft
context (upstream: https://github.com/vllm-project/vllm/pull/47926, DRAFT).

Cache-restored (prefix-cache hit / preempt-resumed) tokens never flow through
the target forward, so the drafter's context KV is never written for them.
Without this patch the draft attends over the full sequence and reads stale KV,
degrading acceptance toward ~1.0 on long shared prefixes. The fix threads a
per-request-slot "num_cached_tokens" count through
states.py -> model_runner.py -> speculator.py, shortens the draft seq_lens by
the restored whole blocks in _prepare_dflash_inputs_kernel, and left-shifts
each draft block-table row to hide the stale slots (shift_draft_block_tables).

NOTE on the Kimi-K3 fork (vllm @881ac39, image vllm-node-kimi3-sm121): the
fork ALREADY carries a later revision of this draft PR in all three files,
with fork-specific evolutions on top of the upstream diff snapshot:
  - states.py also keeps a CPU mirror (num_cached_tokens_np).
  - model_runner.py passes both the GPU tensor and the np mirror to
    set_num_cached_tokens (2-arg signature).
  - speculator.py adds _has_unaligned_cached_prefix() and fails closed
    (drafting disabled, draft_tokens=-1) when a restored prefix is not
    block-aligned; upstream only documents the residual partial block.
  - speculator.py uses block_tables.kernel_block_sizes (upstream: block_sizes).
  - shift_draft_block_tables additionally shifts by the bounded K3 draft-KV
    sliding window (draft_kv_window arg) and hardens the overlapping in-place
    copy (num_stages=1, loop_unroll_factor=1, tl.debug_barrier()).
This script is therefore primarily a verifier/ensurer: every hunk detects as
already present on the fork and is skipped. The apply paths below exist so the
same mod can port the fix (fork-flavored end state) onto a tree that predates
it. Anything this script inserts carries a "# PR_47926" marker; files already
containing the marker are skipped wholesale, and each hunk additionally
detects the fork's (marker-less) code so the fork tree is never dirtied.

Anchor-based, idempotent, warn-and-continue on missing anchors.
The diff's tests/ file is intentionally not ported.

Usage: patch_dspark_prefix_mask.py <VLLM_ROOT>
"""

from __future__ import annotations

import py_compile
import sys
from pathlib import Path

MARKER = "PR_47926"

STATES = "vllm/v1/worker/gpu/states.py"
RUNNER = "vllm/v1/worker/gpu/model_runner.py"
SPEC = "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"

# ---------------------------------------------------------------------------
# Hunk payloads (fork-flavored end state; markers on inserted text)
# ---------------------------------------------------------------------------

S1_TEXT = """
        # PR_47926: Tokens whose KV was restored (e.g. from the prefix cache)
        # rather than computed at the request's most recent (re)admission.
        # The target never runs a forward pass over them, so speculators that
        # derive per-token state from target hidden states (DFlash/DSpark
        # context KV) have nothing for these positions.
        self.num_cached_tokens = StagedWriteTensor(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.num_cached_tokens_np = np.zeros(self.max_num_reqs, dtype=np.int32)
"""

S2_TEXT = """        # PR_47926
        self.num_cached_tokens.stage_write_elem(req_idx, num_computed_tokens)
        self.num_cached_tokens_np[req_idx] = num_computed_tokens
"""

S3_TEXT = """        self.num_cached_tokens.apply_write()  # PR_47926
"""

M1_TEXT = """                # PR_47926: DFlash/DSpark mask cache-restored tokens out of
                # the draft's context (their draft context KV was never
                # computed).
                if hasattr(self.speculator, "set_num_cached_tokens"):
                    self.speculator.set_num_cached_tokens(
                        self.req_states.num_cached_tokens.gpu,
                        self.req_states.num_cached_tokens_np,
                    )
"""

P1_TEXT = """
        # PR_47926: Per-request-slot count of tokens whose KV was restored
        # (e.g. from the prefix cache) at the request's last (re)admission,
        # indexed by req_state_idx. The target never ran a forward pass over
        # them, so their draft context KV was never computed; the prep kernel
        # and the block-table shift in propose() hide them from the draft's
        # attention. The runner replaces this zeros fallback via
        # set_num_cached_tokens.
        self.num_cached_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=device
        )
        self.num_cached_tokens_np = np.zeros(self.max_num_reqs, dtype=np.int32)
"""

P2_TEXT = """
    # PR_47926
    def set_num_cached_tokens(
        self,
        num_cached_tokens: torch.Tensor,
        num_cached_tokens_np: np.ndarray,
    ) -> None:
        \"\"\"Register the runner's per-request-slot cache-restored token counts.

        Indexed by req_state_idx; see the buffer comment in __init__.
        \"\"\"
        self.num_cached_tokens = num_cached_tokens
        self.num_cached_tokens_np = num_cached_tokens_np

    def _has_unaligned_cached_prefix(self, input_batch: InputBatch) -> bool:
        req_state_indices = input_batch.idx_mapping_np[: input_batch.num_reqs]
        cached = self.num_cached_tokens_np[req_state_indices]
        return any(
            np.any(cached % block_size != 0)
            for block_size in self.block_tables.kernel_block_sizes
        )
"""

P3_TEXT = """        # PR_47926: The shared seq_lens buffer carries the cache-shifted
        # draft sequence lengths (see _prepare_dflash_inputs_kernel), which
        # only works if all draft groups shift by the same number of slots
        # per cached block.
        draft_block_sizes = {
            self.block_tables.kernel_block_sizes[gid]
            for gid in self.draft_kv_cache_group_ids
        }
        assert len(draft_block_sizes) == 1, (
            "DFlash requires a uniform block size across draft KV cache "
            f"groups, got {draft_block_sizes}."
        )
"""

P4_TEXT = """        # PR_47926: a block-table shift cannot hide the residual partial
        # block of a block-unaligned cache-restored prefix; fail closed.
        if not dummy_run and self._has_unaligned_cached_prefix(input_batch):
            logger.warning_once(
                "DFlash/DSpark drafting is disabled for a batch containing a "
                "block-unaligned cache-restored prefix because draft KV is "
                "not available for the restored partial block."
            )
            self.draft_tokens[:num_reqs].fill_(-1)
            return self.draft_tokens[:num_reqs]
"""

P6_TEXT = """
        # PR_47926: Cache-restored tokens (e.g. prefix-cache hits) never
        # flowed through the target forward, so their draft context KV was
        # never written and their cache slots hold garbage. Hide them from
        # the draft's attention: shift each draft block-table row left by the
        # restored whole blocks (the prep kernel shortened seq_lens to
        # match). Runs after prepare_dflash_inputs because the slot mappings
        # index the unshifted table; in-place is safe because
        # input_block_tables are regathered from the persistent block tables
        # every step. Block-unaligned restored prefixes fail closed before
        # this path because a block-table shift cannot hide the residual
        # partial block. Skipped for dummy runs, whose idx_mapping does not
        # reference live requests.
        if not dummy_run:
            for gid in self.draft_kv_cache_group_ids:
                shift_draft_block_tables(
                    self.block_tables.input_block_tables[gid],
                    input_batch.idx_mapping,
                    self.num_cached_tokens,
                    self.input_buffers.seq_lens,
                    self.block_tables.kernel_block_sizes[gid],
                    self.draft_kv_window or 0,
                )
"""

P8_NEW = """        # PR_47926: seq_lens is the absolute sequence length the draft
        # attention reads up to (context + query), not just the count of
        # accepted tokens this step — minus the cache-restored whole blocks,
        # which hold no draft KV and are shifted out of the block table (see
        # shift_draft_block_tables).
        num_cached = tl.load(num_cached_tokens_ptr + req_state_idx)
        num_shifted_slots = (num_cached // block_size) * block_size
        # The clamp guards dummy runs, where req_state_idx may point at a
        # stale slot whose cached count exceeds the dummy sequence length.
        tl.store(
            out_seq_lens_ptr + req_idx,
            tl.maximum(
                last_valid_pos + 1 + num_query_per_req - num_shifted_slots,
                num_query_per_req,
            ),
        )
"""

P11_TEXT = """

# PR_47926
@triton.jit
def _shift_draft_block_tables_kernel(
    block_table_ptr,
    block_table_stride,
    idx_mapping_ptr,
    num_cached_tokens_ptr,
    seq_lens_ptr,
    block_size,
    draft_kv_window,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    num_cached = tl.load(num_cached_tokens_ptr + req_state_idx)
    cached_shift = num_cached // block_size
    seq_len = tl.load(seq_lens_ptr + req_idx)
    window_shift = tl.maximum(seq_len - draft_kv_window, 0) // block_size
    window_shift = tl.where(draft_kv_window > 0, window_shift, 0)
    shift = cached_shift + window_shift
    if shift == 0:
        return
    row_ptr = block_table_ptr + req_idx.to(tl.int64) * block_table_stride
    # Only the blocks the shifted sequence still references need to move;
    # seq_lens holds the cache-shifted draft length (written by
    # _prepare_dflash_inputs_kernel, which must run first).
    seq_len -= window_shift * block_size
    tl.store(seq_lens_ptr + req_idx, seq_len)
    num_needed = (seq_len + block_size - 1) // block_size
    num_remaining = tl.minimum(block_table_stride - shift, num_needed)
    # In-place left shift is safe: iterations run in ascending order and each
    # loads its chunk (from offset + shift) before storing (at offset), so no
    # store ever precedes a load of the same element.
    # Keep iterations strictly ordered. Compiler software pipelining may start
    # a store before a later overlapping source load has completed.
    for i in tl.range(
        0,
        num_remaining,
        BLOCK_SIZE,
        num_stages=1,
        loop_unroll_factor=1,
    ):
        offset = i + tl.arange(0, BLOCK_SIZE)
        mask = offset < num_remaining
        block_ids = tl.load(row_ptr + offset + shift, mask=mask, other=0)
        # Source and destination overlap for shifts smaller than BLOCK_SIZE.
        # Ensure every lane has consumed its source before any lane stores.
        tl.debug_barrier()
        tl.store(row_ptr + offset, block_ids, mask=mask)


def shift_draft_block_tables(
    # [max_num_reqs, max_num_blocks]
    block_table: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [max_num_reqs]
    num_cached_tokens: torch.Tensor,
    # [num_reqs] cache-shifted draft sequence lengths
    seq_lens: torch.Tensor,
    block_size: int,
    draft_kv_window: int = 0,
) -> None:
    \"\"\"Shift each request's draft block-table row left by its cache-restored
    whole blocks, hiding slots that hold no draft context KV from the draft's
    attention. Must run after prepare_dflash_inputs (slot mappings index the
    unshifted table, and seq_lens must already hold the shifted lengths).\"\"\"
    num_reqs = idx_mapping.shape[0]
    _shift_draft_block_tables_kernel[(num_reqs,)](
        block_table,
        block_table.stride(0),
        idx_mapping,
        num_cached_tokens,
        seq_lens,
        block_size,
        draft_kv_window,
        BLOCK_SIZE=1024,  # type: ignore
    )
"""

# ---------------------------------------------------------------------------
# Hunk table. mode "insert_after": text is inserted right after anchor.
# mode "replace": old is substituted by new.
# ---------------------------------------------------------------------------

HUNKS = [
    # --- states.py ---------------------------------------------------------
    dict(
        id="states:S1-buffer",
        file=STATES,
        detect="self.num_cached_tokens = StagedWriteTensor(",
        mode="insert_after",
        anchor="""        self.next_prefill_tokens = torch.zeros(
            num_prefill_lookahead,
            self.max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
""",
        text=S1_TEXT,
    ),
    dict(
        id="states:S2-add-request",
        file=STATES,
        detect="self.num_cached_tokens.stage_write_elem(req_idx, num_computed_tokens)",
        mode="insert_after",
        anchor="        self.num_computed_tokens.stage_write_elem(req_idx, num_computed_tokens)\n",
        text=S2_TEXT,
    ),
    dict(
        id="states:S3-apply-writes",
        file=STATES,
        detect="self.num_cached_tokens.apply_write()",
        mode="insert_after",
        anchor="        self.num_computed_tokens.apply_write()\n",
        text=S3_TEXT,
    ),
    # --- model_runner.py ---------------------------------------------------
    # Tightly scoped to the initialize_kv_cache draft-KV region (the
    # DraftModelSpeculator set_attn call site) so it coexists with the
    # pr46932 mod, which patches the memory-estimation region.
    dict(
        id="runner:M1-register-counts",
        file=RUNNER,
        detect='hasattr(self.speculator, "set_num_cached_tokens")',
        mode="insert_after",
        anchor="""                self.speculator.set_attn(
                    self.model_state,
                    self.kv_cache_config,
                    self.block_tables,
                    self.input_buffers,
                    self.attn_groups,
                )
""",
        text=M1_TEXT,
    ),
    # --- speculator.py -----------------------------------------------------
    dict(
        id="spec:P1-init-buffer",
        file=SPEC,
        detect="self.num_cached_tokens = torch.zeros(",
        mode="insert_after",
        anchor="""        self.context_positions = torch.zeros(
            self.max_num_tokens, dtype=torch.int64, device=device
        )
""",
        text=P1_TEXT,
    ),
    dict(
        id="spec:P2-setter-methods",
        file=SPEC,
        detect="def set_num_cached_tokens(",
        mode="insert_after",
        anchor="""        maybe_load_mask_embedding(
            model,
            self.draft_model_config.model,
            self.parallel_drafting_token_id,
        )
        return model
""",
        text=P2_TEXT,
    ),
    dict(
        id="spec:P3-uniform-block-size-assert",
        file=SPEC,
        detect="DFlash requires a uniform block size across draft KV cache",
        mode="insert_after",
        anchor="""        self.draft_kv_cache_group_ids = [
            gid for gid, g in enumerate(self.attn_groups) if g
        ]
        assert self.draft_kv_cache_group_ids, "No draft attention groups found."
        self.draft_kv_cache_group_id = self.draft_kv_cache_group_ids[0]
""",
        text=P3_TEXT,
    ),
    dict(
        id="spec:P4-unaligned-fail-closed",
        file=SPEC,
        detect="_has_unaligned_cached_prefix(input_batch)",
        mode="insert_after",
        anchor="""        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
""",
        text=P4_TEXT,
    ),
    dict(
        id="spec:P5-prepare-call-arg",
        file=SPEC,
        detect="self.block_tables.kernel_block_sizes[gid],\n                self.num_cached_tokens,",
        mode="replace",
        old="""                self.block_tables.kernel_block_sizes[gid],
                self.parallel_drafting_token_id,
""",
        new="""                self.block_tables.kernel_block_sizes[gid],
                self.num_cached_tokens,  # PR_47926
                self.parallel_drafting_token_id,
""",
    ),
    dict(
        id="spec:P6-shift-block-tables",
        file=SPEC,
        detect="Cache-restored tokens (e.g. prefix-cache hits) never flowed through",
        mode="insert_after",
        anchor="""                context_num_tokens_padded=context_num_tokens_padded,
            )
""",
        text=P6_TEXT,
    ),
    dict(
        id="spec:P7-kernel-sig-param",
        file=SPEC,
        detect="num_cached_tokens_ptr,\n    # Scalars",
        mode="replace",
        old="""    block_table_stride,
    # Scalars
    parallel_drafting_token_id,
""",
        new="""    block_table_stride,
    # PR_47926: [max_num_reqs] cache-restored token counts, indexed by
    # req_state_idx.
    num_cached_tokens_ptr,
    # Scalars
    parallel_drafting_token_id,
""",
    ),
    dict(
        id="spec:P8-kernel-seq-lens",
        file=SPEC,
        detect="num_shifted_slots = (num_cached // block_size) * block_size",
        mode="replace",
        old="        tl.store(out_seq_lens_ptr + req_idx, last_valid_pos + 1 + num_query_per_req)\n",
        new=P8_NEW,
    ),
    dict(
        id="spec:P9-wrapper-sig",
        file=SPEC,
        detect="    # [max_num_reqs]\n    num_cached_tokens: torch.Tensor,\n    parallel_drafting_token_id: int,",
        mode="replace",
        old="""    block_size: int,
    parallel_drafting_token_id: int,
""",
        new="""    block_size: int,
    # PR_47926: [max_num_reqs]
    num_cached_tokens: torch.Tensor,
    parallel_drafting_token_id: int,
""",
    ),
    dict(
        id="spec:P10-wrapper-call",
        file=SPEC,
        detect="block_table.stride(0),\n        num_cached_tokens,\n        parallel_drafting_token_id,",
        mode="replace",
        old="""        block_table.stride(0),
        parallel_drafting_token_id,
""",
        new="""        block_table.stride(0),
        num_cached_tokens,  # PR_47926
        parallel_drafting_token_id,
""",
    ),
    dict(
        id="spec:P11-shift-kernel",
        file=SPEC,
        detect="def shift_draft_block_tables(",
        mode="insert_after",
        anchor="""        SAMPLE_FROM_ANCHOR=sample_from_anchor,
        PAD_SLOT_ID=PAD_SLOT_ID,
        BLOCK_SIZE=BLOCK_SIZE,
    )
""",
        text=P11_TEXT,
    ),
]

TOUCHED_FILES = [STATES, RUNNER, SPEC]


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    if not (root / "vllm/v1/worker/gpu").is_dir():
        print(f"[pr47926] ERROR: {root} does not look like a vLLM root", file=sys.stderr)
        return 1

    applied: list[str] = []
    skipped_present: list[str] = []
    skipped_marker: list[str] = []
    warned: list[str] = []
    contents: dict[str, str] = {}
    dirty: set[str] = set()

    for rel in TOUCHED_FILES:
        p = root / rel
        if not p.exists():
            warned.append(f"{rel}: file missing")
            contents[rel] = ""
        else:
            contents[rel] = p.read_text()

    # Marker fast-path is evaluated against the PRISTINE file content, so a
    # hunk applied earlier in this same run cannot mask its siblings.
    marked_files = {rel for rel, c in contents.items() if MARKER in c}

    for h in HUNKS:
        rel = h["file"]
        content = contents[rel]
        if not content:
            warned.append(f"{h['id']}: skipped (file missing)")
            continue
        if rel in marked_files:
            skipped_marker.append(h["id"])
            continue
        if h["detect"] in content:
            skipped_present.append(h["id"])
            continue
        if h["mode"] == "insert_after":
            anchor = h["anchor"]
            if content.count(anchor) != 1:
                warned.append(
                    f"{h['id']}: anchor not unique/found "
                    f"(count={content.count(anchor)}) in {rel}"
                )
                continue
            content = content.replace(anchor, anchor + h["text"], 1)
        else:  # replace
            old = h["old"]
            if content.count(old) != 1:
                warned.append(
                    f"{h['id']}: replace target not unique/found "
                    f"(count={content.count(old)}) in {rel}"
                )
                continue
            content = content.replace(old, h["new"], 1)
        contents[rel] = content
        dirty.add(rel)
        applied.append(h["id"])

    for rel in sorted(dirty):
        (root / rel).write_text(contents[rel])

    # Syntax sanity check on all touched files (patched or not).
    for rel in TOUCHED_FILES:
        p = root / rel
        if p.exists():
            py_compile.compile(str(p), doraise=True)

    print("[pr47926] === summary ===")
    for hid in applied:
        print(f"[pr47926] APPLIED:  {hid}")
    for hid in skipped_present:
        print(f"[pr47926] PRESENT:  {hid} (already in tree, skipped)")
    for hid in skipped_marker:
        print(f"[pr47926] MARKER:   {hid} (PR_47926 marker found, skipped)")
    for msg in warned:
        print(f"[pr47926] WARNING:  {msg}", file=sys.stderr)
    print(
        f"[pr47926] applied={len(applied)} present={len(skipped_present)} "
        f"marker-skipped={len(skipped_marker)} warnings={len(warned)}"
    )
    print("[pr47926] py_compile OK on touched files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
