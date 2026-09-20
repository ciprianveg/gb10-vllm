#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V4PLUS DCPFIX — un-gate the 8-row fused verify family at DCP>1.

Serving now runs DCP8.  Under it, the 8-row fused verify family (nst=6,
v4plus-batch4) silently falls back to flat because batch4 keyed its plan
family to ``dcp_world_size == 1`` (the #565 DCP machinery had not been
audited then).  This mod removes that restriction: the 8-row family is
created at any dcp_size, and the strided scatter/gather path in
forward_mqa no longer rejects dcp>1.

WHY THIS IS SAFE WITHOUT THE #565 mla.py HUNKS (the 4-row DCP8 audit):
the 4-row fused family has been live at DCP8 in prod testing (builder
prints ENABLED; the 64K quality gate passed), and it uses exactly the
same DCP machinery the 8-row family needs — nothing in the B12X fused
verify path consumes the #565 mla.py query-replication hunks:

  1. Per-row visibility under DCP: the DCP branch of
     ``_materialize_query_cache_seq_lens`` (verbatim #565 inline math,
     installed as the batch2 gate repair) computes per-row LOCAL visible
     lengths from ``decode_metadata.dcp_tot_seq_lens`` via the
     round-robin interleave math — per-ROW, hence row-count agnostic:
     it feeds the 4-row family's ``flat_lens`` verbatim and the 8-row
     family's strided ``row_lens`` scatter (build() copies
     ``flat_lens.view(B, query_len)`` into ``row_lens.view(B, 8)[:, :query_len]``)
     identically.
  2. Query heads under DCP: ``forward_mqa``'s DCP branch runs BEFORE the
     strided branch and all-gathers the per-rank head shards into the
     full-head compact q (``dcp_b12x_all_gather_heads`` into
     ``dense_mla_padded_q[:total_q, :effective_heads]``) — row-count
     agnostic.  The strided scatter's source is therefore the gathered
     q; at dcp>1, ``kernel_heads == effective_heads`` (divisibility is
     enforced in ``supports_combination``), so the gathered view is
     contiguous and the scatter shapes hold.
  3. Cross-rank combination: the compact output+LSE gather feeds the
     per-rank LSE reduce (``dcp_a2a_lse_reduce`` / ``cp_lse_ag_out_rs``)
     — exact combination of per-shard partial attentions, row-count
     agnostic, and already proven by the 4-row family at DCP8 (rows
     with no visible local KV contribute the neutral value via the
     -inf LSE convention).
  4. Plans: the 8-row family is created through the same
     ``_create_dense_mla_plan(..., dcp_size=self.dcp_world_size, ...)``
     as the 4-row family — caps sized to the local DCP shard; the b12x
     kernel's ``tiles_per_request`` mapping is dcp-agnostic.

The 8-row path adds ONLY row-layout transforms (strided scatter/gather)
on top of that machinery — no new DCP surface.

TASK-1 STATUS (ported #565 mla.py DCP query-replication hunks): NOT
PORTED — BLOCKED ON ARTIFACTS.  The #565 diff files
(/tmp/opencode/patches/v4plus/, wiped before this task) and the image's
mla.py (never extracted) are both gone, and no surviving session read
those hunks in detail (batch2 excluded them before this conversation's
context begins).  Per the audit above, the B12X fused-verify path does
not consume them — they carry the query-replication machinery for the
NON-B12X MLA backend (DCPGroupColumnParallelLinear plumbing, the
qrep-driven head-check relocation).  If the serving config ever runs
that backend under DCP, the hunks must be ported from a re-extracted
diff; for B12X_MLA at DCP8 this mod is complete without them.  See the
fail-loud section of the report for the exact re-extraction list.

GROUND-TRUTH CAVEAT: the v4img extracts were wiped before this task.
The anchors below are transcribed BYTE-EXACT from this conversation's
own batch4 patch text (patch_vllm_verify_tile.py) and its post-state
reads — the regions they match are text this session authored.  The
simulation runs against a labeled RECONSTRUCTION of the b4 post-state;
re-extract b12x_mla.py from the image and re-run this script (idempotent)
before baking.

Idempotent: both hunks skip when their markers are present.  A missing
anchor prints a NOTE and skips; only a missing prerequisite,
file-not-found, or a broken post-patch compile exits non-zero.
"""

from __future__ import annotations

import os
import py_compile
import sys

SCRIPT_NAME = "patch_ungate_verify8_dcp"
TAG = "# V4PLUS-DCPFIX (un-gate 8-row fused verify at DCP>1)"

VLLM_ROOT = os.environ.get("VLLM_ROOT", "/opt/kimi-k3/vllm/vllm")
B12X_MLA = os.path.join(VLLM_ROOT, "v1", "attention", "backends", "mla", "b12x_mla.py")


def apply_hunks(path: str, hunks: list[tuple[str, str, str, str]]) -> bool:
    """Apply (name, anchor, replacement, present) hunks; True if all well."""
    try:
        with open(path) as f:
            src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {path} not found", file=sys.stderr)
        return False
    changed = False
    ok = True
    for name, anchor, repl, present in hunks:
        if present in src:
            print(f"[{SCRIPT_NAME}] SKIP  {os.path.basename(path)}: {name} (already present)")
            continue
        n = src.count(anchor)
        if n != 1:
            print(
                f"[{SCRIPT_NAME}] NOTE  {os.path.basename(path)}: {name} — "
                f"anchor found {n}x (want 1); hunk skipped"
            )
            ok = False
            continue
        src = src.replace(anchor, repl, 1)
        changed = True
        print(f"[{SCRIPT_NAME}] APPLY {os.path.basename(path)}: {name}")
    if changed:
        try:
            compile(src, path, "exec")
        except SyntaxError as exc:
            print(
                f"[{SCRIPT_NAME}] ERROR: {path} does not compile after patch: {exc}",
                file=sys.stderr,
            )
            return False
        with open(path, "w") as f:
            f.write(src)
        py_compile.compile(path, doraise=True)
    return ok


def check_prerequisites() -> bool:
    """Fail loud unless b12x_mla.py is the batch2+batch4 post-state."""
    try:
        with open(B12X_MLA) as f:
            src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {B12X_MLA} not found", file=sys.stderr)
        return False
    ok = True
    # batch2 markers (B1b + the bucketing fix).
    for marker in (
        "_dense_mla_verify_plans",
        "VLLM_K3_BUCKETED_DECODE",
        "tiled_verify_rows = 8",
    ):
        if marker not in src:
            print(
                f"[{SCRIPT_NAME}] ERROR: {B12X_MLA} lacks the v4plus-batch2/4 "
                f"marker {marker!r} — apply mods/v4plus-batch2 and "
                "mods/v4plus-batch4 first; this mod builds on their "
                "post-state.",
                file=sys.stderr,
            )
            ok = False
    # batch4 markers (the 8-row family + the strided branch).
    for marker in (
        "self._dense_mla_verify8_plans: dict[int, Any] = {}",
        "strided_verify = verify_rows == 8",
        "VLLM_K3_FUSED_TILE",
    ):
        if marker not in src:
            print(
                f"[{SCRIPT_NAME}] ERROR: {B12X_MLA} lacks the v4plus-batch4 "
                f"marker {marker!r} — apply mods/v4plus-batch4 first.",
                file=sys.stderr,
            )
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Hunk 1: un-gate the 8-row family creation (any dcp_size).
# Anchor = the batch4 family block's comment tail + condition (text this
# session authored in patch_vllm_verify_tile.py).
# ---------------------------------------------------------------------------

U_FAMILY_ANCHOR = (
    "        # marks them invalid and skips both their math and their\n"
    "        # output/LSE stores. dcp=1 only: the strided scatter/gather\n"
    "        # is not wired through the DCP head-gather path.\n"
    "        self._dense_mla_verify8_plans: dict[int, Any] = {}\n"
    "        max_verify8_batch = min(\n"
    "            int(vllm_config.scheduler_config.max_num_seqs),\n"
    "            max_dense_mla_rows // 8,\n"
    "        )\n"
    "        if (\n"
    "            fused_verify\n"
    "            and fused_tile == 8\n"
    "            and 4 <= num_spec_tokens <= 7\n"
    "            and self.dcp_world_size == 1\n"
    "            and max_verify8_batch >= 1\n"
    "        ):\n"
)
U_FAMILY_REPLACEMENT = (
    "        # marks them invalid and skips both their math and their\n"
    "        # output/LSE stores. V4PLUS-DCPFIX: the family is now created\n"
    "        # at ANY dcp_size — the DCP composition is the same machinery\n"
    "        # the 4-row family already uses in production at DCP8: the\n"
    "        # DCP branch of _materialize_query_cache_seq_lens produces\n"
    "        # per-row LOCAL visible lengths, forward_mqa's DCP head gather\n"
    "        # runs before the strided scatter (the scatter source is the\n"
    "        # gathered q), and the compact output+LSE gather feeds the\n"
    "        # per-rank LSE reduce (dcp_a2a_lse_reduce / cp_lse_ag_out_rs),\n"
    "        # which combines the per-shard partial attentions exactly.\n"
    "        self._dense_mla_verify8_plans: dict[int, Any] = {}\n"
    "        max_verify8_batch = min(\n"
    "            int(vllm_config.scheduler_config.max_num_seqs),\n"
    "            max_dense_mla_rows // 8,\n"
    "        )\n"
    "        if (\n"
    "            fused_verify\n"
    "            and fused_tile == 8\n"
    "            and 4 <= num_spec_tokens <= 7\n"
    "            and max_verify8_batch >= 1\n"
    "        ):\n"
)
U_FAMILY_PRESENT = "V4PLUS-DCPFIX: the family is now created"

# ---------------------------------------------------------------------------
# Hunk 2: stop rejecting dcp>1 in the strided forward path.
# Anchor = the batch4 strided-branch head with the dcp raise (text this
# session authored).
# ---------------------------------------------------------------------------

U_STRAIGHT_ANCHOR = (
    "        strided_out_view = None\n"
    "        if strided_verify:\n"
    "            # V4PLUS-B4: scatter the compact verify rows into the\n"
    "            # per-request strided layout (query_len real rows of each\n"
    "            # 8-row span). Head padding (kernel_heads > effective)\n"
    "            # stays zero from the buffer's torch.zeros allocation.\n"
    "            if metadata_dcp_world_size != 1:\n"
    "                raise RuntimeError(\n"
    "                    \"B12X_MLA strided fused verify requires a single KV \"\n"
    "                    \"shard (dcp=1).\"\n"
    "                )\n"
    "            strided_q = getattr(attn_metadata, \"dense_mla_verify_q\", None)\n"
)
U_STRAIGHT_REPLACEMENT = (
    "        strided_out_view = None\n"
    "        if strided_verify:\n"
    "            # V4PLUS-B4: scatter the compact verify rows into the\n"
    "            # per-request strided layout (query_len real rows of each\n"
    "            # 8-row span). Head padding (kernel_heads > effective)\n"
    "            # stays zero from the buffer's torch.zeros allocation.\n"
    "            # V4PLUS-DCPFIX: dcp>1 is no longer rejected. The DCP head\n"
    "            # gather above has already produced the full-head compact\n"
    "            # q (gathered_q), so the scatter source is the gathered\n"
    "            # rows; the per-row visibility came from the DCP branch of\n"
    "            # _materialize_query_cache_seq_lens (local shard lens), and\n"
    "            # the compact output+LSE gather below feeds the per-rank\n"
    "            # LSE reduce — the same composition the 4-row family uses\n"
    "            # at DCP8.\n"
    "            strided_q = getattr(attn_metadata, \"dense_mla_verify_q\", None)\n"
)
U_STRAIGHT_PRESENT = "V4PLUS-DCPFIX: dcp>1 is no longer rejected"


def main() -> int:
    print(f"[{SCRIPT_NAME}] {TAG}")
    print(f"[{SCRIPT_NAME}] VLLM_ROOT={VLLM_ROOT}")
    if not check_prerequisites():
        print(
            f"[{SCRIPT_NAME}] PREREQUISITE FAILED: b12x_mla.py is not the "
            "batch2+batch4 post-state; refusing to patch.",
            file=sys.stderr,
        )
        return 1

    ok = apply_hunks(
        B12X_MLA,
        [
            (
                "un-gate 8-row family creation (any dcp_size)",
                U_FAMILY_ANCHOR,
                U_FAMILY_REPLACEMENT,
                U_FAMILY_PRESENT,
            ),
            (
                "strided path: accept dcp>1 (gather->scatter->LSE-reduce)",
                U_STRAIGHT_ANCHOR,
                U_STRAIGHT_REPLACEMENT,
                U_STRAIGHT_PRESENT,
            ),
        ],
    )

    if ok:
        print(
            f"[{SCRIPT_NAME}] NOTE: the 4-row family's DCP8 path is the "
            "template — per-row local lens (the DCP branch of "
            "_materialize_query_cache_seq_lens), the DCP head gather "
            "upstream of the scatter, and the per-rank LSE reduce "
            "downstream of the compact gather. The 8-row path adds only "
            "row-layout transforms on top; no new DCP surface."
        )
        print(
            f"[{SCRIPT_NAME}] *** LOUD NOTE: the #565 mla.py DCP "
            "query-replication hunks were NOT ported — the diff artifacts "
            "and the image's mla.py are missing from /tmp (wiped), and no "
            "surviving session read them. Per the audit, the B12X fused "
            "verify path does not consume them (they serve the non-B12X "
            "MLA backend's DCP verify). If that backend ever runs under "
            "DCP, re-extract the #565 diff + mla.py and port them "
            "separately. ***"
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
