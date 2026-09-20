#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V4PLUS Batch 4 (vllm side) — 8-row fused verify plans + nst auto-select.

Builds on v4plus-batch2's B1b (fused 4-row verify plans, #565 lineage) and
its VLLM_K3_BUCKETED_DECODE fix.  Extends fused DSpark verification from
nst=3 to nst=4..7 (practically nst=6):

  * envs: new VLLM_K3_FUSED_TILE (default "8").  8 = create the 8-row
    verify family for nst=4..7 AND keep the 4-row family for nst=3;
    4 = 4-row family only (nst=3); 0 = fused verify disabled entirely.
    VLLM_K3_FUSED_VERIFY remains the master gate (now meaning "fused
    verify when a matching tile plan exists").  Invalid tile values
    disable fused verify with a warning.  The bucketing gate
    (VLLM_K3_BUCKETED_DECODE) is untouched.
  * builder: a second verify-plan family keyed by batch bucket with
    max_total_q = batch * 8 (b12x maps two 4-row query tiles per
    request — see patch_b12x_verify_tile.py for why a literal 8-row CTA
    tile is impossible).  Auto-selection happens per step in build():
    query_len == 4 -> the 4-row family; 5 <= query_len <= 8 -> the 8-row
    family; anything else stays on the flattened path.  The 8-row family
    is dcp=1-only (the strided scatter/gather is not wired through the
    DCP head-gather path); the 4-row family keeps its DCP support.
  * layout: the 8-row family needs a per-request STRIDED row layout —
    request r occupies rows [8r, 8r + query_len) of preallocated padded
    buffers, with the span tail as padding OUTSIDE the request's
    cu_seqlens entry.  The kernel's existing ragged-tail mechanism flags
    padded rows query_valid == 0: their math is skipped and BOTH their
    output and LSE stores are skipped (verified in b12x _math.py), so
    padded rows are never computed, never written, and never read (the
    impl gathers only the first query_len rows of each span).  The
    per-row visibility (query_cache_seqlens) is scattered into the same
    strided layout with padded rows clamped to zero.
  * impl (forward_mqa): the strided path scatters the compact q rows
    into the padded buffer (one strided view copy), binds the padded
    buffers plus the builder's 8-stride query start locations, and after
    the run gathers the real rows of each span back into the compact
    output/LSE layout the verify consumer expects.  The 4-row path
    (query_len == rows per request) remains copy-free and byte-identical.

The KV-sharing win at nst=6: the 7 flattened full-prefix sweeps collapse
to TWO shared-KV sweeps (one per 4-row tile) — not the theoretical one,
because two CTAs cannot share smem — plus two small row-count-proportional
scatter/gather copies (~0.001x of the KV traffic at 32K context).

PREREQUISITE: v4plus-batch2 (B1b + the bucketing fix) must be applied to
the vllm tree — this script FAILS LOUD if its markers are absent.  The
b12x side (patch_b12x_verify_tile.py) must also be applied for the 8-row
family to actually fuse; without it an 8-row verify plan degrades to
query_tile=1 (correct results, no fusion) — a soft NOTE checks for it.

Idempotent: every hunk is skipped when its marker is already present.
A missing anchor prints a NOTE and skips that hunk; only a missing
prerequisite, file-not-found, or a broken post-patch compile exits
non-zero.
"""

from __future__ import annotations

import os
import py_compile
import sys

SCRIPT_NAME = "patch_vllm_verify_tile"
TAG = "# V4PLUS-B4 (vllm 8-row fused verify plans)"

VLLM_ROOT = os.environ.get("VLLM_ROOT", "/opt/kimi-k3/vllm/vllm")
ENVS = os.path.join(VLLM_ROOT, "envs.py")
B12X_MLA = os.path.join(VLLM_ROOT, "v1", "attention", "backends", "mla", "b12x_mla.py")

B12X_ROOT = os.environ.get("B12X_ROOT")
if not B12X_ROOT:
    try:
        import b12x  # noqa: F401

        B12X_ROOT = os.path.dirname(b12x.__file__)
    except Exception:
        B12X_ROOT = "/opt/kimi-k3/b12x/b12x"


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
    """Fail loud unless the v4plus-batch2 (B1b + fixes) surface is present."""
    ok = True
    try:
        with open(ENVS) as f:
            envs_src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {ENVS} not found", file=sys.stderr)
        return False
    for marker in (
        "    VLLM_K3_FUSED_VERIFY: bool = True\n",
        "    VLLM_K3_BUCKETED_DECODE: bool = False\n",
    ):
        if marker not in envs_src:
            print(
                f"[{SCRIPT_NAME}] ERROR: {ENVS} lacks the v4plus-batch2 "
                f"marker {marker.strip()!r}. Apply mods/v4plus-batch2 "
                "first — this mod builds on its post-state.",
                file=sys.stderr,
            )
            ok = False
    try:
        with open(B12X_MLA) as f:
            mla_src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {B12X_MLA} not found", file=sys.stderr)
        return False
    for marker in (
        "_dense_mla_verify_plans",
        "VLLM_K3_BUCKETED_DECODE",
    ):
        if marker not in mla_src:
            print(
                f"[{SCRIPT_NAME}] ERROR: {B12X_MLA} lacks the "
                f"v4plus-batch2 marker {marker!r}. Apply "
                "mods/v4plus-batch2 first.",
                file=sys.stderr,
            )
            ok = False
    if "tiled_verify = (" not in mla_src and "tiled_verify_rows = 4" not in mla_src:
        # The second form is this mod's own post-state (idempotent re-run).
        print(
            f"[{SCRIPT_NAME}] ERROR: {B12X_MLA} lacks the v4plus-batch2 "
            "tiled-verify gate (nor this mod's post-state). Apply "
            "mods/v4plus-batch2 first.",
            file=sys.stderr,
        )
        ok = False
    return ok


def note_b12x_side() -> None:
    """Soft NOTE when the b12x companion patch is not yet applied."""
    scratch = os.path.join(B12X_ROOT, "attention", "dense_mla", "_scratch.py")
    try:
        with open(scratch) as f:
            src = f.read()
    except FileNotFoundError:
        print(
            f"[{SCRIPT_NAME}] NOTE: cannot read {scratch}; skipped the "
            "b12x-side companion check."
        )
        return
    if "tiles_per_request" not in src:
        print(
            f"[{SCRIPT_NAME}] *** LOUD NOTE: the b12x tree at {B12X_ROOT} "
            "does not carry patch_b12x_verify_tile.py's tiles_per_request "
            "surface. Without it, 8-row verify plans resolve to "
            "query_tile=1 (correct results, NO fusion — the KV-sharing "
            "win is lost). Run patch_b12x_verify_tile.py too. ***"
        )
    else:
        print(
            f"[{SCRIPT_NAME}] b12x companion check: tiles_per_request "
            "surface present."
        )


# ---------------------------------------------------------------------------
# envs.py — VLLM_K3_FUSED_TILE (default 8)
# ---------------------------------------------------------------------------

E_TILE_FIELD_ANCHOR = "    VLLM_K3_BUCKETED_DECODE: bool = False\n"
E_TILE_FIELD_REPLACEMENT = (
    "    VLLM_K3_BUCKETED_DECODE: bool = False\n"
    "    VLLM_K3_FUSED_TILE: int = 8\n"
)
E_TILE_FIELD_PRESENT = "    VLLM_K3_FUSED_TILE: int = 8"

E_TILE_LAMBDA_ANCHOR = (
    '    "VLLM_K3_BUCKETED_DECODE": lambda: bool(\n'
    '        int(os.getenv("VLLM_K3_BUCKETED_DECODE", "0"))\n'
    "    ),\n"
)
E_TILE_LAMBDA_REPLACEMENT = (
    '    "VLLM_K3_BUCKETED_DECODE": lambda: bool(\n'
    '        int(os.getenv("VLLM_K3_BUCKETED_DECODE", "0"))\n'
    "    ),\n"
    "    # V4PLUS-B4: fused verify tile rows per request. 8 = two 4-row\n"
    "    # query tiles per request (nst=4..7; nst=3 keeps the 4-row\n"
    "    # single-tile plans); 4 = nst=3 single-tile plans only; 0\n"
    "    # disables fused verify entirely. Invalid values disable fused\n"
    "    # verify with a warning at builder init.\n"
    '    "VLLM_K3_FUSED_TILE": lambda: int(os.getenv("VLLM_K3_FUSED_TILE", "8")),\n'
)
E_TILE_LAMBDA_PRESENT = '"VLLM_K3_FUSED_TILE": lambda: int('


# ---------------------------------------------------------------------------
# b12x_mla.py — metadata fields
# ---------------------------------------------------------------------------

M_FIELDS_ANCHOR = (
    "    dense_mla_verify_block_table: torch.Tensor | None = None\n"
    "    dense_mla_query_cache_seq_lens: torch.Tensor | None = None\n"
    "    dense_mla_dcp_world_size: int = 1\n"
)
M_FIELDS_REPLACEMENT = (
    "    dense_mla_verify_block_table: torch.Tensor | None = None\n"
    "    dense_mla_query_cache_seq_lens: torch.Tensor | None = None\n"
    "    dense_mla_verify_rows_per_request: int = 0\n"
    "    dense_mla_verify_q: torch.Tensor | None = None\n"
    "    dense_mla_verify_output: torch.Tensor | None = None\n"
    "    dense_mla_verify_query_start_loc: torch.Tensor | None = None\n"
    "    dense_mla_dcp_world_size: int = 1\n"
)
M_FIELDS_PRESENT = "dense_mla_verify_rows_per_request: int = 0"


# ---------------------------------------------------------------------------
# b12x_mla.py — builder __init__: gate generalization + 8-row family
# ---------------------------------------------------------------------------

B_GATE_ANCHOR = (
    "        num_spec_tokens = int(\n"
    "            getattr(vllm_config, \"num_speculative_tokens\", 0) or 0\n"
    "        )\n"
    "        fused_verify = bool(\n"
    "            envs.VLLM_K3_FUSED_VERIFY\n"
    "            and _planned_kv_dtype(vllm_config) == torch.float8_e4m3fn\n"
    "            and num_spec_tokens == 3\n"
    "        )\n"
    "        if envs.VLLM_K3_FUSED_VERIFY and not fused_verify:\n"
    "            logger.info_once(\n"
    "                \"Kimi-K3 fused DSpark verification is configured ON but \"\n"
    "                \"inactive for this launch (requires fp8 KV cache and \"\n"
    "                \"num_speculative_tokens=3; got nst=%d, kv_dtype=%s). \"\n"
    "                \"Serving continues on the flattened verify path.\",\n"
    "                num_spec_tokens,\n"
    "                _planned_kv_dtype(vllm_config),\n"
    "            )\n"
)
B_GATE_REPLACEMENT = (
    "        num_spec_tokens = int(\n"
    "            getattr(vllm_config, \"num_speculative_tokens\", 0) or 0\n"
    "        )\n"
    "        # V4PLUS-B4: VLLM_K3_FUSED_TILE selects the fused verify tile\n"
    "        # rows per request: 8 = two 4-row query tiles per request\n"
    "        # (nst=4..7), 4 = the single-tile 4-row plans (nst=3 only),\n"
    "        # 0 = fused verify disabled. Invalid values disable fused\n"
    "        # verify with a warning.\n"
    "        fused_tile = int(envs.VLLM_K3_FUSED_TILE)\n"
    "        if fused_tile not in (0, 4, 8):\n"
    "            logger.warning_once(\n"
    "                \"VLLM_K3_FUSED_TILE=%d is not one of (0, 4, 8); fused \"\n"
    "                \"verify is disabled for this launch.\",\n"
    "                fused_tile,\n"
    "            )\n"
    "            fused_tile = 0\n"
    "        fused_verify = bool(\n"
    "            envs.VLLM_K3_FUSED_VERIFY\n"
    "            and _planned_kv_dtype(vllm_config) == torch.float8_e4m3fn\n"
    "            and 3 <= num_spec_tokens <= 7\n"
    "        )\n"
    "        if envs.VLLM_K3_FUSED_VERIFY and not fused_verify:\n"
    "            logger.info_once(\n"
    "                \"Kimi-K3 fused DSpark verification is configured ON but \"\n"
    "                \"inactive for this launch (requires fp8 KV cache and \"\n"
    "                \"num_speculative_tokens in 3..7; got nst=%d, kv_dtype=%s). \"\n"
    "                \"Serving continues on the flattened verify path.\",\n"
    "                num_spec_tokens,\n"
    "                _planned_kv_dtype(vllm_config),\n"
    "            )\n"
    "        elif fused_verify and fused_tile == 0:\n"
    "            logger.info_once(\n"
    "                \"Kimi-K3 fused DSpark verification is ON and nst=%d is \"\n"
    "                \"supported, but VLLM_K3_FUSED_TILE=0 disables the fused \"\n"
    "                \"plans; serving continues on the flattened verify path.\",\n"
    "                num_spec_tokens,\n"
    "            )\n"
)
B_GATE_PRESENT = "fused_tile = int(envs.VLLM_K3_FUSED_TILE)"

B_FAMILY4_ANCHOR = "        if fused_verify and max_verify_batch >= 1:\n"
B_FAMILY4_REPLACEMENT = (
    "        if (\n"
    "            fused_verify\n"
    "            and fused_tile in (4, 8)\n"
    "            and num_spec_tokens == 3\n"
    "            and max_verify_batch >= 1\n"
    "        ):\n"
)
B_FAMILY4_PRESENT = (
    "and fused_tile in (4, 8)\n            and num_spec_tokens == 3"
)

B_FAMILY8_ANCHOR = (
    "                for batch in _dense_mla_plan_row_caps(max_verify_batch)\n"
    "            }\n"
    "        all_plans = [\n"
    "            *self._dense_mla_plans.values(),\n"
    "            *self._dense_mla_verify_plans.values(),\n"
    "        ]\n"
)
B_FAMILY8_REPLACEMENT = (
    "                for batch in _dense_mla_plan_row_caps(max_verify_batch)\n"
    "            }\n"
    "        # V4PLUS-B4: the 8-row fused verify family (nst=4..7). Each\n"
    "        # request spans 8 query rows as TWO of the same 4-row query\n"
    "        # tiles (b12x tiles_per_request=2): a request's KV chunks are\n"
    "        # read twice per sweep instead of once per row (nst=6 flat:\n"
    "        # 7x), while the proven 4-row tile kernel, smem layout, and\n"
    "        # thread geometry are untouched. Padded tail rows (nst+1 < 8)\n"
    "        # live outside the request's cu_seqlens span, so the kernel\n"
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
    "            logger.warning_once(\n"
    "                \"Kimi-K3 fused 8-row (two-tile) DSpark verification is \"\n"
    "                \"ENABLED (nst=%d, bucketed verify plans, two 4-row query \"\n"
    "                \"tiles per request). This path is NEW (v4plus-batch4) \"\n"
    "                \"and was never qualified in the fork: the first boot \"\n"
    "                \"and the 64K quality gate must pass before this path \"\n"
    "                \"is trusted.\",\n"
    "                num_spec_tokens,\n"
    "            )\n"
    "            self._dense_mla_verify8_plans = {\n"
    "                batch: _create_dense_mla_plan(\n"
    "                    vllm_config,\n"
    "                    device,\n"
    "                    page_size=self.page_size,\n"
    "                    num_q_heads=self._kernel_heads,\n"
    "                    max_total_q=batch * 8,\n"
    "                    max_batch=batch,\n"
    "                    mode=\"verify\",\n"
    "                    uses_query_cache_seqlens=True,\n"
    "                    dcp_size=self.dcp_world_size,\n"
    "                    max_cache_tokens=max_cache_tokens,\n"
    "                )\n"
    "                for batch in _dense_mla_plan_row_caps(max_verify8_batch)\n"
    "            }\n"
    "            self._dense_mla_verify_q = torch.zeros(\n"
    "                (\n"
    "                    max_verify8_batch * 8,\n"
    "                    self._kernel_heads,\n"
    "                    _K3_ABSORBED_HEAD_DIM,\n"
    "                ),\n"
    "                # V4PLUS-B4 fix: the 8-row fused tile is E4M3-only (see\n"
    "                # _query_tile) and this family is created solely under fp8\n"
    "                # KV, where the b12x MLA path quantizes queries to\n"
    "                # float8_e4m3fn before the kernel. The strided scatter\n"
    "                # buffer must match the live query dtype or the fail-loud\n"
    "                # storage check rejects every step.\n"
    "                dtype=torch.float8_e4m3fn,\n"
    "                device=device,\n"
    "            )\n"
    "            self._dense_mla_verify_output = torch.empty(\n"
    "                (\n"
    "                    max_verify8_batch * 8,\n"
    "                    self._kernel_heads,\n"
    "                    _K3_KV_LORA_RANK,\n"
    "                ),\n"
    "                dtype=torch.bfloat16,\n"
    "                device=device,\n"
    "            )\n"
    "            self._dense_mla_verify_query_start_loc = torch.arange(\n"
    "                0,\n"
    "                max_verify8_batch * 8 + 1,\n"
    "                8,\n"
    "                dtype=torch.int32,\n"
    "                device=device,\n"
    "            )\n"
    "            self._dense_mla_verify_row_lens = torch.zeros(\n"
    "                max_verify8_batch * 8,\n"
    "                dtype=torch.int32,\n"
    "                device=device,\n"
    "            )\n"
    "        all_plans = [\n"
    "            *self._dense_mla_plans.values(),\n"
    "            *self._dense_mla_verify_plans.values(),\n"
    "            *self._dense_mla_verify8_plans.values(),\n"
    "        ]\n"
)
B_FAMILY8_PRESENT = "self._dense_mla_verify8_plans: dict[int, Any] = {}"


# ---------------------------------------------------------------------------
# b12x_mla.py — build(): tiled family auto-selection + the 8-row branch
# ---------------------------------------------------------------------------

V_GATE_ANCHOR = (
    "        verify_plans = getattr(self, \"_dense_mla_verify_plans\", {})\n"
    "        # V4PLUS-B2 (#565 + d461572be): a causal 4-row DSpark verify block\n"
    "        # (nst=3) runs the fused tiled-verify plan — one 4-row query tile\n"
    "        # per request with per-row visibility — instead of flattening to\n"
    "        # independent single-token sweeps.\n"
    "        tiled_verify = (\n"
    "            metadata.causal\n"
    "            and query_len == 4\n"
    "            and bool(verify_plans)\n"
    "            and metadata.num_decodes <= max(verify_plans)\n"
    "        )\n"
    "        if tiled_verify:\n"
)
V_GATE_REPLACEMENT = (
    "        verify_plans = getattr(self, \"_dense_mla_verify_plans\", {})\n"
    "        verify8_plans = getattr(self, \"_dense_mla_verify8_plans\", {})\n"
    "        # V4PLUS-B2 (#565 + d461572be): a causal 4-row DSpark verify block\n"
    "        # (nst=3) runs the fused tiled-verify plan — one 4-row query tile\n"
    "        # per request with per-row visibility — instead of flattening to\n"
    "        # independent single-token sweeps.\n"
    "        # V4PLUS-B4: a causal 5..8-row verify block (nst=4..7) runs the\n"
    "        # 8-row fused family — two 4-row query tiles per request —\n"
    "        # with auto-selection by nst: the family whose tile matches\n"
    "        # the live query length wins; anything else stays flat.\n"
    "        tiled_verify_rows = 0\n"
    "        if (\n"
    "            metadata.causal\n"
    "            and query_len == 4\n"
    "            and bool(verify_plans)\n"
    "            and metadata.num_decodes <= max(verify_plans)\n"
    "        ):\n"
    "            tiled_verify_rows = 4\n"
    "        elif (\n"
    "            metadata.causal\n"
    "            and 5 <= query_len <= 8\n"
    "            and bool(verify8_plans)\n"
    "            and metadata.num_decodes <= max(verify8_plans)\n"
    "        ):\n"
    "            tiled_verify_rows = 8\n"
    "        if tiled_verify_rows == 4:\n"
)
V_GATE_PRESENT = "tiled_verify_rows = 4"

V_BRANCH8_ANCHOR = (
    "            metadata.dense_mla_plan = _select_dense_mla_plan(\n"
    "                verify_plans, metadata.num_decodes\n"
    "            )\n"
    "            metadata.dense_mla_verify_block_table = verify_table\n"
    "            metadata.dense_mla_query_cache_seq_lens = flat_lens\n"
    "            return metadata\n"
)
V_BRANCH8_REPLACEMENT = (
    "            metadata.dense_mla_plan = _select_dense_mla_plan(\n"
    "                verify_plans, metadata.num_decodes\n"
    "            )\n"
    "            metadata.dense_mla_verify_block_table = verify_table\n"
    "            metadata.dense_mla_query_cache_seq_lens = flat_lens\n"
    "            return metadata\n"
    "\n"
    "        if tiled_verify_rows == 8:\n"
    "            # V4PLUS-B4: strided 8-row spans. Each request occupies\n"
    "            # rows [8r, 8r + query_len) of the padded verify buffers;\n"
    "            # the tail rows of each span are padding and live OUTSIDE\n"
    "            # the request's cu_seqlens span, so the kernel flags them\n"
    "            # invalid and never writes their output or LSE. The\n"
    "            # per-row visibility is scattered into the same strided\n"
    "            # layout, with padded rows clamped to zero.\n"
    "            verify_table = self._dense_mla_flat_block_table[: metadata.num_decodes]\n"
    "            source_width = min(\n"
    "                int(source_table.shape[1]),\n"
    "                int(verify_table.shape[1]),\n"
    "            )\n"
    "            verify_table[:, :source_width].copy_(source_table[:, :source_width])\n"
    "            metadata.dense_mla_plan = _select_dense_mla_plan(\n"
    "                verify8_plans, metadata.num_decodes\n"
    "            )\n"
    "            metadata.dense_mla_verify_block_table = verify_table\n"
    "            row_lens = self._dense_mla_verify_row_lens[\n"
    "                : metadata.num_decodes * 8\n"
    "            ]\n"
    "            row_lens.view(metadata.num_decodes, 8)[:, :query_len].copy_(\n"
    "                flat_lens.view(metadata.num_decodes, query_len)\n"
    "            )\n"
    "            row_lens.view(metadata.num_decodes, 8)[:, query_len:].zero_()\n"
    "            metadata.dense_mla_query_cache_seq_lens = row_lens\n"
    "            metadata.dense_mla_verify_rows_per_request = 8\n"
    "            metadata.dense_mla_verify_q = self._dense_mla_verify_q\n"
    "            metadata.dense_mla_verify_output = self._dense_mla_verify_output\n"
    "            metadata.dense_mla_verify_query_start_loc = (\n"
    "                self._dense_mla_verify_query_start_loc[\n"
    "                    : metadata.num_decodes + 1\n"
    "                ]\n"
    "            )\n"
    "            return metadata\n"
)
V_BRANCH8_PRESENT = "if tiled_verify_rows == 8:"


# ---------------------------------------------------------------------------
# b12x_mla.py — forward_mqa(): strided capture, scatter, gather
# ---------------------------------------------------------------------------

V_FWD_CAPTURE_ANCHOR = (
    "        verify_block_table = getattr(\n"
    "            attn_metadata,\n"
    "            \"dense_mla_verify_block_table\",\n"
    "            None,\n"
    "        )\n"
    "        if verify_block_table is not None:\n"
    "            block_table = verify_block_table\n"
)
V_FWD_CAPTURE_REPLACEMENT = (
    "        verify_block_table = getattr(\n"
    "            attn_metadata,\n"
    "            \"dense_mla_verify_block_table\",\n"
    "            None,\n"
    "        )\n"
    "        verify_rows = int(\n"
    "            getattr(attn_metadata, \"dense_mla_verify_rows_per_request\", 0) or 0\n"
    "        )\n"
    "        strided_verify = verify_rows == 8\n"
    "        if verify_block_table is not None:\n"
    "            block_table = verify_block_table\n"
    "            if strided_verify:\n"
    "                # V4PLUS-B4: the strided fused-verify layout addresses\n"
    "                # rows by 8-row request span, so the per-request query\n"
    "                # start locations come from the builder's strided\n"
    "                # table, not the compact query_start_loc.\n"
    "                strided_cu = getattr(\n"
    "                    attn_metadata,\n"
    "                    \"dense_mla_verify_query_start_loc\",\n"
    "                    None,\n"
    "                )\n"
    "                if strided_cu is None:\n"
    "                    raise RuntimeError(\n"
    "                        \"B12X_MLA strided fused verify metadata is missing \"\n"
    "                        \"its query start locations.\"\n"
    "                    )\n"
    "                query_start_loc = strided_cu\n"
)
V_FWD_CAPTURE_PRESENT = "strided_verify = verify_rows == 8"

V_FWD_SCATTER_ANCHOR = (
    "        if kernel_heads == effective_heads and metadata_dcp_world_size == 1:\n"
    "            output = torch.empty(\n"
    "                (total_q, effective_heads, self.kv_lora_rank),\n"
    "                dtype=torch.bfloat16,\n"
    "                device=q.device,\n"
    "            )\n"
    "        else:\n"
)
V_FWD_SCATTER_REPLACEMENT = (
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
    "            strided_out = getattr(\n"
    "                attn_metadata, \"dense_mla_verify_output\", None\n"
    "            )\n"
    "            if strided_q is None or strided_out is None:\n"
    "                raise RuntimeError(\n"
    "                    \"B12X_MLA strided fused verify metadata is missing \"\n"
    "                    \"its caller-owned buffers.\"\n"
    "                )\n"
    "            if (\n"
    "                int(strided_q.shape[0]) < batch * 8\n"
    "                or int(strided_out.shape[0]) < batch * 8\n"
    "            ):\n"
    "                raise ValueError(\n"
    "                    \"B12X_MLA strided fused verify capacity is smaller \"\n"
    "                    f\"than the decode batch: \"\n"
    "                    f\"q={strided_q.shape[0]}, output={strided_out.shape[0]}, \"\n"
    "                    f\"required={batch * 8}.\"\n"
    "                )\n"
    "            if strided_q.dtype != q.dtype:\n"
    "                raise TypeError(\n"
    "                    \"B12X_MLA strided fused verify storage does not \"\n"
    "                    f\"match the live query: buffer={strided_q.dtype}, \"\n"
    "                    f\"query={q.dtype}.\"\n"
    "                )\n"
    "            query_len_rows = total_q // batch\n"
    "            strided_q_view = strided_q[: batch * 8]\n"
    "            strided_q_view.view(\n"
    "                batch, 8, kernel_heads, _K3_ABSORBED_HEAD_DIM\n"
    "            )[:, :query_len_rows, :effective_heads].copy_(\n"
    "                q.view(\n"
    "                    batch,\n"
    "                    query_len_rows,\n"
    "                    effective_heads,\n"
    "                    _K3_ABSORBED_HEAD_DIM,\n"
    "                )\n"
    "            )\n"
    "            q = strided_q_view\n"
    "            strided_out_view = strided_out[: batch * 8]\n"
    "            output = strided_out_view\n"
    "        elif kernel_heads == effective_heads and metadata_dcp_world_size == 1:\n"
    "            output = torch.empty(\n"
    "                (total_q, effective_heads, self.kv_lora_rank),\n"
    "                dtype=torch.bfloat16,\n"
    "                device=q.device,\n"
    "            )\n"
    "        else:\n"
)
V_FWD_SCATTER_PRESENT = "strided_q_view = strided_q[: batch * 8]"

V_FWD_GATHER_ANCHOR = (
    "        output, lse = self._dense_mla.run(binding=binding)\n"
    "        output = output[:, :effective_heads]\n"
    "        lse = lse[:, :effective_heads]\n"
    "        if dcp_group is None:\n"
    "            return output, lse\n"
)
V_FWD_GATHER_REPLACEMENT = (
    "        output, lse = self._dense_mla.run(binding=binding)\n"
    "        if strided_verify:\n"
    "            # V4PLUS-B4: gather the per-request strided rows back into\n"
    "            # the compact layout the verify consumer expects: each\n"
    "            # 8-row span holds query_len real rows; the padded tail\n"
    "            # rows were never written (query_valid == 0 skips both\n"
    "            # the output and the LSE stores).\n"
    "            query_len_rows = total_q // batch\n"
    "            compact_output = torch.empty(\n"
    "                (total_q, effective_heads, self.kv_lora_rank),\n"
    "                dtype=torch.bfloat16,\n"
    "                device=output.device,\n"
    "            )\n"
    "            compact_output.view(\n"
    "                batch, query_len_rows, effective_heads, self.kv_lora_rank\n"
    "            ).copy_(\n"
    "                strided_out_view.view(\n"
    "                    batch, 8, kernel_heads, self.kv_lora_rank\n"
    "                )[:, :query_len_rows, :effective_heads]\n"
    "            )\n"
    "            output = compact_output\n"
    "            compact_lse = torch.empty(\n"
    "                (total_q, effective_heads),\n"
    "                dtype=torch.float32,\n"
    "                device=lse.device,\n"
    "            )\n"
    "            compact_lse.view(batch, query_len_rows, effective_heads).copy_(\n"
    "                lse.view(batch, 8, kernel_heads)[\n"
    "                    :, :query_len_rows, :effective_heads\n"
    "                ]\n"
    "            )\n"
    "            lse = compact_lse\n"
    "        else:\n"
    "            output = output[:, :effective_heads]\n"
    "            lse = lse[:, :effective_heads]\n"
    "        if dcp_group is None:\n"
    "            return output, lse\n"
)
V_FWD_GATHER_PRESENT = "gather the per-request strided rows back into"


def main() -> int:
    print(f"[{SCRIPT_NAME}] {TAG}")
    print(f"[{SCRIPT_NAME}] VLLM_ROOT={VLLM_ROOT}")
    if not check_prerequisites():
        print(
            f"[{SCRIPT_NAME}] PREREQUISITE FAILED: v4plus-batch2 is not "
            "applied to this vllm tree. Apply mods/v4plus-batch2 first; "
            "refusing to patch.",
            file=sys.stderr,
        )
        return 1
    note_b12x_side()

    ok = True
    ok &= apply_hunks(
        ENVS,
        [
            (
                "VLLM_K3_FUSED_TILE field (default 8)",
                E_TILE_FIELD_ANCHOR,
                E_TILE_FIELD_REPLACEMENT,
                E_TILE_FIELD_PRESENT,
            ),
            (
                "VLLM_K3_FUSED_TILE lambda (default 8)",
                E_TILE_LAMBDA_ANCHOR,
                E_TILE_LAMBDA_REPLACEMENT,
                E_TILE_LAMBDA_PRESENT,
            ),
        ],
    )
    ok &= apply_hunks(
        B12X_MLA,
        [
            ("metadata verify-tile fields", M_FIELDS_ANCHOR, M_FIELDS_REPLACEMENT, M_FIELDS_PRESENT),
            ("fused gate generalization", B_GATE_ANCHOR, B_GATE_REPLACEMENT, B_GATE_PRESENT),
            ("4-row family gate", B_FAMILY4_ANCHOR, B_FAMILY4_REPLACEMENT, B_FAMILY4_PRESENT),
            ("8-row verify family + buffers", B_FAMILY8_ANCHOR, B_FAMILY8_REPLACEMENT, B_FAMILY8_PRESENT),
            ("build() tiled auto-selection", V_GATE_ANCHOR, V_GATE_REPLACEMENT, V_GATE_PRESENT),
            ("build() 8-row strided branch", V_BRANCH8_ANCHOR, V_BRANCH8_REPLACEMENT, V_BRANCH8_PRESENT),
            ("forward_mqa strided capture", V_FWD_CAPTURE_ANCHOR, V_FWD_CAPTURE_REPLACEMENT, V_FWD_CAPTURE_PRESENT),
            ("forward_mqa strided scatter", V_FWD_SCATTER_ANCHOR, V_FWD_SCATTER_REPLACEMENT, V_FWD_SCATTER_PRESENT),
            ("forward_mqa strided gather", V_FWD_GATHER_ANCHOR, V_FWD_GATHER_REPLACEMENT, V_FWD_GATHER_PRESENT),
        ],
    )

    if ok:
        print(
            f"[{SCRIPT_NAME}] NOTE: the 8-row fused family is dcp=1-only; "
            "the 4-row family keeps its DCP support. At nst=3 nothing "
            "changes (same 4-row plans, same copy-free path)."
        )
        print(
            f"[{SCRIPT_NAME}] NOTE: watch for the 'Kimi-K3 fused 8-row "
            "(two-tile) DSpark verification is ENABLED' warning at builder "
            "init — it fires only when the 8-row family is actually "
            "created (VLLM_K3_FUSED_TILE=8, nst=4..7, fp8 KV, dcp=1)."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
