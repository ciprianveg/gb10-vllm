#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V4PLUS Batch 4 (b12x side) — 8-row fused verify requests (nst=4..7).

Extends the #271 fused DSpark verification (v4plus-batch2 / B1a) from the
hardcoded 4-row request (nst=3) to an 8-row request span covering
nst=4..7 (practically nst=6), with auto-selection by nst on the vllm side.

DESIGN DECISION (parameterize the REQUEST SPAN, not the CTA tile):
a literal 8-row query tile in one CTA is PHYSICALLY IMPOSSIBLE on this
kernel: MATH_WARPS_PER_QUERY=4 gives block_threads = 4*8*32 + 32 = 1056 >
the 1024-thread CUDA limit, and the fp8 smem footprint would be ~126 KiB
(q_stage alone 8*8*592 = 37,888 B on top of the 75,776-B double-buffered
KV stages) against the ~101 KB device opt-in budget that already pins the
4-row tile to E4M3-only.  Instead, an 8-row verify REQUEST is processed
by TWO of the proven 4-row query tiles: a new integer kernel parameter
``tiles_per_request`` maps ``query_tile_index // tiles_per_request`` to
the request, so both tiles of a request sweep the same KV pages — the KV
traffic per verify step drops from once per row (nst=6 flat: 7x) to twice
per request.  The math/IO/smem/thread geometry of the 4-row tile is
untouched, and a ``const_expr(tiles_per_request == 1)`` branch keeps the
existing 4-row path's generated code byte-identical.

What was 4-specific and how it is handled:
  * _scratch._query_tile's ``max_total_q == max_batch * 4`` rule  -> an
    8-row branch (``max_total_q == max_batch * 8``, E4M3, no window)
    selects the SAME query_tile=4 plus tiles_per_request=2;
  * the kernel's request mapping ``request = query_tile_index`` (one
    tile per request) -> the const_expr divisor above;
  * the binding completeness check ``q.shape[0] == batch * query_tile``
    -> ``batch * query_tile * tiles_per_request``;
  * the 4-branch math-group dispatch / barrier ids 2-5 in the kernel
    body are tile-4-specific and stay EXACTLY as they are (query_tile
    remains 4 in every tiled configuration).

Padded-row semantics (nst+1 < 8): the vllm side places each request's
real rows at [8r, 8r + query_len) of the padded verify buffers and leaves
the span tail OUTSIDE the request's cu_seqlens entry, so the kernel's
existing ragged-tail mechanism flags those rows ``query_valid == 0``:
their math group skips accumulation and write_partial_or_final skips BOTH
the output and the LSE stores (verified in _math.py).  Padded rows are
never computed, never written, and never read (the vllm side gathers only
the first query_len rows of each span).  stage_absorbed_query does stage
padded-row q bytes into smem; the values are garbage but never enter any
arithmetic because the entire math body is behind the query_valid guard.

PREREQUISITE: v4plus-batch2 (B1a) must be applied first — the fused-verify
kernel surface (uses_query_cache_seqlens) and compile-spec version 5 must
already exist.  This script FAILS LOUD if they are absent.

Compile-spec: version bumps 5 -> 6 and gains a ``tiles_per_request`` key
field (plus the _signature tuple entry), so no stale JIT artifact can
serve 8-row requests with the one-tile-per-request mapping.  The version
bump re-JITs the 4-row kernels too — expected first-boot cost.

Excluded: dense_mla/_policy.py (the standalone planner's _query_tile —
not read by the vllm plan path, which uses _scratch._query_tile), the
reference (span-based loop already handles 8-row spans, gap rows, and
per-row query_cache_seqlens unchanged), and _layout.py (query_tile stays
4; the (1, 2, 4) allowlist is untouched).

Idempotent: every hunk is skipped when its marker is already present.
A missing anchor prints a NOTE and skips that hunk; only a missing
prerequisite, file-not-found, or a broken post-patch compile exits
non-zero.
"""

from __future__ import annotations

import os
import py_compile
import sys

SCRIPT_NAME = "patch_b12x_verify_tile"
TAG = "# V4PLUS-B4 (b12x 8-row fused verify requests)"

B12X_ROOT = os.environ.get("B12X_ROOT")
if not B12X_ROOT:
    try:
        import b12x  # noqa: F401

        B12X_ROOT = os.path.dirname(b12x.__file__)
    except Exception:
        B12X_ROOT = "/opt/kimi-k3/b12x/b12x"

DENSE_MLA = os.path.join(B12X_ROOT, "attention", "dense_mla")
SCRATCH = os.path.join(DENSE_MLA, "_scratch.py")
FORWARD = os.path.join(DENSE_MLA, "_forward.py")
KERNEL = os.path.join(DENSE_MLA, "_kernel.py")


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
    """Fail loud unless the v4plus-batch2 (B1a) surface is present."""
    ok = True
    try:
        with open(SCRATCH) as f:
            scratch_src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {SCRATCH} not found", file=sys.stderr)
        return False
    if "    uses_query_cache_seqlens: bool = False\n" not in scratch_src:
        print(
            f"[{SCRIPT_NAME}] ERROR: {SCRATCH} lacks the B1a "
            "(v4plus-batch2) uses_query_cache_seqlens Caps field. "
            "Apply mods/v4plus-batch2 first — this mod builds on its "
            "post-state.",
            file=sys.stderr,
        )
        ok = False
    try:
        with open(FORWARD) as f:
            forward_src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {FORWARD} not found", file=sys.stderr)
        return False
    if "        uses_query_cache_seqlens: bool,\n" not in forward_src:
        print(
            f"[{SCRIPT_NAME}] ERROR: {FORWARD} lacks the B1a "
            "(v4plus-batch2) uses_query_cache_seqlens kernel parameter. "
            "Apply mods/v4plus-batch2 first.",
            file=sys.stderr,
        )
        ok = False
    try:
        with open(KERNEL) as f:
            kernel_src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {KERNEL} not found", file=sys.stderr)
        return False
    if (
        '        "attention.dense_mla.forward",\n        5,\n' not in kernel_src
        # Idempotent re-runs see our own post-state: version 6 plus the
        # tiles_per_request key field.
        and 'key_field(\n            "tiles_per_request",\n            scratch.tiles_per_request,\n        ),'
        not in kernel_src
    ):
        print(
            f"[{SCRIPT_NAME}] ERROR: {KERNEL} does not carry the B1a "
            "(v4plus-batch2) compile-spec version 5 (nor this mod's "
            "version-6 post-state). Apply mods/v4plus-batch2 first.",
            file=sys.stderr,
        )
        ok = False
    return ok


# ---------------------------------------------------------------------------
# _scratch.py
# ---------------------------------------------------------------------------

# Tile rule: the 8-row verify request rides the same 4-row query tile.
S_QT8_ANCHOR = (
    "def _query_tile(caps: Caps) -> int:\n"
    "    if (\n"
    '        caps.mode == "verify"\n'
    "        and caps.max_total_q == caps.max_batch * 4\n"
    "        and caps.window_size is None\n"
    "    ):\n"
    "        return 4 if caps.kv_dtype == _FP8 else 1\n"
)
S_QT8_REPLACEMENT = (
    "def _query_tile(caps: Caps) -> int:\n"
    "    if (\n"
    '        caps.mode == "verify"\n'
    "        and caps.max_total_q == caps.max_batch * 4\n"
    "        and caps.window_size is None\n"
    "    ):\n"
    "        return 4 if caps.kv_dtype == _FP8 else 1\n"
    "    if (\n"
    '        caps.mode == "verify"\n'
    "        and caps.max_total_q == caps.max_batch * 8\n"
    "        and caps.kv_dtype == _FP8\n"
    "        and caps.window_size is None\n"
    "    ):\n"
    "        # V4PLUS-B4: an 8-row verify request (nst=4..7) rides the\n"
    "        # same proven 4-row query tile as two consecutive tiles per\n"
    "        # request (tiles_per_request=2). A literal 8-row CTA tile is\n"
    "        # impossible: MATH_WARPS_PER_QUERY=4 gives 1056 block threads\n"
    "        # (> the 1024 CUDA limit) and ~126 KiB of smem against the\n"
    "        # device opt-in budget that already pins tile=4 to E4M3.\n"
    "        return 4\n"
)
S_QT8_PRESENT = "caps.max_total_q == caps.max_batch * 8"

# tiles_per_request derivation, next to the other caps helpers.
S_TPR_ANCHOR = "def _max_attended_tokens(caps: Caps) -> int:\n"
S_TPR_REPLACEMENT = (
    "def _tiles_per_request(caps: Caps) -> int:\n"
    "    \"\"\"Query tiles covering one verify request's row span.\n"
    "\n"
    "    V4PLUS-B4: an 8-row verify request spans two 4-row query tiles;\n"
    "    the kernel maps tile -> request with this divisor. Non-verify\n"
    "    plans and single-tile mappings keep 1.\n"
    "    \"\"\"\n"
    '    if caps.mode != "verify" or not caps.uses_query_cache_seqlens:\n'
    "        return 1\n"
    "    query_tile = _query_tile(caps)\n"
    "    if query_tile <= 1:\n"
    "        return 1\n"
    "    rows = caps.max_total_q // caps.max_batch\n"
    "    return max(1, rows // query_tile)\n"
    "\n"
    "\n"
    "def _max_attended_tokens(caps: Caps) -> int:\n"
)
S_TPR_PRESENT = "def _tiles_per_request(caps: Caps) -> int:"

# Caps validation: tiled verify requires per-query lengths and a row count
# that is an exact multiple of the tile (appended at the end of
# __post_init__, after int normalization).
S_CAPS_VALIDATE_ANCHOR = (
    "        object.__setattr__(\n"
    "            self,\n"
    '            "uses_query_cache_seqlens",\n'
    "            uses_query_cache_seqlens,\n"
    "        )\n"
    "        if self.window_size is not None:\n"
    '            object.__setattr__(self, "window_size", int(self.window_size))\n'
)
S_CAPS_VALIDATE_REPLACEMENT = (
    "        object.__setattr__(\n"
    "            self,\n"
    '            "uses_query_cache_seqlens",\n'
    "            uses_query_cache_seqlens,\n"
    "        )\n"
    "        if self.window_size is not None:\n"
    '            object.__setattr__(self, "window_size", int(self.window_size))\n'
    "        # V4PLUS-B4: tiled verify plans (query_tile > 1) map query\n"
    "        # tiles to requests positionally, which requires per-query\n"
    "        # cache lengths and a per-request row count that is an exact\n"
    "        # multiple of the tile. Without the lengths the kernel cannot\n"
    "        # resolve the request for tiles beyond the first; without the\n"
    "        # multiple, a request's rows would straddle tile boundaries.\n"
    '        if self.mode == "verify":\n'
    "            if _query_tile(self) > 1 and not uses_query_cache_seqlens:\n"
    "                raise ValueError(\n"
    '                    "tiled verify plans require per-query cache lengths"\n'
    "                )\n"
    "            if uses_query_cache_seqlens:\n"
    "                if self.max_total_q % self.max_batch:\n"
    "                    raise ValueError(\n"
    '                        "verify plans require a uniform per-request "\n'
    '                        "query-row count"\n'
    "                    )\n"
    "                rows = self.max_total_q // self.max_batch\n"
    "                tile = _query_tile(self)\n"
    "                if tile > 1 and rows % tile:\n"
    "                    raise ValueError(\n"
    '                        "tiled verify plans require the per-request row "\n'
    '                        "count to be a multiple of the query tile"\n'
    "                    )\n"
)
S_CAPS_VALIDATE_PRESENT = '"tiled verify plans require per-query cache lengths"'

# Scratch dataclass field.
S_SCRATCH_FIELD_ANCHOR = (
    "    query_tile: int\n"
    "    use_cuda_graph: bool\n"
    "    uses_query_cache_seqlens: bool\n"
)
S_SCRATCH_FIELD_REPLACEMENT = (
    "    query_tile: int\n"
    "    use_cuda_graph: bool\n"
    "    uses_query_cache_seqlens: bool\n"
    "    tiles_per_request: int\n"
)
S_SCRATCH_FIELD_PRESENT = (
    "    uses_query_cache_seqlens: bool\n    tiles_per_request: int"
)

# _materialize passes the derived divisor into Scratch.
S_MATERIALIZE_ANCHOR = (
    "        query_tile=query_tile,\n"
    "        use_cuda_graph=caps.use_cuda_graph,\n"
    "        uses_query_cache_seqlens=caps.uses_query_cache_seqlens,\n"
)
S_MATERIALIZE_REPLACEMENT = (
    "        query_tile=query_tile,\n"
    "        use_cuda_graph=caps.use_cuda_graph,\n"
    "        uses_query_cache_seqlens=caps.uses_query_cache_seqlens,\n"
    "        tiles_per_request=_tiles_per_request(caps),\n"
)
S_MATERIALIZE_PRESENT = "tiles_per_request=_tiles_per_request(caps),"

# Binding completeness check: one complete tile grid per request.
S_COMPLETE_ANCHOR = (
    "    if (\n"
    '        scratch.mode == "verify"\n'
    "        and scratch.query_tile > 1\n"
    "        and int(q.shape[0]) != batch * scratch.query_tile\n"
    "    ):\n"
    "        raise ValueError(\n"
    '            "tiled verify plan requires one complete query tile per request"\n'
    "        )\n"
)
S_COMPLETE_REPLACEMENT = (
    "    if (\n"
    '        scratch.mode == "verify"\n'
    "        and scratch.query_tile > 1\n"
    "        and int(q.shape[0])\n"
    "        != batch * scratch.query_tile * scratch.tiles_per_request\n"
    "    ):\n"
    "        raise ValueError(\n"
    '            "tiled verify plan requires one complete query tile per request"\n'
    "        )\n"
)
S_COMPLETE_PRESENT = "batch * scratch.query_tile * scratch.tiles_per_request"


# ---------------------------------------------------------------------------
# _forward.py
# ---------------------------------------------------------------------------

# Kernel constructor: the new parameter.
F_SIG_ANCHOR = (
    "        window_size: int | None,\n"
    "        uses_query_cache_seqlens: bool,\n"
    "    ):\n"
)
F_SIG_REPLACEMENT = (
    "        window_size: int | None,\n"
    "        uses_query_cache_seqlens: bool,\n"
    "        tiles_per_request: int,\n"
    "    ):\n"
)
F_SIG_PRESENT = "        uses_query_cache_seqlens: bool,\n        tiles_per_request: int,\n"

F_ATTR_ANCHOR = "        self.uses_query_cache_seqlens = bool(uses_query_cache_seqlens)\n"
F_ATTR_REPLACEMENT = (
    "        self.uses_query_cache_seqlens = bool(uses_query_cache_seqlens)\n"
    "        self.tiles_per_request = int(tiles_per_request)\n"
)
F_ATTR_PRESENT = "self.tiles_per_request = int(tiles_per_request)"

# Request mapping: const_expr keeps the 4-row path's code identical.
F_REQUEST_ANCHOR = (
    "        elif cutlass.const_expr(self.uses_query_cache_seqlens):\n"
    "            request = query_tile_index\n"
)
F_REQUEST_REPLACEMENT = (
    "        elif cutlass.const_expr(self.uses_query_cache_seqlens):\n"
    "            # V4PLUS-B4: tiles_per_request > 1 groups consecutive\n"
    "            # query tiles of one verify request (an 8-row request\n"
    "            # spans two 4-row tiles). The == 1 branch keeps the\n"
    "            # proven 4-row path's exact expression so its generated\n"
    "            # code is unchanged.\n"
    "            if cutlass.const_expr(self.tiles_per_request == 1):\n"
    "                request = query_tile_index\n"
    "            else:\n"
    "                request = query_tile_index // Int32(self.tiles_per_request)\n"
)
F_REQUEST_PRESENT = "request = query_tile_index // Int32(self.tiles_per_request)"


# ---------------------------------------------------------------------------
# _kernel.py
# ---------------------------------------------------------------------------

K_SIGNATURE_ANCHOR = (
    "        scratch.chunks_per_split,\n"
    "        scratch.uses_query_cache_seqlens,\n"
    "        scratch.physical_record_width,\n"
)
K_SIGNATURE_REPLACEMENT = (
    "        scratch.chunks_per_split,\n"
    "        scratch.uses_query_cache_seqlens,\n"
    "        scratch.tiles_per_request,\n"
    "        scratch.physical_record_width,\n"
)
K_SIGNATURE_PRESENT = (
    "        scratch.uses_query_cache_seqlens,\n"
    "        scratch.tiles_per_request,\n"
    "        scratch.physical_record_width,\n"
)

K_CTOR_ANCHOR = (
    "        window_size=scratch.window_size,\n"
    "        uses_query_cache_seqlens=scratch.uses_query_cache_seqlens,\n"
    "    )\n"
)
K_CTOR_REPLACEMENT = (
    "        window_size=scratch.window_size,\n"
    "        uses_query_cache_seqlens=scratch.uses_query_cache_seqlens,\n"
    "        tiles_per_request=scratch.tiles_per_request,\n"
    "    )\n"
)
K_CTOR_PRESENT = "tiles_per_request=scratch.tiles_per_request,"

K_VERSION_ANCHOR = (
    "    spec = KernelCompileSpec.from_fields(\n"
    '        "attention.dense_mla.forward",\n'
    "        5,\n"
)
K_VERSION_REPLACEMENT = (
    "    spec = KernelCompileSpec.from_fields(\n"
    '        "attention.dense_mla.forward",\n'
    "        6,\n"
)
K_VERSION_PRESENT = '"attention.dense_mla.forward",\n        6,\n'

K_KEYFIELD_ANCHOR = (
    "        key_field(\n"
    '            "uses_query_cache_seqlens",\n'
    "            scratch.uses_query_cache_seqlens,\n"
    "        ),\n"
)
K_KEYFIELD_REPLACEMENT = (
    "        key_field(\n"
    '            "uses_query_cache_seqlens",\n'
    "            scratch.uses_query_cache_seqlens,\n"
    "        ),\n"
    "        key_field(\n"
    '            "tiles_per_request",\n'
    "            scratch.tiles_per_request,\n"
    "        ),\n"
)
K_KEYFIELD_PRESENT = 'key_field(\n            "tiles_per_request",\n            scratch.tiles_per_request,\n        ),'


def main() -> int:
    print(f"[{SCRIPT_NAME}] {TAG}")
    print(f"[{SCRIPT_NAME}] B12X_ROOT={B12X_ROOT}")
    if not check_prerequisites():
        print(
            f"[{SCRIPT_NAME}] PREREQUISITE FAILED: v4plus-batch2 (B1a) is "
            "not applied to this b12x tree. Apply mods/v4plus-batch2 "
            "first; refusing to patch.",
            file=sys.stderr,
        )
        return 1

    ok = True
    ok &= apply_hunks(
        SCRATCH,
        [
            ("8-row verify tile rule", S_QT8_ANCHOR, S_QT8_REPLACEMENT, S_QT8_PRESENT),
            ("tiles_per_request helper", S_TPR_ANCHOR, S_TPR_REPLACEMENT, S_TPR_PRESENT),
            (
                "Caps tiled-verify validation",
                S_CAPS_VALIDATE_ANCHOR,
                S_CAPS_VALIDATE_REPLACEMENT,
                S_CAPS_VALIDATE_PRESENT,
            ),
            (
                "Scratch tiles_per_request field",
                S_SCRATCH_FIELD_ANCHOR,
                S_SCRATCH_FIELD_REPLACEMENT,
                S_SCRATCH_FIELD_PRESENT,
            ),
            (
                "_materialize tiles_per_request",
                S_MATERIALIZE_ANCHOR,
                S_MATERIALIZE_REPLACEMENT,
                S_MATERIALIZE_PRESENT,
            ),
            (
                "binding completeness x tiles_per_request",
                S_COMPLETE_ANCHOR,
                S_COMPLETE_REPLACEMENT,
                S_COMPLETE_PRESENT,
            ),
        ],
    )
    ok &= apply_hunks(
        FORWARD,
        [
            ("kernel ctor parameter", F_SIG_ANCHOR, F_SIG_REPLACEMENT, F_SIG_PRESENT),
            ("kernel ctor attribute", F_ATTR_ANCHOR, F_ATTR_REPLACEMENT, F_ATTR_PRESENT),
            (
                "request mapping divisor",
                F_REQUEST_ANCHOR,
                F_REQUEST_REPLACEMENT,
                F_REQUEST_PRESENT,
            ),
        ],
    )
    ok &= apply_hunks(
        KERNEL,
        [
            ("_signature entry", K_SIGNATURE_ANCHOR, K_SIGNATURE_REPLACEMENT, K_SIGNATURE_PRESENT),
            ("forward ctor arg", K_CTOR_ANCHOR, K_CTOR_REPLACEMENT, K_CTOR_PRESENT),
            ("compile-spec version 5 -> 6", K_VERSION_ANCHOR, K_VERSION_REPLACEMENT, K_VERSION_PRESENT),
            ("tiles_per_request key field", K_KEYFIELD_ANCHOR, K_KEYFIELD_REPLACEMENT, K_KEYFIELD_PRESENT),
        ],
    )

    if ok:
        print(
            f"[{SCRIPT_NAME}] NOTE: the compile-spec version bump (5 -> 6) "
            "re-JITs every dense-MLA forward kernel on first boot, "
            "including the proven 4-row ones — expect a one-time startup "
            "latency increase."
        )
        print(
            f"[{SCRIPT_NAME}] NOTE: dense_mla/_policy.py (the standalone "
            "planner's _query_tile) is deliberately untouched: the vllm "
            "plan path resolves tiles through _scratch._query_tile, and "
            "the standalone planner never sees verify plans."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
