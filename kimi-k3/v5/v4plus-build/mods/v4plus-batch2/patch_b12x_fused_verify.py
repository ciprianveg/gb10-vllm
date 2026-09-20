#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V4PLUS Batch 2 / B1 (b12x side) — fused 4-query DSpark verify kernel.

Ports the fused-verify subset of b12x PR #271 (b12x_271_fused_verify.diff,
Aug 29) onto the v4-prd image's b12x dense_mla (the #124/#138/#139-era
package at /opt/kimi-k3/b12x, plus v4plus-batch1 which did NOT touch
attention/dense_mla — batch1's f5394625c rewrite lives in
attention/_shared/mla, a different directory, so there is no conflict).

What the kernel change does: a causal DSpark verify block (nst=3 -> 4
uniform query rows per request) currently gets flattened by the vllm side
into 4 independent single-token decode rows, each sweeping the full
prefix.  The fused path instead runs ONE 4-row query tile per request and
gives every row its own visibility bound through a new per-query-row
`query_cache_seqlens` tensor, so each 64-token KV chunk is loaded once and
shared across the 4 rows.

Source lineage — RE-ANCHORED to the REAL image (Build #7 ground truth,
/tmp/opencode/k3spec/b12ximg): the image's b12x dense_mla is commit
b8c7153 ("Merge B12X PR #241 validation fixes into Kimi-K3 runtime"),
whose _forward.py / _kernel.py / _reference.py are BYTE-IDENTICAL to the
#271 diff's base (the dev/kimi-lineage dense_mla) and whose _scratch.py
differs from it by ONE removed line.  The #271 diff itself therefore
applies cleanly to the image files, and the hunks below are the VERBATIM
#271 hunks (fused-verify subset) against that content — not the
9bc5f0c/8596afcf1 shapes the first draft used (11 hunks missed on those).
The pristine image has NO uses_query_cache_seqlens anywhere in dense_mla
(verified), so this patch supplies the entire surface.

Deliberately EXCLUDED (bundled in the same PR diff but NOT this item):
  - dynamic sparsity (sparse_stride / sparse_min_tokens / sparse_sink_chunks
    / sparse_recent_chunks / sparse_refresh_interval, _selected_chunk,
    dynamic_sparse_chunk_indices, the __init__/api/planner/__init__ doc
    hunks): intentionally changes attention semantics — out of scope.
  - dense_mla/_policy.py and api.py/planner.py: out of scope for this
    patch's four files (policy/_query_tile rule lives in _scratch here).
  - benchmarks/ + tests/: not shipped in the image tree.
  - the 9bc5f0c-only fast paths (_bind_prevalidated /
    _materialize_prevalidated): DO NOT EXIST in the b8c7153 lineage
    (bind() calls _validate_binding directly) — their hunks are DROPPED,
    not skipped; the Plan.bind -> _validate_binding chain is complete.

Compile-spec version DECISION: the pristine image's forward spec is
already version 4 (same as the #271 base, which #271 itself bumped 4->6
for its seven new key fields).  Our port adds ONE key field, so the
version bumps 4 -> 5 (the first draft's 3 -> 4 hunk silently SKIPped on
the image because its present-marker matched the pristine version 4).

Dry-run state repair: the first dry-run applied 21 hunks (all verified to
land at correct or functionally-equivalent positions — including the
__init__ attr, which landed inside the real forward kernel's __init__,
where `self.fp8 = bool(fp8)` is unique) but left the __init__ PARAMETER
unapplied, so the image currently has
`self.uses_query_cache_seqlens = bool(uses_query_cache_seqlens)`
referencing an undefined name — a guaranteed NameError on the first
kernel compile.  Re-running this fixed script completes the signature
hunk and resolves it; every already-applied hunk skips by marker.

Idempotent: every hunk is skipped when its marker is already present.
A missing anchor prints a NOTE and skips that hunk (never fails the boot);
only file-not-found or a broken post-patch compile exits non-zero.
"""

from __future__ import annotations

import os
import py_compile
import sys

SCRIPT_NAME = "patch_b12x_fused_verify"
TAG = "# V4PLUS-B2 (b12x #271 fused verify)"

B12X_ROOT = os.environ.get("B12X_ROOT")
if not B12X_ROOT:
    try:
        import b12x  # noqa: F401

        B12X_ROOT = os.path.dirname(b12x.__file__)
    except Exception:
        B12X_ROOT = "/opt/kimi-k3/b12x/b12x"

DENSE_MLA = os.path.join(B12X_ROOT, "attention", "dense_mla")


def apply_hunks(path: str, hunks: list[tuple[str, str, str, str]]) -> bool:
    """Apply (name, anchor, replacement, present) hunks; True if changed."""
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


# ---------------------------------------------------------------------------
# b12x/attention/dense_mla/_scratch.py
# ---------------------------------------------------------------------------

# 4-row verify tiles: one query tile per request when the plan is exactly
# 4 rows per request (VERBATIM #271 — the b8c7153 Caps HAS window_size, so
# the full rule including the window_size clause ports as written; the old
# 9bc5f0c-shaped anchor dropped the clause because that lineage lacked the
# field, and was 0x here because the base condition also carries it).
S_QT_ANCHOR = (
    "def _query_tile(caps: Caps) -> int:\n"
    '    if caps.mode == "decode" or caps.max_batch != 1 or caps.window_size is not None:\n'
    "        return 1\n"
)
S_QT_REPLACEMENT = (
    "def _query_tile(caps: Caps) -> int:\n"
    "    if (\n"
    '        caps.mode == "verify"\n'
    "        and caps.max_total_q == caps.max_batch * 4\n"
    "        and caps.window_size is None\n"
    "    ):\n"
    "        return 4 if caps.kv_dtype == _FP8 else 1\n"
    '    if caps.mode == "decode" or caps.max_batch != 1 or caps.window_size is not None:\n'
    "        return 1\n"
)
S_QT_PRESENT = 'caps.mode == "verify"\n        and caps.max_total_q == caps.max_batch * 4\n        and caps.window_size is None'

# Caps field (verbatim #271; the sparse_* fields that follow it in #271 are
# deliberately not ported).
S_CAPS_FIELD_ANCHOR = (
    "    use_cuda_graph: bool = False\n"
    "    budget: Budget | None = None\n"
)
S_CAPS_FIELD_REPLACEMENT = (
    "    use_cuda_graph: bool = False\n"
    "    uses_query_cache_seqlens: bool = False\n"
    "    budget: Budget | None = None\n"
)
S_CAPS_FIELD_PRESENT = "    uses_query_cache_seqlens: bool = False\n    budget: Budget | None = None"

# Caps.__post_init__ validation (verbatim #271 minus the sparse loop).
S_CAPS_CHECK_ANCHOR = (
    '        if self.mode not in ("decode", "extend", "verify"):\n'
    "            raise ValueError(f\"unsupported dense MLA mode {self.mode!r}\")\n"
)
S_CAPS_CHECK_REPLACEMENT = (
    '        if self.mode not in ("decode", "extend", "verify"):\n'
    "            raise ValueError(f\"unsupported dense MLA mode {self.mode!r}\")\n"
    "        uses_query_cache_seqlens = bool(self.uses_query_cache_seqlens)\n"
    "        if uses_query_cache_seqlens and self.mode != \"verify\":\n"
    "            raise ValueError(\n"
    '                "per-query cache lengths are supported only by verify plans"\n'
    "            )\n"
)
S_CAPS_CHECK_PRESENT = 'per-query cache lengths are supported only by verify plans'

# Caps.__post_init__ normalization (verbatim #271).
S_CAPS_NORM_ANCHOR = (
    '        object.__setattr__(self, "use_cuda_graph", bool(self.use_cuda_graph))\n'
)
S_CAPS_NORM_REPLACEMENT = (
    '        object.__setattr__(self, "use_cuda_graph", bool(self.use_cuda_graph))\n'
    "        object.__setattr__(\n"
    "            self,\n"
    '            "uses_query_cache_seqlens",\n'
    "            uses_query_cache_seqlens,\n"
    "        )\n"
)
S_CAPS_NORM_PRESENT = '            "uses_query_cache_seqlens",\n            uses_query_cache_seqlens,'

# Scratch field (verbatim #271 minus sparse_*).
S_SCRATCH_ANCHOR = (
    "    query_tile: int\n"
    "    use_cuda_graph: bool\n"
    "    partial_output: torch.Tensor | None\n"
)
S_SCRATCH_REPLACEMENT = (
    "    query_tile: int\n"
    "    use_cuda_graph: bool\n"
    "    uses_query_cache_seqlens: bool\n"
    "    partial_output: torch.Tensor | None\n"
)
S_SCRATCH_PRESENT = "    use_cuda_graph: bool\n    uses_query_cache_seqlens: bool\n    partial_output"

# Binding field (verbatim #271).
S_BINDING_ANCHOR = (
    "    cache_seqlens: torch.Tensor\n"
    "    cu_seqlens_q: torch.Tensor\n"
    "    kv_scale: torch.Tensor | None\n"
)
S_BINDING_REPLACEMENT = (
    "    cache_seqlens: torch.Tensor\n"
    "    query_cache_seqlens: torch.Tensor\n"
    "    cu_seqlens_q: torch.Tensor\n"
    "    kv_scale: torch.Tensor | None\n"
)
S_BINDING_PRESENT = "    cache_seqlens: torch.Tensor\n    query_cache_seqlens: torch.Tensor\n    cu_seqlens_q: torch.Tensor"

# _validate_binding signature (disambiguated from _bind_prevalidated by the
# def line; verbatim #271 param addition).
S_VB_SIG_ANCHOR = (
    "def _validate_binding(\n"
    "    *,\n"
    "    scratch: Scratch,\n"
    "    q: torch.Tensor,\n"
    "    kv_cache: torch.Tensor,\n"
    "    output: torch.Tensor,\n"
    "    page_table: torch.Tensor,\n"
    "    cache_seqlens: torch.Tensor,\n"
    "    cu_seqlens_q: torch.Tensor,\n"
    "    kv_scale: torch.Tensor | None,\n"
)
S_VB_SIG_REPLACEMENT = (
    "def _validate_binding(\n"
    "    *,\n"
    "    scratch: Scratch,\n"
    "    q: torch.Tensor,\n"
    "    kv_cache: torch.Tensor,\n"
    "    output: torch.Tensor,\n"
    "    page_table: torch.Tensor,\n"
    "    cache_seqlens: torch.Tensor,\n"
    "    query_cache_seqlens: torch.Tensor | None,\n"
    "    cu_seqlens_q: torch.Tensor,\n"
    "    kv_scale: torch.Tensor | None,\n"
)
S_VB_SIG_PRESENT = "    cache_seqlens: torch.Tensor,\n    query_cache_seqlens: torch.Tensor | None,\n    cu_seqlens_q: torch.Tensor,"

# _validate_binding body: the per-query length contract (verbatim #271 minus
# sparse), inserted between the cache_seqlens/cu_seqlens_q checks and the
# page_table checks.
S_VB_CHECK_ANCHOR = (
    "        if tensor.device != scratch.device or not tensor.is_contiguous():\n"
    '            raise ValueError(f"{name} must be contiguous on the plan device")\n'
    "    if (\n"
    "        page_table.ndim != 2\n"
)
S_VB_CHECK_REPLACEMENT = (
    "        if tensor.device != scratch.device or not tensor.is_contiguous():\n"
    '            raise ValueError(f"{name} must be contiguous on the plan device")\n'
    "    if scratch.uses_query_cache_seqlens:\n"
    "        if query_cache_seqlens is None:\n"
    '            raise ValueError("verify plan requires per-query cache lengths")\n'
    "        if query_cache_seqlens.dtype != torch.int32 or tuple(\n"
    "            query_cache_seqlens.shape\n"
    "        ) != (int(q.shape[0]),):\n"
    "            raise TypeError(\n"
    '                "query_cache_seqlens must be contiguous int32 with shape [total_q]"\n'
    "            )\n"
    "        if (\n"
    "            query_cache_seqlens.device != scratch.device\n"
    "            or not query_cache_seqlens.is_contiguous()\n"
    "        ):\n"
    "            raise ValueError(\n"
    '                "query_cache_seqlens must be contiguous on the plan device"\n'
    "            )\n"
    "    elif query_cache_seqlens is not None:\n"
    '        raise ValueError("plan does not accept per-query cache lengths")\n'
    "    else:\n"
    "        query_cache_seqlens = cache_seqlens\n"
    "    if (\n"
    '        scratch.mode == "verify"\n'
    "        and scratch.query_tile > 1\n"
    "        and int(q.shape[0]) != batch * scratch.query_tile\n"
    "    ):\n"
    "        raise ValueError(\n"
    '            "tiled verify plan requires one complete query tile per request"\n'
    "        )\n"
    "    if (\n"
    "        page_table.ndim != 2\n"
)
S_VB_CHECK_PRESENT = '"tiled verify plan requires one complete query tile per request"'

# _validate_binding Binding construction (verbatim #271).
S_VB_RET_ANCHOR = (
    "        cache_seqlens=cache_seqlens.detach(),\n"
    "        cu_seqlens_q=cu_seqlens_q.detach(),\n"
)
S_VB_RET_REPLACEMENT = (
    "        cache_seqlens=cache_seqlens.detach(),\n"
    "        query_cache_seqlens=query_cache_seqlens.detach(),\n"
    "        cu_seqlens_q=cu_seqlens_q.detach(),\n"
)
S_VB_RET_PRESENT = "        query_cache_seqlens=query_cache_seqlens.detach(),"

# _bind_prevalidated signature (v4-only fast path; same param addition).

# _bind_prevalidated body (adapted: caller-validated path only DEFAULTS the
# per-query lengths; it does not re-validate, matching the path's contract).


# _materialize Scratch construction (verbatim #271 minus sparse).
S_MAT_ANCHOR = (
    "        query_tile=query_tile,\n"
    "        use_cuda_graph=caps.use_cuda_graph,\n"
    "        partial_output=partial_output,\n"
)
S_MAT_REPLACEMENT = (
    "        query_tile=query_tile,\n"
    "        use_cuda_graph=caps.use_cuda_graph,\n"
    "        uses_query_cache_seqlens=caps.uses_query_cache_seqlens,\n"
    "        partial_output=partial_output,\n"
)
S_MAT_PRESENT = (
    "        query_tile=query_tile,\n"
    "        use_cuda_graph=caps.use_cuda_graph,\n"
    "        uses_query_cache_seqlens=caps.uses_query_cache_seqlens,\n"
)

# _materialize_prevalidated Scratch construction (v4-only fast path).

# Plan.bind signature (verbatim #271).
S_BIND_SIG_ANCHOR = (
    "        cache_seqlens: torch.Tensor,\n"
    "        cu_seqlens_q: torch.Tensor,\n"
    "        kv_scale: torch.Tensor | None = None,\n"
)
S_BIND_SIG_REPLACEMENT = (
    "        cache_seqlens: torch.Tensor,\n"
    "        query_cache_seqlens: torch.Tensor | None = None,\n"
    "        cu_seqlens_q: torch.Tensor,\n"
    "        kv_scale: torch.Tensor | None = None,\n"
)
S_BIND_SIG_PRESENT = "        query_cache_seqlens: torch.Tensor | None = None,\n        cu_seqlens_q: torch.Tensor,"

# Plan.bind -> bind_impl pass-through (verbatim #271).
S_BIND_CALL_ANCHOR = (
    "            cache_seqlens=cache_seqlens,\n"
    "            cu_seqlens_q=cu_seqlens_q,\n"
)
S_BIND_CALL_REPLACEMENT = (
    "            cache_seqlens=cache_seqlens,\n"
    "            query_cache_seqlens=query_cache_seqlens,\n"
    "            cu_seqlens_q=cu_seqlens_q,\n"
)
S_BIND_CALL_PRESENT = "            query_cache_seqlens=query_cache_seqlens,\n            cu_seqlens_q=cu_seqlens_q,"

SCRATCH_HUNKS = [
    ("query-tile verify rule", S_QT_ANCHOR, S_QT_REPLACEMENT, S_QT_PRESENT),
    ("Caps field", S_CAPS_FIELD_ANCHOR, S_CAPS_FIELD_REPLACEMENT, S_CAPS_FIELD_PRESENT),
    ("Caps validation", S_CAPS_CHECK_ANCHOR, S_CAPS_CHECK_REPLACEMENT, S_CAPS_CHECK_PRESENT),
    ("Caps normalization", S_CAPS_NORM_ANCHOR, S_CAPS_NORM_REPLACEMENT, S_CAPS_NORM_PRESENT),
    ("Scratch field", S_SCRATCH_ANCHOR, S_SCRATCH_REPLACEMENT, S_SCRATCH_PRESENT),
    ("Binding field", S_BINDING_ANCHOR, S_BINDING_REPLACEMENT, S_BINDING_PRESENT),
    ("_validate_binding signature", S_VB_SIG_ANCHOR, S_VB_SIG_REPLACEMENT, S_VB_SIG_PRESENT),
    ("_validate_binding checks", S_VB_CHECK_ANCHOR, S_VB_CHECK_REPLACEMENT, S_VB_CHECK_PRESENT),
    ("_validate_binding Binding", S_VB_RET_ANCHOR, S_VB_RET_REPLACEMENT, S_VB_RET_PRESENT),
    ("_materialize Scratch", S_MAT_ANCHOR, S_MAT_REPLACEMENT, S_MAT_PRESENT),
    ("Plan.bind signature", S_BIND_SIG_ANCHOR, S_BIND_SIG_REPLACEMENT, S_BIND_SIG_PRESENT),
    ("Plan.bind pass-through", S_BIND_CALL_ANCHOR, S_BIND_CALL_REPLACEMENT, S_BIND_CALL_PRESENT),
]
# NOT APPLICABLE to the b8c7153 lineage (dropped, not skipped): the
# _bind_prevalidated signature/defaulting/Binding and
# _materialize_prevalidated hunks — those fast paths exist only in the
# 9bc5f0c lineage; here bind() calls _validate_binding directly, so the
# Plan.bind -> _validate_binding threading above is the complete chain.

# ---------------------------------------------------------------------------
# b12x/attention/dense_mla/_forward.py  (the CuTe DSL kernel)
# ---------------------------------------------------------------------------

# DenseMlaForwardKernel.__init__ (verbatim #271 position: the param lands
# after window_size; keyword-only, so call-site order is unaffected).
# RE-ANCHORED: the b8c7153 __init__ carries the generalized
# qk_dim/value_dim/window_size params the 9bc5f0c lineage lacked.
F_INIT_SIG_ANCHOR = (
    "        qk_dim: int,\n"
    "        value_dim: int,\n"
    "        window_size: int | None,\n"
    "    ):\n"
)
F_INIT_SIG_REPLACEMENT = (
    "        qk_dim: int,\n"
    "        value_dim: int,\n"
    "        window_size: int | None,\n"
    "        uses_query_cache_seqlens: bool,\n"
    "    ):\n"
)
F_INIT_SIG_PRESENT = "        window_size: int | None,\n        uses_query_cache_seqlens: bool,\n    ):"

F_INIT_ATTR_ANCHOR = "        self.fp8 = bool(fp8)\n"
F_INIT_ATTR_REPLACEMENT = (
    "        self.fp8 = bool(fp8)\n"
    "        self.uses_query_cache_seqlens = bool(uses_query_cache_seqlens)\n"
)
F_INIT_ATTR_PRESENT = "        self.uses_query_cache_seqlens = bool(uses_query_cache_seqlens)"

# __call__ signature — RE-ANCHORED to the b8c7153 signature, which carries
# the generalized cache_record_stride_bytes param the 9bc5f0c lineage
# lacked (that extra param is why the old full-signature anchor was 0x).
# The stream tail disambiguates __call__ from kernel(); verbatim #271
# insertion position (query_cache_seqlens after cu_seqlens_q).
F_CALL_SIG_ANCHOR = (
    "        cache_seqlens: cute.Tensor,\n"
    "        cu_seqlens_q: cute.Tensor,\n"
    "        output: cute.Tensor,\n"
    "        final_lse: cute.Tensor,\n"
    "        partial_output: cute.Tensor,\n"
    "        partial_lse: cute.Tensor,\n"
    "        kv_scale: cute.Tensor,\n"
    "        q_scale: cute.Tensor,\n"
    "        sm_scale_log2: Float32,\n"
    "        q_stride_row_bytes: Int64,\n"
    "        q_stride_head_bytes: Int64,\n"
    "        page_stride_bytes: Int64,\n"
    "        cache_record_stride_bytes: Int64,\n"
    "        page_table_stride: Int64,\n"
    "        total_q: Int32,\n"
    "        batch: Int32,\n"
    "        active_splits: Int32,\n"
    "        stream: cuda.CUstream,\n"
    "    ):\n"
)
F_CALL_SIG_REPLACEMENT = (
    "        cache_seqlens: cute.Tensor,\n"
    "        cu_seqlens_q: cute.Tensor,\n"
    "        query_cache_seqlens: cute.Tensor,\n"
    "        output: cute.Tensor,\n"
    "        final_lse: cute.Tensor,\n"
    "        partial_output: cute.Tensor,\n"
    "        partial_lse: cute.Tensor,\n"
    "        kv_scale: cute.Tensor,\n"
    "        q_scale: cute.Tensor,\n"
    "        sm_scale_log2: Float32,\n"
    "        q_stride_row_bytes: Int64,\n"
    "        q_stride_head_bytes: Int64,\n"
    "        page_stride_bytes: Int64,\n"
    "        cache_record_stride_bytes: Int64,\n"
    "        page_table_stride: Int64,\n"
    "        total_q: Int32,\n"
    "        batch: Int32,\n"
    "        active_splits: Int32,\n"
    "        stream: cuda.CUstream,\n"
    "    ):\n"
)
F_CALL_SIG_PRESENT = (
    "        query_cache_seqlens: cute.Tensor,\n"
    "        output: cute.Tensor,\n"
    "        final_lse: cute.Tensor,\n"
    "        partial_output: cute.Tensor,\n"
    "        partial_lse: cute.Tensor,\n"
    "        kv_scale: cute.Tensor,\n"
    "        q_scale: cute.Tensor,\n"
    "        sm_scale_log2: Float32,\n"
    "        q_stride_row_bytes: Int64,\n"
    "        q_stride_head_bytes: Int64,\n"
    "        page_stride_bytes: Int64,\n"
    "        cache_record_stride_bytes: Int64,\n"
    "        page_table_stride: Int64,\n"
    "        total_q: Int32,\n"
    "        batch: Int32,\n"
    "        active_splits: Int32,\n"
    "        stream: cuda.CUstream,\n"
    "    ):\n"
)

# __call__ -> self.kernel(...) pass-through (verbatim #271 arg order;
# verified to land in the self.kernel call — the only 12-space
# cache_seqlens/cu_seqlens_q/output sequence in the file).
F_CALL_PASS_ANCHOR = (
    "            cache_seqlens,\n"
    "            cu_seqlens_q,\n"
    "            output,\n"
)
F_CALL_PASS_REPLACEMENT = (
    "            cache_seqlens,\n"
    "            cu_seqlens_q,\n"
    "            query_cache_seqlens,\n"
    "            output,\n"
)
F_CALL_PASS_PRESENT = "            query_cache_seqlens,\n            output,"

# kernel() signature — RE-ANCHORED like __call__ (cache_record_stride_bytes;
# ends without the stream tail, which is the disambiguator).
F_KERNEL_SIG_ANCHOR = (
    "        cache_seqlens: cute.Tensor,\n"
    "        cu_seqlens_q: cute.Tensor,\n"
    "        output: cute.Tensor,\n"
    "        final_lse: cute.Tensor,\n"
    "        partial_output: cute.Tensor,\n"
    "        partial_lse: cute.Tensor,\n"
    "        kv_scale: cute.Tensor,\n"
    "        q_scale: cute.Tensor,\n"
    "        sm_scale_log2: Float32,\n"
    "        q_stride_row_bytes: Int64,\n"
    "        q_stride_head_bytes: Int64,\n"
    "        page_stride_bytes: Int64,\n"
    "        cache_record_stride_bytes: Int64,\n"
    "        page_table_stride: Int64,\n"
    "        total_q: Int32,\n"
    "        batch: Int32,\n"
    "        active_splits: Int32,\n"
    "    ):\n"
)
F_KERNEL_SIG_REPLACEMENT = (
    "        cache_seqlens: cute.Tensor,\n"
    "        cu_seqlens_q: cute.Tensor,\n"
    "        query_cache_seqlens: cute.Tensor,\n"
    "        output: cute.Tensor,\n"
    "        final_lse: cute.Tensor,\n"
    "        partial_output: cute.Tensor,\n"
    "        partial_lse: cute.Tensor,\n"
    "        kv_scale: cute.Tensor,\n"
    "        q_scale: cute.Tensor,\n"
    "        sm_scale_log2: Float32,\n"
    "        q_stride_row_bytes: Int64,\n"
    "        q_stride_head_bytes: Int64,\n"
    "        page_stride_bytes: Int64,\n"
    "        cache_record_stride_bytes: Int64,\n"
    "        page_table_stride: Int64,\n"
    "        total_q: Int32,\n"
    "        batch: Int32,\n"
    "        active_splits: Int32,\n"
    "    ):\n"
)
F_KERNEL_SIG_PRESENT = (
    "        query_cache_seqlens: cute.Tensor,\n"
    "        output: cute.Tensor,\n"
    "        final_lse: cute.Tensor,\n"
    "        partial_output: cute.Tensor,\n"
    "        partial_lse: cute.Tensor,\n"
    "        kv_scale: cute.Tensor,\n"
    "        q_scale: cute.Tensor,\n"
    "        sm_scale_log2: Float32,\n"
    "        q_stride_row_bytes: Int64,\n"
    "        q_stride_head_bytes: Int64,\n"
    "        page_stride_bytes: Int64,\n"
    "        cache_record_stride_bytes: Int64,\n"
    "        page_table_stride: Int64,\n"
    "        total_q: Int32,\n"
    "        batch: Int32,\n"
    "        active_splits: Int32,\n"
    "    ):\n"
)

# Request selection (verbatim #271): under per-query lengths every query
# tile IS one request (tiled verify = one complete tile per request).
F_REQUEST_ANCHOR = (
    "            request = lower\n"
    "        query_begin = Int32(cu_seqlens_q[request])\n"
)
F_REQUEST_REPLACEMENT = (
    "            request = lower\n"
    "        elif cutlass.const_expr(self.uses_query_cache_seqlens):\n"
    "            request = query_tile_index\n"
    "        query_begin = Int32(cu_seqlens_q[request])\n"
)
F_REQUEST_PRESENT = "        elif cutlass.const_expr(self.uses_query_cache_seqlens):\n            request = query_tile_index"

# Per-row visibility (verbatim #271): each verify row sees keys through its
# own drafted position instead of the tile-uniform causal bound.
F_VISIBLE_ANCHOR = (
    "            visible_end = cache_length - query_length + local_query + Int32(1)\n"
    "            if visible_end < Int32(0):\n"
)
F_VISIBLE_REPLACEMENT = (
    "            visible_end = cache_length - query_length + local_query + Int32(1)\n"
    "            if cutlass.const_expr(self.uses_query_cache_seqlens):\n"
    "                visible_end = Int32(query_cache_seqlens[query_row])\n"
    "            if visible_end < Int32(0):\n"
)
F_VISIBLE_PRESENT = "                visible_end = Int32(query_cache_seqlens[query_row])"

FORWARD_HUNKS = [
    ("__init__ signature", F_INIT_SIG_ANCHOR, F_INIT_SIG_REPLACEMENT, F_INIT_SIG_PRESENT),
    ("__init__ attr", F_INIT_ATTR_ANCHOR, F_INIT_ATTR_REPLACEMENT, F_INIT_ATTR_PRESENT),
    ("__call__ signature", F_CALL_SIG_ANCHOR, F_CALL_SIG_REPLACEMENT, F_CALL_SIG_PRESENT),
    ("__call__ pass-through", F_CALL_PASS_ANCHOR, F_CALL_PASS_REPLACEMENT, F_CALL_PASS_PRESENT),
    ("kernel signature", F_KERNEL_SIG_ANCHOR, F_KERNEL_SIG_REPLACEMENT, F_KERNEL_SIG_PRESENT),
    ("request selection", F_REQUEST_ANCHOR, F_REQUEST_REPLACEMENT, F_REQUEST_PRESENT),
    ("per-row visibility", F_VISIBLE_ANCHOR, F_VISIBLE_REPLACEMENT, F_VISIBLE_PRESENT),
]

# ---------------------------------------------------------------------------
# b12x/attention/dense_mla/_kernel.py  (launch / compile-spec plumbing)
# ---------------------------------------------------------------------------

# _signature (verbatim #271 minus sparse fields).
K_SIG_ANCHOR = (
    "        scratch.query_tile,\n"
    "        scratch.num_splits,\n"
    "        scratch.chunks_per_split,\n"
)
K_SIG_REPLACEMENT = (
    "        scratch.query_tile,\n"
    "        scratch.num_splits,\n"
    "        scratch.chunks_per_split,\n"
    "        scratch.uses_query_cache_seqlens,\n"
)
K_SIG_PRESENT = "        scratch.chunks_per_split,\n        scratch.uses_query_cache_seqlens,"

# DenseMlaForwardKernel construction — RE-ANCHORED to the b8c7153 ctor,
# which passes the generalized qk_dim/value_dim/window_size kwargs before
# the close paren (the old 9bc5f0c-shaped fp8-tail anchor was 0x here).
# Verbatim #271 insertion position: after window_size.
K_CTOR_ANCHOR = (
    "        value_dim=scratch.v_head_dim,\n"
    "        window_size=scratch.window_size,\n"
    "    )\n"
)
K_CTOR_REPLACEMENT = (
    "        value_dim=scratch.v_head_dim,\n"
    "        window_size=scratch.window_size,\n"
    "        uses_query_cache_seqlens=scratch.uses_query_cache_seqlens,\n"
    "    )\n"
)
K_CTOR_PRESENT = "        uses_query_cache_seqlens=scratch.uses_query_cache_seqlens,\n    )"

# _forward_args: the per-query lengths tensor, between cu_seqlens_q and
# output (verbatim #271; matches the kernel signature order).
K_ARGS_ANCHOR = (
    "        _to_cute(\n"
    "            binding.cu_seqlens_q,\n"
    "            cutlass.Int32,\n"
    "            align=4,\n"
    "            dynamic_layout=True,\n"
    "        ),\n"
    "        _to_cute(\n"
    "            output,\n"
    "            cutlass.BFloat16,\n"
)
K_ARGS_REPLACEMENT = (
    "        _to_cute(\n"
    "            binding.cu_seqlens_q,\n"
    "            cutlass.Int32,\n"
    "            align=4,\n"
    "            dynamic_layout=True,\n"
    "        ),\n"
    "        _to_cute(\n"
    "            binding.query_cache_seqlens,\n"
    "            cutlass.Int32,\n"
    "            align=4,\n"
    "            dynamic_layout=True,\n"
    "        ),\n"
    "        _to_cute(\n"
    "            output,\n"
    "            cutlass.BFloat16,\n"
)
K_ARGS_PRESENT = "            binding.query_cache_seqlens,\n            cutlass.Int32,"

# Compile-spec version bump 4 -> 5.  DECISION: the pristine image's
# _kernel.py ALREADY has version 4 (the b8c7153 lineage matches the #271
# base, which #271 itself bumped 4->6; our port adds ONE key field, so we
# bump to 5).  The old 3->4 hunk silently SKIPped on the image because its
# present marker matched the pristine version 4.  The bump is
# belt-and-suspenders: the new key_field below changes the spec key by
# itself, but an explicit version bump guarantees no stale JIT cache entry
# survives.
K_SPEC_VER_ANCHOR = (
    "    spec = KernelCompileSpec.from_fields(\n"
    '        "attention.dense_mla.forward",\n'
    "        4,\n"
)
K_SPEC_VER_REPLACEMENT = (
    "    spec = KernelCompileSpec.from_fields(\n"
    '        "attention.dense_mla.forward",\n'
    "        5,\n"
)
K_SPEC_VER_PRESENT = '"attention.dense_mla.forward",\n        5,'

K_SPEC_KEY_ANCHOR = (
    '        key_field("chunks_per_split", scratch.chunks_per_split),\n'
)
K_SPEC_KEY_REPLACEMENT = (
    '        key_field("chunks_per_split", scratch.chunks_per_split),\n'
    "        key_field(\n"
    '            "uses_query_cache_seqlens",\n'
    "            scratch.uses_query_cache_seqlens,\n"
    "        ),\n"
)
K_SPEC_KEY_PRESENT = '            "uses_query_cache_seqlens",\n            scratch.uses_query_cache_seqlens,'

KERNELPY_HUNKS = [
    ("_signature", K_SIG_ANCHOR, K_SIG_REPLACEMENT, K_SIG_PRESENT),
    ("forward ctor", K_CTOR_ANCHOR, K_CTOR_REPLACEMENT, K_CTOR_PRESENT),
    ("forward args", K_ARGS_ANCHOR, K_ARGS_REPLACEMENT, K_ARGS_PRESENT),
    ("spec version bump", K_SPEC_VER_ANCHOR, K_SPEC_VER_REPLACEMENT, K_SPEC_VER_PRESENT),
    ("spec key field", K_SPEC_KEY_ANCHOR, K_SPEC_KEY_REPLACEMENT, K_SPEC_KEY_PRESENT),
]

# ---------------------------------------------------------------------------
# b12x/attention/dense_mla/_reference.py  (exact oracle; test-support)
# ---------------------------------------------------------------------------

R_SIG_ANCHOR = (
    "    cu_seqlens_q: torch.Tensor,\n"
    "    *,\n"
    "    kv_scale: torch.Tensor | float | None = None,\n"
)
R_SIG_REPLACEMENT = (
    "    cu_seqlens_q: torch.Tensor,\n"
    "    *,\n"
    "    query_cache_seqlens: torch.Tensor | None = None,\n"
    "    kv_scale: torch.Tensor | float | None = None,\n"
)
R_SIG_PRESENT = "    query_cache_seqlens: torch.Tensor | None = None,\n    kv_scale: torch.Tensor | float | None = None,"

R_CHECK_ANCHOR = (
    '    if tuple(cu_seqlens_q.shape) != (batch + 1,):\n'
    '        raise ValueError("cu_seqlens_q shape must be [batch + 1]")\n'
)
R_CHECK_REPLACEMENT = (
    '    if tuple(cu_seqlens_q.shape) != (batch + 1,):\n'
    '        raise ValueError("cu_seqlens_q shape must be [batch + 1]")\n'
    "    if query_cache_seqlens is not None and (\n"
    "        query_cache_seqlens.dtype != torch.int32\n"
    "        or tuple(query_cache_seqlens.shape) != (int(q.shape[0]),)\n"
    "    ):\n"
    '        raise TypeError("query_cache_seqlens must be int32 with shape [total_q]")\n'
)
R_CHECK_PRESENT = '        raise TypeError("query_cache_seqlens must be int32 with shape [total_q]")'

R_DEV_ANCHOR = (
    "        t.device != q.device for t in (cache, page_table, cache_seqlens, cu_seqlens_q)\n"
    "    ):\n"
)
R_DEV_REPLACEMENT = (
    "        t.device != q.device\n"
    "        for t in (\n"
    "            cache,\n"
    "            page_table,\n"
    "            cache_seqlens,\n"
    "            cu_seqlens_q,\n"
    "            *(() if query_cache_seqlens is None else (query_cache_seqlens,)),\n"
    "        )\n"
    "    ):\n"
)
R_DEV_PRESENT = "            *(() if query_cache_seqlens is None else (query_cache_seqlens,)),"

# Per-row visibility in the oracle loop — RE-ANCHORED: the b8c7153 oracle
# has window handling (visible_begin) between `visible` and `key`, so the
# old 9bc5f0c-shaped anchor (visible directly followed by
# `key = records[:visible]`) was 0x.  Verbatim #271 conditional + range
# check; the sparse selected_positions rewrite is excluded and the
# visible_begin/key lines stay untouched.
R_VIS_ANCHOR = (
    "            visible = kv_len - q_len + local_q + 1\n"
    "            visible_begin = (\n"
)
R_VIS_REPLACEMENT = (
    "            visible = (\n"
    "                int(query_cache_seqlens[q_begin + local_q])\n"
    "                if query_cache_seqlens is not None\n"
    "                else kv_len - q_len + local_q + 1\n"
    "            )\n"
    '            if not 0 < visible <= kv_len:\n'
    '                raise ValueError("per-query visible cache length is out of range")\n'
    "            visible_begin = (\n"
)
R_VIS_PRESENT = '                raise ValueError("per-query visible cache length is out of range")'

REFERENCE_HUNKS = [
    ("signature", R_SIG_ANCHOR, R_SIG_REPLACEMENT, R_SIG_PRESENT),
    ("shape/dtype check", R_CHECK_ANCHOR, R_CHECK_REPLACEMENT, R_CHECK_PRESENT),
    ("device check", R_DEV_ANCHOR, R_DEV_REPLACEMENT, R_DEV_PRESENT),
    ("per-row visibility", R_VIS_ANCHOR, R_VIS_REPLACEMENT, R_VIS_PRESENT),
]


def main() -> int:
    print(f"[{SCRIPT_NAME}] {TAG}")
    print(
        f"[{SCRIPT_NAME}] b12x dense_mla fused 4-query verify kernel "
        f"(PR #271 fused-verify subset; sparse subset excluded)"
    )
    if not os.path.isdir(DENSE_MLA):
        print(
            f"[{SCRIPT_NAME}] ERROR: dense_mla package not found at {DENSE_MLA}",
            file=sys.stderr,
        )
        return 1
    ok = True
    ok &= apply_hunks(os.path.join(DENSE_MLA, "_scratch.py"), SCRATCH_HUNKS)
    ok &= apply_hunks(os.path.join(DENSE_MLA, "_forward.py"), FORWARD_HUNKS)
    ok &= apply_hunks(os.path.join(DENSE_MLA, "_kernel.py"), KERNELPY_HUNKS)
    ok &= apply_hunks(os.path.join(DENSE_MLA, "_reference.py"), REFERENCE_HUNKS)
    if not ok:
        print(
            f"[{SCRIPT_NAME}] NOTE: one or more hunks were skipped — the "
            "fused-verify kernel surface may be INCOMPLETE. The vllm-side "
            "gate (VLLM_K3_FUSED_VERIFY) probes for this surface and stays "
            "OFF when it is absent, so serving remains on the flattened "
            "verify path."
        )
    else:
        print(
            f"[{SCRIPT_NAME}] fused-verify kernel surface complete: "
            "Caps/Scratch/Binding query_cache_seqlens threading, 4-row "
            "verify tiles, per-row visibility, compile-spec key + version "
            "bump, oracle support."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
