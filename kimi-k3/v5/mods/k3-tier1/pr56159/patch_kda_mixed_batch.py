#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PR56159 — backport of upstream vLLM PR #56159.

Avoid the KDA mixed-batch gather/scatter: when the scheduler's stable
partition leaves spec and non-spec tokens CONTIGUOUS, record
``spec_token_start`` / ``non_spec_token_start`` in the metadata and slice
instead of index_select-ing mixed_qkv/g1/beta; the attention kernels then
write straight into the corresponding ``core_attn_out`` slices (``out=``
plumbing for chunk_kda_with_fused_gate and
fused_recurrent_kda_packed_decode), skipping the index_copy_ restore.

FORK ADAPTATIONS (documented):
  * The fork's kda.py _forward differs from upstream's pre-PR context: it
    has no flashinfer/recoverssm prefill branches and its flashkda backend
    cannot write into a caller-provided output slice. When the continuous
    layout is detected but the active non-spec backend did NOT write in
    place (flashkda), the restore step falls back to a plain slice copy
    (core_attn_out[:, non_spec_slice] = ...) — still avoiding the input
    gathers and the index scatter.
  * The spec path keeps the fork's existing prefix-slice ``spec_out`` and
    extends it with the contiguous-slice case.

Pure Python / Triton. Idempotent via marker; py_compile doraise.
"""
import py_compile
import sys

SCRIPT_NAME = "patch_kda_mixed_batch"
TAG = "pr56159 (upstream #56159 backport)"
MARKER = "pr56159 (upstream #56159)"

META = "models/kimi_k3/nvidia/kda_metadata.py"
KDA = "models/kimi_k3/nvidia/kda.py"
CHUNK = "models/kimi_k3/nvidia/ops/third_party/kda/chunk.py"
RECUR = "models/kimi_k3/nvidia/ops/third_party/kda/fused_recurrent.py"

# --- kda_metadata.py hunks --------------------------------------------------

ME_H1_ANCHOR = """@dataclass
class KimiK3KDAMetadata(GDNAttentionMetadata):
    pass
"""
ME_H1_REPL = """@dataclass
class KimiK3KDAMetadata(GDNAttentionMetadata):
    # {MARKER}: contiguous spec/non-spec token offsets; when set, the
    # attention layer slices instead of gathering/scattering.
    spec_token_start: int | None = None
    non_spec_token_start: int | None = None
""".replace("{MARKER}", MARKER)

ME_H2_ANCHOR = """        else:
            assert spec_sequence_masks_cpu is not None
            assert num_accepted_tokens is not None
            query_lens_cpu = query_start_loc_cpu.diff()
"""
ME_H2_REPL = """        else:
            assert spec_sequence_masks_cpu is not None
            assert num_accepted_tokens is not None
            spec_token_start = None
            non_spec_token_start = None
            query_lens_cpu = query_start_loc_cpu.diff()
"""

ME_H3_ANCHOR = """                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]
"""
ME_H3_REPL = """                non_spec_token_indx = index[:num_non_spec_tokens]
                spec_token_indx = index[num_non_spec_tokens:]

                # {MARKER}: when the stable partition leaves the spec /
                # non-spec tokens contiguous, record the offsets so the
                # attention layer can slice instead of gather/scatter.
                active_spec_mask = spec_sequence_masks_cpu[query_lens_cpu > 0]
                if (active_spec_mask[1:] != active_spec_mask[:-1]).sum().item() == 1:
                    spec_first = active_spec_mask[0].item()
                    spec_token_start = 0 if spec_first else num_non_spec_tokens
                    non_spec_token_start = num_spec_decode_tokens if spec_first else 0
""".replace("{MARKER}", MARKER)

ME_H4_ANCHOR = """            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            num_accepted_tokens=num_accepted_tokens,
"""
ME_H4_REPL = """            spec_token_indx=spec_token_indx,
            non_spec_token_indx=non_spec_token_indx,
            spec_token_start=spec_token_start,
            non_spec_token_start=non_spec_token_start,
            num_accepted_tokens=num_accepted_tokens,
"""

# --- kda.py hunks -----------------------------------------------------------

KDA_H1_ANCHOR = """        spec_token_indx = m.spec_token_indx
        non_spec_token_indx = m.non_spec_token_indx
"""
KDA_H1_REPL = """        spec_token_indx = m.spec_token_indx
        non_spec_token_indx = m.non_spec_token_indx
        # {MARKER}: contiguous spec/non-spec token offsets (may be None).
        spec_token_start = getattr(m, "spec_token_start", None)
        non_spec_token_start = getattr(m, "non_spec_token_start", None)
""".replace("{MARKER}", MARKER)

KDA_H2_ANCHOR = """            if m.num_prefills == 0 and m.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                g1_spec, beta_spec = g1, beta
                mixed_qkv_ns = g1_ns = beta_ns = None
            else:
                assert spec_token_indx is not None
                assert non_spec_token_indx is not None
"""
KDA_H2_REPL = """            if m.num_prefills == 0 and m.num_decodes == 0:
                mixed_qkv_spec = mixed_qkv
                g1_spec, beta_spec = g1, beta
                mixed_qkv_ns = g1_ns = beta_ns = None
            elif spec_token_start is not None:
                assert non_spec_token_start is not None
                spec_end = spec_token_start + m.num_spec_decode_tokens
                non_spec_end = (
                    non_spec_token_start + m.num_prefill_tokens + m.num_decode_tokens
                )
                spec_slice = slice(spec_token_start, spec_end)
                non_spec_slice = slice(non_spec_token_start, non_spec_end)
                mixed_qkv_spec = mixed_qkv[spec_slice]
                g1_spec, beta_spec = g1[:, spec_slice], beta[:, spec_slice]
                mixed_qkv_ns = mixed_qkv[non_spec_slice]
                g1_ns, beta_ns = g1[:, non_spec_slice], beta[:, non_spec_slice]
            else:
                assert spec_token_indx is not None
                assert non_spec_token_indx is not None
"""

KDA_H3_ANCHOR = """            spec_out = (
                core_attn_out[:, : q_spec.shape[1]]
                if m.num_prefills == 0 and m.num_decodes == 0
                else None
            )
"""
KDA_H3_REPL = """            if spec_token_start is not None:
                # {MARKER}: contiguous spec tokens write straight into the
                # output slice.
                spec_out = core_attn_out[:, spec_slice]
            else:
                spec_out = (
                    core_attn_out[:, : q_spec.shape[1]]
                    if m.num_prefills == 0 and m.num_decodes == 0
                    else None
                )
""".replace("{MARKER}", MARKER)

KDA_H4_ANCHOR = """        core_attn_out_non_spec = None
        if mixed_qkv_ns is not None:
            assert g1_ns is not None and beta_ns is not None
            if m.num_prefills > 0:
"""
KDA_H4_REPL = """        core_attn_out_non_spec = None
        non_spec_out = None
        if mixed_qkv_ns is not None:
            assert g1_ns is not None and beta_ns is not None
            if non_spec_token_start is not None:
                # {MARKER}: contiguous non-spec tokens write straight into
                # the output slice (chunk / packed-decode backends).
                non_spec_out = core_attn_out[:, non_spec_slice]
            if m.num_prefills > 0:
""".replace("{MARKER}", MARKER)

KDA_H5_ANCHOR = """                        use_qk_l2norm_in_kernel=True,
                        cu_seqlens=non_spec_query_start_loc,
                    )
"""
KDA_H5_REPL = """                        use_qk_l2norm_in_kernel=True,
                        cu_seqlens=non_spec_query_start_loc,
                        out=non_spec_out,
                    )
"""

KDA_H6_ANCHOR = """                    initial_state=recurrent_state,
                    state_indices=decode_conv_indices,
                )
"""
KDA_H6_REPL = """                    initial_state=recurrent_state,
                    state_indices=decode_conv_indices,
                    out=non_spec_out,
                )
"""

KDA_H7_ANCHOR = """        if core_attn_out_spec is not None and core_attn_out_non_spec is not None:
            core_attn_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
            core_attn_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
"""
KDA_H7_REPL = """        if core_attn_out_spec is not None and core_attn_out_non_spec is not None:
            if spec_token_start is None:
                assert spec_token_indx is not None
                assert non_spec_token_indx is not None
                core_attn_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
                core_attn_out.index_copy_(
                    1, non_spec_token_indx, core_attn_out_non_spec
                )
            elif non_spec_out is None:
                # {MARKER}: continuous layout; the spec branch wrote its
                # slice in place. Backends that cannot write in place
                # (flashkda prefill) get a plain slice copy instead of an
                # index scatter.
                core_attn_out[:, non_spec_slice] = core_attn_out_non_spec
""".replace("{MARKER}", MARKER)

# --- chunk.py hunks ---------------------------------------------------------

CH_H1_ANCHOR = """    chunk_size: int = FLA_CHUNK_SIZE,
    safe_gate: bool = False,
):
"""
CH_H1_REPL = """    chunk_size: int = FLA_CHUNK_SIZE,
    safe_gate: bool = False,
    out: torch.Tensor | None = None,  # {MARKER}
):
""".replace("{MARKER}", MARKER)

CH_H2_ANCHOR = """        A=Aqk,
        h=h,
        o=v,
        scale=scale,
"""
CH_H2_REPL = """        A=Aqk,
        h=h,
        o=v if out is None else out,
        scale=scale,
"""

CH_H3_ANCHOR = """    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
):
    chunk_size = FLA_CHUNK_SIZE
"""
CH_H3_REPL = """    lower_bound: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
):
    chunk_size = FLA_CHUNK_SIZE
"""

CH_H4_ANCHOR = """        chunk_size=chunk_size,
        safe_gate=lower_bound is not None,
    )
"""
CH_H4_REPL = """        chunk_size=chunk_size,
        safe_gate=lower_bound is not None,
        out=out,
    )
"""

CH_H5_ANCHOR = """    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **kwargs,
"""
CH_H5_REPL = """    lower_bound: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    **kwargs,
"""

CH_H6_ANCHOR = """        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
    )
    return o, final_state
"""
CH_H6_REPL = """        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        out=out,
    )
    return o, final_state
"""

# --- fused_recurrent.py hunks ----------------------------------------------

FR_H1_ANCHOR = """    state_indices: torch.Tensor,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
"""
FR_H1_REPL = """    state_indices: torch.Tensor,
    scale: float | None = None,
    out: torch.Tensor | None = None,  # {MARKER}
) -> tuple[torch.Tensor, torch.Tensor]:
""".replace("{MARKER}", MARKER)

FR_H2_ANCHOR = """    out = torch.empty((1, B, H, V), dtype=mixed_qkv.dtype, device=device)
    grid = (cdiv(V, BV), B * H)
"""
FR_H2_REPL = """    if out is None:
        out = torch.empty((1, B, H, V), dtype=mixed_qkv.dtype, device=device)
    grid = (cdiv(V, BV), B * H)
"""


def load(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError as exc:
        print(f"[{SCRIPT_NAME}] NOTE  {path}: unreadable ({exc})")
        return None


def save(path, src):
    try:
        with open(path, "w") as f:
            f.write(src)
        py_compile.compile(path, doraise=True)
        return True
    except (OSError, py_compile.PyCompileError) as exc:
        print(f"[{SCRIPT_NAME}] FAIL  {path}: write/compile failed ({exc})")
        return False


def patch_file(vroot, relpath, hunks):
    path = vroot + "/" + relpath
    src = load(path)
    if src is None:
        return False
    if MARKER in src:
        print(f"[{SCRIPT_NAME}] SKIP  {relpath} (already present)")
        return True
    ok = True
    for anchor, repl, desc in hunks:
        n = src.count(anchor)
        if n != 1:
            print(
                f"[{SCRIPT_NAME}] NOTE  {relpath}: anchor for '{desc}' found "
                f"{n}x (want 1); file skipped — stays stock."
            )
            ok = False
        else:
            src = src.replace(anchor, repl, 1)
    if not ok:
        return False
    if not save(path, src):
        return False
    print(f"[{SCRIPT_NAME}] APPLY {relpath}: {len(hunks)} hunk(s)")
    return True


def main():
    print(f"[{SCRIPT_NAME}] {TAG}")
    if len(sys.argv) != 2:
        print(f"usage: {SCRIPT_NAME}.py <VLLM_ROOT>")
        return 2
    vroot = sys.argv[1].rstrip("/")
    ok = patch_file(
        vroot,
        META,
        [
            (ME_H1_ANCHOR, ME_H1_REPL, "metadata spec/non_spec_token_start fields"),
            (ME_H2_ANCHOR, ME_H2_REPL, "build() start init"),
            (ME_H3_ANCHOR, ME_H3_REPL, "build() continuity detection"),
            (ME_H4_ANCHOR, ME_H4_REPL, "metadata constructor fields"),
        ],
    )
    ok &= patch_file(
        vroot,
        KDA,
        [
            (KDA_H1_ANCHOR, KDA_H1_REPL, "_forward reads token starts"),
            (KDA_H2_ANCHOR, KDA_H2_REPL, "contiguous slice branch"),
            (KDA_H3_ANCHOR, KDA_H3_REPL, "spec output slice"),
            (KDA_H4_ANCHOR, KDA_H4_REPL, "non-spec output slice"),
            (KDA_H5_ANCHOR, KDA_H5_REPL, "chunk prefill out="),
            (KDA_H6_ANCHOR, KDA_H6_REPL, "packed decode out="),
            (KDA_H7_ANCHOR, KDA_H7_REPL, "restore: skip scatter when continuous"),
        ],
    )
    ok &= patch_file(
        vroot,
        CHUNK,
        [
            (CH_H1_ANCHOR, CH_H1_REPL, "_chunk_kda_fwd_with_cumulative_g out param"),
            (CH_H2_ANCHOR, CH_H2_REPL, "chunk fwd o= destination"),
            (CH_H3_ANCHOR, CH_H3_REPL, "chunk_kda_with_fused_gate_fwd out param"),
            (CH_H4_ANCHOR, CH_H4_REPL, "fwd out passthrough"),
            (CH_H5_ANCHOR, CH_H5_REPL, "chunk_kda_with_fused_gate out param"),
            (CH_H6_ANCHOR, CH_H6_REPL, "fused gate out passthrough"),
        ],
    )
    ok &= patch_file(
        vroot,
        RECUR,
        [
            (FR_H1_ANCHOR, FR_H1_REPL, "packed decode out param"),
            (FR_H2_ANCHOR, FR_H2_REPL, "packed decode out allocation"),
        ],
    )
    if ok:
        print(
            f"[{SCRIPT_NAME}] done. Mixed spec/non-spec KDA batches with a "
            f"contiguous partition now slice inputs and write outputs in "
            f"place (no index_select / index_copy_ round trip)."
        )
    else:
        print(f"[{SCRIPT_NAME}] NOTE: some hunks skipped — see above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
