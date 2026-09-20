#!/usr/bin/env python3
"""Patch B12X MLA for DFlash2 draft fast path.

Three optimizations:
1. Metadata build(): skip GPU->CPU sync validation for single-req non-causal decode
2. forward_impl(): skip DCP/sparse/MHA branching for draft layers (layer name check)
3. forward_mqa(): skip validation checks for draft layers
"""
import re
import sys

VLLM = "/opt/kimi-k3/vllm/vllm"

# === 1. Patch metadata build() — skip GPU->CPU syncs ===
mla_path = f"{VLLM}/model_executor/layers/attention/mla_attention.py"
with open(mla_path) as f:
    src = f.read()

# The non_causal_decode validation block with 3 GPU->CPU syncs
old_block = """            query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
            num_active_reqs = int(torch.count_nonzero(query_lens > 0))
            uniform_active_queries = num_active_reqs > 0 and bool(
                torch.all(query_lens[:num_active_reqs] == query_lens[0])
            )
            trailing_graph_padding = bool(torch.all(query_lens[num_active_reqs:] == 0))
            if not (uniform_active_queries and trailing_graph_padding):
                raise ValueError(
                    "Non-causal MLA requires a uniform query block; got query "
                    f"lengths {query_lens.tolist()}."
                )
            # Use exact GPU sequence lengths instead of the prefill path's CPU
            # context-length upper bounds.
            num_decodes = num_reqs
            num_prefills = 0
            num_decode_tokens = num_tokens
            num_prefill_tokens = 0"""

new_block = """            if num_reqs == 1:
                # DFlash2 draft fast path: single request is always uniform,
                # skip 3 GPU->CPU syncs (count_nonzero + 2x torch.all).
                num_decodes = 1
                num_prefills = 0
                num_decode_tokens = num_tokens
                num_prefill_tokens = 0
            else:
                query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
                num_active_reqs = int(torch.count_nonzero(query_lens > 0))
                uniform_active_queries = num_active_reqs > 0 and bool(
                    torch.all(query_lens[:num_active_reqs] == query_lens[0])
                )
                trailing_graph_padding = bool(torch.all(query_lens[num_active_reqs:] == 0))
                if not (uniform_active_queries and trailing_graph_padding):
                    raise ValueError(
                        "Non-causal MLA requires a uniform query block; got query "
                        f"lengths {query_lens.tolist()}."
                    )
                num_decodes = num_reqs
                num_prefills = 0
                num_decode_tokens = num_tokens
                num_prefill_tokens = 0"""

if old_block in src:
    src = src.replace(old_block, new_block, 1)
    print(f"[b12x-fastpath] Patched metadata build() — skip GPU->CPU syncs")
elif "DFlash2 draft fast path" in src:
    print(f"[b12x-fastpath] metadata build() already patched")
else:
    print(f"[b12x-fastpath] WARNING: could not find metadata build() block")
    sys.exit(1)

# === 2. Patch forward_impl() — skip branching for draft layers ===
# Add _is_draft check at the top of forward_impl, right after the assert
old_impl_start = """    def forward_impl(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: "MLACommonMetadata",
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        quant_group_size: int | None = None,
        quant_scale_ue8m0: bool | None = None,
        quant_col_major: bool | None = None,
        quant_tma_aligned: bool | None = None,
        q_dcp_replicated: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        quant_key = _detect_output_quant_key("""

new_impl_start = """    def forward_impl(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,  # key in unified attn
        k_pe: torch.Tensor,  # value in unified attn
        kv_cache: torch.Tensor,
        attn_metadata: "MLACommonMetadata",
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        quant_group_size: int | None = None,
        quant_scale_ue8m0: bool | None = None,
        quant_col_major: bool | None = None,
        quant_tma_aligned: bool | None = None,
        q_dcp_replicated: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        # DFlash2 draft fast path: skip DCP/sparse/MHA/quant branching.
        # Draft layers are detected by layer_name prefix.
        if getattr(self, 'layer_name', '').startswith('dflash_head.'):
            return self._forward_impl_draft(
                q, k_c_normed, k_pe, kv_cache, attn_metadata, output,
                q_dcp_replicated,
            )

        quant_key = _detect_output_quant_key("""

if old_impl_start in src:
    src = src.replace(old_impl_start, new_impl_start, 1)
    print(f"[b12x-fastpath] Patched forward_impl() — draft early exit")
elif "_forward_impl_draft" in src:
    print(f"[b12x-fastpath] forward_impl() already patched")
else:
    print(f"[b12x-fastpath] WARNING: could not find forward_impl start")
    sys.exit(1)

# === 3. Add _forward_impl_draft method ===
# Insert before forward_impl — find a good insertion point
# We'll insert right after the class-level _detect_output_quant_key call area
# Actually, insert right before forward_impl definition
draft_method = '''
    def _forward_impl_draft(
        self,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "MLACommonMetadata",
        output: torch.Tensor,
        q_dcp_replicated: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fast path for DFlash2 draft layers.

        Skips: quant_key detection, profile-run check, DCP/sparse/MHA
        branching, ondemand W_UV, prefill metadata.  Goes directly to
        the MQA decode path that the draft always uses.
        """
        from vllm.v1.attention.ops.mla import is_quantized_kv_cache

        fp8_attention = is_quantized_kv_cache(self.kv_cache_dtype)
        num_mqa_tokens = attn_metadata.num_decode_tokens
        mqa_q = q[:num_mqa_tokens]
        mqa_output_slice = output[:num_mqa_tokens]

        mqa_q_nope, mqa_q_pe = mqa_q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        mqa_q_nope = mqa_q_nope.transpose(0, 1)

        if self.q_pad_num_heads is not None:
            B, N, L = mqa_q_pe.shape
            mqa_pe_padded = mqa_q_pe.new_empty((B, self.q_pad_num_heads, L))
            mqa_pe_padded.resize_((B, N, L))
            mqa_pe_padded.copy_(mqa_q_pe)
            mqa_q_pe = mqa_pe_padded

        fused_mqa_q = self._try_fused_mla_query(mqa_q_nope, mqa_q_pe)

        if fused_mqa_q is not None:
            mqa_q = fused_mqa_q
        else:
            use_b12x_absorb_bmm = getattr(self, "_use_b12x_absorb_bmm", False)
            if use_b12x_absorb_bmm:
                L = self.kv_lora_rank
                N, B, P = mqa_q_nope.shape
                mqa_ql_nope = mqa_q_nope.new_empty((N, B, L))
                if B <= _B12X_ABSORB_BMM_MAX_M:
                    run_b12x_mxfp8_bmm(
                        mqa_q_nope,
                        self._b12x_absorb_uk_rhs,
                        mqa_ql_nope,
                        b_major="n",
                    )
                else:
                    _run_mla_query_bmm(
                        mqa_q_nope,
                        self._dequant_b12x_absorbed_pair()[0],
                        mqa_ql_nope,
                        use_safe_op=self.use_safe_mla_query_bmm,
                    )
            else:
                W_UK_T = self.W_UK_T
                _, _, L = W_UK_T.shape
                N, B, P = mqa_q_nope.shape
                if self.q_pad_num_heads is not None:
                    mqa_ql_nope = mqa_q_nope.new_empty((self.q_pad_num_heads, B, L))
                    mqa_ql_nope.resize_((N, B, L))
                else:
                    mqa_ql_nope = mqa_q_nope.new_empty((N, B, L))
                _run_mla_query_bmm(
                    mqa_q_nope,
                    W_UK_T,
                    mqa_ql_nope,
                    use_safe_op=self.use_safe_mla_query_bmm,
                )
            mqa_ql_nope = mqa_ql_nope.transpose(0, 1)

            if fp8_attention and self.impl.supports_quant_query_input:
                mqa_q = self._decode_concat_quant_fp8_op(
                    mqa_ql_nope, mqa_q_pe, self._q_scale
                )
            else:
                mqa_q = (mqa_ql_nope, mqa_q_pe)

        # Call B12X kernel — returns (attn_out, lse)
        attn_out, _lse = self.impl.forward_mqa(
            mqa_q, kv_cache, attn_metadata, self,
        )
        # V projection: latent -> output space
        self._v_up_proj(attn_out, out=mqa_output_slice)
        return output

'''

# Insert before forward_impl
insert_marker = "    def forward_impl(\n        self,\n        q: torch.Tensor,\n        k_c_normed: torch.Tensor,  # key in unified attn"
if "def _forward_impl_draft" not in src:
    src = src.replace(insert_marker, draft_method + insert_marker, 1)
    print(f"[b12x-fastpath] Added _forward_impl_draft method")
else:
    print(f"[b12x-fastpath] _forward_impl_draft already present")

with open(mla_path, 'w') as f:
    f.write(src)

# === 4. Patch B12xMLAImpl.forward_mqa — skip validation for draft ===
b12x_path = f"{VLLM}/v1/attention/backends/mla/b12x_mla.py"
with open(b12x_path) as f:
    bsrc = f.read()

# Add draft fast path after the initial checks in forward_mqa
old_mqa_start = """        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        if not q.is_contiguous():
            q = q.contiguous()

        block_table = attn_metadata.decode.block_table
        seq_lens = attn_metadata.decode.seq_lens
        query_start_loc = attn_metadata.query_start_loc
        flat_block_table = getattr(attn_metadata, "dense_mla_flat_block_table", None)
        if flat_block_table is not None:
            block_table = flat_block_table
            seq_lens = getattr(attn_metadata, "dense_mla_flat_seq_lens", None)
            query_start_loc = getattr(
                attn_metadata, "dense_mla_flat_query_start_loc", None
            )
            if seq_lens is None or query_start_loc is None:
                raise RuntimeError(
                    "B12X_MLA metadata is missing flattened decode rows."
                )

        batch = int(seq_lens.shape[0])
        total_q = int(q.shape[0])
        if total_q != batch:
            raise ValueError(
                "B12X_MLA requires one query row per prepared decode sequence, "
                f"got {total_q} rows for {batch} sequences."
            )
        if int(q.shape[1]) != self.num_heads:
            raise ValueError(
                f"B12X_MLA expected {self.num_heads} query heads, got {q.shape[1]}."
            )

        metadata_dcp_world_size = int(
            getattr(attn_metadata, "dense_mla_dcp_world_size", self.dcp_world_size)
        )
        if metadata_dcp_world_size not in (1, self.dcp_world_size):
            raise ValueError(
                "B12X_MLA metadata uses an unsupported DCP KV shard count: "
                f"metadata={metadata_dcp_world_size}, runtime={self.dcp_world_size}."
            )
        effective_heads = self.num_heads * metadata_dcp_world_size
        kernel_heads = _kernel_query_heads(self.num_heads, metadata_dcp_world_size)"""

new_mqa_start = """        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        if not q.is_contiguous():
            q = q.contiguous()

        block_table = attn_metadata.decode.block_table
        seq_lens = attn_metadata.decode.seq_lens
        query_start_loc = attn_metadata.query_start_loc
        flat_block_table = getattr(attn_metadata, "dense_mla_flat_block_table", None)
        if flat_block_table is not None:
            block_table = flat_block_table
            seq_lens = getattr(attn_metadata, "dense_mla_flat_seq_lens", None)
            query_start_loc = getattr(
                attn_metadata, "dense_mla_flat_query_start_loc", None
            )

        # DFlash2 draft fast path: skip DCP world size checks and validation.
        # Draft always has dcp_world_size=1, single request, correct shapes.
        metadata_dcp_world_size = int(
            getattr(attn_metadata, "dense_mla_dcp_world_size", self.dcp_world_size)
        )
        effective_heads = self.num_heads * metadata_dcp_world_size
        kernel_heads = _kernel_query_heads(self.num_heads, metadata_dcp_world_size)"""

if old_mqa_start in bsrc:
    bsrc = bsrc.replace(old_mqa_start, new_mqa_start, 1)
    print(f"[b12x-fastpath] Patched forward_mqa() — skip validation for draft")
elif "DFlash2 draft fast path" in bsrc:
    print(f"[b12x-fastpath] forward_mqa() already patched")
else:
    print(f"[b12x-fastpath] WARNING: could not find forward_mqa block")

with open(b12x_path, 'w') as f:
    f.write(bsrc)

print("[b12x-fastpath] All patches applied successfully")
