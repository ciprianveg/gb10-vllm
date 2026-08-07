# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K3 dense MLA draft model for DSpark speculative decoding."""

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import vllm._custom_ops as ops
from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed import get_tp_group, tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    get_draft_quant_config,
    maybe_prefix,
)
from vllm.models.common.ops import fused_allreduce_rms_norm
from vllm.models.kimi_k3.nvidia.mla import (
    KimiShardedMergedColumnParallelLinear,
    MultiHeadLatentAttention,
)
from vllm.models.kimi_k3.nvidia.model import KimiMLP
from vllm.utils.torch_utils import is_quantized_kv_cache

logger = init_logger(__name__)


class K3DSparkDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        layer_idx: int,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        quant_config = get_draft_quant_config(vllm_config)
        self.self_attn = MultiHeadLatentAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            prefix=maybe_prefix(
                prefix, f"layers.{start_layer_id + layer_idx}.self_attn"
            ),
            use_rope=True,
            non_causal_multi_token_decode=True,
        )
        # Both row-parallel outputs stay un-reduced; their all-reduces are fused
        # into the RMSNorm that follows via fused_allreduce_rms_norm.
        self.self_attn.o_proj.reduce_results = False
        self.mlp = KimiMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            reduce_results=False,
            prefix=maybe_prefix(prefix, f"layers.{start_layer_id + layer_idx}.mlp"),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        rope_cos_sin_cache: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # First layer: hidden_states is the (already reduced) embedding.
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = fused_allreduce_rms_norm(
                hidden_states,
                residual,
                self.input_layernorm,
                prefer_b12x=envs.VLLM_DSPARK_PREFER_B12X_ALLREDUCE_RMS,
            )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            rope_cos_sin_cache=rope_cos_sin_cache,
        )
        hidden_states, residual = fused_allreduce_rms_norm(
            hidden_states,
            residual,
            self.post_attention_layernorm,
            prefer_b12x=envs.VLLM_DSPARK_PREFER_B12X_ALLREDUCE_RMS,
        )
        # The MLP output is reduced by the next layer's input_layernorm (or by
        # the model's final_norm).
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


def _restore_layer_major_kv_order(
    rank_major: torch.Tensor,
    *,
    num_layers: int,
    local_kv_width: int,
    tp_size: int,
) -> torch.Tensor:
    """Convert ``[rank0(layers), rank1(layers), ...]`` to layer-major KV."""
    expected_width = tp_size * num_layers * local_kv_width
    if int(rank_major.shape[-1]) != expected_width:
        raise ValueError(
            "Unexpected gathered DSpark context-KV width: "
            f"got {rank_major.shape[-1]}, expected {expected_width}."
        )
    return (
        rank_major.unflatten(-1, (tp_size, num_layers, local_kv_width))
        .transpose(-3, -2)
        .flatten(-3)
    )


def _fill_compact_rope_cache(
    positions: torch.Tensor,
    inv_freq: torch.Tensor,
    freqs_workspace: torch.Tensor,
    cache_workspace: torch.Tensor,
    *,
    mscale: float,
) -> torch.Tensor:
    """Materialize only the RoPE rows consumed by the current draft step.

    K3 DSpark uses absolute target positions up to 1M, but a proposal step has
    at most ``max_num_batched_tokens`` rows.  Computing those rows into stable
    workspaces is exact with respect to the normal fp32 table and avoids
    retaining a 256 MiB table on every TP rank.
    """
    num_positions = int(positions.shape[0])
    if num_positions > int(freqs_workspace.shape[0]):
        raise ValueError(
            "K3 DSpark compact RoPE workspace is too small: "
            f"positions={num_positions}, capacity={freqs_workspace.shape[0]}."
        )
    freqs = freqs_workspace[:num_positions]
    cache = cache_workspace[:num_positions]
    half_dim = int(inv_freq.shape[0])
    torch.mul(positions[:, None], inv_freq[None, :], out=freqs)
    torch.cos(freqs, out=cache[:, :half_dim])
    torch.sin(freqs, out=cache[:, half_dim:])
    if mscale != 1.0:
        cache.mul_(mscale)
    return cache


class K3DSparkModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int,
        prefix: str,
    ) -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.quant_config = get_draft_quant_config(vllm_config)

        # The frozen target embedding is aliased after the draft checkpoint loads.
        self.embed_tokens: nn.Module | None = None

        # The concatenated target auxiliaries are identical on every TP rank.
        # Shard the 490 MiB BF16 output rows and gather only the small activation
        # (seven rows at bs=1) instead of storing and evaluating the complete
        # matrix redundantly on all 16 GPUs.
        self.context_proj = ColumnParallelLinear(
            self.config.target_hidden_size * self.config.num_target_layers,
            self.config.hidden_size,
            bias=False,
            gather_output=True,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "context_proj"),
        )
        self.context_norm = RMSNorm(
            self.config.hidden_size, eps=self.config.rms_norm_eps
        )

        self.layers = nn.ModuleList(
            [
                K3DSparkDecoderLayer(
                    vllm_config=vllm_config,
                    config=self.config,
                    layer_idx=layer_idx,
                    start_layer_id=start_layer_id,
                    prefix=prefix,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.final_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.markov_head = DSparkMarkovHead(
            self.config.vocab_size,
            self.config.draft_vocab_size,
            self.config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
            quant_config=self.quant_config,
        )
        self._context_kv_fusion_available: bool | None = None
        self._max_num_context_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self._compact_rope_enabled = bool(envs.VLLM_DSPARK_COMPACT_ROPE)
        if self._compact_rope_enabled:
            self._init_compact_rope()

    def _init_compact_rope(self) -> None:
        rotary_modules = [layer.self_attn.rotary_emb for layer in self.layers]
        if not rotary_modules or any(rotary is None for rotary in rotary_modules):
            raise RuntimeError("K3 DSpark compact RoPE requires rotary draft layers.")
        rotary = rotary_modules[0]
        assert rotary is not None
        if any(candidate is not rotary for candidate in rotary_modules[1:]):
            raise RuntimeError(
                "K3 DSpark compact RoPE requires all draft layers to share one "
                "immutable rotary embedding."
            )
        if not hasattr(rotary, "scaling_factor"):
            raise TypeError(
                "K3 DSpark compact RoPE currently requires a YaRN rotary "
                f"embedding, got {type(rotary).__name__}."
            )

        full_cache = rotary.cos_sin_cache
        if full_cache.dtype != torch.float32 or full_cache.ndim != 2:
            raise TypeError(
                "K3 DSpark compact RoPE requires a 2-D fp32 source cache, got "
                f"shape={tuple(full_cache.shape)}, dtype={full_cache.dtype}."
            )
        # Recompute on the same device on which the original table was built.
        # YaRN's fp32 inverse frequencies can differ slightly when evaluated on
        # CPU and then copied to CUDA, which would defeat exact replacement.
        with torch.device(full_cache.device):
            inv_freq = rotary._compute_inv_freq(  # noqa: SLF001
                rotary.scaling_factor
            )
        inv_freq = inv_freq.to(dtype=torch.float32)
        half_dim = int(inv_freq.shape[0])
        if int(full_cache.shape[1]) != 2 * half_dim:
            raise ValueError(
                "K3 DSpark compact RoPE frequency width does not match its "
                f"source table: inv_freq={half_dim}, cache={full_cache.shape[1]}."
            )

        self.register_buffer("_compact_rope_inv_freq", inv_freq, persistent=False)
        self.register_buffer(
            "_compact_rope_freqs",
            torch.empty(
                (self._max_num_context_tokens, half_dim),
                dtype=torch.float32,
                device=full_cache.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_compact_rope_cache",
            torch.empty(
                (self._max_num_context_tokens, 2 * half_dim),
                dtype=torch.float32,
                device=full_cache.device,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_compact_rope_positions",
            torch.arange(
                self._max_num_context_tokens,
                dtype=torch.int64,
                device=full_cache.device,
            ),
            persistent=False,
        )
        self._compact_rope_mscale = float(rotary.mscale)

        released_bytes = full_cache.numel() * full_cache.element_size()
        # ``get_rope`` interns this module, so replacing its registered buffer
        # releases the single table shared by all five draft layers.
        rotary.cos_sin_cache = torch.empty(
            (0, 2 * half_dim), dtype=torch.float32, device=full_cache.device
        )
        logger.info_once(
            "K3 DSpark compact RoPE released %.2f MiB/rank; materializing at "
            "most %d fp32 rows per forward.",
            released_bytes / (1024**2),
            self._max_num_context_tokens,
        )

    def _get_rope_inputs(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rotary = self.layers[0].self_attn.rotary_emb
        assert rotary is not None
        if not self._compact_rope_enabled:
            return positions, rotary.cos_sin_cache
        cache = _fill_compact_rope_cache(
            positions,
            self._compact_rope_inv_freq,
            self._compact_rope_freqs,
            self._compact_rope_cache,
            mscale=self._compact_rope_mscale,
        )
        return self._compact_rope_positions[: positions.shape[0]], cache

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        assert self.embed_tokens is not None
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.context_norm(self.context_proj(hidden_states))

    @torch.inference_mode()
    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        """Project target-derived context into each draft layer's latent cache."""
        if self._context_kv_fusion_available is None:
            self._build_fused_context_kv_buffers()
        if self._context_kv_fusion_available:
            self._precompute_fused_context_kv(
                context_states, context_positions, context_slot_mapping
            )
            return

        # Quantized fallback. Directly invoking the projection modules preserves
        # their quantization methods, at the cost of also computing unused Q rows.
        rope_positions, rope_cos_sin_cache = self._get_rope_inputs(context_positions)
        for layer_idx, layer in enumerate(self.layers):
            attn = layer.self_attn
            assert attn.fused_qkv_a_proj is not None
            assert attn.q_lora_rank is not None
            assert attn.rotary_emb is not None
            qkv_lora = attn.fused_qkv_a_proj(context_states)[0]
            kv_lora = qkv_lora[..., attn.q_lora_rank :]
            kv_c, k_pe = kv_lora.split(
                [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1
            )
            kv_c = attn.kv_a_layernorm(kv_c)
            k_pe = k_pe.unsqueeze(1)
            # DeepSeek YaRN's FlashInfer path requires paired Q/K tensors.
            # The vLLM CUDA op supports rotating one tensor in place and
            # consumes the same (possibly scaled fp32) cos/sin cache.
            rotary_emb = attn.rotary_emb
            ops.rotary_embedding(
                rope_positions,
                k_pe,
                None,
                rotary_emb.head_size,
                rope_cos_sin_cache,
                rotary_emb.is_neox_style,
            )

            slot_mapping = (
                context_slot_mapping[layer_idx]
                if isinstance(context_slot_mapping, (list, tuple))
                else context_slot_mapping
            )
            if slot_mapping is None:
                continue
            attn.impl.do_kv_cache_update(
                kv_c,
                k_pe,
                attn.kv_cache,
                slot_mapping,
                attn.kv_cache_dtype,
                attn._k_scale,
            )

    def _build_fused_context_kv_buffers(self) -> None:
        """Build a cross-layer KV-only A projection after checkpoint loading."""
        attentions = [layer.self_attn for layer in self.layers]
        if not attentions or any(
            attn.fused_qkv_a_proj is None
            or not hasattr(attn.fused_qkv_a_proj, "weight")
            for attn in attentions
        ):
            self._context_kv_fusion_available = False
            return

        attn0 = attentions[0]
        assert attn0.q_lora_rank is not None
        kv_width = attn0.kv_lora_rank + attn0.qk_rope_head_dim
        sharded = all(
            isinstance(
                attn.fused_qkv_a_proj,
                KimiShardedMergedColumnParallelLinear,
            )
            for attn in attentions
        )
        if sharded:
            tp_size = int(attn0.fused_qkv_a_proj.tp_size)
            if attn0.q_lora_rank % tp_size or kv_width % tp_size:
                self._context_kv_fusion_available = False
                return
            q_weight_offset = attn0.q_lora_rank // tp_size
            stored_kv_width = kv_width // tp_size
        else:
            # A mixed replicated/sharded layer set has no single gather policy.
            if any(
                isinstance(
                    attn.fused_qkv_a_proj,
                    KimiShardedMergedColumnParallelLinear,
                )
                for attn in attentions
            ):
                self._context_kv_fusion_available = False
                return
            tp_size = 1
            q_weight_offset = attn0.q_lora_rank
            stored_kv_width = kv_width
        kv_weights = []
        for attn in attentions:
            assert attn.q_lora_rank is not None
            assert (
                attn.q_lora_rank == attn0.q_lora_rank
                and attn.kv_lora_rank == attn0.kv_lora_rank
                and attn.qk_rope_head_dim == attn0.qk_rope_head_dim
                and attn.kv_a_layernorm.variance_epsilon
                == attn0.kv_a_layernorm.variance_epsilon
            ), "All MLA DSpark layers must share their latent KV geometry."
            weight = attn.fused_qkv_a_proj.weight.detach()
            # Serialized quantized qkv-a weights cannot be consumed directly
            # by F.linear. Selective online MXFP8 leaves this small projection
            # in BF16, allowing the cross-layer KV-only fusion to stay active.
            if weight.element_size() < 2 or not weight.dtype.is_floating_point:
                self._context_kv_fusion_available = False
                return
            if int(weight.shape[0]) < q_weight_offset + stored_kv_width:
                self._context_kv_fusion_available = False
                return
            kv_weights.append(weight.narrow(0, q_weight_offset, stored_kv_width))

        # Replicated layout: [L * kv_width, hidden]. Sharded layout keeps only
        # [L * (kv_width / TP), hidden] and restores layer-major ordering with
        # one all-gather after the fused local GEMM.
        self._fused_context_kv_weight = torch.cat(kv_weights, dim=0)
        self._context_kv_sharded = sharded
        self._context_kv_tp_size = tp_size
        self._context_kv_stored_width = stored_kv_width
        self._context_kv_norm_weights = torch.stack(
            [attn.kv_a_layernorm.weight.detach() for attn in attentions], dim=0
        ).contiguous()
        self._num_context_layers = len(attentions)
        self._context_kv_width = kv_width
        self._context_kv_lora_rank = attn0.kv_lora_rank
        self._context_rope_dim = attn0.qk_rope_head_dim
        self._context_rms_norm_eps = attn0.kv_a_layernorm.variance_epsilon
        self._context_positions_repeated = torch.empty(
            self._num_context_layers * self._max_num_context_tokens,
            dtype=torch.int64,
            device=self._fused_context_kv_weight.device,
        )
        self._context_kv_fusion_available = True

    def _precompute_fused_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None,
    ) -> None:
        num_ctx = context_states.shape[0]
        num_layers = self._num_context_layers

        # One KV-only GEMM replaces five full Q+KV GEMMs. For K3 this projects
        # 5*576 rows rather than 5*2112 rows (72.7% fewer A-projection FLOPs).
        all_kv = F.linear(context_states, self._fused_context_kv_weight)
        if self._context_kv_sharded:
            all_kv = _restore_layer_major_kv_order(
                tensor_model_parallel_all_gather(all_kv),
                num_layers=num_layers,
                local_kv_width=self._context_kv_stored_width,
                tp_size=self._context_kv_tp_size,
            )
        all_kv = all_kv.view(num_ctx, num_layers, self._context_kv_width)
        all_kv_c = all_kv[..., : self._context_kv_lora_rank]
        all_k_pe = all_kv[..., self._context_kv_lora_rank :]

        # Layer-major layout lets the 2-D RMSNorm weights select a distinct row
        # for each draft layer in one grouped kernel.
        all_kv_c = all_kv_c.permute(1, 0, 2).contiguous()
        all_kv_c_normed = torch.empty_like(all_kv_c)
        ops.rms_norm(
            all_kv_c_normed,
            all_kv_c,
            self._context_kv_norm_weights,
            self._context_rms_norm_eps,
        )

        all_k_pe = all_k_pe.permute(1, 0, 2).contiguous()
        all_k_pe_flat = all_k_pe.view(num_layers * num_ctx, 1, self._context_rope_dim)
        repeated_positions = self._context_positions_repeated[: num_layers * num_ctx]
        rope_positions, rope_cos_sin_cache = self._get_rope_inputs(context_positions)
        repeated_positions.view(num_layers, num_ctx).copy_(rope_positions)
        # Keep the single-tensor context RoPE on vLLM's optimized CUDA op;
        # DeepSeek YaRN's FlashInfer wrapper assumes a non-null key tensor.
        rotary_emb = self.layers[0].self_attn.rotary_emb
        assert rotary_emb is not None
        ops.rotary_embedding(
            repeated_positions,
            all_k_pe_flat,
            None,
            rotary_emb.head_size,
            rope_cos_sin_cache,
            rotary_emb.is_neox_style,
        )
        all_k_pe = all_k_pe_flat.view(num_layers, num_ctx, 1, self._context_rope_dim)

        if context_slot_mapping is None:
            return

        cache_layers = [layer.self_attn for layer in self.layers]
        if (
            not is_quantized_kv_cache(cache_layers[0].kv_cache_dtype)
            and self._has_uniform_block_layout(cache_layers)
            and (
                isinstance(context_slot_mapping, torch.Tensor)
                or all(s is not None for s in context_slot_mapping)
            )
        ):
            # Grouped context KV insert only supports unquantized (bf16) KV cache
            # and assumes that all layers share the same block layout.

            if isinstance(context_slot_mapping, (list, tuple)):
                per_layer_slot_mappings = [
                    s for s in context_slot_mapping if s is not None
                ]
                if len({s.data_ptr() for s in per_layer_slot_mappings}) == 1:
                    # All rows alias to the same slot mapping.
                    slot_mapping = (
                        per_layer_slot_mappings[0].unsqueeze(0).expand(num_layers, -1)
                    )
                else:
                    slot_mapping = torch.stack(per_layer_slot_mappings, dim=0)
            else:
                # Broadcast the single shared context_slot_mapping tensor.
                slot_mapping = context_slot_mapping.unsqueeze(0).expand(num_layers, -1)

            ref_cache = cache_layers[0].kv_cache
            ops.concat_and_cache_mla_grouped(
                all_kv_c_normed,
                all_k_pe.squeeze(2),
                self._get_context_kv_cache_ptrs(cache_layers),
                slot_mapping,
                ref_cache.size(1),
                ref_cache.stride(0),
                ref_cache.stride(1),
            )
            return

        for layer_idx, layer in enumerate(self.layers):
            slot_mapping = (
                context_slot_mapping[layer_idx]
                if isinstance(context_slot_mapping, (list, tuple))
                else context_slot_mapping
            )
            if slot_mapping is None:
                continue
            attn = layer.self_attn
            attn.impl.do_kv_cache_update(
                all_kv_c_normed[layer_idx],
                all_k_pe[layer_idx],
                attn.kv_cache,
                slot_mapping,
                attn.kv_cache_dtype,
                attn._k_scale,
            )

    def _has_uniform_block_layout(
        self,
        cache_layers: list[MultiHeadLatentAttention],
    ) -> bool:
        if not hasattr(self, "_layers_share_kv_block_layout"):
            ref_cache = cache_layers[0].kv_cache
            self._layers_share_kv_block_layout = all(
                cl.kv_cache.size(1) == ref_cache.size(1)
                and cl.kv_cache.stride(0) == ref_cache.stride(0)
                and cl.kv_cache.stride(1) == ref_cache.stride(1)
                for cl in cache_layers
            )
        return self._layers_share_kv_block_layout

    def _get_context_kv_cache_ptrs(
        self,
        cache_layers: list[MultiHeadLatentAttention],
    ) -> torch.Tensor:
        # The per-layer KV cache base pointers are stable after allocation, so
        # build the pointer array once and return it on every call.
        if not hasattr(self, "_context_cache_ptrs"):
            ref_cache = cache_layers[0].kv_cache
            cache_ptrs = torch.tensor(
                [cl.kv_cache.data_ptr() for cl in cache_layers],
                dtype=torch.int64,
                device=ref_cache.device,
            )
            self._context_cache_ptrs = cache_ptrs
        return self._context_cache_ptrs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids)

        hidden_states = inputs_embeds
        residual = None
        rope_positions, rope_cos_sin_cache = self._get_rope_inputs(positions)
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=rope_positions,
                hidden_states=hidden_states,
                residual=residual,
                rope_cos_sin_cache=rope_cos_sin_cache,
            )
        hidden_states, _ = fused_allreduce_rms_norm(
            hidden_states,
            residual,
            self.final_norm,
            prefer_b12x=envs.VLLM_DSPARK_PREFER_B12X_ALLREDUCE_RMS,
        )
        return hidden_states


class K3DSparkForCausalLM(nn.Module):
    has_own_embed_tokens = False
    has_own_lm_head = False
    draft_id_to_target_id = None
    checkpoint_skip_substrs = ("confidence_head", "embed_tokens", "lm_head")

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"": "model."},
        orig_to_new_stacked={
            ".gate_proj": (".gate_up_proj", 0),
            ".up_proj": (".gate_up_proj", 1),
            ".q_a_proj": (".fused_qkv_a_proj", 0),
            ".kv_a_proj_with_mqa": (".fused_qkv_a_proj", 1),
        },
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        assert vllm_config.speculative_config is not None
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = K3DSparkModel(
            vllm_config=vllm_config,
            start_layer_id=target_layer_num,
            prefix=maybe_prefix(prefix, "model"),
        )

        # Assigned by load_dspark_model from the target. Keeping no placeholder
        # avoids a transient full-vocabulary allocation for this 163k-vocab model.
        self.lm_head: nn.Module | None = None
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        self._b12x_dspark_argmax_enabled = bool(envs.VLLM_KIMI_K3_B12X_DSPARK_ARGMAX)
        self._b12x_dspark_argmax_max_batch = min(
            vllm_config.scheduler_config.max_num_seqs, 8
        )
        self._b12x_dspark_argmax_runtime: Any = None
        self._b12x_dspark_argmax_output: torch.Tensor | None = None

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.layer_name for layer in self.model.layers]

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert self.lm_head is not None
        return self.logits_processor(self.lm_head, hidden_states)

    def compute_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.compute_logits(hidden_states)

    def supports_local_draft_argmax(self) -> bool:
        """Validate the no-full-vocab-gather path for sharded Markov sampling."""
        markov_head = self.model.markov_head
        if not markov_head.shard_across_tp:
            return False
        if not isinstance(self.lm_head, VocabParallelEmbedding):
            raise TypeError(
                "K3 local draft argmax requires a vocab-parallel target LM head."
            )
        markov_w2 = markov_head.markov_w2
        if not isinstance(markov_w2, VocabParallelEmbedding):
            raise TypeError(
                "K3 local draft argmax requires a vocab-parallel Markov W2."
            )
        layout_fields = (
            "tp_size",
            "tp_rank",
            "num_embeddings",
            "org_vocab_size",
            "num_embeddings_padded",
            "num_embeddings_per_partition",
        )
        mismatches = [
            field
            for field in layout_fields
            if getattr(self.lm_head, field) != getattr(markov_w2, field)
        ]
        if mismatches or self.lm_head.shard_indices != markov_w2.shard_indices:
            raise ValueError(
                "Target LM head and Markov W2 use different TP vocabulary "
                f"layouts (mismatches={mismatches})."
            )
        if self.logits_processor.soft_cap is not None:
            raise ValueError(
                "K3 local draft argmax cannot combine base and Markov logits "
                "exactly when logit soft-capping is enabled."
            )
        if self.logits_processor.scale <= 0.0:
            raise ValueError("K3 local draft sampling requires a positive logit scale.")
        return True

    def compute_local_draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert isinstance(self.lm_head, VocabParallelEmbedding)
        return self.logits_processor._apply_head(self.lm_head, hidden_states, None)

    def compute_local_markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.local_bias(markov_embed, self.logits_processor)

    def _get_b12x_dspark_argmax(
        self,
        base_logits: torch.Tensor,
    ) -> Any:
        runtime = self._b12x_dspark_argmax_runtime
        if runtime is not None:
            return runtime

        from sparkinfer.comm.pcie import VocabParallelArgmax

        tp_group = get_tp_group()
        runtime = VocabParallelArgmax.from_exchange_group(
            # IPC handle exchange is metadata-only. Keep it off the primary
            # TP NCCL communicator whose collective order is captured by the
            # target and draft CUDA graphs.
            exchange_group=tp_group.cpu_group,
            device=base_logits.device,
            local_vocab_size=base_logits.shape[-1],
            max_batch_size=self._b12x_dspark_argmax_max_batch,
        )
        self._b12x_dspark_argmax_output = torch.empty(
            self._b12x_dspark_argmax_max_batch,
            dtype=torch.int64,
            device=base_logits.device,
        )
        self._b12x_dspark_argmax_runtime = runtime
        logger.info_once(
            "Kimi-K3 DSpark uses B12X TP16 fused BF16 add/global argmax "
            "for up to %d requests.",
            self._b12x_dspark_argmax_max_batch,
        )
        return runtime

    def sample_local_draft_logits(
        self,
        base_logits: torch.Tensor,
        markov_bias: torch.Tensor,
    ) -> torch.Tensor:
        assert isinstance(self.lm_head, VocabParallelEmbedding)
        batch_size = int(base_logits.shape[0])
        use_b12x = (
            getattr(self, "_b12x_dspark_argmax_enabled", False)
            and self.lm_head.tp_size == 16
            and base_logits.dtype == torch.bfloat16
            and markov_bias.dtype == torch.bfloat16
            and 0 < batch_size <= self._b12x_dspark_argmax_max_batch
        )
        if use_b12x:
            runtime = self._get_b12x_dspark_argmax(base_logits)
            assert self._b12x_dspark_argmax_output is not None
            output = self._b12x_dspark_argmax_output[:batch_size]
            return runtime.fused_add_argmax(base_logits, markov_bias, out=output)

        # A full gather is faster than a tiny pair gather on this TP16 PCIe
        # topology (0.55 vs 0.79 ms for all seven Markov steps). Base and
        # Markov logits are added locally first, so this remains one collective
        # per step and no collective is captured in the draft CUDA graph.
        logits = base_logits + markov_bias
        logits = tensor_model_parallel_all_gather(logits, dim=-1)
        logits = logits[..., : self.logits_processor.org_vocab_size]
        return logits.argmax(dim=-1)

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_bias(self, markov_embed: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.bias(markov_embed, self.logits_processor)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # confidence_head is training-only. The frozen target embedding and LM
        # head are shared after this draft-specific checkpoint is loaded.
        loader = AutoWeightsLoader(
            self,
            skip_substrs=list(self.checkpoint_skip_substrs),
        )
        loaded_weights = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
        # The replicated Markov embedding is not a LinearBase, so finalize its
        # online MXFP8 representation explicitly after its BF16 checkpoint row
        # has loaded.  This only affects draft acceptance; target verification
        # and final output probabilities remain those of the full model.
        self.model.markov_head.process_weights_after_loading()
        self.model._build_fused_context_kv_buffers()
        return loaded_weights
