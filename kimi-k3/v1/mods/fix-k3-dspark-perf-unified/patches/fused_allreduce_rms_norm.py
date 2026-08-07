# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused all-reduce + residual-add + RMSNorm for eager model paths.

This recovers a fusion that vLLM's torch.compile passes would normally do but
that doesn't fire for models running eager (or under a breakable CUDA graph).
"""

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_allreduce_gemma_rms_norm import (
    _AR_RESIDUAL_RMS_NORM,
    _can_use_flashinfer,
    flashinfer_trtllm_fused_allreduce_norm,
)
from vllm.model_executor.layers.layernorm import RMSNorm


def fused_allreduce_rms_norm(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm: RMSNorm,
    *,
    prefer_b12x: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All-reduce + add residual + standard RMSNorm.

    ``hidden_states`` is the per-rank *partial* output of a row-parallel linear
    run with ``reduce_results=False``; ``norm`` is the RMSNorm applied right
    after. Returns ``(normed_output, new_residual)``, equivalent to
    ``norm(all_reduce(hidden_states), residual)``. The default fast path is
    FlashInfer's fused kernel. ``prefer_b12x`` selects an experimental B12X
    all-reduce followed by the ordinary fused residual/RMSNorm operation.
    Falls back to an explicit all-reduce when neither fast path is available.
    """
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size == 1:
        return norm(hidden_states, residual)

    if prefer_b12x:
        from vllm.distributed.device_communicators.custom_all_reduce import (
            get_active_b12x_pcie_allreduce,
        )

        custom_allreduce = get_active_b12x_pcie_allreduce()
        if custom_allreduce is not None:
            reduced = custom_allreduce.try_all_reduce_for_composed_add_rms_norm(
                hidden_states
            )
            if reduced is not None:
                return norm(reduced, residual)

    if flashinfer_trtllm_fused_allreduce_norm is not None:
        ok, max_token_num = _can_use_flashinfer(hidden_states, tp_size)
        if ok:
            norm_out = torch.empty_like(hidden_states)
            # With norm_out provided, the kernel writes the new residual
            # (all_reduce(hidden_states) + residual) into the hidden_states
            # buffer and the normalized result into norm_out.
            flashinfer_trtllm_fused_allreduce_norm(
                allreduce_in=hidden_states,
                residual=residual,
                rms_gamma=norm.weight,
                rms_eps=norm.variance_epsilon,
                world_size=tp_size,
                weight_bias=0.0,  # standard RMSNorm (Gemma would use 1.0)
                launch_with_pdl=True,
                fp32_acc=True,
                max_token_num=max_token_num,
                pattern_code=_AR_RESIDUAL_RMS_NORM,
                norm_out=norm_out,
            )
            return norm_out, hidden_states

    reduced = tensor_model_parallel_all_reduce(hidden_states)
    return norm(reduced, residual)
