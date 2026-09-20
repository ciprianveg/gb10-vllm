# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Minimal subset of upstream ``vllm/model_executor/layers/fused_moe/moe_output.py``.

Upstream's full module carries the deferred-MoE-finalize protocol
(``MoEOutput``, ``convert_flashinfer_moe_output``, TRT-LLM layout
normalization) which depends on the deferred-finalize FusedMoE config
fields and the FlashInfer TRT-LLM monolithic expert wrappers — none of
which exist in this fork. The K3 latent-MoE tail fusion (upstream #53152)
only needs the ``UnfinalizedMoEOutput`` record, ported here verbatim so
the kernel-side import path matches upstream for future rebases.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class UnfinalizedMoEOutput:
    """A MoE layer's output with its top-k reduction still open.

    Attributes:
        gemm2_permuted: The permuted, un-reduced GEMM2 output. Row ``r``
            holds expert ``expert_ids[r]``'s output for the ``r``-th
            (token, expert) pair.
        expert_weights: Top-k weights for each token, shape
            ``[num_tokens, top_k]``.
        expanded_idx_to_permuted_idx: Map from (token, top-k slot) to the
            permuted row index in ``gemm2_permuted``. Negative entries mark
            dropped routes. Shape ``[num_tokens, top_k]``.
    """

    gemm2_permuted: torch.Tensor
    expert_weights: torch.Tensor
    expanded_idx_to_permuted_idx: torch.Tensor
