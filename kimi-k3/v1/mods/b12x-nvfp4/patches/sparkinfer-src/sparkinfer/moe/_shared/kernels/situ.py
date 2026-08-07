"""Kimi K3 SiTU wrappers for the activation-specialized fused MoE backends.

The shared backend implements ``beta * tanh(gate / beta) * sigmoid(gate)``
times ``linear_beta * tanh(up / linear_beta)`` with K3's beta values.
"""

from __future__ import annotations

from typing import Tuple

from sparkinfer.moe._shared.kernels.activations import SITU
from sparkinfer.moe._shared.kernels.dynamic import MoEDynamicKernelBackend
from sparkinfer.moe._shared.kernels.micro import MoEMicroKernelBackend


class MoEMicroKernelSitu(MoEMicroKernelBackend):
    """Reserved micro specialization.

    SiTU currently stays on the fused dynamic family because the compact
    NVFP4 intermediate quantization has not passed its correctness gate.
    """

    @classmethod
    def is_supported(
        cls,
        m: int,
        k: int,
        n: int,
        num_topk: int,
        weight_E: int,
    ) -> bool:
        del m, k, n, num_topk, weight_E
        return False

    def __init__(self, *args: object, **kwargs: object):
        del args, kwargs
        raise NotImplementedError(
            "SiTU compact micro is disabled; select the fused dynamic kernel"
        )


class MoEDynamicKernelSitu(MoEDynamicKernelBackend):
    def __init__(
        self,
        sf_vec_size: int,
        mma_tiler_mn: Tuple[int, int],
        *,
        fast_math: bool = False,
        dynamic_down_scale: bool = False,
        share_input_across_experts: bool = False,
        deterministic_output: bool = False,
        num_topk: int = 1,
        swap_ab: bool = False,
        quant_recipe: str = "nvfp4",
        w4a8_repacked: bool = False,
        direct_routing: bool = False,
        work_source: str = "materialized_queue",
        materialize_intermediate: bool = False,
        swiglu_limit: float | None = None,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
    ):
        super().__init__(
            sf_vec_size,
            mma_tiler_mn,
            fast_math=fast_math,
            activation=SITU,
            dynamic_down_scale=dynamic_down_scale,
            share_input_across_experts=share_input_across_experts,
            deterministic_output=deterministic_output,
            num_topk=num_topk,
            swap_ab=swap_ab,
            quant_recipe=quant_recipe,
            w4a8_repacked=w4a8_repacked,
            direct_routing=direct_routing,
            work_source=work_source,
            materialize_intermediate=materialize_intermediate,
            swiglu_limit=swiglu_limit,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
        )


__all__ = ["MoEDynamicKernelSitu", "MoEMicroKernelSitu"]
