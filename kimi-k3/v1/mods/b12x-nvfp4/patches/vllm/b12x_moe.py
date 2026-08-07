# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B12X modular fused-MoE backend for DeepSeek V4 native MXFP4 weights."""

from typing import Any, cast

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform


def _run_b12x_moe_fp4(
    *,
    a: torch.Tensor,
    a1_gscale: torch.Tensor,
    w1_fp4: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_alphas: torch.Tensor,
    a2_gscale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_alphas: torch.Tensor,
    output: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    apply_router_weight_on_input: bool,
    input_scales_are_reciprocal: bool,
    input_scales_static: bool,
    activation: str,
    quant_mode: str,
    unit_scale_contract: bool,
    source_format: str,
    w13_layout: str,
    prepared_w4a16: Any,
    swiglu_limit: float | None,
) -> None:
    """Call b12x MoE with caller-owned live scratch."""
    from b12x.integration.tp_moe import (
        TPMoEScratchCaps,
        b12x_moe_fp4,
        plan_tp_moe_scratch,
    )

    weight_E = int(prepared_w4a16.num_experts)
    n = int(prepared_w4a16.intermediate_size)
    tokens = int(a.shape[0])
    plan = plan_tp_moe_scratch(
        TPMoEScratchCaps(
            max_tokens=tokens,
            weight_E=weight_E,
            k=int(a.shape[1]),
            n=n,
            num_topk=int(topk_ids.shape[1]),
            device=a.device,
            dtype=a.dtype,
            core_token_counts=(tokens,),
            route_num_experts=0,
            quant_mode=quant_mode,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            swiglu_limit=swiglu_limit,
            source_format=source_format,
            w13_layout=w13_layout,
            frozen=True,
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=a.device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    binding = plan.bind(
        scratch=scratch,
        a=a,
        a1_gscale=a1_gscale,
        w1_fp4=w1_fp4,
        w1_blockscale=w1_blockscale,
        w1_alphas=w1_alphas,
        a2_gscale=a2_gscale,
        w2_fp4=w2_fp4,
        w2_blockscale=w2_blockscale,
        w2_alphas=w2_alphas,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        apply_router_weight_on_input=apply_router_weight_on_input,
        output=output,
        input_scales_are_reciprocal=input_scales_are_reciprocal,
        input_scales_static=input_scales_static,
        activation=activation,
        quant_mode=quant_mode,
        unit_scale_contract=unit_scale_contract,
        source_format=source_format,
        w13_layout=w13_layout,
        prepared_w4a16=prepared_w4a16,
        swiglu_limit=swiglu_limit,
    )
    b12x_moe_fp4(binding=binding)


def _b12x_activation_name(activation: MoEActivation) -> str:
    if activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI):
        return "silu"
    if activation == MoEActivation.RELU2:
        return "relu2"
    return activation.value


def _prepare_b12x_fp4_moe_weights(**kwargs):
    from b12x.integration import prepare_b12x_fp4_moe_weights

    return prepare_b12x_fp4_moe_weights(**kwargs)


def _replace_parameter_with_empty(
    layer: torch.nn.Module,
    param_name: str,
) -> torch.Tensor | None:
    param = getattr(layer, param_name, None)
    if not isinstance(param, torch.Tensor):
        return None
    empty = torch.empty((0,), dtype=param.dtype, device=param.device)
    replace_parameter(layer, param_name, empty)
    return getattr(layer, param_name)


def _set_quant_config_weight_scale(
    quant_config: FusedMoEQuantConfig,
    weight_name: str,
    scale: torch.Tensor,
) -> None:
    desc = getattr(quant_config, weight_name, None)
    if desc is not None and hasattr(desc, "scale"):
        desc.scale = scale
        return

    public_name = "w1_scale" if weight_name == "_w1" else "w2_scale"
    if hasattr(quant_config, public_name):
        setattr(quant_config, public_name, scale)


def _maybe_release_cuda_cache(device: torch.device) -> None:
    if device.type != "cuda" or _is_current_stream_capturing():
        return
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        accelerator.empty_cache()
    else:
        torch.cuda.empty_cache()


def _raise_if_capture_copy_required(tensor: torch.Tensor, description: str) -> None:
    if tensor.device.type != "cuda" or not _is_current_stream_capturing():
        return
    raise RuntimeError(
        f"B12X MoE {description} would allocate during CUDA graph capture"
    )


def _is_current_stream_capturing() -> bool:
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return False
    is_capturing = getattr(cuda, "is_current_stream_capturing", None)
    return bool(is_capturing is not None and is_capturing())


def _normalize_b12x_moe_topk_ids(topk_ids: torch.Tensor) -> torch.Tensor:
    if topk_ids.dtype != torch.int32:
        _raise_if_capture_copy_required(topk_ids, "topk_ids dtype normalization")
        topk_ids = topk_ids.to(torch.int32)
    if not topk_ids.is_contiguous():
        _raise_if_capture_copy_required(topk_ids, "topk_ids contiguity normalization")
        topk_ids = topk_ids.contiguous()
    return topk_ids


def _normalize_b12x_moe_topk_weights(topk_weights: torch.Tensor) -> torch.Tensor:
    if topk_weights.dtype != torch.float32:
        _raise_if_capture_copy_required(
            topk_weights,
            "topk_weights dtype normalization",
        )
        topk_weights = topk_weights.to(torch.float32)
    if not topk_weights.is_contiguous():
        _raise_if_capture_copy_required(
            topk_weights,
            "topk_weights contiguity normalization",
        )
        topk_weights = topk_weights.contiguous()
    return topk_weights


def _has_b12x() -> bool:
    try:
        from b12x.integration.tp_moe import b12x_moe_fp4  # noqa: F401

        return True
    except ImportError:
        return False


class B12xExperts(mk.FusedMoEExpertsModular):
    """Native DeepSeek V4 MXFP4 MoE backend powered by b12x kernels."""

    def __init__(
        self,
        moe_config: mk.FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config, quant_config)

        assert quant_config.weight_quant_dtype == "mxfp4", (
            "B12xExperts only supports native MXFP4 weights, got "
            f"{quant_config.weight_quant_dtype}"
        )

        self._prepared_fp4_moe_by_dtype: dict[torch.dtype, Any] = {}
        self._released_w4a16_source_scales = False
        self._unit_scale_by_device: dict[torch.device, torch.Tensor] = {}

    def _source_format(self) -> str:
        return "fp4_e8m0_k32"

    def _w13_layout(self) -> str:
        # vLLM DSV4 loading stores fused W13 as [w1/gate, w3/up], which is the
        # row order consumed by b12x for the runtime SwiGLU path.
        return "w31"

    def _unit_expert_scale(
        self, device: torch.device, num_experts: int
    ) -> torch.Tensor:
        scale = self._unit_scale_by_device.get(device)
        if scale is None or scale.numel() != num_experts:
            scale = torch.ones(num_experts, dtype=torch.float32, device=device)
            self._unit_scale_by_device[device] = scale
        return scale

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Prepare b12x-owned W4A16 weights and release one-way sources."""
        device = layer.w13_weight.device
        moe_config = getattr(self, "moe_config", None)
        params_dtype = getattr(moe_config, "in_dtype", torch.bfloat16)
        activation = getattr(layer, "activation", None)
        if activation is None:
            activation = getattr(moe_config, "activation", MoEActivation.SILU)
        activation = cast(MoEActivation, activation)

        self._get_or_prepare_fp4_moe_weights(
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            activation=activation,
            params_dtype=params_dtype,
        )
        self._release_w4a16_source_scales(layer)
        self._release_w4a16_source_weights(layer)
        _maybe_release_cuda_cache(device)

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return p.is_cuda() and p.is_device_capability_family(120) and _has_b12x()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, None)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI)

    @staticmethod
    def _supports_parallel_config(
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            not moe_parallel_config.use_ep
            and moe_parallel_config.ep_size <= 1
            and not moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.enable_eplb
        )

    @staticmethod
    def _supports_routing_method(
        routing_method: RoutingMethodType,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return routing_method == RoutingMethodType.DeepseekV4

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def _get_or_prepare_fp4_moe_weights(
        self,
        *,
        w1: torch.Tensor,
        w2: torch.Tensor,
        activation: MoEActivation,
        params_dtype: torch.dtype,
    ):
        prepared = self._prepared_fp4_moe_by_dtype.get(params_dtype)
        if prepared is not None and getattr(prepared, "w4a16", None) is not None:
            return prepared

        if self._released_w4a16_source_scales:
            prepared_dtypes = ", ".join(
                str(dtype) for dtype in self._prepared_fp4_moe_by_dtype
            )
            raise RuntimeError(
                "B12X W4A16 source block scales were already released; "
                f"cannot prepare FP4 MoE weights for dtype {params_dtype}. "
                f"Prepared dtypes: {prepared_dtypes or 'none'}."
            )

        if w1.device.type == "cuda" and _is_current_stream_capturing():
            raise RuntimeError(
                "B12X FP4 MoE weights were not prepared before CUDA "
                f"graph capture for dtype {params_dtype}."
            )
        assert self.w1_scale is not None and self.w2_scale is not None, (
            "w1_scale and w2_scale must not be None for B12xExperts"
        )

        unit_scale = self._unit_expert_scale(w1.device, int(w1.shape[0]))
        prepared = _prepare_b12x_fp4_moe_weights(
            source_format=self._source_format(),
            w13_layout=self._w13_layout(),
            w1_fp4=w1,
            w1_blockscale=self.w1_scale,
            w1_global_scale=unit_scale,
            a1_gscale=unit_scale,
            w2_fp4=w2,
            w2_blockscale=self.w2_scale,
            w2_global_scale=unit_scale,
            a2_gscale=unit_scale,
            activation=_b12x_activation_name(activation),
            params_dtype=params_dtype,
            prepare_runtime_alphas=False,
            prepare_w4a16=True,
            reuse_input_storage=True,
        )
        self._prepared_fp4_moe_by_dtype[params_dtype] = prepared
        return prepared

    def _lookup_prepared_w4a16(self) -> Any | None:
        for prepared in self._prepared_fp4_moe_by_dtype.values():
            w4a16 = getattr(prepared, "w4a16", None)
            if w4a16 is not None:
                return w4a16
        return None

    def _release_w4a16_source_scales(self, layer: torch.nn.Module) -> None:
        if self._released_w4a16_source_scales:
            return

        w1_scale = _replace_parameter_with_empty(layer, "w13_weight_scale")
        w2_scale = _replace_parameter_with_empty(layer, "w2_weight_scale")
        if w1_scale is not None:
            _set_quant_config_weight_scale(self.quant_config, "_w1", w1_scale)
        if w2_scale is not None:
            _set_quant_config_weight_scale(self.quant_config, "_w2", w2_scale)

        self._released_w4a16_source_scales = True

    def _release_w4a16_source_weights(self, layer: torch.nn.Module) -> None:
        _replace_parameter_with_empty(layer, "w13_weight")
        _replace_parameter_with_empty(layer, "w2_weight")

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        if w1.numel() != 0 and w2.numel() != 0:
            return super().moe_problem_size(a1, w1, w2, topk_ids)

        prepared_w4a16 = self._lookup_prepared_w4a16()
        if prepared_w4a16 is None:
            return super().moe_problem_size(a1, w1, w2, topk_ids)

        if a1.dim() == 2:
            assert topk_ids.size(0) == a1.size(0), f"{topk_ids.size(0)} != {a1.size(0)}"
            m = a1.size(0)
        else:
            assert a1.dim() == 3
            m = a1.size(1)

        intermediate_size = int(prepared_w4a16.intermediate_size)
        n = intermediate_size * 2
        return (
            int(prepared_w4a16.num_experts),
            m,
            n,
            a1.size(-1),
            topk_ids.size(1),
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (0,), (0,), (M, K)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ) -> None:
        prepared = self._get_or_prepare_fp4_moe_weights(
            w1=w1,
            w2=w2,
            activation=activation,
            params_dtype=hidden_states.dtype,
        )
        prepared_w4a16 = prepared.w4a16
        assert prepared_w4a16 is not None
        assert self.w1_scale is not None and self.w2_scale is not None, (
            "w1_scale and w2_scale must not be None for B12xExperts"
        )

        if expert_map is not None:
            raise RuntimeError(
                "B12X MoE does not support expert_map with the current b12x_moe_fp4 API"
            )

        num_experts = int(prepared_w4a16.num_experts)
        unit_scale = self._unit_expert_scale(hidden_states.device, num_experts)
        topk_ids = _normalize_b12x_moe_topk_ids(topk_ids)
        topk_weights = _normalize_b12x_moe_topk_weights(topk_weights)

        _run_b12x_moe_fp4(
            a=hidden_states,
            a1_gscale=unit_scale,
            w1_fp4=w1,
            w1_blockscale=self.w1_scale,
            w1_alphas=unit_scale,
            a2_gscale=unit_scale,
            w2_fp4=w2,
            w2_blockscale=self.w2_scale,
            w2_alphas=unit_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=(
                apply_router_weight_on_input
                if apply_router_weight_on_input is not None
                else False
            ),
            output=output,
            input_scales_are_reciprocal=True,
            input_scales_static=True,
            activation=_b12x_activation_name(activation),
            quant_mode="w4a16",
            unit_scale_contract=True,
            source_format=self._source_format(),
            w13_layout=self._w13_layout(),
            prepared_w4a16=prepared_w4a16,
            swiglu_limit=getattr(self.quant_config, "gemm1_clamp_limit", None),
        )

    def moe_sum(self, input: torch.Tensor, output: torch.Tensor) -> None:
        raise NotImplementedError("LoRA is not supported for B12xExperts")
