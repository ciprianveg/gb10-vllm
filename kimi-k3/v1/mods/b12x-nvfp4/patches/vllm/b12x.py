# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib

import torch

import vllm.envs as envs
from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .BlockScaledMMLinearKernel import (
    FP8ScaledMMLinearLayerConfig,
    Fp8BlockScaledMMLinearKernel,
)


def _import_b12x_block_fp8():
    try:
        return importlib.import_module("b12x.gemm.block_fp8_linear")
    except ImportError:
        return None


def _current_linear_backend() -> str:
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return "auto"
    return str(getattr(vllm_config.kernel_config, "linear_backend", "auto")).lower()


def _register_b12x_linear_layer(layer: torch.nn.Module) -> None:
    prefix = getattr(layer, "prefix", "")
    if not prefix:
        raise RuntimeError("b12x FP8 linear requires a non-empty layer prefix")

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return

    static_forward_context = vllm_config.compilation_config.static_forward_context
    existing = static_forward_context.get(prefix)
    if existing is not None and existing is not layer:
        raise ValueError(f"Duplicate layer name: {prefix}")
    static_forward_context[prefix] = layer


def _empty_plan_scratch(
    plan,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in plan.shapes_and_dtypes()
    )


def _run_b12x_fp8_block_scaled_linear(
    layer: torch.nn.Module,
    input_2d: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
) -> None:
    packed_weight = getattr(layer, "b12x_packed_weight", None)
    if packed_weight is None:
        raise RuntimeError(
            "b12x FP8 packed weights are missing; process_weights_after_loading "
            "did not run for this layer"
        )

    if int(packed_weight.out_features) <= 0:
        raise RuntimeError(
            f"b12x FP8 linear invalid out_features={packed_weight.out_features}"
        )

    expected_output_shape = (input_2d.shape[0], int(packed_weight.out_features))
    if output.shape != expected_output_shape:
        raise RuntimeError(
            "b12x FP8 linear output shape mismatch: "
            f"input={tuple(input_2d.shape)} output={tuple(output.shape)} "
            f"expected={expected_output_shape}"
        )

    block_fp8 = _import_b12x_block_fp8()
    if block_fp8 is None:
        raise ImportError("b12x.gemm.block_fp8_linear is not importable")

    plan = block_fp8.plan_block_fp8_linear_scratch(
        block_fp8.BlockFP8LinearScratchCaps(
            device=input_2d.device,
            max_tokens=int(input_2d.shape[0]),
            in_features=int(packed_weight.in_features),
            out_features=int(packed_weight.out_features),
            output_dtype=input_2d.dtype,
        )
    )
    scratch = _empty_plan_scratch(plan, input_2d.device)
    binding = plan.bind(
        scratch=scratch,
        source=input_2d,
        packed_weight=packed_weight,
        output=output.view(
            int(input_2d.shape[0]),
            int(packed_weight.out_features),
            1,
        ),
        bias=bias,
    )

    block_fp8.block_fp8_linear_mxfp8(binding=binding)


def _b12x_fp8_block_scaled_linear_op(
    input_2d: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    _run_b12x_fp8_block_scaled_linear(layer, input_2d, bias, output)


def _b12x_fp8_block_scaled_linear_op_fake(
    input_2d: torch.Tensor,
    bias: torch.Tensor | None,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    del input_2d, bias, output, layer_name
    return None


direct_register_custom_op(
    op_name="b12x_fp8_block_scaled_linear",
    op_func=_b12x_fp8_block_scaled_linear_op,
    mutates_args=["output"],
    fake_impl=_b12x_fp8_block_scaled_linear_op_fake,
)


class B12xFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    """Block-FP8 linear through the native b12x SM120 MXFP8 GEMM path."""

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not current_platform.is_cuda():
            return False, "b12x FP8 kernels are only available on CUDA"
        if not current_platform.is_device_capability_family(120):
            return False, "b12x FP8 kernels require a Blackwell 12x device"
        if _import_b12x_block_fp8() is None:
            return False, "b12x.gemm.block_fp8_linear is not importable"
        return True, None

    @classmethod
    def can_implement(
        cls,
        config: FP8ScaledMMLinearLayerConfig,
    ) -> tuple[bool, str | None]:
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        if _current_linear_backend() != "b12x" and not envs.VLLM_USE_B12X_FP8_GEMM:
            return False, "b12x FP8 GEMM is not enabled"

        if config.out_dtype not in (torch.bfloat16, torch.float16):
            return False, "Supports only bf16/fp16 output dtype"

        act_quant_desc = config.activation_quant_key.scale
        if act_quant_desc.group_shape != GroupShape(1, 128):
            return (
                False,
                "Supports only dynamic per-token group activation "
                "quantization with group_shape=(1,128).",
            )

        weight_group_shape = config.weight_quant_key.scale.group_shape
        if weight_group_shape != GroupShape(128, 128):
            return False, "Supports only 128x128 block-scaled FP8 weights"

        out_features, in_features = config.weight_shape
        if in_features % 128 != 0 or out_features <= 0:
            return False, "Input features must be a positive multiple of 128"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "b12x_skip_generic_block_fp8_linear", False):
            layer.b12x_packed_weight = None
            return

        params = self._get_layer_params(layer)
        weight_scale = (
            params.weight_scale
            if params.weight_scale_inv is None
            else params.weight_scale_inv
        )
        if weight_scale is None:
            raise ValueError("b12x FP8 linear requires block weight scales")
        assert layer.weight_block_size is not None

        block_fp8 = _import_b12x_block_fp8()
        if block_fp8 is None:
            raise ImportError("b12x.gemm.block_fp8_linear is not importable")
        layer.b12x_packed_weight = block_fp8.pack_block_fp8_linear_weight_mxfp8(
            params.weight.detach(),
            weight_scale.detach(),
            block_size=tuple(layer.weight_block_size),
        )
        _register_b12x_linear_layer(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if getattr(layer, "b12x_skip_generic_block_fp8_linear", False):
            raise RuntimeError(
                "b12x generic FP8 linear was called for a layer owned by a fused "
                "b12x path"
            )
        params = self._get_layer_params(layer)
        input_2d = x.reshape(-1, x.shape[-1]).contiguous()
        output_shape = [*x.shape[:-1], params.weight.shape[0]]
        if input_2d.shape[0] == 0:
            return torch.empty(
                output_shape,
                dtype=self.config.out_dtype,
                device=x.device,
            )
        if input_2d.dtype != self.config.out_dtype:
            raise RuntimeError(
                "b12x FP8 linear currently expects input and output dtype to "
                f"match, got input={input_2d.dtype}, output={self.config.out_dtype}"
            )

        layer_name = getattr(layer, "prefix", "")
        if not layer_name:
            raise RuntimeError("b12x FP8 linear requires a non-empty layer prefix")
        output = torch.empty(
            (input_2d.shape[0], params.weight.shape[0]),
            dtype=self.config.out_dtype,
            device=x.device,
        )
        if is_forward_context_available():
            torch.ops.vllm.b12x_fp8_block_scaled_linear(
                input_2d,
                bias,
                output,
                layer_name,
            )
        else:
            _run_b12x_fp8_block_scaled_linear(layer, input_2d, bias, output)
        return output.view(*output_shape)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        del A, B, As, Bs
        raise NotImplementedError("b12x FP8 linear overrides apply_weights")
