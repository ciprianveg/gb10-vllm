# SPDX-License-Identifier: Apache-2.0
"""Memory-bounded TP reductions for full-width Kimi-K3 projections."""

import torch

import vllm.envs as envs
from vllm.distributed import (
    get_dcp_group,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_reduce_in_place,
)
from vllm.v1.attention.ops.dcp_alltoall import (
    dcp_b12x_all_gather_heads,
    dcp_b12x_all_gather_pair,
    try_dcp_b12x_all_gather_pair_kimi_topk,
)

_KIMI_OUTPUT_BUFFER_REUSE_MIN_TOKENS = 1024
_KIMI_B12X_PAIRED_PROJECTION_MAX_TOKENS = 8


def _get_kimi_projection_group():
    """Return a full-TP group for lossless projection collectives.

    DCP16 and TP16 historically shared the DCP coordinator here.  With a
    smaller DCP group (for example TP16/DCP8), projections are still sharded
    over all TP ranks and must not be gathered over only one DCP replica.
    Retain the established DCP16 coordinator when it already spans TP, and
    otherwise use the independent TP coordinator.
    """
    tp_size = get_tensor_model_parallel_world_size()
    dcp_group = get_dcp_group()
    if dcp_group.world_size == tp_size:
        return dcp_group
    tp_group = get_tp_group()
    if tp_group.world_size != tp_size:
        raise RuntimeError(
            "Kimi projection group does not span tensor parallel ranks: "
            f"group={tp_group.world_size}, TP={tp_size}"
        )
    return tp_group


def _try_b12x_kimi_projection_gather(
    output_parallel: torch.Tensor,
) -> torch.Tensor | None:
    """Gather one decode token over Kimi's full-TP PCIe channel.

    The SparkInfer operation is a copy collective. Naturally aligned BF16/FP16
    projections use their native lanes; an unaligned half projection gets a
    zero tail that is stripped after the gather. FP32 router logits are
    reinterpreted as FP8-sized byte lanes solely for transport and restored to
    FP32 afterwards.
    """
    if (
        not envs.VLLM_KIMI_USE_B12X_PROJECTION_GATHER
        or output_parallel.ndim != 2
        or output_parallel.shape[0] != 1
        or not output_parallel.is_cuda
        or not output_parallel.is_contiguous()
    ):
        return None

    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return None
    projection_group = _get_kimi_projection_group()

    local_width = output_parallel.shape[1]
    restore_dtype: torch.dtype | None = None
    strip_local_width: int | None = None
    if output_parallel.dtype in (torch.float16, torch.bfloat16):
        if local_width % 8 == 0:
            transport = output_parallel.view(1, 1, local_width)
        else:
            padded_width = (local_width + 7) // 8 * 8
            transport = torch.nn.functional.pad(
                output_parallel, (0, padded_width - local_width)
            ).view(1, 1, padded_width)
            strip_local_width = local_width
    elif output_parallel.dtype == torch.float32:
        raw_width = local_width * output_parallel.element_size()
        if raw_width % 8 != 0:
            return None
        # SparkInfer accepts FP8 lanes for a one-byte copy dispatch. This is a
        # dtype view, not a conversion, so NaNs and every FP32 mantissa bit are
        # preserved exactly.
        transport = output_parallel.view(torch.float8_e4m3fn).view(1, 1, raw_width)
        restore_dtype = torch.float32
    elif output_parallel.dtype == torch.float8_e4m3fn:
        if local_width % 16 != 0:
            return None
        transport = output_parallel.view(1, 1, local_width)
    else:
        return None

    gathered = dcp_b12x_all_gather_heads(
        transport,
        projection_group,
        max_batch_size=1,
    )
    if strip_local_width is not None:
        gathered = gathered.narrow(-1, 0, strip_local_width).contiguous()
    gathered = gathered.flatten(1)
    if restore_dtype is not None:
        gathered = gathered.view(restore_dtype)
    return gathered


def gather_kimi_sharded_projection(output_parallel: torch.Tensor) -> torch.Tensor:
    """Gather a rank-major Kimi projection with a lossless fast-path."""
    if get_tensor_model_parallel_world_size() <= 1:
        return output_parallel
    gathered = _try_b12x_kimi_projection_gather(output_parallel)
    if gathered is not None:
        return gathered
    return tensor_model_parallel_all_gather(output_parallel, dim=-1)


def gather_kimi_sharded_projection_pair(
    local_first: torch.Tensor,
    local_second: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather two decode projections with one byte-exact B12X barrier."""
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size <= 1:
        return local_first, local_second
    projection_group = _get_kimi_projection_group()
    if (
        envs.VLLM_KIMI_USE_B12X_PROJECTION_GATHER
        and envs.VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER
        and local_first.ndim == local_second.ndim == 2
        and local_first.shape[0] == local_second.shape[0]
        and 0 < local_first.shape[0] <= _KIMI_B12X_PAIRED_PROJECTION_MAX_TOKENS
        and local_first.is_cuda
        and local_second.is_cuda
        and local_first.is_contiguous()
        and local_second.is_contiguous()
        and projection_group.world_size == tp_size
    ):
        return dcp_b12x_all_gather_pair(
            local_first,
            local_second,
            projection_group,
            max_batch_size=_KIMI_B12X_PAIRED_PROJECTION_MAX_TOKENS,
        )
    return (
        gather_kimi_sharded_projection(local_first),
        gather_kimi_sharded_projection(local_second),
    )


def try_gather_kimi_sharded_projection_pair_topk(
    local_down: torch.Tensor,
    local_router: torch.Tensor,
    correction_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Fuse K3's exact TP16 decode gather with its compact routing result."""
    tp_size = get_tensor_model_parallel_world_size()
    if (
        not envs.VLLM_KIMI_USE_B12X_PROJECTION_GATHER
        or not envs.VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER
        or not envs.VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK
        or tp_size != 16
    ):
        return None
    projection_group = _get_kimi_projection_group()
    batch = int(local_down.shape[0]) if local_down.ndim == 2 else 0
    return try_dcp_b12x_all_gather_pair_kimi_topk(
        local_down,
        local_router,
        correction_bias,
        projection_group,
        max_batch_size=1 if batch == 1 else 8,
    )


def should_reuse_kimi_full_width_output(output_buffer: torch.Tensor) -> bool:
    """Whether a full-width Kimi buffer is large enough to donate safely.

    Decode-sized projections stay on their original backend/cudagraph path.
    During chunked prefill the normalized layer input is dead by the time the
    final row-parallel projection runs, and reusing it avoids a 28 MiB
    allocation for Kimi-K3's [2048, 7168] output.
    """
    return output_buffer.ndim >= 2 and output_buffer.shape[0] >= (
        _KIMI_OUTPUT_BUFFER_REUSE_MIN_TOKENS
    )


def reduce_kimi_full_width_output(
    output: torch.Tensor,
    tp_size: int,
) -> torch.Tensor:
    """Reduce a dead rank-local projection output without a full-size copy."""
    if tp_size <= 1:
        return output
    if should_reuse_kimi_full_width_output(output):
        # Large messages fall through TP16's graph-oriented custom AR to NCCL.
        # The rank-local result is dead, so let NCCL overwrite it instead of
        # allocating another 28 MiB [2048, 7168] output tensor.
        return tensor_model_parallel_all_reduce_in_place(output)
    # Preserve the existing custom/cudagraph path for decode-sized tensors.
    return tensor_model_parallel_all_reduce(output)
