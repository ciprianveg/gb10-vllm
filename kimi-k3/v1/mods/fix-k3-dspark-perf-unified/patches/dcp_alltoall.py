# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
DCP All-to-All communication backend for attention.

Provides All-to-All (A2A) communication as an alternative to
AllGather + ReduceScatter (AG+RS) for Decode Context Parallel (DCP).
Instead of gathering the full Q tensor and scattering partial outputs,
A2A exchanges partial attention outputs and their LSE values across
ranks, then combines them with exact LSE-weighted reduction.

This reduces the number of NCCL calls per attention layer by exchanging
the partial output and LSE in a single packed All-to-All payload.

Usage:
    vllm serve model --tp 16 --dcp 16 --dcp-comm-backend a2a

Reference: https://arxiv.org/abs/2507.07120
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.multi_stream_utils import is_vllm_cudagraph_capture_active

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.attention.ops.common import CPTritonContext


logger = init_logger(__name__)

_B12X_DCP_A2A_POOLS: dict[tuple[int, int, int, int, int, int], Any] = {}
_B12X_DCP_A2A_DISABLED: set[tuple[int, int, int, int, int, int]] = set()
_B12X_DCP_EAGER_CHANNEL_ID = "vllm:eager:dcp"
_B12X_DCP_CAPTURE_SEQUENCES: dict[int, int] = {}
_B12X_DCP_WORLD_SIZES = (2, 4, 8, 16)
_DCP_A2A_GRAPH_BUFFERS: dict[
    tuple[tuple[int, ...], torch.device, torch.dtype],
    tuple[torch.Tensor, torch.Tensor],
] = {}


def _is_supported_bhd_layout(tensor: torch.Tensor) -> bool:
    """Accept packed token-major or capacity-strided head-major BHD views."""
    if tensor.ndim != 3 or int(tensor.stride(2)) != 1:
        return False
    batch, heads, head_dim = (int(value) for value in tensor.shape)
    stride_batch, stride_head, _ = (int(value) for value in tensor.stride())
    packed_token_major = stride_batch == heads * head_dim and stride_head == head_dim
    capacity_strided_head_major = (
        stride_batch == head_dim and stride_head >= batch * head_dim
    )
    return packed_token_major or capacity_strided_head_major


@lru_cache(maxsize=1)
def _load_b12x_dcp_a2a_pool() -> Any | None:
    try:
        from sparkinfer.comm.pcie import DcpAllToAllPool as PCIeDCPA2APool
    except Exception:
        return None
    return PCIeDCPA2APool


def _b12x_dcp_init_failed(
    cp_group: GroupCoordinator,
    device: torch.device,
    init_error: Exception | None,
) -> bool:
    """Reach consensus on pool initialization through its exchange group."""
    failed = torch.tensor(
        [int(init_error is not None)], dtype=torch.int32, device=device
    )
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=cp_group.device_group)
    return bool(failed.item())


def _get_b12x_dcp_a2a_pool(
    cp_group: GroupCoordinator,
    *,
    device: torch.device,
    total_heads: int,
    head_dim: int,
    query_head_dim: int,
    max_batch_size: int,
) -> Any | None:
    device_index = device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    key = (
        id(cp_group.device_group),
        int(device_index),
        int(total_heads),
        int(head_dim),
        int(query_head_dim),
        int(max_batch_size),
    )
    if key in _B12X_DCP_A2A_DISABLED:
        return None

    pool = _B12X_DCP_A2A_POOLS.get(key)
    if pool is not None:
        return pool

    # IPC allocation and handle exchange are not capture-safe. Dedicated
    # kernel warmup normally initializes this channel before graph capture.
    if torch.cuda.is_current_stream_capturing():
        return None
    pool_cls = _load_b12x_dcp_a2a_pool()
    if pool_cls is None:
        _B12X_DCP_A2A_DISABLED.add(key)
        return None

    init_error: Exception | None = None
    try:
        pool = pool_cls.from_exchange_group(
            exchange_group=cp_group.device_group,
            device=device,
            max_batch_size=max_batch_size,
            total_heads=total_heads,
            head_dim=head_dim,
            query_head_dim=query_head_dim,
            single_channel=False,
        )
        # SparkInfer identifies independently replayable channels by a stable
        # semantic name. Stream handles are process-local and can be recycled,
        # so they are not a valid distributed identity.
        pool.prepare_channels((_B12X_DCP_EAGER_CHANNEL_ID,))
        pool.for_stream(channel_id=_B12X_DCP_EAGER_CHANNEL_ID)
    except Exception as exc:
        init_error = exc

    # Keep the status collective ordered with the NCCL group used above for
    # IPC-handle exchange. A separate Gloo group can be at a different startup
    # sequence when vLLM initializes overlapping TP/DCP coordinators.
    any_failed = _b12x_dcp_init_failed(cp_group, device, init_error)

    if any_failed:
        if pool is not None:
            pool.close()
        _B12X_DCP_A2A_DISABLED.add(key)
        if init_error is not None:
            logger.warning(
                "B12X PCIe DCP collective initialization failed; falling "
                "back to NCCL: %s",
                init_error,
            )
        return None

    assert pool is not None
    _B12X_DCP_A2A_POOLS[key] = pool
    logger.info(
        "Using B12X PCIe DCP collectives "
        "(world_size=%d, max_batch_size=%d, heads=%d, "
        "query_head_dim=%d, output_head_dim=%d).",
        cp_group.world_size,
        max_batch_size,
        total_heads,
        query_head_dim,
        head_dim,
    )
    return pool


@contextmanager
def capture_b12x_dcp_a2a(
    cp_group: GroupCoordinator,
    stream: torch.cuda.Stream | None = None,
):
    """Bind registered SparkInfer DCP pools to the graph's owning stream.

    Each graph capture receives independent channels; reusing channels across
    target and draft graphs would make one graph depend on another's lifetime.

    Args:
        cp_group: DCP group whose registered pools should enter capture.
        stream: CUDA stream owned by the enclosing graph capture.
    """
    group_id = id(cp_group.device_group)
    sequence = _B12X_DCP_CAPTURE_SEQUENCES.get(group_id, 0)
    _B12X_DCP_CAPTURE_SEQUENCES[group_id] = sequence + 1
    channel_id = f"vllm:graph:{sequence}"
    matching_pools = sorted(
        (
            (key, pool)
            for key, pool in _B12X_DCP_A2A_POOLS.items()
            if key[0] == group_id
        ),
        key=lambda item: item[0][1:],
    )
    with ExitStack() as stack:
        for _, pool in matching_pools:
            stack.enter_context(pool.capture(stream=stream, channel_id=channel_id))
        yield


def checkpoint_b12x_dcp_a2a_channels(
    cp_group: GroupCoordinator,
) -> tuple[int, dict[Any, tuple[Any, Any]]]:
    """Snapshot SparkInfer DCP pools before a disposable graph capture."""
    group_id = id(cp_group.device_group)
    checkpoints = {
        key: (pool, pool.checkpoint_channels())
        for key, pool in _B12X_DCP_A2A_POOLS.items()
        if key[0] == group_id
    }
    return group_id, checkpoints


def rollback_b12x_dcp_a2a_channels(
    checkpoint: tuple[int, dict[Any, tuple[Any, Any]]],
) -> None:
    """Restore DCP pools after their disposable graphs are destroyed."""
    group_id, checkpoints = checkpoint
    for key, pool in list(_B12X_DCP_A2A_POOLS.items()):
        if key[0] != group_id:
            continue
        saved = checkpoints.get(key)
        if saved is None:
            pool.close()
            del _B12X_DCP_A2A_POOLS[key]
            continue
        saved_pool, channel_checkpoint = saved
        if pool is not saved_pool:
            pool.close()
            _B12X_DCP_A2A_POOLS[key] = saved_pool
        saved_pool.rollback_channels(channel_checkpoint)


def _try_b12x_dcp_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    *,
    return_lse: bool,
    is_lse_base_on_e: bool,
    max_batch_size: int | None,
    query_head_dim: int | None,
) -> torch.Tensor | None:
    """Use the low-latency B12X PCIe path when its contract is satisfied."""
    world_size = cp_group.world_size
    if (
        return_lse
        or not cp_attn_out.is_cuda
        or cp_attn_out.dtype not in (torch.float16, torch.bfloat16)
        or cp_attn_lse.dtype != torch.float32
        or world_size not in _B12X_DCP_WORLD_SIZES
        or cp_attn_out.ndim != 3
        or cp_attn_lse.shape != cp_attn_out.shape[:2]
    ):
        return None

    batch, total_heads, head_dim = cp_attn_out.shape
    if total_heads % world_size != 0 or head_dim % 8 != 0:
        return None

    if max_batch_size is None:
        max_batch_size = batch
    max_batch_size = int(max_batch_size)
    token_cap = envs.VLLM_DCP_A2A_MAX_TOKENS
    if token_cap > 0:
        # Deliberate hybrid dispatch: batches above the cap take a pipelined
        # NCCL collective instead, and the staging pool shrinks to the cap.
        if batch > token_cap:
            return None
        max_batch_size = min(max_batch_size, token_cap)
    if max_batch_size < 1:
        return None
    if query_head_dim is None:
        query_head_dim = head_dim
    query_head_dim = int(query_head_dim)
    if query_head_dim <= 0 or query_head_dim % 8 != 0:
        return None

    pool = _get_b12x_dcp_a2a_pool(
        cp_group,
        device=cp_attn_out.device,
        total_heads=total_heads,
        head_dim=head_dim,
        query_head_dim=query_head_dim,
        max_batch_size=max_batch_size,
    )
    if pool is None:
        return None

    if batch > max_batch_size:
        logger.warning_once(
            "B12X PCIe DCP A2A received batch=%d beyond its configured "
            "max_batch_size=%d; falling back to NCCL.",
            batch,
            max_batch_size,
        )
        return None

    # The channel accepts packed token-major input and the capacity-strided
    # head-major layout produced by B12X sparse MLA. Preserve either layout;
    # only legacy padded-head slices need materialization.
    if not _is_supported_bhd_layout(cp_attn_out):
        cp_attn_out = cp_attn_out.contiguous()
    if not cp_attn_lse.is_contiguous():
        cp_attn_lse = cp_attn_lse.contiguous()

    reduced_storage = torch.empty(
        (total_heads // world_size, batch, head_dim),
        device=cp_attn_out.device,
        dtype=cp_attn_out.dtype,
    )
    reduced = reduced_storage.transpose(0, 1)
    return pool.lse_reduce_scatter(
        cp_attn_out,
        cp_attn_lse,
        out=reduced,
        is_lse_base_on_e=is_lse_base_on_e,
        channel_id=_B12X_DCP_EAGER_CHANNEL_ID,
    )


def _try_b12x_dcp_all_gather_heads(
    local_input: torch.Tensor,
    cp_group: GroupCoordinator,
    *,
    max_batch_size: int | None,
    output_head_dim: int | None,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Gather rank-local query heads with the B12X PCIe channel."""
    world_size = cp_group.world_size
    if (
        not local_input.is_cuda
        or local_input.dtype not in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
        or world_size not in _B12X_DCP_WORLD_SIZES
        or local_input.ndim != 3
        or not local_input.is_contiguous()
    ):
        return None

    batch, local_heads, head_dim = local_input.shape
    gather_alignment = 16 if local_input.dtype == torch.float8_e4m3fn else 8
    if local_heads <= 0 or head_dim % gather_alignment != 0:
        return None
    if max_batch_size is None:
        max_batch_size = batch
    max_batch_size = int(max_batch_size)
    token_cap = envs.VLLM_DCP_A2A_MAX_TOKENS
    if token_cap > 0:
        if batch > token_cap:
            return None
        max_batch_size = min(max_batch_size, token_cap)
    if max_batch_size < 1 or batch > max_batch_size:
        return None
    if output_head_dim is None:
        output_head_dim = head_dim
    output_head_dim = int(output_head_dim)
    if output_head_dim <= 0 or output_head_dim % 8 != 0:
        return None

    pool = _get_b12x_dcp_a2a_pool(
        cp_group,
        device=local_input.device,
        total_heads=local_heads * world_size,
        head_dim=output_head_dim,
        query_head_dim=head_dim,
        max_batch_size=max_batch_size,
    )
    if pool is None:
        return None
    if out is None:
        return pool.all_gather_heads(
            local_input,
            channel_id=_B12X_DCP_EAGER_CHANNEL_ID,
        )
    return pool.all_gather_heads(
        local_input,
        out=out,
        channel_id=_B12X_DCP_EAGER_CHANNEL_ID,
    )


def dcp_b12x_all_gather_heads(
    local_input: torch.Tensor,
    cp_group: GroupCoordinator,
    *,
    max_batch_size: int | None = None,
    output_head_dim: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather query heads with B12X, falling back to the group backend."""
    local_input = local_input.contiguous()
    if envs.VLLM_USE_B12X_DCP_A2A:
        if out is None:
            result = _try_b12x_dcp_all_gather_heads(
                local_input,
                cp_group,
                max_batch_size=max_batch_size,
                output_head_dim=output_head_dim,
            )
        else:
            result = _try_b12x_dcp_all_gather_heads(
                local_input,
                cp_group,
                max_batch_size=max_batch_size,
                output_head_dim=output_head_dim,
                out=out,
            )
        if result is not None:
            return result
    result = cp_group.all_gather(local_input, dim=1)
    if out is None:
        return result
    out.copy_(result)
    return out


def _try_b12x_dcp_all_gather_pair(
    local_first: torch.Tensor,
    local_second: torch.Tensor,
    cp_group: GroupCoordinator,
    *,
    max_batch_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Gather two decode projections behind one SparkInfer IPC barrier."""
    world_size = cp_group.world_size
    supported_dtypes = (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float8_e4m3fn,
    )
    if (
        world_size not in _B12X_DCP_WORLD_SIZES
        or local_first.dtype not in supported_dtypes
        or local_second.dtype not in supported_dtypes
        or not local_first.is_cuda
        or not local_second.is_cuda
        or local_first.device != local_second.device
        or local_first.ndim != 2
        or local_second.ndim != 2
        or local_first.shape[0] != local_second.shape[0]
        or not local_first.is_contiguous()
        or not local_second.is_contiguous()
    ):
        return None
    batch = int(local_first.shape[0])
    first_row_bytes = int(local_first.shape[1]) * local_first.element_size()
    second_row_bytes = int(local_second.shape[1]) * local_second.element_size()
    if batch < 1 or first_row_bytes % 16 != 0 or second_row_bytes % 16 != 0:
        return None
    if max_batch_size is None:
        max_batch_size = batch
    max_batch_size = int(max_batch_size)
    token_cap = envs.VLLM_DCP_A2A_MAX_TOKENS
    if token_cap > 0:
        if batch > token_cap:
            return None
        max_batch_size = min(max_batch_size, token_cap)
    if max_batch_size < batch:
        return None

    combined_row_bytes = first_row_bytes + second_row_bytes
    pool = _get_b12x_dcp_a2a_pool(
        cp_group,
        device=local_first.device,
        total_heads=world_size,
        head_dim=combined_row_bytes,
        query_head_dim=combined_row_bytes,
        max_batch_size=max_batch_size,
    )
    if pool is None or not hasattr(pool, "all_gather_pair"):
        return None
    return pool.all_gather_pair(
        local_first,
        local_second,
        channel_id=_B12X_DCP_EAGER_CHANNEL_ID,
    )


def dcp_b12x_all_gather_pair(
    local_first: torch.Tensor,
    local_second: torch.Tensor,
    cp_group: GroupCoordinator,
    *,
    max_batch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather two rank-local rows together, with exact separate fallbacks."""
    if envs.VLLM_USE_B12X_DCP_A2A:
        result = _try_b12x_dcp_all_gather_pair(
            local_first,
            local_second,
            cp_group,
            max_batch_size=max_batch_size,
        )
        if result is not None:
            return result
    return (
        cp_group.all_gather(local_first, dim=-1),
        cp_group.all_gather(local_second, dim=-1),
    )


def try_dcp_b12x_all_gather_pair_kimi_topk(
    local_down: torch.Tensor,
    local_router: torch.Tensor,
    correction_bias: torch.Tensor,
    cp_group: GroupCoordinator,
    *,
    max_batch_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return Kimi's gathered latent and compact routing payload on TP16.

    The payload is one FP32 tensor with contiguous ``[M, 16]`` weight and ID
    planes; the ID plane is viewed as int32 by the K3 router. This function
    deliberately has no semantic NCCL fallback; its caller retains the
    ordinary paired gather plus router implementation when the exact K3
    contract is unavailable.
    """
    batch = int(local_down.shape[0]) if local_down.ndim == 2 else 0
    if (
        not envs.VLLM_USE_B12X_DCP_A2A
        or cp_group.world_size != 16
        or not 1 <= batch <= 8
        or local_down.shape != (batch, 224)
        or local_down.dtype != torch.bfloat16
        or local_router.shape != (batch, 56)
        or local_router.dtype != torch.float32
        or correction_bias.shape != (896,)
        or correction_bias.dtype != torch.float32
        or not local_down.is_cuda
        or not local_router.is_cuda
        or not correction_bias.is_cuda
        or local_down.device != local_router.device
        or local_down.device != correction_bias.device
        or not local_down.is_contiguous()
        or not local_router.is_contiguous()
        or not correction_bias.is_contiguous()
    ):
        return None
    if max_batch_size is None:
        max_batch_size = 1 if batch == 1 else 8
    if int(max_batch_size) < batch:
        return None

    combined_row_bytes = 224 * 2 + 56 * 4
    pool = _get_b12x_dcp_a2a_pool(
        cp_group,
        device=local_down.device,
        total_heads=16,
        head_dim=combined_row_bytes,
        query_head_dim=combined_row_bytes,
        max_batch_size=int(max_batch_size),
    )
    if pool is None or not hasattr(pool, "all_gather_pair_kimi_topk"):
        return None

    routed_hidden_states = torch.empty(
        (batch, 3584), device=local_down.device, dtype=torch.bfloat16
    )
    routing_payload = torch.empty(
        (batch * 2, 16), device=local_down.device, dtype=torch.float32
    )
    topk_weights = routing_payload[:batch]
    topk_ids = routing_payload[batch:].view(torch.int32)
    pool.all_gather_pair_kimi_topk(
        local_down,
        local_router,
        correction_bias,
        routed_hidden_states,
        topk_weights,
        topk_ids,
        channel_id=_B12X_DCP_EAGER_CHANNEL_ID,
    )
    return routed_hidden_states, routing_payload


def warmup_b12x_kimi_projection_gathers(
    projection_group: GroupCoordinator,
    *,
    device: torch.device,
) -> int:
    """Create Kimi's paired TP channels before CUDA graph capture.

    The M=8 paired projection is first reached by DSpark verification, while
    the M=1 gather+top-k channel is first reached by ordinary decode. Neither
    geometry is guaranteed to run during the large-token memory-profile
    forward. Creating either pool from inside ``graph_capture()`` is too late:
    it misses the graph-specific channel binding established on context entry
    and leaves captured nodes sharing the eager channel.
    """
    if (
        not envs.VLLM_USE_B12X_DCP_A2A
        or not envs.VLLM_KIMI_USE_B12X_PROJECTION_GATHER
        or not envs.VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_GATHER
        or projection_group.world_size != 16
    ):
        return 0

    # A target-only max-num-seqs=1 profile deliberately caps the B12X DCP
    # path at one token and routes larger batches through its exact NCCL
    # fallback.  Do not make that profile initialize an M=8 channel it can
    # never dispatch; DSpark profiles set the cap to at least eight and still
    # precreate this graph channel before verification capture.
    token_cap = envs.VLLM_DCP_A2A_MAX_TOKENS
    warmed = 0
    if token_cap <= 0 or token_cap >= 8:
        local_down = torch.zeros((8, 224), device=device, dtype=torch.bfloat16)
        local_router = torch.zeros((8, 56), device=device, dtype=torch.float32)
        if (
            envs.VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK
            and envs.VLLM_KIMI_USE_B12X_BATCHED_PROJECTION_TOPK
        ):
            correction_bias = torch.zeros(
                (896,), device=device, dtype=torch.float32
            )
            paired = try_dcp_b12x_all_gather_pair_kimi_topk(
                local_down,
                local_router,
                correction_bias,
                projection_group,
                max_batch_size=8,
            )
        else:
            paired = _try_b12x_dcp_all_gather_pair(
                local_down,
                local_router,
                projection_group,
                max_batch_size=8,
            )
        if paired is None:
            raise RuntimeError(
                "B12X Kimi M=8 paired projection gather path is unavailable"
            )
        warmed += 1

    if envs.VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK:
        local_down = torch.zeros((1, 224), device=device, dtype=torch.bfloat16)
        local_router = torch.zeros((1, 56), device=device, dtype=torch.float32)
        correction_bias = torch.zeros((896,), device=device, dtype=torch.float32)
        fused = try_dcp_b12x_all_gather_pair_kimi_topk(
            local_down,
            local_router,
            correction_bias,
            projection_group,
            max_batch_size=1,
        )
        if fused is None:
            raise RuntimeError(
                "B12X Kimi M=1 paired projection gather+top-k is unavailable"
            )
        warmed += 1
    return warmed


def warmup_b12x_dcp_a2a(
    cp_group: GroupCoordinator,
    *,
    device: torch.device,
    dtype: torch.dtype,
    max_batch_size: int,
    total_heads: int,
    head_dim: int,
    query_head_dim: int | None = None,
) -> None:
    """Create and exercise the B12X DCP channel before CUDA graph capture."""
    if not envs.VLLM_USE_B12X_DCP_A2A:
        return
    if cp_group.world_size not in _B12X_DCP_WORLD_SIZES:
        # The PCIe channel only exists for these world sizes. The runtime
        # dispatchers already fall back to NCCL collectives per call, so an
        # unsupported DCP size (e.g. TP6 with DCP3/DCP6) must not fail boot.
        logger.warning_once(
            "B12X PCIe DCP collectives support world sizes 2/4/8/16; "
            "DCP world size %d uses NCCL collectives instead.",
            cp_group.world_size,
        )
        return
    if query_head_dim is None:
        query_head_dim = head_dim
    local_query = torch.empty(
        (1, total_heads // cp_group.world_size, query_head_dim),
        device=device,
        dtype=dtype,
    )
    gathered_query = _try_b12x_dcp_all_gather_heads(
        local_query,
        cp_group,
        max_batch_size=max_batch_size,
        output_head_dim=head_dim,
    )
    if gathered_query is None:
        raise RuntimeError(
            "B12X PCIe DCP query all-gather is unavailable for the configured "
            "attention geometry"
        )
    partial_output = torch.empty(
        (1, total_heads, head_dim),
        device=device,
        dtype=dtype,
    )
    partial_lse = torch.zeros(
        (1, total_heads),
        device=device,
        dtype=torch.float32,
    )
    result = _try_b12x_dcp_lse_reduce(
        partial_output,
        partial_lse,
        cp_group,
        return_lse=False,
        is_lse_base_on_e=True,
        max_batch_size=max_batch_size,
        query_head_dim=query_head_dim,
    )
    if result is None:
        raise RuntimeError(
            "B12X PCIe DCP output reduction is unavailable for the configured "
            "attention geometry"
        )


def _lse_weighted_combine(
    outputs: torch.Tensor,
    lses: torch.Tensor,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    CPU reference implementation for LSE-weighted combination.

    This is a pure PyTorch implementation used for testing and validation.

    Args:
        outputs: Partial attention outputs [N, B, H, D]
                 N = number of KV shards (ranks)
                 B = batch size (num_tokens)
                 H = number of heads per rank
                 D = head dimension
        lses: Log-sum-exp values [N, B, H]
        return_lse: If True, also return the global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H, D], and optionally global LSE [B, H]
    """
    N, B, H, D = outputs.shape

    # Handle NaN and inf in LSEs
    lses = torch.where(
        torch.isnan(lses) | torch.isinf(lses),
        torch.tensor(float("-inf"), device=lses.device, dtype=lses.dtype),
        lses,
    )

    # Compute max LSE for numerical stability
    lse_max, _ = lses.max(dim=0)  # [B, H]
    lse_max = torch.where(
        lse_max == float("-inf"),
        torch.zeros_like(lse_max),
        lse_max,
    )

    # Compute weights: softmax over the N dimension
    if is_lse_base_on_e:
        weights = torch.exp(lses - lse_max.unsqueeze(0))  # [N, B, H]
    else:
        weights = torch.pow(2.0, lses - lse_max.unsqueeze(0))  # [N, B, H]

    # Handle NaN weights
    weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)

    # Normalize weights
    weight_sum = weights.sum(dim=0, keepdim=True)  # [1, B, H]
    weights = weights / weight_sum.clamp(min=1e-10)  # [N, B, H]

    # Weighted combination: sum over N dimension
    result = (outputs * weights.unsqueeze(-1)).sum(dim=0)  # [B, H, D]

    if return_lse:
        if is_lse_base_on_e:
            global_lse = torch.log(weight_sum.squeeze(0)) + lse_max  # [B, H]
        else:
            global_lse = torch.log2(weight_sum.squeeze(0)) + lse_max  # [B, H]
        return result, global_lse

    return result


def _dcp_a2a_lse_pack_dim(output_dtype: torch.dtype) -> int:
    bits = torch.finfo(output_dtype).bits
    if bits == 16:
        return 2
    if bits == 32:
        return 1
    raise ValueError(f"Cannot pack fp32 LSE into output dtype {output_dtype}.")


def _validate_dcp_valid_counts(
    valid_counts: torch.Tensor | None,
    cp_attn_out: torch.Tensor,
) -> None:
    if valid_counts is None:
        return
    expected_shape = (cp_attn_out.shape[0],)
    if valid_counts.shape != expected_shape:
        raise ValueError(
            f"valid_counts must have shape {expected_shape}, got "
            f"{tuple(valid_counts.shape)}."
        )
    if valid_counts.dtype != torch.int32:
        raise TypeError(
            f"valid_counts must have dtype torch.int32, got {valid_counts.dtype}."
        )
    if valid_counts.device != cp_attn_out.device:
        raise ValueError(
            "valid_counts and attention output must be on the same device, got "
            f"{valid_counts.device} and {cp_attn_out.device}."
        )


@triton.jit
def _sanitize_dcp_empty_rows_kernel(
    out_ptr,
    lse_ptr,
    valid_counts_ptr,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    lse_stride_B,
    lse_stride_H,
    valid_counts_stride_B,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < HEAD_DIM
    row_is_valid = tl.load(valid_counts_ptr + batch_idx * valid_counts_stride_B) > 0
    out_offsets = (
        batch_idx * out_stride_B + head_idx * out_stride_H + d_offsets * out_stride_D
    )
    values = tl.load(out_ptr + out_offsets, mask=d_mask)
    tl.store(out_ptr + out_offsets, tl.where(row_is_valid, values, 0.0), mask=d_mask)
    lse_offset = batch_idx * lse_stride_B + head_idx * lse_stride_H
    lse = tl.load(lse_ptr + lse_offset)
    tl.store(lse_ptr + lse_offset, tl.where(row_is_valid, lse, -float("inf")))


def sanitize_dcp_attn_empty_rows(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    valid_counts: torch.Tensor | None,
) -> None:
    """Force locally empty DCP rows to the neutral ``(0, -inf)`` state."""
    _validate_dcp_valid_counts(valid_counts, cp_attn_out)
    if valid_counts is None or cp_attn_out.shape[0] == 0:
        return
    if cp_attn_out.ndim != 3 or cp_attn_lse.shape != cp_attn_out.shape[:2]:
        raise ValueError(
            "Expected attention output [B,H,D] and LSE [B,H], got "
            f"{tuple(cp_attn_out.shape)} and {tuple(cp_attn_lse.shape)}."
        )
    head_dim = cp_attn_out.shape[2]
    grid = (cp_attn_out.shape[0], cp_attn_out.shape[1], 1)
    _sanitize_dcp_empty_rows_kernel[grid](
        cp_attn_out,
        cp_attn_lse,
        valid_counts,
        cp_attn_out.stride(0),
        cp_attn_out.stride(1),
        cp_attn_out.stride(2),
        cp_attn_lse.stride(0),
        cp_attn_lse.stride(1),
        valid_counts.stride(0),
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


def _dcp_a2a_send_recv_buffers(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Don't use the shared WorkspaceManager here. A FULL cudagraph bakes in the
    # buffer address at capture, but a larger eager batch can grow that workspace
    # and free the captured address. Eager calls therefore use ordinary temporary
    # tensors, while captured calls retain fixed-size owners below.
    if device.type == "cuda" and (
        is_vllm_cudagraph_capture_active() or torch.cuda.is_current_stream_capturing()
    ):
        # FULL graphs share a global graph pool. Without a live Python owner,
        # a larger descriptor's staging allocation can be recycled while a
        # smaller descriptor is captured, leaving both NCCL graph nodes bound
        # to the same address. Keep one exact-shape pair alive per device so
        # descriptors cannot alias each other's A2A staging storage. Layers of
        # one graph may reuse the pair because all operations are stream-ordered.
        key = (shape, device, dtype)
        buffers = _DCP_A2A_GRAPH_BUFFERS.get(key)
        if buffers is None:
            buffers = (
                torch.empty(shape, device=device, dtype=dtype),
                torch.empty(shape, device=device, dtype=dtype),
            )
            _DCP_A2A_GRAPH_BUFFERS[key] = buffers
        return buffers

    return (
        torch.empty(shape, device=device, dtype=dtype),
        torch.empty(shape, device=device, dtype=dtype),
    )


@triton.jit
def _dcp_a2a_pack_send_kernel(
    out_ptr,
    lse_ptr,
    send_ptr,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    lse_stride_B,
    lse_stride_H,
    send_stride_N,
    send_stride_B,
    send_stride_H,
    send_stride_D,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    H_PER_RANK: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    local_head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    for rank_idx in tl.static_range(N):
        src_head_idx = rank_idx * H_PER_RANK + local_head_idx
        send_base = (
            rank_idx * send_stride_N
            + batch_idx * send_stride_B
            + local_head_idx * send_stride_H
        )

        out_offsets = (
            batch_idx * out_stride_B
            + src_head_idx * out_stride_H
            + d_offsets * out_stride_D
        )
        tl.store(
            send_ptr + send_base + d_offsets * send_stride_D,
            tl.load(out_ptr + out_offsets),
        )

        lse_val = tl.load(
            lse_ptr + batch_idx * lse_stride_B + src_head_idx * lse_stride_H
        )
        if LSE_PACK_DIM == 1:
            tl.store(
                send_ptr + send_base + HEAD_DIM * send_stride_D,
                lse_val.to(send_ptr.dtype.element_ty),
            )
        else:
            lse_bits = lse_val.to(tl.uint32, bitcast=True)
            lo = (lse_bits & 0xFFFF).to(tl.uint16)
            hi = ((lse_bits >> 16) & 0xFFFF).to(tl.uint16)
            tl.store(
                send_ptr + send_base + HEAD_DIM * send_stride_D,
                lo.to(send_ptr.dtype.element_ty, bitcast=True),
            )
            tl.store(
                send_ptr + send_base + (HEAD_DIM + 1) * send_stride_D,
                hi.to(send_ptr.dtype.element_ty, bitcast=True),
            )


@triton.jit
def _dcp_a2a_unpack_combine_kernel(
    recv_ptr,
    out_ptr,
    out_lse_ptr,
    recv_stride_N,
    recv_stride_B,
    recv_stride_H,
    recv_stride_D,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    out_lse_stride_B,
    out_lse_stride_H,
    N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    LSE_PACK_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_offsets = tl.arange(0, HEAD_DIM)

    lse_max = -float("inf")
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D).to(
                tl.float32
            )
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        lse_max = tl.maximum(lse_max, lse_val)

    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)

    lse_sum = 0.0
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D).to(
                tl.float32
            )
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            lse_sum += tl.exp(lse_val - lse_max)
        else:
            lse_sum += tl.exp2(lse_val - lse_max)

    if IS_BASE_E:  # noqa: SIM108
        global_lse = tl.log(lse_sum) + lse_max
    else:
        global_lse = tl.log2(lse_sum) + lse_max

    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    for rank_idx in tl.static_range(N):
        recv_base = (
            rank_idx * recv_stride_N
            + batch_idx * recv_stride_B
            + head_idx * recv_stride_H
        )
        if LSE_PACK_DIM == 1:
            lse_val = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D).to(
                tl.float32
            )
        else:
            lo_raw = tl.load(recv_ptr + recv_base + HEAD_DIM * recv_stride_D)
            hi_raw = tl.load(recv_ptr + recv_base + (HEAD_DIM + 1) * recv_stride_D)
            lo = lo_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            hi = hi_raw.to(tl.uint16, bitcast=True).to(tl.uint32)
            lse_val = (lo | (hi << 16)).to(tl.float32, bitcast=True)
        lse_val = tl.where(
            (lse_val != lse_val) | (lse_val == float("inf")),
            -float("inf"),
            lse_val,
        )
        if IS_BASE_E:
            weight = tl.exp(lse_val - global_lse)
        else:
            weight = tl.exp2(lse_val - global_lse)
        weight = tl.where(weight != weight, 0.0, weight)
        acc += (
            tl.load(recv_ptr + recv_base + d_offsets * recv_stride_D).to(tl.float32)
            * weight
        )

    final_offsets = (
        batch_idx * out_stride_B + head_idx * out_stride_H + d_offsets * out_stride_D
    )
    tl.store(out_ptr + final_offsets, acc)

    if RETURN_LSE:
        out_lse_offset = batch_idx * out_lse_stride_B + head_idx * out_lse_stride_H
        tl.store(out_lse_ptr + out_lse_offset, global_lse)


def _dcp_a2a_pack_send(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    send_buffer: torch.Tensor,
    world_size: int,
    h_per_rank: int,
    head_dim: int,
    lse_pack_dim: int,
) -> None:
    grid = (cp_attn_out.shape[0], h_per_rank, 1)
    _dcp_a2a_pack_send_kernel[grid](
        cp_attn_out,
        cp_attn_lse,
        send_buffer,
        cp_attn_out.stride(0),
        cp_attn_out.stride(1),
        cp_attn_out.stride(2),
        cp_attn_lse.stride(0),
        cp_attn_lse.stride(1),
        send_buffer.stride(0),
        send_buffer.stride(1),
        send_buffer.stride(2),
        send_buffer.stride(3),
        N=world_size,
        HEAD_DIM=head_dim,
        H_PER_RANK=h_per_rank,
        LSE_PACK_DIM=lse_pack_dim,
    )


def _dcp_a2a_unpack_combine(
    recv_buffer: torch.Tensor,
    head_dim: int,
    lse_pack_dim: int,
    return_lse: bool,
    is_lse_base_on_e: bool,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    world_size, num_tokens, h_per_rank, _ = recv_buffer.shape
    out_storage = torch.empty(
        (h_per_rank, num_tokens, head_dim),
        device=recv_buffer.device,
        dtype=recv_buffer.dtype,
    )
    out = out_storage.transpose(0, 1)
    out_lse = torch.empty(
        (num_tokens, h_per_rank) if return_lse else (1, 1),
        device=recv_buffer.device,
        dtype=torch.float32 if return_lse else recv_buffer.dtype,
    )
    grid = (num_tokens, h_per_rank, 1)
    _dcp_a2a_unpack_combine_kernel[grid](
        recv_buffer,
        out,
        out_lse,
        recv_buffer.stride(0),
        recv_buffer.stride(1),
        recv_buffer.stride(2),
        recv_buffer.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out_lse.stride(0),
        out_lse.stride(1),
        N=world_size,
        HEAD_DIM=head_dim,
        IS_BASE_E=is_lse_base_on_e,
        RETURN_LSE=return_lse,
        LSE_PACK_DIM=lse_pack_dim,
    )
    if return_lse:
        return out, out_lse
    return out


def dcp_a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
    use_b12x: bool = False,
    b12x_max_batch_size: int | None = None,
    b12x_query_head_dim: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Combine partial attention outputs across DCP ranks using All-to-All.

    The output and fp32 LSE are packed into a single output-dtype buffer, sent
    with one All-to-All, then unpacked and combined with exact LSE weighting.

    Args:
        cp_attn_out: [B, H, D] where B=num_tokens, H=total_heads, D=head_dim
        cp_attn_lse: [B, H] log-sum-exp values (fp32)
        cp_group: GroupCoordinator for DCP communication
        ctx: CPTritonContext (unused, for signature compatibility)
        return_lse: If True, also return the combined global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2
        use_b12x: Try the low-latency B12X PCIe path before NCCL A2A
        b12x_max_batch_size: Configured token capacity for B12X staging
        b12x_query_head_dim: Query width when it differs from output width

    Returns:
        Combined output [B, H/N, D] (head-scattered)
        If return_lse=True, also returns global_lse [B, H/N]
    """
    world_size = cp_group.world_size

    if world_size == 1:
        if return_lse:
            return cp_attn_out, cp_attn_lse
        return cp_attn_out

    if use_b12x and envs.VLLM_USE_B12X_DCP_A2A:
        b12x_result = _try_b12x_dcp_lse_reduce(
            cp_attn_out,
            cp_attn_lse,
            cp_group,
            return_lse=return_lse,
            is_lse_base_on_e=is_lse_base_on_e,
            max_batch_size=b12x_max_batch_size,
            query_head_dim=b12x_query_head_dim,
        )
        if b12x_result is not None:
            return b12x_result
        token_cap = envs.VLLM_DCP_A2A_MAX_TOKENS
        if (
            token_cap > 0
            and cp_attn_out.shape[0] > token_cap
            and envs.VLLM_DCP_A2A_LARGE_BACKEND == "ag_rs"
        ):
            # ProcessGroupNCCL lazily reserves a sizable transport workspace
            # on its first all-to-all.  That hidden allocation can fail after
            # an explicitly sized, near-capacity KV cache has been created.
            # PyNccl AG+RS uses the already-warmed DCP communicator and keeps
            # the low-latency B12X path unchanged for decode-sized batches.
            from vllm.v1.attention.ops.common import cp_lse_ag_out_rs

            return cp_lse_ag_out_rs(
                cp_attn_out,
                cp_attn_lse,
                cp_group,
                ctx=ctx,
                return_lse=return_lse,
                is_lse_base_on_e=is_lse_base_on_e,
                head_major_output=True,
            )

    B, H, D = cp_attn_out.shape
    if H % world_size != 0:
        raise ValueError(f"H={H} must be divisible by DCP world size {world_size}.")
    H_per_rank = H // world_size
    # The pack kernel bit-casts the LSE as fp32; some MLA backends return it in
    # the activation dtype (bf16/fp16), so enforce the documented fp32 contract.
    if cp_attn_lse.dtype != torch.float32:
        cp_attn_lse = cp_attn_lse.to(torch.float32)
    lse_pack_dim = _dcp_a2a_lse_pack_dim(cp_attn_out.dtype)

    send_buffer, recv_buffer = _dcp_a2a_send_recv_buffers(
        (world_size, B, H_per_rank, D + lse_pack_dim),
        device=cp_attn_out.device,
        dtype=cp_attn_out.dtype,
    )

    _dcp_a2a_pack_send(
        cp_attn_out,
        cp_attn_lse,
        send_buffer,
        world_size,
        H_per_rank,
        D,
        lse_pack_dim,
    )

    work = dist.all_to_all_single(
        recv_buffer.view(-1),
        send_buffer.view(-1),
        group=cp_group.device_group,
        async_op=True,
    )
    work.wait()

    return _dcp_a2a_unpack_combine(
        recv_buffer, D, lse_pack_dim, return_lse, is_lse_base_on_e
    )
