# SPDX-License-Identifier: Apache-2.0
"""Host-staged RPC for a dedicated Kimi-K3 draft GPU.

The verifier and RTX 3090 do not have CUDA peer access on the target host, so
the first transport deliberately uses ZMQ multipart frames backed by host
memory.  The protocol is small and versioned so a verifier-side proxy can be
added without coupling the standalone process to the generic EAGLE draft
server protocol.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import torch

from vllm.config.vllm import set_current_vllm_config
from vllm.distributed.utils import StatelessProcessGroup, create_tcp_store
from vllm.forward_context import set_forward_context
from vllm.k3_rdma_transport import K3RdmaServer
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonDecodeMetadata,
    MLACommonMetadata,
)
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)
from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.utils import draft_gumbel_pos

if TYPE_CHECKING:
    from vllm.entrypoints.k3_dspark_standalone import StandaloneRuntime

logger = init_logger(__name__)

PROTOCOL_VERSION = 3

# Op codes for the RDMA side-channel header (int64[16]).
OP_PING = 0
OP_PROPOSE = 1
OP_FREE = 2
OP_RECONNECT = 3
OP_CLEAR = 4

# Draft method -> small integer code carried in the PING handshake response.
_METHOD_CODE = {"dspark": 0, "dflash": 1}
_CODE_METHOD = {code: method for method, code in _METHOD_CODE.items()}


def draft_query_len(
    sample_from_anchor: bool,
    num_speculative_tokens: int,
) -> int:
    """Return the draft query rows one proposal step runs.

    Args:
        sample_from_anchor: ``True`` samples every draft token from the anchor
            row, so the query holds ``num_speculative_tokens`` rows; ``False``
            prepends the anchor to the mask rows (one extra row).
        num_speculative_tokens: Draft depth of the step.

    Returns:
        The number of query rows.
    """
    if sample_from_anchor:
        return int(num_speculative_tokens)
    return 1 + int(num_speculative_tokens)


def _required_int_field(request: dict[str, Any], name: str) -> int:
    """Return an integer request field, rejecting a missing or non-integer one.

    Args:
        request: One PROPOSE request header entry.
        name: Field name.

    Returns:
        The field value as ``int``.

    Raises:
        ValueError: The field is absent or not an integer.
    """
    value = request.get(name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"PROPOSE request field {name!r} must be an integer")
    return value


def _required_int(value: Any, name: str) -> int:
    """Return an integer control value, rejecting a bool or non-integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"K3 draft {name} must be an integer")
    return value


# ---------------------------------------------------------------------------
# Raw-verbs RDMA point-to-point transport (kernel-bypass RoCEv2).
#
# The verifier (cluster TP-rank-0 worker) and this T1 draft server form a
# dedicated 2-host channel over libk3rdma (raw ibverbs RC). The TCPStore from
# the StatelessProcessGroup rendezvous is reused ONLY to exchange the small
# ibverbs peer-info strings; all tensor payloads ride the RDMA queue pair.
# This replaced the NCCL P2P side-channel, whose comm-init was unreliable
# across the custom-fork (cluster) / stock (T1) NCCL libraries.
# ---------------------------------------------------------------------------

_TCPSTORE_DEFAULT_PORT = 51230
_TCPSTORE_TIMEOUT_SECONDS = 3600
# Server-side recv window. The old NCCL recv blocked indefinitely; a long
# RDMA wait preserves that idle behaviour without killing the thread.
_RDMA_RECV_TIMEOUT_MS = 3600 * 1000


def _tcpstore_port() -> int:
    return int(
        os.environ.get("VLLM_K3_DRAFT_TCPSTORE_PORT", str(_TCPSTORE_DEFAULT_PORT))
    )


def _rdma_hca() -> str:
    return os.environ.get("VLLM_K3_DRAFT_RDMA_HCA", "mlx5_0")


def _rdma_gid_index() -> int:
    return int(os.environ.get("VLLM_K3_DRAFT_RDMA_GID_INDEX", "5"))


def _rdma_port() -> int:
    return int(os.environ.get("VLLM_K3_DRAFT_RDMA_PORT", "1"))


def _tcpstore_timeout() -> int:
    """Rendezvous window in seconds (env-overridable).

    Must comfortably exceed the cluster's time-to-speculator-init (~8-10 min
    from container start). A timed-out master attempt can leak its listen
    socket inside c10d (observed: later attempts fail EADDRINUSE against our
    own stale listener), so one LONG window beats repeated short ones; the
    retry loop stays only as a backstop for non-timeout failures.
    """
    return int(
        os.environ.get(
            "VLLM_K3_DRAFT_TCPSTORE_TIMEOUT_S", str(_TCPSTORE_TIMEOUT_SECONDS)
        )
    )


def _create_stateless_group(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    is_master: bool,
) -> StatelessProcessGroup:
    """Create a StatelessProcessGroup with an explicit TCPStore master.

    ``StatelessProcessGroup.create`` always makes rank 0 the store master, but
    here the always-on T1 server (rank 1) hosts the rendezvous store so the
    cluster verifier (rank 0) does not need to know its own routable IP ahead
    of time. The verifier already knows the T1 address.

    NOTE: do NOT pre-bind a listen socket and pass ``master_listen_fd``. That
    path double-closes the fd inside ``create_tcp_store`` on failure (EBADF
    masking the real error) and broke the T1 master startup. Plain TCPStore
    construction binds the master socket itself.
    """
    try:
        store = create_tcp_store(
            host,
            port,
            world_size=world_size,
            is_master=is_master,
            timeout=timedelta(seconds=_tcpstore_timeout()),
            use_libuv=False,
        )
    except Exception:
        logger.exception(
            "StatelessProcessGroup TCPStore create failed "
            "(host=%s port=%d rank=%d is_master=%s)",
            host,
            port,
            rank,
            is_master,
        )
        raise
    return StatelessProcessGroup(
        rank=rank,
        world_size=world_size,
        store=store,
    )


class ProjectedContextCache:
    """Bounded, chunked cache of projected DSpark context states.

    The standalone draft server keeps this cache on the draft device.  Keeping
    the projected rows on CPU forced a blocking D2H copy after every proposal,
    even though the rows are normally consumed again by the same GPU during a
    prefix reconnect.  CPU remains the default for lightweight unit tests and
    callers which explicitly want host storage.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        max_tokens: int,
        chunk_size: int = 256,
        initial_position: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        if hidden_size <= 0 or max_tokens <= 0 or chunk_size <= 0:
            raise ValueError("Projected context cache dimensions must be positive")
        if initial_position < 0:
            raise ValueError("Projected context cache position cannot be negative")
        self.hidden_size = hidden_size
        self.max_tokens = max_tokens
        self.chunk_size = chunk_size
        self.device = torch.device(device)
        self.start_position = initial_position
        self.end_position = initial_position
        self._chunks: dict[int, torch.Tensor] = {}

    def _truncate(self, end_position: int) -> None:
        if not self.start_position <= end_position <= self.end_position:
            raise ValueError(
                "Cannot truncate projected context outside its retained range: "
                f"retained=[{self.start_position}, {self.end_position}), "
                f"requested_end={end_position}"
            )
        first_discarded_chunk = (end_position + self.chunk_size - 1) // self.chunk_size
        for chunk_idx in list(self._chunks):
            if chunk_idx >= first_discarded_chunk:
                del self._chunks[chunk_idx]
        self.end_position = end_position

    def append(self, first_position: int, states: torch.Tensor) -> None:
        if states.device != self.device:
            raise ValueError(
                "Projected context cache device mismatch: "
                f"cache={self.device}, states={states.device}"
            )
        if states.dtype != torch.bfloat16 or states.ndim != 2:
            raise ValueError("Projected context states must be a 2D BF16 tensor")
        if states.shape[1] != self.hidden_size:
            raise ValueError(
                f"Projected context width is {states.shape[1]}, expected "
                f"{self.hidden_size}"
            )
        if first_position < self.start_position or first_position > self.end_position:
            raise ValueError(
                "Projected context append is not contiguous with retained state: "
                f"retained=[{self.start_position}, {self.end_position}), "
                f"first={first_position}"
            )
        if first_position < self.end_position:
            self._truncate(first_position)

        num_rows = int(states.shape[0])
        final_end = first_position + num_rows
        new_start = max(self.start_position, final_end - self.max_tokens)
        # Evict before allocating incoming chunks so the configured retained
        # capacity is also a bound on transient device allocation.
        for chunk_idx in list(self._chunks):
            if (chunk_idx + 1) * self.chunk_size <= new_start:
                del self._chunks[chunk_idx]

        offset = max(0, new_start - first_position)
        while offset < num_rows:
            position = first_position + offset
            chunk_idx, chunk_offset = divmod(position, self.chunk_size)
            count = min(num_rows - offset, self.chunk_size - chunk_offset)
            chunk = self._chunks.get(chunk_idx)
            if chunk is None:
                chunk = torch.empty(
                    (self.chunk_size, self.hidden_size),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                self._chunks[chunk_idx] = chunk
            chunk[chunk_offset : chunk_offset + count].copy_(
                states[offset : offset + count]
            )
            offset += count

        self.end_position = final_end
        self.start_position = new_start

    def has_range(self, start_position: int, end_position: int) -> bool:
        return (
            self.start_position <= start_position <= end_position <= self.end_position
        )

    def read(self, start_position: int, end_position: int) -> torch.Tensor:
        if not self.has_range(start_position, end_position):
            raise ValueError(
                "Projected context range is unavailable: "
                f"retained=[{self.start_position}, {self.end_position}), "
                f"requested=[{start_position}, {end_position})"
            )
        output = torch.empty(
            (end_position - start_position, self.hidden_size),
            dtype=torch.bfloat16,
            device=self.device,
        )
        offset = 0
        while start_position + offset < end_position:
            position = start_position + offset
            chunk_idx, chunk_offset = divmod(position, self.chunk_size)
            count = min(
                end_position - position,
                self.chunk_size - chunk_offset,
            )
            chunk = self._chunks.get(chunk_idx)
            if chunk is None:
                raise RuntimeError(
                    f"Projected context chunk {chunk_idx} is unexpectedly missing"
                )
            output[offset : offset + count].copy_(
                chunk[chunk_offset : chunk_offset + count]
            )
            offset += count
        return output

    def truncate(self, end_position: int) -> None:
        if end_position < self.end_position:
            self._truncate(end_position)

    @property
    def allocated_bytes(self) -> int:
        return sum(
            chunk.numel() * chunk.element_size() for chunk in self._chunks.values()
        )


@dataclass
class DraftRequestState:
    request_id: str
    slot: int
    committed_end: int = 0
    context_start: int = 0
    context_cache: ProjectedContextCache | None = None


class DraftKVSlotAllocator:
    """Assign fixed rolling MLA block ranges to a small request batch."""

    def __init__(
        self,
        *,
        num_cache_blocks: int,
        block_size: int,
        window_size: int,
        max_requests: int,
    ) -> None:
        if window_size <= 0 or window_size % block_size != 0:
            raise ValueError(
                "DSpark KV window must be a positive block-size multiple, got "
                f"window={window_size}, block_size={block_size}"
            )
        self.block_size = block_size
        self.window_size = window_size
        self.max_requests = max_requests
        # The vLLM rolling window may retain window + block_size - 1 tokens
        # while it waits for the next whole-block shift.
        self.blocks_per_request = window_size // block_size + 1
        required = 1 + max_requests * self.blocks_per_request
        if required > num_cache_blocks:
            raise ValueError(
                "Dedicated draft KV cache is too small for the requested rolling "
                f"slots: required_blocks={required}, available={num_cache_blocks}"
            )
        self._free_slots = list(range(max_requests))
        self._states: dict[str, DraftRequestState] = {}

    def get_or_allocate(self, request_id: str) -> tuple[DraftRequestState, bool]:
        state = self._states.get(request_id)
        if state is not None:
            return state, False
        if not self._free_slots:
            raise RuntimeError(
                f"DSpark request capacity exhausted (max={self.max_requests})"
            )
        slot = self._free_slots.pop(0)
        state = DraftRequestState(request_id=request_id, slot=slot)
        self._states[request_id] = state
        return state, True

    def free(self, request_id: str) -> DraftRequestState | None:
        state = self._states.pop(request_id, None)
        if state is not None:
            self._free_slots.append(state.slot)
            self._free_slots.sort()
        return state

    def get(self, request_id: str) -> DraftRequestState | None:
        return self._states.get(request_id)

    def rebind(self, source_request_id: str, request_id: str) -> DraftRequestState:
        state = self._states.get(source_request_id)
        if state is None:
            raise KeyError(f"Unknown DSpark source request {source_request_id!r}")
        if source_request_id == request_id:
            return state
        if request_id in self._states:
            raise ValueError(f"DSpark request {request_id!r} already exists")
        del self._states[source_request_id]
        state.request_id = request_id
        self._states[request_id] = state
        return state

    def physical_block(self, state: DraftRequestState, position: int) -> int:
        absolute_block = position // self.block_size
        return (
            1
            + state.slot * self.blocks_per_request
            + absolute_block % self.blocks_per_request
        )

    def cache_slot(self, state: DraftRequestState, position: int) -> int:
        return self.physical_block(state, position) * self.block_size + (
            position % self.block_size
        )

    def cache_slots(
        self,
        state: DraftRequestState,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorize rolling physical-slot mapping for one request."""
        if positions.ndim != 1 or positions.dtype != torch.int64:
            raise ValueError("Draft cache positions must be a 1D int64 tensor")
        absolute_blocks = torch.div(
            positions,
            self.block_size,
            rounding_mode="floor",
        )
        physical_blocks = (
            1
            + state.slot * self.blocks_per_request
            + torch.remainder(absolute_blocks, self.blocks_per_request)
        )
        return physical_blocks * self.block_size + torch.remainder(
            positions, self.block_size
        )

    def block_table(
        self, state: DraftRequestState, sequence_end: int
    ) -> tuple[list[int], int]:
        if sequence_end <= 0:
            raise ValueError(f"sequence_end must be positive, got {sequence_end}")
        first_block = (
            max(state.context_start, sequence_end - self.window_size) // self.block_size
        )
        end_block = (sequence_end + self.block_size - 1) // self.block_size
        blocks = [
            1
            + state.slot * self.blocks_per_request
            + absolute_block % self.blocks_per_request
            for absolute_block in range(first_block, end_block)
        ]
        local_sequence_len = sequence_end - first_block * self.block_size
        if local_sequence_len > self.window_size + self.block_size - 1:
            raise AssertionError("rolling DSpark sequence length exceeded its window")
        if len(blocks) > self.blocks_per_request:
            raise AssertionError("rolling DSpark block table aliases a live block")
        return blocks, local_sequence_len

    def physical_block_range(self, state: DraftRequestState) -> slice:
        start = 1 + state.slot * self.blocks_per_request
        return slice(start, start + self.blocks_per_request)

    @property
    def active_requests(self) -> int:
        return len(self._states)

    @property
    def request_ids(self) -> list[str]:
        return list(self._states)


@dataclass
class _DraftCudaGraphState:
    """Persistent inputs and output for one standalone draft graph shape."""

    batch_size: int
    num_speculative_tokens: int
    query_len: int
    input_ids: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    output_tokens: torch.Tensor
    input_ids_host: torch.Tensor
    positions_host: torch.Tensor
    slots_host: torch.Tensor
    seq_lens_host: torch.Tensor
    block_table_host: torch.Tensor
    attn_metadata: dict[str, Any]
    slot_mapping: dict[str, torch.Tensor]
    graph: torch.cuda.CUDAGraph | None = None
    captured_hidden: torch.Tensor | None = None
    captured_logits: torch.Tensor | None = None

    def stage(
        self,
        *,
        input_ids: list[int],
        positions: list[int],
        slots: list[int],
        block_rows: list[list[int]],
        seq_lens: list[int],
    ) -> None:
        """Copy one request batch into address-stable graph inputs."""
        expected_tokens = self.batch_size * self.query_len
        if not (
            len(input_ids) == len(positions) == len(slots) == expected_tokens
            and len(block_rows) == len(seq_lens) == self.batch_size
        ):
            raise ValueError("Draft CUDA graph input shape mismatch")

        self.input_ids_host.copy_(torch.tensor(input_ids, dtype=torch.int64))
        self.positions_host.copy_(torch.tensor(positions, dtype=torch.int64))
        self.slots_host.copy_(torch.tensor(slots, dtype=torch.int64))
        self.seq_lens_host.copy_(torch.tensor(seq_lens, dtype=torch.int32))
        self.block_table_host.zero_()
        for row_idx, row in enumerate(block_rows):
            if len(row) > self.block_table_host.shape[1]:
                raise ValueError(
                    "Draft block table exceeds CUDA graph capacity: "
                    f"row={len(row)}, capacity={self.block_table_host.shape[1]}"
                )
            self.block_table_host[row_idx, : len(row)].copy_(
                torch.tensor(row, dtype=torch.int32)
            )

        # All copies and the replay are enqueued on the same stream. The
        # proposal's final query event synchronizes before these pinned host
        # buffers can be reused by the next (serialized) RPC.
        self.input_ids.copy_(self.input_ids_host, non_blocking=True)
        self.positions.copy_(self.positions_host, non_blocking=True)
        self.slots.copy_(self.slots_host, non_blocking=True)
        self.seq_lens.copy_(self.seq_lens_host, non_blocking=True)
        self.block_table.copy_(self.block_table_host, non_blocking=True)


class K3DSparkDraftEngine:
    """Minimal greedy K3 draft scheduler backed by the 3090 KV cache."""

    def __init__(
        self,
        runtime: StandaloneRuntime,
        *,
        max_requests: int,
        window_size: int,
        device: torch.device,
    ) -> None:
        self.runtime = runtime
        self.model = runtime.model
        self.method = runtime.method
        self.device = device
        self.max_model_len = int(runtime.vllm_config.model_config.max_model_len)
        self._draft_sample_method = str(
            runtime.vllm_config.speculative_config.draft_sample_method
        )
        first_cache = next(iter(runtime.kv_caches.values()))
        self.allocator = DraftKVSlotAllocator(
            num_cache_blocks=int(first_cache.shape[0]),
            block_size=runtime.kv_cache_block_size,
            window_size=window_size,
            max_requests=max_requests,
        )
        speculative_config = runtime.vllm_config.speculative_config
        assert speculative_config is not None
        draft_config = speculative_config.draft_model_config.hf_config
        self.hidden_size = int(draft_config.hidden_size)
        aux_layers = get_eagle3_aux_layers_from_config(speculative_config)
        if not aux_layers:
            raise ValueError(
                f"K3 {self.method} config does not declare target auxiliary layers"
            )
        self.num_aux_layers = len(aux_layers)
        target_hidden_size = int(
            getattr(draft_config, "target_hidden_size", None)
            or draft_config.hidden_size
        )
        self.raw_context_width = int(target_hidden_size * self.num_aux_layers)
        self.mask_token_id = get_parallel_drafting_token_id(draft_config)
        self.max_speculative_tokens = int(
            runtime.vllm_config.speculative_config.num_speculative_tokens
        )
        self.max_context_tokens = int(
            runtime.vllm_config.scheduler_config.max_num_batched_tokens
        )
        self.max_batch_size = int(runtime.vllm_config.scheduler_config.max_num_seqs)
        if max_requests < self.max_batch_size:
            raise ValueError(
                "Retained request capacity must cover every accepted batch: "
                f"retained={max_requests}, batch={self.max_batch_size}"
            )
        # The draft samples every token from the anchor row when the checkpoint
        # declares sample_from_anchor; otherwise the anchor is prepended to the
        # mask rows. Single source for the engine, CUDA-graph, and standalone
        # query-length formulas.
        self.sample_from_anchor = bool(
            getattr(draft_config, "sample_from_anchor", True)
        )
        self.prefix_cache_tokens = int(
            os.environ.get(
                "VLLM_K3_DRAFT_PREFIX_CACHE_TOKENS",
                os.environ.get("VLLM_K3_DSPARK_PREFIX_CACHE_TOKENS", "131072"),
            )
        )
        if self.prefix_cache_tokens < self.allocator.window_size:
            raise ValueError(
                "VLLM_K3_DRAFT_PREFIX_CACHE_TOKENS must be at least the "
                f"draft KV window ({self.allocator.window_size}), got "
                f"{self.prefix_cache_tokens}"
            )
        self._positions_staging = torch.empty(
            self.max_context_tokens,
            dtype=torch.int64,
            pin_memory=True,
        )
        self._context_staging = torch.empty(
            self.max_context_tokens * self.raw_context_width,
            dtype=torch.bfloat16,
            pin_memory=True,
        )
        self._lock = threading.RLock()
        self.proposal_count = 0
        self.last_latency_ms = 0.0
        self.last_timing_ms: dict[str, float] = {}
        self._timing_totals_ms: dict[str, float] = {}
        self.cold_bootstrap_count = 0
        self.reconnect_count = 0
        self.last_reconnect_latency_ms = 0.0
        self.cuda_graph_enabled = False
        self.cuda_graph_capture_seconds = 0.0
        self.cuda_graph_memory_gib = 0.0
        self.cuda_graph_replay_count = 0
        self.cuda_graph_eager_fallback_count = 0
        self._cuda_graphs: dict[tuple[int, int], _DraftCudaGraphState] = {}

    def _make_cuda_graph_state(
        self,
        batch_size: int,
        num_speculative_tokens: int,
    ) -> _DraftCudaGraphState:
        if self.runtime.is_gqa and self.runtime.attn_metadata_builder is None:
            raise RuntimeError("K3 DFlash attention metadata builder is missing")

        query_len = draft_query_len(
            self.sample_from_anchor,
            num_speculative_tokens,
        )
        if query_len > self.allocator.block_size:
            raise ValueError(
                "Draft CUDA graph dummy sequence must fit in one mapped KV "
                f"block: query_len={query_len}, "
                f"block_size={self.allocator.block_size}"
            )
        num_tokens = batch_size * query_len
        max_blocks = self.allocator.blocks_per_request
        input_ids = torch.empty(num_tokens, dtype=torch.int64, device=self.device)
        positions = torch.empty(num_tokens, dtype=torch.int64, device=self.device)
        slots = torch.empty(num_tokens, dtype=torch.int64, device=self.device)
        seq_lens = torch.empty(batch_size, dtype=torch.int32, device=self.device)
        block_table = torch.zeros(
            (batch_size, max_blocks), dtype=torch.int32, device=self.device
        )
        output_tokens = torch.empty(
            (batch_size, num_speculative_tokens),
            dtype=torch.int64,
            device=self.device,
        )

        input_ids_host = torch.empty(num_tokens, dtype=torch.int64, pin_memory=True)
        positions_host = torch.empty(num_tokens, dtype=torch.int64, pin_memory=True)
        slots_host = torch.empty(num_tokens, dtype=torch.int64, pin_memory=True)
        seq_lens_host = torch.empty(batch_size, dtype=torch.int32, pin_memory=True)
        block_table_host = torch.zeros(
            (batch_size, max_blocks), dtype=torch.int32, pin_memory=True
        )

        # The graph's launch topology is shape-static, while seq_lens remains
        # a live tensor. Triton BF16 attention uses seq_lens to bound the KV
        # scan; this conservative upper bound therefore does not force every
        # replay to scan the full rolling window.
        max_seq_len = self.allocator.window_size + self.allocator.block_size - 1
        query_start_cpu = torch.arange(
            0,
            (batch_size + 1) * query_len,
            query_len,
            dtype=torch.int32,
        )
        query_start_gpu = query_start_cpu.to(self.device)
        if self.runtime.is_gqa:
            common = CommonAttentionMetadata(
                query_start_loc=query_start_gpu,
                query_start_loc_cpu=query_start_cpu,
                seq_lens=seq_lens,
                seq_lens_cpu_upper_bound=torch.full(
                    (batch_size,), max_seq_len, dtype=torch.int32
                ),
                max_seq_len=max_seq_len,
                num_reqs=batch_size,
                num_actual_tokens=num_tokens,
                max_query_len=query_len,
                block_table_tensor=block_table,
                slot_mapping=slots,
                causal=True,
            )
            assert self.runtime.attn_metadata_builder is not None
            metadata = self.runtime.attn_metadata_builder.build(0, common)
        else:
            metadata = MLACommonMetadata(
                num_reqs=batch_size,
                max_query_len=query_len,
                max_seq_len=max_seq_len,
                num_actual_tokens=num_tokens,
                query_start_loc=query_start_gpu,
                slot_mapping=slots,
                num_decodes=batch_size,
                num_decode_tokens=num_tokens,
                num_prefills=0,
                causal=False,
                head_dim=int(next(iter(self.runtime.kv_caches.values())).shape[-1]),
                prefill=None,
                decode=MLACommonDecodeMetadata(
                    block_table=block_table,
                    seq_lens=seq_lens,
                    dcp_tot_seq_lens=None,
                ),
            )
        attn_metadata = {layer_name: metadata for layer_name in self.runtime.kv_caches}
        slot_mapping = {layer_name: slots for layer_name in self.runtime.kv_caches}
        state = _DraftCudaGraphState(
            batch_size=batch_size,
            num_speculative_tokens=num_speculative_tokens,
            query_len=query_len,
            input_ids=input_ids,
            positions=positions,
            slots=slots,
            seq_lens=seq_lens,
            block_table=block_table,
            output_tokens=output_tokens,
            input_ids_host=input_ids_host,
            positions_host=positions_host,
            slots_host=slots_host,
            seq_lens_host=seq_lens_host,
            block_table_host=block_table_host,
            attn_metadata=attn_metadata,
            slot_mapping=slot_mapping,
        )

        # Seed capture with valid, isolated dummy sequences. Runtime replay
        # overwrites every graph input before use.
        dummy_input_ids: list[int] = []
        dummy_positions: list[int] = []
        dummy_slots: list[int] = []
        dummy_rows: list[list[int]] = []
        for request_idx in range(batch_size):
            dummy_input_ids.append(0)
            dummy_input_ids.extend([self.mask_token_id] * (query_len - 1))
            dummy_positions.extend(range(query_len))
            dummy_slots.extend(
                request_idx * self.allocator.block_size + position
                for position in range(query_len)
            )
            dummy_rows.append([request_idx])
        state.stage(
            input_ids=dummy_input_ids,
            positions=dummy_positions,
            slots=dummy_slots,
            block_rows=dummy_rows,
            seq_lens=[query_len] * batch_size,
        )
        return state

    def _run_cuda_graph_state(
        self,
        state: _DraftCudaGraphState,
    ) -> None:
        num_tokens = state.batch_size * state.query_len
        with (
            set_current_vllm_config(self.runtime.draft_vllm_config),
            set_forward_context(
                state.attn_metadata,
                self.runtime.draft_vllm_config,
                num_tokens=num_tokens,
                skip_compiled=True,
                slot_mapping=state.slot_mapping,
            ),
        ):
            hidden = self.model(
                input_ids=state.input_ids,
                positions=state.positions,
            )
        logits = self.model.compute_draft_logits(hidden).view(
            state.batch_size, state.query_len, -1
        )
        # Anchor-start loop (matches _run_query_block): the capture path's
        # input_ids row 0 is the anchor token and query_len ==
        # num_speculative_tokens when the draft samples from the anchor, so
        # every query row produces one draft token.
        previous = state.input_ids.view(state.batch_size, state.query_len)[:, 0]
        for step in range(state.query_len):
            markov = self.model.markov_bias(self.model.markov_embed(previous))
            previous = (logits[:, step] + markov).argmax(dim=-1)
            state.output_tokens[:, step].copy_(previous)
        # Retain graph-owned outputs so their backing allocations cannot be
        # recycled while captured nodes still reference them.
        state.captured_hidden = hidden
        state.captured_logits = logits

    @torch.inference_mode()
    def capture_cuda_graphs(self, *, warmups: int = 2) -> None:
        """Capture all DSpark or DFlash shapes selectable by this server."""
        if warmups < 1:
            raise ValueError("Draft CUDA graph capture requires at least one warmup")
        if self._cuda_graphs:
            return

        started = time.perf_counter()
        allocated_before = torch.cuda.memory_allocated(self.device)
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(torch.cuda.current_stream(self.device))
        # Capture the largest shape first. Triton MLA grows shared workspace
        # buffers on demand; capturing a smaller shape first and then resizing
        # that workspace for B2/K3 leaves the earlier graph with stale device
        # pointers and causes an illegal access on replay.
        shapes = [
            (batch_size, depth)
            for batch_size in range(
                self.runtime.vllm_config.scheduler_config.max_num_seqs,
                0,
                -1,
            )
            for depth in range(self.max_speculative_tokens, 0, -1)
        ]
        logger.info(
            "Capturing %d standalone K3 %s CUDA graphs on %s: %s",
            len(shapes),
            self.method,
            self.device,
            ", ".join(f"B{batch_size}K{depth}" for batch_size, depth in shapes),
        )
        with torch.cuda.stream(capture_stream):
            for batch_size, depth in shapes:
                state = self._make_cuda_graph_state(batch_size, depth)
                for _ in range(warmups):
                    self._run_cuda_graph_state(state)
                capture_stream.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph,
                    stream=capture_stream,
                ):
                    self._run_cuda_graph_state(state)
                state.graph = graph
                self._cuda_graphs[(batch_size, depth)] = state

        torch.cuda.current_stream(self.device).wait_stream(capture_stream)
        torch.cuda.synchronize(self.device)
        # Dummy capture rows only touch these reserved low blocks. Block zero
        # is never assigned; live request allocation clears its own full range.
        max_dummy_blocks = int(self.runtime.vllm_config.scheduler_config.max_num_seqs)
        for cache in self.runtime.kv_caches.values():
            cache[:max_dummy_blocks].zero_()
        torch.cuda.synchronize(self.device)

        self.cuda_graph_enabled = True
        self.cuda_graph_capture_seconds = time.perf_counter() - started
        self.cuda_graph_memory_gib = max(
            0.0,
            (torch.cuda.memory_allocated(self.device) - allocated_before) / 1024**3,
        )
        logger.info(
            "Standalone K3 %s CUDA graphs ready in %.2fs; allocated_delta=%.3f GiB",
            self.method,
            self.cuda_graph_capture_seconds,
            self.cuda_graph_memory_gib,
        )

    @property
    def cuda_graph_shapes(self) -> list[str]:
        return [
            f"B{batch_size}K{depth}" for batch_size, depth in sorted(self._cuda_graphs)
        ]

    def _clear_state_cache(
        self,
        state: DraftRequestState,
        *,
        clear_context: bool = True,
    ) -> None:
        block_range = self.allocator.physical_block_range(state)
        for cache in self.runtime.kv_caches.values():
            cache[block_range].zero_()
        state.committed_end = 0
        state.context_start = 0
        if clear_context:
            state.context_cache = None

    def reset(self, request_ids: list[str]) -> None:
        with self._lock:
            for request_id in request_ids:
                state, _ = self.allocator.get_or_allocate(request_id)
                self._clear_state_cache(state)

    def free(self, request_ids: list[str]) -> None:
        with self._lock:
            for request_id in request_ids:
                self.allocator.free(request_id)

    def clear(self) -> None:
        self.free(self.allocator.request_ids)

    @property
    def prefix_cache_bytes(self) -> int:
        return sum(
            state.context_cache.allocated_bytes
            for request_id in self.allocator.request_ids
            if (state := self.allocator.get(request_id)) is not None
            and state.context_cache is not None
        )

    @property
    def prefix_cache_host_bytes(self) -> int:
        return sum(
            state.context_cache.allocated_bytes
            for request_id in self.allocator.request_ids
            if (state := self.allocator.get(request_id)) is not None
            and state.context_cache is not None
            and state.context_cache.device.type == "cpu"
        )

    @property
    def prefix_cache_device_bytes(self) -> int:
        return self.prefix_cache_bytes - self.prefix_cache_host_bytes

    @property
    def mean_timing_ms(self) -> dict[str, float]:
        if self.proposal_count <= 0:
            return {}
        return {
            key: value / self.proposal_count
            for key, value in self._timing_totals_ms.items()
        }

    def _record_timing(self, timing_ms: dict[str, float]) -> None:
        self.last_timing_ms = timing_ms
        for key, value in timing_ms.items():
            self._timing_totals_ms[key] = self._timing_totals_ms.get(key, 0.0) + value

    def _window_restore_start(self, prefix_end: int) -> int:
        """Return the block-aligned first position a window restore rebuilds."""
        start = max(0, prefix_end - self.allocator.window_size)
        return start // self.allocator.block_size * self.allocator.block_size

    def _restore_projected_context(
        self,
        state: DraftRequestState,
        prefix_end: int,
    ) -> int:
        context_cache = state.context_cache
        if context_cache is None:
            raise ValueError(
                f"No projected context is retained for {state.request_id!r}"
            )
        restore_start = self._window_restore_start(prefix_end)
        if not context_cache.has_range(restore_start, prefix_end):
            raise ValueError(
                f"Projected context for {state.request_id!r} cannot restore "
                f"prefix_end={prefix_end}; retained="
                f"[{context_cache.start_position}, {context_cache.end_position})"
            )

        self._clear_state_cache(state, clear_context=False)
        for start in range(restore_start, prefix_end, self.max_context_tokens):
            end = min(prefix_end, start + self.max_context_tokens)
            context_states = context_cache.read(start, end)
            positions = torch.arange(start, end, dtype=torch.int64)
            context_gpu = context_states.to(self.device, non_blocking=True)
            positions_gpu = positions.to(self.device, non_blocking=True)
            slots = self.allocator.cache_slots(state, positions).to(
                self.device,
                non_blocking=True,
            )
            self.model.precompute_and_store_context_kv(
                context_gpu,
                positions_gpu,
                slots,
            )
        state.committed_end = prefix_end
        state.context_start = restore_start
        context_cache.truncate(prefix_end)
        return restore_start

    def reconnect(
        self,
        source_request_id: str,
        request_id: str,
        prefix_end: int,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        with self._lock:
            state = self.allocator.get(source_request_id)
            if state is None:
                raise KeyError(f"Unknown DSpark source request {source_request_id!r}")
            context_cache = state.context_cache
            restore_start = self._window_restore_start(prefix_end)
            if (
                prefix_end <= 0
                or context_cache is None
                or not context_cache.has_range(restore_start, prefix_end)
            ):
                retained = (
                    None
                    if context_cache is None
                    else [context_cache.start_position, context_cache.end_position]
                )
                raise ValueError(
                    f"Cannot reconnect {source_request_id!r} at {prefix_end}; "
                    f"retained={retained}"
                )
            state = self.allocator.rebind(source_request_id, request_id)
            restored_start = self._restore_projected_context(state, prefix_end)
            torch.accelerator.synchronize()
            self.reconnect_count += 1
            self.last_reconnect_latency_ms = (time.perf_counter() - started) * 1000
            return {
                "ok": True,
                "protocol": PROTOCOL_VERSION,
                "request_id": request_id,
                "restored_start": restored_start,
                "prefix_end": prefix_end,
                "latency_ms": self.last_reconnect_latency_ms,
                "active_requests": self.allocator.active_requests,
            }

    def _decode_host_tensor(
        self,
        frame: bytes,
        *,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        element_size = torch.tensor([], dtype=dtype).element_size()
        expected = element_size
        for dim in shape:
            expected *= dim
        if len(frame) != expected:
            raise ValueError(
                f"Tensor frame has {len(frame)} bytes, expected {expected} for "
                f"shape={shape}, dtype={dtype}"
            )
        source = torch.frombuffer(frame, dtype=dtype)
        if dtype == torch.int64:
            staging = self._positions_staging
        elif dtype == torch.bfloat16:
            staging = self._context_staging
        else:
            raise TypeError(f"Unsupported DSpark RPC tensor dtype: {dtype}")
        if source.numel() > staging.numel():
            raise ValueError(
                f"Tensor frame exceeds pinned staging capacity: "
                f"elements={source.numel()}, capacity={staging.numel()}"
            )
        output = staging[: source.numel()]
        output.copy_(source)
        return output.view(shape)

    def _append_context(
        self,
        states: list[DraftRequestState],
        context_counts: list[int],
        positions_cpu: torch.Tensor,
        context_cpu: torch.Tensor,
        *,
        projected: bool,
    ) -> None:
        if positions_cpu.numel() == 0:
            return
        # The RDMA path stages positions through host memory. The slot loop
        # below does one ``int()`` per token; keep the (small) positions
        # tensor on the host for that, while the (large) context stays on
        # the device and is never staged through host memory.
        if positions_cpu.is_cuda:
            positions_cpu = positions_cpu.cpu()
        positions = positions_cpu.to(self.device, non_blocking=True)
        context_input = context_cpu.to(self.device, non_blocking=True)
        context_states = (
            context_input
            if projected
            else self.model.combine_hidden_states(context_input)
        )

        slots_cpu = torch.empty_like(positions_cpu)
        offset = 0
        for state, count in zip(states, context_counts, strict=True):
            req_positions = positions_cpu[offset : offset + count]
            if count:
                first = int(req_positions[0])
                if first > state.committed_end:
                    raise ValueError(
                        f"Context gap for {state.request_id!r}: "
                        f"expected <= {state.committed_end}, got {first}"
                    )
                expected = torch.arange(first, first + count, dtype=torch.int64)
                if not torch.equal(req_positions, expected):
                    raise ValueError(
                        f"Context positions for {state.request_id!r} are not contiguous"
                    )
                state.committed_end = int(req_positions[-1]) + 1
                slots_cpu[offset : offset + count].copy_(
                    self.allocator.cache_slots(state, req_positions)
                )
            offset += count
        slot_mapping = slots_cpu.to(self.device, non_blocking=True)
        self.model.precompute_and_store_context_kv(
            context_states,
            positions,
            slot_mapping,
        )
        offset = 0
        for state, count in zip(states, context_counts, strict=True):
            if count:
                first_position = int(positions_cpu[offset])
                if state.context_cache is None:
                    state.context_cache = ProjectedContextCache(
                        hidden_size=self.hidden_size,
                        max_tokens=self.prefix_cache_tokens,
                        initial_position=first_position,
                        device=self.device,
                    )
                state.context_cache.append(
                    first_position,
                    context_states[offset : offset + count],
                )
            offset += count

    def _run_query_block(
        self,
        states: list[DraftRequestState],
        anchor_positions: list[int],
        anchor_token_ids: list[int],
        num_speculative_tokens: int,
        temperatures: list[float] | None = None,
        seeds: list[int] | None = None,
    ) -> torch.Tensor:
        batch_size = len(states)
        # V6: match the local DSpark speculator's sample_from_anchor semantics
        # exactly (True for this draft). sample_from_anchor=True ->
        # num_query_per_req = N: the anchor token sits at position 0 and
        # predicts spec token 1, and the Markov loop iterates all N positions.
        # The old not-is_gqa=False path used 1+N positions and skipped the
        # anchor, which degraded positions 2-3.
        query_len = draft_query_len(
            self.sample_from_anchor,
            num_speculative_tokens,
        )
        input_ids: list[int] = []
        positions: list[int] = []
        slots: list[int] = []
        block_rows: list[list[int]] = []
        seq_lens: list[int] = []

        for state, anchor_position, anchor_token_id in zip(
            states,
            anchor_positions,
            anchor_token_ids,
            strict=True,
        ):
            if anchor_position != state.committed_end:
                raise ValueError(
                    f"Anchor position for {state.request_id!r} must equal the "
                    f"committed context end ({state.committed_end}), got "
                    f"{anchor_position}"
                )
            sequence_end = anchor_position + query_len
            if sequence_end > self.max_model_len:
                raise ValueError(
                    f"Draft query exceeds max_model_len={self.max_model_len}: "
                    f"end={sequence_end}"
                )
            input_ids.append(anchor_token_id)
            input_ids.extend([self.mask_token_id] * (query_len - 1))
            req_positions = range(anchor_position, sequence_end)
            positions.extend(req_positions)
            slots.extend(
                self.allocator.cache_slot(state, position)
                for position in range(anchor_position, sequence_end)
            )
            row, local_seq_len = self.allocator.block_table(state, sequence_end)
            block_rows.append(row)
            seq_lens.append(local_seq_len)

        if self.cuda_graph_enabled:
            graph_state = self._cuda_graphs.get((batch_size, num_speculative_tokens))
            if graph_state is not None:
                graph_state.stage(
                    input_ids=input_ids,
                    positions=positions,
                    slots=slots,
                    block_rows=block_rows,
                    seq_lens=seq_lens,
                )
                assert graph_state.graph is not None
                graph_state.graph.replay()
                self.cuda_graph_replay_count += 1
                return graph_state.output_tokens
            self.cuda_graph_eager_fallback_count += 1

        max_blocks = max(len(row) for row in block_rows)
        block_table = torch.zeros(
            (batch_size, max_blocks), dtype=torch.int32, device=self.device
        )
        for row_idx, row in enumerate(block_rows):
            block_table[row_idx, : len(row)] = torch.tensor(
                row, dtype=torch.int32, device=self.device
            )
        input_ids_gpu = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        positions_gpu = torch.tensor(positions, dtype=torch.int64, device=self.device)
        slots_gpu = torch.tensor(slots, dtype=torch.int64, device=self.device)
        seq_lens_gpu = torch.tensor(seq_lens, dtype=torch.int32, device=self.device)
        query_start_loc = torch.arange(
            0,
            (batch_size + 1) * query_len,
            query_len,
            dtype=torch.int32,
            device=self.device,
        )
        if self.runtime.is_gqa:
            query_start_loc_cpu = torch.arange(
                0,
                (batch_size + 1) * query_len,
                query_len,
                dtype=torch.int32,
            )
            common = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens_gpu,
                seq_lens_cpu_upper_bound=torch.tensor(seq_lens, dtype=torch.int32),
                max_seq_len=max(seq_lens),
                num_reqs=batch_size,
                num_actual_tokens=batch_size * query_len,
                max_query_len=query_len,
                block_table_tensor=block_table,
                slot_mapping=slots_gpu,
                causal=True,
            )
            if self.runtime.attn_metadata_builder is None:
                raise RuntimeError("K3 DFlash attention metadata builder is missing")
            metadata = self.runtime.attn_metadata_builder.build(0, common)
        else:
            metadata = MLACommonMetadata(
                num_reqs=batch_size,
                max_query_len=query_len,
                max_seq_len=max(seq_lens),
                num_actual_tokens=batch_size * query_len,
                query_start_loc=query_start_loc,
                slot_mapping=slots_gpu,
                num_decodes=batch_size,
                num_decode_tokens=batch_size * query_len,
                num_prefills=0,
                causal=False,
                head_dim=int(next(iter(self.runtime.kv_caches.values())).shape[-1]),
                prefill=None,
                decode=MLACommonDecodeMetadata(
                    block_table=block_table,
                    seq_lens=seq_lens_gpu,
                    dcp_tot_seq_lens=None,
                ),
            )
        attn_metadata = {layer_name: metadata for layer_name in self.runtime.kv_caches}
        slot_mapping = {layer_name: slots_gpu for layer_name in self.runtime.kv_caches}
        with (
            set_current_vllm_config(self.runtime.draft_vllm_config),
            set_forward_context(
                attn_metadata,
                self.runtime.draft_vllm_config,
                num_tokens=batch_size * query_len,
                skip_compiled=True,
                slot_mapping=slot_mapping,
            ),
        ):
            hidden = self.model(input_ids=input_ids_gpu, positions=positions_gpu)

        if self.runtime.is_gqa:
            # V6: sequential Markov sampling (same as the local DSpark
            # speculator) — the draft predicts each spec token conditioned
            # on the previous token via the markov bias. query_len =
            # 1 + num_speculative_tokens (anchor + N spec); step 0 is the
            # anchor's own prediction, which we skip. The naive parallel
            # compute_logits argmax had no markov bias or autoregressive
            # conditioning, so positions 2+ diverged (per-position
            # acceptance 0.167/0/0). This matches local acceptance.
            base_logits = self.model.compute_draft_logits(hidden).view(
                batch_size, query_len, -1
            )
            previous = torch.tensor(
                anchor_token_ids, dtype=torch.int64, device=self.device
            )
            draft_tokens = torch.empty(
                (batch_size, num_speculative_tokens),
                dtype=torch.int64,
                device=self.device,
            )
            probabilistic = self._draft_sample_method == "probabilistic"
            for step in range(query_len):
                markov = self.model.markov_bias(self.model.markov_embed(previous))
                logits = base_logits[:, step] + markov
                if probabilistic:
                    temp = (
                        torch.tensor(
                            temperatures, dtype=torch.float32, device=self.device
                        )
                        if temperatures
                        else torch.ones(batch_size, device=self.device)
                    )
                    seed = (
                        torch.tensor(
                            seeds, dtype=torch.int64, device=self.device
                        )
                        if seeds
                        else torch.zeros(batch_size, dtype=torch.int64, device=self.device)
                    )
                    # Gumbel-max sampling in draft-vocab space. The draft
                    # stream is keyed by step offset so it stays disjoint from
                    # the target-side verifier stream.
                    previous = gumbel_sample(
                        logits,
                        torch.arange(batch_size, device=self.device),
                        temp,
                        seed,
                        draft_gumbel_pos(
                            torch.full((batch_size,), step, device=self.device)
                        ),
                        apply_temperature=True,
                    )
                else:
                    previous = logits.argmax(dim=-1)
                draft_tokens[:, step].copy_(previous)
            return draft_tokens

        base_logits = self.model.compute_draft_logits(hidden).view(
            batch_size, query_len, -1
        )
        previous = torch.tensor(anchor_token_ids, dtype=torch.int64, device=self.device)
        draft_tokens = torch.empty(
            (batch_size, query_len), dtype=torch.int64, device=self.device
        )
        probabilistic = self._draft_sample_method == "probabilistic"
        for step in range(query_len):
            markov = self.model.markov_bias(self.model.markov_embed(previous))
            logits = base_logits[:, step] + markov
            if probabilistic:
                temp = (
                    torch.tensor(
                        temperatures, dtype=torch.float32, device=self.device
                    )
                    if temperatures
                    else torch.ones(batch_size, device=self.device)
                )
                seed = (
                    torch.tensor(
                        seeds, dtype=torch.int64, device=self.device
                    )
                    if seeds
                    else torch.zeros(batch_size, dtype=torch.int64, device=self.device)
                )
                previous = gumbel_sample(
                    logits,
                    torch.arange(batch_size, device=self.device),
                    temp,
                    seed,
                    draft_gumbel_pos(
                        torch.full((batch_size,), step, device=self.device)
                    ),
                    apply_temperature=True,
                )
            else:
                previous = logits.argmax(dim=-1)
            draft_tokens[:, step].copy_(previous)
        return draft_tokens

    @torch.inference_mode()
    def propose(
        self,
        header: dict[str, Any],
        positions_gpu: torch.Tensor | None = None,
        context_gpu: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        started = time.perf_counter()
        requests = header.get("requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError("PROPOSE requires a non-empty requests list")
        if len(requests) > self.allocator.max_requests:
            raise ValueError(
                f"Batch has {len(requests)} requests, max is "
                f"{self.allocator.max_requests}"
            )
        num_speculative_tokens = int(
            header.get("num_speculative_tokens", self.max_speculative_tokens)
        )
        if not 1 <= num_speculative_tokens <= self.max_speculative_tokens:
            raise ValueError(
                f"num_speculative_tokens must be in [1, {self.max_speculative_tokens}]"
            )
        projected = bool(header.get("projected", False))
        context_counts = [int(req.get("context_count", 0)) for req in requests]
        if any(count < 0 for count in context_counts):
            raise ValueError("context_count cannot be negative")
        total_context = sum(context_counts)
        context_width = self.hidden_size if projected else self.raw_context_width
        if total_context:
            if positions_gpu is None or context_gpu is None:
                raise ValueError(
                    "PROPOSE with context requires GPU positions/context tensors"
                )
            if int(positions_gpu.shape[0]) != total_context:
                raise ValueError(
                    f"PROPOSE positions has {positions_gpu.shape[0]} rows, "
                    f"expected {total_context}"
                )
            if tuple(context_gpu.shape) != (total_context, context_width):
                raise ValueError(
                    f"PROPOSE context has shape {tuple(context_gpu.shape)}, "
                    f"expected {(total_context, context_width)}"
                )
            positions_tensor = positions_gpu
            context_tensor = context_gpu
        else:
            positions_tensor = torch.empty(0, dtype=torch.int64, device=self.device)
            context_tensor = torch.empty(
                (0, context_width), dtype=torch.bfloat16, device=self.device
            )
        decoded_at = time.perf_counter()

        lock_started = time.perf_counter()
        with self._lock:
            lock_acquired = time.perf_counter()
            gpu_start = torch.cuda.Event(enable_timing=True)
            context_end = torch.cuda.Event(enable_timing=True)
            query_end = torch.cuda.Event(enable_timing=True)
            gpu_start.record()
            states: list[DraftRequestState] = []
            for req in requests:
                request_id = req.get("request_id")
                if not isinstance(request_id, str) or not request_id:
                    raise ValueError("Every request requires a non-empty request_id")
                state, created = self.allocator.get_or_allocate(request_id)
                if bool(req.get("reset", False)) or created:
                    self._clear_state_cache(state)
                    reset_position = int(req.get("reset_position", 0))
                    if not 0 <= reset_position <= self.max_model_len:
                        raise ValueError(
                            "Draft reset_position must be in model bounds, got "
                            f"{reset_position}"
                        )
                    state.committed_end = reset_position
                    state.context_start = reset_position
                    if reset_position:
                        self.cold_bootstrap_count += 1
                states.append(state)
            self._append_context(
                states,
                context_counts,
                positions_tensor,
                context_tensor,
                projected=projected,
            )
            context_end.record()
            anchor_positions = [
                _required_int_field(req, "anchor_position") for req in requests
            ]
            anchor_token_ids = [
                _required_int_field(req, "anchor_token_id") for req in requests
            ]
            temperatures = [float(req.get("temperature", 1.0)) for req in requests]
            seeds = [int(req.get("seed", 0)) for req in requests]
            draft_tokens = self._run_query_block(
                states,
                anchor_positions,
                anchor_token_ids,
                num_speculative_tokens,
                temperatures,
                seeds,
            )
            query_end.record()
            submit_done = time.perf_counter()
            query_end.synchronize()
            sync_done = time.perf_counter()
            self.proposal_count += 1
            self.last_latency_ms = (sync_done - started) * 1000
            timing_ms = {
                "decode_frames": (decoded_at - started) * 1000,
                "lock_wait": (lock_acquired - lock_started) * 1000,
                "host_submit": (submit_done - lock_acquired) * 1000,
                "gpu_context": gpu_start.elapsed_time(context_end),
                "gpu_query": context_end.elapsed_time(query_end),
                "gpu_wait": (sync_done - submit_done) * 1000,
                "total": self.last_latency_ms,
            }
            self._record_timing(timing_ms)
        return (
            draft_tokens,
            [
                1,
                int(self.last_latency_ms * 1000),
                self.allocator.active_requests,
                0,
            ],
        )


class K3DSparkNCCLServer:
    """Blocking RDMA request/response loop for the T1 draft server.

    The server joins the dedicated 2-host channel as rank 1 (blocking until
    the cluster's rank-0 verifier connects), then blocks for each request
    message, dispatches by op, slices the payload tensors, computes with the
    existing engine code paths, and sends the response bytes. The HTTP
    /v1/status endpoint is served independently by the standalone process and
    is used only for bootstrap/health.

    The class name is kept for import compatibility; the transport is now the
    raw-verbs RDMA side-channel (formerly the NCCL P2P side-channel).
    """

    def __init__(
        self,
        engine: K3DSparkDraftEngine,
        *,
        stop: threading.Event,
        rank: int = 1,
        world_size: int = 2,
    ) -> None:
        self.engine = engine
        self.stop = stop
        self.rank = rank
        self.world_size = world_size
        self.ready = threading.Event()
        self.error: str | None = None
        self.rdma: K3RdmaServer | None = None
        self._thread = threading.Thread(
            target=self._run,
            name="k3-draft-rdma",
            daemon=True,
        )

    def start(self) -> None:
        # Non-blocking: the endpoint init (rank 1) blocks until the cluster's
        # rank 0 connects, and the HTTP status server (which exposes the store
        # rendezvous) must be up before that. The main thread serves HTTP.
        self._thread.start()

    def join(self, timeout: float = 5.0) -> None:
        self._thread.join(timeout=timeout)

    def _decode_request_id(self, row: list[int]) -> str:
        while row and row[-1] == 0:
            row.pop()
        return bytes(row).decode("utf-8", errors="replace")

    def _decode_request_ids(
        self, req_ids_cpu: torch.Tensor, num_requests: int
    ) -> list[str]:
        rows = req_ids_cpu.view(self.engine.allocator.max_requests, 64).tolist()
        return [self._decode_request_id(list(rows[i])) for i in range(num_requests)]

    @staticmethod
    def _tensor_from_bytes(
        payload: bytes, dtype: torch.dtype, shape: tuple[int, ...]
    ) -> tuple[torch.Tensor, bytes]:
        """Slice one tensor off the front of a raw payload buffer."""
        numel = 1
        for dim in shape:
            numel *= int(dim)
        nbytes = numel * torch.empty(0, dtype=dtype).element_size()
        if len(payload) < nbytes:
            raise ValueError(
                f"K3 draft payload has {len(payload)} bytes, need {nbytes} "
                f"for shape {shape} {dtype}"
            )
        chunk = bytearray(payload[:nbytes])
        tensor = torch.frombuffer(chunk, dtype=dtype).reshape(shape)
        return tensor, payload[nbytes:]

    def _rdma_max_msg(self) -> int:
        """Worst-case request size for the registered recv buffer.

        The client concatenates the int64[16] header, the fixed control
        tensors, the full context payload (rows * width * 2 bytes for BF16)
        and the per-token positions (int64) into one message, so the recv
        buffer must cover all of it.
        """
        max_requests = self.engine.allocator.max_requests
        control_bytes = 128 + max_requests * (8 * 8 + 4 + 64)
        context_bytes = (
            self.engine.max_context_tokens * self.engine.raw_context_width * 2
        )
        positions_bytes = self.engine.max_context_tokens * 8
        return max(
            64 << 20,
            control_bytes + context_bytes + positions_bytes + (1 << 20),
        )

    def _rdma_response_max(self) -> int:
        """Worst-case response size: draft tokens + status, both tiny."""
        max_requests = self.engine.allocator.max_requests
        return max(
            1 << 20,
            max_requests * self.engine.max_speculative_tokens * 8 + 64,
        )

    def _establish(self, rdma: K3RdmaServer) -> bool:
        """Retry ``rdma.start()`` every 10 s until it succeeds or we stop.

        Returns True once the endpoint is up, False if the server is stopping.
        """
        while not self.stop.is_set():
            try:
                rdma.start()
                return True
            except Exception:
                logger.warning(
                    "RDMA side-channel not yet established; retrying in 10 s"
                )
                if self.stop.wait(timeout=10.0):
                    return False
        return False

    def _run(self) -> None:
        try:
            # Host the TCPStore rendezvous as rank 1 (the always-on server).
            # Create the TCPStore ONCE and reuse it across retries to avoid
            # "Address already in use" from TIME_WAIT on the previous socket.
            # The cluster needs ~8-10 min to reach speculator init; one long
            # TCPStore window (1h) covers any boot order. Retry only the RDMA
            # endpoint init (which can fail due to transient HCA/peer issues).
            store = create_tcp_store(
                host="0.0.0.0",
                port=_tcpstore_port(),
                world_size=self.world_size,
                is_master=True,
                timeout=timedelta(seconds=_tcpstore_timeout()),
                use_libuv=False,
            )
            group = StatelessProcessGroup(
                rank=self.rank,
                world_size=self.world_size,
                store=store,
            )

            rdma = K3RdmaServer(
                group,
                hca=_rdma_hca(),
                gid_index=_rdma_gid_index(),
                port=_rdma_port(),
                max_msg=self._rdma_max_msg(),
                send_max=self._rdma_response_max(),
                recv_max=self._rdma_max_msg(),
            )
            if not self._establish(rdma):
                return
            self.rdma = rdma
            self.ready.set()
            logger.info(
                "K3 %s RDMA proposal server ready (rank %d, hca=%s gid=%d)",
                self.engine.method,
                self.rank,
                _rdma_hca(),
                _rdma_gid_index(),
            )
            while not self.stop.is_set():
                try:
                    self._handle_one()
                except Exception:
                    # A transport error leaves a stale WR on the QP; tear the
                    # endpoint down and re-establish rather than killing the
                    # whole server thread.
                    logger.exception(
                        "K3 DSpark RDMA transport error; re-establishing the "
                        "side-channel"
                    )
                    try:
                        rdma.close()
                    except Exception:  # noqa: BLE001 - best-effort teardown
                        pass
                    if not self._establish(rdma):
                        return
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            logger.exception("K3 DSpark RDMA proposal server failed")
            self.ready.set()
        finally:
            if self.rdma is not None:
                self.rdma.close()

    def _handle_one(self) -> None:
        assert self.rdma is not None
        request = self.rdma.recv_request(timeout_ms=_RDMA_RECV_TIMEOUT_MS)
        if len(request) < 128:
            raise ValueError(
                f"K3 draft RDMA request is {len(request)} bytes, missing the "
                "int64[16] header"
            )
        header = torch.frombuffer(
            bytearray(request[:128]), dtype=torch.int64
        ).reshape(16)
        h = header.tolist()
        # Reject a malformed header before dispatch: the raw bytes are
        # attacker-adjacent (a peer could post any int64 pattern), so require
        # real integers rather than letting a bool/float-like value through.
        protocol = _required_int(h[0], "protocol")
        op = _required_int(h[1], "op")
        if protocol != PROTOCOL_VERSION:
            raise ValueError(
                f"Unsupported protocol {protocol}; expected {PROTOCOL_VERSION}"
            )
        num_requests = int(h[2])
        total_context_rows = int(h[3])
        num_spec_tokens = int(h[4])
        projected = bool(h[5])
        payload = request[128:]
        logger.info(
            "K3 RDMA server handling op=%d payload=%d bytes", op, len(payload)
        )
        try:
            if op == OP_PING:
                response = self._handle_ping()
            elif op == OP_CLEAR:
                response = self._handle_clear()
            elif op == OP_FREE:
                response = self._handle_free(payload, num_requests)
            elif op == OP_RECONNECT:
                response = self._handle_reconnect(payload)
            elif op == OP_PROPOSE:
                response = self._handle_propose(
                    payload,
                    num_requests,
                    total_context_rows,
                    num_spec_tokens,
                    projected,
                )
            else:
                raise ValueError(f"Unknown K3 draft RDMA op {op}")
        except Exception:
            logger.exception("K3 DSpark proposal request failed")
            response = self._send_error(op, num_requests, num_spec_tokens)
        self.rdma.send_response(response)

    def _send_error(
        self, op: int, num_requests: int, num_spec_tokens: int
    ) -> bytes:
        """Build a fail-closed response after a request-handling exception.

        The verifier never knows whether a failed proposal mutated remote KV,
        so it gets ``-1`` draft tokens (rejected by the target) and a status
        with ``ok=0``.
        """
        if op == OP_PROPOSE:
            draft_tokens = torch.full(
                (num_requests, num_spec_tokens),
                -1,
                dtype=torch.int64,
            )
            status = torch.tensor(
                [0, 0, self.engine.allocator.active_requests, 0],
                dtype=torch.int64,
            )
            return draft_tokens.numpy().tobytes() + status.numpy().tobytes()
        if op == OP_PING:
            # The PING response is int64[16] (128 bytes), matching _handle_ping;
            # a differently-sized error frame here would desynchronise the
            # client's fixed-length exchange.
            ping = torch.tensor(
                [
                    0,
                    PROTOCOL_VERSION,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                dtype=torch.int64,
            )
            return ping.numpy().tobytes()
        status = torch.tensor(
            [0, 0, self.engine.allocator.active_requests, 0],
            dtype=torch.int64,
        )
        return status.numpy().tobytes()

    def _handle_ping(self) -> bytes:
        logger.info("K3 RDMA ping begin")
        response = torch.tensor(
            [
                1,
                PROTOCOL_VERSION,
                _METHOD_CODE.get(self.engine.method, 0),
                self.engine.allocator.max_requests,
                self.engine.allocator.block_size,
                self.engine.allocator.window_size,
                self.engine.prefix_cache_tokens,
                self.engine.allocator.active_requests,
                # Capacity handshake: the verifier rejects an undersized draft
                # server at bootstrap instead of failing per proposal.
                self.engine.max_batch_size,
                self.engine.max_speculative_tokens,
                self.engine.max_context_tokens,
                0,
                0,
                0,
                0,
                0,
            ],
            dtype=torch.int64,
        )
        logger.info("K3 RDMA ping done")
        return response.numpy().tobytes()

    def _handle_clear(self) -> bytes:
        logger.info("K3 RDMA clear begin")
        self.engine.clear()
        status = torch.tensor([1, 0, 0, 0], dtype=torch.int64)
        logger.info("K3 RDMA clear done")
        return status.numpy().tobytes()

    def _handle_free(self, payload: bytes, num_requests: int) -> bytes:
        logger.info("K3 RDMA free begin")
        max_requests = self.engine.allocator.max_requests
        req_ids, _ = self._tensor_from_bytes(
            payload, torch.int8, (max_requests, 64)
        )
        request_ids = self._decode_request_ids(req_ids, num_requests)
        self.engine.free(request_ids)
        status = torch.tensor(
            [1, self.engine.allocator.active_requests, 0, 0],
            dtype=torch.int64,
        )
        logger.info("K3 RDMA free done")
        return status.numpy().tobytes()

    def _handle_reconnect(self, payload: bytes) -> bytes:
        logger.info("K3 RDMA reconnect begin")
        max_requests = self.engine.allocator.max_requests
        meta, payload = self._tensor_from_bytes(
            payload, torch.int64, (max_requests, 8)
        )
        req_ids, _ = self._tensor_from_bytes(
            payload, torch.int8, (max_requests, 64)
        )
        rows = req_ids.view(max_requests, 64).tolist()
        source_request_id = self._decode_request_id(list(rows[0]))
        request_id = self._decode_request_id(list(rows[1]))
        prefix_end = _required_int(meta[0, 4].item(), "prefix_end")
        result = self.engine.reconnect(source_request_id, request_id, prefix_end)
        status = torch.tensor(
            [
                1,
                int(result.get("restored_start", 0)),
                int(result.get("active_requests", 0)),
                0,
            ],
            dtype=torch.int64,
        )
        logger.info("K3 RDMA reconnect done")
        return status.numpy().tobytes()

    def _handle_propose(
        self,
        payload: bytes,
        num_requests: int,
        total_context_rows: int,
        num_spec_tokens: int,
        projected: bool,
    ) -> bytes:
        device = self.engine.device
        max_requests = self.engine.allocator.max_requests
        meta, payload = self._tensor_from_bytes(
            payload, torch.int64, (max_requests, 8)
        )
        temps, payload = self._tensor_from_bytes(
            payload, torch.float32, (max_requests,)
        )
        req_ids, payload = self._tensor_from_bytes(
            payload, torch.int8, (max_requests, 64)
        )
        positions: torch.Tensor | None = None
        context: torch.Tensor | None = None
        if total_context_rows:
            context_width = (
                self.engine.hidden_size
                if projected
                else self.engine.raw_context_width
            )
            positions_t, payload = self._tensor_from_bytes(
                payload, torch.int64, (total_context_rows,)
            )
            context_t, payload = self._tensor_from_bytes(
                payload, torch.bfloat16, (total_context_rows, context_width)
            )
            # H2D staging: the engine consumes GPU tensors.
            positions = positions_t.to(device)
            context = context_t.to(device)

        meta_cpu = meta.view(max_requests, 8).tolist()
        temps_cpu = temps.tolist()
        req_rows = req_ids.view(max_requests, 64).tolist()
        requests: list[dict[str, Any]] = []
        for i in range(num_requests):
            # Reconstruct the int64 seed from its two 32-bit words without
            # overflowing Python->int64 (the client ships seed & 0xFFFFFFFF
            # and (seed >> 32) & 0xFFFFFFFF; a naive ``lo | (hi << 32)`` can
            # exceed 2**63-1 for large/negative seeds and torch.tensor raises
            # "Overflow when unpacking long long").
            seed_u = (int(meta_cpu[i][5]) & 0xFFFFFFFF) | (
                (int(meta_cpu[i][6]) & 0xFFFFFFFF) << 32
            )
            if seed_u >= (1 << 63):
                seed_u -= 1 << 64
            requests.append(
                {
                    "request_id": self._decode_request_id(list(req_rows[i])),
                    "reset": bool(int(meta_cpu[i][0])),
                    "reset_position": int(meta_cpu[i][1]),
                    "context_count": int(meta_cpu[i][2]),
                    "anchor_token_id": int(meta_cpu[i][3]),
                    "anchor_position": int(meta_cpu[i][4]),
                    "seed": seed_u,
                    "temperature": float(temps_cpu[i]),
                }
            )
        header = {
            "num_speculative_tokens": num_spec_tokens,
            "projected": projected,
            "requests": requests,
        }
        draft_tokens, status = self.engine.propose(header, positions, context)

        # Normalize to exactly the client's expected shape: the engine may
        # return the full configured [max_requests, max_speculative_tokens]
        # buffer, but the client posts a recv of exactly
        # [num_requests, num_spec_tokens] and rejects any length mismatch.
        draft_tokens = draft_tokens[:num_requests, :num_spec_tokens].contiguous()

        # D2H staging: the engine returns GPU tensors; the RDMA send buffer is
        # host memory (no GPUDirect on GB10).
        draft_bytes = draft_tokens.detach().cpu().contiguous().numpy().tobytes()
        status_tensor = torch.tensor(status, dtype=torch.int64)
        return draft_bytes + status_tensor.numpy().tobytes()
