# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verifier-side proxy for a dedicated RTX 3090 K3 draft process."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import zmq

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)
from vllm.v1.worker.gpu.spec_decode.speculator import (
    BaseSpeculator,
)

logger = init_logger(__name__)

PROTOCOL_VERSION = 2


@dataclass
class _RetainedRequestPrefix:
    token_ids: torch.Tensor
    committed_end: int
    context_start: int
    serial: int


def _build_valid_context_plan(
    input_batch: InputBatch,
    rejected_counts: list[int],
) -> tuple[list[int], list[int]]:
    """Return valid row indices and per-request counts."""
    if len(rejected_counts) != input_batch.num_reqs:
        raise ValueError("Rejected-token count does not match the request batch")
    gather_indices: list[int] = []
    valid_counts: list[int] = []
    offset = 0
    for request_idx, (scheduled, rejected) in enumerate(
        zip(input_batch.num_scheduled_tokens.tolist(), rejected_counts)
    ):
        valid = int(scheduled) - int(rejected)
        if not 0 <= valid <= int(scheduled):
            raise ValueError(
                f"Invalid valid-context length for request {request_idx}: "
                f"scheduled={scheduled}, rejected={rejected}"
            )
        gather_indices.extend(range(offset, offset + valid))
        valid_counts.append(valid)
        offset += int(scheduled)
    return gather_indices, valid_counts


def _anchor_positions_from_context(
    context_counts: list[int], context_positions: torch.Tensor
) -> list[int]:
    """Return the position immediately following each request's context."""
    anchors: list[int] = []
    offset = 0
    for count in context_counts:
        if count <= 0:
            raise ValueError("Every remote draft request requires context rows")
        offset += count
        anchors.append(int(context_positions[offset - 1]) + 1)
    if offset != context_positions.numel():
        raise ValueError("Remote draft context counts do not match the position tensor")
    return anchors


def _contiguous_draft_output(
    draft_tokens: torch.Tensor,
    num_reqs: int,
    num_speculative_tokens: int,
) -> torch.Tensor:
    """Return the active TP-broadcast region with a compact row stride."""
    return draft_tokens[:num_reqs, :num_speculative_tokens].contiguous()


class RemoteDSparkSpeculator(BaseSpeculator):
    """Forward target auxiliary states to a standalone draft server.

    Model-generic port of RemoteK3DSparkSpeculator (myshytf/vllm
    agent/k3-remote-dspark, ZMQ/TCP protocol v2), adapted to the vLLM 0.29
    speculator API. Deltas from the K3 original are marked [GENERIC].
    Verified with GLM-5.3-NVFP4 + GLM-5.3-DSpark (DSparkDraftModel).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        *,
        address: str,
    ) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None
        self.method = str(self.speculative_config.method)
        if self.method not in ("dspark", "dflash"):
            raise ValueError(f"Unsupported remote draft method: {self.method}")
        # [GENERIC] accept greedy + probabilistic (the PROPOSE header already
        # threads per-request temperature/seeds for server-side sampling).
        if self.speculative_config.draft_sample_method not in ("greedy", "probabilistic"):
            raise ValueError(
                "Remote DSpark draft currently supports greedy and probabilistic "
                f"drafting only, got {self.speculative_config.draft_sample_method}"
            )
        if self.speculative_config.rejection_sample_method != "block":
            raise ValueError(
                "Remote K3 draft currently requires block rejection sampling"
            )
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("Remote DSpark transport currently requires BF16")
        self.num_speculative_steps = int(self.speculative_config.num_speculative_tokens)
        self.max_num_reqs = int(vllm_config.scheduler_config.max_num_seqs)
        self.max_num_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
        draft_hf_config = self.speculative_config.draft_model_config.hf_config
        aux_layers = get_eagle3_aux_layers_from_config(self.speculative_config)
        if not aux_layers:
            # [GENERIC] RedHatAI-style DSpark drafts declare
            # aux_hidden_state_layer_ids instead of the eagle3/dflash keys.
            aux_layers = tuple(
                getattr(draft_hf_config, "aux_hidden_state_layer_ids", None) or ()
            )
        if not aux_layers:
            raise ValueError(
                f"Remote {self.method} config does not declare auxiliary layers"
            )
        self.num_aux_layers = len(aux_layers)
        target_hidden_size = int(
            getattr(draft_hf_config, "target_hidden_size", None)
            or getattr(draft_hf_config, "hidden_size", None)
            # [GENERIC] some drafts (e.g. GLM-5.3-DSpark) declare neither width;
            # the aux tensors come from the TARGET model, so use its width.
            or vllm_config.model_config.hf_config.hidden_size
        )
        self.raw_context_width = int(target_hidden_size * self.num_aux_layers)
        self.address = address
        self.timeout_ms = int(
            os.environ.get(
                "VLLM_K3_DRAFT_REMOTE_TIMEOUT_MS",
                os.environ.get("VLLM_K3_DSPARK_REMOTE_TIMEOUT_MS", "30000"),
            )
        )
        self.supports_mm_inputs = False
        self.draft_logits: torch.Tensor | None = None
        self.draft_tokens = torch.full(
            (self.max_num_reqs, self.num_speculative_steps),
            -1,
            dtype=torch.int64,
            device=device,
        )
        self._known_requests: set[str] = set()
        self._disabled_requests: set[str] = set()
        self._active_requests: set[str] = set()
        self._retained_prefixes: dict[str, _RetainedRequestPrefix] = {}
        self._retained_serial = 0
        self._remote_max_requests = self.max_num_reqs
        self._remote_block_size = 1
        self._remote_window_size = 0
        self._remote_prefix_cache_tokens = 0
        self._timing_log_interval = int(
            os.environ.get("VLLM_K3_DRAFT_TIMING_LOG_INTERVAL", "0")
        )
        if self._timing_log_interval < 0:
            raise ValueError("VLLM_K3_DRAFT_TIMING_LOG_INTERVAL must be >= 0")
        self._timing_count = 0
        self._timing_totals_ms: dict[str, float] = {}

        tp_group = get_tp_group()
        self._tp_group = tp_group
        self._tp_rank = int(tp_group.rank_in_group)
        self._zmq_context: zmq.Context | None = None
        self._socket: zmq.Socket | None = None
        if self._tp_rank == 0:
            # P2P is unavailable on the target host. Keep the mandatory D2H
            # hop off pageable memory so it can be queued directly after the
            # target forward on the current stream.
            self._positions_staging = torch.empty(
                self.max_num_tokens,
                dtype=torch.int64,
                pin_memory=True,
            )
            self._context_staging = torch.empty(
                (self.max_num_tokens, self.raw_context_width),
                dtype=vllm_config.model_config.dtype,
                pin_memory=True,
            )
            self._rejected_staging = torch.empty(
                self.max_num_reqs,
                dtype=torch.int32,
                pin_memory=True,
            )
            self._anchor_staging = torch.empty(
                self.max_num_reqs,
                dtype=torch.int64,
                pin_memory=True,
            )
            self._zmq_context = zmq.Context()
            self._connect()
            response = self._rpc(
                [json.dumps({"protocol": PROTOCOL_VERSION, "op": "PING"}).encode()]
            )
            if response.get("op") != "PONG":
                raise RuntimeError(f"Unexpected K3 draft health response: {response}")
            if response.get("method") != self.method:
                raise RuntimeError(
                    "Remote K3 draft method mismatch: "
                    f"target={self.method}, server={response.get('method')}"
                )
            self._remote_max_requests = int(
                response.get("max_requests", self.max_num_reqs)
            )
            self._remote_block_size = int(response.get("block_size", 1))
            self._remote_window_size = int(response.get("window_size", 0))
            self._remote_prefix_cache_tokens = int(
                response.get("prefix_cache_tokens", 0)
            )
            if int(response.get("active_requests", 0)):
                # A restarted verifier cannot safely identify state left by an
                # older process, so establish a clean protocol epoch.
                self._rpc(
                    [json.dumps({"protocol": PROTOCOL_VERSION, "op": "CLEAR"}).encode()]
                )

        logger.info(
            "Remote K3 %s proxy initialized: address=%s, TP rank=%d, K=%d",
            self.method,
            address,
            self._tp_rank,
            self.num_speculative_steps,
        )

    def _connect(self) -> None:
        assert self._zmq_context is not None
        if self._socket is not None:
            self._socket.close()
        socket = self._zmq_context.socket(zmq.REQ)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        socket.connect(self.address)
        self._socket = socket

    def _rpc(self, frames: list[bytes]) -> dict[str, Any]:
        assert self._socket is not None
        try:
            self._socket.send_multipart(frames)
            response = self._socket.recv_json()
        except Exception:
            self._connect()
            raise
        if not isinstance(response, dict) or not response.get("ok", False):
            raise RuntimeError(f"Remote DSpark RPC failed: {response}")
        if int(response.get("protocol", -1)) != PROTOCOL_VERSION:
            raise RuntimeError(f"Remote DSpark protocol mismatch: {response}")
        return response

    def init_cudagraph_manager(self, cudagraph_mode=None) -> None:
        """The standalone drafter owns its CUDA graph lifecycle."""

    def capture(self) -> None:
        """The verifier has no local draft graph to capture."""

    def _free_remote_requests(self, request_ids: set[str] | list[str]) -> None:
        remote_request_ids = sorted(set(request_ids) & self._known_requests)
        if not remote_request_ids:
            return
        self._rpc(
            [
                json.dumps(
                    {
                        "protocol": PROTOCOL_VERSION,
                        "op": "FREE",
                        "request_ids": remote_request_ids,
                    }
                ).encode()
            ]
        )
        self._known_requests.difference_update(remote_request_ids)
        for request_id in remote_request_ids:
            self._retained_prefixes.pop(request_id, None)

    def _ensure_remote_capacity(self, current_request_ids: set[str]) -> None:
        while len(self._known_requests) >= self._remote_max_requests:
            candidates = self._known_requests - current_request_ids
            if not candidates:
                raise RuntimeError(
                    "Remote DSpark request capacity is exhausted by active requests"
                )
            request_id = min(
                candidates,
                key=lambda req_id: (
                    self._retained_prefixes[req_id].serial
                    if req_id in self._retained_prefixes
                    else -1
                ),
            )
            self._free_remote_requests({request_id})

    @staticmethod
    def _token_prefix(
        input_batch: InputBatch,
        request_idx: int,
        prefix_end: int,
    ) -> torch.Tensor | None:
        # [GENERIC] Full per-request token history: Kimi-fork InputBatch
        # exposes all_token_ids_cpu; the 0.29 V1 runner names it token_ids_cpu;
        # the V2 runner's InputBatch has no such table at all (token ids live
        # in request state). Returning None is safe: _remember_prefix skips
        # (no server prefix-cache entry) and non-continuing requests fall
        # back to sending the full context — steady-state decode is
        # unaffected.
        token_table = getattr(input_batch, "all_token_ids_cpu", None)
        if token_table is None:
            token_table = getattr(input_batch, "token_ids_cpu", None)
        if token_table is None or prefix_end < 0:
            return None
        state_idx = int(input_batch.idx_mapping_np[request_idx])
        if state_idx < 0 or prefix_end > token_table.shape[1]:
            return None
        return token_table[state_idx, :prefix_end]

    def _can_restore_prefix(
        self,
        retained: _RetainedRequestPrefix,
        prefix_end: int,
    ) -> bool:
        if (
            prefix_end <= 0
            or retained.committed_end < prefix_end
            or self._remote_window_size <= 0
            or self._remote_prefix_cache_tokens < self._remote_window_size
        ):
            return False
        restore_start = max(0, prefix_end - self._remote_window_size)
        restore_start = (
            restore_start // self._remote_block_size * self._remote_block_size
        )
        retained_start = max(
            retained.context_start,
            retained.committed_end - self._remote_prefix_cache_tokens,
        )
        return restore_start >= retained_start

    def _find_reconnect_source(
        self,
        token_prefix: torch.Tensor,
        prefix_end: int,
        current_request_ids: set[str],
    ) -> str | None:
        candidates: list[tuple[int, int, str]] = []
        for request_id in self._known_requests - current_request_ids:
            retained = self._retained_prefixes.get(request_id)
            if retained is None or not self._can_restore_prefix(retained, prefix_end):
                continue
            if torch.equal(retained.token_ids[:prefix_end], token_prefix):
                candidates.append(
                    (
                        retained.committed_end - prefix_end,
                        -retained.serial,
                        request_id,
                    )
                )
        return min(candidates)[2] if candidates else None

    def _reconnect_request(
        self,
        source_request_id: str,
        request_id: str,
        prefix_end: int,
        token_prefix: torch.Tensor,
    ) -> bool:
        try:
            response = self._rpc(
                [
                    json.dumps(
                        {
                            "protocol": PROTOCOL_VERSION,
                            "op": "RECONNECT",
                            "source_request_id": source_request_id,
                            "request_id": request_id,
                            "prefix_end": prefix_end,
                        }
                    ).encode()
                ]
            )
        except Exception:
            logger.exception(
                "Remote K3 DSpark prefix reconnect failed: source=%s, "
                "request=%s, prefix_end=%d",
                source_request_id,
                request_id,
                prefix_end,
            )
            return False

        self._retained_prefixes.pop(source_request_id)
        self._known_requests.discard(source_request_id)
        self._known_requests.add(request_id)
        self._retained_serial += 1
        self._retained_prefixes[request_id] = _RetainedRequestPrefix(
            token_ids=token_prefix.clone(),
            committed_end=prefix_end,
            context_start=int(response.get("restored_start", 0)),
            serial=self._retained_serial,
        )
        logger.info(
            "Remote K3 DSpark prefix reconnected: source=%s, request=%s, "
            "prefix_end=%d, restored_start=%s, latency_ms=%.1f",
            source_request_id,
            request_id,
            prefix_end,
            response.get("restored_start"),
            float(response.get("latency_ms", 0.0)),
        )
        return True

    def _remember_prefix(
        self,
        input_batch: InputBatch,
        request_idx: int,
        request_id: str,
        committed_end: int,
        context_start: int | None = None,
    ) -> None:
        token_prefix = self._token_prefix(input_batch, request_idx, committed_end)
        if token_prefix is None:
            return
        previous = self._retained_prefixes.get(request_id)
        if context_start is None:
            context_start = previous.context_start if previous is not None else 0
        self._retained_serial += 1
        self._retained_prefixes[request_id] = _RetainedRequestPrefix(
            token_ids=token_prefix.clone(),
            committed_end=committed_end,
            context_start=context_start,
            serial=self._retained_serial,
        )

    def _copy_tokens_from_response(
        self,
        response: dict[str, Any],
        active_indices: list[int],
        num_speculative_tokens: int,
    ) -> None:
        tokens = response.get("tokens")
        expected_shape = (len(active_indices), num_speculative_tokens)
        if (
            not isinstance(tokens, list)
            or len(tokens) != expected_shape[0]
            or any(
                not isinstance(row, list) or len(row) != expected_shape[1]
                for row in tokens
            )
        ):
            raise ValueError(
                f"Remote DSpark token response has the wrong shape; "
                f"expected={expected_shape}, got={tokens!r}"
            )
        remote_tokens = torch.tensor(tokens, dtype=torch.int64, device=self.device)
        active_gpu = torch.tensor(active_indices, dtype=torch.int64, device=self.device)
        # ``draft_tokens`` is allocated at the configured maximum depth, while
        # adaptive speculation and the per-batch schedule can request a
        # smaller depth for an individual step.  Copy into the matching width
        # instead of requiring every response to have the maximum width.
        self.draft_tokens[:, :num_speculative_tokens].index_copy_(
            0, active_gpu, remote_tokens
        )

    def _record_timing(self, timing_ms: dict[str, float]) -> None:
        if self._timing_log_interval <= 0:
            return
        self._timing_count += 1
        for key, value in timing_ms.items():
            self._timing_totals_ms[key] = self._timing_totals_ms.get(key, 0.0) + value
        if self._timing_count < self._timing_log_interval:
            return
        means = {
            key: value / self._timing_count
            for key, value in self._timing_totals_ms.items()
        }
        logger.info(
            "Remote K3 %s timing over %d proposals (ms): %s",
            self.method,
            self._timing_count,
            ", ".join(f"{key}={value:.3f}" for key, value in means.items()),
        )
        self._timing_count = 0
        self._timing_totals_ms.clear()

    def _rank0_propose(
        self,
        input_batch: InputBatch,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        num_speculative_tokens: int,
    ) -> None:
        started = time.perf_counter()
        if aux_hidden_states is None or len(aux_hidden_states) != self.num_aux_layers:
            raise ValueError(
                f"Remote K3 {self.method} requires {self.num_aux_layers} configured "
                "target auxiliary hidden states"
            )

        num_reqs = input_batch.num_reqs
        current_request_ids = set(input_batch.req_ids)
        previous_active_requests = self._active_requests
        self._disabled_requests.intersection_update(current_request_ids)
        idx_mapping = input_batch.idx_mapping[:num_reqs].long()
        sampled_counts = num_sampled[:num_reqs]
        sampled_anchors = last_sampled[idx_mapping, 0]
        prefill_anchors = next_prefill_tokens[0, idx_mapping]
        anchor_tokens = torch.where(
            sampled_counts > 0,
            sampled_anchors,
            prefill_anchors,
        ).to(torch.int64)
        # Queue both small D2H copies and wait once.  The same stream owns the
        # preceding token-table update, so this synchronization also makes the
        # UVA-backed table safe for prefix matching below.
        rejected_staging = self._rejected_staging[:num_reqs]
        anchor_staging = self._anchor_staging[:num_reqs]
        rejected_staging.copy_(num_rejected[:num_reqs], non_blocking=True)
        anchor_staging.copy_(anchor_tokens, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        rejected_counts = rejected_staging.tolist()
        anchor_tokens_cpu = anchor_staging.tolist()
        gather_indices, valid_counts = _build_valid_context_plan(
            input_batch, rejected_counts
        )
        metadata_ready = time.perf_counter()

        active_indices: list[int] = []
        requests: list[dict[str, Any]] = []
        request_context_starts: list[int | None] = []
        selected_gather_indices: list[int] = []
        gather_offset = 0
        for request_idx, request_id in enumerate(input_batch.req_ids):
            valid_count = valid_counts[request_idx]
            request_gather = gather_indices[gather_offset : gather_offset + valid_count]
            gather_offset += valid_count
            if request_id in self._disabled_requests or valid_count <= 0:
                continue
            first_position = int(input_batch.num_computed_tokens_np[request_idx])
            is_continuing = (
                request_id in previous_active_requests
                and request_id in self._known_requests
            )
            reset = False
            context_start: int | None = None
            if not is_continuing:
                if first_position == 0:
                    if request_id in self._known_requests:
                        self._free_remote_requests({request_id})
                    self._ensure_remote_capacity(current_request_ids)
                    reset = True
                    context_start = 0
                else:
                    token_prefix = self._token_prefix(
                        input_batch,
                        request_idx,
                        first_position,
                    )
                    source_request_id: str | None = None
                    if token_prefix is not None:
                        retained = self._retained_prefixes.get(request_id)
                        if (
                            request_id in self._known_requests
                            and retained is not None
                            and self._can_restore_prefix(retained, first_position)
                            and torch.equal(
                                retained.token_ids[:first_position], token_prefix
                            )
                        ):
                            source_request_id = request_id
                        else:
                            source_request_id = self._find_reconnect_source(
                                token_prefix,
                                first_position,
                                current_request_ids,
                            )
                    if source_request_id is None or token_prefix is None:
                        if request_id in self._known_requests:
                            self._free_remote_requests({request_id})
                        self._ensure_remote_capacity(current_request_ids)
                        reset = True
                        context_start = first_position
                        logger.warning(
                            "Remote K3 %s cold-bootstrapping cache-restored "
                            "request %s at position %d from %d fresh context "
                            "rows; target verification preserves correctness.",
                            self.method,
                            request_id,
                            first_position,
                            valid_count,
                        )
                    elif not self._reconnect_request(
                        source_request_id,
                        request_id,
                        first_position,
                        token_prefix,
                    ):
                        self._disabled_requests.add(request_id)
                        continue
            requests.append(
                {
                    "request_id": request_id,
                    "reset": reset,
                    "reset_position": first_position if reset else 0,
                    "context_count": valid_count,
                    "anchor_token_id": int(anchor_tokens_cpu[request_idx]),
                }
            )
            self._known_requests.add(request_id)
            active_indices.append(request_idx)
            request_context_starts.append(context_start)
            selected_gather_indices.extend(request_gather)

        self._active_requests = self._known_requests & current_request_ids
        if not active_indices:
            return
        requests_ready = time.perf_counter()

        indices_gpu = torch.tensor(
            selected_gather_indices, dtype=torch.int64, device=self.device
        )
        positions = input_batch.positions.index_select(0, indices_gpu)
        context = torch.cat(
            [hidden.index_select(0, indices_gpu) for hidden in aux_hidden_states],
            dim=-1,
        )
        num_context_rows = int(context.shape[0])
        if num_context_rows > self.max_num_tokens:
            raise ValueError(
                f"Remote DSpark context has {num_context_rows} rows, max is "
                f"{self.max_num_tokens}"
            )
        if context.shape[1] != self.raw_context_width:
            raise ValueError(
                f"Remote DSpark context width is {context.shape[1]}, expected "
                f"{self.raw_context_width}"
            )
        context_ready = time.perf_counter()
        positions_staging = self._positions_staging[:num_context_rows]
        context_staging = self._context_staging[:num_context_rows]
        positions_staging.copy_(positions, non_blocking=True)
        context_staging.copy_(context, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()
        context_copied = time.perf_counter()
        anchor_positions = _anchor_positions_from_context(
            [int(request["context_count"]) for request in requests],
            positions_staging,
        )
        for request, anchor_position in zip(requests, anchor_positions):
            request["anchor_position"] = anchor_position
        positions_frame = positions_staging.numpy().tobytes()
        # NumPy has inconsistent bfloat16 support; preserve its exact bits as u16.
        context_frame = context_staging.view(torch.uint16).numpy().tobytes()
        serialized = time.perf_counter()
        header = {
            "protocol": PROTOCOL_VERSION,
            "op": "PROPOSE",
            "projected": False,
            "num_speculative_tokens": num_speculative_tokens,
            "requests": requests,
        }
        response = self._rpc(
            [json.dumps(header).encode(), positions_frame, context_frame]
        )
        rpc_done = time.perf_counter()
        self._copy_tokens_from_response(
            response,
            active_indices,
            num_speculative_tokens,
        )
        output_copied = time.perf_counter()
        timing_ms = {
            "metadata_d2h": (metadata_ready - started) * 1000,
            "request_plan": (requests_ready - metadata_ready) * 1000,
            "context_gather": (context_ready - requests_ready) * 1000,
            "context_d2h": (context_copied - context_ready) * 1000,
            "serialize": (serialized - context_copied) * 1000,
            "rpc_roundtrip": (rpc_done - serialized) * 1000,
            "tokens_h2d": (output_copied - rpc_done) * 1000,
            "client_total": (output_copied - started) * 1000,
        }
        server_timing = response.get("timing_ms")
        if isinstance(server_timing, dict):
            for key, value in server_timing.items():
                if isinstance(value, int | float):
                    timing_ms[f"server_{key}"] = float(value)
        self._record_timing(timing_ms)
        for request_idx, request, anchor_position, context_start in zip(
            active_indices,
            requests,
            anchor_positions,
            request_context_starts,
        ):
            self._remember_prefix(
                input_batch,
                request_idx,
                str(request["request_id"]),
                anchor_position,
                context_start,
            )

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # [GENERIC] The optional tail of propose() shifted across vLLM
        # versions: 0.29/V2 inserts dp_sync right after seeds; the perf/V1
        # tree does not. Positional callers would misalign every argument
        # after seeds, so accept the tail loosely and resolve it by type
        # (the relative order of the trailing bools is stable across trees).
        dp_sync = kwargs.get("dp_sync")
        num_speculative_tokens = kwargs.get("num_speculative_tokens")
        num_tokens_across_dp = kwargs.get("num_tokens_across_dp")
        dummy_run = kwargs.get("dummy_run", False)
        skip_attn_for_dummy_run = kwargs.get("skip_attn_for_dummy_run", False)
        mm_inputs = kwargs.get("mm_inputs")
        is_profile = kwargs.get("is_profile", False)
        tail_bools: list[bool] = []
        for value in args:
            if isinstance(value, bool):
                tail_bools.append(value)
            elif isinstance(value, int):
                num_speculative_tokens = value
            elif isinstance(value, torch.Tensor):
                num_tokens_across_dp = value
            elif isinstance(value, tuple):
                mm_inputs = value
            elif value is not None:
                # DPSyncState (or a future non-scalar) in the dp_sync slot.
                dp_sync = value
        if tail_bools:
            dummy_run = tail_bools[0]
        if len(tail_bools) > 1:
            skip_attn_for_dummy_run = tail_bools[1]
        if len(tail_bools) > 2:
            is_profile = tail_bools[2]
        del (
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            temperature,
            seeds,
            num_tokens_across_dp,
            skip_attn_for_dummy_run,
            mm_inputs,
        )
        active_k = (
            int(num_speculative_tokens)
            if num_speculative_tokens is not None
            else self.num_speculative_steps
        )
        if not 1 <= active_k <= self.num_speculative_steps:
            raise ValueError(
                f"Remote DSpark depth must be in [1, "
                f"{self.num_speculative_steps}], got {active_k}"
            )
        output = self.draft_tokens[: input_batch.num_reqs, :active_k]
        output.fill_(-1)
        if self._tp_rank == 0 and not (dummy_run or is_profile):
            try:
                self._rank0_propose(
                    input_batch,
                    aux_hidden_states,
                    num_sampled,
                    num_rejected,
                    last_sampled,
                    next_prefill_tokens,
                    active_k,
                )
            except Exception:
                output.fill_(-1)
                # The verifier cannot know whether a timed-out request mutated
                # remote KV. Fail closed for those requests until they leave
                # the active batch; FREE remains safe even if the server never
                # created the state.
                self._disabled_requests.update(input_batch.req_ids)
                logger.exception(
                    "Remote K3 DSpark proposal failed; drafting is disabled for "
                    "this step"
                )
        # Slicing the active depth from the max-width persistent buffer leaves
        # a larger row stride whenever adaptive K is below the configured
        # maximum. NCCL broadcast requires a contiguous tensor. Materialize
        # only the tiny [batch, K] result after rank 0 has populated it.
        output = _contiguous_draft_output(
            self.draft_tokens,
            input_batch.num_reqs,
            active_k,
        )
        self._tp_group.broadcast(output, src=0)
        return output
# V4PLUS-V6-REMOTE-DSPARK


RemoteK3DSparkSpeculator = RemoteDSparkSpeculator  # backwards-compat alias.

# remote-dspark-tcp (marker — do not remove)
