# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verifier-side proxy for a dedicated RTX 3090 K3 draft process."""

from __future__ import annotations

import json
import os
import socket
import time
import urllib.request
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_tp_group
from vllm.distributed.utils import StatelessProcessGroup, create_tcp_store
from vllm.k3_rdma_transport import K3RdmaClient
from vllm.logger import init_logger
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
    get_eagle3_aux_layers_from_config,
)
from vllm.v1.worker.gpu.spec_decode.speculator import (
    BaseSpeculator,
    CUDAGraphCapturePhase,
)

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

# ---------------------------------------------------------------------------
# Raw-verbs RDMA point-to-point transport (kernel-bypass RoCEv2).
#
# The verifier (cluster TP-rank-0 worker) and the T1 draft server form a
# dedicated 2-host channel over libk3rdma (raw ibverbs RC). The TCPStore from
# the StatelessProcessGroup rendezvous is reused ONLY to exchange the small
# ibverbs peer-info strings; all tensor payloads ride the RDMA queue pair.
# This replaced the NCCL P2P side-channel, whose comm-init was unreliable
# across the custom-fork (cluster) / stock (T1) NCCL libraries.
# ---------------------------------------------------------------------------

_TCPSTORE_DEFAULT_PORT = 51230
# Short rendezvous window: a long (300s) TCPStore block stalled warmup when
# the peer was not yet up. Fail fast and retry on the cooldown instead.
_TCPSTORE_TIMEOUT_SECONDS = 15
# Health-check and handshake budgets (kept well under the engine's 60s
# shm_broadcast watchdog so a stuck bootstrap cannot cancel the EngineCore).
_HEALTH_TIMEOUT_MS = 5000
_HANDSHAKE_PING_TIMEOUT_MS = 10000


def _tcpstore_port() -> int:
    return int(
        os.environ.get("VLLM_K3_DRAFT_TCPSTORE_PORT", str(_TCPSTORE_DEFAULT_PORT))
    )


def _rdma_hca() -> str:
    return os.environ.get("VLLM_K3_DRAFT_RDMA_HCA", "rocep1s0f1")


def _rdma_gid_index() -> int:
    return int(os.environ.get("VLLM_K3_DRAFT_RDMA_GID_INDEX", "3"))


def _rdma_port() -> int:
    return int(os.environ.get("VLLM_K3_DRAFT_RDMA_PORT", "1"))


def _tensor_to_raw_bytes(tensor: torch.Tensor) -> bytes:
    """Host-stage a tensor to its raw bytes.

    ``Tensor.numpy()`` has no bfloat16 dtype, so reinterpret the contiguous
    host tensor as uint8 before extracting the bytes. This is the required
    D2H host-staging copy (no GPUDirect on GB10).
    """
    staged = tensor.detach().cpu().contiguous()
    return staged.view(torch.uint8).numpy().tobytes()


def _remote_host(address: str) -> str:
    """Extract the host from the configured draft-server address."""
    return address.split("://", 1)[-1].split(":")[0]


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
            timeout=timedelta(seconds=_TCPSTORE_TIMEOUT_SECONDS),
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


def _remote_status_url(address: str) -> str:
    """Derive the draft server's HTTP /v1/status URL from the configured address.

    The RDMA side-channel carries all tensor data; HTTP is used ONLY for the
    bootstrap/health check (the ibverbs peer-info strings are exchanged over
    the StatelessProcessGroup TCPStore). ``address`` may be the legacy ZMQ
    form (``tcp://host:port``) or an explicit ``http(s)://host:port`` URL.
    """
    override = os.environ.get("VLLM_K3_DRAFT_REMOTE_STATUS_URL")
    if override:
        return override.rstrip("/") + "/v1/status"
    if address.startswith("http://") or address.startswith("https://"):
        return address.rstrip("/") + "/v1/status"
    host = address.split("://", 1)[-1].split(":")[0]
    port = os.environ.get("VLLM_K3_DRAFT_REMOTE_STATUS_PORT", "8091")
    return f"http://{host}:{port}/v1/status"


def _fetch_remote_status(address: str, timeout_ms: int) -> dict[str, Any]:
    """Fetch the draft server's /v1/status JSON with timeout + retries."""
    url = _remote_status_url(address)
    deadline = time.time() + timeout_ms / 1000.0
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - retry any transient failure
            last_exc = exc
            time.sleep(0.5)
    raise RuntimeError(
        f"Could not fetch K3 draft status from {url}: {last_exc}"
    )


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
        zip(
            input_batch.num_scheduled_tokens.tolist(),
            rejected_counts,
            strict=True,
        )
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


class _DraftContentError(RuntimeError):
    """Received draft payload is degenerate (e.g. all zeros).

    The transport itself worked, so the RDMA channel stays up: only the
    request state is discarded, and the stale-release path resyncs it on
    the next step (FREE + reset from fresh context).
    """


class RemoteK3DSparkSpeculator(BaseSpeculator):
    """Forward target auxiliary states to a standalone greedy draft server."""

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
            raise ValueError(f"Unsupported remote K3 draft method: {self.method}")
        if self.speculative_config.draft_sample_method not in ("greedy", "probabilistic"):
            raise ValueError(
                "Remote K3 draft currently supports greedy and probabilistic "
                f"drafting only, got {self.speculative_config.draft_sample_method}"
            )
        if self.speculative_config.rejection_sample_method != "block":
            raise ValueError(
                "Remote K3 draft currently requires block rejection sampling"
            )
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("Remote K3 DSpark transport currently requires BF16")
        self.num_speculative_steps = int(self.speculative_config.num_speculative_tokens)
        self.max_num_reqs = int(vllm_config.scheduler_config.max_num_seqs)
        self.max_num_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
        draft_hf_config = self.speculative_config.draft_model_config.hf_config
        aux_layers = get_eagle3_aux_layers_from_config(self.speculative_config)
        if not aux_layers:
            raise ValueError(
                f"Remote K3 {self.method} config does not declare auxiliary layers"
            )
        self.num_aux_layers = len(aux_layers)
        target_hidden_size = int(
            getattr(draft_hf_config, "target_hidden_size", None)
            or draft_hf_config.hidden_size
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
        # Ids whose remote slot may still exist after a failed exchange. The
        # next proposal frees them first (FREE is a no-op for ids the server
        # does not hold), which reclaims the slot and re-enables the
        # requests: their next proposal then resets the remote state from
        # fresh context rows. Without this, one transient transport failure
        # disables drafting for the affected requests for the rest of their
        # lifetime. Ported from upstream myshytf/vllm 7f37e34ca.
        self._stale_remote_requests: set[str] = set()
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
        self._rdma: K3RdmaClient | None = None
        if self._tp_rank == 0:
            # Pinned staging buffers are retained as fallback scratch; the
            # RDMA path stages positions/context through host memory (no
            # GPUDirect on GB10).
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
            # LAZY channel bootstrap (see _ensure_rdma_channel): the RDMA
            # side-channel is established on the first propose(), NOT in
            # __init__. Worker init runs 16 TP/DCP ranks bootstrapping NCCL
            # simultaneously under model-load memory pressure; doing our
            # extra endpoint init there multiplies the failure modes and a
            # failure kills the whole boot. Post-ready init happens on a
            # settled GPU with a clear error budget, and a channel failure
            # only disables drafting for that step (fail-closed), never the
            # engine. Cooldown between attempts avoids log storms.
            self._rdma_ready = False
            self._rdma_retry_at = 0.0
            self._rdma_retry_interval_s = 30.0

        logger.info(
            "Remote K3 %s proxy initialized: address=%s, TP rank=%d, K=%d",
            self.method,
            address,
            self._tp_rank,
            self.num_speculative_steps,
        )

    def _rdma_max_msg(self) -> int:
        """Worst-case request size for the registered RDMA send/recv buffers.

        The client copies the int64[16] header plus every outbound tensor
        into one contiguous host buffer, so the buffer must cover the full
        context payload (rows * width * 2 bytes for BF16), the per-token
        positions (int64) and the fixed control tensors.
        """
        control_bytes = 128 + self.max_num_reqs * (8 * 8 + 4 + 64)
        context_bytes = self.max_num_tokens * self.raw_context_width * 2
        positions_bytes = self.max_num_tokens * 8
        return max(
            64 << 20,
            control_bytes + context_bytes + positions_bytes + (1 << 20),
        )

    def _rdma_response_max(self) -> int:
        """Worst-case response size: draft tokens + status, both tiny."""
        return max(
            1 << 20,
            self.max_num_reqs * self.num_speculative_steps * 8 + 64,
        )

    def _invalidate_rdma(self) -> None:
        """Drop a broken channel and arm the retry cooldown.

        A timed-out exchange leaves a stale WR on the QP, so the channel must
        not be reused; close it and force a fresh establishment after the
        cooldown.
        """
        rdma = self._rdma
        self._rdma = None
        self._rdma_ready = False
        self._rdma_retry_at = time.monotonic() + self._rdma_retry_interval_s
        if rdma is not None:
            try:
                rdma.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                logger.warning(
                    "Remote K3 %s RDMA channel invalidation failed",
                    self.method,
                    exc_info=True,
                )

    def _ensure_rdma_channel(self) -> bool:
        """Establish the RDMA side-channel on first use (lazy init).

        Worker init runs all TP/DCP ranks bootstrapping NCCL simultaneously
        under model-load memory pressure; doing our extra endpoint init there
        triples the failure modes and any failure kills the whole boot.
        Post-ready init happens on a settled GPU, and a channel failure only
        disables drafting for that step (fail-closed) — never the engine.
        Attempts are cooldown-throttled to avoid log storms.
        """
        if self._rdma_ready:
            return True
        now = time.monotonic()
        if now < self._rdma_retry_at:
            return False
        # Defensively drop any half-open endpoint left by a previous failed
        # attempt before opening a new one (the C lib is a single global
        # endpoint and cannot host two).
        if self._rdma is not None:
            try:
                self._rdma.close()
            except Exception:  # noqa: BLE001 - best-effort teardown
                pass
            self._rdma = None
        try:
            # Health check: confirm the T1 draft server is up. HTTP is used
            # for bootstrap/health ONLY; tensor data rides the RDMA channel.
            _fetch_remote_status(self.address, _HEALTH_TIMEOUT_MS)
            # Bootstrap: join the dedicated 2-host channel. The T1 server
            # hosts the TCPStore rendezvous (rank 1); the verifier connects
            # to it (rank 0) and the two sides exchange their ibverbs
            # peer-info strings over that store before connecting the QP.
            group = _create_stateless_group(
                host=_remote_host(self.address),
                port=_tcpstore_port(),
                rank=0,
                world_size=2,
                is_master=False,
            )
            self._rdma = K3RdmaClient(
                group,
                hca=_rdma_hca(),
                gid_index=_rdma_gid_index(),
                port=_rdma_port(),
                max_msg=self._rdma_max_msg(),
                recv_max=self._rdma_response_max(),
            )
            self._rdma.start()
            logger.info(
                "Remote K3 %s RDMA side-channel ready (rank 0, hca=%s gid=%d)",
                self.method,
                _rdma_hca(),
                _rdma_gid_index(),
            )
            response = self._rdma_ping(timeout_ms=_HANDSHAKE_PING_TIMEOUT_MS)
            if not response.get("ok"):
                raise RuntimeError(
                    f"Unexpected K3 draft health response: {response}"
                )
            if response.get("method") != self.method:
                raise RuntimeError(
                    "Remote K3 draft method mismatch: "
                    f"target={self.method}, server={response.get('method')}"
                )
            # Capacity handshake (fail-closed): reject an undersized draft
            # server up front instead of failing per proposal at runtime. The
            # PONG always publishes these fields; a missing field raises
            # KeyError here and disables drafting rather than silently
            # proceeding with an assumed capacity.
            remote_max_batch_size = int(response["max_batch_size"])
            if self.max_num_reqs > remote_max_batch_size:
                raise RuntimeError(
                    "Remote K3 draft batch capacity is smaller than the target "
                    f"scheduler: remote={remote_max_batch_size}, "
                    f"target={self.max_num_reqs}"
                )
            remote_max_depth = int(response["max_speculative_tokens"])
            if self.num_speculative_steps > remote_max_depth:
                raise RuntimeError(
                    "Remote K3 draft depth is smaller than the target scheduler: "
                    f"remote={remote_max_depth}, "
                    f"target={self.num_speculative_steps}"
                )
            remote_max_context_tokens = int(response["max_context_tokens"])
            if self.max_num_tokens > remote_max_context_tokens:
                raise RuntimeError(
                    "Remote K3 draft context-row capacity is smaller than the "
                    "target scheduler's batch: "
                    f"remote={remote_max_context_tokens}, "
                    f"target={self.max_num_tokens}"
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
                self._rdma_clear()
        except Exception as exc:
            if self._rdma is not None:
                try:
                    self._rdma.close()
                except Exception:  # noqa: BLE001 - best-effort teardown
                    pass
            self._rdma = None
            self._rdma_retry_at = now + self._rdma_retry_interval_s
            logger.warning(
                "Remote K3 %s RDMA side-channel unavailable (%s); drafting "
                "disabled until retry in %.0fs",
                self.method,
                exc,
                self._rdma_retry_interval_s,
            )
            return False
        self._rdma_ready = True
        return True

    def _rdma_exchange(
        self,
        header: torch.Tensor,
        extra_sends: list[torch.Tensor],
        recv_tensors: list[torch.Tensor],
        timeout_ms: int | None = None,
    ) -> None:
        """Run one request/response exchange over the RDMA side-channel.

        No GPUDirect on GB10, so this is host-staged: the header and every
        outbound tensor are copied D2H into one contiguous request buffer,
        and every inbound tensor is copied H2D out of the response buffer.
        """
        assert self._rdma is not None
        parts = [_tensor_to_raw_bytes(header)]
        for tensor in extra_sends:
            parts.append(_tensor_to_raw_bytes(tensor))
        request = b"".join(parts)
        response_len = sum(t.numel() * t.element_size() for t in recv_tensors)
        response = self._rdma.exchange(
            request,
            response_len,
            self.timeout_ms if timeout_ms is None else timeout_ms,
        )
        offset = 0
        for tensor in recv_tensors:
            nbytes = tensor.numel() * tensor.element_size()
            chunk = response[offset : offset + nbytes]
            offset += nbytes
            staged = torch.frombuffer(bytearray(chunk), dtype=tensor.dtype).reshape(
                tensor.shape
            )
            tensor.copy_(staged.to(tensor.device))
        if offset != len(response):
            raise RuntimeError(
                f"K3 RDMA response has {len(response)} bytes, consumed {offset}"
            )

    def _build_header_tensor(
        self,
        op: int,
        num_requests: int = 0,
        total_context_rows: int = 0,
        num_spec_tokens: int = 0,
        projected: bool = False,
    ) -> torch.Tensor:
        return torch.tensor(
            [
                PROTOCOL_VERSION,
                op,
                num_requests,
                total_context_rows,
                num_spec_tokens,
                1 if projected else 0,
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
            device=self.device,
        )

    def _build_request_meta_tensor(self, requests: list[dict[str, Any]]) -> torch.Tensor:
        meta = torch.zeros(
            (self._remote_max_requests, 8), dtype=torch.int64, device=self.device
        )
        for i, req in enumerate(requests):
            meta[i, 0] = 1 if req.get("reset", False) else 0
            meta[i, 1] = int(req.get("reset_position", 0))
            meta[i, 2] = int(req.get("context_count", 0))
            meta[i, 3] = int(req.get("anchor_token_id", 0))
            meta[i, 4] = int(req.get("anchor_position", 0))
            seed = int(req.get("seed", 0))
            meta[i, 5] = seed & 0xFFFFFFFF
            meta[i, 6] = (seed >> 32) & 0xFFFFFFFF
        return meta.reshape(-1)

    def _build_temperature_tensor(self, requests: list[dict[str, Any]]) -> torch.Tensor:
        temps = torch.zeros(
            self._remote_max_requests, dtype=torch.float32, device=self.device
        )
        for i, req in enumerate(requests):
            temps[i] = float(req.get("temperature", 1.0))
        return temps

    def _build_request_id_tensor(self, request_ids: list[str]) -> torch.Tensor:
        ids = torch.zeros(
            (self._remote_max_requests, 64), dtype=torch.int8, device=self.device
        )
        for i, request_id in enumerate(request_ids):
            raw = request_id.encode("utf-8")[:64]
            if raw:
                ids[i, : len(raw)] = torch.tensor(
                    list(raw), dtype=torch.int8, device=self.device
                )
        return ids.reshape(-1)

    def _rdma_ping(self, timeout_ms: int | None = None) -> dict[str, Any]:
        header = self._build_header_tensor(OP_PING)
        # int64[16] mirrors the request-header layout: the first 8 slots are the
        # original PONG fields, slots 8-10 publish the server's capacity so the
        # verifier can reject an undersized draft server at handshake time.
        response = torch.zeros(16, dtype=torch.int64, device=self.device)
        self._rdma_exchange(
            header,
            extra_sends=[],
            recv_tensors=[response],
            timeout_ms=timeout_ms,
        )
        values = response.tolist()
        return {
            "ok": bool(values[0]),
            "protocol": int(values[1]),
            "method": _CODE_METHOD.get(int(values[2]), "unknown"),
            "max_requests": int(values[3]),
            "block_size": int(values[4]),
            "window_size": int(values[5]),
            "prefix_cache_tokens": int(values[6]),
            "active_requests": int(values[7]),
            "max_batch_size": int(values[8]),
            "max_speculative_tokens": int(values[9]),
            "max_context_tokens": int(values[10]),
        }

    def _rdma_clear(self) -> list[int]:
        header = self._build_header_tensor(OP_CLEAR)
        status = torch.zeros(4, dtype=torch.int64, device=self.device)
        self._rdma_exchange(header, extra_sends=[], recv_tensors=[status])
        return status.tolist()

    def _rdma_free(self, request_ids: list[str]) -> list[int]:
        header = self._build_header_tensor(OP_FREE, num_requests=len(request_ids))
        req_ids = self._build_request_id_tensor(request_ids)
        status = torch.zeros(4, dtype=torch.int64, device=self.device)
        self._rdma_exchange(header, extra_sends=[req_ids], recv_tensors=[status])
        return status.tolist()

    def _rdma_reconnect(
        self,
        source_request_id: str,
        request_id: str,
        prefix_end: int,
    ) -> list[int]:
        header = self._build_header_tensor(OP_RECONNECT, num_requests=1)
        meta = torch.zeros(
            (self._remote_max_requests, 8), dtype=torch.int64, device=self.device
        )
        meta[0, 4] = int(prefix_end)
        meta = meta.reshape(-1)
        req_ids = self._build_request_id_tensor([source_request_id, request_id])
        status = torch.zeros(4, dtype=torch.int64, device=self.device)
        self._rdma_exchange(header, extra_sends=[meta, req_ids], recv_tensors=[status])
        return status.tolist()

    def init_cudagraph_manager(self, cudagraph_mode=None) -> None:
        """The standalone drafter owns its CUDA graph lifecycle."""

    def capture(self, *, capture_phase: CUDAGraphCapturePhase) -> None:
        """The verifier has no local draft graph to capture."""

    def _free_remote_requests(
        self,
        request_ids: set[str] | list[str],
        *,
        known_only: bool = True,
    ) -> None:
        """Free remote draft state.

        Args:
            request_ids: Request ids whose remote state is released.
            known_only: Free only ids the verifier still tracks. ``False``
                also frees ids dropped after a failed exchange; the server
                treats an unknown id as already free.
        """
        remote_request_ids = set(request_ids)
        if known_only:
            remote_request_ids &= self._known_requests
        remote_request_ids = sorted(remote_request_ids)
        if not remote_request_ids:
            return
        self._rdma_free(remote_request_ids)
        self._known_requests.difference_update(remote_request_ids)
        for request_id in remote_request_ids:
            self._retained_prefixes.pop(request_id, None)

    def _discard_failed_requests(self, request_ids: list[str]) -> None:
        """Forget all local state whose remote mutation is now uncertain.

        A failed exchange leaves the requests disabled, but records the ids
        whose remote slot may still exist so the next proposal frees them
        first and re-enables the requests (see
        ``_release_stale_remote_requests``). Ported from upstream
        myshytf/vllm 7f37e34ca.
        """
        failed = set(request_ids)
        self._disabled_requests.update(failed)
        self._stale_remote_requests.update(failed & self._known_requests)
        self._known_requests.difference_update(failed)
        self._active_requests.difference_update(failed)
        for request_id in failed:
            self._retained_prefixes.pop(request_id, None)

    def _release_stale_remote_requests(self) -> None:
        """Free remote slots left by failed exchanges and re-enable requests.

        Runs before a proposal. A transport failure here raises to the
        proposal's failure path, which keeps every id stale for a later
        attempt. After a successful FREE the requests draft again: nothing
        local refers to their old remote state, so their next proposal is a
        reset or a cold bootstrap. Ported from upstream myshytf/vllm
        7f37e34ca.
        """
        if not self._stale_remote_requests:
            return
        stale = set(self._stale_remote_requests)
        self._free_remote_requests(stale, known_only=False)
        self._stale_remote_requests.clear()
        self._disabled_requests.difference_update(stale)

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
        token_table = input_batch.all_token_ids_cpu
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
            status = self._rdma_reconnect(
                source_request_id,
                request_id,
                prefix_end,
            )
        except Exception:
            logger.exception(
                "Remote K3 DSpark prefix reconnect failed: source=%s, "
                "request=%s, prefix_end=%d",
                source_request_id,
                request_id,
                prefix_end,
            )
            # The server may or may not have rebound the source slot. Neither
            # id is trusted any more: both are freed before the next proposal,
            # after which the request cold-bootstraps. Ported from upstream
            # myshytf/vllm 7f37e34ca.
            self._retained_prefixes.pop(source_request_id, None)
            self._known_requests.discard(source_request_id)
            self._stale_remote_requests.update({source_request_id, request_id})
            return False
        self._retained_prefixes.pop(source_request_id)
        self._known_requests.discard(source_request_id)
        self._known_requests.add(request_id)
        self._retained_serial += 1
        restored_start = int(status[1]) if len(status) > 1 else 0
        self._retained_prefixes[request_id] = _RetainedRequestPrefix(
            token_ids=token_prefix.clone(),
            committed_end=prefix_end,
            context_start=restored_start,
            serial=self._retained_serial,
        )
        logger.info(
            "Remote K3 DSpark prefix reconnected: source=%s, request=%s, "
            "prefix_end=%d, restored_start=%s, latency_ms=%.1f",
            source_request_id,
            request_id,
            prefix_end,
            restored_start,
            float(status[1] if len(status) > 1 else 0.0),
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

    def _copy_draft_tokens_from_gpu(
        self,
        remote_tokens: torch.Tensor,
        active_indices: list[int],
        num_speculative_tokens: int,
    ) -> None:
        """Copy draft tokens received directly into a GPU tensor.

        Mirrors ``_copy_tokens_from_response`` semantics but sources the rows
        from the RDMA receive staging buffer (already moved H2D) instead of a
        JSON list, avoiding the Python-list round-trip.
        """
        expected_shape = (len(active_indices), num_speculative_tokens)
        if tuple(remote_tokens.shape) != expected_shape:
            raise ValueError(
                f"Remote DSpark token response has the wrong shape; "
                f"expected={expected_shape}, got={tuple(remote_tokens.shape)}"
            )
        # Degenerate-payload guard: the draft server returns an all-zero
        # buffer for dummy/warmup proposes and when its slot state desyncs
        # (seen mid-bench). Zeros are in-vocab so nothing else catches them;
        # verifying them wastes the step, and a desync never recovers without
        # a reset. Fail closed as a CONTENT error (request state is discarded
        # and resynced next step; the RDMA channel itself is left alone).
        if remote_tokens.numel() and bool((remote_tokens == 0).all().item()):
            raise _DraftContentError(
                "Remote K3 DSpark returned all-zero draft tokens; "
                "treating the step as no-draft"
            )
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
        temperature: torch.Tensor | None = None,
        seeds: torch.Tensor | None = None,
    ) -> None:
        started = time.perf_counter()
        if aux_hidden_states is None or len(aux_hidden_states) != self.num_aux_layers:
            raise ValueError(
                f"Remote K3 {self.method} requires {self.num_aux_layers} configured "
                "target auxiliary hidden states"
            )
        # Lazy channel bootstrap (see _ensure_rdma_channel): the first
        # propose() establishes the RDMA side-channel on a settled GPU. If
        # it is not ready, skip drafting this step (output stays filled
        # with -1 from propose(); fail-closed, engine unaffected).
        if not self._ensure_rdma_channel():
            return

        num_reqs = input_batch.num_reqs
        current_request_ids = set(input_batch.req_ids)
        previous_active_requests = self._active_requests
        self._disabled_requests.intersection_update(current_request_ids)
        self._release_stale_remote_requests()
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
            # Context-window cap (RESET-ONLY): the server retains at most
            # `_remote_window_size` tokens, so a FRESH server-side state only
            # needs the last `window_size` rows. This MUST apply solely on
            # reset: continuing appends must stay contiguous with the server's
            # committed_end, and slicing a continuing delta opens a context
            # gap (the 100k-chunked-prefill crash). `want_truncate` is decided
            # here; the slice itself happens in the reset branches below once
            # `reset` is known. Non-reset requests always ship the full delta.
            first_position = int(input_batch.num_computed_tokens_np[request_idx])
            shipped_count = valid_count
            truncated_start = first_position
            want_truncate = bool(
                self._remote_window_size > 0
                and valid_count > self._remote_window_size
            )
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
                    if want_truncate:
                        shipped_count = self._remote_window_size
                        request_gather = request_gather[-shipped_count:]
                        truncated_start = (
                            first_position + valid_count - shipped_count
                        )
                    context_start = truncated_start
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
                        if want_truncate:
                            shipped_count = self._remote_window_size
                            request_gather = request_gather[-shipped_count:]
                            truncated_start = (
                                first_position + valid_count - shipped_count
                            )
                        context_start = truncated_start
                        logger.warning(
                            "Remote K3 %s cold-bootstrapping cache-restored "
                            "request %s at position %d from %d fresh context "
                            "rows; target verification preserves correctness.",
                            self.method,
                            request_id,
                            first_position,
                            shipped_count,
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
                    "reset_position": truncated_start if reset else 0,
                    "context_count": shipped_count,
                    "anchor_token_id": int(anchor_tokens_cpu[request_idx]),
                    "temperature": (
                        float(temperature[request_idx])
                        if temperature is not None
                        else 1.0
                    ),
                    "seed": (
                        int(seeds[request_idx])
                        if seeds is not None
                        else 0
                    ),
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
        # Anchor positions and the ACTUAL first shipped position are read
        # straight from the GPU positions tensor (no D2H staging for the bulk
        # context; only a handful of scalar reads per request).
        anchor_positions = _anchor_positions_from_context(
            [int(request["context_count"]) for request in requests],
            positions,
        )
        # The analytical truncated_start can be wrong when scheduled rows are
        # not [first_position, first_position + valid_count) (chunked prefill,
        # rejected-token holes). The server requires reset_position <= first
        # shipped position, so overwrite it with the ACTUAL first shipped
        # position per request.
        _pos_offset = 0
        for _ri, _req in enumerate(requests):
            _count = int(_req["context_count"])
            if _count > 0:
                _first_shipped = int(positions[_pos_offset])
                if _req.get("reset", False):
                    _req["reset_position"] = _first_shipped
                    request_context_starts[_ri] = _first_shipped
                _pos_offset += _count
        for request, anchor_position in zip(
            requests,
            anchor_positions,
            strict=True,
        ):
            request["anchor_position"] = anchor_position
        # Build the fixed-size control tensors and hand the bulk (positions,
        # context) to the RDMA side-channel, which stages them D2H into the
        # registered host buffer (no GPUDirect on GB10).
        header_tensor = self._build_header_tensor(
            OP_PROPOSE,
            num_requests=len(requests),
            total_context_rows=num_context_rows,
            num_spec_tokens=num_speculative_tokens,
            projected=False,
        )
        meta_tensor = self._build_request_meta_tensor(requests)
        temp_tensor = self._build_temperature_tensor(requests)
        req_id_tensor = self._build_request_id_tensor(
            [str(request["request_id"]) for request in requests]
        )
        draft_tokens_gpu = torch.empty(
            (len(requests), num_speculative_tokens),
            dtype=torch.int64,
            device=self.device,
        )
        status_tensor = torch.zeros(4, dtype=torch.int64, device=self.device)
        extra_sends: list[torch.Tensor] = [meta_tensor, temp_tensor, req_id_tensor]
        if num_context_rows:
            extra_sends.append(positions)
            extra_sends.append(context)
        rdma_started = time.perf_counter()
        self._rdma_exchange(
            header_tensor,
            extra_sends=extra_sends,
            recv_tensors=[draft_tokens_gpu, status_tensor],
        )
        rdma_done = time.perf_counter()
        self._copy_draft_tokens_from_gpu(
            draft_tokens_gpu,
            active_indices,
            num_speculative_tokens,
        )
        output_copied = time.perf_counter()
        timing_ms = {
            "metadata_d2h": (metadata_ready - started) * 1000,
            "request_plan": (requests_ready - metadata_ready) * 1000,
            "context_gather": (context_ready - requests_ready) * 1000,
            "rdma_xfer": (rdma_done - rdma_started) * 1000,
            "tokens_copy": (output_copied - rdma_done) * 1000,
            "client_total": (output_copied - started) * 1000,
        }
        if status_tensor.numel() > 1:
            timing_ms["server_latency_us"] = float(status_tensor[1].item())
        self._record_timing(timing_ms)
        for request_idx, request, anchor_position, context_start in zip(
            active_indices,
            requests,
            anchor_positions,
            request_context_starts,
            strict=True,
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
        num_speculative_tokens: int | None = None,
        num_tokens_across_dp: torch.Tensor | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
    ) -> torch.Tensor:
        del (
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            num_tokens_across_dp,
            skip_attn_for_dummy_run,
            mm_inputs,
        )
        active_k = (
            int(num_speculative_tokens)
            if num_speculative_tokens is not None
            else self.num_speculative_steps
        )
        # A scheduler can intentionally disable speculation for one step (for
        # example, a request with max_tokens=1). ModelRunner treats an empty
        # second dimension as a normal non-speculative step. Return before any
        # RDMA exchange so a K=0 step never posts a zero-length message.
        if active_k == 0:
            return self.draft_tokens[: input_batch.num_reqs, :0].contiguous()
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
                    temperature,
                    seeds,
                )
            except _DraftContentError:
                output.fill_(-1)
                # Degenerate payload (e.g. all-zero drafts): the transport is
                # fine, so keep the channel and only discard the request
                # state. The stale-release path frees the slots and resyncs
                # via reset on the next step.
                self._discard_failed_requests(input_batch.req_ids)
                logger.warning(
                    "Remote K3 DSpark returned degenerate drafts; request "
                    "state discarded, resync on next step"
                )
            except Exception:
                output.fill_(-1)
                # The verifier cannot know whether a timed-out request mutated
                # remote KV. Fail closed for those requests until they leave
                # the active batch; FREE remains safe even if the server never
                # created the state. The ids are recorded as stale so the next
                # proposal frees their remote slots first and re-enables them
                # instead of disabling drafting for the rest of their lifetime
                # (upstream myshytf/vllm 7f37e34ca).
                self._discard_failed_requests(input_batch.req_ids)
                # A failed exchange may leave a stale WR posted on the QP;
                # drop the channel and re-establish on the next cooldown
                # instead of consuming a stale completion.
                self._invalidate_rdma()
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
        # K3 DIAG (TEMP): reference — the draft tokens this propose() returns,
        # to compare against the graph input_ids tail logged in model_runner.
        if output.numel():
            logger.info(
                "K3 DIAG propose output (first=%s last=%s flat_tail=%s)",
                output[0].tolist(),
                output[-1].tolist(),
                output.flatten()[-min(16, output.numel()):].tolist(),
            )
        self._tp_group.broadcast(output, src=0)
        return output
# V4PLUS-V6-REMOTE-DSPARK
# patch_remote_dspark_rdma_transport: applied
