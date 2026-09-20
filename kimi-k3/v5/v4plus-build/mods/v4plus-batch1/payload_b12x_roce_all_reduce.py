# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RoCEnante: adapter for the b12x one-shot RoCE collectives (multi-node DGX Spark TP).

Thin shim: capability voting, construction, and size gating live here; the
protocol lives in ``b12x.comm.roce``.  Enabled with
``VLLM_ENABLE_ROCE_ALLREDUCE=1`` for tensor-parallel groups whose ranks span
nodes; single-node groups keep their existing backends.

Contract with the runtime (``b12x.comm.roce.API_VERSION`` ==
``REQUIRED_B12X_ROCE_API_VERSION``):

- Every rank parses the size limits and checks the API version before the
  vote; the parsed limits are exchanged and must be identical, and the
  runtime itself refuses ranks whose ABI, HCA count, slot geometry, spin
  limit or launch geometry differ.  Any rank that cannot take part disables
  the backend on every rank, at initialization only.
- Dispatch is rank-invariant: eligibility depends on dtype, shape, contiguity
  and size, never on pointer values, so all ranks route the same collective.
- Failures are fail-stop, never a fallback: a wait that times out freezes the
  runtime, later launches do nothing, and ``check_health`` (called by the
  worker after each step's host synchronization) raises so the step's output
  never leaves the worker.  Peers starve on the stalled rank and raise too.
- The runtime orders collectives across streams with an event and requires a
  single stream inside a CUDA graph capture, which is how vLLM captures.
"""

# V4PLUS (fork #597): the fork's b12x_roce_all_reduce.py shim, verbatim except
# the _parse_byte_size import (see the adapted import below).
# Installed into vllm/distributed/device_communicators/b12x_roce_all_reduce.py
# by patch_vllm_rocenante.py.

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
# V4PLUS (fork #597 adapted): the v4 tree has no b12x_pcie_all_reduce module
# (B12X PCIe AR postdates it); _parse_byte_size lives in custom_all_reduce.py.
from vllm.distributed.device_communicators.custom_all_reduce import (
    _parse_byte_size,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger

logger = init_logger(__name__)


REQUIRED_B12X_ROCE_API_VERSION = 1


class B12xRoceAllReduce:
    """Route eligible tensor-parallel all-reduces to ``b12x.comm.roce``."""

    backend_name = "B12X_ROCENANTE"

    def __init__(
        self,
        group: ProcessGroup,
        device_group: ProcessGroup | None,
        device: torch.device,
    ) -> None:
        self.disabled = True
        self.group = group
        self.device_group = device_group
        self.device = device
        self.rank = dist.get_rank(group=group)
        self.world_size = dist.get_world_size(group=group)
        self._runtime = None
        self._announced = False
        self._announced_gather = False

        if device_group is None:
            logger.warning("RoCEnante requires a CUDA process group.")
            return
        if all(in_the_same_node_as(group, source_rank=0)):
            logger.info("RoCEnante skipped: group is single-node.")
            return

        # Vote before the collective constructor so a rank that cannot take
        # part (missing package, wrong API version, unsupported device, or an
        # unparsable limit) disables the backend on every rank instead of
        # leaving peers in the runtime's setup exchange.  The parsed limits
        # travel with the vote and must be identical everywhere.
        reason, limits = self._local_capability()
        verdict = self._exchange_vote(reason, limits)
        if verdict is not None:
            logger.warning("RoCEnante disabled on every rank: %s", verdict)
            return
        max_size, max_gather = limits

        from b12x.comm import roce

        try:
            # Exchange setup over the CPU (gloo) group: using the torch NCCL
            # group would create a torch NCCL communicator that vLLM otherwise
            # never needs, costing ~3.4 GB of unified memory per rank on Spark.
            self._runtime = roce.AllReduce.from_exchange_group(
                exchange_group=group,
                device=device,
                max_size=max_size,
                max_gather_bytes=max_gather,
            )
        except Exception as exc:  # noqa: BLE001 - the runtime already coordinated ranks
            logger.warning("RoCEnante initialization failed: %s", exc)
            return
        self.disabled = False
        if self.rank == 0:
            logger.info(
                "Using RoCEnante (b12x one-shot RoCE collectives): world=%d, hcas=%s, "
                "all-reduce max=%d bytes, all-gather shard max=%d bytes.",
                self.world_size,
                ",".join(self._runtime.hca_names),
                max_size,
                max_gather,
            )

    def _local_capability(self) -> tuple[str | None, tuple[int, int] | None]:
        """Evaluate this rank's ability to take part, without any collective.

        Returns:
            A pair of the reason this rank cannot take part (None when it can)
            and the parsed ``(max_size, max_gather)`` limits (None on failure).
        """
        try:
            from b12x.comm import roce
        except ImportError as exc:  # missing package or a broken native build
            return f"b12x.comm.roce is not importable: {exc}", None
        api = getattr(roce, "API_VERSION", None)
        if api != REQUIRED_B12X_ROCE_API_VERSION:
            needed = REQUIRED_B12X_ROCE_API_VERSION
            return f"b12x.comm.roce API version {api}, adapter needs {needed}", None
        if not roce.is_supported(self.device):
            return "needs an integrated GPU with an active RDMA device", None
        try:
            limits = (
                _parse_byte_size(envs.VLLM_ROCE_ALLREDUCE_MAX_SIZE),
                _parse_byte_size(envs.VLLM_ROCE_ALLGATHER_MAX_SIZE),
            )
        except Exception as exc:  # noqa: BLE001 - reported through the vote
            return f"invalid RoCEnante size limit: {exc}", None
        return None, limits

    def _exchange_vote(
        self, reason: str | None, limits: tuple[int, int] | None
    ) -> str | None:
        """Gather every rank's capability result over the CPU group.

        Args:
            reason: This rank's reason for not taking part, or None.
            limits: This rank's parsed ``(max_size, max_gather)``, or None.

        Returns:
            None when every rank can proceed with identical limits, else the
            text naming the ranks that cannot or whose limits differ.
        """
        votes: list[tuple[str | None, tuple[int, int] | None]] = [
            (None, None)
        ] * self.world_size
        dist.all_gather_object(votes, (reason, limits), group=self.group)
        failures = [f"rank {i}: {r}" for i, (r, _) in enumerate(votes) if r]
        if failures:
            return "; ".join(failures)
        reference = votes[0][1]
        differing = [
            f"rank {i}: {lim}" for i, (_, lim) in enumerate(votes) if lim != reference
        ]
        if differing:
            return (
                f"size limits differ across ranks (rank 0: {reference}; "
                + "; ".join(differing)
                + ")"
            )
        return None

    def check_health(self) -> None:
        """Fail-stop check of the runtime.

        Raises:
            RuntimeError: When a RoCEnante wait timed out or its proxy died.
        """
        if not self.disabled and self._runtime is not None:
            self._runtime.check_health()

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        return not self.disabled and self._runtime.should_allreduce(inp)

    def custom_all_reduce(self, inp: torch.Tensor) -> torch.Tensor | None:
        if not self.should_custom_ar(inp):
            return None
        if not self._announced:
            self._announced = True
            # One confirmation line, rank 0 only; workers keep it at debug.
            log = logger.info if self.rank == 0 else logger.debug
            log(
                "RoCEnante all-reduce is live: first routed all-reduce is %d bytes "
                "(%s); NCCL remains the fallback above %s.",
                inp.numel() * inp.element_size(),
                str(inp.dtype).replace("torch.", ""),
                envs.VLLM_ROCE_ALLREDUCE_MAX_SIZE,
            )
        return self._runtime.all_reduce(inp)

    def should_all_gather(self, inp: torch.Tensor, dim: int) -> bool:
        return not self.disabled and self._runtime.should_all_gather(inp, dim)

    def all_gather(self, inp: torch.Tensor, dim: int) -> torch.Tensor:
        """Concatenate along ``dim`` (0 or last) directly in the output layout."""

        if not self._announced_gather:
            self._announced_gather = True
            log = logger.info if self.rank == 0 else logger.debug
            log(
                "RoCEnante all-gather is live: first routed shard is %s %s "
                "along dim %d.",
                tuple(inp.shape),
                str(inp.dtype).replace("torch.", ""),
                dim,
            )
        return self._runtime.all_gather(inp, dim=dim)

    def supports_fused_add_rms_norm(self) -> bool:
        return False

    @contextmanager
    def capture(self, stream: torch.cuda.Stream | None = None):
        if self.disabled:
            yield
            return
        # Compile and allocate scratch before capture; the runtime refuses both
        # inside a graph.  vLLM's gather shards have 16-byte rows (direct
        # layout), so the padded-gather scratch is not requested here.
        self._runtime.prepare((torch.bfloat16, torch.float16, torch.float32))
        with self._runtime.capture(stream=stream):
            yield

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None
        self.disabled = True
