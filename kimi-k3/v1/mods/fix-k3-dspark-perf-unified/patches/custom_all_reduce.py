# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.distributed.device_communicators.all_reduce_utils import (
    CUSTOM_ALL_REDUCE_MAX_SIZES,
    MiB,
    gpu_p2p_access_check,
)
from vllm.distributed.parallel_state import in_the_same_node_as
from vllm.logger import init_logger
from vllm.platforms import current_platform

try:
    ops.meta_size()
    custom_ar = True
except Exception:
    # For CPUs
    custom_ar = False

try:
    import torch.distributed._symmetric_memory as torch_symm_mem
except ImportError:
    torch_symm_mem = None

logger = init_logger(__name__)


def _get_pcie_allreduce_backend() -> str:
    backend = envs.VLLM_PCIE_ALLREDUCE_BACKEND.lower()
    if backend not in {"b12x", "cpp"}:
        raise ValueError(
            "Invalid VLLM_PCIE_ALLREDUCE_BACKEND: "
            f"{backend!r}. Valid values: b12x, cpp."
        )
    return backend


def _b12x_pcie_allreduce_requested() -> bool:
    return envs.VLLM_ENABLE_PCIE_ALLREDUCE and _get_pcie_allreduce_backend() == "b12x"


def _is_piecewise_cudagraph_runtime() -> bool:
    try:
        from vllm.config import CUDAGraphMode
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )
    except Exception:
        return False
    return (
        is_forward_context_available()
        and get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE
    )


def _parse_byte_size(value: str) -> int:
    normalized = value.upper().strip()
    suffixes = {
        "KB": 1024,
        "K": 1024,
        "MB": 1024 * 1024,
        "M": 1024 * 1024,
    }
    for suffix, multiplier in sorted(suffixes.items(), key=lambda item: -len(item[0])):
        if normalized.endswith(suffix):
            return int(normalized[: -len(suffix)]) * multiplier
    return int(value)


def _b12x_pcie_oneshot_limits() -> tuple[int, int, int]:
    allreduce_max_size = _parse_byte_size(envs.VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE)
    fused_max_size = _parse_byte_size(
        envs.VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE
    )
    if allreduce_max_size < 0 or fused_max_size < 0:
        raise ValueError("b12x PCIe oneshot size limits must be non-negative")
    # The pool only dispatches tensors within these limits. Reserving the
    # scheduler's full prefill tensor capacity per stream wastes hundreds of
    # MiB once target and draft graphs use independent channels.
    buffer_size = max(allreduce_max_size, fused_max_size, 16)
    return allreduce_max_size, fused_max_size, buffer_size


def _b12x_pcie_dma_min_bytes() -> int | None:
    configured = envs.VLLM_PCIE_DMA_MIN_BYTES.strip().lower()
    if configured in {"off", "disabled", "none"}:
        return None
    min_bytes = _parse_byte_size(configured)
    if min_bytes < 0:
        raise ValueError("b12x PCIe DMA minimum size must be non-negative")
    return min_bytes


@lru_cache(maxsize=1)
def _load_b12x_pcie_allreduce() -> Any | None:
    try:
        from sparkinfer.comm.pcie import AllReduce as PCIeAllReduce
    except Exception:
        return None
    return PCIeAllReduce


@lru_cache(maxsize=1)
def _load_b12x_pcie_dma() -> Any | None:
    try:
        from sparkinfer.comm.pcie import DmaAllReduce as PCIeDmaAllReduce
    except Exception:
        return None
    return PCIeDmaAllReduce


def _get_physical_device_numa_node(physical_device_id: int) -> int | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        try:
            numa_node = pynvml.nvmlDeviceGetNumaNodeId(handle)
            if numa_node >= 0 and _numa_node_has_cpus(numa_node):
                return int(numa_node)
        except Exception:
            pass

        for cpu_id in _get_device_cpu_affinity(pynvml, handle):
            numa_node = _get_numa_node_for_cpu(cpu_id)
            if numa_node is not None:
                return numa_node
    except Exception:
        return None
    return None


def _numa_node_has_cpus(node_id: int) -> bool:
    try:
        return (
            Path(f"/sys/devices/system/node/node{node_id}/cpulist")
            .read_text(encoding="utf-8")
            .strip()
            != ""
        )
    except (OSError, ValueError):
        return False


def _get_device_cpu_affinity(pynvml: Any, handle: Any) -> list[int]:
    cpu_count = os.cpu_count()
    if cpu_count is None:
        return []

    cpu_set_size = (cpu_count + 63) // 64
    cpu_affinity_mask = pynvml.nvmlDeviceGetCpuAffinity(handle, cpu_set_size)

    cpu_ids = []
    for i, mask in enumerate(cpu_affinity_mask):
        for bit in range(64):
            cpu_id = i * 64 + bit
            if cpu_id >= cpu_count:
                break
            if mask & (1 << bit):
                cpu_ids.append(cpu_id)
    return cpu_ids


def _get_numa_node_for_cpu(cpu_id: int) -> int | None:
    node_path = Path("/sys/devices/system/node")
    if not node_path.exists():
        return None

    for node_dir in node_path.iterdir():
        if not node_dir.name.startswith("node"):
            continue
        try:
            node_id = int(node_dir.name[4:])
            cpulist_file = node_dir / "cpulist"
            if cpulist_file.exists() and _cpu_in_cpulist(
                cpu_id, cpulist_file.read_text(encoding="utf-8").strip()
            ):
                return node_id
        except (ValueError, OSError):
            continue
    return None


def _cpu_in_cpulist(cpu_id: int, cpulist: str) -> bool:
    for part in cpulist.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            if int(start) <= cpu_id <= int(end):
                return True
        elif part and cpu_id == int(part):
            return True
    return False


def _is_cross_numa_topology(physical_device_ids: list[int]) -> bool:
    numa_nodes: list[int] = []
    for physical_device_id in physical_device_ids:
        numa_node = _get_physical_device_numa_node(physical_device_id)
        if numa_node is not None:
            numa_nodes.append(numa_node)

    return len(set(numa_nodes)) > 1


def _can_p2p(rank: int, world_size: int) -> bool:
    for i in range(world_size):
        if i == rank:
            continue
        if envs.VLLM_SKIP_P2P_CHECK:
            logger.debug("Skipping P2P check and trusting the driver's P2P report.")
            # can_device_access_peer takes visible device ordinals, while
            # rank and i are logical local IDs.
            return torch.cuda.can_device_access_peer(
                current_platform.logical_device_id_to_visible_device_id(rank),
                current_platform.logical_device_id_to_visible_device_id(i),
            )
        if not gpu_p2p_access_check(rank, i):
            return False
    return True


from vllm.distributed.utils import is_weak_contiguous  # noqa: E402


class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8, 16]
    _DEFAULT_ALL_GATHER_MAX_SIZE = 2 * 1024 * 1024
    _DEFAULT_MNNVL_ALL_GATHER_MAX_SIZES = {
        2: 8 * 1024 * 1024,
        4: 4 * 1024 * 1024,
        6: 2 * 1024 * 1024,
        8: 2 * 1024 * 1024,
        16: 2 * 1024 * 1024,
    }
    _DEFAULT_REDUCE_SCATTER_MAX_SIZE = 16 * 1024 * 1024
    _DEFAULT_MNNVL_REDUCE_SCATTER_MAX_SIZE = 16 * 1024 * 1024

    # max_size: max supported allreduce size
    def __init__(
        self,
        group: ProcessGroup,
        device: int | str | torch.device,
        max_size=8192 * 1024,
        max_all_gather_size=_DEFAULT_ALL_GATHER_MAX_SIZE,
        max_mnnvl_all_gather_size=None,
        max_reduce_scatter_size=_DEFAULT_REDUCE_SCATTER_MAX_SIZE,
        max_mnnvl_reduce_scatter_size=_DEFAULT_MNNVL_REDUCE_SCATTER_MAX_SIZE,
        symm_mem_enabled=False,
        nccl_group: ProcessGroup | None = None,
    ) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bound to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self._ptr = 0
        self.disabled = True
        self.mnnvl_buffer = None
        self.mnnvl_handle = None
        self.mnnvl_peer_buffers: list[torch.Tensor] | None = None
        self.mnnvl_multicast_ptr = 0
        self.mnnvl_buffer_size = 0
        self.mnnvl_lamport_ag_local_ptr = 0
        self.mnnvl_lamport_ag_multicast_ptr = 0
        self.mnnvl_lamport_rs_local_ptr = 0
        self.mnnvl_lamport_epochs = None
        self.mnnvl_lamport_ag_epoch_ptr = 0
        self.mnnvl_lamport_rs_epoch_ptr = 0
        self.mnnvl_only = False
        self._pcie_runtime = None
        self._pcie_dma = None
        self._pcie_capture_stream: torch.cuda.Stream | None = None
        self._pcie_allreduce_max_size: int | None = None
        self._pcie_fused_add_rms_norm_max_size: int | None = None
        self._pcie_composed_add_rms_norm_max_size: int | None = None
        self._cpp_ar_cutoff_size: int | None = None
        self._cpp_ar_ignore_cutoff_max_rows = 0
        self._pcie_cpp_backend = False
        self._pcie_logged_first_allreduce = False

        if not custom_ar:
            # disable because of missing custom allreduce library
            # e.g. in a non-GPU environment
            logger.info_once(
                "Custom allreduce is disabled because "
                "of missing custom allreduce library"
            )
            return

        self.group = group
        self.nccl_group = nccl_group

        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "CustomAllreduce should be attached to a non-NCCL group."
        )

        same_node = all(in_the_same_node_as(group, source_rank=0))
        self.mnnvl_only = not same_node

        rank = dist.get_rank(group=self.group)
        self.rank = rank
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        b12x_pcie_configured = _b12x_pcie_allreduce_requested()
        b12x_pcie_requested = b12x_pcie_configured and same_node
        if (
            world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES
            and not b12x_pcie_requested
        ):
            logger.warning_once(
                "Custom allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size,
                str(CustomAllreduce._SUPPORTED_WORLD_SIZES),
            )
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
        # Read once: envs attribute access hits os.environ, and
        # should_custom_ar is on the per-all-reduce dispatch path.
        self.allow_pcie = envs.VLLM_ALLOW_CUSTOM_ALLREDUCE_PCIE and same_node
        # On PCIe-only opt-in topologies the 2-stage kernel keeps beating
        # NCCL well past the default 8 MB ceiling (measured 1.2-1.3x at
        # 8-64 MB and 1.15x at 128-256 MB on 4x RTX PRO 6000), so raise the
        # ceiling to cover chunked-prefill all-reduces (16k tokens x 4k
        # hidden x bf16 = 128 MiB).
        if self.allow_pcie:
            max_size = max(max_size, 256 * MiB)
        device_capability = current_platform.get_device_capability()
        if (
            current_platform.is_cuda()
            and symm_mem_enabled
            and device_capability is not None
        ):
            device_capability_str = device_capability.as_version_str()
            if (
                device_capability_str in CUSTOM_ALL_REDUCE_MAX_SIZES
                and world_size in CUSTOM_ALL_REDUCE_MAX_SIZES[device_capability_str]
            ):
                max_size = min(
                    CUSTOM_ALL_REDUCE_MAX_SIZES[device_capability_str][world_size],
                    max_size,
                )
        # Physical GPU identifiers are meaningful only within one host.
        physical_device_ids: list[int] = []
        fully_connected = False
        use_b12x_pcie_allreduce = False
        if same_node:
            physical_device_id = (
                current_platform.visible_device_id_to_physical_device_id(device.index)
            )
            tensor = torch.tensor([physical_device_id], dtype=torch.int, device="cpu")
            gather_list = [
                torch.tensor([0], dtype=torch.int, device="cpu")
                for _ in range(world_size)
            ]
            dist.all_gather(gather_list, tensor, group=self.group)
            physical_device_ids = [t.item() for t in gather_list]
            assert current_platform.is_cuda_alike()
            fully_connected = current_platform.is_fully_connected(physical_device_ids)
            if b12x_pcie_requested:
                if not current_platform.is_cuda():
                    logger.warning(
                        "Custom allreduce is disabled because b12x PCIe oneshot "
                        "allreduce requires CUDA."
                    )
                    return
                logger.debug(
                    "b12x PCIe oneshot allreduce requested "
                    "(world_size=%d, physical_device_ids=%s, fully_connected=%s).",
                    world_size,
                    physical_device_ids,
                    fully_connected,
                )
                use_b12x_pcie_allreduce = True
            elif not fully_connected:
                if envs.VLLM_ENABLE_PCIE_ALLREDUCE:
                    pcie_backend = _get_pcie_allreduce_backend()
                    if pcie_backend == "cpp" and world_size > 2:
                        logger.debug(
                            "PCIe custom allreduce enabled via "
                            "VLLM_ENABLE_PCIE_ALLREDUCE=1 "
                            "(backend=cpp, using vLLM C++ custom allreduce)."
                        )
                        # Preserve the legacy PCIe opt-in behavior: allow the same
                        # small-tensor C++ custom allreduce path as fully-connected
                        # topologies once the user explicitly enables it.
                        self._pcie_cpp_backend = True
                        fully_connected = True
                if world_size > 2 and not fully_connected and self.allow_pcie:
                    # The sync protocol is atomics-free (peer plain writes + local
                    # polling), so PCIe-only P2P is functionally fine. Keep the
                    # topology non-fully-connected so the C++ dispatch selects the
                    # explicitly requested algorithm below.
                    os.environ.setdefault("VLLM_CUSTOM_ALLREDUCE_ALGO", "2stage")
                    logger.info_once(
                        "Custom allreduce enabled on %d PCIe-only GPUs by "
                        "VLLM_ALLOW_CUSTOM_ALLREDUCE_PCIE (algo=%s).",
                        world_size,
                        os.environ["VLLM_CUSTOM_ALLREDUCE_ALGO"],
                    )
                elif world_size > 2 and not fully_connected:
                    logger.warning(
                        "Custom allreduce is disabled for >2 PCIe-only GPUs. "
                        "Set VLLM_ENABLE_PCIE_ALLREDUCE=1 for the integrated PCIe "
                        "backend or VLLM_ALLOW_CUSTOM_ALLREDUCE_PCIE=1 for the "
                        "generic 2-stage C++ path."
                    )
                    return
        elif b12x_pcie_configured:
            logger.info_once(
                "B12X PCIe allreduce applies only to single-node groups; "
                "using the multi-node custom-collective path instead."
            )
        # test P2P capability, this checks software/cudaruntime support
        # this is expensive to compute at the first time
        # then we cache the result
        # On AMD GPU, p2p is always enabled between XGMI connected GPUs
        if (
            same_node
            and not current_platform.is_rocm()
            and not _can_p2p(rank, world_size)
        ):
            logger.warning(
                "Custom allreduce is disabled because your platform lacks "
                "GPU P2P capability or P2P test failed. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly."
            )
            return

        if use_b12x_pcie_allreduce:
            allow_cross_numa = envs.VLLM_PCIE_ONESHOT_ALLOW_CROSS_NUMA
            if _is_cross_numa_topology(physical_device_ids) and not allow_cross_numa:
                logger.warning(
                    "Custom allreduce is disabled because b12x PCIe oneshot "
                    "allreduce was requested on a cross-NUMA PCIe topology "
                    "(physical_device_ids=%s). Set "
                    "VLLM_PCIE_ONESHOT_ALLOW_CROSS_NUMA=1 or unset it to force it.",
                    physical_device_ids,
                )
                return
            allreduce_cls = _load_b12x_pcie_allreduce()
            if allreduce_cls is None:
                logger.warning(
                    "PCIe custom allreduce was requested, but "
                    "sparkinfer.comm.pcie.AllReduce is unavailable."
                )
                return
            # DMA must accommodate the largest scheduled prefill tensor. The
            # oneshot pool only needs its much smaller dispatch cutoffs.
            try:
                from vllm.config import get_current_vllm_config

                vllm_config = get_current_vllm_config()
                model_hidden_size = vllm_config.model_config.get_hidden_size()
                max_num_batched_tokens = (
                    vllm_config.scheduler_config.max_num_batched_tokens
                )
            except Exception:
                model_hidden_size = 6144
                max_num_batched_tokens = 8192
                logger.warning(
                    "vLLM config unavailable during CustomAllreduce init; "
                    "allocating b12x PCIe buffers for hidden=%d, rows<=%d.",
                    model_hidden_size,
                    max_num_batched_tokens,
                )
            pcie_buffer_size = max_num_batched_tokens * model_hidden_size * 2
            pcie_single_channel = envs.VLLM_PCIE_ONESHOT_SINGLE_CHANNEL
            (
                self._pcie_allreduce_max_size,
                self._pcie_fused_add_rms_norm_max_size,
                pcie_oneshot_buffer_size,
            ) = _b12x_pcie_oneshot_limits()
            self._pcie_composed_add_rms_norm_max_size = (
                self._pcie_fused_add_rms_norm_max_size
            )
            if self.nccl_group is None:
                logger.warning(
                    "Custom allreduce is disabled because b12x PCIe oneshot "
                    "allreduce requires a CUDA/NCCL device process group."
                )
                return
            self.max_size = pcie_buffer_size
            self.rank = rank
            self.world_size = world_size
            self.fully_connected = False
            pcie_runtime = None
            pcie_init_error: Exception | None = None
            try:
                pcie_runtime = allreduce_cls.from_exchange_group(
                    exchange_group=self.nccl_group,
                    device=self.device,
                    eager_buffer_bytes=pcie_oneshot_buffer_size,
                    max_size=pcie_oneshot_buffer_size,
                    single_channel=pcie_single_channel,
                )
                pcie_runtime.for_stream()
            except Exception as exc:
                pcie_init_error = exc

            pcie_failed = torch.tensor(
                [int(pcie_init_error is not None)], dtype=torch.int, device="cpu"
            )
            dist.all_reduce(pcie_failed, op=dist.ReduceOp.MAX, group=self.group)
            if int(pcie_failed.item()) != 0:
                if pcie_runtime is not None:
                    pcie_runtime.close()
                if pcie_init_error is not None:
                    logger.warning(
                        "b12x PCIe oneshot allreduce initialization failed on "
                        "rank %d: %s. Falling back to PyNCCL allreduce.",
                        rank,
                        pcie_init_error,
                    )
                else:
                    logger.warning(
                        "b12x PCIe oneshot allreduce initialization failed on "
                        "another TP rank. Falling back to PyNCCL allreduce."
                    )
                return
            assert pcie_runtime is not None
            self._pcie_runtime = pcie_runtime
            if not hasattr(pcie_runtime, "all_reduce_fused_add_rms_norm"):
                self._pcie_fused_add_rms_norm_max_size = 0

            # Prefill-size DMA allreduce alongside the oneshot. A deployment
            # preflight can tune its crossover or disable it when lossless DMA
            # never beats NCCL on the selected PCIe topology.
            dma_min_bytes = _b12x_pcie_dma_min_bytes()
            supports_dma = pcie_runtime.supports_all_peer_auxiliary
            dma_cls = (
                _load_b12x_pcie_dma()
                if supports_dma and dma_min_bytes is not None
                else None
            )
            if not supports_dma:
                logger.info(
                    "b12x PCIe %s allreduce does not support the auxiliary "
                    "peer topology required by DMA; large allreduces stay on "
                    "PyNCCL.",
                    pcie_runtime.algorithm,
                )
            elif dma_min_bytes is None:
                logger.info(
                    "b12x PCIe DMA allreduce disabled by "
                    "VLLM_PCIE_DMA_MIN_BYTES=off; large allreduces stay on PyNCCL."
                )
            elif dma_cls is None:
                logger.warning(
                    "b12x PCIe DMA allreduce unavailable "
                    "(sparkinfer.comm.pcie.DmaAllReduce not importable); "
                    "large allreduces stay on PyNCCL."
                )
            else:
                dma = None
                dma_error: Exception | None = None
                dma_fp8 = os.getenv(
                    "VLLM_PCIE_DMA_FP8", os.getenv("B12X_PCIE_DMA_FP8", "")
                )
                logger.debug(
                    "b12x PCIe DMA allreduce fp8 request: "
                    "VLLM_PCIE_DMA_FP8=%r B12X_PCIE_DMA_FP8=%r -> %r",
                    os.getenv("VLLM_PCIE_DMA_FP8"),
                    os.getenv("B12X_PCIE_DMA_FP8"),
                    dma_fp8,
                )
                try:
                    dma = dma_cls(
                        exchange_group=self.nccl_group,
                        device=self.device,
                        max_bytes=pcie_buffer_size,
                        fp8=dma_fp8,
                    )
                except Exception as exc:
                    dma_error = exc
                dma_failed = torch.tensor(
                    [int(dma_error is not None)], dtype=torch.int, device="cpu"
                )
                dist.all_reduce(dma_failed, op=dist.ReduceOp.MAX, group=self.group)
                if int(dma_failed.item()) != 0:
                    if dma is not None:
                        dma.close()
                    logger.warning(
                        "b12x PCIe DMA allreduce initialization failed "
                        "(rank %d error: %s); large allreduces stay on PyNCCL.",
                        rank,
                        dma_error,
                    )
                else:
                    assert dma is not None
                    assert dma_min_bytes is not None
                    dma.min_bytes = dma_min_bytes
                    self._pcie_dma = dma
                    logger.debug("b12x PCIe DMA allreduce wire mode: %s", dma.wire_mode)

            if rank == 0:
                logger.info(
                    "Configured b12x PCIe allreduce: algorithm=%s, max=%d, "
                    "fused max=%d, DMA min=%s.",
                    pcie_runtime.algorithm,
                    self._pcie_allreduce_max_size,
                    self._pcie_fused_add_rms_norm_max_size,
                    dma_min_bytes
                    if supports_dma and dma_min_bytes is not None
                    else "off",
                )
            self.disabled = False
            logger.debug(
                "Using b12x PCIe allreduce backend "
                "(world_size=%d, allreduce_max_size=%d, "
                "fused_add_rms_norm_max_size=%d, oneshot_buffer_size=%d, "
                "dma_buffer_size=%d, "
                "single_channel=%s).",
                world_size,
                self._pcie_allreduce_max_size,
                self._pcie_fused_add_rms_norm_max_size,
                pcie_oneshot_buffer_size,
                pcie_buffer_size,
                pcie_single_channel,
            )
            return

        if same_node and world_size > 2 and not fully_connected and not self.allow_pcie:
            logger.warning(
                "Custom allreduce is disabled because this PCIe topology is not "
                "fully connected and b12x PCIe oneshot is unavailable."
            )
            return

        self.disabled = False
        # Buffers memory are owned by this Python class and passed to C++.
        # Metadata composes of two parts: metadata for synchronization and a
        # temporary buffer for storing intermediate allreduce results.
        if same_node:
            self.meta_ptrs = self.create_shared_buffer(
                ops.meta_size() + max_size, group=group, uncached=True
            )
        else:
            meta_ptr, _ = ops.allocate_shared_buffer_and_handle(ops.meta_size())
            self.meta_ptrs = [meta_ptr] * world_size
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before the operation is performed
        legacy_buffer_size = max(max_size, max_all_gather_size, max_reduce_scatter_size)
        if same_node:
            self.buffer_ptrs = self.create_shared_buffer(
                legacy_buffer_size,
                group=group,
            )
        else:
            buffer_ptr, _ = ops.allocate_shared_buffer_and_handle(legacy_buffer_size)
            self.buffer_ptrs = [buffer_ptr] * world_size
        # This stores tuples of pointers to IPC buffers from all ranks.
        # Each registered tuple contains at most 16 addresses.
        # Allocating 8MB is enough for 65536 such tuples. The largest model uses
        # fewer than 10000 registered tuples.
        self.rank_data = torch.empty(
            8 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )
        self.max_size = max_size
        self.max_all_gather_size = max_all_gather_size
        if max_mnnvl_all_gather_size is None:
            max_mnnvl_all_gather_size = self._DEFAULT_MNNVL_ALL_GATHER_MAX_SIZES[
                world_size
            ]
        self.max_mnnvl_all_gather_size = max_mnnvl_all_gather_size
        self.max_reduce_scatter_size = max_reduce_scatter_size
        self.max_mnnvl_reduce_scatter_size = max_mnnvl_reduce_scatter_size
        self.rank = rank
        self.world_size = world_size
        self.fully_connected = fully_connected
        default_cutoff = "56KB" if self._pcie_cpp_backend else None
        cpp_ar_cutoff = os.getenv(
            "VLLM_CPP_AR_1STAGE_NCCL_CUTOFF", default_cutoff or ""
        )
        if cpp_ar_cutoff:
            self._cpp_ar_cutoff_size = _parse_byte_size(cpp_ar_cutoff)
        cpp_ar_ignore_rows = (
            os.getenv(
                "VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS",
                "1" if self._pcie_cpp_backend else "0",
            )
            or "0"
        )
        self._cpp_ar_ignore_cutoff_max_rows = int(cpp_ar_ignore_rows)
        if (
            self._cpp_ar_cutoff_size is not None
            and self._cpp_ar_ignore_cutoff_max_rows > 0
        ):
            logger.info(
                "Using dynamic C++ custom allreduce cutoff "
                "(cutoff=%d bytes, ignore_cutoff_max_rows=%d).",
                self._cpp_ar_cutoff_size,
                self._cpp_ar_ignore_cutoff_max_rows,
            )
        self._ptr = ops.init_custom_ar(
            self.meta_ptrs, self.rank_data, rank, self.fully_connected
        )
        ops.register_buffer(self._ptr, self.buffer_ptrs)
        self._init_mnnvl_buffer(
            max(
                max_mnnvl_all_gather_size * world_size,
                max_mnnvl_reduce_scatter_size,
            )
        )
        if not same_node and not self.mnnvl_multicast_ptr:
            logger.warning(
                "Custom collectives are disabled because this multi-node "
                "group does not support MNNVL multicast."
            )
            self.close()
            self.disabled = True

    def _init_mnnvl_buffer(self, stage_size: int) -> None:
        if torch_symm_mem is None or not current_platform.is_cuda():
            return
        try:
            buffer_size = stage_size * 6
            buffer = torch_symm_mem.empty(
                buffer_size, dtype=torch.uint8, device=self.device
            )
            handle = torch_symm_mem.rendezvous(buffer, self.group.group_name)
            if handle.multicast_ptr == 0:
                return
            peer_buffers = [
                handle.get_buffer(
                    peer,
                    (buffer_size,),
                    torch.uint8,
                    storage_offset=0,
                )
                for peer in range(self.world_size)
            ]
            ptrs = [peer_buffer.data_ptr() for peer_buffer in peer_buffers]
            lamport_ag_offset = 0
            lamport_rs_offset = stage_size * 3
            lamport_ag_ptrs = [ptr + lamport_ag_offset for ptr in ptrs]
            lamport_rs_ptrs = [ptr + lamport_rs_offset for ptr in ptrs]
            ops.register_buffer(self._ptr, lamport_ag_ptrs)
            ops.register_buffer(self._ptr, lamport_rs_ptrs)

            buffer.view(torch.int32).fill_(-2147483648)
            epochs = torch.zeros(
                (2, 32),
                dtype=torch.int32,
                device=self.device,
            )
            torch.accelerator.synchronize()
            dist.barrier(group=self.group)

            self.mnnvl_buffer = buffer
            self.mnnvl_handle = handle
            self.mnnvl_peer_buffers = peer_buffers
            self.mnnvl_multicast_ptr = handle.multicast_ptr
            self.mnnvl_buffer_size = stage_size
            self.mnnvl_lamport_ag_local_ptr = lamport_ag_ptrs[self.rank]
            self.mnnvl_lamport_ag_multicast_ptr = (
                handle.multicast_ptr + lamport_ag_offset
            )
            self.mnnvl_lamport_rs_local_ptr = lamport_rs_ptrs[self.rank]
            self.mnnvl_lamport_epochs = epochs
            self.mnnvl_lamport_ag_epoch_ptr = epochs[0].data_ptr()
            self.mnnvl_lamport_rs_epoch_ptr = epochs[1].data_ptr()
        except RuntimeError as error:
            logger.debug("MNNVL AG/RS initialization failed: %s", error)

    @contextmanager
    def capture(self, stream: torch.cuda.Stream | None = None):
        """Bind communicator resources to the enclosing CUDA graph capture.

        Legacy custom all-reduce registers graph buffers on exit. SparkInfer
        PCIe channels are instead created on the graph's owning stream and
        remain valid for the graph lifetime.
        """
        old_pcie_capture_stream = self._pcie_capture_stream
        try:
            self._IS_CAPTURING = True
            if self._pcie_runtime is None:
                yield
            else:
                self._pcie_capture_stream = stream
                with self._pcie_runtime.capture(stream=stream):
                    yield
        finally:
            self._pcie_capture_stream = old_pcie_capture_stream
            self._IS_CAPTURING = False
            if not self.disabled and self._pcie_runtime is None:
                self.register_graph_buffers()

    def checkpoint_pcie_channels(self) -> Any | None:
        """Snapshot SparkInfer PCIe channels before a disposable capture.

        Returns:
            An opaque runtime checkpoint, or ``None`` when PCIe all-reduce is
            unavailable.
        """
        runtime = self._pcie_runtime
        checkpoint = getattr(runtime, "checkpoint_channels", None)
        if checkpoint is None:
            return None
        return checkpoint()

    def rollback_pcie_channels(self, checkpoint: Any) -> None:
        """Release SparkInfer PCIe channels created after ``checkpoint``.

        Args:
            checkpoint: Opaque state returned by ``checkpoint_pcie_channels``.

        Raises:
            RuntimeError: If the SparkInfer PCIe runtime is unavailable.
        """
        runtime = self._pcie_runtime
        rollback = getattr(runtime, "rollback_channels", None)
        if rollback is None:
            raise RuntimeError("SparkInfer PCIe all-reduce runtime is unavailable")
        rollback(checkpoint)

    def _pcie_runtime_stream(self) -> torch.cuda.Stream | None:
        pinned = self._pcie_capture_stream
        if pinned is None:
            return None
        if not (self._IS_CAPTURING or torch.cuda.is_current_stream_capturing()):
            return None
        # Only pin the all-reduce onto the stored capture stream when that
        # stream is the one actually being captured (the full-CUDA-graph
        # path, where vLLM's graph_capture() makes _pcie_capture_stream the
        # current stream). Piecewise / inductor CUDA graphs (e.g. MTP or
        # spec-decode) capture on a torch-owned stream that is *not* our
        # stored stream; redirecting onto the stored stream there would
        # launch and allocate on a non-capturing stream and raise
        # cudaErrorStreamCaptureUnsupported. In that case return None so the
        # runtime runs inline on the current (capturing) stream.
        if torch.cuda.current_stream().cuda_stream != pinned.cuda_stream:
            return None
        return pinned

    def register_graph_buffers(self):
        if self._pcie_runtime is not None:
            self._pcie_runtime.for_stream(
                self._pcie_runtime_stream()
            ).register_graph_buffers()
            return
        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
        logger.debug("Registering %d cuda graph addresses", len(offset))
        # We cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.
        all_data: list[list[list[int] | None]]
        all_data = [[None, None] for _ in range(dist.get_world_size(group=self.group))]
        all_data[self.rank] = [handle, offset]
        ranks = sorted(dist.get_process_group_ranks(group=self.group))
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(
                all_data[i], src=rank, group=self.group, device="cpu"
            )
        # Unpack list of tuples to tuple of lists.
        handles = cast(list[list[int]], [d[0] for d in all_data])
        offsets = cast(list[list[int]], [d[1] for d in all_data])
        ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        if self._pcie_runtime is not None:
            inp_size = inp.numel() * inp.element_size()
            use_custom = (
                self._pcie_allreduce_max_size is not None
                and inp_size <= self._pcie_allreduce_max_size
                and self._pcie_runtime.for_stream(
                    self._pcie_runtime_stream()
                ).should_allreduce(inp)
            )
            if (
                not use_custom
                and self._pcie_dma is not None
                and self._pcie_dma.should_allreduce(inp)
            ):
                return True
            return use_custom
        if self.world_size > 8:
            return False
        inp_size = inp.numel() * inp.element_size()
        rows = int(inp.shape[0]) if inp.ndim >= 2 else 1
        cutoff_applies = not (
            self._cpp_ar_ignore_cutoff_max_rows > 0
            and rows <= self._cpp_ar_ignore_cutoff_max_rows
        )
        if (
            cutoff_applies
            and self._cpp_ar_cutoff_size is not None
            and inp_size > self._cpp_ar_cutoff_size
        ):
            return False
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # Keep the runtime guard aligned with the initialization contract.
        # The generic PCIe path intentionally remains non-fully-connected and
        # is admitted through its separate opt-in.
        if self.world_size == 2 or self.fully_connected or self.allow_pcie:
            return inp_size < self.max_size
        return False

    def backend_name(self) -> str:
        if self._pcie_runtime is not None:
            algorithm = self._pcie_runtime.algorithm.upper()
            if self._pcie_dma is not None:
                return f"B12X_PCIE_{algorithm}_DMA"
            return f"B12X_PCIE_{algorithm}"
        if self._pcie_cpp_backend:
            return "CUSTOM_CPP_PCIE"
        return "CUSTOM"

    def all_reduce(
        self, inp: torch.Tensor, *, out: torch.Tensor = None, registered: bool = False
    ):
        """Performs an out-of-place all reduce.

        If registered is True, this assumes inp's pointer is already
        IPC-registered. Otherwise, inp is first copied into a pre-registered
        buffer.
        """
        if self._pcie_runtime is not None:
            inp_size = inp.numel() * inp.element_size()
            oneshot_eligible = (
                self._pcie_allreduce_max_size is not None
                and inp_size <= self._pcie_allreduce_max_size
            )
            if (
                not oneshot_eligible
                and self._pcie_dma is not None
                and self._pcie_dma.should_allreduce(inp)
            ):
                stream = self._pcie_runtime_stream()
                if stream is not None:
                    with torch.cuda.stream(stream):
                        return self._pcie_dma.all_reduce(inp, out=out)
                return self._pcie_dma.all_reduce(inp, out=out)
            if not self._pcie_logged_first_allreduce:
                self._pcie_logged_first_allreduce = True
                logger.debug(
                    "b12x PCIe oneshot allreduce first dispatch: "
                    "shape=%s dtype=%s bytes=%d capture_stream=%s.",
                    tuple(inp.shape),
                    inp.dtype,
                    inp.numel() * inp.element_size(),
                    self._pcie_runtime_stream() is not None,
                )
            return self._pcie_runtime.all_reduce(
                inp, out=out, stream=self._pcie_runtime_stream()
            )
        if out is None:
            out = torch.empty_like(inp)
        if registered:
            ops.all_reduce(self._ptr, inp, out, 0, 0)
        else:
            ops.all_reduce(
                self._ptr, inp, out, self.buffer_ptrs[self.rank], self.max_size
            )
        return out

    def supports_fused_add_rms_norm(self) -> bool:
        """Return whether the B12X runtime provides fused AR + RMSNorm."""
        return (
            not self.disabled
            and self._pcie_runtime is not None
            and self._pcie_fused_add_rms_norm_max_size is not None
            and self._pcie_fused_add_rms_norm_max_size > 0
            and hasattr(
                self._pcie_runtime,
                "all_reduce_fused_add_rms_norm",
            )
        )

    def try_all_reduce_for_composed_add_rms_norm(
        self,
        inp: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run B12X AR for a caller-provided fused add + RMSNorm.

        This has an independent cutoff from generic all-reduce dispatch. It
        lets an eager model path select the measured B12X collective without
        changing the target model's all-reduce crossover.
        """
        inp_size = inp.numel() * inp.element_size()
        runtime = self._pcie_runtime
        max_size = self._pcie_composed_add_rms_norm_max_size
        if (
            self.disabled
            or runtime is None
            or max_size is None
            or max_size <= 0
            or inp_size > max_size
        ):
            return None
        stream = self._pcie_runtime_stream()
        channel = runtime.for_stream(stream)
        if not channel.should_allreduce(inp):
            return None
        return runtime.all_reduce(inp, stream=stream)

    def try_fused_add_rms_norm(
        self,
        inp: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        epsilon: float,
    ) -> bool:
        """Run the B12X fused collective when all operands are supported."""
        inp_size = inp.numel() * inp.element_size()
        if (
            not self.supports_fused_add_rms_norm()
            or self._pcie_fused_add_rms_norm_max_size is None
            or inp_size > self._pcie_fused_add_rms_norm_max_size
        ):
            return False
        assert self._pcie_runtime is not None
        if not self._pcie_runtime.for_stream(
            self._pcie_runtime_stream()
        ).should_allreduce(inp):
            return False
        if (
            inp.ndim == 0
            or residual.shape != inp.shape
            or residual.dtype != inp.dtype
            or residual.device != inp.device
            or not is_weak_contiguous(residual)
            or weight.shape != (inp.shape[-1],)
            or weight.dtype != inp.dtype
            or weight.device != inp.device
            or not weight.is_contiguous()
            or inp.shape[-1] * inp.element_size() % 16 != 0
            or inp.data_ptr() == residual.data_ptr()
            or epsilon < 0
        ):
            return False

        self._pcie_runtime.all_reduce_fused_add_rms_norm(
            inp,
            residual,
            weight,
            epsilon,
            out=inp,
            residual_out=residual,
            stream=self._pcie_runtime_stream(),
        )
        return True

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor | None:
        """The main allreduce API that provides support for cuda graph."""
        # When custom allreduce is disabled, this will be None.
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce(input, registered=True)
            else:
                # Piecewise CUDA graph execution can run split ops eagerly while
                # graph capture bookkeeping is active. Those ops need a real
                # all-reduce; returning a placeholder is only valid for warmup.
                if _is_piecewise_cudagraph_runtime():
                    return self.all_reduce(input, registered=False)
                # If warm up, mimic the allocation pattern since custom
                # allreduce is out-of-place.
                return torch.empty_like(input)
        else:
            # Note: outside of cuda graph context, custom allreduce incurs a
            # cost of cudaMemcpy, which should be small (<=1% of overall
            # latency) compared to the performance gain of using custom kernels
            return self.all_reduce(input, registered=False)

    def should_custom_all_gather(self, inp: torch.Tensor) -> bool:
        if self.disabled or not current_platform.is_cuda():
            return False
        if self.world_size == 16 and not self.mnnvl_only:
            return False
        inp_size = inp.nbytes
        if inp.dtype not in (
            torch.float32,
            torch.float16,
            torch.bfloat16,
        ):
            return False
        max_size = (
            self.max_mnnvl_all_gather_size
            if self.mnnvl_multicast_ptr
            else self.max_all_gather_size
        )
        return (
            0 < inp_size <= max_size
            and inp_size % 16 == 0
            and is_weak_contiguous(inp)
            and (self.fully_connected or bool(self.mnnvl_multicast_ptr))
        )

    def custom_all_gather(self, inp: torch.Tensor) -> torch.Tensor | None:
        if not self.should_custom_all_gather(inp):
            return None
        out_shape = (inp.shape[0] * self.world_size,) + inp.shape[1:]
        if self.mnnvl_multicast_ptr:
            logger.info_once(
                "Using the MNNVL Lamport all-gather kernel.",
                scope="global",
            )
            out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
            ops.mnnvl_lamport_all_gather(
                self._ptr,
                inp,
                out,
                self.mnnvl_lamport_ag_local_ptr,
                self.mnnvl_lamport_ag_multicast_ptr,
                self.mnnvl_lamport_ag_epoch_ptr,
                self.mnnvl_buffer_size,
            )
        else:
            out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
            ops.custom_all_gather(
                self._ptr,
                inp,
                out,
                self.buffer_ptrs[self.rank],
                self.max_all_gather_size,
            )
        return out

    def should_custom_reduce_scatter(self, inp: torch.Tensor) -> bool:
        if self.disabled or not current_platform.is_cuda():
            return False
        if self.world_size == 16 and not self.mnnvl_only:
            return False
        inp_size = inp.nbytes
        if inp.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            return False
        if inp.shape[0] % self.world_size != 0:
            return False
        output_size = inp_size // self.world_size
        max_size = (
            self.max_mnnvl_reduce_scatter_size
            if self.mnnvl_multicast_ptr
            else self.max_reduce_scatter_size
        )
        return (
            0 < inp_size <= max_size
            and output_size % 16 == 0
            and is_weak_contiguous(inp)
            and (self.fully_connected or bool(self.mnnvl_multicast_ptr))
        )

    def custom_reduce_scatter(self, inp: torch.Tensor) -> torch.Tensor | None:
        if not self.should_custom_reduce_scatter(inp):
            return None
        out_shape = (inp.shape[0] // self.world_size,) + inp.shape[1:]
        out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
        if self.mnnvl_multicast_ptr:
            logger.info_once(
                "Using the MNNVL Lamport reduce-scatter kernel.",
                scope="global",
            )
            ops.mnnvl_lamport_reduce_scatter(
                self._ptr,
                inp,
                out,
                self.mnnvl_lamport_rs_local_ptr,
                self.mnnvl_lamport_rs_epoch_ptr,
                self.mnnvl_buffer_size,
            )
        else:
            ops.custom_reduce_scatter(
                self._ptr,
                inp,
                out,
                self.buffer_ptrs[self.rank],
                self.max_reduce_scatter_size,
            )
        return out

    def close(self):
        if self._pcie_dma is not None:
            self._pcie_dma.close()
            self._pcie_dma = None
        if self._pcie_runtime is not None:
            self._pcie_runtime.close()
            self._pcie_runtime = None
        if not self.disabled and self._ptr:
            if ops is not None:
                ops.dispose(self._ptr)
            self._ptr = 0
            self.free_shared_buffer(self.meta_ptrs, rank=self.rank)
            self.free_shared_buffer(self.buffer_ptrs, rank=self.rank)
            self.mnnvl_peer_buffers = None
            self.mnnvl_handle = None
            self.mnnvl_buffer = None
            self.mnnvl_lamport_epochs = None

    def __del__(self):
        self.close()

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int,
        group: ProcessGroup | None = None,
        uncached: bool | None = False,
    ) -> list[int]:
        pointer, handle = ops.allocate_shared_buffer_and_handle(size_in_bytes)

        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        handles = [None] * world_size
        dist.all_gather_object(handles, handle, group=group)

        pointers: list[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer)  # type: ignore
            else:
                pointers.append(ops.open_mem_handle(h))
        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: list[int],
        group: ProcessGroup | None = None,
        rank: int | None = None,
    ) -> None:
        if rank is None:
            rank = dist.get_rank(group=group)
        if ops is not None:
            ops.free_shared_buffer(pointers[rank])


def get_b12x_pcie_allreduce() -> CustomAllreduce | None:
    """Return the active TP B12X PCIe all-reduce runtime, if available."""
    try:
        from vllm.distributed.parallel_state import get_tp_group

        device_communicator = get_tp_group().device_communicator
    except (AssertionError, RuntimeError):
        return None
    if device_communicator is None:
        return None
    custom_allreduce = getattr(device_communicator, "ca_comm", None)
    if (
        isinstance(custom_allreduce, CustomAllreduce)
        and custom_allreduce.supports_fused_add_rms_norm()
    ):
        return custom_allreduce
    return None


def get_active_b12x_pcie_allreduce() -> CustomAllreduce | None:
    """Return the active TP B12X runtime, including hierarchical mode."""
    try:
        from vllm.distributed.parallel_state import get_tp_group

        device_communicator = get_tp_group().device_communicator
    except (AssertionError, RuntimeError):
        return None
    if device_communicator is None:
        return None
    custom_allreduce = getattr(device_communicator, "ca_comm", None)
    if (
        isinstance(custom_allreduce, CustomAllreduce)
        and not custom_allreduce.disabled
        and custom_allreduce._pcie_runtime is not None
    ):
        return custom_allreduce
    return None
