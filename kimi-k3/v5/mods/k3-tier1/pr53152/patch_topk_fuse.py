#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PR53152 (+ #53327) — backport of upstream vLLM PRs #53152 / #53327.

Fuse the MXFP4 top-k finalization into the K3 latent-MoE tail collective:
the cute-DSL AllReduce/RMSNorm/ReduceScatter kernel gains a ``top_k`` mode
that gathers the per-route GEMM2 rows, applies the expert weights, and
reduces — instead of the runner materializing the finalized routed output
first. ``KimiK3LatentMoETailOp.initialize(experts_per_token=...)`` enables
it; the op then accepts either a finalized tensor (stock) or an
``UnfinalizedMoEOutput``.

PORTED (kernel side, self-contained):
  * NEW vllm/model_executor/layers/fused_moe/moe_output.py — minimal
    upstream subset: the ``UnfinalizedMoEOutput`` dataclass only (the full
    upstream module's deferred-finalize protocol depends on FusedMoE config
    fields and FlashInfer TRT-LLM expert wrappers this fork does not have).
  * ops/latent_moe_tail.py — contract/initialize ``experts_per_token``,
    ``__call__``/``_validate_inputs`` union handling, and the upstream
    capacity bumps (_MAX_NUM_TOKENS 16->128, _COLLECTIVE_TOKEN_CTAS 8->32).
  * ops/cute_dsl/latent_moe_tail/allreduce_..._early_exit.py — full top_k
    plumbing (kernel params, finalize block, compile/launch keys, runtime
    args, CollectiveKernel validation + dummy tensors).

NOTE-SKIPPED (production plumbing; depends on infrastructure the fork
lacks — do NOT port without the whole deferred-finalize MoE stack):
  * latent_moe_runner.py (nvidia + amd) defer gating — needs
    moe_config.defer_moe_finalize / use_deferred_moe_finalize /
    should_defer_moe_finalize and TrtLlmMxfp4/NvFp4ExpertsMonolithic,
    none of which exist in this fork (no trtllm expert modules at all).
  * fused_moe/config.py, modular_kernel.py, moe_runner.py,
    routed_experts.py, fused_moe_method_base.py, mxfp4.py, trtllm_*_moe.py,
    convert_flashinfer_moe_output — same missing infrastructure.
  * #53327 (bugfix): its only hunk fixes the #53152 runner defer gating
    (decide before the MoE kernel is built); since that gating is not
    ported, the bugfix has no target here. Its substance is moot in the
    kernel-only port.

The deferred mode is exercisable directly via
``KimiK3LatentMoETailOp.initialize(..., experts_per_token=16)`` (as the
upstream tests/benchmarks do).

Idempotent via marker. All touched files are py_compile'd with doraise.
"""
import os
import py_compile
import sys

SCRIPT_NAME = "patch_topk_fuse"
TAG = "pr53152 (upstream #53152 + #53327 backport, kernel side)"
MARKER = "pr53152 (upstream #53152)"

PAYLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "payload")

# (payload repo-relative path, VLLM_ROOT-relative target)
NEW_FILES = [
    (
        "vllm/model_executor/layers/fused_moe/moe_output.py",
        "model_executor/layers/fused_moe/moe_output.py",
    ),
]

TAIL = "models/kimi_k3/nvidia/ops/latent_moe_tail.py"
AR = "models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py"

# --- latent_moe_tail.py hunks ---------------------------------------------

LT_H1_ANCHOR = """from vllm.distributed import get_tp_group
from vllm.model_executor.warmup.cutedsl_warmup import (
"""
LT_H1_REPL = """from vllm.distributed import get_tp_group
# {MARKER}: minimal upstream moe_output subset (UnfinalizedMoEOutput only).
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.warmup.cutedsl_warmup import (
""".replace("{MARKER}", MARKER)

LT_H2_ANCHOR = """_MAX_NUM_TOKENS = 16
_SKINNY_MAX_NUM_TOKENS = 5
_MMA_TILER_MN = (64, 32)
_GEMM_CLUSTER_MN = (1, 8)
_B_PRIME_STAGES = 2
_COLLECTIVE_TOKEN_CTAS = 8
"""
LT_H2_REPL = """_MAX_NUM_TOKENS = 128
_SKINNY_MAX_NUM_TOKENS = 5
_MMA_TILER_MN = (64, 32)
_GEMM_CLUSTER_MN = (1, 8)
_B_PRIME_STAGES = 2
_COLLECTIVE_TOKEN_CTAS = 32
"""

LT_H3_ANCHOR = """    latent_size: int
    max_num_tokens: int
    rms_eps: float
"""
LT_H3_REPL = """    latent_size: int
    max_num_tokens: int
    rms_eps: float
    experts_per_token: int
"""

LT_H4A_ANCHOR = """        dtype: torch.dtype,
        device: torch.device,
        rms_eps: float,
    ) -> tuple[KimiK3LatentMoETailContract, dist.ProcessGroup]:
"""
LT_H4A_REPL = """        dtype: torch.dtype,
        device: torch.device,
        rms_eps: float,
        experts_per_token: int = 0,
    ) -> tuple[KimiK3LatentMoETailContract, dist.ProcessGroup]:
"""

LT_H4B_ANCHOR = """                max_num_tokens=_MAX_NUM_TOKENS,
                rms_eps=float(rms_eps),
            ),
"""
LT_H4B_REPL = """                max_num_tokens=_MAX_NUM_TOKENS,
                rms_eps=float(rms_eps),
                experts_per_token=experts_per_token,
            ),
"""

LT_H5_ANCHOR = """        dtype: torch.dtype,
        device: torch.device,
        rms_eps: float,
    ) -> "KimiK3LatentMoETailOp":
        contract, group = cls._contract_and_group(
            hidden_size=hidden_size,
            latent_size=latent_size,
            dtype=dtype,
            device=device,
            rms_eps=rms_eps,
        )
"""
LT_H5_REPL = """        dtype: torch.dtype,
        device: torch.device,
        rms_eps: float,
        experts_per_token: int = 0,
    ) -> "KimiK3LatentMoETailOp":
        contract, group = cls._contract_and_group(
            hidden_size=hidden_size,
            latent_size=latent_size,
            experts_per_token=experts_per_token,
            dtype=dtype,
            device=device,
            rms_eps=rms_eps,
        )
"""

LT_H6_ANCHOR = """                max_token_ctas=_COLLECTIVE_TOKEN_CTAS,
                rms_eps=contract.rms_eps,
                fp32_internal=False,
            )
"""
LT_H6_REPL = """                max_token_ctas=_COLLECTIVE_TOKEN_CTAS,
                rms_eps=contract.rms_eps,
                fp32_internal=False,
                top_k=contract.experts_per_token,
            )
"""

LT_H7A_ANCHOR = """    def __call__(
        self,
        routed_output: torch.Tensor,
        shared_output: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> torch.Tensor:
"""
LT_H7A_REPL = """    def __call__(
        self,
        routed_output: torch.Tensor | UnfinalizedMoEOutput,
        shared_output: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> torch.Tensor:
"""

LT_H7B_ANCHOR = """        self._up_projection.ensure_compiled(routed_output.shape[0])
        latent, shared_shard = self._collective(
"""
LT_H7B_REPL = """        num_tokens = (
            routed_output.expanded_idx_to_permuted_idx.shape[0]
            if isinstance(routed_output, UnfinalizedMoEOutput)
            else routed_output.shape[0]
        )
        self._up_projection.ensure_compiled(num_tokens)
        latent, shared_shard = self._collective(
"""

LT_H7C_ANCHOR = """        return self._lamport_copy(
            mailbox,
            m=routed_output.shape[0],
        ).squeeze(0)
"""
LT_H7C_REPL = """        return self._lamport_copy(
            mailbox,
            m=num_tokens,
        ).squeeze(0)
"""

LT_H8A_ANCHOR = """    def _validate_inputs(
        self,
        routed_output: torch.Tensor,
        shared_output: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> None:
        contract = self.contract
        if routed_output.ndim != 2:
            raise ValueError("routed_output must be a 2D tensor.")
        num_tokens = routed_output.shape[0]
        if routed_output.shape != (num_tokens, contract.latent_size):
            raise ValueError(
                f"routed_output must have shape [M, {contract.latent_size}]."
            )
"""
LT_H8A_REPL = """    def _validate_inputs(
        self,
        routed_output: torch.Tensor | UnfinalizedMoEOutput,
        shared_output: torch.Tensor,
        rms_weight: torch.Tensor,
        up_weight: torch.Tensor,
    ) -> None:
        contract = self.contract
        if isinstance(routed_output, UnfinalizedMoEOutput):
            num_tokens = routed_output.expanded_idx_to_permuted_idx.shape[0]
        else:
            if routed_output.ndim != 2:
                raise ValueError("routed_output must be a 2D tensor.")
            num_tokens = routed_output.shape[0]
            if routed_output.shape != (num_tokens, contract.latent_size):
                raise ValueError(
                    f"routed_output must have shape [M, {contract.latent_size}]."
                )
"""

LT_H8B_ANCHOR = """        tensors = (routed_output, shared_output, rms_weight, up_weight)
"""
LT_H8B_REPL = """        tensors = (shared_output, rms_weight, up_weight)
"""

# --- allreduce_rmsnorm_reduce_scatter_early_exit.py hunks ------------------

AR_H1_ANCHOR = """import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from .primitives import (
"""
AR_H1_REPL = """import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

# {MARKER}: minimal upstream moe_output subset (UnfinalizedMoEOutput only).
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput

from .primitives import (
""".replace("{MARKER}", MARKER)

AR_H2A_ANCHOR = """        fp32_internal: bool = False,
        include_reduce_scatter: bool = True,
        include_routed: bool = True,
    ):
"""
AR_H2A_REPL = """        fp32_internal: bool = False,
        include_reduce_scatter: bool = True,
        include_routed: bool = True,
        top_k: int = 0,
    ):
"""

AR_H2B_ANCHOR = """        self.fp32_internal = fp32_internal
        self.include_reduce_scatter = include_reduce_scatter
        self.include_routed = include_routed
"""
AR_H2B_REPL = """        self.fp32_internal = fp32_internal
        self.include_reduce_scatter = include_reduce_scatter
        self.include_routed = include_routed
        self.top_k = top_k
"""

AR_H3A_ANCHOR = """        m: Int32,
        epsilon: Float32,
        stream: cuda.CUstream,
    ):
        self.kernel(
"""
AR_H3A_REPL = """        m: Int32,
        epsilon: Float32,
        stream: cuda.CUstream,
        expert_weights: cute.Tensor,
        expanded_idx_to_permuted_idx: cute.Tensor,
    ):
        self.kernel(
"""

AR_H3B_ANCHOR = """            shared_peer_ptrs,
            m,
            epsilon,
        ).launch(
"""
AR_H3B_REPL = """            shared_peer_ptrs,
            m,
            epsilon,
            expert_weights,
            expanded_idx_to_permuted_idx,
        ).launch(
"""

AR_H4A_ANCHOR = """        shared_peer_ptrs: cute.Tensor,
        m: Int32,
        epsilon: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
"""
AR_H4A_REPL = """        shared_peer_ptrs: cute.Tensor,
        m: Int32,
        epsilon: Float32,
        expert_weights: cute.Tensor,
        expanded_idx_to_permuted_idx: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
"""

AR_H4B_ANCHOR = """                m,
                epsilon,
                token,
                token_cta,
                cta_y,
                logical_role,
                cluster_rank,
                tidx,
            )
            token = token + self.token_ctas
"""
AR_H4B_REPL = """                m,
                epsilon,
                token,
                token_cta,
                cta_y,
                logical_role,
                cluster_rank,
                tidx,
                expert_weights,
                expanded_idx_to_permuted_idx,
            )
            token = token + self.token_ctas
"""

AR_H5_ANCHOR = """        token: Int32,
        token_cta: Int32,
        cta_y: Int32,
        role: Int32,
        cluster_rank: Int32,
        tidx: Int32,
    ):
"""
AR_H5_REPL = """        token: Int32,
        token_cta: Int32,
        cta_y: Int32,
        role: Int32,
        cluster_rank: Int32,
        tidx: Int32,
        expert_weights: cute.Tensor,
        expanded_idx_to_permuted_idx: cute.Tensor,
    ):
"""

AR_H6_ANCHOR = """            local_ptr = cute.make_ptr(
                BFloat16,
                (latent_source.iterator + element_offset).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            local_packed = sanitize_negative_zero(
                load_global_u32x4(local_ptr, volatile=False)
            )
            multicast_offset = (
"""
AR_H6_REPL = """            if cutlass.const_expr(self.top_k > 0):
                # finalize topk reduction
                local_values = cute.make_rmem_tensor(
                    cute.make_layout((VEC_BF16,)), BFloat16
                )
                for element in cutlass.range_constexpr(VEC_BF16):
                    local_values[element] = BFloat16(0.0)
                for slot in cutlass.range_constexpr(self.top_k):
                    permuted_idx = expanded_idx_to_permuted_idx[token, slot]
                    if permuted_idx >= Int32(0):
                        permuted_element = (
                            Int64(permuted_idx) * self.latent_dim
                            + Int64(packed_idx) * VEC_BF16
                        )
                        permuted_ptr = cute.make_ptr(
                            BFloat16,
                            (latent_source.iterator + permuted_element).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        values = packed_u32x4_to_bf16x8(
                            load_global_u32x4(permuted_ptr, volatile=False)
                        )
                        weight = expert_weights[token, slot].to(Float32)
                        for element in cutlass.range_constexpr(VEC_BF16):
                            scaled = (values[element].to(Float32) * weight).to(BFloat16)
                            local_values[element] = (
                                local_values[element].to(Float32) + scaled.to(Float32)
                            ).to(BFloat16)
                local_packed = sanitize_negative_zero(
                    bf16x8_to_packed_u32x4(local_values.load())
                )
            else:
                local_ptr = cute.make_ptr(
                    BFloat16,
                    (latent_source.iterator + element_offset).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                local_packed = sanitize_negative_zero(
                    load_global_u32x4(local_ptr, volatile=False)
                )
            multicast_offset = (
"""

AR_H7A_ANCHOR = """    fp32_internal: bool,
    include_reduce_scatter: bool,
    include_routed: bool,
):
    return (
"""
AR_H7A_REPL = """    fp32_internal: bool,
    include_reduce_scatter: bool,
    include_routed: bool,
    *,
    top_k: int,
):
    return (
"""

AR_H7B_ANCHOR = """    return (
        torch.accelerator.current_device_index(),
        rank,
        tp_size,
        latent_dim,
        hidden_dim,
        max_m,
        max_token_ctas,
        fp32_internal,
        include_reduce_scatter,
        include_routed,
    )
"""
AR_H7B_REPL = """    return (
        torch.accelerator.current_device_index(),
        rank,
        tp_size,
        latent_dim,
        hidden_dim,
        max_m,
        max_token_ctas,
        fp32_internal,
        include_reduce_scatter,
        include_routed,
        top_k,
    )
"""

AR_H8A_ANCHOR = """    shared_flags: torch.Tensor,
    shared_peer_ptrs: torch.Tensor,
    rms_eps: float,
):
    return (
"""
AR_H8A_REPL = """    shared_flags: torch.Tensor,
    shared_peer_ptrs: torch.Tensor,
    rms_eps: float,
    *,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
):
    return (
"""

AR_H8B_ANCHOR = """        Int32(latent_source.shape[0]),
        Float32(rms_eps),
        cuda.CUstream(torch.cuda.current_stream(latent_source.device).cuda_stream),
    )
"""
AR_H8B_REPL = """        Int32(shared_source.shape[0]),
        Float32(rms_eps),
        cuda.CUstream(torch.cuda.current_stream(latent_source.device).cuda_stream),
        to_cute_dynamic_m(expert_weights, mode=0, assumed_align=16),
        to_cute_dynamic_m(expanded_idx_to_permuted_idx, mode=0, assumed_align=16),
    )
"""

AR_H9A_ANCHOR = """    rms_eps: float,
    fp32_internal: bool,
    include_reduce_scatter: bool = True,
    include_routed: bool = True,
) -> None:
    \"\"\"Compile the rank/M specialization without retaining caller tensors.\"\"\"
"""
AR_H9A_REPL = """    rms_eps: float,
    fp32_internal: bool,
    include_reduce_scatter: bool = True,
    include_routed: bool = True,
    top_k: int = 0,
) -> None:
    \"\"\"Compile the rank/M specialization without retaining caller tensors.\"\"\"
"""

AR_H9B_ANCHOR = """        include_routed,
    )
    if key in _COMPILED:
        return
"""
AR_H9B_REPL = """        include_routed,
        top_k=top_k,
    )
    if key in _COMPILED:
        return
"""

AR_H9C_ANCHOR = """    device = latent_output.device
    latent = torch.empty((max_m, latent_dim), dtype=torch.bfloat16, device=device)
    gamma = torch.empty((latent_dim,), dtype=torch.bfloat16, device=device)
"""
AR_H9C_REPL = """    device = latent_output.device
    latent_rows = max_m * max(top_k, 1)
    latent = torch.empty((latent_rows, latent_dim), dtype=torch.bfloat16, device=device)
    expert_weights = torch.empty(
        (max_m, max(top_k, 1)), dtype=torch.bfloat16, device=device
    )
    expanded_idx = torch.empty((max_m, max(top_k, 1)), dtype=torch.int32, device=device)
    gamma = torch.empty((latent_dim,), dtype=torch.bfloat16, device=device)
"""

AR_H9D_ANCHOR = """        fp32_internal=fp32_internal,
        include_reduce_scatter=include_reduce_scatter,
        include_routed=include_routed,
    )
    _COMPILED[key] = cute.compile(
"""
AR_H9D_REPL = """        fp32_internal=fp32_internal,
        include_reduce_scatter=include_reduce_scatter,
        include_routed=include_routed,
        top_k=top_k,
    )
    _COMPILED[key] = cute.compile(
"""

AR_H9E_ANCHOR = """            shared_flags,
            shared_peer_ptrs,
            rms_eps,
        ),
    )
"""
AR_H9E_REPL = """            shared_flags,
            shared_peer_ptrs,
            rms_eps,
            expert_weights=expert_weights,
            expanded_idx_to_permuted_idx=expanded_idx,
        ),
    )
"""

AR_H10A_ANCHOR = """    fp32_internal: bool,
    include_reduce_scatter: bool = True,
    include_routed: bool = True,
) -> None:
    compile_kernel(
"""
AR_H10A_REPL = """    fp32_internal: bool,
    include_reduce_scatter: bool = True,
    include_routed: bool = True,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    top_k: int = 0,
) -> None:
    compile_kernel(
"""

AR_H10B_ANCHOR = """        rms_eps=rms_eps,
        fp32_internal=fp32_internal,
        include_reduce_scatter=include_reduce_scatter,
        include_routed=include_routed,
    )
    _COMPILED[
"""
AR_H10B_REPL = """        rms_eps=rms_eps,
        fp32_internal=fp32_internal,
        include_reduce_scatter=include_reduce_scatter,
        include_routed=include_routed,
        top_k=top_k,
    )
    _COMPILED[
"""

AR_H10C_ANCHOR = """            include_routed,
        )
    ](
"""
AR_H10C_REPL = """            include_routed,
            top_k=top_k,
        )
    ](
"""

AR_H10D_ANCHOR = """            shared_flags,
            shared_peer_ptrs,
            rms_eps,
        )
    )
"""
AR_H10D_REPL = """            shared_flags,
            shared_peer_ptrs,
            rms_eps,
            expert_weights=expert_weights,
            expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        )
    )
"""

AR_H11A_ANCHOR = """        max_token_ctas: int,
        rms_eps: float,
        fp32_internal: bool,
    ) -> None:
        validate_shape(
"""
AR_H11A_REPL = """        max_token_ctas: int,
        rms_eps: float,
        fp32_internal: bool,
        top_k: int = 0,
    ) -> None:
        validate_shape(
"""

AR_H11B_ANCHOR = """        self.rms_eps = float(rms_eps)
        self.fp32_internal = fp32_internal
        device = torch.device("cuda", torch.accelerator.current_device_index())

        bytes_per_routed_buffer = max_m * tp_size * latent_dim * 2
"""
AR_H11B_REPL = """        self.rms_eps = float(rms_eps)
        self.fp32_internal = fp32_internal
        self.top_k = top_k
        device = torch.device("cuda", torch.accelerator.current_device_index())

        self._dummy_expert_weights = torch.empty(
            (max_m, max(top_k, 1)), dtype=torch.bfloat16, device=device
        )
        self._dummy_expanded_idx = torch.empty(
            (max_m, max(top_k, 1)), dtype=torch.int32, device=device
        )

        bytes_per_routed_buffer = max_m * tp_size * latent_dim * 2
"""

AR_H11C_ANCHOR = """                    rms_eps=self.rms_eps,
                    fp32_internal=fp32_internal,
                )
            dist.barrier(group=group, device_ids=[device.index])
"""
AR_H11C_REPL = """                    rms_eps=self.rms_eps,
                    fp32_internal=fp32_internal,
                    top_k=top_k,
                )
            dist.barrier(group=group, device_ids=[device.index])
"""

AR_H12A_ANCHOR = """    def __call__(
        self,
        latent_source: torch.Tensor,
        shared_source: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_source.ndim != 2 or shared_source.ndim != 2:
            raise ValueError("latent_source and shared_source must be rank-2")
        m = latent_source.shape[0]
        device = self._routed_workspace.device
        expected = (
            (latent_source, (m, self.latent_dim), "latent_source"),
            (shared_source, (m, self.hidden_dim), "shared_source"),
            (gamma, (self.latent_dim,), "gamma"),
        )
        for tensor, shape, name in expected:
            if (
                tensor.shape != shape
                or tensor.dtype != torch.bfloat16
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise ValueError(f"{name} must be contiguous CUDA BF16 {list(shape)}")
        if not 1 <= m <= self.max_m:
            raise ValueError(f"runtime M={m} must be in [1, {self.max_m}]")

        with torch.accelerator.device_index(device.index):
            launch(
                latent_source,
"""
AR_H12A_REPL = """    def __call__(
        self,
        latent_source: torch.Tensor | UnfinalizedMoEOutput,
        shared_source: torch.Tensor,
        gamma: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if shared_source.ndim != 2:
            raise ValueError("shared_source must be rank-2")
        expected: list[tuple[torch.Tensor, tuple[int, ...], str, torch.dtype]]
        if isinstance(latent_source, UnfinalizedMoEOutput):
            if self.top_k <= 0:
                raise ValueError("collective was not configured for top-k finalize")
            gemm2_permuted = latent_source.gemm2_permuted
            expert_weights = latent_source.expert_weights
            expanded_idx = latent_source.expanded_idx_to_permuted_idx
            m = expanded_idx.shape[0]
            expected = [
                (
                    gemm2_permuted,
                    (gemm2_permuted.shape[0], self.latent_dim),
                    "gemm2_permuted",
                    torch.bfloat16,
                ),
                (
                    expert_weights,
                    (m, self.top_k),
                    "expert_weights",
                    torch.bfloat16,
                ),
                (
                    expanded_idx,
                    (m, self.top_k),
                    "expanded_idx_to_permuted_idx",
                    torch.int32,
                ),
            ]
        else:
            if self.top_k > 0:
                raise ValueError("top-k collective requires an unfinalized output")
            if latent_source.ndim != 2:
                raise ValueError("latent_source must be rank-2")
            m = latent_source.shape[0]
            gemm2_permuted = latent_source
            expert_weights = self._dummy_expert_weights
            expanded_idx = self._dummy_expanded_idx
            expected = [
                (
                    latent_source,
                    (m, self.latent_dim),
                    "latent_source",
                    torch.bfloat16,
                ),
            ]
        device = self._routed_workspace.device
        expected.extend(
            [
                (
                    shared_source,
                    (m, self.hidden_dim),
                    "shared_source",
                    torch.bfloat16,
                ),
                (gamma, (self.latent_dim,), "gamma", torch.bfloat16),
            ]
        )
        for tensor, shape, name, dtype in expected:
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous CUDA {dtype} {list(shape)}"
                )
        if not 1 <= m <= self.max_m:
            raise ValueError(f"runtime M={m} must be in [1, {self.max_m}]")

        with torch.accelerator.device_index(device.index):
            launch(
                gemm2_permuted,
"""

AR_H12B_ANCHOR = """                max_m=self.max_m,
                max_token_ctas=self.max_token_ctas,
                fp32_internal=self.fp32_internal,
            )
        return (
"""
AR_H12B_REPL = """                max_m=self.max_m,
                max_token_ctas=self.max_token_ctas,
                fp32_internal=self.fp32_internal,
                expert_weights=expert_weights,
                expanded_idx_to_permuted_idx=expanded_idx,
                top_k=self.top_k,
            )
        return (
"""


def load(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError as exc:
        print(f"[{SCRIPT_NAME}] NOTE  {path}: unreadable ({exc})")
        return None


def save(path, src):
    try:
        with open(path, "w") as f:
            f.write(src)
        py_compile.compile(path, doraise=True)
        return True
    except (OSError, py_compile.PyCompileError) as exc:
        print(f"[{SCRIPT_NAME}] FAIL  {path}: write/compile failed ({exc})")
        return False


def install_new_file(payload_rel, dest_abs, label):
    payload_path = os.path.join(PAYLOAD_DIR, payload_rel)
    payload = load(payload_path)
    if payload is None:
        return False
    if os.path.exists(dest_abs):
        existing = load(dest_abs)
        if existing is not None and MARKER in existing:
            print(f"[{SCRIPT_NAME}] SKIP  {label} (already present)")
            return True
        print(f"[{SCRIPT_NAME}] NOTE  {label}: exists without marker; not overwritten.")
        return False
    os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
    if not payload.endswith("\n"):
        payload += "\n"
    payload += f"# {MARKER}\n"
    if not save(dest_abs, payload):
        return False
    print(f"[{SCRIPT_NAME}] APPLY {label} (new file)")
    return True


def patch_file(vroot, relpath, hunks):
    path = vroot + "/" + relpath
    src = load(path)
    if src is None:
        return False
    if MARKER in src:
        print(f"[{SCRIPT_NAME}] SKIP  {relpath} (already present)")
        return True
    ok = True
    for anchor, repl, desc in hunks:
        n = src.count(anchor)
        if n != 1:
            print(
                f"[{SCRIPT_NAME}] NOTE  {relpath}: anchor for '{desc}' found "
                f"{n}x (want 1); file skipped — stays stock."
            )
            ok = False
        else:
            src = src.replace(anchor, repl, 1)
    if not ok:
        return False
    if not save(path, src):
        return False
    print(f"[{SCRIPT_NAME}] APPLY {relpath}: {len(hunks)} hunk(s)")
    return True


def main():
    print(f"[{SCRIPT_NAME}] {TAG}")
    if len(sys.argv) != 2:
        print(f"usage: {SCRIPT_NAME}.py <VLLM_ROOT>")
        return 2
    vroot = sys.argv[1].rstrip("/")
    ok = True
    for payload_rel, target_rel in NEW_FILES:
        ok &= install_new_file(payload_rel, os.path.join(vroot, target_rel), target_rel)
    ok &= patch_file(
        vroot,
        TAIL,
        [
            (LT_H1_ANCHOR, LT_H1_REPL, "UnfinalizedMoEOutput import"),
            (LT_H2_ANCHOR, LT_H2_REPL, "capacity: _MAX_NUM_TOKENS=128, _COLLECTIVE_TOKEN_CTAS=32"),
            (LT_H3_ANCHOR, LT_H3_REPL, "contract experts_per_token field"),
            (LT_H4A_ANCHOR, LT_H4A_REPL, "_contract_and_group experts_per_token"),
            (LT_H4B_ANCHOR, LT_H4B_REPL, "contract init experts_per_token"),
            (LT_H5_ANCHOR, LT_H5_REPL, "initialize experts_per_token"),
            (LT_H6_ANCHOR, LT_H6_REPL, "CollectiveKernel top_k"),
            (LT_H7A_ANCHOR, LT_H7A_REPL, "__call__ union annotation"),
            (LT_H7B_ANCHOR, LT_H7B_REPL, "__call__ num_tokens from deferred output"),
            (LT_H7C_ANCHOR, LT_H7C_REPL, "lamport copy m=num_tokens"),
            (LT_H8A_ANCHOR, LT_H8A_REPL, "_validate_inputs union handling"),
            (LT_H8B_ANCHOR, LT_H8B_REPL, "_validate_inputs device/dtype tuple"),
        ],
    )
    ok &= patch_file(
        vroot,
        AR,
        [
            (AR_H1_ANCHOR, AR_H1_REPL, "UnfinalizedMoEOutput import"),
            (AR_H2A_ANCHOR, AR_H2A_REPL, "kernel class top_k param"),
            (AR_H2B_ANCHOR, AR_H2B_REPL, "kernel class top_k attr"),
            (AR_H3A_ANCHOR, AR_H3A_REPL, "__call__ expert tensors params"),
            (AR_H3B_ANCHOR, AR_H3B_REPL, "__call__ kernel args"),
            (AR_H4A_ANCHOR, AR_H4A_REPL, "kernel expert tensors params"),
            (AR_H4B_ANCHOR, AR_H4B_REPL, "kernel _token_device args"),
            (AR_H5_ANCHOR, AR_H5_REPL, "_token_device expert tensors params"),
            (AR_H6_ANCHOR, AR_H6_REPL, "top-k finalize block"),
            (AR_H7A_ANCHOR, AR_H7A_REPL, "_compile_key top_k kwonly"),
            (AR_H7B_ANCHOR, AR_H7B_REPL, "_compile_key tuple"),
            (AR_H8A_ANCHOR, AR_H8A_REPL, "_runtime_args expert kwargs"),
            (AR_H8B_ANCHOR, AR_H8B_REPL, "_runtime_args body (m from shared, expert tensors)"),
            (AR_H9A_ANCHOR, AR_H9A_REPL, "compile_kernel top_k param"),
            (AR_H9B_ANCHOR, AR_H9B_REPL, "compile_kernel key"),
            (AR_H9C_ANCHOR, AR_H9C_REPL, "compile_kernel dummy expert tensors"),
            (AR_H9D_ANCHOR, AR_H9D_REPL, "compile_kernel kernel top_k"),
            (AR_H9E_ANCHOR, AR_H9E_REPL, "compile_kernel runtime args"),
            (AR_H10A_ANCHOR, AR_H10A_REPL, "launch expert tensors + top_k"),
            (AR_H10B_ANCHOR, AR_H10B_REPL, "launch compile_kernel top_k"),
            (AR_H10C_ANCHOR, AR_H10C_REPL, "launch key top_k"),
            (AR_H10D_ANCHOR, AR_H10D_REPL, "launch runtime args"),
            (AR_H11A_ANCHOR, AR_H11A_REPL, "CollectiveKernel top_k param"),
            (AR_H11B_ANCHOR, AR_H11B_REPL, "CollectiveKernel top_k + dummies"),
            (AR_H11C_ANCHOR, AR_H11C_REPL, "CollectiveKernel compile top_k"),
            (AR_H12A_ANCHOR, AR_H12A_REPL, "CollectiveKernel.__call__ union validation"),
            (AR_H12B_ANCHOR, AR_H12B_REPL, "CollectiveKernel.__call__ launch args"),
        ],
    )
    print(
        f"[{SCRIPT_NAME}] NOTE  production defer plumbing NOT ported "
        f"(latent_moe_runner defer gating, FusedMoE config fields, "
        f"modular_kernel passthrough, trtllm experts, "
        f"convert_flashinfer_moe_output): the fork lacks the entire "
        f"deferred-finalize MoE stack those hunks build on. The kernel-side "
        f"fusion is exercisable via KimiK3LatentMoETailOp.initialize("
        f"..., experts_per_token=16)."
    )
    print(
        f"[{SCRIPT_NAME}] NOTE  #53327 (bugfix to the #53152 runner defer "
        f"gating) has no target in this port; skipped with the same hunks."
    )
    if ok:
        print(
            f"[{SCRIPT_NAME}] done. Tail collective now supports top-k "
            f"finalize fusion; capacity raised to 128 tokens / 32 token CTAs."
        )
    else:
        print(f"[{SCRIPT_NAME}] NOTE: some hunks skipped — see above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
