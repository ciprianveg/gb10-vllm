#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PR54168 — backport of upstream vLLM PR #54168.

Optimize the low-M fused latent-MoE tail:
  * 7-CTA / 64-thread collective geometry for M<=4 at (TP8, 3584, 7168)
    with a bf16 top-16 finalize (wide inline-PTX metadata loads +
    fma.rn.f32.bf16), single-token schedule skipping redundant Lamport
    arrival waits, parity-alternating DSM reduction slots so consecutive
    token waves overlap, multicast stores via multimem.st (NVLS), and
    compact ReduceScatter roles (one CTA covers several destinations).
  * top_k==16 fast path in the collective's finalize (finalize_top16_bf16);
    the generic path now accumulates in FP32.
  * fused_add_multicast_skinny_gemm: M<=5 uses vector_width=16 / 224
    threads, 32-byte alignment, and fma.rn.f32.bf16 accumulation.
  * lamport_copy: launch_dependents before polling (Lamport marker carries
    producer readiness), grid trimmed to the fragment count, copy+cleanup
    fused into one pass; _LAMPORT_COPY_THREADS 224 -> 128.

PREREQUISITE: pr53152 must be applied first — this PR builds on the top_k
finalize block and the CollectiveKernel top_k plumbing it adds.

No FUSED_TOPK16 interaction: the fork has no VLLM_KIMI_FUSED_TOPK16
env/implementation (verified by grep over the reference tree), so this
port lands unmodified.

Pure Python / CuTe DSL. Idempotent via marker; py_compile doraise.
"""
import py_compile
import sys

SCRIPT_NAME = "patch_lowm_tail"
TAG = "pr54168 (upstream #54168 backport)"
MARKER = "pr54168 (upstream #54168)"
PREREQ_MARKER = "pr53152 (upstream #53152)"

PRIM = "models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/primitives.py"
AR = "models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py"
FG = "models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/fused_add_multicast_skinny_gemm.py"
LC = "models/kimi_k3/nvidia/ops/cute_dsl/latent_moe_tail/lamport_copy.py"
TAIL = "models/kimi_k3/nvidia/ops/latent_moe_tail.py"

# --- primitives.py hunks ----------------------------------------------------

PR_H1_ANCHOR = """@dsl_user_op
def load_global_u32x4(
"""
PR_H1_REPL = '''@dsl_user_op
def fma_f32_bf16(  # {MARKER}
    a: BFloat16,
    b: BFloat16,
    acc: Float32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    a_bits = llvm.bitcast(T.i16(), a.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    b_bits = llvm.bitcast(T.i16(), b.ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    result = llvm.inline_asm(
        T.f32(),
        [a_bits, b_bits, acc.ir_value(loc=loc, ip=ip)],
        "fma.rn.f32.bf16 $0, $1, $2, $3;",
        "=f,h,h,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Float32(result)


@dsl_user_op
def load_global_u32x4(
'''.replace("{MARKER}", MARKER)

PR_H2_ANCHOR = """@dsl_user_op
def store_global_u32x4(
"""
PR_H2_REPL = '''def _make_top16_bf16_finalize_asm() -> str:
    lines = [
        "{",
        ".reg .pred _valid<16>;",
        ".reg .s32 _index<16>;",
        ".reg .b32 _weight_words<8>;",
        ".reg .b16 _weights<16>;",
        ".reg .b32 _data<64>;",
        ".reg .b16 _lo, _hi;",
        ".reg .f32 _acc<8>;",
        ".reg .u64 _address<16>;",
        "ld.global.v4.u32 {_index0, _index1, _index2, _index3}, [$5];",
        "ld.global.v4.u32 {_index4, _index5, _index6, _index7}, [$5+16];",
        "ld.global.v4.u32 {_index8, _index9, _index10, _index11}, [$5+32];",
        "ld.global.v4.u32 {_index12, _index13, _index14, _index15}, [$5+48];",
        "ld.global.v4.u32 "
        "{_weight_words0, _weight_words1, _weight_words2, "
        "_weight_words3}, [$6];",
        "ld.global.v4.u32 "
        "{_weight_words4, _weight_words5, _weight_words6, "
        "_weight_words7}, [$6+16];",
    ]
    for pair in range(8):
        lines.append(
            f"mov.b32 {{_weights{2 * pair}, _weights{2 * pair + 1}}}, "
            f"_weight_words{pair};"
        )
    for element in range(8):
        lines.append(f"mov.f32 _acc{element}, 0f00000000;")
    for route in range(16):
        data = 4 * route
        lines.extend(
            [
                f"setp.ge.s32 _valid{route}, _index{route}, 0;",
                f"@_valid{route} mad.wide.s32 _address{route}, "
                f"_index{route}, 7168, $4;",
                f"@_valid{route} ld.global.v4.u32 "
                f"{{_data{data}, _data{data + 1}, _data{data + 2}, "
                f"_data{data + 3}}}, [_address{route}];",
            ]
        )
    for route in range(16):
        for pair in range(4):
            data = 4 * route + pair
            element = 2 * pair
            lines.extend(
                [
                    f"@_valid{route} mov.b32 {{_lo, _hi}}, _data{data};",
                    f"@_valid{route} fma.rn.f32.bf16 _acc{element}, _lo, "
                    f"_weights{route}, _acc{element};",
                    f"@_valid{route} fma.rn.f32.bf16 _acc{element + 1}, _hi, "
                    f"_weights{route}, _acc{element + 1};",
                ]
            )
    for pair in range(4):
        lines.append(f"cvt.rn.bf16x2.f32 ${pair}, _acc{2 * pair + 1}, _acc{2 * pair};")
    lines.append("}")
    return "\\n".join(lines)


_TOP16_BF16_FINALIZE_ASM = _make_top16_bf16_finalize_asm()


@dsl_user_op
def finalize_top16_bf16(
    gemm2_vector: cute.Pointer,
    route_indices: cute.Pointer,
    route_weights: cute.Pointer,
    *,
    loc=None,
    ip=None,
):
    """Finalize one Kimi K3 top-16 vector with wide metadata loads."""

    addresses = [
        pointer.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
        for pointer in (gemm2_vector, route_indices, route_weights)
    ]
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        addresses,
        _TOP16_BF16_FINALIZE_ASM,
        "=r,=r,=r,=r,l,l,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def store_global_u32x4(
'''.replace("{MARKER}", MARKER)

PR_H3_ANCHOR = """@dsl_user_op
def store_lamport_sentinel_128(pointer: cute.Pointer, *, loc=None, ip=None) -> None:
"""
PR_H3_REPL = '''@dsl_user_op
def stmc_bf16x8(address: Int64, packed, *, loc=None, ip=None) -> None:
    """Publish eight BF16 values through an NVLS multicast mapping."""

    words = [packed[i].ir_value(loc=loc, ip=ip) for i in range(4)]
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        "multimem.st.relaxed.sys.global.v4.bf16x2 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_lamport_sentinel_128(pointer: cute.Pointer, *, loc=None, ip=None) -> None:
'''.replace("{MARKER}", MARKER)

PR_H4_ANCHOR = """@dsl_user_op
def packed_u32x4_to_bf16x8(packed, *, loc=None, ip=None):
"""
PR_H4_REPL = '''@dsl_user_op
def load_shared_f32x2(pointer: cute.Pointer, *, loc=None, ip=None):
    """Load two aligned FP32 DSM partials from local shared memory."""

    address = Int32(pointer.toint(loc=loc, ip=ip))
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(2)
    )


@dsl_user_op
def load_shared_f32x4(pointer: cute.Pointer, *, loc=None, ip=None):
    """Load four aligned FP32 DSM partials from local shared memory."""

    address = Int32(pointer.toint(loc=loc, ip=ip))
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4)
    )


@dsl_user_op
def packed_u32x4_to_bf16x8(packed, *, loc=None, ip=None):
'''.replace("{MARKER}", MARKER)

# --- allreduce file hunks ---------------------------------------------------

AR_H1_ANCHOR = """from .primitives import (
    NUM_LAMPORT_BUFFERS,
    PACKED_BYTES,
    VEC_BF16,
    bf16x8_to_packed_u32x4,
    block_sum_specialized,
    fragment_is_dirty,
    load_global_u32x4,
    load_volatile_u32,
    map_shared_to_peer,
    packed_u32x4_to_bf16x8,
    red_async_release_gpu_add_u32,
    sanitize_negative_zero,
    store_global_u32x4,
    store_lamport_sentinel_128,
    store_shared_cluster_f32,
    to_cute,
    to_cute_dynamic_m,
)
"""
AR_H1_REPL = """from .primitives import (
    NUM_LAMPORT_BUFFERS,
    PACKED_BYTES,
    VEC_BF16,
    bf16x8_to_packed_u32x4,
    finalize_top16_bf16,
    fragment_is_dirty,
    load_global_u32x4,
    load_shared_f32x2,
    load_shared_f32x4,
    load_volatile_u32,
    map_shared_to_peer,
    packed_u32x4_to_bf16x8,
    red_async_release_gpu_add_u32,
    sanitize_negative_zero,
    stmc_bf16x8,
    store_global_u32x4,
    store_lamport_sentinel_128,
    store_shared_cluster_f32,
    to_cute,
    to_cute_dynamic_m,
    warp_sum_specialized,
)

_SEVEN_CTA_MAX_M = 4  # {MARKER}
""".replace("{MARKER}", MARKER)

AR_H2_ANCHOR = """        if include_reduce_scatter:
            self.cluster_ctas = mapped_cluster
            self.threads = mapped_threads
        else:
            self.threads, self.cluster_ctas = _select_routed_schedule(
                tp_size, latent_dim, hidden_dim, max_m
            )
"""
AR_H2_REPL = """        seven_cta_geometry = (
            include_routed
            and max_m <= _SEVEN_CTA_MAX_M
            and max_token_ctas == max_m
            and (tp_size, latent_dim, hidden_dim) == (8, 3584, 7168)
        )
        self.seven_cta_geometry = seven_cta_geometry
        self.single_token_geometry = seven_cta_geometry and max_m == 1
        if include_reduce_scatter:
            if seven_cta_geometry:
                self.cluster_ctas = 7
                self.threads = 64
                shared_roles = (tp_size + self.cluster_ctas - 1) // self.cluster_ctas
            else:
                self.cluster_ctas = mapped_cluster
                self.threads = mapped_threads
        else:
            if seven_cta_geometry:
                self.threads, self.cluster_ctas = 64, 7
            else:
                self.threads, self.cluster_ctas = _select_routed_schedule(
                    tp_size, latent_dim, hidden_dim, max_m
                )
        self.shared_roles = 1 if include_reduce_scatter else shared_roles
        self.shared_destination_stride = self.shared_roles * self.cluster_ctas
        self.shared_destinations_per_cta = (
            self.tp_size + self.shared_destination_stride - 1
        ) // self.shared_destination_stride
        self.shard_vectors = self.shard_dim // VEC_BF16
"""

AR_H3_ANCHOR = """        if include_routed and include_reduce_scatter:
            self.roles = 1 + shared_roles
        elif include_routed:
            self.roles = 1
        else:
            self.roles = shared_roles
"""
AR_H3_REPL = """        if include_routed and include_reduce_scatter:
            self.roles = 1 + self.shared_roles
        elif include_routed:
            self.roles = 1
        else:
            self.roles = self.shared_roles
"""

AR_H4A_ANCHOR = """        expert_weights: cute.Tensor,
        expanded_idx_to_permuted_idx: cute.Tensor,
    ):
        self.kernel(
"""
AR_H4A_REPL = """        expert_weights: cute.Tensor,
        expanded_idx_to_permuted_idx: cute.Tensor,
    ):
        grid_x = m if cutlass.const_expr(self.seven_cta_geometry) else self.token_ctas
        self.kernel(
"""

AR_H4B_ANCHOR = """        ).launch(
            grid=(self.token_ctas, self.cluster_ctas, self.roles),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster_ctas, 1),
            smem=(self.warps + self.cluster_ctas) * 4,
            stream=stream,
            use_pdl=True,
        )
"""
AR_H4B_REPL = """        ).launch(
            grid=(grid_x, self.cluster_ctas, self.roles),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster_ctas, 1),
            smem=2 * self.cluster_ctas * self.warps * 4,
            stream=stream,
            use_pdl=True,
        )
"""

AR_H5A_ANCHOR = """        cute.arch.griddepcontrol_wait()
        token = token_cta
        while token < m:
"""
AR_H5A_REPL = """        cute.arch.griddepcontrol_wait()
        token = token_cta
        parity = Int32(0)
        while token < m:
"""

AR_H5B_ANCHOR = """                m,
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
AR_H5B_REPL = """                m,
                epsilon,
                token,
                parity,
                token_cta,
                cta_y,
                logical_role,
                cluster_rank,
                tidx,
                expert_weights,
                expanded_idx_to_permuted_idx,
            )
            token = token + self.token_ctas
            parity = parity ^ Int32(1)
"""

AR_H6_ANCHOR = """        m: Int32,
        epsilon: Float32,
        token: Int32,
        token_cta: Int32,
"""
AR_H6_REPL = """        m: Int32,
        epsilon: Float32,
        token: Int32,
        parity: Int32,
        token_cta: Int32,
"""

AR_H7_ANCHOR = """            if cutlass.const_expr(self.top_k > 0):
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
"""
AR_H7_REPL = """            if cutlass.const_expr(self.top_k > 0):
                # finalize topk reduction
                if cutlass.const_expr(
                    self.top_k == 16
                    and self.latent_dim == 3584
                    and expert_weights.element_type == BFloat16
                ):
                    gemm2_vector = cute.make_ptr(
                        BFloat16,
                        (
                            latent_source.iterator + Int64(packed_idx) * VEC_BF16
                        ).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    route_indices = cute.make_ptr(
                        Int32,
                        (
                            expanded_idx_to_permuted_idx.iterator
                            + Int64(token) * self.top_k
                        ).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    route_weights = cute.make_ptr(
                        BFloat16,
                        (expert_weights.iterator + Int64(token) * self.top_k).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    local_packed = sanitize_negative_zero(
                        finalize_top16_bf16(
                            gemm2_vector,
                            route_indices,
                            route_weights,
                        )
                    )
                else:
                    local_values = cute.make_rmem_tensor(
                        cute.make_layout((VEC_BF16,)), Float32
                    )
                    for element in cutlass.range_constexpr(VEC_BF16):
                        local_values[element] = Float32(0.0)
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
                                local_values[element] = (
                                    local_values[element]
                                    + values[element].to(Float32) * weight
                                )
                    local_packed = sanitize_negative_zero(
                        bf16x8_to_packed_u32x4(local_values.load().to(BFloat16))
                    )
"""

AR_H8_ANCHOR = """            store_global_u32x4(
                latent_multicast_ptr + multicast_offset,
                local_packed,
                volatile=False,
            )

            cute.arch.cluster_arrive()
            if cluster_rank == 0 and tidx < 32:
                cute.arch.cluster_wait()
                if tidx == 0:
                    red_async_release_gpu_add_u32(latent_flags.iterator + 8, Uint32(1))
"""
AR_H8_REPL = """            stmc_bf16x8(
                latent_multicast_ptr + multicast_offset,
                local_packed,
            )
            cute.arch.griddepcontrol_launch_dependents()

            if cutlass.const_expr(not self.single_token_geometry):
                cute.arch.cluster_arrive()
                if cluster_rank == 0 and tidx < 32:
                    cute.arch.cluster_wait()
                    if tidx == 0:
                        red_async_release_gpu_add_u32(
                            latent_flags.iterator + 8, Uint32(1)
                        )
"""

AR_H9_ANCHOR = """            # Preserve the original early PDL point before RMSNorm.
            cute.arch.griddepcontrol_launch_dependents()

            if cutlass.const_expr(self.fp32_internal):
"""
AR_H9_REPL = """            if cutlass.const_expr(self.fp32_internal):
"""

AR_H10_ANCHOR = """            smem = cutlass.utils.SmemAllocator()
            warp_sums = smem.allocate_tensor(
                Float32, cute.make_layout((self.warps,)), byte_alignment=4
            )
            cluster_sums = smem.allocate_tensor(
                Float32,
                cute.make_layout((self.cluster_ctas,)),
                byte_alignment=4,
            )
            block_sum = block_sum_specialized(
                thread_sum,
                warp_sums,
                tidx,
                self.warps,
                self.last_warp_lanes,
                self.last_warp_mask,
            )
            if tidx < self.cluster_ctas:
                local_slot = cluster_sums.iterator + cluster_rank
                remote_slot = map_shared_to_peer(local_slot, Int32(tidx))
                store_shared_cluster_f32(remote_slot, block_sum)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            full_sum = Float32(0.0)
            for peer in cutlass.range_constexpr(self.cluster_ctas):
                full_sum = full_sum + cluster_sums[peer]
"""
AR_H10_REPL = """            smem = cutlass.utils.SmemAllocator()
            cluster_sums = smem.allocate_tensor(
                Float32,
                cute.make_layout((2 * self.cluster_ctas * self.warps,)),
                byte_alignment=16,
            )
            lane = cute.arch.lane_idx()
            warp_idx = cute.arch.warp_idx()
            warp_sum = warp_sum_specialized(
                thread_sum,
                warp_idx,
                lane,
                self.warps,
                self.last_warp_lanes,
                self.last_warp_mask,
            )
            # Alternate DSM slots until the next cluster synchronization so
            # peers may safely begin publishing the following token wave.
            parity_offset = parity * Int32(self.cluster_ctas * self.warps)
            if lane < self.cluster_ctas:
                local_slot = (
                    cluster_sums.iterator
                    + parity_offset
                    + cluster_rank * self.warps
                    + warp_idx
                )
                remote_slot = map_shared_to_peer(local_slot, lane)
                store_shared_cluster_f32(remote_slot, warp_sum)
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            full_sum = Float32(0.0)
            for peer in cutlass.range_constexpr(self.cluster_ctas):
                peer_slot = cluster_sums.iterator + parity_offset + peer * self.warps
                if cutlass.const_expr(self.warps == 4):
                    sum0, sum1, sum2, sum3 = load_shared_f32x4(peer_slot)
                    full_sum = full_sum + sum0 + sum1 + sum2 + sum3
                elif cutlass.const_expr(self.warps == 2):
                    sum0, sum1 = load_shared_f32x2(peer_slot)
                    full_sum = full_sum + sum0 + sum1
                else:
                    for peer_warp in cutlass.range_constexpr(self.warps):
                        full_sum = (
                            full_sum
                            + cluster_sums[
                                parity_offset + peer * self.warps + peer_warp
                            ]
                        )
"""

AR_H11_ANCHOR = """            # The x=0 CTA rotates only after reaching its final token wave.
            # Waiting for all M arrivals then guarantees every token-wave CTA
            # loaded the current generation before the metadata is advanced.
            if (
                token_cta == 0
                and token + self.token_ctas >= m
                and cta_y == 0
                and tidx == 0
            ):
                access_counter = latent_flags.iterator + 8
                arrived = load_volatile_u32(access_counter)
                while arrived < Uint32(m):
                    arrived = load_volatile_u32(access_counter)
                next_index = (current_index + Uint32(1)) % Uint32(NUM_LAMPORT_BUFFERS)
"""
AR_H11_REPL = """            # The general schedule waits until every token cluster has loaded
            # this generation. The M=1 schedule has only this cluster, whose
            # DSM barrier above already covers all routed CTAs.
            if (
                token_cta == 0
                and token + self.token_ctas >= m
                and cta_y == 0
                and tidx == 0
            ):
                access_counter = latent_flags.iterator + 8
                if cutlass.const_expr(not self.single_token_geometry):
                    arrived = load_volatile_u32(access_counter)
                    while arrived < Uint32(m):
                        arrived = load_volatile_u32(access_counter)
                next_index = (current_index + Uint32(1)) % Uint32(NUM_LAMPORT_BUFFERS)
"""

AR_H12_ANCHOR = """            source_element = (
                Int64(token) * self.hidden_dim
                + Int64(destination) * self.shard_dim
                + Int64(tidx) * VEC_BF16
            )
            source_ptr = cute.make_ptr(
                BFloat16,
                (shared_source.iterator + source_element).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            local_packed = sanitize_negative_zero(
                load_global_u32x4(source_ptr, volatile=False)
            )
            peer_base = cute.arch.load(
                (shared_peer_ptrs.iterator + destination).llvm_ptr,
                Int64,
            )
            destination_element = current_elements + (
                (Int64(token) * self.tp_size + self.rank) * self.shard_dim
                + Int64(tidx) * VEC_BF16
            )
            store_global_u32x4(
                peer_base + destination_element * 2,
                local_packed,
                volatile=False,
            )

            # One arrival per shared destination group and token.
            cute.arch.cluster_arrive()
"""
AR_H12_REPL = """            for destination_round in cutlass.range_constexpr(
                self.shared_destinations_per_cta
            ):
                round_destination = destination + Int32(
                    destination_round * self.shared_destination_stride
                )
                if round_destination < Int32(self.tp_size):
                    peer_base = cute.arch.load(
                        (shared_peer_ptrs.iterator + round_destination).llvm_ptr,
                        Int64,
                    )
                    vector = tidx
                    while vector < self.shard_vectors:
                        source_element = (
                            Int64(token) * self.hidden_dim
                            + Int64(round_destination) * self.shard_dim
                            + Int64(vector) * VEC_BF16
                        )
                        source_ptr = cute.make_ptr(
                            BFloat16,
                            (shared_source.iterator + source_element).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        local_packed = sanitize_negative_zero(
                            load_global_u32x4(source_ptr, volatile=False)
                        )
                        destination_element = current_elements + (
                            (Int64(token) * self.tp_size + self.rank) * self.shard_dim
                            + Int64(vector) * VEC_BF16
                        )
                        store_global_u32x4(
                            peer_base + destination_element * 2,
                            local_packed,
                            volatile=False,
                        )
                        vector = vector + self.threads

            cute.arch.griddepcontrol_launch_dependents()

            # One arrival per shared cluster and token.
            cute.arch.cluster_arrive()
"""

AR_H13_ANCHOR = """            global_tid = (
                Int64(token) * self.tp_size + Int64(destination)
            ) * self.threads + Int64(tidx)
            total_threads = Int64(m) * self.tp_size * self.threads
            clear_fragments = (Int64(bytes_to_clear) + PACKED_BYTES - 1) // PACKED_BYTES
            clear_idx = global_tid
            if dirty_num_stages > Uint32(0):
                while clear_idx < clear_fragments:
                    clear_ptr = cute.make_ptr(
                        BFloat16,
                        (
                            shared_workspace.iterator
                            + dirty_elements
                            + clear_idx * VEC_BF16
                        ).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    store_lamport_sentinel_128(clear_ptr)
                    clear_idx = clear_idx + total_threads

            if destination == self.rank:
                rank_words = cute.make_rmem_tensor(
                    cute.make_layout((self.tp_size, 4), stride=(4, 1)), Uint32
                )
                valid = False
                while not valid:
                    valid = True
                    for source_rank in cutlass.range_constexpr(self.tp_size):
                        remote_element = current_elements + (
                            (Int64(token) * self.tp_size + source_rank) * self.shard_dim
                            + Int64(tidx) * VEC_BF16
                        )
                        remote_ptr = cute.make_ptr(
                            BFloat16,
                            (shared_workspace.iterator + remote_element).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        remote = load_global_u32x4(remote_ptr, volatile=True)
                        for word in cutlass.range_constexpr(4):
                            rank_words[source_rank, word] = remote[word]
                        valid = valid & (not fragment_is_dirty(remote))

                accum = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
                for element in cutlass.range_constexpr(VEC_BF16):
                    accum[element] = Float32(0.0)
                for source_rank in cutlass.range_constexpr(self.tp_size):
                    values = packed_u32x4_to_bf16x8(
                        rank_words[source_rank, None].load()
                    ).to(Float32)
                    for element in cutlass.range_constexpr(VEC_BF16):
                        accum[element] = accum[element] + values[element]
                result = accum.load().to(BFloat16)
                output_element = (
                    Int64(token) * self.hidden_dim
                    + self.rank * self.shard_dim
                    + Int64(tidx) * VEC_BF16
                )
                store_global_u32x4(
                    Int64((shared_output.iterator + output_element).toint()),
                    bf16x8_to_packed_u32x4(result),
                    volatile=False,
                )

            if destination == self.rank:
                cute.arch.barrier()

            cute.arch.griddepcontrol_launch_dependents()
"""
AR_H13_REPL = """            total_threads = Int64(m) * self.tp_size * self.threads
            clear_fragments = (Int64(bytes_to_clear) + PACKED_BYTES - 1) // PACKED_BYTES

            for destination_round in cutlass.range_constexpr(
                self.shared_destinations_per_cta
            ):
                round_destination = destination + Int32(
                    destination_round * self.shared_destination_stride
                )
                global_tid = (
                    Int64(token) * self.tp_size + Int64(round_destination)
                ) * self.threads + Int64(tidx)
                clear_idx = global_tid
                if round_destination < Int32(
                    self.tp_size
                ) and dirty_num_stages > Uint32(0):
                    while clear_idx < clear_fragments:
                        clear_ptr = cute.make_ptr(
                            BFloat16,
                            (
                                shared_workspace.iterator
                                + dirty_elements
                                + clear_idx * VEC_BF16
                            ).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        store_lamport_sentinel_128(clear_ptr)
                        clear_idx = clear_idx + total_threads

                if round_destination == self.rank:
                    vector = tidx
                    while vector < self.shard_vectors:
                        rank_words = cute.make_rmem_tensor(
                            cute.make_layout((self.tp_size, 4), stride=(4, 1)),
                            Uint32,
                        )
                        valid = False
                        while not valid:
                            valid = True
                            for source_rank in cutlass.range_constexpr(self.tp_size):
                                remote_element = current_elements + (
                                    (Int64(token) * self.tp_size + source_rank)
                                    * self.shard_dim
                                    + Int64(vector) * VEC_BF16
                                )
                                remote_ptr = cute.make_ptr(
                                    BFloat16,
                                    (
                                        shared_workspace.iterator + remote_element
                                    ).llvm_ptr,
                                    cute.AddressSpace.gmem,
                                    assumed_align=16,
                                )
                                remote = load_global_u32x4(remote_ptr, volatile=True)
                                for word in cutlass.range_constexpr(4):
                                    rank_words[source_rank, word] = remote[word]
                                valid = valid & (not fragment_is_dirty(remote))

                        accum = cute.make_rmem_tensor(
                            cute.make_layout((VEC_BF16,)), Float32
                        )
                        for element in cutlass.range_constexpr(VEC_BF16):
                            accum[element] = Float32(0.0)
                        for source_rank in cutlass.range_constexpr(self.tp_size):
                            values = packed_u32x4_to_bf16x8(
                                rank_words[source_rank, None].load()
                            ).to(Float32)
                            for element in cutlass.range_constexpr(VEC_BF16):
                                accum[element] = accum[element] + values[element]
                        result = accum.load().to(BFloat16)
                        output_element = (
                            Int64(token) * self.hidden_dim
                            + self.rank * self.shard_dim
                            + Int64(vector) * VEC_BF16
                        )
                        store_global_u32x4(
                            Int64((shared_output.iterator + output_element).toint()),
                            bf16x8_to_packed_u32x4(result),
                            volatile=False,
                        )
                        vector = vector + self.threads

                    cute.arch.barrier()
"""

AR_H14_ANCHOR = """                target = Uint32(m) * Uint32(self.tp_size // self.cluster_ctas)
"""
AR_H14_REPL = """                target = Uint32(m) * Uint32(self.shared_roles)
"""

AR_H15_ANCHOR = """        self._dummy_expanded_idx = torch.empty(
            (max_m, max(top_k, 1)), dtype=torch.int32, device=device
        )

        bytes_per_routed_buffer = max_m * tp_size * latent_dim * 2
"""
AR_H15_REPL = """        self._dummy_expanded_idx = torch.empty(
            (max_m, max(top_k, 1)), dtype=torch.int32, device=device
        )
        self._seven_cta_max_m = (
            min(max_m, _SEVEN_CTA_MAX_M)
            if (tp_size, latent_dim, hidden_dim) == (8, 3584, 7168)
            and top_k == 16
            and self._dummy_expert_weights.dtype == torch.bfloat16
            and torch.cuda.get_device_capability(device)[0] == 10
            else 0
        )

        bytes_per_routed_buffer = max_m * tp_size * latent_dim * 2
"""

AR_H16_ANCHOR = """        torch.accelerator.synchronize(device)
        dist.barrier(group=group, device_ids=[device.index])
        for owner in range(tp_size):
"""
AR_H16_REPL = """        torch.accelerator.synchronize(device)
        dist.barrier(group=group, device_ids=[device.index])
        if self._seven_cta_max_m:
            specializations = (
                (1, 1),
                (self._seven_cta_max_m, self._seven_cta_max_m),
            )
            for owner in range(tp_size):
                if rank == owner:
                    for compile_max_m, compile_token_ctas in specializations:
                        if (compile_max_m, compile_token_ctas) == (
                            max_m,
                            max_token_ctas,
                        ):
                            continue
                        compile_kernel(
                            rank=rank,
                            tp_size=tp_size,
                            latent_dim=latent_dim,
                            hidden_dim=hidden_dim,
                            max_m=compile_max_m,
                            max_token_ctas=compile_token_ctas,
                            latent_output=self._latent_output,
                            routed_workspace=self._routed_workspace,
                            routed_flags=self._routed_flags,
                            routed_multicast_ptr=self._routed_multicast_ptr,
                            shared_output=self._shared_output,
                            shared_workspace=self._shared_workspace,
                            shared_flags=self._shared_flags,
                            shared_peer_ptrs=self._shared_peer_ptrs,
                            rms_eps=self.rms_eps,
                            fp32_internal=fp32_internal,
                            top_k=top_k,
                        )
                dist.barrier(group=group, device_ids=[device.index])
        for owner in range(tp_size):
"""

AR_H17A_ANCHOR = """        if not 1 <= m <= self.max_m:
            raise ValueError(f"runtime M={m} must be in [1, {self.max_m}]")

        with torch.accelerator.device_index(device.index):
"""
AR_H17A_REPL = """        if not 1 <= m <= self.max_m:
            raise ValueError(f"runtime M={m} must be in [1, {self.max_m}]")

        launch_max_m = self.max_m
        launch_token_ctas = self.max_token_ctas
        if self._seven_cta_max_m and m <= self._seven_cta_max_m:
            if m == 1:
                launch_max_m = 1
                launch_token_ctas = 1
            else:
                launch_max_m = self._seven_cta_max_m
                launch_token_ctas = self._seven_cta_max_m
        with torch.accelerator.device_index(device.index):
"""

AR_H17B_ANCHOR = """                max_m=self.max_m,
                max_token_ctas=self.max_token_ctas,
                fp32_internal=self.fp32_internal,
                expert_weights=expert_weights,
"""
AR_H17B_REPL = """                max_m=launch_max_m,
                max_token_ctas=launch_token_ctas,
                fp32_internal=self.fp32_internal,
                expert_weights=expert_weights,
"""

# --- fused_add_multicast_skinny_gemm.py hunks -------------------------------

FG_H1_ANCHOR = """def config_for_m(num_rows: int, shard_dim: int = 896) -> SkinnyConfig:
    if shard_dim == 448:
"""
FG_H1_REPL = """def config_for_m(num_rows: int, shard_dim: int = 896) -> SkinnyConfig:
    if num_rows <= 5 and shard_dim in (448, 896):  # {MARKER}
        return SkinnyConfig(
            block_size=224,
            outputs_per_block=2,
            k_unroll=1,
            vector_width=16,
        )
    if shard_dim == 448:
""".replace("{MARKER}", MARKER)

FG_H2_ANCHOR = """def _as_cute(tensor: torch.Tensor):
    return from_dlpack(
        CUDAGraphCompatibleWrapper(tensor.detach()),
        assumed_align=16,
    )
"""
FG_H2_REPL = """def _as_cute(tensor: torch.Tensor):
    return from_dlpack(
        CUDAGraphCompatibleWrapper(tensor.detach()),
        assumed_align=32,
    )
"""

FG_H3A_ANCHOR = """                    acc[mi, ni] = acc[mi, ni] + a_regs[mi, vi].to(Float32) * b_regs[
                        ni, vi
                    ].to(Float32)
"""
FG_H3A_REPL = """                    acc[mi, ni] = fma_f32_bf16(
                        a_regs[mi, vi],
                        b_regs[ni, vi],
                        acc[mi, ni],
                    )
"""

FG_H3B_ANCHOR = """                        acc[mi, ni] = acc[mi, ni] + a_regs[mi, vi].to(Float32) * b_regs[
                            ni, vi
                        ].to(Float32)
"""
FG_H3B_REPL = """                        acc[mi, ni] = fma_f32_bf16(
                            a_regs[mi, vi],
                            b_regs[ni, vi],
                            acc[mi, ni],
                        )
"""

FG_H4_ANCHOR = """    a = make_fake_tensor(
        BFloat16,
        (num_rows, latent_dim),
        stride=(latent_dim, 1),
        assumed_align=16,
    )
    b = make_fake_tensor(
        BFloat16,
        (shard_dim, latent_dim),
        stride=(latent_dim, 1),
        assumed_align=16,
    )
    shared = make_fake_tensor(
        BFloat16,
        (num_rows, shard_dim),
        stride=(hidden_dim, 1),
        assumed_align=16,
    )
"""
FG_H4_REPL = """    a = make_fake_tensor(
        BFloat16,
        (num_rows, latent_dim),
        stride=(latent_dim, 1),
        assumed_align=32,
    )
    b = make_fake_tensor(
        BFloat16,
        (shard_dim, latent_dim),
        stride=(latent_dim, 1),
        assumed_align=32,
    )
    shared = make_fake_tensor(
        BFloat16,
        (num_rows, shard_dim),
        stride=(hidden_dim, 1),
        assumed_align=32,
    )
"""

FG_H5_ANCHOR = """            raise ValueError("skinny up-projection inputs have unsupported strides")
        if mailbox.shape[1] < self.num_rows:
"""
FG_H5_REPL = """            raise ValueError("skinny up-projection inputs have unsupported strides")
        if any(tensor.data_ptr() % 32 for tensor in (latent, weight, shared_shard)):
            raise ValueError("skinny up-projection inputs must be 32-byte aligned")
        if mailbox.shape[1] < self.num_rows:
"""

FG_H6_ANCHOR = """from .primitives import (
    CUDAGraphCompatibleWrapper,
    bf16x2_to_u32,
    bf16x4_to_packed_u32x2,
    bf16x8_to_packed_u32x4,
    sanitize_negative_zero,
"""
FG_H6_REPL = """from .primitives import (
    CUDAGraphCompatibleWrapper,
    bf16x2_to_u32,
    bf16x4_to_packed_u32x2,
    bf16x8_to_packed_u32x4,
    fma_f32_bf16,  # {MARKER}
    sanitize_negative_zero,
"""

# --- lamport_copy.py hunks ---------------------------------------------------

LC_H1_ANCHOR = """        m: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(symmetric_mailbox, local_output, m).launch(
            grid=(self.ctas, 1, 1),
"""
LC_H1_REPL = """        m: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        fragments = m * cutlass.Int32(self.hidden_dim // VEC_BF16)
        grid_ctas = cutlass.min(
            cutlass.Int32(self.ctas),
            cute.ceil_div(fragments, self.threads),
        )
        self.kernel(symmetric_mailbox, local_output, m).launch(
            grid=(grid_ctas, 1, 1),
"""

LC_H2_ANCHOR = """        # The CTA may be scheduled early, but mailbox inspection must not pass
        # the producer GEMM's programmatic completion point.
        cute.arch.griddepcontrol_wait()

        tidx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        thread = cutlass.Int64(block * self.threads + tidx)
        stride = cutlass.Int64(self.ctas * self.threads)
        fragments = cutlass.Int64(m) * cutlass.Int64(self.hidden_dim // VEC_BF16)

        fragment = thread
        while fragment < fragments:
            element = fragment * VEC_BF16
            source = cute.make_ptr(
                cutlass.BFloat16,
                (symmetric_mailbox.iterator + element).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            packed = load_global_u32x4(source, volatile=True)
            while fragment_is_dirty(packed):
                packed = load_global_u32x4(source, volatile=True)

            destination = cutlass.Int64((local_output.iterator + element).toint())
            store_global_u32x4(destination, packed, volatile=False)
            fragment = fragment + stride

        # The returned ordinary tensor is complete. A same-stream successor
        # may overlap the mailbox cleanup below.
        cute.arch.griddepcontrol_launch_dependents()

        fragment = thread
        while fragment < fragments:
            element = fragment * VEC_BF16
            source = cute.make_ptr(
                cutlass.BFloat16,
                (symmetric_mailbox.iterator + element).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            store_lamport_sentinel_128(source)
            fragment = fragment + stride
"""
LC_H2_REPL = """        # {MARKER}: the Lamport marker carries producer readiness, so polling
        # can begin before the producer grid reaches ordinary completion.
        cute.arch.griddepcontrol_launch_dependents()

        tidx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()
        thread = cutlass.Int64(block * self.threads + tidx)
        stride = cutlass.Int64(grid_x * self.threads)
        fragments = cutlass.Int64(m) * cutlass.Int64(self.hidden_dim // VEC_BF16)

        fragment = thread
        while fragment < fragments:
            element = fragment * VEC_BF16
            source = cute.make_ptr(
                cutlass.BFloat16,
                (symmetric_mailbox.iterator + element).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            packed = load_global_u32x4(source, volatile=True)
            while fragment_is_dirty(packed):
                packed = load_global_u32x4(source, volatile=True)

            destination = cutlass.Int64((local_output.iterator + element).toint())
            store_global_u32x4(destination, packed, volatile=False)
            store_lamport_sentinel_128(source)
            fragment = fragment + stride
""".replace("{MARKER}", MARKER)

# --- latent_moe_tail.py hunk -------------------------------------------------

LT_H1_ANCHOR = """_LAMPORT_COPY_THREADS = 224
"""
LT_H1_REPL = """_LAMPORT_COPY_THREADS = 128  # {MARKER}
""".replace("{MARKER}", MARKER)


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


def patch_file(vroot, relpath, hunks):
    """hunks: (anchor, replacement, description[, expected_count])."""
    path = vroot + "/" + relpath
    src = load(path)
    if src is None:
        return False
    if MARKER in src:
        print(f"[{SCRIPT_NAME}] SKIP  {relpath} (already present)")
        return True
    ok = True
    for hunk in hunks:
        anchor, repl, desc = hunk[0], hunk[1], hunk[2]
        want = hunk[3] if len(hunk) > 3 else 1
        n = src.count(anchor)
        if n != want:
            print(
                f"[{SCRIPT_NAME}] NOTE  {relpath}: anchor for '{desc}' found "
                f"{n}x (want {want}); file skipped — stays stock."
            )
            ok = False
        else:
            src = src.replace(anchor, repl)
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

    # PREREQUISITE: pr53152 must already be applied (upstream dependency —
    # this PR's finalize/geometry hunks build on the top_k block).
    ar_src = load(vroot + "/" + AR)
    if ar_src is None:
        return 1
    if PREREQ_MARKER not in ar_src:
        print(
            f"[{SCRIPT_NAME}] PREREQUISITE FAILED: pr53152 marker not found in "
            f"{AR}. Upstream #54168 builds on #53152's top_k finalize block — "
            f"apply mods/k3-tier1/pr53152 first."
        )
        return 1

    ok = patch_file(
        vroot,
        PRIM,
        [
            (PR_H1_ANCHOR, PR_H1_REPL, "fma_f32_bf16"),
            (PR_H2_ANCHOR, PR_H2_REPL, "finalize_top16_bf16 + asm builder"),
            (PR_H3_ANCHOR, PR_H3_REPL, "stmc_bf16x8"),
            (PR_H4_ANCHOR, PR_H4_REPL, "load_shared_f32x2/x4"),
        ],
    )
    ok &= patch_file(
        vroot,
        AR,
        [
            (AR_H1_ANCHOR, AR_H1_REPL, "imports + _SEVEN_CTA_MAX_M"),
            (AR_H2_ANCHOR, AR_H2_REPL, "seven-CTA geometry + shared roles"),
            (AR_H3_ANCHOR, AR_H3_REPL, "roles from self.shared_roles"),
            (AR_H4A_ANCHOR, AR_H4A_REPL, "__call__ grid_x"),
            (AR_H4B_ANCHOR, AR_H4B_REPL, "__call__ grid/smem"),
            (AR_H5A_ANCHOR, AR_H5A_REPL, "kernel parity init"),
            (AR_H5B_ANCHOR, AR_H5B_REPL, "kernel parity arg + flip"),
            (AR_H6_ANCHOR, AR_H6_REPL, "_token_device parity param"),
            (AR_H7_ANCHOR, AR_H7_REPL, "top_k==16 bf16 finalize fast path"),
            (AR_H8_ANCHOR, AR_H8_REPL, "stmc multicast store + early PDL + arrival gate"),
            (AR_H9_ANCHOR, AR_H9_REPL, "remove old pre-RMSNorm PDL point"),
            (AR_H10_ANCHOR, AR_H10_REPL, "parity DSM warp-sum reduction"),
            (AR_H11_ANCHOR, AR_H11_REPL, "single-token Lamport arrival gate"),
            (AR_H12_ANCHOR, AR_H12_REPL, "shared ReduceScatter destination rounds"),
            (AR_H13_ANCHOR, AR_H13_REPL, "shared clear/rank-reduce destination rounds"),
            (AR_H14_ANCHOR, AR_H14_REPL, "shared arrival target = m * shared_roles"),
            (AR_H15_ANCHOR, AR_H15_REPL, "_seven_cta_max_m"),
            (AR_H16_ANCHOR, AR_H16_REPL, "seven-CTA warmup specializations"),
            (AR_H17A_ANCHOR, AR_H17A_REPL, "__call__ launch_max_m selection"),
            (AR_H17B_ANCHOR, AR_H17B_REPL, "__call__ launch max_m/token_ctas"),
        ],
    )
    ok &= patch_file(
        vroot,
        FG,
        [
            (FG_H6_ANCHOR, FG_H6_REPL, "fma_f32_bf16 import"),
            (FG_H1_ANCHOR, FG_H1_REPL, "low-M vector_width=16 config"),
            (FG_H2_ANCHOR, FG_H2_REPL, "_as_cute 32-byte align"),
            (FG_H3A_ANCHOR, FG_H3A_REPL, "fma accumulation (mainloop)"),
            (FG_H3B_ANCHOR, FG_H3B_REPL, "fma accumulation (k-tile loop)"),
            (FG_H4_ANCHOR, FG_H4_REPL, "compile fake tensors 32-byte align"),
            (FG_H5_ANCHOR, FG_H5_REPL, "32-byte alignment validation"),
        ],
    )
    ok &= patch_file(
        vroot,
        LC,
        [
            (LC_H1_ANCHOR, LC_H1_REPL, "grid trimmed to fragment count"),
            (LC_H2_ANCHOR, LC_H2_REPL, "fused copy+cleanup, early launch_dependents"),
        ],
    )
    ok &= patch_file(
        vroot,
        TAIL,
        [(LT_H1_ANCHOR, LT_H1_REPL, "_LAMPORT_COPY_THREADS 224 -> 128")],
    )
    if ok:
        print(
            f"[{SCRIPT_NAME}] done. Low-M tail: 7-CTA/64-thread geometry with "
            f"bf16 top-16 finalize (TP8 3584/7168), parity DSM slots, NVLS "
            f"multicast stores, compact ReduceScatter, faster lamport copy."
        )
    else:
        print(f"[{SCRIPT_NAME}] NOTE: some hunks skipped — see above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
