"""Standalone BF16→FP4 TMA quantize kernel. 2-stage pipeline, warp-split."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch, cutlass, cutlass.cute as cute, cutlass.pipeline as pipeline
import cutlass.utils as utils, cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils.blackwell_helpers as sm120_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cuda.bindings.driver as cuda
from cutlass.cutlass_dsl import Int32, Uint8, Uint32, Uint64, Float32, T, dsl_user_op
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync

@dsl_user_op
def fabs_f32(a, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip)], "abs.f32 $0, $1;", "=f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))
@dsl_user_op
def fmax_f32(a, b, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)], "max.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))
@dsl_user_op
def fmin_f32(a, b, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)], "min.f32 $0, $1, $2;", "=f,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))
@dsl_user_op
def rcp_approx_ftz(a, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip)], "rcp.approx.ftz.f32 $0, $1;", "=f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))
@dsl_user_op
def cvt_f32_to_e4m3(a, *, loc=None, ip=None):
    return Uint32(llvm.inline_asm(T.i32(), [Float32(a).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 fp8_pair;\n.reg .f32 zero;\nmov.f32 zero, 0f00000000;\ncvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $1;\ncvt.u32.u16 $0, fp8_pair;\n}",
        "=r,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))
@dsl_user_op
def fp8_e4m3_to_f32_and_rcp(fp8_val, *, loc=None, ip=None):
    return Float32(llvm.inline_asm(T.f32(), [Uint32(fp8_val).ir_value(loc=loc, ip=ip)],
        "{\n.reg .pred p_zero;\n.reg .u32 exp_u, mant_u;\n.reg .s32 exp_s;\n.reg .f32 exp_f, mant_f, fp8_float, result;\n"
        "setp.eq.u32 p_zero, $1, 0;\nand.b32 mant_u, $1, 7;\nshr.b32 exp_u, $1, 3;\nand.b32 exp_u, exp_u, 15;\n"
        "sub.s32 exp_s, exp_u, 7;\ncvt.rn.f32.s32 exp_f, exp_s;\nex2.approx.f32 exp_f, exp_f;\n"
        "cvt.rn.f32.u32 mant_f, mant_u;\nfma.rn.f32 mant_f, mant_f, 0f3E000000, 0f3F800000;\n"
        "mul.f32 fp8_float, exp_f, mant_f;\nrcp.approx.ftz.f32 result, fp8_float;\n"
        "selp.f32 $0, 0f00000000, result, p_zero;\n}",
        "=f,r", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))
@dsl_user_op
def cvt_e2m1x8_f32(v0, v1, v2, v3, v4, v5, v6, v7, *, loc=None, ip=None):
    args = [Float32(v).ir_value(loc=loc, ip=ip) for v in [v0, v1, v2, v3, v4, v5, v6, v7]]
    return Uint32(llvm.inline_asm(T.i32(), args,
        "{\n.reg .b8 byte0, byte1, byte2, byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;\ncvt.rn.satfinite.e2m1x2.f32 byte1, $4, $3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte2, $6, $5;\ncvt.rn.satfinite.e2m1x2.f32 byte3, $8, $7;\n"
        "mov.b32 $0, {byte0, byte1, byte2, byte3};\n}",
        "=r,f,f,f,f,f,f,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))
@dsl_user_op
def sm120_make_smem_layout_sfa(tiled_mma, tile_shape_mnk, sf_vec_size, num_stages, *, loc=None, ip=None):
    blk_mn, blk_sf, blk_elems = 128, 4, 128 * 4
    mma_nsf = tiled_mma.shape_mnk[2] // sf_vec_size
    mn_bs, mn_st = (32, 4), (16, 4)
    k_bs, k_st = (sf_vec_size, mma_nsf), (0, 1)
    sM, stM = (mn_bs, tile_shape_mnk[0] // blk_mn), (mn_st, blk_elems)
    sK = (k_bs, blk_sf // mma_nsf, tile_shape_mnk[2] // sf_vec_size // blk_sf)
    stK = (k_st, mma_nsf, tile_shape_mnk[0] // blk_mn * blk_elems)
    sl = cute.make_layout((sM, sK), stride=(stM, stK))
    return cute.append(sl, cute.make_layout(num_stages, stride=cute.cosize(cute.filter_zeros(sl))))
@dsl_user_op
def shared_ptr_to_u32(ptr, *, loc=None, ip=None):
    return Int32(llvm.ptrtoint(T.i32(), ptr.llvm_ptr, loc=loc, ip=ip))
@dsl_user_op
def st_shared_u8(smem_addr, value, *, loc=None, ip=None):
    llvm.inline_asm(None, [Int32(smem_addr).ir_value(loc=loc, ip=ip), Uint8(value).ir_value(loc=loc, ip=ip)],
        "st.shared.u8 [$0], $1;", "r,r", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT)

@cute.jit
def quantize_block_fp4_fast(values, max_abs, global_scale_val):
    scale_u32 = Uint32(0)
    scale_byte = Uint8(0)
    packed64 = Uint64(0)
    if global_scale_val != Float32(0.0):
        fp4_max_rcp = rcp_approx_ftz(Float32(6.0))
        gs_recip = rcp_approx_ftz(global_scale_val)
        scale_float = gs_recip * (max_abs * fp4_max_rcp)
        scale_float = fmin_f32(scale_float, Float32(448.0))
        scale_u32 = cvt_f32_to_e4m3(scale_float)
        scale_byte = Uint8(scale_u32 & Uint32(0xFF))
        inv_quantized_scale = fp8_e4m3_to_f32_and_rcp(scale_u32)
        if inv_quantized_scale != Float32(0.0):
            q = cute.make_rmem_tensor((16,), Float32)
            for i in cutlass.range_constexpr(16):
                q[i] = values[i] * inv_quantized_scale * gs_recip
            packed_lo = cvt_e2m1x8_f32(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])
            packed_hi = cvt_e2m1x8_f32(q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15])
            packed64 = (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)
    return packed64, scale_byte

import ctypes
from cutlass._mlir import ir
import cutlass._mlir.dialects.cute as _cute_ir
from cutlass.cute.typing import Pointer, Numeric, AddressSpace, Type

class _Pointer(Pointer):
    def __init__(self, pointer, dtype, mem_space=_cute_ir.AddressSpace.generic, assumed_align=None):
        self._pointer = pointer; self._dtype = dtype; self._addr_space = mem_space
        self._assumed_align = dtype.width // 8 if assumed_align is None else assumed_align
        self._desc = None; self._c_pointer = None
    def size_in_bytes(self): return ctypes.sizeof(ctypes.c_void_p(int(self._pointer)))
    def __get_mlir_types__(self): return [self.mlir_type]
    def __c_pointers__(self):
        if self._c_pointer is None:
            self._desc = ctypes.c_void_p(int(self._pointer)); self._c_pointer = ctypes.addressof(self._desc)
        return [self._c_pointer]
    def __new_from_mlir_values__(self, values): return values[0]
    @property
    def mlir_type(self): return _cute_ir.PtrType.get(self._dtype.mlir_type, self._addr_space, self._assumed_align)
    @property
    def dtype(self): return self._dtype
    @property
    def memspace(self): return self._addr_space
    def align(self, min_align, *, loc=None, ip=None): raise NotImplementedError
    def verify(self, e): return e is Pointer
    @property
    def __cache_key__(self): return (self._dtype, self._addr_space, self._assumed_align)

def make_ptr(dtype, value, mem_space=AddressSpace.generic, assumed_align=None):
    return _Pointer(value if isinstance(value, int) else ctypes.cast(value, ctypes.c_void_p).value,
                    dtype, mem_space, assumed_align=assumed_align)

_NUM_STAGES = 1

class TestKernel:
    def __init__(self):
        self.tile_shape_mnk = (128, 128, 128)
        self.threads_per_cta = 160
        self.num_mma_warps = 4
        self.tma_warp = 4
        self.num_stages = _NUM_STAGES
        self.load_register_requirement = 40

    @cute.jit
    def __call__(self, bf16_input: cute.Tensor, global_scale: cute.Tensor,
                 packed_a: cute.Tensor, sfa_ptr: cute.Pointer,
                 mac: cutlass.Constexpr, stream: cuda.CUstream):
        ab, bf, sf = cutlass.Float4E2M1FN, cutlass.BFloat16, cutlass.Float8E4M3FN
        fp4_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.COL_MAJOR, ab, 128), ab)
        fp4_staged = cute.tile_to_shape(fp4_atom, (128, 128, 1), order=(0, 1, 2))
        bf16_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(utils.LayoutEnum.COL_MAJOR, bf, 128), bf)
        # 2-stage BF16 smem for load/compute overlap.
        bf16_staged = cute.tile_to_shape(bf16_atom, (128, 128, self.num_stages), order=(0, 1, 2))
        mma_op = cute.nvgpu.warp.MmaMXF4NVF4Op(ab, cutlass.Float32, sf)
        perm = sm120_utils.get_permutation_mnk(self.tile_shape_mnk, 16, False)
        tiled_mma = cute.make_tiled_mma(mma_op, cute.make_layout((2,2,1)), permutation_mnk=perm)
        sfa_staged = sm120_make_smem_layout_sfa(tiled_mma, self.tile_shape_mnk, 16, 1)
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(packed_a.shape, 16)
        sfa_tensor = cute.make_tensor(sfa_ptr, sfa_layout)
        bf16_smem1 = cute.slice_(bf16_staged, (None, None, 0))
        tma_load, gInput = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), bf16_input, bf16_smem1, (128,128), num_multicast=1)
        fp4_smem1 = cute.slice_(fp4_staged, (None, None, 0))
        tile_mk = cute.slice_(self.tile_shape_mnk, (None, 0, None))
        tma_store_a, gOA = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), packed_a, fp4_smem1, tile_mk)
        sfa_smem1 = cute.slice_(sfa_staged, (None, None, 0))
        tma_store_sfa, gOSFA = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), sfa_tensor, sfa_smem1, tile_mk, internal_type=cutlass.Int16)
        self.kernel(
            tma_load, gInput, tma_store_a, gOA, tma_store_sfa, gOSFA,
            global_scale,
            bf16_staged, fp4_staged, sfa_staged,
            cute.cosize(bf16_staged), cute.cosize(fp4_staged), cute.cosize(sfa_staged),
        ).launch(grid=(mac,1,1), block=[self.threads_per_cta,1,1], cluster=[1,1,1], stream=stream)

    @cute.kernel
    def kernel(self,
               tma_load: cute.CopyAtom, mInput: cute.Tensor,
               tma_store_a: cute.CopyAtom, mOA: cute.Tensor,
               tma_store_sfa: cute.CopyAtom, mOSFA: cute.Tensor,
               global_scale: cute.Tensor,
               bf16_smem: cute.ComposedLayout, fp4_smem: cute.ComposedLayout, sfa_smem: cute.Layout,
               bf16_cs: cutlass.Constexpr, fp4_cs: cutlass.Constexpr, sfa_cs: cutlass.Constexpr):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        warp_idx = tidx // Int32(32)

        M = Int32(mInput.shape[0])
        K = Int32(mInput.shape[1])
        k_tiles = K // Int32(128)
        total_tiles = (M // Int32(128)) * k_tiles

        smem = cutlass.utils.SmemAllocator()
        @cute.struct
        class S:
            pmem: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sBF16: cute.struct.Align[cute.struct.MemRange[cutlass.BFloat16, bf16_cs], 1024]
            sFP4: cute.struct.Align[cute.struct.MemRange[cutlass.Float4E2M1FN, fp4_cs], 1024]
            sSFA: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, sfa_cs], 1024]
        st = smem.allocate(S)
        sBF16 = st.sBF16.get_tensor(bf16_smem.outer, swizzle=bf16_smem.inner)
        sFP4 = st.sFP4.get_tensor(fp4_smem.outer, swizzle=fp4_smem.inner)
        sSFA = st.sSFA.get_tensor(sfa_smem)
        sA_u8 = cute.recast_tensor(sFP4[None, None, 0], cutlass.Uint8)
        sfa_base = shared_ptr_to_u32(st.sSFA.data_ptr())
        gs_value = global_scale[Int32(0)].to(cutlass.Float32)

        cta_layout = cute.make_layout(1)
        gI = cute.local_tile(mInput, (128, 128), (None, None))
        tLsI, tLgI = cpasync.tma_partition(tma_load, 0, cta_layout,
            cute.group_modes(sBF16, 0, 2), cute.group_modes(gI, 0, 2))
        tile_mk = cute.slice_(self.tile_shape_mnk, (None, 0, None))
        gOA = cute.local_tile(mOA, tile_mk, (None, None, None))
        bSsA, bSgA = cpasync.tma_partition(tma_store_a, 0, cta_layout,
            cute.group_modes(sFP4, 0, 2), cute.group_modes(gOA, 0, 2))
        gOSFA = cute.local_tile(mOSFA, tile_mk, (None, None, None))
        bSsSFA, bSgSFA = cpasync.tma_partition(tma_store_sfa, 0, cta_layout,
            cute.group_modes(sSFA, 0, 2), cute.group_modes(gOSFA, 0, 2))

        # 2-stage load pipeline for load/compute overlap.
        cta_layout_vmnk = cute.make_layout((1, 1, 1, 1))
        load_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps),
            tx_count=128 * 128 * cutlass.BFloat16.width // 8,
            barrier_storage=st.pmem.data_ptr(),
            cta_layout_vmnk=cta_layout_vmnk)
        store_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_mma_warps * 32))

        if tidx == Int32(0):
            cpasync.prefetch_descriptor(tma_load)
            cpasync.prefetch_descriptor(tma_store_a)
            cpasync.prefetch_descriptor(tma_store_sfa)
        cute.arch.sync_threads()

        tile_idx = Int32(bidx)

        # MMA warps 0-3: consume BF16, quantize, TMA store.
        if warp_idx < Int32(self.num_mma_warps):
            cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
            while tile_idx < total_tiles:
                mt = tile_idx // k_tiles
                kt = tile_idx % k_tiles
                load_pipeline.consumer_wait(cs)
                stage = cs.index

                blk = Int32(tidx)
                while blk < Int32(128 * 8):
                    row = blk // Int32(8)
                    sf_col = blk % Int32(8)
                    col0 = sf_col * Int32(16)
                    vals = cute.make_rmem_tensor((16,), cutlass.Float32)
                    bmax = cutlass.Float32(0.0)
                    for e in cutlass.range_constexpr(16):
                        v = cutlass.Float32(sBF16[row, col0 + Int32(e), stage])
                        vals[e] = v
                        bmax = fmax_f32(bmax, fabs_f32(v))
                    p64, sbyte = quantize_block_fp4_fast(vals, bmax, gs_value)
                    pb = sf_col << Int32(3)
                    dpc = row & Int32(63)
                    xor = ((dpc >> Int32(1)) & Int32(0x3)) << Int32(4)
                    rhi = row >> Int32(6)
                    for bi in cutlass.range_constexpr(8):
                        spc = pb + Int32(bi)
                        dr = ((spc ^ xor) << Int32(1)) + rhi
                        flat = dr * Int32(64) + dpc
                        sA_u8[flat] = Uint8((p64 >> Uint64(bi * 8)) & Uint64(0xFF))
                    om = row % Int32(32)
                    im = row // Int32(32)
                    ik = sf_col % Int32(4)
                    ktile = sf_col // Int32(4)
                    sf_off = ktile * Int32(512) + om * Int32(16) + im * Int32(4) + ik
                    st_shared_u8(sfa_base + sf_off, sbyte)
                    blk += Int32(self.num_mma_warps * 32)

                load_pipeline.consumer_release(cs)
                cs.advance()

                # TMA store.
                cute.arch.fence_proxy("async.shared", space="cta")
                if warp_idx == Int32(0) and (tidx & Int32(31)) == Int32(0):
                    cute.copy(tma_store_a, bSsA[(None, Int32(0))], bSgA[(None, mt, kt, Int32(0))])
                    cute.copy(tma_store_sfa, bSsSFA[(None, Int32(0))], bSgSFA[(None, mt, kt, Int32(0))])
                    store_pipeline.producer_commit()
                    store_pipeline.producer_acquire()

                tile_idx += Int32(gdim)

        # DMA warp 4: TMA loads only.
        elif warp_idx == Int32(self.tma_warp):
            cute.arch.setmaxregister_decrease(self.load_register_requirement)
            ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
            while tile_idx < total_tiles:
                mt = tile_idx // k_tiles
                kt = tile_idx % k_tiles
                load_pipeline.producer_acquire(ps)
                cute.copy(tma_load, tLgI[(None, mt, kt)], tLsI[(None, ps.index)],
                          tma_bar_ptr=load_pipeline.producer_get_barrier(ps))
                load_pipeline.producer_commit(ps)
                ps.advance()
                tile_idx += Int32(gdim)
            load_pipeline.producer_tail(ps)


