#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PERF-PR55356-54896-MLA-CACHE-KERNELS — build-time port of upstream
vllm-project/vllm#55356 + #54896 (same MLA cache kernel family) to the
v0.26.1rc0+kimi.k3.aligned tree.

#55356 — grouped MLA context-KV cache insert gains plain-FP8 support:
  per-layer kv_scales + kv_cache_dtype plumbed through the kernel
  (template <scalar_t, cache_t, kv_dt> + fp8::scaled_convert store), the
  host launcher (bf16 fast path unchanged, FP8 via LAUNCH_GROUPED_FP8),
  ops.h, the torch_bindings schema (Tensor? kv_scales=None,
  str kv_cache_dtype='auto') and the vllm/_custom_ops.py wrapper.
  New schema args have defaults, so the existing bf16 caller
  (dspark_mla._precompute_fused_context_kv) keeps working unchanged.

#54896 — MLA decode concat/cache epilogue restructure:
  writeLatent576 generalised to (lane, lane_stride) so SPLIT warps can
  cooperate on one 576-wide row; fusedKimiK3MLADecodeQConcatKVCacheKernel
  gains a SPLIT template param (3-warp row split for <= 64 decode tokens),
  moves cudaGridDependencySynchronize out of the kernel prologue into the
  q-branch only (cache-slot warps skip the wait; non-dependent inputs are
  read before it), and the two host launchers dispatch split/unsplit via
  a new launchPdlSlots helper (upstream has it; our tree only had
  launchPdl, so it is ported here adapted to our conventions: explicit
  total-warp-count signature).

Dropped as out of scope / not-portable (see run.sh header):
  - tests/kernels/attention/test_cache.py (test file — skipped)
  - tests/kernels/attention/test_kimi_k3_mla_fused_epilogue.py (test file)
  - vllm/models/kimi_k3/nvidia/dspark_mla.py model-side fp8 wiring
    (upstream also moved _build_fused_context_kv_metadata into
    process_weights_after_loading — depends on newer upstream code shape;
    our older caller keeps the bf16 grouped path via schema defaults)

This is a BUILD-TIME patch mod: the .cu/.h/.cpp changes require the vLLM
C++ extension (.so) to be recompiled. Applying it to a source tree whose
extension is already built does NOT change the running .so. There is no
nvcc on the head node — CUDA compilation is NOT attempted or verified
here; only anchor application, idempotency and (for the .py) py_compile
are checked.

Modes:
  apply    <src_root> [--skip-python]
           patch files in place under src_root; idempotent via the marker
           string; missing files/anchors NO-OP with a loud NOTE and exit 0
           (never block boot). --skip-python leaves vllm/_custom_ops.py
           untouched (used by run.sh when the built .so has the OLD schema
           so the wrapper must not outrun the extension).
  --simulate <src_root>
           never writes the targets; applies every hunk to temp copies,
           py_compiles the .py, prints unified diffs; exits 1 if any file
           or hunk is missing (pre-flight check).

Idempotency marker: "perf-pr55356-54896-mla-cache-kernels: applied".
"""
import difflib
import py_compile
import shutil
import sys
import tempfile
from pathlib import Path

SCRIPT_NAME = "perf-pr55356-54896-mla-cache-kernels"
PR55356 = "vllm-project/vllm#55356"
PR54896 = "vllm-project/vllm#54896"
MARKER = f"{SCRIPT_NAME}: applied"


def _b(s: str) -> str:
    """Raw triple-quoted block -> exact text (strip the leading newline).

    Raw strings keep literal backslashes (needed for the LAUNCH_GROUPED_FP8
    macro lines) and never interpret escapes.
    """
    assert s.startswith("\n"), "block must start with a newline after \"\"\""
    return s[1:]


def loud(msg: str) -> None:
    print(f"=====> [{SCRIPT_NAME}] {msg}")


# ────────────────────────────────────────────────────────────────────────────
# csrc/libtorch_stable/cache_kernels.cu  (#55356)
# ────────────────────────────────────────────────────────────────────────────

CK_OLD_1 = _b(r"""
// Grouped variant of concat_and_cache_mla: inserts the context K/V for every
// draft layer in a single launch. Grid is (num_tokens, num_layers); each layer
// reads its own cache base pointer from kv_cache_ptrs (same pointer-array
// pattern as copy_blocks_kernel). bf16 only, so it is a raw 16-bit copy with no
// scaling or quantization; scalar_t is uint16_t for portability.
template <typename scalar_t>
__global__ void concat_and_cache_mla_grouped_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_layers, num_tokens,
                                        // kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_layers, num_tokens, pe_dim]
    const int64_t* __restrict__ kv_cache_ptrs,  // [num_layers]
    const int64_t* __restrict__ slot_mapping,   // [num_layers, num_tokens]
""")
CK_NEW_1 = _b(r"""
// Grouped variant of concat_and_cache_mla: inserts the context K/V for every
// draft layer in a single launch. Grid is (num_tokens, num_layers); each layer
// reads its own cache base pointer from kv_cache_ptrs (same pointer-array
// pattern as copy_blocks_kernel).
// MARKER_LINE (port of #55356)
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_mla_grouped_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_layers, num_tokens,
                                        // kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_layers, num_tokens, pe_dim]
    const int64_t* __restrict__ kv_cache_ptrs,  // [num_layers]
    const float* __restrict__ kv_scales,        // [num_layers] or nullptr
    const int64_t* __restrict__ slot_mapping,   // [num_layers, num_tokens]
""").replace("MARKER_LINE", MARKER)

CK_OLD_2 = _b(r"""
  scalar_t* __restrict__ kv_cache =
      reinterpret_cast<scalar_t*>(kv_cache_ptrs[layer_idx]);
  const scalar_t* __restrict__ kv_c_layer =
      kv_c + layer_idx * kv_c_layer_stride;
  const scalar_t* __restrict__ k_pe_layer =
      k_pe + layer_idx * k_pe_layer_stride;
""")
CK_NEW_2 = _b(r"""
  cache_t* __restrict__ kv_cache =
      reinterpret_cast<cache_t*>(kv_cache_ptrs[layer_idx]);
  const scalar_t* __restrict__ kv_c_layer =
      kv_c + layer_idx * kv_c_layer_stride;
  const scalar_t* __restrict__ k_pe_layer =
      k_pe + layer_idx * k_pe_layer_stride;
  float scale = 0.0f;
  if constexpr (kv_dt != Fp8KVCacheDataType::kAuto) {
    scale = kv_scales[layer_idx];
  }
""")

CK_OLD_3 = _b(r"""
      const int64_t dst_idx =
          block_idx * block_stride + block_offset * entry_stride + i + offset;
      kv_cache[dst_idx] = src[src_idx];
""")
CK_NEW_3 = _b(r"""
      const int64_t dst_idx =
          block_idx * block_stride + block_offset * entry_stride + i + offset;
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        kv_cache[dst_idx] = src[src_idx];
      } else {
        kv_cache[dst_idx] =
            fp8::scaled_convert<cache_t, scalar_t, kv_dt>(src[src_idx], scale);
      }
""")

CK_OLD_4 = _b(r"""
    torch::stable::Tensor& slot_mapping,   // [num_layers, num_tokens] int64
    int64_t block_size, int64_t block_stride, int64_t entry_stride) {
  int num_layers = kv_c.size(0);
  int num_tokens = kv_c.size(1);
  int kv_lora_rank = kv_c.size(2);
  int pe_dim = k_pe.size(2);

  STD_TORCH_CHECK(
      kv_c.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          k_pe.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "concat_and_cache_mla_grouped only supports a bf16 KV cache; got kv_c=",
      kv_c.scalar_type(), ", k_pe=", k_pe.scalar_type());
  STD_TORCH_CHECK(
      kv_cache_ptrs.scalar_type() == torch::headeronly::ScalarType::Long,
      "kv_cache_ptrs must be int64");

  if (num_tokens == 0 || num_layers == 0) {
""")
# Adapted from upstream #55356: kv_scales->is_cuda() replaced with
# kv_scales->device().is_cuda() (the device().is_cuda() form is what this
# older tree uses on stable tensors, e.g. fused_kimi_k3 kernel checks).
CK_NEW_4 = _b(r"""
    torch::stable::Tensor& slot_mapping,   // [num_layers, num_tokens] int64
    int64_t block_size, int64_t block_stride, int64_t entry_stride,
    std::optional<torch::stable::Tensor> kv_scales,  // [num_layers] or None
    const std::string& kv_cache_dtype) {
  const bool use_fp8 = kv_cache_dtype == "fp8" ||
                       kv_cache_dtype == "fp8_e4m3" ||
                       kv_cache_dtype == "fp8_e5m2";
#ifdef USE_ROCM
  STD_TORCH_CHECK(kv_cache_dtype != "fp8_e5m2",
                  "concat_and_cache_mla_grouped does not support fp8_e5m2 "
                  "KV cache on ROCm");
#endif
  STD_TORCH_CHECK(
      use_fp8 || kv_cache_dtype == "auto" || kv_cache_dtype == "bfloat16",
      "concat_and_cache_mla_grouped only supports BF16 and plain "
      "FP8 KV cache; got ",
      kv_cache_dtype);

  STD_TORCH_CHECK(
      kv_c.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          k_pe.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "concat_and_cache_mla_grouped requires BF16 inputs; got kv_c=",
      kv_c.scalar_type(), ", k_pe=", k_pe.scalar_type());
  STD_TORCH_CHECK(
      kv_cache_ptrs.scalar_type() == torch::headeronly::ScalarType::Long &&
          slot_mapping.scalar_type() == torch::headeronly::ScalarType::Long,
      "cache pointers and slot mapping must be int64");

  const int num_layers = kv_c.size(0);
  const int num_tokens = kv_c.size(1);
  const int kv_lora_rank = kv_c.size(2);
  const int pe_dim = k_pe.size(2);
  const float* kv_scales_ptr = nullptr;
  if (use_fp8) {
    STD_TORCH_CHECK(kv_scales.has_value(),
                    "FP8 grouped cache insert requires kv_scales");
    STD_TORCH_CHECK(
        kv_scales->scalar_type() == torch::headeronly::ScalarType::Float,
        "kv_scales must be float32");
    STD_TORCH_CHECK(kv_scales->numel() == num_layers,
                    "kv_scales must contain one scale per layer");
    STD_TORCH_CHECK(kv_scales->device().is_cuda() &&
                        kv_scales->get_device_index() ==
                            kv_c.get_device_index(),
                    "kv_scales must be on the same CUDA device as kv_c");
    STD_TORCH_CHECK(kv_scales->is_contiguous(), "kv_scales must be contiguous");
    kv_scales_ptr = kv_scales->const_data_ptr<float>();
  } else {
    STD_TORCH_CHECK(!kv_scales.has_value(),
                    "BF16 grouped cache insert does not use kv_scales");
  }

  if (num_tokens == 0 || num_layers == 0) {
""")

CK_OLD_5 = _b(r"""
  dim3 grid(num_tokens, num_layers);
  dim3 block(std::min(kv_lora_rank, 512));
  vllm::concat_and_cache_mla_grouped_kernel<uint16_t>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<const uint16_t*>(kv_c.data_ptr()),
          reinterpret_cast<const uint16_t*>(k_pe.data_ptr()),
          kv_cache_ptrs.const_data_ptr<int64_t>(),
          slot_mapping.const_data_ptr<int64_t>(), kv_c_layer_stride,
          kv_c_token_stride, k_pe_layer_stride, k_pe_token_stride,
          slot_layer_stride, block_stride, entry_stride, kv_lora_rank, pe_dim,
          block_size);
""")
CK_NEW_5 = _b(r"""
  const dim3 grid(num_tokens, num_layers);
  const dim3 block(std::min(kv_lora_rank, 512));

  if (!use_fp8) {
    vllm::concat_and_cache_mla_grouped_kernel<uint16_t, uint16_t,
                                              vllm::Fp8KVCacheDataType::kAuto>
        <<<grid, block, 0, stream>>>(
            reinterpret_cast<const uint16_t*>(kv_c.data_ptr()),
            reinterpret_cast<const uint16_t*>(k_pe.data_ptr()),
            kv_cache_ptrs.const_data_ptr<int64_t>(), nullptr,
            slot_mapping.const_data_ptr<int64_t>(), kv_c_layer_stride,
            kv_c_token_stride, k_pe_layer_stride, k_pe_token_stride,
            slot_layer_stride, block_stride, entry_stride, kv_lora_rank, pe_dim,
            block_size);
    return;
  }

#define LAUNCH_GROUPED_FP8(KV_DTYPE)                                           \
  vllm::concat_and_cache_mla_grouped_kernel<__nv_bfloat16, uint8_t, KV_DTYPE>  \
      <<<grid, block, 0, stream>>>(                                            \
          reinterpret_cast<const __nv_bfloat16*>(kv_c.data_ptr()),             \
          reinterpret_cast<const __nv_bfloat16*>(k_pe.data_ptr()),             \
          kv_cache_ptrs.const_data_ptr<int64_t>(), kv_scales_ptr,              \
          slot_mapping.const_data_ptr<int64_t>(), kv_c_layer_stride,           \
          kv_c_token_stride, k_pe_layer_stride, k_pe_token_stride,             \
          slot_layer_stride, block_stride, entry_stride, kv_lora_rank, pe_dim, \
          block_size)

  if (kv_cache_dtype == "fp8_e5m2") {
    LAUNCH_GROUPED_FP8(vllm::Fp8KVCacheDataType::kFp8E5M2);
  } else {
    LAUNCH_GROUPED_FP8(vllm::Fp8KVCacheDataType::kFp8E4M3);
  }
#undef LAUNCH_GROUPED_FP8
""")

# ────────────────────────────────────────────────────────────────────────────
# csrc/libtorch_stable/ops.h  (#55356)
# ────────────────────────────────────────────────────────────────────────────

OPS_OLD_1 = _b(r"""
void concat_and_cache_mla_grouped(torch::stable::Tensor& kv_c,
                                  torch::stable::Tensor& k_pe,
                                  torch::stable::Tensor& kv_cache_ptrs,
                                  torch::stable::Tensor& slot_mapping,
                                  int64_t block_size, int64_t block_stride,
                                  int64_t entry_stride);
""")
OPS_NEW_1 = _b(r"""
// MARKER_LINE (port of #55356)
void concat_and_cache_mla_grouped(
    torch::stable::Tensor& kv_c, torch::stable::Tensor& k_pe,
    torch::stable::Tensor& kv_cache_ptrs, torch::stable::Tensor& slot_mapping,
    int64_t block_size, int64_t block_stride, int64_t entry_stride,
    std::optional<torch::stable::Tensor> kv_scales,
    const std::string& kv_cache_dtype);
""").replace("MARKER_LINE", MARKER)

# ────────────────────────────────────────────────────────────────────────────
# csrc/libtorch_stable/torch_bindings.cpp  (#55356)
# ────────────────────────────────────────────────────────────────────────────

TB_OLD_1 = _b(r"""
  // Grouped concat_and_cache_mla across all layers (bf16 only). Each
  // layer's cache base pointer is read from kv_cache_ptrs.
  ops.def(
      "concat_and_cache_mla_grouped(Tensor kv_c, Tensor k_pe,"
      "                             Tensor kv_cache_ptrs,"
      "                             Tensor slot_mapping,"
      "                             int block_size, int block_stride,"
      "                             int entry_stride) -> ()");
""")
TB_NEW_1 = _b(r"""
  // Grouped concat_and_cache_mla across all layers. Each layer's cache base
  // pointer and optional plain-FP8 scale are read from device tensors.
  // MARKER_LINE (port of #55356)
  ops.def(
      "concat_and_cache_mla_grouped(Tensor kv_c, Tensor k_pe,"
      "                             Tensor kv_cache_ptrs,"
      "                             Tensor slot_mapping,"
      "                             int block_size, int block_stride,"
      "                             int entry_stride,"
      "                             Tensor? kv_scales=None,"
      "                             str kv_cache_dtype='auto') -> ()");
""").replace("MARKER_LINE", MARKER)

# ────────────────────────────────────────────────────────────────────────────
# csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu  (#54896)
# ────────────────────────────────────────────────────────────────────────────

KK_OLD_1 = _b(r"""
constexpr int kVecElems = 8;  // 8 bf16 == one uint4 load / one uint2 fp8 store
""")
KK_NEW_1 = _b(r"""
constexpr int kVecElems = 8;  // 8 bf16 == one uint4 load / one uint2 fp8 store
// Max token count for the decode epilogue's 3-warps-per-row split; larger
// batches have enough rows for one warp per row.
// MARKER_LINE (port of #54896)
constexpr int kDecodeRowSplitMaxTokens = 64;
""").replace("MARKER_LINE", MARKER)

KK_OLD_2 = _b(r"""
// Concat + store a 576-wide latent: dst[e] = [a512 | b64], e in [0, 576). Used
// for the decode query mqa_q = [ql_nope | q_pe] and the plain latent cache
// entry [kv_c | k_pe]. FP8 packs to E4M3 (dst_elem_size 1); bf16 stores uint4.
template <typename scalar_t, bool FP8, bool APPLY_ROPE = false>
__device__ __forceinline__ void writeLatent576(void* dst, const scalar_t* a512,
                                               const scalar_t* b64, int laneId,
                                               int dst_elem_size,
                                               float scale_inv,
                                               const float* cos_sin = nullptr) {
  auto* d = reinterpret_cast<uint8_t*>(dst);
  for (int e = laneId * kVecElems; e < kCacheEntry; e += 32 * kVecElems) {
""")
KK_NEW_2 = _b(r"""
// Concat + store a 576-wide latent: dst[e] = [a512 | b64], e in [0, 576). Used
// for the decode query mqa_q = [ql_nope | q_pe] and the plain latent cache
// entry [kv_c | k_pe]. FP8 packs to E4M3 (dst_elem_size 1); bf16 stores uint4.
// A single warp passes (laneId, 32); a row split across W warps passes
// (part * 32 + laneId, W * 32). __restrict__ lets the compiler batch loads
// ahead of stores past the byte-typed dst.
template <typename scalar_t, bool FP8, bool APPLY_ROPE = false>
__device__ __forceinline__ void writeLatent576(
    void* __restrict__ dst, const scalar_t* __restrict__ a512,
    const scalar_t* __restrict__ b64, int lane, int lane_stride,
    int dst_elem_size, float scale_inv,
    const float* __restrict__ cos_sin = nullptr) {
  auto* d = reinterpret_cast<uint8_t*>(dst);
  for (int e = lane * kVecElems; e < kCacheEntry;
       e += lane_stride * kVecElems) {
""")

KK_OLD_3 = _b(r"""
      writeLatent576<scalar_t, false, APPLY_ROPE>(
          row, kv_c + tokenIdx * kv_c_tok_stride,
          k_pe + tokenIdx * k_pe_tok_stride, laneId, sizeof(scalar_t), 1.0f,
          rope_cache);
""")
KK_NEW_3 = _b(r"""
      writeLatent576<scalar_t, false, APPLY_ROPE>(
          row, kv_c + tokenIdx * kv_c_tok_stride,
          k_pe + tokenIdx * k_pe_tok_stride, laneId, 32, sizeof(scalar_t),
          1.0f, rope_cache);
""")

KK_OLD_4 = _b(r"""
      writeLatent576<scalar_t, true, APPLY_ROPE>(
          row, kv_c + tokenIdx * kv_c_tok_stride,
          k_pe + tokenIdx * k_pe_tok_stride, laneId, 1, ksi, rope_cache);
""")
KK_NEW_4 = _b(r"""
      writeLatent576<scalar_t, true, APPLY_ROPE>(
          row, kv_c + tokenIdx * kv_c_tok_stride,
          k_pe + tokenIdx * k_pe_tok_stride, laneId, 32, 1, ksi, rope_cache);
""")

KK_OLD_5 = _b(r"""
// ────────────────────────────────────────────────────────────────────────────
// Decode epilogue: concat mqa_q = [ql_nope | q_pe] (576) + latent cache insert,
// run right before forward_mqa. Q_FP8 quantizes mqa_q; KV_FP8 quantizes the
// plain per-tensor cache. (ds_mla cache uses the separate kernel below.)
// ────────────────────────────────────────────────────────────────────────────
template <typename scalar_t, bool Q_FP8, bool KV_FP8, bool APPLY_ROPE>
__global__ void fusedKimiK3MLADecodeQConcatKVCacheKernel(
""")
KK_NEW_5 = _b(r"""
// ────────────────────────────────────────────────────────────────────────────
// Decode epilogue: concat mqa_q = [ql_nope | q_pe] (576) + latent cache insert,
// run right before forward_mqa. Q_FP8 quantizes mqa_q; KV_FP8 quantizes the
// plain per-tensor cache. (ds_mla cache uses the separate kernel below.)
//
// SPLIT warps cooperate on each 576-wide row: decode batches have too few
// rows to hide the load->store latency with one warp per row, so batches up
// to kDecodeRowSplitMaxTokens launch SPLIT=3 (one 8-element chunk per lane).
//
// Only ql_nope / q_pe come from the producing GEMM chain, so everything else
// (slot_mapping, scales, rope table, cache-row inputs) is read before the
// grid-dependency wait, and the cache-slot warps skip the wait entirely.
// The dependent-launch trigger stays at the end of the kernel: it releases
// the consuming FMHA's grid-dependency wait, so it must not fire before this
// kernel's stores to mqa_q are issued.
// ────────────────────────────────────────────────────────────────────────────
template <typename scalar_t, bool Q_FP8, bool KV_FP8, bool APPLY_ROPE,
          int SPLIT = 1>
__global__ void fusedKimiK3MLADecodeQConcatKVCacheKernel(
""")

KK_OLD_6 = _b(r"""
  constexpr int kMqElem = Q_FP8 ? 1 : sizeof(scalar_t);
  constexpr int kCacheElem = KV_FP8 ? 1 : sizeof(scalar_t);
  int const warpsPerBlock = blockDim.x / 32;
  int const laneId = threadIdx.x % 32;
  int const globalWarpIdx = blockIdx.x * warpsPerBlock + threadIdx.x / 32;
  int const slotsPerToken = num_heads + 1;
  int const tokenIdx = globalWarpIdx / slotsPerToken;
  int const slotIdx = globalWarpIdx % slotsPerToken;
  if (tokenIdx >= num_tokens) return;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  cudaGridDependencySynchronize();
#endif

  const float* rope_cache = nullptr;
""")
KK_NEW_6 = _b(r"""
  constexpr int kMqElem = Q_FP8 ? 1 : sizeof(scalar_t);
  constexpr int kCacheElem = KV_FP8 ? 1 : sizeof(scalar_t);
  int const warpsPerBlock = blockDim.x / 32;
  int const laneId = threadIdx.x % 32;
  int const globalWarpIdx = blockIdx.x * warpsPerBlock + threadIdx.x / 32;
  int const slotsPerToken = num_heads + 1;
  int const globalSlotIdx = globalWarpIdx / SPLIT;
  int const part = globalWarpIdx % SPLIT;
  int const tokenIdx = globalSlotIdx / slotsPerToken;
  int const slotIdx = globalSlotIdx % slotsPerToken;
  if (tokenIdx >= num_tokens) return;
  int const lane = part * 32 + laneId;
  constexpr int kLaneStride = SPLIT * 32;

  const float* rope_cache = nullptr;
""")

KK_OLD_7 = _b(r"""
  if (slotIdx < num_heads) {
    float const qsi = Q_FP8 ? __ldg(q_scale_inv) : 1.0f;
    writeLatent576<scalar_t, Q_FP8, APPLY_ROPE>(
        reinterpret_cast<uint8_t*>(mqa_q) +
            (tokenIdx * mq_tok_stride + slotIdx * mq_head_stride) * kMqElem,
        ql_nope + tokenIdx * qn_tok_stride + slotIdx * qn_head_stride,
        q_pe + tokenIdx * qpe_tok_stride + slotIdx * qpe_head_stride, laneId,
        kMqElem, qsi, rope_cache);
""")
KK_NEW_7 = _b(r"""
  if (slotIdx < num_heads) {
    float const qsi = Q_FP8 ? __ldg(q_scale_inv) : 1.0f;
    uint8_t* const dst =
        reinterpret_cast<uint8_t*>(mqa_q) +
        (tokenIdx * mq_tok_stride + slotIdx * mq_head_stride) * kMqElem;
    const scalar_t* const nope_src =
        ql_nope + tokenIdx * qn_tok_stride + slotIdx * qn_head_stride;
    const scalar_t* const pe_src =
        q_pe + tokenIdx * qpe_tok_stride + slotIdx * qpe_head_stride;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cudaGridDependencySynchronize();
#endif
    writeLatent576<scalar_t, Q_FP8, APPLY_ROPE>(
        dst, nope_src, pe_src, lane, kLaneStride, kMqElem, qsi, rope_cache);
""")

KK_OLD_8 = _b(r"""
      writeLatent576<scalar_t, KV_FP8, APPLY_ROPE>(
          reinterpret_cast<uint8_t*>(k_cache) +
              (slot_id / cache_block_size * cache_block_stride +
               slot_id % cache_block_size * cache_token_stride) *
                  kCacheElem,
          kv_c + tokenIdx * kv_c_tok_stride, k_pe + tokenIdx * k_pe_tok_stride,
          laneId, kCacheElem, ksi, rope_cache);
""")
KK_NEW_8 = _b(r"""
      writeLatent576<scalar_t, KV_FP8, APPLY_ROPE>(
          reinterpret_cast<uint8_t*>(k_cache) +
              (slot_id / cache_block_size * cache_block_stride +
               slot_id % cache_block_size * cache_token_stride) *
                  kCacheElem,
          kv_c + tokenIdx * kv_c_tok_stride, k_pe + tokenIdx * k_pe_tok_stride,
          lane, kLaneStride, kCacheElem, ksi, rope_cache);
""")

KK_OLD_9 = _b(r"""
    writeLatent576<scalar_t, false, APPLY_ROPE>(
        mqa_q + tokenIdx * mq_tok_stride + slotIdx * mq_head_stride,
        ql_nope + tokenIdx * qn_tok_stride + slotIdx * qn_head_stride,
        q_pe + tokenIdx * qpe_tok_stride + slotIdx * qpe_head_stride, laneId,
        sizeof(scalar_t), 1.0f, rope_cache);
""")
KK_NEW_9 = _b(r"""
    writeLatent576<scalar_t, false, APPLY_ROPE>(
        mqa_q + tokenIdx * mq_tok_stride + slotIdx * mq_head_stride,
        ql_nope + tokenIdx * qn_tok_stride + slotIdx * qn_head_stride,
        q_pe + tokenIdx * qpe_tok_stride + slotIdx * qpe_head_stride, laneId,
        32, sizeof(scalar_t), 1.0f, rope_cache);
""")

# Port of upstream #54896's launchPdlSlots helper. Upstream's signature is
# launchPdlSlots(kernel, num_tokens, slots, 1, 0, stream, ...); the (1, 0)
# args belong to a newer launchPdl shape our tree does not have, so this is
# adapted to our launchPdl conventions: explicit total warp count.
KK_OLD_10 = _b(r"""
  kernel<<<grid, kBlockSize, 0, stream>>>(args...);
  // clang-format on
#endif
}

void checkBfloat16Support(torch::headeronly::ScalarType dtype) {
""")
KK_NEW_10 = _b(r"""
  kernel<<<grid, kBlockSize, 0, stream>>>(args...);
  // clang-format on
#endif
}

// PDL-aware launch of an explicit total-warp-count grid. Row-split variant
// of launchPdl (port of #54896's launchPdlSlots, adapted to this tree):
// the decode epilogue splits each 576-wide row across SPLIT warps, so the
// warp count is num_tokens * (num_heads + 1) * SPLIT.
template <typename KernelT, typename... Args>
static void launchPdlSlots(KernelT kernel, int64_t total_warps,
                           cudaStream_t stream, Args... args) {
  constexpr int kBlockSize = 256;
  constexpr int kWarpsPerBlock = kBlockSize / 32;
  int const grid =
      static_cast<int>((total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock);
#ifndef USE_ROCM
  static int const sm_version = getSMVersion();
  cudaLaunchConfig_t config;
  config.gridDim = dim3(grid);
  config.blockDim = dim3(kBlockSize);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = (sm_version >= 90) ? 1 : 0;
  cudaLaunchKernelEx(&config, kernel, args...);
#else
  // clang-format off
  // hipify's CUDA->HIP regex catastrophically backtracks on "> > >"; keep the
  // launch closer as ">>>". clang-format would otherwise re-split it (it does
  // not parse the CUDA launch syntax in this libtorch_stable file).
  kernel<<<grid, kBlockSize, 0, stream>>>(args...);
  // clang-format on
#endif
}

void checkBfloat16Support(torch::headeronly::ScalarType dtype) {
""")

KK_OLD_11 = _b(r"""
  VLLM_STABLE_DISPATCH_HALF_TYPES(
      dt, "fused_kimi_k3_mla_decode_q_concat_kv_cache_insert", [&] {
        auto launch = [&](auto kernel) {
          kk3::launchPdl(
              kernel, num_tokens, num_heads, stream,
""")
KK_NEW_11 = _b(r"""
  VLLM_STABLE_DISPATCH_HALF_TYPES(
      dt, "fused_kimi_k3_mla_decode_q_concat_kv_cache_insert", [&] {
        auto launch = [&](auto kernel, int row_split) {
          kk3::launchPdlSlots(
              kernel,
              static_cast<int64_t>(num_tokens) * (num_heads + 1) * row_split,
              stream,
""")

KK_OLD_12 = _b(r"""
        if (apply_rope) {
          launch(kk3::fusedKimiK3MLADecodeQConcatKVCacheKernel<scalar_t, false,
                                                               false, true>);
        } else {
          launch(kk3::fusedKimiK3MLADecodeQConcatKVCacheKernel<scalar_t, false,
                                                               false, false>);
        }
      });
}
""")
KK_NEW_12 = _b(r"""
        // Split decode-sized rows across 3 warps (see the kernel comment).
        bool const split_rows = num_tokens <= kk3::kDecodeRowSplitMaxTokens;
        if (apply_rope) {
          if (split_rows) {
            launch(
                kk3::fusedKimiK3MLADecodeQConcatKVCacheKernel<scalar_t, false,
                                                              false, true, 3>,
                3);
          } else {
            launch(
                kk3::fusedKimiK3MLADecodeQConcatKVCacheKernel<scalar_t, false,
                                                              false, true>,
                1);
          }
        } else {
          if (split_rows) {
            launch(
                kk3::fusedKimiK3MLADecodeQConcatKVCacheKernel<scalar_t, false,
                                                              false, false, 3>,
                3);
          } else {
            launch(
                kk3::fusedKimiK3MLADecodeQConcatKVCacheKernel<scalar_t, false,
                                                              false, false>,
                1);
          }
        }
      });
}
""")

KK_OLD_13 = _b(r"""
  VLLM_STABLE_DISPATCH_HALF_TYPES(
      dt, "fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert", [&] {
        auto launch = [&](auto kernel) {
          kk3::launchPdl(
              kernel, num_tokens, num_heads, stream,
""")
KK_NEW_13 = _b(r"""
  VLLM_STABLE_DISPATCH_HALF_TYPES(
      dt, "fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert", [&] {
        auto launch = [&](auto kernel, int row_split) {
          kk3::launchPdlSlots(
              kernel,
              static_cast<int64_t>(num_tokens) * (num_heads + 1) * row_split,
              stream,
""")

KK_OLD_14 = _b(r"""
        if (apply_rope) {
          launch(kk3::fusedKimiK3MLADecodeQConcatKVCacheKernel<scalar_t, true,
                                                               true, true>);
        } else {
          launch(kk3::fusedKimiK3MLADecodeQConcatKVCacheKernel<scalar_t, true,
                                                               true, false>);
        }
      });
}
""")
KK_NEW_14 = _b(r"""
        // Split decode-sized rows across 3 warps (see the kernel comment).
        bool const split_rows = num_tokens <= kk3::kDecodeRowSplitMaxTokens;
        if (apply_rope) {
          if (split_rows) {
            launch(
                kk3::fusedKimiK3MLADecodeQConcatKVCacheKernel<scalar_t, true,
                                                              true, true, 3>,
                3);
          } else {
            launch(
                kk3::fusedKimiK3MLADecodeQConcatKVCacheKernel<scalar_t, true,
                                                              true, true>,
                1);
          }
        } else {
          if (split_rows) {
            launch(
                kk3::fusedKimiK3MLADecodeQConcatKVCacheKernel<scalar_t, true,
                                                              true, false, 3>,
                3);
          } else {
            launch(
                kk3::fusedKimiK3MLADecodeQConcatKVCacheKernel<scalar_t, true,
                                                              true, false>,
                1);
          }
        }
      });
}
""")

# ────────────────────────────────────────────────────────────────────────────
# vllm/_custom_ops.py  (#55356)
# ────────────────────────────────────────────────────────────────────────────

PY_OLD_1 = _b(r"""
def concat_and_cache_mla_grouped(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache_ptrs: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    block_stride: int,
    entry_stride: int,
) -> None:
    torch.ops._C_cache_ops.concat_and_cache_mla_grouped(
        kv_c,
        k_pe,
        kv_cache_ptrs,
        slot_mapping,
        block_size,
        block_stride,
        entry_stride,
    )
""")
PY_NEW_1 = _b(r"""
def concat_and_cache_mla_grouped(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache_ptrs: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    block_stride: int,
    entry_stride: int,
    kv_scales: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
) -> None:
    # MARKER_LINE (port of #55356)
    torch.ops._C_cache_ops.concat_and_cache_mla_grouped(
        kv_c,
        k_pe,
        kv_cache_ptrs,
        slot_mapping,
        block_size,
        block_stride,
        entry_stride,
        kv_scales,
        kv_cache_dtype,
    )
""").replace("MARKER_LINE", MARKER)

# relpath -> list of (hunk_name, old, new)
FILES: dict[str, list[tuple[str, str, str]]] = {
    "csrc/libtorch_stable/cache_kernels.cu": [
        ("ck1 grouped kernel comment/template/kv_scales param", CK_OLD_1, CK_NEW_1),
        ("ck2 grouped kernel cache_t pointer + per-layer scale", CK_OLD_2, CK_NEW_2),
        ("ck3 grouped kernel fp8 scaled store", CK_OLD_3, CK_NEW_3),
        ("ck4 host fn signature + dtype/scale checks", CK_OLD_4, CK_NEW_4),
        ("ck5 host fn bf16/FP8 launch dispatch", CK_OLD_5, CK_NEW_5),
    ],
    "csrc/libtorch_stable/ops.h": [
        ("ops1 declaration", OPS_OLD_1, OPS_NEW_1),
    ],
    "csrc/libtorch_stable/torch_bindings.cpp": [
        ("tb1 schema def", TB_OLD_1, TB_NEW_1),
    ],
    "csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu": [
        ("kk1 kDecodeRowSplitMaxTokens constant", KK_OLD_1, KK_NEW_1),
        ("kk2 writeLatent576 (lane, lane_stride) generalisation", KK_OLD_2, KK_NEW_2),
        ("kk3 insert-kernel writeLatent576 call", KK_OLD_3, KK_NEW_3),
        ("kk4 fp8-qkv kernel writeLatent576 call", KK_OLD_4, KK_NEW_4),
        ("kk5 decode kernel comment + SPLIT template param", KK_OLD_5, KK_NEW_5),
        ("kk6 decode kernel row-split indexing + gdc_wait removal", KK_OLD_6, KK_NEW_6),
        ("kk7 decode kernel q-branch pre-wait reads + gdc_wait", KK_OLD_7, KK_NEW_7),
        ("kk8 decode kernel cache-branch lane args", KK_OLD_8, KK_NEW_8),
        ("kk9 ds_mla decode kernel writeLatent576 call", KK_OLD_9, KK_NEW_9),
        ("kk10 launchPdlSlots helper", KK_OLD_10, KK_NEW_10),
        ("kk11 bf16 host launcher head", KK_OLD_11, KK_NEW_11),
        ("kk12 bf16 host launcher split dispatch", KK_OLD_12, KK_NEW_12),
        ("kk13 fp8 host launcher head", KK_OLD_13, KK_NEW_13),
        ("kk14 fp8 host launcher split dispatch", KK_OLD_14, KK_NEW_14),
    ],
    "vllm/_custom_ops.py": [
        ("py1 wrapper kwargs + passthrough", PY_OLD_1, PY_NEW_1),
    ],
}

TOTAL_HUNKS = sum(len(h) for h in FILES.values())


def patch_text(src: str, hunks: list[tuple[str, str, str]]) -> tuple[str, list[str], list[str]]:
    """Apply hunks to src. Returns (new_src, applied_names, missing_names)."""
    applied: list[str] = []
    missing: list[str] = []
    for name, old, new in hunks:
        if old in src:
            src = src.replace(old, new, 1)
            applied.append(name)
        elif new in src:
            applied.append(name + " [already]")
        else:
            missing.append(name)
    return src, applied, missing


def process_root(root: Path, skip_python: bool, simulate: bool) -> int:
    loud(f"target root: {root}")
    loud(f"hunks: {TOTAL_HUNKS} across {len(FILES)} files"
         + (" (vllm/_custom_ops.py SKIPPED via --skip-python)" if skip_python and not simulate else ""))

    any_missing = False
    tmpdir = Path(tempfile.mkdtemp(prefix="mla_cache_kernels_sim_"))

    for relpath, hunks in FILES.items():
        if skip_python and relpath == "vllm/_custom_ops.py" and not simulate:
            loud(f"SKIP (--skip-python): {relpath}")
            continue
        path = root / relpath
        if not path.is_file():
            loud(f"NOTE: PREREQUISITE MISSING — {path} not found. NO-OP"
                 + ("" if simulate else ", not blocking boot."))
            any_missing = True
            continue

        src = path.read_text(encoding="utf-8")

        if MARKER in src:
            loud(f"SKIP (already applied, marker present): {relpath}")
            continue

        new_src, applied, missing = patch_text(src, hunks)
        if not applied and missing:
            loud(f"NOTE: PREREQUISITE MISSING — no anchors found in {relpath}; "
                 "tree shape does not match the expected "
                 "v0.26.1rc0+kimi.k3.aligned sources. NO-OP"
                 + ("" if simulate else ", not blocking boot.")
                 + f" (missing: {', '.join(missing)})")
            any_missing = True
            continue
        if missing:
            loud(f"WARNING: PARTIAL — {len(applied)}/{len(hunks)} hunks applied to "
                 f"{relpath}; missing: {', '.join(missing)}. Inspect manually — a "
                 "partial CUDA patch will NOT compile.")
            any_missing = True

        if simulate:
            sim_path = tmpdir / relpath.replace("/", "_")
            sim_path.write_text(new_src, encoding="utf-8")
            if relpath.endswith(".py"):
                try:
                    py_compile.compile(str(sim_path), doraise=True)
                except py_compile.PyCompileError as exc:
                    loud(f"SIMULATE COMPILE FAILED ({relpath}): {exc}")
                    return 1
            loud(f"SIMULATE OK: {len(applied)} hunk(s) would apply to {relpath}"
                 + (f" (patched temp copy: {sim_path})"))
            diff = difflib.unified_diff(
                src.splitlines(keepends=True),
                new_src.splitlines(keepends=True),
                fromfile=f"{relpath} (before)",
                tofile=f"{relpath} (after, simulated)",
            )
            sys.stdout.writelines(diff)
            continue

        path.write_text(new_src, encoding="utf-8")
        if relpath.endswith(".py"):
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                loud(f"COMPILE FAILED ({relpath}, file left patched, fix "
                     f"manually): {exc}")
                return 1
        loud(f"APPLY: {relpath} ({len(applied)} hunk(s))")

    if simulate:
        if any_missing:
            loud("SIMULATE RESULT: FAILED — file(s)/hunk(s) missing (see NOTEs above).")
            return 1
        loud(f"SIMULATE RESULT: OK — all {TOTAL_HUNKS} hunks apply cleanly; "
             "vllm/_custom_ops.py temp copy py_compiled.")
        loud("NOTE: CUDA compilation NOT attempted (no nvcc on this host) — "
             "hunk application only. The extension must be rebuilt in the "
             "image build for the .cu/.h/.cpp changes to take effect.")
        return 0

    loud("INFO: this mod ports upstream " + PR55356 + " (grouped MLA cache "
         "insert fp8 support) + " + PR54896 + " (decode epilogue 3-warp row "
         "split + pre-wait input reads). BUILD-TIME mod: the vLLM C++ "
         "extension (.so) must be (re)compiled from the patched csrc for the "
         "kernel changes to take effect.")
    loud("dry-run checklist:")
    loud("  1. APPLY line for each of the 5 files (or SKIP: already applied), "
         "no anchor NOTEs / PARTIAL WARNINGs.")
    loud("  2. Pre-flight: ./run.sh simulate — all hunks must apply to a temp "
         "copy; _custom_ops.py must py_compile.")
    loud("  3. Image build: bake this mod BEFORE the vLLM extension build "
         "step so the .so is compiled from patched csrc.")
    loud("  4. Serving: concat_and_cache_mla_grouped schema gains "
         "(Tensor? kv_scales, str kv_cache_dtype='auto'); decode epilogue "
         "launches 3 warps/row for <= 64 decode tokens.")
    return 0


def main() -> int:
    args = sys.argv[1:]
    skip_python = "--skip-python" in args
    args = [a for a in args if a != "--skip-python"]
    if len(args) == 2 and args[0] == "--simulate":
        return process_root(Path(args[1]), skip_python=False, simulate=True)
    if len(args) == 1 and not args[0].startswith("--"):
        return process_root(Path(args[0]), skip_python, simulate=False)
    print(f"usage: {SCRIPT_NAME}.py <src_root> [--skip-python] | "
          f"--simulate <src_root>")
    return 2


if __name__ == "__main__":
    sys.exit(main())
