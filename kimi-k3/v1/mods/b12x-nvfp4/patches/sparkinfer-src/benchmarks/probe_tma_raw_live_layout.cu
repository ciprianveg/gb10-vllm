#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr int kRows = 64;
constexpr int kCols = 256;
constexpr int kMaxElemBytes = 2;
constexpr int kMaxStageBytes = kRows * kCols * kMaxElemBytes;
constexpr int kMaxWords = kMaxStageBytes / int(sizeof(uint32_t));

struct ProbeConfig {
  const char* dtype_name;
  int elem_bytes;
  int subtile_cols;
  int subtiles;
  CUtensorMapDataType tensor_dtype;
};

__device__ __forceinline__ uint32_t smem_ptr(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* bar, uint32_t expected_count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" : : "r"(smem_ptr(bar)), "r"(expected_count) : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [%0], %1;"
      :
      : "r"(smem_ptr(bar)), "r"(bytes)
      : "memory");
}

__device__ __forceinline__ bool mbarrier_try_wait_parity(uint64_t* bar, uint32_t phase) {
  uint32_t ready = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred p;\n\t"
      "mbarrier.try_wait.parity.shared.b64 p, [%1], %2;\n\t"
      "selp.b32 %0, 1, 0, p;\n\t"
      "}\n\t"
      : "=r"(ready)
      : "r"(smem_ptr(bar)), "r"(phase)
      : "memory");
  return ready != 0;
}

__device__ __forceinline__ void cp_async_bulk_tensor_2d(
    void* dst,
    const void* tensor_map,
    int c0,
    int c1,
    uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes "
      "[%0], [%1, {%2, %3}], [%4];"
      :
      : "r"(smem_ptr(dst)), "l"(tensor_map), "r"(c0), "r"(c1), "r"(smem_ptr(bar))
      : "memory");
}

__global__ void probe_tma_live_layout_kernel(
    const CUtensorMap* __restrict__ tensor_map,
    uint32_t* __restrict__ out_words,
    bool plane_slabs,
    int subtile_cols,
    int subtiles,
    int row_bytes,
    int subtile_bytes,
    int stage_words) {
  __shared__ alignas(1024) uint8_t stage[kMaxStageBytes];
  __shared__ alignas(8) uint64_t bars[4];

  if (threadIdx.x == 0) {
    for (int g = 0; g < subtiles; ++g) {
      mbarrier_init(&bars[g], 1);
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    for (int g = 0; g < subtiles; ++g) {
      uint8_t* dst =
          plane_slabs ? (stage + g * kRows * subtile_bytes) : (stage + g * subtile_bytes);
      mbarrier_arrive_expect_tx(&bars[g], kRows * subtile_bytes);
      cp_async_bulk_tensor_2d(
          dst,
          tensor_map,
          g * subtile_cols,
          0,
          &bars[g]);
    }
  }
  __syncthreads();

  for (int g = 0; g < subtiles; ++g) {
    while (!mbarrier_try_wait_parity(&bars[g], 0)) {
    }
  }
  __syncthreads();

  const uint32_t* stage_words_ptr = reinterpret_cast<const uint32_t*>(stage);
  for (int word_idx = threadIdx.x; word_idx < stage_words; word_idx += blockDim.x) {
    out_words[word_idx] = stage_words_ptr[word_idx];
  }
}

CUtensorMapSwizzle parse_swizzle(const std::string& name) {
  if (name == "none") {
    return CU_TENSOR_MAP_SWIZZLE_NONE;
  }
  if (name == "128b") {
    return CU_TENSOR_MAP_SWIZZLE_128B;
  }
  if (name == "atom32") {
    return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
  }
  if (name == "atom64") {
    return CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B;
  }
  std::fprintf(stderr, "unsupported swizzle: %s\n", name.c_str());
  std::exit(2);
}

const char* swizzle_name(CUtensorMapSwizzle swizzle) {
  switch (swizzle) {
    case CU_TENSOR_MAP_SWIZZLE_NONE:
      return "none";
    case CU_TENSOR_MAP_SWIZZLE_128B:
      return "128b";
    case CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B:
      return "atom32";
    case CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B:
      return "atom64";
    default:
      return "unknown";
  }
}

ProbeConfig parse_dtype(const std::string& name) {
  if (name == "bf16") {
    return ProbeConfig{
        .dtype_name = "bf16",
        .elem_bytes = 2,
        .subtile_cols = 64,
        .subtiles = 4,
        .tensor_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT16,
    };
  }
  if (name == "fp8") {
    return ProbeConfig{
        .dtype_name = "fp8",
        .elem_bytes = 1,
        .subtile_cols = 128,
        .subtiles = 2,
        .tensor_dtype = CU_TENSOR_MAP_DATA_TYPE_UINT8,
    };
  }
  std::fprintf(stderr, "unsupported dtype: %s\n", name.c_str());
  std::exit(2);
}

void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
    std::exit(1);
  }
}

void check_cu(CUresult err, const char* what) {
  if (err != CUDA_SUCCESS) {
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(err, &name);
    cuGetErrorString(err, &text);
    std::fprintf(stderr, "%s failed: %s (%s)\n", what, name ? name : "?", text ? text : "?");
    std::exit(1);
  }
}

uint8_t source_byte(int idx) {
  uint32_t x = static_cast<uint32_t>(idx) * 1103515245u + 12345u;
  x ^= x >> 11;
  x ^= x << 7;
  x ^= x >> 13;
  return static_cast<uint8_t>((x >> 16) & 0xFFu);
}

void pack_words(const std::vector<uint8_t>& bytes, std::vector<uint32_t>* words_out) {
  const int word_count = static_cast<int>(bytes.size()) / 4;
  words_out->assign(word_count, 0);
  for (int idx = 0; idx < word_count; ++idx) {
    (*words_out)[idx] =
        static_cast<uint32_t>(bytes[idx * 4 + 0]) |
        (static_cast<uint32_t>(bytes[idx * 4 + 1]) << 8) |
        (static_cast<uint32_t>(bytes[idx * 4 + 2]) << 16) |
        (static_cast<uint32_t>(bytes[idx * 4 + 3]) << 24);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string dtype_arg = argc > 1 ? argv[1] : "bf16";
  const std::string swizzle_arg = argc > 2 ? argv[2] : "128b";
  const std::string layout_mode = argc > 3 ? argv[3] : "fullrow";
  const ProbeConfig cfg = parse_dtype(dtype_arg);
  const CUtensorMapSwizzle swizzle = parse_swizzle(swizzle_arg);
  const bool plane_slabs = layout_mode == "planes";
  if (!plane_slabs && layout_mode != "fullrow") {
    std::fprintf(stderr, "unsupported layout mode: %s\n", layout_mode.c_str());
    return 2;
  }

  const int row_bytes = kCols * cfg.elem_bytes;
  const int subtile_bytes = cfg.subtile_cols * cfg.elem_bytes;
  const int stage_bytes = kRows * row_bytes;
  const int stage_words = stage_bytes / int(sizeof(uint32_t));

  check_cuda(cudaSetDevice(0), "cudaSetDevice");
  check_cuda(cudaFree(nullptr), "cudaFree");
  check_cu(cuInit(0), "cuInit");

  std::vector<uint8_t> host_src_bytes(stage_bytes);
  for (int idx = 0; idx < stage_bytes; ++idx) {
    host_src_bytes[idx] = source_byte(idx);
  }

  void* dev_src = nullptr;
  uint32_t* dev_out = nullptr;
  CUtensorMap* dev_tensor_map = nullptr;
  check_cuda(cudaMalloc(&dev_src, stage_bytes), "cudaMalloc(dev_src)");
  check_cuda(cudaMalloc(&dev_out, stage_words * sizeof(uint32_t)), "cudaMalloc(dev_out)");
  check_cuda(cudaMemcpy(dev_src, host_src_bytes.data(), stage_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(dev_src)");

  alignas(64) CUtensorMap host_tensor_map{};
  std::array<uint64_t, 2> global_dim = {
      static_cast<uint64_t>(kCols),
      static_cast<uint64_t>(kRows),
  };
  std::array<uint64_t, 1> global_stride = {
      static_cast<uint64_t>(kCols * cfg.elem_bytes),
  };
  std::array<uint32_t, 2> box_dim = {
      static_cast<uint32_t>(cfg.subtile_cols),
      static_cast<uint32_t>(kRows),
  };
  std::array<uint32_t, 2> element_stride = {1u, 1u};

  check_cu(
      cuTensorMapEncodeTiled(
          &host_tensor_map,
          cfg.tensor_dtype,
          2,
          dev_src,
          global_dim.data(),
          global_stride.data(),
          box_dim.data(),
          element_stride.data(),
          CU_TENSOR_MAP_INTERLEAVE_NONE,
          swizzle,
          CU_TENSOR_MAP_L2_PROMOTION_NONE,
          CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
      "cuTensorMapEncodeTiled");

  check_cuda(cudaMalloc(&dev_tensor_map, sizeof(CUtensorMap)), "cudaMalloc(dev_tensor_map)");
  check_cuda(
      cudaMemcpy(dev_tensor_map, &host_tensor_map, sizeof(CUtensorMap), cudaMemcpyHostToDevice),
      "cudaMemcpy(dev_tensor_map)");

  probe_tma_live_layout_kernel<<<1, 128>>>(
      dev_tensor_map,
      dev_out,
      plane_slabs,
      cfg.subtile_cols,
      cfg.subtiles,
      row_bytes,
      subtile_bytes,
      stage_words);
  check_cuda(cudaGetLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

  std::vector<uint32_t> host_out(stage_words);
  check_cuda(
      cudaMemcpy(host_out.data(), dev_out, stage_words * sizeof(uint32_t), cudaMemcpyDeviceToHost),
      "cudaMemcpy(dev_out)");

  std::vector<uint8_t> expected_bytes(stage_bytes, 0);
  const int chunks_per_row = row_bytes / 16;
  const int chunks_per_plane = subtile_bytes / 16;
  if (plane_slabs) {
    for (int plane = 0; plane < cfg.subtiles; ++plane) {
      const int plane_dst_byte_base = plane * kRows * subtile_bytes;
      const int plane_src_byte_base = plane * subtile_bytes;
      for (int row = 0; row < kRows; ++row) {
        for (int chunk = 0; chunk < chunks_per_plane; ++chunk) {
          const int src_chunk = chunk ^ (row % 8);
          const int dst_byte = plane_dst_byte_base + row * subtile_bytes + chunk * 16;
          const int src_byte = row * row_bytes + plane_src_byte_base + src_chunk * 16;
          for (int byte_idx = 0; byte_idx < 16; ++byte_idx) {
            expected_bytes[dst_byte + byte_idx] = host_src_bytes[src_byte + byte_idx];
          }
        }
      }
    }
  } else {
    for (int row = 0; row < kRows; ++row) {
      for (int chunk = 0; chunk < chunks_per_row; ++chunk) {
        const int src_chunk = chunk ^ (row % 8);
        const int dst_byte = row * row_bytes + chunk * 16;
        const int src_byte = row * row_bytes + src_chunk * 16;
        for (int byte_idx = 0; byte_idx < 16; ++byte_idx) {
          expected_bytes[dst_byte + byte_idx] = host_src_bytes[src_byte + byte_idx];
        }
      }
    }
  }

  std::vector<uint32_t> expected_words;
  pack_words(expected_bytes, &expected_words);

  int mismatch_count = 0;
  int first_mismatch = -1;
  for (int idx = 0; idx < stage_words; ++idx) {
    if (host_out[idx] != expected_words[idx]) {
      ++mismatch_count;
      if (first_mismatch == -1) {
        first_mismatch = idx;
      }
    }
  }

  std::printf("{\n");
  std::printf("  \"dtype\": \"%s\",\n", cfg.dtype_name);
  std::printf("  \"swizzle\": \"%s\",\n", swizzle_name(swizzle));
  std::printf("  \"layout_mode\": \"%s\",\n", plane_slabs ? "planes" : "fullrow");
  std::printf("  \"mismatch_count\": %d,\n", mismatch_count);
  if (first_mismatch >= 0) {
    const int words_per_row = row_bytes / 4;
    const int row = first_mismatch / words_per_row;
    const int word_in_row = first_mismatch - row * words_per_row;
    std::printf("  \"first_mismatch\": {\n");
    std::printf("    \"word\": %d,\n", first_mismatch);
    std::printf("    \"row\": %d,\n", row);
    std::printf("    \"word_in_row\": %d,\n", word_in_row);
    std::printf("    \"got\": \"0x%08x\",\n", host_out[first_mismatch]);
    std::printf("    \"expected\": \"0x%08x\"\n", expected_words[first_mismatch]);
    std::printf("  },\n");
  } else {
    std::printf("  \"first_mismatch\": null,\n");
  }
  std::printf("  \"first_words\": [");
  for (int idx = 0; idx < 16; ++idx) {
    std::printf("%s\"0x%08x\"", idx == 0 ? "" : ", ", host_out[idx]);
  }
  std::printf("]\n");
  std::printf("}\n");

  cudaFree(dev_tensor_map);
  cudaFree(dev_out);
  cudaFree(dev_src);
  return mismatch_count == 0 ? 0 : 3;
}
