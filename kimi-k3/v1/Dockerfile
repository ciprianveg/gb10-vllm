# syntax=docker/dockerfile:1.6

# =========================================================
# KIMI-K3 B12X_MLA + DSpark Image for GB10 (SM121)
# =========================================================
# 
# Base image: nvidia/cuda:13.0.2-devel-ubuntu24.04
# 
# Credits & Attribution:
# - NVIDIA: CUDA 13.0, GB10 (SM121) architecture support
# - vLLM Project (https://github.com/vllm-project/vllm): Upstream inference engine
# - local-inference-lab/vllm (fork): B12X_MLA, DSpark, PP2 aux-forward, KDA optimizations
# - voipmonitor/vllm (https://github.com/voipmonitor/vllm): Base image inspiration
# - eugr/spark-vllm-docker (https://github.com/eugr/spark-vllm-docker): Build system + mod-apply infrastructure
# - NCCL mesh support: zyang-dev/nccl (dgxspark-3node-ring branch)
# - FlashInfer: FlashInfer team (https://github.com/flashinfer-ai/flashinfer)
# - b12x/sparkinfer: NVIDIA CuTe DSL kernels for SM120/SM121
# - Marlin MoE: Marlin team (https://github.com/IST-DASLab/marlin)
# - InstantTensor: InstantTensor team (https://github.com/instanttensor)
# - Kimi-K3 model: Moonshot AI (https://www.moonshot.ai/)
# - DSpark draft: Inferact/Kimi-K3-DSpark (https://huggingface.co/Inferact/Kimi-K3-DSpark)
# =========================================================

# =========================================================
# KIMI-K3 B12X_MLA + DSpark Image for GB10 (SM121)
# =========================================================
# Based on nvidia/cuda:13.0.2-devel-ubuntu24.04
# Includes: B12X_MLA, DSpark, InstantTensor, Marlin MoE, NCCL mesh support

ARG BUILD_JOBS=16
ARG CUDA_IMAGE=nvidia/cuda:13.0.2-devel-ubuntu24.04
ARG NCCL_NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"

# =========================================================
# STAGE 1: Base Build Image
# =========================================================
FROM ${CUDA_IMAGE} AS base

ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"
ENV DG_JIT_USE_NVRTC=0
ENV USE_CUDNN=1

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy
ENV UV_HTTP_TIMEOUT=600
ENV UV_HTTP_RETRIES=10

ENV VLLM_BASE_DIR=/workspace/vllm

RUN apt update && \
    apt install -y --no-install-recommends \
    curl vim cmake build-essential ninja-build \
    libcudnn9-cuda-13 libcudnn9-dev-cuda-13 \
    python3-dev python3-pip git wget \
    libibverbs1 libibverbs-dev rdma-core \
    ccache devscripts debhelper fakeroot \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
     uv pip install torch==2.11.0 torchvision torchaudio triton --index-url https://download.pytorch.org/whl/cu130 && \
     uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2" filelock pynvml requests tqdm

ENV PATH=/usr/lib/ccache:$PATH
ENV CCACHE_DIR=/root/.ccache
ENV CCACHE_MAXSIZE=50G
ENV CCACHE_COMPRESS=1
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

ARG TORCH_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG NCCL_NVCC_GENCODE
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

WORKDIR $VLLM_BASE_DIR

# Build NCCL with mesh support for GB10 (DGX Spark)
RUN git clone -b dgxspark-3node-ring https://github.com/zyang-dev/nccl.git && \
    cd nccl && make -j ${BUILD_JOBS} src.build NVCC_GENCODE="${NCCL_NVCC_GENCODE}" && \
    make pkg.debian.build && apt install -y --no-install-recommends --allow-downgrades ./build/pkg/deb/*.deb

# =========================================================
# STAGE 2: FlashInfer Builder
# =========================================================
FROM base AS flashinfer-builder

ARG FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
WORKDIR $VLLM_BASE_DIR
ARG FLASHINFER_REF=main
ARG CACHEBUST_FLASHINFER=1

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install --no-build-isolation flashinfer-python --index-url https://flashinfer.ai/whl/cu130/torch2.11

# =========================================================
# STAGE 3: vLLM Builder (local-inference-lab/vllm fork)
# =========================================================
FROM base AS vllm-builder

WORKDIR $VLLM_BASE_DIR
ARG VLLM_REF=codex/hh-kimi-k3-dspark-dcp16-20260804
ARG CACHEBUST_VLLM=1

RUN git clone -b ${VLLM_REF} --depth 1 https://github.com/local-inference-lab/vllm.git vllm-fork && \
    cd vllm-fork && pip install -e . --no-build-isolation -v

# =========================================================
# STAGE 4: FlashInfer Export
# =========================================================
FROM base AS flashinfer-export

COPY --from=flashinfer-builder /usr/local/lib/python3.12/dist-packages/flashinfer* /usr/local/lib/python3.12/dist-packages/

# =========================================================
# STAGE 5: vLLM Export
# =========================================================
FROM base AS vllm-export

COPY --from=vllm-builder /usr/local/lib/python3.12/dist-packages/vllm* /usr/local/lib/python3.12/dist-packages/

# =========================================================
# STAGE 6: Runner (Final Image)
# =========================================================
FROM base AS runner

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV VLLM_BASE_DIR=/workspace/vllm

ENV PIP_CACHE_DIR=/root/.cache/pip
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy

# Install runtime dependencies
RUN apt update && \
    apt install -y --no-install-recommends \
    python3 python3-pip python3-dev vim curl git wget \
    libcudnn9-cuda-13 \
    libibverbs1 libibverbs-dev rdma-core \
    libxcb1 earlyoom \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Copy NCCL from base
COPY --from=base /workspace/nccl/build/pkg/deb /workspace/nccl-pkg
RUN cd /workspace/nccl-pkg && apt install -y --no-install-recommends --allow-downgrades --allow-change-held-packages ./*.deb \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv

# Copy FlashInfer and vLLM from export stages
COPY --from=flashinfer-export /usr/local/lib/python3.12/dist-packages/flashinfer* /usr/local/lib/python3.12/dist-packages/
COPY --from=vllm-export /usr/local/lib/python3.12/dist-packages/vllm* /usr/local/lib/python3.12/dist-packages/

# Download Tiktoken files
RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

# Install Python dependencies
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
     uv pip install torch==2.11.0 torchvision torchaudio triton --index-url https://download.pytorch.org/whl/cu130 && \
     uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2"

# Prebuilt wheels are not bundled — the in-image vLLM/FlashInfer builds below are
# authoritative. (Removed the host ./wheels/ bind-mount: the dir was always empty.)
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    echo "wheels not bundled — using in-image vLLM/FlashInfer builds";

# Setup environment for runtime
ARG TORCH_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ARG FLASHINFER_CUDA_ARCH_LIST="12.1a"
ENV FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TIKTOKEN_ENCODINGS_BASE=/workspace/vllm/tiktoken_encodings
ENV PATH=/workspace/vllm:$PATH

# Install b12x package (SM120/121 CuTe DSL kernels for NVFP4 GEMM and MoE)
# Provides B12X_MLA_SPARSE attention backend + b12x MoE backend
ARG INSTALL_B12X=1
RUN --mount=type=bind,source=mods/b12x-nvfp4/patches/sparkinfer-src,target=/tmp/sparkinfer-readonly \
    if [ "$INSTALL_B12X" = "1" ]; then \
        cp -a /tmp/sparkinfer-readonly /tmp/sparkinfer && \
        pip install /tmp/sparkinfer --no-deps && \
        python3 -c "import sparkinfer; import importlib.metadata as m; print(f'sparkinfer {m.version(\"sparkinfer\")} installed OK')" && \
        python3 -c "from sparkinfer.attention.dense_mla import api; print('dense_mla: OK')"; \
    fi

# Fix NCCL library path
RUN rm /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 && \
    ln -s /usr/lib/aarch64-linux-gnu/libnccl.so.2 /usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2

# Build metadata
COPY build-metadata.yaml /workspace/build-metadata.yaml

# Default entrypoint
ENTRYPOINT ["vllm"]
