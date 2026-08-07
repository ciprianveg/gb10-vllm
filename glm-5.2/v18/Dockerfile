# Dockerfile — Reference Build for vllm-node-tf5-glm52-v18 (Gilded Gnosis, aarch64)
#
# Adapted from local-inference-lab/blackwell-llm-docker @ 7f3cbc6
# for DGX Spark GB10 (SM121 / aarch64).
#
# Key adaptations:
#   - CUDA arch: sm_120 (NOT sm_120a — a suffix cubins don't load on SM121)
#   - NVFP4 SM120 disabled via CMakeLists.txt sed
#   - MAX_JOBS=12 (not 64 — GB10 has 121 GB RAM)
#   - Architecture paths use dpkg-architecture for aarch64 compat
#
# To build:
#   git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
#   cd blackwell-llm-docker && git checkout 7f3cbc6
#   cp /path/to/gb10-glm-5.2/v18/Dockerfile Dockerfile.vllm-b12x-cu132
#   IMAGE=vllm-node-tf5-glm52-v18:latest \
#     BUILD_BASE_IMAGE=1 PUSH_BASE_IMAGE=0 \
#     MAX_JOBS=12 VLLM_MAX_JOBS=12 NVCC_THREADS=1 \
#     ./build-gilded-gnosis-v18-cu132.sh
#
# For production (multi-node, wheel caching, SCP deploy):
#   Use spark-vllm-docker build system with v18 recipe.

ARG BASE_IMAGE=nvidia/cuda:13.2.1-devel-ubuntu24.04
ARG RUNNER_IMAGE=nvidia/cuda:13.2.1-runtime-ubuntu24.04
ARG PYTHON_VERSION=3.12

# =============================================================================
# STAGE 1: Base
# =============================================================================
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    git curl wget ca-certificates \
    build-essential cmake ninja-build \
    libibverbs-dev librdmacm-dev \
    libnuma-dev numactl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && mv /root/.local/bin/uv /usr/local/bin/uv

# =============================================================================
# STAGE 2: vLLM Build (with v18 source patches)
# =============================================================================
FROM base AS vllm-builder

ARG VLLM_REPO=https://github.com/local-inference-lab/vllm.git
ARG VLLM_REF=build/gilded-gnosis-v18-final-20260718
ARG TORCH_CUDA_ARCH_LIST=12.0
ARG FLASHINFER_CUDA_ARCH_LIST=12.1a
ARG BUILD_JOBS=12
ARG CACHEBUST_VLLM=0

# Install PyTorch 2.12.0 (v18 stack)
RUN uv pip install --system \
    --index-url https://download.pytorch.org/whl/cu132 \
    torch==2.12.0 torchvision==0.17.0 torchaudio==2.12.0

RUN uv pip install --system numpy ninja cmake pybind11 packaging wheel setuptools

# Install FlashInfer (voipmonitor fork, prebuilt sm_120)
RUN uv pip install --system \
    git+https://github.com/voipmonitor/flashinfer@801d57a

# Install DeepGEMM
RUN uv pip install --system \
    git+https://github.com/deepseek-ai/deepgemm.git@a6b593d

# Install InstantTensor
RUN uv pip install --system \
    git+https://github.com/local-inference-lab/instant-tensor.git@85e7c5f

# Clone vLLM v18
RUN git clone --recursive ${VLLM_REPO} /workspace/vllm && \
    cd /workspace/vllm && \
    git checkout ${VLLM_REF} && \
    git submodule update --init --recursive

# Disable NVFP4 SM120 (sm_120a cubins don't load on SM121)
RUN sed -i 's/cuda_archs_loose_intersection(FP4_SM120_ARCHS "12.0f" "${CUDA_ARCHS}")/cuda_archs_loose_intersection(FP4_SM120_ARCHS "99.0f" "${CUDA_ARCHS}")/' /opt/vllm/CMakeLists.txt

# Build vLLM wheel
RUN cd /workspace/vllm && \
    MAX_JOBS=${BUILD_JOBS} \
    VLLM_MAX_JOBS=${BUILD_JOBS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CMAKE_CUDA_ARCHITECTURES=120 \
    NVCC_THREADS=1 \
    python3 -m pip wheel --no-deps -w /wheels . --verbose

# =============================================================================
# STAGE 3: Runner
# =============================================================================
FROM ${RUNNER_IMAGE} AS runner

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    LD_LIBRARY_PATH=/opt/libnccl-local-inference.so.2.30.4:$LD_LIBRARY_PATH \
    VLLM_USE_V2_MODEL_RUNNER=1 \
    VLLM_USE_B12X_SPARSE_INDEXER=1 \
    VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    CUTE_DSL_ARCH=sm_121a

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv \
    libibverbs1 librdmacm1 libnuma1 numactl \
    libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Copy vLLM wheel
COPY --from=vllm-builder /wheels/*.whl /tmp/wheels/
RUN uv pip install --system /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Install B12X (voipmonitor fork, v18)
RUN uv pip install --system --no-deps \
    git+https://github.com/voipmonitor/b12x@bc85ef3

# NCCL 2.30.4 (local-inference-lab fork)
RUN uv pip install --system \
    --index-url https://pypi.org/simple \
    nvidia-nccl-cu12==2.30.4

# OCI provenance labels (show up on the GHCR package page)
LABEL org.opencontainers.image.source="https://github.com/ciprianveg/gb10-glm-5.2"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.title="gb10-glm-5.2 v18 (base)"
LABEL org.opencontainers.image.description="GLM-5.2-Int4-Int8Mix on 8× DGX Spark (GB10, sm_121), TP8+PP1, MTP k=4 — compiled stack, no runtime mods"
LABEL org.opencontainers.image.authors="ciprian"
LABEL org.opencontainers.image.url="https://github.com/ciprianveg/gb10-glm-5.2/blob/master/ATTRIBUTION.md"
LABEL org.opencontainers.image.version="v18"

ENTRYPOINT ["vllm"]
CMD ["serve", "--help"]
