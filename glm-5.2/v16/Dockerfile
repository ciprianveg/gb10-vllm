# Dockerfile — Reference Build for vllm-node-tf5-glm52-v16 (TP8+PP1 production)
#
# This is a REFERENCE Dockerfile showing the build configuration.
# The actual build is performed by eugr/spark-vllm-docker's build-and-copy.sh
# which uses a multi-stage Dockerfile with wheel caching.
#
# To build locally (single node, no cache):
#   docker build -t vllm-node-tf5-glm52-v16 --build-arg VLLM_REF=codex/fathomless-firmament-v16-unified-20260712 --build-arg VLLM_REPO=https://github.com/local-inference-lab/vllm.git --build-arg VLLM_LOCAL_PATCHES_DIR=v16/patches --build-arg CACHEBUST_VLLM=$(date +%s) -f Dockerfile.vllm .
#
# For production (multi-node, cached wheels, SCP deploy):
#   cd eugr/spark-vllm-docker
#   ./build-and-copy.sh -t vllm-node-tf5-glm52-v16 --tf5 --vllm-repo https://github.com/local-inference-lab/vllm.git --vllm-ref codex/fathomless-firmament-v16-unified-20260712 --b12x-ref master --vllm-local-patches patches/v16-final --rebuild-vllm -c

# =============================================================================
# STAGE 1: Base CUDA + System Deps
# =============================================================================
FROM nvidia/cuda:13.2.0-devel-ubuntu24.04 AS base

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

# Install uv (fast pip replacement)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && mv /root/.local/bin/uv /usr/local/bin/uv

# =============================================================================
# STAGE 2: FlashInfer (prebuilt wheels for sm_121)
# =============================================================================
FROM base AS flashinfer-builder

ARG FLASHINFER_VERSION=0.2.0
ARG FLASHINFER_CUDA_ARCH_LIST=12.1a

RUN uv pip install --system \
    --index-url https://flashinfer.ai/whl/cu121 \
    flashinfer-python==${FLASHINFER_VERSION}

# =============================================================================
# STAGE 3: vLLM Build (with patches)
# =============================================================================
FROM base AS vllm-builder

ARG VLLM_REPO=https://github.com/local-inference-lab/vllm.git
ARG VLLM_REF=codex/fathomless-firmament-v16-unified-20260712
ARG VLLM_LOCAL_PATCHES_DIR=v16/patches
ARG VLLM_APPLY_PRESET_PRS=0
ARG TORCH_CUDA_ARCH_LIST=12.1a
ARG FLASHINFER_CUDA_ARCH_LIST=12.1a
ARG NCCL_NVCC_GENCODE=-gencode=arch=compute_121,code=sm_121
ARG BUILD_JOBS=16
ARG CACHEBUST_VLLM=0

# Copy FlashInfer from previous stage
COPY --from=flashinfer-builder /usr/local/lib/python3.12/dist-packages/ /usr/local/lib/python3.12/dist-packages/

# Install PyTorch 2.11.0 (pinned)
RUN uv pip install --system \
    --index-url https://download.pytorch.org/whl/cu121 \
    torch==2.11.0 torchvision==0.16.0 torchaudio==2.11.0

# Install build deps
RUN uv pip install --system \
    numpy ninja cmake pybind11 packaging wheel setuptools

# Clone vLLM
RUN git clone --recursive ${VLLM_REPO} /workspace/vllm && \
    cd /workspace/vllm && \
    git checkout ${VLLM_REF} && \
    git submodule update --init --recursive

# Apply local patches
COPY ${VLLM_LOCAL_PATCHES_DIR}/ /tmp/vllm-local-patches/
RUN if [ -d "/tmp/vllm-local-patches" ] && [ "$(ls /tmp/vllm-local-patches/*.patch 2>/dev/null | wc -l)" -gt 0 ]; then \
    cd /workspace/vllm && \
    for p in $(ls /tmp/vllm-local-patches/*.patch | sort); do \
        echo "Applying $(basename $p)..."; \
        if git apply --check "$p" 2>/dev/null; then \
            git apply "$p" && echo "  ✓ applied"; \
        else \
            echo "  ⚠ already applied or context mismatch — skipping"; \
        fi; \
    done; \
    echo "Local patches done."; \
    else \
    echo "No local patches to apply."; \
    fi

# Build vLLM wheel
RUN cd /workspace/vllm && \
    MAX_JOBS=${BUILD_JOBS} \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    FLASHINFER_CUDA_ARCH_LIST=${FLASHINFER_CUDA_ARCH_LIST} \
    NCCL_NVCC_GENCODE=${NCCL_NVCC_GENCODE} \
    python3 -m pip wheel --no-deps -w /wheels . --verbose

# =============================================================================
# STAGE 4: Runner (minimal runtime image)
# =============================================================================
FROM nvidia/cuda:13.2.0-runtime-ubuntu24.04 AS runner

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/conda/bin:$PATH \
    LD_LIBRARY_PATH=/opt/nccl-2.30.4/lib:$LD_LIBRARY_PATH \
    NCCL_NET=IB \
    NCCL_IB_GID_INDEX=3 \
    NCCL_CROSS_NIC=1 \
    NCCL_CUMEM_ENABLE=0 \
    NCCL_IGNORE_CPU_AFFINITY=1 \
    NCCL_DEBUG=WARN \
    VLLM_USE_V2_MODEL_RUNNER=1 \
    VLLM_USE_B12X_SPARSE_INDEXER=1 \
    VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1 \
    VLLM_DCP_GLOBAL_TOPK=1 \
    VLLM_DCP_SHARD_DRAFT=1 \
    TORCH_CUDA_ARCH_LIST=12.1a \
    CUTE_DSL_ARCH=sm_121a

# Install runtime deps + NCCL 2.30.4
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv \
    libibverbs1 librdmacm1 libnuma1 numactl \
    libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# NCCL 2.30.4 (aarch64 wheel)
RUN uv pip install --system \
    --index-url https://pypi.org/simple \
    nvidia-nccl-cu12==2.30.4

# Copy vLLM wheel and install
COPY --from=vllm-builder /wheels/*.whl /tmp/wheels/
RUN uv pip install --system /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Copy FlashInfer
COPY --from=flashinfer-builder /usr/local/lib/python3.12/dist-packages/ /usr/local/lib/python3.12/dist-packages/

# Install b12x from master
RUN uv pip install --system --no-deps \
    git+https://github.com/lukealonso/b12x@master

# Default entrypoint
ENTRYPOINT ["vllm"]
CMD ["serve", "--help"]