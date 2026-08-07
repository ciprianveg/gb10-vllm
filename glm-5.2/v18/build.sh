#!/usr/bin/env bash
# build.sh — Build the vllm-node-tf5-glm52-v18 image for 8× GB10 (Gilded Gnosis)
#
# Usage:
#   ./v18/build.sh --solo             # build base image only (no copy, no push)
#   ./v18/build.sh                    # build + copy to all 7 workers
#   ./v18/build.sh --push             # build base + push both tags to GHCR
#   ./v18/build.sh --push-prod        # build prod variant only (base must exist)
#   ./v18/build.sh --tag <name>       # override local image tag
#
# GHCR push requires:
#   docker login ghcr.io -u ciprianveg
#   (use a PAT with write:packages scope)
#
# Requires: local-inference-lab/blackwell-llm-docker cloned at ../blackwell-llm-docker
#           v18/Dockerfile adapted for aarch64/SM121
#
# Build time on GB10: ~30-40 min (cached base stages) or ~80-90 min (full rebuild)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
BLACKWELL_DIR="${REPO_DIR}/../blackwell-llm-docker"

# Local image tag (what blackwell build produces)
TAG="${TAG:-vllm-node-tf5-glm52-v18}"
# GHCR target (set REGISTRY to override, e.g. for a fork)
GHCR_OWNER="${GHCR_OWNER:-ciprianveg}"
GHCR_REPO="ghcr.io/${GHCR_OWNER}/gb10-glm-5.2"

SOLO=false
PUSH=false
PUSH_PROD_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --solo) SOLO=true; shift ;;
        --push) PUSH=true; SOLO=true; shift ;;
        --push-prod) PUSH_PROD_ONLY=true; shift ;;
        --tag) TAG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Prod-only path: build Dockerfile.prod on top of an existing base image ──
if [[ "$PUSH_PROD_ONLY" == true ]]; then
    BASE_TAG="${TAG}"
    echo "Building prod image FROM ${BASE_TAG}:latest"
    docker build -f "$SCRIPT_DIR/Dockerfile.prod" \
        --build-arg BASE_IMAGE="${BASE_TAG}:latest" \
        -t "${GHCR_REPO}:v18-prod" \
        "$REPO_DIR"
    echo "Pushing ${GHCR_REPO}:v18-prod"
    docker push "${GHCR_REPO}:v18-prod"
    echo ""
    echo "Done. Image: ${GHCR_REPO}:v18-prod"
    exit 0
fi

# ── Full base build path ──
if [[ ! -d "$BLACKWELL_DIR" ]]; then
    echo "ERROR: blackwell-llm-docker not found at $BLACKWELL_DIR"
    echo "Clone it: git clone https://github.com/local-inference-lab/blackwell-llm-docker.git ../blackwell-llm-docker"
    echo "Then: cd ../blackwell-llm-docker && git checkout 7f3cbc6"
    exit 1
fi

# Copy adapted Dockerfile into blackwell-llm-docker
cp "$SCRIPT_DIR/Dockerfile" "$BLACKWELL_DIR/Dockerfile.vllm-b12x-cu132"

cd "$BLACKWELL_DIR"

echo "Building v18 base image: $TAG"
echo "  Source: local-inference-lab/blackwell-llm-docker @ 7f3cbc6"
echo "  CUDA arch: sm_120 (forward-compat with SM121)"
echo "  MAX_JOBS: 12  VLLM_MAX_JOBS: 12"
echo "  Copy to workers: $([ "$SOLO" == false ] && echo yes || echo no)"
echo ""

BUILD_ARGS=(
    IMAGE="$TAG"
    BUILD_BASE_IMAGE=1
    PUSH_BASE_IMAGE=0
    MAX_JOBS=12
    VLLM_MAX_JOBS=12
    NVCC_THREADS=1
)

if [[ "$SOLO" == false ]]; then
    # Build + copy to workers
    export IMAGE="$TAG"
    export BUILD_BASE_IMAGE=1
    export PUSH_BASE_IMAGE=0
    export MAX_JOBS=12
    export VLLM_MAX_JOBS=12
    export NVCC_THREADS=1

    ./build-gilded-gnosis-v18-cu132.sh

    echo ""
    echo "=== Distributing to workers ==="
    echo "Save image: docker save $TAG | gzip > /tmp/v18.tar.gz"
    echo "Copy to each worker sequentially, removing old image first:"
    echo "  for host in 192.168.177.12 ... 192.168.177.18; do"
    echo "      ssh \"\$host\" 'docker rmi $TAG 2>/dev/null; docker image prune -f'"
    echo '      ssh "$host" "docker load" < /tmp/v18.tar.gz'
    echo "  done"
else
    # Build only
    ./build-gilded-gnosis-v18-cu132.sh
fi

echo ""
echo "Base image: ${TAG}:latest"

# ── Post-build verification (sm_120a cubin check) ──
echo ""
echo "Post-build verification:"
docker run --rm "$TAG" bash -c '
    so=/opt/venv/lib/python3.12/site-packages/vllm/_C_stable_libtorch.abi3.so
    echo "sm_120a (must be 0): $(cuobjdump --list-elf "$so" 2>/dev/null | grep -c sm_120a)"
    echo "sm_120 (should be >0): $(cuobjdump --list-elf "$so" 2>/dev/null | grep -c sm_120)"
'

# ── Push to GHCR if requested ──
if [[ "$PUSH" == true ]]; then
    BASE_TAG="${GHCR_REPO}:v18-base"
    PROD_TAG="${GHCR_REPO}:v18-prod"

    echo ""
    echo "=== Pushing to GHCR ==="
    echo "  base: ${BASE_TAG}"
    echo "  prod: ${PROD_TAG}"

    # Tag + push base
    docker tag "${TAG}:latest" "${BASE_TAG}"
    docker push "${BASE_TAG}"

    # Build + push prod variant
    docker build -f "$SCRIPT_DIR/Dockerfile.prod" \
        --build-arg BASE_IMAGE="${TAG}:latest" \
        -t "${PROD_TAG}" \
        "$REPO_DIR"
    docker push "${PROD_TAG}"

    echo ""
    echo "Done. Published:"
    echo "  ${BASE_TAG}"
    echo "  ${PROD_TAG}"
    echo ""
    echo "Make the package public at:"
    echo "  https://github.com/users/${GHCR_OWNER}/packages/container/gb10-glm-5.2/settings"
    exit 0
fi

echo ""
echo "Done."
