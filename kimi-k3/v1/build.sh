#!/usr/bin/env bash
# build.sh — Build the KIMI-K3 B12X_MLA + DSpark image for GB10 (SM121)
#
# Usage:
#   ./kimi-k3/v1/build.sh              # build local tag (recipe image)
#   ./kimi-k3/v1/build.sh --push       # build + push to GHCR
#   ./kimi-k3/v1/build.sh --tag <name> # override image tag
#   ./kimi-k3/v1/build.sh --no-cache   # rebuild without Docker cache
#
# Prerequisites:
#   - wheels/ populated (prebuilt vLLM + FlashInfer wheels, see README) or prebuilt via build args
#   - docker login ghcr.io -u ciprianveg   # PAT with write:packages scope for --push
#
# The Dockerfile bind-mounts ./wheels and ./mods/b12x-nvfp4/patches/sparkinfer-src
# so this script must be run from the repo root (./kimi-k3/v1/build.sh).
#
# Build time on GB10 head: ~20-40 min (cached stages) or ~80-90 min (full --no-cache).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Local image tag (what the recipes reference as `container:`)
TAG="${TAG:-vllm-node-kimi3-hh}"
# GHCR target (set GHCR_OWNER to override, e.g. for a fork)
GHCR_OWNER="${GHCR_OWNER:-ciprianveg}"
GHCR_REPO="ghcr.io/${GHCR_OWNER}/gb10-vllm"

# Build args
VLLM_REF="${VLLM_REF:-codex/hh-kimi-k3-dspark-dcp16-20260804}"
BUILD_JOBS="${BUILD_JOBS:-16}"
NO_CACHE=false
PUSH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --push) PUSH=true; shift ;;
        --tag) TAG="$2"; shift 2 ;;
        --vllm-ref) VLLM_REF="$2"; shift 2 ;;
        --jobs) BUILD_JOBS="$2"; shift 2 ;;
        --no-cache) NO_CACHE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! -d "$SCRIPT_DIR/wheels" ]]; then
    echo "WARNING: $SCRIPT_DIR/wheels not found — build will compile from source (slow)."
elif ! compgen -G "$SCRIPT_DIR/wheels/*.whl" >/dev/null; then
    echo "NOTE: $SCRIPT_DIR/wheels is empty — using in-image vLLM/FlashInfer builds."
fi

echo "Building KIMI-K3 B12X image"
echo "  Dockerfile: $SCRIPT_DIR/Dockerfile"
echo "  Target:     ${TAG}"
echo "  vLLM ref:   ${VLLM_REF}"
echo "  Jobs:       ${BUILD_JOBS}"
echo "  Push:       $([ "$PUSH" == true ] && echo yes || echo no)"
echo ""

DOCKER_BUILD=(docker build -f "$SCRIPT_DIR/Dockerfile" \
    --build-arg BUILD_JOBS="${BUILD_JOBS}" \
    --build-arg VLLM_REF="${VLLM_REF}" \
    --build-arg INSTALL_B12X=1 \
    --build-arg CACHEBUST_FLASHINFER=1 \
    --build-arg CACHEBUST_VLLM=1 \
    -t "${TAG}")

if [[ "$NO_CACHE" == true ]]; then
    DOCKER_BUILD+=(--no-cache)
fi

"${DOCKER_BUILD[@]}" "$SCRIPT_DIR"

echo ""
echo "Built: ${TAG}"

# ── Post-build verification: b12x installed + vllm imports ─────────────────────
echo ""
echo "Post-build verification:"
docker run --rm --entrypoint bash "${TAG}" -c '
    python3 -c "import sparkinfer; import importlib.metadata as m; print(f'\''sparkinfer {m.version(\"sparkinfer\")}: OK'\'')" && \
    python3 -c "import vllm; print(f'\''vLLM {vllm.__version__}: OK'\'')"
' 2>&1 | grep -E "sparkinfer|vLLM" || true

if [[ "$PUSH" == true ]]; then
    GHCR_TAG="${GHCR_REPO}/kimi-k3:latest"
    echo ""
    echo "=== Pushing to GHCR ==="
    docker tag "${TAG}" "${GHCR_TAG}"
    docker push "${GHCR_TAG}"
    if [[ "$TAG" != "vllm-node-kimi3-hh" ]]; then
        docker tag "${TAG}" "${GHCR_REPO}/kimi-k3:${TAG}"
        docker push "${GHCR_REPO}/kimi-k3:${TAG}"
    fi
    echo ""
    echo "Done. Published: ${GHCR_TAG}"
    echo ""
    echo "Make the package public at:"
    echo "  https://github.com/users/${GHCR_OWNER}/packages/container/gb10-vllm/settings"
fi

echo ""
echo "Done."