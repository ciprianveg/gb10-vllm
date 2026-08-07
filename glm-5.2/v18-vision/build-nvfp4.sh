#!/usr/bin/env bash
# build-nvfp4.sh — Build the v18.1-vision image (native SM121 cubins + NVFP4 KV-cache capability)
#
# Rebuilds vLLM in-place on top of v18-vision. The resulting v18.1-vision image is the
# default for v18 workloads and supports BOTH fp8_ds_mla (default) and nvfp4_ds_mla
# (opt-in, for workloads needing more context than fp8 provides) KV cache.
#
# Does NOT touch the published v18-vision image. Produces a new local tag by default.
#
# Usage:
#   ./v18-vision/build-nvfp4.sh                # build only (local tag v18.1-vision)
#   ./v18-vision/build-nvfp4.sh --push         # build + push to GHCR (REQUIRES APPROVAL)
#   ./v18-vision/build-nvfp4.sh --tag <name>   # override local tag
#   ./v18-vision/build-nvfp4.sh --base <img>   # override base image
#
# Prerequisites:
#   docker pull ghcr.io/ciprianveg/gb10-glm-5.2:v18-vision
#
# Build time on GB10 head: ~30-60 min (full vLLM CUDA extension rebuild, MAX_JOBS=12).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

GHCR_OWNER="${GHCR_OWNER:-ciprianveg}"
GHCR_REPO="ghcr.io/${GHCR_OWNER}/gb10-glm-5.2"
TAG="${TAG:-${GHCR_REPO}:v18.1-vision}"
BASE_IMAGE="${BASE_IMAGE:-${GHCR_REPO}:v18-vision}"
PUSH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --push) PUSH=true; shift ;;
        --tag) TAG="$2"; shift 2 ;;
        --base) BASE_IMAGE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Building v18.1-vision (native SM121 cubins + NVFP4 KV-cache capability)"
echo "  Base:   ${BASE_IMAGE}"
echo "  Target: ${TAG}"
echo "  Push:   $([ "$PUSH" == true ] && echo yes || echo no)"
echo ""

# Ensure base image is present locally
docker pull "${BASE_IMAGE}" 2>/dev/null || true

docker build -f "$SCRIPT_DIR/Dockerfile.nvfp4" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${TAG}" \
    "$REPO_DIR"

echo ""
echo "Built: ${TAG}"

# ── Post-build verification: native sm_121 + NVFP4 cubins ────────────────────────
echo ""
echo "Post-build verification:"
docker run --rm --entrypoint bash "${TAG}" -c '
    so=/opt/venv/lib/python3.12/site-packages/vllm/_C_stable_libtorch.abi3.so
    echo "sm_121 cubins (must be >0): $(cuobjdump --list-elf "$so" 2>/dev/null | grep -c sm_121)"
    echo "sm_120 cubins:               $(cuobjdump --list-elf "$so" 2>/dev/null | grep -c sm_120)"
    echo "vision overlay glm5v.py:     $([ -f /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/glm5v.py ] && echo present || echo MISSING)"
' 2>&1 | grep -E "sm_121|sm_120|vision overlay"

if [[ "$PUSH" == true ]]; then
    echo ""
    echo "=== Pushing to GHCR ==="
    docker push "${TAG}"
    echo "Published: ${TAG}"
fi

echo ""
echo "Done."
