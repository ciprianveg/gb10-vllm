#!/usr/bin/env bash
# build.sh — Build the v18-vision overlay image on top of v18-prod
#
# Usage:
#   ./v18-vision/build.sh              # build only (local tag)
#   ./v18-vision/build.sh --push       # build + push to GHCR
#   ./v18-vision/build.sh --tag <name> # override local tag
#
# Prerequisites:
#   docker pull ghcr.io/ciprianveg/gb10-glm-5.2:v18-prod
#   docker login ghcr.io -u ciprianveg   # PAT with write:packages for --push
#
# The Dockerfile COPYs from ./v18-vision/overlay/... so this script must be
# run from the repo root (./v18-vision/build.sh), not from inside v18-vision/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# GHCR target (set REGISTRY/owner to override, e.g. for a fork)
GHCR_OWNER="${GHCR_OWNER:-ciprianveg}"
GHCR_REPO="ghcr.io/${GHCR_OWNER}/gb10-glm-5.2"
TAG="${TAG:-${GHCR_REPO}:v18-vision}"
BASE_IMAGE="${BASE_IMAGE:-${GHCR_REPO}:v18-prod}"

PUSH=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --push) PUSH=true; shift ;;
        --tag) TAG="$2"; shift 2 ;;
        --base) BASE_IMAGE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Building v18-vision overlay image"
echo "  Base:   ${BASE_IMAGE}"
echo "  Target: ${TAG}"
echo "  Push:   $([ "$PUSH" == true ] && echo yes || echo no)"
echo ""

# Ensure base image is present locally
docker pull "${BASE_IMAGE}" 2>/dev/null || true

docker build -f "$SCRIPT_DIR/Dockerfile" \
    --build-arg BASE_IMAGE="${BASE_IMAGE}" \
    -t "${TAG}" \
    "$REPO_DIR"

echo ""
echo "Built: ${TAG}"

if [[ "$PUSH" == true ]]; then
    echo ""
    echo "=== Pushing to GHCR ==="
    docker push "${TAG}"
    echo ""
    echo "Done. Published: ${TAG}"
    echo ""
    echo "Make the package public at:"
    echo "  https://github.com/users/${GHCR_OWNER}/packages/container/gb10-glm-5.2/settings"
fi

echo ""
echo "Done."
