#!/usr/bin/env bash
# build.sh — Build the vllm-node-tf5-glm52-v16 image for 8× GB10 (TP8+PP1 production)
#
# Usage:
#   ./build.sh                    # build + copy to all 7 workers
#   ./build.sh --solo             # build only (no copy)
#   ./build.sh --push             # build + push to registry (set REGISTRY env)
#
# Requires: eugr/spark-vllm-docker cloned at ../spark-vllm-docker
#           patches/ directory populated from this repo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EUROOT="${SCRIPT_DIR}/../spark-vllm-docker"
PATCHES_DIR="${SCRIPT_DIR}/patches"

TAG="${TAG:-vllm-node-tf5-glm52-v16}"
VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
VLLM_REF="${VLLM_REF:-codex/fathomless-firmament-v16-unified-20260712}"
B12X_REF="${B12X_REF:-master}"

SOLO=false
PUSH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --solo) SOLO=true; shift ;;
        --push) PUSH=true; shift ;;
        --tag) TAG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ! -d "$EUROOT" ]]; then
    echo "ERROR: eugr/spark-vllm-docker not found at $EUROOT"
    echo "Clone it: git clone https://github.com/eugr/spark-vllm-docker ../spark-vllm-docker"
    exit 1
fi

if [[ ! -d "$PATCHES_DIR" ]]; then
    echo "ERROR: patches/ not found at $PATCHES_DIR"
    exit 1
fi

cd "$EUROOT"

BUILD_ARGS=(
    -t "$TAG"
    --tf5
    --vllm-repo "$VLLM_REPO"
    --vllm-ref "$VLLM_REF"
    --b12x-ref "$B12X_REF"
    --vllm-local-patches "$PATCHES_DIR"
    --rebuild-vllm
)

if [[ "$SOLO" == false ]]; then
    BUILD_ARGS+=(-c)
fi

if [[ "$PUSH" == true ]]; then
    if [[ -z "${REGISTRY:-}" ]]; then
        echo "ERROR: REGISTRY env var required for --push"
        exit 1
    fi
    BUILD_ARGS+=(--push --registry "$REGISTRY")
fi

echo "Building $TAG..."
echo "  VLLM: $VLLM_REPO @ $VLLM_REF"
echo "  b12x: $B12X_REF"
echo "  Patches: $PATCHES_DIR"
echo "  Copy to workers: $([ "$SOLO" == false ] && echo yes || echo no)"

./build-and-copy.sh "${BUILD_ARGS[@]}"

echo "Done. Image: $TAG:latest"