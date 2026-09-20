#!/usr/bin/env bash
# build.sh — Build the KIMI-K3 v4 sm121 image
#            (public v3 base + r36-era tree advance + 5-mod overlay)
#
# Usage:
#   ./kimi-k3/v4/build.sh              # build local tag ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v4-prd
#   ./kimi-k3/v4/build.sh --push       # build + push to GHCR
#   ./kimi-k3/v4/build.sh --tag <ref>  # override image ref
#
# Base is the published v3 image; the Dockerfile advances the vLLM + b12x
# trees to the pinned upstream fork commits (public voipmonitor refs) and
# bakes the 5 mods. No CUDA recompilation — the base *.abi3.so extensions are
# preserved (they are untracked files, untouched by the git tree advance).
# GHCR push requires: docker login ghcr.io  (token with write:packages scope)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_IMAGE="${BASE_IMAGE:-ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v3-sm121}"
GHCR_OWNER="${GHCR_OWNER:-ciprianveg}"
TAG="${TAG:-ghcr.io/${GHCR_OWNER}/gb10-vllm/kimi-k3:v4-prd}"

PUSH=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --push) PUSH=true; shift ;;
        --tag) TAG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Building KIMI-K3 v4 sm121 image (v3 base + tree advance + mods overlay)"
echo "  Base:   ${BASE_IMAGE}"
echo "  Target: ${TAG}"
echo ""

# ── Step 1: pull base image ──────────────────────────────────────────────────
docker pull "$BASE_IMAGE"

# ── Step 2: build (tree advance + mod overlay) ───────────────────────────────
DOCKER_BUILDKIT=1 docker build \
    -f "$SCRIPT_DIR/build-sm121/Dockerfile" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -t "$TAG" \
    "$SCRIPT_DIR"

echo "Built: ${TAG}"

# ── Step 3: verify ───────────────────────────────────────────────────────────
echo ""
echo "Post-build verification:"
docker run --rm --entrypoint bash "$TAG" -c '
    SRC=/opt/kimi-k3/vllm/vllm
    ls $SRC/_C_stable_libtorch.abi3.so >/dev/null && echo "compiled .so: preserved OK"
    grep -q "fix-k3-r29-mamba-debug" $SRC/v1/worker/gpu/model_states/mamba_hybrid.py && echo "mamba cadence fix: OK"
    grep -q "mamba_aligned_state_indices" $SRC/models/kimi_k3/nvidia/kda_metadata.py && echo "PR #52388 (mamba metadata): OK"
    grep -q "stale_spec_reqs" $SRC/v1/attention/backends/gdn_attn.py && echo "PR #51508 (stale zero-accept): OK"
    grep -q "PR_50169" $SRC/v1/core/kv_cache_utils.py && echo "PR #50169 (drafter KV pool): OK"
    ls $SRC/v1/worker/gpu/spec_decode/dflash2/speculator.py >/dev/null && echo "DFlash2 support: OK"
' || { echo "VERIFICATION FAILED"; exit 1; }

if [[ "$PUSH" == true ]]; then
    echo ""
    echo "Pushing ${TAG} ..."
    docker push "$TAG"
    echo "Published: ${TAG}"
    echo ""
    echo "Make the package public at:"
    echo "  https://github.com/users/${GHCR_OWNER}/packages/container/gb10-vllm/settings"
fi

echo ""
echo "Done."
