#!/usr/bin/env bash
# build.sh — Build the KIMI-K3 v3 sm121 image (thin overlay on the v2 image)
#
# Usage:
#   ./kimi-k3/v3/build.sh              # build local tag ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v3-sm121
#   ./kimi-k3/v3/build.sh --push       # build + push to GHCR
#   ./kimi-k3/v3/build.sh --tag <ref>  # override image ref
#
# The base v2 image must be published (or built locally via ../v2/build.sh).
# GHCR push requires: docker login ghcr.io  (token with write:packages scope)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VLLM_BASE_COMMIT="881ac39a4fb6c5bbfa14f3944db560e0a27f3ffe"
VLLM_TARGET_COMMIT="0232bce6158378f23d06ef71faa23afd4059a5d7"
VLLM_REPO="https://github.com/local-inference-lab/vllm.git"

BASE_IMAGE="${BASE_IMAGE:-ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v2-sm121}"
GHCR_OWNER="${GHCR_OWNER:-ciprianveg}"
TAG="${TAG:-ghcr.io/${GHCR_OWNER}/gb10-vllm/kimi-k3:v3-sm121}"

PUSH=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --push) PUSH=true; shift ;;
        --tag) TAG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Building KIMI-K3 v3 sm121 image (thin overlay)"
echo "  Base:   ${BASE_IMAGE}"
echo "  Target: ${TAG}"
echo ""

# ── Step 1: generate the vLLM overlay patch (881ac39 -> 0232bce6) ───────────
VLLM_CLONE="$(mktemp -d /tmp/vllm-v3-patch.XXXXXX)"
trap 'rm -rf "$VLLM_CLONE"' EXIT

echo "Fetching vLLM tree for overlay patch..."
git clone -q --filter=blob:none --no-checkout "$VLLM_REPO" "$VLLM_CLONE/vllm"
git -C "$VLLM_CLONE/vllm" fetch -q --depth=300 origin dev/infernal-invocation
git -C "$VLLM_CLONE/vllm" diff "$VLLM_BASE_COMMIT" "$VLLM_TARGET_COMMIT" -- vllm/ \
    > "$SCRIPT_DIR/build-sm121/vllm-overlay.patch"
echo "  overlay patch: $(wc -l < "$SCRIPT_DIR/build-sm121/vllm-overlay.patch") lines"

# ── Step 2: pull base image and build the overlay ────────────────────────────
docker pull "$BASE_IMAGE"

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
    ls /opt/kimi-k3/vllm/vllm/_C_stable_libtorch.abi3.so >/dev/null && echo "compiled .so: preserved OK"
    ls /opt/kimi-k3/vllm/vllm/models/kimi_k3/nvidia/tp_projection.py >/dev/null && echo "MoE fusion: OK"
    [ "$(grep -c "Recurrent state has one token-position shard" /opt/kimi-k3/vllm/vllm/v1/kv_cache_interface.py)" = "1" ] && echo "PR #418 (DCP loop fix): OK"
    [ "$(grep -c "VLLM_SHM_BROADCAST_BUSY_LOOP_S" /opt/kimi-k3/vllm/vllm/envs.py)" = "1" ] && echo "PR #421 (shm busy-wait): OK"
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
