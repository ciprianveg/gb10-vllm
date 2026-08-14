#!/usr/bin/env bash
# build.sh — Build the KIMI-K3 v2 `vllm-node-kimi3-sm121` image for GB10 (SM121)
#
# Usage:
#   ./kimi-k3/v2/build.sh            # build local tag vllm-node-kimi3-sm121
#   ./kimi-k3/v2/build.sh --push     # build + push to GHCR
#   ./kimi-k3/v2/build.sh --tag <name>   # override image tag
#
# GHCR push requires:
#   docker login ghcr.io -u ciprianveg   # PAT with write:packages scope
#
# This wraps the two-step sm121 build:
#   1. blackwell-llm-docker base image  (PyTorch 2.13 / NCCL 2.31 / cu133)
#   2. blackwell-llm-docker runtime     (vLLM@881ac39 + B12X + InstantTensor + FlashInfer)
# The sm121-adapted Dockerfiles + build scripts live in ./build-sm121/ and are
# copied into the blackwell-llm-docker checkout before building.
#
# Build time on GB10: ~70-100 min (base ~40-60 min + runtime ~30-40 min).
# See BUILD-SM121-IMAGE.md for the full guide.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BLLMD="${BLLMD_DIR:-${REPO_DIR}/../blackwell-llm-docker}"
BLLMD_COMMIT="697f50ff644f2c418645c64a50828dccce597d38"

# Local image tag (what the recipes reference as `container:`)
TAG="${TAG:-vllm-node-kimi3-sm121}"
# GHCR target (set GHCR_OWNER to override, e.g. for a fork)
GHCR_OWNER="${GHCR_OWNER:-ciprianveg}"
GHCR_REPO="ghcr.io/${GHCR_OWNER}/gb10-vllm"

PUSH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --push) PUSH=true; shift ;;
        --tag) TAG="$2"; shift 2 ;;
        --blackwell-dir) BLLMD="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Prepare blackwell-llm-docker checkout ───────────────────────────────────
if [[ ! -d "$BLLMD/.git" ]]; then
    echo "Cloning local-inference-lab/blackwell-llm-docker..."
    mkdir -p "$(dirname "$BLLMD")"
    git clone https://github.com/local-inference-lab/blackwell-llm-docker.git "$BLLMD"
fi
cd "$BLLMD"
git fetch origin "$BLLMD_COMMIT" 2>/dev/null || true
git checkout -q "$BLLMD_COMMIT"
git clean -qfd 2>/dev/null || true

echo "Building KIMI-K3 v2 sm121 image"
echo "  blackwell-llm-docker @ $BLLMD_COMMIT"
echo "  Target:      ${TAG}"
echo "  GHCR:        ${GHCR_REPO}/kimi-k3:v2-sm121"
echo "  Push:        $([ "$PUSH" == true ] && echo yes || echo no)"
echo ""

# ── Step 1: copy sm121-adapted Dockerfiles + build scripts into the checkout ─
cp "$SCRIPT_DIR/build-sm121/Dockerfile.kimi-k3-cu133-torch213-base" .
cp "$SCRIPT_DIR/build-sm121/Dockerfile.kimi-k3-infernal-invocation-cu133-torch213" .
cp "$SCRIPT_DIR/build-sm121/build-kimi-k3-cu133-torch213-base.sh" .
cp "$SCRIPT_DIR/build-sm121/build-kimi-k3-infernal-invocation-cu133-torch213.sh" .
chmod +x build-kimi-k3-cu133-torch213-base.sh \
        build-kimi-k3-infernal-invocation-cu133-torch213.sh

# ── Step 2: build the base image (PyTorch 2.13, NCCL 2.31.2, XGrammar) ──────
IMAGE="vllm-node-kimi3-sm121-base" \
RELEASE_DATE="$(date -u +%Y%m%d)" \
REVISION=r1 \
ALLOW_DIRTY_BUILD=1 \
bash build-kimi-k3-cu133-torch213-base.sh

# ── Step 3: build the runtime image (vLLM + B12X + InstantTensor + FlashInfer)
BASE_IMAGE="vllm-node-kimi3-sm121-base" \
IMAGE="${TAG}" \
RELEASE_DATE="$(date -u +%Y%m%d)" \
REVISION=r1 \
ALLOW_DIRTY_BUILD=1 \
bash build-kimi-k3-infernal-invocation-cu133-torch213.sh

echo ""
echo "Built: ${TAG}"

# ── Post-build verification ──────────────────────────────────────────────────
echo ""
echo "Post-build verification:"
docker run --rm --entrypoint bash "${TAG}" -c '
    python3 -c "import torch; print(f\"torch {torch.__version__}, cuda {torch.version.cuda}\")"
    python3 -c "import vllm; print(f\"vLLM {vllm.__version__}: OK\")"
    python3 -c "import instanttensor; print(\"instanttensor OK\")"
    python3 -c "import flashinfer; print(f\"flashinfer {flashinfer.__version__}\")"
' 2>&1 | grep -E "torch|vLLM|instanttensor|flashinfer" || true

if [[ "$PUSH" == true ]]; then
    GHCR_TAG="${GHCR_REPO}/kimi-k3:v2-sm121"
    echo ""
    echo "=== Pushing to GHCR ==="
    docker tag "${TAG}" "${GHCR_TAG}"
    docker push "${GHCR_TAG}"
    echo ""
    echo "Done. Published: ${GHCR_TAG}"
    echo ""
    echo "Make the package public at:"
    echo "  https://github.com/users/${GHCR_OWNER}/packages/container/gb10-vllm/settings"
fi

echo ""
echo "Done."
