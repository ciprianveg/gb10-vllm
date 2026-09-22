#!/usr/bin/env bash
# build.sh — Build the KIMI-K3 v5 sm121 image (v4-prd base + v4plus batches +
#            v6 mod set + _C rebuild + perf layers)
#
# Usage:
#   ./kimi-k3/v5/build.sh              # build local tag ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v5-prd
#   ./kimi-k3/v5/build.sh --push       # build + push to GHCR
#   ./kimi-k3/v5/build.sh --tag <ref>  # override image ref
#
# If v4plus-build/ is present, the v4plus batch chain (v4-plus -> b3 -> b4 ->
# b4d -> b4e -> b4f) is built first and used as the base. Otherwise the
# published v4-prd image is used directly (missing the batch-layer speedups).
# GHCR push requires: docker login ghcr.io  (token with write:packages scope)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASE_IMAGE="${BASE_IMAGE:-ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v4-prd}"
GHCR_OWNER="${GHCR_OWNER:-ciprianveg}"
TAG="${TAG:-ghcr.io/${GHCR_OWNER}/gb10-vllm/kimi-k3:v5-prd}"

PUSH=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --push) PUSH=true; shift ;;
        --tag) TAG="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Building KIMI-K3 v5 sm121 image"
echo "  Base:   ${BASE_IMAGE}"
echo "  Target: ${TAG}"
echo ""

# ── Step 1: v4plus batch chain (if the batch layer sources are present) ──────
if [[ -d "$SCRIPT_DIR/v4plus-build" ]]; then
    echo "Building v4plus batch chain (v4-plus -> b3 -> b4 -> b4d -> b4e -> b4f)..."
    for df in Dockerfile.v4-plus Dockerfile.v4-plus-b3 Dockerfile.v4-plus-b4 \
              Dockerfile.v4-plus-b4d Dockerfile.v4-plus-b4e Dockerfile.v4-plus-b4f; do
        tag="${df#Dockerfile.}"
        DOCKER_BUILDKIT=1 docker build \
            -f "$SCRIPT_DIR/v4plus-build/$df" -t "$tag" "$SCRIPT_DIR/v4plus-build"
    done
    BASE_IMAGE="v4-plus-b4f"
    echo "  base switched to: ${BASE_IMAGE}"
else
    echo "NOTE: v4plus-build/ not present — building on plain v4-prd base"
    echo "      (no RoCEnante collectives / fused verify batch layers)."
fi

# ── Step 2: ensure base image is available, then build the overlay ──────────
docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || docker pull "$BASE_IMAGE"

DOCKER_BUILDKIT=1 docker build \
    -f "$SCRIPT_DIR/build-sm121/Dockerfile" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" \
    -t "$TAG" \
    "$SCRIPT_DIR"

echo "Built: ${TAG}"

# ── Step 3: verify (markers emitted by the mods in mods/) ───────────────────
echo ""
echo "Post-build verification:"
docker run --rm --entrypoint bash "$TAG" -c '
    V=/opt/kimi-k3/vllm
    check() { grep -rq "$2" "$V/vllm" && echo "$1: OK" || { echo "$1: MISSING"; exit 1; }; }
    ls $V/vllm/_C_stable_libtorch.abi3.so >/dev/null && echo "compiled .so: present" || { echo "compiled .so: MISSING"; exit 1; }
    check "v6-remote-dspark"        "V4PLUS-V6-REMOTE-DSPARK"
    check "pr53524 ll-bf16-prefetch" "pr53524 (upstream #53524)"
    check "pr53525 kda-pdl"         "pr53525 (upstream #53525)"
    check "pr53942 eh-proj"         "pr53942 (upstream #53942)"
    check "pr52388 mamba-metadata"  "def get_aligned_state_indices_multi_group_kernel("
    check "pr53152 topk-fuse"       "pr53152 (upstream #53152)"
    check "pr54168 lowm-tail"       "pr54168 (upstream #54168)"
    check "pr54697 kda-overlap"     "pr54697 (upstream #54697)"
    check "pr56159 kda-mixed-batch" "pr56159 (upstream #56159)"
    check "retention-dense"         "fix-k3-retention-dense"
    check "kda-spec-token-init"     "fix-k3-kda-spec-token-init"
    grep -q "perf-pr55356-54896-mla-cache-kernels" $V/vllm/_custom_ops.py && echo "MLA cache kernels (#55356/#54896): OK" || { echo "MLA cache kernels: MISSING"; exit 1; }
    grep -rlq "perf-pr55180-fp8-cta-swizzle: applied" $V/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/ && echo "FP8 CTA swizzle (#55180): OK" || { echo "FP8 CTA swizzle: MISSING"; exit 1; }
    check "autotune-import guard"   "def set_autotune_process_group(group):"
    check "marlin-nopad"            "fix-k3-marlin-nopad"
    check "triton SM12x unlock"     "SM12X-UNLOCK"
    check "nan-gumbel guard"        "fix-k3-nan-gumbel"
    check "adaptive-nst"            "fix-dspark-adaptive-nst"
    check "record-stream fence"     "fix-multistream-record-stream"
    check "grammar-stream fence"    "fix-grammar-stream-fence"
    check "draft-noeplb"            "fix-dspark-draft-noeplb"
    check "adaptive-min-depth"      "fix-adaptive-min-depth"
    check "kv-dedup"                "fix-kv-dedup-retained-endpoints"
    check "mamba-align-state-free"  "_two_steps_ago_block_idx"
    check "moe-skip-padding"        "fix-moe-skip-padding-producer"
    check "long-prefill-singleton"  "fix-long-prefill-singleton"
    check "sm121-cublas-oob"        "fix-sm121-cublas-oob"
    check "kda-first-chunk"         "fix-k3-kda-first-chunk"
    check "gb10-kv-sizing"          "fix-gb10-kv-sizing"
    check "gb10-nvml-fallback"      "fix-gb10-nvml-fallback"
    check "apc-drain-hardening"     "harden-apc-drain"
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
