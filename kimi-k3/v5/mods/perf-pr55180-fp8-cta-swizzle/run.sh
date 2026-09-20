#!/usr/bin/env bash
# perf-pr55180-fp8-cta-swizzle — port of upstream vllm-project/vllm#55180.
#
# SM12.x (GB10 / DGX Spark, sm121) blockwise-FP8 CUTLASS GEMM: swizzled CTA
# order (max_swizzle_size=8) when the FP8 weight exceeds the device L2
# (GB10: 24 MiB) — weight tiles re-read from L2 instead of re-streamed from
# DRAM per row of M tiles. Bit-identical results; auto-gated on
# get_device_prop()->l2CacheSize (bigger-L2 parts keep the default order).
#
# BUILD-TIME PATCH MOD (CUDA sources — needs a vLLM .so rebuild):
#   * at image-build time it patches the vLLM source tree
#     (/opt/kimi-k3/vllm/csrc/...) BEFORE the extension build, so the baked
#     .so carries the fix;
#   * at container start it is no-op-safe: the source tree is either already
#     marked (SKIP) or absent (NOTE + exit 0). Patching sources after the
#     .so is built has no runtime effect — the mod prints a loud NOTE saying
#     so instead of pretending.
#
# Idempotent via the marker "perf-pr55180-fp8-cta-swizzle: applied".
# Missing anchors / missing scheduler surface NO-OP with a loud NOTE (never
# block boot or the rest of the build).
#
# usage: run.sh [apply|simulate]
#   apply    (default) patch the vLLM source tree in place
#   simulate never write; patch temp copies, print the diffs (falls back to
#            the /var/tmp/v6-csrc extracted copies or /var/tmp/vllm-src when
#            no source tree is resolvable — useful on the head node)
#
# env: VLLM_SRC_ROOT  force the vLLM source root (tree containing csrc/)
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-apply}"

if [ "$MODE" != "apply" ] && [ "$MODE" != "simulate" ]; then
  echo "usage: run.sh [apply|simulate]"
  exit 2
fi

C3X_REL="csrc/libtorch_stable/quantization/w8a8/cutlass/c3x"

# Resolve the vLLM SOURCE root (the tree with csrc/ — NOT site-packages).
VLLM_SRC_ROOT="${VLLM_SRC_ROOT:-}"
if [ -z "$VLLM_SRC_ROOT" ]; then
  for cand in /opt/kimi-k3/vllm /workspace/vllm; do
    if [ -f "$cand/$C3X_REL/scaled_mm_blockwise_sm120_fp8.cu" ]; then
      VLLM_SRC_ROOT="$cand"
      break
    fi
  done
fi
# Fallback: a dev checkout whose installed package sits inside the source
# tree (site-packages layout "…/vllm/vllm/__init__.py" with csrc/ alongside).
if [ -z "$VLLM_SRC_ROOT" ]; then
  pkg="$(python3 -c "import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))" 2>/dev/null)"
  if [ -n "$pkg" ] && [ -f "$pkg/$C3X_REL/scaled_mm_blockwise_sm120_fp8.cu" ]; then
    VLLM_SRC_ROOT="$pkg"
  fi
fi

if [ "$MODE" = "simulate" ]; then
  SIM_ROOT="$VLLM_SRC_ROOT"
  if [ -z "$SIM_ROOT" ]; then
    for cand in /var/tmp/vllm-src /var/tmp/v6-csrc; do
      if [ -f "$cand/$C3X_REL/scaled_mm_blockwise_sm120_fp8.cu" ]; then
        SIM_ROOT="$cand"
        break
      fi
    done
  fi
  if [ -z "$SIM_ROOT" ]; then
    echo "=====> [perf-pr55180-fp8-cta-swizzle] NOTE: no source tree and no"
    echo "      /var/tmp fallback copy available for simulate. NO-OP."
    exit 1
  fi
  echo "=====> [perf-pr55180-fp8-cta-swizzle] SIMULATE mode (no writes)"
  python3 "$SCRIPT_DIR/patch_fp8_cta_swizzle.py" --simulate "$SIM_ROOT"
  exit $?
fi

# apply mode
if [ -z "$VLLM_SRC_ROOT" ]; then
  echo "=====> [perf-pr55180-fp8-cta-swizzle] NOTE: PREREQUISITE MISSING —"
  echo "      no vLLM source tree with $C3X_REL found"
  echo "      (looked in /opt/kimi-k3/vllm, /workspace/vllm, \$VLLM_SRC_ROOT,"
  echo "      installed-package parent). NO-OP, not blocking boot."
  exit 0
fi
echo "=====> [perf-pr55180-fp8-cta-swizzle] VLLM_SRC_ROOT=$VLLM_SRC_ROOT"
echo "=====> [perf-pr55180-fp8-cta-swizzle] build-time CUDA patch — the .so"
echo "      must be (re)built after this for the change to take effect."

python3 "$SCRIPT_DIR/patch_fp8_cta_swizzle.py" "$VLLM_SRC_ROOT"
rc=$?

echo "=====> [perf-pr55180-fp8-cta-swizzle] dry-run checklist:"
echo "  1. APPLY lines for both files (or SKIP: already applied), no anchor NOTEs."
echo "  2. Wire this mod into the image build BEFORE the vLLM extension"
echo "     compile (v6-cluster-build/Dockerfile patch stage); at container"
echo "     start it is a no-op (marker SKIP or NOTE)."
echo "  3. Serving on GB10: blockwise-FP8 GEMMs with weights > 24 MiB L2"
echo "     (e.g. 16384x2560 FP8) hold 150-174 TFLOPS at all M instead of"
echo "     collapsing to ~52 TFLOPS at M>=16384; outputs bit-identical."
echo "  4. Parts with larger L2 (GB202, 96-128 MiB) keep the default order"
echo "     (l2CacheSize gate) — no behaviour change there."
exit $rc
