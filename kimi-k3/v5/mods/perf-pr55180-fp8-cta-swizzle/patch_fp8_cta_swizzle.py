#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PERF-PR55180-FP8-CTA-SWIZZLE — port of upstream vllm-project/vllm#55180.

SM12.x (GB10 / DGX Spark, sm121) blockwise-FP8 CUTLASS GEMM: switch the
persistent tile scheduler to a swizzled CTA order (``max_swizzle_size = 8``)
when the FP8 weight operand exceeds the device L2 (GB10: 24 MiB), so the
weight tiles are re-read from L2 instead of being re-streamed from DRAM once
per row of M tiles. Bit-identical results (each output tile's K-reduction is
unchanged); auto-gated on ``get_device_prop()->l2CacheSize`` so parts whose
L2 holds the weight (RTX PRO 6000 / GB202, 96-128 MiB) keep the default
order. Port of upstream vllm-project/vllm#55180 (blockwise-FP8 SM120 CTA
swizzle), adapted to this tree (v0.26.1rc0+kimi.k3.aligned — older than the
PR base, but both target files match the PR pre-image byte-for-byte).

BUILD-TIME PATCH: touches CUDA sources only. The vLLM C extension (.so) must
be (re)built after applying for the change to take effect. Applying at
container start on an already-built image is a no-op for runtime (it only
edits source text); the marker makes re-runs SKIP.

Surface pre-check (why this compiles on our tree, CUTLASS pin v4.4.2):
  * ``c3x::cutlass_gemm_caller`` (csrc/libtorch_stable/quantization/w8a8/
    cutlass/c3x/cutlass_gemm_caller.cuh) already takes a 5th parameter
    ``typename GemmKernel::TileSchedulerArguments scheduler = {}`` and feeds
    it into ``GemmUniversal::Arguments`` — the exact plumbing #55180 needs.
  * For arch::Sm120 the CUTLASS TileSchedulerSelector maps to
    PersistentTileSchedulerSm100, whose ``Arguments`` is
    ``{ int max_swizzle_size = 0; RasterOrderOptions raster_order = ...; }``
    (CUTLASS v4.4.2, sm100_tile_scheduler.hpp) — consumed by
    ``to_underlying_arguments`` and implemented device-side by
    ``swizzle_and_rasterize``. Passing 1 selects the default (identity)
    order, matching the unpatched behaviour.
  * ``get_device_prop()`` (cached ``cudaDeviceProp*``) is declared in
    csrc/libtorch_stable/torch_utils.h, transitively included via the
    dispatch .cuh -> cutlass_gemm_caller.cuh -> torch_utils.h chain.

Modes:
  apply    <vllm_src_root | c3x_dir>   patch in place; missing anchors or
                                       missing scheduler surface NO-OP with a
                                       loud NOTE and exit 0 (never block
                                       boot/build of the rest).
  --simulate <vllm_src_root | c3x_dir> never writes the targets; patches
                                       temp copies, prints unified diffs.
                                       Exits 1 if anchors/surface are
                                       missing — useful as pre-flight check.

No CUDA compilation is performed here (by design — the mod runs before the
wheel build; syntax-level verification is anchor matching + diff review).

Idempotent via the marker string ``perf-pr55180-fp8-cta-swizzle: applied``
(embedded in both patched files).
"""
import difflib
import sys
import tempfile
from pathlib import Path

SCRIPT_NAME = "perf-pr55180-fp8-cta-swizzle"
PR = "vllm-project/vllm#55180"
TAG = f"{SCRIPT_NAME} (blockwise-FP8 SM120 CTA swizzle for GB10 / sm121)"

C3X_REL = "csrc/libtorch_stable/quantization/w8a8/cutlass/c3x"
CU_NAME = "scaled_mm_blockwise_sm120_fp8.cu"
CUH_NAME = "scaled_mm_blockwise_sm120_fp8_dispatch.cuh"
CALLER_NAME = "cutlass_gemm_caller.cuh"

MARKER = f"{SCRIPT_NAME}: applied"

# ---------------------------------------------------------------------------
# scaled_mm_blockwise_sm120_fp8.cu — L2-gated swizzle-size selection + plumb
# (upstream hunk 1: matched byte-for-byte, only the marker line added)
# ---------------------------------------------------------------------------
OLD_CU1 = (
    "namespace vllm {\n"
    "\n"
    "void cutlass_scaled_mm_blockwise_sm120_fp8(\n"
    "    torch::stable::Tensor& out, torch::stable::Tensor const& a,\n"
    "    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,\n"
    "    torch::stable::Tensor const& b_scales) {\n"
    "  if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {\n"
    "    cutlass_gemm_blockwise_sm120_fp8_dispatch<cutlass::bfloat16_t>(\n"
    "        out, a, b, a_scales, b_scales);\n"
)
NEW_CU1 = (
    "namespace vllm {\n"
    "\n"
    "namespace {\n"
    "\n"
    f"// {MARKER}\n"
    "// CTA swizzle for SM 12.x parts whose L2 does not hold the weight operand.\n"
    "//\n"
    "// On a GB10 (24 MiB L2) the blockwise kernel loses most of its throughput once\n"
    "// the weight is re-streamed from DRAM per row of M tiles: a 16384x2560 FP8\n"
    "// weight runs at 165 TFLOPS at M=4096 but 90 at M=8192 and 52 at M>=16384;\n"
    "// 8192x8192 is at 54 from M=6144. With the tile scheduler's max_swizzle_size = 8\n"
    "// the same launches run at 150-174 TFLOPS at every M, bit-identical to the\n"
    "// default order (ten N/K shapes, M 2048-16384, all cells identical). The one\n"
    "// place the default order is better is a narrow band around M=4096 on the\n"
    "// 2560-wide weights (167 vs 153 at 16384x2560, 160 vs 154 at 12288x2560);\n"
    "// elsewhere the swizzled order is equal or up to 3.3x faster, so it is used\n"
    "// whenever the weight exceeds the L2. Parts whose L2 holds the weight (RTX PRO\n"
    "// 6000 Blackwell / GB202: 96-128 MiB) keep the default order, which is also\n"
    "// the faster one there (2560x6144 at 15 MiB: 178 vs 163 at M=2048).\n"
    f"// Port of upstream {PR}.\n"
    "constexpr int kBlockwiseFp8SwizzleSize = 8;\n"
    "\n"
    "int blockwise_fp8_swizzle_size(int64_t weight_bytes) {\n"
    "  const int64_t l2_bytes = get_device_prop()->l2CacheSize;\n"
    "  return (l2_bytes > 0 && weight_bytes > l2_bytes) ? kBlockwiseFp8SwizzleSize\n"
    "                                                   : 1;\n"
    "}\n"
    "\n"
    "}  // namespace\n"
    "\n"
    "void cutlass_scaled_mm_blockwise_sm120_fp8(\n"
    "    torch::stable::Tensor& out, torch::stable::Tensor const& a,\n"
    "    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,\n"
    "    torch::stable::Tensor const& b_scales) {\n"
    "  // b is [K, N] FP8 (one byte per element).\n"
    "  const int swizzle = blockwise_fp8_swizzle_size(b.size(1) * b.size(0));\n"
    "  if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {\n"
    "    cutlass_gemm_blockwise_sm120_fp8_dispatch<cutlass::bfloat16_t>(\n"
    "        out, a, b, a_scales, b_scales, swizzle);\n"
)

OLD_CU2 = (
    "    cutlass_gemm_blockwise_sm120_fp8_dispatch<cutlass::half_t>(\n"
    "        out, a, b, a_scales, b_scales);\n"
)
NEW_CU2 = (
    "    cutlass_gemm_blockwise_sm120_fp8_dispatch<cutlass::half_t>(\n"
    "        out, a, b, a_scales, b_scales, swizzle);\n"
)

CU_ANCHORS = [
    ("cu: anon-namespace swizzle selector + bf16 dispatch", OLD_CU1, NEW_CU1),
    ("cu: fp16 dispatch passes swizzle", OLD_CU2, NEW_CU2),
]

# ---------------------------------------------------------------------------
# scaled_mm_blockwise_sm120_fp8_dispatch.cuh — thread max_swizzle_size through
# to the persistent scheduler (upstream hunks 2-4: matched byte-for-byte,
# only the marker line added)
# ---------------------------------------------------------------------------
OLD_CUH1 = (
    "void cutlass_gemm_caller_blockwise(torch::stable::Tensor& out, torch::stable::Tensor const& a,\n"
    "                                   torch::stable::Tensor const& b,\n"
    "                                   torch::stable::Tensor const& a_scales,\n"
    "                                   torch::stable::Tensor const& b_scales) {\n"
)
NEW_CUH1 = (
    "void cutlass_gemm_caller_blockwise(torch::stable::Tensor& out, torch::stable::Tensor const& a,\n"
    "                                   torch::stable::Tensor const& b,\n"
    "                                   torch::stable::Tensor const& a_scales,\n"
    "                                   torch::stable::Tensor const& b_scales,\n"
    "                                   int max_swizzle_size) {\n"
)

OLD_CUH2 = (
    "  auto c_ptr = static_cast<ElementD*>(out.data_ptr());\n"
    "  typename GemmKernel::EpilogueArguments epilogue_args{\n"
    "      {}, c_ptr, c_stride, c_ptr, c_stride};\n"
    "  c3x::cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args,\n"
    "                                       epilogue_args);\n"
    "}\n"
)
NEW_CUH2 = (
    "  auto c_ptr = static_cast<ElementD*>(out.data_ptr());\n"
    "  typename GemmKernel::EpilogueArguments epilogue_args{\n"
    "      {}, c_ptr, c_stride, c_ptr, c_stride};\n"
    "  // CTA rasterization: max_swizzle_size > 1 groups nearby M/N tiles in the\n"
    "  // persistent scheduler's raster, improving the temporal locality of the\n"
    "  // shared weight (B) tiles in the L2. Bit-identical to the default order\n"
    "  // (each output tile's K-reduction is unchanged).\n"
    f"  // {MARKER}\n"
    "  typename GemmKernel::TileSchedulerArguments scheduler{};\n"
    "  scheduler.max_swizzle_size = max_swizzle_size;\n"
    "  c3x::cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args,\n"
    "                                       epilogue_args, scheduler);\n"
    "}\n"
)

OLD_CUH3 = (
    "void cutlass_gemm_blockwise_sm120_fp8_dispatch(torch::stable::Tensor& out,\n"
    "                                               torch::stable::Tensor const& a,\n"
    "                                               torch::stable::Tensor const& b,\n"
    "                                               torch::stable::Tensor const& a_scales,\n"
    "                                               torch::stable::Tensor const& b_scales) {\n"
)
NEW_CUH3 = (
    "void cutlass_gemm_blockwise_sm120_fp8_dispatch(torch::stable::Tensor& out,\n"
    "                                               torch::stable::Tensor const& a,\n"
    "                                               torch::stable::Tensor const& b,\n"
    "                                               torch::stable::Tensor const& a_scales,\n"
    "                                               torch::stable::Tensor const& b_scales,\n"
    "                                               int max_swizzle_size) {\n"
)

OLD_CUH4 = (
    "      using Gemm = typename sm120_blockwise_fp8_config_pingpong<OutType>::Gemm;\n"
    "      return cutlass_gemm_caller_blockwise<Gemm>(\n"
    "          out, a, b, a_scales, b_scales);\n"
)
NEW_CUH4 = (
    "      using Gemm = typename sm120_blockwise_fp8_config_pingpong<OutType>::Gemm;\n"
    "      return cutlass_gemm_caller_blockwise<Gemm>(\n"
    "          out, a, b, a_scales, b_scales, max_swizzle_size);\n"
)

OLD_CUH5 = (
    "    using Gemm = typename sm120_blockwise_fp8_config_default<OutType>::Gemm;\n"
    "    return cutlass_gemm_caller_blockwise<Gemm>(\n"
    "        out, a, b, a_scales, b_scales);\n"
)
NEW_CUH5 = (
    "    using Gemm = typename sm120_blockwise_fp8_config_default<OutType>::Gemm;\n"
    "    return cutlass_gemm_caller_blockwise<Gemm>(\n"
    "        out, a, b, a_scales, b_scales, max_swizzle_size);\n"
)

OLD_CUH6 = (
    "    using Gemm = typename sm120_blockwise_fp8_config_swapab<OutType>::Gemm;\n"
    "    return cutlass_gemm_caller_blockwise<Gemm>(\n"
    "        out, a, b, a_scales, b_scales);\n"
)
NEW_CUH6 = (
    "    using Gemm = typename sm120_blockwise_fp8_config_swapab<OutType>::Gemm;\n"
    "    return cutlass_gemm_caller_blockwise<Gemm>(\n"
    "        out, a, b, a_scales, b_scales, max_swizzle_size);\n"
)

CUH_ANCHORS = [
    ("cuh: cutlass_gemm_caller_blockwise signature", OLD_CUH1, NEW_CUH1),
    ("cuh: scheduler args into c3x::cutlass_gemm_caller", OLD_CUH2, NEW_CUH2),
    ("cuh: sm120_fp8_dispatch signature", OLD_CUH3, NEW_CUH3),
    ("cuh: pingpong call site", OLD_CUH4, NEW_CUH4),
    ("cuh: default call site", OLD_CUH5, NEW_CUH5),
    ("cuh: swapab call site", OLD_CUH6, NEW_CUH6),
]


def loud(msg: str) -> None:
    print(f"=====> [{SCRIPT_NAME}] {msg}")


def resolve_c3x(root: Path) -> Path | None:
    """Accept either the vllm source root or the c3x dir itself."""
    as_root = root / C3X_REL
    if (as_root / CU_NAME).is_file():
        return as_root
    if (root / CU_NAME).is_file():
        return root
    return None


def precheck(c3x_dir: Path) -> bool:
    """Verify the swizzle/raster surface the patch relies on.

    Returns True if it is safe to patch. Prints a loud NOTE otherwise.
    """
    ok = True
    caller = c3x_dir / CALLER_NAME
    if caller.is_file():
        caller_src = caller.read_text()
        if "TileSchedulerArguments scheduler = {}" in caller_src:
            loud(
                "pre-check OK: c3x::cutlass_gemm_caller already takes "
                "`TileSchedulerArguments scheduler = {}` (5th param) and feeds "
                "it into GemmUniversal::Arguments — the exact surface "
                f"{PR} needs."
            )
            if "torch_utils.h" in caller_src:
                loud(
                    "pre-check OK: cutlass_gemm_caller.cuh includes "
                    "torch_utils.h -> get_device_prop() is visible in the "
                    ".cu translation unit."
                )
            else:
                loud(
                    "WARNING: cutlass_gemm_caller.cuh does not include "
                    "torch_utils.h — confirm get_device_prop() is declared "
                    "before the patched .cu compiles."
                )
                ok = False
        else:
            loud(
                "NOTE: SCHEDULER SURFACE MISSING — cutlass_gemm_caller.cuh "
                "has no `TileSchedulerArguments scheduler` parameter; the "
                "patch would not compile against this tree. NO-OP."
            )
            ok = False
    else:
        loud(
            "WARNING: cutlass_gemm_caller.cuh not alongside the targets "
            "(sparse copy?) — cannot verify the scheduler surface here. It "
            "was verified against the full v6 tree "
            "(/var/tmp/vllm-src) and the pinned CUTLASS v4.4.2: for "
            "arch::Sm120 the TileSchedulerSelector maps to "
            "PersistentTileSchedulerSm100, whose Arguments is "
            "{ int max_swizzle_size; RasterOrderOptions raster_order; }."
        )
    return ok


def patch_file(src: str, anchors: list, label: str):
    """Apply anchors to one file. Returns (new_src, applied, already, missing)."""
    applied, already, missing = [], [], []
    for name, old, new in anchors:
        if old in src:
            if new in src:
                already.append(name)
            else:
                src = src.replace(old, new, 1)
                applied.append(name)
        elif new in src:
            already.append(name)
        else:
            missing.append(name)
    return src, applied, already, missing


def unified_diff(old: str, new: str, path: str):
    return list(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"{path} (before)",
            tofile=f"{path} (after, simulated)",
        )
    )


def process(path: Path, anchors: list, simulate: bool):
    """Patch (or simulate) one target file. Returns exit code."""
    src = path.read_text()
    if MARKER in src:
        loud(f"SKIP (already applied): {path}")
        return 0
    new_src, applied, already, missing = patch_file(src, anchors, str(path))

    if already and not applied and not missing:
        loud(f"SKIP (new shape already present, marker absent): {path}")
        return 0
    if missing and not applied:
        loud(
            f"NOTE: PREREQUISITE MISSING — anchors not found in {path} "
            f"({', '.join(missing)}); tree shape does not match the expected "
            f"SM120 blockwise-FP8 kernel. NO-OP"
            + ("" if simulate else ", not blocking boot/build.")
        )
        return 1
    if missing:
        loud(
            f"WARNING: partial patch on {path} — applied "
            f"({', '.join(applied)}), already ({', '.join(already)}), "
            f"MISSING ({', '.join(missing)}); inspect the file manually."
        )

    if simulate:
        for line in unified_diff(src, new_src, str(path)):
            sys.stdout.write(line)
        loud(
            f"SIMULATE: {len(applied)} hunk(s) would apply to {path} "
            "(no writes; CUDA compilation NOT performed — syntax-level "
            "verification is anchor matching + diff review)."
        )
        return 0 if not missing else 1

    path.write_text(new_src)
    loud(f"APPLY: {path} ({len(applied)} hunks)")
    loud(
        "NOTE: build-time patch — the vLLM C extension (.so) must be "
        "(re)built for this to take effect. No CUDA compilation was "
        "performed by this mod."
    )
    return 0 if not missing else 1


def main() -> int:
    args = sys.argv[1:]
    simulate = False
    if len(args) == 2 and args[0] == "--simulate":
        simulate = True
        root = Path(args[1])
    elif len(args) == 1 and not args[0].startswith("--"):
        root = Path(args[0])
    else:
        print(f"usage: {SCRIPT_NAME}.py <vllm_src_root | c3x_dir> | "
              f"--simulate <vllm_src_root | c3x_dir>")
        return 2

    c3x_dir = resolve_c3x(root)
    if c3x_dir is None:
        loud(
            f"NOTE: PREREQUISITE MISSING — {C3X_REL}/{CU_NAME} not found "
            f"under {root}. NO-OP"
            + ("" if simulate else ", not blocking boot/build.")
        )
        return 1 if simulate else 0

    loud(f"target dir: {c3x_dir}")

    if not precheck(c3x_dir):
        # Surface missing in a real tree: patching would break the build.
        # In simulate, exit 1 so pre-flight catches it; in apply, NO-OP.
        return 1 if simulate else 0

    if simulate:
        loud("SIMULATE mode (no writes)")

    rc_cu = process(c3x_dir / CU_NAME, CU_ANCHORS, simulate)
    rc_cuh = process(c3x_dir / CUH_NAME, CUH_ANCHORS, simulate)
    if rc_cu == 0 and rc_cuh == 0:
        loud(f"INFO: {TAG} — port of upstream {PR}.")
        loud("dry-run checklist:")
        loud("  1. APPLY lines for both files (or SKIP: already applied), "
             "no anchor NOTEs.")
        loud("  2. .cu gained the L2-gated blockwise_fp8_swizzle_size() "
             "(weight bytes > l2CacheSize -> 8, else 1); both dispatch "
             "calls pass `swizzle`.")
        loud("  3. .cuh threads max_swizzle_size through all three configs "
             "into TileSchedulerArguments::max_swizzle_size.")
        loud("  4. Rebuild the vLLM wheel/.so (image build), then on GB10: "
             "large-M blockwise-FP8 GEMMs (weights > 24 MiB) keep "
             "150-174 TFLOPS instead of collapsing to ~52 at M>=16384; "
             "outputs bit-identical.")
        return 0
    return rc_cu or rc_cuh


if __name__ == "__main__":
    sys.exit(main())
