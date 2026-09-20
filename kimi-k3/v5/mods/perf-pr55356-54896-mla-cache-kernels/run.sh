#!/usr/bin/env bash
# perf-pr55356-54896-mla-cache-kernels — BUILD-TIME port of upstream
# vllm-project/vllm#55356 + #54896 (same MLA cache kernel family) to the
# v0.26.1rc0+kimi.k3.aligned tree.
#
# #55356: concat_and_cache_mla_grouped gains plain-FP8 KV cache support
#   (per-layer kv_scales + kv_cache_dtype; kernel template
#   <scalar_t, cache_t, kv_dt>, fp8::scaled_convert store, bf16 fast path
#   unchanged). Schema args have defaults, so the existing bf16 caller
#   (dspark_mla fused context-KV insert) keeps working unchanged.
# #54896: MLA decode concat/cache epilogue restructure — writeLatent576
#   (lane, lane_stride) generalisation, SPLIT=3 warps per 576-wide row for
#   <= 64 decode tokens, non-dependent inputs read before the PDL
#   grid-dependency wait (cache-slot warps skip the wait entirely),
#   launchPdlSlots helper (ported; our tree only had launchPdl).
#
# Dropped as out of scope / not-portable:
#   - upstream test files (test_cache.py, test_kimi_k3_mla_fused_epilogue.py)
#   - vllm/models/kimi_k3/nvidia/dspark_mla.py model-side fp8 wiring
#     (upstream also moved _build_fused_context_kv_metadata into
#     process_weights_after_loading — depends on newer upstream code shape;
#     the fp8 grouped path is exposed at op level but not yet wired into the
#     model; bf16 grouped path is unchanged via schema defaults)
#
# This is a BUILD-TIME patch mod: the .cu/.h/.cpp changes require the vLLM
# C++ extension (.so) to be recompiled. Intended flow: bake/apply this mod
# in the image build BEFORE the extension compile step (patch
# /opt/kimi-k3/vllm/csrc/... + vllm/_custom_ops.py, then the build
# recompiles). CUDA is NOT compiled here (no nvcc on this host).
#
# Behaviour:
#   - source tree found (build time, or editable install): patch all target
#     files, idempotent via marker; missing anchors NOTE and exit 0.
#   - source tree found but the already-built .so has the OLD schema and
#     torch can resolve it (container start on an unbaked image): csrc is
#     patched, but vllm/_custom_ops.py is NOT (--skip-python) so the Python
#     wrapper cannot outrun the stale extension; LOUD note to rebuild.
#   - no source tree (runtime-only install): if the built .so schema (or the
#     marker in the installed _custom_ops.py) already contains the change,
#     SKIP loudly; otherwise NOTE that this is a build-time mod and exit 0.
#
# usage: run.sh [apply|simulate]
#   apply    (default) patch the vLLM source tree in place
#   simulate never write; patch temp copies of all 5 targets, py_compile the
#            .py, print unified diffs (falls back to /var/tmp/v6-csrc when
#            no source tree is resolvable — useful on the head node)
set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-apply}"

if [ "$MODE" != "apply" ] && [ "$MODE" != "simulate" ]; then
  echo "usage: run.sh [apply|simulate]"
  exit 2
fi

# ── Resolve the vLLM SOURCE root (must contain csrc/ + vllm/) ────────────────
VLLM_SRC=""
for cand in "${VLLM_SRC_ROOT:-}" /opt/kimi-k3/vllm /workspace/vllm; do
  if [ -n "$cand" ] && [ -f "$cand/csrc/libtorch_stable/cache_kernels.cu" ]; then
    VLLM_SRC="$cand"
    break
  fi
done
if [ -z "$VLLM_SRC" ]; then
  pkg_dir="$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)"
  if [ -n "$pkg_dir" ] && [ -f "$pkg_dir/../csrc/libtorch_stable/cache_kernels.cu" ]; then
    VLLM_SRC="$(cd "$pkg_dir/.." && pwd)"
  fi
fi

# ── Probe the built extension schema (best-effort; empty if unresolvable) ───
# "new"  -> .so already contains the #55356 schema change
# "old"  -> .so resolvable and lacks the change
# ""     -> torch/ops not loadable here (typical at image-build time)
SCHEMA_STATE="$(python3 -c "
import torch
try:
    s = str(torch.ops._C_cache_ops.concat_and_cache_mla_grouped.default._schema)
except Exception:
    print(''); raise SystemExit
print('new' if 'kv_scales' in s else 'old')" 2>/dev/null)"

# ── simulate ─────────────────────────────────────────────────────────────────
if [ "$MODE" = "simulate" ]; then
  SIM_ROOT="$VLLM_SRC"
  if [ -z "$SIM_ROOT" ] && [ -f /var/tmp/v6-csrc/csrc/libtorch_stable/cache_kernels.cu ]; then
    SIM_ROOT=/var/tmp/v6-csrc
  fi
  if [ -z "$SIM_ROOT" ]; then
    echo "=====> [perf-pr55356-54896-mla-cache-kernels] NOTE: PREREQUISITE MISSING — no vLLM source root resolvable and no /var/tmp/v6-csrc fallback."
    exit 1
  fi
  echo "=====> [perf-pr55356-54896-mla-cache-kernels] SIMULATE mode (no writes)"
  python3 "$SCRIPT_DIR/patch_mla_cache_kernels.py" --simulate "$SIM_ROOT"
  exit $?
fi

# ── apply ────────────────────────────────────────────────────────────────────
if [ -n "$VLLM_SRC" ]; then
  echo "=====> [perf-pr55356-54896-mla-cache-kernels] VLLM_SRC=$VLLM_SRC (schema probe: '${SCHEMA_STATE:-unresolvable}')"
  if [ "$SCHEMA_STATE" = "new" ]; then
    echo "=====> [perf-pr55356-54896-mla-cache-kernels] SKIP loudly: the built extension already contains the patched schema (kv_scales present). Source markers keep it idempotent."
  fi
  EXTRA_ARGS=()
  if [ "$SCHEMA_STATE" = "old" ]; then
    echo "=====> [perf-pr55356-54896-mla-cache-kernels] WARNING: built .so has the OLD schema but a source tree is present — patching csrc ONLY (--skip-python) so the Python wrapper cannot outrun the stale extension. This is a BUILD-TIME mod: rebuild the extension (bake the mod into the image) before serving."
    EXTRA_ARGS+=(--skip-python)
  fi
  python3 "$SCRIPT_DIR/patch_mla_cache_kernels.py" "$VLLM_SRC" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
  rc=$?
  echo "=====> [perf-pr55356-54896-mla-cache-kernels] reminder: CUDA changes need an extension rebuild — this mod does NOT compile CUDA itself (no nvcc here)."
  exit $rc
fi

# ── apply, no source tree (container-start safety check) ─────────────────────
pkg_dir="$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)"
if [ -n "$pkg_dir" ] && grep -q "perf-pr55356-54896-mla-cache-kernels: applied" "$pkg_dir/_custom_ops.py" 2>/dev/null; then
  echo "=====> [perf-pr55356-54896-mla-cache-kernels] SKIP loudly: marker present in installed $pkg_dir/_custom_ops.py — image already baked with this mod."
  exit 0
fi
if [ "$SCHEMA_STATE" = "new" ]; then
  echo "=====> [perf-pr55356-54896-mla-cache-kernels] SKIP loudly: built extension schema already contains kv_scales — change is compiled in."
  exit 0
fi
echo "=====> [perf-pr55356-54896-mla-cache-kernels] NOTE: no vLLM source tree (csrc/) present and the compiled change was not detected. This is a BUILD-TIME mod: apply it at image-build time (before the extension compile) and rebuild. NO-OP, not blocking boot."
exit 0
