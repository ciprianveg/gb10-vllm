#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V4PLUS CGFIX — upstream fork PR #617: synchronize auxiliary warmup streams.

Ports local-inference-lab/vllm #617 (commit 7cf2edd45, Sep 3 2026) onto the
v4-plus-b4 image lineage.  ROOT CAUSE it fixes: model forwards fork kernels
onto auxiliary streams and enqueue only event waits; CUDA graph capture must
not begin while an uncaptured auxiliary kernel still owns temporary allocator
storage — starting capture at that boundary races allocator reuse and
surfaces as a cudaErrorIllegalAddress at the capture-begin boundary (the
stale async fault is polled at is_current_stream_capturing).  This is the
crash-at-capture-begin signature with cudagraph_capture_sizes [1,2,4,8],
DSpark spec + B12X_MLA + marlin MXFP4.

What this script applies (against the REAL b4-image extracts in
/tmp/opencode/k3spec/cgfix):

1. cudagraph_utils.py, capture loop — VERBATIM #617: after every uncaptured
   per-descriptor warmup forward (``forward_fn(CUDAGraphMode.NONE)`` at the
   "# Warmup" site), ``torch.accelerator.synchronize()`` before the capture
   section begins.  The upstream anchor context ("# Warmup" / blank /
   "# Capture" / logger.debug) exists byte-for-byte in our tree.

2. cudagraph_utils.py, B12X prewarm site — LINEAGE-ADAPTED #617: our tree
   has a SECOND uncaptured forward the upstream tree lacks — the B12X
   prewarm (``b12x_cuda_graph_prewarm_enabled()`` ->
   ``forward_fn(CUDAGraphMode.NONE)``) that re-warms the exact fresh state
   FULL capture will use, immediately before ``torch.cuda.CUDAGraph()``.
   The same auxiliary-stream completion requirement applies: it gets the
   same synchronize.

3. compilation/monitor.py — VERBATIM #617: expose
   ``is_cudagraph_capturing_enabled()`` so model-specific auxiliary-stream
   code can distinguish uncaptured warmup from regular serving and FULL
   capture.  No caller in our tree yet (the consumers are aux-stream
   overlap features we do not carry — see the #619 finding in run.sh);
   ported as the upstream API surface.

NOT ported from the same PR: the +65-line test
(tests/v1/cudagraph/test_cudagraph_manager.py) — the image tree ships no
tests directory.

PREREQUISITE (fail loud): the target files must carry the kimi.k3.aligned
lineage markers (the b12x_capture import and the CudaGraphManager capture
descriptor structure).  This script refuses to patch a foreign lineage.

Idempotent: every hunk is skipped when its marker is already present.
A missing anchor prints a NOTE and skips that hunk; only a missing
prerequisite, file-not-found, or a broken post-patch compile exits
non-zero.
"""

from __future__ import annotations

import os
import py_compile
import sys

SCRIPT_NAME = "patch_cudagraph_sync"
TAG = "# V4PLUS-CGFIX (fork #617: synchronize auxiliary warmup streams)"

VLLM_ROOT = os.environ.get("VLLM_ROOT", "/opt/kimi-k3/vllm/vllm")
CUDAGRAPH_UTILS = os.path.join(VLLM_ROOT, "v1", "worker", "gpu", "cudagraph_utils.py")
MONITOR = os.path.join(VLLM_ROOT, "compilation", "monitor.py")


def apply_hunks(path: str, hunks: list[tuple[str, str, str, str]]) -> bool:
    """Apply (name, anchor, replacement, present) hunks; True if all well."""
    try:
        with open(path) as f:
            src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {path} not found", file=sys.stderr)
        return False
    changed = False
    ok = True
    for name, anchor, repl, present in hunks:
        if present in src:
            print(f"[{SCRIPT_NAME}] SKIP  {os.path.basename(path)}: {name} (already present)")
            continue
        n = src.count(anchor)
        if n != 1:
            print(
                f"[{SCRIPT_NAME}] NOTE  {os.path.basename(path)}: {name} — "
                f"anchor found {n}x (want 1); hunk skipped"
            )
            ok = False
            continue
        src = src.replace(anchor, repl, 1)
        changed = True
        print(f"[{SCRIPT_NAME}] APPLY {os.path.basename(path)}: {name}")
    if changed:
        try:
            compile(src, path, "exec")
        except SyntaxError as exc:
            print(
                f"[{SCRIPT_NAME}] ERROR: {path} does not compile after patch: {exc}",
                file=sys.stderr,
            )
            return False
        with open(path, "w") as f:
            f.write(src)
        py_compile.compile(path, doraise=True)
    return ok


def check_prerequisites() -> bool:
    """Fail loud unless the targets carry the kimi.k3.aligned lineage."""
    ok = True
    try:
        with open(CUDAGRAPH_UTILS) as f:
            src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {CUDAGRAPH_UTILS} not found", file=sys.stderr)
        return False
    if "from vllm.compilation.b12x_capture import (" not in src:
        print(
            f"[{SCRIPT_NAME}] ERROR: {CUDAGRAPH_UTILS} lacks the "
            "vllm.compilation.b12x_capture import — this is not the "
            "kimi.k3.aligned lineage the CGFIX mod targets.",
            file=sys.stderr,
        )
        ok = False
    if (
        "self._capture_descs: dict[CUDAGraphMode, list[BatchExecutionDescriptor]] = {}"
        not in src
    ):
        print(
            f"[{SCRIPT_NAME}] ERROR: {CUDAGRAPH_UTILS} lacks the "
            "CudaGraphManager._capture_descs structure the CGFIX hunks "
            "anchor against.",
            file=sys.stderr,
        )
        ok = False
    try:
        with open(MONITOR) as f:
            mon_src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {MONITOR} not found", file=sys.stderr)
        return False
    if "def set_cudagraph_capturing_enabled(enabled: bool) -> None:" not in mon_src:
        print(
            f"[{SCRIPT_NAME}] ERROR: {MONITOR} lacks "
            "set_cudagraph_capturing_enabled — unexpected lineage.",
            file=sys.stderr,
        )
        ok = False
    return ok


# ---------------------------------------------------------------------------
# cudagraph_utils.py — capture loop: synchronize after the warmup forward
# (VERBATIM #617; the anchor context matches the upstream patch byte-for-byte)
# ---------------------------------------------------------------------------

CU_SYNC_WARMUP_ANCHOR = (
    "                    # Warmup\n"
    "                    forward_fn(CUDAGraphMode.NONE)\n"
    "\n"
    "                    # Capture\n"
)
CU_SYNC_WARMUP_REPLACEMENT = (
    "                    # Warmup\n"
    "                    forward_fn(CUDAGraphMode.NONE)\n"
    "                    # A model forward may fork work onto auxiliary streams and\n"
    "                    # join them with events queued on the compute stream.  CUDA\n"
    "                    # graph capture must not begin while those warmup kernels\n"
    "                    # are still executing, even though the queued event waits\n"
    "                    # preserve normal stream ordering.\n"
    "                    torch.accelerator.synchronize()\n"
    "\n"
    "                    # Capture\n"
)
CU_SYNC_WARMUP_PRESENT = (
    "                    # A model forward may fork work onto auxiliary streams and\n"
    "                    # join them with events queued on the compute stream."
)

# ---------------------------------------------------------------------------
# cudagraph_utils.py — B12X prewarm site: same synchronize before
# torch.cuda.CUDAGraph() (LINEAGE-ADAPTED #617: this uncaptured forward is
# specific to our tree)
# ---------------------------------------------------------------------------

CU_SYNC_PREWARM_ANCHOR = (
    "                        if b12x_cuda_graph_prewarm_enabled():\n"
    "                            # B12X kernels use caller-owned scratch views in\n"
    "                            # the CuTe launcher contract. Re-warm the exact\n"
    "                            # fresh state that FULL capture will use, so CUDA\n"
    "                            # graph capture only records resolved launches.\n"
    "                            forward_fn(CUDAGraphMode.NONE)\n"
    "                        graph = torch.cuda.CUDAGraph()\n"
)
CU_SYNC_PREWARM_REPLACEMENT = (
    "                        if b12x_cuda_graph_prewarm_enabled():\n"
    "                            # B12X kernels use caller-owned scratch views in\n"
    "                            # the CuTe launcher contract. Re-warm the exact\n"
    "                            # fresh state that FULL capture will use, so CUDA\n"
    "                            # graph capture only records resolved launches.\n"
    "                            forward_fn(CUDAGraphMode.NONE)\n"
    "                            # V4PLUS-CGFIX (#617, lineage-adapted): the B12X\n"
    "                            # prewarm forward is also uncaptured; the same\n"
    "                            # auxiliary-stream completion requirement applies\n"
    "                            # before capture begins.\n"
    "                            torch.accelerator.synchronize()\n"
    "                        graph = torch.cuda.CUDAGraph()\n"
)
CU_SYNC_PREWARM_PRESENT = (
    "                            # V4PLUS-CGFIX (#617, lineage-adapted): the B12X"
)

# ---------------------------------------------------------------------------
# compilation/monitor.py — expose the graph-preparation state (VERBATIM #617)
# ---------------------------------------------------------------------------

MONITOR_GETTER_ANCHOR = (
    "def set_cudagraph_capturing_enabled(enabled: bool) -> None:\n"
    "    global cudagraph_capturing_enabled\n"
    "    cudagraph_capturing_enabled = enabled\n"
)
MONITOR_GETTER_REPLACEMENT = (
    "def set_cudagraph_capturing_enabled(enabled: bool) -> None:\n"
    "    global cudagraph_capturing_enabled\n"
    "    cudagraph_capturing_enabled = enabled\n"
    "\n"
    "\n"
    "def is_cudagraph_capturing_enabled() -> bool:\n"
    '    """Return whether the model runner is preparing or capturing CUDA graphs."""\n'
    "    return cudagraph_capturing_enabled\n"
)
MONITOR_GETTER_PRESENT = "def is_cudagraph_capturing_enabled() -> bool:"


def main() -> int:
    print(f"[{SCRIPT_NAME}] {TAG}")
    print(f"[{SCRIPT_NAME}] VLLM_ROOT={VLLM_ROOT}")
    if not check_prerequisites():
        print(
            f"[{SCRIPT_NAME}] PREREQUISITE FAILED: the target files are not "
            "the expected kimi.k3.aligned lineage; refusing to patch.",
            file=sys.stderr,
        )
        return 1

    ok = True
    ok &= apply_hunks(
        CUDAGRAPH_UTILS,
        [
            (
                "synchronize after warmup forward (#617 verbatim)",
                CU_SYNC_WARMUP_ANCHOR,
                CU_SYNC_WARMUP_REPLACEMENT,
                CU_SYNC_WARMUP_PRESENT,
            ),
            (
                "synchronize after B12X prewarm forward (#617 lineage-adapted)",
                CU_SYNC_PREWARM_ANCHOR,
                CU_SYNC_PREWARM_REPLACEMENT,
                CU_SYNC_PREWARM_PRESENT,
            ),
        ],
    )
    ok &= apply_hunks(
        MONITOR,
        [
            (
                "is_cudagraph_capturing_enabled getter (#617 verbatim)",
                MONITOR_GETTER_ANCHOR,
                MONITOR_GETTER_REPLACEMENT,
                MONITOR_GETTER_PRESENT,
            ),
        ],
    )

    if ok:
        print(
            f"[{SCRIPT_NAME}] NOTE: the synchronize is a full device sync "
            "after every uncaptured warmup/prewarm forward — capture "
            "startup gets slower by one sync per descriptor (microseconds "
            "each); serving throughput is unaffected."
        )
        print(
            f"[{SCRIPT_NAME}] NOTE: is_cudagraph_capturing_enabled() has no "
            "caller in this tree yet (its consumers are aux-stream overlap "
            "features we do not carry); it is ported as the upstream API "
            "surface for future aux-stream code."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
