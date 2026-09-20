#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V4PLUS CGFIX — upstream fork PR #628 (functional): register speculative
CUDA-graph row counts for the B12X warmup contract.

Ports the FUNCTION of local-inference-lab/vllm #628 (commit cbb66bdff,
Sep 3 2026) onto the v4-plus-b4 image lineage.  ROOT CAUSE it fixes
upstream: B12X launch plans must exist for every model-row count a CUDA
graph executes before capture begins; speculative decode graph descriptors
carry verifier-row counts (batch x (nst+1)) that the scheduler
capture-size list never registers, so upstream's centralized B12X warmup
(prepared from the scheduler list only) left those counts unplanned and a
lazy first-use path ran INSIDE capture (fork issue #451 contract: every
B12X route/shape must initialize outside graph capture — lazy init inside
capture takes buffers from the graph-private pool -> corruption/illegal
access).

LINEAGE ANALYSIS (why this port differs from upstream's diff):
  * Upstream's consumer file vllm/model_executor/warmup/b12x_warmup.py
    DOES NOT EXIST in our tree (no model_executor/warmup/ directory at
    all — our Build #7 lineage is older).  Upstream's own raw patch in
    hand only carries the test + docstring side; planned_token_counts()
    already existed there.
  * Our lineage decentralizes the same contract at the capture site:
    CudaGraphManager.capture() runs an uncaptured warmup forward for
    EVERY staged descriptor (which already includes the verifier row
    counts — decode descriptors round each capture size up to
    decode_query_len = nst+1, plus the 1..32-request small-batch grid),
    the optional B12X prewarm re-warms the exact fresh capture state,
    guard_b12x_kernel_resolution() freezes b12x kernel resolution across
    the captured region, and the B12X_MLA backend fail-louds on any
    uncompiled layout at capture time.  The structural
    "scheduler-list-only" gap #628 fixes therefore cannot occur here —
    but nothing exposes the staged row counts, and nothing states the
    contract at the capture boundary.

What this script applies (against the REAL b4-image extracts):

1. CudaGraphManager.planned_token_counts() — the #628 API, reconstructed
   exactly from the upstream test semantics (sorted, unique num_tokens
   across ALL staged capture descriptors — spec verify shapes included;
   verified against the upstream test vector: capture sizes
   [1,2,4,8,16,24] with decode_query_len=6, nst=5 -> [1,2,4,6,8,12,16,18,
   24], and 12 absent for a target-only manager).

2. CudaGraphManager.capture() — the registration point for OUR lineage:
   before any capture begins, log the full planned row-count set with the
   #451 contract.  This is the boot-time artifact proving which row
   counts the per-descriptor warmups must cover (including the
   batch x (nst+1) verifier shapes), and the hook any future centralized
   warmup consumes planned_token_counts() from.

PREREQUISITE (fail loud): the kimi.k3.aligned lineage markers, and (for a
clean apply) that #617's synchronize hunks are either applied or absent —
this script's anchors do not overlap #617's regions, so either order
works; run patch_cudagraph_sync.py first for the documented pairing.

Idempotent: every hunk is skipped when its marker is already present.
A missing anchor prints a NOTE and skips that hunk; only a missing
prerequisite, file-not-found, or a broken post-patch compile exits
non-zero.
"""

from __future__ import annotations

import os
import py_compile
import sys

SCRIPT_NAME = "patch_planned_token_counts"
TAG = "# V4PLUS-CGFIX (fork #628 functional: planned token counts)"

VLLM_ROOT = os.environ.get("VLLM_ROOT", "/opt/kimi-k3/vllm/vllm")
CUDAGRAPH_UTILS = os.path.join(VLLM_ROOT, "v1", "worker", "gpu", "cudagraph_utils.py")


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
    """Fail loud unless the target carries the kimi.k3.aligned lineage."""
    try:
        with open(CUDAGRAPH_UTILS) as f:
            src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {CUDAGRAPH_UTILS} not found", file=sys.stderr)
        return False
    ok = True
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
            "CudaGraphManager._capture_descs structure this port anchors "
            "against.",
            file=sys.stderr,
        )
        ok = False
    if "def planned_token_counts" in src and "V4PLUS-CGFIX" not in src:
        # A foreign planned_token_counts already exists (a NEWER upstream
        # lineage): refuse rather than double-define.
        print(
            f"[{SCRIPT_NAME}] ERROR: {CUDAGRAPH_UTILS} already carries a "
            "planned_token_counts() that is not ours — this lineage "
            "likely already includes #628; refusing to double-define.",
            file=sys.stderr,
        )
        ok = False
    return ok


# ---------------------------------------------------------------------------
# CudaGraphManager.planned_token_counts() — the #628 API
# ---------------------------------------------------------------------------

CU_PTC_ANCHOR = (
    "    def needs_capture(self) -> bool:\n"
    "        return len(self._capture_descs) > 0\n"
)
CU_PTC_REPLACEMENT = (
    "    def needs_capture(self) -> bool:\n"
    "        return len(self._capture_descs) > 0\n"
    "\n"
    "    def planned_token_counts(self) -> list[int]:\n"
    "        \"\"\"Return model-row counts staged for decoder graph capture.\n"
    "\n"
    "        V4PLUS-CGFIX (upstream #628): every ``num_tokens`` across the\n"
    "        staged capture descriptors — including the speculative\n"
    "        verifier row counts (batch x (nst + 1)) that the scheduler\n"
    "        capture-size list never registers.  B12X plan warmup must\n"
    "        cover every count here BEFORE capture begins (fork issue\n"
    "        #451 contract: no B12X lazy initialization inside a captured\n"
    "        graph).\n"
    "\n"
    "        Returns:\n"
    "            Sorted, unique ``num_tokens`` values from the capture\n"
    "            descriptors.\n"
    "        \"\"\"\n"
    "        return sorted(\n"
    "            {\n"
    "                desc.num_tokens\n"
    "                for descs in self._capture_descs.values()\n"
    "                for desc in descs\n"
    "            }\n"
    "        )\n"
)
CU_PTC_PRESENT = "    def planned_token_counts(self) -> list[int]:"

# ---------------------------------------------------------------------------
# CudaGraphManager.capture() — the registration point: log the planned
# row-count set (with the #451 contract) before any capture begins
# ---------------------------------------------------------------------------

CU_REGISTER_ANCHOR = (
    "        # Keep event handles created by descriptor warmups alive together with\n"
    "        # the graph artifacts captured below. Some multi-stream custom ops run\n"
    "        # on joined auxiliary streams where CUDA's per-current-stream capture\n"
    "        # query is false even though later graph nodes retain those handles.\n"
    "        with (\n"
    "            graph_capture(device=self.device, channel_id=channel_id),\n"
    "            vllm_cudagraph_capture_scope(),\n"
    "        ):\n"
)
CU_REGISTER_REPLACEMENT = (
    "        # V4PLUS-CGFIX (upstream #628, lineage-adapted registration):\n"
    "        # every model-row count the graphs below will execute — the\n"
    "        # scheduler capture sizes AND the speculative verifier shapes\n"
    "        # batch x (nst + 1).  The per-descriptor warmup/prewarm\n"
    "        # forwards in this loop must resolve every B12X plan for these\n"
    "        # counts BEFORE the captured forward runs (fork issue #451\n"
    "        # contract: B12X lazy initialization inside a captured graph\n"
    "        # takes buffers from the graph-private pool and corrupts\n"
    "        # memory).  planned_token_counts() is the hook a centralized\n"
    "        # warmup consumes; in this lineage the warmups below are the\n"
    "        # registration.\n"
    "        planned_counts = self.planned_token_counts()\n"
    "        if planned_counts:\n"
    "            logger.info(\n"
    "                \"CUDA graph capture preparing model-row counts %s; the \"\n"
    "                \"per-descriptor warmup forwards must resolve every B12X \"\n"
    "                \"plan for these counts before capture begins.\",\n"
    "                planned_counts,\n"
    "            )\n"
    "        # Keep event handles created by descriptor warmups alive together with\n"
    "        # the graph artifacts captured below. Some multi-stream custom ops run\n"
    "        # on joined auxiliary streams where CUDA's per-current-stream capture\n"
    "        # query is false even though later graph nodes retain those handles.\n"
    "        with (\n"
    "            graph_capture(device=self.device, channel_id=channel_id),\n"
    "            vllm_cudagraph_capture_scope(),\n"
    "        ):\n"
)
CU_REGISTER_PRESENT = (
    "        planned_counts = self.planned_token_counts()"
)


def main() -> int:
    print(f"[{SCRIPT_NAME}] {TAG}")
    print(f"[{SCRIPT_NAME}] VLLM_ROOT={VLLM_ROOT}")
    if not check_prerequisites():
        print(
            f"[{SCRIPT_NAME}] PREREQUISITE FAILED: the target file is not "
            "the expected kimi.k3.aligned lineage; refusing to patch.",
            file=sys.stderr,
        )
        return 1

    ok = apply_hunks(
        CUDAGRAPH_UTILS,
        [
            (
                "CudaGraphManager.planned_token_counts() (#628 API)",
                CU_PTC_ANCHOR,
                CU_PTC_REPLACEMENT,
                CU_PTC_PRESENT,
            ),
            (
                "capture() planned-count registration (#628 lineage-adapted)",
                CU_REGISTER_ANCHOR,
                CU_REGISTER_REPLACEMENT,
                CU_REGISTER_PRESENT,
            ),
        ],
    )

    if ok:
        print(
            f"[{SCRIPT_NAME}] NOTE: our lineage has no centralized B12X "
            "warmup to union these counts into (upstream's "
            "model_executor/warmup/b12x_warmup.py does not exist here); "
            "the per-descriptor warmup/prewarm forwards in capture() ARE "
            "the registration, and the B12X_MLA backend fail-louds on any "
            "uncompiled layout at capture time. planned_token_counts() + "
            "the capture-start log line make the covered row-count set "
            "visible at boot."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
