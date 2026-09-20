#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V4PLUS CGFIX3 — run the DSpark draft speculator EAGERLY, keeping the
TARGET model's FULL CUDA graphs.

Boot evidence (v6 = b5 + cgfix2 eager-context-KV): every capture phase
completes cleanly (target FULL graphs, DSpark speculator graphs 2/2), then
the first post-capture ``warmup_kernels`` decode step faults — fork issue
#396: the replay of the speculator's own captured draft-decode FULL graph
crashes (``_multi_step_decode -> run_fullgraph -> illegal access``).  The
issue is open and unfixed upstream, and the fork's own K3 production recipe
still runs ``--enforce-eager``.  Pragmatic fix: the TARGET model keeps its
FULL graphs (they capture cleanly and dominate step time at nst<=6); the
DRAFT speculator runs fully eagerly.

The change (one gate, reusing proven machinery): in
``init_cudagraph_manager``, ``VLLM_K3_EAGER_DRAFT`` (default "1") forces
``wants_full = False`` BEFORE the mode selection.  That selects the exact
configuration the existing "draft attention does not support full CUDA
graphs; running the draft eagerly" path already produces in production
for backends without UNIFORM_BATCH support:

  * ``query_cudagraph_manager`` is still CONSTRUCTED (non-None — no
    None-safety surface is created) but with ``CUDAGraphMode.NONE``;
  * ``CudaGraphManager._init_candidates`` returns early for a falsy mode
    (``if not (self.cudagraph_mode and capture_sizes): return``), so no
    capture descriptors are staged, ``needs_capture()`` is False, and
    ``capture()`` is a no-op — the "Capturing dspark CUDA graphs" phase
    disappears from the boot;
  * ``CudaGraphManager.dispatch`` never finds captured graphs
    (``self._graphs_captured`` guards the candidate lookup) and returns a
    ``CUDAGraphMode.NONE`` descriptor, so the draft decode runs the eager
    branch — ``run_fullgraph`` is unreachable;
  * the cgfix2 context-KV manager gate also reads ``wants_full``, so an
    eager draft keeps the speculator graph-free end to end (neither decode
    nor context-KV graphs), regardless of
    VLLM_K3_DISABLE_CONTEXT_GRAPHS.

Env semantics — ``VLLM_K3_EAGER_DRAFT`` (direct ``os.getenv``, like the
cgfix2 knob; ``import os`` is already present from cgfix2):
  * "1" (DEFAULT): draft speculator EAGER, target keeps FULL graphs.  Live
    by default because the speculator graph replay crashes the first
    decode step (fork #396).
  * "0" / "false" / "False": restore the full-graph speculator (A/B only
    on images where #396 is fixed).

A loud INFO fires at init whenever the gate actually downgrades a FULL
request: "DSpark draft speculator runs EAGERLY (VLLM_K3_EAGER_DRAFT=1);
target model keeps FULL CUDA graphs."

PREREQUISITE (fail loud): speculator.py must carry the cgfix2 post-state
markers (``VLLM_K3_DISABLE_CONTEXT_GRAPHS`` + ``import os``) and the
DFlash/DSpark lineage markers.  This mod builds on v6 = b5 + cgfix2.

GROUND-TRUTH CAVEAT: the /tmp/opencode/k3spec extracts were wiped before
this task was started.  The anchor below is transcribed BYTE-EXACT from
the cgfix2-session read of the real b5 file (a region cgfix2 does not
modify), and the simulation in the report runs against a labeled
RECONSTRUCTION of the v6 post-state — re-extract speculator.py from the
image and re-run this script against it before baking (the fail-loud
prereqs and the anchor-count guard make a mismatched real file refuse to
patch rather than corrupt it).

Idempotent: the hunk is skipped when its marker is already present.  A
missing anchor prints a NOTE and skips; only a missing prerequisite,
file-not-found, or a broken post-patch compile exits non-zero.
"""

from __future__ import annotations

import os
import py_compile
import sys

SCRIPT_NAME = "patch_eager_draft"
TAG = "# V4PLUS-CGFIX3 (eager DSpark draft, target keeps FULL graphs)"

VLLM_ROOT = os.environ.get("VLLM_ROOT", "/opt/kimi-k3/vllm/vllm")
SPECULATOR = os.path.join(
    VLLM_ROOT, "v1", "worker", "gpu", "spec_decode", "dflash", "speculator.py"
)


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
    """Fail loud unless speculator.py is the v6 (b5 + cgfix2) post-state."""
    try:
        with open(SPECULATOR) as f:
            src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {SPECULATOR} not found", file=sys.stderr)
        return False
    ok = True
    # cgfix2 post-state markers (this mod builds on v6).
    for marker in (
        "VLLM_K3_DISABLE_CONTEXT_GRAPHS",
        "import copy\nimport os\n",
    ):
        if marker not in src:
            print(
                f"[{SCRIPT_NAME}] ERROR: {SPECULATOR} lacks the cgfix2 "
                f"marker {marker!r} — apply mods/v4plus-cgfix2 first; this "
                "mod builds on the v6 post-state.",
                file=sys.stderr,
            )
            ok = False
    # DFlash/DSpark lineage markers.
    for marker in (
        "DFlashCudaGraphManager",
        "DFlashContextCudaGraphManager",
        "def _precompute_context_kv(",
        "precompute_and_store_context_kv",
        "envs.VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH",
    ):
        if marker not in src:
            print(
                f"[{SCRIPT_NAME}] ERROR: {SPECULATOR} lacks the lineage "
                f"marker {marker!r} — this is not the DFlash/DSpark "
                "speculator lineage this mod targets.",
                file=sys.stderr,
            )
            ok = False
    return ok


# ---------------------------------------------------------------------------
# init_cudagraph_manager: force wants_full off for the eager draft.
# Anchor transcribed byte-exact from the real b5 file (cgfix2 does not
# modify this region).
# ---------------------------------------------------------------------------

S_GATE_ANCHOR = (
    "    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:\n"
    "        wants_full = cudagraph_mode.decode_mode() == CUDAGraphMode.FULL\n"
    "        supports_full = (\n"
    "            self.attn_cg_support.min_cg_support.value\n"
    "            >= AttentionCGSupport.UNIFORM_BATCH.value\n"
    "        )\n"
    "        if wants_full and not supports_full:\n"
    "            logger.warning(\n"
    '                "%s draft attention (%s) does not support full CUDA graphs; "\n'
    '                "running the draft eagerly.",\n'
    "                self._speculator_name,\n"
    "                self.attn_cg_support.min_cg_attn_backend,\n"
    "            )\n"
    "        # PIECEWISE cudagraphs are not supported for dflash.\n"
    "        if wants_full and supports_full:\n"
    "            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY\n"
    "        else:\n"
    "            cudagraph_mode = CUDAGraphMode.NONE\n"
)
S_GATE_REPLACEMENT = (
    "    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:\n"
    "        wants_full = cudagraph_mode.decode_mode() == CUDAGraphMode.FULL\n"
    "        supports_full = (\n"
    "            self.attn_cg_support.min_cg_support.value\n"
    "            >= AttentionCGSupport.UNIFORM_BATCH.value\n"
    "        )\n"
    "        # V4PLUS-CGFIX3: fork issue #396 — the replay of the\n"
    "        # speculator's own captured draft-decode FULL graph faults in\n"
    "        # the first post-capture warmup decode step\n"
    "        # (_multi_step_decode -> run_fullgraph -> illegal access);\n"
    "        # open and unfixed upstream, and the fork's own K3 production\n"
    "        # recipe still runs --enforce-eager. VLLM_K3_EAGER_DRAFT\n"
    "        # selects the mode: \"1\" (default) = the draft speculator\n"
    "        # runs fully EAGERLY (no speculator graphs at all — neither\n"
    "        # decode nor context-KV; forcing wants_full off also keeps\n"
    "        # the cgfix2 context manager unconstructed) while the TARGET\n"
    "        # model keeps its FULL CUDA graphs. \"0\" restores the\n"
    "        # full-graph speculator for A/B on images where #396 is\n"
    "        # fixed. Forcing wants_full off selects the exact\n"
    "        # configuration the not-supports-full path below already\n"
    "        # produces in production: the query manager is constructed\n"
    "        # with mode NONE, stages no descriptors, captures nothing,\n"
    "        # and dispatch always returns NONE — the eager draft branch.\n"
    "        eager_draft = os.getenv(\"VLLM_K3_EAGER_DRAFT\", \"1\") not in (\n"
    "            \"0\",\n"
    "            \"false\",\n"
    "            \"False\",\n"
    "        )\n"
    "        if eager_draft and wants_full:\n"
    "            logger.info(\n"
    "                \"%s draft speculator runs EAGERLY (VLLM_K3_EAGER_DRAFT=1); \"\n"
    "                \"target model keeps FULL CUDA graphs.\",\n"
    "                self._speculator_name,\n"
    "            )\n"
    "            wants_full = False\n"
    "        if wants_full and not supports_full:\n"
    "            logger.warning(\n"
    '                "%s draft attention (%s) does not support full CUDA graphs; "\n'
    '                "running the draft eagerly.",\n'
    "                self._speculator_name,\n"
    "                self.attn_cg_support.min_cg_attn_backend,\n"
    "            )\n"
    "        # PIECEWISE cudagraphs are not supported for dflash.\n"
    "        if wants_full and supports_full:\n"
    "            cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY\n"
    "        else:\n"
    "            cudagraph_mode = CUDAGraphMode.NONE\n"
)
S_GATE_PRESENT = "VLLM_K3_EAGER_DRAFT"


def main() -> int:
    print(f"[{SCRIPT_NAME}] {TAG}")
    print(f"[{SCRIPT_NAME}] VLLM_ROOT={VLLM_ROOT}")
    if not check_prerequisites():
        print(
            f"[{SCRIPT_NAME}] PREREQUISITE FAILED: speculator.py is not the "
            "expected v6 (b5 + cgfix2) post-state; refusing to patch.",
            file=sys.stderr,
        )
        return 1

    ok = apply_hunks(
        SPECULATOR,
        [
            (
                "eager-draft gate (wants_full override + INFO)",
                S_GATE_ANCHOR,
                S_GATE_REPLACEMENT,
                S_GATE_PRESENT,
            ),
        ],
    )

    if ok:
        print(
            f"[{SCRIPT_NAME}] NOTE: the query manager is still constructed "
            "(non-None) with mode NONE — the proven not-supports-full "
            "configuration: _init_candidates stages nothing, needs_capture() "
            "is False, capture() is a no-op, and dispatch() always returns "
            "a NONE descriptor so the draft decode runs eagerly and "
            "run_fullgraph is unreachable. No None-safety surface is "
            "created."
        )
        print(
            f"[{SCRIPT_NAME}] NOTE: the target model's CudaGraphManager is "
            "a separate object owned by the model runner; this gate does "
            "not touch it. Expect the 'Capturing dspark CUDA graphs' phase "
            "to disappear from the boot log while the target FULL capture "
            "phases remain."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
