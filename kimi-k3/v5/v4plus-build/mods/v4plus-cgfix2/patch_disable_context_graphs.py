#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V4PLUS CGFIX2 — disable the DFlash context-KV CUDA graph capture phase.

Boot evidence (b5 image, cgfix #617+#628 applied): "Capturing dspark CUDA
graphs (FULL): 2/2" completes, then CUDA illegal memory access at
"Capturing DFlash context-KV CUDA graphs (FULL): 0%".  That capture phase
is fork-PR-#251-specific; the fork's newer lineages and upstream run the
context-KV precompute EAGERLY outside any graph, and this tree already
carries the native eager fallback.

The change (surgical, one gate): stop constructing
``DFlashContextCudaGraphManager`` unless explicitly re-enabled.  With the
manager absent, every downstream site already does the right thing —
verified against the real b5 image file (speculator.py):

  * ``capture()`` line ~248: ``if self.context_cudagraph_manager is not
    None:`` guards the ``capture_context(...)`` call — the crashing phase
    never starts (no hunk needed there);
  * ``get_cudagraph_managers()`` line ~260: a None manager is not
    registered, so no context-graph capture phase is scheduled at all;
  * ``_dispatch_context_batch()`` lines ~586-596: ``if
    self.context_cudagraph_manager is not None and ...`` fails short and
    returns ``BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, ...)``;
  * ``_precompute_context_kv()`` lines ~567-575: with a NONE-mode
    descriptor the FULL branch (which asserts the manager) is unreachable
    and the eager path runs:
    ``self.model.precompute_and_store_context_kv(hidden_states,
    context_positions, context_slots)`` — exactly the fork-converged
    design.

The main DSpark FULL graphs (``query_cudagraph_manager``) are untouched.

Env semantics — ``VLLM_K3_DISABLE_CONTEXT_GRAPHS`` (read directly via
``os.getenv`` in speculator.py):
  * "1" (DEFAULT): context-KV graphs DISABLED — eager precompute, matching
    the fork-converged/upstream design.  The mod is live by default
    because the current capture phase crashes the boot.
  * "0" / "false" / "False": restore the #251 context-KV graph capture
    (A/B only — known to crash at capture-begin on this image).
A loud INFO at speculator init states which mode is active.

Why ``os.getenv`` instead of an envs.py field: the b5 image's envs.py was
not extracted for this mod, and writing an envs.py hunk blind would
violate the anchor-against-real-files convention.  A direct read is one
fewer anchor and equally robust; the name keeps the VLLM_K3_ prefix our
other knobs use.

PREREQUISITE (fail loud): speculator.py must carry the DFlash/DSpark
lineage markers (DFlashContextCudaGraphManager, _precompute_context_kv,
precompute_and_store_context_kv).  This script refuses to patch a foreign
lineage.

Idempotent: every hunk is skipped when its marker is already present.
A missing anchor prints a NOTE and skips that hunk; only a missing
prerequisite, file-not-found, or a broken post-patch compile exits
non-zero.
"""

from __future__ import annotations

import os
import py_compile
import sys

SCRIPT_NAME = "patch_disable_context_graphs"
TAG = "# V4PLUS-CGFIX2 (disable DFlash context-KV graph capture)"

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
    """Fail loud unless speculator.py carries the DFlash/DSpark lineage."""
    try:
        with open(SPECULATOR) as f:
            src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {SPECULATOR} not found", file=sys.stderr)
        return False
    ok = True
    for marker in (
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
# import os (the file reads the new env directly)
# ---------------------------------------------------------------------------

S_IMPORT_ANCHOR = "import copy\n"
S_IMPORT_REPLACEMENT = "import copy\nimport os\n"
S_IMPORT_PRESENT = "import copy\nimport os\n"

# ---------------------------------------------------------------------------
# init_cudagraph_manager: gate the context manager construction + loud INFO
# ---------------------------------------------------------------------------

S_GATE_ANCHOR = (
    "        if wants_full and supports_full and self._speculator_name == \"DSpark\":\n"
    "            self.context_cudagraph_manager = DFlashContextCudaGraphManager(\n"
    "                self.vllm_config,\n"
    "                self.device,\n"
    "                max_num_context_tokens=self.max_num_tokens,\n"
    "            )\n"
)
S_GATE_REPLACEMENT = (
    "        # V4PLUS-CGFIX2: the DFlash context-KV capture phase (fork PR\n"
    "        # #251 lineage) crashes at capture-begin under FULL graphs on\n"
    "        # this image (CUDA illegal memory access right after the main\n"
    "        # DSpark graphs capture cleanly). The fork's newer lineages and\n"
    "        # upstream run the context-KV precompute EAGERLY outside any\n"
    "        # graph, and this tree already carries that native fallback:\n"
    "        # with no context manager, _dispatch_context_batch returns a\n"
    "        # NONE-mode descriptor and _precompute_context_kv calls\n"
    "        # model.precompute_and_store_context_kv directly.\n"
    "        # VLLM_K3_DISABLE_CONTEXT_GRAPHS selects the mode: \"1\"\n"
    "        # (default) = eager context-KV precompute (the fork-converged\n"
    "        # design); \"0\" = restore the #251 context-KV graph capture\n"
    "        # (A/B only; known to crash at capture-begin here).\n"
    "        context_graphs_disabled = os.getenv(\n"
    "            \"VLLM_K3_DISABLE_CONTEXT_GRAPHS\", \"1\"\n"
    "        ) not in (\"0\", \"false\", \"False\")\n"
    "        if (\n"
    "            wants_full\n"
    "            and supports_full\n"
    "            and self._speculator_name == \"DSpark\"\n"
    "            and not context_graphs_disabled\n"
    "        ):\n"
    "            self.context_cudagraph_manager = DFlashContextCudaGraphManager(\n"
    "                self.vllm_config,\n"
    "                self.device,\n"
    "                max_num_context_tokens=self.max_num_tokens,\n"
    "            )\n"
    "        if wants_full and supports_full and self._speculator_name == \"DSpark\":\n"
    "            logger.info(\n"
    "                \"%s context-KV precompute runs %s \"\n"
    "                \"(VLLM_K3_DISABLE_CONTEXT_GRAPHS=%s).\",\n"
    "                self._speculator_name,\n"
    "                (\n"
    "                    \"EAGERLY outside CUDA graphs\"\n"
    "                    if context_graphs_disabled\n"
    "                    else \"inside dedicated context-KV CUDA graphs\"\n"
    "                ),\n"
    "                \"1\" if context_graphs_disabled else \"0\",\n"
    "            )\n"
)
S_GATE_PRESENT = "VLLM_K3_DISABLE_CONTEXT_GRAPHS"


def main() -> int:
    print(f"[{SCRIPT_NAME}] {TAG}")
    print(f"[{SCRIPT_NAME}] VLLM_ROOT={VLLM_ROOT}")
    if not check_prerequisites():
        print(
            f"[{SCRIPT_NAME}] PREREQUISITE FAILED: speculator.py is not the "
            "expected DFlash/DSpark lineage; refusing to patch.",
            file=sys.stderr,
        )
        return 1

    ok = apply_hunks(
        SPECULATOR,
        [
            (
                "import os",
                S_IMPORT_ANCHOR,
                S_IMPORT_REPLACEMENT,
                S_IMPORT_PRESENT,
            ),
            (
                "context-KV graph gate + mode INFO",
                S_GATE_ANCHOR,
                S_GATE_REPLACEMENT,
                S_GATE_PRESENT,
            ),
        ],
    )

    if ok:
        print(
            f"[{SCRIPT_NAME}] NOTE: the capture() context phase needs no "
            "hunk — it is already guarded by "
            "'if self.context_cudagraph_manager is not None:', so a None "
            "manager skips capture_context() and the crashing phase never "
            "starts. get_cudagraph_managers() likewise registers nothing."
        )
        print(
            f"[{SCRIPT_NAME}] NOTE: with the manager absent, "
            "_dispatch_context_batch() returns a NONE-mode descriptor and "
            "_precompute_context_kv() runs the eager "
            "model.precompute_and_store_context_kv(...) path — the "
            "fork-converged design. The main DSpark FULL query graphs are "
            "untouched."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
