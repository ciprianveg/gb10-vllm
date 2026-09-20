#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V4PLUS-V6-REMOTE-DSPARK — port of myshytf/vllm@a653e74 (II fork line).

Remote-DSpark-draft feature: run the Kimi-K3 DSpark/DFlash draft model in a
dedicated standalone process on a separate GPU (e.g. an RTX 3090) and let the
verifier's model runner use it through a ZMQ-backed RemoteK3DSparkSpeculator
proxy, selected at init_speculator() time via environment variables:

  - VLLM_K3_DRAFT_REMOTE_ADDRESS   (both dspark and dflash methods)
  - VLLM_K3_DSPARK_REMOTE_ADDRESS  (dspark method only; lower precedence)

New files (written verbatim from the commit, payload dir mirrors the repo
layout; `vllm/...` payloads land under VLLM_ROOT, `tests/...` payloads land
under the repo root when a tests/ tree is detectable):

  - vllm/entrypoints/k3_dspark_standalone.py        (847 lines)
  - vllm/entrypoints/k3_dspark_rpc.py               (1363 lines)
  - vllm/v1/worker/gpu/spec_decode/dspark/remote_speculator.py (770 lines)
  - tests/v1/spec_decode/test_k3_dspark_remote_speculator.py   (126 lines)
  - tests/v1/spec_decode/test_k3_dspark_standalone.py          (172 lines)

Modified files (anchor-based replacement, anchors adapted from the II fork
context lines to this fork's b4f text):

  - v1/worker/gpu/buffer_utils.py     — StagedWriteTensor.cpu property
  - v1/worker/gpu/input_batch.py      — InputBatch.all_token_ids_cpu field
  - v1/worker/gpu/model_runner.py     — pass all_token_ids_cpu into InputBatch
  - v1/worker/gpu/spec_decode/__init__.py — env-gated RemoteK3DSparkSpeculator

Idempotent via the V4PLUS-V6-REMOTE-DSPARK marker. Every touched file is
py_compiled with doraise=True. Prints APPLY/SKIP/NOTE per file.
"""
import os
import py_compile
import sys

SCRIPT_NAME = "patch_remote_dspark"
TAG = "v6-remote-dspark (myshytf/vllm@a653e74 port, image v4-plus-b4f)"
MARKER = "V4PLUS-V6-REMOTE-DSPARK"
PAYLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "payload")

# ---------------------------------------------------------------------------
# New files: payload path (repo-relative) -> install destination
# ---------------------------------------------------------------------------

NEW_FILES = [
    # (payload repo-relative path, VLLM_ROOT-relative target)
    ("vllm/entrypoints/k3_dspark_standalone.py", "entrypoints/k3_dspark_standalone.py"),
    ("vllm/entrypoints/k3_dspark_rpc.py", "entrypoints/k3_dspark_rpc.py"),
    (
        "vllm/v1/worker/gpu/spec_decode/dspark/remote_speculator.py",
        "v1/worker/gpu/spec_decode/dspark/remote_speculator.py",
    ),
]

NEW_TEST_FILES = [
    "tests/v1/spec_decode/test_k3_dspark_remote_speculator.py",
    "tests/v1/spec_decode/test_k3_dspark_standalone.py",
]

# ---------------------------------------------------------------------------
# Modified files: anchor-based hunks (anchors verified against b4f)
# ---------------------------------------------------------------------------

BUFFER_UTILS = "v1/worker/gpu/buffer_utils.py"
INPUT_BATCH = "v1/worker/gpu/input_batch.py"
MODEL_RUNNER = "v1/worker/gpu/model_runner.py"
SPEC_DECODE_INIT = "v1/worker/gpu/spec_decode/__init__.py"

# Hunk: StagedWriteTensor.__init__ explicit UVA-buffer field
# (upstream 7f37e34ca: initialize `_uva_buf` in __init__ and read it directly,
# instead of guarding every read with getattr).
BU_INIT_ANCHOR = """        if not uva_instead_of_gpu:
            # Create a GPU tensor (default)
            self.gpu = torch.zeros(size, dtype=dtype, device=device)
"""
BU_INIT_REPLACEMENT = """        self._uva_buf: UvaBuffer | None = None
        if not uva_instead_of_gpu:
            # Create a GPU tensor (default)
            self.gpu = torch.zeros(size, dtype=dtype, device=device)
"""

# Hunk: StagedWriteTensor.cpu property (II anchor == b4f anchor, buffer_utils.py:153).
BU_ANCHOR = """        self.write_cu_lens = new_buffer(self.num_rows, dtype=torch.int32)

    def stage_write(
"""
BU_REPLACEMENT = """        self.write_cu_lens = new_buffer(self.num_rows, dtype=torch.int32)

    # V4PLUS-V6-REMOTE-DSPARK: host view for remote-speculator prefix checks.
    @property
    def cpu(self) -> torch.Tensor | None:
        \"\"\"Return the host backing tensor when this tensor uses UVA.\"\"\"
        return None if self._uva_buf is None else self._uva_buf.cpu

    def stage_write(
"""

# Hunk: InputBatch.all_token_ids_cpu field (II anchor == b4f anchor, input_batch.py:105).
IB_ANCHOR = """    max_req_tokens: int | None = None
    valid_num_draft_tokens_per_req: np.ndarray | None = None

    # When > 0, dummy batches carry seeded-random token ids instead of zeros.
"""
IB_REPLACEMENT = """    max_req_tokens: int | None = None
    valid_num_draft_tokens_per_req: np.ndarray | None = None

    # V4PLUS-V6-REMOTE-DSPARK: Optional host view of the request token table.
    # Remote speculators use this to verify that a target prefix-cache hit
    # belongs to retained draft state before reconnecting it. The tensor is
    # shared, not copied.
    all_token_ids_cpu: torch.Tensor | None = None

    # When > 0, dummy batches carry seeded-random token ids instead of zeros.
"""

# Hunk: pass all_token_ids_cpu into InputBatch (II anchor == b4f anchor,
# model_runner.py:1558-1561, inside prepare_inputs).
MR_ANCHOR = """            prompt_lens=prompt_lens,
            max_req_tokens=max_req_tokens,
            valid_num_draft_tokens_per_req=valid_num_draft_tokens_per_req,
        )
"""
MR_REPLACEMENT = """            prompt_lens=prompt_lens,
            max_req_tokens=max_req_tokens,
            valid_num_draft_tokens_per_req=valid_num_draft_tokens_per_req,
            # V4PLUS-V6-REMOTE-DSPARK: host token table for remote speculators.
            all_token_ids_cpu=self.req_states.all_token_ids.cpu,
        )
"""

# Hunk: K3 DIAG (TEMPORARY) — dump the draft-token region of the graph input
# right before FULL-mode replay, to confirm the remotely-received draft tokens
# actually landed in the buffer the captured graph reads. If they did not, the
# graph replays stale/placeholder tokens -> 0% acceptance + garbage output.
# model_runner.py ~1997-2006 (FULL graph replay block).
MR_GRAPH_DIAG_ANCHOR = """            with record_function_or_nullcontext(
                f"vllm:v2/target/{phase}/full_graph_replay"
            ):
                self.kv_connector.pre_forward(scheduler_output)
                model_output = self.cudagraph_manager.run_fullgraph(batch_desc)"""
MR_GRAPH_DIAG_REPLACEMENT = """            with record_function_or_nullcontext(
                f"vllm:v2/target/{phase}/full_graph_replay"
            ):
                self.kv_connector.pre_forward(scheduler_output)
                # K3 DIAG (TEMP): confirm the graph sees the freshly-written
                # draft tokens, not a capture-time snapshot.
                _diag_ids = input_batch.input_ids
                if _diag_ids is not None and _diag_ids.numel():
                    _diag_tail = _diag_ids[-min(16, _diag_ids.numel()):].tolist()
                    logger.info(
                        "K3 DIAG graph input_ids tail (first=last16): %s",
                        _diag_tail,
                    )
                model_output = self.cudagraph_manager.run_fullgraph(batch_desc)"""

# Hunk: `import os` header (spec_decode/__init__.py:1-3).
SD_HEADER_ANCHOR = """# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
"""
SD_HEADER_REPLACEMENT = """# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# V4PLUS-V6-REMOTE-DSPARK: env-gated remote draft speculator selection.
import os

import torch
"""

# Hunk: dflash branch remote hook. NOTE: unlike the II fork, b4f's dflash
# branch first dispatches DFlash2DraftModel to DFlash2Speculator; the remote
# hook is placed at the top of the branch so the env var wins deterministically
# over both local speculators.
SD_DFLASH_ANCHOR = """    if speculative_config.method == "dflash":
        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:
"""
SD_DFLASH_REPLACEMENT = """    if speculative_config.method == "dflash":
        remote_address = os.environ.get("VLLM_K3_DRAFT_REMOTE_ADDRESS")
        if remote_address:
            from vllm.v1.worker.gpu.spec_decode.dspark.remote_speculator import (
                RemoteK3DSparkSpeculator,
            )

            return RemoteK3DSparkSpeculator(
                vllm_config,
                device,
                address=remote_address,
            )
        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:
"""

# Hunk: dspark branch remote hook (accepts both env vars, like the II fork).
SD_DSPARK_ANCHOR = """    elif speculative_config.method == "dspark":
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )
"""
SD_DSPARK_REPLACEMENT = """    elif speculative_config.method == "dspark":
        remote_address = os.environ.get(
            "VLLM_K3_DRAFT_REMOTE_ADDRESS"
        ) or os.environ.get("VLLM_K3_DSPARK_REMOTE_ADDRESS")
        if remote_address:
            from vllm.v1.worker.gpu.spec_decode.dspark.remote_speculator import (
                RemoteK3DSparkSpeculator,
            )

            return RemoteK3DSparkSpeculator(
                vllm_config,
                device,
                address=remote_address,
            )
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )
"""


def load(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError as exc:
        print(f"[{SCRIPT_NAME}] NOTE  {path}: unreadable ({exc})")
        return None


def save(path, src):
    try:
        with open(path, "w") as f:
            f.write(src)
        py_compile.compile(path, doraise=True)
        return True
    except (OSError, py_compile.PyCompileError) as exc:
        print(f"[{SCRIPT_NAME}] FAIL  {path}: write/compile failed ({exc})")
        return False


def install_new_file(payload_rel, dest_abs, label):
    """Write a payload file verbatim (+ trailing marker comment) if absent."""
    payload_path = os.path.join(PAYLOAD_DIR, payload_rel)
    payload = load(payload_path)
    if payload is None:
        return False
    if os.path.exists(dest_abs):
        existing = load(dest_abs)
        if existing is not None and MARKER in existing:
            print(f"[{SCRIPT_NAME}] SKIP  {label} (marker present)")
            return True
        print(
            f"[{SCRIPT_NAME}] NOTE  {label}: exists without marker; "
            f"not overwritten — inspect manually."
        )
        return False
    os.makedirs(os.path.dirname(dest_abs), exist_ok=True)
    if not payload.endswith("\n"):
        payload += "\n"
    payload += f"# {MARKER}\n"
    if not save(dest_abs, payload):
        return False
    print(f"[{SCRIPT_NAME}] APPLY {label} (new file, verbatim from a653e74)")
    return True


def patch_file(vroot, relpath, hunks, label):
    """Apply anchor-based hunks. Each hunk is (anchor, replacement, desc)."""
    path = os.path.join(vroot, relpath)
    src = load(path)
    if src is None:
        return False
    if MARKER in src:
        print(f"[{SCRIPT_NAME}] SKIP  {relpath} (marker present)")
        return True
    out = src
    for anchor, replacement, desc in hunks:
        count = out.count(anchor)
        if count != 1:
            print(
                f"[{SCRIPT_NAME}] NOTE  {relpath}: anchor for '{desc}' found "
                f"{count}x (want 1); hunk skipped."
            )
            return False
        out = out.replace(anchor, replacement, 1)
    if not save(path, out):
        return False
    print(f"[{SCRIPT_NAME}] APPLY {relpath}: {label}")
    return True


def main():
    print(f"[{SCRIPT_NAME}] {TAG}")
    if len(sys.argv) != 2:
        print(f"usage: {SCRIPT_NAME}.py <VLLM_ROOT>")
        return 2
    vroot = sys.argv[1].rstrip("/")
    if not os.path.isdir(vroot):
        print(f"[{SCRIPT_NAME}] FAIL  VLLM_ROOT is not a directory: {vroot}")
        return 1

    ok = True
    notes = []

    # --- new vllm package files ---
    for payload_rel, target_rel in NEW_FILES:
        ok &= install_new_file(payload_rel, os.path.join(vroot, target_rel), target_rel)

    # --- new test files (repo-root relative, best effort) ---
    repo_root = os.path.dirname(vroot)
    tests_root = os.path.join(repo_root, "tests")
    if os.path.isdir(tests_root):
        for payload_rel in NEW_TEST_FILES:
            dest_rel = payload_rel[len("tests/"):]
            ok &= install_new_file(
                payload_rel, os.path.join(tests_root, dest_rel), payload_rel
            )
    else:
        notes.append(
            f"tests/ tree not found at {tests_root}; the 2 new test files "
            f"were NOT installed (runtime feature is unaffected)."
        )
        for payload_rel in NEW_TEST_FILES:
            print(f"[{SCRIPT_NAME}] NOTE  {payload_rel}: no tests/ tree at repo root; skipped.")

    # --- modified files ---
    ok &= patch_file(
        vroot,
        BUFFER_UTILS,
        [
            (BU_INIT_ANCHOR, BU_INIT_REPLACEMENT, "StagedWriteTensor._uva_buf init"),
            (BU_ANCHOR, BU_REPLACEMENT, "StagedWriteTensor.cpu"),
        ],
        "StagedWriteTensor.cpu host-view property + explicit _uva_buf init",
    )
    ok &= patch_file(
        vroot, INPUT_BATCH, [(IB_ANCHOR, IB_REPLACEMENT, "all_token_ids_cpu field")],
        "InputBatch.all_token_ids_cpu field",
    )
    ok &= patch_file(
        vroot, MODEL_RUNNER, [(MR_ANCHOR, MR_REPLACEMENT, "all_token_ids_cpu kwarg")],
        "prepare_inputs passes all_token_ids_cpu to InputBatch",
    )
    ok &= patch_file(
        vroot,
        MODEL_RUNNER,
        [(MR_GRAPH_DIAG_ANCHOR, MR_GRAPH_DIAG_REPLACEMENT, "K3 DIAG graph input_ids")],
        "K3 DIAG: dump graph input_ids before FULL replay",
    )
    ok &= patch_file(
        vroot,
        SPEC_DECODE_INIT,
        [
            (SD_HEADER_ANCHOR, SD_HEADER_REPLACEMENT, "import os header"),
            (SD_DFLASH_ANCHOR, SD_DFLASH_REPLACEMENT, "dflash remote hook"),
            (SD_DSPARK_ANCHOR, SD_DSPARK_REPLACEMENT, "dspark remote hook"),
        ],
        "env-gated RemoteK3DSparkSpeculator selection (dspark + dflash)",
    )

    for note in notes:
        print(f"[{SCRIPT_NAME}] NOTE  {note}")
    if ok:
        print(
            f"[{SCRIPT_NAME}] done. Remote draft engages when "
            f"VLLM_K3_DRAFT_REMOTE_ADDRESS (or VLLM_K3_DSPARK_REMOTE_ADDRESS "
            f"for dspark) is set at verifier boot; unset = stock local "
            f"DSpark/DFlash/DFlash2 speculators."
        )
    else:
        print(f"[{SCRIPT_NAME}] NOTE: one or more files skipped — see above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
