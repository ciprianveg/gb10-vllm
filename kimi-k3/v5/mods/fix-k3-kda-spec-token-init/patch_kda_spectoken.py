#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""FIX-K3-KDA-SPEC-TOKEN-INIT — initialize spec partition offsets on the
no-spec path of the KDA metadata builder.

Root cause (v6-prd boot failure, 2026-09-11): in
models/kimi_k3/nvidia/kda_metadata.py build(), the `num_spec_decodes == 0`
branch assigns every other metadata field but never assigns
spec_token_start / non_spec_token_start — only the `else` branch does.
build() then passes both unconditionally into the metadata constructor
(~line 513). Any capture/build with zero spec sequences (notably the
cudagraph-memory profiling capture during determine_available_memory, but
also any runtime no-spec capture) dies with:
  UnboundLocalError: cannot access local variable 'spec_token_start'

None is the semantically correct value on the no-spec path (no spec
tokens, no contiguous spec/non-spec partition to record), and it matches
what the else branch passes through when the partition is non-contiguous —
the downstream constructor already accepts None.

Idempotent via marker. No env gates: this is a pure bug fix.
"""
import py_compile
import sys

SCRIPT_NAME = "patch_kda_spectoken"
TAG = "fix-k3-kda-spec-token-init"

COORD = "models/kimi_k3/nvidia/kda_metadata.py"

R_ANCHOR = '''            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
        else:
'''

R_REPLACEMENT = '''            non_spec_query_start_loc = query_start_loc
            non_spec_query_start_loc_cpu = query_start_loc_cpu
            num_accepted_tokens = None
            # fix-k3-kda-spec-token-init: the no-spec branch never assigned
            # these, but build() passes them unconditionally below. None is
            # correct here (no spec tokens, no partition offsets) and matches
            # the else branch when the partition is non-contiguous.
            spec_token_start = None
            non_spec_token_start = None
        else:
'''

R_PRESENT = "fix-k3-kda-spec-token-init"


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


def patch_kda(vroot):
    """Insert the no-spec offset init. Returns True on OK/SKIP."""
    path = vroot + "/" + COORD
    src = load(path)
    if src is None:
        return False
    if R_PRESENT in src:
        print(f"[{SCRIPT_NAME}] SKIP  kda_metadata.py (already present)")
        return True
    if src.count(R_ANCHOR) != 1:
        print(
            f"[{SCRIPT_NAME}] NOTE  kda_metadata.py: anchor found "
            f"{src.count(R_ANCHOR)}x (want 1); hunk skipped."
        )
        return False
    print(f"[{SCRIPT_NAME}] APPLY kda_metadata.py: no-spec spec_token_start init")
    return save(path, src.replace(R_ANCHOR, R_REPLACEMENT))


def main():
    if len(sys.argv) != 2:
        print(f"usage: {SCRIPT_NAME}.py <vllm-root>")
        return 2
    ok = patch_kda(sys.argv[1].rstrip("/"))
    print(f"[{SCRIPT_NAME}] {'OK' if ok else 'FAIL'}  {TAG}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
