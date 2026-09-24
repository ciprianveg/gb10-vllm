#!/usr/bin/env bash
# fix-k3-moe-serial-smalln — serialize the MoE router gate + routed-expert
# down-projection overlap for tiny decode steps (num_tokens <= 8).
#
# Bug: KimiMoE._maybe_overlap_router_and_down_proj runs gate.forward_local
# (torch.mm bf16->fp32) and down_proj.forward_local concurrently on separate
# CUDA streams via maybe_execute_in_parallel for steps with 0 < num_tokens
# <= 8. Under TP16/DCP16 + spec-decode, single-stream decode steps land in
# this branch constantly (7 tokens in the observed crash), and the concurrent
# small-M cuBLAS calls fault: synchronous CUBLAS_STATUS_EXECUTION_FAILED
# inside the gate/down_proj GEMMs, then async illegal-memory-access, worker
# death. Same SM121 batched-GEMM family as lab#710, different call sites
# (no upstream coverage: #710 covers MLA bmm + DCP storage only).
#
# Fix: pass aux_stream=None in the <=8 branch so the helper runs gate then
# down_proj sequentially on the current stream — exactly what the fallback
# branch below already does when its own conditions fail (aux_stream=None
# -> sequential). Numerics are bit-identical (same ops, consumer reads the
# same values); expected cost ~1% decode (two tiny launch-bound GEMMs per
# MoE layer serialized instead of overlapped).
#
# The fallback branch (non-TP-sharded aux projections path) is untouched.
# Upstream #695/#706 allocator-lifetime equivalents are already present in
# this tree and did not prevent the crash, so the concurrent path itself
# goes, not just its fencing.
#
# All-or-nothing per file: exit 3 = no anchors found / all copies skipped,
# exit 4 = ambiguous anchor (count>1), exit 5 = patched text fails
# compile/py_compile. Idempotent via the marker string.

set -euo pipefail

MARKER="fix-k3-moe-serial-smalln"
TARGET_REL="vllm/models/kimi_k3/nvidia/model.py"

python3 - <<'PYEOF'
import os, subprocess, sys, py_compile

MARKER = "fix-k3-moe-serial-smalln"
TARGET_REL = "vllm/models/kimi_k3/nvidia/model.py"
ROOTS = os.environ.get(
    "MOD_FIND_ROOTS", "/opt/kimi-k3 /opt/venv /usr/local/lib").split()

# --- serialize the <=8 dual-stream branch: aux_stream None ---
old_branch = '''            (router_local, _), (down_local, _) = maybe_execute_in_parallel(
                lambda: self.gate.forward_local(hidden_states),
                lambda: down_proj.forward_local(hidden_states),
                self._down_proj_events[0],
                self._down_proj_events[1],
                self._down_proj_stream,
            )'''

new_branch = '''            # fix-k3-moe-serial-smalln: run gate then down_proj sequentially
            # on the current stream (aux_stream=None). The concurrent
            # small-M cuBLAS calls fault on SM121 (EXECUTION_FAILED ->
            # illegal access, worker death); serial order is bit-identical
            # for the consumer and costs ~1% decode on launch-bound GEMMs.
            (router_local, _), (down_local, _) = maybe_execute_in_parallel(
                lambda: self.gate.forward_local(hidden_states),
                lambda: down_proj.forward_local(hidden_states),
                self._down_proj_events[0],
                self._down_proj_events[1],
                None,
            )'''

applied = 0
skipped = 0
found = []
for root in ROOTS:
    try:
        out = subprocess.run(
            ["find", root, "-path", "*" + TARGET_REL],
            capture_output=True, text=True, timeout=120).stdout
    except Exception:
        continue
    for line in out.splitlines():
        p = line.strip()
        if p and "__pycache__" not in p and p not in found:
            found.append(p)

for p in sorted(set(found)):
    try:
        with open(p) as f:
            s = f.read()
    except FileNotFoundError:
        continue
    if MARKER in s:
        print(f"[{MARKER}] already applied in {p}")
        applied += 1
        continue
    _n = s.count(old_branch)
    if _n == 0:
        print(f"[{MARKER}] WARNING: serial anchor not found in {p}, skipping")
        skipped += 1
        continue
    if _n > 1:
        print(f"[{MARKER}] ERROR: serial anchor count={_n} in {p}")
        sys.exit(4)
    s = s.replace(old_branch, new_branch, 1)
    try:
        compile(s, p, "exec")
    except SyntaxError as e:
        print(f"[{MARKER}] ERROR: patched syntax invalid for {p}: {e}")
        sys.exit(5)
    with open(p, "w") as f:
        f.write(s)
    try:
        py_compile.compile(p, doraise=True)
    except Exception as e:
        print(f"[{MARKER}] ERROR: py_compile failed for {p}: {e}")
        sys.exit(5)
    print(f"[{MARKER}] APPLIED + py_compile OK: {p}")
    applied += 1

if applied == 0:
    print(f"[{MARKER}] WARNING: nothing applied (skipped {skipped}), exiting 3")
    sys.exit(3)
print(f"[{MARKER}] complete ({applied} files, {skipped} skipped)")
PYEOF
