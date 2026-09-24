#!/usr/bin/env bash
# fix-k3-moe-gate-bf16-epilogue — route the MoE router gate around the fused
# bf16->fp32 cuBLAS epilogue kernel on SM121a.
#
# Bug: KimiColumnParallelGate.forward_local computes router logits with
# torch.mm(x, weight.T, out_dtype=torch.float32) — a fused bf16-input/fp32-
# output cublasLt epilogue. On SM121a this kernel family faults on small-M
# decode steps (synchronous CUBLAS_STATUS_EXECUTION_FAILED, then async
# illegal-memory-access, worker death; same SM121 batched-GEMM family as
# lab#710, which covered only MLA bmm + DCP storage). Serializing the
# dual-stream overlap did NOT prevent a later crash at the same 7-token
# shape, and allocator-lifetime equivalents (#695/#706) are already present
# — leaving kernel selection at this call site as the remaining lever.
#
# Fix: compute in bf16 and cast explicitly, selecting the plain (non-fused)
# cublasLt kernel instead of the epilogue variant:
#     output = torch.mm(x, self.weight.T).float()
# This mirrors the pre-existing else-branch below (F.linear(...).float())
# and matches what grouped_topk consumes (fp32 either way); routing is
# top-k over fp32 logits in both cases. The lm_head path needs no change:
# our LogitsProcessor is built without head_dtype, so it already takes the
# F.linear bf16 path, never the fused fp32 epilogue.
#
# Unconditional for the test (no env gate to forget); bake only if the
# TP16 crash stops recurring with it live.
#
# All-or-nothing per file: exit 3 = no anchors found / all copies skipped,
# exit 4 = ambiguous anchor (count>1), exit 5 = patched text fails
# compile/py_compile. Idempotent via the marker string.

set -euo pipefail

MARKER="fix-k3-moe-gate-bf16-epilogue"
TARGET_REL="vllm/models/kimi_k3/nvidia/model.py"

python3 - <<'PYEOF'
import os, subprocess, sys, py_compile

MARKER = "fix-k3-moe-gate-bf16-epilogue"
TARGET_REL = "vllm/models/kimi_k3/nvidia/model.py"
ROOTS = os.environ.get(
    "MOD_FIND_ROOTS", "/opt/kimi-k3 /opt/venv /usr/local/lib").split()

# --- gate epilogue: bf16 mm + explicit cast instead of fused fp32 epilogue ---
old_gate = '''        if x.is_cuda and x.dtype == self.weight.dtype == torch.bfloat16:
            output = torch.mm(x, self.weight.T, out_dtype=torch.float32)'''

new_gate = '''        # fix-k3-moe-gate-bf16-epilogue: plain bf16 mm + explicit cast
        # instead of the fused bf16->fp32 cublasLt epilogue, which faults
        # on small-M decode steps on SM121a (EXECUTION_FAILED -> illegal
        # access). Same pattern as the else-branch below; grouped_topk
        # consumes fp32 either way.
        if x.is_cuda and x.dtype == self.weight.dtype == torch.bfloat16:
            output = torch.mm(x, self.weight.T).float()'''

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
    _n = s.count(old_gate)
    if _n == 0:
        print(f"[{MARKER}] WARNING: gate anchor not found in {p}, skipping")
        skipped += 1
        continue
    if _n > 1:
        print(f"[{MARKER}] ERROR: gate anchor count={_n} in {p}")
        sys.exit(4)
    s = s.replace(old_gate, new_gate, 1)
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
