#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PR53942 — backport of upstream vLLM PR #53942.

eh_proj optimization for the K3 MTP draft block:
  * mtp.py: eh_proj becomes a ReplicatedLinear (return_bias=False) so
    enable_kimi_k3_low_latency_gemm can install the low-latency linear
    method on it (a bare nn.Linear is invisible to that machinery).
  * low_latency_gemm.py: new measured (N=7168, K=14336) projection spec
    (the eh_proj shape) with a static-K C=1 config
    (SkinnyGemmConfig(1, 256, 2, vector_width=4, static_k=14336)).

Pure Python / CuTe DSL. Idempotent via marker; py_compile doraise.
"""
import py_compile
import sys

SCRIPT_NAME = "patch_eh_proj"
TAG = "pr53942 (upstream #53942 backport)"
MARKER = "pr53942 (upstream #53942)"

MTP = "models/kimi_k3/nvidia/mtp.py"
K3_GEMM = "models/kimi_k3/nvidia/low_latency_gemm.py"

MTP_H1_ANCHOR = """from vllm.model_executor.layers.layernorm import RMSNorm
"""
MTP_H1_REPL = """from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear  # {MARKER}
""".replace("{MARKER}", MARKER)

MTP_H2_ANCHOR = """        self.eh_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
"""
MTP_H2_REPL = """        # {MARKER}: ReplicatedLinear so the K3 low-latency GEMM installer
        # can replace the unquantized method with the measured (7168, 14336)
        # plan; return_bias=False keeps the call site returning a bare tensor.
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=None,
            prefix=maybe_prefix(prefix, "eh_proj"),
            return_bias=False,
        )
""".replace("{MARKER}", MARKER)

K3_H1_ANCHOR = """    ),
    (20480, 7168): ProjectionSpec(
"""
K3_H1_REPL = """    ),
    # {MARKER}: MTP eh_proj (hidden_size*2 -> hidden_size), measured on B300.
    (7168, 14336): ProjectionSpec(
        7168,
        14336,
        cute_configs=(
            (1, SkinnyGemmConfig(1, 256, 2, vector_width=4, static_k=14336)),
            (2, _cute(2, 224, 4, 2)),
        ),
    ),
    (20480, 7168): ProjectionSpec(
""".replace("{MARKER}", MARKER)


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


def patch_file(vroot, relpath, hunks):
    path = vroot + "/" + relpath
    src = load(path)
    if src is None:
        return False
    if MARKER in src:
        print(f"[{SCRIPT_NAME}] SKIP  {relpath} (already present)")
        return True
    ok = True
    for anchor, repl, desc in hunks:
        n = src.count(anchor)
        if n != 1:
            print(
                f"[{SCRIPT_NAME}] NOTE  {relpath}: anchor for '{desc}' found "
                f"{n}x (want 1); file skipped — stays stock."
            )
            ok = False
        else:
            src = src.replace(anchor, repl, 1)
    if not ok:
        return False
    if not save(path, src):
        return False
    print(f"[{SCRIPT_NAME}] APPLY {relpath}: {len(hunks)} hunk(s)")
    return True


def main():
    print(f"[{SCRIPT_NAME}] {TAG}")
    if len(sys.argv) != 2:
        print(f"usage: {SCRIPT_NAME}.py <VLLM_ROOT>")
        return 2
    vroot = sys.argv[1].rstrip("/")
    ok = patch_file(
        vroot,
        MTP,
        [
            (MTP_H1_ANCHOR, MTP_H1_REPL, "ReplicatedLinear import"),
            (MTP_H2_ANCHOR, MTP_H2_REPL, "eh_proj -> ReplicatedLinear"),
        ],
    )
    ok &= patch_file(
        vroot,
        K3_GEMM,
        [(K3_H1_ANCHOR, K3_H1_REPL, "(7168, 14336) eh_proj projection spec")],
    )
    if ok:
        print(
            f"[{SCRIPT_NAME}] done. MTP eh_proj now routes through the "
            f"measured K3 low-latency GEMM plan (static-K C=1 config)."
        )
    else:
        print(f"[{SCRIPT_NAME}] NOTE: some hunks skipped — see above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
