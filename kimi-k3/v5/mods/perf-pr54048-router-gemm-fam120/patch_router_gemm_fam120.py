#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PERF-PR54048-ROUTER-GEMM-FAM120 — port of upstream vllm-project/vllm#54048.

Un-gates the cuBLAS bf16->fp32 router GEMM (``torch.mm`` out_dtype epilogue)
for family-120 Blackwell (GB10 / DGX Spark, sm121).

Why: on GB10 ``is_device_capability_family(100)`` is False (family 12 != 10),
so ``can_use_specialized_kernels`` is False, so
``allow_specialized_router_gemm`` is False, so ``allow_cublas_router_gemm``
is False. The router then falls through to Tier 6 (``F.linear``): bf16-rounded
logits plus a separate bf16->fp32 copy kernel before grouped_topk — every MoE
layer, every decode step. The plain cuBLAS out_dtype epilogue has no SM90+
requirement, so tier 5 is re-gated on ``not bias and current_platform.is_cuda()``
(port of the ``_router_gemm_cublas_capable`` capability check from #54048,
adapted to this older, simpler tree shape which has no ROCm disjunct and no
pre-existing ``_router_gemm_no_bias`` attribute).

Only the cuBLAS tier gate changes. ``allow_specialized_router_gemm`` and the
cuteDSL ``allow_ll_bf16_gemm`` path (SM90+) are untouched.

Modes:
  apply     <vllm_root>            patch in place; missing anchors NO-OP with
                                   a loud NOTE and exit 0 (never block boot).
  --simulate <gate_linear.py>      never writes the target; patches a temp
                                   copy, py_compiles it, prints a unified
                                   diff; exits 1 if anchors are missing so it
                                   is useful as a pre-flight check.

Idempotent via the marker string ``perf-pr54048-router-gemm-fam120: applied``.
"""
import difflib
import py_compile
import re
import sys
import tempfile
from pathlib import Path

SCRIPT_NAME = "perf-pr54048-router-gemm-fam120"
PR = "vllm-project/vllm#54048"
TAG = f"{SCRIPT_NAME} (cuBLAS out_dtype router GEMM for family-120 / GB10 sm121)"

TARGET = "model_executor/layers/fused_moe/router/gate_linear.py"

MARKER = f"{SCRIPT_NAME}: applied"

# --- Anchor 1: __init__ cuBLAS eligibility block -----------------------------
OLD_INIT = (
    "        # cuBLAS bf16→fp32 eligibility\n"
    "        self.allow_cublas_router_gemm = (\n"
    "            self.allow_specialized_router_gemm\n"
    "            and self.weight.dtype == torch.bfloat16\n"
    "            and self.out_dtype == torch.float32\n"
    "        )\n"
)
NEW_INIT = (
    "        # cuBLAS bf16→fp32 eligibility\n"
    f"        # {MARKER}\n"
    "        # Plain cuBLAS out_dtype epilogue (torch.mm), no SM90+ requirement.\n"
    "        # The specialized-kernel gate (allow_specialized_router_gemm)\n"
    "        # excludes family-120 Blackwell (GB10 / DGX Spark, sm121), which\n"
    "        # this tier still covers. No bias: torch.mm has no bias term.\n"
    f"        # Port of upstream {PR}.\n"
    "        self._router_gemm_cublas_capable = (\n"
    "            not bias and current_platform.is_cuda()\n"
    "        )\n"
    "        self.allow_cublas_router_gemm = (\n"
    "            self._router_gemm_cublas_capable\n"
    "            and self.weight.dtype == torch.bfloat16\n"
    "            and self.out_dtype == torch.float32\n"
    "        )\n"
)

# --- Anchor 2: set_out_dtype cuBLAS re-check ---------------------------------
OLD_SET_OUT = (
    "        if (\n"
    "            not self.allow_cublas_router_gemm\n"
    "            and self.allow_specialized_router_gemm\n"
    "            and out_dtype == torch.float32\n"
    "        ):\n"
    "            self.allow_cublas_router_gemm = self.weight.dtype == torch.bfloat16\n"
)
NEW_SET_OUT = (
    "        if (\n"
    "            not self.allow_cublas_router_gemm\n"
    "            and self._router_gemm_cublas_capable\n"
    "            and out_dtype == torch.float32\n"
    "        ):\n"
    "            self.allow_cublas_router_gemm = self.weight.dtype == torch.bfloat16\n"
)


def loud(msg: str) -> None:
    print(f"=====> [{SCRIPT_NAME}] {msg}")


def precheck(src: str) -> None:
    """Verify allow_specialized_router_gemm is the gate that blocks GB10.

    Prints the actual assignment found in the file so the operator can
    confirm the mod flips the gate on sm121 rather than being a no-op.
    """
    assign = re.search(
        r"^([ \t]*)self\.allow_specialized_router_gemm = (.+)$", src, re.M
    )
    if assign is None:
        loud(
            "WARNING: no `self.allow_specialized_router_gemm = ...` assignment "
            "found — cannot confirm the GB10 gate; proceeding on anchors only."
        )
        return
    loud(f"pre-check: allow_specialized_router_gemm assignment: "
         f"self.allow_specialized_router_gemm = {assign.group(2).strip()!r}")
    if assign.group(2).strip() != "can_use_specialized_kernels":
        loud(
            "WARNING: unexpected assignment shape (expected "
            "`can_use_specialized_kernels`); check that the patch still "
            "un-gates family-120."
        )
    fam = re.search(
        r"^([ \t]*)is_blackwell = current_platform\.is_device_capability_family\((\d+)\)",
        src,
        re.M,
    )
    if fam:
        loud(
            f"pre-check: is_blackwell = is_device_capability_family("
            f"{fam.group(2)}) — family-120 (sm121: 121//10=12 != {int(fam.group(2))//10}) "
            "is EXCLUDED, so allow_specialized_router_gemm is False on GB10. "
            "This mod re-gates the cuBLAS tier on "
            "(`not bias and current_platform.is_cuda()`) → True on sm121 → "
            "the gate genuinely flips."
        )
    else:
        loud(
            "WARNING: is_device_capability_family(100) probe not found; "
            "confirm manually that family-120 is excluded from the "
            "specialized-kernel gate."
        )


def apply_patch(src: str) -> tuple[str, int]:
    changed = 0
    if OLD_INIT in src:
        src = src.replace(OLD_INIT, NEW_INIT, 1)
        changed += 1
    if OLD_SET_OUT in src:
        src = src.replace(OLD_SET_OUT, NEW_SET_OUT, 1)
        changed += 1
    return src, changed


def already_applied(src: str) -> bool:
    return MARKER in src or (
        "self._router_gemm_cublas_capable" in src
        and "self.allow_specialized_router_gemm\n" not in src.split(
            "def set_out_dtype"
        )[0].split("def forward")[0]
    )


def main() -> int:
    args = sys.argv[1:]
    simulate = False
    if len(args) == 2 and args[0] == "--simulate":
        simulate = True
        target = args[1]
    elif len(args) == 1 and not args[0].startswith("--"):
        target = f"{args[0]}/{TARGET}"
    else:
        print(f"usage: {SCRIPT_NAME}.py <vllm_root> | --simulate <gate_linear.py>")
        return 2

    path = Path(target)
    if not path.is_file():
        loud(f"NOTE: PREREQUISITE MISSING — {target} not found. NO-OP"
             + ("" if simulate else ", not blocking boot."))
        return 1 if simulate else 0

    src = path.read_text()

    if MARKER in src:
        loud(f"SKIP (already applied): {target}")
        return 0

    precheck(src)
    new_src, changed = apply_patch(src)

    if changed == 0:
        if "self._router_gemm_cublas_capable" in src:
            loud(f"SKIP (new shape already present, marker absent): {target}")
            return 0
        loud(
            f"NOTE: PREREQUISITE MISSING — anchors not found in {target}; "
            "tree shape does not match the expected GateLinear. NO-OP"
            + ("" if simulate else ", not blocking boot.")
        )
        return 1 if simulate else 0
    if changed == 1:
        loud(
            "WARNING: only one of two anchors matched — partial shape; "
            "inspect the file manually."
        )

    if simulate:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="_gate_linear_sim.py", delete=False
        ) as tf:
            tf.write(new_src)
            sim_path = tf.name
        try:
            py_compile.compile(sim_path, doraise=True)
        except py_compile.PyCompileError as exc:
            loud(f"SIMULATE COMPILE FAILED: {exc}")
            return 1
        loud(f"SIMULATE OK: {changed} hunk(s) would apply to {target}")
        loud(f"SIMULATE: patched temp copy (py_compile passed): {sim_path}")
        loud(f"INFO: this mod ports upstream {PR} — cuBLAS bf16→fp32 router "
             "GEMM un-gated for family-120 (GB10 / DGX Spark, sm121).")
        diff = difflib.unified_diff(
            src.splitlines(keepends=True),
            new_src.splitlines(keepends=True),
            fromfile=f"{target} (before)",
            tofile=f"{target} (after, simulated)",
        )
        sys.stdout.writelines(diff)
        return 0

    path.write_text(new_src)
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as exc:
        loud(f"COMPILE FAILED (file left patched, fix manually): {exc}")
        return 1

    loud(f"APPLY: {target} ({changed} hunks)")
    loud(f"INFO: applied {TAG} — port of upstream {PR}.")
    loud("dry-run checklist:")
    loud("  1. APPLY line above (or SKIP: already applied), no anchor NOTEs.")
    loud("  2. Only the cuBLAS tier gate changed: __init__ gained "
         "_router_gemm_cublas_capable; set_out_dtype now re-checks it.")
    loud("  3. allow_specialized_router_gemm and the cuteDSL ll_bf16 path "
         "(SM90+) are untouched.")
    loud("  4. Serving on GB10: router GEMM hits Tier 5 "
         "(torch.mm out_dtype=fp32); the separate bf16→fp32 copy kernel "
         "before grouped_topk disappears; router logits are no longer "
         "bf16-rounded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
