#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PR53525 — backport of upstream vLLM PR #53525 (Python side only).

Optimize the C=1 KDA PDL pipeline:
  * cute skinny GEMM: new ``early_pdl_trigger`` config flag — the kernel
    fires ``griddepcontrol_launch_dependents()`` right after the mainloop
    (activation A fully consumed) instead of after the epilogue.
  * K3 low-latency plan: M=1 entries for in_proj_qkvgfab/o_proj (cute) and
    f_b_proj/fused_qkv_a_proj (dsv3) request the early trigger.
  * ``_runtime_ok`` accepts a non-packed leading stride for size-1 inputs.

UPSTREAM CSRCS PARTS NOT PORTED (require C++ rebuild of
_C_stable_libtorch.abi3.so; the fork ships a prebuilt binary):
  * csrc/libtorch_stable/dsv3_fused_a_gemm.cu  (early_pdl_trigger template
    param + relaxed single-row stride check)
  * csrc/libtorch_stable/kimi_k3/fused_kda_decode_kernel.cu (deferred
    cudaGridDependencySynchronize for B==1)
  * csrc/libtorch_stable/torch_bindings.cpp (op schema arg)

ADAPTATION: the native-only pieces are gated at runtime on the compiled
op's schema containing ``early_pdl_trigger`` (a proxy for "the .so was
built from post-#53525 csrc"). On the current fork binary the dsv3 early
trigger and the single-row stride relaxation stay disabled (stock
behavior); the cute-DSL early trigger engages immediately because those
kernels are JIT-compiled from Python. Once the image is rebuilt with the
csrc changes, everything engages with no further patching.

Also NOTE-skipped: upstream's model.py hunk is a pure formatting no-op.

Idempotent via marker. All touched files are py_compile'd with doraise.
"""
import py_compile
import sys

SCRIPT_NAME = "patch_c1_kda_pdl"
TAG = "pr53525 (upstream #53525 backport, Python side)"
MARKER = "pr53525 (upstream #53525)"

SKINNY_INT = "model_executor/kernels/linear/cute_dsl/_skinny_gemm.py"
SKINNY = "model_executor/kernels/linear/cute_dsl/skinny_gemm.py"
CUSTOM_OPS = "_custom_ops.py"
K3_GEMM = "models/kimi_k3/nvidia/low_latency_gemm.py"

# --- _skinny_gemm.py hunks ------------------------------------------------

SG_H1_ANCHOR = """        has_residual: bool = False,
        use_pdl: bool = False,
        static_k: int | None = None,
    ) -> None:
"""
SG_H1_REPL = """        has_residual: bool = False,
        use_pdl: bool = False,
        early_pdl_trigger: bool = False,  # {MARKER}
        static_k: int | None = None,
    ) -> None:
""".replace("{MARKER}", MARKER)

SG_H2_ANCHOR = """        self.has_residual = has_residual
        self.use_pdl = use_pdl
        self.static_k = static_k
"""
SG_H2_REPL = """        self.has_residual = has_residual
        self.use_pdl = use_pdl
        self.early_pdl_trigger = early_pdl_trigger
        self.static_k = static_k
"""

SG_H3_ANCHOR = """                for vi in cutlass.range_constexpr(vector_width):
                    for mi in cutlass.range_constexpr(num_rows):
                        for ni in cutlass.range_constexpr(outputs_per_block):
                            acc[mi, ni] = acc[mi, ni] + a_regs[mi, vi].to(
                                cutlass.Float32
                            ) * b_regs[ni, vi].to(cutlass.Float32)

        for mi in cutlass.range_constexpr(num_rows):
"""
SG_H3_REPL = """                for vi in cutlass.range_constexpr(vector_width):
                    for mi in cutlass.range_constexpr(num_rows):
                        for ni in cutlass.range_constexpr(outputs_per_block):
                            acc[mi, ni] = acc[mi, ni] + a_regs[mi, vi].to(
                                cutlass.Float32
                            ) * b_regs[ni, vi].to(cutlass.Float32)

        # Activation A is fully consumed.  The K3 C=1 specialization exposes
        # reduction and stores to the dependent without changing other shapes.
        if const_expr(self.use_pdl and self.early_pdl_trigger):
            cute.arch.griddepcontrol_launch_dependents()

        for mi in cutlass.range_constexpr(num_rows):
"""

SG_H4_ANCHOR = """        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
"""
SG_H4_REPL = """        if const_expr(self.use_pdl and not self.early_pdl_trigger):
            cute.arch.griddepcontrol_launch_dependents()
"""

# --- skinny_gemm.py hunks -------------------------------------------------

SK_H1_ANCHOR = """    k_unroll: int = 1
    vector_width: int = 8
    static_k: int | None = None
"""
SK_H1_REPL = """    k_unroll: int = 1
    vector_width: int = 8
    static_k: int | None = None
    early_pdl_trigger: bool = False  # {MARKER}
""".replace("{MARKER}", MARKER)

SK_H2_ANCHOR = """            has_residual=has_residual,
            use_pdl=self._use_pdl(),
            static_k=config.static_k,
        )
"""
SK_H2_REPL = """            has_residual=has_residual,
            use_pdl=self._use_pdl(),
            early_pdl_trigger=config.early_pdl_trigger,
            static_k=config.static_k,
        )
"""

SK_H3_ANCHOR = """                    item[1].k_unroll,
                    item[1].vector_width,
                    item[2],
"""
SK_H3_REPL = """                    item[1].k_unroll,
                    item[1].vector_width,
                    item[1].early_pdl_trigger,
                    item[2],
"""

# --- _custom_ops.py hunks -------------------------------------------------

CO_H1_ANCHOR = """def dsv3_fused_a_gemm(
    output: torch.Tensor,
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    enable_pdl: bool = False,
) -> None:
    \"\"\"Low-latency fused-A-style GEMM (SM 9.0+, BF16, 1-16 tokens).
"""
CO_H1_REPL = """_dsv3_early_pdl_schema: bool | None = None


def dsv3_fused_a_gemm_early_pdl_supported() -> bool:
    \"\"\"{MARKER}: True when the native op schema takes early_pdl_trigger.

    The csrc half of upstream #53525 (early-trigger template param, relaxed
    single-row stride check, new schema arg) requires rebuilding
    _C_stable_libtorch.abi3.so. This probe lets the Python side engage
    automatically once the binary is rebuilt, and stay stock before.
    \"\"\"
    global _dsv3_early_pdl_schema
    if _dsv3_early_pdl_schema is None:
        try:
            schema = str(torch.ops._C.dsv3_fused_a_gemm.default._schema)
            _dsv3_early_pdl_schema = "early_pdl_trigger" in schema
        except Exception:
            _dsv3_early_pdl_schema = False
    return _dsv3_early_pdl_schema


def dsv3_fused_a_gemm(
    output: torch.Tensor,
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    enable_pdl: bool = False,
    early_pdl_trigger: bool = False,
) -> None:
    \"\"\"Low-latency fused-A-style GEMM (SM 9.0+, BF16, 1-16 tokens).
""".replace("{MARKER}", MARKER)

CO_H2_ANCHOR = """    torch.ops._C.dsv3_fused_a_gemm(output, mat_a, mat_b, enable_pdl)
"""
CO_H2_REPL = """    if dsv3_fused_a_gemm_early_pdl_supported():
        torch.ops._C.dsv3_fused_a_gemm(
            output, mat_a, mat_b, enable_pdl, early_pdl_trigger
        )
    else:
        # Pre-#53525 native binary: no early_pdl_trigger schema arg. The
        # flag cannot be forwarded without a C++ rebuild; behavior is stock.
        torch.ops._C.dsv3_fused_a_gemm(output, mat_a, mat_b, enable_pdl)
"""

# --- low_latency_gemm.py hunks --------------------------------------------

K3_H1_ANCHOR = """from dataclasses import dataclass
from typing import Literal
"""
K3_H1_REPL = """from dataclasses import dataclass, replace
from typing import Literal
"""

K3_H2_ANCHOR = """# A resolved per-token-count call: the backend plus its CuTe config (None for
# dsv3, which needs no config).
ResolvedCall = tuple[Backend, SkinnyGemmConfig | None]
"""
K3_H2_REPL = """# {MARKER}: a resolved per-token-count call: the backend, its CuTe config
# (None for dsv3), and whether the dsv3 path may use the C=1 early PDL
# trigger.
ResolvedCall = tuple[Backend, SkinnyGemmConfig | None, bool]
""".replace("{MARKER}", MARKER)

K3_H3_ANCHOR = """    for num_tokens in range(1, 17):
        backend = _backend_for(spec, num_tokens, has_residual=False)
        if backend == "cute":
            plan[num_tokens] = ("cute", spec.cute_config(num_tokens))
        elif backend == "dsv3_fused_a":
            plan[num_tokens] = ("dsv3_fused_a", None)
    return plan
"""
K3_H3_REPL = """    for num_tokens in range(1, 17):
        backend = _backend_for(spec, num_tokens, has_residual=False)
        if backend == "cute":
            config = spec.cute_config(num_tokens)
            if num_tokens == 1 and spec.name in {"in_proj_qkvgfab", "o_proj"}:
                assert config is not None
                config = replace(config, early_pdl_trigger=True)
            plan[num_tokens] = ("cute", config, False)
        elif backend == "dsv3_fused_a":
            early_trigger = num_tokens == 1 and spec.name in {
                "f_b_proj",
                "fused_qkv_a_proj",
            }
            plan[num_tokens] = ("dsv3_fused_a", None, early_trigger)
    return plan
"""

K3_H4_ANCHOR = """def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        _is_packed_row_major(x)
        and _is_packed_row_major(weight)
"""
K3_H4_REPL = """def _runtime_ok(x: torch.Tensor, weight: torch.Tensor) -> bool:
    # {MARKER}: a size-1 leading dimension has no inter-row access, so the
    # C=1 plan accepts a non-packed leading stride. Gated on the native dsv3
    # op supporting early_pdl_trigger: the matching csrc stride-check
    # relaxation landed in the same PR, and the pre-#53525 binary rejects
    # such views (the cute path would then route them into dsv3 shapes).
    from vllm._custom_ops import dsv3_fused_a_gemm_early_pdl_supported

    x_ok = _is_packed_row_major(x) or (
        dsv3_fused_a_gemm_early_pdl_supported()
        and x.dim() == 2
        and x.shape[0] == 1
        and x.stride(1) == 1
    )
    return (
        x_ok
        and _is_packed_row_major(weight)
""".replace("{MARKER}", MARKER)

K3_H5_ANCHOR = """    entry = plan.get(x.shape[0])
    if entry is None:
        return None
    backend, config = entry
"""
K3_H5_REPL = """    entry = plan.get(x.shape[0])
    if entry is None:
        return None
    backend, config, early_pdl_trigger = entry
"""

K3_H6_ANCHOR = """    ops.dsv3_fused_a_gemm(output, x, weight.t(), enable_pdl=True)
    return output
"""
K3_H6_REPL = """    ops.dsv3_fused_a_gemm(
        output,
        x,
        weight.t(),
        enable_pdl=True,
        early_pdl_trigger=early_pdl_trigger,
    )
    return output
"""

K3_H7_ANCHOR = """        warmup_configs.update(config for _, config in spec.cute_configs)
        residual_warmup_configs.update(config for _, config in spec.residual_configs)
"""
K3_H7_REPL = """        plan = _build_plan(spec)
        warmup_configs.update(
            config
            for backend, config, _ in plan.values()
            if backend == "cute" and config is not None
        )
        residual_warmup_configs.update(config for _, config in spec.residual_configs)
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
        SKINNY_INT,
        [
            (SG_H1_ANCHOR, SG_H1_REPL, "__init__ early_pdl_trigger param"),
            (SG_H2_ANCHOR, SG_H2_REPL, "early_pdl_trigger attr"),
            (SG_H3_ANCHOR, SG_H3_REPL, "post-mainloop early PDL trigger"),
            (SG_H4_ANCHOR, SG_H4_REPL, "final trigger gated on early flag"),
        ],
    )
    ok &= patch_file(
        vroot,
        SKINNY,
        [
            (SK_H1_ANCHOR, SK_H1_REPL, "SkinnyGemmConfig.early_pdl_trigger"),
            (SK_H2_ANCHOR, SK_H2_REPL, "_compile flag passthrough"),
            (SK_H3_ANCHOR, SK_H3_REPL, "warmup sort key"),
        ],
    )
    ok &= patch_file(
        vroot,
        CUSTOM_OPS,
        [
            (CO_H1_ANCHOR, CO_H1_REPL, "schema probe + early_pdl_trigger param"),
            (CO_H2_ANCHOR, CO_H2_REPL, "schema-gated native call"),
        ],
    )
    ok &= patch_file(
        vroot,
        K3_GEMM,
        [
            (K3_H1_ANCHOR, K3_H1_REPL, "dataclasses.replace import"),
            (K3_H2_ANCHOR, K3_H2_REPL, "ResolvedCall 3-tuple"),
            (K3_H3_ANCHOR, K3_H3_REPL, "_build_plan C=1 early triggers"),
            (K3_H4_ANCHOR, K3_H4_REPL, "_runtime_ok single-row relaxation (gated)"),
            (K3_H5_ANCHOR, K3_H5_REPL, "_run_plan 3-tuple unpack"),
            (K3_H6_ANCHOR, K3_H6_REPL, "dsv3 early_pdl_trigger passthrough"),
            (K3_H7_ANCHOR, K3_H7_REPL, "warmup configs from plan"),
        ],
    )
    print(
        f"[{SCRIPT_NAME}] NOTE  csrc hunks NOT ported (dsv3_fused_a_gemm.cu "
        f"early-trigger + stride check, fused_kda_decode_kernel.cu deferred "
        f"grid-dep wait, torch_bindings.cpp schema): require rebuilding "
        f"_C_stable_libtorch.abi3.so. Python side auto-engages the native "
        f"parts once the binary carries the early_pdl_trigger schema arg."
    )
    print(
        f"[{SCRIPT_NAME}] NOTE  model.py hunk skipped: upstream formatting "
        f"no-op."
    )
    if ok:
        print(
            f"[{SCRIPT_NAME}] done. Cute skinny-GEMM C=1 early PDL trigger is "
            f"live; dsv3-side pieces await the csrc rebuild."
        )
    else:
        print(f"[{SCRIPT_NAME}] NOTE: some hunks skipped — see above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
