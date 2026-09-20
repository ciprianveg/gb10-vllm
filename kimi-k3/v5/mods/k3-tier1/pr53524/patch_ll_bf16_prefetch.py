#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PR53524 — backport of upstream vLLM PR #53524.

Prefetch the static ll_bf16 router weight stripe into registers *before*
the PDL grid-dependency wait when M==1: the weight B is independent of the
producer kernel, so it can be resident while PDL waits for the activation A.
A second LLBf16Gemm instance (ll_bf16_gemm_c1_pdl_kernel,
prefetch_pdl_weights=True) is selected automatically for M==1 in
ll_bf16_gemm().

Pure Python / CuTe DSL (JIT-compiled at runtime) — no C++ rebuild needed.
Idempotent via marker. All touched files are py_compile'd with doraise.
"""
import py_compile
import sys

SCRIPT_NAME = "patch_ll_bf16_prefetch"
TAG = "pr53524 (upstream #53524 backport)"
MARKER = "pr53524 (upstream #53524)"

DOTPROD = "model_executor/kernels/linear/cute_dsl/_ll_bf16_dotprod.py"
LLBF16 = "model_executor/kernels/linear/cute_dsl/ll_bf16.py"

# --- _ll_bf16_dotprod.py hunks -------------------------------------------

DP_H1_ANCHOR = """        main_vec_width: int = 8,
        tail_vec_width: int = 4,
        use_pdl: bool = False,
    ):
"""
DP_H1_REPL = """        main_vec_width: int = 8,
        tail_vec_width: int = 4,
        use_pdl: bool = False,
        prefetch_pdl_weights: bool = False,  # {MARKER}
    ):
""".replace("{MARKER}", MARKER)

DP_H2_ANCHOR = """        self.use_pdl = use_pdl
        self.num_warps = bs // cute.arch.WARP_SIZE
        self._init_k_tiles(k)
"""
DP_H2_REPL = """        self.use_pdl = use_pdl
        self.prefetch_pdl_weights = prefetch_pdl_weights
        self.num_warps = bs // cute.arch.WARP_SIZE
        self._init_k_tiles(k)
        self.main_prefetch_tiles = min(self.main_tiles, 8)
"""

DP_H3_ANCHOR = """                for v in cutlass.range_constexpr(vec_width):
                    acc[m] = acc[m] + ar[v].to(cutlass.Float32) * br_f32[v]

    def _make_thread_vector_slice(
"""
DP_H3_REPL = """                for v in cutlass.range_constexpr(vec_width):
                    acc[m] = acc[m] + ar[v].to(cutlass.Float32) * br_f32[v]

    @cute.jit
    def _vector_dotprod_prefetched(
        self,
        acc: cute.Tensor,
        tA: cute.Tensor,
        tB: cute.Tensor,
        rB: cute.Tensor,
        M: cutlass.Constexpr,
        num_tiles: cutlass.Constexpr,
        prefetch_tiles: cutlass.Constexpr,
    ):
        for tile in cutlass.range_constexpr(num_tiles):
            if const_expr(tile < prefetch_tiles):
                br_f32 = rB[tile, None].load().to(cutlass.Float32)
            else:
                bt = tB[None, tile]
                br = cute.make_rmem_tensor_like(bt)
                cute.autovec_copy(bt, br)
                br_f32 = br.load().to(cutlass.Float32)
            for m in cutlass.range_constexpr(M):
                at = tA[m, None, tile]
                ar = cute.make_rmem_tensor_like(at)
                cute.autovec_copy(at, ar)
                vec_width: cutlass.Constexpr = cute.size(ar)
                for v in cutlass.range_constexpr(vec_width):
                    acc[m] = acc[m] + ar[v].to(cutlass.Float32) * br_f32[v]

    def _make_thread_vector_slice(
"""

DP_H4_ANCHOR = """        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        # 128-bit vectorized main loop
        if const_expr(k_main_elems > 0):
            gA_main = self._make_k_slice(gA, 0, k_main_elems)
            gB_main = self._make_k_slice(gB, 0, k_main_elems)
            gA_vec = cute.logical_divide(gA_main, (None, main_vec_width))
            gB_vec = cute.logical_divide(gB_main, (None, main_vec_width))
            tA, tB = self._make_thread_vector_slice(gA_vec, gB_vec, tidx, n_idx, bs)
            self._vector_dotprod(acc, tA, tB, M, main_tiles, 16)
"""
DP_H4_REPL = """        # In the K3 C=1 specialization, the static weight stripe is independent
        # of the producer and can be resident while PDL waits for activation A.
        if const_expr(k_main_elems > 0):
            gA_main = self._make_k_slice(gA, 0, k_main_elems)
            gB_main = self._make_k_slice(gB, 0, k_main_elems)
            gA_vec = cute.logical_divide(gA_main, (None, main_vec_width))
            gB_vec = cute.logical_divide(gB_main, (None, main_vec_width))
            tA, tB = self._make_thread_vector_slice(gA_vec, gB_vec, tidx, n_idx, bs)
            if const_expr(self.use_pdl and self.prefetch_pdl_weights):
                prefetch_tiles: cutlass.Constexpr = self.main_prefetch_tiles
                prefetched_b = cute.make_rmem_tensor(
                    cute.make_layout(
                        (prefetch_tiles, main_vec_width),
                        stride=(main_vec_width, 1),
                    ),
                    cutlass.BFloat16,
                )
                for tile in cutlass.range_constexpr(prefetch_tiles):
                    cute.autovec_copy(tB[None, tile], prefetched_b[tile, None])

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        # 128-bit vectorized main loop
        if const_expr(k_main_elems > 0):
            if const_expr(self.use_pdl and self.prefetch_pdl_weights):
                self._vector_dotprod_prefetched(
                    acc,
                    tA,
                    tB,
                    prefetched_b,
                    M,
                    main_tiles,
                    prefetch_tiles,
                )
            else:
                self._vector_dotprod(acc, tA, tB, M, main_tiles, 16)
"""

# --- ll_bf16.py hunks -----------------------------------------------------

LB_H1_ANCHOR = """    def __init__(self) -> None:
        # Dot-prod: keyed on (M, K, bs), because M and K are Constexpr.
"""
LB_H1_REPL = """    def __init__(self, *, prefetch_pdl_weights: bool = False) -> None:
        # {MARKER}: C=1 PDL weight-prefetch specialization flag.
        self._prefetch_pdl_weights = prefetch_pdl_weights
        # Dot-prod: keyed on (M, K, bs), because M and K are Constexpr.
""".replace("{MARKER}", MARKER)

LB_H2_ANCHOR = """        gemm = LLBf16Dotprod(k=compile_key.K, bs=compile_key.bs, use_pdl=_use_pdl())
"""
LB_H2_REPL = """        gemm = LLBf16Dotprod(
            k=compile_key.K,
            bs=compile_key.bs,
            use_pdl=_use_pdl(),
            prefetch_pdl_weights=self._prefetch_pdl_weights,
        )
"""

LB_H3_ANCHOR = """ll_bf16_gemm_kernel = LLBf16Gemm()
"""
LB_H3_REPL = """ll_bf16_gemm_kernel = LLBf16Gemm()
ll_bf16_gemm_c1_pdl_kernel = LLBf16Gemm(prefetch_pdl_weights=True)
"""

LB_H4_ANCHOR = """    return ll_bf16_gemm_kernel(hidden_states, router_weight, output_dtype)
"""
LB_H4_REPL = """    kernel = (
        ll_bf16_gemm_c1_pdl_kernel
        if hidden_states.shape[0] == 1
        else ll_bf16_gemm_kernel
    )
    return kernel(hidden_states, router_weight, output_dtype)
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
    """Apply (anchor, replacement, description) hunks; all-or-nothing."""
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
        DOTPROD,
        [
            (DP_H1_ANCHOR, DP_H1_REPL, "__init__ prefetch_pdl_weights param"),
            (DP_H2_ANCHOR, DP_H2_REPL, "prefetch attrs + main_prefetch_tiles"),
            (DP_H3_ANCHOR, DP_H3_REPL, "_vector_dotprod_prefetched jit method"),
            (DP_H4_ANCHOR, DP_H4_REPL, "kernel prefetch-before-PDL restructure"),
        ],
    )
    ok &= patch_file(
        vroot,
        LLBF16,
        [
            (LB_H1_ANCHOR, LB_H1_REPL, "LLBf16Gemm __init__ flag"),
            (LB_H2_ANCHOR, LB_H2_REPL, "_compile_dotprod flag passthrough"),
            (LB_H3_ANCHOR, LB_H3_REPL, "ll_bf16_gemm_c1_pdl_kernel instance"),
            (LB_H4_ANCHOR, LB_H4_REPL, "ll_bf16_gemm M==1 dispatch"),
        ],
    )
    if ok:
        print(
            f"[{SCRIPT_NAME}] done. M==1 ll_bf16 router GEMMs now prefetch the "
            f"weight stripe ahead of the PDL wait."
        )
    else:
        print(f"[{SCRIPT_NAME}] NOTE: some hunks skipped — see above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
