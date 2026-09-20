#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V4PLUS MARLIN-NOPAD (part 2) — skip the expert-layer marlin roundup.

The SECOND pad that kills marlin-noEP: RoutedExperts.__init__ calls
Mxfp4MoEMethod.maybe_roundup_sizes -> mxfp4_round_up_hidden_size_and_
intermediate_size, whose MARLIN branch does round_up(intermediate, 128).
At TP16 noEP the per-partition intermediate is 192 -> rounded to 256,
so create_weights allocates 896 experts x 256 = the same ~118.9 GiB/rank
even when the model-level pad (part 1) is skipped.

Marlin's actual tile constraints are satisfied at raw 192:
  w13: n=2*192=384 (%128==0), k=7168 (%64==0)
  w2:  n=7168      (%128==0), k=192  (%64==0)
(gptq_marlin_repack tiles 16x64; moe marlin requires n%128==0, k%64==0.)
So for K3 the roundup is pure memory waste. VLLM_K3_MARLIN_NOPAD=1 (same
knob as part 1) returns the sizes unchanged for MARLIN/BATCHED_MARLIN.
Default OFF = stock roundup, bit-identical.
"""
import py_compile
import sys

SCRIPT_NAME = "patch_marlin_roundup"
TAG = ("V4PLUS marlin-noEP expert-layer roundup skip "
       "(VLLM_K3_MARLIN_NOPAD, default OFF)")

ORACLE = "model_executor/layers/fused_moe/oracle/mxfp4.py"

U_ANCHOR = '''    elif backend in (Mxfp4MoeBackend.MARLIN, Mxfp4MoeBackend.BATCHED_MARLIN):
        intermediate_size = round_up(intermediate_size, 128)
        if current_platform.is_xpu():
            hidden_size = round_up(hidden_size, 128)
        else:
            hidden_size = round_up(hidden_size, 256)
'''

U_REPLACEMENT = '''    elif backend in (Mxfp4MoeBackend.MARLIN, Mxfp4MoeBackend.BATCHED_MARLIN):
        # fix-k3-marlin-nopad (part 2): the expert-layer roundup is the
        # second pad defeating noEP — round_up(192, 128) = 256 re-inflates
        # create_weights to the same ~118.9 GiB/rank the model-level pad
        # caused. K3's raw 192 already satisfies marlin tiles
        # (w13 n=384%128=0 k=7168%64=0; w2 n=7168%128=0 k=192%64=0),
        # so with VLLM_K3_MARLIN_NOPAD=1 return sizes unchanged.
        import os
        if os.getenv("VLLM_K3_MARLIN_NOPAD", "0") == "1":
            return hidden_size, intermediate_size
        intermediate_size = round_up(intermediate_size, 128)
        if current_platform.is_xpu():
            hidden_size = round_up(hidden_size, 128)
        else:
            hidden_size = round_up(hidden_size, 256)
'''

U_PRESENT = "fix-k3-marlin-nopad (part 2)"


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


def patch_oracle(vroot):
    """Add the env-gated roundup skip. Returns True on OK/SKIP."""
    path = vroot + "/" + ORACLE
    src = load(path)
    if src is None:
        return False
    if U_PRESENT in src:
        print(f"[{SCRIPT_NAME}] SKIP  oracle/mxfp4.py (already present)")
        return True
    if src.count(U_ANCHOR) != 1:
        print(
            f"[{SCRIPT_NAME}] NOTE  oracle/mxfp4.py: roundup anchor found "
            f"{src.count(U_ANCHOR)}x (want 1); hunk skipped — serving stays "
            f"on the roundup path."
        )
        return False
    src = src.replace(U_ANCHOR, U_REPLACEMENT, 1)
    if not save(path, src):
        return False
    print(f"[{SCRIPT_NAME}] APPLY oracle/mxfp4.py: marlin roundup skip "
          f"(VLLM_K3_MARLIN_NOPAD, default OFF)")
    return True


def main():
    print(f"[{SCRIPT_NAME}] {TAG}")
    if len(sys.argv) != 2:
        print(f"usage: {SCRIPT_NAME}.py <VLLM_ROOT>")
        return 2
    vroot = sys.argv[1].rstrip("/")
    ok = patch_oracle(vroot)
    if ok:
        print(
            f"[{SCRIPT_NAME}] done. Same knob as part 1: "
            f"VLLM_K3_MARLIN_NOPAD=1. Both pads must be skipped together."
        )
    else:
        print(f"[{SCRIPT_NAME}] NOTE: hunk skipped — see above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
