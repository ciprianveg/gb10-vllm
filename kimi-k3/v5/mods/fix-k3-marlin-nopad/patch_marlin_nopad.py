#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V4PLUS MARLIN-NOPAD — env-gated skip of the KimiMoE 192->256 pad.

KimiMoE.__init__ pads moe_intermediate 3072->4096 when the TP16
per-partition value 192 < min_moe_intermediate_per_partition (256), giving
~118.9 GiB MoE/rank without EP — certain pool death at load on the 121 GiB
GB10 (documented in kimi-k3-full-tp16.yaml:(c2)). b12x skips it via
use_native_b12x_intermediate; EP skips it via the baked EP-aware guard.

Marlin's tile math is satisfied at unpadded 192 (gate-up n=384 %128==0,
k=7168 %64==0; down n=7168 %128==0, k=192 %64==0), so for marlin-noEP the
pad only inflates memory. VLLM_K3_MARLIN_NOPAD=1 skips it (-> ~89 GiB/rank,
checkpoint-native 3072 shapes, no zero-fill block). Default OFF = stock.
Must be stacked with fix-marlin-chunked-repack (the 2x repack transient on
89 GiB still needs the 1x path).
"""
import py_compile
import sys

SCRIPT_NAME = "patch_marlin_nopad"
TAG = ("V4PLUS marlin-noEP nopad guard "
       "(VLLM_K3_MARLIN_NOPAD, default OFF)")

MODEL = "models/kimi_k3/nvidia/model.py"

N_ANCHOR = '''        if (
            self.tp_size > 1
            and not vllm_config.parallel_config.enable_expert_parallel
            and not use_native_b12x_intermediate
        ):
'''

N_REPLACEMENT = '''        # fix-k3-marlin-nopad: env-gated skip of the 192->256 pad for
        # marlin-noEP (VLLM_K3_MARLIN_NOPAD=1). Marlin tiles are satisfied
        # at unpadded 192; the pad only inflates per-rank MoE 89->118.9 GiB
        # and kills the 121 GiB pool at load. Default OFF (stock).
        if (
            self.tp_size > 1
            and not vllm_config.parallel_config.enable_expert_parallel
            and not use_native_b12x_intermediate
            and os.getenv("VLLM_K3_MARLIN_NOPAD", "0") != "1"
        ):
'''

N_PRESENT = "fix-k3-marlin-nopad"


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


def patch_model(vroot):
    """Add the env-gated nopad condition. Returns True on OK/SKIP."""
    path = vroot + "/" + MODEL
    src = load(path)
    if src is None:
        return False
    if N_PRESENT in src:
        print(f"[{SCRIPT_NAME}] SKIP  model.py (already present)")
        return True
    if src.count(N_ANCHOR) != 1:
        print(
            f"[{SCRIPT_NAME}] NOTE  model.py: nopad anchor found "
            f"{src.count(N_ANCHOR)}x (want 1); hunk skipped — serving stays "
            f"on the padded 4096 path."
        )
        return False
    src = src.replace(N_ANCHOR, N_REPLACEMENT, 1)
    if not save(path, src):
        return False
    print(f"[{SCRIPT_NAME}] APPLY model.py: nopad guard "
          f"(VLLM_K3_MARLIN_NOPAD, default OFF)")
    return True


def main():
    print(f"[{SCRIPT_NAME}] {TAG}")
    if len(sys.argv) != 2:
        print(f"usage: {SCRIPT_NAME}.py <VLLM_ROOT>")
        return 2
    vroot = sys.argv[1].rstrip("/")
    ok = patch_model(vroot)
    if ok:
        print(
            f"[{SCRIPT_NAME}] done. Serving knob: VLLM_K3_MARLIN_NOPAD=1 "
            f"skips the 192->256 pad (marlin-noEP only; stack with the "
            f"in-place repack mod). Watch for padded=3072 shapes at load."
        )
    else:
        print(f"[{SCRIPT_NAME}] NOTE: hunk skipped — see above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
