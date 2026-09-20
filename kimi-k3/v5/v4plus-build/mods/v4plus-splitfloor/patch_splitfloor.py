#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V4PLUS SPLITFLOOR — env-gated minimum-split floor for b12x dense-MLA.

The GB10 has 48 SMs. _active_dense_mla_splits() trims splits to KV need,
which underfills the grid at small batch (fused-8: ~16 CTAs vs 48 SMs).
VLLM_K3_MIN_SPLITS (default 0 = off) floors the returned split count so
small-batch grids fill ~48-96 CTAs. Extra splits cost only redundant
q-staging + merge (cheap when KV fits in L2).

SAFETY: the floor is clamped to plan.num_splits (scratch/merge are sized
for the built splits; the floor can only omit FEWER trailing splits).
Under CUDA graph capture the caller already uses all planned splits, so
the floor only affects the eager path. Recommended start: "2" at batch 1-4.
"""
import py_compile
import sys

SCRIPT_NAME = "patch_splitfloor"
TAG = "V4PLUS split-floor for 48-SM fill (VLLM_K3_MIN_SPLITS, default OFF)"


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


# ---------------------------------------------------------------------------
# envs.py — field + lambda, mirroring the VLLM_K3_FUSED_TILE pattern
# ---------------------------------------------------------------------------

E_FIELD_ANCHOR = '    VLLM_K3_FUSED_TILE: int = 8\n'
E_FIELD_REPLACEMENT = (
    '    VLLM_K3_FUSED_TILE: int = 8\n'
    '    VLLM_K3_MIN_SPLITS: int = 0\n'
)
E_FIELD_PRESENT = '    VLLM_K3_MIN_SPLITS: int = 0\n'

E_LAMBDA_ANCHOR = (
    '    "VLLM_K3_FUSED_TILE": lambda: int(os.getenv('
    '"VLLM_K3_FUSED_TILE", "8")),\n'
)
E_LAMBDA_REPLACEMENT = (
    '    "VLLM_K3_FUSED_TILE": lambda: int(os.getenv('
    '"VLLM_K3_FUSED_TILE", "8")),\n'
    '    "VLLM_K3_MIN_SPLITS": lambda: int(os.getenv('
    '"VLLM_K3_MIN_SPLITS", "0")),\n'
)
E_LAMBDA_PRESENT = '"VLLM_K3_MIN_SPLITS": lambda: int(os.getenv('


def patch_envs(vroot):
    """Add the VLLM_K3_MIN_SPLITS env field. Returns True on OK/SKIP."""
    path = vroot + "/envs.py"
    src = load(path)
    if src is None:
        return False
    changed = False
    if E_FIELD_PRESENT in src:
        print(f"[{SCRIPT_NAME}] SKIP  envs.py: field (already present)")
    elif src.count(E_FIELD_ANCHOR) == 1:
        src = src.replace(E_FIELD_ANCHOR, E_FIELD_REPLACEMENT, 1)
        changed = True
        print(f"[{SCRIPT_NAME}] APPLY envs.py: VLLM_K3_MIN_SPLITS field")
    else:
        print(
            f"[{SCRIPT_NAME}] NOTE  envs.py: field anchor found "
            f"{src.count(E_FIELD_ANCHOR)}x (want 1); hunk skipped"
        )
        return False
    if E_LAMBDA_PRESENT in src:
        print(f"[{SCRIPT_NAME}] SKIP  envs.py: lambda (already present)")
    elif src.count(E_LAMBDA_ANCHOR) == 1:
        src = src.replace(E_LAMBDA_ANCHOR, E_LAMBDA_REPLACEMENT, 1)
        changed = True
        print(f"[{SCRIPT_NAME}] APPLY envs.py: VLLM_K3_MIN_SPLITS lambda")
    else:
        print(
            f"[{SCRIPT_NAME}] NOTE  envs.py: lambda anchor found "
            f"{src.count(E_LAMBDA_ANCHOR)}x (want 1); hunk skipped"
        )
        return False
    if changed and not save(path, src):
        return False
    return True


# ---------------------------------------------------------------------------
# b12x_mla.py — floor inside _active_dense_mla_splits
# ---------------------------------------------------------------------------

MLA = "v1/attention/backends/mla/b12x_mla.py"

S_ANCHOR = (
    "    if max_seq_len is None:\n"
    "        return num_splits\n"
    "    valid_chunks = max(1, (max(0, int(max_seq_len)) + 63) // 64)\n"
    "    return min(\n"
    "        num_splits,\n"
    "        (valid_chunks + chunks_per_split - 1) // chunks_per_split,\n"
    "    )\n"
)
S_REPLACEMENT = (
    "    if max_seq_len is None:\n"
    "        return num_splits\n"
    "    valid_chunks = max(1, (max(0, int(max_seq_len)) + 63) // 64)\n"
    "    splits = min(\n"
    "        num_splits,\n"
    "        (valid_chunks + chunks_per_split - 1) // chunks_per_split,\n"
    "    )\n"
    "    # V4PLUS-SPLITFLOOR: floor the split count so small-batch grids\n"
    "    # fill ~48 SMs (GB10). Trim-to-KV-need underfills at small batch;\n"
    "    # extra splits cost only redundant q-staging + merge (cheap when KV\n"
    "    # fits in L2). Clamped to the plan's built splits: scratch/merge\n"
    "    # are sized for num_splits, so the floor only omits fewer trailing\n"
    "    # splits. Under graph capture the caller already uses all splits.\n"
    "    try:\n"
    "        _floor = int(envs.VLLM_K3_MIN_SPLITS)\n"
    "    except (ValueError, TypeError):\n"
    "        _floor = 0\n"
    "    if _floor > splits:\n"
    "        _floor = min(_floor, num_splits)\n"
    "        logger.info_once(\n"
    '            "B12X_MLA split floor active: splits %d -> %d (plan max "\n'
    '            "%d, VLLM_K3_MIN_SPLITS=%d).",\n'
    "            splits,\n"
    "            _floor,\n"
    "            num_splits,\n"
    "            int(envs.VLLM_K3_MIN_SPLITS),\n"
    "        )\n"
    "        return _floor\n"
    "    return splits\n"
)
S_PRESENT = "V4PLUS-SPLITFLOOR"


def patch_mla(vroot):
    """Apply the floor hunk. Returns True on OK/SKIP."""
    path = vroot + "/" + MLA
    src = load(path)
    if src is None:
        return False
    if S_PRESENT in src:
        print(f"[{SCRIPT_NAME}] SKIP  b12x_mla.py: split floor (already present)")
        return True
    if src.count(S_ANCHOR) != 1:
        print(
            f"[{SCRIPT_NAME}] NOTE  b12x_mla.py: split-floor anchor found "
            f"{src.count(S_ANCHOR)}x (want 1); hunk skipped — the floor is "
            f"NOT active, serving stays on trim-to-KV-need."
        )
        return False
    src = src.replace(S_ANCHOR, S_REPLACEMENT, 1)
    if not save(path, src):
        return False
    print(f"[{SCRIPT_NAME}] APPLY b12x_mla.py: split floor (VLLM_K3_MIN_SPLITS)")
    return True


def main():
    print(f"[{SCRIPT_NAME}] {TAG}")
    if len(sys.argv) != 2:
        print(f"usage: {SCRIPT_NAME}.py <VLLM_ROOT>")
        return 2
    vroot = sys.argv[1].rstrip("/")
    ok = patch_envs(vroot) and patch_mla(vroot)
    if ok:
        print(
            f"[{SCRIPT_NAME}] done. Serving knobs: VLLM_K3_MIN_SPLITS "
            f"(default 0 = off). Recommended start at batch 1-4: \"2\". "
            f"A/B vs off on the same recipe; watch the 'split floor active' "
            f"log line to confirm engagement."
        )
    else:
        print(f"[{SCRIPT_NAME}] NOTE: one or more hunks skipped — see above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
