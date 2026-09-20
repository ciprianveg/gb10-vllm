#!/usr/bin/env python3
# V4PLUS M1 — b12x PR #295 (RoCEnante kernel side): add b12x/comm/roce.
#
# One-shot RDMA all-reduce/all-gather for multi-node DGX Spark TP over the
# ConnectX-7 RoCE ports. Purely ADDITIVE against the v4-era b12x: its comm/
# contains only pcie/ (verified at both 8596afcf1 and 9bc5f0cd9). Runtime-JIT
# CuTe DSL kernels + a C proxy built with the host cc at first use — no .so
# rebuild, so a source-tree mod is sufficient.
#
# Files (all VERBATIM from the b12x#295 patch, extracted to payload_roce/):
#   b12x/comm/roce/{__init__,api,roce_oneshot,_oneshot_cute,_allgather_cute,
#   _cute_intrinsics,_proxy}.py + _roce_proxy.c   (8 new files)
# Registration (minimal #295 additions outside comm/roce/):
#   b12x/__init__.py      : _OPS += "comm.roce"
#   b12x/comm/__init__.py : docstring + _OP_MODULES += "roce"
# SKIPPED from #295 (noted): pyproject.toml package-data for the .c file
#   (irrelevant for a source-tree mod — no rebuild/reinstall happens);
#   README/docs/benchmarks/tests.
#
# Anchor provenance: both registration anchors verified count==1 against the
# local b12x clones at 8596afcf1 AND 9bc5f0cd9 (identical content for these
# files at both revisions). The "comm.pcie" _OPS anchor is lineage-tolerant
# (the #295 diff context includes ops that postdate v4; anchoring on the
# single "comm.pcie", line instead).
#
# Idempotent: payload files skip when present with identical content (hash);
# a differing existing file is left UNCHANGED with a loud note. Registration
# hunks are marker-based. py_compile per touched .py. Missing anchors skip
# with notes, never fail.

import hashlib
import os
import py_compile
import shutil
import sys

SCRIPT_NAME = "patch_b12x_roce_kernel"

B12X_ROOT = os.environ.get("B12X_ROOT", "/opt/kimi-k3/b12x/b12x")
MOD_DIR = os.path.dirname(os.path.abspath(__file__))
PAYLOAD_DIR = os.path.join(MOD_DIR, "payload_roce")

ROCE_FILES = [
    "__init__.py",
    "api.py",
    "roce_oneshot.py",
    "_oneshot_cute.py",
    "_allgather_cute.py",
    "_cute_intrinsics.py",
    "_proxy.py",
    "_roce_proxy.c",
]

INIT_PATH = os.path.join(B12X_ROOT, "__init__.py")
COMM_INIT_PATH = os.path.join(B12X_ROOT, "comm", "__init__.py")

# b12x/__init__.py: register the op. Lineage-tolerant anchor (see header).
OPS_ANCHOR = '    "comm.pcie",\n'
OPS_REPLACEMENT = (
    '    "comm.pcie",\n'
    '    "comm.roce",  # V4PLUS (b12x #295)\n'
)
OPS_PRESENT = '    "comm.roce",  # V4PLUS (b12x #295)\n'

# b12x/comm/__init__.py: docstring + lazy module list (verbatim #295 hunks).
COMM_DOC_ANCHOR = (
    "``pcie``: collectives for consumer PCIe fabrics (no NVLink) — one-shot and\n"
    "  DMA/CE-ring all-reduce, FP8-transport two-shot reduce-scatter, and the DCP\n"
    "  attention all-to-all with fused LSE merge.\n"
    '"""\n'
)
COMM_DOC_REPLACEMENT = (
    "``pcie``: collectives for consumer PCIe fabrics (no NVLink) — one-shot and\n"
    "  DMA/CE-ring all-reduce, FP8-transport two-shot reduce-scatter, and the DCP\n"
    "  attention all-to-all with fused LSE merge.\n"
    "- ``roce`` (RoCEnante): one-shot RDMA all-reduce and all-gather for tensor\n"
    "  parallelism across DGX Spark nodes over their ConnectX-7 RoCE ports.\n"
    '"""\n'
)
COMM_DOC_PRESENT = "``roce`` (RoCEnante): one-shot RDMA all-reduce and all-gather for tensor"

COMM_MODS_ANCHOR = '_OP_MODULES = ("pcie",)\n'
COMM_MODS_REPLACEMENT = '_OP_MODULES = ("pcie", "roce")  # V4PLUS (b12x #295)\n'
COMM_MODS_PRESENT = '_OP_MODULES = ("pcie", "roce")  # V4PLUS (b12x #295)\n'


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def install_roce_files():
    dst_dir = os.path.join(B12X_ROOT, "comm", "roce")
    os.makedirs(dst_dir, exist_ok=True)
    for fname in ROCE_FILES:
        src = os.path.join(PAYLOAD_DIR, fname)
        dst = os.path.join(dst_dir, fname)
        if not os.path.isfile(src):
            print(f"  ERROR: payload missing: {src}", file=sys.stderr)
            return False
        if os.path.isfile(dst):
            if md5(src) == md5(dst):
                print(f"  SKIP (identical): comm/roce/{fname}")
                continue
            print(
                f"  *** LOUD NOTE: {dst} already exists with DIFFERENT content; "
                "left unchanged — review before baking ***"
            )
            continue
        shutil.copyfile(src, dst)
        print(f"  APPLIED: installed comm/roce/{fname} (verbatim b12x #295)")
        if fname.endswith(".py"):
            py_compile.compile(dst, doraise=True)
    return True


def patch_registration():
    for label, path, hunks in (
        (
            "b12x/__init__.py",
            INIT_PATH,
            [("register comm.roce op", OPS_ANCHOR, OPS_REPLACEMENT, OPS_PRESENT)],
        ),
        (
            "b12x/comm/__init__.py",
            COMM_INIT_PATH,
            [
                ("docstring", COMM_DOC_ANCHOR, COMM_DOC_REPLACEMENT, COMM_DOC_PRESENT),
                ("_OP_MODULES", COMM_MODS_ANCHOR, COMM_MODS_REPLACEMENT, COMM_MODS_PRESENT),
            ],
        ),
    ):
        try:
            with open(path) as f:
                src = f.read()
        except FileNotFoundError:
            print(f"[{SCRIPT_NAME}] ERROR: {path} not found", file=sys.stderr)
            return 1
        changed = False
        for name, anchor, replacement, present in hunks:
            if present in src:
                print(f"  SKIP (already patched) [{label}]: {name}")
                continue
            count = src.count(anchor)
            if count != 1:
                print(
                    f"  NOTE (not applicable) [{label}]: {name}: anchor found "
                    f"{count}x (expected 1); hunk skipped — review before baking"
                )
                continue
            src = src.replace(anchor, replacement, 1)
            print(f"  APPLIED [{label}]: {name}")
            changed = True
        if changed:
            with open(path, "w") as f:
                f.write(src)
            py_compile.compile(path, doraise=True)
            print(f"[{SCRIPT_NAME}] {label} py_compile OK")
    return 0


def main() -> int:
    print(f"[{SCRIPT_NAME}] b12x #295 RoCEnante kernel side (B12X_ROOT={B12X_ROOT})")
    if not install_roce_files():
        return 1
    rc = patch_registration()
    if rc:
        return rc
    if not os.path.isfile(os.path.join(B12X_ROOT, "comm", "roce", "roce_oneshot.py")):
        print(
            f"[{SCRIPT_NAME}] *** LOUD NOTE: comm/roce/roce_oneshot.py is absent "
            "after install — RoCEnante will not import; the vLLM adapter (M2) "
            "will default OFF. ***"
        )
    print(
        f"[{SCRIPT_NAME}] NOTE: pyproject.toml package-data (\"b12x.comm.roce\" "
        "= [\"*.c\"]) intentionally NOT applied — this is a source-tree mod, no "
        "rebuild/reinstall happens; _proxy.py builds the .c at first use."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
