#!/usr/bin/env python3
# V4PLUS M5 — b12x f5394625c "gate speculative chunk copies + split-merge
# rewrite" over b12x/attention/_shared/mla/{decode_math,io,kernel,merge,smem}.py.
#
# Gates KV chunk copies that spec steps don't need and restructures the
# per-split merge on the B12X MLA decode/verify path under speculation.
# Helps at BOTH nst=6 and nst=3. CuTe DSL = runtime JIT — no .so rebuild.
#
# PORT METHOD: the commit is a rewrite (large context drift between the
# diff base and the v4-era b12x), so the 63 source hunks were extracted
# VERBATIM into payload_f5394625c_hunks.json as (pre-image, post-image)
# pairs, located by search (git-apply-style positional offsets), made
# unique by context extension, and validated BYTE-FOR-BYTE against a real
# `git apply` on BOTH v4-era reference checkouts (8596afcf1 and 9bc5f0cd9
# — the five target files are identical between them). No paraphrasing:
# every post-image is the commit's own text.
#
# APPLY SEMANTICS: PER-FILE ATOMIC. The hunks are interdependent (kernel.py
# rewrites call_extra_pertok and its callers together); a partial apply
# could compile yet be semantically broken. So each file is either fully
# patched or untouched:
#   * all post-images present  -> SKIP (already patched)
#   * some post-images present -> LOUD NOTE (inconsistent state), untouched
#   * every pre-image count==1 -> apply all, py_compile
#   * any anchor fails         -> whole file skipped with per-hunk notes
#
# SKIPPED from the commit (noted): the two test files (the task excludes
# tests; the follow-up 296cb8647 test-hardening patch is likewise skipped).
#
# Idempotent; missing anchors skip with notes, never fail; py_compile per
# patched file.

import json
import os
import py_compile
import sys

SCRIPT_NAME = "patch_b12x_f5394625c"

B12X_ROOT = os.environ.get("B12X_ROOT", "/opt/kimi-k3/b12x/b12x")
MOD_DIR = os.path.dirname(os.path.abspath(__file__))
HUNKS_PATH = os.path.join(MOD_DIR, "payload_f5394625c_hunks.json")


def patch_file(fname, info):
    # The JSON stores repo-relative paths ("b12x/attention/..."); B12X_ROOT
    # is the PACKAGE dir (…/b12x/b12x), so strip the leading "b12x/".
    rel_path = info["path"]
    assert rel_path.startswith("b12x/"), rel_path
    rel_path = rel_path[len("b12x/"):]
    path = os.path.join(B12X_ROOT, rel_path)
    try:
        with open(path) as f:
            src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {path} not found", file=sys.stderr)
        return 1

    hunks = info["hunks"]
    patched = sum(1 for h in hunks if h["post"] in src)
    if patched == len(hunks):
        print(f"  SKIP (already patched): {rel_path} ({len(hunks)} hunks)")
        return 0
    if patched > 0:
        print(
            f"  *** LOUD NOTE: {rel_path} is in an INCONSISTENT state "
            f"({patched}/{len(hunks)} post-images present). Left untouched — "
            "restore the pristine file or complete the patch by hand. ***"
        )
        return 0

    missing = [
        h["name"] for h in hunks if src.count(h["pre"]) != 1
    ]
    if missing:
        for name in missing:
            print(
                f"  NOTE (not applicable) [{rel_path}]: {name}: pre-image "
                "not found exactly once — file drifted from the v4-era base"
            )
        print(
            f"  *** LOUD NOTE: {rel_path} NOT PATCHED ({len(missing)}/{len(hunks)} "
            "anchors failed) — the split-merge rewrite is incomplete for this "
            "file. Since the hunks are interdependent, nothing was applied. ***"
        )
        return 0

    for h in hunks:
        src = src.replace(h["pre"], h["post"], 1)
    with open(path, "w") as f:
        f.write(src)
    py_compile.compile(path, doraise=True)
    print(f"  APPLIED: {rel_path} ({len(hunks)} hunks) — py_compile OK")
    return 0


def main() -> int:
    print(f"[{SCRIPT_NAME}] b12x f5394625c spec MLA merge rewrite (B12X_ROOT={B12X_ROOT})")
    try:
        with open(HUNKS_PATH) as f:
            table = json.load(f)["files"]
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {HUNKS_PATH} missing", file=sys.stderr)
        return 1

    rc = 0
    total = 0
    for fname in ["decode_math.py", "io.py", "kernel.py", "merge.py", "smem.py"]:
        info = table[fname]
        total += len(info["hunks"])
        r = patch_file(fname, info)
        rc = rc or r
    if rc:
        print(
            f"[{SCRIPT_NAME}] ERROR: at least one file was missing — see above"
        )
        return rc
    print(
        f"[{SCRIPT_NAME}] NOTE: the commit's two test files and the "
        "follow-up 296cb8647 test-hardening patch are intentionally SKIPPED "
        "(Batch 1 excludes tests). CuTe DSL kernels JIT at first use — "
        "purge/prime the JIT caches when benchmarking."
    )
    print(f"[{SCRIPT_NAME}] done ({total} hunks across 5 files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
