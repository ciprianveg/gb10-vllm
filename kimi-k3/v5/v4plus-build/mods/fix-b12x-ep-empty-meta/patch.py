#!/usr/bin/env python3
"""Patch B12X EP workspace planning to tolerate empty expert metadata.

b12x_ep_moe.py::workspace_shapes raises ValueError when the expert
metadata reports 0 local experts while weights are prepared for the full
local set (seen: metadata=0, prepared=56 at TP16+DCP8+EP boot, warmup
forward). The plan itself is built from `prepared`, and `local_num_experts`
is not consumed downstream of the check — so planning for prepared
capacity when metadata is empty is safe; real mismatches (local > 0 and
different) still raise.

Usage: patch.py <path-to-b12x_ep_moe.py>  (path passed by run.sh)
"""
import py_compile
import sys

MARKER = "FIX-B12X-EP-EMPTY-META"

OLD = """\
        if prepared.num_experts != int(local_num_experts):
            raise ValueError(
                "B12X EP local expert metadata does not match prepared weights: "
                f"metadata={int(local_num_experts)}, "
                f"prepared={prepared.num_experts}"
            )
"""

NEW = """\
        if prepared.num_experts != int(local_num_experts):
            if int(local_num_experts) == 0:
                # FIX-B12X-EP-EMPTY-META: warmup/empty microbatches (or ranks
                # with no live experts this step) report 0 local experts
                # while weights are prepared for the full local set.
                # Workspace planning must cover capacity (prepared), not live
                # occupancy: proceed with the prepared plan instead of
                # aborting boot. Real mismatches (local > 0) still raise.
                if not globals().get("_FIX_B12X_EP_EMPTY_META_WARNED", False):
                    globals()["_FIX_B12X_EP_EMPTY_META_WARNED"] = True
                    print(
                        "[fix-b12x-ep-empty-meta] B12X EP empty expert "
                        f"metadata (0) with {prepared.num_experts} prepared "
                        "experts: planning workspace for prepared capacity.",
                        flush=True,
                    )
            else:
                raise ValueError(
                    "B12X EP local expert metadata does not match prepared weights: "
                    f"metadata={int(local_num_experts)}, "
                    f"prepared={prepared.num_experts}"
                )
"""


def main():
    path = sys.argv[1]
    with open(path) as f:
        content = f.read()
    if MARKER in content:
        print("  SKIP: already applied")
        return 0
    if content.count(OLD) != 1:
        print(f"  ERROR: anchor found {content.count(OLD)}x (want 1); refusing to patch")
        return 1
    content = content.replace(OLD, NEW, 1)
    with open(path, "w") as f:
        f.write(content)
    py_compile.compile(path, doraise=True)
    print("  OK: applied empty-metadata tolerance (+py_compile clean)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
