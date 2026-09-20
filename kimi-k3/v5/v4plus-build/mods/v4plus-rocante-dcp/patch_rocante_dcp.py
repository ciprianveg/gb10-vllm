#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V4PLUS ROCANTE-DCP — RoCEnante (b12x.comm.roce one-shot RDMA) for the
DCP-group collectives.

At DCP8, per-step DCP-group collectives tax decode (short-ctx 13.85 vs
cp=1's 15.86): the query head-gather, the LSE reduce legs, and shard
gathers — decode-sized, latency-bound, exactly RoCEnante's win case
(-5-19% step latency at TP4 measured by batch1).

The fork's newer lineage wires DCP transports via a3fee72f5, but its fast
paths are CUDA-IPC (b12x.comm.pcie.DcpAllToAllPool — same-node only; on
cross-node DCP8 the IPC handle exchange fails and the pool self-disables
to NCCL, see dcp_alltoall.py's init-failure consensus).  This mod adapts
the same collectives to b12x.comm.roce (RDMA one-shot), which batch1
already ships, using the M2 adapter pattern verbatim: the DCP group's
CudaCommunicator gets its OWN B12xRoceAllReduce instance, so its
all_reduce/all_gather dispatch (per-instance, cuda_communicator.py
~315-334 / ~427-440) rides the identical path TP already uses — the TP
hook is untouched.

COLLECTIVE MAP (decode path at DCP8, B12X_MLA + dspark):
  1. Query head-gather (all_gather, dim=1):
     b12x_mla.py ~1103 dcp_b12x_all_gather_heads -> dcp_alltoall.py:473
     fallback cp_group.all_gather(local, dim=1) -> the DCP communicator's
     all_gather.  Decode-sized (local q ~0.77 MB at B=8/nst=6, gathered
     ~6.2 MB — under the shim's 16 MB AG cap).  WIRED: the PCIe pool
     stays preferred when it can initialize (same-node); cross-node it
     self-disables and the RoCEnante AG serves the fallback.
  2. LSE reduce, a2a backend: b12x_mla.py ~1296 dcp_a2a_lse_reduce ->
     PCIe pool lse_reduce_scatter (cross-node: disabled) -> NCCL
     dist.all_to_all_single (dcp_alltoall.py:1441).  FALLBACK NCCL —
     a direct torch.distributed call that bypasses the communicator, and
     b12x.comm.roce has no a2a surface (#295 is AR + AG only).
  3. LSE reduce, ag_rs variant / non-a2a backend: cp_lse_ag_out_rs
     (v1/attention/ops/common — NOT extracted).  Its all-gather legs ride
     the communicator hook automatically once wired; its reduce-scatter
     legs stay NCCL (no roce RS surface).  Flagged for the dry-run.
  4. KV shard gather (context/prefill path, not decode):
     dcp_utils.py init_kv_gather fallback
     torch.distributed.all_gather_into_tensor(group=device_group) —
     direct call, context-sized (above caps anyway).  FALLBACK NCCL.
  5. Pool-init consensus all_reduce (dcp_alltoall.py:110): 4 bytes,
     boot-time only.  NCCL.
  6. Generic-MLA query gather (dcp_utils.py:696 group.all_gather,
     dim=1): WIRED automatically — same communicator, benefits the
     non-b12x MLA backend too.
  TP-group collectives: untouched.

GROUP BINDING: B12xRoceAllReduce's constructor is group-parameterized
(``B12xRoceAllReduce(group=cpu_group, device_group=device_group,
device=device)`` — the same pattern as PyNcclCommunicator /
CustomAllreduce in this file), so a second instance bound to the DCP
group is the designed use.  The shim's own safety net makes the wiring
fail-safe: if construction fails or the underlying kernel declines the
group geometry, the instance comes up ``.disabled`` (or raises, caught
below) and every dispatch site falls through to exact NCCL.  This script
PROBES the image's shim at patch time (the batch1 payload artifact
vllm/distributed/device_communicators/b12x_roce_all_reduce.py): it must
exist, expose the group-parameterized ctor plus should_custom_ar /
custom_all_reduce / should_all_gather / all_gather, and show no
module-level singleton that would break a second instance.  The probe
bakes the VLLM_ENABLE_ROCE_DCP default ("1" healthy / "0" otherwise,
M2-style); the runtime env overrides the bake.

ENV: VLLM_ENABLE_ROCE_DCP (direct os.getenv — envs.py is not part of
this mod's extracted surface; the cgfix2 precedent).  Size caps are
REUSED: the shim enforces VLLM_ROCE_ALLREDUCE_MAX_SIZE /
VLLM_ROCE_ALLGATHER_MAX_SIZE per instance, so the DCP group inherits
them automatically.

PREREQUISITE (fail loud): cuda_communicator.py must carry the batch1 M2
post-state markers (the V4PLUS fork #597 hook).  The TP wiring is never
modified.

Idempotent: every hunk skips when its marker is present.  A missing
anchor prints a NOTE and skips; only a missing prerequisite,
file-not-found, or a broken post-patch compile exits non-zero.
"""

from __future__ import annotations

import os
import py_compile
import sys

SCRIPT_NAME = "patch_rocante_dcp"
TAG = "# V4PLUS-ROCANTE-DCP (RoCEnante for the DCP group)"

VLLM_ROOT = os.environ.get("VLLM_ROOT", "/opt/kimi-k3/vllm/vllm")
CUDA_COMMUNICATOR = os.path.join(
    VLLM_ROOT, "distributed", "device_communicators", "cuda_communicator.py"
)
SHIM = os.path.join(
    VLLM_ROOT,
    "distributed",
    "device_communicators",
    "b12x_roce_all_reduce.py",
)


def apply_hunks(path: str, hunks: list[tuple[str, str, str, str]]) -> bool:
    """Apply (name, anchor, replacement, present) hunks; True if all well."""
    try:
        with open(path) as f:
            src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {path} not found", file=sys.stderr)
        return False
    changed = False
    ok = True
    for name, anchor, repl, present in hunks:
        if present in src:
            print(f"[{SCRIPT_NAME}] SKIP  {os.path.basename(path)}: {name} (already present)")
            continue
        n = src.count(anchor)
        if n != 1:
            print(
                f"[{SCRIPT_NAME}] NOTE  {os.path.basename(path)}: {name} — "
                f"anchor found {n}x (want 1); hunk skipped"
            )
            ok = False
            continue
        src = src.replace(anchor, repl, 1)
        changed = True
        print(f"[{SCRIPT_NAME}] APPLY {os.path.basename(path)}: {name}")
    if changed:
        try:
            compile(src, path, "exec")
        except SyntaxError as exc:
            print(
                f"[{SCRIPT_NAME}] ERROR: {path} does not compile after patch: {exc}",
                file=sys.stderr,
            )
            return False
        with open(path, "w") as f:
            f.write(src)
        py_compile.compile(path, doraise=True)
    return ok


def check_prerequisites() -> bool:
    """Fail loud unless cuda_communicator.py is the batch1 M2 post-state."""
    try:
        with open(CUDA_COMMUNICATOR) as f:
            src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {CUDA_COMMUNICATOR} not found", file=sys.stderr)
        return False
    ok = True
    for marker in (
        "use_roce_allreduce = False  # V4PLUS (fork #597)",
        "B12xRoceAllReduce",
        "self.b12x_ar_comm",
        "V4PLUS (fork #597 adapted): RoCEnante all-gather",
    ):
        if marker not in src:
            print(
                f"[{SCRIPT_NAME}] ERROR: {CUDA_COMMUNICATOR} lacks the "
                f"batch1 M2 marker {marker!r} — this mod builds on the "
                "batch1 RoCEnante post-state.",
                file=sys.stderr,
            )
            ok = False
    return ok


def probe_roce_shim() -> tuple[bool, str]:
    """Healthy-probe the image's B12xRoceAllReduce shim (M2-style bake).

    Returns (healthy, report).  Healthy means: the shim exists and exposes
    the group-parameterized ctor plus the AR/AG dispatch surface, with no
    module-level singleton that would break a second (DCP-group) instance.
    """
    try:
        with open(SHIM) as f:
            shim = f.read()
    except FileNotFoundError:
        return False, f"shim not found at {SHIM}"
    missing = [
        marker
        for marker in (
            "class B12xRoceAllReduce",
            "def __init__(",
            "group",
            "device_group",
            "def should_custom_ar(",
            "def custom_all_reduce(",
            "def should_all_gather(",
            "def all_gather(",
        )
        if marker not in shim
    ]
    if missing:
        return False, f"shim lacks required surface: {missing}"
    # Singleton red flags: a module-level instance/global comm cache that
    # would make a second group binding unsound.
    for flag in (
        "_GLOBAL_",
        "_SINGLETON",
        "= B12xRoceAllReduce(",  # module-level instantiation
    ):
        if flag in shim:
            return False, f"singleton red flag {flag!r} in the shim"
    return True, "shim surface OK (group-parameterized ctor, AR+AG, no singleton flags)"


# ---------------------------------------------------------------------------
# Hunk 1: import os (for the direct env read; envs.py is not in scope).
# ---------------------------------------------------------------------------

C_IMPORT_ANCHOR = (
    "import torch\n"
    "from torch.distributed import ProcessGroup\n"
)
C_IMPORT_REPLACEMENT = (
    "import os\n"
    "\n"
    "import torch\n"
    "from torch.distributed import ProcessGroup\n"
)
C_IMPORT_PRESENT = "import os\n\nimport torch\n"

# ---------------------------------------------------------------------------
# Hunk 2: module-level baked enable for the DCP wiring.
# ---------------------------------------------------------------------------

C_ENABLE_ANCHOR = "logger = init_logger(__name__)\n"
C_ENABLE_TEMPLATE = (
    "logger = init_logger(__name__)\n"
    "\n"
    "# V4PLUS-ROCANTE-DCP: RoCEnante for the DCP-group collectives\n"
    "# (decode-path query head-gather + all-gather/all-reduce legs).\n"
    "# Default baked at mod time from a healthy-probe of the image's\n"
    "# B12xRoceAllReduce shim; VLLM_ENABLE_ROCE_DCP=0/false disables at\n"
    "# runtime. The shim's own size caps (VLLM_ROCE_ALLREDUCE_MAX_SIZE /\n"
    "# VLLM_ROCE_ALLGATHER_MAX_SIZE) apply per instance and are reused\n"
    "# unchanged for the DCP group.\n"
    "_ROCE_DCP_ENABLED = os.getenv(\n"
    '    "VLLM_ENABLE_ROCE_DCP", "{default}"\n'
    ") not in (\"0\", \"false\", \"False\")\n"
)
C_ENABLE_PRESENT = "_ROCE_DCP_ENABLED = os.getenv("

# ---------------------------------------------------------------------------
# Hunk 3: the DCP slot in CudaCommunicator.__init__ (non-TP branch).
# ---------------------------------------------------------------------------

C_GATE_ANCHOR = (
    "        if \"tp\" not in unique_name:\n"
    "            # custom allreduce or torch symm mem can be used only by tp\n"
    "            use_custom_allreduce = False\n"
    "            use_torch_symm_mem = False\n"
    "            use_flashinfer_allreduce = False\n"
    "            use_aiter_allreduce = False\n"
    "            use_roce_allreduce = False  # V4PLUS (fork #597)\n"
    "        else:\n"
)
C_GATE_REPLACEMENT = (
    "        if \"tp\" not in unique_name:\n"
    "            # custom allreduce or torch symm mem can be used only by tp\n"
    "            use_custom_allreduce = False\n"
    "            use_torch_symm_mem = False\n"
    "            use_flashinfer_allreduce = False\n"
    "            use_aiter_allreduce = False\n"
    "            use_roce_allreduce = False  # V4PLUS (fork #597)\n"
    "            # V4PLUS-ROCANTE-DCP: the DCP group's communicator gets\n"
    "            # its own RoCEnante slot — the same B12xRoceAllReduce\n"
    "            # shim batch1 wired for TP (the ctor binds the given\n"
    "            # group; the all_reduce/all_gather dispatch is\n"
    "            # per-instance). This routes the decode-path DCP\n"
    "            # collectives that fall through to cp_group.all_gather\n"
    "            # or a group all-reduce through the RDMA one-shot path\n"
    "            # under the shim's existing size caps. The a2a LSE\n"
    "            # reduce (dist.all_to_all_single) and any\n"
    "            # reduce-scatter legs have no b12x.comm.roce surface\n"
    "            # and stay NCCL. The shim's health gating (.disabled /\n"
    "            # should_* declination) falls back to exact NCCL above\n"
    "            # caps or when the surface is absent.\n"
    "            if (\n"
    "                \"cp\" in unique_name\n"
    "                and \"tp\" not in unique_name\n"
    "                and _ROCE_DCP_ENABLED\n"
    "                and self.world_size > 1\n"
    "            ):\n"
    "                use_roce_allreduce = True\n"
    "                logger.info(\n"
    "                    \"RoCEnante DCP collectives enabled for group %r \"\n"
    "                    \"(VLLM_ENABLE_ROCE_DCP default baked from the \"\n"
    "                    \"shim probe).\",\n"
    "                    unique_name,\n"
    "                )\n"
    "        else:\n"
)
C_GATE_PRESENT = "V4PLUS-ROCANTE-DCP: the DCP group's communicator gets"


def main() -> int:
    print(f"[{SCRIPT_NAME}] {TAG}")
    print(f"[{SCRIPT_NAME}] VLLM_ROOT={VLLM_ROOT}")
    if not check_prerequisites():
        print(
            f"[{SCRIPT_NAME}] PREREQUISITE FAILED: cuda_communicator.py is "
            "not the batch1 M2 post-state; refusing to patch.",
            file=sys.stderr,
        )
        return 1

    healthy, probe_report = probe_roce_shim()
    default = "1" if healthy else "0"
    print(
        f"[{SCRIPT_NAME}] b12x_roce_all_reduce shim probe: "
        f"{'HEALTHY' if healthy else 'UNHEALTHY'} -> "
        f"VLLM_ENABLE_ROCE_DCP default '{default}' ({probe_report})"
    )
    if not healthy:
        print(
            f"[{SCRIPT_NAME}] *** LOUD NOTE: the shim probe failed — the "
            "DCP wiring is baked OFF (exact-NCCL behavior preserved). "
            "Fix the shim or force VLLM_ENABLE_ROCE_DCP=1 at runtime to "
            "override the bake. ***"
        )

    ok = apply_hunks(
        CUDA_COMMUNICATOR,
        [
            ("import os", C_IMPORT_ANCHOR, C_IMPORT_REPLACEMENT, C_IMPORT_PRESENT),
            (
                "baked DCP enable constant",
                C_ENABLE_ANCHOR,
                C_ENABLE_TEMPLATE.format(default=default),
                C_ENABLE_PRESENT,
            ),
            (
                "DCP RoCEnante slot in __init__",
                C_GATE_ANCHOR,
                C_GATE_REPLACEMENT,
                C_GATE_PRESENT,
            ),
        ],
    )

    if ok:
        print(
            f"[{SCRIPT_NAME}] NOTE: the TP-group hook is untouched; the "
            "construction site (elif use_roce_allreduce and world_size > 1 "
            "-> B12xRoceAllReduce(group=self.cpu_group, ...)) and both "
            "dispatch sites are per-instance and now serve the DCP "
            "communicator too."
        )
        print(
            f"[{SCRIPT_NAME}] NOTE: the group-name gate matches any "
            "non-TP group whose unique_name contains 'cp' (covers 'cp' "
            "and 'dcp' naming). The boot INFO line prints the actual "
            "unique_name — if the DCP group is named differently, report "
            "it and the gate will be adjusted."
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
