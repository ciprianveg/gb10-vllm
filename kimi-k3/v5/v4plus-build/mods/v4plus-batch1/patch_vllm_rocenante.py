#!/usr/bin/env python3
# V4PLUS M2 — fork #597 (RoCEnante vLLM adapter), adapted to the v4 tree.
#
# The fork's #597 wires the shim into a cuda_communicator that already has a
# b12x AR slot (dev/kimi B12X PCIe AR lineage). The v4 tree has NO b12x slot
# (verified: no b12x* files in device_communicators/), so this script hand-
# wires the equivalent:
#   * NEW  vllm/distributed/device_communicators/b12x_roce_all_reduce.py
#     (the fork's shim verbatim; ONE adaptation: _parse_byte_size imported
#     from custom_all_reduce.py — v4 has no b12x_pcie_all_reduce.py).
#   * vllm/envs.py: VLLM_ENABLE_ROCE_ALLREDUCE / VLLM_ROCE_ALLREDUCE_MAX_SIZE /
#     VLLM_ROCE_ALLGATHER_MAX_SIZE (#597 verbatim placement). The enable
#     default is BAKED from a b12x.comm.roce probe (API_VERSION 1): ON when
#     healthy, OFF + loud note otherwise. The shim's own capability voting
#     handles per-rank failures at runtime.
#   * cuda_communicator.py (adapted from #597): use_roce_allreduce flag,
#     b12x_ar_comm slot, RoCEnante constructed INSTEAD of the IPC
#     CustomAllreduce (which cannot span nodes), all_reduce dispatch before
#     the IPC/PCIe custom backends, all_gather dispatch, backend logging.
#   * vllm/v1/worker/gpu_worker.py: #597-verbatim fail-stop health check
#     (_B12xRoceCheckedAsyncOutput + _b12x_roce_guarded around
#     sample_tokens / execute_model outputs).
#
# Anchor provenance: ALL anchors verified against the local fork clone at
# 881ac39a4 (== the image tree). gpu_worker hunks are #597-verbatim (its
# anchors exist unchanged in v4). cuda_communicator hunks are adapted to v4's
# dispatch chain (no b12x/pcie branches here).
#
# Staged gating: envs -> shim -> flag/attr/constructor -> dispatch/logging,
# so a partial apply can never reference an undefined name.
# Idempotent; missing anchors skip with notes; py_compile per touched file.

import hashlib
import os
import py_compile
import shutil
import subprocess
import sys

SCRIPT_NAME = "patch_vllm_rocenante"

VLLM_ROOT = os.environ.get("VLLM_ROOT", "/opt/kimi-k3/vllm/vllm")
MOD_DIR = os.path.dirname(os.path.abspath(__file__))

ENVS_PATH = os.path.join(VLLM_ROOT, "envs.py")
CC_PATH = os.path.join(VLLM_ROOT, "distributed", "device_communicators", "cuda_communicator.py")
GW_PATH = os.path.join(VLLM_ROOT, "v1", "worker", "gpu_worker.py")
SHIM_DST = os.path.join(VLLM_ROOT, "distributed", "device_communicators", "b12x_roce_all_reduce.py")
SHIM_SRC = os.path.join(MOD_DIR, "payload_b12x_roce_all_reduce.py")


def probe_roce_kernel():
    """True/False when determinable, None when the probe itself failed."""
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                "from b12x.comm import roce; "
                "import sys; sys.exit(0 if getattr(roce, 'API_VERSION', None) == 1 else 1)",
            ],
            capture_output=True,
            timeout=120,
        )
        return proc.returncode == 0
    except Exception:
        return None


# ----------------------------------------------------------------------
# envs.py (#597-verbatim placement; enable default baked per probe)
# ----------------------------------------------------------------------
ENVS_FIELD_ANCHOR = (
    '    VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: str = "84KB"\n'
)
ENVS_FIELD_REPLACEMENT = (
    '    VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: str = "84KB"\n'
    "    # V4PLUS (fork #597)\n"
    "    VLLM_ENABLE_ROCE_ALLREDUCE: bool = False\n"
    '    VLLM_ROCE_ALLREDUCE_MAX_SIZE: str = "2MB"\n'
    '    VLLM_ROCE_ALLGATHER_MAX_SIZE: str = "16MB"\n'
)
ENVS_FIELD_PRESENT = '    VLLM_ROCE_ALLGATHER_MAX_SIZE: str = "16MB"\n'

ENVS_LAMBDA_ANCHOR = (
    '    "VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE": lambda: os.getenv(\n'
    '        "VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE", "84KB"\n'
    "    ),\n"
)
ENVS_LAMBDA_TEMPLATE = (
    "    # V4PLUS (fork #597): enable the b12x one-shot RoCE all-reduce for\n"
    "    # multi-node TP (DGX Spark).\n"
    "    \"VLLM_ENABLE_ROCE_ALLREDUCE\": lambda: bool(\n"
    "        int(os.getenv(\"VLLM_ENABLE_ROCE_ALLREDUCE\", \"{default}\"))\n"
    "    ),\n"
    "    \"VLLM_ROCE_ALLREDUCE_MAX_SIZE\": lambda: os.getenv(\n"
    "        \"VLLM_ROCE_ALLREDUCE_MAX_SIZE\", \"2MB\"\n"
    "    ),\n"
    "    # Largest per-rank shard routed to the RoCE all-gather\n"
    "    # (e.g. logits [rows, vocab/tp]).\n"
    "    \"VLLM_ROCE_ALLGATHER_MAX_SIZE\": lambda: os.getenv(\n"
    "        \"VLLM_ROCE_ALLGATHER_MAX_SIZE\", \"16MB\"\n"
    "    ),\n"
)
ENVS_LAMBDA_PRESENT = '    "VLLM_ENABLE_ROCE_ALLREDUCE": lambda: bool(\n'

# ----------------------------------------------------------------------
# cuda_communicator.py (adapted to v4's chain)
# ----------------------------------------------------------------------
CC_NONTP_ANCHOR = (
    "        if \"tp\" not in unique_name:\n"
    "            # custom allreduce or torch symm mem can be used only by tp\n"
    "            use_custom_allreduce = False\n"
    "            use_torch_symm_mem = False\n"
    "            use_flashinfer_allreduce = False\n"
    "            use_aiter_allreduce = False\n"
    "        else:\n"
)
CC_NONTP_REPLACEMENT = (
    "        if \"tp\" not in unique_name:\n"
    "            # custom allreduce or torch symm mem can be used only by tp\n"
    "            use_custom_allreduce = False\n"
    "            use_torch_symm_mem = False\n"
    "            use_flashinfer_allreduce = False\n"
    "            use_aiter_allreduce = False\n"
    "            use_roce_allreduce = False  # V4PLUS (fork #597)\n"
    "        else:\n"
)
CC_NONTP_PRESENT = "            use_roce_allreduce = False  # V4PLUS (fork #597)\n"

CC_TP_ANCHOR = (
    "            use_aiter_allreduce = use_custom_allreduce and bool(\n"
    "                rocm_aiter_ops.is_custom_all_reduce_enabled()\n"
    "            )\n"
    "\n"
    "        self.use_custom_allreduce = use_custom_allreduce\n"
)
CC_TP_REPLACEMENT = (
    "            use_aiter_allreduce = use_custom_allreduce and bool(\n"
    "                rocm_aiter_ops.is_custom_all_reduce_enabled()\n"
    "            )\n"
    "            # V4PLUS (fork #597 adapted): RoCEnante one-shot RDMA\n"
    "            # all-reduce for multi-node DGX Spark TP (b12x.comm.roce).\n"
    "            use_roce_allreduce = (\n"
    "                use_custom_allreduce and envs.VLLM_ENABLE_ROCE_ALLREDUCE\n"
    "            )\n"
    "            if use_roce_allreduce:\n"
    "                use_flashinfer_allreduce = False\n"
    "\n"
    "        self.use_custom_allreduce = use_custom_allreduce\n"
)
CC_TP_PRESENT = "            use_roce_allreduce = (\n"

CC_ATTR_ANCHOR = "        self.use_aiter_allreduce = use_aiter_allreduce\n"
CC_ATTR_REPLACEMENT = (
    "        self.use_aiter_allreduce = use_aiter_allreduce\n"
    "        self.use_roce_allreduce = use_roce_allreduce  # V4PLUS (fork #597)\n"
)
CC_ATTR_PRESENT = "        self.use_roce_allreduce = use_roce_allreduce  # V4PLUS (fork #597)\n"

CC_SLOT_ANCHOR = "        self.aiter_ar_comm: AiterCustomAllreduce | None = None\n"
CC_SLOT_REPLACEMENT = (
    "        self.aiter_ar_comm: AiterCustomAllreduce | None = None\n"
    "        self.b12x_ar_comm = None  # V4PLUS (fork #597): B12xRoceAllReduce | None\n"
)
CC_SLOT_PRESENT = "        self.b12x_ar_comm = None  # V4PLUS (fork #597): B12xRoceAllReduce | None\n"

CC_CA_GATE_ANCHOR = (
    "        if use_custom_allreduce and self.aiter_ar_comm is None and self.world_size > 1:\n"
    "            # Initialize a custom fast all-reduce implementation.\n"
    "            self.ca_comm = CustomAllreduce(\n"
)
CC_CA_GATE_REPLACEMENT = (
    "        if (\n"
    "            use_custom_allreduce\n"
    "            and not use_roce_allreduce  # V4PLUS (fork #597)\n"
    "            and self.aiter_ar_comm is None\n"
    "            and self.world_size > 1\n"
    "        ):\n"
    "            # Initialize a custom fast all-reduce implementation.\n"
    "            self.ca_comm = CustomAllreduce(\n"
)
CC_CA_GATE_PRESENT = "            and not use_roce_allreduce  # V4PLUS (fork #597)\n"

CC_CONSTRUCT_ANCHOR = (
    "                symm_mem_enabled=(\n"
    "                    self.symm_mem_comm is not None and not self.symm_mem_comm.disabled\n"
    "                ),\n"
    "                nccl_group=self.device_group,\n"
    "            )\n"
)
CC_CONSTRUCT_REPLACEMENT = (
    "                symm_mem_enabled=(\n"
    "                    self.symm_mem_comm is not None and not self.symm_mem_comm.disabled\n"
    "                ),\n"
    "                nccl_group=self.device_group,\n"
    "            )\n"
    "        elif use_roce_allreduce and self.world_size > 1:\n"
    "            # V4PLUS (fork #597 adapted): RoCEnante — multi-node DGX Spark\n"
    "            # one-shot RDMA collectives from b12x.comm.roce. Replaces the\n"
    "            # IPC CustomAllreduce, which cannot span nodes.\n"
    "            from vllm.distributed.device_communicators.b12x_roce_all_reduce import (\n"
    "                B12xRoceAllReduce,\n"
    "            )\n"
    "\n"
    "            self.b12x_ar_comm = B12xRoceAllReduce(\n"
    "                group=self.cpu_group,\n"
    "                device_group=self.device_group,\n"
    "                device=self.device,\n"
    "            )\n"
)
CC_CONSTRUCT_PRESENT = "            self.b12x_ar_comm = B12xRoceAllReduce(\n"

CC_AR_DISPATCH_ANCHOR = (
    "        if self.pynccl_comm is not None and should_nccl_symm_mem_allreduce(\n"
    "            self.pynccl_comm.world_size, input_\n"
    "        ):\n"
    "            out = torch.ops.vllm.all_reduce_symmetric_with_copy(input_)\n"
    "            if out is not None:\n"
    "                return out\n"
)
CC_AR_DISPATCH_REPLACEMENT = (
    "        if self.pynccl_comm is not None and should_nccl_symm_mem_allreduce(\n"
    "            self.pynccl_comm.world_size, input_\n"
    "        ):\n"
    "            out = torch.ops.vllm.all_reduce_symmetric_with_copy(input_)\n"
    "            if out is not None:\n"
    "                return out\n"
    "        # V4PLUS (fork #597 adapted): RoCEnante one-shot RDMA all-reduce\n"
    "        # (multi-node TP over ConnectX-7). Tried before the IPC/PCIe\n"
    "        # custom backends, which cannot span nodes.\n"
    "        b12x_ar_comm = self.b12x_ar_comm\n"
    "        if (\n"
    "            b12x_ar_comm is not None\n"
    "            and not b12x_ar_comm.disabled\n"
    "            and b12x_ar_comm.should_custom_ar(input_)\n"
    "        ):\n"
    "            out = b12x_ar_comm.custom_all_reduce(input_)\n"
    "            assert out is not None\n"
    "            return out\n"
)
CC_AR_DISPATCH_PRESENT = "            out = b12x_ar_comm.custom_all_reduce(input_)\n"

CC_LOG_POTENTIAL_ANCHOR = (
    "        all_potential_ar_backends = [\n"
    '            "NCCL_SYMM_MEM",\n'
)
CC_LOG_POTENTIAL_REPLACEMENT = (
    "        all_potential_ar_backends = [\n"
    '            "B12X_ROCENANTE",  # V4PLUS (fork #597)\n'
    '            "NCCL_SYMM_MEM",\n'
)
CC_LOG_POTENTIAL_PRESENT = '            "B12X_ROCENANTE",  # V4PLUS (fork #597)\n'

CC_LOG_ENABLED_ANCHOR = "        enabled_ar_backends: list[str] = []\n"
CC_LOG_ENABLED_REPLACEMENT = (
    "        enabled_ar_backends: list[str] = []\n"
    "        if self.b12x_ar_comm is not None and not self.b12x_ar_comm.disabled:\n"
    "            enabled_ar_backends.append(\n"
    '                getattr(self.b12x_ar_comm, "backend_name", "B12X_ROCENANTE")\n'
    "            )\n"
)
CC_LOG_ENABLED_PRESENT = '                getattr(self.b12x_ar_comm, "backend_name", "B12X_ROCENANTE")\n'

CC_AG_ANCHOR = (
    "        if dim < 0:\n"
    "            dim += input_.dim()\n"
    "        if dim == 0 and should_nccl_symm_mem_ag_rs():\n"
)
CC_AG_REPLACEMENT = (
    "        if dim < 0:\n"
    "            dim += input_.dim()\n"
    "        # V4PLUS (fork #597 adapted): RoCEnante all-gather (writes the\n"
    "        # concatenated layout directly, so no reshape/copy follows).\n"
    "        b12x_ar_comm = self.b12x_ar_comm\n"
    "        if (\n"
    "            b12x_ar_comm is not None\n"
    "            and not b12x_ar_comm.disabled\n"
    "            and getattr(b12x_ar_comm, \"should_all_gather\", None) is not None\n"
    "            and b12x_ar_comm.should_all_gather(input_, dim)\n"
    "        ):\n"
    "            return b12x_ar_comm.all_gather(input_, dim)\n"
    "        if dim == 0 and should_nccl_symm_mem_ag_rs():\n"
)
CC_AG_PRESENT = "            return b12x_ar_comm.all_gather(input_, dim)\n"

# ----------------------------------------------------------------------
# gpu_worker.py (#597-verbatim)
# ----------------------------------------------------------------------
GW_CLASS_ANCHOR = (
    "class Worker(WorkerBase):\n"
    "    def __init__(\n"
)
GW_CLASS_REPLACEMENT = (
    "# V4PLUS (fork #597): verbatim from fork PR #597.\n"
    "class _B12xRoceCheckedAsyncOutput(AsyncModelRunnerOutput):\n"
    "    \"\"\"An asynchronous output whose completion is followed by the RoCEnante check.\"\"\"\n"
    "\n"
    "    def __init__(\n"
    "        self, inner: AsyncModelRunnerOutput, check: Callable[[], None]\n"
    "    ) -> None:\n"
    "        self._inner = inner\n"
    "        self._check = check\n"
    "\n"
    "    def get_output(self) -> ModelRunnerOutput:\n"
    "        \"\"\"Wait for the wrapped output, then run the fail-stop check.\n"
    "\n"
    "        Returns:\n"
    "            The completed ModelRunnerOutput.\n"
    "\n"
    "        Raises:\n"
    "            RuntimeError: When a RoCEnante wait timed out or its proxy died.\n"
    "        \"\"\"\n"
    "        output = self._inner.get_output()\n"
    "        self._check()\n"
    "        return output\n"
    "\n"
    "\n"
    "class Worker(WorkerBase):\n"
    "    def __init__(\n"
)
GW_CLASS_PRESENT = "class _B12xRoceCheckedAsyncOutput(AsyncModelRunnerOutput):\n"

GW_SAMPLE_ANCHOR = (
    "    def sample_tokens(\n"
    "        self, grammar_output: \"GrammarOutput | None\"\n"
    "    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:\n"
    "        return self.model_runner.sample_tokens(grammar_output)\n"
)
GW_SAMPLE_REPLACEMENT = (
    "    def sample_tokens(\n"
    "        self, grammar_output: \"GrammarOutput | None\"\n"
    "    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:\n"
    "        return self._b12x_roce_guarded(  # V4PLUS (fork #597)\n"
    "            self.model_runner.sample_tokens(grammar_output)\n"
    "        )\n"
    "\n"
    "    def _b12x_roce_health_check(self) -> Callable[[], None] | None:\n"
    "        \"\"\"The RoCEnante health check of the TP communicator, if one is active.\n"
    "\n"
    "        Returns:\n"
    "            The check callable, or None when RoCEnante is not in use.\n"
    "        \"\"\"\n"
    "        communicator = get_tp_group().device_communicator\n"
    '        comm = getattr(communicator, "b12x_ar_comm", None)\n'
    '        return getattr(comm, "check_health", None)\n'
    "\n"
    "    def _b12x_roce_guarded(self, output):\n"
    "        \"\"\"Fail-stop RoCEnante check once the step's output is on the host.\n"
    "\n"
    "        A RoCEnante wait that timed out records itself and freezes the runtime.\n"
    "        A synchronous output already holds the sampled tokens on the host, so\n"
    "        every collective of the step has completed and the check runs now; an\n"
    "        asynchronous output is wrapped so the check runs right after its\n"
    "        ``get_output()`` completes the copy.  Either way a failed collective's\n"
    "        output never leaves the worker.  Every rank reaches the same state on\n"
    "        its own (a stalled rank starves its peers' waits), so the raise is\n"
    "        coordinated without a supervisor.  Two pinned-memory reads; no added\n"
    "        synchronization.\n"
    "\n"
    "        Args:\n"
    "            output: The model runner's output for this step, possibly None.\n"
    "\n"
    "        Returns:\n"
    "            The same output, or a wrapper for an asynchronous output.\n"
    "\n"
    "        Raises:\n"
    "            RuntimeError: When a RoCEnante wait timed out or its proxy died.\n"
    "        \"\"\"\n"
    "        check = self._b12x_roce_health_check()\n"
    "        if check is None:\n"
    "            return output\n"
    "        if isinstance(output, AsyncModelRunnerOutput):\n"
    "            return _B12xRoceCheckedAsyncOutput(output, check)\n"
    "        check()\n"
    "        return output\n"
)
GW_SAMPLE_PRESENT = "    def _b12x_roce_guarded(self, output):\n"

GW_EXEC_ANCHOR = (
    "            if isinstance(\n"
    "                output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType\n"
    "            ):\n"
    "                return output\n"
)
GW_EXEC_REPLACEMENT = (
    "            if isinstance(\n"
    "                output, ModelRunnerOutput | AsyncModelRunnerOutput | NoneType\n"
    "            ):\n"
    "                return self._b12x_roce_guarded(output)  # V4PLUS (fork #597)\n"
)
GW_EXEC_PRESENT = "                return self._b12x_roce_guarded(output)  # V4PLUS (fork #597)\n"


def apply_hunks(path, hunks, label):
    try:
        with open(path) as f:
            src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {path} not found", file=sys.stderr)
        return None
    done = set()
    changed = False
    for name, anchor, replacement, present in hunks:
        if present in src:
            print(f"  SKIP (already patched) [{label}]: {name}")
            done.add(name)
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
        done.add(name)
        changed = True
    if changed:
        with open(path, "w") as f:
            f.write(src)
        py_compile.compile(path, doraise=True)
        print(f"[{SCRIPT_NAME}] {label} py_compile OK")
    return done


def md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def install_shim() -> bool:
    os.makedirs(os.path.dirname(SHIM_DST), exist_ok=True)
    if os.path.isfile(SHIM_DST):
        if md5(SHIM_SRC) == md5(SHIM_DST):
            print("  SKIP (identical): b12x_roce_all_reduce.py")
            return True
        print(
            "  *** LOUD NOTE: b12x_roce_all_reduce.py exists with DIFFERENT "
            "content; left unchanged — review before baking ***"
        )
        return True  # present; construction can import it
    shutil.copyfile(SHIM_SRC, SHIM_DST)
    py_compile.compile(SHIM_DST, doraise=True)
    print("  APPLIED: installed b12x_roce_all_reduce.py (fork #597, adapted import)")
    return True


def main() -> int:
    print(f"[{SCRIPT_NAME}] fork #597 RoCEnante adapter (VLLM_ROOT={VLLM_ROOT})")
    kernel = probe_roce_kernel()
    default = "1" if kernel is True else "0"
    if kernel is True:
        print(f"[{SCRIPT_NAME}] b12x.comm.roce probe: HEALTHY — VLLM_ENABLE_ROCE_ALLREDUCE default ON")
    elif kernel is False:
        print(
            f"[{SCRIPT_NAME}] *** LOUD NOTE: b12x.comm.roce is absent or broken "
            "(probe failed). VLLM_ENABLE_ROCE_ALLREDUCE defaults to OFF; even "
            "when enabled, the shim's capability vote disables the backend on "
            "every rank. Apply M1 (patch_b12x_roce_kernel.py) first. ***"
        )
    else:
        print(
            f"[{SCRIPT_NAME}] NOTE: b12x.comm.roce probe could not run; "
            "VLLM_ENABLE_ROCE_ALLREDUCE defaults to OFF."
        )

    # 1. envs.py first: the tp-branch flag computation reads the env.
    envs_ok = False
    done = apply_hunks(
        ENVS_PATH,
        [
            ("ROCE env fields", ENVS_FIELD_ANCHOR, ENVS_FIELD_REPLACEMENT, ENVS_FIELD_PRESENT),
            (
                "ROCE env lambdas",
                ENVS_LAMBDA_ANCHOR,
                ENVS_LAMBDA_TEMPLATE.format(default=default),
                ENVS_LAMBDA_PRESENT,
            ),
        ],
        "envs.py",
    )
    if done is not None:
        with open(ENVS_PATH) as f:
            envs_ok = ENVS_LAMBDA_PRESENT in f.read()

    # 2. shim payload.
    shim_ok = install_shim()

    # 3. cuda_communicator.py — staged.
    cc_hunks = []
    if envs_ok:
        cc_hunks = [
            ("non-tp use_roce_allreduce = False", CC_NONTP_ANCHOR, CC_NONTP_REPLACEMENT, CC_NONTP_PRESENT),
            ("tp-branch flag + flashinfer off", CC_TP_ANCHOR, CC_TP_REPLACEMENT, CC_TP_PRESENT),
        ]
    else:
        print(
            "  NOTE (not applicable) [cuda_communicator.py]: flag hunks skipped "
            "— envs.py lambda hunk did not apply (would AttributeError on "
            "envs.VLLM_ENABLE_ROCE_ALLREDUCE)"
        )
    done = apply_hunks(CC_PATH, cc_hunks, "cuda_communicator.py")
    if done is None:
        return 1
    with open(CC_PATH) as f:
        cc_src = f.read()
    flags_ok = CC_NONTP_PRESENT in cc_src and CC_TP_PRESENT in cc_src
    slot_hunks = [("b12x_ar_comm slot", CC_SLOT_ANCHOR, CC_SLOT_REPLACEMENT, CC_SLOT_PRESENT)]
    if flags_ok:
        slot_hunks += [
            ("use_roce_allreduce attr", CC_ATTR_ANCHOR, CC_ATTR_REPLACEMENT, CC_ATTR_PRESENT),
            ("ca_comm gated off under roce", CC_CA_GATE_ANCHOR, CC_CA_GATE_REPLACEMENT, CC_CA_GATE_PRESENT),
        ]
    else:
        print(
            "  NOTE (not applicable) [cuda_communicator.py]: attr + ca_comm "
            "gate hunks skipped — flag hunks incomplete"
        )
    apply_hunks(CC_PATH, slot_hunks, "cuda_communicator.py")
    with open(CC_PATH) as f:
        cc_src = f.read()
    ctor_hunks = []
    if (
        CC_ATTR_PRESENT in cc_src
        and CC_CA_GATE_PRESENT in cc_src
        and CC_SLOT_PRESENT in cc_src
        and shim_ok
    ):
        ctor_hunks = [
            ("B12xRoceAllReduce construction", CC_CONSTRUCT_ANCHOR, CC_CONSTRUCT_REPLACEMENT, CC_CONSTRUCT_PRESENT),
        ]
    else:
        print(
            "  NOTE (not applicable) [cuda_communicator.py]: construction "
            "hunk skipped — attr/slot/gate hunks or shim payload incomplete"
        )
    dispatch_hunks = []
    if CC_SLOT_PRESENT in cc_src:
        dispatch_hunks = [
            ("all_reduce dispatch", CC_AR_DISPATCH_ANCHOR, CC_AR_DISPATCH_REPLACEMENT, CC_AR_DISPATCH_PRESENT),
            ("log potential backends", CC_LOG_POTENTIAL_ANCHOR, CC_LOG_POTENTIAL_REPLACEMENT, CC_LOG_POTENTIAL_PRESENT),
            ("log enabled backends", CC_LOG_ENABLED_ANCHOR, CC_LOG_ENABLED_REPLACEMENT, CC_LOG_ENABLED_PRESENT),
            ("all_gather dispatch", CC_AG_ANCHOR, CC_AG_REPLACEMENT, CC_AG_PRESENT),
        ]
    else:
        print(
            "  NOTE (not applicable) [cuda_communicator.py]: dispatch/log "
            "hunks skipped — b12x_ar_comm slot attribute missing"
        )
    apply_hunks(CC_PATH, ctor_hunks + dispatch_hunks, "cuda_communicator.py")

    # 4. gpu_worker.py (#597-verbatim).
    if apply_hunks(
        GW_PATH,
        [
            ("_B12xRoceCheckedAsyncOutput", GW_CLASS_ANCHOR, GW_CLASS_REPLACEMENT, GW_CLASS_PRESENT),
            ("sample_tokens guard + helpers", GW_SAMPLE_ANCHOR, GW_SAMPLE_REPLACEMENT, GW_SAMPLE_PRESENT),
            ("execute_model guard", GW_EXEC_ANCHOR, GW_EXEC_REPLACEMENT, GW_EXEC_PRESENT),
        ],
        "gpu_worker.py",
    ) is None:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
