#!/usr/bin/env bash
# fix-gb10-kv-sizing — size the KV cache from THIS process's device memory,
# not the device-wide free-memory delta (upstream vLLM PR #55828).
#
# On GB10 / DGX Spark (UMA, several instances can share the device) the
# device-wide free-memory delta charges other processes' allocations to this
# instance's non-KV-cache budget (or crashes the init assert when they free
# memory mid-profile). Upstream #55828 measures per-process device memory via
# NVML's compute-process list (the one NVML query that DOES work on GB10) and
# uses that delta for total_consumed, falling back to the device-wide delta
# when NVML cannot attribute memory.
#
# Files (adapted to our diverged tree via in-image extract anchors):
#  - vllm/platforms/cuda.py       : NvmlCudaPlatform.get_process_memory_usage()
#  - vllm/utils/mem_utils.py      : MemorySnapshot.process_memory, profiling
#                                   process_scoped + per-process total_consumed
#  - vllm/v1/worker/gpu_worker.py : don't assert/crash when OTHER processes
#                                   freed memory while we were profiling and
#                                   the budget is process-scoped
# Skipped upstream hunks:
#  - vllm/platforms/interface.py default get_process_memory_usage -> no
#    in-image extract; mem_utils calls the hook via getattr(..., None), which
#    is behaviorally identical on CUDA and crash-proof elsewhere.
#  - tests/* (not shipped in the image).
#
# All three files must end up patched-or-already-marked; otherwise exit 1.
# All-or-nothing per file: every anchor must match exactly once or the file
# is left untouched (exit 3 = no anchors, skip copy; exit 4 = partial/
# duplicate match or invalid syntax, die; exit 5 = py_compile failure).
# Marker: fix-gb10-kv-sizing

set -euo pipefail

MOD="fix-gb10-kv-sizing"
# MOD_FIND_ROOTS override exists for offline validation against /tmp copies;
# production default is the in-image vllm tree (+ conventional fallbacks).
FIND_ROOTS="${MOD_FIND_ROOTS:-/opt/kimi-k3 /opt/venv /usr/local/lib}"

ok_cuda=0
ok_mem=0
ok_worker=0

# ---------------- vllm/platforms/cuda.py ----------------
for FILE in $(find $FIND_ROOTS -path "*vllm/platforms/cuda.py" 2>/dev/null | grep -v __pycache__ | sort -u || true); do
  if grep -q "$MOD" "$FILE" 2>/dev/null; then
    echo "[$MOD] already applied in $FILE"
    ok_cuda=1
    continue
  fi
  if python3 - "$FILE" <<'PYEOF'
import py_compile
import sys

p = sys.argv[1]
s = open(p).read()

sites = [
    # Insert get_process_memory_usage into NvmlCudaPlatform, anchored on the
    # NVML is_fully_connected block (kept disjoint from fix-gb10-nvml-fallback's
    # edit of get_device_total_memory so mod order does not matter).
    (
        "nvml-get-process-memory-usage",
        '''    @classmethod
    @with_nvml_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """
        query if the set of gpus are fully connected by nvlink (1 hop)
        """
''',
        '''    @classmethod
    @with_nvml_context
    def get_process_memory_usage(cls, device_id: int = 0) -> int | None:
        """fix-gb10-kv-sizing (upstream #55828): device memory used by this
        process on the visible device ``device_id`` as reported by NVML.

        Returns ``None`` whenever NVML cannot attribute memory to this
        process (no entry for our PID, e.g. a container whose PID namespace
        NVML does not see, WDDM, MIG, or an NVML error) so that callers fall
        back to device-level accounting.

        Keep this on the per-process query: ``nvmlDeviceGetMemoryInfo`` is
        ``NVMLError_NotSupported`` on integrated parts such as GB10 (DGX
        Spark), where this per-process path is exactly what still works.
        """
        try:
            physical_device_id = cls.visible_device_id_to_physical_device_id(device_id)
            handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except (pynvml.NVMLError, IndexError, ValueError):
            return None
        pid = os.getpid()
        for proc in processes:
            if proc.pid == pid:
                used = proc.usedGpuMemory
                return int(used) if isinstance(used, int) else None
        return None

    @classmethod
    @with_nvml_context
    def is_fully_connected(cls, physical_device_ids: list[int]) -> bool:
        """
        query if the set of gpus are fully connected by nvlink (1 hop)
        """
''',
    ),
]

found = [n for n, o, _ in sites if s.count(o) == 1]
missing = [n for n, o, _ in sites if s.count(o) == 0]
dup = [n for n, o, _ in sites if s.count(o) > 1]

if not found and not dup:
    print("no anchors present", file=sys.stderr)
    sys.exit(3)
if dup or missing:
    print(
        f"partial/duplicate anchors: found={found} missing={missing} duplicate={dup}",
        file=sys.stderr,
    )
    sys.exit(4)

out = s
for _name, old, new in sites:
    out = out.replace(old, new, 1)

try:
    compile(out, p, "exec")
except SyntaxError as e:
    print(f"patched source does not parse: {e}", file=sys.stderr)
    sys.exit(4)

open(p, "w").write(out)
try:
    py_compile.compile(p, doraise=True)
except py_compile.PyCompileError as e:
    print(f"py_compile failed: {e}", file=sys.stderr)
    sys.exit(5)
print(f"applied {len(sites)} sites: {[n for n, _, _ in sites]}")
PYEOF
  then
    echo "[$MOD] APPLIED + py_compile OK: $FILE"
    ok_cuda=1
  else
    rc=$?
    if [ "$rc" = "3" ]; then
      echo "[$MOD] WARNING: anchors not found in $FILE (different vllm version?), skipping"
    else
      echo "[$MOD] ERROR: refusing to patch $FILE (exit $rc: partial/duplicate anchors or compile failure)"
      exit 1
    fi
  fi
done

# ---------------- vllm/utils/mem_utils.py ----------------
for FILE in $(find $FIND_ROOTS -path "*vllm/utils/mem_utils.py" 2>/dev/null | grep -v __pycache__ | sort -u || true); do
  if grep -q "$MOD" "$FILE" 2>/dev/null; then
    echo "[$MOD] already applied in $FILE"
    ok_mem=1
    continue
  fi
  if python3 - "$FILE" <<'PYEOF'
import py_compile
import sys

p = sys.argv[1]
s = open(p).read()

sites = [
    # (m1) warning threshold constant
    (
        "const",
        """logger = init_logger(__name__)


def format_kib(b: int) -> str:
""",
        """logger = init_logger(__name__)

# fix-gb10-kv-sizing (upstream #55828): device-wide vs. per-process
# consumption differences below this are not reported (CUDA runtime
# bookkeeping, other processes' minor fluctuations).
_OTHER_PROCESS_DELTA_WARN_BYTES = 128 * MiB_bytes


def format_kib(b: int) -> str:
""",
    ),
    # (m2) MemorySnapshot.process_memory field
    (
        "snapshot-field",
        """    non_torch_memory: int = 0
    timestamp: float = 0.0
""",
        """    non_torch_memory: int = 0
    # fix-gb10-kv-sizing: memory used by this process alone (per-process
    # accounting, e.g. NVML); None when the platform cannot attribute device
    # memory to processes.
    process_memory: int | None = None
    timestamp: float = 0.0
""",
    ),
    # (m3) measure(): record per-process usage. getattr instead of the
    # upstream interface.py default (that file has no in-image extract).
    (
        "measure",
        """        self.non_torch_memory = self.cuda_memory - self.torch_memory
        self.timestamp = time.time()
""",
        """        self.non_torch_memory = self.cuda_memory - self.torch_memory
        # fix-gb10-kv-sizing (upstream #55828): record this process's own
        # device memory so profiling can charge only our own growth. The
        # getattr keeps platforms without the hook working (the interface.py
        # default from upstream is not patched here).
        _get_process_memory_usage = getattr(
            current_platform, "get_process_memory_usage", None
        )
        process_memory = (
            _get_process_memory_usage(device.index or 0)
            if _get_process_memory_usage is not None
            else None
        )
        self.process_memory = (
            process_memory if isinstance(process_memory, int) else None
        )
        self.timestamp = time.time()
""",
    ),
    # (m4) __sub__ carries process_memory deltas
    (
        "sub",
        """            non_torch_memory=self.non_torch_memory - other.non_torch_memory,
            timestamp=self.timestamp - other.timestamp,
""",
        """            non_torch_memory=self.non_torch_memory - other.non_torch_memory,
            process_memory=(
                self.process_memory - other.process_memory
                if self.process_memory is not None and other.process_memory is not None
                else None
            ),
            timestamp=self.timestamp - other.timestamp,
""",
    ),
    # (m5a) _format_process_memory helper before MemorySnapshot.__repr__
    (
        "repr-method",
        """    def __repr__(self) -> str:
        return (
            f"torch_peak={format_gib(self.torch_peak)}GiB, "
""",
        """    def _format_process_memory(self) -> str:
        if self.process_memory is None:
            return "n/a"
        return f"{format_gib(self.process_memory)}GiB"

    def __repr__(self) -> str:
        return (
            f"torch_peak={format_gib(self.torch_peak)}GiB, "
""",
    ),
    # (m5b) show process_memory in the snapshot repr
    (
        "repr-line",
        """            f"non_torch_memory={format_gib(self.non_torch_memory)}GiB, "
            f"timestamp={self.timestamp}, "
""",
        """            f"non_torch_memory={format_gib(self.non_torch_memory)}GiB, "
            f"process_memory={self._format_process_memory()}, "
            f"timestamp={self.timestamp}, "
""",
    ),
    # (m6) MemoryProfilingResult.process_scoped field
    (
        "profiling-field",
        """    total_consumed: int = 0
    transient_peak_headroom: int = 0
""",
        """    total_consumed: int = 0
    # fix-gb10-kv-sizing: True when total_consumed comes from this process's
    # own device memory usage rather than from the device-wide free-memory
    # delta.
    process_scoped: bool = False
    transient_peak_headroom: int = 0
""",
    ),
    # (m7) process-scoped total_consumed with device-wide fallback
    (
        "total-consumed",
        """    # Measure total consumption via mem_get_info() instead of
    # memory_reserved(), which goes negative when pluggable allocators
    # (e.g. cumem) bypass PyTorch's tracking.
    result.total_consumed = (
        result.before_create.free_memory - result.after_profile.free_memory
    )
""",
        """    # Measure total consumption via mem_get_info() instead of
    # memory_reserved(), which goes negative when pluggable allocators
    # (e.g. cumem) bypass PyTorch's tracking.
    device_consumed = (
        result.before_create.free_memory - result.after_profile.free_memory
    )
    before_process = result.before_create.process_memory
    after_process = result.after_profile.process_memory
    if before_process is not None and after_process is not None:
        # fix-gb10-kv-sizing (upstream #55828): per-process accounting -
        # memory that other processes on the same device allocate or release
        # while this instance loads and profiles must not be charged to (or
        # credited against) this instance.
        result.total_consumed = after_process - before_process
        result.process_scoped = True
        other_processes_delta = device_consumed - result.total_consumed
        if abs(other_processes_delta) >= _OTHER_PROCESS_DELTA_WARN_BYTES:
            logger.warning(
                "Other processes on %s changed their device memory usage by "
                "%s GiB while this instance was loading and profiling. The "
                "KV cache budget is based on this process's own usage "
                "(%s GiB) rather than the device-wide change (%s GiB); make "
                "sure the instances sharing this device do not request more "
                "memory than it has in total.",
                result.before_create.device_,
                format_gib(other_processes_delta),
                format_gib(result.total_consumed),
                format_gib(device_consumed),
            )
    else:
        result.total_consumed = device_consumed
""",
    ),
]

found = [n for n, o, _ in sites if s.count(o) == 1]
missing = [n for n, o, _ in sites if s.count(o) == 0]
dup = [n for n, o, _ in sites if s.count(o) > 1]

if not found and not dup:
    print("no anchors present", file=sys.stderr)
    sys.exit(3)
if dup or missing:
    print(
        f"partial/duplicate anchors: found={found} missing={missing} duplicate={dup}",
        file=sys.stderr,
    )
    sys.exit(4)

out = s
for _name, old, new in sites:
    out = out.replace(old, new, 1)

try:
    compile(out, p, "exec")
except SyntaxError as e:
    print(f"patched source does not parse: {e}", file=sys.stderr)
    sys.exit(4)

open(p, "w").write(out)
try:
    py_compile.compile(p, doraise=True)
except py_compile.PyCompileError as e:
    print(f"py_compile failed: {e}", file=sys.stderr)
    sys.exit(5)
print(f"applied {len(sites)} sites: {[n for n, _, _ in sites]}")
PYEOF
  then
    echo "[$MOD] APPLIED + py_compile OK: $FILE"
    ok_mem=1
  else
    rc=$?
    if [ "$rc" = "3" ]; then
      echo "[$MOD] WARNING: anchors not found in $FILE (different vllm version?), skipping"
    else
      echo "[$MOD] ERROR: refusing to patch $FILE (exit $rc: partial/duplicate anchors or compile failure)"
      exit 1
    fi
  fi
done

# ---------------- vllm/v1/worker/gpu_worker.py ----------------
for FILE in $(find $FIND_ROOTS -path "*vllm/v1/worker/gpu_worker.py" 2>/dev/null | grep -v __pycache__ | sort -u || true); do
  if grep -q "$MOD" "$FILE" 2>/dev/null; then
    echo "[$MOD] already applied in $FILE"
    ok_worker=1
    continue
  fi
  if python3 - "$FILE" <<'PYEOF'
import py_compile
import sys

p = sys.argv[1]
s = open(p).read()

sites = [
    # determine_available_memory: our fork has no rocm_fallback here, so the
    # upstream guard collapses to: process-scoped budget -> warn instead of
    # assert when other processes freed memory during profiling.
    (
        "assert-block",
        """        free_gpu_memory = profile_result.after_profile.free_memory
        # NOTE(woosuk): Here we assume that the other processes using the same
        # GPU did not change their memory usage during the profiling.
        assert self.init_snapshot.free_memory >= free_gpu_memory, (
            "Error in memory profiling. "
            f"Initial free memory {format_gib(self.init_snapshot.free_memory)} GiB, "
            f"current free memory {format_gib(free_gpu_memory)} GiB. "
            "This happens when other processes sharing the same container "
            "release GPU memory while vLLM is profiling during initialization. "
            "To fix this, ensure consistent GPU memory allocation or "
            "isolate vLLM in its own container."
        )
""",
        """        free_gpu_memory = profile_result.after_profile.free_memory
        init_free_memory = self.init_snapshot.free_memory
        if profile_result.process_scoped:
            # fix-gb10-kv-sizing (upstream #55828): the budget is based on
            # this process's own memory usage, so other processes releasing
            # memory during profiling is harmless.
            if init_free_memory < free_gpu_memory:
                logger.warning(
                    "Other processes released %s GiB on the device while this "
                    "instance was profiling; ignored because the KV cache "
                    "budget uses this process's own memory usage.",
                    format_gib(free_gpu_memory - init_free_memory),
                )
        else:
            # NOTE(woosuk): Here we assume that the other processes using the same
            # GPU did not change their memory usage during the profiling.
            assert init_free_memory >= free_gpu_memory, (
                "Error in memory profiling. "
                f"Initial free memory {format_gib(self.init_snapshot.free_memory)} GiB, "
                f"current free memory {format_gib(free_gpu_memory)} GiB. "
                "This happens when other processes sharing the same container "
                "release GPU memory while vLLM is profiling during initialization. "
                "To fix this, ensure consistent GPU memory allocation or "
                "isolate vLLM in its own container."
            )
""",
    ),
]

found = [n for n, o, _ in sites if s.count(o) == 1]
missing = [n for n, o, _ in sites if s.count(o) == 0]
dup = [n for n, o, _ in sites if s.count(o) > 1]

if not found and not dup:
    print("no anchors present", file=sys.stderr)
    sys.exit(3)
if dup or missing:
    print(
        f"partial/duplicate anchors: found={found} missing={missing} duplicate={dup}",
        file=sys.stderr,
    )
    sys.exit(4)

out = s
for _name, old, new in sites:
    out = out.replace(old, new, 1)

try:
    compile(out, p, "exec")
except SyntaxError as e:
    print(f"patched source does not parse: {e}", file=sys.stderr)
    sys.exit(4)

open(p, "w").write(out)
try:
    py_compile.compile(p, doraise=True)
except py_compile.PyCompileError as e:
    print(f"py_compile failed: {e}", file=sys.stderr)
    sys.exit(5)
print(f"applied {len(sites)} sites: {[n for n, _, _ in sites]}")
PYEOF
  then
    echo "[$MOD] APPLIED + py_compile OK: $FILE"
    ok_worker=1
  else
    rc=$?
    if [ "$rc" = "3" ]; then
      echo "[$MOD] WARNING: anchors not found in $FILE (different vllm version?), skipping"
    else
      echo "[$MOD] ERROR: refusing to patch $FILE (exit $rc: partial/duplicate anchors or compile failure)"
      exit 1
    fi
  fi
done

fail=0
[ "$ok_cuda" = "1" ] || { echo "[$MOD] ERROR: vllm/platforms/cuda.py not patched"; fail=1; }
[ "$ok_mem" = "1" ] || { echo "[$MOD] ERROR: vllm/utils/mem_utils.py not patched"; fail=1; }
[ "$ok_worker" = "1" ] || { echo "[$MOD] ERROR: vllm/v1/worker/gpu_worker.py not patched"; fail=1; }
[ "$fail" = "0" ] || exit 1
echo "[$MOD] all three targets patched-or-applied"
