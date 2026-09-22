#!/usr/bin/env bash
# fix-gb10-nvml-fallback — fall back to torch when the device-level NVML
# memory query is unsupported (upstream vLLM PR #57378).
#
# On integrated parts (GB10 / DGX Spark) NVML does not implement
# nvmlDeviceGetMemoryInfo (nvidia-smi shows [N/A] for memory.total), so
# NvmlCudaPlatform.get_device_total_memory raised NVMLError. This wraps the
# query in try/except and falls back to the same source NonNvmlCudaPlatform
# reads: torch.cuda.get_device_properties(device_id).total_memory.
#
# Target: vllm/platforms/cuda.py (NvmlCudaPlatform only; the anchor is the
# NVML variant of get_device_total_memory). Kept disjoint from
# fix-gb10-kv-sizing's edit (anchored on is_fully_connected) so mod order
# does not matter.
#
# All-or-nothing per file: every anchor must match exactly once or the file
# is left untouched (exit 3 = no anchors, skip copy; exit 4 = partial/
# duplicate match or invalid syntax, die; exit 5 = py_compile failure).
# Marker: fix-gb10-nvml-fallback

set -euo pipefail

MOD="fix-gb10-nvml-fallback"
TARGET_REL="vllm/platforms/cuda.py"
# MOD_FIND_ROOTS override exists for offline validation against /tmp copies;
# production default is the in-image vllm tree (+ conventional fallbacks).
FIND_ROOTS="${MOD_FIND_ROOTS:-/opt/kimi-k3 /opt/venv /usr/local/lib}"

PATCHED=0
for FILE in $(find $FIND_ROOTS -path "*$TARGET_REL" 2>/dev/null | grep -v __pycache__ | sort -u || true); do
  if grep -q "$MOD" "$FILE" 2>/dev/null; then
    echo "[$MOD] already applied in $FILE"
    PATCHED=1
    continue
  fi
  if python3 - "$FILE" <<'PYEOF'
import py_compile
import sys

p = sys.argv[1]
s = open(p).read()

sites = [
    # NvmlCudaPlatform.get_device_total_memory -> torch fallback on NVMLError
    (
        "nvml-total-memory-fallback",
        """    @classmethod
    @with_nvml_context
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)
""",
        """    @classmethod
    @with_nvml_context
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        physical_device_id = cls.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device_id)
        try:
            return int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)
        except pynvml.NVMLError:
            # fix-gb10-nvml-fallback (upstream #57378): integrated parts (e.g.
            # GB10 / DGX Spark) do not implement the device-level NVML memory
            # query - nvidia-smi reports [N/A] for memory.total there. Fall
            # back to the same source NonNvmlCudaPlatform reads.
            return int(torch.cuda.get_device_properties(device_id).total_memory)
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
    PATCHED=1
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
[ "$PATCHED" = "1" ] || { echo "[$MOD] ERROR: primary file not patched (no $TARGET_REL with anchors found under: $FIND_ROOTS)"; exit 1; }
