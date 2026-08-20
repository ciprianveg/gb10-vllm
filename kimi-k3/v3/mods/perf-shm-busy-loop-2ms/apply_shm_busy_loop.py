#!/usr/bin/env python3
"""Apply PR #421: shm busy_loop_s 1s -> 2ms via env var. Python patcher."""
import re, sys

shm_path = "vllm/distributed/device_communicators/shm_broadcast.py"
envs_path = "vllm/envs.py"

# --- shm_broadcast.py ---
with open(shm_path) as f:
    shm = f.read()

if "VLLM_SHM_BROADCAST_BUSY_LOOP_S" in shm:
    print("[perf-shm-busy-loop-2ms] already present in shm_broadcast.py")
else:
    # 1. Add import math after SPDX header
    shm = shm.replace(
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\nimport copyreg",
        "# SPDX-FileCopyrightText: Copyright contributors to the vLLM project\nimport math\nimport copyreg",
        1)

    # 2. Change default busy_loop_s from 1 to None
    shm = shm.replace("busy_loop_s: float = 1,", "busy_loop_s: float | None = None,", 1)

    # 3. Replace self.busy_loop_s = busy_loop_s with env var logic + validation
    old = "            self.busy_loop_s = busy_loop_s\n"
    new = (
        "            self.busy_loop_s = (\n"
        "                busy_loop_s\n"
        "                if busy_loop_s is not None\n"
        "                else envs.VLLM_SHM_BROADCAST_BUSY_LOOP_S\n"
        "            )\n"
        "            if not math.isfinite(self.busy_loop_s) or self.busy_loop_s < 0:\n"
        "                raise ValueError(\n"
        '                    "busy_loop_s must be finite and non-negative, got "\n'
        '                    f"{self.busy_loop_s!r} (VLLM_SHM_BROADCAST_BUSY_LOOP_S)"\n'
        "                )\n"
    )
    shm = shm.replace(old, new, 1)

    with open(shm_path, "w") as f:
        f.write(shm)
    print("[perf-shm-busy-loop-2ms] patched shm_broadcast.py")

# --- envs.py ---
with open(envs_path) as f:
    envs = f.read()

if "VLLM_SHM_BROADCAST_BUSY_LOOP_S" in envs:
    print("[perf-shm-busy-loop-2ms] already present in envs.py")
else:
    # 1. Add dataclass field after RINGBUFFER_WARNING_INTERVAL
    envs = envs.replace(
        "    VLLM_RINGBUFFER_WARNING_INTERVAL: int = 60\n",
        "    VLLM_RINGBUFFER_WARNING_INTERVAL: int = 60\n"
        "    VLLM_SHM_BROADCAST_BUSY_LOOP_S: float = 0.002\n",
        1)

    # 2. Add resolver lambda after RINGBUFFER_WARNING_INTERVAL resolver
    envs = envs.replace(
        '    "VLLM_RINGBUFFER_WARNING_INTERVAL": lambda: int(\n'
        '        os.environ.get("VLLM_RINGBUFFER_WARNING_INTERVAL", "60")\n'
        "    ),\n",
        '    "VLLM_RINGBUFFER_WARNING_INTERVAL": lambda: int(\n'
        '        os.environ.get("VLLM_RINGBUFFER_WARNING_INTERVAL", "60")\n'
        "    ),\n"
        '    "VLLM_SHM_BROADCAST_BUSY_LOOP_S": lambda: float(\n'
        '        os.environ.get("VLLM_SHM_BROADCAST_BUSY_LOOP_S", "0.002")\n'
        "    ),\n",
        1)

    with open(envs_path, "w") as f:
        f.write(envs)
    print("[perf-shm-busy-loop-2ms] patched envs.py")

# Verify
import py_compile
py_compile.compile(shm_path, doraise=True)
py_compile.compile(envs_path, doraise=True)
print("[perf-shm-busy-loop-2ms] syntax check OK")
