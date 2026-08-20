#!/usr/bin/env bash
# perf-shm-busy-loop-2ms — PR #421
# Reduce shm_broadcast busy_loop_s from 1.0s to 2ms (configurable via env).
# On GB10 (shared CPU/GPU thermal budget), the 1s default pins a CPU core
# at max clock for the entire decode phase, throttling the GPU.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /opt/kimi-k3/vllm

python3 "$SCRIPT_DIR/apply_shm_busy_loop.py" && echo "[perf-shm-busy-loop-2ms] applied (default busy_loop_s=2ms)"
