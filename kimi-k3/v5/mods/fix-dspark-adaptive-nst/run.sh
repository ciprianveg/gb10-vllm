#!/bin/bash
# fix-dspark-adaptive-nst — drive SchedulerOutput.num_spec_tokens_to_schedule from
# observed acceptance when VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH=1 (Kimi-K3 dspark)
#
# With VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH=1 the dspark proposer accepts a
# per-step draft count (v1/worker/gpu/spec_decode/dflash/speculator.py:165
# `dynamic_physical_depth`; :539-543 uses the scheduler-provided
# `num_speculative_tokens`), and the V2 model runner forwards
# SchedulerOutput.num_spec_tokens_to_schedule into it
# (v1/worker/gpu/model_runner.py:2068-2070, 2119-2125, 2243). But the
# scheduler only varies that field when an AcceptanceLengthController
# exists, and the controller is instantiated exclusively from the
# `adaptive_speculative_tokens_window` speculative-config field
# (v1/core/sched/scheduler.py:262-269) — which the serving config does not
# set. Result: the scheduler echoes the static ceiling every step
# (metrics: tokens/draft exactly 7.00), so nothing drives the dynamic path.
#
# Fix (scheduler-local, minimal): when VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH is
# set, the method is dspark, and no explicit window config was given,
# instantiate the fork's own AcceptanceLengthController with window
# VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH_WINDOW (default 8). The scheduler's
# existing observation path (update_from_output: per-request
# accepted/draft accounting at :1840-1861, observe_batch at :2069-2088)
# then feeds it; the decided depth is emitted as
# num_spec_tokens_to_schedule at :1245-1252 and as the async-scheduling
# placeholder length (async_scheduler.py:23-25). Depth policy (the fork's
# acceptance-length rule): start at the config ceiling, target =
# floor(mean accepted draft tokens per draft + 1.5) clamped to
# [1, ceiling], step down directly to target, step up by one per window.
# Per-depth full-cudagraph capture already happens for dspark+env
# (v1/worker/gpu/cudagraph_utils.py:262-275), so reduced depths verify
# fewer tokens instead of replaying the max-depth graph.
#
# No-op unless VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH is set; an explicit
# adaptive_speculative_tokens_window config keeps priority (elif).
#
# NOTE: the Kimi-K3 image has WORKDIR=/opt/kimi-k3/vllm, so `import vllm`
# resolves to the SOURCE TREE (/opt/kimi-k3/vllm/vllm), shadowing the
# site-packages install. Patch every scheduler.py copy that exists.
set -euo pipefail

echo "--- Applying scheduler-side adaptive dspark draft depth (fix-dspark-adaptive-nst)..."

python3 << 'PYTHON_PATCH'
import os, subprocess, sys

candidates = []

# 1) Import-resolved package (what the engine core actually loads from cwd).
try:
    d = subprocess.run(
        ["python3", "-c", "import vllm, os; print(os.path.dirname(vllm.__file__))"],
        capture_output=True, text=True,
    ).stdout.strip()
    if d:
        candidates.append(os.path.join(d, "v1", "core", "sched", "scheduler.py"))
except Exception:
    pass

# 2) Known install locations (source-tree and venv layouts).
for base in (
    "/opt/kimi-k3/vllm/vllm",
    "/opt/venv/lib/python3.12/site-packages/vllm",
    "/usr/local/lib/python3.12/dist-packages/vllm",
):
    candidates.append(os.path.join(base, "v1", "core", "sched", "scheduler.py"))

# De-duplicate by realpath, keep order.
seen, files = set(), []
for f in candidates:
    rp = os.path.realpath(f)
    if rp not in seen and os.path.isfile(f):
        seen.add(rp)
        files.append(f)

if not files:
    print("  ⚠ No scheduler.py found in any known vllm location — nothing to patch")
    sys.exit(0)

OLD_IMPORT = """from vllm.compilation.cuda_graph import CUDAGraphStat"""

NEW_IMPORT = """from vllm import envs
from vllm.compilation.cuda_graph import CUDAGraphStat"""

OLD_INIT = """            if (
                observation_window := (
                    speculative_config.adaptive_speculative_tokens_window
                )
            ) is not None:
                self.acceptance_length_controller = AcceptanceLengthController(
                    max_num_spec_tokens=self.num_spec_tokens,
                    observation_window=observation_window,
                )
"""

NEW_INIT = """            if (
                observation_window := (
                    speculative_config.adaptive_speculative_tokens_window
                )
            ) is not None:
                self.acceptance_length_controller = AcceptanceLengthController(
                    max_num_spec_tokens=self.num_spec_tokens,
                    observation_window=observation_window,
                )
            elif (
                envs.VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH
                and speculative_config.use_dspark()
                and self.num_spec_tokens > 1
            ):
                # fix-dspark-adaptive-nst: without this the scheduler echoes
                # the static ceiling in num_spec_tokens_to_schedule every
                # step, so the dspark proposer is always handed the max
                # count even with VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH=1. Drive
                # the per-step count from the rolling acceptance window
                # (VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH_WINDOW, default 8):
                # target = floor(mean accepted draft tokens + 1.5), step
                # down to target, step up by one, clamp [1, ceiling].
                self.acceptance_length_controller = AcceptanceLengthController(
                    max_num_spec_tokens=self.num_spec_tokens,
                    observation_window=max(
                        1, envs.VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH_WINDOW
                    ),
                )
                logger.info(
                    "fix-dspark-adaptive-nst: DSpark scheduler-side adaptive "
                    "draft depth enabled (ceiling=%d, window=%d).",
                    self.num_spec_tokens,
                    self.acceptance_length_controller.observation_window,
                )
"""

patched_any = False
for FILE in files:
    with open(FILE) as f:
        content = f.read()

    if "fix-dspark-adaptive-nst" in content:
        print(f"  Already patched: {FILE}")
        patched_any = True
        continue

    if OLD_INIT not in content or OLD_IMPORT not in content:
        print(f"  ⚠ Anchor not found (file shape changed?) — skipping: {FILE}")
        idx = content.find("acceptance_length_controller = AcceptanceLengthController(")
        if idx >= 0:
            print(content[max(0, idx - 400):idx + 200])
        continue

    content = content.replace(OLD_IMPORT, NEW_IMPORT, 1)
    content = content.replace(OLD_INIT, NEW_INIT, 1)
    with open(FILE, "w") as f:
        f.write(content)
    print(f"  ✓ Patched Scheduler.__init__ (env-gated AcceptanceLengthController): {FILE}")
    patched_any = True

if not patched_any:
    raise SystemExit(1)
PYTHON_PATCH

# Validate every patched copy still compiles.
for CAND in \
    /opt/kimi-k3/vllm/vllm/v1/core/sched/scheduler.py \
    /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py
do
    if [ -f "$CAND" ] && grep -q "fix-dspark-adaptive-nst" "$CAND"; then
        python3 -m py_compile "$CAND"
        echo "  ✓ py_compile OK: $CAND"
    fi
done

# Drop stale bytecode for the patched module so the engine core cannot load
# a pre-patch .pyc if source mtime granularity is ever defeated.
find /opt/kimi-k3/vllm/vllm/v1/core/sched/__pycache__ /opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/__pycache__ \
    -name 'scheduler.cpython-*.pyc' -delete 2>/dev/null || true

echo "=== fix-dspark-adaptive-nst complete ==="
