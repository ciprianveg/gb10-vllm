#!/usr/bin/env bash
# fix-no-mixed-steps — never batch a LONG prefill chunk with decode tokens
# in one engine step (SM121 interleaved-batched-MMA OOB class: mixed
# chunked-prefill + spec-decode steps fault cuBLAS/CUTLASS/FP8 kernels
# with illegal memory access on GB10).
#
# Mechanism: when any in-flight prefill chunk belongs to a request with
# more than VLLM_NO_MIX_LONG_PREFILL_TOKENS (default 32768) tokens still
# un-computed, decode requests are skipped for that step (prefill-only
# step). Decode-only steps recur via the fix-prefill-decode-share cadence
# (VLLM_PREFILL_COMPUTE_SHARE_INTERVAL) — the no-mix gate REQUIRES the
# cadence to be active (N>1), otherwise it stays off (no decode freeze).
# Also stays off on decode-only steps (defer_prefills) to avoid empty steps.
#
# Deadlock analysis:
# - only decodes running: no chunk exists -> gate off -> normal.
# - only prefills running: skipping decodes is vacuous -> normal.
# - both + cadence active: prefill steps skip decodes, decode steps skip
#   prefills -> both sides progress. No deadlock.
# - backlog saturated (prefill_capacity_bound): defer off; long-chunk steps
#   skip decodes; short-chunk steps still mix (bounded stall, documented).
# - waiting loop untouched: new requests are prefills; admitting them on a
#   prefill-only step is normal multi-prefill packing (tested geometry).

set -euo pipefail

MARKER="fix-no-mixed-steps"
TARGET="vllm/v1/core/sched/scheduler.py"

already() { grep -q "$MARKER" "$1" 2>/dev/null; }

python3 - <<'PYEOF'
import subprocess, sys, py_compile

MARKER = "fix-no-mixed-steps"
TARGET_REL = "vllm/v1/core/sched/scheduler.py"
ROOTS = __import__("os").environ.get(
    "MOD_FIND_ROOTS", "/opt/kimi-k3 /opt/venv /usr/local/lib").split()

# NOTE: no `import os` patch — the gate uses __import__("os") inline, and
# fix-prefill-decode-share owns the import-os hunk. Keeps both orders working.

# --- 1. no-mix gate: computed right AFTER the bare defer_prefills block ---
# (bare block WITHOUT the comment lines: the interleave mod inserts its
# throttle_prefills rebind between the DP comment and the block, so a
# comment-anchored hunk breaks depending on application order).
old_gate = '''        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)'''

new_gate = '''        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)

        # fix-no-mixed-steps: never batch a LONG prefill chunk with decode
        # tokens in one step (SM121 interleaved-batched-MMA OOB class).
        # Requires the interleave cadence (else decodes would freeze for
        # the whole prefill); off on decode-only steps (else empty steps).
        _nm_interval = int(
            __import__("os").environ.get("VLLM_PREFILL_COMPUTE_SHARE_INTERVAL", "0"))
        _nm_min_tokens = int(
            __import__("os").environ.get("VLLM_NO_MIX_LONG_PREFILL_TOKENS", "32768"))
        no_mix_skip_decodes = (
            _nm_interval > 1
            and not defer_prefills
            and any(
                r.is_prefill_chunk
                and (r.num_prompt_tokens - r.num_computed_tokens) > _nm_min_tokens
                for r in self.running
            )
        )'''

# --- 3. skip-decodes branch: right AFTER the defer branch in the loop ---
old_branch = '''            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue'''

new_branch = '''            if defer_prefills and request.is_prefill_chunk:
                # DP prefill balancing: defer this in-progress prefill chunk to a
                # cadence-aligned step; decodes still run to fill this step.
                req_index += 1
                continue

            if no_mix_skip_decodes and not request.is_prefill_chunk:
                # fix-no-mixed-steps: this step carries a long prefill chunk;
                # decode-only steps recur via the interleave cadence.
                req_index += 1
                continue'''

applied = 0
skipped = 0
found = []
for root in ROOTS:
    try:
        out = subprocess.run(
            ["find", root, "-path", "*" + TARGET_REL],
            capture_output=True, text=True, timeout=120).stdout
    except Exception:
        continue
    for line in out.splitlines():
        p = line.strip()
        if p and "__pycache__" not in p and p not in found:
            found.append(p)

for p in sorted(set(found)):
    try:
        with open(p) as f:
            s = f.read()
    except FileNotFoundError:
        continue
    if MARKER in s:
        print(f"[{MARKER}] already applied in {p}")
        applied += 1
        continue
    # gate (0 matches: warn+skip this copy; >1: fail-loud)
    _n = s.count(old_gate)
    if _n == 0:
        print(f"[{MARKER}] WARNING: gate anchor not found in {p}, skipping")
        skipped += 1
        continue
    if _n > 1:
        print(f"[{MARKER}] ERROR: gate anchor count={_n} in {p}")
        sys.exit(4)
    s = s.replace(old_gate, new_gate, 1)
    # branch (0 matches: warn+skip this copy; >1: fail-loud)
    _n = s.count(old_branch)
    if _n == 0:
        print(f"[{MARKER}] WARNING: branch anchor not found in {p}, skipping")
        skipped += 1
        continue
    if _n > 1:
        print(f"[{MARKER}] ERROR: branch anchor count={_n} in {p}")
        sys.exit(4)
    s = s.replace(old_branch, new_branch, 1)
    # syntax check the PATCHED string BEFORE writing (fail-loud, no write)
    try:
        compile(s, p, "exec")
    except SyntaxError as e:
        print(f"[{MARKER}] ERROR: patched syntax invalid for {p}: {e}")
        sys.exit(5)
    with open(p, "w") as f:
        f.write(s)
    try:
        py_compile.compile(p, doraise=True)
    except Exception as e:
        print(f"[{MARKER}] ERROR: py_compile failed for {p}: {e}")
        sys.exit(5)
    print(f"[{MARKER}] APPLIED + py_compile OK: {p}")
    applied += 1

if applied == 0:
    print(f"[{MARKER}] WARNING: nothing applied (skipped {skipped}), exiting 3")
    sys.exit(3)
print(f"[{MARKER}] complete ({applied} files, {skipped} skipped)")
PYEOF
