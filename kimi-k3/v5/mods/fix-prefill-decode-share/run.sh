#!/usr/bin/env bash
# fix-prefill-decode-share — decode-favoring prefill interleave for the
# KimiK3 fork (the honest equivalent of DS4.1's --prefill-compute-share,
# which this fork lacks).
#
# During long chunked prefills, concurrent decode streams stall in
# compute-bound mixed steps. The fork's scheduler already has a DP-gated
# defer mechanism: when defer_prefills is True, in-flight prefill chunks
# are skipped for the step ("decodes still run to fill this step") and
# waiting prefills break out — but throttle_prefills is hardwired False at
# DP=1 (EngineCore._should_throttle_prefills returns False
# unconditionally), so the mechanism never fires for us.
#
# This mod adds an env-driven cadence to the defer_prefills computation in
# vllm/v1/core/sched/scheduler.py (oracle-spec'd):
#   VLLM_PREFILL_COMPUTE_SHARE_INTERVAL=N (N>1): every step where
#   current_step % N != 0 defers prefill chunks (decodes run alone —
#   decode-only steps are ~10-20x cheaper than chunk steps); every Nth
#   step runs prefills. Env unset/0 = exact current behavior (no-op).
# Existing guards are untouched: any(not r.is_prefill_chunk ...) keeps a
# defer from starving a prefill-only batch (no deadlock), and
# prefill_capacity_bound auto-disables the defer when the waiting queue
# is saturated.
#
# IMPLEMENTATION NOTE (deliberate, oracle-semantics-preserving): the oracle
# spec rewrites the defer_prefills expression in place. We instead keep the
# existing defer_prefills block BYTE-EXACT and rebind throttle_prefills
# immediately before it:
#     _pcs_interval = int(os.environ.get("VLLM_PREFILL_COMPUTE_SHARE_INTERVAL", "0"))
#     throttle_prefills = throttle_prefills or (
#         _pcs_interval > 0 and self.current_step % _pcs_interval != 0
#     )
# This is boolean-identical to the oracle expression — throttle_prefills is
# a schedule() parameter consumed ONLY by the defer_prefills block in this
# file (verified against the baked extract), so the rebind is
# side-effect-free — and it keeps the block intact so that
# mods/fix-long-prefill-singleton (whose "guard" anchor spans this block)
# remains applicable in EITHER order. Verified offline both ways; the
# combined file is byte-identical regardless of application order.
#
# Also adds `import os` (missing from the baked scheduler's import block).
#
# Inert by construction at PP=1: the per-request
# `current_step < request.next_decode_eligible_step` guard (scheduler line
# ~582) is never advanced anywhere in the scheduler — it only fires on the
# V2+PP+async path — so it stays at its Request init value 0 and cannot
# interact with this step-level cadence.
#
# All-or-nothing per file: every anchor must match exactly once or the file
# is left untouched (exit 3 = no anchors, skip copy; exit 4 = partial/
# duplicate match or invalid syntax, die; exit 5 = py_compile failure).
# Marker: fix-prefill-decode-share

set -euo pipefail

MOD="fix-prefill-decode-share"
TARGET_REL="vllm/v1/core/sched/scheduler.py"
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
    # (a) the baked scheduler does not import os; the cadence reads the env
    # directly so no second file (vllm/envs.py) needs patching.
    (
        "os-import",
        """import itertools
import math
import time
""",
        """import itertools
import math
import os
import time
""",
    ),
    # (b) decode-favoring cadence: rebind throttle_prefills right before the
    # defer_prefills block (kept byte-exact — see header note). Anchor is
    # the bare block so it matches BOTH the pre- and post-
    # fix-long-prefill-singleton states of this file.
    (
        "defer-cadence",
        """        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)
""",
        """        # fix-prefill-decode-share (oracle-spec'd; DS4.1
        # --prefill-compute-share equivalent): env-driven decode-favoring
        # cadence. With VLLM_PREFILL_COMPUTE_SHARE_INTERVAL=N (N>1), every
        # step where current_step % N != 0 defers in-flight prefill chunks
        # so decodes run alone; every Nth step runs prefills. Env unset/0
        # = exact current behavior. throttle_prefills is consumed only by
        # the defer_prefills block below, so rebinding it here is
        # side-effect-free. Existing guards unchanged: no defer when only
        # prefill chunks run (no deadlock); defer auto-disables when the
        # waiting queue is saturated (prefill_capacity_bound).
        _pcs_interval = int(os.environ.get("VLLM_PREFILL_COMPUTE_SHARE_INTERVAL", "0"))
        throttle_prefills = throttle_prefills or (
            _pcs_interval > 0 and self.current_step % _pcs_interval != 0
        )
        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)
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
