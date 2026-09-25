#!/usr/bin/env bash
# fix-k3-nomix-standalone — arm the baked no-mix crash guard WITHOUT the
# decode-favoring interleave cadence (fix-prefill-decode-share).
#
# CONTEXT (2026-09-25): prod crash on .11 — Xid 31 + NV_ERR_NO_MEMORY (0x51)
# on a marlin MoE workspace aten::new_empty during a MIXED step (long
# prefill chunk batched with decode + speculative tokens). The baked
# fix-no-mixed-steps guard (FIX4) closes this class but only arms when
# VLLM_PREFILL_COMPUTE_SHARE_INTERVAL > 1, because without the cadence the
# skipped decodes would have no decode-only steps to recur on — the original
# design assumed decodes must keep flowing during long prefills.
#
# The user measured the interleave as a net loss for their (single-request
# opencode) workload and wants it OFF. For that traffic shape there are no
# concurrent decodes to protect during a prefill, so letting decodes skip
# long-chunk steps outright is free; for concurrent-stream workloads
# (game-bench) decodes now STALL (rather than crawl) for the duration of a
# long prefill — the documented tradeoff of running standalone.
#
# MECHANISM: the baked gate
#     no_mix_skip_decodes = (
#         _nm_interval > 1
#         and not defer_prefills
#         and any(<long prefill chunk running>))
# becomes
#     _nm_arm = int(env VLLM_NO_MIX_ARM, "0")
#     no_mix_skip_decodes = (
#         (_nm_interval > 1 or _nm_arm > 0)
#         and not defer_prefills
#         and any(<long prefill chunk running>))
# VLLM_NO_MIX_ARM=1 arms the guard with the cadence unset. The threshold
# stays VLLM_NO_MIX_LONG_PREFILL_TOKENS (default 32768; prod sets 4096 so
# any multi-chunk prefill is guarded — the 2026-09-25 crash request had
# only ~10K un-computed tokens after a 92.6% endpoint-cache hit, which the
# 32768 default would NOT have caught).
#
# DEADLOCK ANALYSIS (unchanged from fix-no-mixed-steps, minus the cadence
# clause): decodes are skipped only on steps that carry a long prefill
# chunk; the prefill always progresses, decodes resume when it completes.
# No empty steps (the chunk fills the step). No deadlock.
#
# All-or-nothing per file: anchor must match exactly once or the file is
# left untouched (exit 3 = no anchors, skip copy; exit 4 = partial/
# duplicate match or invalid syntax, die; exit 5 = py_compile failure).
# Marker: fix-k3-nomix-standalone

set -euo pipefail

MOD="fix-k3-nomix-standalone"
TARGET_REL="vllm/v1/core/sched/scheduler.py"
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
    (
        "arm-decouple",
        """        _nm_interval = int(
            __import__("os").environ.get("VLLM_PREFILL_COMPUTE_SHARE_INTERVAL", "0"))
        _nm_min_tokens = int(
            __import__("os").environ.get("VLLM_NO_MIX_LONG_PREFILL_TOKENS", "32768"))
        no_mix_skip_decodes = (
            _nm_interval > 1
            and not defer_prefills
""",
        """        _nm_interval = int(
            __import__("os").environ.get("VLLM_PREFILL_COMPUTE_SHARE_INTERVAL", "0"))
        _nm_min_tokens = int(
            __import__("os").environ.get("VLLM_NO_MIX_LONG_PREFILL_TOKENS", "32768"))
        # fix-k3-nomix-standalone: VLLM_NO_MIX_ARM=1 arms the no-mix guard
        # without the interleave cadence. Decodes then skip long-chunk steps
        # outright (no decode-only recurrence) — correct for single-request
        # workloads; concurrent decodes stall for the prefill duration
        # (documented tradeoff). The prefill always progresses; no deadlock.
        _nm_arm = int(
            __import__("os").environ.get("VLLM_NO_MIX_ARM", "0"))
        no_mix_skip_decodes = (
            (_nm_interval > 1 or _nm_arm > 0)
            and not defer_prefills
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
      echo "[$MOD] WARNING: anchors not found in $FILE (no baked no-mix guard?), skipping"
    else
      echo "[$MOD] ERROR: refusing to patch $FILE (exit $rc: partial/duplicate anchors or compile failure)"
      exit 1
    fi
  fi
done
[ "$PATCHED" = "1" ] || { echo "[$MOD] ERROR: primary file not patched (no $TARGET_REL with anchors found under: $FIND_ROOTS)"; exit 1; }
