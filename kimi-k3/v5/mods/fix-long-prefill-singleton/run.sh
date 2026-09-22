#!/usr/bin/env bash
# fix-long-prefill-singleton — don't let long_prefill_token_threshold cap a
# lone request (adapted from upstream vLLM PR #57951).
#
# The threshold exists to stop a long prefill from starving OTHER requests of
# the token budget. When it is the only request in the system (running +
# waiting + skipped_waiting), there is nobody to starve, so the cap only
# needlessly splits the prefill into threshold-sized chunks. This mod computes
# an effective threshold once per schedule() call — the configured value when
# more than one request exists, 0 (disabled) otherwise — and uses it at both
# cap sites in vllm/v1/core/sched/scheduler.py.
#
# Out of scope (NOT touched): the block-alignment region using the local
# `long_prefill_threshold` (mamba slot-alignment cap) — different purpose.
#
# All-or-nothing per file: every anchor must match exactly once or the file is
# left untouched (exit 3 = no anchors, skip copy; exit 4 = partial match, die).
# Marker: fix-long-prefill-singleton

set -e

MARKER="fix-long-prefill-singleton"
TARGET_REL="vllm/v1/core/sched/scheduler.py"
# MOD_FIND_ROOTS override exists for offline validation against /tmp copies;
# production default matches the debug-kv-groups convention.
FIND_ROOTS="${MOD_FIND_ROOTS:-/opt/kimi-k3 /opt/venv /usr/local/lib}"

PATCHED=0
for FILE in $(find $FIND_ROOTS -path "*$TARGET_REL" 2>/dev/null | grep -v __pycache__ | sort -u); do
  if grep -q "$MARKER" "$FILE" 2>/dev/null; then
    echo "[fix-long-prefill-singleton] already applied in $FILE"
    PATCHED=1
    continue
  fi
  if python3 - "$FILE" <<'PYEOF'
import sys

p = sys.argv[1]
s = open(p).read()

sites = [
    # (a) compute the effective threshold once per schedule() call
    (
        "guard",
        """        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)

        # First, schedule the RUNNING requests.
""",
        """        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)

        # fix-long-prefill-singleton (adapted upstream #57951): the threshold
        # exists to stop a long prefill starving others; alone, let it use the budget.
        long_prefill_token_threshold = (
            self.scheduler_config.long_prefill_token_threshold
            if len(self.running) + len(self.waiting) + len(self.skipped_waiting) > 1
            else 0
        )

        # First, schedule the RUNNING requests.
""",
    ),
    # (b) RUNNING-requests cap site -> use the effective threshold
    (
        "running-cap",
        """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
""",
        """            if 0 < long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = long_prefill_token_threshold
""",
    ),
    # (c) WAITING-requests cap site -> use the effective threshold
    (
        "waiting-cap",
        """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
""",
        """                    if 0 < long_prefill_token_threshold < num_new_tokens:
                        num_new_tokens = long_prefill_token_threshold
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

for _name, old, new in sites:
    s = s.replace(old, new, 1)
open(p, "w").write(s)
print(f"applied {len(sites)} sites: {[n for n, _, _ in sites]}")
PYEOF
  then
    python3 -m py_compile "$FILE" && echo "[fix-long-prefill-singleton] APPLIED + py_compile OK: $FILE"
    PATCHED=1
  else
    rc=$?
    if [ "$rc" = "3" ]; then
      echo "[fix-long-prefill-singleton] WARNING: anchors not found in $FILE (different vllm version?), skipping"
    else
      echo "[fix-long-prefill-singleton] ERROR: refusing to patch $FILE (partial/duplicate anchors)"
      exit 1
    fi
  fi
done
[ "$PATCHED" = "1" ] || { echo "[fix-long-prefill-singleton] ERROR: primary file not patched (no $TARGET_REL with anchors found under: $FIND_ROOTS)"; exit 1; }
