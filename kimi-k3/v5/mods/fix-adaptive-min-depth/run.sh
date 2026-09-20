#!/usr/bin/env bash
# fix-adaptive-min-depth — raise the adaptive draft-depth floor via env
# (default 4 instead of the fork's hardcoded 1).
#
# The fork's AcceptanceLengthController targets floor(mean_accepted + 1.5)
# clamped to [1, ceiling]: with typical means ~3 it parks at 4 anyway, but
# transient dips collapse it to 1-2 where verify rows go idle and recovery
# climbs +1/window. A floor of 4 keeps the verify pipeline fed through dips
# while preserving adaptation in the productive 4..ceiling band.
# Env-gated: VLLM_DSPARK_DYNAMIC_MIN_DEPTH (default "4"); unset/empty keeps
# the new default, explicit "1" restores upstream behavior.
#
# Marker: fix-adaptive-min-depth

set -e
PATCHED=0
for FILE in $(find /opt/kimi-k3 /opt/venv /usr/local/lib -path "*vllm/v1/spec_decode/dynamic/acceptance_length.py" 2>/dev/null | grep -v __pycache__ | sort -u); do
  if grep -q "fix-adaptive-min-depth" "$FILE" 2>/dev/null; then
    echo "[fix-adaptive-min-depth] already applied in $FILE"
    PATCHED=1
    continue
  fi
  if ! grep -q "max(1, floor(mean_num_accepted_tokens" "$FILE" 2>/dev/null; then
    echo "[fix-adaptive-min-depth] WARNING: anchor not found in $FILE, skipping"
    continue
  fi
  python3 - "$FILE" <<'PYEOF'
import sys
p = sys.argv[1]
s = open(p).read()
old = """        target_num_spec_tokens = min(
            self.max_num_spec_tokens,
            max(1, floor(mean_num_accepted_tokens + 1.5)),
        )"""
new = """        # fix-adaptive-min-depth: floor the adaptive target via env
        # (VLLM_DSPARK_DYNAMIC_MIN_DEPTH, default 4) instead of 1, so
        # transient dips do not idle the verify pipeline below the
        # productive band.
        import os as _min_os
        try:
            _min_depth = max(
                1, int(_min_os.environ.get("VLLM_DSPARK_DYNAMIC_MIN_DEPTH", "4"))
            )
        except ValueError:
            _min_depth = 4
        target_num_spec_tokens = min(
            self.max_num_spec_tokens,
            max(_min_depth, floor(mean_num_accepted_tokens + 1.5)),
        )"""
assert old in s, "anchor not found"
open(p, "w").write(s.replace(old, new, 1))
print("patched", p)
PYEOF
  python3 -m py_compile "$FILE" && echo "[fix-adaptive-min-depth] APPLIED + py_compile OK: $FILE"
  PATCHED=1
done
[ "$PATCHED" = "1" ] || { echo "[fix-adaptive-min-depth] nothing patched"; exit 1; }
