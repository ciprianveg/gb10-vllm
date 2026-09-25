#!/usr/bin/env bash
# fix-k3-dflash-unaligned-endpoint-restore — stop disabling DFlash/DSpark
# drafting for the whole batch on a block-unaligned cache-restored prefix.
#
# ROOT CAUSE (oracle-verified 2026-09-25): the dflash speculator's propose()
# bails for the ENTIRE batch when any request's num_cached_tokens is not a
# multiple of the draft block size (768 on this config), claiming "draft KV
# is not available for the restored partial block". That claim is FALSE for
# every unaligned restore source on this fleet:
#   - Normal APC can never be unaligned: SlidingWindowManager rejects
#     partial hits (alignment assert) and hybrid min-reconciliation only
#     lowers hits to other aligned boundaries.
#   - The request-endpoint cache (fix-k3-request-endpoint-cache) restores
#     the draft sliding-window TAIL blocks — including the partial block,
#     whose slots [0, E%768) hold VALID draft KV from the finished request.
#   - Streaming-input session resume likewise keeps the session's own draft
#     tail blocks.
# The shift machinery already floors the restore count for ANY num_cached:
#   _prepare_dflash_inputs_kernel:  num_shifted_slots = (num_cached //
#     block_size) * block_size
#   _shift_draft_block_tables_kernel: cached_shift = num_cached // block_size
# so the draft frame is the contiguous suffix [floor(num_cached), total) and
# the geometry is unaligned-tolerant by construction. Draft proposals remain
# target-verified (block rejection sampling), so residual risk is bounded to
# wasted drafts, never corrupted output.
#
# EFFECT: with the endpoint cache ON, drafting was disabled on essentially
# every opencode follow-up turn (88-92% prefix hit rate traffic restored at
# arbitrary token boundaries → Per-position acceptance 0.000 across the
# board). After this mod, drafting runs through unaligned restores at zero
# extra recompute cost (the target still resumes exactly at the endpoint).
#
# NOTE: scheduler-side flooring of the restore count was investigated and
# REJECTED — the restore count is the position of the restored mamba/KDA
# recurrent state; flooring it either double-advances the state (silent
# corruption) or drops the state (defeats the endpoint cache's whole win).
# Upstream fork #482 (local-inference-lab/vllm) truncates at admission but
# is gated to DCP=1 / no-mamba-align and is inapplicable here.
#
# Hunks:
#   1. dflash/speculator.py propose(): replace the batch-wide bail-out
#      (warning + fill(-1) + return) with a one-time info log.
#   2. dspark/speculator.py propose(): drop the _has_unaligned_cached_prefix
#      term from _last_proposal_confidence_valid (capacity confidence stays
#      honest while drafting through unaligned restores).
#
# All-or-nothing per file: every anchor must match exactly once or the file
# is left untouched (exit 3 = no anchors, skip copy; exit 4 = partial/
# duplicate match or invalid syntax, die; exit 5 = py_compile failure).
# Marker: fix-k3-dflash-unaligned-endpoint-restore

set -euo pipefail

MOD="fix-k3-dflash-unaligned-endpoint-restore"
FIND_ROOTS="${MOD_FIND_ROOTS:-/opt/kimi-k3 /opt/venv /usr/local/lib}"

PATCHED=0
for FILE in $(find $FIND_ROOTS -type f \( -path "*vllm/v1/worker/gpu/spec_decode/dflash/speculator.py" -o -path "*vllm/v1/worker/gpu/spec_decode/dspark/speculator.py" \) 2>/dev/null | grep -v __pycache__ | sort -u || true); do
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

is_dflash = p.endswith("/dflash/speculator.py")

sites = []

if is_dflash:
    sites.append((
        "bailout-removal",
        """        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        if not dummy_run and self._has_unaligned_cached_prefix(input_batch):
            logger.warning_once(
                "DFlash/DSpark drafting is disabled for a batch containing a "
                "block-unaligned cache-restored prefix because draft KV is "
                "not available for the restored partial block."
            )
            self.draft_tokens[:num_reqs].fill_(-1)
            return self.draft_tokens[:num_reqs]
""",
        """        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        # fix-k3-dflash-unaligned-endpoint-restore: do NOT disable drafting
        # for the whole batch on a block-unaligned cache-restored prefix.
        # The shift path already floors the restore count
        # (_prepare_dflash_inputs_kernel: num_shifted_slots =
        # (num_cached // block_size) * block_size; _shift_draft_block_tables_
        # kernel: cached_shift = num_cached // block_size), so the draft
        # frame is the contiguous suffix [floor(num_cached), total) for any
        # restore count. The residual partial block carries VALID draft KV
        # for every unaligned restore source on this config: the
        # request-endpoint cache (fix-k3-request-endpoint-cache) and
        # streaming-input session resume both restore the draft
        # sliding-window tail blocks, while normal APC hits are always
        # whole-block (SlidingWindowManager rejects partial hits). Draft
        # proposals remain target-verified either way.
        if not dummy_run and self._has_unaligned_cached_prefix(input_batch):
            logger.info_once(
                "DFlash/DSpark drafting through a block-unaligned "
                "cache-restored prefix (request-endpoint cache restore); "
                "the block-table shift floors the restore count and the "
                "partial block holds restored draft KV."
            )
""",
    ))
else:
    sites.append((
        "capacity-gate-term",
        """        self._last_proposal_confidence_valid = bool(
            self.use_draft_token_capacity
            and not kwargs.get("is_profile", False)
            and not kwargs.get("dummy_run", False)
            and not self._has_unaligned_cached_prefix(input_batch)
            and (
                self.capacity_activation_batch_size <= 0
                or input_batch.num_reqs >= self.capacity_activation_batch_size
            )
        )
""",
        """        self._last_proposal_confidence_valid = bool(
            self.use_draft_token_capacity
            and not kwargs.get("is_profile", False)
            and not kwargs.get("dummy_run", False)
            # fix-k3-dflash-unaligned-endpoint-restore: an unaligned
            # restore (request-endpoint cache) no longer invalidates the
            # proposal confidence — drafting proceeds through it (see the
            # DFlash propose() bail-out removal).
            and (
                self.capacity_activation_batch_size <= 0
                or input_batch.num_reqs >= self.capacity_activation_batch_size
            )
        )
""",
    ))

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
[ "$PATCHED" = "1" ] || { echo "[$MOD] ERROR: no speculator files patched under: $FIND_ROOTS"; exit 1; }
