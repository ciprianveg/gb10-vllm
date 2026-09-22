#!/usr/bin/env bash
# harden-apc-drain — hardening on TOP of the two APC fixes baked into v5-prd
# (fix-kv-dedup-retained-endpoints in kv_cache_manager.py,
#  fix-mamba-align-state-free in single_type_kv_cache_manager.py).
# BOOT ORDER: run AFTER both fix mods — every anchor here is the POST-fix
# (baked) state; on the pre-fix state the anchors are absent and this mod
# fails loudly (exit 1, "not hardened in any candidate").
#
# Items (spec /tmp/opencode/harden-apc-drain-spec.md, oracle-reviewed):
#  1a  block_pool.free_blocks: ref_cnt underflow tripwire. ref_cnt<=0 at the
#      decrement means a double/over-free (CoW fence vs table-ref accounting
#      bug). Log-and-skip: skip the decrement, do not enqueue to the free
#      queue. VLLM_APC_DRAIN_ABORT=1 escalates to RuntimeError (soak only).
#  1b+3c (ONE replacement, same span) single_type align free: free the
#      two-steps-ago block only when it is strictly behind the committed
#      frontier (cdiv(processed_computed_tokens, block_size) - 1) — a
#      transient frontier lag defers the free instead of freeing a live
#      block; verify saved block identity vs the table slot (fork drift ->
#      drop bookkeeping, do NOT free); underflow tripwire on the slot block.
#      Converts the only silent-corruption path into a bounded leak.
#  3a  __init__: _two_steps_ago_block dict (identity saved alongside index).
#      NOTE: spec's literal two-line anchor does not exist in the baked
#      new_init (the _allocated_block_reqs pair sits between the decls);
#      anchored on the real baked span, dict kept adjacent to
#      _two_steps_ago_block_idx.
#  3b  allocate_new_blocks save site: record the block identity too.
#  3d  pop_blocks_for_free: clean up _two_steps_ago_block.
#  2   kv_cache_manager drain: raw-vs-dedup pair counts + free-pool level,
#      every 256th drain, only when VLLM_APC_DRAIN_DEBUG=1 (debug-only).
#
# Trip policy: explicit if (never assert); rate-limited logger.error, one
# line per block id per 1000 trips, module-level total counter; the table
# slot is still nulled on trip (corruption barrier).
# Env: VLLM_APC_DRAIN_ABORT=1 (abort on trip), VLLM_APC_DRAIN_DEBUG=1 (Item 2).
#
# All-or-nothing per file: every anchor must match exactly once or the file
# is left untouched (exit 3 = no anchors, skip copy; exit 4 = partial/
# duplicate anchors or syntax error, die; exit 5 = py_compile fail).
# Marker: harden-apc-drain
#
# MULTI-COPY REALITY (v5-prd boot #1 post-mortem): the image ships three
# single_type_kv_cache_manager.py copies (source tree /opt/kimi-k3/vllm/vllm,
# a build/lib.linux-* leftover, and /opt/venv site-packages), but
# fix-mamba-align-state-free bakes ONLY the /opt/kimi-k3/vllm/vllm copy —
# the other two are PRE-fix and must not be treated as patch targets (their
# import lines match the helper anchor but none of the baked-fix anchors do,
# which turned a skippable rc=3 into a fatal rc=4 and killed boot #1). The
# single_type target therefore requires the baked prerequisite marker
# fix-mamba-align-state-free in a candidate before attempting it: copies
# without it are skipped with a warning; a copy WITH the marker but
# partial/duplicate anchors is real fork drift and stays fatal; if NO copy
# carries the prerequisite the mod still dies (fix mods missing / wrong boot
# order). The same applies to kv_cache_manager.py (fix-kv-dedup's own scan
# never covers build/lib leftovers, so its copy there is pre-dedup).
# block_pool.py needs no gate: all three image copies are byte-identical and
# this mod is the first to touch it.

set -euo pipefail

MARKER="harden-apc-drain"
# MOD_FIND_ROOTS override exists for offline validation against /tmp copies;
# production default matches the debug-kv-groups convention.
FIND_ROOTS="${MOD_FIND_ROOTS:-/opt/kimi-k3 /opt/venv /usr/local/lib}"
TARGET_RELS=(
  "vllm/v1/core/block_pool.py"
  "vllm/v1/core/single_type_kv_cache_manager.py"
  "vllm/v1/core/kv_cache_manager.py"
)

PATCHER="$(mktemp /tmp/harden-apc-drain-patcher.XXXXXX.py)"
trap 'rm -f "$PATCHER"' EXIT

cat > "$PATCHER" <<'PYEOF'
# harden-apc-drain patcher: all-or-nothing anchor replacements, dispatched by
# path suffix. Exit codes: 0 applied | 3 no anchors | 4 partial/duplicate or
# syntax error (file untouched) | 5 py_compile fail.
import sys
import py_compile

p = sys.argv[1]
s = open(p).read()

FULL_HELPER = '''# ---- harden-apc-drain: module-scoped tripwire helpers ----
# Lazy one-shot binds: the patched modules do not all import os or define a
# logger name at top level. VLLM_APC_DRAIN_ABORT=1 escalates an underflow
# trip to RuntimeError (soak mode only); default policy is log-and-skip.
globals().setdefault("_os", __import__("os"))
globals().setdefault(
    "_harden_log",
    globals().get("logger")
    or __import__("vllm.logger", fromlist=["init_logger"]).init_logger(__name__),
)
_HARDEN_TRIPS: dict = {"counts": {}, "total": 0}


def _harden_underflow(block, req_id=None):
    # harden-apc-drain: ref_cnt underflow tripwire — a double/over-free.
    # Caller skips the decrement and still nulls the table slot (corruption
    # barrier). Rate-limited: one error log per block id per 1000 trips for
    # that id, plus a module-level total counter.
    _HARDEN_TRIPS["total"] += 1
    n = _HARDEN_TRIPS["counts"].get(block.block_id, 0) + 1
    _HARDEN_TRIPS["counts"][block.block_id] = n
    if _os.environ.get("VLLM_APC_DRAIN_ABORT") == "1":
        raise RuntimeError(
            f"harden-apc-drain: ref_cnt underflow block={block.block_id} "
            f"ref_cnt={block.ref_cnt} req={req_id}"
        )
    if n == 1 or n % 1000 == 0:
        _harden_log.error(
            "harden-apc-drain: ref_cnt underflow block=%d ref_cnt=%d req=%s "
            "trips=%d total=%d (log-and-skip)",
            block.block_id, block.ref_cnt, req_id, n,
            _HARDEN_TRIPS["total"],
        )'''

LIGHT_HELPER = '''# ---- harden-apc-drain: lazy one-shot binds for drain debug counters ----
# This module does not import os at top level; bind once at module scope.
globals().setdefault("_os", __import__("os"))
globals().setdefault(
    "_harden_log",
    globals().get("logger")
    or __import__("vllm.logger", fromlist=["init_logger"]).init_logger(__name__),
)'''

if p.endswith("vllm/v1/core/block_pool.py"):
    sites = [
        # module-scoped helpers (after the existing logger binding)
        (
            "helper",
            """logger = init_logger(__name__)""",
            """logger = init_logger(__name__)


""" + FULL_HELPER,
        ),
        # Item 1a: underflow tripwire before the free_blocks decrement
        (
            "free_blocks-tripwire",
            """        for block in ordered_blocks:
            block.ref_cnt -= 1
""",
            """        for block in ordered_blocks:
            # harden-apc-drain: underflow tripwire. ref_cnt==0 here means a
            # double/over-free (CoW fence vs table-ref accounting bug).
            if block.ref_cnt <= 0 and not block.is_null:
                _harden_underflow(block)   # log-and-skip policy, see below
                continue                   # SKIP the decrement; do not enqueue
            block.ref_cnt -= 1
""",
        ),
    ]
elif p.endswith("vllm/v1/core/single_type_kv_cache_manager.py"):
    sites = [
        # module-scoped helpers (after the import block; this file has no
        # top-level logger name, the helper binds one lazily)
        (
            "helper",
            """from vllm.v1.request import Request
""",
            """from vllm.v1.request import Request


""" + FULL_HELPER,
        ),
        # Item 3a: identity dict in __init__ (anchored on the real baked
        # new_init span; kept adjacent to _two_steps_ago_block_idx)
        (
            "init-identity-dict",
            """            self._two_steps_ago_block_idx: dict[str, int] = {}
            # The set of the requests that have been allocated blocks
            self._allocated_block_reqs: set[str] = set()
""",
            """            self._two_steps_ago_block_idx: dict[str, int] = {}
            # harden-apc-drain: block identity saved alongside the index so the
            # free can verify slot contents and the frontier before releasing.
            self._two_steps_ago_block: dict[str, object] = {}
            # The set of the requests that have been allocated blocks
            self._allocated_block_reqs: set[str] = set()
""",
        ),
        # Item 3b: save the block identity alongside the index
        (
            "alloc-save-identity",
            """                    old_last = self.last_state_block_idx.get(request_id)
                    if old_last is not None:
                        self._two_steps_ago_block_idx[request_id] = old_last
""",
            """                    old_last = self.last_state_block_idx.get(request_id)
                    if old_last is not None:
                        self._two_steps_ago_block_idx[request_id] = old_last
                        _hb = self.req_to_blocks[request_id]
                        self._two_steps_ago_block[request_id] = (
                            _hb[old_last] if old_last < len(_hb) else None
                        )
""",
        ),
        # Items 1b+3c (ONE replacement): frontier-guarded free + identity
        # check + underflow tripwire. processed_computed_tokens is the
        # enclosing remove_skipped_blocks parameter (same source the pre-fix
        # guard used); cdiv is already imported in this file.
        (
            "align-free-guarded",
            """            two_steps_ago_idx = self._two_steps_ago_block_idx.get(request_id)
            if two_steps_ago_idx is not None:
                # Blocks allocated during prefill may be non-contiguous. Use
                # the saved index to free the appropriate block and replace it
                # with a null block.
                blocks = self.req_to_blocks[request_id]
                if two_steps_ago_idx < len(blocks):
                    blk = blocks[two_steps_ago_idx]
                    if blk != self._null_block:
                        self.block_pool.free_blocks([blk])
                        blocks[two_steps_ago_idx] = self._null_block
                # Clean up the saved index
                self._two_steps_ago_block_idx.pop(request_id, None)
""",
            """            two_steps_ago_idx = self._two_steps_ago_block_idx.get(request_id)
            if two_steps_ago_idx is not None:
                blocks = self.req_to_blocks[request_id]
                # harden-apc-drain: free only when the saved block is strictly
                # behind the committed frontier. Transient frontier lag
                # (in-flight/spec tokens) defers the free to a later step
                # instead of dropping the entry; a persistent lag degrades to
                # the pre-fix leak (bounded, freed at request end), never to a
                # use-after-free.
                if two_steps_ago_idx < (
                    cdiv(processed_computed_tokens, self.block_size) - 1
                ):
                    if two_steps_ago_idx < len(blocks):
                        blk = blocks[two_steps_ago_idx]
                        if blk != self._null_block:
                            saved = self._two_steps_ago_block.get(request_id)
                            if saved is not None and blk is not saved:
                                # index/slot disagreement: fork drift. Drop
                                # bookkeeping, do NOT free (log once).
                                _harden_log.error(
                                    "harden-apc-drain align-free identity "
                                    "mismatch req=%s idx=%d", request_id,
                                    two_steps_ago_idx,
                                )
                            elif blk.ref_cnt <= 0:
                                # Item 1b tripwire: already pool-owned.
                                _harden_underflow(blk, request_id)
                                blocks[two_steps_ago_idx] = self._null_block
                            else:
                                self.block_pool.free_blocks([blk])
                                blocks[two_steps_ago_idx] = self._null_block
                    self._two_steps_ago_block_idx.pop(request_id, None)
                    self._two_steps_ago_block.pop(request_id, None)
""",
        ),
        # Item 3d: cleanup on request free
        (
            "pop-cleanup-identity",
            """            self._two_steps_ago_block_idx.pop(request_id, None)
            self._producer_partial_tail_reqs.pop(request_id, None)
""",
            """            self._two_steps_ago_block_idx.pop(request_id, None)
            self._two_steps_ago_block.pop(request_id, None)
            self._producer_partial_tail_reqs.pop(request_id, None)
""",
        ),
    ]
elif p.endswith("vllm/v1/core/kv_cache_manager.py"):
    sites = [
        # lazy _os/_harden_log binds (this module imports neither at top)
        (
            "helper",
            """logger = init_logger(__name__)""",
            """logger = init_logger(__name__)


""" + LIGHT_HELPER,
        ),
        # Item 2 anchor A: raw pair counter init
        (
            "drain-raw-init",
            """        pending_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
        seen_pairs: set[tuple[int, int]] = set()
""",
            """        pending_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
        seen_pairs: set[tuple[int, int]] = set()
        raw_pairs = 0
""",
        ),
        # Item 2 anchor B: count every raw pair before dedup
        (
            "drain-raw-count",
            """        for mgr in self.coordinator.single_type_managers:
            for source_block, cow_block in mgr.take_pending_cow_copies():
                pair_key = (source_block.block_id, cow_block.block_id)
""",
            """        for mgr in self.coordinator.single_type_managers:
            for source_block, cow_block in mgr.take_pending_cow_copies():
                raw_pairs += 1
                pair_key = (source_block.block_id, cow_block.block_id)
""",
        ),
        # Item 2 anchor C: debug-only drain stats before the return
        (
            "drain-debug-stats",
            """        retained: dict[int, KVCacheBlock] = {}
        for source_block, cow_block in pending_copies:
            retained.setdefault(source_block.block_id, source_block)
            retained.setdefault(cow_block.block_id, cow_block)
        return copies, list(retained.values())
""",
            """        retained: dict[int, KVCacheBlock] = {}
        for source_block, cow_block in pending_copies:
            retained.setdefault(source_block.block_id, source_block)
            retained.setdefault(cow_block.block_id, cow_block)
        # harden-apc-drain (debug-only): raw vs deduped pair counts + free-pool
        # level at matched load; off unless VLLM_APC_DRAIN_DEBUG=1.
        if _os.environ.get("VLLM_APC_DRAIN_DEBUG") == "1":
            self._harden_drain_n = getattr(self, "_harden_drain_n", 0) + 1
            if self._harden_drain_n % 256 == 0:
                _harden_log.warning(
                    "[apc-drain] drains=%d raw=%d uniq=%d retained=%d free=%d",
                    self._harden_drain_n, raw_pairs, len(seen_pairs),
                    len(retained), self.block_pool.get_num_free_blocks(),
                )
        return copies, list(retained.values())
""",
        ),
    ]
else:
    print(f"no site table for {p}", file=sys.stderr)
    sys.exit(3)

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

# syntax-check BEFORE writing so a bad patch never lands on disk
try:
    compile(s, p, "exec")
except SyntaxError as e:
    print(f"post-patch syntax error: {e}", file=sys.stderr)
    sys.exit(4)

open(p, "w").write(s)
try:
    py_compile.compile(p, doraise=True)
except py_compile.PyCompileError as e:
    print(f"py_compile failed: {e}", file=sys.stderr)
    sys.exit(5)
print(f"applied {len(sites)} sites: {[n for n, _, _ in sites]}")
PYEOF

# Baked prerequisite per target: a candidate lacking it is a pre-fix straggler
# (build/lib leftover, unpatched site-packages copy), NOT a patch target.
prereq_marker() {
  case "$1" in
    vllm/v1/core/single_type_kv_cache_manager.py) echo "fix-mamba-align-state-free" ;;
    vllm/v1/core/kv_cache_manager.py) echo "fix-kv-dedup-retained-endpoints" ;;
    *) echo "" ;;
  esac
}

for TARGET_REL in "${TARGET_RELS[@]}"; do
  PREREQ="$(prereq_marker "$TARGET_REL")"
  # FIND_ROOTS unquoted on purpose: space-separated list of search roots
  FOUND_FILES="$(find $FIND_ROOTS -path "*$TARGET_REL" 2>/dev/null | grep -v __pycache__ | sort -u || true)"
  if [ -z "$FOUND_FILES" ]; then
    echo "[$MARKER] ERROR: $TARGET_REL not found under: $FIND_ROOTS" >&2
    exit 1
  fi
  COVERED=0
  for FILE in $FOUND_FILES; do
    if grep -q "$MARKER" "$FILE" 2>/dev/null; then
      echo "[$MARKER] already applied in $FILE"
      COVERED=1
      continue
    fi
    if [ -n "$PREREQ" ] && ! grep -q "$PREREQ" "$FILE" 2>/dev/null; then
      echo "[$MARKER] WARNING: skipping copy without baked prerequisite '$PREREQ' (pre-fix straggler, not a patch target): $FILE" >&2
      continue
    fi
    if python3 "$PATCHER" "$FILE"; then
      python3 -m py_compile "$FILE" \
        || { echo "[$MARKER] ERROR: py_compile failed after patch: $FILE" >&2; exit 5; }
      echo "[$MARKER] APPLIED + py_compile OK: $FILE"
      COVERED=1
    else
      rc=$?
      if [ "$rc" = "3" ]; then
        echo "[$MARKER] WARNING: anchors not found in $FILE (baked fix mods missing or different vllm version?), skipping" >&2
      else
        echo "[$MARKER] ERROR: refusing to patch $FILE (rc=$rc: partial/duplicate anchors or syntax)" >&2
        exit 1
      fi
    fi
  done
  if [ "$COVERED" != "1" ]; then
    echo "[$MARKER] ERROR: $TARGET_REL not hardened in any candidate under: $FIND_ROOTS${PREREQ:+ (no copy carried baked prerequisite: $PREREQ)}" >&2
    echo "[$MARKER] ERROR: boot must apply fix-mamba-align-state-free and fix-kv-dedup-retained-endpoints BEFORE harden-apc-drain" >&2
    exit 1
  fi
done

echo "=== $MARKER complete ==="
