#!/usr/bin/env bash
# fix-k3-revert-apc-leak — revert the leaky CoW-drain dedup back to pristine
# per-occurrence retention (oracle-confirmed root cause of native prefix-cache
# collapse + hangs).
#
# Root cause (audited): fix-kv-dedup-retained-endpoints replaced the
# per-occurrence retained-endpoint list with a dict keyed by block_id, so a
# source block shared by >=2 CoW pairs in one drain (the defining event of
# prefix-heavy traffic: partial hits on the shared cached tail block) is
# released once instead of once per pair. Leaked +1 ref per shared block per
# step -> permanently pinned blocks -> pool exhaustion -> cached prefixes
# evicted to satisfy allocations (hit-rate collapse ~9%) and eventual
# ValueError/hang. The endpoint cache masked it by bypassing the CoW path.
#
# This mod restores the pristine upstream functions:
#   take_kv_cache_block_copies: per-occurrence pairs + per-occurrence retained
#   take_partial_tail_offloads: per-occurrence pins (upstream behavior)
# It also removes harden-apc-drain's kv_cache_manager sites (helper binds,
# raw-pair counters, drain debug stats), which anchor on the DEDUPED text
# and cannot coexist with the pristine functions. harden-apc-drain's
# block_pool tripwire + single_type mamba guard are SEPARATE files/sites and
# stay in place (protective, unrelated to the leak).
#
# Application order inside the file (single pass, atomic write):
#   Phase 1 (harden-kv removals): restores the exact deduped text the
#   kv-dedup anchors were written against.
#   Phase 2 (dedup reversal): restores pristine upstream text.
# Either phase failing -> file untouched (exit 4), so a tree that never had
# the mods (or has only one half) is skipped safely, never half-patched.
# Copies lacking EITHER mod are skipped with a warning (pre-fix stragglers).
#
# All-or-nothing per file: exit 3 = no anchors found / all copies skipped,
# exit 4 = ambiguous anchor (count>1) or phase-2 mismatch after phase-1,
# exit 5 = patched text fails compile/py_compile. Idempotent via marker;
# also detects the already-reverted state (neither old marker present AND
# pristine anchors present -> reports already-reverted).
# Marker: fix-k3-revert-apc-leak

set -euo pipefail

MARKER="fix-k3-revert-apc-leak"
TARGET_REL="vllm/v1/core/kv_cache_manager.py"

python3 - <<'PYEOF'
import os, subprocess, sys, py_compile

MARKER = "fix-k3-revert-apc-leak"
TARGET_REL = "vllm/v1/core/kv_cache_manager.py"
ROOTS = os.environ.get(
    "MOD_FIND_ROOTS", "/opt/kimi-k3 /opt/venv /usr/local/lib").split()
OLD_MARKERS = ("fix-kv-dedup-retained-endpoints", "harden-apc-drain")

# ---------- Phase 1: remove harden-apc-drain kv_cache_manager sites ----------
HARDEN_HELPER_OLD = '''logger = init_logger(__name__)


# ---- harden-apc-drain: lazy one-shot binds for drain debug counters ----
# This module does not import os at top level; bind once at module scope.
globals().setdefault("_os", __import__("os"))
globals().setdefault(
    "_harden_log",
    globals().get("logger")
    or __import__("vllm.logger", fromlist=["init_logger"]).init_logger(__name__),
)'''
HARDEN_HELPER_NEW = '''logger = init_logger(__name__)'''

HARDEN_RAW_INIT_OLD = '''        pending_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
        seen_pairs: set[tuple[int, int]] = set()
        raw_pairs = 0
'''
HARDEN_RAW_INIT_NEW = '''        pending_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
        seen_pairs: set[tuple[int, int]] = set()
'''

HARDEN_RAW_COUNT_OLD = '''        for mgr in self.coordinator.single_type_managers:
            for source_block, cow_block in mgr.take_pending_cow_copies():
                raw_pairs += 1
                pair_key = (source_block.block_id, cow_block.block_id)
'''
HARDEN_RAW_COUNT_NEW = '''        for mgr in self.coordinator.single_type_managers:
            for source_block, cow_block in mgr.take_pending_cow_copies():
                pair_key = (source_block.block_id, cow_block.block_id)
'''

HARDEN_DEBUG_OLD = '''        retained: dict[int, KVCacheBlock] = {}
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
'''
HARDEN_DEBUG_NEW = '''        retained: dict[int, KVCacheBlock] = {}
        for source_block, cow_block in pending_copies:
            retained.setdefault(source_block.block_id, source_block)
            retained.setdefault(cow_block.block_id, cow_block)
        return copies, list(retained.values())
'''

# ---------- Phase 2: restore pristine take_* functions (dedup reversal) ----------
DEDUP_COPIES_OLD = '''        # fix-kv-dedup-retained-endpoints: managers can report the same CoW
        # pair (lockstep peers mirror one physical copy; allocation retries
        # can re-queue a pair). Emit each (src, dst) copy once.
        pending_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
        seen_pairs: set[tuple[int, int]] = set()
        for mgr in self.coordinator.single_type_managers:
            for source_block, cow_block in mgr.take_pending_cow_copies():
                pair_key = (source_block.block_id, cow_block.block_id)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                pending_copies.append((source_block, cow_block))
        copies = [
            KVCacheBlockCopy(
                src_block_id=source_block.block_id,
                dst_block_id=cow_block.block_id,
            )
            for source_block, cow_block in pending_copies
        ]
        # fix-kv-dedup-retained-endpoints: retain each unique endpoint once,
        # keyed by block_id (dict preserves first-seen order). The deferred
        # free in the scheduler decrements ref_cnt per occurrence, so a
        # duplicate here would double-free a live block; retaining once is
        # the safe direction (a stale extra bump only delays reuse).
        retained: dict[int, KVCacheBlock] = {}
        for source_block, cow_block in pending_copies:
            retained.setdefault(source_block.block_id, source_block)
            retained.setdefault(cow_block.block_id, cow_block)
        return copies, list(retained.values())'''
DEDUP_COPIES_NEW = '''        pending_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
        for mgr in self.coordinator.single_type_managers:
            pending_copies.extend(mgr.take_pending_cow_copies())
        copies = [
            KVCacheBlockCopy(
                src_block_id=source_block.block_id,
                dst_block_id=cow_block.block_id,
            )
            for source_block, cow_block in pending_copies
        ]
        retained_blocks = [block for pair in pending_copies for block in pair]
        return copies, retained_blocks'''

DEDUP_OFFLOADS_OLD = '''        # fix-kv-dedup-retained-endpoints: dedup hand-offs by
        # (req_id, group_id, block_id, boundary) across managers and pin each
        # unique block once — a duplicate entry would double-touch the pin
        # and double-free it when the request's blocks are freed,
        # underflowing ref_cnt. Producer content retention needs no second
        # slot here: the boundary state was moved onto cow_block by
        # block_pool.move_block_hashes (running CoW) / registered via
        # cache_partial_block, so the prefix-cache entry alone keeps it
        # reachable; the touch() below only guards the bytes against
        # overwrite until the connector reads them.
        #
        # TODO(fix-kv-dedup-retained-endpoints): convert this connector pin
        # toward an early unpin (block_pool exposes no unpin_blocks API;
        # unpin == free_blocks) drained only when safe — the request's
        # num_in_flight_tokens == 0 (the queued CoW copy has run) and its
        # computed boundary still matches boundary_tokens. The anchor for
        # that check does not exist at this drain: _partial_tail_pins does
        # not record boundary_tokens and no per-step hook carries the
        # Request, so pins stay released at request free (unchanged
        # behavior).
        offloads: dict[str, list[tuple[int, int, int]]] = {}
        seen_offloads: set[tuple[str, int, int, int]] = set()
        pinned_here: dict[int, KVCacheBlock] = {}
        for mgr in self.coordinator.single_type_managers:
            for (
                req_id,
                group_id,
                block,
                boundary_tokens,
            ) in mgr.take_pending_partial_tail_offloads():
                key = (req_id, group_id, block.block_id, boundary_tokens)
                if key in seen_offloads:
                    continue
                seen_offloads.add(key)
                if block.block_id not in pinned_here:
                    pinned_here[block.block_id] = block
                    self.block_pool.touch((block,))
                    self._partial_tail_pins.setdefault(req_id, []).append(block)
                offloads.setdefault(req_id, []).append(
                    (group_id, block.block_id, boundary_tokens)
                )
        return offloads'''
DEDUP_OFFLOADS_NEW = '''        offloads: dict[str, list[tuple[int, int, int]]] = {}
        for mgr in self.coordinator.single_type_managers:
            for (
                req_id,
                group_id,
                block,
                boundary_tokens,
            ) in mgr.take_pending_partial_tail_offloads():
                self.block_pool.touch((block,))
                self._partial_tail_pins.setdefault(req_id, []).append(block)
                offloads.setdefault(req_id, []).append(
                    (group_id, block.block_id, boundary_tokens)
                )
        return offloads'''

PHASE1 = [
    ("harden-helper", HARDEN_HELPER_OLD, HARDEN_HELPER_NEW),
    ("harden-raw-init", HARDEN_RAW_INIT_OLD, HARDEN_RAW_INIT_NEW),
    ("harden-raw-count", HARDEN_RAW_COUNT_OLD, HARDEN_RAW_COUNT_NEW),
    ("harden-debug-stats", HARDEN_DEBUG_OLD, HARDEN_DEBUG_NEW),
]
PHASE2 = [
    ("dedup-copies", DEDUP_COPIES_OLD, DEDUP_COPIES_NEW),
    ("dedup-offloads", DEDUP_OFFLOADS_OLD, DEDUP_OFFLOADS_NEW),
]

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
    has_old = any(m in s for m in OLD_MARKERS)
    if not has_old:
        # Pristine tree (or already reverted): verify it looks pristine.
        if ("retained_blocks = [block for pair in pending_copies for block in pair]" in s
                and "seen_pairs" not in s):
            print(f"[{MARKER}] already reverted (pristine) in {p}")
            applied += 1
            continue
        print(f"[{MARKER}] WARNING: no mod markers in {p}, skipping (unknown state)")
        skipped += 1
        continue
    # Phase 1: harden-kv removals (restores exact dedup text for phase 2).
    ok = True
    for name, old, new in PHASE1:
        _n = s.count(old)
        if _n == 0:
            print(f"[{MARKER}] WARNING: phase1 {name} anchor not found in {p}, skipping file")
            ok = False
            break
        if _n > 1:
            print(f"[{MARKER}] ERROR: phase1 {name} anchor count={_n} in {p}")
            sys.exit(4)
        s = s.replace(old, new, 1)
    if not ok:
        skipped += 1
        continue
    # Phase 2: dedup reversal to pristine.
    for name, old, new in PHASE2:
        _n = s.count(old)
        if _n == 0:
            print(f"[{MARKER}] ERROR: phase2 {name} anchor not found in {p} after phase1")
            sys.exit(4)
        if _n > 1:
            print(f"[{MARKER}] ERROR: phase2 {name} anchor count={_n} in {p}")
            sys.exit(4)
        s = s.replace(old, new, 1)
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
    # Post-conditions: neither old marker may remain.
    with open(p) as f:
        check = f.read()
    for m in OLD_MARKERS:
        if m in check:
            print(f"[{MARKER}] ERROR: old marker {m} still present in {p}")
            sys.exit(4)
    print(f"[{MARKER}] APPLIED + py_compile OK: {p}")
    applied += 1

if applied == 0:
    print(f"[{MARKER}] WARNING: nothing applied (skipped {skipped}), exiting 3")
    sys.exit(3)
print(f"[{MARKER}] complete ({applied} files, {skipped} skipped)")
PYEOF
