#!/bin/bash
# fix-kv-dedup-retained-endpoints — dedup CoW copy pairs + retained endpoints at the
# KVCacheManager drain (Kimi-K3 fork; upstream has NO dedup here — see #49675, #56794)
#
# take_kv_cache_block_copies() aggregates _pending_cow_copies across ALL
# single_type_managers and hands the scheduler a flat list of copy pairs plus
# a "retain these endpoints until the copy ran" block list. The scheduler
# defers free_blocks(retained) to the fence step (_free_cow_retained_blocks),
# which decrements ref_cnt once PER OCCURRENCE. With no dedup at aggregation,
# a pair reported by more than one manager (lockstep peers mirror one
# physical copy; allocation retries can re-queue) or an endpoint shared by
# several pairs double-decrements ref_cnt on release — a live block can hit
# zero early and be recycled under an active request (fork #807 accounting).
# The worker also re-runs identical copies.
#
# Fix (librarian sketch, applied at our coordinator drain only):
#  1) dedup copy pairs by (src_id, dst_id) across managers at aggregation;
#  2) retain unique endpoints once via a dict keyed by block_id (set
#     semantics: the list means "keep alive until fence", not "ref +1 per
#     occurrence" — under-releasing vs per-pair bumps is the safe direction:
#     a leaked ref delays reuse, a double free corrupts);
#  3) producer partial-tail retention already needs no second slot: the
#     boundary state is moved onto cow_block by block_pool.move_block_hashes
#     (running CoW) / registered via cache_partial_block, so the prefix-cache
#     entry alone keeps it reachable; the aggregation now pins each unique
#     block once instead of once per duplicate entry;
#  4) early unpin of the connector pin: block_pool has NO unpin_blocks API
#     (unpin == free_blocks) and _partial_tail_pins does not record
#     boundary_tokens, so the safe-drain anchor (num_in_flight_tokens == 0
#     and boundary match) does not exist — marked TODO, behavior unchanged.
#
# Healthy runs unaffected: without duplicates the output lists are identical
# to before (dict/set dedup preserves first-seen order).
set -euo pipefail

echo "--- Applying CoW drain dedup at KVCacheManager (fix-kv-dedup-retained-endpoints)..."

python3 << 'PYTHON_PATCH'
import os, subprocess, sys

candidates = []

# 1) Import-resolved package (what `python3 -m vllm...` actually loads).
try:
    d = subprocess.run(
        ["python3", "-c", "import vllm, os; print(os.path.dirname(vllm.__file__))"],
        capture_output=True, text=True,
    ).stdout.strip()
    if d:
        candidates.append(os.path.join(d, "v1", "core", "kv_cache_manager.py"))
except Exception:
    pass

# 2) Known install locations (source-tree and site-packages layouts).
for base in (
    "/opt/kimi-k3/vllm/vllm",
    "/opt/venv/lib/python3.12/site-packages/vllm",
    "/usr/local/lib/python3.12/dist-packages/vllm",
):
    candidates.append(os.path.join(base, "v1", "core", "kv_cache_manager.py"))

# De-duplicate by realpath, keep order.
seen, files = set(), []
for f in candidates:
    rp = os.path.realpath(f)
    if rp not in seen and os.path.isfile(f):
        seen.add(rp)
        files.append(f)

if not files:
    print("  ⚠ No kv_cache_manager.py found in any known vllm location — nothing to patch")
    sys.exit(0)

MARKER = "fix-kv-dedup-retained-endpoints"

# --- Site 1: take_kv_cache_block_copies — dedup pairs + unique retained endpoints.
OLD1 = """        pending_copies: list[tuple[KVCacheBlock, KVCacheBlock]] = []
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
        return copies, retained_blocks"""

NEW1 = """        # fix-kv-dedup-retained-endpoints: managers can report the same CoW
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
        return copies, list(retained.values())"""

# --- Site 2: take_partial_tail_offloads — pin each unique block once + TODO.
OLD2 = """        offloads: dict[str, list[tuple[int, int, int]]] = {}
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
        return offloads"""

NEW2 = """        # fix-kv-dedup-retained-endpoints: dedup hand-offs by
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
        return offloads"""

patched_any = False
for FILE in files:
    with open(FILE) as f:
        content = f.read()

    if MARKER in content:
        print(f"  Already patched: {FILE}")
        patched_any = True
        continue

    n_applied = 0
    for name, old, new in (("take_kv_cache_block_copies", OLD1, NEW1),
                           ("take_partial_tail_offloads", OLD2, NEW2)):
        if old not in content:
            print(f"  ⚠ Anchor not found ({name}) — skipping site in: {FILE}")
            probe = old.splitlines()[0].strip()
            idx = content.find(probe)
            if idx >= 0:
                print(content[max(0, idx - 200):idx + 400])
            continue
        content = content.replace(old, new, 1)
        n_applied += 1
        print(f"  ✓ Patched {name}: {FILE}")

    if n_applied:
        with open(FILE, "w") as f:
            f.write(content)
        patched_any = True

if not patched_any:
    raise SystemExit(1)
PYTHON_PATCH

# Validate every patched copy still compiles.
for CAND in \
    /opt/kimi-k3/vllm/vllm/v1/core/kv_cache_manager.py \
    /opt/venv/lib/python3.12/site-packages/vllm/v1/core/kv_cache_manager.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_manager.py
do
    if [ -f "$CAND" ] && grep -q "fix-kv-dedup-retained-endpoints" "$CAND"; then
        python3 -m py_compile "$CAND"
        echo "  ✓ py_compile OK: $CAND"
    fi
done

echo "=== fix-kv-dedup-retained-endpoints complete ==="
