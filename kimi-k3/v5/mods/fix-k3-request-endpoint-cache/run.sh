#!/usr/bin/env bash
# fix-k3-request-endpoint-cache — resume the next turn at a finished
# request's last computed token (ports upstream local-inference-lab/vllm
# PR#732, the "request-endpoint cache"; the two upstream test files are
# intentionally NOT ported — we validate live).
#
# Feature: a finished request publishes its end position — the attention
# pages and draft-window pages holding its last computed token plus one pool
# block per recurrent group that receives the normalized recurrent state —
# as a prefix-cache entry hanging off the chain hash of the last full hash
# unit below the endpoint. A later prompt that extends the same token
# sequence resumes at that token (not at the last hash boundary) through the
# ordinary partial-hit copy-on-write path.
#
# Env gating (preserved exactly from upstream; OFF by default — the mod is a
# runtime no-op unless VLLM_K3_REQUEST_ENDPOINT_CACHE=1):
#   VLLM_K3_REQUEST_ENDPOINT_CACHE                (default 0)
#   VLLM_K3_REQUEST_ENDPOINT_CACHE_MAX_ENTRIES    (default 4)
#   VLLM_K3_REQUEST_ENDPOINT_CACHE_DISABLE_FILE   (default "")
#   VLLM_K3_REQUEST_ENDPOINT_CACHE_DEBUG          (default 0)
#   VLLM_K3_REQUEST_ENDPOINT_CACHE_TORCH_COPY     (default 0)
#   VLLM_K3_REQUEST_ENDPOINT_CACHE_TORCH_COPY_FILE (default "")
# The scheduler additionally requires mamba layers + max_concurrent_batches
# <= 2 and warns+disables otherwise.
#
# Ported hunks (12 non-test files): vllm/envs.py,
# vllm/v1/core/{block_pool,kv_cache_manager,kv_cache_utils}.py,
# vllm/v1/core/sched/{output,scheduler}.py, vllm/v1/request.py,
# vllm/v1/worker/gpu/model_runner.py (v2 runner: materializes endpoints in
# finish_requests), vllm/v1/worker/gpu/model_states/interface.py (base
# materialize_request_endpoints), vllm/v1/worker/gpu/model_states/
# mamba_hybrid.py (shadow snapshot + materialization),
# vllm/v1/worker/gpu_model_runner.py (v1 runner: warns, cannot materialize),
# vllm/v1/worker/mamba_utils.py (address-based copy refactor, shadow
# snapshot kernel, materialize kernel + torch reference, context shadow).
#
# Two bugs fixed during the port (from the PR's CodeRabbit review, unfixed
# upstream) — search "bug (a)" / "bug (b)":
#   (a) scheduler.py: endpoint_final_step is guarded on observed_spec_decode
#       so non-speculative requests publish the neutral (0, 0) draft geometry
#       instead of reading num_draft_tokens/num_accepted before assignment.
#   (b) endpoint entries are WITHDRAWN whenever the worker skips
#       materialization (shadow unavailable / unmapped slot / bad block
#       counts / non-recurrent model state / v1 runner / uninitialized
#       context), via a withdrawal callback registry in kv_cache_utils; a
#       later prompt can never resume from unwritten recurrent state.
#       In-process workers get full withdrawal; under a process-separated
#       worker (ray actors) the registry is empty in the worker's process
#       and the call degrades to a logged no-op.
#
# Adaptations vs the upstream diff (our pin differs; every anchor was
# verified against the live /opt/kimi-k3 tree and the venv tree):
#   - scheduler.py: upstream's `from vllm import envs` import hunk is NOT
#     ported (our pin already imports envs at module top).
#   - mamba_hybrid.py: the _get_mamba_group_info hunk anchors on the tail
#     shared by both trees (the live tree carries a baked fix-k3-r29-mamba-
#     debug block between the assert and the assignments; the venv tree has
#     an assert instead).
#   - mamba_utils.py: _copy_mamba_state_addrs keeps OUR pin's memmove-
#     hardened copy body (token-wise DS/SD paths, same-block left-shift
#     safety) instead of upstream's simpler pre-hardening body; only the
#     parameterization (block-table columns -> resolved addresses) is taken
#     from upstream.
#   - block_pool.py reset_prefix_cache additionally clears the withdrawal
#     registry (bug (b) support).
#
# All-or-nothing per file: exit 3 = no anchors found / all copies skipped,
# exit 4 = ambiguous anchor (count>1), exit 5 = patched text fails
# compile/py_compile. Idempotent via the marker string.

set -euo pipefail

MARKER="fix-k3-request-endpoint-cache"
TARGET="vllm/envs.py vllm/v1/core/block_pool.py vllm/v1/core/kv_cache_manager.py vllm/v1/core/kv_cache_utils.py vllm/v1/core/sched/output.py vllm/v1/core/sched/scheduler.py vllm/v1/request.py vllm/v1/worker/gpu/model_runner.py vllm/v1/worker/gpu/model_states/interface.py vllm/v1/worker/gpu/model_states/mamba_hybrid.py vllm/v1/worker/gpu_model_runner.py vllm/v1/worker/mamba_utils.py"

already() { grep -q "$MARKER" "$1" 2>/dev/null; }

python3 - <<'PYEOF'
import os, subprocess, sys, py_compile

MARKER = "fix-k3-request-endpoint-cache"
ROOTS = os.environ.get(
    "MOD_FIND_ROOTS", "/opt/kimi-k3 /opt/venv /usr/local/lib").split()

# (relative path, [(site name, anchor, replacement), ...]) — all-or-nothing
# per file: every anchor must match exactly once before anything is written.
HUNKS = [
    (
        "vllm/envs.py",
        [
            (
                "env field declarations",
                '''    VLLM_USE_DIRECT_DCP_KV_GATHER: bool | None = None
    VLLM_DEEP_GEMM_WARMUP: Literal[
''',
                '''    VLLM_USE_DIRECT_DCP_KV_GATHER: bool | None = None
    # fix-k3-request-endpoint-cache: request-endpoint cache env flags
    # (marker comment; the flags themselves are verbatim from PR#732).
    VLLM_K3_REQUEST_ENDPOINT_CACHE: bool = False
    VLLM_K3_REQUEST_ENDPOINT_CACHE_MAX_ENTRIES: int = 4
    VLLM_K3_REQUEST_ENDPOINT_CACHE_DISABLE_FILE: str = ""
    VLLM_K3_REQUEST_ENDPOINT_CACHE_DEBUG: bool = False
    VLLM_K3_REQUEST_ENDPOINT_CACHE_TORCH_COPY: bool = False
    VLLM_K3_REQUEST_ENDPOINT_CACHE_TORCH_COPY_FILE: str = ""
    VLLM_DEEP_GEMM_WARMUP: Literal[
''',
            ),
            (
                "env parser lambdas",
                '''    "VLLM_USE_DIRECT_DCP_KV_GATHER": lambda: maybe_convert_bool(
        os.getenv("VLLM_USE_DIRECT_DCP_KV_GATHER")
    ),
    # Whether to enable dual cuda streams for LoRA computation
''',
                '''    "VLLM_USE_DIRECT_DCP_KV_GATHER": lambda: maybe_convert_bool(
        os.getenv("VLLM_USE_DIRECT_DCP_KV_GATHER")
    ),
    # Publish every finished request's end state (attention pages, draft
    # pages and a normalized recurrent state) under a request-endpoint
    # prefix-cache entry, so the next turn that extends the same sequence
    # resumes at the last computed token instead of the last hash boundary.
    # Needs a hybrid (recurrent + attention) cache in align mode and at
    # most two concurrent batches; the scheduler disables it otherwise.
    "VLLM_K3_REQUEST_ENDPOINT_CACHE": lambda: bool(
        int(os.getenv("VLLM_K3_REQUEST_ENDPOINT_CACHE", "0"))
    ),
    # Upper bound on live request-endpoint entries. Each entry pins one
    # pool block per recurrent cache group for its state, so the bound caps
    # the KV capacity the endpoint cache can hold back from attention pages.
    "VLLM_K3_REQUEST_ENDPOINT_CACHE_MAX_ENTRIES": lambda: int(
        os.getenv("VLLM_K3_REQUEST_ENDPOINT_CACHE_MAX_ENTRIES", "4")
    ),
    # File whose presence switches request-endpoint registration and lookup
    # off without an engine restart.
    "VLLM_K3_REQUEST_ENDPOINT_CACHE_DISABLE_FILE": lambda: os.getenv(
        "VLLM_K3_REQUEST_ENDPOINT_CACHE_DISABLE_FILE", ""
    ),
    # Log endpoint registration/hit geometry and, on the worker, checksums of
    # the recurrent-state pages read and written by the endpoint copies.
    "VLLM_K3_REQUEST_ENDPOINT_CACHE_DEBUG": lambda: bool(
        int(os.getenv("VLLM_K3_REQUEST_ENDPOINT_CACHE_DEBUG", "0"))
    ),
    # Materialize request-endpoint recurrent states with per-state torch
    # copies instead of the fused Triton kernel (same results; a fallback
    # and reference path). The file, when named, enables it while present.
    "VLLM_K3_REQUEST_ENDPOINT_CACHE_TORCH_COPY": lambda: bool(
        int(os.getenv("VLLM_K3_REQUEST_ENDPOINT_CACHE_TORCH_COPY", "0"))
    ),
    "VLLM_K3_REQUEST_ENDPOINT_CACHE_TORCH_COPY_FILE": lambda: os.getenv(
        "VLLM_K3_REQUEST_ENDPOINT_CACHE_TORCH_COPY_FILE", ""
    ),
    # Whether to enable dual cuda streams for LoRA computation
''',
            ),
        ],
    ),
    (
        "vllm/v1/core/kv_cache_utils.py",
        [
            (
                "MambaEndpointStateCopy + endpoint helpers + withdrawal registry",
                '''class KVCacheBlockCopy(NamedTuple):
    src_block_id: int
    dst_block_id: int


class FreeKVCacheBlockQueue:
''',
                '''class KVCacheBlockCopy(NamedTuple):
    src_block_id: int
    dst_block_id: int


class MambaEndpointStateCopy(NamedTuple):
    """Materialize a finished request's recurrent state into durable blocks.

    ``dst_block_ids`` holds one pool block per recurrent cache group, in
    ascending group-id order. A block id names the same physical page in
    every group (each raw KV tensor is shared by one layer of every group),
    so groups must not share a destination. Every block receives the state
    after the request's endpoint token. When ``from_shadow`` is set the
    state is read from the worker's per-request shadow pages (written before
    the request's final in-flight step could overwrite its state slots): the
    unshifted conv window shifted by ``token_bias`` and temporal shadow slot
    ``token_bias``. Otherwise it is read from the request's own blocks: the
    temporal state of ``temporal_src_block_ids`` and the convolution window
    of ``conv_src_block_ids`` shifted by ``token_bias``, one entry per
    recurrent cache group in ascending group-id order.
    """

    req_id: str
    dst_block_ids: tuple[int, ...]
    from_shadow: bool
    conv_src_block_ids: tuple[int, ...]
    temporal_src_block_ids: tuple[int, ...]
    token_bias: int


# Prefix-cache keys of request-endpoint entries reuse the block hash map with
# the group id lifted into this namespace, so a page can be reachable both at
# its hash boundaries and at a request's end position.
ENDPOINT_GROUP_ID_OFFSET = 1 << 24


def request_endpoint_cache_enabled() -> bool:
    """Whether finished requests publish request-endpoint cache entries.

    Gated by ``VLLM_K3_REQUEST_ENDPOINT_CACHE`` and switched off, without a
    restart, by the file named in
    ``VLLM_K3_REQUEST_ENDPOINT_CACHE_DISABLE_FILE``.

    An entry restores the end position of every cache group, a speculative
    draft's sliding window included, so a deployment that hides restored KV
    from its draft group must keep the flag off.
    """
    if not envs.VLLM_K3_REQUEST_ENDPOINT_CACHE:
        return False
    disable_file = envs.VLLM_K3_REQUEST_ENDPOINT_CACHE_DISABLE_FILE
    return not (disable_file and os.path.exists(disable_file))


def request_endpoint_cache_debug() -> bool:
    """Whether endpoint registration, hits and worker copies are logged."""
    return bool(envs.VLLM_K3_REQUEST_ENDPOINT_CACHE_DEBUG)


def request_endpoint_cache_torch_copy() -> bool:
    """Whether endpoint states are materialized by torch copies (env flag or
    the presence of the named file) instead of the fused kernel."""
    if envs.VLLM_K3_REQUEST_ENDPOINT_CACHE_TORCH_COPY:
        return True
    path = envs.VLLM_K3_REQUEST_ENDPOINT_CACHE_TORCH_COPY_FILE
    return bool(path) and os.path.exists(path)


def request_endpoint_cache_max_entries() -> int:
    """Upper bound on live request-endpoint entries (each pins one block)."""
    return max(int(envs.VLLM_K3_REQUEST_ENDPOINT_CACHE_MAX_ENTRIES), 0)


# fix-k3-request-endpoint-cache (bug (b), flagged by the PR's review but
# unfixed upstream): a published endpoint entry must never survive a SKIPPED
# worker materialization (shadow unavailable, unmapped request slot, or bad
# block counts) — otherwise a later prompt resumes from unwritten recurrent
# state. The scheduler arms a withdrawal callback per published request; the
# worker calls withdraw_request_endpoints() on every skip path and
# confirm_endpoint_materialized() once the copies are enqueued. The
# callbacks are plain in-process closures: they cover the in-process worker
# (solo / mp executors); under a process-separated worker (e.g. ray actors)
# this registry is empty in the worker's process and the calls degrade to a
# no-op — the skip is still logged loudly by the worker either way.
_ENDPOINT_WITHDRAW_CALLBACKS: dict[str, Callable[[], None]] = {}


def register_endpoint_withdrawal(req_id: str, callback: Callable[[], None]) -> None:
    """Arm ``callback`` (drop the req's endpoint entry) for a worker skip."""
    _ENDPOINT_WITHDRAW_CALLBACKS[req_id] = callback


def withdraw_request_endpoints(req_id: str) -> None:
    """Worker skip path: drop the req's published endpoint entry (if any)."""
    callback = _ENDPOINT_WITHDRAW_CALLBACKS.pop(req_id, None)
    if callback is not None:
        try:
            callback()
        except Exception:
            logger.exception(
                "request-endpoint withdrawal failed for %s", req_id
            )


def confirm_endpoint_materialized(req_id: str) -> None:
    """Worker success path: forget the withdrawal callback (entry stays)."""
    _ENDPOINT_WITHDRAW_CALLBACKS.pop(req_id, None)


def clear_endpoint_withdrawals() -> None:
    """Drop every armed withdrawal callback (prefix-cache reset)."""
    _ENDPOINT_WITHDRAW_CALLBACKS.clear()


class FreeKVCacheBlockQueue:
''',
            ),
        ],
    ),
    (
        "vllm/v1/core/block_pool.py",
        [
            (
                "collections import (deque, Iterator)",
                '''from collections.abc import Iterable, Sequence
from typing import Any
''',
                '''from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from typing import Any
''',
            ),
            (
                "kv_cache_utils import (ENDPOINT_GROUP_ID_OFFSET, clear_endpoint_withdrawals)",
                '''from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    generate_block_hash_extra_keys,
''',
                '''from vllm.v1.core.kv_cache_utils import (
    ENDPOINT_GROUP_ID_OFFSET,
    BlockHash,
    BlockHashWithGroupId,
    ExternalBlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    clear_endpoint_withdrawals,
    generate_block_hash_extra_keys,
''',
            ),
            (
                "module-level EndpointEntry + helpers",
                '''        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


class BlockPool:
''',
                '''        raise AssertionError(f"Invalid KV cache block type {type(blocks)}")


def _iter_group_blocks(
    group_blocks: dict[int, tuple[int, list[KVCacheBlock]]],
) -> Iterator[tuple[int, KVCacheBlock]]:
    for group_id, (_, blocks) in group_blocks.items():
        for block in blocks:
            yield group_id, block


class EndpointEntry:
    """A finished request's end position, reachable by a later prefix.

    ``num_tokens`` tokens of the producing request are restorable. For each
    KV cache group ``group_blocks[group_id] = (first_block_index, blocks)``
    lists the blocks that must be handed to a consumer starting at block
    index ``first_block_index``: the page holding position ``num_tokens - 1``
    for attention groups, that page plus the window before it for
    sliding-window groups, and for every recurrent group the one pool block
    holding the state after ``num_tokens`` tokens. The entry hangs off
    ``parent_hash``, the chain hash of the last full hash unit below
    ``num_tokens``; ``tail_tokens`` and ``extra_keys`` describe the tokens
    between that unit and ``num_tokens`` and must match the consumer's.
    """

    __slots__ = (
        "parent_hash",
        "num_tokens",
        "tail_tokens",
        "extra_keys",
        "group_blocks",
    )

    def __init__(
        self,
        parent_hash: BlockHash,
        num_tokens: int,
        tail_tokens: tuple[int, ...],
        extra_keys: tuple[Any, ...] | None,
        group_blocks: dict[int, tuple[int, list[KVCacheBlock]]],
    ) -> None:
        self.parent_hash = parent_hash
        self.num_tokens = num_tokens
        self.tail_tokens = tail_tokens
        self.extra_keys = extra_keys
        self.group_blocks = group_blocks

    def blocks(self) -> Iterator[tuple[int, KVCacheBlock]]:
        for group_id, (_, blocks) in self.group_blocks.items():
            for block in blocks:
                yield group_id, block


# Endpoint entries kept per parent hash; older entries are dropped first.
_MAX_ENDPOINTS_PER_PARENT = 2


class BlockPool:
''',
            ),
            (
                "__init__ endpoint bookkeeping",
                '''        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
        self.cached_block_hashes_by_block: dict[int, set[BlockHashWithGroupId]] = {}

        # To represent a placeholder block with block_id=0.
''',
                '''        self.cached_block_hash_to_block: BlockHashToBlockMap = BlockHashToBlockMap()
        self.cached_block_hashes_by_block: dict[int, set[BlockHashWithGroupId]] = {}

        # Request-endpoint entries by parent hash, the entries each block takes
        # part in (an entry is dropped when any of its blocks is reused), and
        # their registration order for the global cap.
        self.endpoint_entries: dict[BlockHash, list[EndpointEntry]] = {}
        self._endpoints_by_block: dict[int, list[EndpointEntry]] = {}
        self._endpoint_order: deque[EndpointEntry] = deque()

        # To represent a placeholder block with block_id=0.
''',
            ),
            (
                "endpoint pool methods",
                '''        # Each hash_block_size hash chains over its full prefix, so the partial
        # entry for any group block size is the hash at that prefix boundary.
        return request.block_hashes[num_hash_blocks - 1]

    def _get_partial_block_parent_hash_and_start(
''',
                '''        # Each hash_block_size hash chains over its full prefix, so the partial
        # entry for any group block size is the hash at that prefix boundary.
        return request.block_hashes[num_hash_blocks - 1]

    def cache_endpoint(
        self,
        parent_hash: BlockHash,
        num_tokens: int,
        tail_tokens: tuple[int, ...],
        extra_keys: tuple[Any, ...] | None,
        group_blocks: dict[int, tuple[int, list[KVCacheBlock]]],
        max_entries: int,
    ) -> EndpointEntry | None:
        """Register a request-endpoint entry over the given per-group blocks.

        Every block gains an endpoint key (the parent hash in the endpoint
        group-id namespace) so it is retained like a cached block and so
        eviction, reset and promotion drop the entry together with the block.
        Older entries beyond ``max_entries`` (globally) or beyond the
        per-parent cap are dropped. Returns ``None`` when caching is disabled,
        ``max_entries`` is not positive, or a block is the null block.
        """
        if not self.enable_caching or not group_blocks or max_entries <= 0:
            return None
        if any(block.is_null for _, block in _iter_group_blocks(group_blocks)):
            return None
        entry = EndpointEntry(
            parent_hash, num_tokens, tail_tokens, extra_keys, group_blocks
        )
        for group_id, block in entry.blocks():
            key = make_block_hash_with_group_id(
                parent_hash, ENDPOINT_GROUP_ID_OFFSET + group_id
            )
            self._insert_block_hash(key, block, num_tokens=num_tokens)
            refs = self._endpoints_by_block.setdefault(block.block_id, [])
            if entry not in refs:
                refs.append(entry)
        entries = self.endpoint_entries.setdefault(parent_hash, [])
        entries.append(entry)
        self._endpoint_order.append(entry)
        while len(entries) > _MAX_ENDPOINTS_PER_PARENT:
            self._drop_endpoint_entry(entries[0])
        while len(self._endpoint_order) > max_entries:
            self._drop_endpoint_entry(self._endpoint_order[0])
        return entry

    def find_endpoints(self, parent_hash: BlockHash) -> list[EndpointEntry]:
        """Return the endpoint entries registered under ``parent_hash``,
        longest first."""
        entries = self.endpoint_entries.get(parent_hash)
        if not entries:
            return []
        return sorted(entries, key=lambda entry: entry.num_tokens, reverse=True)

    @property
    def num_endpoint_entries(self) -> int:
        return len(self._endpoint_order)

    def _drop_endpoint_entry(self, entry: EndpointEntry) -> None:
        """Forget ``entry`` and remove its endpoint keys from its blocks (a key
        shared with a surviving entry of the same parent stays)."""
        parent_hash = entry.parent_hash
        entries = self.endpoint_entries.get(parent_hash)
        if entries is not None:
            if entry in entries:
                entries.remove(entry)
            if not entries:
                del self.endpoint_entries[parent_hash]
        if entry in self._endpoint_order:
            self._endpoint_order.remove(entry)
        for group_id, block in entry.blocks():
            refs = self._endpoints_by_block.get(block.block_id)
            if refs is None:
                continue
            refs[:] = [ref for ref in refs if ref is not entry]
            if not refs:
                del self._endpoints_by_block[block.block_id]
            if any(
                ref.parent_hash == parent_hash
                and any(
                    peer is block for peer in ref.group_blocks.get(group_id, (0, ()))[1]
                )
                for ref in refs
            ):
                continue
            self._remove_endpoint_key(
                block,
                make_block_hash_with_group_id(
                    parent_hash, ENDPOINT_GROUP_ID_OFFSET + group_id
                ),
            )

    def _remove_endpoint_key(
        self, block: KVCacheBlock, key: BlockHashWithGroupId
    ) -> None:
        if block.block_hash == key:
            # The endpoint key was the block's primary hash: promote one of
            # its secondary keys (if any) so the block stays a cached block.
            secondary = self.cached_block_hashes_by_block.pop(block.block_id, set())
            self.cached_block_hash_to_block.pop(key, block.block_id)
            block.reset_hash()
            for other in secondary:
                self.cached_block_hash_to_block.pop(other, block.block_id)
                self._insert_block_hash(other, block, num_tokens=None)
            return
        remaining = self.cached_block_hashes_by_block.get(block.block_id)
        if remaining is not None:
            remaining.discard(key)
            if not remaining:
                del self.cached_block_hashes_by_block[block.block_id]
        self.cached_block_hash_to_block.pop(key, block.block_id)

    def _drop_endpoints_of_block(self, block: KVCacheBlock) -> None:
        """Forget every endpoint entry that used ``block`` (block reuse)."""
        refs = self._endpoints_by_block.get(block.block_id)
        if not refs:
            return
        for entry in list(refs):
            self._drop_endpoint_entry(entry)

    def _get_partial_block_parent_hash_and_start(
''',
            ),
            (
                "_remove_cached_block_hashes endpoint keys + entry drop",
                '''        removed_hashes: list[BlockHashWithGroupId] = []
        for block_hash in block_hashes:
            if (
                self.cached_block_hash_to_block.pop(block_hash, block.block_id)
                is not None
            ):
                removed_hashes.append(block_hash)
        block.reset_hash()
        return removed_hashes
''',
                '''        removed_hashes: list[BlockHashWithGroupId] = []
        for block_hash in block_hashes:
            if (
                self.cached_block_hash_to_block.pop(block_hash, block.block_id)
                is not None
            ) and get_group_id(block_hash) < ENDPOINT_GROUP_ID_OFFSET:
                # Endpoint keys never emit events; their entries are dropped
                # with the block below.
                removed_hashes.append(block_hash)
        block.reset_hash()
        self._drop_endpoints_of_block(block)
        return removed_hashes
''',
            ),
            (
                "reset_prefix_cache clears endpoints (+ bug (b) callbacks)",
                '''        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        self.cached_block_hashes_by_block.clear()

        # Remove all hashes from all blocks.
''',
                '''        # Remove all hashes so that no new blocks will hit.
        self.cached_block_hash_to_block = BlockHashToBlockMap()
        self.cached_block_hashes_by_block.clear()
        self.endpoint_entries.clear()
        self._endpoints_by_block.clear()
        self._endpoint_order.clear()
        # fix-k3-request-endpoint-cache (bug (b)): drop any armed withdrawal
        # callbacks along with the entries they protected.
        clear_endpoint_withdrawals()

        # Remove all hashes from all blocks.
''',
            ),
        ],
    ),
    (
        "vllm/v1/core/kv_cache_manager.py",
        [
            (
                "imports (endpoint helpers + manager classes)",
                '''from vllm.v1.core.kv_cache_utils import KVCacheBlock, KVCacheBlockCopy
from vllm.v1.kv_cache_interface import (
''',
                '''from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    KVCacheBlockCopy,
    MambaEndpointStateCopy,
    generate_block_hash_extra_keys,
    register_endpoint_withdrawal,
    request_endpoint_cache_debug,
    request_endpoint_cache_enabled,
    request_endpoint_cache_max_entries,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
    RSWAManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import (
''',
            ),
            (
                "__init__ endpoint copy queues",
                '''        # Off-table cow blocks handed to a KV connector for partial-tail
        # offload; pinned until the request's blocks are freed.
        self._partial_tail_pins: dict[str, list[KVCacheBlock]] = {}

    @property
    def usage(self) -> float:
''',
                '''        # Off-table cow blocks handed to a KV connector for partial-tail
        # offload; pinned until the request's blocks are freed.
        self._partial_tail_pins: dict[str, list[KVCacheBlock]] = {}

        # Request-endpoint cache: state copies for the worker and the blocks
        # they read or write, retained until the step that runs them.
        self._pending_endpoint_copies: list[MambaEndpointStateCopy] = []
        self._endpoint_retained_blocks: list[KVCacheBlock] = []

    @property
    def usage(self) -> float:
''',
            ),
            (
                "get_computed_blocks endpoint hit",
                '''        computed_blocks, num_new_computed_tokens, num_uncached = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        # When kv_cache_report_mode is "full", emit BlockStored events
''',
                '''        computed_blocks, num_new_computed_tokens, num_uncached = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )
        endpoint_hit = self._find_endpoint_hit(
            request, max_cache_hit_length, num_new_computed_tokens
        )
        if endpoint_hit is not None:
            computed_blocks, num_new_computed_tokens = endpoint_hit
            num_uncached = 0

        # When kv_cache_report_mode is "full", emit BlockStored events
''',
            ),
            (
                "get_computed_blocks events skip endpoint hits",
                '''        if (
            num_new_computed_tokens > 0
            and self.enable_kv_cache_events
            and getattr(request, "kv_cache_report_mode", "incremental") == "full"
        ):
''',
                '''        if (
            num_new_computed_tokens > 0
            and endpoint_hit is None
            and self.enable_kv_cache_events
            and getattr(request, "kv_cache_report_mode", "incremental") == "full"
        ):
''',
            ),
            (
                "get_computed_blocks_for_connector endpoint hit",
                '''        num_local = per_group_hits[fa_group_id]
        blocks = self.create_kv_cache_blocks(computed)
''',
                '''        num_local = per_group_hits[fa_group_id]
        endpoint_hit = self._find_endpoint_hit(
            request, request.num_tokens - 1, num_local
        )
        if endpoint_hit is not None:
            endpoint_blocks, num_endpoint = endpoint_hit
            return self.create_kv_cache_blocks(endpoint_blocks), num_endpoint, 0, False
        blocks = self.create_kv_cache_blocks(computed)
''',
            ),
            (
                "endpoint cache methods (cache_endpoint / take / find / hit blocks)",
                '''                blocks = mgr.req_to_blocks[request_id]
                mgr.new_block_ids.extend(blk.block_id for blk in blocks[start_idx:])

    def take_kv_cache_block_copies(
''',
                '''                blocks = mgr.req_to_blocks[request_id]
                mgr.new_block_ids.extend(blk.block_id for blk in blocks[start_idx:])

    # ------------------------------------------------------------------
    # Request-endpoint cache
    #
    # A finished request publishes its end position: the attention pages and
    # draft-window blocks holding its last computed token, plus one pool block
    # that receives every recurrent group's state after that token. A later
    # prompt that extends the same token sequence resumes at that position
    # instead of at the last hash boundary.
    # ------------------------------------------------------------------

    def cache_endpoint(
        self,
        request: Request,
        final_step: tuple[int, int],
        in_flight: bool,
    ) -> bool:
        """Publish a finished request's end position as a prefix-cache entry.

        Args:
            request: The finished request; its blocks must still be held.
            final_step: (scheduled draft tokens, accepted draft tokens) of the
                step that produced the request's last computed token; locates
                the recurrent-state slot of the endpoint within the request's
                blocks.
            in_flight: Whether a later step of the request is still executing.
                That step overwrites the request's state slots, so the
                recurrent state is then taken from the worker's shadow pages
                (written before the step ran) instead of the blocks.

        Returns:
            True if an entry was registered. The recurrent state is written by
            the worker from the copy record drained by
            ``take_mamba_endpoint_copies``; the destination block and the
            source blocks stay retained until that step completes.
        """
        if not self.enable_caching or not request_endpoint_cache_enabled():
            return False
        max_entries = request_endpoint_cache_max_entries()
        if max_entries <= 0:
            return False
        unit = self.block_pool.hash_block_size
        # The endpoint is the last token with KV. A stop inside the accepted
        # tokens of the final step trims the sequence below the committed
        # position; the recurrent state of every accepted row is kept, so the
        # endpoint may sit up to ``num_accepted`` rows before it.
        num_tokens = request.num_tokens - 1
        committed = request.num_computed_tokens - request.num_in_flight_tokens
        num_draft_tokens, num_accepted = final_step
        num_before = committed - num_accepted - 1
        accepted_at_endpoint = num_tokens - num_before - 1
        if (
            num_tokens < unit
            or num_tokens > committed
            or not 0 <= accepted_at_endpoint <= num_accepted
        ):
            return False
        num_units = num_tokens // unit
        if num_units > len(request.block_hashes):
            return False
        parent_hash = request.block_hashes[num_units - 1]
        tail_start = num_units * unit
        tail_tokens = tuple(request.all_token_ids[tail_start:num_tokens])
        extra_keys, _ = generate_block_hash_extra_keys(
            request, tail_start, num_tokens, 0
        )

        req_id = request.request_id
        group_blocks: dict[int, tuple[int, list[KVCacheBlock]]] = {}
        mamba_managers: list[MambaManager] = []
        mamba_group_ids: list[int] = []
        for group_id, (manager, group) in enumerate(
            zip(
                self.coordinator.single_type_managers,
                self.kv_cache_config.kv_cache_groups,
                strict=True,
            )
        ):
            req_blocks = manager.req_to_blocks.get(req_id)
            if not req_blocks:
                return False
            if isinstance(manager, MambaManager):
                if manager.mamba_cache_mode != "align":
                    return False
                mamba_managers.append(manager)
                mamba_group_ids.append(group_id)
                continue
            block_size = manager.block_size
            last_idx = (num_tokens - 1) // block_size
            if isinstance(manager, SlidingWindowManager):
                window = get_kv_cache_spec_sliding_window(group.kv_cache_spec)
                if window is None:
                    return False
                first_idx = max(0, num_tokens - window + 1) // block_size
            elif isinstance(manager, FullAttentionManager) and not isinstance(
                manager, RSWAManager
            ):
                first_idx = last_idx
            else:
                return False
            if last_idx >= len(req_blocks):
                return False
            blocks = list(req_blocks[first_idx : last_idx + 1])
            if any(block.is_null for block in blocks):
                return False
            group_blocks[group_id] = (first_idx, blocks)
        if not mamba_managers:
            return False

        mamba_block_size = mamba_managers[0].block_size
        retained: list[KVCacheBlock] = []
        conv_src_ids: list[int] = []
        temporal_src_ids: list[int] = []
        # The final step scheduled ``num_draft_tokens + 1`` rows from the
        # committed position ``num_before`` and wrote row ``j`` of its state
        # into column ``base + j``; the conv window lives in ``base`` and is
        # read with the accepted count as its shift. When acceptance landed
        # exactly on a block boundary inside the running column the post-step
        # kernel normalized the committed state into slot 0 and shifted the
        # window in place, which leaves no state for an earlier endpoint.
        base = (num_before + num_draft_tokens) // mamba_block_size
        aligned = committed // mamba_block_size * mamba_block_size
        normalized = (
            aligned >= num_before + 1 and aligned // mamba_block_size - 1 == base
        )
        if normalized and accepted_at_endpoint != num_accepted:
            return False
        token_bias = 0 if normalized else accepted_at_endpoint
        if not in_flight:
            temporal_idx = base + token_bias
            for manager in mamba_managers:
                req_blocks = manager.req_to_blocks[req_id]
                if temporal_idx >= len(req_blocks) or base >= len(req_blocks):
                    return False
                conv_src = req_blocks[base]
                temporal_src = req_blocks[temporal_idx]
                if conv_src.is_null or temporal_src.is_null:
                    return False
                conv_src_ids.append(conv_src.block_id)
                temporal_src_ids.append(temporal_src.block_id)
                retained.append(conv_src)
                retained.append(temporal_src)

        # One durable block per recurrent group: a block id names the same
        # physical page in every group (each raw KV tensor is shared by one
        # layer of every group), so groups must not share a block. Keep one
        # block spare so publishing never takes the pool's last block.
        num_dst = len(mamba_managers)
        if self.block_pool.get_num_free_blocks() < num_dst + 1:
            return False
        dsts = self.block_pool.get_new_blocks(num_dst)
        state_idx = (num_tokens - 1) // mamba_block_size
        for group_id, dst in zip(mamba_group_ids, dsts):
            group_blocks[group_id] = (state_idx, [dst])
        entry = self.block_pool.cache_endpoint(
            parent_hash, num_tokens, tail_tokens, extra_keys, group_blocks, max_entries
        )
        if entry is None:
            self.block_pool.free_blocks(dsts)
            return False
        # fix-k3-request-endpoint-cache (bug (b), flagged by the PR's review
        # but unfixed upstream): arm a withdrawal so this entry is dropped if
        # the worker SKIPS the copy's materialization (shadow unavailable,
        # unmapped slot, bad block counts). The worker calls
        # withdraw_request_endpoints()/confirm_endpoint_materialized() from
        # its materialization path; with a process-separated worker (e.g.
        # ray actors) the registry is empty in the worker's process and the
        # withdrawal degrades to a no-op — the skip is still logged there.
        register_endpoint_withdrawal(
            req_id, lambda: self.block_pool._drop_endpoint_entry(entry)
        )
        self.block_pool.touch(retained)
        self._pending_endpoint_copies.append(
            MambaEndpointStateCopy(
                req_id=req_id,
                dst_block_ids=tuple(dst.block_id for dst in dsts),
                from_shadow=in_flight,
                conv_src_block_ids=tuple(conv_src_ids),
                temporal_src_block_ids=tuple(temporal_src_ids),
                token_bias=token_bias,
            )
        )
        self._endpoint_retained_blocks.extend(dsts)
        self._endpoint_retained_blocks.extend(retained)
        if request_endpoint_cache_debug():
            logger.info(
                "[endpoint] register req=%s L=%d committed=%d drafts=%d accepted=%d "
                "accepted_at_endpoint=%d base=%d normalized=%s in_flight=%s "
                "bias=%d dst=%s conv_src=%s temporal_src=%s groups=%s",
                req_id,
                num_tokens,
                committed,
                num_draft_tokens,
                num_accepted,
                accepted_at_endpoint,
                base,
                normalized,
                in_flight,
                token_bias,
                [dst.block_id for dst in dsts],
                conv_src_ids,
                temporal_src_ids,
                {
                    gid: (first, [b.block_id for b in blocks])
                    for gid, (first, blocks) in group_blocks.items()
                },
            )
        return True

    def take_mamba_endpoint_copies(
        self,
    ) -> tuple[list[MambaEndpointStateCopy], list[KVCacheBlock]]:
        """Drain endpoint state copies and the blocks retained for them."""
        copies = self._pending_endpoint_copies
        retained = self._endpoint_retained_blocks
        self._pending_endpoint_copies = []
        self._endpoint_retained_blocks = []
        return copies, retained

    def _find_endpoint_hit(
        self,
        request: Request,
        max_cache_hit_length: int,
        min_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int] | None:
        """Find the longest request-endpoint entry the request can resume from.

        Entries hang off the chain hash of their last full hash unit, so the
        candidate parents are the request's hashes from the longest one whose
        endpoint could fit under ``max_cache_hit_length`` down to the one whose
        endpoint could still exceed ``min_hit_length``. An entry matches when
        its tail tokens and extra keys equal the request's, and every
        attention page before its endpoint page is still cached.

        Returns:
            Per-group block lists and the endpoint length, or ``None``.
        """
        if not request_endpoint_cache_enabled():
            return None
        if self.block_pool.num_endpoint_entries == 0:
            return None
        unit = self.block_pool.hash_block_size
        block_hashes = request.block_hashes
        token_ids = request.all_token_ids
        highest = min(len(block_hashes), max_cache_hit_length // unit) - 1
        lowest = max(0, min_hit_length // unit - 1)
        for parent_idx in range(highest, lowest - 1, -1):
            entries = self.block_pool.find_endpoints(block_hashes[parent_idx])
            if not entries:
                continue
            tail_start = (parent_idx + 1) * unit
            for entry in entries:
                num_tokens = entry.num_tokens
                if num_tokens > max_cache_hit_length or num_tokens <= min_hit_length:
                    continue
                if tuple(token_ids[tail_start:num_tokens]) != entry.tail_tokens:
                    continue
                extra_keys, _ = generate_block_hash_extra_keys(
                    request, tail_start, num_tokens, 0
                )
                if extra_keys != entry.extra_keys:
                    continue
                blocks = self._endpoint_hit_blocks(entry, block_hashes)
                if blocks is not None:
                    if request_endpoint_cache_debug():
                        logger.info(
                            "[endpoint] hit req=%s L=%d parent_idx=%d prompt=%d "
                            "aligned_hit=%d blocks=%s",
                            request.request_id,
                            num_tokens,
                            parent_idx,
                            request.num_tokens,
                            min_hit_length,
                            [[b.block_id for b in group] for group in blocks],
                        )
                    return blocks, num_tokens
        return None

    def _endpoint_hit_blocks(
        self, entry, block_hashes
    ) -> tuple[list[KVCacheBlock], ...] | None:
        unit = self.block_pool.hash_block_size
        null_block = self.block_pool.null_block
        result: list[list[KVCacheBlock]] = []
        for group_id, manager in enumerate(self.coordinator.single_type_managers):
            group_entry = entry.group_blocks.get(group_id)
            if group_entry is None:
                return None
            first_idx, blocks = group_entry
            if isinstance(manager, (MambaManager, SlidingWindowManager)):
                result.append([null_block] * first_idx + list(blocks))
                continue
            # Full attention: every page before the endpoint page must still
            # be cached under the request's chain.
            scale = manager.block_size // unit
            chain: list[KVCacheBlock] = []
            for page_idx in range(first_idx):
                hash_idx = (page_idx + 1) * scale - 1
                if hash_idx >= len(block_hashes):
                    return None
                cached = self.block_pool.get_cached_block(
                    block_hashes[hash_idx], [group_id]
                )
                if not cached:
                    return None
                chain.append(cached[0])
            result.append(chain + list(blocks))
        return tuple(result)

    def take_kv_cache_block_copies(
''',
            ),
        ],
    ),
    (
        "vllm/v1/core/sched/output.py",
        [
            (
                "TYPE_CHECKING import + fallback",
                '''    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy
    from vllm.v1.request import Request
else:
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    KVCacheBlockCopy = object
''',
                '''    from vllm.v1.core.kv_cache_utils import KVCacheBlockCopy, MambaEndpointStateCopy
    from vllm.v1.request import Request
else:
    ECConnectorMetadata = object
    KVConnectorMetadata = object
    KVCacheBlockCopy = object
    MambaEndpointStateCopy = object
''',
            ),
            (
                "mamba_endpoint_copies field",
                '''    # CoW copies to apply after zeroing new blocks and before forward.
    kv_cache_block_copies: list[KVCacheBlockCopy] | None = None

    # Producer partial-tail offload hand-off for external KV connectors:
''',
                '''    # CoW copies to apply after zeroing new blocks and before forward.
    kv_cache_block_copies: list[KVCacheBlockCopy] | None = None

    # fix-k3-request-endpoint-cache: finished requests' recurrent states to
    # write into their durable request-endpoint blocks. The worker applies
    # them while the finished requests are still mapped, before this step's
    # CoW copies (a consumer may already copy from a destination block).
    mamba_endpoint_copies: list[MambaEndpointStateCopy] | None = None

    # Producer partial-tail offload hand-off for external KV connectors:
''',
            ),
        ],
    ),
    (
        "vllm/v1/core/sched/scheduler.py",
        [
            (
                "__init__ request_endpoint_cache flag",
                '''        self.hash_block_size = hash_block_size
        self.kv_cache_manager = KVCacheManager(
''',
                '''        self.hash_block_size = hash_block_size
        # Request-endpoint cache: a finished request's recurrent state is read
        # either from its blocks or from the worker's shadow slot written
        # before the one step that may still be in flight. More concurrent
        # batches would leave states neither source covers.
        self.request_endpoint_cache = (
            self.kv_cache_config.has_mamba_layers
            and bool(envs.VLLM_K3_REQUEST_ENDPOINT_CACHE)
            and self.vllm_config.max_concurrent_batches <= 2
        )
        if envs.VLLM_K3_REQUEST_ENDPOINT_CACHE and not self.request_endpoint_cache:
            logger.warning(
                "VLLM_K3_REQUEST_ENDPOINT_CACHE is set but unsupported here "
                "(mamba layers: %s, concurrent batches: %d); disabled.",
                self.kv_cache_config.has_mamba_layers,
                self.vllm_config.max_concurrent_batches,
            )
        self.kv_cache_manager = KVCacheManager(
''',
            ),
            (
                "schedule() drains endpoint copies",
                '''        pending_kv_cache_block_copies = kv_cache_block_copies or None

        # Dynamic speculative decoding: compute optimal K
''',
                '''        pending_kv_cache_block_copies = kv_cache_block_copies or None

        # Request-endpoint state copies run before this step's CoW copies (a
        # consumer may already copy from a destination block); the blocks they
        # read and write stay retained until the step completes.
        mamba_endpoint_copies, endpoint_retained_blocks = (
            self.kv_cache_manager.take_mamba_endpoint_copies()
        )
        if mamba_endpoint_copies:
            self._free_cow_retained_blocks(
                endpoint_retained_blocks, self.sched_step_seq + 1
            )
        pending_mamba_endpoint_copies = mamba_endpoint_copies or None

        # Dynamic speculative decoding: compute optimal K
''',
            ),
            (
                "SchedulerOutput mamba_endpoint_copies",
                '''            kv_cache_block_copies=pending_kv_cache_block_copies,
            partial_tail_offloads=pending_partial_tail_offloads,
''',
                '''            kv_cache_block_copies=pending_kv_cache_block_copies,
            mamba_endpoint_copies=pending_mamba_endpoint_copies,
            partial_tail_offloads=pending_partial_tail_offloads,
''',
            ),
            (
                "endpoint_final_step (with bug (a) guard)",
                '''                    prefill_stats.finalize(
                        self.kv_cache_manager.estimate_cached_tokens(request)
                    )

            finish_reason = None
''',
                '''                    prefill_stats.finalize(
                        self.kv_cache_manager.estimate_cached_tokens(request)
                    )

            # The request-endpoint cache locates the committed recurrent state
            # from the last step's draft geometry.
            # fix-k3-request-endpoint-cache (bug (a), flagged by the PR's
            # review but unfixed upstream): num_draft_tokens / num_accepted
            # are only meaningful on the speculative path. Our pin
            # pre-initializes them to (0, 0) per request, but guard on
            # observed_spec_decode so a non-speculative request publishes the
            # neutral (0, 0) geometry and can never read unbound or stale
            # draft values.
            request.endpoint_final_step = (
                (num_draft_tokens, num_accepted) if observed_spec_decode else (0, 0)
            )

            finish_reason = None
''',
            ),
            (
                "_free_request publishes the endpoint",
                '''        self._inflight_prefills.discard(request)
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
''',
                '''        self._inflight_prefills.discard(request)
        self._cache_request_endpoint(request)
        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
''',
            ),
            (
                "_cache_request_endpoint method",
                '''        return kv_xfer_params, ec_xfer_params

    def _free_blocks(self, request: Request):
''',
                '''        return kv_xfer_params, ec_xfer_params

    def _cache_request_endpoint(self, request: Request) -> None:
        """Publish the finished request's end position to the prefix cache.

        Runs at finish time, while the request still holds its blocks: a
        connector may delay the free, and the worker slot the recurrent state
        is copied from is recycled once the request is reported finished.
        """
        if not self.request_endpoint_cache:
            return
        self.kv_cache_manager.cache_endpoint(
            request,
            request.endpoint_final_step,
            in_flight=request.num_in_flight_tokens > 0,
        )

    def _free_blocks(self, request: Request):
''',
            ),
        ],
    ),
    (
        "vllm/v1/request.py",
        [
            (
                "endpoint_final_step field",
                '''        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.cache_salt: str | None = cache_salt
''',
                '''        self.spec_token_ids: list[int] = []
        self.num_computed_tokens = 0
        # fix-k3-request-endpoint-cache: (scheduled draft tokens, accepted
        # draft tokens) of the step that finished the request; the
        # request-endpoint cache uses them to locate the committed
        # recurrent state slot.
        self.endpoint_final_step: tuple[int, int] = (0, 0)
        self.cache_salt: str | None = cache_salt
''',
            ),
        ],
    ),
    (
        "vllm/v1/worker/gpu/model_runner.py",
        [
            (
                "finish_requests materializes endpoints first",
                '''    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        finished_req_ids = scheduler_output.finished_req_ids
''',
                '''    def finish_requests(self, scheduler_output: SchedulerOutput) -> None:
        # fix-k3-request-endpoint-cache: finished requests' end states are
        # read from their (still mapped) request slots, so materialize them
        # before the slots are recycled.
        if scheduler_output.mamba_endpoint_copies:
            self.model_state.materialize_request_endpoints(
                scheduler_output.mamba_endpoint_copies,
                self.req_states.req_id_to_index,
            )
        finished_req_ids = scheduler_output.finished_req_ids
''',
            ),
        ],
    ),
    (
        "vllm/v1/worker/gpu/model_states/interface.py",
        [
            (
                "logger import",
                '''from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.tasks import GenerationTask
''',
                '''from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.tasks import GenerationTask
''',
            ),
            (
                "kv_cache_utils import (withdrawal, bug (b))",
                '''from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.core.sched.output import NewRequestData
''',
                '''from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.core.kv_cache_utils import withdraw_request_endpoints
from vllm.v1.core.sched.output import NewRequestData
''',
            ),
            (
                "logger instance",
                '''from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


class ModelSpecificAttnMetadata:
''',
                '''from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


class ModelSpecificAttnMetadata:
''',
            ),
            (
                "base materialize_request_endpoints (withdraws on bug (b))",
                '''    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        return None

    @abstractmethod
    def get_mm_embeddings(
''',
                '''    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
        num_computed_tokens: torch.Tensor | None = None,
    ) -> None:
        return None

    def materialize_request_endpoints(
        self,
        copies: list[Any],
        req_id_to_index: dict[str, int],
    ) -> None:
        """Write finished requests' committed recurrent states into the pool
        blocks named by ``copies`` (``MambaEndpointStateCopy`` records). Runs
        before the finished requests are removed from the worker. Only hybrid
        recurrent model states implement it; the scheduler emits copies only
        when such a model publishes request-endpoint cache entries."""
        if copies:
            logger.warning_once(
                "Request-endpoint copies received by a model state without "
                "recurrent state; the entries stay unmaterialized."
            )
            # fix-k3-request-endpoint-cache (bug (b), flagged by the PR's
            # review but unfixed upstream): a model state that cannot
            # materialize recurrent states must not leave the published
            # entries resumable — withdraw them so no later prompt resumes
            # from unwritten state.
            for copy in copies:
                withdraw_request_endpoints(getattr(copy, "req_id", ""))
        return None

    @abstractmethod
    def get_mm_embeddings(
''',
            ),
        ],
    ),
    (
        "vllm/v1/worker/gpu/model_states/mamba_hybrid.py",
        [
            (
                "imports (envs, parallel_state, logger, conv layout)",
                '''import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.triton_utils import tl, triton
''',
                '''import torch
import torch.nn as nn

from vllm import envs
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.triton_utils import tl, triton
''',
            ),
            (
                "kv_cache_utils imports (endpoint switches + bug (b) hooks)",
                '''from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.core.sched.output import NewRequestData
''',
                '''from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.core.kv_cache_utils import (
    confirm_endpoint_materialized,
    request_endpoint_cache_debug,
    request_endpoint_cache_torch_copy,
    withdraw_request_endpoints,
)
from vllm.v1.core.sched.output import NewRequestData
''',
            ),
            (
                "mamba_utils imports + logger + page fingerprints",
                '''from vllm.v1.worker.mamba_utils import (
    MambaSpecDecodeGPUContext,
    preprocess_mamba_align_fused_kernel,
)
from vllm.v1.worker.utils import AttentionGroup


@dataclass
class MambaHybridAttnMetadata(ModelSpecificAttnMetadata):
''',
                '''from vllm.v1.worker.mamba_utils import (
    ENDPOINT_COPY_META_FIXED,
    EndpointCopyRecord,
    MambaSpecDecodeGPUContext,
    endpoint_layer_states,
    materialize_endpoints_torch,
    preprocess_mamba_align_fused_kernel,
    verify_endpoints,
)
from vllm.v1.worker.utils import AttentionGroup

logger = init_logger(__name__)


def _page_sig(tensor: torch.Tensor, block: int) -> str:
    """A cheap fingerprint of one block page for endpoint debugging."""
    page = tensor[block].float()
    flat = page.flatten()
    return (
        f"sum={page.sum().item():.6e} abs={page.abs().sum().item():.6e} "
        f"head={[round(v, 5) for v in flat[:3].tolist()]} "
        f"tail={[round(v, 5) for v in flat[-3:].tolist()]}"
    )


def _bytes_sig(raw: torch.Tensor, dtype: torch.dtype) -> str:
    """Fingerprint of a raw (uint8) shadow page viewed as ``dtype``."""
    elem = torch.empty((), dtype=dtype).element_size()
    usable = raw.numel() // elem * elem
    page = raw[:usable].view(dtype).float()
    return (
        f"sum={page.sum().item():.6e} abs={page.abs().sum().item():.6e} "
        f"head={[round(v, 5) for v in page[:3].tolist()]} "
        f"tail={[round(v, 5) for v in page[-3:].tolist()]}"
    )


@dataclass
class MambaHybridAttnMetadata(ModelSpecificAttnMetadata):
''',
            ),
            (
                "__init__ layer states + endpoint shadow",
                '''            self._mamba_ctx: MambaSpecDecodeGPUContext | None = None
            self._mamba_group_ids: list[int] = []
            self._mamba_spec: MambaSpec | None = None
''',
                '''            self._mamba_ctx: MambaSpecDecodeGPUContext | None = None
            # Every recurrent layer's state views in the context's state
            # order (torch reference path of the endpoint materialization).
            self._mamba_layer_states: list[tuple[torch.Tensor, ...]] = []
            self._mamba_group_ids: list[int] = []
            self._mamba_spec: MambaSpec | None = None
            # Request-endpoint cache: keep every request's committed state in
            # a shadow slot so a finished request's state survives the step
            # that was already in flight when it finished.
            self._endpoint_shadow = bool(envs.VLLM_K3_REQUEST_ENDPOINT_CACHE)
''',
            ),
            (
                "_get_mamba_group_info first layer name + _debug_mamba_states",
                '''            self._mamba_group_ids = group_ids
            self._mamba_spec = specs[0]
        return self._mamba_group_ids, self._mamba_spec

    def _ensure_align_ctx(
''',
                '''            self._mamba_group_ids = group_ids
            self._mamba_spec = specs[0]
            self._first_mamba_layer_name = kv_cache_config.kv_cache_groups[
                group_ids[0]
            ].layer_names[0]
        return self._mamba_group_ids, self._mamba_spec

    def _debug_mamba_states(self) -> list[torch.Tensor] | None:
        """The first recurrent layer's ``[conv_state, temporal_state]``."""
        name = getattr(self, "_first_mamba_layer_name", None)
        if name is None:
            return None
        forward_context = self.vllm_config.compilation_config.static_forward_context
        attention = forward_context.get(name)
        return getattr(attention, "kv_cache", None)

    def _ensure_align_ctx(
''',
            ),
            (
                "_ensure_align_ctx layer states + shadow",
                '''                [block_tables[gid] for gid in mamba_group_ids],
            )
        return ctx
''',
                '''                [block_tables[gid] for gid in mamba_group_ids],
            )
            self._mamba_layer_states = endpoint_layer_states(
                kv_cache_config, forward_context, mamba_group_ids
            )
        if self._endpoint_shadow and not ctx.has_endpoint_shadow:
            ctx.ensure_endpoint_shadow(self.max_num_reqs)
        return ctx
''',
            ),
            (
                "preprocess_state snapshots to shadow",
                '''            self._mamba_src_off_gpu,
            input_batch.idx_mapping,
        )
''',
                '''            self._mamba_src_off_gpu,
            input_batch.idx_mapping,
            snapshot_to_shadow=self._endpoint_shadow,
        )
''',
            ),
            (
                "materialize_request_endpoints + debug helpers (bug (b) withdrawals)",
                '''    def prepare_attn(
        self,
        input_batch: InputBatch,
''',
                '''    def materialize_request_endpoints(
        self,
        copies: list[Any],
        req_id_to_index: dict[str, int],
    ) -> None:
        """Write finished requests' committed states into pool blocks.

        Each ``MambaEndpointStateCopy`` names the destination block and either
        the request's shadow pages (``from_shadow``; ``token_bias`` selects the
        conv shift and the temporal slot) or its own state blocks per
        recurrent group. Runs on the current stream before the finished
        requests' slots are recycled, so it is ordered after the in-flight
        step that may have overwritten their state slots and before any
        consumer's copy-on-write of the destination block.

        The fused kernel performs the copies; with the torch-copy switch the
        per-state torch reference performs them instead. In debug mode the
        kernel result is verified against the reference for every state and
        the mismatches are logged on the first tensor-parallel rank.

        fix-k3-request-endpoint-cache (bug (b), flagged by the PR's review
        but unfixed upstream): every skip path WITHDRAWS the request's
        published endpoint entry (via the withdrawal registry in
        kv_cache_utils) so a later prompt can never resume from unwritten
        recurrent state; successful copies are confirmed so their entries
        survive.
        """
        if not copies:
            return
        if not self._align_mode:
            # fix-k3-request-endpoint-cache (bug (b)): copies on a non-align
            # model can never be materialized here; withdraw them all.
            for copy in copies:
                withdraw_request_endpoints(copy.req_id)
            return
        ctx = self._mamba_ctx
        if ctx is None or not ctx.is_initialized:
            logger.warning_once(
                "Request-endpoint copies arrived before the recurrent-state "
                "context was initialized; %d entr(ies) stay unmaterialized.",
                len(copies),
            )
            # fix-k3-request-endpoint-cache (bug (b)): the context may never
            # initialize; withdraw every entry.
            for copy in copies:
                withdraw_request_endpoints(copy.req_id)
            return
        num_groups = ctx.num_groups
        width = ENDPOINT_COPY_META_FIXED + 3 * num_groups
        meta = np.full((len(copies), width), -1, dtype=np.int32)
        records: list[EndpointCopyRecord] = []
        rows = 0
        # fix-k3-request-endpoint-cache (bug (b)): requests whose copy is
        # enqueued (confirmed below once the launch has happened).
        materialized_req_ids: list[str] = []
        for copy in copies:
            req_idx = req_id_to_index.get(copy.req_id)
            if copy.from_shadow:
                if req_idx is None or not ctx.has_endpoint_shadow:
                    logger.warning(
                        "Request-endpoint copy for %s skipped: shadow source "
                        "unavailable (slot=%s, shadow=%s)",
                        copy.req_id,
                        req_idx,
                        ctx.has_endpoint_shadow,
                    )
                    # fix-k3-request-endpoint-cache (bug (b)): the state was
                    # never written; drop the published entry.
                    withdraw_request_endpoints(copy.req_id)
                    continue
                meta[rows, 0] = 1
                meta[rows, 1] = req_idx
                meta[rows, 2] = copy.token_bias
            else:
                if (
                    len(copy.conv_src_block_ids) != num_groups
                    or len(copy.temporal_src_block_ids) != num_groups
                ):
                    logger.warning(
                        "Request-endpoint copy for %s skipped: %d/%d source "
                        "blocks for %d recurrent groups",
                        copy.req_id,
                        len(copy.conv_src_block_ids),
                        len(copy.temporal_src_block_ids),
                        num_groups,
                    )
                    withdraw_request_endpoints(copy.req_id)
                    continue
                meta[rows, 0] = 0
                meta[rows, 1] = 0
                meta[rows, 2] = copy.token_bias
                meta[rows, 4 + num_groups : 4 + 2 * num_groups] = (
                    copy.conv_src_block_ids
                )
                meta[rows, 4 + 2 * num_groups : width] = copy.temporal_src_block_ids
            if len(copy.dst_block_ids) != num_groups:
                logger.warning(
                    "Request-endpoint copy for %s skipped: %d destination "
                    "blocks for %d recurrent groups",
                    copy.req_id,
                    len(copy.dst_block_ids),
                    num_groups,
                )
                meta[rows] = -1
                withdraw_request_endpoints(copy.req_id)
                continue
            meta[rows, 4 : 4 + num_groups] = copy.dst_block_ids
            records.append(
                EndpointCopyRecord(
                    from_shadow=bool(copy.from_shadow),
                    req_idx=req_idx if copy.from_shadow and req_idx is not None else 0,
                    token_bias=int(copy.token_bias),
                    dst_block_ids=tuple(int(b) for b in copy.dst_block_ids),
                    conv_src_block_ids=tuple(copy.conv_src_block_ids),
                    temporal_src_block_ids=tuple(copy.temporal_src_block_ids),
                )
            )
            materialized_req_ids.append(copy.req_id)
            rows += 1
        if rows == 0:
            return
        torch_copy = request_endpoint_cache_torch_copy()
        debug = request_endpoint_cache_debug() and get_tensor_model_parallel_rank() == 0
        conv_dim_first = is_conv_state_dim_first()
        layer_states = self._mamba_layer_states
        if debug:
            self._debug_log_endpoint_sources(copies, req_id_to_index, ctx)
        if not torch_copy or debug:
            copy_meta = torch.from_numpy(meta[:rows]).to(self.device)
            trace = None
            if debug:
                trace = torch.zeros(
                    (rows, ctx.num_layers * ctx.num_state_types, 4),
                    dtype=torch.int64,
                    device=self.device,
                )
            ctx.run_endpoint_materialize(copy_meta, trace)
            if debug:
                torch.accelerator.synchronize()
                self._debug_log_trace(records, trace)
                self._debug_log_verification(
                    "kernel", records, layer_states, conv_dim_first
                )
        if torch_copy:
            materialize_endpoints_torch(ctx, layer_states, records, conv_dim_first)
            if debug:
                torch.accelerator.synchronize()
                self._debug_log_verification(
                    "torch", records, layer_states, conv_dim_first
                )
        # fix-k3-request-endpoint-cache (bug (b)): the copies are enqueued on
        # the current stream (ordered before any consumer CoW); release the
        # withdrawal hooks so these entries survive.
        for req_id in materialized_req_ids:
            confirm_endpoint_materialized(req_id)

    def _debug_log_trace(
        self, records: list[EndpointCopyRecord], trace: torch.Tensor | None
    ) -> None:
        """Compare the addresses the kernel resolved with the host
        expectation for every record and state."""
        ctx = self._mamba_ctx
        if ctx is None or trace is None:
            return
        expected = ctx.expected_endpoint_addresses(records)
        actual = trace.cpu().tolist()
        mismatches: list[dict[str, Any]] = []
        for record_idx, per_state in enumerate(expected):
            for state_idx, want in enumerate(per_state):
                got = tuple(actual[record_idx][state_idx])
                if got != want:
                    mismatches.append(
                        {
                            "record": record_idx,
                            "state_idx": state_idx,
                            "kernel": [hex(v) for v in got],
                            "expected": [hex(v) for v in want],
                        }
                    )
        logger.info(
            "[endpoint] trace records=%d states=%d address_mismatches=%d first=%s",
            len(records),
            len(expected[0]) if expected else 0,
            len(mismatches),
            mismatches[0] if mismatches else None,
        )

    def _debug_log_verification(
        self,
        path: str,
        records: list[EndpointCopyRecord],
        layer_states: list[tuple[torch.Tensor, ...]],
        conv_dim_first: bool,
    ) -> None:
        ctx = self._mamba_ctx
        assert ctx is not None
        mismatches = verify_endpoints(ctx, layer_states, records, conv_dim_first)
        total = len(records) * len(layer_states) * ctx.num_state_types
        by_record: dict[int, int] = {}
        for entry in mismatches:
            by_record[entry["record"]] = by_record.get(entry["record"], 0) + 1
        logger.info(
            "[endpoint] verify path=%s records=%d states=%d mismatched=%d "
            "per_record=%s first=%s",
            path,
            len(records),
            total,
            len(mismatches),
            {records[i].dst_block_ids[0]: n for i, n in sorted(by_record.items())},
            mismatches[0] if mismatches else None,
        )

    def _debug_log_endpoint_sources(self, copies, req_id_to_index, ctx) -> None:
        states = self._debug_mamba_states()
        if states is None or len(states) < 2 or not ctx.is_initialized:
            return
        torch.accelerator.synchronize()
        strides = ctx.shadow_page_strides_cpu[:2]
        base0 = int(ctx.state_base_addrs[0].item())
        base1 = int(ctx.state_base_addrs[1].item())
        logger.info(
            "[endpoint] views conv_ptr=%#x ctx_base0=%#x temporal_ptr=%#x "
            "ctx_base1=%#x "
            "shadow0_ptr=%#x ctx_shadow0=%#x shadow1_ptr=%#x ctx_shadow1=%#x layers=%d",
            states[0].data_ptr(),
            base0,
            states[1].data_ptr(),
            base1,
            ctx.shadow_buffers[0].data_ptr() if ctx.has_endpoint_shadow else 0,
            int(ctx.shadow_base_addrs[0].item()),
            ctx.shadow_buffers[1].data_ptr() if ctx.has_endpoint_shadow else 0,
            int(ctx.shadow_base_addrs[1].item()),
            len(self._mamba_layer_states),
        )
        for copy in copies:
            if copy.from_shadow:
                req_idx = req_id_to_index.get(copy.req_id)
                if req_idx is None or not ctx.has_endpoint_shadow:
                    continue
                bufs = ctx.shadow_buffers
                conv_page = bufs[0][req_idx * strides[0] : (req_idx + 1) * strides[0]]
                slot = req_idx * ctx.shadow_temporal_slots + copy.token_bias
                temporal_page = bufs[1][slot * strides[1] : (slot + 1) * strides[1]]
                conv_sig = _bytes_sig(conv_page, states[0].dtype)
                temporal_sig = _bytes_sig(temporal_page, states[1].dtype)
                src_desc = f"shadow slot={req_idx} temporal_slot={slot}"
            else:
                conv_sig = _page_sig(states[0], copy.conv_src_block_ids[0])
                temporal_sig = _page_sig(states[1], copy.temporal_src_block_ids[0])
                src_desc = (
                    f"blocks conv={copy.conv_src_block_ids[0]} "
                    f"temporal={copy.temporal_src_block_ids[0]}"
                )
            logger.info(
                "[endpoint] source req=%s %s bias=%d dst0=%d conv=%s temporal=%s "
                "state0=(base=%#x stride=%d) state1=(base=%#x stride=%d) "
                "shape0=%s stride0=%s shape1=%s stride1=%s",
                copy.req_id,
                src_desc,
                copy.token_bias,
                copy.dst_block_ids[0],
                conv_sig,
                temporal_sig,
                base0,
                int(strides[0]),
                base1,
                int(strides[1]),
                tuple(states[0].shape),
                tuple(states[0].stride()),
                tuple(states[1].shape),
                tuple(states[1].stride()),
            )

    def prepare_attn(
        self,
        input_batch: InputBatch,
''',
            ),
        ],
    ),
    (
        "vllm/v1/worker/gpu_model_runner.py",
        [
            (
                "v1 runner warns + withdraws (bug (b))",
                '''        if scheduler_output.new_block_ids_to_zero:
            self._zero_block_ids(scheduler_output.new_block_ids_to_zero)
        if scheduler_output.kv_cache_block_copies:
''',
                '''        if scheduler_output.new_block_ids_to_zero:
            self._zero_block_ids(scheduler_output.new_block_ids_to_zero)
        if scheduler_output.mamba_endpoint_copies:
            # The request-endpoint cache needs the v2 runner's recurrent-state
            # shadow; this runner leaves the entries unmaterialized.
            logger.warning_once(
                "Request-endpoint state copies are not applied by this runner; "
                "run with the v2 model runner to use the request-endpoint cache."
            )
            # fix-k3-request-endpoint-cache (bug (b), flagged by the PR's
            # review but unfixed upstream): withdraw the entries this runner
            # cannot materialize so no later prompt resumes from unwritten
            # recurrent state.
            from vllm.v1.core.kv_cache_utils import withdraw_request_endpoints

            for copy in scheduler_output.mamba_endpoint_copies:
                withdraw_request_endpoints(copy.req_id)
        if scheduler_output.kv_cache_block_copies:
''',
            ),
        ],
    ),
    (
        "vllm/v1/worker/mamba_utils.py",
        [
            (
                "_copy_mamba_state_addrs refactor + endpoint kernels",
                '''@triton.jit
def _copy_mamba_state_block(
    state_idx,
    bt_row_idx,
    src_col,
    dst_col,
    token_bias,
    block_table_ptrs_ptr,
    block_table_stride_req,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    # DS conv row metadata. Zero keeps the single-region copy path.
    state_dim_row_count_ptr,
    state_dim_row_stride_ptr,
    tile_idx,
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr,
):
    """Copy one (layer, state-type) mamba state block between block columns.

    Shared copy body of ``postprocess_mamba_fused_kernel`` and
    ``precopy_mamba_align_fused_kernel``, mirroring the V1 copy specs
    (``get_conv_copy_spec`` / ``get_temporal_copy_spec``):
    - conv state (conv_width > 0): shift the window by ``token_bias`` tokens,
      ``state[bt[src_col], token_bias:] ->
      state[bt[dst_col], :conv_width - token_bias]``
    - temporal state: ``token_bias`` selects the accepted speculative column,
      ``state[bt[src_col + token_bias]] -> state[bt[dst_col]]``

    The caller owns the decision logic (which columns, whether to copy); this
    device function only performs the byte copy for the given metadata slot.

    ``tile_idx`` in ``[0, TEMPORAL_TILES)`` partitions the temporal state's
    u64 range into ``TEMPORAL_TILES`` contiguous, COPY_BLOCK_SIZE-aligned
    slices, giving more CTAs to fill the SMs at small batch (multi-MiB
    temporal copies otherwise leave the GPU under-filled). Conv states are
    small; only ``tile_idx == 0`` copies them. ``TEMPORAL_TILES == 1`` and
    ``tile_idx == 0`` reproduces the untiled behavior.
    """
    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)
    state_elem_size = tl.load(state_elem_sizes_ptr + state_idx)
    state_inner_size = tl.load(state_inner_sizes_ptr + state_idx)
    conv_width = tl.load(state_conv_widths_ptr + state_idx)

    # Load the group index for this state, then index into the correct
    # group's block table. Each mamba group has independently allocated
    # physical blocks. Reinterpret as int32* since block ids are int32.
    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)
    group_base_addr = tl.load(block_table_ptrs_ptr + group_idx)
    block_table_typed = group_base_addr.to(tl.pointer_type(tl.int32))
    block_table_base = block_table_typed + bt_row_idx * block_table_stride_req

    # Widen block ids to int64 before they reach `block_id * state_block_stride`
    # below: state_block_stride can exceed 2**31 bytes for large mamba caches,
    # and Triton would otherwise do the multiply in int32 and wrap.
    dest_block_id = tl.load(block_table_base + dst_col).to(tl.int64)
    dst_addr = state_base_addr + dest_block_id * state_block_stride

    is_conv_state = conv_width > 0

    if CONV_STATE_DIM_FIRST and is_conv_state:
        # Conv states are small; only tile 0 does the copy. Higher tiles
        # early-return so they contribute nothing beyond a bounds check.
        if tile_idx > 0:
            return
        # DS conv layout: state_len is the slide axis; copy per dim row.
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        dim_rows = tl.load(state_dim_row_count_ptr + state_idx)
        row_stride = tl.load(state_dim_row_stride_ptr + state_idx)
        src_block_addr = state_base_addr + src_block_id * state_block_stride
        offsets = tl.arange(0, COPY_BLOCK_SIZE)

        # Stable row-to-lane ownership makes left shifts memmove-safe while
        # exposing the dimension rows in parallel. All addresses retain
        # state_elem_size alignment: tensor strides and token offsets are
        # measured in whole elements before conversion to bytes.
        num_dst_tokens = conv_width - token_bias
        for token_idx in range(0, num_dst_tokens):
            for row_base in range(0, dim_rows, COPY_BLOCK_SIZE):
                rows = row_base + offsets
                mask = rows < dim_rows
                src_byte_addr = (
                    src_block_addr
                    + rows * row_stride
                    + (token_idx + token_bias) * state_elem_size
                )
                dst_byte_addr = (
                    dst_addr + rows * row_stride + token_idx * state_elem_size
                )
                if state_elem_size == 2:
                    src_u16 = src_byte_addr.to(tl.pointer_type(tl.uint16))
                    dst_u16 = dst_byte_addr.to(tl.pointer_type(tl.uint16))
                    data_u16 = tl.load(src_u16, mask=mask)
                    tl.store(dst_u16, data_u16, mask=mask)
                elif state_elem_size == 4:
                    src_u32 = src_byte_addr.to(tl.pointer_type(tl.uint32))
                    dst_u32 = dst_byte_addr.to(tl.pointer_type(tl.uint32))
                    data_u32 = tl.load(src_u32, mask=mask)
                    tl.store(dst_u32, data_u32, mask=mask)
                else:
                    for byte_idx in range(0, state_elem_size):
                        src_u8 = (src_byte_addr + byte_idx).to(
                            tl.pointer_type(tl.uint8)
                        )
                        dst_u8 = (dst_byte_addr + byte_idx).to(
                            tl.pointer_type(tl.uint8)
                        )
                        data_u8 = tl.load(src_u8, mask=mask)
                        tl.store(dst_u8, data_u8, mask=mask)
        return

    if is_conv_state:
        if tile_idx > 0:
            return
        # SD conv: copy
        #   state[bt[src_col], token_bias:] ->
        #   state[bt[dst_col], :conv_width - token_bias]
        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        src_block_addr = state_base_addr + src_block_id * state_block_stride
        token_bytes = state_inner_size * state_elem_size
        num_dst_tokens = conv_width - token_bias

        # Distinct blocks and exact self-copies cannot have a destructive
        # overlap, so retain the u64-vectorized single-CTA copy.
        if src_block_id != dest_block_id or token_bias == 0:
            src_addr = src_block_addr + token_bias.to(tl.int64) * token_bytes
            copy_size = num_dst_tokens.to(tl.int64) * token_bytes
            _memcpy_u64_tiled(
                src_addr,
                dst_addr,
                copy_size,
                tile_idx,
                COPY_BLOCK_SIZE=COPY_BLOCK_SIZE,
                NUM_TILES=1,
            )
            return

        # Copy tokens from low to high. Each token-sized source and destination
        # region is disjoint, so same-block left shifts are memmove-safe
        # without a barrier.
        for token_idx in range(0, num_dst_tokens):
            src_token = src_block_addr + (token_idx + token_bias) * token_bytes
            dst_token = dst_addr + token_idx * token_bytes
            _memcpy_u64_tiled(
                src_token,
                dst_token,
                token_bytes,
                tile_idx,
                COPY_BLOCK_SIZE=COPY_BLOCK_SIZE,
                NUM_TILES=1,
            )
        return

    # Temporal state: copy state[bt[src_col + token_bias]] -> state[bt[dst_col]]
    # Body u64 range is partitioned across TEMPORAL_TILES CTAs to keep the
    # SMs filled at small batch.
    actual_src_block_id = tl.load(block_table_base + src_col + token_bias).to(tl.int64)
    src_addr = state_base_addr + actual_src_block_id * state_block_stride
    # Use natural block data size (inner_size * elem_size), NOT
    # state_block_stride which is the page stride and can exceed the
    # actual data when the state tensor uses as_strided page padding.
    copy_size = state_inner_size * state_elem_size
    _memcpy_u64_tiled(
        src_addr,
        dst_addr,
        copy_size,
        tile_idx,
        COPY_BLOCK_SIZE=COPY_BLOCK_SIZE,
        NUM_TILES=TEMPORAL_TILES,
    )


''',
                '''@triton.jit
def _copy_mamba_state_addrs(
    state_idx,
    conv_src_addr,
    temporal_src_addr,
    dst_addr,
    token_bias,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    # DS conv row metadata. Zero keeps the single-region copy path.
    state_dim_row_count_ptr,
    state_dim_row_stride_ptr,
    tile_idx,
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr,
):
    """Copy one (layer, state-type) mamba state between resolved block
    addresses (V1 ``get_conv_copy_spec`` / ``get_temporal_copy_spec``
    semantics):
    - conv state (conv_width > 0): shift the window by ``token_bias`` tokens,
      ``conv_src[token_bias:] -> dst[:conv_width - token_bias]``
    - temporal state: ``temporal_src -> dst`` (the caller resolves the
      accepted speculative column into ``temporal_src_addr``)

    fix-k3-request-endpoint-cache: port of upstream PR#732's
    ``_copy_mamba_state_addrs`` ADAPTED to our pin — the byte-copy body is
    our memmove-hardened variant (token-wise DS/SD paths with same-block
    left-shift safety), re-parameterized from block-table columns to
    resolved addresses, instead of upstream's simpler pre-hardening body.

    ``tile_idx`` in ``[0, TEMPORAL_TILES)`` partitions the temporal state's
    u64 range into ``TEMPORAL_TILES`` contiguous, COPY_BLOCK_SIZE-aligned
    slices, giving more CTAs to fill the SMs at small batch (multi-MiB
    temporal copies otherwise leave the GPU under-filled). Conv states are
    small; only ``tile_idx == 0`` copies them. ``TEMPORAL_TILES == 1`` and
    ``tile_idx == 0`` reproduces the untiled behavior.
    """
    state_elem_size = tl.load(state_elem_sizes_ptr + state_idx)
    state_inner_size = tl.load(state_inner_sizes_ptr + state_idx)
    conv_width = tl.load(state_conv_widths_ptr + state_idx)

    is_conv_state = conv_width > 0

    if CONV_STATE_DIM_FIRST and is_conv_state:
        # Conv states are small; only tile 0 does the copy. Higher tiles
        # early-return so they contribute nothing beyond a bounds check.
        if tile_idx > 0:
            return
        # DS conv layout: state_len is the slide axis; copy per dim row.
        dim_rows = tl.load(state_dim_row_count_ptr + state_idx)
        row_stride = tl.load(state_dim_row_stride_ptr + state_idx)
        offsets = tl.arange(0, COPY_BLOCK_SIZE)

        # Stable row-to-lane ownership makes left shifts memmove-safe while
        # exposing the dimension rows in parallel. All addresses retain
        # state_elem_size alignment: tensor strides and token offsets are
        # measured in whole elements before conversion to bytes.
        num_dst_tokens = conv_width - token_bias
        for token_idx in range(0, num_dst_tokens):
            for row_base in range(0, dim_rows, COPY_BLOCK_SIZE):
                rows = row_base + offsets
                mask = rows < dim_rows
                src_byte_addr = (
                    conv_src_addr
                    + rows * row_stride
                    + (token_idx + token_bias) * state_elem_size
                )
                dst_byte_addr = (
                    dst_addr + rows * row_stride + token_idx * state_elem_size
                )
                if state_elem_size == 2:
                    src_u16 = src_byte_addr.to(tl.pointer_type(tl.uint16))
                    dst_u16 = dst_byte_addr.to(tl.pointer_type(tl.uint16))
                    data_u16 = tl.load(src_u16, mask=mask)
                    tl.store(dst_u16, data_u16, mask=mask)
                elif state_elem_size == 4:
                    src_u32 = src_byte_addr.to(tl.pointer_type(tl.uint32))
                    dst_u32 = dst_byte_addr.to(tl.pointer_type(tl.uint32))
                    data_u32 = tl.load(src_u32, mask=mask)
                    tl.store(dst_u32, data_u32, mask=mask)
                else:
                    for byte_idx in range(0, state_elem_size):
                        src_u8 = (src_byte_addr + byte_idx).to(
                            tl.pointer_type(tl.uint8)
                        )
                        dst_u8 = (dst_byte_addr + byte_idx).to(
                            tl.pointer_type(tl.uint8)
                        )
                        data_u8 = tl.load(src_u8, mask=mask)
                        tl.store(dst_u8, data_u8, mask=mask)
        return

    if is_conv_state:
        if tile_idx > 0:
            return
        # SD conv: copy conv_src[token_bias:] -> dst[:conv_width - token_bias]
        token_bytes = state_inner_size * state_elem_size
        num_dst_tokens = conv_width - token_bias

        # Distinct addresses and exact self-copies cannot have a destructive
        # overlap, so retain the u64-vectorized single-CTA copy.
        if conv_src_addr != dst_addr or token_bias == 0:
            src_addr = conv_src_addr + token_bias.to(tl.int64) * token_bytes
            copy_size = num_dst_tokens.to(tl.int64) * token_bytes
            _memcpy_u64_tiled(
                src_addr,
                dst_addr,
                copy_size,
                tile_idx,
                COPY_BLOCK_SIZE=COPY_BLOCK_SIZE,
                NUM_TILES=1,
            )
            return

        # Copy tokens from low to high. Each token-sized source and destination
        # region is disjoint, so same-block left shifts are memmove-safe
        # without a barrier.
        for token_idx in range(0, num_dst_tokens):
            src_token = conv_src_addr + (token_idx + token_bias) * token_bytes
            dst_token = dst_addr + token_idx * token_bytes
            _memcpy_u64_tiled(
                src_token,
                dst_token,
                token_bytes,
                tile_idx,
                COPY_BLOCK_SIZE=COPY_BLOCK_SIZE,
                NUM_TILES=1,
            )
        return

    # Temporal state: copy temporal_src -> dst. Body u64 range is
    # partitioned across TEMPORAL_TILES CTAs to keep the SMs filled at
    # small batch.
    # Use natural block data size (inner_size * elem_size), NOT
    # state_block_stride which is the page stride and can exceed the
    # actual data when the state tensor uses as_strided page padding.
    copy_size = state_inner_size * state_elem_size
    _memcpy_u64_tiled(
        temporal_src_addr,
        dst_addr,
        copy_size,
        tile_idx,
        COPY_BLOCK_SIZE=COPY_BLOCK_SIZE,
        NUM_TILES=TEMPORAL_TILES,
    )


@triton.jit
def _copy_mamba_state_block(
    state_idx,
    bt_row_idx,
    src_col,
    dst_col,
    token_bias,
    block_table_ptrs_ptr,
    block_table_stride_req,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    # DS conv row metadata. Zero keeps the single-region copy path.
    state_dim_row_count_ptr,
    state_dim_row_stride_ptr,
    tile_idx,
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr,
):
    """Copy one (layer, state-type) mamba state block between block columns.

    Shared copy body of ``postprocess_mamba_fused_kernel`` and
    ``precopy_mamba_align_fused_kernel``, mirroring the V1 copy specs
    (``get_conv_copy_spec`` / ``get_temporal_copy_spec``):
    - conv state (conv_width > 0): shift the window by ``token_bias`` tokens,
      ``state[bt[src_col], token_bias:] ->
      state[bt[dst_col], :conv_width - token_bias]``
    - temporal state: ``token_bias`` selects the accepted speculative column,
      ``state[bt[src_col + token_bias]] -> state[bt[dst_col]]``

    The caller owns the decision logic (which columns, whether to copy); this
    device function resolves the block-table columns to addresses and
    delegates the byte copy to ``_copy_mamba_state_addrs``.
    """
    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)

    # Load the group index for this state, then index into the correct
    # group's block table. Each mamba group has independently allocated
    # physical blocks. Reinterpret as int32* since block ids are int32.
    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)
    group_base_addr = tl.load(block_table_ptrs_ptr + group_idx)
    block_table_typed = group_base_addr.to(tl.pointer_type(tl.int32))
    block_table_base = block_table_typed + bt_row_idx * block_table_stride_req

    # Widen block ids to int64 before they reach `block_id * state_block_stride`
    # below: state_block_stride can exceed 2**31 bytes for large mamba caches,
    # and Triton would otherwise do the multiply in int32 and wrap.
    dest_block_id = tl.load(block_table_base + dst_col).to(tl.int64)
    dst_addr = state_base_addr + dest_block_id * state_block_stride
    # The conv window lives in the source column; the accepted temporal state
    # in the column ``token_bias`` slots after it (same row, in range for
    # every state of the request).
    conv_src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
    temporal_src_block_id = tl.load(block_table_base + src_col + token_bias).to(
        tl.int64
    )
    conv_src_addr = state_base_addr + conv_src_block_id * state_block_stride
    temporal_src_addr = state_base_addr + temporal_src_block_id * state_block_stride

    _copy_mamba_state_addrs(
        state_idx,
        conv_src_addr,
        temporal_src_addr,
        dst_addr,
        token_bias,
        state_elem_sizes_ptr,
        state_inner_sizes_ptr,
        state_conv_widths_ptr,
        state_dim_row_count_ptr,
        state_dim_row_stride_ptr,
        tile_idx,
        COPY_BLOCK_SIZE,
        CONV_STATE_DIM_FIRST,
        TEMPORAL_TILES,
    )


@triton.jit
def _snapshot_mamba_state_to_shadow(
    state_idx,
    bt_row_idx,
    src_col,
    token_bias,
    shadow_slot,
    shadow_temporal_slots,
    block_table_ptrs_ptr,
    block_table_stride_req,
    state_base_addrs_ptr,
    state_block_strides_ptr,
    shadow_base_addrs_ptr,
    shadow_page_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    state_dim_row_count_ptr,
    state_dim_row_stride_ptr,
    tile_idx,
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr,
):
    """Copy a request's committed state into its shadow pages, compact pages
    (one state's data bytes each) outside the block tables that only this
    snapshot writes.

    The conv window of ``src_col`` is copied unshifted (the materialization
    applies the shift), and the temporal states of the columns ``src_col +
    t`` for ``t`` in ``[0, token_bias]`` land in the request's temporal
    shadow slots ``t``, so any acceptance count up to the committed one can
    be materialized later (a stop inside the accepted tokens keeps fewer).
    """
    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)
    conv_width = tl.load(state_conv_widths_ptr + state_idx)
    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)
    group_base_addr = tl.load(block_table_ptrs_ptr + group_idx)
    block_table_typed = group_base_addr.to(tl.pointer_type(tl.int32))
    block_table_base = block_table_typed + bt_row_idx * block_table_stride_req
    shadow_base_addr = tl.load(shadow_base_addrs_ptr + state_idx)
    shadow_page_stride = tl.load(shadow_page_strides_ptr + state_idx)

    if conv_width > 0:
        conv_src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        conv_src_addr = state_base_addr + conv_src_block_id * state_block_stride
        dst_addr = shadow_base_addr + shadow_slot.to(tl.int64) * shadow_page_stride
        _copy_mamba_state_addrs(
            state_idx,
            conv_src_addr,
            conv_src_addr,
            dst_addr,
            token_bias - token_bias,
            state_elem_sizes_ptr,
            state_inner_sizes_ptr,
            state_conv_widths_ptr,
            state_dim_row_count_ptr,
            state_dim_row_stride_ptr,
            tile_idx,
            COPY_BLOCK_SIZE,
            CONV_STATE_DIM_FIRST,
            TEMPORAL_TILES,
        )
        return

    state_elem_size = tl.load(state_elem_sizes_ptr + state_idx)
    state_inner_size = tl.load(state_inner_sizes_ptr + state_idx)
    copy_size = state_inner_size * state_elem_size
    slot_base = shadow_slot.to(tl.int64) * shadow_temporal_slots.to(tl.int64)
    for t in range(0, token_bias + 1):
        src_block_id = tl.load(block_table_base + src_col + t).to(tl.int64)
        src_addr = state_base_addr + src_block_id * state_block_stride
        dst_addr = shadow_base_addr + (slot_base + t) * shadow_page_stride
        _memcpy_u64_tiled(
            src_addr,
            dst_addr,
            copy_size,
            tile_idx,
            COPY_BLOCK_SIZE=COPY_BLOCK_SIZE,
            NUM_TILES=TEMPORAL_TILES,
        )


class EndpointCopyRecord(NamedTuple):
    """One request-endpoint materialization (host form of a kernel record)."""

    from_shadow: bool
    req_idx: int
    token_bias: int
    dst_block_ids: tuple[int, ...]
    conv_src_block_ids: tuple[int, ...]
    temporal_src_block_ids: tuple[int, ...]


def endpoint_layer_states(
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    mamba_group_ids: list[int],
) -> list[tuple[torch.Tensor, ...]]:
    """Every recurrent layer's state views in the context's state order
    (groups in ``mamba_group_ids`` order, layers in group order)."""
    return [
        tuple(forward_context[layer_name].kv_cache)
        for group_id in mamba_group_ids
        for layer_name in kv_cache_config.kv_cache_groups[group_id].layer_names
    ]


def _conv_window_copy(
    dst: torch.Tensor, src: torch.Tensor, token_bias: int, conv_dim_first: bool
) -> None:
    """``src[token_bias:] -> dst[:width - token_bias]`` along the conv slide
    axis (V1 ``get_conv_copy_spec`` semantics for both conv layouts)."""
    if conv_dim_first:
        width = src.shape[1]
        dst[:, : width - token_bias].copy_(src[:, token_bias:])
    else:
        width = src.shape[0]
        dst[: width - token_bias].copy_(src[token_bias:])


def materialize_endpoints_torch(
    ctx: "MambaSpecDecodeGPUContext",
    layer_states: list[tuple[torch.Tensor, ...]],
    records: list[EndpointCopyRecord],
    conv_dim_first: bool,
) -> None:
    """Torch reference of ``materialize_mamba_endpoint_kernel``: one tensor
    copy per (record, layer, state type) with identical results."""
    for record in records:
        sources = ctx.endpoint_source_pages(record, layer_states)
        for layer, states in enumerate(layer_states):
            for type_idx, state in enumerate(states):
                state_idx = layer * ctx.num_state_types + type_idx
                src = sources[layer][type_idx]
                group = ctx.state_group_indices_cpu[state_idx]
                dst = state[record.dst_block_ids[group]]
                if ctx.state_conv_widths_cpu[state_idx] > 0:
                    _conv_window_copy(dst, src, record.token_bias, conv_dim_first)
                else:
                    dst.copy_(src)


def verify_endpoints(
    ctx: "MambaSpecDecodeGPUContext",
    layer_states: list[tuple[torch.Tensor, ...]],
    records: list[EndpointCopyRecord],
    conv_dim_first: bool,
) -> list[dict[str, Any]]:
    """Compare every record's destination block with the copy-spec reference
    (the conv window a reader consumes and the whole temporal page). Returns
    one entry per mismatching (record, layer, state type)."""
    mismatches: list[dict[str, Any]] = []
    for record_idx, record in enumerate(records):
        sources = ctx.endpoint_source_pages(record, layer_states)
        for layer, states in enumerate(layer_states):
            for type_idx, state in enumerate(states):
                state_idx = layer * ctx.num_state_types + type_idx
                src = sources[layer][type_idx]
                group = ctx.state_group_indices_cpu[state_idx]
                dst = state[record.dst_block_ids[group]]
                if ctx.state_conv_widths_cpu[state_idx] > 0:
                    bias = record.token_bias
                    if conv_dim_first:
                        expected = src[:, bias:]
                        actual = dst[:, : src.shape[1] - bias]
                    else:
                        expected = src[bias:]
                        actual = dst[: src.shape[0] - bias]
                else:
                    expected, actual = src, dst
                if torch.equal(expected, actual):
                    continue
                mismatches.append(
                    {
                        "record": record_idx,
                        "layer": layer,
                        "state_idx": state_idx,
                        "kind": "conv"
                        if ctx.state_conv_widths_cpu[state_idx] > 0
                        else "temporal",
                        "expected_head": expected.flatten()[:3].float().tolist(),
                        "actual_head": actual.flatten()[:3].float().tolist(),
                        "expected_sum": float(expected.float().sum().item()),
                        "actual_sum": float(actual.float().sum().item()),
                    }
                )
    return mismatches


# Per-copy int32 record of ``materialize_mamba_endpoint_kernel``:
# [from_shadow, req_idx, token_bias, reserved, dst[num_groups],
#  conv_src[num_groups], temporal_src[num_groups]]. Every recurrent group has
# its own destination block because block ids alias the same physical pages
# across groups. ``token_bias`` selects the conv shift and, for a shadow
# source, the temporal shadow slot; the block-source columns are resolved by
# the scheduler and carry the temporal slot already.
ENDPOINT_COPY_META_FIXED = 4


@triton.jit(do_not_specialize=["num_copies"])
def materialize_mamba_endpoint_kernel(
    copy_meta_ptr,  # int32 [num_copies, copy_meta_stride]
    copy_meta_stride: tl.int64,
    shadow_base_addrs_ptr,  # int64 [total_states]; 0 when no shadow exists
    shadow_page_strides_ptr,  # int64 [total_states]; bytes per shadow page
    shadow_temporal_slots,  # temporal shadow slots per request
    state_base_addrs_ptr,
    state_block_strides_ptr,
    state_elem_sizes_ptr,
    state_inner_sizes_ptr,
    state_conv_widths_ptr,
    state_group_indices_ptr,
    state_dim_row_count_ptr,
    state_dim_row_stride_ptr,
    num_copies,
    NUM_GROUPS: tl.constexpr,
    COPY_BLOCK_SIZE: tl.constexpr,
    CONV_STATE_DIM_FIRST: tl.constexpr,
    TEMPORAL_TILES: tl.constexpr = 1,
    # Optional address trace: int64 [num_copies, total_states, 4] receiving
    # (conv source, temporal source, destination, from_shadow) per state.
    trace_ptr=None,
    trace_copy_stride=0,
    HAS_TRACE: tl.constexpr = False,
):
    """Write finished requests' committed recurrent states into pool blocks.

    Grid: (num_copies, num_layers * num_state_types [, TEMPORAL_TILES]). A
    record reads either the request's shadow pages (``from_shadow``: the
    unshifted conv window shifted here by ``token_bias`` and temporal shadow
    slot ``token_bias``) or its own blocks per recurrent group (the conv
    window of ``conv_src`` shifted by ``token_bias`` and the temporal slot
    ``temporal_src``), and writes the destination block's page of every
    recurrent cache group.
    """
    copy_idx = tl.program_id(0)
    state_idx = tl.program_id(1)
    tile_idx = tl.program_id(2)
    if copy_idx >= num_copies:
        return

    meta = copy_meta_ptr + copy_idx * copy_meta_stride
    from_shadow = tl.load(meta + 0)
    req_idx = tl.load(meta + 1).to(tl.int64)
    token_bias = tl.load(meta + 2)
    # Block ids alias the same physical pages across cache groups (one raw
    # tensor per layer index is shared by every group), so each recurrent
    # group has its own destination block.
    group_idx = tl.load(state_group_indices_ptr + state_idx).to(tl.int64)
    dst_block_id = tl.load(meta + 4 + group_idx).to(tl.int64)
    if dst_block_id < 0:
        return

    state_base_addr = tl.load(state_base_addrs_ptr + state_idx)
    state_block_stride = tl.load(state_block_strides_ptr + state_idx)
    dst_addr = state_base_addr + dst_block_id * state_block_stride

    conv_src_block_id = tl.load(meta + 4 + NUM_GROUPS + group_idx).to(tl.int64)
    temporal_src_block_id = tl.load(meta + 4 + 2 * NUM_GROUPS + group_idx).to(tl.int64)
    slot_conv_addr = state_base_addr + conv_src_block_id * state_block_stride
    slot_temporal_addr = state_base_addr + temporal_src_block_id * state_block_stride

    # Shadow layout: one unshifted conv page per request; temporal pages
    # ``req_idx * shadow_temporal_slots + t`` for accepted-slot ``t``.
    shadow_base_addr = tl.load(shadow_base_addrs_ptr + state_idx)
    shadow_page_stride = tl.load(shadow_page_strides_ptr + state_idx)
    shadow_conv_addr = shadow_base_addr + req_idx * shadow_page_stride
    shadow_temporal_addr = (
        shadow_base_addr
        + (req_idx * shadow_temporal_slots.to(tl.int64) + token_bias.to(tl.int64))
        * shadow_page_stride
    )

    use_shadow = from_shadow != 0
    conv_src_addr = tl.where(use_shadow, shadow_conv_addr, slot_conv_addr)
    temporal_src_addr = tl.where(use_shadow, shadow_temporal_addr, slot_temporal_addr)
    if HAS_TRACE:  # noqa: SIM102 (constexpr guard kept separate for Triton)
        if tile_idx == 0:
            trace = trace_ptr + copy_idx * trace_copy_stride + state_idx * 4
            tl.store(trace + 0, conv_src_addr.to(tl.int64))
            tl.store(trace + 1, temporal_src_addr.to(tl.int64))
            tl.store(trace + 2, dst_addr.to(tl.int64))
            tl.store(trace + 3, from_shadow.to(tl.int64))

    _copy_mamba_state_addrs(
        state_idx,
        conv_src_addr,
        temporal_src_addr,
        dst_addr,
        token_bias,
        state_elem_sizes_ptr,
        state_inner_sizes_ptr,
        state_conv_widths_ptr,
        state_dim_row_count_ptr,
        state_dim_row_stride_ptr,
        tile_idx,
        COPY_BLOCK_SIZE,
        CONV_STATE_DIM_FIRST,
        TEMPORAL_TILES,
    )


''',
            ),
            (
                "precopy kernel shadow params",
                '''    # TEMPORAL_TILES: see postprocess_mamba_fused_kernel. Default 1 preserves
    # the 2D-grid contract; > 1 requires a 3D grid.
    TEMPORAL_TILES: tl.constexpr = 1,
):
    """Pre-copy mamba "align" state across block boundaries.
''',
                '''    # TEMPORAL_TILES: see postprocess_mamba_fused_kernel. Default 1 preserves
    # the 2D-grid contract; > 1 requires a 3D grid.
    TEMPORAL_TILES: tl.constexpr = 1,
    # Request-endpoint shadow: before the boundary decision, copy every
    # request's committed state into its shadow slot (indexed by req_idx) so
    # a request that finishes this step keeps a readable state even though
    # this step overwrites its state slots.
    shadow_base_addrs_ptr=None,
    shadow_temporal_slots=1,
    HAS_SHADOW: tl.constexpr = False,
    shadow_page_strides_ptr=None,
):
    """Pre-copy mamba "align" state across block boundaries.
''',
            ),
            (
                "precopy kernel snapshot + split early-return",
                '''    src_col = tl.load(src_col_ptr + req_idx)
    dst_col = tl.load(mamba_state_idx_ptr + req_idx)
    # Fresh state, or still writing the same block: kernels locate the initial
    # state in-block via num_accepted (preserved when no boundary is crossed),
    # so there is nothing to copy.
    if src_col < 0 or src_col == dst_col:
        return

    token_bias = tl.load(token_bias_ptr + req_idx)
    _copy_mamba_state_block(
''',
                '''    src_col = tl.load(src_col_ptr + req_idx)
    dst_col = tl.load(mamba_state_idx_ptr + req_idx)
    # Fresh state: nothing committed yet.
    if src_col < 0:
        return

    token_bias = tl.load(token_bias_ptr + req_idx)
    if HAS_SHADOW:
        _snapshot_mamba_state_to_shadow(
            state_idx,
            batch_idx,
            src_col,
            token_bias,
            req_idx,
            shadow_temporal_slots,
            block_table_ptrs_ptr,
            block_table_stride_req,
            state_base_addrs_ptr,
            state_block_strides_ptr,
            shadow_base_addrs_ptr,
            shadow_page_strides_ptr,
            state_elem_sizes_ptr,
            state_inner_sizes_ptr,
            state_conv_widths_ptr,
            state_group_indices_ptr,
            state_dim_row_count_ptr,
            state_dim_row_stride_ptr,
            tile_idx,
            COPY_BLOCK_SIZE,
            CONV_STATE_DIM_FIRST,
            TEMPORAL_TILES,
        )

    # Still writing the same block: kernels locate the initial state in-block
    # via num_accepted (preserved when no boundary is crossed), so there is
    # nothing to migrate.
    if src_col == dst_col:
        return

    _copy_mamba_state_block(
''',
            ),
            (
                "context num_state_slots field",
                '''    block_table_ptrs: torch.Tensor
    block_table_stride_req: int = 0

''',
                '''    block_table_ptrs: torch.Tensor
    block_table_stride_req: int = 0
    # Recurrent state slots a request can hold at once: the committed state
    # plus one per speculative token.
    num_state_slots: int = 1

''',
            ),
            (
                "context shadow fields",
                '''    precopy_src_col_buf: CpuGpuBuffer | None = None
    precopy_token_bias_buf: CpuGpuBuffer | None = None

    # Flag to track if metadata has been populated
    is_initialized: bool = False
''',
                '''    precopy_src_col_buf: CpuGpuBuffer | None = None
    precopy_token_bias_buf: CpuGpuBuffer | None = None

    # Request-endpoint shadow: one block-strided slot per request state index
    # for every (layer, state type), written by the fused pre-copy launch and
    # read by ``materialize_mamba_endpoint_kernel``. ``shadow_base_addrs`` is
    # all zeros until ``ensure_endpoint_shadow`` allocates the slots.
    shadow_base_addrs: torch.Tensor | None = None
    # Bytes per shadow page per state: the state's data bytes per block
    # rounded up to 64, smaller than the pool page stride (which pads every
    # state to the page shared by all groups' layers).
    shadow_page_strides: torch.Tensor | None = None
    shadow_buffers: list[torch.Tensor] | None = None
    shadow_num_slots: int = 0
    # Temporal shadow pages per request: one per speculative state slot, so a
    # stop inside the accepted tokens can still be materialized.
    shadow_temporal_slots: int = 1
    # Host copies of the per-state metadata (filled by
    # ``initialize_from_forward_context``) for the torch reference path of
    # the request-endpoint materialization and its verification.
    state_block_strides_cpu: list[int] = dataclasses.field(default_factory=list)
    state_conv_widths_cpu: list[int] = dataclasses.field(default_factory=list)
    state_group_indices_cpu: list[int] = dataclasses.field(default_factory=list)
    state_base_addrs_cpu: list[int] = dataclasses.field(default_factory=list)
    shadow_base_addrs_cpu: list[int] = dataclasses.field(default_factory=list)
    shadow_page_strides_cpu: list[int] = dataclasses.field(default_factory=list)

    # Flag to track if metadata has been populated
    is_initialized: bool = False
''',
            ),
            (
                "create() num_state_slots",
                '''            num_groups=len(mamba_group_ids),
            num_accepted_tokens_out=torch.zeros(
''',
                '''            num_groups=len(mamba_group_ids),
            num_state_slots=1 + mamba_spec.num_speculative_blocks,
            num_accepted_tokens_out=torch.zeros(
''',
            ),
            (
                "create() shadow tensors",
                '''            precopy_src_col_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            precopy_token_bias_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            is_initialized=False,
        )
''',
                '''            precopy_src_col_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            precopy_token_bias_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            shadow_base_addrs=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            shadow_page_strides=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            is_initialized=False,
        )
''',
            ),
            (
                "context endpoint methods (shadow alloc + source pages)",
                '''            is_initialized=False,
        )

    def initialize_from_forward_context(
''',
                '''            is_initialized=False,
        )

    def state_page_bytes(self, state_idx: int) -> int:
        """Data bytes of one block of state ``state_idx`` (conv window or
        temporal state), without the pool page padding."""
        elem = int(self.state_elem_sizes[state_idx].item())
        inner = int(self.state_inner_sizes[state_idx].item())
        conv_width = int(self.state_conv_widths[state_idx].item())
        if conv_width > 0 and is_conv_state_dim_first():
            rows = int(self.state_dim_row_count[state_idx].item())
            return rows * conv_width * elem
        if conv_width > 0:
            return conv_width * inner * elem
        return inner * elem

    def ensure_endpoint_shadow(self, num_slots: int) -> None:
        """Allocate the request-endpoint shadow slots (idempotent).

        Each (layer, state type) gets ``num_slots`` compact pages (temporal
        states: ``num_slots`` × temporal slots) of the state's data bytes
        rounded up to 64, so the copy kernels address a shadow page like a
        pool block with the state's own page stride. Must run after
        ``initialize_from_forward_context``.
        """
        if self.shadow_buffers is not None:
            return
        assert self.is_initialized, "state metadata must be populated first"
        assert self.shadow_base_addrs is not None
        assert self.shadow_page_strides is not None
        device = self.shadow_base_addrs.device
        conv_widths = self.state_conv_widths.tolist()
        temporal_slots = self.num_state_slots
        buffers: list[torch.Tensor] = []
        addrs: list[int] = []
        strides: list[int] = []
        total_bytes = 0
        for state_idx, conv_width in enumerate(conv_widths):
            pages = num_slots * (1 if conv_width > 0 else temporal_slots)
            stride = (self.state_page_bytes(state_idx) + 63) // 64 * 64
            nbytes = stride * pages
            # Slots are zero-initialized so a never-written slot reads as an
            # all-zero state rather than as stale device memory.
            buf = torch.zeros(max(nbytes, 1), dtype=torch.uint8, device=device)
            buffers.append(buf)
            addrs.append(buf.data_ptr())
            strides.append(stride)
            total_bytes += nbytes
        self.shadow_base_addrs.copy_(
            torch.tensor(addrs, dtype=torch.int64, device=device)
        )
        self.shadow_page_strides.copy_(
            torch.tensor(strides, dtype=torch.int64, device=device)
        )
        self.shadow_base_addrs_cpu = list(addrs)
        self.shadow_page_strides_cpu = list(strides)
        self.shadow_buffers = buffers
        self.shadow_num_slots = num_slots
        self.shadow_temporal_slots = temporal_slots
        logger.info(
            "Request-endpoint recurrent-state shadow: %d slot(s) x %d state(s), "
            "%d temporal page(s) per slot, %.1f MiB",
            num_slots,
            len(strides),
            temporal_slots,
            total_bytes / (1 << 20),
        )

    @property
    def has_endpoint_shadow(self) -> bool:
        return self.shadow_buffers is not None

    def shadow_page(
        self, state_idx: int, page: int, like: torch.Tensor
    ) -> torch.Tensor:
        """Shadow page ``page`` of state ``state_idx`` viewed as one block of
        ``like`` (the state tensor whose blocks the shadow mirrors)."""
        assert self.shadow_buffers is not None
        stride = self.shadow_page_strides_cpu[state_idx]
        nbytes = like[0].numel() * like.element_size()
        raw = self.shadow_buffers[state_idx][page * stride : page * stride + nbytes]
        return raw.view(like.dtype).view(like.shape[1:])

    def endpoint_source_pages(
        self,
        record: "EndpointCopyRecord",
        layer_states: list[tuple[torch.Tensor, ...]],
    ) -> list[list[torch.Tensor]]:
        """Per layer and state type, the page a request-endpoint copy reads:
        the request's shadow pages (conv unshifted, temporal slot
        ``token_bias``) or its own state blocks per recurrent group."""
        pages: list[list[torch.Tensor]] = []
        for layer, states in enumerate(layer_states):
            per_layer: list[torch.Tensor] = []
            for type_idx, state in enumerate(states):
                state_idx = layer * self.num_state_types + type_idx
                is_conv = self.state_conv_widths_cpu[state_idx] > 0
                if record.from_shadow:
                    page = record.req_idx
                    if not is_conv:
                        page = page * self.shadow_temporal_slots + record.token_bias
                    per_layer.append(self.shadow_page(state_idx, page, state))
                else:
                    group = self.state_group_indices_cpu[state_idx]
                    block = (
                        record.conv_src_block_ids[group]
                        if is_conv
                        else record.temporal_src_block_ids[group]
                    )
                    per_layer.append(state[block])
            pages.append(per_layer)
        return pages

    def initialize_from_forward_context(
''',
            ),
            (
                "initialize_from_forward_context cpu metadata",
                '''        for i, bt in enumerate(block_tables):
            self.block_table_ptrs[i] = bt.data_ptr()

        self.is_initialized = True
''',
                '''        for i, bt in enumerate(block_tables):
            self.block_table_ptrs[i] = bt.data_ptr()

        self.state_block_strides_cpu = [
            int(v) for v in self.state_block_strides.tolist()
        ]
        self.state_conv_widths_cpu = [int(v) for v in self.state_conv_widths.tolist()]
        self.state_group_indices_cpu = [
            int(v) for v in self.state_group_indices.tolist()
        ]
        self.state_base_addrs_cpu = [int(v) for v in self.state_base_addrs.tolist()]
        self.is_initialized = True
''',
            ),
            (
                "run_fused_precopy signature",
                '''        token_bias_gpu: torch.Tensor,
        idx_mapping: torch.Tensor | None,
    ) -> None:
        """Pre-copy each request's previous running block into its new window
''',
                '''        token_bias_gpu: torch.Tensor,
        idx_mapping: torch.Tensor | None,
        snapshot_to_shadow: bool = False,
    ) -> None:
        """Pre-copy each request's previous running block into its new window
''',
            ),
            (
                "run_fused_precopy docstring + has_shadow",
                '''            idx_mapping: optional [num_reqs] batch_idx -> req_state_idx.
                None means V1 batch order already equals request state order.
        """
        if num_reqs == 0 or not self.is_initialized:
            return
        total_states = self.num_layers * self.num_state_types
        grid = (num_reqs, total_states, _TEMPORAL_TILES)
        precopy_mamba_align_fused_kernel[grid](
''',
                '''            idx_mapping: optional [num_reqs] batch_idx -> req_state_idx.
                None means V1 batch order already equals request state order.
            snapshot_to_shadow: also copy every request's committed state into
                its request-endpoint shadow slot (requires
                ``ensure_endpoint_shadow`` and an ``idx_mapping``).
        """
        if num_reqs == 0 or not self.is_initialized:
            return
        has_shadow = snapshot_to_shadow and self.shadow_buffers is not None
        if has_shadow:
            assert idx_mapping is not None, "shadow slots are indexed by req idx"
        total_states = self.num_layers * self.num_state_types
        grid = (num_reqs, total_states, _TEMPORAL_TILES)
        precopy_mamba_align_fused_kernel[grid](
''',
            ),
            (
                "run_fused_precopy shadow args + materialize/expected methods",
                '''            HAS_IDX_MAPPING=idx_mapping is not None,
            TEMPORAL_TILES=_TEMPORAL_TILES,
        )

    def run_fused_postprocess_align(
''',
                '''            HAS_IDX_MAPPING=idx_mapping is not None,
            TEMPORAL_TILES=_TEMPORAL_TILES,
            shadow_base_addrs_ptr=self.shadow_base_addrs,
            shadow_temporal_slots=self.shadow_temporal_slots,
            HAS_SHADOW=has_shadow,
            shadow_page_strides_ptr=self.shadow_page_strides,
        )

    def run_endpoint_materialize(
        self, copy_meta: torch.Tensor, trace: torch.Tensor | None = None
    ) -> None:
        """Run ``materialize_mamba_endpoint_kernel`` over ``copy_meta``, an
        int32 ``[num_copies, ENDPOINT_COPY_META_FIXED + 2 * num_groups]``
        device tensor of per-copy records. ``trace``, when given, is an
        int64 ``[num_copies, total_states, 4]`` device tensor that receives
        the addresses the kernel resolved (conv source, temporal source,
        destination) and the source kind per state."""
        num_copies = int(copy_meta.shape[0])
        if num_copies == 0 or not self.is_initialized:
            return
        assert copy_meta.dtype == torch.int32 and copy_meta.dim() == 2
        assert copy_meta.shape[1] == ENDPOINT_COPY_META_FIXED + 3 * self.num_groups
        assert copy_meta.is_contiguous()
        assert self.shadow_base_addrs is not None
        assert self.shadow_page_strides is not None
        total_states = self.num_layers * self.num_state_types
        if trace is not None:
            assert trace.dtype == torch.int64 and trace.is_contiguous()
            assert tuple(trace.shape) == (num_copies, total_states, 4)
        grid = (num_copies, total_states, _TEMPORAL_TILES)
        materialize_mamba_endpoint_kernel[grid](
            copy_meta,
            copy_meta.stride(0),
            self.shadow_base_addrs,
            self.shadow_page_strides,
            self.shadow_temporal_slots,
            self.state_base_addrs,
            self.state_block_strides,
            self.state_elem_sizes,
            self.state_inner_sizes,
            self.state_conv_widths,
            self.state_group_indices,
            self.state_dim_row_count,
            self.state_dim_row_stride,
            num_copies,
            NUM_GROUPS=self.num_groups,
            COPY_BLOCK_SIZE=1024,
            CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
            TEMPORAL_TILES=_TEMPORAL_TILES,
            trace_ptr=trace,
            trace_copy_stride=0 if trace is None else trace.stride(0),
            HAS_TRACE=trace is not None,
        )

    def expected_endpoint_addresses(
        self, records: list["EndpointCopyRecord"]
    ) -> list[list[tuple[int, int, int, int]]]:
        """Per record and state, the (conv source, temporal source,
        destination, from_shadow) addresses ``materialize_mamba_endpoint_kernel``
        must resolve, computed on the host from the captured metadata."""
        total_states = self.num_layers * self.num_state_types
        out: list[list[tuple[int, int, int, int]]] = []
        for record in records:
            per_state: list[tuple[int, int, int, int]] = []
            for state_idx in range(total_states):
                base = self.state_base_addrs_cpu[state_idx]
                stride = self.state_block_strides_cpu[state_idx]
                group = self.state_group_indices_cpu[state_idx]
                dst = base + record.dst_block_ids[group] * stride
                if record.from_shadow:
                    shadow = self.shadow_base_addrs_cpu[state_idx]
                    sstride = self.shadow_page_strides_cpu[state_idx]
                    conv_src = shadow + record.req_idx * sstride
                    temporal_src = (
                        shadow
                        + (
                            record.req_idx * self.shadow_temporal_slots
                            + record.token_bias
                        )
                        * sstride
                    )
                else:
                    conv_src = base + record.conv_src_block_ids[group] * stride
                    temporal_src = base + record.temporal_src_block_ids[group] * stride
                per_state.append((conv_src, temporal_src, dst, int(record.from_shadow)))
            out.append(per_state)
        return out

    def run_fused_postprocess_align(
''',
            ),
        ],
    ),
]

applied = 0
skipped = 0
for rel, hunks in HUNKS:
    found = []
    for root in ROOTS:
        try:
            out = subprocess.run(
                ["find", root, "-path", "*" + rel],
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
        ok = True
        for name, old, new in hunks:
            # per-site (0 matches: warn+skip this copy; >1: fail-loud)
            _n = s.count(old)
            if _n == 0:
                print(f"[{MARKER}] WARNING: {name} anchor not found in {p}, skipping")
                ok = False
                break
            if _n > 1:
                print(f"[{MARKER}] ERROR: {name} anchor count={_n} in {p}")
                sys.exit(4)
            s = s.replace(old, new, 1)
        if not ok:
            skipped += 1
            continue
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
