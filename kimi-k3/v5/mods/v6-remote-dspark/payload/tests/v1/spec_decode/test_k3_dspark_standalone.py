# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch

from vllm.entrypoints.k3_dspark_rpc import (
    DraftKVSlotAllocator,
    ProjectedContextCache,
)
from vllm.entrypoints.k3_dspark_standalone import (
    EMBED_TENSOR,
    LM_HEAD_TENSOR,
    resolve_shared_weight_files,
)


def _write_index(root, weight_map):
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


def test_resolve_shared_weight_files_requires_both_target_tensors(tmp_path):
    shard = tmp_path / "shared.safetensors"
    shard.touch()
    _write_index(tmp_path, {EMBED_TENSOR: shard.name})

    with pytest.raises(KeyError, match=LM_HEAD_TENSOR):
        resolve_shared_weight_files(tmp_path)


def test_resolve_shared_weight_files_resolves_checkpoint_shards(tmp_path):
    shard = tmp_path / "shared.safetensors"
    shard.touch()
    _write_index(
        tmp_path,
        {
            EMBED_TENSOR: shard.name,
            LM_HEAD_TENSOR: shard.name,
        },
    )

    resolved = resolve_shared_weight_files(tmp_path)

    assert resolved == {
        EMBED_TENSOR: shard.resolve(),
        LM_HEAD_TENSOR: shard.resolve(),
    }


def test_resolve_shared_weight_files_rejects_path_escape(tmp_path):
    outside = tmp_path.parent / "outside.safetensors"
    outside.touch()
    _write_index(
        tmp_path,
        {
            EMBED_TENSOR: f"../{outside.name}",
            LM_HEAD_TENSOR: f"../{outside.name}",
        },
    )

    with pytest.raises(ValueError, match="escapes"):
        resolve_shared_weight_files(tmp_path)


def test_draft_kv_slot_allocator_keeps_rolling_blocks_unique():
    allocator = DraftKVSlotAllocator(
        num_cache_blocks=11,
        block_size=4,
        window_size=16,
        max_requests=2,
    )
    first, created = allocator.get_or_allocate("first")
    second, _ = allocator.get_or_allocate("second")

    assert created
    assert allocator.physical_block_range(first) == slice(1, 6)
    assert allocator.physical_block_range(second) == slice(6, 11)
    blocks, local_len = allocator.block_table(first, 21)
    assert blocks == [2, 3, 4, 5, 1]
    assert len(blocks) == len(set(blocks))
    assert local_len == 17


def test_draft_kv_slot_allocator_reuses_freed_request_slot():
    allocator = DraftKVSlotAllocator(
        num_cache_blocks=6,
        block_size=4,
        window_size=16,
        max_requests=1,
    )
    original, _ = allocator.get_or_allocate("original")
    with pytest.raises(RuntimeError, match="capacity exhausted"):
        allocator.get_or_allocate("other")

    assert allocator.free("original") is original
    replacement, created = allocator.get_or_allocate("replacement")
    assert created
    assert replacement.slot == original.slot


def test_draft_kv_slot_allocator_rebinds_without_changing_slot():
    allocator = DraftKVSlotAllocator(
        num_cache_blocks=6,
        block_size=4,
        window_size=16,
        max_requests=1,
    )
    original, _ = allocator.get_or_allocate("original")

    rebound = allocator.rebind("original", "replacement")

    assert rebound is original
    assert rebound.request_id == "replacement"
    assert allocator.get("original") is None
    assert allocator.get("replacement") is rebound


def test_projected_context_cache_rewinds_and_overwrites_exact_prefix():
    cache = ProjectedContextCache(hidden_size=2, max_tokens=8, chunk_size=4)
    initial = torch.arange(16, dtype=torch.bfloat16).view(8, 2)
    cache.append(0, initial)

    replacement = torch.tensor([[100, 101], [102, 103]], dtype=torch.bfloat16)
    cache.append(5, replacement)

    assert cache.start_position == 0
    assert cache.end_position == 7
    assert torch.equal(cache.read(0, 5), initial[:5])
    assert torch.equal(cache.read(5, 7), replacement)


def test_projected_context_cache_evicts_old_rows_without_regaining_them():
    cache = ProjectedContextCache(hidden_size=1, max_tokens=6, chunk_size=4)
    cache.append(0, torch.arange(8, dtype=torch.bfloat16).view(8, 1))

    assert cache.start_position == 2
    assert cache.has_range(2, 8)
    assert not cache.has_range(1, 8)

    cache.append(5, torch.tensor([[50], [60]], dtype=torch.bfloat16))
    assert cache.start_position == 2
    assert cache.end_position == 7
    with pytest.raises(ValueError, match="unavailable"):
        cache.read(1, 7)


def test_projected_context_cache_tracks_configured_device():
    cache = ProjectedContextCache(
        hidden_size=2,
        max_tokens=4,
        chunk_size=2,
        device=torch.device("cpu"),
    )
    states = torch.arange(8, dtype=torch.bfloat16).view(4, 2)

    cache.append(0, states)

    assert cache.device == torch.device("cpu")
    assert cache.read(0, 4).device == cache.device
    assert cache.allocated_bytes == states.numel() * states.element_size()


def test_projected_context_cache_rejects_device_mismatch():
    cache = ProjectedContextCache(hidden_size=2, max_tokens=4)
    states = torch.empty((1, 2), dtype=torch.bfloat16, device="meta")

    with pytest.raises(ValueError, match="device mismatch"):
        cache.append(0, states)
