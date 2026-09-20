# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode.dspark.remote_speculator import (
    RemoteK3DSparkSpeculator,
    _anchor_positions_from_context,
    _build_valid_context_plan,
    _contiguous_draft_output,
    _RetainedRequestPrefix,
)


def test_build_valid_context_plan_drops_rejected_tail_rows():
    batch = SimpleNamespace(
        num_reqs=2,
        num_scheduled_tokens=np.array([4, 3], dtype=np.int32),
        num_computed_tokens_np=np.array([10, 20], dtype=np.int32),
    )

    indices, counts = _build_valid_context_plan(batch, [2, 0])

    assert indices == [0, 1, 4, 5, 6]
    assert counts == [2, 3]


def test_anchor_positions_follow_actual_valid_context_rows():
    positions = torch.tensor([24, 25, 26, 80, 81], dtype=torch.int64)

    anchors = _anchor_positions_from_context([3, 2], positions)

    assert anchors == [27, 82]


def test_remote_tokens_copy_supports_adaptive_depth():
    proxy = RemoteK3DSparkSpeculator.__new__(RemoteK3DSparkSpeculator)
    proxy.device = torch.device("cpu")
    proxy.draft_tokens = torch.full((3, 8), -1, dtype=torch.int64)

    proxy._copy_tokens_from_response(
        {"tokens": [[11, 12], [21, 22]]},
        active_indices=[0, 2],
        num_speculative_tokens=2,
    )

    assert proxy.draft_tokens.tolist() == [
        [11, 12, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1],
        [21, 22, -1, -1, -1, -1, -1, -1],
    ]


def test_adaptive_depth_output_is_contiguous_for_tp_broadcast():
    draft_tokens = torch.arange(24, dtype=torch.int64).view(3, 8)

    output = _contiguous_draft_output(draft_tokens, 2, 3)

    assert output.is_contiguous()
    assert output.tolist() == [[0, 1, 2], [8, 9, 10]]


@pytest.mark.parametrize("rejected", [[5, 0], [-1, 0]])
def test_build_valid_context_plan_rejects_invalid_counts(rejected):
    batch = SimpleNamespace(
        num_reqs=2,
        num_scheduled_tokens=np.array([4, 3], dtype=np.int32),
        num_computed_tokens_np=np.array([0, 0], dtype=np.int32),
    )

    with pytest.raises(ValueError, match="Invalid valid-context length"):
        _build_valid_context_plan(batch, rejected)


def _make_prefix_matcher() -> RemoteK3DSparkSpeculator:
    proxy = RemoteK3DSparkSpeculator.__new__(RemoteK3DSparkSpeculator)
    proxy._known_requests = {"old"}
    proxy._remote_block_size = 16
    proxy._remote_window_size = 32
    proxy._remote_prefix_cache_tokens = 128
    proxy._retained_prefixes = {
        "old": _RetainedRequestPrefix(
            token_ids=torch.arange(96, dtype=torch.int32),
            committed_end=96,
            context_start=0,
            serial=1,
        )
    }
    return proxy


def test_remote_prefix_match_requires_exact_token_identity():
    proxy = _make_prefix_matcher()
    matching = torch.arange(80, dtype=torch.int32)

    assert proxy._find_reconnect_source(matching, 80, {"new"}) == "old"

    mismatched = matching.clone()
    mismatched[40] = -1
    assert proxy._find_reconnect_source(mismatched, 80, {"new"}) is None


def test_remote_prefix_match_rejects_range_evicted_from_projected_cache():
    proxy = _make_prefix_matcher()
    proxy._remote_prefix_cache_tokens = 48
    matching = torch.arange(40, dtype=torch.int32)

    assert proxy._find_reconnect_source(matching, 40, {"new"}) is None


def test_remote_prefix_match_rejects_history_before_cold_bootstrap():
    proxy = _make_prefix_matcher()
    proxy._retained_prefixes["old"].context_start = 64

    assert (
        proxy._find_reconnect_source(torch.arange(80, dtype=torch.int32), 80, {"new"})
        is None
    )
    assert (
        proxy._find_reconnect_source(torch.arange(96, dtype=torch.int32), 96, {"new"})
        == "old"
    )
