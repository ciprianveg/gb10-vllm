from __future__ import annotations

import pytest
import torch

from b12x.distributed.pcie_oneshot import (
    PCIeOneshotAllReduce,
    PCIeOneshotAllReducePool,
    _compute_crossover_size,
    parse_pcie_oneshot_max_size,
)


class _FakeExt:
    def __init__(self):
        self.init_calls = []
        self.register_pcie_buffers_calls = []
        self.register_buffer_calls = []
        self.all_reduce_calls = []
        self.dispose_calls = []
        self.register_graph_buffers_calls = []
        self.handle_bytes = [1, 2, 3]
        self.offsets = [0, 64]

    def init_custom_ar(self, signal_ptrs, rank_data, rank):
        self.init_calls.append((tuple(signal_ptrs), rank_data.device.type, rank))
        return 12345

    def register_pcie_buffers(self, ptr, ptrs0, ptrs1):
        self.register_pcie_buffers_calls.append((ptr, tuple(ptrs0), tuple(ptrs1)))

    def register_buffer(self, ptr, peer_input_ptrs):
        self.register_buffer_calls.append((ptr, tuple(peer_input_ptrs)))

    def all_reduce(self, ptr, inp, out, reg_buffer, reg_buffer_bytes):
        self.all_reduce_calls.append((ptr, int(inp.data_ptr()), int(out.data_ptr()), reg_buffer, reg_buffer_bytes))
        out.copy_(inp)

    def dispose(self, ptr):
        self.dispose_calls.append(ptr)

    def meta_size(self):
        return 256

    def get_graph_buffer_ipc_meta(self, ptr):
        return list(self.handle_bytes), list(self.offsets)

    def register_graph_buffers(self, ptr, handles, offsets):
        self.register_graph_buffers_calls.append((ptr, handles, offsets))


def _make_runtime(
    *,
    rank=0,
    world_size=2,
    exchange_group=None,
    max_size=8 * 1024 * 1024,
    eager=False,
    ext=None,
):
    ext = ext or _FakeExt()
    kwargs = {}
    if eager:
        kwargs["eager_buffer_ptrs0"] = tuple(range(200, 200 + world_size))
        kwargs["eager_buffer_ptrs1"] = tuple(range(300, 300 + world_size))
    return PCIeOneshotAllReduce(
        rank=rank,
        world_size=world_size,
        device=torch.device("cpu"),
        signal_ptrs=tuple(range(100, 100 + world_size)),
        exchange_group=exchange_group,
        max_size=max_size,
        ext_module=ext,
        **kwargs,
    )


def test_parse_pcie_oneshot_max_size_accepts_auto_and_suffixes():
    assert parse_pcie_oneshot_max_size(None) is None
    assert parse_pcie_oneshot_max_size("auto") is None
    assert parse_pcie_oneshot_max_size("64KB") == 64 * 1024
    assert parse_pcie_oneshot_max_size("2m") == 2 * 1024 * 1024
    assert parse_pcie_oneshot_max_size(4096) == 4096


def test_compute_crossover_size_runs_fine_sweep():
    seen_sizes = []

    def benchmark(size_bytes: int) -> tuple[float, float]:
        seen_sizes.append(size_bytes)
        if size_bytes <= 48 * 1024:
            return 1.0, 2.0
        return 3.0, 2.0

    crossover, results = _compute_crossover_size(
        benchmark,
        ceiling_bytes=64 * 1024,
        fine_step_bytes=8 * 1024,
    )

    assert crossover == 48 * 1024
    assert 40 * 1024 in seen_sizes
    assert 48 * 1024 in seen_sizes
    assert 56 * 1024 in seen_sizes
    assert results[-1].size_bytes == 64 * 1024


def test_register_buffer_is_idempotent_for_same_mapping():
    runtime = _make_runtime()
    ext = runtime._ext

    runtime.register_buffer((111, 222))
    runtime.register_buffer((111, 222))

    assert ext.register_buffer_calls == [(12345, (111, 222))]


def test_register_buffer_rejects_mismatched_mapping_for_same_local_ptr():
    runtime = _make_runtime()

    runtime.register_buffer((111, 222))

    with pytest.raises(ValueError, match="already registered"):
        runtime.register_buffer((111, 333))


def test_all_reduce_registers_explicit_peer_ptrs_once():
    runtime = _make_runtime()
    ext = runtime._ext
    inp = torch.arange(8, dtype=torch.bfloat16)

    out0 = runtime.all_reduce(inp, peer_input_ptrs=(inp.data_ptr(), 222))
    out1 = runtime.all_reduce(inp, peer_input_ptrs=(inp.data_ptr(), 222))

    assert torch.equal(out0, inp)
    assert torch.equal(out1, inp)
    assert ext.register_buffer_calls == [(12345, (inp.data_ptr(), 222))]
    assert len(ext.all_reduce_calls) == 2


def test_all_reduce_requires_registration_without_eager_buffers():
    runtime = _make_runtime()
    inp = torch.arange(8, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="peer_input_ptrs are required"):
        runtime.all_reduce(inp)


def test_eager_buffers_allow_all_reduce_without_peer_ptrs():
    runtime = _make_runtime(eager=True)
    ext = runtime._ext
    inp = torch.arange(8, dtype=torch.bfloat16)

    out = runtime.all_reduce(inp)

    assert torch.equal(out, inp)
    assert ext.register_pcie_buffers_calls == [(12345, (200, 201), (300, 301))]
    assert ext.register_buffer_calls == []
    assert len(ext.all_reduce_calls) == 1


def test_runtime_rejects_reuse_from_another_stream_key(monkeypatch):
    runtime = _make_runtime(eager=True)
    inp = torch.arange(8, dtype=torch.bfloat16)
    state = {"stream_key": 11}

    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot._current_stream_key",
        lambda device, stream=None: state["stream_key"],
    )

    runtime.all_reduce(inp)
    state["stream_key"] = 22

    with pytest.raises(RuntimeError, match="stream-affine"):
        runtime.all_reduce(inp)


def test_should_allreduce_checks_device_dtype_size_alignment_and_contiguity():
    runtime = _make_runtime(max_size=16)

    good = torch.arange(8, dtype=torch.bfloat16)
    assert runtime.should_allreduce(good) is True
    assert runtime.should_allreduce(torch.arange(4, dtype=torch.int32)) is False
    assert runtime.should_allreduce(torch.arange(16, dtype=torch.bfloat16)) is False
    assert runtime.should_allreduce(torch.arange(7, dtype=torch.bfloat16)) is False
    assert runtime.should_allreduce(torch.arange(16, dtype=torch.bfloat16)[::2]) is False


def test_graph_buffer_api_exposes_explicit_registration_hooks():
    runtime = _make_runtime()
    ext = runtime._ext

    assert runtime.get_graph_buffer_ipc_meta() == ([1, 2, 3], [0, 64])

    runtime.register_graph_buffers_from_ranks(
        ([1, 2, 3], [4, 5, 6]),
        ([0, 64], [8, 72]),
    )

    assert ext.register_graph_buffers_calls == [
        (12345, [[1, 2, 3], [4, 5, 6]], [[0, 64], [8, 72]])
    ]


def test_register_graph_buffers_uses_exchange_group_broadcast(monkeypatch):
    remote_meta = {
        0: ([1, 2, 3], [0, 64]),
        1: ([9, 8, 7], [16, 80]),
    }

    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr("torch.distributed.get_process_group_ranks", lambda group=None: [0, 1])
    monkeypatch.setattr("b12x.distributed.pcie_oneshot._object_broadcast_device", lambda group: "cpu")

    def fake_broadcast(object_list, src, group=None, device=None):
        object_list[0] = remote_meta[src]

    monkeypatch.setattr("torch.distributed.broadcast_object_list", fake_broadcast)

    runtime = _make_runtime(exchange_group=object())
    ext = runtime._ext
    runtime.register_graph_buffers()

    assert ext.register_graph_buffers_calls == [
        (12345, [[1, 2, 3], [9, 8, 7]], [[0, 64], [16, 80]])
    ]


def test_register_graph_buffers_noops_when_no_rank_registered_buffers(monkeypatch):
    monkeypatch.setattr("torch.distributed.get_world_size", lambda group=None: 2)
    monkeypatch.setattr("torch.distributed.get_rank", lambda group=None: 0)
    monkeypatch.setattr("torch.distributed.get_process_group_ranks", lambda group=None: [0, 1])
    monkeypatch.setattr("b12x.distributed.pcie_oneshot._object_broadcast_device", lambda group: "cpu")
    monkeypatch.setattr(
        "torch.distributed.broadcast_object_list",
        lambda object_list, src, group=None, device=None: object_list.__setitem__(0, ([], [])),
    )

    runtime = _make_runtime(exchange_group=object())
    ext = runtime._ext
    ext.handle_bytes = []
    ext.offsets = []

    runtime.register_graph_buffers()

    assert ext.register_graph_buffers_calls == []


def test_capture_registers_graph_buffers_after_context(monkeypatch):
    runtime = _make_runtime(exchange_group=object())
    calls = []

    monkeypatch.setattr(runtime, "register_graph_buffers", lambda: calls.append("registered"))

    with runtime.capture():
        pass

    assert calls == ["registered"]


def test_eager_capture_skips_graph_buffer_registration(monkeypatch):
    runtime = _make_runtime(eager=True, exchange_group=object())
    calls = []

    monkeypatch.setattr(runtime, "register_graph_buffers", lambda: calls.append("registered"))

    with runtime.capture():
        pass

    assert calls == []


def test_pool_creates_distinct_channels_per_stream_key(monkeypatch):
    created = []

    def make_channel(stream_key):
        runtime = _make_runtime(eager=True)
        created.append((stream_key, runtime))
        return runtime

    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=make_channel,
    )

    monkeypatch.setattr(
        "b12x.distributed.pcie_oneshot._current_stream_key",
        lambda device, stream=None: 7 if stream is None else int(stream),
    )

    ch7 = pool.for_stream()
    ch8 = pool.for_stream(8)

    assert pool.for_stream() is ch7
    assert pool.for_stream(8) is ch8
    assert ch7 is not ch8
    assert [entry[0] for entry in created] == [7, 8]


def test_pool_requires_precreated_channel_during_capture(monkeypatch):
    pool = PCIeOneshotAllReducePool(
        rank=0,
        world_size=2,
        device=torch.device("cpu"),
        channel_factory=lambda stream_key: _make_runtime(eager=True),
    )

    monkeypatch.setattr("b12x.distributed.pcie_oneshot._current_stream_key", lambda device, stream=None: 7)
    monkeypatch.setattr("b12x.distributed.pcie_oneshot._is_current_stream_capturing", lambda device: True)

    with pytest.raises(RuntimeError, match="before capture starts"):
        pool.for_stream()

    pool._channels[7] = _make_runtime(eager=True)

    assert pool.for_stream() is pool._channels[7]
