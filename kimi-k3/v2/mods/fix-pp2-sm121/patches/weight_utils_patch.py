#!/usr/bin/env python3
"""Patch weight_utils.py: use TP group instead of WORLD for weight loading.

Under PP2, the draft model loads only on PP1 ranks. InstantTensor and
fastsafetensors broadcast on group.WORLD (all ranks), but PP0 isn't
loading the draft → NCCL broadcast deadlock (Issue #50959).

Fix: scope the process group to get_tp_group(), which only includes
ranks within the current PP stage.

Idempotent: skips if already patched.
"""
import sys
from pathlib import Path

WEIGHT_UTILS = Path(
    "/opt/kimi-k3/vllm/vllm/model_executor/model_loader/weight_utils.py"
)


def main() -> None:
    src = WEIGHT_UTILS.read_text()

    if "get_tp_group" in src and "fix-dspark-pp2-weight-loading" in src:
        print("[weight_utils_patch] already patched")
        return

    # 1. Patch instanttensor_weights_iterator: get_world_group() → get_tp_group()
    old_instant = """    try:
        world_group = get_world_group()
    except AssertionError:
        # Entering here only in unit tests where the world group is not initialized.
        process_group = None
    else:
        process_group = world_group.device_group if world_group.world_size > 1 else None
"""
    new_instant = """    # fix-dspark-pp2-weight-loading: use TP group instead of WORLD so
    # draft model loading under PP doesn't deadlock on broadcast (Issue #50959).
    try:
        from vllm.distributed.parallel_state import get_tp_group as _get_tp_group
        _tp_group = _get_tp_group()
    except (AssertionError, ImportError):
        process_group = None
    else:
        process_group = _tp_group.device_group if _tp_group.world_size > 1 else None
"""
    if old_instant in src:
        src = src.replace(old_instant, new_instant, 1)
        print("[weight_utils_patch] patched instanttensor_weights_iterator")
    else:
        print("[weight_utils_patch] WARNING: instanttensor anchor not found")

    # 2. Patch fastsafetensors_weights_iterator: group.WORLD → get_tp_group()
    old_fsst = """    if torch.distributed.is_initialized():
        pg = torch.distributed.group.WORLD
    else:
        pg = SingleGroup()
"""
    new_fsst = """    # fix-dspark-pp2-weight-loading: use TP group instead of WORLD
    if torch.distributed.is_initialized():
        from vllm.distributed.parallel_state import get_tp_group as _get_tp_group
        pg = _get_tp_group().device_group
    else:
        pg = SingleGroup()
"""
    if old_fsst in src:
        src = src.replace(old_fsst, new_fsst, 1)
        print("[weight_utils_patch] patched fastsafetensors_weights_iterator")
    else:
        print("[weight_utils_patch] WARNING: fastsafetensors anchor not found")

    WEIGHT_UTILS.write_text(src)
    print("[weight_utils_patch] applied successfully")


if __name__ == "__main__":
    main()
