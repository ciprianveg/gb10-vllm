#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Idempotent patch: skip spec-decode warmup decode steps that trigger CUDA
# indexSelectSmallIndex asserts in Kimi-K3's model + KDA metadata builder.
#
# Two known crash sites (different vLLM versions):
# 1. Optimized image (vLLM main @72cd5424d): kda_metadata.py:342
#    torch.repeat_interleave with mismatched query_lens/spec_sequence_masks.
# 2. Old image (vLLM 0.26.1rc1.dev160): model.py forward pass during warmup
#    with spec decode — Triton kernel index out of bounds.
#
# Both are triggered by the warmup's synthetic SchedulerOutput for spec-decode
# batches. The fix: skip ALL spec-decode warmup decode steps. The non-spec
# warmup path still runs normally. First spec-decode batch at inference time
# JIT-compiles on the fly (minor one-time cost, no correctness impact).
#
# Root cause: vLLM warmup.py + K3 model mismatch. Related upstream:
# vllm-project/vllm#47029 (same assert class, embedding path),
# #50851 (DSpark nightly), #50001 (K3 tracking). No upstream fix exists yet.
import ast
import sys
from pathlib import Path

MARKER = "# fix-k3-dspark-warmup:"

OLD = """        # Decode steps to warm, as (request indices, per-request spec flag).
        # Under spec decoding the scheduler drops requests the drafter proposed
        # nothing for, so warm each batch shape with and without draft tokens.
        decode_steps: list[tuple[list[int], list[bool]]] = [
            (all_indices, [use_spec_decode] * num_reqs),
        ]
        if num_reqs >= 2:
            # Mixed spec / non-spec: GDN and KDA reclassify the non-spec decode
            # as a prefill and split the batch into spec/non-spec token indices.
            decode_steps.append(([0, 1], [use_spec_decode, False]))
            if use_spec_decode:
                # Exercise the model paths that split a batch by whether each
                # request received draft tokens.
                decode_steps.append(([0, 1], [False, False]))
        if num_reqs > 1:
            decode_steps.append(([0], [use_spec_decode]))
            if use_spec_decode:
                decode_steps.append(([0], [False]))
        elif use_spec_decode:
            decode_steps.append(([0], [False]))"""

NEW = """        # fix-k3-dspark-warmup: skip ALL spec-decode warmup decode steps.
        # Kimi-K3's model + KDA metadata builder crash with indexSelectSmallIndex
        # asserts during spec-decode warmup (different crash sites per vLLM
        # version: kda_metadata.py:342 in main @72cd5424d, model.py forward in
        # 0.26.1rc1.dev160). Only non-spec warmup steps run; spec-decode paths
        # JIT-compile on first inference batch (minor one-time cost).
        decode_steps: list[tuple[list[int], list[bool]]] = []
        if num_reqs >= 2:
            decode_steps.append(([0, 1], [False, False]))
        if num_reqs > 1:
            decode_steps.append(([0], [False]))
        elif not use_spec_decode:
            decode_steps.append(([0], [False]))"""


def main() -> int:
    try:
        from vllm import __file__ as vllm_init
    except Exception as e:
        print(f"ERROR: cannot locate vllm install: {e}", file=sys.stderr)
        return 1
    vllm_dir = Path(vllm_init).parent
    target = vllm_dir / "v1" / "worker" / "gpu" / "warmup.py"
    if not target.exists():
        print(f"ERROR: {target} not found", file=sys.stderr)
        return 1

    text = target.read_text()
    if MARKER in text:
        print(f"{target}: already patched, skipping")
        return 0

    if OLD not in text:
        print(f"ERROR: expected warmup decode_steps block not found in {target}", file=sys.stderr)
        print("--- context around 'decode_steps' ---")
        for i, line in enumerate(text.splitlines(), start=1):
            if "decode_steps" in line:
                print(f"{i}: {line}")
        return 1

    updated = text.replace(OLD, NEW, 1)
    try:
        ast.parse(updated)
    except SyntaxError as e:
        print(f"ERROR: patched file has syntax error: {e}", file=sys.stderr)
        return 1

    target.write_text(updated)
    print(f"{target}: applied fix-k3-dspark-warmup (skip all spec-decode warmup)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
