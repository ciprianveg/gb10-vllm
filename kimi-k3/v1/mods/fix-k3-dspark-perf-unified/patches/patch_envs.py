#!/usr/bin/env python3
"""Patch vllm/envs.py: add DSpark perf-unified env vars.

Adds the env-var declarations and resolver lambdas introduced by the
five cherry-picked commits from codex/hh-kimi-k3-unified-20260805:

  - VLLM_KIMI_K3_B12X_DSPARK_ARGMAX        (commit 03f9c5719a78)
  - VLLM_DSPARK_PREFER_B12X_ALLREDUCE_RMS  (commit aacd3aab5500)
  - VLLM_DSPARK_CAPTURE_SHARDED_MARKOV     (commit 142d21a06326)
  - VLLM_KIMI_PRELAUNCH_SHARED_EXPERTS     (commit 27399cda9b52)
  - VLLM_KIMI_USE_B12X_BATCHED_PROJECTION_TOPK (dependency of commit
    49c1d79eed6e; referenced by the unified dcp_alltoall.py warmup path)
  - VLLM_DSPARK_REPLICATE_MARKOV_W1        (sibling declaration present
    on the unified branch; harmless default-off)

All default to off, so the mod is a no-op unless the operator opts in.

Idempotent: skips any var that is already declared.
"""
import sys
from pathlib import Path

ENVS = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/envs.py"
)

# (var_name, annotation_line, resolver_block)
# annotation_line / resolver_block are inserted as-is after the anchor.
DSPARK_VARS = [
    (
        "VLLM_DSPARK_REPLICATE_MARKOV_W1",
        "    VLLM_DSPARK_REPLICATE_MARKOV_W1: bool = False\n",
        '    "VLLM_DSPARK_REPLICATE_MARKOV_W1": lambda: bool(\n'
        '        int(os.getenv("VLLM_DSPARK_REPLICATE_MARKOV_W1", "0"))\n'
        "    ),\n",
    ),
    (
        "VLLM_KIMI_K3_B12X_DSPARK_ARGMAX",
        "    VLLM_KIMI_K3_B12X_DSPARK_ARGMAX: bool = False\n",
        '    "VLLM_KIMI_K3_B12X_DSPARK_ARGMAX": lambda: bool(\n'
        '        int(os.getenv("VLLM_KIMI_K3_B12X_DSPARK_ARGMAX", "0"))\n'
        "    ),\n",
    ),
    (
        "VLLM_DSPARK_PREFER_B12X_ALLREDUCE_RMS",
        "    VLLM_DSPARK_PREFER_B12X_ALLREDUCE_RMS: bool = False\n",
        '    "VLLM_DSPARK_PREFER_B12X_ALLREDUCE_RMS": lambda: bool(\n'
        '        int(os.getenv("VLLM_DSPARK_PREFER_B12X_ALLREDUCE_RMS", "0"))\n'
        "    ),\n",
    ),
    (
        "VLLM_DSPARK_CAPTURE_SHARDED_MARKOV",
        "    VLLM_DSPARK_CAPTURE_SHARDED_MARKOV: bool = False\n",
        '    "VLLM_DSPARK_CAPTURE_SHARDED_MARKOV": lambda: bool(\n'
        '        int(os.getenv("VLLM_DSPARK_CAPTURE_SHARDED_MARKOV", "0"))\n'
        "    ),\n",
    ),
]

KIMI_VARS = [
    (
        "VLLM_KIMI_USE_B12X_BATCHED_PROJECTION_TOPK",
        "    VLLM_KIMI_USE_B12X_BATCHED_PROJECTION_TOPK: bool = False\n",
        '    "VLLM_KIMI_USE_B12X_BATCHED_PROJECTION_TOPK": lambda: bool(\n'
        '        int(os.getenv("VLLM_KIMI_USE_B12X_BATCHED_PROJECTION_TOPK", "0"))\n'
        "    ),\n",
    ),
    (
        "VLLM_KIMI_PRELAUNCH_SHARED_EXPERTS",
        "    VLLM_KIMI_PRELAUNCH_SHARED_EXPERTS: bool = False\n",
        '    "VLLM_KIMI_PRELAUNCH_SHARED_EXPERTS": lambda: bool(\n'
        '        int(os.getenv("VLLM_KIMI_PRELAUNCH_SHARED_EXPERTS", "0"))\n'
        "    ),\n",
    ),
]

# Anchors: insert DSPARK vars after SHARD_MARKOV_HEAD; KIMI vars after
# PAIRED_PROJECTION_TOPK. Both the class-annotation anchor and the
# resolver anchor are the *last line* of the preceding entry.
CLASS_ANCHOR_DSPARK = "    VLLM_DSPARK_SHARD_MARKOV_HEAD: bool = False\n"
CLASS_ANCHOR_KIMI = "    VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK: bool = False\n"
RESOLVER_ANCHOR_DSPARK = (
    '    "VLLM_DSPARK_SHARD_MARKOV_HEAD": lambda: bool(\n'
    '        int(os.getenv("VLLM_DSPARK_SHARD_MARKOV_HEAD", "0"))\n'
    "    ),\n"
)
RESOLVER_ANCHOR_KIMI = (
    '    "VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK": lambda: bool(\n'
    '        int(os.getenv("VLLM_KIMI_USE_B12X_PAIRED_PROJECTION_TOPK", "0"))\n'
    "    ),\n"
)


def _insert_block(
    src: str, anchor: str, block: str, present_marker: str, label: str
) -> str:
    """Insert `block` immediately after the *first* occurrence of `anchor`.

    `present_marker` is a distinctive substring of `block` used for the
    idempotency check (so adding the class annotation does not fool the
    resolver insertion into skipping).
    """
    if present_marker in src:
        print(f"[patch_envs] {label}: already present, skipping")
        return src
    idx = src.find(anchor)
    if idx == -1:
        print(f"[patch_envs] ERROR: anchor not found for {label}", file=sys.stderr)
        sys.exit(1)
    end = idx + len(anchor)
    return src[:end] + block + src[end:]


def main() -> None:
    src = ENVS.read_text()

    # Class annotations — idempotency marker is the annotation line itself.
    for var_name, ann, _ in DSPARK_VARS:
        src = _insert_block(src, CLASS_ANCHOR_DSPARK, ann, ann.strip(), f"{var_name} (ann)")
    for var_name, ann, _ in KIMI_VARS:
        src = _insert_block(src, CLASS_ANCHOR_KIMI, ann, ann.strip(), f"{var_name} (ann)")

    # Resolver lambdas — idempotency marker is the resolver's dict key.
    for var_name, _, res in DSPARK_VARS:
        marker = f'"{var_name}": lambda'
        src = _insert_block(src, RESOLVER_ANCHOR_DSPARK, res, marker, f"{var_name} (resolver)")
    for var_name, _, res in KIMI_VARS:
        marker = f'"{var_name}": lambda'
        src = _insert_block(src, RESOLVER_ANCHOR_KIMI, res, marker, f"{var_name} (resolver)")

    ENVS.write_text(src)
    print("[patch_envs] applied successfully")


if __name__ == "__main__":
    main()
