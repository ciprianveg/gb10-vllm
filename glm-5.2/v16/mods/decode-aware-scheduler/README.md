# Decode-Aware Custom Scheduler

Prevents long-prefill requests from blocking decode streams under concurrent load.

## Problem

With chunked prefill enabled, multiple concurrent prefill requests can starve
ongoing decode streams. A single long prefill (e.g. 8K tokens) occupies the
GPU for the entire chunk duration, and decode tokens stop flowing until the
prefill chunk completes. Under DCP>1 this is severe (decode drops to 0.0–0.2
tok/s); under DCP=1 it is milder but still causes multi-second stalls.

## Solution

Dynamic prefill budgets that adapt to decode activity:

- **No active decode** → prefill gets the full idle budget (default 16384 tokens
  per step, capped by `--max-num-batched-tokens`). Prefill throughput stays high.
- **Active decode** → all prefill requests share a small decode budget (default
  1024, tunable). Decode keeps progressing at reduced but non-zero throughput.
- **At most 1 long prefill per step** → round-robin selection by least-recently-
  scheduled age, so no single prefill monopolizes the budget.
- **Idle budget returns instantly** when decode finishes → no manual tuning.

The feature is a scheduler-level change. It does not increase KV-cache capacity,
maximum context length, or concurrency. It can be disabled at runtime without
removing the patch.

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--enable-decode-aware-prefill` | false | Enable decode-aware prefill scheduling |
| `--decode-prefill-token-budget N` | 1024 | Max aggregate prefill tokens per step when decode is active |
| `--idle-prefill-token-budget N` | 16384 | Max aggregate prefill tokens per step when no decode is active (capped by `--max-num-batched-tokens`) |
| `--max-long-prefills-per-step N` | 1 | Max long-prefill requests selected per step |

**Required companions:**
- `--enable-chunked-prefill` (already enabled in production)
- `--long-prefill-token-threshold 2048` (must be > 0; controls what counts as "long")

## Tuning Guide

The `--decode-prefill-token-budget` controls the tradeoff between decode
responsiveness and prefill completion time under pressure:

| Budget | Decode stall (approx.) | Use case |
|-------:|------------------------:|----------|
| 1024   | ~1.6–2.0s               | Forum default (DCP4) |
| 512    | ~1.0s                   | Balanced |
| 256    | ~0.5s                   | Decode-priority (chat/agent) |

Lower budgets = shorter decode stalls but slower prefill completion. The idle
budget is unaffected, so solo prefill throughput stays at `--max-num-batched-tokens`.

**Recommendation:** Start at 256 for interactive/agent workloads where decode
latency matters most. Raise to 512–1024 for batch workloads where prefill
completion time matters more.

## Production Configuration (this project)

```
--max-num-batched-tokens 4096
--long-prefill-token-threshold 2048
--enable-decode-aware-prefill
--decode-prefill-token-budget 256
--idle-prefill-token-budget 16384
--max-long-prefills-per-step 1
```

With `--max-num-batched-tokens 4096`, the idle budget (16384) is effectively
capped at 4096. To fully utilize the idle budget, raise `--max-num-batched-tokens`
to 16384 (requires GPU memory re-tuning).

## Rollback

Remove `--enable-decode-aware-prefill` from the serve command (or set it to
`false`). The patched code path is not entered when disabled. The patch can
remain installed.

## Patch Compatibility

The patch modifies 5 vLLM files (3 source, 2 test):

| File | Changes |
|------|---------|
| `vllm/config/scheduler.py` | 4 new `SchedulerConfig` fields + validation |
| `vllm/engine/arg_utils.py` | 4 new CLI flags + `EngineArgs` fields |
| `vllm/v1/core/sched/scheduler.py` | Dynamic budget logic, round-robin selection, prefill tracking |
| `tests/v1/core/test_scheduler.py` | 3 new behavior tests |
| `tests/v1/core/utils.py` | Test helper parameters |

**Validated baseline:** `local-inference-lab/vllm@a663653d` (branch `spark4-overlay`)

**This mod's adaptation:** The v16 fork (`fathomless-firmament-v16-unified`)
has a spec-decode padding block in `scheduler.py` not present in the original
patch baseline. The `scheduler.py` changes are applied via a Python script
(`apply_scheduler_patch.py`) with targeted string replacements instead of a
unified diff, making them robust against line-number shifts and fork-specific
modifications. The other two files apply cleanly via `patch -p1`.

## Credits

**Original patch and on-hardware validation:** [penguinchang](https://forums.developer.nvidia.com/u/penguinchang)
on the [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/glm-5-2-int4-int8-on-8x-gb10-1-200-t-s-prefill-33-54-t-s-avg-decode-generic-coding-structured/376831)
(July 15, 2026). The patch bundle, test report, and tuning data are from
penguinchang's post and accompanying `glm52-decode-aware-custom-scheduler-patch-20260715.tar.gz`.

**v16 fork adaptation:** This mod's `apply_scheduler_patch.py` adapts the
scheduler changes for the `fathomless-firmament-v16-unified` fork's spec-decode
padding block, which caused one hunk in the original unified diff to fail
context matching.
