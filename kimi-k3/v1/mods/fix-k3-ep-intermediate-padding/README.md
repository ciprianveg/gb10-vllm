# fix-k3-ep-intermediate-padding

Make `KimiMoE.__init__` intermediate-size padding expert-parallel-aware so
`--enable-expert-parallel` actually reduces per-rank MoE memory from ~118.4
GiB to ~89.1 GiB on K3 TP16 (fits the 121 GiB GB10 UMA pool).

## The bug

`vllm/models/kimi_k3/nvidia/model.py` `KimiMoE.__init__` computes
`padded_moe_intermediate_size` from `moe_intermediate_size // self.tp_size`
and pads up to `min_moe_intermediate_per_partition (256) * self.tp_size` when
the per-partition value is below 256. `self.tp_size` is
`get_tensor_model_parallel_world_size()` (global TP), so this logic is
**EP-unaware**.

Under `--enable-expert-parallel` (ep_size=16) FusedMoE gives each rank the
FULL intermediate per local expert (EP shards experts, not the intermediate).
With tp=16 and `moe_intermediate_size=3072`, the per-partition value is
192 < 256, so the model pads to 4096. Each of the 56 local experts then
receives the full padded 4096, and:

```
56 local experts * 4096 = 229,376 inter-units/rank
896 experts * 256       = 229,376 inter-units/rank (no-EP)
```

Byte-for-byte identical. The EP memory savings are annihilated: per-rank MoE
footprint stays at ~118.4 GiB instead of the ~89.1 GiB EP would deliver, and
the 121 GiB GB10 UMA pool is exhausted during weight load
(`NVRM NV_ERR_NO_MEMORY` -> driver wedge -> node reboot).

Verified on the 16-node cluster with `--enable-expert-parallel` deployed:
the launch log confirmed EP groups were created
(`[EP Rank 0/16] Local/global number of experts: 56/896`) but `check-mem`
showed all 15 fully-loaded nodes at ~120 GiB used / ~1 GiB available — the
replicated/padded footprint, not the EP footprint.

## The fix

Skip the `min_moe_intermediate_per_partition` padding entirely when
`vllm_config.parallel_config.enable_expert_parallel` is True. Under EP the
intermediate is not TP-sharded, so the padding is both unnecessary (3072 is
already a valid kernel size) and harmful (it inflates per-rank memory). With
the fix:

```
56 local experts * 3072 = 172,032 inter-units/rank
                        = ~0.969 GiB/layer * 92 layers
                        = ~89.1 GiB/rank
```

That leaves ~25-30 GiB headroom per node for KV cache and the drafter.

The downstream `if padded != moe_intermediate_size` block (which zeroes the
padded region and records `intermediate_size_per_partition_unpadded =
moe_intermediate_size // self.tp_size`) does NOT execute when padded stays at
3072. `FusedMoEConfig.__post_init__` defaults
`intermediate_size_per_partition_unpadded = intermediate_size_per_partition`,
which under EP with `tp_size=1` is 3072 — correct, no mismatch.

## What this mod does NOT change

- No-EP path (TP16 without `--enable-expert-parallel`): the original padding
  block runs unchanged. `padded_moe_intermediate_size` still becomes 4096.
- MegaMoE path (`--moe-backend deep_gemm_mega_moe`): the same guard applies;
  under EP the padding is skipped, under no-EP it runs. The MegaMoE backend
  already requires EP (`KimiMoE.__init__` raises `NotImplementedError` if
  EP is missing), so this mod's guard is always active for MegaMoE.
- Weight shapes: under EP with the fix, per-local-expert shapes are
  `(3072, 1792)` for w1/w3 and `(3584, 1536)` for w2 — exactly matching the
  checkpoint (`experts.N.w{1,2,3}.weight_packed` are stored at the
  unpadded 3072 intermediate). No zero-fill of a padded region is needed.

## Why a local mod instead of a vLLM rebuild

Upstream `vllm-project/vllm` `main` (verified by fetching the raw file) has
**byte-for-byte identical padding logic** — the bug is not fixed in any
merged PR as of 2026-08-03. The K3 perf PRs in
[vllm-project/vllm#50587](https://github.com/vllm-project/vllm/issues/50587)
are compute/throughput optimizations (dspark fused kv, addmm inplace, ROCm
decode gates, latent up-proj shard) or memory wins gated behind paths we
can't use on GB10 (shared-expert sharding requires
`deep_gemm_mega_moe` + sequence parallelism; DeepGEMM is sm100-only). A
rebuild does not fix the OOM.

## Applies to

- `vllm-node-kimi3` image (vLLM `0.26.1rc1.dev160+g2ac91211d.d20260730`).
- Any future image build that still ships the un-patched
  `vllm/models/kimi_k3/nvidia/model.py` KimiMoE padding block.

## Idempotency

The patch is idempotent: `patch_model.py` checks for the marker comment
`# fix-k3-ep-intermediate-padding:` in the file and skips if already
applied. Re-running the mod is safe.

## Verification

- `python3 -c "import ast; ast.parse(open('.../model.py').read())"` — syntax
  check (run by `run.sh`).
- `grep -q 'fix-k3-ep-intermediate-padding' model.py` — guard present (run
  by `run.sh`).
- End-to-end: relaunch the cluster via `./run-recipe.sh kimi-k3-full-hh-b12x-tp16.yaml`
  (eugr harness) and watch `check-mem` — expect ~90-97 GiB used per node (25-30 GiB free)
  instead of the prior ~120 GiB / ~1 GiB free.
