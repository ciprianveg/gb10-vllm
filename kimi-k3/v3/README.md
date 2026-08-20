# KIMI-K3 on 16× GB10 DGX Spark — v3 (sm121 image + MoE fusion + RedHat DSpark)

**Recommended Kimi-K3 solution.** Full [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
(MXFP4 experts, BF16 attention) served with the sm121
`ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v3-sm121` image and **RedHat DSpark**
speculative decoding on 16-node DGX Spark GB10 (sm121) clusters.

| Version | Stack | Status |
|---------|-------|--------|
| **v3** | v2 image + vLLM@0232bce6 overlay (MoE fusion #385/#386) + 11 mods baked | **Current production** 🚀 |
| v2 | `vllm-node-kimi3-sm121` (vLLM@881ac39 + B12X) + RedHat DSpark, runtime mods | [Superseded](../v2/README.md) |
| v1 | `vllm-node-kimi3-hh` (B12X_MLA + Inferact DSpark) | [Historic artifact](../v1/README.md) |

v3 is a thin source overlay on the published v2 image: it advances the vLLM tree
to `0232bce6` (MoE projection-transport fusion, KDA/DCP fixes) and bakes the v2
runtime mods into the image, so the recipe needs **no runtime mods** at all.

## What's new in v3

- **Faster decode**: MoE projection-transport fusion (#385/#386) removes 92
  kernel launches per decode step. Combined with the shm busy-wait fix (#421)
  this gives **~+10% no-spec decode** and **~+5-8% under DSpark** vs v2.
- **Long-context loop fixed**: the NaN/generation-loop bug observed past
  **~270K context** under DCP is fixed by keeping the recurrent KDA
  token-position cache unsharded under DCP — a fix found and upstreamed by the
  [`local-inference-lab/vllm`](https://github.com/local-inference-lab/vllm)
  project as **PR #418**, which v3 carries natively. Credit for identifying and
  fixing this issue goes to that repository's authors.
- **2M total context capacity**: the TP16+DCP8 recipe holds ~2M tokens of KV —
  4 concurrent requests can each use a 500K context.
- **No runtime mods**: recipes run with `mods: []`.

## Quick Start

```bash
# 1. Pull the prebuilt image (linux/arm64, sm121) — on every node
docker pull ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v3-sm121

# 2. Deploy from spark-vllm-docker (weights already on NFS — no download needed)
./run-recipe.sh <gb10-vllm>/kimi-k3/v3/recipes/kimi-k3-dcp-tp16-prd.yaml
```

> First boot JIT-compiles CuTe DSL + Triton kernels per batch shape — expect a
> slow first ~10-20 requests, then full speed. Mount persistent cache dirs
> (see [Cache requirements](#cache-requirements)) to avoid re-JIT on restart.

### Build the image yourself (optional)

Short reproducible build guide (thin overlay on the published v2 image, no CUDA
recompile): [`BUILD-SM121-IMAGE.md`](BUILD-SM121-IMAGE.md)

```bash
./kimi-k3/v3/build.sh            # build local tag
./kimi-k3/v3/build.sh --push     # build + push to GHCR
```

## Recipes

| Recipe | TP | DCP | Draft | Context | Notes |
|---|---|---|---|---|---|
| [`kimi-k3-dcp-tp16-prd.yaml`](recipes/kimi-k3-dcp-tp16-prd.yaml) | 16 | 8 | RedHat DSpark (Qwen3-GQA), s6 | **2M total** (4 × 500K concurrent) | **Recommended** — improved long-context decode speed |

The recipe splits the MLA latent KV 8× across ranks (DCP8) so the B12X_MLA
spec-verify flattening re-reads KV/8 per row instead of the full prefix. This
directly targets the long-context decode decline (KV re-read bound at 100K+).

Model weights default to `/root/models/models115/Kimi-K3`, the RedHat DSpark
draft to `/root/models/models11/RedHatAI-Kimi-K3-dspark`; both are
`{model_path}` / `{draft_model_path}` defaults you can override.

## Benchmarks (TP16 + DCP8, 16× GB10)

From llama-benchy (coherent corpus, tg=1500, up to 300K context). This is a
general-purpose corpus, so DSpark acceptance is on the lower side:

| test | t/s | peak t/s | ttfr (ms) | est_ppt (ms) | e2e_ttft (ms) |
|:---|---:|---:|---:|---:|---:|
| pp2048 @ d4000 | **675.10** | — | 8,260.59 | 8,256.52 | 8,260.59 |
| tg1500 @ d4000 | **19.82** | 35.00 | — | — | — |
| pp2048 @ d50000 | **748.43** | — | 62,921.28 | 62,917.21 | 62,921.28 |
| tg1500 @ d50000 | **20.09** | 39.00 | — | — | — |
| pp2048 @ d100000 | **712.79** | — | 129,595.93 | 129,591.86 | 129,600.92 |
| tg1500 @ d100000 | **17.41** | 37.00 | — | — | — |
| pp2048 @ d200000 | **657.64** | — | 277,772.00 | 277,767.93 | 277,780.66 |
| tg1500 @ d200000 | **15.66** | 33.00 | — | — | — |
| pp2048 @ d300000 | **607.73** | — | 449,404.11 | 449,400.05 | 449,417.24 |
| tg1500 @ d300000 | **14.85** | 28.00 | — | — | — |

Single-stream coding benchmark (temp=0, prompt 125 tok, tg=1500) — coding
workloads see higher DSpark acceptance than the coherent corpus above:

```
Completion tokens : 1500
Prompt tokens     : 125
Wall time         : 59.09s
Decode tok/s      : 25.38
```

## Mods baked into the image

All v2 runtime mods are baked in, plus the vLLM source tree is advanced to
`0232bce6`. The image runs with `mods: []`.

| Mod | Upstream PR | What it does |
|---|---|---|
| `pr51508-stale-zero-accept` | #51508 | Skip GDN/KDA recurrent-state updates for stale (zero-accept) spec rows |
| `pr51036-repetition-detection` | #51036 | Allowlist `repetition_detection` override-generation config (mitigates KDA-NaN loops) |
| `pr46324-piecewise-spec-capture` | #46324 | Align spec-decode CUDA-graph capture sizes in PIECEWISE mode |
| `pr50169-drafter-kv-pool` | #50169 | Dedicated KV groups for sliding-window drafters (GB10 pool 415k → 871k) |
| `pr46932-uma-cudagraph-mem` | #46932 | Fix CUDA-graph memory accounting on unified memory (GB10) |
| `pr47926-dspark-prefix-mask` | #47926 | Mask prefix-cache-restored tokens out of the DSpark draft context |
| `fix-mamba-cow-external-hit` | #395 | Preserve Mamba/KDA copy-on-write after external prefix-cache hits |
| `fix-k3-renderer-split-turns` | #395 | Merge split Kimi-K3 prose+tool_calls turns; preserve reasoning |
| `fix-dcp-hybrid-prefix-cache-hits` | #401 | Allow hash-aligned DCP hybrid prefix-cache hits (prefill savings under DCP) |
| `fix-dcp-recurrent-state-unsharded` | #418 | Keep recurrent KDA token-position state unsharded under DCP — fixes the >270K-context loop |
| `perf-shm-busy-loop-2ms` | #421 | Shm broadcast busy-wait 1s → 2ms (frees GB10 CPU thermal headroom; +3.2% decode) |

Key PRs arriving with the `0232bce6` tree itself:

| PR | What it does |
|---|---|
| #385/#386 | Fuse Kimi routed-MoE projection transport — removes 92 kernel launches per decode step |
| #387 | Execute Kimi dense MLA with DCP |
| #388/#389/#390 | DSpark runtime support: external drafts, bounded/sharded draft state, vocab-sharded sampling |
| #413 | Stream Kimi-K3 tool arguments incrementally |
| #414 | Preserve Mamba CoW after external prefix hits (in-tree version of the #395 mod) |
| #415 | Preserve the DSpark CUDA-graph capture contract |
| #418 | Recurrent cache position state unsharded under DCP (the >270K Context loop fix) |
| #419 | Keep Kimi K3 protocol markers out of streamed content |
| #422 | Handle KV load failures across hybrid cache groups |
| #433 | MRV2 sampler / thinking-budget fix |

## Cache requirements

| Cache | Host path | Env var |
|---|---|---|
| CuTe DSL | `~/.cache/huggingface/b12x/cute_compile` | `B12X_CUTE_COMPILE_CACHE_DIR` |
| Triton | `/cache/huggingface/triton-cache` | `TRITON_CACHE_DIR` |
| TorchInductor | `/cache/huggingface/torchinductor-cache` | `TORCHINDUCTOR_CACHE_DIR` |
| Torch extensions | `/cache/huggingface/torch_extensions` | `TORCH_EXTENSIONS_DIR` |

## Requirements

- 16× DGX Spark GB10 (SM121, aarch64), RoCE v2 (ConnectX-7, 100 Gbit)
- ~600 GB model weights on NFS (`shared_weights_nfs: true`)
- `eugr/spark-vllm-docker` recipe runner

## Credits

Full credits in [`../../ATTRIBUTION.md`](../../ATTRIBUTION.md). Key upstreams:
[moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) (model weights),
[local-inference-lab/vllm](https://github.com/local-inference-lab/vllm),
[local-inference-lab/b12x](https://github.com/local-inference-lab/b12x),
[voipmonitor/InstantTensor](https://github.com/voipmonitor/InstantTensor),
[RedHatAI/Kimi-K3-speculator.dspark](https://huggingface.co/RedHatAI/Kimi-K3-speculator.dspark).
