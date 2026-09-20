# KIMI-K3 on 16× GB10 DGX Spark — v5 (v5-prd: RoCE collectives + fused verify + baked perf stack)

**Current recommended Kimi-K3 solution.** Full [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
(MXFP4 experts, BF16 attention) served with the sm121
`ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v5-prd` image + RedHat DSpark speculative
decoding on 16-node DGX Spark GB10 (sm121) clusters.

| Version | Stack | Status |
|---------|-------|--------|
| **v5** | v4-prd + RoCEnante collectives, fused verify, fp8 draft, b12x spec-merge + v6 KDA/MoE kernel mods + `_C` rebuild (#54896/#55180) + perf-layer stack | **Current production** 🚀 |
| v4 | v4 base (`v4-sm121-r36`) + 5 baked mods (DFlash2 MLA drafting, #52388/#51508/#50169) | [Superseded](../v4/README.md) |
| v3 | v2 image + vLLM@0232bce6 overlay (MoE fusion #385/#386) + 11 mods baked | [Superseded](../v3/README.md) |
| v2 | `vllm-node-kimi3-sm121` (vLLM@881ac39 + B12X) + RedHat DSpark, runtime mods | [Superseded](../v2/README.md) |
| v1 | `vllm-node-kimi3-hh` (B12X_MLA + Inferact DSpark) | [Historic artifact](../v1/README.md) |

## What's new in v5

- **RoCEnante TP+DCP collectives**: one-shot all-reduce (TP) and all-gather
  (DCP) over the RoCE v2 fabric, replacing NCCL for the small decode-size
  messages.
- **Fused verify TILE8 + nst6**: 8-row fused spec-verify kernel, 6 speculative
  tokens, DCP-un-gated.
- **Recompiled `_C` extension**: MLA cache-kernel epilogue (#54896) + SM12.x
  blockwise-FP8 CUTLASS CTA swizzle (#55180) baked into the native build.
- **Marlin MoE + nopad**: env-gated skip of the KimiMoE 192→256 expert pad
  (`VLLM_K3_MARLIN_NOPAD=1`).
- **Adaptive draft depth** (opt-in recipe): acceptance-length-driven draft
  depth, floor 4 (`VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH=1`).
- **Stability guards**: NaN-gumbel mask, multistream record-stream + grammar
  stream fences, draft-noeplb for DCP16, flashinfer autotune-import guard,
  Triton MXFP4 MoE unlocked on SM12x.

## Quick Start

```bash
# 1. Pull the prebuilt image (linux/arm64, sm121) — on every node
docker pull ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v5-prd

# 2. Deploy from spark-vllm-docker (weights already on NFS — no download needed)
./run-recipe.sh <gb10-vllm>/kimi-k3/v5/recipes/kimi-k3-dcp-tp16-prd.yaml
```

> First boot JIT-compiles CuTe DSL + Triton kernels per batch shape — expect a
> slow first ~10-20 requests, then full speed. Mount persistent cache dirs
> (see [Cache requirements](#cache-requirements)) to avoid re-JIT on restart.

### Build the image yourself (optional)

Thin overlay build guide (from v4-prd, ordered mod application):
[`BUILD-SM121-IMAGE.md`](BUILD-SM121-IMAGE.md)

```bash
./kimi-k3/v5/build.sh            # build local tag
./kimi-k3/v5/build.sh --push     # build + push to GHCR
```

## Recipes

| Recipe | TP | DCP | Draft | Notes |
|---|---|---|---|---|
| [`kimi-k3-dcp-tp16-prd.yaml`](recipes/kimi-k3-dcp-tp16-prd.yaml) | 16 | 8 | DSpark nst6 static, TILE8, kv3G, 8 seqs | **Recommended** |
| [`kimi-k3-dcp-tp16-prd-adaptive.yaml`](recipes/kimi-k3-dcp-tp16-prd-adaptive.yaml) | 16 | 8 | DSpark adaptive 4-6 (`VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH=1`) | acceptance-driven draft depth |

Model weights default to `/root/models/models115/Kimi-K3`, the RedHat DSpark
draft to `/root/models/models11/RedHatAI-Kimi-K3-dspark`; both are
`{model_path}` / `{draft_model_path}` defaults you can override.

## Benchmarks (v5-prd, TP16 + DCP8, static nst6, TILE8, kv3G, 16× GB10)

Coding game-bench, 3000-token runs:

| Concurrency | Throughput |
|---|---|
| C1 | **29.81 tok/s** |
| C2 | 42.00 agg / 21.00 mean stream |
| C4 | 58.00 agg / 14.50 mean stream |
| C8 | **87.12 agg** / 10.89 mean stream |

* 136t/s peak speed at C8

llama-bench (coherent corpus — DSpark acceptance is on the lower side here):

| test | t/s | peak t/s |
|:---|---:|---:|
| pp2048 @ d4000 | **880.35** | — |
| tg2048 @ d4000 | **23.59** | 46.00 |
| pp2048 @ d100000 | **812.56** | — |
| tg2048 @ d100000 | **21.13** | 33.00 |
| pp2048 @ d200000 | **848.73** | — |
| tg2048 @ d200000 | **20.03** | 38.00 |

## Speed mods baked into v5-prd

| Mod / layer | What it does |
|---|---|
| RoCEnante TP+DCP collectives | one-shot all-reduce (TP) + all-gather (DCP) over RoCE v2 |
| Fused verify K=3 (upstream b12x#271, via v4-prd) | nst-3-only fused DFlash verify from the upstream fork — the base the 8-row kernel below extends to nst 4–7 |
| Fused verify TILE8 + nst6  | 8-row fused spec-verify kernel as 2×4-row tiles (true 8-row CTA impossible on sm121), 6 speculative tokens, DCP-un-gated |
| fp8 draft | fp8 draft path + workspace-reserve fixes |
| `fix-k3-marlin-nopad` | skip the KimiMoE 192→256 expert pad (`VLLM_K3_MARLIN_NOPAD=1`) |
| `_C` rebuild (`perf-pr55356-54896-mla-cache-kernels`, `perf-pr55180-fp8-cta-swizzle`) | native rebuild with #54896 MLA cache epilogue + #55180 FP8 CTA swizzle — csrc-only PRs can't ship as Python runtime mods, so the extension was recompiled (~47 min), no regression |
| `perf-pr54048-router-gemm-fam120` (runtime-only) | un-gates the fused router GEMM on sm121 (fam120); measured neutral, kept opt-in |
| adaptive draft depth (`fix-dspark-adaptive-nst`, `fix-adaptive-min-depth`) | acceptance-length-driven draft depth with env floor (`VLLM_DSPARK_DYNAMIC_MIN_DEPTH`) |
| `fix-mxfp4-triton-sm121` | unlock OAI Triton MXFP4 MoE experts on SM12x (GB10) |
| `fix-k3-nan-gumbel` | NaN→-inf guard in `gumbel_block_argmax` (kills async IMA at nst=5) |
| `fix-multistream-record-stream` + `fix-grammar-stream-fence` | stream fences for aux-stream outputs and grammar bitmask copies |
| `fix-dspark-draft-noeplb` | draft model never inherits EPLB config (required at DCP16) |
| `drop-caches` | runtime cache-drop loop, stays recipe-level (nothing to bake) |

## Dynamic FP8 quantization (about a third of the v4 → v5 gain)

At load time the image quantizes selected dense linears to MXFP8 (`"linear":{"weight":"mxfp8"}`)
and the shared experts to per-block FP8 (`"shared_experts":{"weight":"fp8_per_block_static"}`).
Quality-sensitive layers are explicitly excluded and stay BF16:

- `re:.*self_attn.*` (all attention projections), `re:.*lm_head` (output head)
- `re:.*vision_tower.*`, `re:.*mm_projector.*` (vision path)
- `re:.*block_sparse_moe.gate` (MoE routing)

That exclusion is why quality is conserved while the at-the-wall dense GEMMs move
roughly half the weight bytes. Attribution of the speedup: ~1/3 dynamic FP8,
the rest RoCEnante DCP collectives + kernel mods (`_C` rebuild, tier-1) + draft
optimizations (nst6/TILE8 fused verify, adaptive depth, fp8 draft path).
Remove `--quantization-config` to run the BF16-dense baseline.

## Tuning knobs

| Knob | Settings | Effect |
|---|---|---|
| `VLLM_K3_FUSED_TILE` | `8` (recipe) / `4` / `0` | `8` = fused verify for nst 4–7 (8-row, 2 tiles); `4` = fused for nst 3 only; `0` = flat verify (wins long-context, e.g. 64K+) |
| `VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH` | `0` (recipe) / `1` | `1` = acceptance-driven draft depth (adaptive recipe); window `VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH_WINDOW` (default 8), floor `VLLM_DSPARK_DYNAMIC_MIN_DEPTH` (default 4) |
| `VLLM_K3_FP8_DRAFT` | `0` (recipe) / `1` | `1` = fp8 draft — slightly faster, less draft KV; `0` = bf16 draft (served config above) |
| `VLLM_ENABLE_ROCE_DCP` | `1` (recipe) / `0` | RoCEnante one-shot all-gather for the DCP group; `0` = pure NCCL |
| `--disable-custom-all-reduce` | set (recipe) / remove | set = TP collectives on NCCL (served at TP16); remove = RoCEnante TP one-shot all-reduce (won at TP8, regressed at TP16 — re-test per cluster) |
| `VLLM_ROCE_ALLGATHER_MAX_SIZE` / `VLLM_ROCE_ALLREDUCE_MAX_SIZE` | `8M` / `1M` (recipe) | per-rank shard caps; above the cap falls back to NCCL (smaller AG cap = decay-free stability, slightly lower peak) |
| `NCCL_MIN/MAX_NCHANNELS` | `3` / `3` (recipe) | optimum depends on RoCE state; re-sweep if hook posture changes |
| `VLLM_K3_MIN_SPLITS` | `0` (recipe) | 48-SM split floor; raise if SM under-fill is suspected |
| `VLLM_K3_MARLIN_NOPAD` / `VLLM_KIMI_FUSED_TOPK16` / `VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT` / `VLLM_MARLIN_USE_ATOMIC_ADD` | all `1` (recipe) | MoE fast paths — keep on |

Memory posture (recipe): `--gpu-memory-utilization 0.82`, `--kv-cache-memory-bytes 3000000000` (3G unblocked 8-way concurrency; raise toward 8–10G only with headroom), `drop-caches` runtime mod against unified-memory pressure.

## Cache requirements

| Cache | Host path | Env var |
|---|---|---|
| CuTe DSL | `~/.cache/huggingface/b12x/cute_compile` | `B12X_CUTE_COMPILE_CACHE_DIR` |
| Triton | `/cache/huggingface/triton-cache` | `TRITON_CACHE_DIR` |
| TorchInductor | `/cache/huggingface/torchinductor-cache` | `TORCHINDUCTOR_CACHE_DIR` |
| Torch extensions | `/cache/huggingface/torch_extensions` | `TORCH_EXTENSIONS_DIR` |

## Requirements

- 16× DGX Spark GB10 (SM121, aarch64), RoCE v2 (ConnectX-7, 100 Gbit)
- ~1.56 TB model weights on NFS (`shared_weights_nfs: true`)
- `eugr/spark-vllm-docker` recipe runner

## Credits

Full credits in [`../../ATTRIBUTION.md`](../../ATTRIBUTION.md). Key upstreams:
[moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) (model weights),
[local-inference-lab/vllm](https://github.com/local-inference-lab/vllm),
[local-inference-lab/b12x](https://github.com/local-inference-lab/b12x),
[voipmonitor/InstantTensor](https://github.com/voipmonitor/InstantTensor
[RedHatAI/Kimi-K3-speculator.dspark](https://huggingface.co/RedHatAI/Kimi-K3-speculator.dspark),
- **eugr/spark-vllm-docker**: build harness + cluster launcher + mod-apply infrastructure.
