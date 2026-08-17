# KIMI-K3 on 16× GB10 DGX Spark — v2 (sm121 image + RedHat DSpark)

**Recommended Kimi-K3 solution.** Full [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
(MXFP4 experts, BF16 attention) served with the native-sm121
`vllm-node-kimi3-sm121` image and **RedHat DSpark** speculative decoding on
16-node DGX Spark GB10 (sm121) clusters.

| Version | Stack | Status |
|---------|-------|--------|
| **v2** | `vllm-node-kimi3-sm121` (PyTorch 2.13 / CUDA 13.3 / vLLM@881ac39 + B12X) + RedHat DSpark | **Current production** 🚀 |
| v1 | `vllm-node-kimi3-hh` (B12X_MLA + Inferact DSpark) | [Historic artifact](v1/) — no longer maintained |

v2 is faster and needs **no runtime mods** for the recommended TP16 path: the v1
runtime-mod fixes are baked into the fork's integration branch. v1 is kept only
for reference.

## Quick Start (v2 — recommended)

```bash
# 1. Pull the prebuilt image (linux/arm64, sm121)
docker pull ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v2-sm121
docker tag ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v2-sm121 vllm-node-kimi3-sm121

# 2. Deploy TP16 (recommended, ~500K ctx) from spark-vllm-docker.
#    Weights already live on NFS (see Requirements) — no --setup download needed.
./run-recipe.sh <gb10-vllm>/kimi-k3/v2/recipes/kimi-k3-tp16.yaml

#    or TP8+PP2 (max context, ~20% slower) — needs the fix-pp2-sm121 mod
./run-recipe.sh <gb10-vllm>/kimi-k3/v2/recipes/kimi-k3-tp8pp2.yaml
```

> First boot JIT-compiles CuTe DSL + Triton kernels per batch shape — expect a
> slow first ~10-20 requests, then full speed. Mount persistent cache dirs
> (see [Cache requirements](#cache-requirements)) to avoid re-JIT on restart.

### Build the image yourself (optional)

Full reproducible two-step build guide:
[`BUILD-SM121-IMAGE.md`](BUILD-SM121-IMAGE.md)

```bash
# Build + push local tag vllm-node-kimi3-sm121 (or --push for GHCR)
./kimi-k3/v2/build.sh
```

## Recipes

| Recipe | TP | PP | Draft | Max ctx | Notes |
|---|---|---|---|---|---|
| [`kimi-k3-tp16.yaml`](recipes/kimi-k3-tp16.yaml) | 16 | 1 | RedHat DSpark (Qwen3-GQA) | **~500K** | **Recommended** — full speed, ~21-25 tok/s single-stream |
| [`kimi-k3-tp8pp2.yaml`](recipes/kimi-k3-tp8pp2.yaml) | 8 | 2 | RedHat DSpark (Qwen3-GQA) | **~1M** | Max context, ~20% slower than TP16 |
| [`kimi-k3-dcp-tp16.yaml`](recipes/kimi-k3-dcp-tp16.yaml) | 16 (DCP16) | 1 | RedHat DSpark (Qwen3-GQA) | **~270K** | DCP16 KV split — targets long-context decode decline (KV re-read bound); context capped at 270K |

Context on TP8+PP2: **~1M tokens** with a slightly increased
`--gpu-memory-utilization` and `instanttensor` disabled, **~750K tokens** with
`--load-format instanttensor`.

Both reference `container: vllm-node-kimi3-sm121` and are `cluster_only`
(16 nodes). Model weights live at `/root/models/models115/Kimi-K3`, the RedHat
DSpark draft at `/root/models/models11/RedHatAI-Kimi-K3-dspark`; the recipes use
`{model_path}` / `{draft_model_path}` defaults you can override.

## Benchmarks (TP16, 16× GB10)

> From llama-benchy (coherent corpus, tg=1500, ctx=120000), run twice — first run
> ignored to exclude the warmup phase. Endpoint: `http://<head>:5002/v1`.

| test | t/s | peak t/s | ttfr (ms) | est_ppt (ms) | e2e_ttft (ms) |
|:---|---:|---:|---:|---:|---:|
| pp2048 | **610.91** | — | 3,098.73 | 3,097.03 | 3,098.73 |
| tg1500 | **21.05** | 37.00 | — | — | — |
| pp2048 @ d60000 | **684.17** | — | 81,825.38 | 81,823.69 | 81,827.58 |
| tg1500 @ d60000 | **13.04** | 29.00 | — | — | — |
| pp2048 @ d120000 | **649.73** | — | 170,239.21 | 170,237.52 | 170,244.05 |
| tg1500 @ d120000 | **11.31** | 20.00 | — | — | — |

Single-stream coding benchmark (temp=0, prompt 125 tok, tg=1500):

```
Completion tokens : 1500
Prompt tokens     : 125
Decode time       : 61.185s
Overall decode tok/s : 24.87
```

> Note: coding and structured-coding workloads typically see a **greater DSpark
> boost** than the coherent corpus used in llama-benchy — expect higher decode
> tok/s on those workloads than the numbers above.

## Tuning knobs

| Knob | Change | Effect |
|---|---|---|
| `--max-num-batched-tokens` | 2048 → 4096 | Faster prefill, **decreases** max context |
| DCP | 1 → 2/4 (TP16) | More context, **slower** decode |
| `--load-format` | drop `instanttensor` | Higher GMU + context, but ~3× slower model load (once) |
| `--moe-backend` | marlin+eager → `b12x` + CUDAGraph | ~10% faster **without** DSpark; needs CUDAGraph tuned for DSpark `num_speculative_tokens` |
| `num_speculative_tokens` | 3–8 (default 6) | Tune to your workload type |
| `--enable-expert-parallel` | remove | Lower concurrency, higher single-stream speed |

## Mods

Both recipes apply a set of **performance mods** at container start — upstream vLLM
PRs backported to the `881ac39` fork. Together they give **~7% decode speed** on
both TP16 and TP8+PP2, and both enable **prefix caching + chunked prefill**
(`--enable-prefix-caching --enable-chunked-prefill`) so the prefix-cache mods
above are active. Command flags follow the eugr harness's tuned config
(`--kda-prefill-backend triton`, `--max-num-batched-tokens 4096`):

| Mod | Upstream PR | What it does |
|---|---|---|
| `pr51508-stale-zero-accept` | #51508 | Skip GDN/KDA recurrent-state updates for stale (zero-accept) spec rows |
| `pr51036-repetition-detection` | #51036 | Allowlist `repetition_detection` override-generation config (mitigates KDA-NaN loops) |
| `pr46324-piecewise-spec-capture` | #46324 | Align spec-decode CUDA-graph capture sizes in PIECEWISE mode |
| `pr50169-drafter-kv-pool` | #50169 | Dedicated KV groups for sliding-window drafters (GB10 pool 415k → 871k) |
| `pr46932-uma-cudagraph-mem` | #46932 | Fix CUDA-graph memory accounting on unified-memory (GB10) |
| `pr47926-dspark-prefix-mask` | #47926 | Mask prefix-cache-restored tokens out of the DSpark draft context |
| `fix-mamba-cow-external-hit` | #395 | Preserve Mamba/KDA copy-on-write after external prefix-cache hits |
| `fix-k3-renderer-split-turns` | #395 | Merge split Kimi-K3 prose+tool_calls turns; preserve reasoning (agentic transcript fix) |
| `fix-dcp-hybrid-prefix-cache-hits` | #401 | Allow hash-aligned DCP hybrid prefix hits (active once prefix caching is enabled) |
| `fix-pp2-sm121` | — (TP8+PP2 only) | PP2 + DSpark infra fixes: draft PP=1, aux-hidden-state-over-PP, EAGLE3+PP guard removal, embed-tokens sharing |

Copy the mods into your eugr harness's `mods/` dir before running (the recipes'
`mods:` entries resolve relative to the harness):

```bash
cp -r <gb10-vllm>/kimi-k3/v2/mods/* ~/eugr/spark-vllm-docker/mods/
```

TP16 needs **no image-level mods** — the v1 runtime fixes are baked into the fork's
integration branch; these mods are runtime performance backports only.

## Cache requirements

| Cache | Host path | Env var |
|---|---|---|
| CuTe DSL | `~/.cache/huggingface/b12x/cute_compile` | `B12X_CUTE_COMPILE_CACHE_DIR` |
| Triton | `/cache/huggingface/triton-cache` | `TRITON_CACHE_DIR` |
| TorchInductor | `/cache/huggingface/torchinductor-cache` | `TORCHINDUCTOR_CACHE_DIR` |
| Torch extensions | `/cache/huggingface/torch_extensions` | `TORCH_EXTENSIONS_DIR` |

## What changed: v1 → v2

- **Image** rebuilt from `local-inference-lab/blackwell-llm-docker` (rtx6kpro
  lineage): PyTorch **2.13.0** (from source, CUDA 13.3), NCCL **2.31.2**,
  FlashInfer **0.6.15.post1**, InstantTensor **0.1.9**, Triton kernels **v3.5.1**,
  all compiled for **sm121**.
- **vLLM** `integration/kimi-k3-ii-cu133-torch213-20260811` @ `881ac39` — the
  v1 runtime mods (B12X MLA/smem, flash-KDA prefill, fused-KV, mamba idx-int64,
  mamba-MRv2 race, spec-decode block-table, EP padding, DSpark warmup/perf-unified)
  are **baked in**, so TP16 runs with `mods: []`.
- **Draft** switched from Inferact DSpark (MLA) to **RedHat DSpark** (Qwen3-GQA)
  with `TRITON_ATTN` — the v2 recipes use `num_speculative_tokens: 6`.
- **TP16 is now the recommended path** (~500K ctx, fastest). v1 recommended PP2
  because TP16 only fit ~200K; v2's TP16 fits ~500K.
- **Performance mods** (both recipes): 6 upstream PR backports + 3 PR #395/#401
  fixes applied at container start (~7% decode speedup over the unmodded v2
  config). Prefix caching + chunked prefill enabled to match the eugr harness.

## Directory structure

```
kimi-k3/
├── v1/                        ← historic (B12X_MLA + Inferact DSpark)
└── v2/
    ├── README.md              this guide
    ├── BUILD-SM121-IMAGE.md   reproducible sm121 image build guide
    ├── build.sh               wrapper for the two-step sm121 build (+ --push)
    ├── build-sm121/           sm121-adapted Dockerfiles + build scripts
    ├── mods/fix-pp2-sm121/    PP2 + DSpark infra mod (external, TP8+PP2 only)
    └── recipes/               kimi-k3-tp16.yaml + kimi-k3-tp8pp2.yaml
```

## Requirements

- 16× DGX Spark GB10 (SM121, aarch64), RoCE v2 (ConnectX-7, 100 Gbit)
- ~600 GB model weights on NFS (`shared_weights_nfs: true`)
- `eugr/spark-vllm-docker` recipe runner
- Optionally `local-inference-lab/blackwell-llm-docker` to rebuild the image

## Credits

Full credits in [`BUILD-SM121-IMAGE.md`](BUILD-SM121-IMAGE.md#credits) and
[`ATTRIBUTION.md`](../../ATTRIBUTION.md). Key upstreams: [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
(model weights), [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm),
[local-inference-lab/blackwell-llm-docker](https://github.com/local-inference-lab/blackwell-llm-docker),
[local-inference-lab/b12x](https://github.com/local-inference-lab/b12x),
[voipmonitor/InstantTensor](https://github.com/voipmonitor/InstantTensor),
[RedHatAI/Kimi-K3-speculator.dspark](https://huggingface.co/RedHatAI/Kimi-K3-speculator.dspark),
[eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker).
