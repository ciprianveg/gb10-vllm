# KIMI-K3 on 16× GB10 DGX Spark — v4 (v4-prd: DFlash2 MLA drafting + baked upstream mods)

**Current published Kimi-K3 solution.** Full [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
(MXFP4 experts, BF16 attention) served with the sm121
`ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v4-prd` image, with a choice of two
speculative drafters — **RedHat DSpark** or **DFlash2** — on 16-node DGX Spark
GB10 (sm121) clusters.

| Version | Stack | Status |
|---------|-------|--------|
| **v4** | v3 image + upstream fork-tree advance (vLLM `e08d796f` / b12x `b8c7153c`) + 5 baked mods (DFlash2 MLA drafting, #52388/#51508/#50169) + b12x/vLLM optimizations | **Current published production** 🚀 |
| v3 | v2 image + vLLM@0232bce6 overlay (MoE fusion #385/#386) + 11 mods baked | [Superseded](../v3/README.md) |
| v2 | `vllm-node-kimi3-sm121` (vLLM@881ac39 + B12X) + RedHat DSpark, runtime mods | [Superseded](../v2/README.md) |
| v1 | `vllm-node-kimi3-hh` (B12X_MLA + Inferact DSpark) | [Historic artifact](../v1/README.md) |

v4-prd is a **thin overlay** on the published v3 image: the vLLM + b12x trees
are advanced to the upstream fork pins (vLLM `e08d796f`, b12x `b8c7153c` —
public `voipmonitor` branch tips) and the 5 mods are baked in. No CUDA
recompilation; see [`BUILD-SM121-IMAGE.md`](BUILD-SM121-IMAGE.md) for the
exact fetch commands.

## What's new in v4

- **DFlash2 drafting (MLA flavor)**: the `dflash2-support` mod backports vLLM
  PR #52816 (+ #52883/#53122/#53435/#53662) so vLLM can serve
  [`lightseekorg/kimi-k3-dflash2`](https://huggingface.co/lightseekorg/kimi-k3-dflash2)
  drafts (`DFlash2DraftModel`) over the **B12X_MLA** backend. Note: this is the
  dense-MLA draft path — K3 uses pure MLA (no `index_topk` config), so the
  sparse-MLA draft variant does not apply on K3.
- **Upstream mods baked**: #52388 (K3 mamba metadata prep, 6.6–7.6× kernel),
  #51508 (stale zero-accept recurrent-state guard), #50169 (drafter KV pool —
  GB10 pool 415k → 871k tokens), plus the r29 Mamba cadence fix.
- **Prefill/decode optimizations**: b12x #271 fused DFlash K=3 verification
  (fixes the long-context decode regression at `nst=3`), myshytf/b12x #3 MoE
  plan memoization (~45% prefill), vllm #52502 GB10 fused-MoE FP8 tuning configs.
- **Engine**: `vLLM v0.26.1rc0+kimi.k3.aligned` (upstream fork tree vLLM
  `e08d796f` / b12x `b8c7153c`), torch 2.13.0, FlashInfer 0.6.15.post1, lazy CuTeDSL compile.

## Quick Start

```bash
# 1. Pull the prebuilt image (linux/arm64, sm121) — on every node
docker pull ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v4-prd

# 2. Deploy from spark-vllm-docker (weights already on NFS — no download needed)
./run-recipe.sh <gb10-vllm>/kimi-k3/v4/recipes/kimi-k3-dcp-tp16-prd.yaml
```

> First boot JIT-compiles CuTe DSL + Triton kernels per batch shape — expect a
> slow first ~10-20 requests, then full speed. Mount persistent cache dirs
> (see [Cache requirements](#cache-requirements)) to avoid re-JIT on restart.

### Build the image yourself (optional)

Thin overlay build guide (base provenance + reproducibility notes):
[`BUILD-SM121-IMAGE.md`](BUILD-SM121-IMAGE.md)

```bash
./kimi-k3/v4/build.sh            # build local tag (pulls the public v3 base)
./kimi-k3/v4/build.sh --push     # build + push to GHCR
```

## Cache requirements

| Cache | Host path | Env var |
|---|---|---|
| CuTe DSL | `~/.cache/huggingface/b12x/cute_compile` | `B12X_CUTE_COMPILE_CACHE_DIR` |
| Triton | `/cache/huggingface/triton-cache` | `TRITON_CACHE_DIR` |
| TorchInductor | `/cache/huggingface/torchinductor-cache` | `TORCHINDUCTOR_CACHE_DIR` |
| Torch extensions | `/cache/huggingface/torch_extensions` | `TORCH_EXTENSIONS_DIR` |

## Requirements

- 16× DGX Spark GB10 (SM121, aarch64), RoCE v2 (ConnectX-7, 100 Gbit)
- `eugr/spark-vllm-docker` recipe runner

## Credits

Full credits in [`../../ATTRIBUTION.md`](../../ATTRIBUTION.md). Key upstreams:
[moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) (model weights),
[local-inference-lab/vllm](https://github.com/local-inference-lab/vllm),
[local-inference-lab/b12x](https://github.com/local-inference-lab/b12x),
[voipmonitor/InstantTensor](https://github.com/voipmonitor/InstantTensor),
[RedHatAI/Kimi-K3-speculator.dspark](https://huggingface.co/RedHatAI/Kimi-K3-speculator.dspark),
[lightseekorg/kimi-k3-dflash2](https://huggingface.co/lightseekorg/kimi-k3-dflash2).
