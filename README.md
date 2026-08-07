# gb10-vllm

vLLM inference solutions optimized for NVIDIA GB10 (DGX Spark / DGX Spark+) with Blackwell SM121 architecture.

## Overview

This repository contains Docker images, recipes, and mods for running vLLM on NVIDIA GB10 (DGX Spark / DGX Spark+) with Blackwell SM121 architecture. It builds upon upstream vLLM and the local-inference-lab/vllm fork to enable B12X_MLA attention, DSpark speculative decoding, and pipeline parallelism on GB10.

## Repository Structure

```
gb10-vllm/
├── kimi-k3/
│   └── v1/                      # KIMI-K3 B12X solutions
│       ├── build.sh
│       ├── Dockerfile
│       ├── recipes/
│       │   ├── kimi-k3-full-hh-b12x-tp16.yaml   # TP16, PP1, EP enabled
│       │   └── kimi-k3-full-hh-b12x-pp2.yaml    # TP8+PP2, 800k+ context
│       ├── mods/                  # KIMI-K3 specific mods
│       ├── wheels/                # Prebuilt vLLM + FlashInfer wheels
│       └── scripts/               # Manage scripts
└── glm-5.2/                       # GLM-5.2 solutions (from gb10-glm-5.2)
    ├── v16/
    ├── v18/
    └── v18-vision/
```

## Models

Each model lives in its own subtree with a self-contained build, recipes, and mods.
Open the model page for full customization guides:

| Model | Page | What's inside |
|-------|------|---------------|
| **KIMI-K3** (Full, B12X_MLA + DSpark) | [kimi-k3/v1](kimi-k3/v1/) | Dockerfile + build.sh, recipes (TP16, TP8+PP2), runtime mods |
| **GLM-5.2** (Int4-Int8, v16/v18/v18-vision) | [glm-5.2](glm-5.2/) | per-version builds, vision overlay, production recipes |

## Quick Start

### Build the KIMI-K3 Image

```bash
./kimi-k3/v1/build.sh                   # build local tag vllm-node-kimi3-hh
./kimi-k3/v1/build.sh --push            # build + push to GHCR
```

Or manually (from repo root):

```bash
docker build \
  --build-arg VLLM_REF=codex/hh-kimi-k3-dspark-dcp16-20260804 \
  --build-arg INSTALL_B12X=1 \
  -t vllm-node-kimi3-hh \
  -f kimi-k3/v1/Dockerfile kimi-k3/v1
```

### Run KIMI-K3 TP16 (Full Speed, 200K Context)

```bash
# Stop GLM-5.2 on .11-.18 first!
./kimi-k3/v1/scripts/manage-kimi-k3-full-tp16.sh start
```

### Run KIMI-K3 TP8+PP2 (800K+ Context)

```bash
# Stop GLM-5.2 on .11-.18 first!
./kimi-k3/v1/scripts/manage-kimi-k3-full-pp2.sh start
```

## Architecture

### Hardware
- **GPU**: NVIDIA GB10 (DGX Spark / DGX Spark+) — 1 GPU per node, Blackwell SM121
- **Interconnect**: InfiniBand RoCE v2 (100 Gbit)
- **Topology**: 16 nodes (2×8 clusters), NVLink-C2C internal fabric

### Software Stack
- **Base**: nvidia/cuda:13.0.2-devel-ubuntu24.04
- **PyTorch**: 2.11.0 + cu130
- **vLLM**: local-inference-lab/vllm fork (codex/hh-kimi-k3-dspark-dcp16-20260804)
- **NCCL**: zyang-dev/nccl (dgxspark-3node-ring branch)
- **FlashInfer**: flashinfer-ai/flashinfer (cu130/torch2.11)
- **Python**: 3.12, uv package manager

### Key Technologies
- **B12X_MLA**: Dense MLA attention via CuTe DSL (sparkinfer/b12x)
- **DSpark**: Speculative decoding with Kimi-K3-DSpark draft
- **PP2**: Pipeline parallelism for 800K+ context
- **Expert Parallel**: MoE expert sharding (TP16)
- **InstantTensor**: Fast weight loading from NFS
- **Marlin MoE**: Quantized expert parallel backend

## Recipes

| Recipe | TP | PP | EP | Context | Model |
|--------|----|----|----|---------|-------|
| `kimi-k3-full-hh-b12x-tp16.yaml` | 16 | 1 | Yes | 200K | Full K3 |
| `kimi-k3-full-hh-b12x-pp2.yaml` | 8 | 2 | No | 800K+ | Full K3 |

## Mods

| Mod | Purpose |
|-----|---------|
| `fix-gb10-mla-smem` | Reduces B12X MLA smem from 131KB to 98KB for GB10 |
| `fix-flashinfer-mla-sm121` | Enables FlashInfer MLA on SM121 (CC gate patch) |
| `fix-k3-fp8-triton-mla` | Enables FP8 KV cache with TRITON_MLA |
| `fix-k3-ep-intermediate-padding` | EP-aware MoE padding |
| `fix-k3-dspark-warmup` | Skips spec-decode warmup |
| `fix-mla-spec-decode-block-table` | Expands block_table for spec decode |
| `fix-dspark-fused-kv` | PR #50585: 4.5x draft MLA kernel speedup |
| `fix-dspark-pp2-aux-forward` | PR #50514: PP2 aux-hidden-state forwarding |
| `fix-k3-flash-kda-prefill` | Flash KDA prefill (1.1-1.4x speedup) |
| `fix-k3-dspark-perf-unified` | Unified DSpark perf (fused router, TP16 greedy, composed reduce, Markov tail, shared-expert prelaunch) |

## Environment Variables

Key env vars set in recipes:
- `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1` — Force decode path for spec verification
- `VLLM_KIMI_K3_B12X_DSPARK_ARGMAX=1` — Fused TP16 greedy sampling
- `VLLM_DSPARK_PREFER_B12X_ALLREDUCE_RMS=1` — Composed all-reduce + RMS norm
- `VLLM_KIMI_PRELAUNCH_SHARED_EXPERTS=1` — Shared-expert prelaunch overlap
- `VLLM_KIMI_USE_B12X_BATCHED_PROJECTION_TOPK=1` — Fused router for DSpark verification
- `VLLM_KIMI_K3_SHARD_SP_SHARED_EXPERT=1` — Shard shared expert
- `VLLM_MARLIN_USE_ATOMIC_ADD=1` — Marlin atomic add optimization

## Building & Publishing

```bash
# Build KIMI-K3 locally (recipe image tag) then publish to GHCR
./kimi-k3/v1/build.sh
./kimi-k3/v1/build.sh --push

# GLM-5.2 versions have their own build scripts:
./glm-5.2/v18/build.sh            # local tag (vllm-node-tf5-glm52-v18)
./glm-5.2/v18-vision/build-nvfp4.sh --push

# Available on GHCR: ghcr.io/ciprianveg/gb10-vllm/kimi-k3:latest
#                    ghcr.io/ciprianveg/gb10-glm-5.2:<v18-prod|v18-vision|v18.1-vision>
```

## Credits

- **NVIDIA**: CUDA 13, GB10/SM121, NCCL, Marlin MoE
- **vLLM Project**: Upstream inference engine (https://github.com/vllm-project/vllm)
- **local-inference-lab/vllm**: B12X_MLA, DSpark, PP2, KDA (https://github.com/local-inference-lab/vllm)
- **NCCL**: zyang-dev/nccl (dgxspark-3node-ring branch)
- **FlashInfer**: FlashInfer team (https://github.com/flashinfer-ai/flashinfer)
- **b12x/sparkinfer**: NVIDIA CuTe DSL kernels for SM120/SM121
- **Marlin MoE**: Marlin team (https://github.com/IST-DASLab/marlin)
- **InstantTensor**: InstantTensor team
- **Kimi-K3 model**: Moonshot AI
- **DSpark draft**: Inferact/Kimi-K3-DSpark

## License

Apache 2.0 — See individual components for their licenses.
EOF