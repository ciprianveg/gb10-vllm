# Credits & Attribution

## Core Dependencies

| Component | Source | License | Notes |
|-----------|--------|---------|-------|
| **CUDA 13.0** | NVIDIA | Proprietary | GB10/SM121 support |
| **vLLM** | vllm-project/vllm | Apache 2.0 | Upstream inference engine |
| **local-inference-lab/vllm** | local-inference-lab/vllm | Apache 2.0 | B12X_MLA, DSpark, PP2, KDA |
| **NCCL** | NVIDIA / zyang-dev | BSD-3-Clause | dgxspark-3node-ring branch |
| **FlashInfer** | flashinfer-ai/flashinfer | Apache 2.0 | cu130/torch2.11 |
| **b12x/sparkinfer** | NVIDIA | Apache 2.0 | CuTe DSL SM120/SM121 kernels |
| **Marlin MoE** | IST-DASLab/marlin | Apache 2.0 | Quantized expert parallel |
| **InstantTensor** | instanttensor | Apache 2.0 | Fast safetensors loading |
| **Kimi-K3 Model** | Moonshot AI | Custom | 1M context |
| **DSpark Draft** | Inferact | Apache 2.0 | Kimi-K3-DSpark draft model |
| **Kimi-K3 Model** | Moonshot AI | Custom | 1M context window |

## Upstream Projects

| Project | Repository | Branch/Commit |
|---------|------------|---------------|
| vLLM (upstream) | https://github.com/vllm-project/vllm | main |
| vLLM (fork) | https://github.com/local-inference-lab/vllm | codex/hh-kimi-k3-dspark-dcp16-20260804 |
| NCCL (mesh) | https://github.com/zyang-dev/nccl | dgxspark-3node-ring |
| FlashInfer | https://github.com/flashinfer-ai/flashinfer | cu130/torch2.11 |
| b12x/sparkinfer | https://github.com/NVIDIA/sparkinfer | main |
| Marlin | https://github.com/IST-DASLab/marlin | main |
| InstantTensor | https://github.com/instanttensor/instanttensor | main |

## Model Weights

| Model | Source | Format | Context |
|-------|--------|--------|---------|
| Kimi-K3 Full | Moonshot AI | MXFP4 | 1M tokens |
| Kimi-K3 REAP-320 | Moonshot AI | MXFP4 | 320K |
| Kimi-K3 DSpark | Inferact | MXFP4 | 5 layers |

## Infrastructure

| Component | Provider |
|-----------|----------|
| GPU Hardware | NVIDIA GB10 (DGX Spark) |
| Interconnect | InfiniBand RoCE v2 (100 Gbit) |
| NFS | models115 (full K3), models16 (REAP-320, DSpark) |
| Container Runtime | Docker + nvidia-container-toolkit |
| Orchestration | Ray (distributed-executor-backend) |

## Modifications & Optimizations (Custom Mods)

Original patches for KIMI-K3 on GB10, applied at container runtime through the
mod-apply infrastructure provided by `eugr/spark-vllm-docker` (each mod is a
directory with a `run.sh` that the launcher executes before `vllm serve`).
All mods below are original work unless the patch header or comment identifies
a specific upstream source.

| Mod | Description | Source |
|-----|-------------|--------|
| `fix-gb10-mla-smem` | MLA smem 131KB→98KB for GB10 | Original |
| `fix-flashinfer-mla-sm121` | FlashInfer MLA CC gate for SM121 | Original |
| `fix-k3-fp8-triton-mla` | FP8 KV with TRITON_MLA | Original |
| `fix-k3-ep-intermediate-padding` | EP-aware MoE padding | Original |
| `fix-k3-dspark-warmup` | Skip spec-decode warmup | Original |
| `fix-mla-spec-decode-block-table` | Expand block_table for spec decode | Original |
| `fix-dspark-fused-kv` | 4.5x draft MLA kernel speedup | Based on `local-inference-lab/vllm` PR #50585 |
| `fix-dspark-pp2-aux-forward` | PP2 aux-hidden-state forwarding | Based on `local-inference-lab/vllm` PR #50514 |
| `fix-k3-flash-kda-prefill` | Flash KDA prefill 1.1-1.4x | Original |
| `fix-k3-dspark-perf-unified` | 5 DSpark perf commits unified | Based on `local-inference-lab/vllm` unified branch |

> **Note** `dspark-fused-kv` and `dspark-pp2-aux-forward` carry DSpark/PP2 patches
> whose upstream code appears in `local-inference-lab/vllm` PRs; this project
> adapts them for GB10/SM121, C2C, and the pipeline-parallel configuration used here.

## Model Optimizations

| Optimization | Config | Effect |
|--------------|--------|--------|
| `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1` | Force decode path for spec extend | Faster verification |
| `VLLM_KIMI_K3_B12X_DSPARK_ARGMAX=1` | Fused TP16 greedy sampling | Faster draft sampling |
| `VLLM_DSPARK_PREFER_B12X_ALLREDUCE_RMS` | Composed all-reduce + RMS | Reduced comm |
| `VLLM_KIMI_PRELAUNCH_SHARED_EXPERTS=1` | Shared expert prelaunch | Overlap compute |
| `VLLM_KIMI_USE_B12X_BATCHED_PROJECTION_TOPK=1` | Fused router | Faster DSpark verification |

## Hardware

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA GB10 (DGX Spark / DGX Spark+) |
| Architecture | Blackwell SM121 |
| Memory | 128GB unified (per node) |
| Interconnect | InfiniBand RoCE v2, 100 Gbit |
| Topology | 16 nodes × 1 GPU, 2×8 clusters |
| Fabric | NVLink-C2C internal (Grace-Blackwell) |

---

*Generated for gb10-vllm repository*
