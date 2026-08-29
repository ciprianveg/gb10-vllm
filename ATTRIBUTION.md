# Attribution & Credits

This project builds on the following open-source works. Full credit to their authors.

## Foundational Work

**CosmicRaisins/glm-5.2-gb10** pioneered GLM-5.2 serving on DGX Spark (GB10, sm_121). This project builds directly on their work:

- Identified the `--hf-overrides '{"index_topk_pattern":"FFFSSS..."}'` requirement (78-char pattern derived from `indexer_types`; without it, 56/78 layers top-k through uninitialized weights — coherent under ~2k tokens, garbage beyond)
- Created the DCP stack patches (PR #72: draft config propagation, `topk_scores_buffer` for B12X, `build_for_drafting`)
- Established the `VLLM_USE_V2_MODEL_RUNNER=1` + `VLLM_USE_B12X_SPARSE_INDEXER=1` + `--attention-backend B12X_MLA_SPARSE` serving configuration
- Identified the `draft-quant-packed-mapping` fix (without it, quantized NextN drafts silently build unquantized and MTP acceptance collapses)
- Created the `eugr/spark-vllm-docker` fork (multi-stage Docker build, wheel caching, SCP deploy, recipe runner, autodiscovery)

Full credit to **CosmicRaisins** for the foundational GLM-5.2-on-GB10 serving stack. See [CosmicRaisins/glm-5.2-gb10](https://github.com/CosmicRaisins/glm-5.2-gb10) and their [ATTRIBUTION.md](https://github.com/CosmicRaisins/glm-5.2-gb10/blob/master/ATTRIBUTION.md) for the complete lineage.

## Upstream Sources

### v16 Stack

| Project | Repo | Commit/Branch | License | Used For |
|---------|------|---------------|---------|----------|
| **vLLM** | `local-inference-lab/vllm` | `codex/fathomless-firmament-v16-unified-20260712` @ `5dffea8` | Apache-2.0 | Core inference engine, V2 runner, MTP, DCP, B12X integration |
| **b12x** | `lukealonso/b12x` | `master` @ `97b3d64` | Apache-2.0 | MoE kernels, W4A8 quantization, unified SM120 sparse MLA, PCIe DCP collectives |
| **FlashInfer** | `flashinfer-ai/flashinfer` | Prebuilt wheels (sm_121) | Apache-2.0 | Sparse MLA attention kernels, page attention |
| **DeepGEMM** | `deepseek-ai/DeepGEMM` | `nv_dev` branch | Apache-2.0 | SM120 GEMM kernels for MoE |
| **NCCL** | `zyang-dev/nccl` | `dgxspark-3node-ring` | BSD-3-Clause | 3-node ring collectives over RoCE |

### v18 Stack Additions (Gilded Gnosis)

| Project | Repo | Commit | License | Used For |
|---------|------|--------|---------|----------|
| **vLLM v18** | `local-inference-lab/vllm` | `build/gilded-gnosis-v18-final-20260718` @ `264bce1d` | Apache-2.0 | v18 inference engine with DCP fast-path, NVFP4 MLA, NF3 Grid188, MTP target-revision |
| **B12X v18** | `voipmonitor/b12x` | `codex/nf3-grid188-decode-20260717` @ `bc85ef3` | Apache-2.0 | v18 B12X with Grid188, deterministc CuTe cache keys |
| **FlashInfer v18** | `voipmonitor/flashinfer` | `801d57a` | Apache-2.0 | Updated FlashInfer for v18 stack |
| **DeepGEMM** | `deepseek-ai/DeepGEMM` | `a6b593d` | Apache-2.0 | SM120 GEMM kernels for MoE (new in v18) |
| **InstantTensor** | `local-inference-lab/instant-tensor` | `85e7c5f` | Apache-2.0 | Fast model loader (new in v18; not used on GB10/NFS — falls back to safetensors) |
| **NCCL v18** | `local-inference-lab/nccl-canonical` | 2.30.4 | BSD-3-Clause | NCCL fork for v18 stack |

### Shared Components (both v16 and v18)

| Project | Used For |
|---------|----------|
| **QuantTrio GLM-5.2-Int4-Int8Mix** (MIT) | Model weights (256-expert, in-checkpoint MTP) |
| **eugr/spark-vllm-docker** (Apache-2.0) | Multi-stage Docker build, wheel caching, SCP deploy, recipe runner |
| **CosmicRaisins/glm-5.2-gb10** (Apache-2.0) | Foundational GLM-5.2-on-GB10 stack |

### v18-vision Additions (vision + adaptive MTP 2/4)

The `v18-vision` image is a 9-file overlay on top of `v18-prod`. Full credit to the following authors:

| Project | Repo / Source | License | Used For |
|---------|---------------|---------|----------|
| **CosmicRaisins** | [CosmicRaisins/glm-5.2-gb10](https://github.com/CosmicRaisins/glm-5.2-gb10) | Apache-2.0 | Adaptive MTP 2/4/5 controller (`acceptance_length.py`), GLM-5.2 vision model overlay (`glm5v.py`, `Glm5vConfig`, registry registration, MTP-compatible multimodal wrapper), composite checkpoint assembler (`scripts/assemble_quanttrio_glm5v.py`), pr72-1 DCP draft config propagation patch (`llm_base_proposer.py`), rebased adaptive-depth scheduler hooks (`scheduler.py`) |
| **aidendle94 / Aiden Le** | [huggingface.co/aidendle94](https://huggingface.co/aidendle94) | — | Original acceptance-length adaptive speculative decoding controller concept that CosmicRaisins forward-ported and tuned into the 2/4/5 production policy |
| **baseten** | [baseten/GLM-5.2-Vision-NVFP4](https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4) @ `f6eab6117386a0c69152fdf272dc65bfd0254f9f` | — | GLM-5.2 Vision weights — frozen MoonViT-3d vision tower (from Kimi-K2.6) + trained 49.5M-param PatchMerger projector (`1152 → 4607 → 6144`) |
| **QuantTrio** | [QuantTrio/GLM-5.2-Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix) | MIT | Int4-Int8Mix base text checkpoint (256 experts, in-checkpoint MTP layer 78) — the text half of the composite model |
| **Tech2wild / Tony (@Tech2Wild)** | [Tech2wild/GLM-5.3-Int4-Int8Mix](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix) | — | GLM-5.3 Int4-Int8Mix quantized checkpoint (compressed-tensors) — text half of the GLM-5.3 composite (`glm-5.3/`) |
| **local-inference-lab/vllm** | branch `gilded-gnosis-v18` | Apache-2.0 | vLLM fork with DCP + `B12X_MLA_SPARSE` (the v18-prod base the overlay sits on) |
| **eugr/spark-vllm-docker** | [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) | Apache-2.0 | Build harness + cluster launcher (recipe runner) |

The `cudagraph_utils.py` overlay includes a one-line guard (`and self.decode_query_len > 1`) on the adaptive-spec CUDAGraph branch to prevent a `ZeroDivisionError` when v18's speculator runs with `decode_query_len = 1`; the guard falls through to the single-graph replay path, matching CosmicRaisins' intended behavior. See [`glm-5.2/v18-vision/README.md`](glm-5.2/v18-vision/README.md#the-cudagraph_utilspy-fix) for details.

## Key Upstream PRs Incorporated

| PR | Author | Repo | Scope | What It Fixes |
|----|--------|------|-------|---------------|
| **#72** | m9e | `local-inference-lab/vllm` | v16 | DCP draft config propagation, `topk_scores_buffer` for B12X, `build_for_drafting` |
| **#46994** | eastwood-c | `vllm-project/vllm` | v16 | V2+MTP+PP: SupportsPP, broadcast padding, draft relay, embed_tokens, stale topk buffer |
| **#109** | voipmonitor | `local-inference-lab/vllm` | v18 | DSpark hardening |
| **#111** | voipmonitor | `local-inference-lab/vllm` | v18 | TP8 full-CKV DCP prefill |
| **#113** | voipmonitor | `local-inference-lab/vllm` | v18 | NF3 Grid188 integration |
| **#115** | voipmonitor | `local-inference-lab/vllm` | v18 | NVFP4 MLA KV cache support |
| **#116** | voipmonitor | `local-inference-lab/vllm` | v18 | B12X scratch-format guard |
| **#117** | voipmonitor | `local-inference-lab/vllm` | v18 | DCP A2A CUDA graph buffer lifetime |
| **#118** | voipmonitor | `local-inference-lab/vllm` | v18 | MTP target-revision inheritance |
| **#47979** | vllm-project | `vllm-project/vllm` | v18 | SM120 PCIe serving stack |
| **B12X #41** | voipmonitor | `voipmonitor/b12x` | v18 | Deterministic CuTe cache keys |
| **#44993** | vllm-project | `vllm-project/vllm` | both | FSM `should_advance` `new_token_ids` fix |

### v16 Patches (v16/patches/)

| File | Origin | Description |
|------|--------|-------------|
| `01-pr72-1-draft-dcp-config-propagation.patch` | PR #72 part 1 | Propagates `decode_context_parallel_size` to draft config |
| `03-draft-quant-packed-mapping.patch` | PR #72 related | Maps quantized NextN draft tokens correctly |
| `04-v16-essential.patch` | PR #46994 Fix #1, #4 (flashinfer), 6f54d3c | DeepSeekMTP `SupportsPP` + stale `topk_indices_buffer` in `flashinfer_mla_sparse_sm120` + MTP `embed_tokens` loading under PP |
| `06-b12x-stale-topk-buffer.patch` | PR #46994 Fix #4 adapted | **Critical:** B12X_MLA_SPARSE stale `topk_indices_buffer` |
| `05-pp-mtp-broadcast-and-draft-relay.patch` | PR #46994 Fix #2 + #3 | PPHandler broadcast padding + draft token relay (PP2 only) |
| `07-draft-pp-size-fix.patch` | New (same class as PR #72) | `create_draft_parallel_config()` sets `pipeline_parallel_size=1` for draft (PP2 only) |

## Community Contributions

| Contribution | Author | Source | What It Does |
|-------------|--------|--------|--------------|
| **Decode-Aware Custom Scheduler** | [penguinchang](https://forums.developer.nvidia.com/u/penguinchang) | [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/glm-5-2-int4-int8-on-8x-gb10-1-200-t-s-prefill-33-54-t-s-avg-decode-generic-coding-structured/376831) (2026-07-15) | Scheduler patch that prevents long-prefill requests from starving decode streams. Adds dynamic prefill budgets, round-robin long-prefill selection, and runtime enable/disable. |

## Model Provenance

```
GLM-5.2 (744B/40B MoE, GlmMoeDsa)
  └─ Z.ai (original)
      └─ QuantTrio/GLM-5.2-Int4-Int8Mix (w4a16/w8a16, 256 experts, in-checkpoint MTP layer 78)
          └─ cyankiwi (quantization)
```

## KIMI-K3 v2 (sm121 image + RedHat DSpark)

The `kimi-k3/v2` image is built from the **rtx6kpro / blackwell-llm-docker** lineage
(qualified runtime, see [`kimi-k3/v2/BUILD-SM121-IMAGE.md`](kimi-k3/v2/BUILD-SM121-IMAGE.md)):

| Project | Repo / Source | License | Used For |
|---------|---------------|---------|----------|
| **rtx6kpro** | [local-inference-lab/rtx6kpro](https://github.com/local-inference-lab/rtx6kpro) | — | Qualified runtime reference (PyTorch 2.13 / CUDA 13.3 stack) |
| **blackwell-llm-docker** | [local-inference-lab/blackwell-llm-docker](https://github.com/local-inference-lab/blackwell-llm-docker) @ `697f50ff` | — | Build scripts + integration-lock patch system |
| **vLLM** | [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) branch `integration/kimi-k3-ii-cu133-torch213-20260811` @ `881ac39` | Apache-2.0 | vLLM fork with B12X + DSpark + DFlash (sm121 unified-memory fixes) |
| **b12x** | [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) tree `2e6092a` (PRs #124/#138/#139) | Apache-2.0 | CuTe DSL kernels for SM120/SM121 MLA |
| **FlashInfer** | flashinfer `0.6.15.post1` | Apache-2.0 | Attention kernels |
| **InstantTensor** | [voipmonitor/InstantTensor](https://github.com/voipmonitor/InstantTensor) @ `49b4010` | — | Fast model weight loading |
| **Triton kernels** | [triton-lang/triton](https://github.com/triton-lang/triton) `v3.5.1` | MIT | Triton kernels for GB10 |
| **NCCL** | [local-inference-lab/nccl-canonical](https://github.com/local-inference-lab/nccl-canonical) `canonical/cu133-nccl2312-amd-turin` | BSD-3-Clause | NCCL 2.31.2 |
| **RedHat DSpark draft** | [RedHatAI/Kimi-K3-speculator.dspark](https://huggingface.co/RedHatAI/Kimi-K3-speculator.dspark) | — | DSpark draft model (used by v2 recipes) |
| **Inferact DSpark draft** | [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark) | — | Inferact DSpark draft (v1 recipes) |
| **DFlash draft** | [modal-labs/Kimi-K3-DFlash](https://huggingface.co/modal-labs/Kimi-K3-DFlash) | — | DFlash draft (alt backend) |
| **Kimi-K3 weights** | [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | — | Kimi-K3 model weights (BF16 attention + MXFP4 experts) |
| **eugr/spark-vllm-docker** | [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) | Apache-2.0 | Build harness + cluster launcher (recipe runner) |

### KIMI-K3 v2 Runtime Mods

TP16 needs no mods (all fixes baked into the fork's integration branch). The
TP8+PP2 recipe applies the external [`fix-pp2-sm121`](kimi-k3/v2/mods/fix-pp2-sm121/)
mod (PP2 + DSpark infra fixes) at container start.

## Build Systems

- **v16 build:** `eugr/spark-vllm-docker` (multi-stage Dockerfile, wheel caching, SCP parallel deploy, recipe runner)
- **v18 build:** `local-inference-lab/blackwell-llm-docker` (adapted for aarch64/SM121)
- **Deploy (both):** `eugr/spark-vllm-docker` recipe runner

## License

This repository: Apache-2.0  
Served weights: MIT (GLM-5.2 by Z.ai → QuantTrio quants)
