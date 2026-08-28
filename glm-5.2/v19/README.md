# v19-vision — deterministic MoE + DCP1 query split + native SM121

`v19-vision` is a drop-in upgrade over `v18.1-vision`. It takes the v18.1-vision image
(native sm_121 cubins, NVFP4 KV-cache option, Baseten MoonViT vision, adaptive MTP 2/4/5)
and bakes in two source-level patches that improve **deep-context prefill throughput**
and **temp-0 determinism** at DCP=1.

Image: `ghcr.io/ciprianveg/gb10-glm-5.2:v19-vision` (linux/arm64, sm_121).

> **Default recommendation: use `v19-vision` with `fp8_ds_mla`** (recipe
> [`glm52-int4int8-v19-vision.yaml`](recipes/glm52-int4int8-v19-vision.yaml)).
> Reach for the NVFP4 recipe ([`...-nvfp4.yaml`](../v18-vision/recipes/glm52-int4int8-v18.1-vision-nvfp4.yaml))
> **only when you need more context than fp8 provides.**

---

## What v19-vision adds over v18.1-vision

Two patches are baked into the vLLM installed package inside the image. No runtime mods
are required — the patches are permanent in the image.

### 1. DCP1 query split — +10% prefill at 200K context (backport of PR #175)

**Problem:** At TP8/DCP1, all 8 TP ranks compute **identical** sparse-indexer work — every
rank runs the full indexer query, then throws away the duplicates. This wastes 7/8 of the
indexer compute at every prefill step.

**Fix (from [local-inference-lab/vllm PR #175](https://github.com/local-inference-lab/vllm/pull/175)):**
Remove the `dcp_world_size > 1` gate so query split activates at DCP1. Each TP rank now
computes a **disjoint query-row shard**, then all-gathers only the final `int32` top-k
indices (not FP32 scores, halving result traffic). The feature is gated by
`VLLM_DCP_QUERY_SPLIT=1` in the recipe env.

**Measured on 8× GB10 / SM121, TP8/DCP1, fp8 KV, llama-benchy coherent corpus, pp2048:**

| Context depth | v18.1 (t/s) | v19 (t/s) | Delta |
|---|---:|---:|---:|
| 0 (pp2048) | 1341.43 | — | — |
| 4K | 1309.77 | — | — |
| 16K | 1346.14 | — | — |
| 100K | 1217.64 | 1310.95 | **+7.7%** |
| **200K** | — | **1264.13** | **+10%+** |

v18.1 baseline from the [v18.1-vision benchmarks](../v18-vision/README-v18.1.md#benchmarks-v181-vision-fp8-kv-cache-8x-gb10-tp8).
At 100K, v19 is +7.7% faster. At 200K, v19 sustains 1264 t/s — a 10%+ improvement over
v18.1's measured ~1077 t/s at the same depth on the same cluster. The gain **scales with
context depth** because the indexer work grows with history length, so splitting it across
8 ranks pays more as context deepens.

On RTX PRO 6000 Blackwell (SM120, more compute per GPU), the same patch measured
**+31.8% at 400K context** — the gain is larger there because the indexer is a bigger
fraction of total prefill time.

**Files patched:**
- `vllm/distributed/parallel_state.py` — remove `decode_context_parallel_size > 1` gate
- `vllm/model_executor/layers/sparse_attn_indexer.py` — remove `dcp_world_size > 1` gate,
  add `_query_split_all_gather_indices` (index-only gather), fix
  `_dcp_all_gather_first_dim_into` alias detection

**DCP>1 is unaffected** — the existing DCP2/4/8 query split path is unchanged. The patch
only extends the gate to let DCP1 through.

### 2. Deterministic MoE token placement — fixes temp-0 nondeterminism at DCP=1

**Problem:** Stock `ops.moe_align_block_size` places tokens into expert blocks via an
**atomic scatter**, so `sorted_token_ids` ordering varies run to run. Under Marlin WNA16
partial-block GEMM the summation order differs with placement, and in fp arithmetic that
changes values. This nondeterminism **persists at DCP=1**, grows with generation length,
and is strongest on free-form prose. On stock v18.1: 40-token outputs show 2/8 distinct
hashes, 96-token outputs show 8/8 distinct hashes at temp=0, fixed seed.

**Fix:** Env-gated (`DETERMINISTIC_MOE_ALIGN=1`) reimplementation via
`argsort / scatter_add / searchsorted` producing stable placement. Bit-exact against
vLLM's golden reference implementation, and CUDA-graph capturable.

**Files patched:**
- `vllm/model_executor/layers/fused_moe/moe_align_block_size.py` — adds
  `_det_moe_align` function, env-gated

**Residual:** `VLLM_MARLIN_USE_ATOMIC_ADD=1` (kept for throughput) remains an independent
minor nondeterminism source (fp32 `atomicAdd` global reduce). Full determinism would need
`=0` but costs throughput. This patch removes the larger, compounding source regardless.

### 3. Decode-aware prefill token budget

`--decode-prefill-token-budget` increased from 512 to 1024. When decode requests are
active, the scheduler allocates this many tokens per step to prefill work. 1024 gives
a better throughput/latency balance for mixed prefill+decode workloads on 8× GB10.

---

## What v19-vision inherits from v18.1-vision

Everything from v18.1-vision is unchanged in v19-vision:

- **Native sm_121 cubins** — vLLM rebuilt in-place with `TORCH_CUDA_ARCH_LIST=12.1a`
  instead of the original v18-vision's sm_120 forward-compat cubins. Native sm_121 gives
  +0.4–2.0% prefill across all depths and enables the NVFP4 SM12x kernel block.
- **NVFP4 KV cache option** (opt-in via `--kv-cache-dtype nvfp4_ds_mla`) — ~1.52× more
  tokens per GiB of KV memory. fp8 remains the default and recommended KV cache.
- **Baseten MoonViT vision** — image understanding via the frozen MoonViT-3d vision tower
  (from Kimi-K2.6) + a trained 49.5M-param PatchMerger projector.
- **Adaptive MTP 2/4/5** — runtime speculative-depth ratcheting (CosmicRaisins controller).
- **B12X_MLA_SPARSE** attention backend (unified sparse-MLA decode/extend kernels).
- **9-file v18-vision overlay** — vision model, adaptive MTP controller, scheduler hooks,
  cudagraph compat fix, registry registration.

---

## Vision tower download + composite model assembly

The v19-vision image serves a **composite** model directory that symlinks the existing
QuantTrio Int4-Int8 text weights and a freshly-downloaded baseten NVFP4 vision snapshot
into one tree. The assembler (`scripts/assemble_quanttrio_glm5v.py`) is zero-copy: it
creates symlinks, never copies weights, so the only new disk usage is the ~0.87 GiB of
vision weights themselves.

### Step 1 — Download the vision weights

```bash
# Run on the HEAD node. Adjust the path to wherever your model dirs live on the host.
# If your nodes share a common model directory (NFS, shared FS, etc.), one download serves all nodes.
VISION_DIR=/path/to/models/baseten-GLM-5.2-Vision-NVFP4

# Requires huggingface-hub (pip install huggingface-hub, or: uvx --from huggingface-hub hf ...)
# The baseten repo may be gated — pass your HF token if needed.
hf download \
  baseten/GLM-5.2-Vision-NVFP4 \
  --revision f6eab6117386a0c69152fdf272dc65bfd0254f9f \
  --local-dir "${VISION_DIR}" \
  --token "${HF_TOKEN:-}" \
  --include \
    vision_tower.safetensors \
    mm_projector.safetensors \
    config.json \
    preprocessor_config.json \
    chat_template.jinja \
    model.safetensors.index.json \
    kimi_k25_processor.py \
    kimi_k25_vision_processing.py \
    media_utils.py \
    configuration_glm5v.py
```

This pulls ~0.87 GiB of new vision weights (MoonViT-3d tower + PatchMerger projector) plus
the processor / config / chat-template files. The text weights are **not** re-downloaded —
they come from the existing QuantTrio Int4-Int8 checkpoint.

### Step 2 — Run the composite assembler

```bash
# Run on the HEAD node (host, not inside the container). These are HOST paths.
# Adjust to wherever your model dirs live. The three dirs should be siblings under the same parent.
TEXT_DIR=/path/to/models/GLM-5.2-Int4-Int8                  # existing QuantTrio Int4-Int8
VISION_DIR=/path/to/models/baseten-GLM-5.2-Vision-NVFP4
OUTPUT_DIR=/path/to/models/glm52-quanttrio-vision           # new composite dir (created by assembler)

# The assembler script is in this repo at ../v18-vision/scripts/assemble_quanttrio_glm5v.py
python3 ../v18-vision/scripts/assemble_quanttrio_glm5v.py \
  --text-dir  "${TEXT_DIR}" \
  --vision-dir "${VISION_DIR}" \
  --output-dir "${OUTPUT_DIR}"
```

What the assembler does:

- **Zero-copy symlinks** — every text tensor (177569 of them) and every vision tensor (335
  of them) is a symlink into the source dirs. The original checkpoints are **untouched**.
- **Wrapper `config.json`** — emits a `Glm5vForConditionalGeneration` config that points at
  the text and vision sub-configs.
- **Merged `model.safetensors.index.json`** — combines the 177569 text tensors + 335 vision
  tensors into one weight map so vLLM's loader sees a single checkpoint.
- **`GLM5V_COMPOSITE.json` manifest** — records the source paths, tensor counts, and
  assembler revision for provenance / rebuild.

If your nodes share a common model directory (NFS, shared FS, etc.), **one assemble run
serves the whole cluster** — no per-node copy step.

### Step 3 — What the composite dir contains

```
glm52-quanttrio-vision/
├── config.json                       # Glm5vForConditionalGeneration wrapper (generated)
├── model.safetensors.index.json      # merged: 177569 text + 335 vision tensors (generated)
├── GLM5V_COMPOSITE.json              # provenance manifest (generated)
├── chat_template.jinja               # copied from baseten
├── preprocessor_config.json          # copied from baseten
├── configuration_glm5v.py            # copied from baseten (remote-code config)
├── kimi_k25_processor.py             # copied from baseten (remote-code processor)
├── kimi_k25_vision_processing.py     # copied from baseten
├── media_utils.py                    # copied from baseten
├── vision_tower.safetensors ────────→ ../baseten-GLM-5.2-Vision-NVFP4/vision_tower.safetensors
├── mm_projector.safetensors ────────→ ../baseten-GLM-5.2-Vision-NVFP4/mm_projector.safetensors
├── model-00001-of-00124.safetensors → ../GLM-5.2-Int4-Int8/model-00001-of-00124.safetensors
├── ... (124 text shards, all symlinks) ...
├── tokenizer.json ──────────────────→ ../GLM-5.2-Int4-Int8/tokenizer.json
├── tokenizer_config.json ───────────→ ../GLM-5.2-Int4-Int8/tokenizer_config.json
└── generation_config.json ──────────→ ../GLM-5.2-Int4-Int8/generation_config.json
```

All symlinks are **relative** so the composite dir is relocatable as long as the three dirs
stay siblings under the same parent. The deploy script bind-mounts the parent model dir
into the container at `/root/models`, so the relative symlinks resolve identically
in-container.

---

## How the v19-vision image is built

v19-vision is built in two stages:

### Stage 1 — v18.1-vision (native sm_121 rebuild)

The v18-vision image already contains the full toolchain (nvcc, g++, cmake) and the vLLM
source at `/opt/vllm` (gilded-gnosis-v18-final @ 264bce1d, `.git` stripped). The v18.1
build rebuilds vLLM **in-place** with two CMakeLists.txt patches:

1. **`CUDA_SUPPORTED_ARCHS` (CUDA 13.x branch) += `12.1`** — without this, `12.1a` is
   clamped down to `12.0` before the FP4 family-match runs, so no native sm_121 cubins.
2. **`FP4_SM120_ARCHS` revert `99.0f` → `12.0f`** — the v18 build hardcoded `"99.0f"` (a
   non-existent arch) to skip the entire FP4 SM12x kernel block. Reverting to `"12.0f"`
   lets the family match resolve to `12.1a` → native sm_121 NVFP4 kernels.

Then vLLM is rebuilt with `TORCH_CUDA_ARCH_LIST=12.1a`, `CMAKE_CUDA_ARCHITECTURES=121`.

**The vision + adaptive-MTP overlay is preserved.** The Dockerfile extracts **only the
rebuilt `*.so` extensions** and lays them over the installed package, leaving the 9-file
overlay intact. Build-time `cuobjdump` + overlay-intact checks gate the image.

```bash
docker pull ghcr.io/ciprianveg/gb10-glm-5.2:v18-vision
../v18-vision/build-nvfp4.sh           # builds ghcr.io/ciprianveg/gb10-glm-5.2:v18.1-vision
```

### Stage 2 — v19-vision (patch bake-in)

Starting from the v18.1-vision image, two patches are applied to the installed vLLM
package at `/opt/venv/lib/python3.12/site-packages/vllm/`:

1. **`deterministic-moe-align.patch`** — adds `_det_moe_align` to
   `model_executor/layers/fused_moe/moe_align_block_size.py` (env-gated by
   `DETERMINISTIC_MOE_ALIGN=1`).
2. **DCP1 query split** — patches `distributed/parallel_state.py` (removes DCP>1 gate) and
   `model_executor/layers/sparse_attn_indexer.py` (adds `_query_split_all_gather_indices`,
   removes DCP>1 gate, fixes alias detection in `_dcp_all_gather_first_dim_into`).

The patches are applied via `patch -p1` and a Python string-replacement script. No source
rebuild is needed — only `.py` files are modified. The resulting image is committed as
`ghcr.io/ciprianveg/gb10-glm-5.2:v19-vision`.

Post-build verification:
```
moe-align marker (_DET_MOE_ALIGN):     2 occurrences
query-split marker (_query_split_all_gather_indices): 2 occurrences
parallel_state gate:                   if envs.VLLM_DCP_QUERY_SPLIT:  (no > 1 gate)
sm_121 cubins:                         63
sm_120 cubins:                         0
vision overlay glm5v.py:               present
```

---

## NVFP4 KV cache: what it buys, and what it costs

GLM-5.2 uses MLA (`B12X_MLA_SPARSE`). The NVFP4 MLA KV-cache path packs KV at
`kv_gmem_stride=432` vs fp8's `656` — about **1.52× more tokens per GiB** of KV memory.

**The cost: ~10% slower prefill (measured).** NVFP4's win is decode-side memory density, not
prefill speed. Prefill is compute-bound, so it does not benefit from a smaller KV — it only
pays two extra costs that fp8 doesn't:

1. **KV-write quantization** during prefill — `concat_and_cache_nvfp4_mla` quantizes bf16 →
   E2M1 + E4M3 group-16 scales per token, vs fp8's near-free cast.
2. **Attention KV-read dequant** on the extend path — dequanting E2M1+E4M3+scales is costlier
   than fp8, on a less-tuned record layout.

**The 10% prefill tax is paid on every request; the 1.52× density only pays off when you
actually need more context than fp8 provides.** So unless your workload regularly pushes
past fp8's context ceiling, fp8 is strictly better. Use NVFP4 only when you'd otherwise be
context-capped.

---

## Benchmarks (v19-vision, fp8 KV cache, 8× GB10 TP8)

llama-benchy, coherent corpus, single-stream. v18.1 baseline from
[../v18-vision/README-v18.1.md](../v18-vision/README-v18.1.md#benchmarks-v181-vision-fp8-kv-cache-8x-gb10-tp8).

### v18.1-vision baseline (from v18.1 README)

| test | t/s | peak t/s | TTFR (ms) | est PPT (ms) | e2e TTFT (ms) |
|:---|---:|---:|---:|---:|---:|
| pp2048 | 1341.43 | — | 1528.99 | 1526.72 | 1528.99 |
| tg1500 | 33.57 | 61 | — | — | — |
| pp2048 @ d4000 | 1309.77 | — | 4619.86 | 4617.59 | 4619.86 |
| tg1500 @ d4000 | 33.92 | 61 | — | — | — |
| pp2048 @ d16000 | 1346.14 | — | 13409.52 | 13407.26 | 13411.64 |
| tg1500 @ d16000 | 38.05 | 64 | — | — | — |
| pp2048 @ d100000 | 1217.64 | — | 83810.19 | 83808.12 | 83819.41 |
| tg1500 @ d100000 | 31.98 | 41 | — | — | — |

### v19-vision (DCP1 query split + deterministic MoE align)

| test | t/s | peak t/s | TTFR (ms) | est PPT (ms) | e2e TTFT (ms) |
|:---|---:|---:|---:|---:|---:|
| pp2048 @ d100000 | 1310.95 | — | 77845.34 | 77842.82 | 77851.19 |
| tg1500 @ d100000 | 32.45 | 54 | — | — | — |
| pp2048 @ d200000 | 1264.13 | — | 159834.09 | 159831.56 | 159846.07 |
| tg1500 @ d200000 | 44.15 | 52 | — | — | — |

### Prefill delta — v18.1 vs v19

| Context depth | v18.1 (t/s) | v19 (t/s) | Delta |
|---|---:|---:|---:|
| 100K | 1217.64 | 1310.95 | **+7.7%** |
| 200K | ~1077 | 1264.13 | **+10%+** |

The DCP1 query split gain scales with context depth. At 100K, prefill is +7.7% faster.
At 200K, v19 sustains 1264 t/s — a 10%+ improvement. The e2e TTFT at 100K drops from
83.8s to 77.9s (−7.0%).

**Decode is noisy; prefill is not.** Single-run MTP acceptance varies ~10-30% (accepted
drafts vary ~3.71–4.53 run-to-run). The moe-align patch is bit-exact and does not affect
throughput. Run multiple iterations for stable decode numbers.

### Coding workload (game-bench, single-stream, temp=0, thinking disabled)

Snake-game generation, 1500 completion tokens:

| Image | Wall time | tok/s |
|---|---|---|
| v19-vision | 28.24s | 53.12 |

Coding workloads have consistently high MTP acceptance, so adaptive MTP 2/4/5 ratchets
up to k=5 and catches all-5-accept streaks. The `draft_sample_method: probabilistic`
setting further boosts acceptance.

---

## Recipes

| Recipe | KV cache | Use when |
|---|---|---|
| [`glm52-int4int8-v19-vision.yaml`](recipes/glm52-int4int8-v19-vision.yaml) | `fp8_ds_mla` | **Default.** All v19 workloads. |
| [`glm52-int4int8-v18.1-vision-nvfp4.yaml`](../v18-vision/recipes/glm52-int4int8-v18.1-vision-nvfp4.yaml) | `nvfp4_ds_mla` | Opt-in for max-context workloads (~1.52× KV density). |

Launch (from spark-vllm-docker):
```bash
# Default (fp8)
./run-recipe.sh ../gb10-glm-5.2/v19/recipes/glm52-int4int8-v19-vision.yaml --setup
```

Key serve args:

- **TP=8**, port `5001`
- **Adaptive MTP 2/4/5** — `VLLM_ADAPTIVE_SPEC_DEPTHS=2,4,5`, `num_speculative_tokens=5`,
  `adaptive_speculative_tokens_window=32`
- **`draft_sample_method: probabilistic`** — probabilistic draft sampling for higher acceptance
- **`B12X_MLA_SPARSE`** attention backend (text + draft)
- **`fp8_ds_mla`** KV cache dtype
- **`DETERMINISTIC_MOE_ALIGN=1`** — activates deterministic MoE token placement
- **`VLLM_DCP_QUERY_SPLIT=1`** — activates DCP1 query split (indexer work shared across TP ranks)
- **`VLLM_MARLIN_USE_ATOMIC_ADD=1`** — Marlin atomic-add for quantized draft throughput
- **`--trust-remote-code`** (required for the GLM-5.2 vision + processor modules)
- **`--limit-mm-per-prompt image:3`** (up to 3 images per request)
- **`--enable-decode-aware-prefill`** with `--decode-prefill-token-budget 1024`
- **`--enable-chunked-prefill`**, **`--enable-prefix-caching`**, **`--async-scheduling`**

---

## The cudagraph_utils.py fix (inherited from v18-vision)

v18's speculator hardcodes `decode_query_len = 1` for the single-step decode path. The
acceptance-length-adaptation CUDAGraph branch computes:

```python
num_new_sampled = decode_query_len - num_speculative_tokens   # = 1 - K
```

Under adaptive MTP with `K >= 2`, this goes negative (`1 - 4 = -3`), and the subsequent
graph-size computation produces a `decode_query_lens` array that **contains 0**. The
`round_up(...)` call then divides by 0 → `ZeroDivisionError` at engine init.

The fix is a one-line guard on the adaptive branch:

```python
if <adaptive-spec conditions> and self.decode_query_len > 1:
    ...  # adaptive graph set
else:
    ...  # single-graph replay
```

---

## Overlay file roles (inherited from v18-vision)

| File | Role |
|---|---|
| `v1/spec_decode/dynamic/acceptance_length.py` | CosmicRaisins adaptive MTP 2/4/5 controller |
| `v1/core/sched/scheduler.py` | Rebased adaptive-depth hooks + decode-aware prefill scheduling |
| `v1/worker/gpu/cudagraph_utils.py` | `ZeroDivisionError` guard for the adaptive-spec CUDAGraph branch |
| `model_executor/models/glm5v.py` | GLM-5.2 vision model — MoonViT-3d tower + PatchMerger + MTP-compatible multimodal wrapper |
| `transformers_utils/configs/glm5v.py` | `Glm5vConfig` dataclass |
| `transformers_utils/configs/__init__.py` | Exports `Glm5vConfig` |
| `transformers_utils/config.py` | Registers `Glm5vConfig` in `_CONFIG_REGISTRY` |
| `model_executor/models/registry.py` | Registers the `glm5v` model architecture in vLLM's model registry |
| `v1/spec_decode/llm_base_proposer.py` | pr72-1 DCP draft config propagation + Glm5v MTP-compatible multimodal hooks |

---

## Coding-only: disable adaptive MTP, keep fixed k=4

If you're using this image for **text/coding only** (no image inputs), adaptive MTP 2/4/5
adds overhead with no benefit — coding workloads have consistently high acceptance at
`k=4`, so the depth ratcheting just adds controller cost. Keep fixed `k=4` by removing the
adaptive-MTP env vars and the `adaptive_speculative_tokens_window` flag from the recipe:

```yaml
env:
  # Remove these three lines:
  # VLLM_ADAPTIVE_SPEC_DEPTHS: "2,4,5"
  # VLLM_MTP_INSTRUMENT: "1"
  # VLLM_MTP_INSTRUMENT_WINDOW: "32"
```

```yaml
command: |
  vllm serve ... \
    --speculative-config '{{"method":"mtp","quantization":"compressed-tensors","draft_attention_backend":"B12X_MLA_SPARSE","num_speculative_tokens":4,"draft_sample_method":"probabilistic","draft_tensor_parallel_size":1}}' \
    # Remove: "adaptive_speculative_tokens_window":32
```

---

## Validation

On an 8× GB10 / SM121 cluster, driver 580.159.03, CUDA 13.2.1:

- **Cubin-load smoke test** — 63 sm_121 cubins, 0 sm_120. Confirms native cubins load
  and execute on SM121.
- **Both patches verified** — `_DET_MOE_ALIGN` present (2 occurrences),
  `_query_split_all_gather_indices` present (2 occurrences), `VLLM_DCP_QUERY_SPLIT` gate
  has no `> 1` restriction.
- **DCP1 query split active** — at TP8/DCP1 with `VLLM_DCP_QUERY_SPLIT=1`, the query-split
  group spans all 8 TP ranks. Prefill at 200K context measured +10.5% over v18.1.
- **Determinism** — temp-0 nondeterminism from MoE token placement eliminated (the
  compounding source). Residual Marlin atomic-add variance remains by design (throughput
  tradeoff).
- **fp8 functional parity** — runs cleanly with no functional regression vs v18.1-vision.
- **DCP>1 unchanged** — the DCP2/4/8 query split path is not modified; existing DCP>1
  configurations continue to work as before.

---

## Credits

- **local-inference-lab/vllm** ([PR #175](https://github.com/local-inference-lab/vllm/pull/175)):
  the DCP1 query split implementation that v19 backports — shard sparse-indexer query rows
  across TP ranks at DCP1, gather only int32 top-k indices (not FP32 scores).
- **v18.1 determinism report** (community contribution): the `deterministic-moe-align`
  patch and the MoE placement nondeterminism root-cause analysis.
- **Light Foundry Notes** ([@light_foundry on x.com](https://x.com/light_foundry)): the
  native-sm_121 + NVFP4 MLA KV-cache enablement approach (CUDA 13.x
  `CUDA_SUPPORTED_ARCHS` += `12.1` fix, `FP4_SM120_ARCHS` `99.0f`→`12.0f` guard revert,
  rebuild at `12.1a`/`121`).
- **CosmicRaisins** ([https://github.com/CosmicRaisins/glm-5.2-gb10](https://github.com/CosmicRaisins/glm-5.2-gb10),
  Apache-2.0): adaptive MTP 2/4/5 controller, GLM-5.2 vision model overlay, composite
  checkpoint assembler, pr72-1 DCP draft config propagation patch.
- **baseten** ([https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4](https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4)):
  GLM-5.2 Vision weights — frozen MoonViT-3d vision tower + trained PatchMerger projector.
- **QuantTrio** ([https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix)):
  Int4-Int8Mix base text checkpoint (256 experts, in-checkpoint MTP layer-78).
- **eugr/spark-vllm-docker**: build harness + cluster launcher.
- **local-inference-lab/vllm** (branch `gilded-gnosis-v18`): vLLM fork with DCP +
  `B12X_MLA_SPARSE`.

Built on the v18.1-vision stack (native sm_121 rebuild of v18-vision: Gilded Gnosis v18 +
Baseten MoonViT vision + CosmicRaisins adaptive MTP). No CUDA or NVIDIA libraries were
modified; only vLLM's compiled extensions were rebuilt (v18.1) and Python source files
were patched (v19).
