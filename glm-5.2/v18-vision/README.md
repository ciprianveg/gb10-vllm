# GLM-5.2 Int4-Int8Mix + Vision + Adaptive MTP on 8× GB10 DGX Spark

## What this is

`v18-prod` base + GLM-5.2 vision (MoonViT/PatchMerger from baseten) + adaptive MTP 2/4/5 (from CosmicRaisins).

Image: `ghcr.io/ciprianveg/gb10-glm-5.2:v18-vision` (linux/arm64, sm_121).

This is a thin 9-file overlay on top of the v18-prod production image. It adds:

- **Vision** — image understanding via the frozen MoonViT-3d vision tower (from Kimi-K2.6) + a trained 49.5M-param PatchMerger projector (`1152 → 4607 → 6144`).
- **Adaptive MTP 2/4** — runtime speculative-depth ratcheting (2 → 4 based on acceptance), vs the v18-prod fixed `k=4`.
- **Cudagraph compat fix** for the adaptive-spec branch on v18 (see [The cudagraph_utils.py fix](#the-cudagraph_utilspy-fix)).

The 9 overlaid files live in [`overlay/vllm/`](overlay/vllm/) and are COPYed into `/opt/venv/lib/python3.12/site-packages/vllm/...` at build time. See [`Dockerfile`](Dockerfile) for the exact destinations.

## Vision tower download + composite model assembly

The v18-vision image serves a **composite** model directory that symlinks the existing QuantTrio Int4-Int8 text weights and a freshly-downloaded baseten NVFP4 vision snapshot into one tree. The assembler (`scripts/assemble_quanttrio_glm5v.py`) is zero-copy: it creates symlinks, never copies weights, so the only new disk usage is the ~0.87 GiB of vision weights themselves.

### Step 1 — Download the vision weights

Pull the baseten GLM-5.2-Vision-NVFP4 snapshot at the pinned revision into a sibling directory next to your existing QuantTrio Int4-Int8 checkpoint. The `--include` list is exhaustive — these are the only files needed by the vLLM vision model + processor:

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

This pulls ~0.87 GiB of new vision weights (MoonViT-3d tower + PatchMerger projector) plus the processor / config / chat-template files. The text weights are **not** re-downloaded — they come from the existing QuantTrio Int4-Int8 checkpoint.

### Step 2 — Run the composite assembler

```bash
# Run on the HEAD node (host, not inside the container). These are HOST paths.
# Adjust to wherever your model dirs live. The three dirs should be siblings under the same parent.
TEXT_DIR=/path/to/models/GLM-5.2-Int4-Int8                  # existing QuantTrio Int4-Int8
VISION_DIR=/path/to/models/baseten-GLM-5.2-Vision-NVFP4
OUTPUT_DIR=/path/to/models/glm52-quanttrio-vision           # new composite dir (created by assembler)

# The assembler script is in this repo at v18-vision/scripts/assemble_quanttrio_glm5v.py
python3 v18-vision/scripts/assemble_quanttrio_glm5v.py \
  --text-dir  "${TEXT_DIR}" \
  --vision-dir "${VISION_DIR}" \
  --output-dir "${OUTPUT_DIR}"
```

What the assembler does:

- **Zero-copy symlinks** — every text tensor (177569 of them) and every vision tensor (335 of them) is a symlink into the source dirs. The original QuantTrio and baseten checkpoints are **untouched**.
- **Wrapper `config.json`** — emits a `Glm5vForConditionalGeneration` config that points at the text and vision sub-configs.
- **Merged `model.safetensors.index.json`** — combines the 177569 text tensors + 335 vision tensors into one weight map so vLLM's loader sees a single checkpoint.
- **`GLM5V_COMPOSITE.json` manifest** — records the source paths, tensor counts, and assembler revision for provenance / rebuild.

If your nodes share a common model directory (NFS, shared FS, etc.), **one assemble run serves the whole cluster** — no per-node copy step. Otherwise, rsync the composite dir (or re-run the assembler) on each node.

### Step 3 — What the composite dir contains

```
glm52-quanttrio-vision/
├── config.json                       # Glm5vForConditionalGeneration wrapper (generated)
├── model.safetensors.index.json      # merged: 177569 text + 335 vision tensors (generated)
├── GLM5V_COMPOSITE.json              # provenance manifest (generated)
├── chat_template.jinja               # copied from baseten (reasoning_effort overridden to medium-high)
├── preprocessor_config.json          # copied from baseten
├── configuration_glm5v.py            # copied from baseten (remote-code config)
├── kimi_k25_processor.py             # copied from baseten (remote-code processor)
├── kimi_k25_vision_processing.py     # copied from baseten
├── media_utils.py                    # copied from baseten
├── vision_tower.safetensors ────────→ ../baseten-GLM-5.2-Vision-NVFP4/vision_tower.safetensors
├── mm_projector.safetensors ────────→ ../baseten-GLM-5.2-Vision-NVFP4/mm_projector.safetensors
├── model-00001-of-00124.safetensors → ../GLM-5.2-Int4-Int8/model-00001-of-00124.safetensors
├── model-00002-of-00124.safetensors → ../GLM-5.2-Int4-Int8/model-00002-of-00124.safetensors
├── ... (124 text shards, all symlinks) ...
├── model.safetensors.index.json ────→ (generated, not symlinked)
├── tokenizer.json ──────────────────→ ../GLM-5.2-Int4-Int8/tokenizer.json
├── tokenizer_config.json ───────────→ ../GLM-5.2-Int4-Int8/tokenizer_config.json
└── generation_config.json ──────────→ ../GLM-5.2-Int4-Int8/generation_config.json
```

All symlinks are **relative** (`../GLM-5.2-Int4-Int8/...`, `../baseten-GLM-5.2-Vision-NVFP4/...`) so the composite dir is relocatable as long as the three dirs stay siblings under the same parent. The deploy script bind-mounts the parent model dir (e.g. `/path/to/models`) into the container at `/root/models`, so the relative symlinks resolve identically in-container.

## Building the image

Layer over `v18-prod`. From the repo root:

```bash
docker pull ghcr.io/ciprianveg/gb10-glm-5.2:v18-prod
./v18-vision/build.sh           # build only
```

The Dockerfile COPYs the 9 overlay files into `/opt/venv/lib/python3.12/site-packages/vllm/...` and runs `python3 -m compileall -q` on them. No source rebuild — this is a sub-second layer on top of the v18-prod image.

Overlay file roles:

| File | Role |
|---|---|
| `v1/spec_decode/dynamic/acceptance_length.py` | CosmicRaisins adaptive MTP 2/4/5 controller — picks the spec depth per step from the rolling acceptance window |
| `v1/core/sched/scheduler.py` | Rebased adaptive-depth hooks into the v18 scheduler |
| `v1/worker/gpu/cudagraph_utils.py` | `ZeroDivisionError` guard for the adaptive-spec CUDAGraph branch (see below) |
| `model_executor/models/glm5v.py` | GLM-5.2 vision model — MoonViT-3d tower + PatchMerger + MTP-compatible multimodal wrapper |
| `transformers_utils/configs/glm5v.py` | `Glm5vConfig` dataclass |
| `transformers_utils/configs/__init__.py` | Exports `Glm5vConfig` |
| `transformers_utils/config.py` | Registers `Glm5vConfig` in `_CONFIG_REGISTRY` |
| `model_executor/models/registry.py` | Registers the `glm5v` model architecture in vLLM's model registry |
| `v1/spec_decode/llm_base_proposer.py` | pr72-1 DCP draft config propagation + Glm5v MTP-compatible multimodal hooks |

## The cudagraph_utils.py fix

v18's speculator hardcodes `decode_query_len = 1` for the single-step decode path. The acceptance-length-adaptation CUDAGraph branch, however, computes:

```python
num_new_sampled = decode_query_len - num_speculative_tokens   # = 1 - K
```

Under adaptive MTP with `K >= 2`, this goes negative (`1 - 4 = -3`), and the subsequent graph-size computation produces a `decode_query_lens` array that **contains 0**. The `round_up(...)` call inside the CUDAGraph sizing then divides by that 0 → `ZeroDivisionError` at engine init.

The fix is a one-line guard on the adaptive branch:

```python
if <adaptive-spec conditions> and self.decode_query_len > 1:
    ...  # adaptive graph set
else:
    ...  # single-graph replay (matches CosmicRaisins' intended behavior)
```

With `decode_query_len == 1` (the v18 speculator's case), the guard falls through to the `else` branch, which replays the single pre-captured decode graph — exactly what CR's original adaptive controller expects. The adaptive path only engages when the speculator actually runs multi-token draft queries (`decode_query_len > 1`), which is the DCP≥2 case the original CR code was written for.

## Launch

Recipe: [`recipes/glm52-int4int8-v18-vision.yaml`](recipes/glm52-int4int8-v18-vision.yaml).

```bash
# From spark-vllm-docker
./run-recipe.sh ../gb10-glm-5.2/v18-vision/recipes/glm52-int4int8-v18-vision.yaml --setup
```

Key serve args:

- **TP=8**, port `5001`
- **Adaptive MTP 2/4** — `VLLM_ADAPTIVE_SPEC_DEPTHS=2,4`, `num_speculative_tokens=4`, `adaptive_speculative_tokens_window=32`
- **`B12X_MLA_SPARSE`** attention backend (text + draft)
- **`fp8_ds_mla`** KV cache dtype
- **`--trust-remote-code`** (required for the GLM-5.2 vision + processor modules)
- **`--limit-mm-per-prompt image:4`** (up to 4 images per request)
- **`draft_tensor_parallel_size:1`** (draft runs on a single rank)
- `--max-model-len 550000`, `--gpu-memory-utilization 0.79` (conservative first-boot; see recipe header comment)

## What this adds over v18-prod

- **Vision** — image understanding via MoonViT-3d + PatchMerger projector (baseten NVFP4 weights, ~0.87 GiB).
- **Adaptive MTP 2/4** — runtime depth ratcheting based on acceptance (vs v18-prod's fixed `k=4`).
- **Cudagraph compat fix** for the adaptive-spec branch on v18's `decode_query_len = 1` speculator.

## Coding-only: disable adaptive MTP, keep fixed k=4

If you're using this image for **text/coding only** (no image inputs), adaptive MTP 2/4 adds overhead with no benefit — coding workloads have consistently high acceptance at `k=4`, so the depth ratcheting just adds controller cost. Keep fixed `k=4` (same as v18-prod) by removing the adaptive-MTP env vars and the `adaptive_speculative_tokens_window` flag from the recipe:

```yaml
env:
  # Remove these two lines:
  # VLLM_ADAPTIVE_SPEC_DEPTHS: "2,4"
  # VLLM_MTP_INSTRUMENT: "1"
  # VLLM_MTP_INSTRUMENT_WINDOW: "32"
```

```yaml
command: |
  vllm serve ... \
    --speculative-config '{{"method":"mtp","quantization":"compressed-tensors","draft_attention_backend":"B12X_MLA_SPARSE","num_speculative_tokens":4,"draft_tensor_parallel_size":1}}' \
    # Remove: "adaptive_speculative_tokens_window":32
```

With adaptive MTP disabled, the overlay's `acceptance_length.py` controller is inert (no `VLLM_ADAPTIVE_SPEC_DEPTHS` env → no adaptation), and the speculator runs fixed `k=4` exactly like v18-prod. You still get vision support — just without the adaptive-depth overhead.

Alternatively, for pure text/coding workloads with no vision needs at all, just use `ghcr.io/ciprianveg/gb10-glm-5.2:v18-prod` directly.

## Credits

- **CosmicRaisins** ([https://github.com/CosmicRaisins/glm-5.2-gb10](https://github.com/CosmicRaisins/glm-5.2-gb10), Apache-2.0): adaptive MTP 2/4/5 controller (`acceptance_length.py`), GLM-5.2 vision model overlay (`glm5v.py`, `Glm5vConfig`, registry registration, MTP-compatible multimodal wrapper), composite checkpoint assembler, pr72-1 DCP draft config propagation patch.
- **aidendle94 / Aiden Le** ([https://huggingface.co/aidendle94](https://huggingface.co/aidendle94)): the original acceptance-length adaptive speculative decoding controller concept that CosmicRaisins forward-ported and tuned into the 2/4/5 production policy.
- **baseten** ([https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4](https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4)): GLM-5.2 Vision weights — frozen MoonViT-3d vision tower (from Kimi-K2.6) + trained 49.5M-param PatchMerger projector (`1152 → 4607 → 6144`). Revision `f6eab6117386a0c69152fdf272dc65bfd0254f9f`.
- **QuantTrio** ([https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix)): Int4-Int8Mix base text checkpoint (256 experts, in-checkpoint MTP layer-78).
- **eugr/spark-vllm-docker**: build harness + cluster launcher.
- **local-inference-lab/vllm** (branch `gilded-gnosis-v18`): vLLM fork with DCP + `B12X_MLA_SPARSE`.
