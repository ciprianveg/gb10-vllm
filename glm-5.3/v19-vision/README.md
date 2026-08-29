# v19-vision — GLM-5.3 + Baseten MoonViT vision (same method as GLM-5.2)

GLM-5.3 ([Tech2wild/GLM-5.3-Int4-Int8Mix](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix))
with image understanding on 8× GB10, using the
[`ghcr.io/ciprianveg/gb10-glm-5.2:v19-vision`](https://github.com/ciprianveg/gb10-vllm/pkgs/container/gb10-glm-5.2)
image **unchanged**.

This is the **same vision-adding method as GLM-5.2 v19-vision**
([../../glm-5.2/v19/README.md](../../glm-5.2/v19/README.md)) — a zero-copy composite
directory that symlinks the 5.3 text checkpoint and the Baseten GLM-5.2 vision weights
into one tree. **No model weights are altered**: the assembler only creates relative
symlinks and generates the wrapper metadata (config, weight index, chat template,
preprocessor config, provenance manifest).

Why the 5.2 vision weights work with 5.3: GLM-5.3 shares GLM-5.2's architecture
(`glm_moe_dsa`, hidden 6144, same vocab), so the frozen MoonViT-3d tower + trained
PatchMerger projector (`1152 → 4607 → 6144`) outputs land exactly where the 5.3 text
backbone expects them — the projector transfers with no retraining.

## Recipe

[`recipes/glm53-int4int8-v19-vision.yaml`](recipes/glm53-int4int8-v19-vision.yaml) — the
GLM-5.2 v19-vision recipe with the model swapped to the 5.3 composite. Everything else
(DCP1 query split, deterministic MoE align, adaptive MTP 2/4/5, `B12X_MLA_SPARSE`,
fp8 KV cache, vision flags) is identical.

## Vision tower download + composite assembly

Identical steps to GLM-5.2 — full walkthrough in the
[GLM-5.2 v19 guide](../../glm-5.2/v19/README.md#vision-tower-download--composite-model-assembly).
Summary:

### Step 1 — Download the vision weights (~0.87 GiB)

```bash
VISION_DIR=/path/to/models/baseten-GLM-5.2-Vision-NVFP4

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

### Step 2 — Run the composite assembler

```bash
# HOST paths. The assembler is in this repo at scripts/assemble_glm53_vision.py —
# identical logic to glm-5.2/v18-vision/scripts/assemble_quanttrio_glm5v.py.
TEXT_DIR=/path/to/models/Tech2wild/GLM-5.3-Int4-Int8Mix
VISION_DIR=/path/to/models/baseten-GLM-5.2-Vision-NVFP4
OUTPUT_DIR=/path/to/models/glm53-int4int8-vision      # created by the assembler

python3 scripts/assemble_glm53_vision.py \
  --text-dir  "${TEXT_DIR}" \
  --vision-dir "${VISION_DIR}" \
  --output-dir "${OUTPUT_DIR}"
```

What the assembler does (all zero-copy):

- **Relative symlinks** for all 177569 text tensors + 335 vision/projector tensors —
  the source checkpoints are **untouched**.
- **Wrapper `config.json`** — `Glm5vForConditionalGeneration` config wrapping the 5.3
  text config + Baseten vision config.
- **Merged `model.safetensors.index.json`** — one weight map so vLLM's loader sees a
  single checkpoint.
- **`chat_template.jinja` / `preprocessor_config.json`** — from Baseten (reasoning-effort
  stanza patched as in the 5.2 guide).
- **`GLM5V_COMPOSITE.json`** — provenance manifest (sources, tensor counts, sha256 of
  the vision files).

### Step 3 — Serve

```bash
docker pull ghcr.io/ciprianveg/gb10-glm-5.2:v19-vision
./run-recipe.sh glm-5.3/v19-vision/recipes/glm53-int4int8-v19-vision.yaml
```

Up to 3 images per prompt (`--limit-mm-per-prompt image:3`). The composite dir is
relocatable as long as the three dirs stay siblings; on a shared model directory
(NFS) one assemble run serves the whole cluster.

## Credits

- **Tech2wild / Tony (@Tech2Wild)** —
  [Tech2wild/GLM-5.3-Int4-Int8Mix](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix):
  GLM-5.3 Int4-Int8Mix text checkpoint (text half of the composite).
- **baseten** —
  [baseten/GLM-5.2-Vision-NVFP4](https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4):
  frozen MoonViT-3d vision tower + trained PatchMerger projector (vision half).
- **GLM-5.2 v19** — image, patches, and the composite method itself:
  [../../glm-5.2/v19/README.md](../../glm-5.2/v19/README.md#credits).
