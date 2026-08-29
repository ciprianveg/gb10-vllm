# GLM-5.3 (Int4-Int8Mix) on GB10 — the v19 stack, text or text+vision

GLM-5.3 inference on NVIDIA DGX Spark (GB10, SM121) using the **proven GLM-5.2 v19 image,
unchanged** — no rebuild, no runtime mods.

**Model:** [Tech2wild/GLM-5.3-Int4-Int8Mix](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix)
(compressed-tensors Int4-Int8Mix) — thanks to **Tony (@Tech2Wild)** for the quant.

**Why it just works:** GLM-5.3 is the same architecture as GLM-5.2 — `glm_moe_dsa`,
hidden 6144, same vocab, same MTP layout, same sparse-indexer config. Everything the
v19 stack does (DCP1 query split, deterministic MoE align, `B12X_MLA_SPARSE` attention,
adaptive MTP 2/4/5, native sm_121 cubins) applies to 5.3 verbatim. The v19 long-context
prefill gains measured on 5.2 ([+7.7% @ 100K, +10%+ @ 200K](../glm-5.2/v19/README.md#prefill-delta--v181-vs-v19))
translate directly to 5.3.

**Image:** [`ghcr.io/ciprianveg/gb10-glm-5.2:v19-vision`](https://github.com/ciprianveg/gb10-vllm/pkgs/container/gb10-glm-5.2)
(linux/arm64, sm_121) — the exact image tested with GLM-5.2 v19.

## Versions

| Version | Guide | Recipe | What |
|---|---|---|---|
| **v19** (text-only) | [v19/README.md](v19/README.md) | [`v19/recipes/glm53-int4int8-v19.yaml`](v19/recipes/glm53-int4int8-v19.yaml) | Serves the Tech2wild checkpoint directly. TP=8 + PP=1, fp8 KV cache, adaptive MTP 2/4/5. |
| **v19-vision** (text + vision) | [v19-vision/README.md](v19-vision/README.md) | [`v19-vision/recipes/glm53-int4int8-v19-vision.yaml`](v19-vision/recipes/glm53-int4int8-v19-vision.yaml) | Zero-copy composite: 5.3 text + frozen Baseten MoonViT-3d vision tower + PatchMerger projector. Same vision-adding method as GLM-5.2. |

Both recipes are the [GLM-5.2 v19 recipe](../glm-5.2/v19/recipes/glm52-int4int8-v19-vision.yaml)
with only the model swapped (and the mm flags removed for text-only).

## Quick start

```bash
docker pull ghcr.io/ciprianveg/gb10-glm-5.2:v19-vision

# Text-only (8× GB10 cluster)
./run-recipe.sh glm-5.3/v19/recipes/glm53-int4int8-v19.yaml

# Text + vision (assemble the composite first — one time, ~0.87 GiB download)
# See v19-vision/README.md for the full guide.
./run-recipe.sh glm-5.3/v19-vision/recipes/glm53-int4int8-v19-vision.yaml
```

## Credits

- **Tech2wild / Tony (@Tech2Wild)** —
  [Tech2wild/GLM-5.3-Int4-Int8Mix](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix):
  the GLM-5.3 Int4-Int8Mix quantized checkpoint (text half of the stack).
- **baseten** —
  [baseten/GLM-5.2-Vision-NVFP4](https://huggingface.co/baseten/GLM-5.2-Vision-NVFP4):
  GLM-5.2 Vision weights — frozen MoonViT-3d vision tower + trained PatchMerger
  projector. Because 5.3 shares 5.2's text hidden size (6144), the 5.2 projector
  transfers with no retraining.
- **GLM-5.2 v19** — the entire stack (image, patches, recipes, vision method) comes
  from [glm-5.2/v19](../glm-5.2/v19/README.md); see its
  [credits section](../glm-5.2/v19/README.md#credits) for the full upstream list
  (local-inference-lab/vllm PR #175, CosmicRaisins, QuantTrio, Light Foundry Notes, …).
- **eugr/spark-vllm-docker** — build harness + cluster launcher.

## License

Apache-2.0 (this repo). Model weights are subject to their respective providers'
licenses — GLM-5.3 (Z.ai via Tech2wild quant), baseten vision weights.
