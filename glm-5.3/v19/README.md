# v19 — GLM-5.3 Int4-Int8Mix text-only on the GLM-5.2 v19 image

GLM-5.3 ([Tech2wild/GLM-5.3-Int4-Int8Mix](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix))
served text-only on 8× GB10, TP=8 + PP=1, using the
[`ghcr.io/ciprianveg/gb10-glm-5.2:v19-vision`](https://github.com/ciprianveg/gb10-vllm/pkgs/container/gb10-glm-5.2)
image **unchanged**.

GLM-5.3 shares GLM-5.2's architecture (`glm_moe_dsa`, hidden 6144, same vocab, same MTP
layout), so the full v19 stack — DCP1 query split, deterministic MoE align,
`B12X_MLA_SPARSE`, adaptive MTP 2/4/5, native sm_121 cubins — carries over verbatim.
The v19 long-context prefill gains measured on 5.2
([+7.7% @ 100K, +10%+ @ 200K](../../glm-5.2/v19/README.md#prefill-delta--v181-vs-v19))
translate directly to 5.3.

For the stack details, patches, and upstream credits, see
[../../glm-5.2/v19/README.md](../../glm-5.2/v19/README.md) — nothing was modified for 5.3.

## Recipe

[`recipes/glm53-int4int8-v19.yaml`](recipes/glm53-int4int8-v19.yaml) — the GLM-5.2 v19
recipe with the model swapped to the Tech2wild checkpoint, minus the multimodal flags.

```bash
docker pull ghcr.io/ciprianveg/gb10-glm-5.2:v19-vision
./run-recipe.sh glm-5.3/v19/recipes/glm53-int4int8-v19.yaml
```

Key serve args (identical to GLM-5.2 v19):

- TP=8, port `5001`, `fp8_ds_mla` KV cache, `--max-model-len 410000`
- `B12X_MLA_SPARSE` attention backend (text + draft)
- Adaptive MTP 2/4/5 — `VLLM_ADAPTIVE_SPEC_DEPTHS=2,4,5`, `num_speculative_tokens=5`,
  `draft_sample_method: probabilistic`
- `DETERMINISTIC_MOE_ALIGN=1` — deterministic MoE token placement (temp-0 reproducibility)
- `VLLM_DCP_QUERY_SPLIT=1` — DCP1 query split (indexer work shared across TP ranks)
- `--enable-decode-aware-prefill` with `--decode-prefill-token-budget 1024`

For vision, see [../v19-vision/README.md](../v19-vision/README.md).
