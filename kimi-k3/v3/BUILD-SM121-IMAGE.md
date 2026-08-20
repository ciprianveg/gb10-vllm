# Building the KIMI-K3 v3 sm121 Image

> **Image:** `ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v3-sm121` (linux/arm64, sm121, ~31.5 GB)

v3 is a **thin overlay** on the published [v2 image](../v2/BUILD-SM121-IMAGE.md):
the vLLM Python source inside the v2 image is advanced from `881ac39` to
`0232bce6` and the v2 runtime mods are baked in (they become no-ops when the
tree already contains them). No CUDA recompilation happens — the compiled
`.abi3.so` extensions and the b12x tree from v2 are preserved untouched.

## What the overlay adds

- **vLLM tree `0232bce6`** (from
  [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm)
  `dev/infernal-invocation`): MoE projection-transport fusion (#385/#386),
  dense-MLA-over-DCP (#387), DSpark runtime support (#388/#389/#390),
  prefix-cache / renderer / streaming fixes (#413/#414/#415/#419/#422/#433),
  and the **>270K-context loop fix** (#418).
- **11 baked mods** — see the [README mods table](README.md#mods-baked-into-the-image).
  Several (#395, #401, #418) are already in the `0232bce6` tree; their mods
  detect that and skip cleanly.

The vLLM tree is intentionally **not** taken past `0232bce6`: the next relevant
merge (#427) moves CuTe-DSL JIT into the startup profiling pass and causes a
multi-minute stall on GB10/sm121. Deferred (lazy) CuTe-DSL compilation is a
property of the v2 b12x tree (`2e6092a`) that v3 preserves.

## Quick build

```bash
# Thin overlay on the published v2 image — takes seconds (no CUDA build)
./kimi-k3/v3/build.sh            # local tag: ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v3-sm121
./kimi-k3/v3/build.sh --push     # build + push to GHCR
```

The script:

1. Pulls `ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v2-sm121` as the base.
2. Generates the vLLM overlay patch by diffing `881ac39..0232bce6` from
   `local-inference-lab/vllm` (needs `git` on the host).
3. Runs the Dockerfile in [`build-sm121/`](build-sm121/), which applies the
   patch and executes every mod's `run.sh` in [`mods/`](mods/).
4. Verifies: `.so` files preserved, MoE-fusion file present, #418 present,
   #427 absent.

If the v2 image is not published yet, build it first via
[`../v2/build.sh`](../v2/build.sh) and tag it
`ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v2-sm121`.

## Verify a pulled/built image

```bash
docker run --rm --entrypoint bash ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v3-sm121 -c '
    ls /opt/kimi-k3/vllm/vllm/_C_stable_libtorch.abi3.so && echo "SO OK"
    ls /opt/kimi-k3/vllm/vllm/models/kimi_k3/nvidia/tp_projection.py && echo "MoE fusion OK"
    grep -c "Recurrent state has one token-position shard" /opt/kimi-k3/vllm/vllm/v1/kv_cache_interface.py
    grep -c "VLLM_SHM_BROADCAST_BUSY_LOOP_S" /opt/kimi-k3/vllm/vllm/envs.py
'
```

Expected: `SO OK`, `MoE fusion OK`, `1`, `1`.
