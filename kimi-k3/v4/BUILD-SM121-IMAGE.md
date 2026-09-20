# Building the KIMI-K3 v4 sm121 Image

> **Image:** `ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v4-prd` (linux/arm64, sm121, ~32 GB)

v4 is a **thin mods overlay**: five patch-style mods are baked into the
`v4-sm121-r36` base line (the published `v4-prd` image is that base with the
mods already applied). No CUDA recompilation happens — the compiled
`.abi3.so` extensions and the b12x tree from the base are preserved untouched.

## What the overlay adds

**5 baked mods** (patch-style, idempotent — see [`mods/`](mods/)):

| Mod | Upstream | What it does |
|---|---|---|
| `fix-k3-r29-mamba-debug` | fork fix | r29 Mamba cadence-assertion fix (syncs block sizes) |
| `perf-pr52388-ii-backport` | [vllm#52388](https://github.com/vllm-project/vllm) | K3 Mamba metadata-preparation optimization (II-branch backport) |
| `pr51508-stale-zero-accept` | [vllm#51508](https://github.com/vllm-project/vllm) | skip GDN/KDA recurrent-state updates for stale (zero-accept) spec rows |
| `pr50169-drafter-kv-pool` | [vllm#50169](https://github.com/vllm-project/vllm) | dedicated KV groups for sliding-window drafters |
| `dflash2-support` (Fix A) | [vllm#52816](https://github.com/vllm-project/vllm) | DFlash2 draft-model support over the B12X_MLA backend |

Sep-1 optimizations already baked into the published `v4-prd` image:

- **b12x #271** — fused DFlash K=3 verification; activates only at `nst=3`
  (3 speculative + 1 bonus token).
- **myshytf/b12x #3** — MoE plan memoization + direct dynamic launch;
  `B12X_MOE_PLAN_CACHE_SIZE=0` disables.
- **vllm #52502** — GB10 fused-MoE FP8 tuning configs (E=256, E=512, N=512).

`drop-caches` stays **runtime-only** (recipe mod, not baked into the image).

## Quick build

```bash
# Thin mods overlay on the v4 base — takes seconds (no CUDA build)
./kimi-k3/v4/build.sh            # local tag: ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v4-prd
./kimi-k3/v4/build.sh --push     # build + push to GHCR
```

The script:

1. Pulls `ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v4-prd` as the base.
2. Runs the Dockerfile in [`build-sm121/`](build-sm121/), which executes every
   mod's `run.sh` in [`mods/`](mods/) (each mod skips cleanly when its change
   is already in the tree).
3. Verifies the baked markers.

## Verify a pulled/built image

```bash
docker run --rm --entrypoint bash ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v4-prd -c '
    SRC=/opt/kimi-k3/vllm/vllm
    grep -q "fix-k3-r29-mamba-debug" $SRC/v1/worker/gpu/model_states/mamba_hybrid.py && echo "mamba cadence fix OK"
    grep -q "mamba_aligned_state_indices" $SRC/models/kimi_k3/nvidia/kda_metadata.py && echo "#52388 OK"
    grep -q "stale_spec_reqs" $SRC/v1/attention/backends/gdn_attn.py && echo "#51508 OK"
    grep -q "PR_50169" $SRC/v1/core/kv_cache_utils.py && echo "#50169 OK"
    ls $SRC/v1/worker/gpu/spec_decode/dflash2/speculator.py >/dev/null && echo "DFlash2 OK"
'
```

Expected: `mamba cadence fix OK`, `#52388 OK`, `#51508 OK`, `#50169 OK`, `DFlash2 OK`.
