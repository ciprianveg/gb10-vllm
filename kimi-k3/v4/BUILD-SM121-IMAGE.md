# Building the KIMI-K3 v4 sm121 Image

> **Image:** `ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v4-prd` (linux/arm64, sm121, ~32 GB)

v4 = the published **v3 image** + an **upstream fork-tree advance** to the
r36-era pins + a **5-mod overlay**. No CUDA recompilation: the compiled
`.abi3.so` extensions and the vendored `vllm/third_party/triton_kernels` are
preserved from the v3 base (they are untracked files, untouched by the git
tree advance — same `.so`-preservation mechanism as the
[v3 overlay](../v3/BUILD-SM121-IMAGE.md)).

## Tree pins

The published v4-prd image carries exactly these commits (verified against its
in-image git HEAD and the `blackwell-llm-docker` source locks). Both are
public branch tips, fetchable with:

```bash
# vLLM tree (checked out at /opt/kimi-k3/vllm):
git fetch --depth=1 https://github.com/voipmonitor/vllm.git \
    refs/heads/integration/ii-kimi-k3-r35-dflash-active-staged-pack-20260822
# -> e08d796fbe7987fc41683963c22740eaf2515a01

# b12x tree (checked out at /opt/kimi-k3/b12x):
git fetch --depth=1 https://github.com/voipmonitor/b12x.git \
    refs/heads/build/kimi-k3-r35-dflash-active-staged-pack-20260822
# -> b8c7153c3e1ca13b1e0ef373d9a382a513ab892f
```

(The image's version string suffix `vllme755f87.b12x2d466e` is a build-time
stamp of the local merge stack, not a git object — the shipped trees are the
commits above.)

## Quick build

```bash
# v3 base + tree advance + mods overlay (no CUDA build; git fetch of the two
# pinned branch tips dominates the time)
./kimi-k3/v4/build.sh            # local tag: ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v4-prd
./kimi-k3/v4/build.sh --push     # build + push to GHCR
```

The script:

1. Pulls `ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v3-sm121` as the base.
2. Runs the Dockerfile in [`build-sm121/`](build-sm121/): `git fetch` +
   forced `git checkout` of both trees to the pins above (untracked build
   artifacts survive), compileall + `.so` sanity check, then executes every
   mod's `run.sh` in [`mods/`](mods/).
3. Verifies the baked markers.

## What the overlay adds

**5 baked mods** (patch-style, idempotent — see [`mods/`](mods/)); all five
were dry-applied to pristine checkouts of the pins above before publishing
this guide:

| Mod | Upstream | What it does |
|---|---|---|
| `fix-k3-r29-mamba-debug` | fork fix | r29 Mamba cadence-assertion fix (syncs block sizes) |
| `perf-pr52388-ii-backport` | [vllm#52388](https://github.com/vllm-project/vllm) | K3 Mamba metadata-preparation optimization (II-branch backport) |
| `pr51508-stale-zero-accept` | [vllm#51508](https://github.com/vllm-project/vllm) | skip GDN/KDA recurrent-state updates for stale (zero-accept) spec rows |
| `pr50169-drafter-kv-pool` | [vllm#50169](https://github.com/vllm-project/vllm) | dedicated KV groups for sliding-window drafters |
| `dflash2-support` (Fix A) | [vllm#52816](https://github.com/vllm-project/vllm) | DFlash2 draft-model support over the B12X_MLA backend |

The published `v4-prd` image additionally carries the Sep-1 optimizations
(b12x #271 fused DFlash K=3 verify at `nst=3`, myshytf/b12x #3 MoE plan
memoization, vllm #52502 GB10 fused-MoE FP8 tuning configs); they are not part
of the public pins above and are not reproduced by this rebuild path.

`drop-caches` stays **runtime-only** (recipe mod, not baked into the image).

## ABI-mismatch risk and fallback

The preserved `.abi3.so` extensions were compiled from the v3-base C++ tree.
The pinned trees are newer, so if a boot fails immediately with an
import-time `undefined symbol` / missing-symbol error referencing a
`*.abi3.so`, the base extensions are ABI-incompatible with the advanced tree.
In that case do the full native rebuild instead:
[`../v2/BUILD-SM121-IMAGE.md`](../v2/BUILD-SM121-IMAGE.md) (compiles the
extensions from the pinned source).

## Verify a pulled/built image

```bash
docker run --rm --entrypoint bash ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v4-prd -c '
    SRC=/opt/kimi-k3/vllm/vllm
    ls $SRC/_C_stable_libtorch.abi3.so >/dev/null && echo "SO OK"
    grep -q "fix-k3-r29-mamba-debug" $SRC/v1/worker/gpu/model_states/mamba_hybrid.py && echo "mamba cadence fix OK"
    grep -q "mamba_aligned_state_indices" $SRC/models/kimi_k3/nvidia/kda_metadata.py && echo "#52388 OK"
    grep -q "stale_spec_reqs" $SRC/v1/attention/backends/gdn_attn.py && echo "#51508 OK"
    grep -q "PR_50169" $SRC/v1/core/kv_cache_utils.py && echo "#50169 OK"
    ls $SRC/v1/worker/gpu/spec_decode/dflash2/speculator.py >/dev/null && echo "DFlash2 OK"
'
```

Expected: `SO OK`, `mamba cadence fix OK`, `#52388 OK`, `#51508 OK`,
`#50169 OK`, `DFlash2 OK`.
