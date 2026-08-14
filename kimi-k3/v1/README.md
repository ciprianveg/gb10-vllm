# KIMI-K3 B12X_MLA + DSpark on 16× GB10 DGX Spark

> **HISTORIC ARTIFACT** — superseded by [`kimi-k3/v2`](../v2/README.md)
> (native-sm121 image + RedHat DSpark). v2 is faster, fits more context on the
> recommended TP16 path, and needs no runtime mods. This page is kept for
> reference only.

Full KIMI-K3 (BF16 attention + MXFP4 experts) served with **B12X_MLA** attention (CuTe DSL
`dense_mla` via sparkinfer) + **DSpark** speculative decoding on 16-node DGX Spark GB10
(sm121) clusters, single-node TP16 or TP8+PP2.

Image: `vllm-node-kimi3-hh` (linux/arm64, sm121).

## What this is

`v1` builds a vLLM image from the [`local-inference-lab/vllm`](https://github.com/local-inference-lab/vllm)
fork at branch `codex/hh-kimi-k3-dspark-dcp16-20260804`, adds sparkinfer 1.0.1 (b12x successor,
provides the `B12X_MLA` dense-MLA attention backend), and serves KIMI-K3 with:

- **B12X_MLA** attention — dense MLA kernels JIT-compiled by CuTe DSL at runtime
  (`CUTE_DSL_ARCH=sm_121a`), so no prebuilt cubins are baked in.
- **DSpark** speculative decoding (draft model + `num_speculative_tokens=7`).
- All Phase 2-4 optimizations: `flash-kda-prefill`, DSpark perf-unified, EP
  intermediate-padding fix, fused-KV, PP2 aux-forward, spec-decode block table.

| Recipe | TP | PP | EP | Max ctx | Notes |
|---|---|---|---|---|---|
| [`kimi-k3-full-hh-b12x-tp16.yaml`](recipes/kimi-k3-full-hh-b12x-tp16.yaml) | 16 | 1 | yes | ~200K | full-speed path, GMU 0.86 |
| [`kimi-k3-full-hh-b12x-pp2.yaml`](recipes/kimi-k3-full-hh-b12x-pp2.yaml) | 8 | 2 | no | ~800K+ | max-context path, GMU 0.82 |

## Building the image

### Self-contained (this repo)

```bash
./kimi-k3/v1/build.sh                    # build local tag vllm-node-kimi3-hh
```

The [`build.sh`](build.sh) and [`Dockerfile`](Dockerfile) are self-contained: they clone the
fork, build vLLM + FlashInfer in-image, and bind-mount
`mods/b12x-nvfp4/patches/sparkinfer-src` to install sparkinfer. `wheels/` is optional — when
empty, vLLM/FlashInfer are compiled from source.

### Via the eugr build harness (the original build path)

The image used by the recipes was built with `eugr/spark-vllm-docker`'s `build-and-copy.sh`.
Other eugr-vllm users can reproduce it exactly with:

```bash
cd ~/eugr/spark-vllm-docker
./build-and-copy.sh \
  --vllm-ref codex/hh-kimi-k3-dspark-dcp16-20260804 \
  --vllm-repo https://github.com/local-inference-lab/vllm.git \
  --install-b12x \
  --rebuild-vllm \
  --gpu-arch 12.1a \
  -t vllm-node-kimi3-hh \
  -j 12 \
  -c
```


Verified build parameters from the built image (`/workspace/build-metadata.yaml`):

| Param | Value |
|---|---|
| `vllm_ref` | `codex/hh-kimi-k3-dspark-dcp16-20260804` |
| base image | `nvidia/cuda:13.0.2-devel-ubuntu24.04` |
| gpu_arch | `12.1a` (sm121) |
| build_jobs | 12 |
| sparkinfer | 1.0.1 (from source, `--no-deps`) |

## Launch

Recipes are `cluster_only` (16 nodes) and reference `container: vllm-node-kimi3-hh`. Model
in my case weights live under `/root/models/models115/Kimi-K3` (full) and the DSpark draft under
`/root/models/models16/K3-DSpark-Inferact`; `shared_weights_nfs: true` expects an NFS-mounted
model tree.

```bash
# From the eugr harness
./run-recipe.sh kimi-k3-full-hh-b12x-tp16.yaml --setup   # TP16 path (stop GLM-5.2 on .11-.18 first)
./run-recipe.sh kimi-k3-full-hh-b12x-pp2.yaml --setup    # TP8+PP2 path (800K+ ctx)

# Or the bundled manage scripts (start/stop/status/kill across all 16 nodes, port 5002)
./kimi-k3/v1/scripts/manage-kimi-k3-full-tp16.sh start
./kimi-k3/v1/scripts/manage-kimi-k3-full-pp2.sh  start
```


## Mods

The 10 runtime mods applied by the recipes (in `mods/`, each a directory with a `run.sh`
applied by the eugr harness) fix B12X_MLA / DSpark / FP8-MLA issues on sm121. The
`b12x-nvfp4` mod additionally carries the sparkinfer source used at build time.

## Credits

- **local-inference-lab/vllm** (branch `codex/hh-kimi-k3-dspark-dcp16-20260804`): vLLM fork with
  B12X_MLA, DSpark, and the Phase 2-4 optimizations.
- **local-inference-lab/b12x** (branch `codex/hh-kimi-k3-dcp16-20260804`): sparkinfer 1.0.1
  source.
- **eugr/spark-vllm-docker**: build harness + cluster launcher + mod-apply infrastructure.
