# KIMI-K3 B12X_MLA + DSpark on 16× GB10 DGX Spark

Full [Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) (BF16 attention + MXFP4 experts)
served with **B12X_MLA** attention (CuTe DSL `dense_mla` via sparkinfer) + **DSpark**
speculative decoding (draft: [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark))
on 16-node DGX Spark GB10 (sm121) clusters, TP16 or the recommended TP8+PP2.

Image: `vllm-node-kimi3-hh` (linux/arm64, sm121).

## Recommended path: TP8+PP2

**Use the PP2 recipe for real workloads.** TP8+PP2 fits **~800K context** and is the layout to
follow with this model. The TP16 path is **too tight on 16× GB10 for this model to be usable
for coding** — it fits only ~200K because KIMI-K3's 640K+ effective window (~378K built-in +
sliding window) exceeds what 16 single-rank KV caches can hold. Start your deployment from
[`kimi-k3-full-hh-b12x-pp2.yaml`](recipes/kimi-k3-full-hh-b12x-pp2.yaml).

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
| [`kimi-k3-full-hh-b12x-pp2.yaml`](recipes/kimi-k3-full-hh-b12x-pp2.yaml) | 8 | 2 | no | **~800K** | **Recommended path** — max context, usable for coding, GMU 0.82 |
| [`kimi-k3-full-hh-b12x-tp16.yaml`](recipes/kimi-k3-full-hh-b12x-tp16.yaml) | 16 | 1 | yes | ~200K | Too tight for coding workloads on 16× GB10; keep for short-context benchmarks |

> **InstantTensor loader**: works on sm121 and cuts model load from ~15 min to ~4.2 min
> (`--load-format instanttensor`). It trades memory for speed: it headrooms **GMU down**
> (and with it max context) to make room for the faster load path, so it's the right choice
> for **quick iteration on optimization tasks**, while the default loader is used for
> production context ceilings.

## Building the image

### Prebuilt image

The recipes reference `container: vllm-node-kimi3-hh`. A prebuilt image is published at:

```bash
docker pull ghcr.io/ciprianveg/gb10-vllm/kimi-k3:latest
```

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
Other eugr-infrastructure users can reproduce it exactly with:

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

Prerequisites on the build host:
- `mods/b12x-nvfp4/patches/sparkinfer-src/` present (sparkinfer 1.0.1 source; cloned from
  `https://github.com/local-inference-lab/b12x.git` branch `codex/hh-kimi-k3-dcp16-20260804`).
  In this repo it is tracked under `mods/b12x-nvfp4/patches/sparkinfer-src/` so a fresh clone
  resolves the bind mount.
- The fork ref must exist upstream (branch `codex/hh-kimi-k3-dspark-dcp16-20260804` on
  `local-inference-lab/vllm`).

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
weights live under `/root/models/models115/Kimi-K3` (full) and the DSpark draft under
`/root/models/models16/K3-DSpark-Inferact`; `shared_weights_nfs: true` expects an NFS-mounted
model tree.

```bash
# From the eugr harness — PP2 is the primary/recommended recipe
./run-recipe.sh kimi-k3-full-hh-b12x-pp2.yaml --setup    # TP8+PP2, ~800K ctx (stop GLM-5.2 on .11-.18 first)
./run-recipe.sh kimi-k3-full-hh-b12x-tp16.yaml --setup   # TP16 ~200K — benchmarks only

# Or the bundled manage scripts (start/stop/status/kill across all 16 nodes, port 5002)
./kimi-k3/v1/scripts/manage-kimi-k3-full-pp2.sh  start
./kimi-k3/v1/scripts/manage-kimi-k3-full-tp16.sh start
```

Key serve args (both recipes):

- `--attention-backend B12X_MLA`, `--moe-backend marlin`, `--enable-expert-parallel` (TP16)
- DSpark spec decode: `{"method":"dspark","num_speculative_tokens":7,"attention_backend":"B12X_MLA"}`
- `--reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice`
- `--port 5002` (this image's cluster uses 5002; GLM-5.2 on `.11-.18` uses 5001)

## Mods

The 10 runtime mods applied by the recipes (in `mods/`, each a directory with a `run.sh`
applied by the eugr harness) fix B12X_MLA / DSpark / FP8-MLA issues on sm121. The
`b12x-nvfp4` mod additionally carries the sparkinfer source used at build time.

## TODO / Known issues

- **DSpark acceptance drop on B12X_MLA**: after switching the draft target's attention
  backend from `TRITON_MLA` to `B12X_MLA`, DSpark acceptance dropped and the speculative
  token count had to be **reduced from the recommended 7 to 5** — both recipes now use
  `num_speculative_tokens: 5`. Investigate why B12X_MLA loses acceptance, and re-tune the
  budget (and the sampling/scheduling knobs) for B12X.
- **CUDAGraph impact with B12X_MLA**: CUDAGraph now works with the `B12X_MLA` path (it
  did not with the native-FlashInfer kernel for this model). Quantify the graph-replay vs.
  eager speedup for B12X, and check for any correctness/peak-memory regressions when
  `--enforce-eager` is dropped.

## Credits

- **local-inference-lab/vllm** (branch `codex/hh-kimi-k3-dspark-dcp16-20260804`): vLLM fork with
  B12X_MLA, DSpark, and the Phase 2-4 optimizations.
- **local-inference-lab/b12x** (branch `codex/hh-kimi-k3-dcp16-20260804`): sparkinfer 1.0.1
  source.
- **eugr/spark-vllm-docker**: build harness + cluster launcher + mod-apply infrastructure.
