# Building the vLLM sm121 Image for GB10 DGX Spark

> Reproducible build guide for the `vllm-node-kimi3-sm121` image — the vLLM runtime
> used by the Kimi-K3 v2 recipes (`kimi-k3-tp16.yaml`, `kimi-k3-tp8pp2.yaml`) on
> NVIDIA DGX Spark (GB10, sm121), adapted from the
> [rtx6kpro](https://github.com/local-inference-lab/rtx6kpro) qualified runtime.
>
> **Image:** `vllm-node-kimi3-sm121` (linux/arm64, sm121, 31.2 GB)
> **Published:** `ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v2-sm121`

## What this image provides

- **vLLM** from [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm)
  branch `integration/kimi-k3-ii-cu133-torch213-20260811` @881ac39
- **B12X** PRs #124, #138, #139 (tree 2e6092a) — dense MLA decode + GQA paged + DFlash wiring
- **PyTorch 2.13.0** (built from source, CUDA 13.3)
- **NCCL 2.31.2** (from rtx6kpro, patched)
- **FlashInfer 0.6.15.post1**
- **InstantTensor 0.1.9** (fast weight loading)
- **Triton kernels v3.5.1**
- All compiled for **sm121** (CUTE_DSL_ARCH=sm_121a, TORCH_CUDA_ARCH_LIST=12.1a)

The v1 Kimi-K3 runtime mods (B12X MLA/smem fixes, flash-KDA prefill, fused-KV,
mamba `idx_mapping` int64 cast, mamba-MRv2 race, spec-decode block-table, EP
intermediate padding, DSpark warmup/perf-unified) are **baked into the fork's
integration branch** — the v2 recipes need **no runtime mods** for TP16. The
`fix-pp2-sm121` mod is only required for the TP8+PP2 recipe (see
[`mods/fix-pp2-sm121`](mods/fix-pp2-sm121/)).

## Prerequisites

- An arm64 (aarch64) Linux host with Docker and BuildKit
- ~60 GB free disk space (base image + runtime build + temp)
- NVIDIA driver (for post-build GPU verification, not needed during build)
- `git`, `python3`, `jq`, `sha256sum`
- If building with other GPU workloads running: limit parallelism to avoid OOM

## Quick start (two-step build)

### Step 1: Clone the build repo

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 697f50ff644f2c418645c64a50828dccce597d38
```

### Step 2: Adapt Dockerfiles for sm121

The rtx6kpro Dockerfiles target sm120 (RTX PRO 6000). For GB10 (sm121), apply
these changes:

**`Dockerfile.kimi-k3-cu133-torch213-base`:**
```
sm_120 → sm_121          (NVCC_GENCODE)
compute_120 → compute_121
12.0a → 12.1a            (TORCH_CUDA_ARCH_LIST, 3 occurrences)
```

**`Dockerfile.kimi-k3-infernal-invocation-cu133-torch213`:**
```
12.0a → 12.1a            (TORCH_CUDA_ARCH_LIST)
120a → 121a              (CMAKE_CUDA_ARCHITECTURES)
fmha-sm120 → fmha-sm121  (MINFER_FMHA_CACHE_DIR)
--parallel 48 → --parallel 12  (or your preferred job count)
```

One-liner:
```bash
sed -i 's/compute_120,code=sm_120/compute_121,code=sm_121/g;
        s/compute_120,code=compute_120/compute_121,code=compute_121/g;
        s/TORCH_CUDA_ARCH_LIST=12\.0a/TORCH_CUDA_ARCH_LIST=12.1a/g;
        s/CMAKE_CUDA_ARCHITECTURES=120a/CMAKE_CUDA_ARCHITECTURES=121a/g;
        s/fmha-sm120/fmha-sm121/g;
        s/--parallel 48/--parallel 12/' \
  Dockerfile.kimi-k3-cu133-torch213-base \
  Dockerfile.kimi-k3-infernal-invocation-cu133-torch213
```

Already-adapted copies of both Dockerfiles and the build scripts are committed in
[`build-sm121/`](build-sm121/). You can copy them over the cloned repo instead of
editing by hand.

### Step 3: Build the base image

```bash
IMAGE=vllm-node-kimi3-sm121-base \
RELEASE_DATE=$(date -u +%Y%m%d) \
REVISION=r1 \
ALLOW_DIRTY_BUILD=1 \
  bash build-kimi-k3-cu133-torch213-base.sh
```

This builds PyTorch 2.13.0, NCCL 2.31.2, Torchvision 0.28.0, and XGrammar 0.2.5
from source. **Build time: ~40-60 min** with 12 parallel jobs.

**Limiting parallelism (avoid OOM):**
```bash
NCCL_MAX_JOBS=12 TORCH_MAX_JOBS=12 TORCHVISION_MAX_JOBS=12 XGRAMMAR_MAX_JOBS=12 \
IMAGE=vllm-node-kimi3-sm121-base \
ALLOW_DIRTY_BUILD=1 \
  bash build-kimi-k3-cu133-torch213-base.sh
```

The build scripts here already pass `--build-arg "*_MAX_JOBS=12"` for these four
components.

**Verification (built into the script):**
```
Kimi-K3 CUDA base contract: PASS torch=2.13.0 torchvision=0.28.0 cuda=13.3 nccl=23102 xgrammar=0.2.5
```

### Step 4: Build the runtime image

```bash
BASE_IMAGE=vllm-node-kimi3-sm121-base \
IMAGE=vllm-node-kimi3-sm121 \
RELEASE_DATE=$(date -u +%Y%m%d) \
REVISION=r1 \
ALLOW_DIRTY_BUILD=1 \
  bash build-kimi-k3-infernal-invocation-cu133-torch213.sh
```

This clones vLLM@881ac39 + B12X (PRs #124/#138/#139) + InstantTensor + FlashInfer,
applies integration patches (SHA-256 verified), builds native extensions, and
packages serve launchers. **Build time: ~30-40 min** with 12 parallel jobs.

**Verification (built into the script):**
```
Kimi-K3 infernal-invocation runtime contract: PASS
torch=2.13.0 cuda=13.3 nccl=23102 instanttensor=0.1.9
flashinfer=0.6.15.post1 torchvision=0.28.0 cutlass-dsl=4.6.0 xgrammar=0.2.5
```

### Step 5: Post-build verification

```bash
docker run --rm --gpus all vllm-node-kimi3-sm121 bash -c "
  python3 -c 'import torch; print(f\"torch {torch.__version__}, cuda {torch.version.cuda}\")'
  python3 -c 'import vllm; print(f\"vLLM {vllm.__version__}\")'
  python3 -c 'import instanttensor; print(\"instanttensor OK\")'
  python3 -c 'import flashinfer; print(f\"flashinfer {flashinfer.__version__}\")'
"
```

## Distributing to cluster nodes

### Using eugr/spark-vllm-docker (recommended for GB10 Spark clusters)

If you use the [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)
build harness (common among GB10 DGX Spark users):

```bash
cd ~/eugr/spark-vllm-docker

# Copy image to all nodes in .env CLUSTER_NODES
./build-and-copy.sh --no-build -c -t vllm-node-kimi3-sm121
```

This uses `docker save | ssh docker load` to copy the image to all nodes
listed in `COPY_HOSTS` from `.env`. It skips nodes that already have the
same image ID.

### Manual distribution

```bash
# Save image to tar
docker save vllm-node-kimi3-sm121 | gzip > vllm-node-kimi3-sm121.tar.gz

# Copy to each node
for node in 192.168.177.111 192.168.177.112 ...; do
  ssh $node "docker load" < vllm-node-kimi3-sm121.tar.gz
done
```

## Serving Kimi-K3

The built image (`vllm-node-kimi3-sm121`) is used by the Kimi-K3 v2 recipes
in [`recipes/`](recipes/) (`kimi-k3-tp16.yaml`, `kimi-k3-tp8pp2.yaml`).
Serving configuration, benchmarks, and tuning knobs are documented in
[`README.md`](README.md) — this guide covers the image build only.

## Credits

- [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) — Kimi-K3 model weights (BF16 attention + MXFP4 experts)
- [local-inference-lab/rtx6kpro](https://github.com/local-inference-lab/rtx6kpro) — qualified runtime reference
- [local-inference-lab/blackwell-llm-docker](https://github.com/local-inference-lab/blackwell-llm-docker) — build scripts
- [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) — vLLM fork with B12X + DSpark + DFlash
- [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) — CuTe DSL kernels for SM120/SM121
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — GB10 cluster build harness + mod system
- [voipmonitor/InstantTensor](https://github.com/voipmonitor/InstantTensor) — fast weight loading
- [RedHatAI/Kimi-K3-speculator.dspark](https://huggingface.co/RedHatAI/Kimi-K3-speculator.dspark) — RedHat DSpark draft
- [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark) — Inferact DSpark draft
- [modal-labs/Kimi-K3-DFlash](https://huggingface.co/modal-labs/Kimi-K3-DFlash) — DFlash draft
