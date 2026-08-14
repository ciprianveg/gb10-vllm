# gb10-vllm

vLLM inference solutions for NVIDIA **DGX Spark / DGX Spark+ (GB10)** — Blackwell **SM121**.

This repo is a **model-agnostic platform**: a common GB10/SM121 vLLM stack (B12X_MLA attention,
speculative decoding, PP/TP/EP) plus per-model subtrees for building images, deploying recipes,
and applying runtime mods. New models slot in as their own subtree — see [Adding a model](#adding-a-model).

## Models index

| Model | Subtree | Highlights |
|-------|---------|------------|
| **KIMI-K3** (Full, sm121 image + RedHat DSpark) | [kimi-k3/v2](kimi-k3/v2/) | **Recommended** — TP16 (~500K ctx) / TP8+PP2 (~1M ctx), MXFP4, no runtime mods on TP16 |
| **KIMI-K3** (v1, B12X_MLA + DSpark) | [kimi-k3/v1](kimi-k3/v1/) | Historic artifact — kept for reference only |
| **GLM-5.2** (Int4-Int8) | [glm-5.2](glm-5.2/) | v16 / v18 / v18-vision / v18.1-vision, ~1,330 t/s prefill, vision | 

Each subtree is self-contained: a `README.md` model page, build script + Dockerfile, deploy
recipes (`*.yaml`), and runtime mods. Open the model page for its full guide.

## Architecture

### Hardware

- **GPU**: 1× NVIDIA GB10 (DGX Spark / DGX Spark+) per node — Blackwell **SM121**
- **Cluster**: 16 nodes (2×8), RoCE v2 (ConnectX-7, 100 Gbit) + NVLink-C2C, unified memory
- **Target arch**: sm_121 (`TORCH_CUDA_ARCH_LIST=12.1a`, `CUTE_DSL_ARCH=sm_121a`)

### Software stack

- **Base**: `nvidia/cuda:13.2.0-devel-ubuntu24.04`, Python 3.12, `uv`
- **PyTorch**: 2.11.0 + cu130
- **vLLM**: `local-inference-lab/vllm` fork (B12X_MLA, DSpark, PP, KDA)
- **b12x/sparkinfer**: CuTe DSL kernels for SM120/SM121 MLA
- **NCCL**: `zyang-dev/nccl` (dgxspark-3node-ring)
- **FlashInfer**: cu130/torch2.11 builds

### Key technologies

| Technology | What it does |
|------------|--------------|
| **B12X_MLA** | Dense & sparse MLA attention via CuTe DSL (sparkinfer/b12x) |
| **DSpark** | Speculative decoding (draft model + acceptance) |
| **MTP** | Multi-token prediction drafting (GLM-5.2) |
| **PP / TP / EP** | Pipeline / tensor / expert parallelism across nodes |
| **InstantTensor** | Fast weights loading from NFS |
| **Marlin MoE** | Quantized expert parallel backend |
| **Mods** | Runtime patches applied at container start (`mods/<name>/run.sh`) |

## Repository structure

```
gb10-vllm/
├── kimi-k3/v1/               KIMI-K3 v1 (historic artifact) — see kimi-k3/v1/README.md
├── kimi-k3/v2/               KIMI-K3 v2 (recommended) — sm121 image + RedHat DSpark
├── glm-5.2/                  GLM-5.2 v16/v18/v18-vision — see glm-5.2/README.md
├── README.md                 this file (platform overview + model index)
├── ATTRIBUTION.md            upstream credits
└── CREDITS.md                component licenses
```

Every model subtree follows one convention:

```
my-model/
├── README.md             model page (quick start, build, bench, known issues)
├── build.sh              image build (→ GHCR or local tag)
├── Dockerfile            multi-stage SM121 build
├── recipes/*.yaml        run-recipe deployables (recipe-schema YAML)
├── mods/<name>           runtime mods (run.sh applied at container start)
└── (optional) wheels/    NOT used — vLLM/FlashInfer compiled in-image
```

## Recipes & the run-recipe harness

Each model's deploy config is a set of YAML recipes run with the
eugr **spark-vllm-docker** harness (GLM-5.2 and KIMI-K3 both use `./run-recipe.sh <name>.yaml` from
eugr). There are no per-model launcher scripts. Recipes encode model path, image tag, mods,
env vars, and the full `vllm serve` command.

```bash
./run-recipe.sh kimi-k3/v1/recipes/kimi-k3-full-hh-b12x-pp2.yaml                    # KIMI-K3 TP8+PP2
./run-recipe.sh glm-5.2/v18-vision/recipes/glm52-int4int8-v18.1-vision.yaml         # GLM-5.2 v18.1 (fp8)
```

See each model page for the exact recipe list and pre-stop requirements.

## Adding a model

1. Create `my-model/` with `README.md`, `build.sh`, `Dockerfile`, `recipes/`, `mods/`
   (copy the structure from `kimi-k3/` or `glm-5.2/`).
2. Add the build step that installs the b12x/sparkinfer source for SM121 ML attention.
3. Write `.yaml` recipes the same schema as existing ones (`recipe_version`, `container`,
   `command`, `mods`, `env`).
4. Add one row to the [Models](#models) table.
5. Push the image to GHCR and add the tag to [Prebuilt images](#building--publishing).

## Building & publishing

```bash
./kimi-k3/v2/build.sh             # KIMI-K3 v2 local tag (vllm-node-kimi3-sm121)
./kimi-k3/v2/build.sh --push     # → ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v2-sm121
./kimi-k3/v1/build.sh --push      # → ghcr.io/ciprianveg/gb10-vllm/kimi-k3:latest (v1, historic)
./glm-5.2/v18-vision/build-nvfp4.sh --push   # → ghcr.io/ciprianveg/gb10-glm-5.2:v18.1-vision
```

Prebuilt images:

- `ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v2-sm121` (recommended)
- `ghcr.io/ciprianveg/gb10-vllm/kimi-k3:latest` (v1, historic)
- `ghcr.io/ciprianveg/gb10-glm-5.2:v18.1-vision` (and other v18 tags)

## Environment conventions

Recipes set the same per-image env knobs (see each model page). Common ones:

- `TORCH_CUDA_ARCH_LIST=12.1a`, `CUTE_DSL_ARCH=sm_121a`
- `VLLM_USE_V2_MODEL_RUNNER=1`
- `HF_HOME=/cache/huggingface`, offline mode
- NCCL: `NCCL_NET=IB`, `NCCL_IB_GID_INDEX=3`, `NCCL_BUFFSIZE=16777216`, `NCCL_MAX_NCHANNELS=8`

## Credits

Model-specific credits and full attribution: see [ATTRIBUTION.md](ATTRIBUTION.md) and
the model READMEs. Upstream:
vLLM ([vllm-project/vllm](https://github.com/vllm-project/vllm)),
local-inference-lab/vllm (B12X_MLA/DSpark/PP),
b12x/sparkinfer, NCCL, FlashInfer, Marlin, InstantTensor.

## License

Apache-2.0 (this repo). Model weights are subject to the respective providers' licenses —
see each model page and [CREDITS](CREDITS.md).