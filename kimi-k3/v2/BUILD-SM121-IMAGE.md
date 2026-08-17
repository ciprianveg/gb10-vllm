# Building the vLLM sm121 Image for GB10 DGX Spark

> Reproducible build + update guide for the `vllm-node-kimi3-sm121` image — the vLLM
> runtime used by the Kimi-K3 recipes on NVIDIA DGX Spark (GB10, sm121), adapted from
> the [rtx6kpro](https://github.com/local-inference-lab/rtx6kpro) qualified runtime.
>
> This document covers two things:
> 1. **Building the current image** (the pinned, known-good recipe).
> 2. **Updating to a newer upstream release** — what to do when rtx6kpro publishes a new
>    sm120 image / vLLM fork drop and we need to rebuild it for sm121 and push it to the
>    cluster.
>
> **Image:** `vllm-node-kimi3-sm121` (linux/arm64, sm121, ~31 GB)
> **Published (rollback):** `ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v2-sm121`

---

## 0. Rollback (read this first)

The current known-good image is published. If an update goes wrong, any node can be
reverted in two commands — no local backup of the old image is required:

```bash
docker pull ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v2-sm121
docker tag  ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v2-sm121 vllm-node-kimi3-sm121
```

Always keep the *previous* published tag working before you overwrite `:v2-sm121`. Prefer
publishing a new tag (e.g. `:v3-sm121`) and only re-pointing recipes once it is verified,
so rollback is a tag change, not a re-pull.

---

## 1. What the current image carries (r1)

| Component | Pin | Notes |
|---|---|---|
| vLLM fork | `local-inference-lab/vllm` branch `integration/kimi-k3-ii-cu133-torch213-20260811` @ `881ac39` (tree `de04f08`) | integration branch, no separate PRs |
| B12X | `local-inference-lab/b12x` master @ `184d7d52` (tree `2e6092a`) | PRs #124, #138, #139 |
| PyTorch | 2.13.0 (from source) | commit `cf30153c` |
| CUDA | 13.3 | |
| NCCL | 2.31.2 (rtx6kpro patched) | commit `fb6f4099` |
| Torchvision | 0.28.0 | commit `8fb87713` |
| XGrammar | 0.2.5 | commit `2ea71da4` |
| FlashInfer | 0.6.15.post1 | |
| InstantTensor | 0.1.9 | commit `49b4010` |
| CUTLASS DSL | 4.6.0 | |
| Triton kernels | 3.5.1 | commit `0add6826` |
| Arch | sm121 (`CUTE_DSL_ARCH=sm_121a`, `TORCH_CUDA_ARCH_LIST=12.1a`) | |

**Build recipe (blackwell-llm-docker):** commit `697f50ff644f2c418645c64a50828dccce597d38`
**Build scripts:** `build-kimi-k3-cu133-torch213-base.sh` + `build-kimi-k3-infernal-invocation-cu133-torch213.sh`
**Composition roots:** `patches/releases/kimi-k3-infernal-invocation-runtime-r1` (vLLM) + `patches/releases/kimi-k3-hh-runtime-r1` (B12X)

The v1 runtime mods (B12X MLA/smem fixes, flash-KDA prefill, fused-KV, mamba int64 cast,
mamba-MRv2 race, spec-decode block-table, EP padding, DSpark warmup) are **baked into the
fork's integration branch** — the TP16 recipe needs no runtime mods. Runtime yaml mods
(`mods/…` in eugr/spark-vllm-docker) are applied on top at launch for PRs the fork does
not yet carry (see §7).

---

## 2. How the upstream build system works (read before updating)

rtx6kpro images are built from a **separate repo**,
[blackwell-llm-docker](https://github.com/local-inference-lab/blackwell-llm-docker), which
is pinned to a commit per release. Understanding three concepts makes every future update
mechanical:

### 2.1 Two-stage build

- **Base image** (`Dockerfile.kimi-k3-cu133-torch213-base`): builds PyTorch, NCCL,
  Torchvision, XGrammar (and, depending on the release, FlashInfer / CUTLASS /
  InstantTensor) from source. Slow (~40–60 min). Changes only when the upstream *base
  contract* (torch/cuda/nccl/flashinfer/cutlass versions) changes.
- **Runtime image** (`Dockerfile.kimi-k3-<release>-runtime`): clones the vLLM fork + B12X
  at the pinned commits, applies the SHA-256-verified integration patch, builds native
  extensions, packages the serve launchers. Faster (~30–40 min). This is where vLLM/B12X
  updates land.

### 2.2 Composition locks (the heart of a release)

Each release is a directory under `patches/releases/<release-name>/` containing, per
component (`vllm/`, `b12x/`):

- `integration.lock.json` — pins `.base.repository`, `.base.ref`, `.base.commit`, the list
  of `.pull_requests[]` (number + head), and the `.result.tree` (the Git tree hash after
  the PRs are applied) + `.result.patch_sha256`.
- `integration.patch` — the combined diff; its SHA-256 must match the lock.

The build script's `read_lock()` verifies the patch hash, then exports
`VLLM_REPO/REF/COMMIT/PRS/INTEGRATION_TREE` (and the B12X equivalents) as `--build-arg`s.
The image is labeled with the tree hashes, and the post-build verifier checks them. **The
tree hash is the source of truth for "what's in the image."**

### 2.3 Where to find the newest release

- The rtx6kpro [kimi-k3 README](https://github.com/local-inference-lab/rtx6kpro/tree/master/models/kimi-k3)
  "Qualified artifact" table lists the newest image tag, its vLLM/B12X composed trees, and
  the blackwell-llm-docker commit that built it.
- The set of release dirs is `patches/releases/` in blackwell-llm-docker — a new upstream
  drop adds a new `<release>` dir (and usually a new `build-kimi-k3-<release>-runtime.sh`).

---

## 3. Building the current image (pinned recipe)

### Prerequisites

- arm64 (aarch64) Linux host with Docker + BuildKit
- ~60 GB free disk
- `git`, `python3`, `jq`, `sha256sum`
- NVIDIA driver only for post-build GPU verification (not needed to build)
- If building alongside GPU workloads, cap parallelism to avoid host OOM

### Step 1 — clone the pinned build recipe

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 697f50ff644f2c418645c64a50828dccce597d38
```

### Step 2 — adapt the Dockerfiles for sm121

The rtx6kpro Dockerfiles target sm120 (RTX PRO 6000). Apply the sm121 mapping (see §6 for
the full reference). Already-adapted copies live in [`build-sm121/`](build-sm121/) — copy
them over the clone instead of editing by hand.

### Step 3 — build the base image (~40–60 min)

```bash
IMAGE=vllm-node-kimi3-sm121-base \
RELEASE_DATE=$(date -u +%Y%m%d) REVISION=r1 ALLOW_DIRTY_BUILD=1 \
  bash build-kimi-k3-cu133-torch213-base.sh
```

Cap parallelism if needed:
```bash
NCCL_MAX_JOBS=12 TORCH_MAX_JOBS=12 TORCHVISION_MAX_JOBS=12 XGRAMMAR_MAX_JOBS=12 \
IMAGE=vllm-node-kimi3-sm121-base ALLOW_DIRTY_BUILD=1 \
  bash build-kimi-k3-cu133-torch213-base.sh
```

Expected: `Kimi-K3 CUDA base contract: PASS torch=2.13.0 torchvision=0.28.0 cuda=13.3 nccl=23102 xgrammar=0.2.5`

### Step 4 — build the runtime image (~30–40 min)

```bash
BASE_IMAGE=vllm-node-kimi3-sm121-base \
IMAGE=vllm-node-kimi3-sm121 \
RELEASE_DATE=$(date -u +%Y%m%d) REVISION=r1 ALLOW_DIRTY_BUILD=1 \
  bash build-kimi-k3-infernal-invocation-cu133-torch213.sh
```

Expected: `Kimi-K3 infernal-invocation runtime contract: PASS torch=2.13.0 cuda=13.3 nccl=23102 instanttensor=0.1.9 flashinfer=0.6.15.post1 torchvision=0.28.0 cutlass-dsl=4.6.0 xgrammar=0.2.5`

### Step 5 — post-build smoke check

```bash
docker run --rm --gpus all vllm-node-kimi3-sm121 bash -c "
  python3 -c 'import torch; print(torch.__version__, torch.version.cuda)'
  python3 -c 'import vllm; print(vllm.__version__)'
  python3 -c 'import instanttensor, flashinfer; print(\"loaders OK\")'
"
```

---

## 4. Updating to a newer upstream release (the playbook)

Use this when rtx6kpro publishes a new image / fork drop. Worked example is r1 → r2 in §5.

### 4.1 Detect and identify the new release

1. Read the rtx6kpro kimi-k3 README "Qualified artifact" table. Note the new:
   - **blackwell-llm-docker commit** (the build recipe),
   - **vLLM composed tree** hash and **base ref/commit**,
   - **B12X composed tree** hash and **base ref/commit**,
   - any **component version bumps** (FlashInfer, CUTLASS DSL, InstantTensor, torch, …).
2. Clone blackwell-llm-docker at that commit. Find the new `patches/releases/<release>/`
   dir and the new `build-kimi-k3-<release>-runtime.sh`.

### 4.2 Diff the composition locks (what actually changed)

```bash
# new release locks
jq -r '{ref:.base.ref, commit:.base.commit, tree:.result.tree, prs:[.pull_requests[].number]}' \
  patches/releases/<NEW>/vllm/integration.lock.json
jq -r '{ref:.base.ref, commit:.base.commit, tree:.result.tree, prs:[.pull_requests[].number]}' \
  patches/releases/<NEW>/b12x/integration.lock.json
```

Compare against the current pins (§1). This tells you exactly which PRs were added/removed
and whether the vLLM/B12X base commits moved.

### 4.3 Decide: base rebuild needed?

Compare the new release's expected component versions against ours (§1). The runtime
image's post-build verifier (`verify_kimi_k3_cu133_runtime.py`) hard-fails on a version
mismatch, so the base must provide what the runtime expects.

- **Only vLLM/B12X trees changed** → reuse the existing sm121 base; rebuild only the runtime.
- **FlashInfer / CUTLASS / torch / nccl / etc. bumped** → rebuild the base for sm121 first
  (Step 3 with the new recipe's `Dockerfile.kimi-k3-cu133-torch213-base`), then the runtime.

> Note: newer releases sometimes move FlashInfer/CUTLASS/InstantTensor into a fatter
> prebuilt base image (a `voipmonitor/vllm@sha256:…` digest referenced as `BASE_IMAGE`).
> That digest is built for **sm120** — you cannot use it directly. Build the equivalent
> sm121 base from the new recipe's base Dockerfile instead, and pass it via `BASE_IMAGE=`.

### 4.4 Re-apply the sm121 adaptation

The new recipe's Dockerfiles are sm120. Re-apply the §6 mapping to **both** the base and
runtime Dockerfiles (paths/arch flags may have moved between releases — re-grep, don't
assume). Also re-check for any new sm120-only kernel references in the runtime Dockerfile
(e.g. FMHA cache dir, CuTeDSL arch).

### 4.5 Handle a new/changed source overlay

Newer runtimes may ship a `runtime/<name>/source-overlay/` (e.g. a `sitecustomize.py`)
that is hash-checked into the image (`SOURCE_OVERLAY_SHA256`). Keep it — it is part of the
qualified runtime. Confirm the runtime Dockerfile's `PYTHONPATH` still points at the fork
and overlay locations the recipe expects.

### 4.6 Build, verify, smoke-test

Run the new `build-kimi-k3-<release>-runtime.sh` with `BASE_IMAGE=` set to your sm121 base
and `ALLOW_DIRTY_BUILD=1` if the recipe isn't committed. The script self-verifies labels +
the runtime contract. Then do the §3 Step-5 smoke check, and a short real serve on one node
before cluster-wide rollout.

### 4.7 Publish + distribute

Publish under a **new** tag first, verify, then repoint:

```bash
docker tag vllm-node-kimi3-sm121 ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v3-sm121
docker push ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v3-sm121
```

Then either push to nodes (§5 in the original flow / § "Distributing") or have nodes pull
the new tag and re-tag to `vllm-node-kimi3-sm121`. Keep `:v2-sm121` intact for rollback (§0).

---

## 5. Worked example: r1 → r2

The r2 release (rtx6kpro README, image `kimi-k3-ii-vllmf21a391-b12x3be8bc7-...-20260816-r2`)
is the current newest. Delta vs our r1:

| | r1 (current) | r2 (target) |
|---|---|---|
| blackwell-llm-docker commit | `697f50ff` | `2c469ba2c54827d82b96b57450374b9c46f163ac` |
| runtime build script | `build-kimi-k3-infernal-invocation-cu133-torch213.sh` | `build-kimi-k3-qsrt-tp16-runtime.sh` |
| runtime Dockerfile | `Dockerfile.kimi-k3-infernal-invocation-cu133-torch213` | `Dockerfile.kimi-k3-qsrt-tp16-runtime` |
| composition root | `…/kimi-k3-infernal-invocation-runtime-r1` (vllm) + `…/kimi-k3-hh-runtime-r1` (b12x) | `…/kimi-k3-qsrt-ii-r3` (both) |
| vLLM base ref | `integration/kimi-k3-ii-cu133-torch213-20260811` | `dev/infernal-invocation` |
| vLLM base commit | `881ac39a4` | `ad848fc4141f201489db18d5453c50b312245a0a` |
| vLLM tree | `de04f08` | `f21a391` |
| vLLM PRs | (integration branch) | **#382–#391** |
| B12X base commit | `184d7d52` | `e68f812f15e6b06420cc649eb9caccfa42d1b9c4` |
| B12X tree | `2e6092a` | `3be8bc7` |
| B12X PRs | #124, #138, #139 | **#215, #220** |
| FlashInfer | 0.6.15.post1 | 0.6.18+cu133 |
| CUTLASS DSL | 4.6.0 | 4.6.2 |
| New mechanism | — | `runtime/kimi-k3-qsrt/source-overlay/sitecustomize.py` (hash-checked) |

**What r2 adds for us (TP16/DCP16 long-context):** #387 empty-shard-mask bypass (removes
192 kernel launches/decode token), #386 fused projection+top-16 routing, #390 vocab-sharded
DSpark sampling, #389 bounded draft KV + sharded DSpark state (memory → more target KV),
#385 projection sharding (~0.63 GiB/GPU), #388 external-draft runtime, plus the RedHat
DSpark DCP correctness fix (#310, via #388/#389). B12X #215 is an FP8 alignment correctness
fix. **#220 / #391 (island_rs all-reduce) are single-node PCIe-IPC only and do not apply to
our multi-node RoCE fabric.**

**Base rebuild:** required — FlashInfer and CUTLASS DSL both bumped, so build the r2 sm121
base first (§4.3), then the r2 runtime.

**Layer PR #395 on top of r2 (mandatory).** r2 carries PRs #382–#391; **#395 is newer and
NOT in r2**. Its commit `ab49145` ("fix(cache): preserve Mamba CoW after external hits") is a
**one-line fix in `vllm/v1/core/single_type_kv_cache_manager.py`** that fixes a confirmed
production crash on our cluster:

> `VLLM_KIMI_DEBUG_FINITE=1` caught `B12X_MLA produced invalid local output/LSE:
> output_finite=False, lse_valid=True` at ~273k tokens under TP16/DCP16/ag_rs with
> `--enable-prefix-caching`. Root cause: when an external mid-block prefix-cache hit
> becomes a running request but its first continuation needs no new Mamba block, the
> request was never registered in `_allocated_block_reqs`, so subsequent copy-on-write
> mishandled the Mamba (KDA) recurrent state. The state corruption accumulates across
> chunked-prefill continuations and eventually feeds NaN into the B12X MLA kernel output.

The fix:
```diff
             if num_required_blocks <= len(req_blocks) and not has_partial_hit:
+                self._allocated_block_reqs.add(request_id)
                 return []
```
Apply this on top of the r2 vLLM tree (or as a runtime mod — see
`mods/fix-mamba-cow-external-hit/` in eugr/spark-vllm-docker, which ports `ab49145`
and was verified to apply cleanly to the r1 tree @881ac39).

**#395 also carries the Kimi-K3 renderer fix** (commits `826e306` + `a5004ad`,
"merge split assistant prose+tool_calls turns" + "preserve Kimi reasoning during
turn merging") in `vllm/renderers/kimi_k3.py`. This is **directly relevant to an
opencode/agentic workload**: OpenAI-compatible gateways (Anthropic-Messages
adapters, Hermes-style agent loops, opencode tool loops) may split one logical
assistant turn into a prose-only message followed by a tool_calls-only message;
K3's XTML encoder then renders each as its own assistant block, teaching the model
in-context that prose-only terminal turns are normal → measurably more premature
prose-only stops mid-workflow. The fix merges the exact split shape (before
`parse_chat_messages`, so reasoning survives). Runtime mod:
`mods/fix-k3-renderer-split-turns/` (production-only diff; verified to apply
cleanly to r1 @881ac39). **Include both parts** of #395 in the next build.

**Layer PR #401 on top of r2 (mandatory) — DCP prefix-cache hybrid hits.** Also
by myshytf (same author as #395):
"[fix(prefix-cache): allow hash-aligned DCP hybrid hits](https://github.com/local-inference-lab/vllm/pull/401)",
one hunk in `vllm/v1/core/kv_cache_coordinator.py`. `HybridKVCacheCoordinator`
previously enabled fine-grained prefix hash hits only when `dcp_world_size == 1`;
under DCP>1 it coarsened local APC hits up to the DCP×hash attention block
(12288 at DCP8, **24576 at DCP16** — so the wasted-prefill regression is *worse*
on DCP16). The fix enables fine-grained hits under **any** DCP size when every
aligned Mamba (KDA) manager's `block_size == hash_block_size`. The logic is
fully `dcp_world_size`-generic — **no DCP8 hardcoding; DCP16 is covered by the
same condition** (the "fix it for DCP16" concern is already satisfied). Raises
prefix-cache effectiveness → less prefill recompute on resumed/shared-prefix
requests (opencode multi-turn). Activation requires K3's aligned KDA block to
equal the hash block; harmless no-op otherwise (verify via prefix-cache hit rate
at launch). Runtime mod: `mods/fix-dcp-hybrid-prefix-cache-hits/` (verified to
apply + reverse cleanly to r1 @881ac39). #401 is newer than r2's #382–#391 set,
so it must be layered on top of r2.

---

## 6. sm121 adaptation reference

Apply to any sm120-targeted Dockerfile from the build recipe. Re-grep each new release —
line locations move.

| sm120 (upstream) | sm121 (ours) | Where |
|---|---|---|
| `compute_120,code=sm_120` | `compute_121,code=sm_121` | `NVCC_GENCODE` |
| `compute_120,code=compute_120` | `compute_121,code=compute_121` | `NVCC_GENCODE` (PTX) |
| `TORCH_CUDA_ARCH_LIST=12.0a` | `12.1a` | base + runtime Dockerfiles (several) |
| `CMAKE_CUDA_ARCHITECTURES=120a` | `121a` | runtime Dockerfile |
| `fmha-sm120` | `fmha-sm121` | `MINFER_FMHA_CACHE_DIR` |
| `--parallel 48` | `--parallel 12` | build job count (host RAM) |

One-liner (adjust filenames per release):

```bash
sed -i 's/compute_120,code=sm_120/compute_121,code=sm_121/g;
        s/compute_120,code=compute_120/compute_121,code=compute_121/g;
        s/TORCH_CUDA_ARCH_LIST=12\.0a/TORCH_CUDA_ARCH_LIST=12.1a/g;
        s/CMAKE_CUDA_ARCHITECTURES=120a/CMAKE_CUDA_ARCHITECTURES=121a/g;
        s/fmha-sm120/fmha-sm121/g;
        s/--parallel 48/--parallel 12/' \
  Dockerfile.kimi-k3-cu133-torch213-base \
  Dockerfile.kimi-k3-<release>-runtime
```

`CUTE_DSL_ARCH` should be `sm_121a` at runtime. GB10 exposes 48 SMs — any B12X path that
requires ≥128 SMs (e.g. the PCIe DCP all-to-all pool behind `VLLM_USE_B12X_DCP_A2A=1`) is
unavailable on GB10 and must stay on the NCCL fallback (`--dcp-comm-backend ag_rs`, no
`VLLM_USE_B12X_DCP_A2A`).

---

## 7. FlashInfer on sm121 (build, bump, gotchas)

**Version.** The sm121 image (blackwell path) builds **FlashInfer 0.6.15.post1** from
source inside the runtime Dockerfile. The r2 upstream bumps to **0.6.18+cu133** — one
reason the r2 base must be rebuilt (§4.3). The separate eugr/spark-vllm-docker default
image (`vllm-node-kimi3-hh`) ships **0.6.18** via prebuilt wheels.

**sm121 build flags (both paths):** `FLASHINFER_CUDA_ARCH_LIST=12.1a`,
`TORCH_CUDA_ARCH_LIST=12.1a`, NVCC `compute_121/sm_121`. The blackwell runtime Dockerfile
compiles FlashInfer from source with these. The eugr path downloads prebuilt `12.1a`
wheels from the GitHub release `prebuilt-flashinfer-current` (3 aarch64 wheels:
`flashinfer_python`, `flashinfer_cubin`, `flashinfer_jit_cache`) or builds them from source.

**Bumping FlashInfer without a full image rebuild (eugr path only):**
- `./build-and-copy.sh --rebuild-flashinfer` — rebuilds only the `flashinfer-export`
  target, then reassembles the runner from cached layers (vLLM untouched). Old wheels are
  backed up / restored on failure.
- `--flashinfer-ref <sha|branch|tag>` / `--apply-flashinfer-pr <n>` — targeted source build.
- Wheel swap: drop new `flashinfer_*.whl` into `./wheels/` — the runner globs `*.whl`, and
  the downloader keeps newer local wheels.
- For the blackwell-path sm121 image, FlashInfer is compiled *into* the runtime image, so
  a bump means rebuilding the runtime (§4) — there is no wheel-swap shortcut.

**sm121 gotchas (important):**
- **Dense FlashInfer MLA is Hopper-only (sm10x)** — only the *sparse/compressed* MLA has an
  sm121 path. So **B12X_MLA is the only viable dense MLA backend on sm121** (why our recipes
  use `--attention-backend B12X_MLA`). The `fix-flashinfer-mla-sm121` runtime mod relaxes
  the vLLM gate `capability.major == 10` → `in (10, 12)` for the sparse path.
- **FlashInfer Mamba/SSM (incl. flash-KDA) is blocked on sm121** — a `static_assert` in the
  KDA binding rejects sm121 (SM100/103 only); `--mamba-backend FLASHINFER` measured ~−28%.
  Use the Triton KDA path (`--kda-prefill-backend triton`).
- **`flashinfer_mxfp4` MoE is SM90/SM100-only** — will not load on sm121.
- **Keep `cutlass-dsl` at 4.6.x** — the 4.4.2 pin breaks FlashInfer import
  (`OperandMajorMode` missing).
- **`FLASHINFER_DISABLE_VERSION_CHECK=1`** is required for the b12x MoE path.
- **Arch note (K3 plan §14.7):** sm_120 and sm_121 emit identical `sm_120f`-family SASS, so
  family mode (`12.0f`/`12.1f`) is optimal for dense kernels; only sparse-MMA benefits from
  `.a`, and some FP4 ops on GB10 may actually require `sm_121f` (plain `sm_121`/`sm_121a`
  can give "Feature not supported").

---

## 8. Relationship to the eugr/spark-vllm-docker mod system

The image is the base runtime. The serving harness
([eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)) applies **runtime
mods** (`mods/<name>/run.sh`, executed inside the container at launch) for fork PRs we want
but that are not yet in the image's integration branch.

- When the image moves to a newer fork tree, **re-audit the mod list** in the active recipe
  (`recipes/8x-spark-cluster/kimi-k3-*.yaml`): a mod whose PR is now baked in becomes a
  no-op (idempotent mods print "already carries"), and a mod whose anchor moved may fail to
  apply. Remove mods that the new tree subsumes; keep the rest.
- Mods are per-launch and not baked into the image, so they survive an image swap unchanged
  — but their anchors target a specific fork tree, so re-verify them after any update.

---

## 9. Distributing to the cluster

### Using eugr/spark-vllm-docker (recommended)

```bash
cd ~/eugr/spark-vllm-docker
./build-and-copy.sh --no-build -c -t vllm-node-kimi3-sm121
```

Uses `docker save | ssh docker load` to all `COPY_HOSTS` from `.env`, skipping nodes that
already have the same image ID. (Alternatively, with the published tag, each node can
`docker pull … && docker tag … vllm-node-kimi3-sm121`.)

### Manual

```bash
docker save vllm-node-kimi3-sm121 | gzip > vllm-node-kimi3-sm121.tar.gz
for node in 192.168.177.111 192.168.177.112 …; do
  ssh $node "docker load" < vllm-node-kimi3-sm121.tar.gz
done
```

---

## 10. Quick checklist for a new upstream release

1. [ ] Read rtx6kpro README → new blackwell-llm-docker commit + vLLM/B12X trees + version bumps.
2. [ ] Clone recipe @ that commit; locate new `patches/releases/<rel>/` + build script.
3. [ ] `jq` the vLLM + B12X `integration.lock.json` → note PR set and tree hashes.
4. [ ] Decide base rebuild (component versions changed?).
5. [ ] Re-apply §6 sm121 mapping to base + runtime Dockerfiles.
6. [ ] Keep any new `source-overlay/`; verify `PYTHONPATH`.
7. [ ] Build base (if needed) → build runtime (self-verifies labels + contract).
8. [ ] Smoke-test imports; short single-node serve.
9. [ ] Re-audit recipe mod list against the new fork tree (§7).
10. [ ] Publish under a NEW tag; verify; only then repoint; keep old tag for rollback (§0).
11. [ ] Distribute to cluster (§8).

---

## Credits

- [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) — model weights (BF16 attention + MXFP4 experts)
- [local-inference-lab/rtx6kpro](https://github.com/local-inference-lab/rtx6kpro) — qualified runtime reference
- [local-inference-lab/blackwell-llm-docker](https://github.com/local-inference-lab/blackwell-llm-docker) — build scripts + composition locks
- [local-inference-lab/vllm](https://github.com/local-inference-lab/vllm) — vLLM fork (B12X + DSpark + DFlash + DCP)
- [local-inference-lab/b12x](https://github.com/local-inference-lab/b12x) — CuTe DSL kernels for SM120/SM121
- [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) — GB10 cluster build harness + mod system
- [voipmonitor/InstantTensor](https://github.com/voipmonitor/InstantTensor) — fast weight loading
- [RedHatAI/Kimi-K3-speculator.dspark](https://huggingface.co/RedHatAI/Kimi-K3-speculator.dspark) — RedHat DSpark draft
- [Inferact/Kimi-K3-DSpark](https://huggingface.co/Inferact/Kimi-K3-DSpark) — Inferact DSpark draft
- [modal-labs/Kimi-K3-DFlash](https://huggingface.co/modal-labs/Kimi-K3-DFlash) — DFlash draft
