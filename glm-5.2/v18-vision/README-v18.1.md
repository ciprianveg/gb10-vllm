# v18.1-vision — native SM121 rebuild + optional NVFP4 KV cache

`v18.1-vision` is a drop-in upgrade over `v18-vision`. It rebuilds vLLM **in-place** on the
v18-vision image so the compiled extensions target **native sm_121** cubins instead of the
original image's sm_120 forward-compat cubins, and it enables the **NVFP4 MLA KV-cache** kernel
block that the v18 build had disabled.

Two things change vs v18-vision, and they are independent:

1. **Native sm_121 cubins** (always on in v18.1-vision) — the mechanism that enables the NVFP4
   SM12x kernel block. 
2. **NVFP4 KV cache** (opt-in via `--kv-cache-dtype nvfp4_ds_mla`) — more context per GiB of KV
   memory, for workloads that need it. **fp8 remains the default and recommended KV cache.**

> **Default recommendation: use `v18.1-vision` with `fp8_ds_mla`** (recipe
> [`glm52-int4int8-v18.1-vision.yaml`](recipes/glm52-int4int8-v18.1-vision.yaml)).
> Reach for the NVFP4 recipe ([`...-nvfp4.yaml`](recipes/glm52-int4int8-v18.1-vision-nvfp4.yaml))
> **only when you need more context than fp8 provides.**

Image: `ghcr.io/ciprianveg/gb10-glm-5.2:v18.1-vision` (linux/arm64, sm_121).

---

## Why native sm_121: it's required for the NVFP4 kernel path

The original v18-vision image was built with `TORCH_CUDA_ARCH_LIST=12.0` (sm_120 forward-compat
cubins) and a comment asserting "a-suffix cubins don't load on SM121". That assertion is
**outdated** on driver 580.x / CUDA 13.2.1: native sm_121 cubins load and execute on GB10 fine

Rebuilding for native sm_121 is **the mechanism that enables the NVFP4 MLA KV-cache kernels** —
the v18 image's sm_120 cubins skip the FP4 SM12x kernel set entirely (the v18 build hardcoded
`FP4_SM120_ARCHS="99.0f"` to disable it). With `12.1a` + the arch-list fix, CMake emits the NVFP4
SM12x kernels for native sm_121. That is the reason v18.1-vision exists.

**Native sm_121 over sm_120.** 
**Prefill** — prefill is consistently slightly faster on native sm_121, across depths and
runs:

| prefill test | v18 (sm_120) | v18.1 (sm_121) | delta |
|---|---|---|---|
| pp2048 | 1329.72 | 1341.43 | +0.9% |
| pp2048 @ d16000 | 1319.37 | 1346.14 | +2.0% |
| pp2048 @ d100000 | 1202.08 | 1206.56 / 1217.64 (two runs) | +0.4% / +1.3% |

That's +0.4% to +2.0%, positive at every depth and on both runs — a small but real sm_121 gain
(native cubins avoid sm_120 forward-compat overhead and can use sm_121-specific scheduling on the
attention/GEMM kernels). The reason to adopt v18.1-vision is the **NVFP4 KV-cache option** it
enables, with a modest prefill bump as a bonus; decode is a wash within MTP noise.

---

## NVFP4 KV cache: what it buys, and what it costs

GLM-5.2 uses MLA (`B12X_MLA_SPARSE`). The NVFP4 MLA KV-cache path packs KV at
`kv_gmem_stride=432` vs fp8's `656` — about **1.52× more tokens per GiB** of KV memory.

**The cost: ~10% slower prefill (measured).** NVFP4's win is decode-side memory density, not
prefill speed. Prefill is compute-bound, so it does not benefit from a smaller KV — it only pays
two extra costs that fp8 doesn't:

1. **KV-write quantization** during prefill — `concat_and_cache_nvfp4_mla` quantizes bf16 → E2M1 +
   E4M3 group-16 scales per token, vs fp8's near-free cast.
2. **Attention KV-read dequant** on the extend path — dequanting E2M1+E4M3+scales is costlier than
   fp8, on a less-tuned record layout.

The fp8_ds_mla layout (656 B/token) *mirrors the FlashMLA / SPARSE_MLA_SM120 layout* — it is the
mature, V32-packed, 576-byte-aligned reference path the b12x unified SM120 backend was built
around. The nvfp4_ds_mla path (432 B/token, E2M1+E4M3) is the newer addition (PR #115); 

**The 10% prefill tax is paid on every request; the 1.52× density only pays off when you actually
need more context than fp8 provides.** So unless your workload regularly pushes past fp8's context
ceiling, fp8 is strictly better (faster prefill, same decode band, longer stability track). Use
NVFP4 only when you'd otherwise be context-capped.

**Use NVFP4 only when you need that extra context.** fp8 is faster per-token at shallow context
and has a longer stability track record. 

---

## Benchmarks (v18.1-vision, fp8 KV cache, 8× GB10 TP8)

llama-benchy, coherent corpus, `tg=1500`, single-stream. 

| test | t/s | peak t/s | TTFR (ms) | est PPT (ms) | e2e TTFT (ms) |
|:---|---:|---:|---:|---:|---:|
| pp2048 | 1341.43 | — | 1528.99 | 1526.72 | 1528.99 |
| tg1500 | 33.57 | 61 | — | — | — |
| pp2048 @ d4000 | 1309.77 | — | 4619.86 | 4617.59 | 4619.86 |
| tg1500 @ d4000 | 33.92 | 61 | — | — | — |
| pp2048 @ d16000 | 1346.14 | — | 13409.52 | 13407.26 | 13411.64 |
| tg1500 @ d16000 | 38.05 | 64 | — | — | — |
| pp2048 @ d100000 | 1217.64 | — | 83810.19 | 83808.12 | 83819.41 |
| tg1500 @ d100000 | 31.98 | 41 | — | — | — |

**Decode is noisy; prefill is not.** `tg1500 @ d100000` was measured at 52.40 t/s (peak 69) in one run and
31.98 t/s (peak 41) in another, with nothing else changed — a ~64% swing driven
purely by MTP acceptance (accepted drafts vary ~3.71–4.53 run-to-run). Prefill is stable run-to-run and is
consistently +0.4–2.0% over the v18 sm_120 baseline. 
To not take into consideration the warmup tine needed for various shapes compilation, ignore the first run results
of a specific benchmark.

### Coding workload (game-bench, single-stream, temp=0, thinking disabled)

Snake-game generation, 1500 completion tokens:

```
Completion tokens: 1500   Prompt tokens: 43   Total: 1543
Wall time: 26.16s   tok/s: 57.33
```

Coding workloads have consistently high MTP acceptance, so adaptive MTP 2/4/5 ratchets up to k=5
and catches all-5-accept streaks that k=4 can't — a measured **~7% decode speedup on coding** vs
~2,4, with **no benefit on general prose** (prose lives at k=2/4 with the 5th draft position almost
always rejected, `p4 gain ≈ 0`). The 57.33 tok/s above reflects that coding boost.


---

## How the image is built (Dockerfile.nvfp4)

The v18-vision image already contains the full toolchain (nvcc, g++, cmake) and the vLLM source at
`/opt/vllm` (gilded-gnosis-v18-final @ 264bce1d, `.git` stripped). So the build rebuilds vLLM
**in-place** on the existing image — no external build harness required.

Two CMakeLists.txt patches enable native sm_121 + the NVFP4 SM12x kernel block:

1. **`CUDA_SUPPORTED_ARCHS` (CUDA 13.x branch) += `12.1`** — without this, `12.1a` is clamped down
   to `12.0` before the FP4 family-match runs, so no native sm_121 cubins are emitted.
2. **`FP4_SM120_ARCHS` revert `99.0f` → `12.0f`** — the v18 build hardcoded `"99.0f"` (a
   non-existent arch) to skip the entire FP4 SM12x kernel block. Reverting to `"12.0f"` lets the
   family match resolve to `12.1a` → native sm_121 NVFP4 kernels.

Then vLLM is rebuilt with `TORCH_CUDA_ARCH_LIST=12.1a`, `CMAKE_CUDA_ARCHITECTURES=121`. CMake
confirms the intent:

```
-- CUDA target architectures: 12.1a
-- CUDA supported target architectures: 12.1a
-- Building SM12x NVFP4 for archs: 12.1a
```

**The vision + adaptive-MTP overlay is preserved.** A naive `pip install --force-reinstall` of the
rebuilt wheel would clobber the 9 v18-vision overlay `.py` files with base-v18 source. The
Dockerfile instead extracts **only the rebuilt `*.so` extensions** and lays them over the installed
package, leaving the overlay intact. Build-time `cuobjdump` + overlay-intact checks gate the image.

Build:
```bash
docker pull ghcr.io/ciprianveg/gb10-glm-5.2:v18-vision
./v18-vision/build-nvfp4.sh           # builds ghcr.io/ciprianveg/gb10-glm-5.2:v18.1-vision
```

Post-build verification (built into the script):
```
sm_121 cubins (must be >0): 63
sm_120 cubins:               0
vision overlay glm5v.py:     present
```

---

## Recipes

Both recipes use the **same v18.1-vision image** and the **same adaptive MTP 2/4/5 /
`num_speculative_tokens=5`** settings as the v18-vision production recipe. They differ only in KV
cache dtype.

| Recipe | KV cache | Use when |
|---|---|---|
| [`glm52-int4int8-v18.1-vision.yaml`](recipes/glm52-int4int8-v18.1-vision.yaml) | `fp8_ds_mla` | **Default.** All v18 workloads unless you need more context than fp8 provides. |
| [`glm52-int4int8-v18.1-vision-nvfp4.yaml`](recipes/glm52-int4int8-v18.1-vision-nvfp4.yaml) | `nvfp4_ds_mla` | Opt-in for max-context workloads (~1.52× KV density). Validate correctness before trusting. |

Launch (from spark-vllm-docker):
```bash
# Default (fp8)
./run-recipe.sh ../gb10-glm-5.2/v18-vision/recipes/glm52-int4int8-v18.1-vision.yaml --setup

# NVFP4 opt-in
./run-recipe.sh ../gb10-glm-5.2/v18-vision/recipes/glm52-int4int8-v18.1-vision-nvfp4.yaml --setup
```

---

## Validation

On an 8× GB10 / SM121 cluster, driver 580.159.03, CUDA 13.2.1:

- **Cubin-load smoke test** — `vllm import` + a CUDA matmul on a single GB10 with the rebuilt
  `_C_stable_libtorch.abi3.so` (63 sm_121 cubins, 0 sm_120). Confirms sm_121 cubins load and
  execute on SM121 (resolves the old "a-suffix cubins don't load" comment).
- **NVFP4 boot** — `kv_cache_dtype=nvfp4_ds_mla`, `B12X GLM MLA KV format: KV_FP8_ROPE=0
  kv_gmem_stride=432 kv_cache_dtype=nvfp4_ds_mla` (stride 432 matches the expected NVFP4 MLA
  layout). KV pool 888,152 tokens @ 410K/0.69.
- **fp8 functional parity** — the v18.1-vision image with `fp8_ds_mla` runs cleanly with no
  functional regression vs the v18-vision fp8 baseline. 


---

## Credits

- **Light Foundry Notes** ([@light_foundry on x.com](https://x.com/light_foundry), [original post](https://x.com/light_foundry/status/2082302264480579891)): the native-sm_121 + NVFP4 MLA KV-cache enablement approach this image follows — the CUDA 13.x `CUDA_SUPPORTED_ARCHS` += `12.1` fix, the `FP4_SM120_ARCHS` `99.0f`→`12.0f` guard revert, and the rebuild at `12.1a`/`121`. Their writeup is also the source for the 1.52× density math (stride 432 vs 656, BF16 RoPE block uncompressed), the `--speculative-config` model-path vision gotcha, and the single-run MTP-acceptance noise caveat (~10%) that this guide relies on.

Built on the v18-vision stack (Gilded Gnosis v18 + Baseten MoonViT vision + CosmicRaisins adaptive
MTP). No CUDA or NVIDIA libraries were modified; only vLLM's compiled extensions were rebuilt.
