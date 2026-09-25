# Building the KIMI-K3 v5 sm121 Image

> **Canonical image (2026-09-25, "0920R1"):** GHCR `sha256:336ae7d1…` —
> tags `v5-prd` / `v5-prd-sm121` / `latest`. Composition = the 09-20 bake
> (this file as of commit `65dda8f`) PLUS three runtime-validated mods
> baked 2026-09-25:
> - `fix-k3-request-endpoint-cache` — request endpoint cache (lab#732;
>   env-gated: set `VLLM_K3_REQUEST_ENDPOINT_CACHE=1` to engage; restores
>   prefixes at arbitrary token boundaries for token-continuous sessions)
> - `fix-k3-dflash-unaligned-endpoint-restore` — DFlash/DSpark drafting
>   proceeds through block-unaligned cache-restored prefixes (the shift
>   kernels already floor the restore count; the old whole-batch bail-out
>   killed drafting on ~every endpoint-cache follow-up)
> - `fix-mamba-align-state-free` — frees the TWO-steps-ago KDA align state
>   block (sawtooth KV inflation fix, 7.6x on APC-on long prefill)
>
> The FIX1-FIX7 bake chain (kv-dedup, harden-apc-drain, backport batch,
> no-mix, serial, epilogue, revert) is ABANDONED: it cost ~30% base decode
> speed and its kv-dedup member was a CoW-ref leak. The base keeps pristine
> per-occurrence CoW retention — no revert mod needed. Crash-guard note:
> this base has NO baked no-mix gate; if mixed-step crashes appear, apply
> `fix-no-mixed-steps` + `fix-k3-nomix-standalone` at runtime with
> `VLLM_NO_MIX_ARM=1` (see the mods' headers). `drop-caches` stays a
> recipe-level runtime mod (a loop, nothing to bake).

> **Image:** `ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v5-prd` (linux/arm64, sm121, ~29.5 GB, flattened)

## Easy route (recommended)

```bash
docker pull ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v5-prd
```

## From-v4 route

`build.sh` reproduces the production lineage on top of the published
`v4-prd` base, applying layers in this exact order:

1. **Batch layers** ([`v4plus-build/`](v4plus-build/), built by
   `build.sh` automatically when present):
   `Dockerfile.v4-plus` → `b3` → `b4` → `b4d` → `b4e` → `b4f`
   (RoCEnante TP one-shot + DCP collectives, b12x spec-merge rewrite,
   fused verify TILE8, fp8 draft fix, splitfloor, ep-empty-meta).
2. **v6-cluster-build mod order** (Python patchers, prerequisite
   pr53152 BEFORE pr54168):
   `v6-remote-dspark`, k3-tier1 `pr53524`, `pr53525`, `pr53942`, `pr52388`,
   `pr53152`, `pr54168`, `pr54896`, `pr54697`, `pr56159`,
   `fix-k3-retention-dense`, `fix-k3-kda-spec-token-init`.
3. **`_C` rebuild**: apply the two build-time CUDA mods
   (`perf-pr55356-54896-mla-cache-kernels`, `perf-pr55180-fp8-cta-swizzle`)
   to `csrc/`, then rebuild `_C_stable_libtorch.abi3.so` in-image
   (~47 min on GB10). Flatten with `docker export | docker import` between
   stages only if the base has deep layer history (>250 `docker history`
   lines — the original chain hit the overlayfs max-depth wall this way).
4. **Perf layers in commit order**:
   `fix-flashinfer-autotune-import` → `fix-k3-marlin-nopad` →
   `fix-mxfp4-triton-sm121` → `fix-k3-nan-gumbel` →
   `fix-dspark-adaptive-nst` → `fix-multistream-record-stream` +
   `fix-grammar-stream-fence` → `fix-dspark-draft-noeplb` +
   `fix-adaptive-min-depth`.

```bash
./kimi-k3/v5/build.sh            # local tag: ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v5-prd
./kimi-k3/v5/build.sh --push     # build + push to GHCR
```

`perf-pr54048-router-gemm-fam120` and `drop-caches` stay runtime-only mods
(never baked).

## From scratch

Build the v2 base image ([`../v2/BUILD-SM121-IMAGE.md`](../v2/BUILD-SM121-IMAGE.md)),
advance the trees to the r36 pins — vLLM `e755f87` / b12x `2d466e3` — then
follow the same from-v4 order above with `v4-prd` replaced by that base.

## Verify a pulled/built image

```bash
docker run --rm --entrypoint bash ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v5-prd -c '
    V=/opt/kimi-k3/vllm
    ls $V/vllm/_C_stable_libtorch.abi3.so >/dev/null && echo "SO OK"
    grep -qc "perf-pr55356-54896-mla-cache-kernels" $V/vllm/_custom_ops.py && echo "MLA epilogue OK"
    grep -rlq "perf-pr55180-fp8-cta-swizzle: applied" $V/csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/ && echo "FP8 swizzle OK"
    grep -rlq "fix-k3-marlin-nopad" $V/vllm && echo "marlin-nopad OK"
    grep -rlq "fix-adaptive-min-depth" $V/vllm && echo "min-depth OK"
    grep -rlq "fix-k3-request-endpoint-cache" $V/vllm && echo "endpoint-cache OK"
    grep -rlq "fix-k3-dflash-unaligned-endpoint-restore" $V/vllm && echo "dflash-unaligned-restore OK"
    grep -rlq "_two_steps_ago_block_idx" $V/vllm && echo "mamba-align-state-free OK"
'
```

Expected: `SO OK`, `MLA epilogue OK`, `FP8 swizzle OK`, `marlin-nopad OK`,
`min-depth OK`, `endpoint-cache OK`, `dflash-unaligned-restore OK`,
`mamba-align-state-free OK`.
