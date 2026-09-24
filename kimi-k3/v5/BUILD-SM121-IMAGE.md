# Building the KIMI-K3 v5 sm121 Image

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
   `fix-adaptive-min-depth` → `fix-kv-dedup-retained-endpoints` +
   `fix-mamba-align-state-free` (APC-loop fixes: CoW drain dedup +
   free two-steps-ago KDA align state; fixes 7.6× KV inflation —
   400K APC-on completes at 22% peak, 0 preemptions) →
   **FIX2 batch (2026-09-22)**:
   `fix-moe-skip-padding-producer` (upstream #56079: mark SP/padding rows so
   MoE routing invalidates them) → `fix-long-prefill-singleton` (upstream
   #57951: long-prefill threshold not applied to singleton requests) →
    `fix-sm121-cublas-oob` (lab#710, both halves: DCP head-major
    reduce-scatter in dcp_utils.py + the MLA half added in FIX3 —
    `_bmm_with_disjoint_batches` wrapper and 4 call-site wraps in
    mla_attention.py, guarding the SM120/121 cuBLAS reads-past-allocation
    on interleaved batched-MMAs; crash class: mixed chunked-prefill +
    spec-decode step → illegal memory access at TP16) → `fix-k3-kda-first-chunk`
   (upstream #51483: stateless first chunk misclassified as decode, read
   unmasked conv/recurrent state) → `fix-gb10-kv-sizing` (upstream #55828:
   process-scoped KV/memory accounting for GB10 UMA; inert while
   `--kv-cache-memory-bytes` is pinned) → `fix-gb10-nvml-fallback`
   (upstream #57378: NVML→torch fallback for GB10 device memory query) →
   `harden-apc-drain` (oracle-spec'd hardening of the two APC-loop fixes:
   ref_cnt underflow tripwire in `BlockPool.free_blocks`, guarded align-free
   with block-identity + frontier check, env-gated drain debug counters
   `VLLM_APC_DRAIN_DEBUG` / `VLLM_APC_DRAIN_ABORT`).
   The last 7 must run AFTER the APC-loop fixes (harden-apc-drain anchors on
   their output text). All are Python-only — no `_C` rebuild needed.

> **Canonical FIX2 image:** `0887adc360e9` (stamp
> `V5-PRD-FIX2-BAKE-STAMP 20260922150708`), produced as a fast overlay
> commit on the FIX1 image `8d94b1da2cc4` and pushed as GHCR digest
> `sha256:146fae23…` under `v5-prd` / `latest` / `v5-prd-sm121`. A full
> `build.sh` run (order above) reproduces the same mod set from `v4-prd`.

> **Canonical FIX3 image:** `c7ca72aff22c` (stamp
> `V5-PRD-FIX3-BAKE-STAMP 20260922212739`), overlay on FIX2 adding the
> lab#710 **MLA half** (the `_bmm_with_disjoint_batches` wrapper + 4
> call-site wraps in mla_attention.py — the crash-class fix for the
> TP16/DCP16 mixed-batch cuBLAS OOB). GHCR digest `sha256:af81b0d8…`,
> same three tags. The mod in `mods/fix-sm121-cublas-oob/` now carries
> BOTH halves, so a full `build.sh` run reproduces FIX3 content directly.

> **Canonical FIX4 image:** `6867b9d40e19` (stamp
> `V5-PRD-FIX4-BAKE-STAMP 20260923115137`), overlay on FIX3 baking the
> scheduler-level **no-mix** crash-class guard (`mods/fix-no-mixed-steps`:
> decode requests skip steps carrying a prefill chunk with >32K tokens
> remaining — closes the mixed-batch OOB class that kernel patches alone
> could not cover; production canary PASS, 0 crash signature). GHCR digest
> `sha256:02d51170…`, same three tags. A full `build.sh` run (order above,
> with `fix-no-mixed-steps` last) reproduces FIX4 content directly.

> **Canonical FIX5 image:** `d3f4c14ca83a` (stamp
> `V5-PRD-FIX5-BAKE-STAMP 20260924012142`), overlay on FIX4 baking the
> request-endpoint cache (`mods/fix-k3-request-endpoint-cache`, lab#732
> port: finished requests publish end positions so token-continuous turns
> resume at the last computed token; both CodeRabbit bugs fixed in-port).
> Validated on .111: crash-test PASS + token-level extension 8.8s vs 20.1s.
> GHCR digest `sha256:02d51170…` (manifest), same three tags.

> **Canonical FIX6 image:** `cba6b4e37b0a` (stamp
> `V5-PRD-FIX6-BAKE-STAMP 20260924074851`), overlay on FIX5 baking the MoE
> dual-stream serialization (`mods/fix-k3-moe-serial-smalln`: gate+down_proj
> run sequentially for <=8-token decode steps — SM121 cuBLAS
> EXECUTION_FAILED class). Validated live on TP16 prod: tool-bench c2
> clean. GHCR digest `sha256:36e3e8d9…`, same three tags.

> **Canonical FIX7 image (current):** `6c0f809b9cdb` (stamp
> `V5-PRD-FIX7-BAKE-STAMP 20260924125125`), overlay on FIX6 baking the gate
> epilogue swap (`mods/fix-k3-moe-gate-bf16-epilogue`: bf16 mm + explicit
> .float() instead of the fused fp32 cuBLAS epilogue at small-M decode
> steps). Validated on TP16 prod (tool-bench clean). GHCR digest
> `sha256:f8792856…`, same three tags. A full `build.sh` run (order above,
> with the three crash mods last) reproduces FIX7 content directly.

> **Revert (no image tag — runtime-equivalent removal):**
> `mods/fix-k3-revert-apc-leak` restores pristine per-occurrence CoW
> retention by reversing `fix-kv-dedup-retained-endpoints` + removing
> `harden-apc-drain`'s kv_cache_manager sites (which anchor on deduped
> text, hence last in the loop). The dedup leaked +1 ref per shared block
> per step (pinned pool → hangs + hit collapse); harden's block_pool
> tripwire + single_type guard stay (other files, untouched). Safe on trees
> that never had the leak (detects pristine state, skips). Recipes list it
> as a runtime mod (harmless no-op where already reverted).

```bash
./kimi-k3/v5/build.sh            # local tag: ghcr.io/ciprianveg/gb10-vllm/kimi-k3:v5-prd
./kimi-k3/v5/build.sh --push     # build + push to GHCR
```

`perf-pr54048-router-gemm-fam120`, `drop-caches`, and
`fix-prefill-decode-share` (the decode-favoring interleave — env-gated,
needs the recipe's `VLLM_PREFILL_COMPUTE_SHARE_INTERVAL`) stay runtime-only
mods (never baked).

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
    grep -rlq "fix-kv-dedup-retained-endpoints" $V/vllm && echo "kv-dedup OK"
    grep -rlq "_two_steps_ago_block_idx" $V/vllm && echo "mamba-align-state-free OK"
    grep -rlq "fix-moe-skip-padding-producer" $V/vllm && echo "moe-skip-padding OK"
    grep -rlq "fix-long-prefill-singleton" $V/vllm && echo "long-prefill-singleton OK"
    grep -rlq "fix-sm121-cublas-oob" $V/vllm && echo "sm121-cublas-oob OK"
    grep -rlq "_bmm_with_disjoint_batches" $V/vllm/model_executor && echo "mla-bmm-disjoint (lab#710 MLA half) OK"
    grep -rlq "fix-k3-kda-first-chunk" $V/vllm && echo "kda-first-chunk OK"
    grep -rlq "fix-gb10-kv-sizing" $V/vllm && echo "gb10-kv-sizing OK"
    grep -rlq "fix-gb10-nvml-fallback" $V/vllm && echo "gb10-nvml-fallback OK"
    grep -rlq "harden-apc-drain" $V/vllm && echo "apc-drain-hardening OK"
    grep -rlq "fix-no-mixed-steps" $V/vllm && echo "no-mixed-steps OK"
    grep -rlq "fix-k3-request-endpoint-cache" $V/vllm && echo "endpoint-cache OK"
    grep -rlq "fix-k3-moe-serial-smalln" $V/vllm && echo "moe-serial-smalln OK"
    grep -rlq "fix-k3-moe-gate-bf16-epilogue" $V/vllm && echo "moe-gate-bf16-epilogue OK"
    ! grep -rlq "fix-kv-dedup-retained-endpoints" $V/vllm/v1/core/kv_cache_manager.py && echo "revert-apc-leak OK (no leak markers)"
'
```

Expected: `SO OK`, `MLA epilogue OK`, `FP8 swizzle OK`, `marlin-nopad OK`,
`min-depth OK`, `kv-dedup OK`, `mamba-align-state-free OK`,
`moe-skip-padding OK`, `long-prefill-singleton OK`, `sm121-cublas-oob OK`,
`mla-bmm-disjoint (lab#710 MLA half) OK`,
`kda-first-chunk OK`, `gb10-kv-sizing OK`, `gb10-nvml-fallback OK`,
`apc-drain-hardening OK`, `no-mixed-steps OK`, `endpoint-cache OK`,
`moe-serial-smalln OK`, `moe-gate-bf16-epilogue OK`,
`revert-apc-leak OK (no leak markers)`.
