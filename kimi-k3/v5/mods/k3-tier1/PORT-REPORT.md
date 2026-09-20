# k3-tier1 — Upstream vLLM Kernel PR Ports (HH fork, image v4-plus-b4f, r29-based)

Port of 9 upstream vLLM kernel PRs (post-Aug-20 small-batch decode
optimizations) as individual mods under `mods/k3-tier1/`, following the
`fix-k3-retention-dense` / `v6-remote-dspark` pattern: anchor-based
replace, MARKER idempotency, `py_compile(doraise=True)` on every touched
file, APPLY/SKIP/NOTE semantics, VLLM_ROOT auto-resolution
(`python3 -c "import vllm..."` → `/opt/kimi-k3/vllm/vllm` → `/opt/vllm/vllm`).

Reference tree: `/var/tmp/vllm-src/vllm/` (b4f + v6-remote-dspark base).
All dry-run testing done against fresh copies under
`/var/tmp/tier1_fakeroot/`.

## Summary

| Mod | Upstream PR | Status | Files touched |
|---|---|---|---|
| `pr53524` | #53524 — prefetch ll_bf16 router weights (M=1) | **PORTED** (full) | `_ll_bf16_dotprod.py`, `ll_bf16.py` |
| `pr53525` | #53525 — C=1 KDA PDL pipeline | **PORTED** (Python side; csrc NOTE-skipped, native parts schema-gated) | `_skinny_gemm.py`, `skinny_gemm.py`, `_custom_ops.py`, `low_latency_gemm.py` |
| `pr53942` | #53942 — eh_proj optimization | **PORTED** (full) | `mtp.py`, `low_latency_gemm.py` |
| `pr52388` | #52388 — Mamba metadata prep | **SKIPPED — already in fork** (verification-only no-op mod) | none |
| `pr54168` | #54168 — low-M fused latent MoE tail | **PORTED** (full; requires pr53152 first) | `primitives.py`, allreduce collective, `fused_add_multicast_skinny_gemm.py`, `lamport_copy.py`, `latent_moe_tail.py` |
| `pr53152` | #53152 + #53327 — fuse MXFP4 top-k finalize into latent tail | **PORTED** (kernel side; production defer plumbing NOTE-skipped — fork lacks the infra) | NEW `moe_output.py` (minimal), `latent_moe_tail.py`, allreduce collective |
| `pr54896` | #54896 — MLA decode concat/cache epilogue | **SKIPPED — csrc-only** (not portable as a Python mod; documented no-op mod) | none |
| `pr54697` | #54697 — overlap low-M KDA projections | **PORTED** (TP8-shape-gated as upstream; TP16 flagged, see below) | NEW `kda_skinny_gemm.py`, `low_latency_gemm.py`, `kda.py`, `model.py`, `kernel_warmup.py` |
| `pr56159` | #56159 — avoid KDA mixed-batch gather/scatter | **PORTED** (full, with fork adaptations) | `kda_metadata.py`, `kda.py`, `chunk.py`, `fused_recurrent.py` |

---

## Per-PR details

### pr53524 — prefetch ll_bf16 router weights for M=1 (#53524) — PORTED
- **Hunks adapted**: none needed — fork anchors match upstream pre-PR context
  exactly (r29 `ll_bf16` is identical to upstream's).
  - `_ll_bf16_dotprod.py`: `LLBf16Dotprod.__init__` gains
    `prefetch_pdl_weights` + `main_prefetch_tiles = min(main_tiles, 8)`;
    new `_vector_dotprod_prefetched` jit method; kernel restructured to
    prefetch the static B stripe into rmem *before*
    `griddepcontrol_wait()`.
  - `ll_bf16.py`: `LLBf16Gemm.__init__(*, prefetch_pdl_weights=False)`;
    `_compile_dotprod` passes the flag; new
    `ll_bf16_gemm_c1_pdl_kernel = LLBf16Gemm(prefetch_pdl_weights=True)`;
    `ll_bf16_gemm()` auto-selects it for `shape[0] == 1`.
- **Dependencies ported**: none. Pure CuTe DSL (JIT), no C++.
- Fork divergences: `_compile_dotprod`'s extra
  `--ptxas-options -maxrregcount=64` kept (fork-only tuning).

### pr53525 — optimize C=1 KDA PDL pipeline (#53525) — PORTED (Python side)
- **Ported**:
  - `_skinny_gemm.py`: `early_pdl_trigger` config/attr;
    `griddepcontrol_launch_dependents()` moved to right after the mainloop
    when enabled; final trigger gated on `not early_pdl_trigger`.
  - `skinny_gemm.py`: `SkinnyGemmConfig.early_pdl_trigger`; compile
    passthrough; warmup sort key.
  - `low_latency_gemm.py`: `ResolvedCall` → 3-tuple; `_build_plan` sets the
    early trigger for M=1 `in_proj_qkvgfab`/`o_proj` (cute) and
    `f_b_proj`/`fused_qkv_a_proj` (dsv3); `_run_plan` unpacks and forwards;
    warmup configs now derived from the plan.
- **NOTE-skipped (csrc — requires rebuilding `_C_stable_libtorch.abi3.so`)**:
  - `dsv3_fused_a_gemm.cu`: `early_pdl_trigger` template param + relaxed
    single-row stride check.
  - `fused_kda_decode_kernel.cu`: deferred `cudaGridDependencySynchronize`
    for B==1.
  - `torch_bindings.cpp`: `early_pdl_trigger` schema arg.
- **Adaptation (fork binary compatibility)**: `_custom_ops.py` gains a
  cached `dsv3_fused_a_gemm_early_pdl_supported()` schema probe; the native
  call and the `_runtime_ok` single-row-stride relaxation are **gated on
  it**. On the current pre-#53525 `.so` the dsv3 early trigger and stride
  relaxation stay off (stock behavior, no crash); the cute-DSL early
  trigger engages immediately. Once the image is rebuilt with the csrc
  changes, everything engages with no further patching.
- Also NOTE-skipped: upstream `model.py` hunk (pure formatting no-op).
- **Hunks adapted**: upstream `_runtime_ok` was unconditional; fork port
  gates the relaxation (see above) because the fork's dsv3 binary would
  reject non-packed single-row views (`STD_TORCH_CHECK` crash) — a
  regression risk upstream never had (its csrc shipped in the same PR).

### pr53942 — eh_proj optimization (#53942) — PORTED
- `mtp.py`: `eh_proj` `nn.Linear` → `ReplicatedLinear(..., quant_config=None,
  prefix=maybe_prefix(prefix, "eh_proj"), return_bias=False)` so
  `enable_kimi_k3_low_latency_gemm` installs the measured plan on it
  (`return_bias=False` keeps the existing call site returning a bare
  tensor; verified against the fork's `ReplicatedLinear.forward`).
- `low_latency_gemm.py`: new `(7168, 14336)` projection spec
  (`SkinnyGemmConfig(1, 256, 2, vector_width=4, static_k=14336)` for M=1,
  `_cute(2, 224, 4, 2)` for M=2) inserted before the `(20480, 7168)` entry.
- **Dependencies ported**: none. Fork already imports `maybe_prefix` in
  `mtp.py`.

### pr52388 — Mamba metadata prep optimization (#52388) — SKIPPED (already present)
The fork (b4f) **already contains** all three upstream hunks verbatim:
- `kda_metadata.py`: `KimiK3KDAMetadataBuilder.mamba_aligned_state_indices`
  attribute + the align-mode precomputed-indices branch (with the MRV1
  fallback assert).
- `mamba_utils.py`: `get_aligned_state_indices_multi_group_kernel` triton
  kernel, `MambaSpecDecodeGPUContext.aligned_state_indices` buffer,
  `compute_aligned_state_indices()`.
- `mamba_hybrid.py`: the `prepare_attn` hook distributing all-group aligned
  state indices to builders.
The mod is a **verification-only no-op**: it greps for the 5 required
snippets and fails loudly if the base image ever loses the feature. No
changes made.

### pr53152 (+ #53327) — fuse MXFP4 top-k finalization into latent tail — PORTED (kernel side)
- **FUSED_TOPK16 check (per instructions)**: grepped the reference tree
  (and pristine b4f) for `FUSED_TOPK16` / `VLLM_KIMI_FUSED_TOPK16` /
  `fused_topk16` — **no matches anywhere**. The fork has no such
  env/implementation, so the fusion is complementary → both PRs ported
  into `mods/k3-tier1/pr53152/` (53327 applied logically with 53152).
- **Ported (self-contained kernel side)**:
  - NEW `model_executor/layers/fused_moe/moe_output.py` — **minimal
    upstream subset**: only the `UnfinalizedMoEOutput` dataclass (the full
    module's protocol depends on infra the fork lacks). Import path matches
    upstream for future rebases.
  - `latent_moe_tail.py`: contract/`initialize` gain `experts_per_token`;
    `__call__`/`_validate_inputs` accept `Tensor | UnfinalizedMoEOutput`;
    capacity bumps `_MAX_NUM_TOKENS 16→128`, `_COLLECTIVE_TOKEN_CTAS 8→32`
    (upstream's own change — tier-0 tail fusion now engages up to M=128).
  - allreduce collective: full `top_k` plumbing — kernel params, the
    top-k finalize block (gather permuted GEMM2 rows × expert weights),
    `_compile_key`/`compile_kernel`/`launch` keys and args,
    `CollectiveKernel` dummy expert tensors + union validation.
- **NOTE-skipped (production defer plumbing — fork lacks the entire
  deferred-finalize MoE stack)**: `latent_moe_runner.py` defer gating
  (nvidia + amd), `fused_moe/config.py` fields
  (`defer_moe_finalize*` / `use_deferred_moe_finalize` /
  `should_defer_moe_finalize`), `modular_kernel.py`
  `supports_deferred_moe_finalize` passthrough, `moe_runner.py` `_unpack`
  widening, `routed_experts.py` / `fused_moe_method_base.py` /
  `mxfp4.py` annotations, all three `trtllm_*_moe.py` expert wrappers
  (the fork has **no trtllm expert modules at all**), and
  `convert_flashinfer_moe_output`. Porting these would mean inventing the
  whole protocol around absent infrastructure — skipped per the
  "never guess" rule.
- **#53327 (bugfix)**: its only hunk moves the #53152 defer decision
  before MoE-kernel setup in `latent_moe_runner`. Since that gating is not
  ported, the bugfix has no target; its substance is moot in the
  kernel-only port. Documented in the patch output.
- The deferred mode is exercisable directly via
  `KimiK3LatentMoETailOp.initialize(..., experts_per_token=16)` (as
  upstream's tests/benchmarks do).

### pr54168 — optimize low-M fused latent MoE tail (#54168) — PORTED
- **FUSED_TOPK16 interaction check (per instructions)**: none — the env/
  implementation does not exist in the fork (see pr53152 above). Port
  lands unmodified.
- **PREREQUISITE**: `pr53152` must be applied first (upstream #54168
  builds on #53152's `top_k` finalize block and `CollectiveKernel` top_k
  plumbing). The patch script hard-fails with `PREREQUISITE FAILED` if the
  pr53152 marker is absent.
- **Ported**:
  - `primitives.py`: `fma_f32_bf16`, `finalize_top16_bf16` (+ inline-PTX
    asm builder), `stmc_bf16x8` (multimem.st), `load_shared_f32x2/x4`.
  - allreduce collective: 7-CTA/64-thread geometry for M≤4 at
    (TP8, 3584, 7168); `top_k==16` bf16 finalize fast path (generic path
    now accumulates FP32); parity-alternating DSM reduction slots;
    single-token schedule skipping the redundant Lamport arrival wait;
    `stmc` multicast publish + early PDL; compact ReduceScatter roles
    (destination rounds; arrival target `m * shared_roles`); seven-CTA
    warmup specializations; `__call__` launch_max_m selection.
  - `fused_add_multicast_skinny_gemm.py`: M≤5 config
    (block 224, vector_width 16), 32-byte alignment (runtime check +
    fake tensors + `_as_cute`), `fma_f32_bf16` accumulation.
  - `lamport_copy.py`: `launch_dependents` before polling, grid trimmed to
    fragment count, copy+cleanup fused into one pass.
  - `latent_moe_tail.py`: `_LAMPORT_COPY_THREADS 224→128`.

### pr54896 — cut MLA decode concat/cache epilogue (#54896) — SKIPPED (csrc-only)
The entire PR is CUDA C++ (`fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu`:
`writeLatent576` lane/lane_stride split so decode rows (≤64 tokens) use
SPLIT=3 warps, deferred grid-dependency wait, `launchPdl` →
`launchPdlSlots`; plus tests). It touches **no Python** — the fork ships a
prebuilt `_C_stable_libtorch.abi3.so`, which mods cannot patch. The mod is
a documented no-op; to engage, rebuild the image with the #54896 csrc
patch (no Python-side changes needed).

### pr54697 — overlap low-M KDA projections (#54697) — PORTED (TP8-gated)
- **Ported**:
  - NEW `ops/cute_dsl/kda_skinny_gemm.py`: TP8 skinny GEMMs for the KDA
    F_A/beta (144×7168) and F_B (1536×128) projections (payload, verbatim
    from upstream).
  - `low_latency_gemm.py`: `KDA_*` configs/constants,
    `run_kda_projection_overlap()` (two-stream fork/join via the fork's
    existing `maybe_execute_in_parallel`), `autotune_kda_qkvg()`,
    `_enable_kda_projection_overlap()` installer + warmup block.
  - `kda.py`: `aux_stream` param + events/`_projection_overlap_max_tokens`
    attrs; overlap fast path in `forward()` (capture-only,
    packed-stride-only).
  - `model.py`: `KimiDecoderLayer` passes its existing `aux_stream` into
    `KimiK3DeltaAttention`.
  - `kernel_warmup.py`: `_autotune_kimi_k3_kda_qkvg` helper + call inside
    `flashinfer_autotune` (adapted to the fork's `runner._dummy_run(...)`
    autotune body — upstream's context has
    `_run_flashinfer_autotune_dummy_runs`/`replayssm_autotune_warmup`,
    which the fork's warmup file doesn't).
- **TP8 vs TP16 (per instructions)**: the two-stream fork/join mechanism is
  TP-agnostic, but every measured constant is TP8-specific (packed weight
  6288×7168, F_B 1536×128, `_KDA_TP_SIZE = 8`, QKVG/skinny split tables).
  At TP16 the fork shards `f_a` (`in_proj_qkvgfab` is 3216×7168), so those
  constants do not transfer. The port keeps upstream's runtime shape gate:
  the overlap engages **only** at TP8-exact shapes and stays stock (zero
  behavioral change) at TP16. Enabling TP16 would require re-measuring the
  config/split tables on TP16 shapes — flagged, not guessed.
- **Fork adaptations**: the `forward()` restructure preserves the fork's
  `split_mixed_precision_input` and `shard_f_a` branches (upstream context
  has neither); the overlap path skips the `f_a` gather + `f_b_proj` since
  it produces `g1` directly.

### pr56159 — avoid KDA mixed-batch gather/scatter (#56159) — PORTED
- **Ported**:
  - `kda_metadata.py`: `KimiK3KDAMetadata.spec_token_start` /
    `non_spec_token_start` fields; `build()` detects a contiguous spec /
    non-spec partition (single transition in the active spec mask) and
    records the offsets.
  - `kda.py` `_forward`: contiguous-slice branch replaces
    `index_select` gathers; spec and non-spec attention outputs write
    straight into `core_attn_out` slices; restore step skips
    `index_copy_` when the layout was continuous.
  - `chunk.py` / `fused_recurrent.py`: `out=` plumbing for
    `chunk_kda_with_fused_gate(_fwd)` and
    `fused_recurrent_kda_packed_decode`.
- **Fork adaptations**: the fork's `_forward` lacks upstream's
  flashinfer/recoverssm prefill branches, and its flashkda backend cannot
  write into a caller-provided slice. When the layout is continuous but
  the active non-spec backend did not write in place (flashkda), the
  restore falls back to a plain slice copy
  (`core_attn_out[:, non_spec_slice] = ...`) instead of upstream's
  in-place-only assumption — still eliminating the input gathers and the
  index scatter. The spec path keeps the fork's prefix-slice `spec_out`
  and extends it with the contiguous-slice case. `_forward` reads the new
  metadata fields via `getattr(..., None)` for safety against older
  metadata objects.

---

## Combined-order dry test

**Dependency-corrected order** (see conflict note below):
`pr53524 → pr53525 → pr53942 → pr52388 → pr53152 → pr54168 → pr54896 → pr54697 → pr56159`

- Fresh copy of the reference tree at `/var/tmp/tier1_fakeroot/combined_ok/vllm`.
- Result: **all 9 mods APPLY clean** (pr52388 SKIP-verified, pr54896
  documented no-op), **zero anchor failures, zero file conflicts**.
- Re-running all 9 on the patched tree: **all SKIP** (idempotency holds).
- `ast.parse` passes on all 20 touched files.

**Shared-file composition** (no anchor conflicts):
- `low_latency_gemm.py` — pr53525 (7 hunks) + pr53942 (1) + pr54697 (5):
  disjoint anchors (pr54697's warmup block anchors on the
  `residual_warmup_configs` line, which pr53525 preserves).
- `kda.py` — pr54697 (`forward`/`__init__`) + pr56159 (`_forward`):
  disjoint functions.
- allreduce collective + `latent_moe_tail.py` — pr53152 then pr54168
  (sequential dependency by design).

### CONFLICT FOUND: orchestrator order vs pr54168 ⇢ pr53152 dependency

The specified combined order places **pr54168 before pr53152**, but
upstream #54168 builds on #53152's top-k finalize block (its diff context
literally contains `expanded_idx_to_permuted_idx` — a #53152 addition).
Applying pr54168 first would leave its topk-dependent hunks without
anchors.

Resolution: `pr54168` declares **pr53152 as a hard prerequisite** (its
patch script exits 1 with `PREREQUISITE FAILED` if the pr53152 marker is
absent) and the combined test uses the corrected order above. Demonstrated
on a second fresh copy (`/var/tmp/tier1_fakeroot/combined_origorder/vllm`)
with the original order: pr54168 refuses cleanly (exit 1, no partial
apply); every other mod applies; swapping the two makes everything pass.
**Recommended apply order is the corrected one.**

## Verification matrix (per mod, fresh-copy dry runs)

| Mod | APPLY | re-run SKIP | ast.parse |
|---|---|---|---|
| pr53524 | 2 files, 8 hunks | yes | pass |
| pr53525 | 4 files, 16 hunks | yes | pass |
| pr53942 | 2 files, 3 hunks | yes | pass |
| pr52388 | — (5× SKIP already-present) | yes | n/a (no changes) |
| pr53152 | new file + 2 files, 39 hunks | yes | pass |
| pr54168 | 5 files, 34 hunks (on pr53152 base) | yes | pass |
| pr54896 | — (documented no-op) | yes | n/a (no changes) |
| pr54697 | new file + 4 files, 11 hunks | yes | pass |
| pr56159 | 4 files, 19 hunks | yes | pass |

All patch scripts `py_compile` every touched file with `doraise=True` at
write time (compile failure ⇒ file not written, FAIL reported).

## Engagement notes ( serving-side )

- pr53524 / pr53525(cute) / pr53942 / pr54168 / pr54697(TP8) / pr56159:
  engage automatically at warmup/first use; no env vars.
- pr53525 (dsv3 native half): engages only after the image is rebuilt with
  the #53525 csrc (schema-gated; see above).
- pr54896: requires the image rebuild with the #54896 csrc patch.
- pr53152: kernel-side fusion is available via
  `KimiK3LatentMoETailOp.initialize(..., experts_per_token=16)`; the
  runner-side deferred path needs the deferred-finalize MoE stack the fork
  doesn't have. Note the capacity bump (tail fusion up to M=128) applies to
  the existing finalized path too — watch tier-0 engagement after warmup.
