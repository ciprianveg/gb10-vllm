# PORT-REPORT — v6-remote-dspark

Port of the remote-DSpark-draft feature from **myshytf/vllm@a653e74** (II fork
line) onto the HH fork (**image v4-plus-b4f**), as mod directory
`mods/v6-remote-dspark/`.

- Diff source (read-only): `/var/tmp/a653e74.diff` (3402 lines, 9 files)
- b4f reference tree (read-only): `/var/tmp/b4f_vllmtree/` (vllm package root)
- Dry-run fakeroot: `/var/tmp/port_fakeroot/` (copy of b4f) +
  `/var/tmp/port_fakeroot_repo/` (thin repo layout: `vllm` → symlink into the
  fakeroot, `tests/v1/spec_decode/` real dirs, so both the package-relative
  and repo-relative install paths are exercised)

## (a) Files ported verbatim (5 new files)

Content taken verbatim from the diff's `+` lines; payload files live in
`mods/v6-remote-dspark/payload/` mirroring the repo layout. The patch script
installs each with a single trailing `# V4PLUS-V6-REMOTE-DSPARK` marker
comment appended (the only deviation from verbatim — required for marker
idempotency).

| Diff path | Lines | Install target |
|---|---|---|
| `vllm/entrypoints/k3_dspark_standalone.py` | 847 | `<VLLM_ROOT>/entrypoints/k3_dspark_standalone.py` |
| `vllm/entrypoints/k3_dspark_rpc.py` | 1363 | `<VLLM_ROOT>/entrypoints/k3_dspark_rpc.py` |
| `vllm/v1/worker/gpu/spec_decode/dspark/remote_speculator.py` | 770 | `<VLLM_ROOT>/v1/worker/gpu/spec_decode/dspark/remote_speculator.py` |
| `tests/v1/spec_decode/test_k3_dspark_remote_speculator.py` | 126 | `<REPO_ROOT>/tests/v1/spec_decode/…` (only if a `tests/` tree exists next to VLLM_ROOT; otherwise NOTE + skip — runtime feature unaffected) |
| `tests/v1/spec_decode/test_k3_dspark_standalone.py` | 172 | same as above |

Verbatimness was machine-checked: extraction line counts match the diff hunk
headers exactly (126/172/1363/847/770), and installed bytes equal
payload + marker line for all 5 files.

## (b) Modified-file hunks: II anchor vs b4f anchor used

All four modified files were adapted. Three of the four II hunks had anchors
that exist identically in b4f; one (`spec_decode/__init__.py`) required real
adaptation because b4f's `dflash` branch dispatches `DFlash2DraftModel` to
`DFlash2Speculator` first — a branch that does not exist in the II fork's
version of the file.

### 1. `v1/worker/gpu/buffer_utils.py` — `StagedWriteTensor.cpu` property

- II anchor: after `self.write_cu_lens = new_buffer(self.num_rows, dtype=torch.int32)` (II buffer_utils.py:152-ish), before `def stage_write(`
- b4f anchor used: **identical text**, `/var/tmp/b4f_vllmtree/v1/worker/gpu/buffer_utils.py:153` → `:155` (`StagedWriteTensor.__init__` tail, before `stage_write`). Unique (1 occurrence).
- Semantics check: b4f's `StagedWriteTensor` sets `self._uva_buf = UvaBuffer(...)` only when `uva_instead_of_gpu=True`, and `UvaBuffer` exposes `.cpu` (b4f buffer_utils.py:48) — the property's `getattr(self, "_uva_buf", None)` guard is correct for both paths.

### 2. `v1/worker/gpu/input_batch.py` — `InputBatch.all_token_ids_cpu` field

- II anchor: between `valid_num_draft_tokens_per_req: np.ndarray | None = None` and the `# When > 0, dummy batches carry seeded-random token ids...` comment (II input_batch.py:105-ish)
- b4f anchor used: **identical text**, `/var/tmp/b4f_vllmtree/v1/worker/gpu/input_batch.py:105-108`. Unique (1 occurrence). `InputBatch` is a `@dataclass` (b4f input_batch.py:37), so the new defaulted field is constructor-compatible.

### 3. `v1/worker/gpu/model_runner.py` — pass `all_token_ids_cpu` to `InputBatch`

- II anchor: `prompt_lens=… / max_req_tokens=… / valid_num_draft_tokens_per_req=… / )` inside `prepare_inputs` (II model_runner.py:1558-ish)
- b4f anchor used: **identical text**, `/var/tmp/b4f_vllmtree/v1/worker/gpu/model_runner.py:1558-1561` (the `InputBatch(` construction inside `prepare_inputs`, begun at b4f :1526). Unique (1 occurrence).
- Semantics check: b4f's `self.req_states.all_token_ids` is a `StagedWriteTensor(..., uva_instead_of_gpu=True)` (b4f `v1/worker/gpu/states.py:34-40`), so `.cpu` resolves to the pinned host table via the hunk-1 property.

### 4. `v1/worker/gpu/spec_decode/__init__.py` — env-gated remote speculator (ADAPTED)

Three sub-hunks:

| Sub-hunk | II anchor | b4f anchor used |
|---|---|---|
| `import os` header | after the 2 SPDX lines, before `import torch` (II :1-3) | identical text, b4f `spec_decode/__init__.py:1-3` |
| dflash remote hook | before `from ...dflash.speculator import DFlashSpeculator` (II :11-ish) — **II has no DFlash2 branch** | top of the `dflash` branch, before the `if "DFlash2DraftModel" in ...` check — b4f `spec_decode/__init__.py:11-12`. Rationale: placing the env-gated remote return first makes `VLLM_K3_DRAFT_REMOTE_ADDRESS` win deterministically over **both** local speculators (DFlash2 and DFlash). |
| dspark remote hook | before `from ...dspark.speculator import DSparkSpeculator` (II :23-ish) | identical text, b4f `spec_decode/__init__.py:23-26`. Both env vars accepted (`VLLM_K3_DRAFT_REMOTE_ADDRESS` OR `VLLM_K3_DSPARK_REMOTE_ADDRESS`), exactly as in II. |

All anchors unique (1 occurrence each) in b4f. No hunk was skipped.

## (c) Import verification — unresolved imports (escalation items)

Every `import`/`from … import` of `vllm.*` modules in the 5 new files was
extracted via AST (including function-level and `TYPE_CHECKING` imports) and
checked against the b4f tree (module file exists + symbol defined/imported at
top level, following star-import re-exports).

**Result: ALL vllm.\* imports resolve in b4f. There are NO unresolved
imports — no escalation items.**

Notable confirmations (symbols specifically called out as at-risk):

| Import | Status in b4f |
|---|---|
| `vllm.v1.worker.gpu.spec_decode.speculator :: BaseSpeculator, CUDAGraphCapturePhase` | OK — both defined; `BaseSpeculator` exposes `init_cudagraph_manager`, `capture(*, capture_phase:)`, `propose` matching the proxy's overrides |
| `vllm.model_executor.layers.attention.mla_attention :: MLACommonDecodeMetadata, MLACommonMetadata` | OK |
| `vllm.v1.worker.workspace :: init_workspace_manager` | OK |
| `vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils :: get_eagle3_aux_layers_from_config` | OK |
| `vllm.v1.worker.gpu.spec_decode.eagle.utils :: _create_draft_vllm_config` | OK |
| `vllm.distributed :: get_tp_group` | OK — resolves via `from .parallel_state import *` in `distributed/__init__.py:5` (no `__all__` restriction; `def get_tp_group` at parallel_state.py:1460) |
| `vllm.distributed.parallel_state :: ensure_model_parallel_initialized, init_distributed_environment, model_parallel_is_initialized` | OK |
| `vllm.engine.arg_utils :: EngineArgs` | OK — and `create_engine_config(headless=True)` exists (arg_utils.py:1938/1941) |
| `vllm.v1.worker.utils :: AttentionGroup` (incl. `create_metadata_builders`, `get_metadata_builder`) | OK |
| `vllm.v1.worker.gpu.spec_decode.dflash.utils :: load_dflash_model, maybe_load_mask_embedding` | OK |
| `vllm.v1.worker.gpu.spec_decode.dspark.utils :: load_dspark_model` | OK |
| `vllm.config.vllm :: set_current_vllm_config` | OK |
| `vllm.forward_context :: set_forward_context` | OK |
| `vllm.model_executor.layers.vocab_parallel_embedding :: ParallelLMHead, VocabParallelEmbedding` | OK |
| `vllm.utils.torch_utils :: set_default_torch_dtype`, `vllm.utils.network_utils :: get_open_port` | OK |
| `vllm.config :: VllmConfig`, `vllm.logger :: init_logger`, `vllm.v1.attention.backend :: CommonAttentionMetadata`, `vllm.v1.worker.gpu.input_batch :: InputBatch`, `vllm.v1.worker.gpu.spec_decode.utils :: get_parallel_drafting_token_id` | OK |
| intra-port (`k3_dspark_rpc`, `k3_dspark_standalone`, `dspark.remote_speculator`) | provided by this port itself |

Non-import attribute dependencies spot-checked in b4f (all present):
`SpeculativeConfig.draft_sample_method` (default `"greedy"`, speculative.py:348),
`rejection_sample_method` (`Literal["standard","synthetic","block"]`, :83/:231),
`draft_model_config.hf_config` (used throughout b4f spec_decode).

## (d) Dry-run results (fakeroot `/var/tmp/port_fakeroot`)

Procedure: `cp -a /var/tmp/b4f_vllmtree /var/tmp/port_fakeroot` (plus a thin
`/var/tmp/port_fakeroot_repo/` wrapper providing `tests/`), then:

1. **First run** — `patch_remote_dspark.py /var/tmp/port_fakeroot_repo/vllm`:
   - APPLY × 9 (3 new vllm files, 2 new test files, 4 modified files), exit 0.
   - Every touched file `py_compile`d with `doraise=True` inside the script — all passed.
2. **Second run (idempotency)** — same command: SKIP × 9 ("marker present"), exit 0. No double-application.
3. **AST parse** — `ast.parse` passed on all 9 touched files (3 new vllm + 2 new tests + 4 modified).
4. **Byte-exactness** — all 5 installed new files equal payload + single trailing marker line.
5. **Hunk diffs** — `diff` of the 4 modified files (b4f vs patched) shows exactly the intended insertions and nothing else (verified visually; output preserved in the port session).
6. **Anchor-uniqueness** — every anchor occurs exactly once in b4f (machine-checked: counts all 1).

No NOTEs, no skipped hunks in the fakeroot run. The only conditional NOTE
path is the test-file install when no `tests/` tree exists next to VLLM_ROOT
(e.g. a pure site-packages install) — by design, recorded as NOTE, non-fatal.

## Caveats / notes for review

- The `cpu` property on `StagedWriteTensor` returns the **UVA host tensor**
  only for tensors constructed with `uva_instead_of_gpu=True`; for
  GPU-backed instances it returns `None` (same as II). `RequestState.all_token_ids`
  is the only intended consumer and is always UVA-backed.
- `RemoteK3DSparkSpeculator.propose` deletes `num_tokens_across_dp` /
  `skip_attn_for_dummy_run` etc. via `del (...)` — its signature was matched
  against b4f's `BaseSpeculator.propose` parameter list only structurally
  (AST); runtime call-compatibility with b4f's model_runner call site was not
  executed (no GPU here). The II fork line and b4f share the same
  speculator interface, so risk is low, but the first live boot should be
  watched for a `TypeError` on `propose()` kwargs.
- `zmq` (`pyzmq`) is a new runtime dependency for both the verifier-side proxy
  and the standalone server — confirm it is present in the b4f image or add
  to the mod's install step if missing.
- The standalone server's `EngineArgs(..., kernel_config={"ir_op_priority": ...,
  "linear_backend": "torch"}, compilation_config={"custom_ops": ["none"]})`
  kwargs were NOT verified against b4f's `EngineArgs` fields (beyond
  `headless`); if b4f's arg surface differs, `--method dspark` server startup
  will fail at config build — check on first standalone boot.
