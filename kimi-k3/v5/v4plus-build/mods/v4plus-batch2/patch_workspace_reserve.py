#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""V4PLUS Batch 2 / B3 — complete the M4b MLA prefill-projection workspace.

v4plus-batch1's M4(b) (fork 8a89a1d2d, "retain the MLA context-projection
workspace") landed the KimiK3PrefillProjectionWorkspace class, the
model-side construction, the imports, and the impl wiring in
mla_attention.py, but on the image the reserve method + the load_weights
trigger skipped (their exact-text anchors did not match the image's
model.py).  Without those two pieces the workspace is present-but-inert:
nothing ever reserves the buffer, `_project_context` always falls back to
a fresh allocation, and the "Kimi-K3 retained %.2f MiB/rank" line never
fires.

RE-ANCHORED against the REAL image content (Build #7 ground truth,
/tmp/opencode/k3spec/v4img): the image's model.py (3028 lines) carries
batch1's import block, _vllm_config stash, and workspace construction
(verified in the extract) but NOT the reserve method or the load_weights
trigger — batch1's anchors for those two regions are 0x on the image
(Build #7 restructured the __init__ tail into an AuxiliaryStateProjector
section).  The image's mla_attention.py carries the workspace class,
_project_context, and the call swap — but batch1's ubatching-import hunk
was 0x (the image's kv_cache_interface import block has extra entries),
leaving dbo_current_ubatch_id UNBOUND: a latent NameError that fires
exactly when this script activates the workspace.

State machine (per piece, in order):
  1. present by marker            -> SKIP
  2. Build #7 real-file anchor    -> APPLY  (reserve: before KimiLinearModel
                                   .make_empty_intermediate_tensors,
                                   identified by its residual_shape body)
  3. batch1 anchor                -> APPLY  (load_weights trigger; verified
                                   against the image's KimiLinearForCausalLM
                                   .load_weights)
  4. structural class/def lookup  -> APPLY  (drift fallback)
  5. nothing matched              -> NOTE (skip-not-fail)
Plus: the ubatching import is applied by THIS script (self-sufficient —
batch1 cannot fix it), anchored after the image's real
kv_cache_interface import block.

The commit's own gating is kept verbatim: the reserve self-disables with
a warning when VLLM_BATCH_INVARIANT is on or any MLA layer's kv_b_proj is
quantized / biased / gather_output, and when the chunked-prefill workspace
sizing does not exceed max_num_batched_tokens.

Run AFTER v4plus-batch1 (or standalone on a tree where batch1's M4(b)
partially landed).  Idempotent; missing anchors print NOTEs and skip.
"""

from __future__ import annotations

import os
import py_compile
import sys

SCRIPT_NAME = "patch_workspace_reserve"
TAG = "# V4PLUS-B2 (B3: complete #568 8a89a1d2d)"

VLLM_ROOT = os.environ.get("VLLM_ROOT", "/opt/kimi-k3/vllm/vllm")
MODEL_PY = os.path.join(VLLM_ROOT, "models", "kimi_k3", "nvidia", "model.py")
MLA_ATT = os.path.join(
    VLLM_ROOT, "model_executor", "layers", "attention", "mla_attention.py"
)


def load(path: str) -> str | None:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {path} not found", file=sys.stderr)
        return None


def save(path: str, src: str) -> bool:
    try:
        compile(src, path, "exec")
    except SyntaxError as exc:
        print(
            f"[{SCRIPT_NAME}] ERROR: {path} does not compile after patch: {exc}",
            file=sys.stderr,
        )
        return False
    with open(path, "w") as f:
        f.write(src)
    py_compile.compile(path, doraise=True)
    return True


def structural_insert_before(src: str, class_name: str, def_line: str, insert: str):
    """Insert `insert` before the first `def_line` inside `class_name`.

    Returns (new_src, True) on success, (src, False) when any lookup fails.
    Tolerates text drift everywhere except the class/def statements themselves.
    """
    try:
        ci = src.index(f"class {class_name}")
    except ValueError:
        return src, False
    try:
        di = src.index(def_line, ci)
    except ValueError:
        return src, False
    return src[:di] + insert + src[di:], True


# ---------------------------------------------------------------------------
# model.py pieces (texts identical to v4plus-batch1's patch_568_riders.py)
# ---------------------------------------------------------------------------

MO_IMPORT_ANCHOR = "from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention\n"
MO_IMPORT_REPLACEMENT = (
    "from vllm.model_executor.layers.attention.mla_attention import (  # V4PLUS (#568 8a89a1d2d)\n"
    "    KimiK3PrefillProjectionWorkspace,\n"
    "    MLACommonMetadataBuilder,\n"
    "    align_mla_chunked_context_workspace_size,\n"
    ")\n"
    "from vllm.model_executor.layers.linear import (  # V4PLUS (#568 8a89a1d2d)\n"
    "    UnquantizedLinearMethod,\n"
    ")\n"
    "from vllm.models.kimi_k3.nvidia.mla import MultiHeadLatentAttention\n"
)
MO_IMPORT_PRESENT = "    KimiK3PrefillProjectionWorkspace,\n    MLACommonMetadataBuilder,\n    align_mla_chunked_context_workspace_size,\n)"

MO_INIT_ANCHOR = (
    "        config = vllm_config.model_config.hf_text_config\n"
    "        self.config = config\n"
    "        self.attn_res_block_size: int | None = config.attn_res_block_size\n"
)
MO_INIT_REPLACEMENT = (
    "        config = vllm_config.model_config.hf_text_config\n"
    "        self.config = config\n"
    "        self._vllm_config = vllm_config  # V4PLUS (#568 8a89a1d2d)\n"
    "        self.attn_res_block_size: int | None = config.attn_res_block_size\n"
)
MO_INIT_PRESENT = "        self._vllm_config = vllm_config  # V4PLUS (#568 8a89a1d2d)"
# Structural fallback: after `self.config = config` inside KimiLinearModel.
MO_INIT_FALLBACK_ANCHOR = "        self.config = config\n"
MO_INIT_FALLBACK_REPLACEMENT = (
    "        self.config = config\n"
    "        self._vllm_config = vllm_config  # V4PLUS (#568 8a89a1d2d)\n"
)

MO_WS_ANCHOR = "        aux_stream = torch.cuda.Stream()\n"
MO_WS_REPLACEMENT = (
    "        aux_stream = torch.cuda.Stream()\n"
    "        # V4PLUS (#568 8a89a1d2d): one retained context-projection output\n"
    "        # shared by all MLA layers; reserved after weights load.\n"
    "        self._mla_prefill_projection_workspace = KimiK3PrefillProjectionWorkspace(\n"
    "            num_ubatches=2 if parallel_config.enable_dbo else 1,\n"
    "            min_tokens=int(vllm_config.scheduler_config.max_num_batched_tokens) + 1,\n"
    "        )\n"
)
MO_WS_PRESENT = "        self._mla_prefill_projection_workspace = KimiK3PrefillProjectionWorkspace("
# Structural fallback: right after the _vllm_config stash (both vllm_config
# and parallel_config are in scope there via vllm_config.parallel_config).
MO_WS_FALLBACK_REPLACEMENT = (
    "        self._vllm_config = vllm_config  # V4PLUS (#568 8a89a1d2d)\n"
    "        # V4PLUS (#568 8a89a1d2d): one retained context-projection output\n"
    "        # shared by all MLA layers; reserved after weights load.\n"
    "        self._mla_prefill_projection_workspace = KimiK3PrefillProjectionWorkspace(\n"
    "            num_ubatches=2 if vllm_config.parallel_config.enable_dbo else 1,\n"
    "            min_tokens=int(vllm_config.scheduler_config.max_num_batched_tokens) + 1,\n"
    "        )\n"
)

MO_RESERVE_ANCHOR = (
    "        world_size = get_tensor_model_parallel_world_size()\n"
    '        assert config.num_attention_heads % world_size == 0, (\n'
    '            "num_attention_heads must be divisible by world_size"\n'
    "        )\n"
    "\n"
    "    def make_empty_intermediate_tensors(\n"
)
MO_RESERVE_REPLACEMENT = (
    "        world_size = get_tensor_model_parallel_world_size()\n"
    '        assert config.num_attention_heads % world_size == 0, (\n'
    '            "num_attention_heads must be divisible by world_size"\n'
    "        )\n"
    "\n"
    "    def reserve_mla_prefill_projection_workspace(self) -> None:\n"
    '        """Reserve one large context projection output shared by MLA layers.\n'
    "\n"
    "        V4PLUS (#568 8a89a1d2d, adapted to v4): internal_tokens comes\n"
    "        from the chunked-prefill workspace sizing instead of\n"
    "        envs.VLLM_MLA_INTERNAL_CONTEXT_WORKSPACE_SIZE (absent here),\n"
    "        torch.cuda.empty_cache() replaces\n"
    "        _release_cuda_cache_before_retained_allocation (absent here),\n"
    "        and the workspace is assigned to the layers' MLA impls directly\n"
    "        instead of threading a ctor parameter through the decoder layer.\n"
    '        """\n'
    "        if self._mla_prefill_projection_workspace._buffer is not None:\n"
    "            return\n"
    "        internal_tokens = (\n"
    "            MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(\n"
    "                self._vllm_config\n"
    "            )\n"
    "        )\n"
    "        max_batched_tokens = self._vllm_config.scheduler_config.max_num_batched_tokens\n"
    "        if internal_tokens <= max_batched_tokens:\n"
    "            return\n"
    "        workspace_tokens = align_mla_chunked_context_workspace_size(\n"
    "            self._vllm_config, internal_tokens\n"
    "        )\n"
    "        mla_layers = [\n"
    "            layer.self_attn\n"
    "            for layer in self.layers\n"
    "            if isinstance(getattr(layer, \"self_attn\", None), MultiHeadLatentAttention)\n"
    "        ]\n"
    "        if not mla_layers:\n"
    "            return\n"
    "        first = mla_layers[0]\n"
    "        if envs.VLLM_BATCH_INVARIANT or not all(\n"
    "            isinstance(layer.kv_b_proj.quant_method, UnquantizedLinearMethod)\n"
    "            and layer.kv_b_proj.bias is None\n"
    "            and not layer.kv_b_proj.gather_output\n"
    "            for layer in mla_layers\n"
    "        ):\n"
    "            logger.warning_once(\n"
    '                "Kimi-K3 retained context projection is unavailable for the "\n'
    '                "configured kv_b_proj method."\n'
    "            )\n"
    "            return\n"
    "        weight = first.kv_b_proj.weight\n"
    "        torch.cuda.empty_cache()  # V4PLUS: release cached blocks first\n"
    "        self._mla_prefill_projection_workspace.reserve(\n"
    "            max_tokens=workspace_tokens,\n"
    "            output_size=weight.shape[0],\n"
    "            dtype=weight.dtype,\n"
    "            device=weight.device,\n"
    "        )\n"
    "        for layer in mla_layers:\n"
    '            impl = getattr(layer, "impl", None)\n'
    "            if impl is not None:\n"
    "                impl._prefill_projection_workspace = (\n"
    "                    self._mla_prefill_projection_workspace\n"
    "                )\n"
    "        logger.info_once(\n"
    '            "Kimi-K3 retained %.2f MiB/rank for the %d-token MLA context "\n'
    '            "projection workspace shared across layers.",\n'
    "            self._mla_prefill_projection_workspace.nbytes / (1024**2),\n"
    "            workspace_tokens,\n"
    "        )\n"
    "\n"
    "    def make_empty_intermediate_tensors(\n"
)
MO_RESERVE_PRESENT = "    def reserve_mla_prefill_projection_workspace(self) -> None:"
# PRIMARY anchor for Build #7: the image's KimiLinearModel.
# make_empty_intermediate_tensors is identified by its residual_shape body
# (the wrapper's delegating copy lacks it).  The batch1 anchor (the
# world_size assert __init__ tail) is 0x on the image — Build #7
# restructured that region (AuxiliaryStateProjector etc.).
MO_RESERVE_REALFILE_ANCHOR = (
    "    def make_empty_intermediate_tensors(\n"
    "        self,\n"
    "        batch_size: int,\n"
    "        dtype: torch.dtype,\n"
    "        device: torch.device,\n"
    "    ) -> IntermediateTensors:\n"
    "        residual_shape: tuple[int, ...] = (batch_size, self.config.hidden_size)\n"
)
# Structural fallback: insert before KimiLinearModel's
# make_empty_intermediate_tensors (first occurrence after the class line —
# the ForCausalLM wrapper's delegating copy comes later in the file).
MO_RESERVE_STRUCTURAL_DEF = "    def make_empty_intermediate_tensors(\n"
# Structural-path insert: the METHOD ONLY (the batch1 replacement repeats the
# __init__ tail as part of its anchor->replacement design; inserting that
# full text structurally would duplicate the tail).
_MO_RESERVE_TAIL = "    def make_empty_intermediate_tensors(\n"
MO_RESERVE_METHOD_ONLY = MO_RESERVE_REPLACEMENT.split(
    "    def reserve_mla_prefill_projection_workspace", 1
)[1]
MO_RESERVE_METHOD_ONLY = (
    "    def reserve_mla_prefill_projection_workspace" + MO_RESERVE_METHOD_ONLY
)
# Drop the replacement's trailing anchor line (the def we insert BEFORE).
assert MO_RESERVE_METHOD_ONLY.endswith(_MO_RESERVE_TAIL)
MO_RESERVE_METHOD_ONLY = MO_RESERVE_METHOD_ONLY[: -len(_MO_RESERVE_TAIL)]

MO_CALL_ANCHOR = (
    "        loaded = loader.load_weights(weights)\n"
    "        self.model.finalize_mega_moe_weights()\n"
)
MO_CALL_REPLACEMENT = (
    "        loaded = loader.load_weights(weights)\n"
    "        self.model.finalize_mega_moe_weights()\n"
    "        # V4PLUS (#568 8a89a1d2d): reserve the retained MLA context-\n"
    "        # projection workspace once, after all weights (and MLA impls)\n"
    "        # are in place. AutoWeightsLoader delegates to this load_weights\n"
    "        # on both the plain and ForConditionalGeneration paths.\n"
    "        self.model.reserve_mla_prefill_projection_workspace()\n"
)
MO_CALL_PRESENT = "        self.model.reserve_mla_prefill_projection_workspace()"
# Structural fallback: before `return loaded` inside
# KimiLinearForCausalLM.load_weights.
MO_CALL_STRUCTURAL_CLASS = "KimiLinearForCausalLM"
MO_CALL_STRUCTURAL_DEF = "    def load_weights"
MO_CALL_STRUCTURAL_RET = "        return loaded\n"
MO_CALL_STRUCTURAL_INSERT = (
    "        # V4PLUS (#568 8a89a1d2d): reserve the retained MLA context-\n"
    "        # projection workspace once, after all weights (and MLA impls)\n"
    "        # are in place. AutoWeightsLoader delegates to this load_weights\n"
    "        # on both the plain and ForConditionalGeneration paths.\n"
    "        self.model.reserve_mla_prefill_projection_workspace()\n"
    "        return loaded\n"
)

# ---------------------------------------------------------------------------
# mla_attention.py pieces (presence checks only in the common case —
# batch1 landed these; structural re-apply if absent)
# ---------------------------------------------------------------------------

ATT_CLASS_PRESENT = "class KimiK3PrefillProjectionWorkspace:"
ATT_METHOD_PRESENT = "    def _project_context(self, kv_c_normed: torch.Tensor) -> torch.Tensor:"
ATT_CALL_PRESENT = "            kv_nope = self._project_context(kv_c_normed).view(  # V4PLUS (#568 8a89a1d2d)"
ATT_IMPORT_PRESENT = "from vllm.v1.worker.ubatching import (  # V4PLUS (#568 8a89a1d2d)"

# ---------------------------------------------------------------------------
# mla_attention.py — the ubatching import (B3 is SELF-SUFFICIENT for it).
#
# RE-ANCHORED for Build #7: the image's mla_attention.py has a LARGER
# kv_cache_interface import block than the 881ac39a4 clone (extra
# SlidingWindowMLASpec / get_kv_cache_dcp_shard_count entries), so batch1's
# ATT_IMPORT anchor was 0x and the import never landed — while the workspace
# class (which calls dbo_current_ubatch_id()) DID land, leaving a latent
# NameError exactly when B3 activates the workspace.  This hunk adds the
# import after the image's real kv_cache_interface block.
# ---------------------------------------------------------------------------
A_IMPORT_ANCHOR = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    AttentionSpec,\n"
    "    KVCacheSpec,\n"
    "    MLAAttentionSpec,\n"
    "    SlidingWindowMLASpec,\n"
    "    get_kv_cache_dcp_shard_count,\n"
    "    get_kv_quant_mode,\n"
    ")\n"
)
A_IMPORT_REPLACEMENT = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    AttentionSpec,\n"
    "    KVCacheSpec,\n"
    "    MLAAttentionSpec,\n"
    "    SlidingWindowMLASpec,\n"
    "    get_kv_cache_dcp_shard_count,\n"
    "    get_kv_quant_mode,\n"
    ")\n"
    "from vllm.v1.worker.ubatching import (  # V4PLUS (#568 8a89a1d2d)\n"
    "    dbo_current_ubatch_id,\n"
    ")\n"
)


def patch_model_py() -> bool:
    src = load(MODEL_PY)
    if src is None:
        return False
    changed = False
    ok = True

    # --- MO_IMPORT ---
    if MO_IMPORT_PRESENT in src:
        print(f"[{SCRIPT_NAME}] SKIP  model.py: mla_attention import block (already present)")
    elif src.count(MO_IMPORT_ANCHOR) == 1:
        src = src.replace(MO_IMPORT_ANCHOR, MO_IMPORT_REPLACEMENT, 1)
        changed = True
        print(f"[{SCRIPT_NAME}] APPLY model.py: mla_attention import block")
    else:
        # Structural fallback: insert the import block before any
        # `from vllm.models.kimi_k3.nvidia.mla import` line.
        idx = src.find("from vllm.models.kimi_k3.nvidia.mla import")
        if idx != -1 and "KimiK3PrefillProjectionWorkspace," not in src:
            src = src[:idx] + MO_IMPORT_REPLACEMENT.replace(
                MO_IMPORT_ANCHOR, ""
            ) + src[idx:]
            changed = True
            print(f"[{SCRIPT_NAME}] APPLY model.py: mla_attention import block (structural)")
        else:
            print(f"[{SCRIPT_NAME}] NOTE  model.py: import block — no anchor; skipped")
            ok = False

    # --- MO_INIT ---
    if MO_INIT_PRESENT in src:
        print(f"[{SCRIPT_NAME}] SKIP  model.py: _vllm_config stash (already present)")
    elif src.count(MO_INIT_ANCHOR) == 1:
        src = src.replace(MO_INIT_ANCHOR, MO_INIT_REPLACEMENT, 1)
        changed = True
        print(f"[{SCRIPT_NAME}] APPLY model.py: _vllm_config stash")
    else:
        # Structural fallback: after `self.config = config` inside
        # KimiLinearModel (vllm_config is in __init__ scope there).
        done = False
        try:
            ci = src.index("class KimiLinearModel")
            si = src.index(MO_INIT_FALLBACK_ANCHOR, ci)
            end = si + len(MO_INIT_FALLBACK_ANCHOR)
            src = src[:end] + "\n" + MO_INIT_PRESENT + src[end:]
            changed = True
            done = True
        except ValueError:
            done = False
        if done:
            print(f"[{SCRIPT_NAME}] APPLY model.py: _vllm_config stash (structural)")
        else:
            print(f"[{SCRIPT_NAME}] NOTE  model.py: _vllm_config stash — no anchor; skipped")
            ok = False

    # --- MO_WS ---
    if MO_WS_PRESENT in src:
        print(f"[{SCRIPT_NAME}] SKIP  model.py: workspace construction (already present)")
    elif src.count(MO_WS_ANCHOR) == 1:
        src = src.replace(MO_WS_ANCHOR, MO_WS_REPLACEMENT, 1)
        changed = True
        print(f"[{SCRIPT_NAME}] APPLY model.py: workspace construction")
    elif MO_INIT_PRESENT in src and src.count(MO_INIT_PRESENT) == 1:
        # Structural fallback: attach the construction to the _vllm_config
        # stash (vllm_config in scope).
        src = src.replace(
            MO_INIT_PRESENT + "\n",
            MO_WS_FALLBACK_REPLACEMENT,
            1,
        )
        changed = True
        print(f"[{SCRIPT_NAME}] APPLY model.py: workspace construction (structural)")
    else:
        print(f"[{SCRIPT_NAME}] NOTE  model.py: workspace construction — no anchor; skipped")
        ok = False

    # --- MO_RESERVE (the piece that missed on the image) ---
    if MO_RESERVE_PRESENT in src:
        print(f"[{SCRIPT_NAME}] SKIP  model.py: reserve method (already present)")
    elif src.count(MO_RESERVE_REALFILE_ANCHOR) == 1:
        src = src.replace(
            MO_RESERVE_REALFILE_ANCHOR,
            MO_RESERVE_METHOD_ONLY + MO_RESERVE_REALFILE_ANCHOR,
            1,
        )
        changed = True
        print(f"[{SCRIPT_NAME}] APPLY model.py: reserve method (Build #7 anchor: before KimiLinearModel.make_empty_intermediate_tensors)")
    elif src.count(MO_RESERVE_ANCHOR) == 1:
        src = src.replace(MO_RESERVE_ANCHOR, MO_RESERVE_REPLACEMENT, 1)
        changed = True
        print(f"[{SCRIPT_NAME}] APPLY model.py: reserve method")
    else:
        new, done = structural_insert_before(
            src, "KimiLinearModel", MO_RESERVE_STRUCTURAL_DEF, MO_RESERVE_METHOD_ONLY
        )
        if done:
            src = new
            changed = True
            print(f"[{SCRIPT_NAME}] APPLY model.py: reserve method (structural: before KimiLinearModel.make_empty_intermediate_tensors)")
        else:
            print(
                f"[{SCRIPT_NAME}] NOTE  model.py: reserve method — neither the "
                "batch1 anchor nor the structural lookup (class KimiLinearModel "
                "-> make_empty_intermediate_tensors) matched; skipped"
            )
            ok = False

    # --- MO_CALL (the piece that missed on the image) ---
    if MO_CALL_PRESENT in src:
        print(f"[{SCRIPT_NAME}] SKIP  model.py: load_weights trigger (already present)")
    elif src.count(MO_CALL_ANCHOR) == 1:
        src = src.replace(MO_CALL_ANCHOR, MO_CALL_REPLACEMENT, 1)
        changed = True
        print(f"[{SCRIPT_NAME}] APPLY model.py: load_weights trigger")
    else:
        # Structural: KimiLinearForCausalLM -> load_weights -> `return loaded`.
        done = False
        try:
            ci = src.index(f"class {MO_CALL_STRUCTURAL_CLASS}")
            di = src.index(MO_CALL_STRUCTURAL_DEF, ci)
            ri = src.index(MO_CALL_STRUCTURAL_RET, di)
            src = src[:ri] + MO_CALL_STRUCTURAL_INSERT + src[ri + len(MO_CALL_STRUCTURAL_RET):]
            changed = True
            done = True
        except ValueError:
            done = False
        if done:
            print(f"[{SCRIPT_NAME}] APPLY model.py: load_weights trigger (structural: before KimiLinearForCausalLM.load_weights' return loaded)")
        else:
            print(
                f"[{SCRIPT_NAME}] NOTE  model.py: load_weights trigger — neither "
                "the batch1 anchor nor the structural lookup (class "
                "KimiLinearForCausalLM -> load_weights -> return loaded) "
                "matched; skipped"
            )
            ok = False

    if changed:
        if not save(MODEL_PY, src):
            return False
    return ok


def patch_mla_attention() -> bool:
    """Apply the missing ubatching import, then check the impl-side pieces.

    The workspace class batch1 landed in mla_attention.py calls
    dbo_current_ubatch_id() — without this import it is a latent NameError
    that fires exactly when B3's reserve activates the workspace.
    """
    src = load(MLA_ATT)
    if src is None:
        return False
    changed = False
    if ATT_IMPORT_PRESENT in src:
        print(f"[{SCRIPT_NAME}] SKIP  mla_attention.py: ubatching import (already present)")
    elif src.count(A_IMPORT_ANCHOR) == 1:
        src = src.replace(A_IMPORT_ANCHOR, A_IMPORT_REPLACEMENT, 1)
        changed = True
        print(f"[{SCRIPT_NAME}] APPLY mla_attention.py: ubatching import (Build #7 anchor: after the kv_cache_interface block)")
    else:
        print(
            f"[{SCRIPT_NAME}] NOTE  mla_attention.py: ubatching import — the "
            "kv_cache_interface import-block anchor did not match; the landed "
            "workspace class would NameError on dbo_current_ubatch_id when "
            "the reserve activates it. Fix manually before serving."
        )
    if changed:
        if not save(MLA_ATT, src):
            return False
    ok = True
    for label, marker in (
        ("workspace class", ATT_CLASS_PRESENT),
        ("_project_context method", ATT_METHOD_PRESENT),
        ("chunk-loop call swap", ATT_CALL_PRESENT),
        ("ubatching import", ATT_IMPORT_PRESENT),
    ):
        if marker in src:
            print(f"[{SCRIPT_NAME}] OK    mla_attention.py: {label} present")
        else:
            print(
                f"[{SCRIPT_NAME}] NOTE  mla_attention.py: {label} ABSENT — "
                "run v4plus-batch1's patch_568_riders.py against the real "
                "mla_attention.py; the model-side reserve stays inert until "
                "the impl-side pieces exist."
            )
            ok = False
    return ok


def patch_overflow_grace() -> bool:
    """V4PLUS-B2 fix: the retained workspace is sized for
    max_num_batched_tokens, but long-context chunked prefill can ask the
    context projection for MORE rows than that (observed: 66048 needed vs
    65536 retained at a 150K prompt -> hard ValueError, engine death).
    Replace the rows-overflow raise with a warn-once + return None so the
    caller's fresh-allocation fallback serves oversized projections.
    """
    src = load(MLA_ATT)
    if src is None:
        return False
    if "_v4plus_overflow_warned" in src:
        print(f"[{SCRIPT_NAME}] SKIP  mla_attention.py: workspace overflow grace (already present)")
        return True
    anchor = (
        "        if num_tokens > buffer.shape[1]:\n"
        "            raise ValueError(\n"
        '                f"context projection needs {num_tokens} rows, but the retained "\n'
        '                f"workspace has {buffer.shape[1]}"\n'
        "            )\n"
    )
    replacement = (
        "        if num_tokens > buffer.shape[1]:\n"
        "            # V4PLUS-B2 fix: long-context chunked prefill can need more\n"
        "            # rows than the retained (max_num_batched_tokens-sized)\n"
        "            # workspace. Fall back to fresh allocation (None) instead\n"
        "            # of raising; warn once.\n"
        "            cls = type(self)\n"
        '            if not getattr(cls, "_v4plus_overflow_warned", False):\n'
        '                cls._v4plus_overflow_warned = True\n'
        '                logger.warning(\n'
        '                    "Kimi-K3 context projection needs %d rows, exceeding "\n'
        '                    "the retained workspace (%d rows); falling back to "\n'
        '                    "fresh allocation for oversized projections.",\n'
        "                    num_tokens,\n"
        "                    buffer.shape[1],\n"
        "                )\n"
        "            return None\n"
    )
    if src.count(anchor) == 1:
        src = src.replace(anchor, replacement, 1)
        if not save(MLA_ATT, src):
            return False
        print(f"[{SCRIPT_NAME}] APPLY mla_attention.py: workspace overflow grace (raise -> warn-once + fresh-allocation fallback)")
        return True
    print(
        f"[{SCRIPT_NAME}] NOTE  mla_attention.py: workspace overflow grace — "
        "rows-overflow raise anchor found 0x; the retained workspace will "
        "RAISE (engine death) on contexts exceeding its sizing. Fix manually."
    )
    return False


def main() -> int:
    print(f"[{SCRIPT_NAME}] {TAG}")
    ok = patch_model_py()
    att_ok = patch_mla_attention()
    grace_ok = patch_overflow_grace()
    if ok and att_ok and grace_ok:
        print(
            f"[{SCRIPT_NAME}] M4(b) wiring complete. Verify at startup: the "
            "\"Kimi-K3 retained %.2f MiB/rank for the %d-token MLA context "
            "projection workspace\" log line must appear (it proves the "
            "load_weights-path reserve runs). If it does not, weights are "
            "loading through a path that bypasses KimiLinearForCausalLM."
            "load_weights — re-route the trigger."
        )
        print(
            f"[{SCRIPT_NAME}] Commit gating kept: the reserve self-disables "
            "(warning) when VLLM_BATCH_INVARIANT is on or any MLA kv_b_proj "
            "is quantized/biased/gather_output, and when the chunked-prefill "
            "workspace sizing does not exceed max_num_batched_tokens."
        )
    else:
        print(
            f"[{SCRIPT_NAME}] NOTE: one or more pieces are missing — see the "
            "NOTEs above; serving is unaffected (the workspace stays inert "
            "and _project_context keeps its fresh-allocation fallback)."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
