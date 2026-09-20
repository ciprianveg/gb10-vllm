#!/usr/bin/env python3
# V4PLUS M4 — fork #568 riders:
#   (a) ee429fb01 — bf16 kv_b_proj cast BUGFIX (correctness for stock bf16
#       kv_b_proj checkpoints + fp8 KV cache, the exact s6 checkpoint class).
#   (b) 8a89a1d2d — retained MLA context-projection workspace (removes the
#       per-chunk kv_b_proj output allocation, ~144 MiB of hot-path
#       allocations per chunk; TTTT/interleave win, decode-neutral).
#
# (a) VERDICT: ALREADY IN THE v4 TREE — skip with a presence check. The
# ee429fb01 commit fixes the LATER tree, where the chunked-context loop
# moved from the MLA impl into the model (mla.py _compute_prefill_context
# with run_chunk). The v4 tree still has the loop in
# MLACommonBaseImpl._compute_prefill_context (mla_attention.py), and that
# version ALREADY computes kv_b_proj_input_dtype via
# _get_kv_b_proj_input_dtype() and casts per chunk (verified against the
# clone). The assert the commit removes (process_weights_after_loading)
# does not exist in v4's mla.py either. Nothing to do.
#
# (b) PORTED WITH ADAPTATIONS (the later-tree target sites do not exist in
# v4; the semantics are ported onto v4's equivalents):
#   * KimiK3PrefillProjectionWorkspace class: VERBATIM from 8a89a1d2d,
#     placed in mla_attention.py next to the impl that uses it (the commit
#     puts it in models/kimi_k3/nvidia/mla.py, which in the later tree owns
#     the chunk loop).
#   * The retained-projection logic (project_context) becomes
#     MLACommonBaseImpl._project_context, swapped into the chunk loop of
#     _compute_prefill_context (v4's equivalent of the commit's run_chunk
#     edit). num_heads replaces num_local_heads (impl-side naming).
#   * The workspace is assigned to the layers' impls by the model's reserve
#     method (instead of threading a ctor parameter through
#     KimiDecoderLayer, which would need 3 more hunks across diverged
#     constructors).
#   * internal_tokens comes from
#     MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size
#     (v4's chunked-workspace sizing) instead of
#     envs.VLLM_MLA_INTERNAL_CONTEXT_WORKSPACE_SIZE (absent in v4).
#   * torch.cuda.empty_cache() replaces _release_cuda_cache_before_retained_
#     allocation (absent in v4).
#   * The reserve call lives at the end of KimiLinearForCausalLM.
#     load_weights (v4 has no model-level process_weights_after_loading);
#     AutoWeightsLoader delegates to child load_weights, so this runs on
#     both the plain and the ForConditionalGeneration paths.
#   * num_ubatches honors parallel_config.enable_dbo (present in v4).
#
# Anchor provenance: ALL anchors verified against the fork clone at
# 881ac39a4 (== the image tree). Idempotent; missing anchors skip with
# notes; py_compile per touched file.

import os
import py_compile
import sys

SCRIPT_NAME = "patch_568_riders"

VLLM_ROOT = os.environ.get("VLLM_ROOT", "/opt/kimi-k3/vllm/vllm")
MLA_ATT_PATH = os.path.join(
    VLLM_ROOT, "model_executor", "layers", "attention", "mla_attention.py"
)
MODEL_PATH = os.path.join(VLLM_ROOT, "models", "kimi_k3", "nvidia", "model.py")
MLA_MODEL_PATH = os.path.join(VLLM_ROOT, "models", "kimi_k3", "nvidia", "mla.py")


def apply_hunks(path, hunks, label):
    try:
        with open(path) as f:
            src = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {path} not found", file=sys.stderr)
        return None
    done = set()
    changed = False
    for name, anchor, replacement, present in hunks:
        if present in src:
            print(f"  SKIP (already patched) [{label}]: {name}")
            done.add(name)
            continue
        count = src.count(anchor)
        if count != 1:
            print(
                f"  NOTE (not applicable) [{label}]: {name}: anchor found "
                f"{count}x (expected 1); hunk skipped — review before baking"
            )
            continue
        src = src.replace(anchor, replacement, 1)
        print(f"  APPLIED [{label}]: {name}")
        done.add(name)
        changed = True
    if changed:
        with open(path, "w") as f:
            f.write(src)
        py_compile.compile(path, doraise=True)
        print(f"[{SCRIPT_NAME}] {label} py_compile OK")
    return done


# ======================================================================
# (a) ee429fb01 — presence check (expected: already in-tree)
# ======================================================================
def rider_a():
    print(f"[{SCRIPT_NAME}] (a) ee429fb01 bf16 kv_b_proj cast — presence check")
    try:
        with open(MLA_ATT_PATH) as f:
            att = f.read()
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] ERROR: {MLA_ATT_PATH} not found", file=sys.stderr)
        return 1
    has_helper = "def _get_kv_b_proj_input_dtype(" in att
    has_cast = (
        "kv_b_proj_input_dtype = _get_kv_b_proj_input_dtype(\n"
        "            self.kv_b_proj, use_fp8_prefill\n"
        "        )" in att
        and "kv_c_normed = kv_c_normed.to(kv_b_proj_input_dtype)" in att
    )
    if has_helper and has_cast:
        print(
            "  SKIP (already in-tree): the per-chunk kv_b_proj input cast is "
            "present in MLACommonBaseImpl._compute_prefill_context — the "
            "ee429fb01 fix predates the v4 tree's impl-side chunk loop."
        )
    else:
        print(
            "  *** LOUD NOTE: the kv_b_proj input-dtype cast was NOT found in "
            f"{MLA_ATT_PATH} (helper={has_helper}, cast={has_cast}). A stock "
            "bf16 kv_b_proj checkpoint with an fp8 KV cache would feed the "
            "gathered latent to kv_b_proj uncast. Apply ee429fb01's cast by "
            "hand. ***"
        )
    # the assert the commit removes should NOT exist in v4's model-side mla.py
    try:
        with open(MLA_MODEL_PATH) as f:
            mla = f.read()
        if "needs a kv_b_proj that" in mla:
            print(
                "  *** LOUD NOTE: the later-tree fp8 kv_b_proj input assert "
                "exists in v4's mla.py process_weights_after_loading — "
                "ee429fb01 removes it; apply by hand. ***"
            )
    except FileNotFoundError:
        print(f"[{SCRIPT_NAME}] NOTE: {MLA_MODEL_PATH} not found; assert check skipped")
    return 0


# ======================================================================
# (b) 8a89a1d2d — retained MLA context-projection workspace (adapted)
# ======================================================================

# mla_attention.py: import (dbo ubatch id)
ATT_IMPORT_ANCHOR = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    AttentionSpec,\n"
    "    KVCacheSpec,\n"
    "    MLAAttentionSpec,\n"
    "    get_kv_quant_mode,\n"
    ")\n"
)
ATT_IMPORT_REPLACEMENT = (
    "from vllm.v1.kv_cache_interface import (\n"
    "    AttentionSpec,\n"
    "    KVCacheSpec,\n"
    "    MLAAttentionSpec,\n"
    "    get_kv_quant_mode,\n"
    ")\n"
    "from vllm.v1.worker.ubatching import (  # V4PLUS (#568 8a89a1d2d)\n"
    "    dbo_current_ubatch_id,\n"
    ")\n"
)
ATT_IMPORT_PRESENT = "from vllm.v1.worker.ubatching import (  # V4PLUS (#568 8a89a1d2d)\n"

# mla_attention.py: the workspace class (verbatim from 8a89a1d2d), inserted
# before MLACommonBaseImpl.
ATT_CLASS_ANCHOR = "class MLACommonBaseImpl(MLAAttentionImpl[A], Generic[A]):\n"
ATT_CLASS_REPLACEMENT = (
    "class KimiK3PrefillProjectionWorkspace:  # V4PLUS (#568 8a89a1d2d)\n"
    '    """Retained output storage for large dense context projections."""\n'
    "\n"
    "    def __init__(self, num_ubatches: int, min_tokens: int) -> None:\n"
    "        if num_ubatches < 1:\n"
    '            raise ValueError("num_ubatches must be positive")\n'
    "        if min_tokens < 0:\n"
    '            raise ValueError("min_tokens must be non-negative")\n'
    "        self.num_ubatches = num_ubatches\n"
    "        self.min_tokens = min_tokens\n"
    "        self._buffer: torch.Tensor | None = None\n"
    "\n"
    "    def reserve(\n"
    "        self,\n"
    "        max_tokens: int,\n"
    "        output_size: int,\n"
    "        dtype: torch.dtype,\n"
    "        device: torch.device,\n"
    "    ) -> None:\n"
    "        if max_tokens < self.min_tokens:\n"
    "            raise ValueError(\n"
    '                f"max_tokens ({max_tokens}) must be at least min_tokens "\n'
    '                f"({self.min_tokens})"\n'
    "            )\n"
    "        self._buffer = torch.empty(\n"
    "            (self.num_ubatches, max_tokens, output_size),\n"
    "            dtype=dtype,\n"
    "            device=device,\n"
    "        )\n"
    "\n"
    "    @property\n"
    "    def nbytes(self) -> int:\n"
    "        buffer = self._buffer\n"
    "        return 0 if buffer is None else buffer.numel() * buffer.element_size()\n"
    "\n"
    "    def get(\n"
    "        self,\n"
    "        num_tokens: int,\n"
    "        output_size: int,\n"
    "        dtype: torch.dtype,\n"
    "        device: torch.device,\n"
    "    ) -> torch.Tensor | None:\n"
    "        if num_tokens < self.min_tokens:\n"
    "            return None\n"
    "        buffer = self._buffer\n"
    "        if buffer is None:\n"
    '            raise RuntimeError("Kimi-K3 prefill projection workspace is not reserved")\n'
    "        if num_tokens > buffer.shape[1]:\n"
    "            raise ValueError(\n"
    '                f"context projection needs {num_tokens} rows, but the retained "\n'
    '                f"workspace has {buffer.shape[1]}"\n'
    "            )\n"
    "        if output_size != buffer.shape[2]:\n"
    "            raise ValueError(\n"
    '                f"context projection needs {output_size} columns, but the retained "\n'
    '                f"workspace has {buffer.shape[2]}"\n'
    "            )\n"
    "        if dtype != buffer.dtype or device != buffer.device:\n"
    "            raise ValueError(\n"
    "                \"context projection input and retained workspace must have the \"\n"
    '                "same dtype and device"\n'
    "            )\n"
    "        ubatch_id = dbo_current_ubatch_id()\n"
    "        if ubatch_id >= self.num_ubatches:\n"
    "            raise RuntimeError(\n"
    '                f"ubatch {ubatch_id} has no Kimi-K3 prefill projection workspace; "\n'
    '                f"configured slots: {self.num_ubatches}"\n'
    "            )\n"
    "        return buffer[ubatch_id, :num_tokens]\n"
    "\n"
    "\n"
    "class MLACommonBaseImpl(MLAAttentionImpl[A], Generic[A]):\n"
)
ATT_CLASS_PRESENT = "class KimiK3PrefillProjectionWorkspace:  # V4PLUS (#568 8a89a1d2d)\n"

# mla_attention.py: impl attr (anchor disambiguated to the BaseImpl __init__
# — the other self.kv_b_proj assignment is followed by dcp_q_replicate).
ATT_ATTR_ANCHOR = (
    "        self.v_head_dim = v_head_dim\n"
    "        self.kv_b_proj = kv_b_proj\n"
    "\n"
    "    def _concat_k_nope_k_pe(\n"
)
ATT_ATTR_REPLACEMENT = (
    "        self.v_head_dim = v_head_dim\n"
    "        self.kv_b_proj = kv_b_proj\n"
    "        self._prefill_projection_workspace = None  # V4PLUS (#568 8a89a1d2d)\n"
    "\n"
    "    def _concat_k_nope_k_pe(\n"
)
ATT_ATTR_PRESENT = "        self._prefill_projection_workspace = None  # V4PLUS (#568 8a89a1d2d)\n"

# mla_attention.py: _project_context method (adapted from the commit's
# nested project_context; num_heads instead of num_local_heads), inserted
# before _compute_prefill_context.
ATT_METHOD_ANCHOR = (
    "    def _compute_prefill_context(\n"
    "        self,\n"
    "        q: torch.Tensor,\n"
    "        kv_c_and_k_pe_cache: torch.Tensor,\n"
    "        attn_metadata: MLACommonMetadata,\n"
    "        k_scale: torch.Tensor,\n"
    "    ):\n"
)
ATT_METHOD_REPLACEMENT = (
    "    def _project_context(self, kv_c_normed: torch.Tensor) -> torch.Tensor:\n"
    "        \"\"\"kv_b_proj with an optional retained output workspace.\n"
    "\n"
    "        V4PLUS (#568 8a89a1d2d, adapted): routes large dense context\n"
    "        projections through the model's retained workspace (assigned by\n"
    "        reserve_mla_prefill_projection_workspace) so the per-chunk\n"
    "        kv_b_proj output is not freshly allocated every chunk.\n"
    "        \"\"\"\n"
    "        workspace = self._prefill_projection_workspace\n"
    '        weight = getattr(self.kv_b_proj, "weight", None)\n'
    "        if workspace is None or not isinstance(weight, torch.Tensor):\n"
    "            return self.kv_b_proj(kv_c_normed)[0]\n"
    "        rows = kv_c_normed.numel() // self.kv_lora_rank\n"
    "        projection = workspace.get(\n"
    "            rows,\n"
    "            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),\n"
    "            kv_c_normed.dtype,\n"
    "            kv_c_normed.device,\n"
    "        )\n"
    "        if projection is None:\n"
    "            return self.kv_b_proj(kv_c_normed)[0]\n"
    "        if not isinstance(self.kv_b_proj.quant_method, UnquantizedLinearMethod):\n"
    "            raise RuntimeError(\n"
    '                "Kimi-K3 retained context projection requires an "\n'
    '                "unquantized kv_b_proj"\n'
    "            )\n"
    "        if self.kv_b_proj.bias is not None or self.kv_b_proj.gather_output:\n"
    "            raise RuntimeError(\n"
    '                "Kimi-K3 retained context projection requires a local, "\n'
    '                "bias-free kv_b_proj"\n'
    "            )\n"
    "        torch.mm(\n"
    "            kv_c_normed.reshape(rows, self.kv_lora_rank),\n"
    "            weight.t(),\n"
    "            out=projection,\n"
    "        )\n"
    "        return projection\n"
    "\n"
    "    def _compute_prefill_context(\n"
    "        self,\n"
    "        q: torch.Tensor,\n"
    "        kv_c_and_k_pe_cache: torch.Tensor,\n"
    "        attn_metadata: MLACommonMetadata,\n"
    "        k_scale: torch.Tensor,\n"
    "    ):\n"
)
ATT_METHOD_PRESENT = "    def _project_context(self, kv_c_normed: torch.Tensor) -> torch.Tensor:\n"

# mla_attention.py: swap the chunk-loop kv_b_proj call (the non-DCP loop —
# disambiguated by the k_pe line that only precedes it).
ATT_CALL_ANCHOR = (
    "            k_pe = workspace[:toks][..., self.kv_lora_rank :].unsqueeze(1)\n"
    "            kv_nope = self.kv_b_proj(kv_c_normed)[0].view(\n"
    "                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim\n"
    "            )\n"
)
ATT_CALL_REPLACEMENT = (
    "            k_pe = workspace[:toks][..., self.kv_lora_rank :].unsqueeze(1)\n"
    "            kv_nope = self._project_context(kv_c_normed).view(  # V4PLUS (#568 8a89a1d2d)\n"
    "                -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim\n"
    "            )\n"
)
ATT_CALL_PRESENT = "            kv_nope = self._project_context(kv_c_normed).view(  # V4PLUS (#568 8a89a1d2d)\n"

# mla_attention.py: UnquantizedLinearMethod import (extend the linear import).
ATT_LIN_IMPORT_ANCHOR = (
    "from vllm.model_executor.layers.linear import (\n"
    "    ColumnParallelLinear,\n"
    ")\n"
)
ATT_LIN_IMPORT_REPLACEMENT = (
    "from vllm.model_executor.layers.linear import (\n"
    "    ColumnParallelLinear,\n"
    "    UnquantizedLinearMethod,  # V4PLUS (#568 8a89a1d2d)\n"
    ")\n"
)
ATT_LIN_IMPORT_PRESENT = "    UnquantizedLinearMethod,  # V4PLUS (#568 8a89a1d2d)\n"

# model.py: import block.
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
MO_IMPORT_PRESENT = "    KimiK3PrefillProjectionWorkspace,\n"

# model.py: stash vllm_config + construct the workspace in the model __init__.
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
MO_INIT_PRESENT = "        self._vllm_config = vllm_config  # V4PLUS (#568 8a89a1d2d)\n"

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
MO_WS_PRESENT = "        self._mla_prefill_projection_workspace = KimiK3PrefillProjectionWorkspace(\n"

# model.py: the reserve method, inserted before the MODEL class's
# make_empty_intermediate_tensors (disambiguated from the CausalLM wrapper's
# delegating copy at line ~1815 by the __init__ tail above it).
MO_RESERVE_ANCHOR = (
    '        world_size = get_tensor_model_parallel_world_size()\n'
    '        assert config.num_attention_heads % world_size == 0, (\n'
    '            "num_attention_heads must be divisible by world_size"\n'
    "        )\n"
    "\n"
    "    def make_empty_intermediate_tensors(\n"
    "        self,\n"
    "        batch_size: int,\n"
    "        dtype: torch.dtype,\n"
    "        device: torch.device,\n"
    "    ) -> IntermediateTensors:\n"
)
MO_RESERVE_REPLACEMENT = (
    '        world_size = get_tensor_model_parallel_world_size()\n'
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
    "        self,\n"
    "        batch_size: int,\n"
    "        dtype: torch.dtype,\n"
    "        device: torch.device,\n"
    "    ) -> IntermediateTensors:\n"
)
MO_RESERVE_PRESENT = "    def reserve_mla_prefill_projection_workspace(self) -> None:\n"

# model.py: trigger from KimiLinearForCausalLM.load_weights.
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
MO_CALL_PRESENT = "        self.model.reserve_mla_prefill_projection_workspace()\n"


def rider_b():
    print(f"[{SCRIPT_NAME}] (b) 8a89a1d2d retained MLA context-projection workspace")
    done = apply_hunks(
        MLA_ATT_PATH,
        [
            ("ubatching import", ATT_IMPORT_ANCHOR, ATT_IMPORT_REPLACEMENT, ATT_IMPORT_PRESENT),
            ("UnquantizedLinearMethod import", ATT_LIN_IMPORT_ANCHOR, ATT_LIN_IMPORT_REPLACEMENT, ATT_LIN_IMPORT_PRESENT),
            ("workspace class", ATT_CLASS_ANCHOR, ATT_CLASS_REPLACEMENT, ATT_CLASS_PRESENT),
            ("impl workspace attr", ATT_ATTR_ANCHOR, ATT_ATTR_REPLACEMENT, ATT_ATTR_PRESENT),
            ("_project_context method", ATT_METHOD_ANCHOR, ATT_METHOD_REPLACEMENT, ATT_METHOD_PRESENT),
            ("chunk-loop call swap", ATT_CALL_ANCHOR, ATT_CALL_REPLACEMENT, ATT_CALL_PRESENT),
        ],
        "mla_attention.py",
    )
    if done is None:
        return 1
    att_ok = ATT_CLASS_PRESENT in open(MLA_ATT_PATH).read() and ATT_METHOD_PRESENT in open(
        MLA_ATT_PATH
    ).read()

    mo_hunks = []
    if att_ok:
        mo_hunks = [
            ("mla_attention imports", MO_IMPORT_ANCHOR, MO_IMPORT_REPLACEMENT, MO_IMPORT_PRESENT),
            ("model stashes vllm_config", MO_INIT_ANCHOR, MO_INIT_REPLACEMENT, MO_INIT_PRESENT),
            ("workspace construction", MO_WS_ANCHOR, MO_WS_REPLACEMENT, MO_WS_PRESENT),
            ("reserve method", MO_RESERVE_ANCHOR, MO_RESERVE_REPLACEMENT, MO_RESERVE_PRESENT),
        ]
    else:
        print(
            "  NOTE (not applicable) [model.py]: workspace hunks skipped — "
            "mla_attention.py class/method hunks did not apply (the model "
            "would import a nonexistent symbol)"
        )
    done = apply_hunks(MODEL_PATH, mo_hunks, "model.py")
    if done is None:
        return 1
    with open(MODEL_PATH) as f:
        mo_src = f.read()
    if (
        MO_WS_PRESENT in mo_src
        and MO_RESERVE_PRESENT in mo_src
        and "UnquantizedLinearMethod" in mo_src
    ):
        apply_hunks(
            MODEL_PATH,
            [("load_weights reserve call", MO_CALL_ANCHOR, MO_CALL_REPLACEMENT, MO_CALL_PRESENT)],
            "model.py",
        )
    else:
        print(
            "  NOTE (not applicable) [model.py]: load_weights reserve call "
            "skipped — workspace construction or reserve method missing"
        )
    return 0


def main() -> int:
    rc = rider_a()
    if rc:
        return rc
    rc = rider_b()
    if rc:
        return rc
    print(
        f"[{SCRIPT_NAME}] NOTE: verify at dry-run that the 'Kimi-K3 retained "
        "%.2f MiB/rank' log line appears at startup (it proves the reserve "
        "call runs on the actual serving load path) and that "
        "VLLM_BATCH_INVARIANT is off."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
