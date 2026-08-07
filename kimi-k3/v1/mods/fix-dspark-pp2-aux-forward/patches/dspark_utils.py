# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import glob
import json
import os
from collections.abc import Iterable

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from vllm.config import VllmConfig, replace
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf,
    maybe_download_from_modelscope,
)
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _create_draft_vllm_config,
    _should_share,
    get_target_lm_head,
)

logger = init_logger(__name__)


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal

    backend = speculative_config.attention_backend
    if backend is None:
        # DSpark needs a non-causal-capable backend for its drafts;
        # auto-selection may pick one that downgrades the spec-decode
        # cudagraph to PIECEWISE.
        backend = AttentionBackendEnum.FLASH_ATTN
    # Build from the draft parallel/model config, not the target config.  In
    # particular, an external DSpark draft is DCP1 by default even when the
    # target uses DCP: every TP rank keeps the complete five-layer draft KV
    # sequence while the target KV remains DCP-sharded.  Reusing the target
    # VllmConfig here silently made the draft DCP8 as well and corrupted its
    # context, collapsing first-token acceptance from ~75% to ~11%.
    draft_vllm_config = _create_draft_vllm_config(vllm_config)
    draft_vllm_config = replace(
        draft_vllm_config,
        attention_config=replace(
            draft_vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=backend,
        ),
    )

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model

    target_embed = getattr(target_inner, "embed_tokens", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    # Last PP stage only has PPMissingLayer for target embed; load real weights.
    if isinstance(target_embed, PPMissingLayer):
        target_embed = _load_target_embed_tokens_for_pp(vllm_config, target_model)
    if target_embed is not None and _should_share(
        draft_model, "has_own_embed_tokens", draft_embed, target_embed
    ):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    if target_lm_head is not None and _should_share(
        draft_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    # Opt-in rowwise-fp8 draft head (VLLM_DSPARK_FP8_DRAFT_HEAD). Must run
    # after the lm_head aliasing above and BEFORE CUDA graph capture: the
    # draft step is captured whole, so the fp8 copy is materialized eagerly.
    maybe_init_fp8_draft_head = getattr(draft_model, "maybe_init_fp8_draft_head", None)
    if maybe_init_fp8_draft_head is not None:
        maybe_init_fp8_draft_head()

    return draft_model


def _checkpoint_to_param_names(
    names: Iterable[str], target_model: nn.Module
) -> dict[str, str]:
    """Map checkpoint tensor names to the parameter names they load into.

    Keyed by parameter name so a caller that recognises a parameter can still
    read the tensor under its original name. Checkpoints do not have to use
    vLLM's naming -- DeepSeek-V4 stores the embedding as ``embed.weight`` -- and
    the model's own ``hf_to_vllm_mapper`` is the translation it applies at load
    time, so searching for a parameter name without it finds nothing.
    """
    identity = {name: name for name in names}
    mapper = getattr(target_model, "hf_to_vllm_mapper", None)
    if mapper is None:
        return identity
    return mapper.apply_dict(identity)


def _load_target_embed_tokens_for_pp(
    vllm_config: VllmConfig, target_model: nn.Module
) -> nn.Module:
    """Load target embed_tokens on a non-first PP stage for the DSpark drafter."""
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )
    from vllm.platforms import current_platform

    model_config = vllm_config.model_config
    load_config = vllm_config.load_config
    text_config = model_config.hf_text_config

    model_name_or_path = model_config.model_weights or model_config.model
    model_name_or_path = (
        maybe_download_from_modelscope(model_name_or_path, model_config.revision)
        or model_name_or_path
    )
    if os.path.isdir(model_name_or_path):
        model_dir = model_name_or_path
    else:
        model_dir = download_weights_from_hf(
            model_name_or_path,
            load_config.download_dir,
            # Index included so the lookup below can go straight to the shard
            # holding the embedding instead of opening every shard in turn.
            allow_patterns=["*.safetensors", SAFE_WEIGHTS_INDEX_NAME],
            revision=model_config.revision,
            ignore_patterns=load_config.ignore_patterns,
        )

    def find_embed(names: Iterable[str]) -> str | None:
        by_param = _checkpoint_to_param_names(names, target_model)
        for param_name, ckpt_name in by_param.items():
            if param_name.endswith("embed_tokens.weight"):
                return ckpt_name
        return None

    key = None
    shard_path = None
    index_path = os.path.join(model_dir, SAFE_WEIGHTS_INDEX_NAME)
    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        key = find_embed(weight_map)
        if key is not None:
            shard_path = os.path.join(model_dir, weight_map[key])
    else:
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safe_open(path, framework="pt") as f:
                key = find_embed(f.keys())
            if key is not None:
                shard_path = path
                break
    if key is None:
        raise RuntimeError(
            "DSpark under pipeline parallelism needs the target's embed_tokens "
            f"on the last stage, but no weight mapping to *embed_tokens.weight "
            f"was found in {model_dir}"
        )

    with torch.device(current_platform.current_device()):
        embed = VocabParallelEmbedding(
            text_config.vocab_size,
            text_config.hidden_size,
            params_dtype=model_config.dtype,
            prefix="dspark_pp.embed_tokens",
        )
    with safe_open(shard_path, framework="pt") as f:
        embed.weight_loader(embed.weight, f.get_tensor(key))
    logger.info(
        "Loaded draft embed_tokens %s from %s (key %s) for PP",
        tuple(embed.weight.shape),
        os.path.basename(shard_path),
        key,
    )
    return embed
