# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2 with Baseten's MoonViT/PatchMerger vision adapter.

This is intentionally a thin specialization of vLLM's Kimi-K2.5 multimodal
implementation.  The language model is the existing GlmMoeDsaForCausalLM and
is initialized with the original root prefix, preserving QuantTrio's exact
``model.layers.*`` compressed-tensors target matching.
"""

from collections.abc import Mapping

from torch import nn

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.kimi_k25 import (
    KimiK25DummyInputsBuilder,
    KimiK25ForConditionalGeneration,
    KimiK25MultiModalProcessor,
    KimiK25ProcessingInfo,
    MaxImageTokenMeta,
)
from vllm.model_executor.models.kimi_k25_vit import (
    KimiK25MultiModalProjector,
    MoonViT3dPretrainedModel,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import VisionChunkImage
from vllm.multimodal.processing import BaseProcessingInfo, InputProcessingContext
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.glm5v import Glm5vConfig
from vllm.transformers_utils.processor import cached_get_image_processor
from vllm.transformers_utils.processors.kimi_k25 import KimiK25Processor

from .utils import WeightsMapper, init_vllm_registered_model, maybe_prefix

logger = init_logger(__name__)


class Glm5vProcessingInfo(KimiK25ProcessingInfo):
    """GLM-specific token resolution with Kimi's image preprocessing."""

    def __init__(self, ctx: InputProcessingContext) -> None:
        # Skip KimiK25ProcessingInfo.__init__: it resolves <|media_pad|>,
        # whereas GLM-5V expands GLM's existing <|image|> token.
        BaseProcessingInfo.__init__(self, ctx)

        self.hf_config = hf_config = self.get_hf_config()
        tokenizer = self.get_tokenizer()
        image_processor = cached_get_image_processor(
            self.ctx.model_config.model,
            revision=self.ctx.model_config.revision,
            trust_remote_code=self.ctx.model_config.trust_remote_code,
        )

        config_token_id = hf_config.media_placeholder_token_id
        resolved_token_id = tokenizer.convert_tokens_to_ids("<|image|>")
        is_valid_resolved = isinstance(resolved_token_id, int) and (
            tokenizer.unk_token_id is None
            or resolved_token_id != tokenizer.unk_token_id
        )
        if not is_valid_resolved:
            raise ValueError("GLM-5V tokenizer does not contain <|image|>")
        if resolved_token_id != config_token_id:
            raise ValueError(
                "GLM-5V image-token mismatch: config has "
                f"{config_token_id}, tokenizer has {resolved_token_id}"
            )

        self.media_token_id = resolved_token_id
        self.media_token = "<|image|>"
        self.image_processor = image_processor
        self.hf_processor = KimiK25Processor(
            tokenizer=tokenizer,
            image_processor=image_processor,
            media_token_id=resolved_token_id,
        )
        self.media_tokens_calculator = image_processor.media_tokens_calculator

    def get_hf_config(self) -> Glm5vConfig:
        return self.ctx.get_hf_config(Glm5vConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"vision_chunk": None}


class Glm5vDummyInputsBuilder(KimiK25DummyInputsBuilder):
    """Profile the supported image path, not the larger video path."""

    def get_dummy_mm_items(self):
        return [
            VisionChunkImage(
                type="image",
                image=self._get_dummy_images(
                    height=MaxImageTokenMeta.height,
                    width=MaxImageTokenMeta.width,
                    num_images=1,
                )[0],
            )
        ]


class Glm5vMultiModalProcessor(KimiK25MultiModalProcessor):
    pass


@MULTIMODAL_REGISTRY.register_processor(
    Glm5vMultiModalProcessor,
    info=Glm5vProcessingInfo,
    dummy_inputs=Glm5vDummyInputsBuilder,
)
class Glm5vForConditionalGeneration(KimiK25ForConditionalGeneration):
    """QuantTrio GLM-5.2 language model plus MoonViT/PatchMerger."""

    @property
    def lm_head(self) -> nn.Module:
        """Expose the nested head so MTP shares the target output projection."""
        return self.language_model.lm_head

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            # QuantTrio keeps the native text-only checkpoint names.
            "model.": "language_model.model.",
            "lm_head.": "language_model.lm_head.",
            # Compatibility with alternate Kimi-style text exports.
            "language_model.layers.": "language_model.model.layers.",
            # Compatibility with older sequential projector exports.
            "mm_projector.proj.0": "mm_projector.linear_1",
            "mm_projector.proj.2": "mm_projector.linear_2",
        }
    )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return "<|begin_of_image|><|image|><|end_of_image|>"
        if modality == "video":
            return "<|glm5v_video_placeholder|>"
        raise ValueError(f"Unsupported modality: {modality}")

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Do not call KimiK25ForConditionalGeneration.__init__: it selects a
        # DeepSeek text class and adds a language_model prefix to quantization
        # matching.  QuantTrio's mixed INT4/INT8 config requires the native
        # model.layers.* prefixes.
        nn.Module.__init__(self)

        model_config = vllm_config.model_config
        config: Glm5vConfig = model_config.hf_config
        self.config = config
        quant_config = vllm_config.quant_config

        self.use_data_parallel = (
            model_config.multimodal_config.mm_encoder_tp_mode == "data"
        )
        self.hidden_size = config.text_config.hidden_size
        self.device = current_platform.current_device()

        with self._mark_tower_model(vllm_config, "vision_chunk"):
            vision_quant_config = self._maybe_ignore_quant_config(quant_config)
            self.vision_tower = MoonViT3dPretrainedModel(
                config.vision_config,
                quant_config=vision_quant_config,
                prefix=maybe_prefix(prefix, "vision_tower"),
            )
            if vision_quant_config is not None:
                self.vision_tower = self.vision_tower.to(device=self.device)
            else:
                self.vision_tower = self.vision_tower.to(
                    device=self.device, dtype=model_config.dtype
                )

            self.mm_projector = KimiK25MultiModalProjector(
                config=config.vision_config,
                use_data_parallel=self.use_data_parallel,
                quant_config=vision_quant_config,
                prefix=maybe_prefix(prefix, "mm_projector"),
            )
            self.mm_projector = self.mm_projector.to(
                device=self.device, dtype=model_config.dtype
            )

        self.quant_config = quant_config
        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                # Critical QuantTrio invariant: compressed-tensors sees
                # model.layers.*, not language_model.model.layers.*.
                prefix=prefix,
                architectures=["GlmMoeDsaForCausalLM"],
            )

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )
        self.media_placeholder = config.media_placeholder_token_id
