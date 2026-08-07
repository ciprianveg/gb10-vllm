#!/usr/bin/env python3
"""Register the local GLM-5V model/config in an exact pinned vLLM install."""

from pathlib import Path


VLLM_ROOT = Path("/usr/local/lib/python3.12/dist-packages/vllm")


def replace_once(relative_path: str, old: str, new: str) -> None:
    path = VLLM_ROOT / relative_path
    source = path.read_text()
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"{relative_path}: expected one registration anchor, found {count}"
        )
    path.write_text(source.replace(old, new, 1))


replace_once(
    "transformers_utils/config.py",
    '    kimi_k25="KimiK25Config",\n',
    '    kimi_k25="KimiK25Config",\n'
    '    glm5v="Glm5vConfig",\n',
)

replace_once(
    "transformers_utils/configs/__init__.py",
    '    "KimiK25Config": "vllm.transformers_utils.configs.kimi_k25",\n',
    '    "KimiK25Config": "vllm.transformers_utils.configs.kimi_k25",\n'
    '    "Glm5vConfig": "vllm.transformers_utils.configs.glm5v",\n',
)

replace_once(
    "transformers_utils/configs/__init__.py",
    '    "KimiK25Config",\n',
    '    "KimiK25Config",\n'
    '    "Glm5vConfig",\n',
)

replace_once(
    "model_executor/models/registry.py",
    '    "KimiK25ForConditionalGeneration": ("kimi_k25", '
    '"KimiK25ForConditionalGeneration"),\n',
    '    "KimiK25ForConditionalGeneration": ("kimi_k25", '
    '"KimiK25ForConditionalGeneration"),\n'
    '    "Glm5vForConditionalGeneration": ("glm5v", '
    '"Glm5vForConditionalGeneration"),\n',
)

replace_once(
    "v1/spec_decode/llm_base_proposer.py",
    '            elif self.get_model_name(target_model) == '
    '"KimiK25ForConditionalGeneration":\n'
    "                self.model.config.image_token_index = (\n"
    "                    target_model.config.media_placeholder_token_id\n"
    "                )\n",
    '            elif self.get_model_name(target_model) in (\n'
    '                "KimiK25ForConditionalGeneration",\n'
    '                "Glm5vForConditionalGeneration",\n'
    "            ):\n"
    "                self.model.config.image_token_index = (\n"
    "                    target_model.config.media_placeholder_token_id\n"
    "                )\n",
)

replace_once(
    "v1/spec_decode/llm_base_proposer.py",
    "        if self.supports_mm_inputs:\n"
    "            # Even if the target model is multimodal, we can also use\n",
    "        if (\n"
    '            self.get_model_name(target_model) == "Glm5vForConditionalGeneration"\n'
    '            and self.method == "mtp"\n'
    "        ):\n"
    "            # The GLM MTP head is text-only. Its permissive embed_input_ids\n"
    "            # signature otherwise makes the probe below select the\n"
    "            # multimodal inputs_embeds path, which yields zero acceptance.\n"
    "            self.supports_mm_inputs = False\n"
    "\n"
    "        if self.supports_mm_inputs:\n"
    "            # Even if the target model is multimodal, we can also use\n",
)

print("Registered Glm5v config, model, and multimodal speculative-decoding hook")
