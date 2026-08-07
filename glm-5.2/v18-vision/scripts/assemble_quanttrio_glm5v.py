#!/usr/bin/env python3
"""Assemble a zero-copy QuantTrio + Baseten GLM-5V checkpoint directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


BASE_TEN_REVISION = "f6eab6117386a0c69152fdf272dc65bfd0254f9f"
VISION_FILES = ("vision_tower.safetensors", "mm_projector.safetensors")
REMOTE_CODE_FILES = (
    "preprocessor_config.json",
    "kimi_k25_processor.py",
    "kimi_k25_vision_processing.py",
    "media_utils.py",
    "configuration_glm5v.py",
)
TEXT_SUPPORT_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "configuration.json",
    ".mdl",
    ".msc",
    ".mv",
)


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def relative_symlink(source: Path, destination: Path) -> None:
    source = source.resolve()
    relative_target = os.path.relpath(source, start=destination.parent.resolve())
    temporary = destination.with_name(destination.name + ".tmp-link")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(relative_target)
    temporary.replace(destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(text_dir: Path, vision_dir: Path) -> dict:
    text_config = load_json(text_dir / "config.json")
    vision_config = load_json(vision_dir / "config.json")

    if text_config.get("model_type") != "glm_moe_dsa":
        raise ValueError("text checkpoint is not glm_moe_dsa")
    if text_config.get("hidden_size") != 6144:
        raise ValueError("QuantTrio hidden_size is not 6144")
    if text_config.get("quantization_config", {}).get("quant_method") != (
        "compressed-tensors"
    ):
        raise ValueError("text checkpoint is not compressed-tensors QuantTrio")

    merged = dict(vision_config)
    merged.pop("auto_map", None)
    merged["architectures"] = ["Glm5vForConditionalGeneration"]
    merged["model_type"] = "glm5v"
    merged["text_config"] = text_config
    merged["quantization_config"] = text_config["quantization_config"]
    merged["media_placeholder_token_id"] = 154854
    merged["language_only"] = False
    merged["encoder_only"] = False

    output_width = merged["vision_config"]["text_hidden_size"]
    if output_width != text_config["hidden_size"]:
        raise ValueError(
            f"projector output {output_width} != text width "
            f"{text_config['hidden_size']}"
        )
    return merged


def build_index(text_dir: Path, vision_dir: Path) -> tuple[dict, int, int]:
    text_index = load_json(text_dir / "model.safetensors.index.json")
    vision_index = load_json(vision_dir / "model.safetensors.index.json")

    weight_map = dict(text_index["weight_map"])
    text_count = len(weight_map)
    vision_entries = {
        name: filename
        for name, filename in vision_index["weight_map"].items()
        if filename in VISION_FILES
    }
    if len(vision_entries) != 335:
        raise ValueError(
            f"expected 335 vision/projector tensors, found {len(vision_entries)}"
        )

    overlap = set(weight_map).intersection(vision_entries)
    if overlap:
        raise ValueError(f"vision/text tensor collision: {sorted(overlap)[:5]}")
    weight_map.update(vision_entries)

    total_size = int(text_index.get("metadata", {}).get("total_size", 0))
    total_size += sum((vision_dir / name).stat().st_size for name in VISION_FILES)
    return (
        {"metadata": {"total_size": total_size}, "weight_map": weight_map},
        text_count,
        len(vision_entries),
    )


def build_chat_template(vision_dir: Path) -> str:
    template = (vision_dir / "chat_template.jinja").read_text()
    baseten_default = (
        "{%- set effective_reasoning_effort = 'high' if reasoning_effort is "
        "defined and reasoning_effort == 'high' else 'max' -%}"
    )
    quanttrio_default = (
        "{# {%- set effective_reasoning_effort = 'high' if reasoning_effort "
        "is defined and reasoning_effort == 'high' else 'max' -%} #}\n"
        "{%- set effective_reasoning_effort = reasoning_effort if "
        "reasoning_effort is defined and reasoning_effort is not none else "
        "'medium-high' -%}"
    )
    if template.count(baseten_default) != 1:
        raise ValueError("unexpected Baseten chat-template reasoning stanza")
    template = template.replace(baseten_default, quanttrio_default, 1)
    required = "<|begin_of_image|><|image|><|end_of_image|>"
    if template.count(required) != 1:
        raise ValueError("unexpected GLM-5V image placeholder in chat template")
    return template


def build_preprocessor_config(vision_dir: Path, patch_limit: int) -> dict:
    config = load_json(vision_dir / "preprocessor_config.json")
    media_config = config.get("media_proc_cfg", {})
    if media_config.get("in_patch_limit") != 16384:
        raise ValueError("unexpected Baseten image patch limit")
    if not 1 <= patch_limit <= 16384:
        raise ValueError("image patch limit must be in [1, 16384]")
    media_config["in_patch_limit"] = patch_limit
    media_config["in_patch_limit_each_frame"] = patch_limit
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-dir", type=Path, required=True)
    parser.add_argument("--vision-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-patch-limit", type=int, default=16384)
    args = parser.parse_args()

    text_dir = args.text_dir.resolve()
    vision_dir = args.vision_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(text_dir, vision_dir)
    index, text_count, vision_count = build_index(text_dir, vision_dir)
    chat_template = build_chat_template(vision_dir)
    preprocessor_config = build_preprocessor_config(
        vision_dir, args.image_patch_limit
    )

    shard_names = sorted(set(index["weight_map"].values()) - set(VISION_FILES))
    for shard_name in shard_names:
        source = text_dir / shard_name
        if not source.is_file():
            raise FileNotFoundError(source)
        relative_symlink(source, output_dir / shard_name)

    for filename in VISION_FILES:
        source = vision_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        relative_symlink(source, output_dir / filename)

    for filename in TEXT_SUPPORT_FILES:
        source = text_dir / filename
        if source.is_file():
            relative_symlink(source, output_dir / filename)

    for filename in REMOTE_CODE_FILES:
        source = vision_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        if filename == "preprocessor_config.json":
            continue
        shutil.copy2(source, output_dir / filename)

    write_json(output_dir / "preprocessor_config.json", preprocessor_config)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "model.safetensors.index.json", index)
    (output_dir / "chat_template.jinja").write_text(chat_template)

    manifest = {
        "format": "quanttrio-glm5v-composite-v1",
        "text_source": str(text_dir),
        "vision_source": "baseten/GLM-5.2-Vision-NVFP4",
        "vision_revision": BASE_TEN_REVISION,
        "text_tensor_count": text_count,
        "vision_tensor_count": vision_count,
        "total_tensor_count": text_count + vision_count,
        "image_patch_limit": args.image_patch_limit,
        "vision_files": {
            name: {
                "size": (vision_dir / name).stat().st_size,
                "sha256": sha256(vision_dir / name),
            }
            for name in VISION_FILES
        },
    }
    write_json(output_dir / "GLM5V_COMPOSITE.json", manifest)

    print(
        f"assembled {output_dir}: {text_count} text + "
        f"{vision_count} vision/projector tensors"
    )


if __name__ == "__main__":
    main()
