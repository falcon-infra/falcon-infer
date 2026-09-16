# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Startup-only fixture helpers for explicitly selected SFA workers.

Normal serving does not import this module or remap real checkpoint weights.
"""

import json
from pathlib import Path

import torch

TARGET_LAYERS = 8
HASH_CHUNK_BYTES = 8 * 1024 * 1024


def remap_mtp_quant_description(
    description: dict,
    source_start: int,
    num_mtp_layers: int,
) -> dict:
    """Move original MTP quantization namespaces into the dummy fixture."""
    if type(source_start) is not int or source_start < TARGET_LAYERS:
        raise ValueError("Original model num_hidden_layers must be an integer >= 8")
    if type(num_mtp_layers) is not int or num_mtp_layers < 1:
        raise ValueError("The parity fixture requires at least one MTP layer")
    destinations = tuple(
        f"model.layers.{TARGET_LAYERS + i}." for i in range(num_mtp_layers)
    )
    result = {
        key: value
        for key, value in description.items()
        if not key.startswith(destinations)
    }
    for offset, destination in enumerate(destinations):
        source = f"model.layers.{source_start + offset}."
        if (
            source + "head.weight" not in description
            and source + "shared_head.head.weight" not in description
        ):
            raise ValueError(
                f"Missing original MTP head quantization at {source}; "
                "refusing a fallback"
            )
        result.update(
            (destination + key[len(source) :], value)
            for key, value in description.items()
            if key.startswith(source)
        )
    return result


def deterministic_dummy_load(original, loader, model, model_config) -> None:
    """Initialize integer dummy weights before quantization post-processing."""
    original(loader, model, model_config)
    with torch.no_grad():
        for value in model.parameters():
            if (
                value.is_floating_point()
                or value.is_complex()
                or value.dtype == torch.bool
            ):
                continue
            generator = torch.Generator(device="cpu").manual_seed(1234)
            if value.ndim == 0:
                value = value.reshape(1)
            row_elements = value[0].numel() if value.shape[0] else 1
            chunk_rows = max(1, HASH_CHUNK_BYTES // max(1, row_elements * 8))
            for chunk in value.split(chunk_rows):
                sample = torch.randint(
                    0,
                    8,
                    chunk.shape,
                    generator=generator,
                    dtype=torch.int64,
                )
                chunk.copy_(sample.to(dtype=value.dtype))


def prepare_dummy_quant_config(config) -> None:
    """Remap quantization only for the eight-layer dummy fixture."""
    from vllm.model_executor.layers.quantization.ascend.modelslim_config import (
        AscendModelSlimConfig,
    )

    if (
        config.load_config.load_format != "dummy"
        or config.model_config.hf_config.num_hidden_layers != TARGET_LAYERS
    ):
        raise ValueError(
            "MTP quantization remapping is only for the eight-layer dummy fixture"
        )
    speculative = config.speculative_config
    if speculative is None or speculative.num_speculative_tokens != 1:
        raise ValueError("SFA parity requires MTP=1")
    draft = speculative.draft_model_config.hf_config
    if (
        draft.model_type != "deepseek_mtp"
        or not isinstance(config.quant_config, AscendModelSlimConfig)
    ):
        raise ValueError(
            "SFA parity requires a DeepSeek MTP draft with Ascend ModelSlim quantization"
        )
    original = json.loads(
        (Path(config.model_config.model) / "config.json").read_text(
            encoding="utf-8"
        )
    )
    description = remap_mtp_quant_description(
        config.quant_config.quant_description,
        original["num_hidden_layers"],
        draft.num_nextn_predict_layers,
    )
    config.quant_config = AscendModelSlimConfig(description)
