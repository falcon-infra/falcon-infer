# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rotary Positional Embeddings."""

from typing import Any

import torch

from .base import RotaryEmbedding
from .deepseek_scaling_rope import DeepseekScalingRotaryEmbedding
from .linear_scaling_rope import LinearScalingRotaryEmbedding
from .yarn_scaling_rope import YaRNScalingRotaryEmbedding

_ROPE_DICT: dict[tuple[Any, ...], RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    max_position: int,
    is_neox_style: bool = True,
    rope_parameters: dict[str, Any] | None = None,
    dtype: torch.dtype | None = None,
    dual_chunk_attention_config: dict[str, Any] | None = None,
) -> RotaryEmbedding:
    if dual_chunk_attention_config is not None or any(
        key in (rope_parameters or {})
        for key in ("mrope_section", "xdrope_section", "use_fope")
    ):
        raise ValueError(
            "P4 supports only GLM text RoPE, not other-model/media variants"
        )
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_parameters is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_parameters_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in rope_parameters.items()
        }
        rope_parameters_args = tuple(rope_parameters_tuple.items())
    else:
        rope_parameters_args = None

    dual_chunk_attention_args = None

    rope_parameters = rope_parameters or {}
    base = rope_parameters.get("rope_theta", 10000)
    scaling_type = rope_parameters.get("rope_type", "default")
    partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)

    if partial_rotary_factor <= 0.0 or partial_rotary_factor > 1.0:
        raise ValueError(f"{partial_rotary_factor=} must be between 0.0 and 1.0")
    rotary_dim = int(head_size * partial_rotary_factor)

    key = (
        head_size,
        rotary_dim,
        max_position,
        is_neox_style,
        rope_parameters_args,
        dual_chunk_attention_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if scaling_type == "default":
        rotary_emb = RotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            dtype,
        )
    if scaling_type == "linear":
        scaling_factor = rope_parameters["factor"]
        rotary_emb = LinearScalingRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            scaling_factor,
            dtype,
        )
    if scaling_type == "yarn":
        scaling_factor = rope_parameters["factor"]
        original_max_position = rope_parameters["original_max_position_embeddings"]
        extra_kwargs = {
            k: v
            for k, v in rope_parameters.items()
            if k
            in (
                "extrapolation_factor",
                "attn_factor",
                "beta_fast",
                "beta_slow",
                "apply_yarn_scaling",
                "truncate",
            )
        }
        rotary_emb = YaRNScalingRotaryEmbedding(
            head_size,
            rotary_dim,
            original_max_position,
            base,
            is_neox_style,
            scaling_factor,
            dtype,
            **extra_kwargs,
        )
    elif scaling_type in ["deepseek_yarn", "deepseek_llama_scaling"]:
        scaling_factor = rope_parameters["factor"]
        original_max_position = rope_parameters["original_max_position_embeddings"]
        # assert max_position == original_max_position * scaling_factor
        extra_kwargs = {
            k: v
            for k, v in rope_parameters.items()
            if k
            in (
                "extrapolation_factor",
                "attn_factor",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            )
        }
        rotary_emb = DeepseekScalingRotaryEmbedding(
            head_size,
            rotary_dim,
            original_max_position,
            base,
            is_neox_style,
            scaling_factor,
            dtype,
            **extra_kwargs,
        )
    raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb
