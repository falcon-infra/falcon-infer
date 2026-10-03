# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retired configuration schema for explicit P4 compatibility errors."""

from typing import Any, Literal, TypeAlias

from pydantic import Field
from vllm.config.utils import config
from vllm.v1.attention.backends.registry import AttentionBackendEnum

MMEncoderTPMode = Literal["weights", "data"]
MMCacheType = Literal["shm", "lru"]
MMDummyOptions: TypeAlias = dict[str, Any]


@config
class MultiModalConfig:
    """Legacy fields only; constructing this retired feature always fails."""

    language_model_only: bool = False
    limit_per_prompt: MMDummyOptions = Field(default_factory=dict)
    enable_mm_embeds: bool = False
    media_io_kwargs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    mm_processor_kwargs: dict[str, object] | None = None
    mm_processor_cache_gb: float = Field(default=4, ge=0)
    mm_processor_cache_type: MMCacheType = "lru"
    mm_shm_cache_max_object_size_mb: int = Field(default=128, ge=0)
    mm_encoder_only: bool = False
    mm_encoder_tp_mode: MMEncoderTPMode = "weights"
    mm_encoder_attn_backend: AttentionBackendEnum | None = None
    interleave_mm_strings: bool = False
    skip_mm_profiling: bool = False
    video_pruning_rate: float | None = Field(default=None, ge=0.0, lt=1.0)

    def __post_init__(self) -> None:
        raise ValueError("Ascend P4 does not support multimodal")
