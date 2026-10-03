# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retired configuration schema for explicit P4 compatibility errors."""

from typing import Literal

import torch
from pydantic import ConfigDict, Field
from vllm.config.utils import config

LoRADType = Literal["auto", "float16", "bfloat16"]
MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]
LoRAExtraVocabSize = Literal[256, 512]


@config(config=ConfigDict(arbitrary_types_allowed=True))
class LoRAConfig:
    """Legacy fields only; constructing this retired feature always fails."""

    max_lora_rank: MaxLoRARanks = 16
    max_loras: int = Field(default=1, ge=1)
    fully_sharded_loras: bool = False
    max_cpu_loras: int | None = None
    lora_dtype: torch.dtype | LoRADType = "auto"
    default_mm_loras: dict[str, str] | None = None
    enable_tower_connector_lora: bool = False
    specialize_active_lora: bool = False

    def __post_init__(self) -> None:
        raise ValueError("Ascend P4 does not support LoRA")
