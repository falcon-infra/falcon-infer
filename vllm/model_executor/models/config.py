# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration hook contract for the native GLM text profile.

GLM-5.2 and its internal MTP use the base configuration plus native Ascend
validation. There are no extra per-model rewrites in this profile.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig


class VerifyAndUpdateConfig:
    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        return

    @staticmethod
    def verify_and_update_model_config(model_config: "ModelConfig") -> None:
        return


MODELS_CONFIG_MAP: dict[str, type[VerifyAndUpdateConfig]] = {}
