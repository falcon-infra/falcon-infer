# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm.config import VllmConfig


def init_model_state(
    vllm_config: VllmConfig,
    model: nn.Module,
    encoder_cache: None,
    device: torch.device,
):

    from vllm.v1.worker.npu.v2.common.model_states.default import DefaultModelState

    return DefaultModelState(vllm_config, model, encoder_cache, device)
