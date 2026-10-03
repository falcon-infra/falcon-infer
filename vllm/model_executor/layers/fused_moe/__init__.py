# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from typing import Any

from vllm.model_executor.layers.fused_moe.activation import (
    MoEActivation,
    activation_without_mul,
    apply_moe_activation,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.layer import (
    FusedMoE,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular,
)
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import (
    FusedMoERouter,
)
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)
from vllm.triton_utils import HAS_TRITON

_config: dict[str, Any] | None = None


@contextmanager
def override_config(config):
    global _config
    old_config = _config
    _config = config
    yield
    _config = old_config


def get_config() -> dict[str, Any] | None:
    return _config


__all__ = [
    "FusedMoE",
    "FusedMoERouter",
    "FusedMoEConfig",
    "FusedMoEMethodBase",
    "MoEActivation",
    "UnquantizedFusedMoEMethod",
    "FusedMoeWeightScaleSupported",
    "FusedMoEExpertsModular",
    "FusedMoEActivationFormat",
    "FusedMoEPrepareAndFinalizeModular",
    "GateLinear",
    "RoutingMethodType",
    "SharedFusedMoE",
    "activation_without_mul",
    "apply_moe_activation",
    "override_config",
    "get_config",
]

from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import GroupedTopk

__all__ += ["fused_topk", "GroupedTopk"]
