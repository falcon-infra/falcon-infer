# SPDX-License-Identifier: Apache-2.0
"""Native Ascend MoE selection; no other-device kernel discovery."""

from enum import Enum


class UnquantizedMoeBackend(Enum):
    NPU = "NPU"


def select_unquantized_moe_backend(moe_config, use_ep: bool, use_dp: bool):
    """Select native Ascend compute; HCCL strategy belongs to AscendMoERunner."""
    if moe_config.moe_backend != "auto":
        raise ValueError("P4 requires moe_backend=auto for native Ascend MoE")
    return UnquantizedMoeBackend.NPU
