# SPDX-License-Identifier: Apache-2.0
"""P4 operator tests use installed native owners, never download model fixtures."""

import pytest


@pytest.fixture(scope="session", autouse=True)
def require_910b3() -> None:
    """Fail before execution when the dedicated test device is not a 910B3."""
    import torch
    import torch_npu  # noqa: F401 -- registers the NPU device

    from vllm.inference_profile import validate_device_name

    if not torch.npu.is_available():
        pytest.fail("P4 operator tests require an available Ascend910B3")
    validate_device_name(torch.npu.get_device_name())
