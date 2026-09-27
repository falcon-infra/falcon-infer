# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Temporary import compatibility for the platform moved into vLLM in P2.

The class is implemented in ``vllm.platforms.npu``. This module does not
register a plugin, replace modules, or maintain another platform implementation.
"""

from vllm.platforms.npu import NPUPlatform, config_deprecated_logging

__all__ = ["NPUPlatform", "config_deprecated_logging"]
