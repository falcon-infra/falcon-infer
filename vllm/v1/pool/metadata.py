# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retired pooling type slot for shared request annotations; no execution."""


class PoolingStates:
    def __init__(self) -> None:
        raise ValueError("Pooling was removed from the P4 text profile")
