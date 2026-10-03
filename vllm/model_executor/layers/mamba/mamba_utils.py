# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import torch


@dataclass
class MambaCopySpec:
    """
    Data class specifying the memory-copy parameters for Mamba states used for
    prefix caching in align mode.

    Attributes:
        start_addr (int): Starting address for the memory copy operation.
        num_elements (int): Number of elements to copy from the starting address.
    """

    start_addr: int
    num_elements: int


MambaStateCopyFunc: TypeAlias = Callable[
    [torch.Tensor, list[int], int, int], MambaCopySpec
]
"""
Type alias for a function that computes a MambaCopySpec for copying state slices.
Parameters:
  state: torch.Tensor - the Mamba state tensor (e.g., conv or temporal states).
  block_ids: list[int] - the list of block indices for the state to copy.
  cur_block_idx: int - current block index within `block_ids` to copy from.
  num_accepted_tokens: int - number of accepted tokens used to compute the copy offset.
      Range: 1 .. 1 + num_speculative_tokens (inclusive).
"""
