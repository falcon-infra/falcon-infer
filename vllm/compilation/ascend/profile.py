# SPDX-License-Identifier: Apache-2.0
"""Configuration helpers for native Ascend sequence parallelism."""


def get_sequence_parallelism_threshold(
    hidden_size: int, tp_size: int, element_size: int
) -> int | None:
    """No GPU-derived threshold; Ascend's pass manager owns SP selection.

    Preserve the previous non-CUDA result used by common compile-range setup.
    Explicit native SP settings are handled in ``vllm.config.ascend``.
    """
    return None
