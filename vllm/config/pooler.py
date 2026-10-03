# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retired configuration schema for explicit P4 compatibility errors."""

from typing import Literal, get_args

from vllm.config.utils import config

SequencePoolingType = Literal["CLS", "LAST", "MEAN"]
SEQ_POOLING_TYPES: tuple[SequencePoolingType, ...] = get_args(SequencePoolingType)
TokenPoolingType = Literal["ALL", "STEP"]
TOK_POOLING_TYPES: tuple[TokenPoolingType, ...] = get_args(TokenPoolingType)


@config
class PoolerConfig:
    """Legacy fields only; constructing this retired feature always fails."""

    pooling_type: SequencePoolingType | TokenPoolingType | None = None
    seq_pooling_type: SequencePoolingType | None = None
    tok_pooling_type: TokenPoolingType | None = None
    use_activation: bool | None = None
    dimensions: int | None = None
    enable_chunked_processing: bool = False
    max_embed_len: int | None = None
    logit_bias: float | None = None
    step_tag_id: int | None = None
    returned_token_ids: list[int] | None = None

    def __post_init__(self) -> None:
        raise ValueError("Ascend P4 does not support pooling")
