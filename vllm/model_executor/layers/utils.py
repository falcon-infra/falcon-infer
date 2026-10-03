# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Utility methods for model layers."""

from collections.abc import Callable

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

MOE_LAYER_ROUTER_GATE_SUFFIXES = {
    "gate",
    "router",
    "router_gate",
    "shared_expert_gate",
    "expert_gate",
}


def is_layer_moe_router_gate(prefix: str) -> bool:
    if not prefix:
        return False
    return prefix.rsplit(".", 1)[-1] in MOE_LAYER_ROUTER_GATE_SUFFIXES


def get_token_bin_counts_and_mask(
    tokens: torch.Tensor,
    vocab_size: int,
    num_seqs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Compute the bin counts for the tokens.
    # vocab_size + 1 for padding.
    bin_counts = torch.zeros(
        (num_seqs, vocab_size + 1), dtype=torch.long, device=tokens.device
    )
    bin_counts.scatter_add_(1, tokens, torch.ones_like(tokens))
    bin_counts = bin_counts[:, :vocab_size]
    mask = bin_counts > 0

    return bin_counts, mask


def apply_penalties(
    logits: torch.Tensor,
    prompt_tokens_tensor: torch.Tensor,
    output_tokens_tensor: torch.Tensor,
    presence_penalties: torch.Tensor,
    frequency_penalties: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> torch.Tensor:
    """
    Applies penalties in place to the logits tensor
    logits : The input logits tensor of shape [num_seqs, vocab_size]
    prompt_tokens_tensor: A tensor containing the prompt tokens. The prompts
        are padded to the maximum prompt length within the batch using
        `vocab_size` as the padding value. The value `vocab_size` is used
        for padding because it does not correspond to any valid token ID
        in the vocabulary.
    output_tokens_tensor: The output tokens tensor.
    presence_penalties: The presence penalties of shape (num_seqs, )
    frequency_penalties: The frequency penalties of shape (num_seqs, )
    repetition_penalties: The repetition penalties of shape (num_seqs, )
    """
    num_seqs, vocab_size = logits.shape
    _, prompt_mask = get_token_bin_counts_and_mask(
        prompt_tokens_tensor, vocab_size, num_seqs
    )
    output_bin_counts, output_mask = get_token_bin_counts_and_mask(
        output_tokens_tensor, vocab_size, num_seqs
    )

    # Apply repetition penalties as a custom op
    # Portable PyTorch operations dispatch to NPU for generation penalties.
    repeated = prompt_mask | output_mask
    adjusted = torch.where(
        logits > 0,
        logits / repetition_penalties[:, None],
        logits * repetition_penalties[:, None],
    )
    logits.copy_(torch.where(repeated, adjusted, logits))

    # We follow the definition in OpenAI API.
    # Refer to https://platform.openai.com/docs/api-reference/parameter-details
    logits -= frequency_penalties.unsqueeze(dim=1) * output_bin_counts
    logits -= presence_penalties.unsqueeze(dim=1) * output_mask
    return logits


def default_unquantized_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.device.type == "npu":
        return torch.ops.vllm.unquantized_gemm(x, weight, bias)
    else:
        return torch.nn.functional.linear(x, weight, bias)


def dispatch_unquantized_gemm() -> Callable[..., torch.Tensor]:
    return default_unquantized_gemm


def unquantized_gemm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight, bias)


def unquantized_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    output_shape = (x.shape[0], weight.shape[0])
    return torch.empty(output_shape, dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="unquantized_gemm",
    op_func=unquantized_gemm,
    fake_impl=unquantized_gemm_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
