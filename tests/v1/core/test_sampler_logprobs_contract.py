# SPDX-License-Identifier: Apache-2.0
"""Contract for preserving parsed speculative logprobs in sync output."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_sampler_output_uses_parser_logprobs_directly():
    source = (ROOT / "vllm/v1/worker/npu_model_runner.py").read_text(
        encoding="utf-8"
    )
    assert "valid_sampled_token_ids, logprobs_lists" in source
    assert "logprobs_tensors.tolists()" in source
    assert "logprobs_tensors.tolists(cu_num_tokens)" not in source
