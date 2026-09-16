# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static contracts for the fused fixed-layout async-MTP preparation path."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def _methods(path: Path, names: set[str]):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in names
    }


def test_proposer_reuses_fixed_q2_outputs_between_preparation_stages():
    path = ROOT / "vllm/v1/spec_decode/ascend/eagle_proposer.py"
    methods = _methods(
        path,
        {
            "prepare_next_token_ids_padded",
            "prepare_inputs_padded",
            "set_inputs_first_pass",
            "warmup_next_mtp_tokens",
        },
    )
    prepare_next = methods["prepare_next_token_ids_padded"]
    prepare_inputs = methods["prepare_inputs_padded"]
    first_pass = methods["set_inputs_first_pass"]
    warmup = methods["warmup_next_mtp_tokens"]
    assert "spec_decode_metadata" in prepare_next
    assert "fixed_q2" in prepare_next
    assert "_prepared_mtp_inputs" in prepare_next
    assert "prepared[3:]" in prepare_inputs
    assert "pack_mtp_tokens_positions_kernel" in first_pass
    assert "_fixed_mtp_pipeline_ready" in warmup
    assert "FIXED_Q2" in warmup


def test_async_token_kernel_uses_contiguous_interleaved_store():
    path = ROOT / "vllm/model_executor/layers/ascend/triton/spec_decode/async_mtp.py"
    source = _methods(path, {"prepare_async_mtp_tokens_kernel"})[
        "prepare_async_mtp_tokens_kernel"
    ]
    assert "tokens =" in source
    assert "tokens // 2" in source
    assert "active_tokens" in source
    assert "input_ids + tokens" in source
    assert "2 * rows" not in source


def test_runner_passes_metadata_and_reuses_identity_logits_view():
    path = ROOT / "vllm/v1/worker/npu_model_runner.py"
    methods = _methods(
        path,
        {"_select_sample_hidden_states", "_run_speculative_decoding"},
    )
    select = methods["_select_sample_hidden_states"]
    run = _methods(path, {"propose_draft_token_ids"})["propose_draft_token_ids"]
    assert "data_ptr" in select
    assert "hidden_states[:logits_indices.numel()]" in select
    assert "spec_decode_metadata" in run
