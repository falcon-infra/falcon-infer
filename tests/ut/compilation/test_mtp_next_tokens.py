# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static contracts for the native fused MTP next-token path."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_native_kernel_has_strided_width_and_count_contract():
    path = ROOT / "vllm/model_executor/layers/ascend/triton/spec_decode/utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "prepare_next_mtp_tokens_kernel"
    )
    names = {argument.arg for argument in kernel.args.args}
    assert {
        "sampled",
        "backup",
        "next_tokens",
        "valid_counts",
        "row_stride",
        "token_stride",
        "WIDTH",
        "BLOCK_SIZE",
    } <= names
    source = ast.unparse(kernel)
    assert "count == 2" in source
    assert "count > 0" in source
    assert "row_stride" in source and "token_stride" in source


def test_native_eagle_path_is_strictly_gated():
    path = ROOT / "vllm/v1/spec_decode/ascend/eagle_proposer.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "prepare_next_token_ids_padded"
    )
    body = ast.unparse(method)
    assert "HAS_TRITON" in body
    assert "self.method == 'mtp'" in body
    assert "self.num_speculative_tokens == 1" in body
    assert "num_discarded_requests == 0" in body
    assert "prepare_next_mtp_tokens_kernel" in body
