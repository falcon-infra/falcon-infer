# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static contracts for native MTP proposal staging reductions."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def test_one_draft_proposal_avoids_unneeded_position_and_table_work():
    path = ROOT / "vllm/v1/spec_decode/ascend/eagle_proposer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    propose = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_propose"
    )
    source = ast.unparse(propose)
    assert "use_staged_mtp_draft_graph or self.num_speculative_tokens > 1" in source
    assert "not use_staged_mtp_draft_graph and self.num_speculative_tokens > 1" in source


def test_startup_warmup_uses_private_token_and_count_buffers():
    path = ROOT / "vllm/v1/spec_decode/ascend/eagle_proposer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    warmup = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "warmup_next_mtp_tokens"
    )
    source = ast.unparse(warmup)
    assert "torch.zeros" in source
    assert "torch.empty_like" in source
    assert "prepare_next_mtp_tokens_kernel" in source
    assert "for width in (1, 2)" in source
