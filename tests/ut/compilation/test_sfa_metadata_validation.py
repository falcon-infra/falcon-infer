# SPDX-License-Identifier: Apache-2.0
"""Static contracts for startup-only full-graph validation."""

import ast
from pathlib import Path


def test_live_prepare_layer_does_not_call_static_checker_or_return_inputs():
    path = Path(__file__).resolve().parents[3] / "vllm/v1/attention/backends/ascend/sfa_v1.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSFAImpl")
    method = next(node for node in cls.body if getattr(node, "name", "") == "prepare_full_graph_layer")
    source = ast.get_source_segment(path.read_text(encoding="utf-8"), method)
    assert source is not None
    assert "if context.staged_sfa_graph_dummy_run" in source
    assert "if not context.staged_sfa_graph_dummy_run:\n            return {}" in source


def test_live_prepare_run_uses_builder_signature_instead_of_walking_inputs():
    path = Path(__file__).resolve().parents[3] / "vllm/compilation/ascend/sfa_full_graph.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SFAFullGraph")
    method = next(node for node in cls.body if getattr(node, "name", "") == "prepare_run")
    calls_validate = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "validate_inputs"
    ]
    assert len(calls_validate) == 1
    assert any(isinstance(node, ast.Attribute) and node.attr == "signature" for node in ast.walk(method))


def test_shared_indexer_layer_guards_are_not_collapsed_by_metadata_memoization():
    path = Path(__file__).resolve().parents[3] / "vllm/v1/attention/backends/ascend/sfa_v1.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_cross_layer_ineligible_reason"
    )
    body = ast.unparse(method)
    assert "has_indexer" in body
    assert "topk_indices_buffer" in body
    assert "indexer_block_table" in body or "indexer" in body
