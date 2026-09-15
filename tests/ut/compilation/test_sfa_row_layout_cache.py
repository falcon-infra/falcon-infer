# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static owner contracts for native SFA general row-layout reuse."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "vllm/v1/attention/backends/ascend/sfa_v1.py"


def test_builder_owns_general_layout_signature_and_row_count():
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    builder = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "AscendSFAMetadataBuilder"
    )
    init = next(node for node in builder.body if getattr(node, "name", "") == "__init__")
    build = next(node for node in builder.body if getattr(node, "name", "") == "build")
    init_source = ast.unparse(init)
    build_source = ast.unparse(build)
    assert "_dsa_general_layout_signature" in init_source
    assert "_dsa_general_decode_rows" in init_source
    assert "general_signature" in build_source
    assert "_dsa_general_layout_signature" in build_source
    assert "_dsa_general_decode_rows" in build_source


def test_general_cache_requires_stable_request_widths_and_prompt_layout():
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    build = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "build"
    )
    source = ast.unparse(build)
    assert "np.diff(qsl)" in source
    assert "widths >= 1" in source
    assert "widths <= 2" in source
    assert "tuple(common_attn_metadata.request_ids or ())" in source
    assert "np.all(computed >= plens_cpu[:n_real])" in source
