# SPDX-License-Identifier: Apache-2.0
"""Contracts for native GLM tool-call streaming backfill."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_glm_parser_owns_cumulative_argument_backfill():
    parser = (ROOT / "vllm/tool_parsers/glm4_moe_tool_parser.py").read_text(
        encoding="utf-8"
    )
    abstract = (ROOT / "vllm/tool_parsers/abstract_tool_parser.py").read_text(
        encoding="utf-8"
    )
    serving = (
        ROOT / "vllm/entrypoints/openai/chat_completion/serving.py"
    ).read_text(encoding="utf-8")
    assert "_streaming_text" in parser
    assert "partial_tag_overlap" in parser
    assert "get_remaining_unstreamed_args" in abstract
    assert "get_remaining_unstreamed_args" in serving
    assert "_append_remaining_tool_args" in serving
    assert "_should_check_for_unstreamed_tool_arg_tokens" not in serving
    assert "if content:" in parser
    assert "DeltaMessage(content=content)" in parser


def test_glm_schema_dependencies_are_declared_for_native_runtime():
    requirements = (ROOT / "requirements/ascend.txt").read_text(encoding="utf-8")
    assert "jsonschema >= 4.23.0" in requirements
    assert "referencing >= 0.35.1" in requirements
