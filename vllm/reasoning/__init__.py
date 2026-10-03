# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.reasoning.abs_reasoning_parsers import ReasoningParser, ReasoningParserManager

__all__ = [
    "ReasoningParser",
    "ReasoningParserManager",
]
"""
Register a lazy module mapping.

Example:
    ReasoningParserManager.register_lazy_module(
        name="qwen3",
        module_path="vllm.reasoning.qwen3_reasoning_parser",
        class_name="Qwen3ReasoningParser",
    )
"""


_REASONING_PARSERS_TO_REGISTER = {
    "glm45": ("deepseek_v3_reasoning_parser", "DeepSeekV3ReasoningWithThinkingParser")
}


def register_lazy_reasoning_parsers():
    for name, (file_name, class_name) in _REASONING_PARSERS_TO_REGISTER.items():
        module_path = f"vllm.reasoning.{file_name}"
        ReasoningParserManager.register_lazy_module(name, module_path, class_name)


register_lazy_reasoning_parsers()
