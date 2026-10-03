# SPDX-License-Identifier: Apache-2.0
"""GLM uses its checkpoint tokenizer chat template; no other-model fallback."""

from pathlib import Path

# Kept for user-supplied template path diagnostics; no fallback assets shipped.
CHAT_TEMPLATES_DIR = Path(__file__).parent


def get_chat_template_fallback_path(
    model_type: str, tokenizer_name_or_path: str
) -> Path | None:
    return None
