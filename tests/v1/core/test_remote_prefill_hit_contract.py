# SPDX-License-Identifier: Apache-2.0
"""Contract for propagating remote prefill cache hits to output accounting."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_scheduler_and_output_processor_preserve_remote_prefill_hits():
    scheduler = (ROOT / "vllm/v1/core/sched/scheduler.py").read_text(
        encoding="utf-8"
    )
    output = (ROOT / "vllm/v1/engine/output_processor.py").read_text(
        encoding="utf-8"
    )
    assert '"remote_prefill_cached_tokens"' in scheduler
    assert "remote_prefill_cached_tokens" in output
    assert "if req_state.is_prefilling:" in output
    assert "req_state.num_cached_tokens = req_state.remote_prefill_cached_tokens" in output
