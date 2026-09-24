# SPDX-License-Identifier: Apache-2.0
"""Dependency-light contract for compact boundary block admission."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_compact_boundary_reserves_restore_destination_without_scheduling_work():
    source = (ROOT / "vllm/v1/core/sched/scheduler.py").read_text(
        encoding="utf-8"
    )
    assert "request.num_tokens % self.block_size == 1" in source
    assert "num_external_computed_tokens == request.num_tokens - 1" in source
    assert "effective_lookahead_tokens = max(effective_lookahead_tokens, 1)" in source
