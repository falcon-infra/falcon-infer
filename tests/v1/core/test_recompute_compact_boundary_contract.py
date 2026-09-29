# SPDX-License-Identifier: Apache-2.0
"""Contract for compact-boundary admission in the recompute scheduler."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_recompute_scheduler_reserves_the_final_restore_slot():
    source = (ROOT / "vllm/v1/core/sched/recompute_scheduler.py").read_text(
        encoding="utf-8"
    )
    assert "dsa_compact_external_load" in source
    assert "num_external_computed_tokens == request.num_tokens - 1" in source
    assert "effective_lookahead_tokens = max(effective_lookahead_tokens, 1)" in source
