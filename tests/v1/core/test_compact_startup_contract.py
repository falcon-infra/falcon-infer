# SPDX-License-Identifier: Apache-2.0
"""Contracts for explicit compact-D startup sizing and admission."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_compact_startup_capacity_is_explicit_and_shared_pool_only():
    utils = (ROOT / "vllm/v1/core/kv_cache_utils.py").read_text(encoding="utf-8")
    scheduler = (ROOT / "vllm/v1/core/sched/scheduler.py").read_text(
        encoding="utf-8"
    )
    assert "def _dsa_compact_startup_bundles" in utils
    assert "def dsa_dense_request_token_capacity" in utils
    assert "_reject_impossible_initial_dsa_request" in scheduler
