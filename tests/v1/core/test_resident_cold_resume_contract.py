# SPDX-License-Identifier: Apache-2.0
"""Contracts preserving resident cold-load proofs through graph routing."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_resident_cold_load_carries_an_independent_computed_end():
    utils = (ROOT / "vllm/v1/attention/backends/ascend/utils.py").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "vllm/v1/worker/npu_model_runner.py").read_text(
        encoding="utf-8"
    )
    assert "resident_computed_ends" in utils
    assert "dsa_cold_resume_computed_end" in utils
    assert "cold_computed_ends" in runner
    assert "ColdResumeMarkers(cold_resumes, cold_computed_ends)" in runner
