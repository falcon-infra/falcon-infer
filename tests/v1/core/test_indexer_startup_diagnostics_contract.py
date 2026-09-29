# SPDX-License-Identifier: Apache-2.0
"""Contract for startup-only native C8 mapper diagnostics."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_startup_diagnostics_wrap_capture_and_profiling_only():
    mapper = (ROOT / "vllm/v1/worker/dsa_shared_pool.py").read_text(
        encoding="utf-8"
    )
    runner = (ROOT / "vllm/v1/worker/npu_model_runner.py").read_text(
        encoding="utf-8"
    )
    assert "def diagnose_indexer_metadata" in mapper
    assert "indexer_metadata_startup" in mapper
    assert "from vllm.v1.worker.dsa_shared_pool import diagnose_indexer_metadata" in runner
    assert "from vllm.v1.core.dsa_shared_pool import diagnose_indexer_metadata" not in runner
    assert 'diagnose_indexer_metadata("capture")' in runner
    assert 'diagnose_indexer_metadata("profiling")' in runner
