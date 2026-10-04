#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the explicit source/host subset; never install dependencies or use NPU."""

from __future__ import annotations

# Standard
import argparse
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TENSOR_TESTS = {
    "vllm": ("test_cold_resume_native_metadata.py", "test_glm52_topk_ownership.py"),
    "lmcache": (
        "test_glm52_metadata.py",
        "test_capture_submission.py",
        "test_checkpoint_allocation_budget.py",
        "test_checkpoint_miss_retry.py",
        "test_checkpoint_page_keys.py",
        "test_dense_checkpoint_partial_pages.py",
        "test_local_checkpoint_restore.py",
        "test_preemption_checkpoint.py",
        "test_checkpoint_initialization.py",
    ),
}


def plan() -> dict:
    """Return the reviewed host-only commands and explicitly deferred tensor tests."""
    primary = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["name"]
    deferred = ["tests/standalone/" + name for name in TENSOR_TESTS[primary]]
    tests = [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "-q",
        "-o",
        "log_cli=false",
        "--noconftest",
        "-p",
        "no:cacheprovider",
        "tests/standalone",
    ]
    tests += ["--ignore=" + path for path in deferred]
    if primary == "vllm":
        tests += ["-k", "not production_draft_expansion_can_exceed_target_capacity"]
        deferred.append(
            "test_staged_dummy_capacity.py::test_production_draft_expansion_can_exceed_target_capacity"
        )
    else:
        tests += [
            "tests/v1/test_cache_engine_close_cpu.py",
            "tests/v1/test_direct_store_plan.py",
            "tests/v1/test_remote_fill_config.py",
        ]
    return {
        "scope": "selected_host_contracts_not_full_test_tree_or_NPU_acceptance",
        "commands": [
            [sys.executable, "-B", "tools/check_native_layout.py", "--source-only"],
            [sys.executable, "-B", "tools/check_p4_profile.py"],
            [sys.executable, "-B", "tools/check_npu_native.py"],
            tests,
        ],
        "deferred_tensor_tests": deferred,
        "note": (
            "Requires the paired repositories as sibling vllm/ and LMCache/ checkouts. "
            "Full tensor/NPU tests remain intranet work."
        ),
    }


def main() -> None:
    """Print the plan or run it fail-fast; no network or implicit installation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show commands and deferred cases without execution",
    )
    args = parser.parse_args()
    report = plan()
    print(json.dumps(report, indent=2), flush=True)
    if args.list:
        return
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONPATH=str(ROOT),
    )
    for command in report["commands"]:
        subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
