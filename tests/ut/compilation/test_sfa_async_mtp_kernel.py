# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Static contracts for async MTP device metadata preparation."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
KERNEL = ROOT / "vllm/model_executor/layers/ascend/triton/spec_decode/async_mtp.py"


def test_kernel_updates_bases_lengths_positions_and_both_slot_groups():
    tree = ast.parse(KERNEL.read_text(encoding="utf-8"))
    kernel = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "prepare_async_mtp_kernel"
    )
    names = {arg.arg for arg in kernel.args.args}
    assert {
        "bases",
        "counts",
        "positions",
        "seq_common",
        "seq_target",
        "table0",
        "table1",
        "slots0",
        "slots1",
    } <= names
    source = ast.unparse(kernel)
    assert "tl.static_range(2)" in source
    assert "seq_common" in source and "seq_target" in source
    assert "slots0" in source and "slots1" in source
    assert "position // block0" in source and "position // block1" in source


def test_kernel_advances_mixed_c8_indexer_slots():
    source = KERNEL.read_text(encoding="utf-8")
    assert "slots_c8" in source
    assert "MIXED_C8" in source
    assert "block_id1 * 2" in source
    assert "table_c8" in source
    assert "stride_c8" in source
