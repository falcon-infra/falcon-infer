# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for immutable one-draft MTP metadata arrays."""

import ast
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "vllm/v1/worker/npu_model_runner.py"


@dataclass
class Metadata:
    draft_token_ids: torch.Tensor
    num_draft_tokens: list[int]
    cu_num_draft_tokens: torch.Tensor
    cu_num_sampled_tokens: torch.Tensor
    target_logits_indices: torch.Tensor
    bonus_logits_indices: torch.Tensor
    logits_indices: torch.Tensor


@pytest.fixture(scope="module")
def methods():
    tree = ast.parse(PATH.read_text(encoding="utf-8"))
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fixed_mtp_metadata_arrays"
    )
    calc = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_calc_spec_decode_metadata"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.extend([helper, calc])
    namespace = {"torch": torch, "np": np, "SpecDecodeMetadata": Metadata}
    exec(compile(ast.fix_missing_locations(module), str(PATH), "exec"), namespace)
    return namespace


def runner(methods, requests=8):
    fixed = methods["_fixed_mtp_metadata_arrays"](requests, torch.device("cpu"))
    return SimpleNamespace(
        _fixed_mtp_metadata=fixed,
        pcp_size=1,
        max_num_reqs=requests,
        _fixed_decode_cu_num_tokens=np.arange(2, 2 * requests + 1, 2),
        input_ids=SimpleNamespace(
            gpu=torch.arange(2 * requests + 4, dtype=torch.int32) + 1000
        ),
    )


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_fixed_metadata_matches_request_major_layout(methods, dtype):
    subject = runner(methods)
    result = methods["_calc_spec_decode_metadata"](
        subject,
        np.ones(4, dtype=np.int32),
        np.arange(2, 9, 2, dtype=dtype),
        None,
    )
    assert result.num_draft_tokens == [1, 1, 1, 1]
    assert result.cu_num_draft_tokens.tolist() == [1, 2, 3, 4]
    assert result.cu_num_sampled_tokens.tolist() == [2, 4, 6, 8]
    assert result.target_logits_indices.tolist() == [0, 2, 4, 6]
    assert result.bonus_logits_indices.tolist() == [1, 3, 5, 7]
    assert result.logits_indices.tolist() == list(range(8))


def test_structural_arrays_are_reused_but_token_ids_are_fresh(methods):
    subject = runner(methods)
    fixed = subject._fixed_mtp_metadata
    first = methods["_calc_spec_decode_metadata"](
        subject,
        np.ones(4, dtype=np.int32),
        np.arange(2, 9, 2, dtype=np.int64),
        None,
    )
    subject.input_ids.gpu.add_(100)
    second = methods["_calc_spec_decode_metadata"](
        subject,
        np.ones(4, dtype=np.int32),
        np.arange(2, 9, 2, dtype=np.int64),
        None,
    )
    assert second.draft_token_ids.tolist() == [1101, 1103, 1105, 1107]
    assert first.draft_token_ids.tolist() == [1001, 1003, 1005, 1007]
    assert first.cu_num_draft_tokens.data_ptr() == fixed[0].data_ptr()
    assert second.logits_indices.data_ptr() == fixed[5].data_ptr()
