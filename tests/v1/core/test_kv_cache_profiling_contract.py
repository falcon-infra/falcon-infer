# SPDX-License-Identifier: Apache-2.0
"""Dependency-light contracts for minimal KV graph profiling allocation."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load_sizing():
    path = ROOT / "vllm/v1/core/kv_cache_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "get_kv_cache_config_from_groups"
    )
    module = ast.parse("from __future__ import annotations")
    module.body.append(function)
    namespace = {
        "KVCacheConfig": lambda **kwargs: NS(**kwargs),
    }
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[function.name]


def _config(override):
    return NS(
        cache_config=NS(num_gpu_blocks_override=override),
        model_config=NS(hf_text_config=NS(index_topk=None)),
        num_speculative_tokens=0,
    )


def test_profiling_sizing_requires_zero_budget_and_positive_override():
    sizing = _load_sizing()
    result = sizing(_config(4), [], available_memory=0, for_profiling=True)
    assert result.num_blocks == 1

    with pytest.raises(ValueError, match="zero budget sentinel"):
        sizing(_config(4), [], available_memory=1, for_profiling=True)
    with pytest.raises(ValueError, match="zero budget sentinel"):
        sizing(_config(None), [], available_memory=0, for_profiling=True)


def test_profiling_budget_bypass_and_override_restore_are_explicit():
    sizing = ROOT.joinpath("vllm/v1/core/kv_cache_utils.py").read_text(
        encoding="utf-8"
    )
    runner = ROOT.joinpath("vllm/v1/worker/npu_runner_state.py").read_text(
        encoding="utf-8"
    )
    sfa = ROOT.joinpath(
        "vllm/v1/attention/backends/ascend/sfa_v1.py"
    ).read_text(encoding="utf-8")
    assert "and not for_profiling" in sizing
    assert "for_profiling=True" in runner
    assert "finally:" in runner
    assert "and not self._layerwise_prefill_p_node" in sfa
