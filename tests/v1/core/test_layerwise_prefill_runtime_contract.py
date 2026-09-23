# SPDX-License-Identifier: Apache-2.0
"""Host contracts for source 92's native runtime adaptation."""

import ast
import os
from collections.abc import Mapping
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[3]


class _FakeArray:
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        if isinstance(key, _FakeArray):
            key = key.values
        if isinstance(key, list):
            return _FakeArray(self.values[int(index)] for index in key)
        value = self.values[key]
        return _FakeArray(value) if isinstance(key, slice) else value

    def __setitem__(self, key, value):
        self.values[key] = value


class _FakeNumpy:
    int8 = int
    int64 = int
    ndarray = _FakeArray

    @staticmethod
    def zeros(size, dtype=None):
        del dtype
        return _FakeArray([0] * size)

    @staticmethod
    def asarray(values, dtype=None):
        del dtype
        return _FakeArray(values)


def source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_native_transfer_window_hooks_are_explicit():
    base = source("vllm/distributed/kv_transfer/kv_connector/v1/base.py")
    utils = source("vllm/v1/attention/backends/ascend/utils.py")
    sfa = source("vllm/v1/attention/backends/ascend/sfa_v1.py")
    for name in (
        "submit_layerwise_prefill_load",
        "finish_layerwise_prefill_save",
        "save_kv_layer_in_layerwise_prefill_transfer_window",
        "save_kv_layer_outside_layerwise_prefill_transfer_window",
    ):
        assert name in base
    assert "layerwise_prefill_transfer_window_supported" in utils
    assert "_submit_sfa_layerwise_transfer_window" in sfa
    assert "sfa_layerwise_prefill_pending" in sfa


def test_npu_runtime_uses_two_physical_block_tables_and_layer_refs():
    input_batch = source("vllm/v1/worker/npu_input_batch.py")
    runner = source("vllm/v1/worker/npu_model_runner.py")
    assert "layerwise_prefill_block_tables" in input_batch
    assert "LAYERWISE_PREFILL_BANK_COUNT - 1" in input_batch
    assert "_refresh_layerwise_prefill_block_tables" in runner
    assert "build_prefill_layer_refs" in runner
    assert "_layerwise_prefill_common_attn_metadata" in runner


def _bank_phase_methods():
    tree = ast.parse(source("vllm/v1/worker/npu_model_runner.py"))
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
    )
    names = {
        "_capture_layerwise_prefill_bank_offsets",
        "_prepare_layerwise_prefill_bank_phase_masks",
    }
    methods = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    namespace = {"Any": Any, "Mapping": Mapping, "np": _FakeNumpy}
    exec(compile(ast.Module(body=methods, type_ignores=[]), "runner", "exec"), namespace)
    return namespace


def test_chunk_phase_metadata_is_captured_for_mapping_and_nested_connectors():
    methods = _bank_phase_methods()
    runner = SimpleNamespace(
        layerwise_prefill_p_node=True,
        _layerwise_prefill_bank_offsets_by_req={},
    )
    runner._capture_layerwise_prefill_bank_offsets = MethodType(
        methods["_capture_layerwise_prefill_bank_offsets"], runner
    )

    nested = SimpleNamespace(
        requests={
            "remote": SimpleNamespace(layerwise_prefill_bank_offset=3),
            "local": SimpleNamespace(layerwise_prefill_bank_offset=0),
        }
    )
    scheduler_output = SimpleNamespace(
        kv_connector_metadata=SimpleNamespace(metadata=(nested,))
    )
    runner._capture_layerwise_prefill_bank_offsets(scheduler_output)

    assert runner._layerwise_prefill_bank_offsets_by_req == {
        "remote": 1,
        "local": 0,
    }


def test_chunk_phase_masks_follow_request_and_token_order():
    methods = _bank_phase_methods()
    runner = SimpleNamespace(
        layerwise_prefill_p_node=True,
        pcp_size=1,
        input_batch=SimpleNamespace(req_ids=["r0", "r1"]),
        _layerwise_prefill_bank_offsets_by_req={"r0": 1, "r1": 0},
    )
    runner._prepare_layerwise_prefill_bank_phase_masks = MethodType(
        methods["_prepare_layerwise_prefill_bank_phase_masks"], runner
    )
    runner._prepare_layerwise_prefill_bank_phase_masks(
        [0, 1, 1], 2
    )

    assert runner._layerwise_prefill_bank_phase_by_request_cpu.values == [
        1,
        0,
    ]
    assert runner._layerwise_prefill_bank_phase_by_token_cpu.values == [
        1,
        0,
        0,
    ]


def test_invalid_chunk_phase_fails_closed():
    methods = _bank_phase_methods()
    runner = SimpleNamespace(
        layerwise_prefill_p_node=True,
        _layerwise_prefill_bank_offsets_by_req={},
    )
    runner._capture_layerwise_prefill_bank_offsets = MethodType(
        methods["_capture_layerwise_prefill_bank_offsets"], runner
    )
    with pytest.raises(RuntimeError, match="invalid layerwise-prefill bank phase"):
        runner._capture_layerwise_prefill_bank_offsets(
            SimpleNamespace(
                kv_connector_metadata=SimpleNamespace(
                    requests={
                        "bad": SimpleNamespace(
                            layerwise_prefill_bank_offset="not-an-int"
                        )
                    }
                )
            )
        )


def test_startup_constraints_and_smoke_tool_are_native():
    platform = source("vllm/platforms/npu.py")
    assert "Layerwise-prefill P nodes require pipeline_parallel_size == 1" in platform
    assert "VLLM_ASCEND_ENABLE_MATMUL_ALLREDUCE" in platform
    tool = source("tools/layerwise_prefill_model_test.py")
    assert '"LMCacheConnectorV1"' in tool
    assert "LMCacheAscendConnector" not in tool
    assert "kv_connector_module_path" not in tool
    ast.parse(tool)


def test_p_node_environment_marker_is_strict_boolean():
    envs = source("vllm/envs_ascend.py")
    assert '"VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE": lambda: _strict_bool_env(' in envs
    tree = ast.parse(envs)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_strict_bool_env"
    )
    namespace = {"os": os}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), "envs_ascend", "exec"),
        namespace,
    )
    parse_bool = namespace["_strict_bool_env"]
    name = "VLLM_ASCEND_LAYERWISE_PREFILL_P_NODE"
    previous = os.environ.get(name)
    try:
        for raw, expected in (("true", True), ("false", False), (" TRUE ", True)):
            os.environ[name] = raw
            assert parse_bool(name) is expected
        for raw in ("1", "yes", "on", "ambiguous", ""):
            os.environ[name] = raw
            with pytest.raises(ValueError, match="must be 'true' or 'false'"):
                parse_bool(name)
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def test_chunk_phase_selects_physical_bank_rows_without_rebuilding_tables():
    runner = source("vllm/v1/worker/npu_model_runner.py")
    assert "_capture_layerwise_prefill_bank_offsets" in runner
    assert "physical_bank = bank ^ phase_value" in runner
    assert "phase_by_request_device" in runner
    assert "torch.where(" in runner
