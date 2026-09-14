# SPDX-License-Identifier: Apache-2.0
"""CPU execution of native SFA query-topology routing contracts."""

import ast
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

np = pytest.importorskip("numpy")


@dataclass(frozen=True)
class BatchDescriptor:
    num_tokens: int


def definitions(path, names, namespace, *, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if class_name:
        tree = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
    body = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(body) == len(names)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            *body,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def routing():
    root = Path(__file__).resolve().parents[3]
    env = SimpleNamespace(VLLM_ASCEND_SFA_FULL_GRAPH=True)
    modes = SimpleNamespace(NONE="none", PIECEWISE="piecewise", FULL="full")
    states = SimpleNamespace(
        DecodeOnly="decode", SpecDecoding="spec", ChunkedPrefill="prefill"
    )
    ns = dict(
        np=np,
        dataclass=dataclass,
        Enum=Enum,
        Any=object,
        BatchDescriptor=BatchDescriptor,
        CUDAGraphMode=modes,
        AscendAttentionState=states,
        envs_ascend=env,
        staged_sfa_graph_configured=lambda _: True,
        sfa_full_graph_enabled=lambda _: env.VLLM_ASCEND_SFA_FULL_GRAPH,
        cold_perf_enabled=lambda: False,
        logger=SimpleNamespace(info=lambda *args: None),
        unwrap_staged_sfa_connector_metadata=lambda value: value,
        staged_sfa_metadata_sparse_route=lambda metadata, request_ids: (
            ns["StagedSFARouteReason"].ELIGIBLE,
            (4096,) * len(request_ids),
            (False,) * len(request_ids),
        ),
    )
    definitions(
        root / "vllm/ascend_forward_context.py",
        {"StagedSFAQueryProfile", "StagedSFAGraphKey"},
        ns,
    )
    definitions(
        root / "vllm/utils/ascend.py",
        {
            "StagedSFARouteAction",
            "StagedSFARouteReason",
            "StagedSFARouteDecision",
        },
        ns,
    )
    names = {
        "_staged_sfa_local_route",
        "_staged_sfa_live_route",
        "_apply_staged_sfa_route",
        "_coordinate_sfa_full_graph_preparation",
        "_staged_sfa_capture_keys",
        "_staged_sfa_dummy_graph_key",
        "_warmup_and_capture",
    }
    definitions(
        root / "vllm/v1/worker/npu_model_runner.py",
        names,
        ns,
        class_name="NPUModelRunner",
    )
    runner_type = type("RunnerRouting", (), {name: ns[name] for name in names})
    runner = runner_type()
    runner.speculative_config = SimpleNamespace(num_speculative_tokens=1)
    runner.decode_threshold = 2
    runner.attn_state = states.SpecDecoding
    runner.vllm_config = SimpleNamespace(lora_config=None)
    runner.parallel_config = SimpleNamespace(data_parallel_size=4)
    runner._staged_sfa_graph_capture_sizes = (2, 4, 8)
    ns["NPUModelRunnerState"] = SimpleNamespace(
        _warmup_and_capture=Mock(return_value="captured")
    )
    return runner, ns, env, modes, states


def route(runner, *, widths):
    return runner._staged_sfa_local_route(
        num_tokens_unpadded=int(sum(widths)),
        num_reqs=len(widths),
        num_scheduled_tokens=np.asarray(widths, dtype=np.int32),
        index_topk=2048,
        has_cascade_attention=False,
        request_ids=list(map(str, range(len(widths)))),
        kv_connector_metadata=None,
    )


@pytest.mark.parametrize(
    "widths,profile",
    [
        ([2, 2, 2], "spec_fixed"),
        ([1, 1, 1], "decode_bounded"),
        ([2, 1, 2], "decode_bounded"),
    ],
)
def test_every_request_width_selects_the_matching_profile(routing, widths, profile):
    runner, ns, _, modes, _ = routing
    runner._staged_sfa_graph_capture_sizes = (8,)
    runner.attn_state = (
        ns["AscendAttentionState"].DecodeOnly
        if max(widths) == 1
        else ns["AscendAttentionState"].SpecDecoding
    )
    local = route(runner, widths=widths)
    assert local.action == ns["StagedSFARouteAction"].STAGED
    assert local.uniform_query_len == (2 if widths == [2, 2, 2] else 0)
    live = runner._staged_sfa_live_route(
        local_route=local,
        dp_route_action=local.action,
        cudagraph_mode=modes.PIECEWISE,
        batch_descriptor=BatchDescriptor(8),
        num_tokens_unpadded=sum(widths),
        num_tokens_padded=8,
        num_reqs=len(widths),
        should_ubatch=False,
    )
    assert live.graph_key.query_profile.value == profile


@pytest.mark.parametrize("widths", [[2, 0], [2, 3], [2, -1], [2]])
def test_invalid_width_vectors_cannot_select_uniform_profile(routing, widths):
    runner, ns, _, _, _ = routing
    runner._staged_sfa_graph_capture_sizes = (4,)
    local = runner._staged_sfa_local_route(
        num_tokens_unpadded=4,
        num_reqs=2,
        num_scheduled_tokens=np.asarray(widths, dtype=np.int32),
        index_topk=2048,
        has_cascade_attention=False,
        request_ids=["a", "b"],
        kv_connector_metadata=None,
    )
    assert local.reason == ns["StagedSFARouteReason"].NON_Q1
    assert local.uniform_query_len == 0


def test_capture_keys_share_capacity_but_not_topology(routing):
    runner, _, _, _, _ = routing
    keys = runner._staged_sfa_capture_keys(8)
    assert [key.query_profile.value for key in keys] == [
        "spec_fixed",
        "decode_bounded",
    ]
    assert {key.token_capacity for key in keys} == {8}
    assert {key.request_capacity for key in keys} == {4}


def test_warmup_captures_both_profiles_and_restores_selector(routing):
    runner, ns, _, modes, _ = routing
    parent = Mock(side_effect=lambda owner, desc, **kwargs: kwargs["profile_seq_lens"])
    ns["NPUModelRunnerState"] = SimpleNamespace(_warmup_and_capture=parent)
    result = runner._warmup_and_capture(
        SimpleNamespace(num_tokens=8, uniform=False),
        modes.PIECEWISE,
        profile_seq_lens=6144,
    )
    assert result is None
    assert parent.call_count == 2
    assert [call.args[1].num_tokens for call in parent.call_args_list] == [8, 8]
    assert getattr(runner, "_sfa_capture_graph_key", None) is None
    runner._sfa_capture_graph_key = runner._staged_sfa_capture_keys(8)[0]
    assert runner._staged_sfa_dummy_graph_key(8, dp_idle=False) == runner._sfa_capture_graph_key
    assert runner._staged_sfa_dummy_graph_key(8, dp_idle=True).query_profile.value == "decode_bounded"


def test_warmup_selector_is_restored_after_capture_failure(routing):
    runner, ns, _, modes, _ = routing
    ns["NPUModelRunnerState"] = SimpleNamespace(
        _warmup_and_capture=Mock(side_effect=RuntimeError("capture error"))
    )
    with pytest.raises(RuntimeError, match="capture error"):
        runner._warmup_and_capture(
            SimpleNamespace(num_tokens=8, uniform=False), modes.PIECEWISE
        )
    assert getattr(runner, "_sfa_capture_graph_key", None) is None
