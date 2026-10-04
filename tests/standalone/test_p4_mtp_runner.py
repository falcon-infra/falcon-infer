# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute production MTP control flow with host-only dependency doubles.

The full runner-state constructor and NPU draft setup/proposal methods are used;
tensor allocation, collectives and model execution are not hardware-validated.
"""

from __future__ import annotations

import ast
import runpy
import threading
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
STATE = "vllm/v1/worker/npu_runner_state.py"
RUNNER = "vllm/v1/worker/npu_model_runner.py"


def load_body(relative: str, owner: str | None, name: str, namespace: dict):
    """Compile an entire production function, not a rewritten branch fragment."""
    source = ROOT / relative
    tree = ast.parse(source.read_text())
    body = tree.body
    if owner is not None:
        body = next(
            node.body
            for node in body
            if isinstance(node, ast.ClassDef) and node.name == owner
        )
    node = next(
        node for node in body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[name]


def fixture(method="mtp", *, last_rank=True, asynchronous=False, tokens=1):
    """Supply explicit config fields and mocks only at device/API boundaries."""
    use_eagle = load_body(
        "vllm/config/speculative.py", "SpeculativeConfig", "use_eagle", {}
    )
    spec = (
        None
        if method is None
        else NS(
            method=method,
            num_speculative_tokens=tokens,
            draft_model_config=NS(max_model_len=64),
            disable_padded_drafter_batch=True,
        )
    )
    if spec is not None:
        spec.use_eagle = lambda: use_eagle(spec)
    model = NS(
        dtype="bfloat16",
        runner_type="generate",
        enable_prompt_embeds=False,
        is_multimodal_raw_input_only_model=False,
        max_model_len=64,
        get_num_attention_heads=lambda _: 8,
        get_inputs_embeds_size=lambda: 32,
        attention_chunk_size=None,
        uses_alibi=False,
        disable_cascade_attn=True,
        is_mm_prefix_lm=False,
        uses_mrope=False,
        uses_xdrope_dim=0,
        logprobs_mode="raw_logprobs",
        logits_processors=None,
        get_vocab_size=lambda: 128,
    )
    config = NS(
        model_config=model,
        cache_config=NS(
            cache_dtype="auto",
            calculate_kv_scales=False,
            block_size=16,
            kv_sharing_fast_prefill=False,
        ),
        offload_config=NS(),
        compilation_config=NS(cudagraph_capture_sizes=[], cudagraph_mode=0),
        lora_config=None,
        load_config=NS(),
        parallel_config=NS(
            decode_context_parallel_size=1,
            distributed_executor_backend="mp",
            cp_kv_cache_interleave_size=1,
        ),
        scheduler_config=NS(
            max_num_batched_tokens=32,
            max_num_seqs=2,
            async_scheduling=asynchronous,
        ),
        speculative_config=spec,
        observability_config=NS(),
    )
    namespace = {
        "np": np,
        "threading": threading,
        "torch": NS(
            npu=NS(Stream=Mock(), Event=Mock()),
            int32="int32",
            int64="int64",
            bool="bool",
            empty=Mock(),
            zeros=Mock(),
            Tensor=np.ndarray,
        ),
        "is_pin_memory_available": lambda: False,
        "kv_cache_dtype_str_to_dtype": lambda *_: "bfloat16",
        "get_pp_group": lambda: NS(is_last_rank=last_rank, ranks=[0]),
        "CacheConfig": NS(DEFAULT_BLOCK_SIZE=16),
        "CUDAGraphMode": NS(NONE=0),
    }
    for name in (
        "Sampler",
        "EagleProposer",
        "RejectionSampler",
        "InputBatch",
        "build_logitsprocs",
        "CudagraphDispatcher",
        "create_offloader",
        "set_offloader",
        "AscendEagleProposer",
        "AscendRejectionSampler",
    ):
        namespace[name] = Mock(name=name)
    runner = NS(_make_buffer=Mock(), _init_device_properties=Mock())
    return config, namespace, runner


def initialize(config, namespace, runner):
    """Run the complete parent constructor followed by native draft setup."""
    load_body(STATE, "NPUModelRunnerState", "__init__", namespace)(
        runner, config, "npu:0"
    )
    namespace["get_spec_decode_method"] = load_body(
        "vllm/v1/spec_decode/ascend/__init__.py",
        None,
        "get_spec_decode_method",
        namespace,
    )
    get_drafter = load_body(RUNNER, "NPUModelRunner", "_get_drafter", namespace)
    runner._get_drafter = lambda: get_drafter(runner)
    load_body(RUNNER, "NPUModelRunner", "_set_up_drafter", namespace)(runner)


@pytest.mark.parametrize("tokens", [1, 2])
@pytest.mark.parametrize("last_rank", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_mtp_runner_constructor_and_native_setup(tokens, last_rank, asynchronous):
    """Valid MTP reaches all post-drafter state and the native NPU proposer."""
    config, ns, runner = fixture(
        tokens=tokens, last_rank=last_rank, asynchronous=asynchronous
    )
    initialize(config, ns, runner)
    assert runner.num_spec_tokens == tokens
    assert runner.uniform_decode_query_len == 1 + tokens
    assert runner.decode_token_per_req == 1 + tokens
    assert runner.effective_drafter_max_model_len == 64
    assert runner.execute_model_state is None
    assert runner.use_aux_hidden_state_outputs is False
    if last_rank:
        ns["EagleProposer"].assert_called_once_with(config, "npu:0", runner)
        ns["RejectionSampler"].assert_called_once_with(runner.sampler)
        ns["AscendEagleProposer"].assert_called_once_with(config, "npu:0", runner)
        assert runner.drafter is ns["AscendEagleProposer"].return_value
        assert runner.rejection_sampler is ns["AscendRejectionSampler"].return_value
    else:
        for name in ("EagleProposer", "RejectionSampler", "AscendEagleProposer"):
            ns[name].assert_not_called()
        assert runner.drafter is None


@pytest.mark.parametrize("asynchronous", [False, True])
def test_non_mtp_runner_does_not_create_draft_components(asynchronous):
    """Ordinary generation still initializes without a speculative config."""
    config, ns, runner = fixture(None, asynchronous=asynchronous)
    initialize(config, ns, runner)
    assert runner.num_spec_tokens == 0
    assert runner.uniform_decode_query_len == 1
    assert runner.decode_token_per_req == 1
    assert runner.drafter is None
    assert runner.execute_model_state is None
    for name in (
        "EagleProposer",
        "RejectionSampler",
        "AscendEagleProposer",
        "AscendRejectionSampler",
    ):
        ns[name].assert_not_called()


@pytest.mark.parametrize(
    "method", ["ngram", "suffix", "medusa", "draft_model", "eagle", "eagle3"]
)
def test_retired_methods_fail_before_proposer_allocation(method):
    """Repairing MTP does not restore standalone EAGLE or other draft methods."""
    config, ns, runner = fixture(method)
    with pytest.raises(ValueError, match="speculative decoding method"):
        initialize(config, ns, runner)
    ns["EagleProposer"].assert_not_called()
    ns["RejectionSampler"].assert_not_called()
    ns["AscendEagleProposer"].assert_not_called()


@pytest.mark.parametrize("decode_step", [False, True])
@pytest.mark.parametrize("padded_batch", [False, True])
def test_mtp_proposal_keeps_staged_graph_identity(decode_step, padded_batch):
    """Execute first-step and decode proposal control flow with host arrays."""
    config, ns, runner = fixture()
    spec = config.speculative_config
    spec.disable_padded_drafter_batch = not padded_batch
    drafter = Mock()
    attention, graph_key = object(), object()
    draft_ids = [[101], [201]]
    drafter._propose.return_value = draft_ids
    drafter.prepare_next_token_ids_cpu.return_value = np.array([10, 20])
    drafter.prepare_next_token_ids_padded.return_value = (
        np.array([10, 20]),
        np.array([1, 1]),
    )
    drafter.prepare_inputs.return_value = (attention, np.array([0, 2]))
    drafter.prepare_inputs_padded.return_value = (
        attention,
        np.array([0, 2]),
        np.array([0, 1]),
        np.array([0, 0]),
    )
    runner.__dict__.update(
        drafter=drafter,
        speculative_config=spec,
        vllm_config=config,
        requests={},
        input_batch=NS(num_reqs=2),
        use_cp=False,
        pcp_size=1,
        input_ids=NS(gpu=np.arange(4)),
        discard_request_indices=NS(gpu=np.array([], dtype=np.int64)),
        num_discarded_requests=0,
        _copy_valid_sampled_token_count=Mock(),
        _get_positions=lambda indices: (
            np.arange(4)[:indices]
            if isinstance(indices, int)
            else np.arange(4)[indices]
        ),
        use_aux_hidden_state_outputs=False,
    )
    propose = load_body(RUNNER, "NPUModelRunner", "propose_draft_token_ids", ns)
    result = propose(
        runner,
        np.array([[10], [20]]) if padded_batch else [[10], [20]],
        object(),
        NS(num_scheduled_tokens={"r0": 2, "r1": 2}),
        NS(num_draft_tokens=[1, 1]) if decode_step else None,
        attention,
        np.arange(4),
        4,
        np.ones((4, 2)),
        target_staged_sfa_graph_key=graph_key,
    )
    assert result is draft_ids
    drafter._propose.assert_called_once()
    assert drafter._propose.call_args.kwargs["target_staged_sfa_graph_key"] is graph_key
    prepare = drafter.prepare_inputs_padded if padded_batch else drafter.prepare_inputs
    if decode_step:
        prepare.assert_called_once()
    else:
        prepare.assert_not_called()


def test_proposal_without_drafter_returns_none():
    """No-MTP/earlier PP ranks must not dereference an absent draft config."""
    _, ns, runner = fixture(None)
    runner.drafter = None
    runner.speculative_config = None
    propose = load_body(RUNNER, "NPUModelRunner", "propose_draft_token_ids", ns)
    assert propose(runner, [], None, None, None, None, None, 0, None) is None


def test_bootstrap_identity_includes_mtp_startup_files():
    """Every container must load the fixed source, not only a matching version."""
    tool = runpy.run_path(str(ROOT / "tools/check_npu_bootstrap.py"))
    assert {
        "config/speculative.py",
        "v1/worker/npu_runner_state.py",
        "v1/worker/npu_model_runner.py",
        "v1/spec_decode/ascend/__init__.py",
    } <= set(tool["IDENTITY_FILES"])
