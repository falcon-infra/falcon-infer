# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute production RoPE/load dispatch with host-only dependency doubles.

No model weights, torch, CANN, Ray cluster or network are used. These tests cover
branch selection, argument preservation and iteration, not tensor/kernel math.
"""

from __future__ import annotations

import ast
import fnmatch
import posixpath
import runpy
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import Mock, mock_open

import pytest

ROOT = Path(__file__).resolve().parents[2]
ROPE = "vllm/model_executor/layers/rotary_embedding/__init__.py"
LOADER = "vllm/model_executor/model_loader/default_loader.py"
WEIGHTS = "vllm/model_executor/model_loader/weight_utils.py"
RAY = "vllm/v1/executor/ray_utils.py"


def load_function(relative, name, namespace, owner=None):
    """Use the complete production body, leaving all branch structure intact."""
    path = ROOT / relative
    body = ast.parse(path.read_text()).body
    if owner:
        body = next(
            n.body for n in body if isinstance(n, ast.ClassDef) and n.name == owner
        )
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


@pytest.fixture
def rope_factory():
    names = (
        "RotaryEmbedding",
        "LinearScalingRotaryEmbedding",
        "YaRNScalingRotaryEmbedding",
        "DeepseekScalingRotaryEmbedding",
    )
    constructors = {
        name: Mock(name=name, side_effect=lambda *a, **kw: NS(args=a, kwargs=kw))
        for name in names
    }
    namespace = {
        **constructors,
        "torch": NS(get_default_dtype=lambda: "float32"),
        "_ROPE_DICT": {},
    }
    return (
        load_function(ROPE, "get_rope", namespace),
        constructors,
        namespace["_ROPE_DICT"],
    )


@pytest.mark.parametrize(
    "kind,owner",
    [
        ("default", "RotaryEmbedding"),
        ("linear", "LinearScalingRotaryEmbedding"),
        ("yarn", "YaRNScalingRotaryEmbedding"),
        ("deepseek_yarn", "DeepseekScalingRotaryEmbedding"),
        ("deepseek_llama_scaling", "DeepseekScalingRotaryEmbedding"),
    ],
)
@pytest.mark.parametrize("neox", [False, True])
def test_retained_rope_constructs_and_caches_once(rope_factory, kind, owner, neox):
    get_rope, constructors, cache = rope_factory
    parameters = {
        "rope_type": kind,
        "rope_theta": 1000000,
        "partial_rotary_factor": 0.5,
    }
    if kind != "default":
        parameters.update(factor=4.0, original_max_position_embeddings=8192)
    if kind in ("yarn", "deepseek_yarn", "deepseek_llama_scaling"):
        parameters.update(beta_fast=32, beta_slow=1, attn_factor=1.2)
    original = parameters.copy()
    result = get_rope(64, 32768, neox, parameters, "bfloat16")
    assert get_rope(64, 32768, neox, parameters.copy(), "bfloat16") is result
    position = (
        8192 if kind in ("yarn", "deepseek_yarn", "deepseek_llama_scaling") else 32768
    )
    expected_args = (64, 32, position, 1000000, neox)
    expected_args += ("bfloat16",) if kind == "default" else (4.0, "bfloat16")
    expected_kwargs = {
        key: parameters[key]
        for key in ("beta_fast", "beta_slow", "attn_factor")
        if key in parameters
    }
    constructors[owner].assert_called_once_with(*expected_args, **expected_kwargs)
    for name, constructor in constructors.items():
        if name != owner:
            constructor.assert_not_called()
    assert list(cache.values()) == [result]
    assert parameters == original


@pytest.mark.parametrize("parameters", [None, {}, {"rope_theta": 500000}])
def test_missing_rope_type_uses_default_and_default_dtype(rope_factory, parameters):
    get_rope, constructors, _ = rope_factory
    get_rope(64, 128, rope_parameters=parameters)
    constructors["RotaryEmbedding"].assert_called_once_with(
        64, 64, 128, (parameters or {}).get("rope_theta", 10000), True, "float32"
    )


@pytest.mark.parametrize(
    "kind",
    [
        "llama3",
        "mllama4",
        "ntk",
        "dynamic",
        "xdrope",
        "longrope",
        "openpangu",
        "unknown",
    ],
)
def test_retired_and_unknown_rope_types_still_fail(rope_factory, kind):
    get_rope, constructors, cache = rope_factory
    with pytest.raises(ValueError, match="Unknown RoPE scaling type"):
        get_rope(64, 128, rope_parameters={"rope_type": kind})
    assert cache == {}
    assert not any(constructor.called for constructor in constructors.values())


@pytest.mark.parametrize(
    "parameters,dual_chunk",
    [
        ({"mrope_section": [8, 8]}, None),
        ({"xdrope_section": [8, 8]}, None),
        ({"use_fope": True}, None),
        ({}, {}),
    ],
)
def test_other_model_rope_features_still_fail(rope_factory, parameters, dual_chunk):
    get_rope, constructors, _ = rope_factory
    with pytest.raises(ValueError, match="only GLM text RoPE"):
        get_rope(
            64, 128, rope_parameters=parameters, dual_chunk_attention_config=dual_chunk
        )
    assert not any(constructor.called for constructor in constructors.values())


@pytest.mark.parametrize("factor", [-1.0, 0.0, 1.1])
def test_invalid_partial_rotary_factor_still_fails(rope_factory, factor):
    get_rope, _, _ = rope_factory
    with pytest.raises(ValueError, match="partial_rotary_factor"):
        get_rope(64, 128, rope_parameters={"partial_rotary_factor": factor})


@pytest.mark.parametrize(
    "change",
    [
        {"head_size": 128},
        {"max_position": 256},
        {"is_neox_style": False},
        {"dtype": "bfloat16"},
        {"rope_parameters": {"rope_theta": 500000}},
    ],
)
def test_rope_cache_separates_incompatible_inputs(rope_factory, change):
    get_rope, constructors, cache = rope_factory
    kwargs = {"head_size": 64, "max_position": 128}
    first = get_rope(**kwargs)
    second = get_rope(**(kwargs | change))
    assert first is not second
    assert len(cache) == constructors["RotaryEmbedding"].call_count == 2


def prepare_weights_fixture(format_name, files, fallback=False):
    """Keep actual loader decisions; replace only filesystem/HF boundaries."""
    folder = "/approved/glm52"
    namespace = {
        "os": NS(path=NS(isdir=lambda _: True, join=posixpath.join)),
        "glob": NS(
            glob=lambda pattern: [
                folder + "/" + name
                for name in files
                if fnmatch.fnmatch(folder + "/" + name, pattern)
            ]
        ),
        "maybe_download_from_modelscope": lambda *_: None,
        "list_filtered_repo_files": lambda **_: [],
        "SAFE_WEIGHTS_INDEX_NAME": "model.safetensors.index.json",
        "filter_duplicate_safetensors_files": lambda found, *_: found,
        "filter_files_not_needed_for_inference": lambda found: found,
    }
    runner = NS(load_config=NS(load_format=format_name))
    prepare = load_function(LOADER, "_prepare_weights", namespace, "DefaultModelLoader")
    return prepare(runner, folder, None, None, fallback, None)


@pytest.mark.parametrize(
    "format_name,files,fallback,expected_safe",
    [
        ("auto", ["model.safetensors"], False, True),
        ("hf", ["model.safetensors"], False, True),
        ("safetensors", ["model.safetensors"], False, True),
        ("auto", ["model.bin"], False, False),
        ("hf", ["model.bin"], False, False),
        ("pt", ["model.pt"], False, False),
        ("npcache", ["model.bin"], False, False),
        ("hf", ["model.pt"], True, False),
    ],
)
def test_retained_weight_formats_reach_file_selection(
    format_name, files, fallback, expected_safe
):
    folder, actual, is_safe = prepare_weights_fixture(format_name, files, fallback)
    assert actual == [folder + "/" + name for name in files]
    assert is_safe is expected_safe


def test_loader_unknown_format_and_missing_weights_still_fail():
    with pytest.raises(ValueError, match="Unknown load_format"):
        prepare_weights_fixture("unknown", ["model.safetensors"])
    with pytest.raises(RuntimeError, match="Cannot find any model weights"):
        prepare_weights_fixture("auto", [])


@pytest.mark.parametrize("strategy", ["eager", "lazy", "prefetch"])
def test_weight_iterator_yields_each_selected_tensor_once(strategy):
    tensors = {"shared": object(), "local_expert": object(), "other_expert": object()}
    safe_open = Mock(
        side_effect=lambda *_, **__: nullcontext(
            NS(keys=lambda: tensors.keys(), get_tensor=tensors.__getitem__)
        )
    )
    load = Mock(return_value=tensors)
    prefetch = Mock()
    namespace = {
        "_natural_sort_key": str,
        "_prefetch_all_checkpoints": prefetch,
        "tqdm": lambda values, **_: values,
        "enable_tqdm": lambda _: False,
        "_BAR_FORMAT": "",
        "open": mock_open(read_data=b"fixture, not real weights"),
        "load": load,
        "safe_open": safe_open,
        "should_skip_weight": lambda name, _: name == "other_expert",
    }
    iterator = load_function(WEIGHTS, "safetensors_weights_iterator", namespace)
    assert list(iterator(["model.safetensors"], False, strategy, {0})) == [
        ("shared", tensors["shared"]),
        ("local_expert", tensors["local_expert"]),
    ]
    assert load.call_count == (1 if strategy == "eager" else 0)
    assert safe_open.call_count == (0 if strategy == "eager" else 1)
    assert prefetch.call_count == (1 if strategy == "prefetch" else 0)


@pytest.mark.parametrize("initialized", [False, True])
def test_ray_connection_is_initialized_at_most_once(monkeypatch, initialized):
    platform_module = ModuleType("vllm.platforms")
    platform_module.current_platform = NS(ray_device_key="NPU")
    monkeypatch.setitem(sys.modules, "vllm.platforms", platform_module)
    ray = NS(is_initialized=lambda: initialized, init=Mock())
    namespace = {
        "ray": ray,
        "assert_ray_available": Mock(),
        "logger": Mock(),
        "_verify_bundles": Mock(),
    }
    group = NS(bundle_specs=[{"NPU": 1} for _ in range(8)])
    config = NS(placement_group=group, world_size=8, ray_runtime_env={})
    initialize = load_function(RAY, "initialize_ray_cluster", namespace)
    initialize(config, "existing-cluster")
    assert ray.init.call_count == (0 if initialized else 1)
    assert config.placement_group is group
    namespace["_verify_bundles"].assert_called_once_with(group, config, "NPU")


def test_bootstrap_identity_covers_model_loading_files():
    tool = runpy.run_path(str(ROOT / "tools/check_npu_bootstrap.py"))
    assert {path.removeprefix("vllm/") for path in (ROPE, LOADER, WEIGHTS, RAY)} <= set(
        tool["IDENTITY_FILES"]
    )
