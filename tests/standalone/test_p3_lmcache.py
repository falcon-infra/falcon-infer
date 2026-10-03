# SPDX-License-Identifier: Apache-2.0
"""Formal native LMCache lifecycle contracts without importing either runtime."""

import ast
import dataclasses
import importlib
import logging
import uuid
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Literal, get_args
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py"
CONFIG_SOURCE = ROOT / "vllm/config/kv_transfer.py"
LEGACY_CONNECTOR = "LMCacheAscendConnectorV1Dynamic"
LEGACY_MODULE = "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
NATIVE_MODULE = "lmcache.integration.vllm.lmcache_connector_v1"


def load_transfer_config(decorator=dataclasses.dataclass):
    """Execute the production config without importing either device runtime."""
    tree = ast.parse(CONFIG_SOURCE.read_text())
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        or (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id in ("KVProducer", "KVConsumer", "KVRole")
                for target in node.targets
            )
        )
    ]
    namespace = {
        "Any": Any,
        "Literal": Literal,
        "get_args": get_args,
        "field": dataclasses.field,
        "config": decorator,
        "kv_buffer_device_default_factory": lambda: "npu",
        "uuid": uuid,
        "logger": logging.getLogger("test.p3.kv_transfer"),
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(CONFIG_SOURCE), "exec"),
        namespace,
    )
    return namespace["KVTransferConfig"]


@pytest.fixture
def transfer_config():
    return load_transfer_config()


def test_pydantic_validation_and_serialization_preserve_migrated_fields(caplog):
    pydantic = pytest.importorskip("pydantic", minversion="2")
    from pydantic.dataclasses import dataclass

    config = load_transfer_config(
        lambda cls: dataclass(cls, config=pydantic.ConfigDict(extra="forbid"))
    )
    adapter = pydantic.TypeAdapter(config)
    parameters = dict(
        kv_connector=LEGACY_CONNECTOR,
        kv_connector_module_path=LEGACY_MODULE,
        kv_role="kv_both",
        engine_id="p3-pydantic-roundtrip",
        kv_buffer_device="npu",
        kv_connector_extra_config={"use_layerwise": True},
    )
    result = adapter.validate_python(parameters)
    serialized = adapter.dump_python(result)
    assert serialized["kv_connector"] == "LMCacheConnectorV1"
    assert serialized["kv_connector_module_path"] is None
    for key in ("engine_id", "kv_role", "kv_connector_extra_config"):
        assert serialized[key] == parameters[key]
    assert "P3 migrated retired LMCache connector" in caplog.text
    caplog.clear()
    roundtrip = adapter.validate_json(adapter.dump_json(result))
    assert adapter.dump_python(roundtrip) == serialized
    assert not caplog.records
    with pytest.raises(pydantic.ValidationError, match="P3 removed"):
        adapter.validate_python(
            {**parameters, "kv_connector_module_path": "lmcache_ascend.unknown"}
        )
    with pytest.raises(pydantic.ValidationError, match="P3 requires use_native=false"):
        adapter.validate_python(
            {**serialized, "kv_connector_extra_config": {"use_native": True}}
        )
    with pytest.raises(pydantic.ValidationError, match="Unexpected keyword argument"):
        adapter.validate_python({**serialized, "unknown_option": True})


@pytest.mark.parametrize("role", ["kv_producer", "kv_consumer", "kv_both"])
@pytest.mark.parametrize("module_path", [None, LEGACY_MODULE])
def test_legacy_launch_config_migrates_without_changing_runtime_parameters(
    transfer_config, caplog, role, module_path
):
    parameters = dict(
        kv_connector=LEGACY_CONNECTOR,
        kv_connector_module_path=module_path,
        engine_id="p3-instance-dp1-tp8",
        kv_buffer_device="npu",
        kv_buffer_size=123456.0,
        kv_role=role,
        kv_rank=1,
        kv_parallel_size=2,
        kv_ip="127.0.0.2",
        kv_port=14580,
        kv_connector_extra_config={"use_layerwise": True, "nested": {"keep": 42}},
        enable_permute_local_kv=False,
        kv_load_failure_policy="fail",
    )
    result = transfer_config(**parameters)
    expected = {
        **parameters,
        "kv_connector": "LMCacheConnectorV1",
        "kv_connector_module_path": None,
    }
    assert dataclasses.asdict(result) == expected
    assert result.kv_connector_extra_config is parameters["kv_connector_extra_config"]
    assert "P3 migrated retired LMCache connector" in caplog.text
    assert "do not install lmcache-ascend" in caplog.text
    caplog.clear()
    assert dataclasses.asdict(transfer_config(**expected)) == expected
    assert not caplog.records


@pytest.mark.parametrize(
    "name,module_path",
    [
        ("LMCacheConnectorV1", None),
        ("LMCacheAscendConnector", None),
        ("LMCacheConnectorV1Dynamic", NATIVE_MODULE),
        ("CustomConnector", "custom.connector"),
        (LEGACY_CONNECTOR, "custom.connector"),
        (None, None),
    ],
)
def test_native_custom_and_disabled_configs_are_unchanged(
    transfer_config, caplog, name, module_path
):
    result = transfer_config(
        kv_connector=name,
        kv_connector_module_path=module_path,
        kv_role="kv_both" if name else None,
    )
    assert (result.kv_connector, result.kv_connector_module_path) == (name, module_path)
    assert not caplog.records


@pytest.mark.parametrize(
    "name,module_path",
    [
        (LEGACY_CONNECTOR, "lmcache_ascend.custom_connector"),
        ("CustomConnector", LEGACY_MODULE),
        ("CustomConnector", "lmcache_ascend"),
        ("LMCacheConnectorV1", LEGACY_MODULE),
        ("LMCacheConnectorV1Dynamic", LEGACY_MODULE),
    ],
)
def test_unknown_retired_module_config_fails_before_worker_start(
    transfer_config, name, module_path
):
    with pytest.raises(ValueError, match="P3 removed the lmcache_ascend package"):
        transfer_config(
            kv_connector=name,
            kv_connector_module_path=module_path,
            kv_role="kv_both",
        )


@pytest.mark.parametrize(
    "name,module_path",
    [
        (LEGACY_CONNECTOR, LEGACY_MODULE),
        ("LMCacheConnectorV1", None),
        ("LMCacheAscendConnector", None),
        ("LMCacheConnectorV1Dynamic", NATIVE_MODULE),
    ],
)
def test_retired_vendored_adapter_is_rejected_at_config_time(
    transfer_config, name, module_path
):
    with pytest.raises(ValueError, match="P3 requires use_native=false"):
        transfer_config(
            kv_connector=name,
            kv_connector_module_path=module_path,
            kv_role="kv_both",
            kv_connector_extra_config={"use_native": True},
        )


def test_migrated_config_resolves_native_factory_without_old_module_import(
    transfer_config,
):
    path = ROOT / "vllm/distributed/kv_transfer/kv_connector/factory.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    registrations = [
        n
        for n in tree.body
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and ast.unparse(n.value.func) == "KVConnectorFactory.register_connector"
    ]
    native = type("LMCacheConnectorV1", (), {})
    imports = Mock(spec=importlib)
    imports.import_module.return_value = NS(LMCacheConnectorV1=native)
    namespace = {"importlib": imports}
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[prefix, cls, *registrations], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    config = transfer_config(
        kv_connector=LEGACY_CONNECTOR,
        kv_connector_module_path=LEGACY_MODULE,
        kv_role="kv_both",
    )
    imports.import_module.assert_not_called()
    assert namespace["KVConnectorFactory"].get_connector_class(config) is native
    imports.import_module.assert_called_once_with(
        "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector"
    )


def connector():
    tree = ast.parse(SOURCE.read_text())
    classes = [
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef)
        and n.name in ("LMCacheKVEvents", "LMCacheConnectorV1")
    ]
    aggregator_path = ROOT / "vllm/distributed/kv_events.py"
    aggregator = next(
        n
        for n in ast.parse(aggregator_path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "KVEventAggregator"
    )
    namespace = {
        "KVConnectorBase_V1": type("Base", (), {}),
        "SupportsHMA": type("HMA", (), {}),
        "KVConnectorKVEvents": type("Events", (), {}),
        "Counter": Counter,
    }
    prefix = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[prefix, aggregator, *classes], type_ignores=[])
            ),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    wrapper = object.__new__(namespace["LMCacheConnectorV1"])
    wrapper._kv_cache_events = None
    return wrapper


def kv_events(wrapper, values):
    """Use the production event container/aggregator with hashable host fixtures."""
    cls = wrapper.update_connector_output.__globals__["LMCacheKVEvents"]
    events = cls(num_workers=1)
    events.add_events(values)
    return events


@pytest.mark.parametrize("kind", ["none", "falsey", "foreign", "empty", "events"])
def test_worker_output_is_forwarded_even_without_kv_events(kind):
    wrapper = connector()
    events = {
        "none": None,
        "falsey": [],
        "foreign": object(),
        "empty": kv_events(wrapper, []),
        "events": kv_events(wrapper, ["block"]),
    }[kind]
    output = NS(
        finished_recving={"cold-request"},
        finished_sending={"sent-request"},
        invalid_block_ids={7},
        completed_decode_window_saves={"decode-request": 256},
        kv_cache_events=events,
    )
    before = vars(output).copy()
    wrapper._lmcache_engine = NS(update_connector_output=Mock())
    wrapper.update_connector_output(output)
    wrapper._lmcache_engine.update_connector_output.assert_called_once_with(output)
    assert wrapper._lmcache_engine.update_connector_output.call_args.args[0] is output
    assert vars(output) == before
    assert wrapper._kv_cache_events is (events if kind in ("empty", "events") else None)


def test_output_forwarding_preserves_kv_event_aggregation_and_drain():
    wrapper = connector()
    first = kv_events(wrapper, ["common", "worker-1"])
    second = kv_events(wrapper, ["common", "worker-2"])
    outputs = [NS(kv_cache_events=first), NS(kv_cache_events=second)]
    observed = []
    wrapper._lmcache_engine = NS(
        update_connector_output=lambda out: observed.append(
            (out, wrapper._kv_cache_events)
        )
    )
    for output in outputs:
        wrapper.update_connector_output(output)
    assert observed == [(outputs[0], None), (outputs[1], first)]
    assert wrapper._kv_cache_events is first
    assert first.get_number_of_workers() == 2
    assert Counter(first.get_all_events()) == Counter(
        ["common", "worker-1", "common", "worker-2"]
    )
    assert list(wrapper.take_events()) == ["common"]
    assert wrapper._kv_cache_events is None
    assert list(wrapper.take_events()) == []


@pytest.mark.parametrize("has_events", [False, True])
def test_adapter_completion_errors_are_not_hidden_by_kv_events(has_events):
    wrapper = connector()
    existing = kv_events(wrapper, ["existing"])
    wrapper._kv_cache_events = existing
    output = NS(kv_cache_events=kv_events(wrapper, ["new"]) if has_events else None)
    wrapper._lmcache_engine = NS(
        update_connector_output=Mock(side_effect=RuntimeError("invalid completion"))
    )
    with pytest.raises(RuntimeError, match="invalid completion"):
        wrapper.update_connector_output(output)
    wrapper._lmcache_engine.update_connector_output.assert_called_once_with(output)
    assert wrapper._kv_cache_events is existing
    assert existing.get_all_events() == ["existing"]
    assert existing.get_number_of_workers() == 1


@pytest.mark.parametrize(
    "layerwise,role,dsa,sparse,expected",
    [
        (True, "kv_both", True, True, True),
        (True, "kv_consumer", True, True, True),
        (True, "kv_producer", True, True, False),
        (False, "kv_both", True, True, False),
        (True, "kv_both", False, True, False),
        (True, "kv_both", True, False, False),
    ],
)
def test_staged_sfa_capability_is_explicit(layerwise, role, dsa, sparse, expected):
    wrapper = connector()
    wrapper._lmcache_engine = NS(
        use_layerwise=layerwise,
        kv_role=role,
        config=NS(dsa_two_groups=dsa, enable_sparse_attention=sparse),
    )
    assert wrapper.supports_dsa_index_lmcache
    assert wrapper.uses_layerwise_model_callbacks == layerwise
    assert wrapper.supports_staged_sfa_sparse_load == expected


def test_preemption_failure_still_clears_metadata():
    wrapper = connector()
    calls = []

    def fail(ids):
        calls.append(ids)
        raise RuntimeError("fixture")

    wrapper._lmcache_engine = NS(handle_preemptions=fail)
    wrapper.bind_connector_metadata = lambda value: calls.append(value)
    wrapper.clear_connector_metadata = lambda: calls.append("clear")
    with pytest.raises(RuntimeError, match="fixture"):
        wrapper.handle_preemptions_with_metadata({"r"}, "snapshot")
    assert calls == ["snapshot", {"r"}, "clear"]


def test_seal_handoff_and_restart_are_forwarded_without_patching():
    wrapper = connector()
    calls = []
    wrapper._lmcache_engine = NS(
        capture_live_source_event_handoff=lambda ctx: (calls.append(ctx), True)[1],
        seal_sparse_destination_layout=lambda: calls.append("seal"),
        lmcache_engine=NS(
            remote_fill_requires_paired_restart=lambda: True,
            get_remote_fill_placement_info=lambda: {"tp": 8},
            get_remote_fill_metrics=lambda: {"pending": 1},
        ),
    )
    assert wrapper.capture_live_source_event_handoff("producer_event")
    wrapper.seal_sparse_destination_layout()
    assert wrapper.remote_fill_requires_paired_restart()
    assert wrapper.get_remote_fill_placement_info() == {"tp": 8}
    assert wrapper.get_remote_fill_metrics() == {"pending": 1}
    assert calls == ["producer_event", "seal"]


def test_optional_lmcache_is_not_imported_at_module_scope():
    tree = ast.parse(SOURCE.read_text())
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("lmcache")
        elif isinstance(node, ast.Import):
            assert not any(a.name.startswith("lmcache") for a in node.names)
    factory = (
        ROOT / "vllm/distributed/kv_transfer/kv_connector/factory.py"
    ).read_text()
    assert (
        '"LMCacheAscendConnector",\n    '
        '"vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector"' in factory
    )
    assert "lmcache_ascend_connector" not in factory
