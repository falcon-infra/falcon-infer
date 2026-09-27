# SPDX-License-Identifier: Apache-2.0
"""Host contracts for native NPU ownership and state; no torch/CANN import."""

from __future__ import annotations

import ast
import copy
import dataclasses
import importlib.util
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


def source(relative):
    return ast.parse((ROOT / relative).read_text())


def execute(nodes, namespace):
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, "<native production contract>", "exec"), namespace)
    return namespace


def load(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location("p2_contract_module", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def stub(name, **attrs):
    value = ModuleType(name)
    value.__dict__.update(attrs)
    return value


class NativeContracts(unittest.TestCase):
    def test_all_layer_targets_exist_and_tables_are_immutable(self):
        registry = load("vllm/model_executor/layers/ascend/registry.py")
        for entries in (registry.NPU_LAYERS, registry.NPU_310P_LAYERS):
            for name, (module, cls) in entries.items():
                with self.subTest(layer=name):
                    declarations = source(module.replace(".", "/") + ".py")
                    self.assertIn(
                        cls,
                        {
                            n.name
                            for n in declarations.body
                            if isinstance(n, ast.ClassDef)
                        },
                    )
            with self.assertRaises(TypeError):
                entries["RMSNorm"] = ("invalid", "invalid")

    def test_router_binding_tracks_each_model_without_shared_registration(self):
        registry = load("vllm/model_executor/layers/ascend/registry.py")
        native = type("AscendGateLinear", (), {})
        config = SimpleNamespace(model_config=SimpleNamespace(hf_text_config=None))
        modules = {
            "vllm.config": stub("vllm.config", get_current_vllm_config=lambda: config),
            "vllm.utils.ascend": stub("vllm.utils.ascend", is_310p=lambda: False),
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(
                registry,
                "import_module",
                return_value=SimpleNamespace(AscendGateLinear=native),
            ),
        ):
            for model, dtype, expected in [
                ("glm_moe_dsa", None, native),
                ("other_moe", None, None),
                ("other_moe", "float32", native),
                ("deepseek_v3", None, None),
            ]:
                config.model_config.hf_text_config = SimpleNamespace(
                    model_type=model, moe_router_dtype=dtype
                )
                self.assertIs(registry.get_npu_layer_class("GateLinear"), expected)
            self.assertIsNone(registry.get_npu_layer_class("AscendGateLinear"))

    def test_mla_layout_and_merge_preserve_separate_latent_and_indexer(self):
        names = {
            "KVCacheSpec",
            "AttentionSpec",
            "FullAttentionSpec",
            "MLAAttentionSpec",
        }
        nodes = [
            n
            for n in source("vllm/v1/kv_cache_interface.py").body
            if isinstance(n, ast.ClassDef) and n.name in names
        ]
        from abc import ABC, abstractmethod

        types = SimpleNamespace(int8="int8", float16="float16")
        ns = execute(
            nodes,
            {
                "dataclass": dataclasses.dataclass,
                "copy": copy,
                "torch": types,
                "ABC": ABC,
                "abstractmethod": abstractmethod,
                "get_dtype_size": lambda dtype: 1 if dtype == "int8" else 2,
            },
        )
        spec = ns["MLAAttentionSpec"]
        latent = spec(block_size=128, num_kv_heads=1, head_size=576, dtype="bfloat16")
        indexer = spec(block_size=128, num_kv_heads=1, head_size=128, dtype="bfloat16")
        self.assertEqual(latent.page_size_bytes, 128 * 576 * 2)
        self.assertEqual(indexer.page_size_bytes, 128 * 128 * 2)
        self.assertEqual(spec.merge([latent, latent]), latent)
        with self.assertRaisesRegex(AssertionError, "head_size"):
            spec.merge([latent, indexer])
        quantized = spec(
            block_size=128,
            num_kv_heads=1,
            head_size=704,
            dtype="bfloat16",
            sparse_head_dim=(512, 64, 128),
            cache_sparse_c8=True,
        )
        self.assertEqual(quantized.page_size_bytes, 128 * (576 * 2 + 128 + 2))

    def test_metadata_is_delivered_before_reading_token_outputs(self):
        scheduler = next(
            n
            for n in source("vllm/v1/core/sched/scheduler.py").body
            if isinstance(n, ast.ClassDef) and n.name == "Scheduler"
        )
        method = next(
            n
            for n in scheduler.body
            if getattr(n, "name", None) == "update_from_output"
        )
        connector = SimpleNamespace(update_connector_worker_metadata=Mock())
        state = SimpleNamespace(
            connector=connector,
            requests={
                "active": SimpleNamespace(is_finished=lambda: False),
                "finished": SimpleNamespace(is_finished=lambda: True),
            },
        )
        metadata = object()
        output = SimpleNamespace(
            kv_connector_output=SimpleNamespace(kv_connector_worker_meta=metadata)
        )
        execute(method.body[:2], {"self": state, "model_runner_output": output})
        connector.update_connector_worker_metadata.assert_called_once_with(
            metadata, {"active"}
        )
        self.assertIn("sampled_token_ids", ast.unparse(method.body[2]))
        output.kv_connector_output = None
        execute(method.body[:2], {"self": state, "model_runner_output": output})
        self.assertEqual(connector.update_connector_worker_metadata.call_count, 1)

    def test_async_intermediate_tensors_wait_exactly_once_before_access(self):
        cls = next(
            n
            for n in source("vllm/v1/worker/runner_output.py").body
            if isinstance(n, ast.ClassDef) and n.name == "AsyncIntermediateTensors"
        )

        class Intermediate:
            def __init__(self, tensors):
                self.tensors = tensors

        ns = execute([cls], {"IntermediateTensors": Intermediate})
        events = []
        pending = ns[cls.name](
            {"hidden": "value"},
            [SimpleNamespace(wait=lambda: events.append("wait"))],
            [lambda: events.append("postprocess")],
        )
        self.assertEqual(events, [])
        self.assertEqual(pending.tensors, {"hidden": "value"})
        pending.wait_for_comm()
        self.assertEqual(pending.tensors["hidden"], "value")
        self.assertEqual(events, ["wait", "postprocess"])

    def test_current_stream_does_not_cache_a_stale_npu_stream(self):
        fn = next(
            n
            for n in source("vllm/utils/torch_utils.py").body
            if getattr(n, "name", None) == "current_stream"
        )
        streams = iter([object(), object()])
        platform = SimpleNamespace(is_npu=lambda: True)
        get_stream = Mock(side_effect=lambda: next(streams))
        ns = execute(
            [fn],
            {"torch": SimpleNamespace(npu=SimpleNamespace(current_stream=get_stream))},
        )
        with patch.dict(
            sys.modules,
            {"vllm.platforms": stub("vllm.platforms", current_platform=platform)},
        ):
            first, second = ns[fn.name](), ns[fn.name]()
        self.assertIsNot(first, second)
        self.assertEqual(get_stream.call_count, 2)

    def test_compiler_dependency_bindings_are_isolated_across_threads(self):
        bind = load("vllm/compilation/ascend/torchair_backend.py").bind_callable
        scope = {"converter": lambda value: ("original", value)}
        exec("def compile_graph(value=0):\n    return converter(value)\n", scope)
        original = scope["compile_graph"]
        first = bind(original, converter=lambda value: ("first", value))
        second = bind(original, converter=lambda value: ("second", value))
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(lambda item: item[0](item[1]), [(first, 1), (second, 2)])
            )
        self.assertEqual(results, [("first", 1), ("second", 2)])
        self.assertEqual(original(), ("original", 0))
        with self.assertRaisesRegex(RuntimeError, "dependency ABI"):
            bind(original, absent=lambda: None)

    def test_connector_registration_does_not_import_lmcache(self):
        declarations = source("vllm/distributed/kv_transfer/kv_connector/factory.py")
        registrations = [
            n
            for n in declarations.body
            if isinstance(n, ast.Expr)
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == "register_connector"
        ]
        entries = {}

        def register(name, module, cls):
            self.assertNotIn(name, entries)
            entries[name] = (module, cls)

        before = set(sys.modules)
        execute(
            registrations,
            {"KVConnectorFactory": SimpleNamespace(register_connector=register)},
        )
        self.assertEqual(set(sys.modules), before)
        self.assertEqual(
            entries["MultiConnector"],
            (
                "vllm.distributed.kv_transfer.ascend.ascend_multi_connector",
                "AscendMultiConnector",
            ),
        )
        self.assertIn("LMCacheAscendConnector", entries)


if __name__ == "__main__":
    unittest.main()
