# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host contracts for P2 platform activation and dispatch; no torch/NPU needed.

Run directly with ``python -B tests/standalone/test_p2_platform.py -v``.
Discovery and plugin modules execute in full against dependency stubs. Platform
methods and dispatch branches execute from their production AST. These checks
do not substitute for full imports, device initialization, or NPU integration.
"""

from __future__ import annotations

import ast
import enum
import importlib.util
import logging
import sys
import tomllib
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


def module(name: str, **attributes) -> ModuleType:
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


def execute_nodes(nodes: list[ast.stmt], namespace: dict) -> dict:
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    tree = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(tree, "<P2 production source>", "exec"), namespace)
    return namespace


def source_node(relative: str, name: str) -> ast.stmt:
    tree = ast.parse((ROOT / relative).read_text())
    return next(node for node in ast.walk(tree) if getattr(node, "name", None) == name)


def load_file(name: str, relative: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


class NativePlatformTests(unittest.TestCase):
    def setUp(self) -> None:
        # Restore every dependency stub after each test, including loaded modules.
        self.enterContext(patch.dict(sys.modules))
        interface = "vllm/platforms/interface.py"
        base = source_node(interface, "Platform")
        base.body = [
            node
            for node in base.body
            if isinstance(node, ast.FunctionDef)
            and (
                node.name.startswith("is_")
                or node.name == "register_builtin_components"
            )
        ]
        namespace = execute_nodes(
            [source_node(interface, "PlatformEnum"), base], {"enum": enum}
        )
        namespace.update(
            ASCEND_QUANTIZATION_METHOD="ascend",
            COMPRESSED_TENSORS_METHOD="compressed-tensors",
        )
        execute_nodes([source_node("vllm/platforms/npu.py", "NPUPlatform")], namespace)
        self.platform = namespace["NPUPlatform"]()
        self.calls = []
        self.envs = module("vllm.envs", VLLM_PLUGINS=None)
        sys.modules.update(
            {
                "vllm": module("vllm", __path__=[], envs=self.envs),
                "vllm.envs": self.envs,
                "vllm.platforms.interface": module(
                    "vllm.platforms.interface",
                    Platform=namespace["Platform"],
                    PlatformEnum=namespace["PlatformEnum"],
                    CpuArchEnum=object,
                ),
                "vllm.utils.import_utils": module(
                    "vllm.utils.import_utils",
                    resolve_obj_by_qualname=Mock(return_value=type(self.platform)),
                ),
                "vllm.utils.torch_utils": module(
                    "vllm.utils.torch_utils", supports_xccl=Mock()
                ),
            }
        )
        self.registry = load_file("vllm.platforms", "vllm/platforms/__init__.py")


    def component_stubs(self) -> None:
        name = "vllm.utils.ascend_profiling_config"
        sys.modules[name] = module(name, generate_service_profiling_config=lambda: self.calls.append("profiling"))

    def plugin_loader(self, entries: list) -> ModuleType:
        self.component_stubs()
        sys.modules["vllm.platforms"] = module(
            "vllm.platforms", current_platform=self.platform
        )
        self.enterContext(
            patch("importlib.metadata.entry_points", return_value=entries)
        )
        return load_file("vllm.plugins", "vllm/plugins/__init__.py")

    def test_native_identity_does_not_impersonate_oot_or_cuda(self) -> None:
        self.assertTrue(self.platform.is_npu())
        self.assertFalse(self.platform.is_out_of_tree())
        self.assertFalse(self.platform.is_cuda_alike())
        self.assertFalse(self.platform.is_cpu())

    def test_discovery_needs_no_plugin_metadata_or_device_probe(self) -> None:
        with (
            patch.object(self.registry, "find_spec", return_value=object()) as find,
            patch("importlib.metadata.entry_points", side_effect=AssertionError),
        ):
            self.assertEqual(
                self.registry.resolve_current_platform_cls_qualname(),
                "vllm.platforms.npu.NPUPlatform",
            )
            find.assert_called_once_with("torch_npu")

    def test_disabling_plugins_does_not_disable_native_platform(self) -> None:
        self.envs.VLLM_PLUGINS = []
        with patch.object(self.registry, "find_spec", return_value=object()):
            self.assertEqual(self.registry.current_platform.device_type, "npu")

    def test_missing_runtime_fails_without_cpu_or_cuda_fallback(self) -> None:
        with (
            patch.object(self.registry, "find_spec", return_value=None),
            patch.object(
                self.registry, "cuda_platform_plugin", side_effect=AssertionError
            ),
            patch.object(
                self.registry, "cpu_platform_plugin", side_effect=AssertionError
            ),
            self.assertRaisesRegex(RuntimeError, "requires torch_npu"),
        ):
            self.registry.resolve_current_platform_cls_qualname()

    def test_runtime_discovery_error_is_not_silenced(self) -> None:
        with (
            patch.object(
                self.registry, "find_spec", side_effect=RuntimeError("broken")
            ),
            self.assertRaisesRegex(RuntimeError, "broken"),
        ):
            self.registry.resolve_current_platform_cls_qualname()

    def test_platform_is_constructed_lazily_once(self) -> None:
        with patch.object(self.registry, "find_spec", return_value=object()) as find:
            find.assert_not_called()
            first = self.registry.current_platform
            self.assertIs(first, self.registry.current_platform)
            find.assert_called_once_with("torch_npu")

    def test_required_components_run_with_empty_metadata_once(self) -> None:
        loader = self.plugin_loader([])
        loader.load_general_plugins()
        loader.load_general_plugins()
        self.assertEqual(self.calls, ["profiling"])

    def test_optional_plugin_filter_does_not_filter_required_components(self) -> None:
        entry = SimpleNamespace(name="optional", value="example:register", load=Mock())
        self.envs.VLLM_PLUGINS = []
        self.plugin_loader([entry]).load_general_plugins()
        entry.load.assert_not_called()
        self.assertEqual(self.calls, ["profiling"])

    def test_optional_plugins_run_after_native_components(self) -> None:
        entry = SimpleNamespace(
            name="optional",
            value="example:register",
            load=lambda: lambda: self.calls.append("optional"),
        )
        self.plugin_loader([entry]).load_general_plugins()
        self.assertEqual(
            self.calls, ["profiling", "optional"]
        )

    def test_required_component_failure_stops_loading(self) -> None:
        loader = self.plugin_loader([])
        with (
            patch.object(
                type(self.platform),
                "register_builtin_components",
                side_effect=RuntimeError("registration failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "registration failed"),
        ):
            loader.load_general_plugins()


    def test_native_package_has_no_legacy_platform_shim(self) -> None:
        self.assertFalse((ROOT / "ascend/vllm_ascend/platform.py").exists())
        self.assertTrue((ROOT / "vllm/platforms/npu.py").is_file())

    def test_npu_custom_op_keeps_ascend_binding(self) -> None:
        method = source_node("vllm/model_executor/custom_op.py", "dispatch_forward")
        namespace = execute_nodes(
            [method],
            {
                "current_platform": self.platform,
                "get_cached_compilation_config": lambda: SimpleNamespace(
                    enabled_custom_ops=set(), disabled_custom_ops=set()
                ),
            },
        )

        class BoundOp:
            name = "test_op"
            _enforce_enable = True
            forward_npu = object()
            forward_cuda = object()

        op = BoundOp()
        self.assertIs(namespace["dispatch_forward"](op, False), op.forward_npu)

    def test_npu_group_devices_keep_local_rank(self) -> None:
        for relative, classname in (
            ("vllm/distributed/parallel_state.py", "GroupCoordinator"),
            ("vllm/distributed/stateless_coordinator.py", "StatelessGroupCoordinator"),
        ):
            cls = source_node(relative, classname)
            constructor = next(
                node for node in cls.body if getattr(node, "name", None) == "__init__"
            )
            if classname == "GroupCoordinator":
                branch = next(
                    node for node in constructor.body
                    if isinstance(node, ast.Assign)
                    and ast.unparse(node.targets[0]) == "self.device"
                )
            else:
                branch = next(
                    node for node in constructor.body
                    if isinstance(node, ast.If)
                    and ast.unparse(node.test) == "current_platform.is_cuda_alike()"
                )
            for rank in (0, 3, 7):
                with self.subTest(file=relative, rank=rank):
                    coordinator = SimpleNamespace()
                    execute_nodes(
                        [branch],
                        {
                            "self": coordinator,
                            "current_platform": self.platform,
                            "torch": SimpleNamespace(device=lambda device: device),
                            "local_rank": rank,
                        },
                    )
                    self.assertEqual(coordinator.device, f"npu:{rank}")

    def test_npu_moe_keeps_existing_ascend_backend(self) -> None:
        relative = "vllm/model_executor/layers/fused_moe/oracle/unquantized.py"
        node = source_node(relative, "select_unquantized_moe_backend")
        backend = enum.Enum("Backend", "NPU OOT CUDA CPU TPU XPU TRITON AITER")
        namespace = execute_nodes(
            [node],
            {
                "current_platform": self.platform,
                "UnquantizedMoeBackend": backend,
                "mk": SimpleNamespace(
                    FusedMoEActivationFormat=SimpleNamespace(Standard=0)
                ),
                "is_supported_config_trtllm_bf16": lambda **kwargs: (
                    False,
                    "unsupported",
                ),
                "has_flashinfer": lambda: False,
                "has_flashinfer_cutlass_fused_moe": lambda: False,
                "rocm_aiter_ops": SimpleNamespace(is_fused_moe_enabled=lambda: False),
                "logger": SimpleNamespace(info_once=lambda *args, **kwargs: None),
            },
        )
        config = SimpleNamespace(
            moe_parallel_config=SimpleNamespace(use_batched_activation_format=False),
            moe_backend="auto",
        )
        self.assertIs(
            namespace["select_unquantized_moe_backend"](config, True, True), backend.NPU
        )

    def test_metadata_no_longer_requires_ascend_entry_points(self) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        self.assertEqual(project["version"], "0.18.0+ascend.p2")
        entries = project["entry-points"]
        self.assertNotIn("vllm.platform_plugins", entries)
        self.assertFalse(
            any(name.startswith("ascend") for name in entries["vllm.general_plugins"])
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    unittest.main()
