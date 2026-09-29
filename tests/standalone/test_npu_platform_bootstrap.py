# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host cold-import contracts for the native platform bootstrap.

Each child executes the complete production platforms/__init__.py and npu.py,
including their imports. Torch, the base interface and the configuration/model
dependency edges are fixtures, not a real installed-runtime or NPU test.
"""

from __future__ import annotations

import ast
import enum
import importlib
import importlib.abc
import importlib.util
import json
import logging
import subprocess
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def stub(name: str, **attributes) -> ModuleType:
    """Install a dependency fixture only in this short-lived child process."""
    result = ModuleType(name)
    result.__dict__.update(attributes)
    sys.modules[name] = result
    return result


def edge(relative: str, target: str) -> str:
    """Read an actual eager import edge without executing unrelated dependencies."""
    tree = ast.parse((ROOT / relative).read_text())
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == target
    )
    return ast.unparse(node)


def run_child(order: str) -> dict:
    """Exercise full platform import with cyclic dependency-edge fixtures."""
    calls = []

    def no_device_probe(*args, **kwargs):
        raise AssertionError("Platform bootstrap must not probe/initialize an NPU")

    def device(name: str):
        assert name == "npu" and "torch_npu" in calls, "NPU type was not registered"
        return SimpleNamespace(type=name)

    torch = stub("torch", device=device)
    stub("vllm", __path__=[str(ROOT / "vllm")])
    stub("vllm.utils", __path__=[str(ROOT / "vllm/utils")])
    stub("vllm.envs", VLLM_PLUGINS=[])
    stub("vllm.logger", logger=logging.getLogger("bootstrap-test"))
    stub("vllm.utils.torch_utils", supports_xccl=lambda: False)

    class PlatformEnum(enum.Enum):
        NPU = 1

    class Platform:
        def is_npu(self):
            return self._enum == PlatformEnum.NPU

    stub(
        "vllm.platforms.interface",
        Platform=Platform,
        PlatformEnum=PlatformEnum,
        CpuArchEnum=object,
    )
    # Use the production resolver, not a mock returning a preconstructed class.
    tree = ast.parse((ROOT / "vllm/utils/import_utils.py").read_text())
    resolver = next(
        n for n in tree.body if getattr(n, "name", None) == "resolve_obj_by_qualname"
    )
    namespace = {"importlib": importlib, "Any": object}
    exec(
        compile(ast.Module(body=[resolver], type_ignores=[]), "<resolver>", "exec"),
        namespace,
    )
    stub("vllm.utils.import_utils", resolve_obj_by_qualname=namespace[resolver.name])

    sources = {
        "vllm.config": edge("vllm/config/__init__.py", "vllm.config.compilation"),
        "vllm.config.compilation": edge("vllm/config/compilation.py", "vllm.platforms")
        + "\nCompilationConfig = CompilationMode = CUDAGraphMode = PassConfig = object\n",
        "vllm.config.ascend": "init_ascend_config = WeightPrefetchConfig = get_ascend_config = object\n",
        "vllm.utils.ascend": edge("vllm/utils/ascend.py", "vllm.config.ascend"),
        # Model-executor-first reproduces the parent-package path of python -m
        # registry. Only the actual dependency edges run; no model is loaded.
        "vllm.model_executor": edge(
            "vllm/model_executor/__init__.py", "vllm.model_executor.parameter"
        ),
        "vllm.model_executor.parameter": edge(
            "vllm/model_executor/parameter.py", "vllm.distributed"
        )
        + "\nBasevLLMParameter = PackedvLLMParameter = object\n",
        "vllm.distributed": edge("vllm/distributed/__init__.py", "communication_op")
        + "\nget_tensor_model_parallel_rank = get_tensor_model_parallel_world_size = object\n",
        "vllm.distributed.communication_op": edge(
            "vllm/distributed/communication_op.py", "parallel_state"
        ),
        "vllm.distributed.parallel_state": edge(
            "vllm/distributed/parallel_state.py", "vllm.distributed.utils"
        )
        + "\nget_tp_group = object\n",
        "vllm.distributed.utils": edge(
            "vllm/distributed/utils.py", "vllm.utils.system_utils"
        )
        + "\nStatelessProcessGroup = object\n",
        "vllm.utils.system_utils": edge("vllm/utils/system_utils.py", "vllm.platforms")
        + "\nsuppress_stdout = object\n",
    }

    class Fixtures(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch_npu" or fullname in sources:
                return importlib.util.spec_from_loader(
                    fullname,
                    self,
                    is_package=fullname
                    in {"vllm.config", "vllm.model_executor", "vllm.distributed"},
                )
            return None

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            calls.append(module.__name__)
            if module.__name__ == "torch_npu":
                torch.npu = SimpleNamespace(
                    is_available=no_device_probe,
                    device_count=no_device_probe,
                    set_device=no_device_probe,
                    init=no_device_probe,
                )
                return
            exec(
                compile(sources[module.__name__], "<dependency-edge fixture>", "exec"),
                module.__dict__,
            )

    sys.meta_path.insert(0, Fixtures())
    first = {
        "platform": "vllm.platforms",
        "config": "vllm.config",
        "model_executor": "vllm.model_executor",
        "ascend_utils": "vllm.utils.ascend",
    }[order]
    importlib.import_module(first)
    from vllm.platforms import current_platform
    from vllm.platforms.npu import NPUPlatform

    assert type(current_platform) is NPUPlatform
    assert current_platform.device_type == "npu"
    assert current_platform.is_npu()
    assert current_platform.supported_quantization == ["ascend", "compressed-tensors"]
    assert current_platform.pass_key == "graph_fusion_manager"
    assert torch.device("npu").type == "npu"
    if order == "platform":
        assert not any(name.startswith("vllm.config") for name in calls)
        assert "vllm.utils.ascend" not in calls
    return {"order": order, "calls": calls, "passed": True}


class ColdPlatformImportTests(unittest.TestCase):
    def probe(self, order: str) -> None:
        """Run one isolated host-fixture import order with a bounded timeout."""
        result = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--child", order],
            cwd=ROOT.parent,
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(json.loads(result.stdout)["passed"])

    def test_platform_first(self):
        self.probe("platform")

    def test_config_first(self):
        self.probe("config")

    def test_registry_parent_first(self):
        self.probe("model_executor")

    def test_ascend_utils_first(self):
        self.probe("ascend_utils")


class BootstrapToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "bootstrap_check_tool", ROOT / "tools/check_npu_bootstrap.py"
        )
        cls.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.tool)

    def test_shared_constants_keep_public_values_and_exports(self):
        tree = ast.parse((ROOT / "vllm/platforms/ascend_constants.py").read_text())
        actual = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
        }
        self.assertEqual(
            actual,
            {
                "ASCEND_QUANTIZATION_METHOD": "ascend",
                "COMPRESSED_TENSORS_METHOD": "compressed-tensors",
                "COMPILATION_PASS_KEY": "graph_fusion_manager",
            },
        )
        utils = ast.parse((ROOT / "vllm/utils/ascend.py").read_text())
        exports = {
            item.asname or item.name
            for node in utils.body
            if isinstance(node, ast.ImportFrom)
            and node.module == "vllm.platforms.ascend_constants"
            for item in node.names
        }
        self.assertEqual(exports, set(actual))
        self.assertFalse(
            any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body)
        )

    def test_command_preserves_failure_and_output(self):
        code, output = self.tool.run_command(
            [
                sys.executable,
                "-B",
                "-c",
                "print('failure detail'); raise SystemExit(7)",
            ],
            timeout=5,
        )
        self.assertEqual(code, 7)
        self.assertIn("failure detail", output)

    def test_timeout_is_a_failure(self):
        code, output = self.tool.run_command(
            [sys.executable, "-B", "-c", "import time; time.sleep(30)"], timeout=1
        )
        self.assertEqual(code, 124)
        self.assertIn("timed out", output)

    def test_help_does_not_require_torch(self):
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "tools/check_npu_bootstrap.py"),
                "--help",
            ],
            text=True,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--inspect-glm", result.stdout)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        print(json.dumps(run_child(sys.argv[2])))
    else:
        unittest.main()
