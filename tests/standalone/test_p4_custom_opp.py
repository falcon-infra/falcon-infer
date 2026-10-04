# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise the real NPU bootstrap against packaged-resource/torch fixtures.

No torch, CANN or model is loaded. The complete platform, plugin loader and
native-ops initializer execute; only device and operator dependencies are stubs.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENV = "ASCEND_CUSTOM_OPP_PATH"


def load_module(monkeypatch, name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def bootstrap(monkeypatch, tmp_path: Path, *, payload: bool = True):
    """Model strict editable __init__ links with resources in the link tree."""
    source = tmp_path / "source/vllm"
    package = tmp_path / "build/__editable__.vllm/vllm"
    source.mkdir(parents=True)
    package.mkdir(parents=True)
    (source / "__init__.py").write_text("")
    (package / "__init__.py").symlink_to(source / "__init__.py")
    vendor = package / "_cann_ops_custom/vendors/vllm-ascend"
    if payload:
        vendor.mkdir(parents=True)
    calls = []

    def stub(name: str, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def no_device(*args, **kwargs):
        raise AssertionError("OPP registration must not load/initialize an NPU")

    envs = stub("vllm.envs", VLLM_PLUGINS=[])
    stub(
        "vllm",
        __file__=str(package / "__init__.py"),
        __path__=[str(ROOT / "vllm")],
        envs=envs,
    )
    stub("vllm.envs_ascend")
    stub("vllm.logger", logger=logging.getLogger("custom-opp-test"))
    stub("vllm.platforms", __path__=[str(ROOT / "vllm/platforms")])
    stub("vllm.platforms.interface", Platform=object, PlatformEnum=NS(NPU="npu"))
    stub("torch", npu=NS(init=no_device, set_device=no_device))
    stub("torch_npu")
    stub("vllm.utils", __path__=[])

    def profiling():
        assert os.environ.get(ENV, "").split(os.pathsep)[0] == str(vendor)
        calls.append("profiling")

    stub(
        "vllm.utils.ascend_profiling_config",
        generate_service_profiling_config=profiling,
    )
    # npu.py sets this unrelated option during import; restore it after the test.
    monkeypatch.setenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "1")
    module = load_module(monkeypatch, "vllm.platforms.npu", "vllm/platforms/npu.py")
    platform = module.NPUPlatform()
    sys.modules["vllm.platforms"].current_platform = platform
    loader = load_module(monkeypatch, "vllm.plugins", "vllm/plugins/__init__.py")
    monkeypatch.setattr("importlib.metadata.entry_points", lambda **kw: [])
    monkeypatch.delenv(ENV, raising=False)
    return NS(platform=platform, loader=loader, vendor=vendor, calls=calls)


@pytest.mark.parametrize("existing", [None, "", "/other/vendor", "/first:/second"])
def test_normal_bootstrap_registers_before_components(monkeypatch, tmp_path, existing):
    runtime = bootstrap(monkeypatch, tmp_path)
    if existing is not None:
        monkeypatch.setenv(ENV, existing)
    runtime.loader.load_general_plugins()
    runtime.loader.load_general_plugins()
    suffix = os.pathsep + existing if existing else ""
    assert os.environ[ENV] == str(runtime.vendor) + suffix
    assert runtime.calls == ["profiling"]
    assert "vllm._ascend_C" not in sys.modules
    assert "vllm._custom_ops" not in sys.modules


@pytest.mark.parametrize("existing", ["first", "last", "duplicate"])
def test_registration_is_ordered_and_idempotent(monkeypatch, tmp_path, existing):
    runtime = bootstrap(monkeypatch, tmp_path)
    own = str(runtime.vendor)
    values = {
        "first": f"{own}:/other",
        "last": f"/other:{own}",
        "duplicate": f"/other:{own}:{own}",
    }
    monkeypatch.setenv(ENV, values[existing])
    runtime.platform.import_kernels()
    runtime.platform.import_kernels()
    assert os.environ[ENV] == f"{own}:/other"


def test_reregister_after_environment_changes(monkeypatch, tmp_path):
    runtime = bootstrap(monkeypatch, tmp_path)
    runtime.platform.import_kernels()
    monkeypatch.setenv(ENV, "/later/vendor")
    runtime.platform.import_kernels()
    assert os.environ[ENV] == f"{runtime.vendor}:/later/vendor"


def test_missing_payload_fails_before_components_and_can_be_retried(
    monkeypatch, tmp_path
):
    runtime = bootstrap(monkeypatch, tmp_path, payload=False)
    monkeypatch.setenv(ENV, "/unrelated/vendor")
    with pytest.raises(RuntimeError, match="packaged CANN custom operators"):
        runtime.platform.register_builtin_components()
    assert runtime.calls == []
    assert os.environ[ENV] == "/unrelated/vendor"
    runtime.vendor.mkdir(parents=True)
    runtime.platform.register_builtin_components()
    assert os.environ[ENV] == f"{runtime.vendor}:/unrelated/vendor"


def test_non_directory_payload_is_rejected(monkeypatch, tmp_path):
    runtime = bootstrap(monkeypatch, tmp_path, payload=False)
    runtime.vendor.parent.mkdir(parents=True)
    runtime.vendor.write_text("not a vendor directory")
    with pytest.raises(RuntimeError, match="packaged CANN custom operators"):
        runtime.platform.import_kernels()
    assert ENV not in os.environ


@pytest.mark.parametrize("has_triton", [False, True])
def test_worker_initializer_registers_before_native_imports(
    monkeypatch, tmp_path, has_triton
):
    runtime = bootstrap(monkeypatch, tmp_path)
    triton = ModuleType("vllm.triton_utils")
    triton.HAS_TRITON = has_triton
    monkeypatch.setitem(sys.modules, triton.__name__, triton)
    ops = load_module(
        monkeypatch,
        "vllm.model_executor.layers.ascend",
        "vllm/model_executor/layers/ascend/__init__.py",
    )

    def native_import(name):
        assert os.environ.get(ENV, "").split(os.pathsep)[0] == str(runtime.vendor)
        runtime.calls.append(name)

    monkeypatch.setattr(ops, "import_module", native_import)
    ops.initialize_native_ops()
    assert len(runtime.calls) == (8 if has_triton else 6)
    assert runtime.calls[0].endswith("fused_moe.fused_moe")
    assert runtime.calls[-1].endswith("rotary_embedding")


@pytest.mark.parametrize("inherited", ["", "/previous-install/vendors/vllm-ascend"])
def test_fresh_interpreter_registers_its_own_package(tmp_path, inherited):
    result = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--child", str(tmp_path)],
        env={**os.environ, ENV: inherited},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS: fresh-process OPP registration" in result.stdout


@pytest.mark.parametrize(
    "condition", ["valid", "unregistered", "wrong-first", "missing", "invalid"]
)
def test_installed_gate_observes_without_repair(monkeypatch, tmp_path, condition):
    runtime = bootstrap(monkeypatch, tmp_path)
    package = runtime.vendor.parents[2]
    config = (
        runtime.vendor
        / "op_impl/ai_core/tbe/kernel/config/ascend910b/binary_info_config.json"
    )
    if condition != "missing":
        config.parent.mkdir(parents=True)
        config.write_text("{" if condition == "invalid" else '{"fixture": true}')
    value = {
        "unregistered": "",
        "wrong-first": f"/unrelated:{runtime.vendor}",
    }.get(condition, str(runtime.vendor))
    monkeypatch.setenv(ENV, value)
    gate = load_module(
        monkeypatch, "opp_bootstrap_gate", "tools/check_npu_bootstrap.py"
    )
    if condition == "valid":
        report = gate.check_custom_opp_registration(package)
        assert report["registered_first"]
        assert report["binary_info_config"] == str(config)
    else:
        with pytest.raises((RuntimeError, ValueError)):
            gate.check_custom_opp_registration(package)
    assert os.environ[ENV] == value


def test_runtime_probe_does_not_mask_missing_registration():
    tree = ast.parse((ROOT / "tools/validate_npu_native.py").read_text())
    probe = next(n for n in tree.body if getattr(n, "name", None) == "probe")
    assert not any(
        isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "import_kernels"
        for n in ast.walk(probe)
    )
    gated = next(
        n
        for n in ast.walk(probe)
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Name)
        and n.test.id == "device_smoke"
    )
    assert any(
        isinstance(n, ast.Call)
        and getattr(n.func, "id", None) == "check_add_rms_norm_bias"
        for n in ast.walk(gated)
    )


if __name__ == "__main__":
    assert len(sys.argv) == 3 and sys.argv[1] == "--child"
    inherited = os.environ.get(ENV, "")
    with pytest.MonkeyPatch.context() as patch:
        runtime = bootstrap(patch, Path(sys.argv[2]))
        patch.setenv(ENV, inherited)
        runtime.loader.load_general_plugins()
        assert os.environ[ENV] == str(runtime.vendor) + (
            ":" + inherited if inherited else ""
        )
    print("PASS: fresh-process OPP registration")
