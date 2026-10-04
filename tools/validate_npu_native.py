#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Intranet P4 installed-runtime probe; never installs packages or loads a model.

Run outside the source checkout in the prepared Ascend environment. Optional
device operations require an idle NPU. This does not certify inference or P/D.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.abc
import importlib.metadata
import importlib.util
import json
import multiprocessing
import os
import sys
import traceback
from pathlib import Path


class RetiredImports(importlib.abc.MetaPathFinder):
    """Make disabled LMCache and removed plugin dependencies observable."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"vllm_ascend", "lmcache", "lmcache_ascend"}:
            raise ImportError(f"P2 no-cache probe forbids dependency: {fullname}")
        return None


def check_add_rms_norm_bias(torch, torch_npu) -> list[dict]:
    """Run a tiny real custom-op check on the explicitly selected idle NPU.

    Compare against the built-in AddRmsNorm plus optional bias. This verifies
    operator discovery and selected tensor cases, not model accuracy/performance.
    """
    cases = []
    for dtype in (torch.float32, torch.bfloat16):
        values = torch.arange(256, dtype=torch.float32).reshape(8, 32) / 128 - 1
        x = values.to(device="npu", dtype=dtype)
        residual = (values * 0.125).to(device="npu", dtype=dtype)
        weight = torch.linspace(0.75, 1.25, 32, device="npu", dtype=dtype)
        for with_bias in (False, True):
            bias = (
                torch.full((32,), 0.125, device="npu", dtype=dtype)
                if with_bias
                else None
            )
            expected, _, expected_residual = torch_npu.npu_add_rms_norm(
                x, residual, weight, 1e-6
            )
            if bias is not None:
                expected = expected + bias
            actual, _, actual_residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                x, residual, weight, bias, 1e-6
            )
            torch.npu.synchronize()
            tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-4
            torch.testing.assert_close(
                actual.cpu(), expected.cpu(), rtol=tolerance, atol=tolerance
            )
            torch.testing.assert_close(actual_residual.cpu(), expected_residual.cpu())
            cases.append({"dtype": str(dtype), "bias": with_bias, "passed": True})
    return cases


def probe(*, device_smoke=False, torchair_abi=False):
    result = {"scope": "installed_import_contracts", "passed": False}
    guard = RetiredImports()
    try:
        # Explicitly test the built-in path with optional plugins disabled.
        os.environ["VLLM_PLUGINS"] = ""
        if importlib.util.find_spec("vllm_ascend") is not None:
            raise RuntimeError("Retired vllm_ascend package is still discoverable")
        for name in ("vllm-ascend", "lmcache-ascend"):
            try:
                importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                continue
            raise RuntimeError(f"Retired distribution is installed: {name}")
        sys.meta_path.insert(0, guard)

        import torch
        import torch_npu

        import vllm
        from vllm.platforms import current_platform
        from vllm.plugins import load_general_plugins

        if importlib.metadata.version("vllm") != "0.18.0+ascend.p4":
            raise RuntimeError("Expected vllm 0.18.0+ascend.p4")
        if not current_platform.is_npu() or current_platform.is_out_of_tree():
            raise RuntimeError("Native NPU platform was not selected")
        load_general_plugins()
        # Observe the same entry used by serve/LLM/WorkerWrapper. Do not repair
        # a missing production call here: that hid the initial P4 regression.
        vendor = (
            Path(vllm.__file__).absolute().parent
            / "_cann_ops_custom/vendors/vllm-ascend"
        )
        if not vendor.is_dir() or os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(
            os.pathsep
        )[0] != str(vendor):
            raise RuntimeError(
                f"Normal startup did not register packaged CANN OPP: {vendor}"
            )
        result["custom_opp"] = {
            "vendor_path": str(vendor),
            "registered_before_native_ops": True,
        }

        # Register actual native operators before importing worker/model owners.
        from vllm.model_executor.layers.ascend import initialize_native_ops

        initialize_native_ops()
        owners = (
            ("vllm.v1.worker.npu_model_runner", "NPUModelRunner"),
            ("vllm.v1.worker.npu.v2.model_runner", "NPUModelRunner"),
        )
        result["runners"] = {}
        for name, symbol in owners:
            cls = getattr(importlib.import_module(name), symbol)
            mro = [f"{base.__module__}.{base.__name__}" for base in cls.__mro__]
            if any("GPUModelRunner" in base or ".worker.gpu" in base for base in mro):
                raise RuntimeError(f"GPU Runner inherited by {name}: {mro}")
            result["runners"][name] = mro
        if any(name.startswith("vllm.v1.worker.gpu") for name in sys.modules):
            raise RuntimeError("Native import closure loaded a GPU worker module")
        for name in (
            "vllm.v1.attention.backends.ascend.sfa_v1",
            "vllm.v1.spec_decode.ascend.eagle_proposer",
            "vllm.v1.core.sched.recompute_scheduler",
            "vllm.distributed.device_communicators.npu_communicator",
            "vllm.distributed.kv_transfer.kv_connector.factory",
        ):
            importlib.import_module(name)

        if torchair_abi:
            from torchair.configs.compiler_config import CompilerConfig

            from vllm.compilation.ascend.torchair_backend import get_npu_backend

            config = CompilerConfig()
            config.mode = "npugraph_ex"
            backend = get_npu_backend(config)
            if not callable(backend):
                raise RuntimeError("TorchAir backend factory returned no callable")
            result["torchair_factory_abi"] = "passed_not_graph_execution"

        if device_smoke:
            from vllm.utils.ascend import enable_custom_op

            torch.npu.set_device(0)
            if not enable_custom_op():
                raise RuntimeError("Native custom extension did not load")
            extension = importlib.import_module("vllm._ascend_C")
            value = torch.ones((8, 32), device="npu", dtype=torch.float32)
            stream, event = torch.npu.Stream(), torch.npu.Event()
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                actual = value * 3
                event.record()
            event.synchronize()
            torch.testing.assert_close(actual.cpu(), torch.full((8, 32), 3.0))
            result["device_smoke"] = {
                "extension": extension.__file__,
                "device": torch.npu.get_device_name(0),
                "stream_event_tensor": "passed",
                "add_rms_norm_bias": check_add_rms_norm_bias(torch, torch_npu),
            }

        result.update(
            platform=f"{type(current_platform).__module__}.{type(current_platform).__name__}",
            vllm_path=vllm.__file__,
            torch_version=torch.__version__,
            torch_npu_version=torch_npu.__version__,
            no_cache_imports=[
                name
                for name in sys.modules
                if name.split(".")[0] in {"lmcache", "lmcache_ascend"}
            ],
        )
        if result["no_cache_imports"]:
            raise RuntimeError("LMCache imported while disabled")
        result["passed"] = True
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        if guard in sys.meta_path:
            sys.meta_path.remove(guard)
    return result


def child_probe(pipe, options):
    try:
        pipe.send(probe(**options))
    finally:
        pipe.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device-smoke", action="store_true")
    parser.add_argument("--torchair-abi", action="store_true")
    parser.add_argument("--spawn", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output file to preserve previous results")
    options = {"device_smoke": args.device_smoke, "torchair_abi": args.torchair_abi}
    result = probe(**options)
    if args.spawn:
        ctx = multiprocessing.get_context("spawn")
        receiver, sender = ctx.Pipe(duplex=False)
        worker = ctx.Process(target=child_probe, args=(sender, options))
        worker.start()
        sender.close()
        try:
            child = (
                receiver.recv()
                if receiver.poll(300)
                else {"passed": False, "error": "spawn probe timed out"}
            )
        except EOFError:
            child = {"passed": False, "error": "spawn worker exited without a report"}
        finally:
            receiver.close()
        worker.join(10)
        if worker.is_alive():
            worker.terminate()
            worker.join()
        result["spawn"] = child
        result["passed"] = result["passed"] and child["passed"] and worker.exitcode == 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
