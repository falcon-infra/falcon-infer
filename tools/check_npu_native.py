#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static P2 source gate. Does not import torch, probe NPU, or compile kernels."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE_AREAS = (
    "vllm/platforms/npu.py",
    "vllm/platforms/ascend_device",
    "vllm/platforms/ascend_310p",
    "vllm/config/ascend.py",
    "vllm/envs_ascend.py",
    "vllm/ascend_forward_context.py",
    "vllm/compilation/ascend",
    "vllm/compilation/xlite",
    "vllm/compilation/flash_common3_context.py",
    "vllm/v1/worker/npu",
    "vllm/v1/worker/npu_model_runner.py",
    "vllm/v1/worker/npu_runner_state.py",
    "vllm/v1/worker/npu_worker.py",
    "vllm/v1/worker/npu_input_batch.py",
    "vllm/v1/worker/npu_block_table.py",
    "vllm/v1/worker/npu_pcp_utils.py",
    "vllm/v1/worker/npu_serving_perf.py",
    "vllm/v1/worker/dsa_shared_pool.py",
    "vllm/v1/worker/runner_output.py",
    "vllm/model_executor/layers/ascend",
    "vllm/model_executor/layers/ascend_batch_invariant.py",
    "vllm/model_executor/layers/quantization/ascend",
    "vllm/model_executor/model_loader/ascend",
    "vllm/lora/ascend",
    "vllm/device_allocator/ascend",
    "vllm/v1/attention/backends/ascend",
    "vllm/v1/sample/ascend",
    "vllm/v1/spec_decode/ascend",
    "vllm/v1/kv_offload/ascend",
    "vllm/v1/core/ascend",
    "vllm/v1/core/mc2_recovery.py",
    "vllm/v1/core/sched/recompute_scheduler.py",
    "vllm/v1/core/sched/dynamic_batch_scheduler.py",
    "vllm/v1/core/sched/balance_scheduler.py",
    "vllm/v1/engine/balance_core.py",
    "vllm/v1/executor/ascend_multiproc_executor.py",
    "vllm/distributed/ascend",
    "vllm/distributed/device_communicators/ascend",
    "vllm/distributed/device_communicators/npu_communicator.py",
    "vllm/distributed/eplb/ascend",
    "vllm/distributed/kv_transfer/ascend",
    "vllm/distributed/kv_transfer/live_source_handoff.py",
    "vllm/distributed/kv_transfer/lmcache_diagnostics.py",
    "vllm/utils/ascend.py",
    "vllm/utils/ascend_profiling_config.py",
    "vllm/utils/npu_cpu_binding.py",
    "vllm/utils/diagnostic_utils.py",
    "vllm/utils/serving_perf.py",
)


def native_files():
    paths = set()
    for area in NATIVE_AREAS:
        path = ROOT / area
        if path.is_file():
            paths.add(path)
        elif path.is_dir():
            paths.update(path.rglob("*.py"))
        else:
            raise AssertionError(f"Missing native owner: {area}")
    return sorted(paths)


def audit():
    errors = []
    files = native_files()
    for path in files:
        relative = str(path.relative_to(ROOT))
        text = path.read_text()
        tree = ast.parse(text)
        compile(tree, relative, "exec")
        module = relative.removesuffix(".py").replace("/", ".")
        package = (
            module.removesuffix(".__init__")
            if path.name == "__init__.py"
            else module.rpartition(".")[0]
        )
        for node in ast.walk(tree):
            imports = []
            if isinstance(node, ast.ImportFrom):
                name = node.module or ""
                if node.level:
                    name = importlib.util.resolve_name("." * node.level + name, package)
                imports = [name]
            elif isinstance(node, ast.Import):
                imports = [a.name for a in node.names]
            for name in imports:
                if name.startswith(
                    (
                        "vllm_ascend",
                        "vllm.patch",
                        "vllm.v1.worker.gpu",
                        "vllm.compilation.cuda_graph",
                    )
                ):
                    errors.append(
                        f"{relative}:{node.lineno}: retired runtime import {name}"
                    )
                if name.startswith("vllm.") and name not in (
                    "vllm._ascend_C",
                    "vllm._build_info",
                    "vllm._version",
                ):
                    target = ROOT.joinpath(*name.split("."))
                    if (
                        not target.with_suffix(".py").is_file()
                        and not (target / "__init__.py").is_file()
                    ):
                        errors.append(
                            f"{relative}:{node.lineno}: unresolved local module {name}"
                        )
            if isinstance(node, ast.ClassDef) and any(
                "GPUModelRunner" in ast.unparse(base) for base in node.bases
            ):
                errors.append(f"{relative}:{node.lineno}: GPU Runner inheritance")
            if isinstance(node, ast.Attribute) and ast.unparse(node).startswith(
                "torch.cuda."
            ):
                errors.append(f"{relative}:{node.lineno}: CUDA API in native owner")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "register_oot"
            ):
                errors.append(f"{relative}:{node.lineno}: OOT registration")
            if isinstance(node, ast.Name) and node.id in (
                "adapt_patch",
                "_torch_cuda_wrapper",
                "torch_cuda_wrapper",
                "_replace_gpu_model_runner_function_wrapper",
            ):
                errors.append(
                    f"{relative}:{node.lineno}: retired runtime hook {node.id}"
                )

    # Also reject legacy package imports anywhere in the installed Python package.
    for path in (ROOT / "vllm").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                continue
            if any(name.startswith(("vllm_ascend", "vllm.patch")) for name in names):
                errors.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: retired package import"
                )

    return {
        "scope": "static_source_not_runtime_or_ABI",
        "native_python_files": len(files),
        "errors": errors,
        "passed": not errors,
    }


if __name__ == "__main__":
    report = audit()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)
