# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intranet P2/P3 cold-import and uncached registry inspection gate.

Run outside source checkouts after installation. No weights are loaded and no
inference, installation or network service is started. Imports can load native
libraries; registry initialization retains normal profiling-config generation.
Each order uses a new interpreter. Only this tool's timed-out children are killed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import tomllib
import traceback
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORDERS = ("platform", "config", "registry", "ascend_utils", "registry_subprocess")
IDENTITY_FILES = (
    "platforms/__init__.py",
    "platforms/npu.py",
    "platforms/ascend_constants.py",
    "utils/ascend.py",
    "config/__init__.py",
    "config/compilation.py",
)
RESULT_PREFIX = "NPU_BOOTSTRAP_RESULT="


def check_order(order: str) -> dict:
    """Check one real installed import order without prewarming vLLM or torch."""
    expected = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "version"
    ]
    installed = metadata.version("vllm")
    if installed != expected:
        raise RuntimeError(f"Installed {installed}; this checkout expects {expected}")
    for name in ("vllm-ascend", "lmcache-ascend"):
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        raise RuntimeError(f"Retired distribution is installed: {name}=={version}")

    first = {
        "platform": "vllm.platforms",
        "config": "vllm.config",
        "registry": "vllm.model_executor.models.registry",
        "ascend_utils": "vllm.utils.ascend",
        "registry_subprocess": "vllm.model_executor.models.registry",
        "glm_inspect": "vllm.model_executor.models.registry",
    }[order]
    importlib.import_module(first)
    import vllm
    from vllm.platforms import current_platform

    qualified_name = (
        f"{type(current_platform).__module__}.{type(current_platform).__name__}"
    )
    if (
        qualified_name != "vllm.platforms.npu.NPUPlatform"
        or current_platform.device_type != "npu"
    ):
        raise RuntimeError(f"Unexpected platform: {qualified_name}")
    if not current_platform.is_npu() or current_platform.is_out_of_tree():
        raise RuntimeError("The native NPU platform was not selected")
    import torch

    if torch.device("npu").type != "npu":
        raise RuntimeError("torch_npu did not register the NPU device type")
    package = Path(vllm.__file__).absolute().parent
    if package == ROOT / "vllm":
        raise RuntimeError(
            "Source checkout shadows the installed package; run outside it"
        )
    fingerprints = {}
    for relative in IDENTITY_FILES:
        actual = (package / relative).read_bytes()
        expected_source = (ROOT / "vllm" / relative).read_bytes()
        if actual != expected_source:
            raise RuntimeError(
                f"Installed source differs from this checkout: {relative}"
            )
        fingerprints[relative] = hashlib.sha256(actual).hexdigest()
    result = {
        "order": order,
        "version": installed,
        "platform": qualified_name,
        "vllm_path": str(package),
        "source_sha256": fingerprints,
    }
    if order in ("registry_subprocess", "glm_inspect"):
        from vllm.model_executor.models import registry

        if order == "registry_subprocess":
            # Use the production python -m registry protocol (with its required
            # cloudpickle stdin), not a warm-process substitute or empty stdin.
            actual = registry._run_in_subprocess(lambda: "registry-protocol-ok")
            if actual != "registry-protocol-ok":
                raise RuntimeError("Registry child returned an unexpected result")
        else:
            model = registry.ModelRegistry.models["GlmMoeDsaForCausalLM"]
            info = registry._run_in_subprocess(
                lambda: registry._ModelInfo.from_model_cls(model.load_model_cls())
            )
            if (
                info.architecture != "GlmMoeDsaForCausalLM"
                or not info.is_text_generation_model
            ):
                raise RuntimeError(f"Unexpected GLM model inspection: {info}")
            result["architecture"] = info.architecture
            result["modelinfo_cache_used"] = False
    result["passed"] = True
    return result


def run_command(command: list[str], timeout: int) -> tuple[int, str]:
    """Bound a check and its registry children, never a preexisting worker."""
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env={**os.environ, "VLLM_PLUGINS": "", "PYTHONDONTWRITEBYTECODE": "1"},
    ) as child:
        try:
            output, _ = child.communicate(timeout=timeout)
            return child.returncode, output
        except subprocess.TimeoutExpired:
            # A fresh process group belongs only to this check and its children.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(child.pid, signal.SIGKILL)
            output, _ = child.communicate()
            return 124, output + f"\nCold-import check timed out after {timeout}s\n"


def main() -> int:
    """Write a fresh report; any import, source identity or timeout error fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--inspect-glm",
        action="store_true",
        help="uncached GLM class inspection; no weights",
    )
    parser.add_argument(
        "--child", choices=(*ORDERS, "glm_inspect"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.child is not None:
        try:
            result = check_order(args.child)
        except Exception:
            result = {
                "order": args.child,
                "passed": False,
                "error": traceback.format_exc(),
            }
        print(RESULT_PREFIX + json.dumps(result))
        return 0 if result["passed"] else 1
    if args.output is None or args.timeout <= 0:
        parser.error("--output and a positive --timeout are required")
    if os.name != "posix":
        parser.error("This Ascend validation tool requires a POSIX environment")
    args.output.mkdir(parents=True, exist_ok=False)
    checks = []
    orders = (*ORDERS, "glm_inspect") if args.inspect_glm else ORDERS
    for order in orders:
        command = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--child",
            order,
        ]
        code, output = run_command(command, args.timeout)
        (args.output / f"{order}.log").write_text(output)
        markers = [
            line[len(RESULT_PREFIX) :]
            for line in output.splitlines()
            if line.startswith(RESULT_PREFIX)
        ]
        try:
            result = (
                json.loads(markers[-1])
                if markers
                else {"passed": False, "error": "No child report"}
            )
        except json.JSONDecodeError:
            result = {"passed": False, "error": "Invalid child report"}
        result.update(order=order, returncode=code, log=f"{order}.log", command=command)
        result["passed"] = code == 0 and result.get("passed") is True
        checks.append(result)
        print(f"{order}: {'PASS' if result['passed'] else 'FAIL'}", flush=True)
    report = {
        "scope": "installed_cold_import_and_optional_class_inspection_not_inference",
        "python": sys.executable,
        "optional_plugins_disabled": True,
        "checks": checks,
        "passed": all(item["passed"] for item in checks),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
