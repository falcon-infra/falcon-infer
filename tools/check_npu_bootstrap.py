# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Intranet cold-import, KV-binding and optional LMCache completion/GLM gate.

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
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ORDERS = (
    "platform",
    "config",
    "registry",
    "ascend_utils",
    "registry_subprocess",
    "kv_cache_bind",
)
IDENTITY_FILES = (
    "platforms/__init__.py",
    "platforms/npu.py",
    "platforms/ascend_constants.py",
    "utils/ascend.py",
    "config/__init__.py",
    "config/compilation.py",
    "config/kv_transfer.py",
    "config/speculative.py",
    "v1/worker/utils.py",
    "v1/worker/npu_runner_state.py",
    "v1/worker/npu_model_runner.py",
    "v1/spec_decode/ascend/__init__.py",
    "distributed/kv_transfer/kv_connector/v1/lmcache_connector.py",
)
RESULT_PREFIX = "NPU_BOOTSTRAP_RESULT="


def check_kv_cache_binding(
    bind_kv_cache: Callable[..., None],
    latent_cache: object,
    indexer_cache: object,
    consumer_cache: object,
) -> dict:
    """Check binding order/identity with caller-provided caches, not kernels.

    The installed check passes CPU tensors to the actual native NPU binding
    function. Host tests pass object fixtures and deliberately broken binders.
    """
    cases = []
    for prefix in ("model.layers.7", "model.layers.78.mtp_block"):
        latent_name = f"{prefix}.self_attn.attn"
        indexer_name = f"{prefix}.self_attn.indexer.k_cache"
        for indexer_first in (False, True):
            pair = [(latent_name, latent_cache), (indexer_name, indexer_cache)]
            if indexer_first:
                pair.reverse()
            # The consumer sorts before the producer but is inserted last.
            caches = dict(pair + [("model.layers.5.self_attn.attn", consumer_cache)])
            cases.append((caches, [consumer_cache, latent_cache, indexer_cache]))
    cases.extend(
        [
            ({"model.layers.7.self_attn.attn": latent_cache}, [latent_cache]),
            ({}, []),
        ]
    )
    for case_index, (caches, expected) in enumerate(cases):
        context = {name: SimpleNamespace() for name in caches}
        runner = []
        bind_kv_cache(caches, context, runner)
        if len(runner) != len(expected) or any(
            actual is not wanted for actual, wanted in zip(runner, expected)
        ):
            raise RuntimeError(
                f"KV binding case {case_index}: wrong runner order/identity"
            )
        for name, value in caches.items():
            bound = getattr(context[name], "kv_cache", None)
            if not isinstance(bound, list) or len(bound) != 1 or bound[0] is not value:
                raise RuntimeError(
                    f"KV binding case {case_index}: wrong context for {name}"
                )
    return {"cases_passed": len(cases), "contract": "order_and_reference_identity"}


def check_lmcache_completion(
    connector_cls: type, adapter_cls: type, output_cls: type
) -> dict:
    """Exercise real scheduler methods on isolated host state, without __init__.

    No cache engine, lookup client, NPU allocation or service is created. This
    checks completion-to-resume state transitions, not tensors or inference.
    """
    wrapper = object.__new__(connector_cls)
    adapter = object.__new__(adapter_cls)
    wrapper._lmcache_engine = adapter
    wrapper._kv_cache_events = None
    adapter.load_specs = {}

    def receive(finished=None, invalid=()):
        wrapper.update_connector_output(
            output_cls(
                finished_recving=finished,
                finished_sending=None,
                kv_cache_events=None,
                invalid_block_ids=set(invalid),
                completed_decode_window_saves={},
            )
        )

    for index, failed in enumerate((False, False, True)):
        req_id = f"bootstrap-cold-{index}"
        spec = SimpleNamespace(dsa_cold_compact_load=True, can_load=False)
        adapter.load_specs[req_id] = spec
        adapter._dsa_group1_direct_hbm_active_req_id = req_id
        adapter._dsa_cold_indexer_block_ids = {req_id: {index}}
        # An unrelated completion must not release the active request's slot.
        receive({"unrelated"})
        receive(invalid={index} if failed else ())
        if getattr(adapter, "_dsa_group1_direct_hbm_active_req_id", None) != req_id:
            raise RuntimeError("LMCache released a cold-load slot before completion")
        if adapter._take_completed_cold_load(req_id, spec):
            raise RuntimeError("LMCache resumed a cold load before completion")
        receive({req_id})
        if getattr(adapter, "_dsa_group1_direct_hbm_active_req_id", None) is not None:
            raise RuntimeError("LMCache completion did not release the cold-load slot")
        ready = adapter._take_completed_cold_load(req_id, spec)
        if ready != (not failed):
            raise RuntimeError("LMCache cold-load validation/resume state is incorrect")
        if not failed and (
            getattr(spec, "dsa_cold_compact_load", False)
            or not getattr(spec, "dsa_cold_compact_resume", False)
            or not spec.can_load
        ):
            raise RuntimeError("LMCache completion did not set sparse resume markers")
        if adapter._take_completed_cold_load(req_id, spec):
            raise RuntimeError("LMCache consumed a cold-load completion twice")
        if wrapper._kv_cache_events is not None:
            raise RuntimeError("LMCache control completion generated KV events")
        # Keep prior LoadSpecs alive: B must proceed before A finishes decoding.

    return {
        "cases_passed": 3,
        "contract": "cold_load_slot_release_sparse_resume_and_invalid_block_rejection",
        "kv_events_enabled": False,
        "constructors_called": False,
        "device_allocation": False,
    }


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
        "kv_cache_bind": "vllm.v1.worker.utils",
        "lmcache_completion": "vllm.config",
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
    if order == "config":
        from vllm.config.kv_transfer import KVTransferConfig
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

        transfer = KVTransferConfig(
            kv_connector="LMCacheAscendConnectorV1Dynamic",
            kv_connector_module_path=(
                "lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1"
            ),
            kv_role="kv_both",
            kv_buffer_device="npu",
        )
        if (
            transfer.kv_connector != "LMCacheConnectorV1"
            or transfer.kv_connector_module_path is not None
        ):
            raise RuntimeError("Retired P3 LMCache launch fields were not migrated")
        connector_cls = KVConnectorFactory.get_connector_class(transfer)
        owner = f"{connector_cls.__module__}.{connector_cls.__name__}"
        if owner != (
            "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector."
            "LMCacheConnectorV1"
        ):
            raise RuntimeError(f"Unexpected native LMCache connector: {owner}")
        result["lmcache_config_migration"] = {
            "connector": owner,
            "instance_created": False,
        }
    if order == "lmcache_completion":
        # Opt-in only: no-KV installations must not acquire an LMCache dependency.
        from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl

        from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
            LMCacheConnectorV1,
        )
        from vllm.v1.outputs import KVConnectorOutput

        result["lmcache_completion"] = check_lmcache_completion(
            LMCacheConnectorV1, LMCacheConnectorV1Impl, KVConnectorOutput
        )
        result["lmcache_version"] = metadata.version("lmcache")
        result["lmcache_adapter_path"] = importlib.import_module(
            LMCacheConnectorV1Impl.__module__
        ).__file__
    if order == "kv_cache_bind":
        from vllm.v1.worker.utils import bind_kv_cache

        # Explicit CPU allocation: no NPU memory, streams or kernels are used.
        # Views share one storage and must be bound by reference, not copied.
        storage = torch.empty(4, dtype=torch.float32, device="cpu")
        result["kv_binding"] = check_kv_cache_binding(
            bind_kv_cache, (storage[:1], storage[1:2]), storage[2:3], storage[3:]
        )
        result["tensor_device"] = str(storage.device)
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
        "--check-lmcache",
        action="store_true",
        help="paired P4 LMCache completion/resume check on host state; no NPU tensors",
    )
    parser.add_argument(
        "--child",
        choices=(*ORDERS, "glm_inspect", "lmcache_completion"),
        help=argparse.SUPPRESS,
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
    if args.check_lmcache:
        orders = (*orders, "lmcache_completion")
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
        "scope": (
            "installed_cold_import_KV_binding_optional_LMCache_completion_and_GLM_not_inference"
        ),
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
