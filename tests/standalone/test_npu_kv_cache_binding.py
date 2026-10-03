# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run production KV-binding bodies with host object fixtures, without torch.

These tests cover ordering and reference identity, not device/tensor behavior.
The archived donor is read as an oracle; its import-time patch is never executed.
"""

from __future__ import annotations

import ast
import importlib.util
import itertools
import subprocess
import unittest
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def load_function(relative: str, name: str, namespace: dict):
    """Compile the complete named function, without importing device backends."""
    path = ROOT / relative
    if relative.startswith("ascend/legacy_patches/"):
        # P4 removes the duplicate plugin tree. The immutable P3 tag is the
        # historical oracle; no donor module or import-time patch is executed.
        source = subprocess.check_output(
            ["git", "-C", str(ROOT), "show", f"p3-frozen-20261003:{relative}"],
            text=True,
        )
    else:
        source = path.read_text()
    tree = ast.parse(source)
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


@lru_cache
def binding(platform: str = "npu", *, donor: bool = False):
    """Return a real binding function with an explicit platform fixture."""
    namespace = {
        "defaultdict": defaultdict,
        "extract_layer_index": load_function(
            "vllm/model_executor/models/utils.py", "extract_layer_index", {}
        ),
        "current_platform": SimpleNamespace(
            is_npu=lambda: platform == "npu",
            is_cuda_alike=lambda: platform == "cuda",
            is_cpu=lambda: platform == "cpu",
            is_xpu=lambda: platform == "xpu",
        ),
    }
    relative = (
        "ascend/legacy_patches/worker/patch_qwen3_next_mtp.py"
        if donor
        else "vllm/v1/worker/utils.py"
    )
    return load_function(relative, "bind_kv_cache", namespace)


class NativeKVCacheBindingTests(unittest.TestCase):
    def bind(self, items, *, platform="npu", donor=False, num_attn_module=1):
        caches = dict(items)
        context = {name: SimpleNamespace() for name in caches}
        runner = []
        binding(platform, donor=donor)(caches, context, runner, num_attn_module)
        for name, value in caches.items():
            self.assertIsInstance(context[name].kv_cache, list)
            self.assertEqual(len(context[name].kv_cache), 1)
            self.assertIs(context[name].kv_cache[0], value)
        return runner

    def test_dsa_pair_order_and_identity(self):
        for prefix, indexer_first, tuple_value in itertools.product(
            ("model.layers.7", "model.layers.78.mtp_block"),
            (False, True),
            (False, True),
        ):
            with self.subTest(
                prefix=prefix, indexer_first=indexer_first, tuple=tuple_value
            ):
                latent = (object(), object()) if tuple_value else object()
                indexer = object()
                items = [
                    (f"{prefix}.self_attn.attn", latent),
                    (f"{prefix}.self_attn.indexer.k_cache", indexer),
                ]
                if indexer_first:
                    items.reverse()
                actual = self.bind(items)
                self.assertEqual(len(actual), 2)
                self.assertIs(actual[0], latent)
                self.assertIs(actual[1], indexer)

    def test_numeric_layer_order_with_consumer_and_mtp(self):
        values = [object() for _ in range(5)]
        names = [
            "model.layers.2.self_attn.attn",
            "model.layers.2.self_attn.indexer.k_cache",
            "model.layers.10.self_attn.attn",  # shared-indexer consumer
            "model.layers.78.mtp_block.self_attn.attn",
            "model.layers.78.mtp_block.self_attn.indexer.k_cache",
        ]
        items = list(zip(names, values))
        for order in itertools.permutations(items):
            with self.subTest(names=[name for name, _ in order]):
                self.assertEqual(self.bind(order), values)

    def test_non_dsa_duplicate_indices_keep_first_entry(self):
        cases = [
            ("self_attn", "cross_attn"),
            ("cross_attn.attn", "self_attn.indexer.k_cache"),
            ("other.self_attn.attn", "self_attn.indexer.k_cache"),
            ("self_attn.attn", "self_attn.indexer.k_cache", "cross_attn"),
        ]
        for suffixes in cases:
            for order in itertools.permutations(suffixes):
                with self.subTest(order=order):
                    items = [(f"model.layers.7.{suffix}", object()) for suffix in order]
                    self.assertEqual(self.bind(items), [items[0][1]])

    def test_single_cache_and_aliases_are_not_copied_or_deduplicated(self):
        shared = (object(), object())
        items = [
            ("model.layers.7.self_attn.attn", shared),
            ("model.layers.78.mtp_block.self_attn.attn", shared),
        ]
        runner = self.bind(items)
        self.assertEqual(len(runner), 2)
        self.assertIs(runner[0], shared)
        self.assertIs(runner[1], shared)
        self.assertEqual(
            self.bind([("model.layers.7.self_attn.indexer.k_cache", shared)]), [shared]
        )

    def test_empty_caches(self):
        self.assertEqual(self.bind([]), [])

    def test_nonempty_runner_is_rejected_before_mutation(self):
        runner = [object()]
        context = {"model.layers.1.self_attn.attn": SimpleNamespace()}
        with self.assertRaises(AssertionError):
            binding()({next(iter(context)): object()}, context, runner)
        self.assertEqual(len(runner), 1)
        self.assertFalse(hasattr(next(iter(context.values())), "kv_cache"))

    def test_invalid_layer_index_is_still_rejected(self):
        for name in ("model.self_attn.attn", "model.layers.7.blocks.2.self_attn.attn"):
            with self.subTest(name=name), self.assertRaises(AssertionError):
                self.bind([(name, object())])

    def test_num_attn_module_is_forwarded(self):
        items = [
            ("model.layers.7.attn.1", object()),
            ("model.layers.7.attn.0", object()),
        ]
        self.assertEqual(
            self.bind(items, num_attn_module=2), [items[1][1], items[0][1]]
        )

    def test_binding_has_only_native_npu_ordering(self):
        items = [
            ("model.layers.7.self_attn.indexer.k_cache", object()),
            ("model.layers.7.self_attn.attn", object()),
        ]
        for platform in ("cuda", "cpu", "xpu"):
            with self.subTest(platform=platform):
                self.assertEqual(
                    self.bind(items, platform=platform), [items[1][1], items[0][1]]
                )
        # Platform rejection happens before a worker is created, not here.
        self.assertNotIn("current_platform", binding().__code__.co_names)

    def test_matches_archived_donor_without_running_its_patch(self):
        names = [
            "model.layers.7.self_attn.attn",
            "model.layers.7.self_attn.indexer.k_cache",
            "model.layers.8.self_attn.attn",
            "model.layers.78.mtp_block.self_attn.attn",
            "model.layers.78.mtp_block.self_attn.indexer.k_cache",
        ]
        values = [object() for _ in names]
        for order in itertools.permutations(zip(names, values)):
            with self.subTest(names=[name for name, _ in order]):
                self.assertEqual(self.bind(order), self.bind(order, donor=True))

    def test_intranet_gate_accepts_correct_binding(self):
        spec = importlib.util.spec_from_file_location(
            "npu_bootstrap_tool", ROOT / "tools/check_npu_bootstrap.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.check_kv_cache_binding(
            binding(), (object(), object()), object(), object()
        )
        self.assertEqual(result["cases_passed"], 6)
        self.assertIn("v1/worker/utils.py", module.IDENTITY_FILES)
        self.assertIn("kv_cache_bind", module.ORDERS)

    def test_intranet_gate_rejects_incomplete_or_reordered_binding(self):
        spec = importlib.util.spec_from_file_location(
            "npu_bootstrap_tool", ROOT / "tools/check_npu_bootstrap.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        native = binding()
        for fault in ("drop-indexer", "reverse", "lose-context"):

            def broken(caches, context, runner):
                native(caches, context, runner)
                if fault == "drop-indexer":
                    runner.pop()
                elif fault == "reverse":
                    runner.reverse()
                else:
                    next(iter(context.values())).kv_cache = [object()]

            with self.subTest(fault=fault), self.assertRaises(RuntimeError):
                module.check_kv_cache_binding(
                    broken, (object(), object()), object(), object()
                )


if __name__ == "__main__":
    unittest.main()
