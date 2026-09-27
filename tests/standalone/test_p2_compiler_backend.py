# SPDX-License-Identifier: Apache-2.0
"""Exercise compiler dependency binding against a small, explicit TorchAir ABI."""

import importlib.util
import sys
import unittest
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "native_torchair_backend", ROOT / "vllm/compilation/ascend/torchair_backend.py"
)
NATIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(NATIVE)


class ValuePack:
    def __init__(self, meta, npu=None):
        self.meta = meta
        self.npu = meta if npu is None else npu


class Interpreter:
    def placeholder(self, target, args, kwargs):
        return args[0]


def module(name, **attributes):
    result = ModuleType(name)
    result.__dict__.update(attributes)
    return result


def compiler_fixture():
    api = module("torchair.npu_fx_compiler", ValuePack=ValuePack)
    # Globals intentionally emulate the third-party dependency points. No real
    # framework is imported; calling the backend actually exercises both binds.
    exec(
        """
def _unpack_meta(args, kwargs):
    return args, kwargs

class _NpuGraphConverter:
    def __init__(self, graph):
        self._graph = graph

    def placeholder(self, target, args, kwargs):
        return ValuePack(args[0], kwargs['npu'])

    def _wrap(self, method):
        def convert(target, args, kwargs):
            meta_args, meta_kwargs = _unpack_meta(args, kwargs)
            npu_args, npu_kwargs = self._unpack_npu(args, kwargs)
            # This marker represents bookkeeping in the installed converter
            # that the native adapter must preserve.
            self._graph.visits += 1
            return self._get_value_pack(
                target(*meta_args, **meta_kwargs),
                target(*npu_args, **npu_kwargs),
            )
        return convert

class _NpuFxCompiler:
    def __init__(self, compiler_config):
        self.config = compiler_config

    def _gen_compiled_gm(self, graph):
        return _NpuGraphConverter(graph)

def get_compiler(compiler_config):
    return _NpuFxCompiler(compiler_config)

def _npu_backend(graph, example_inputs, compiler_config=None, decompositions=None):
    compiler = get_compiler(compiler_config)
    return compiler._gen_compiled_gm(graph), compiler.config, decompositions
""",
        api.__dict__,
    )
    decompositions = {"preserved": object()}
    air = module(
        "torchair",
        npu_fx_compiler=api,
        get_npu_backend=lambda *, compiler_config: partial(
            api._npu_backend,
            compiler_config=compiler_config,
            decompositions=decompositions,
        ),
    )
    modules = {
        "torch": module("torch"),
        "torch.fx": module("torch.fx", Interpreter=Interpreter),
        "torchair": air,
        "torchair.npu_fx_compiler": api,
        "torchair.core": module("torchair.core"),
        "torchair.core._concrete_graph": module(
            "torchair.core._concrete_graph",
            ValuePack=ValuePack,
            _is_symlist=lambda value: value == ["symbol"],
        ),
    }
    return api, air, modules, decompositions


class CompilerBackendTests(unittest.TestCase):
    def test_nested_dictionary_metadata_and_npu_values_use_private_converter(self):
        api, air, modules, decompositions = compiler_fixture()
        original = dict(api.__dict__)
        config = object()
        graph = SimpleNamespace(
            visits=0,
            parse_symlist=lambda value: ("parsed", value),
            parse_output=lambda target, args, kwargs, meta: meta,
        )
        with patch.dict(sys.modules, modules):
            first = NATIVE.get_npu_backend(config)
            second = NATIVE.get_npu_backend(config)
            converter, actual_config, actual_decompositions = first(graph, [])
            other, _, _ = second(graph, [])
            self.assertIsNot(type(converter), type(other))
            packed = converter.placeholder("input", ({"x": 3},), {"npu": {"x": 7}})
            self.assertEqual(packed["x"], 3)
            self.assertEqual(
                converter._unpack_npu((["symbol"],), {}), ([("parsed", ["symbol"])], {})
            )
            result = converter._wrap("call_function")(
                lambda values: values["nested"][0]["x"] + 1,
                ({"nested": [packed]},),
                {},
            )
            self.assertEqual((result.meta, result.npu), (4, 8))
            self.assertEqual(graph.visits, 1)
            self.assertEqual(converter.output("output", (result,), {}), 4)
        self.assertIs(actual_config, config)
        self.assertIs(actual_decompositions, decompositions)
        self.assertEqual(api.__dict__, original)
        self.assertFalse(hasattr(ValuePack, "__getitem__"))

    def test_incompatible_backend_factory_fails_without_patching_third_party(self):
        api, air, modules, _ = compiler_fixture()
        original = dict(api.__dict__)
        air.get_npu_backend = lambda **kwargs: lambda graph: graph
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "backend factory ABI"):
                NATIVE.get_npu_backend(object())
        self.assertEqual(api.__dict__, original)


if __name__ == "__main__":
    unittest.main()
