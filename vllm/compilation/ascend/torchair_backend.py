# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Per-compiler Triton dictionary support for the pinned TorchAir interface.

TorchAir's compiler factory does not expose a converter argument. Bind its
converter dependency in a private callable scope, preserving the installed
compiler's optimizations and ABI. No module, class or torch API is modified;
two compiler instances can compile concurrently with different converters.
This replaces the old process-wide ValuePack assignment and module reloads.
"""

from functools import partial
from types import CodeType, FunctionType


def _global_names(code):
    names = set(code.co_names)
    for constant in code.co_consts:
        if isinstance(constant, CodeType):
            names.update(_global_names(constant))
    return names


def bind_callable(function, **dependencies):
    """Construct a callable with explicit local dependencies, leaving its owner intact."""
    if not isinstance(function, FunctionType):
        raise RuntimeError(
            "Unsupported TorchAir callable ABI; expected a Python function"
        )
    missing = set(dependencies).difference(_global_names(function.__code__))
    if missing:
        raise RuntimeError(f"Unsupported TorchAir dependency ABI: {sorted(missing)}")
    scope = dict(function.__globals__)
    scope.update(dependencies)
    bound = FunctionType(
        function.__code__,
        scope,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    bound.__kwdefaults__ = function.__kwdefaults__
    return bound


def get_npu_backend(compiler_config):
    """Build an isolated backend using the installed, approved TorchAir version."""
    import torchair
    from torch.fx import Interpreter
    from torchair import npu_fx_compiler as compiler_api
    from torchair.core._concrete_graph import ValuePack, _is_symlist

    class DictionaryValuePack(ValuePack):
        def __getitem__(self, key):
            if isinstance(self.meta, dict):
                return self.meta.get(key)
            raise ValueError(f"Unsupported dictionary metadata: {type(self.meta)}")

    def unpack(value, part):
        if isinstance(value, ValuePack):
            return getattr(value, part)
        if isinstance(value, dict):
            return {key: unpack(item, part) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(unpack(item, part) for item in value)
        return value

    def unpack_meta(args, kwargs):
        return unpack(args, "meta"), unpack(kwargs, "meta")

    # Retain the installed converter's fake mode, tracing and IR bookkeeping.
    wrap = bind_callable(
        compiler_api._NpuGraphConverter._wrap, _unpack_meta=unpack_meta
    )

    class NativeGraphConverter(compiler_api._NpuGraphConverter):
        def placeholder(self, target, args, kwargs):
            value = super().placeholder(target, args, kwargs)
            return DictionaryValuePack(value.meta, value.npu)

        def _unpack_npu(self, args, kwargs):
            def npu_value(value):
                if isinstance(value, (list, tuple)) and value and _is_symlist(value):
                    return self._graph.parse_symlist(value)
                return unpack(value, "npu")

            return [npu_value(value) for value in args], {
                key: npu_value(value) for key, value in kwargs.items()
            }

        def _get_value_pack(self, meta_outputs, npu_outputs):
            if isinstance(npu_outputs, (list, tuple)):
                return [
                    self._get_value_pack(meta, npu)
                    for meta, npu in zip(meta_outputs, npu_outputs)
                ]
            return DictionaryValuePack(meta_outputs, npu_outputs)

        def _wrap(self, method):
            return wrap(self, method)

        def output(self, target, args, kwargs):
            meta = Interpreter.placeholder(
                self, target, unpack(args, "meta"), unpack(kwargs, "meta")
            )
            return self._graph.parse_output(target, args, kwargs, meta)

    # Source-preserving dependency injection keeps all installed TorchAir graph
    # passes, static-kernel options, input mutation handling and code generation.
    generate = bind_callable(
        compiler_api._NpuFxCompiler._gen_compiled_gm,
        _NpuGraphConverter=NativeGraphConverter,
    )

    class NativeCompiler(compiler_api._NpuFxCompiler):
        def _gen_compiled_gm(self, *args, **kwargs):
            return generate(self, *args, **kwargs)

    backend = torchair.get_npu_backend(compiler_config=compiler_config)
    if (
        not isinstance(backend, partial)
        or backend.func is not compiler_api._npu_backend
    ):
        raise RuntimeError(
            "Unsupported TorchAir backend factory ABI; verify the approved intranet build"
        )
    compile_graph = bind_callable(backend.func, get_compiler=NativeCompiler)
    return partial(compile_graph, *backend.args, **backend.keywords)
