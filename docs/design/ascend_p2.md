# Native Ascend migration: P2 platform batch

This fork starts P2 from the preserved P1 commit
`230fbc0218e656cb90f8a8c2261150345a1d8ee5`. The local `p1` branch retains that
input; development uses `p2`. P1 acceptance and baseline defect repairs remain
pending. This batch establishes the native platform entry, not the whole P2 exit.

`NPUPlatform` now lives in `vllm.platforms.npu` with `PlatformEnum.NPU` and
`Platform.is_npu()`. The Ascend-only resolver checks for `torch_npu` without a
device probe and resolves the built-in class without platform entry-point
metadata. Missing `torch_npu` raises an error. Optional `VLLM_PLUGINS` filtering
does not disable the built-in platform or its required components.

The process component loader invokes `register_builtin_components()` before
optional general plugins. It registers the existing KV connectors, netloader,
rfork loader, and profiling configuration directly. Connector registration is
lazy; it does not import LMCache. The distribution no longer advertises Ascend
platform/general entry points. Optional LoRA resolver entries remain.

The old `vllm_ascend.platform` module temporarily re-exports the same class. The
remaining configuration, quantization, graph, communication, Runner, and patch
implementations still use `vllm_ascend` and will migrate in later batches.
Custom-op and MoE dispatch explicitly retain their existing Ascend bindings when
`is_out_of_tree()` becomes false. Both group coordinators explicitly select the
NPU for their local rank. Native kernel resources remain under the packaged
Ascend directory, including the strict editable link tree.

The vLLM package version is `0.18.0+ascend.p2`, paired with LMCache
`0.4.3+ascend.p1`. Both repositories' historical `p1_dev.py` helpers validate this
pair on `p2`; the copies on `p1` retain their original versions. Build and install
only in the prepared intranet environment. Do not mix old plugin distributions
or copy P2 modules into a running P1 installation.

Run the relevant host contracts without importing torch or using an NPU:

```bash
python3 -B tests/standalone/test_p2_platform.py -v
python3 -B ascend/tests/standalone/test_p1_resources.py -v
python3 -B tests/standalone/test_p1_development.py -v
```

These use dependency stubs, production AST methods, and synthetic build fixtures.
They do not establish full imports, wheel/ABI correctness, model inference, or
performance equivalence. NPU validation must later cover clean wheel/editable
imports, optional-plugin filtering, spawn startup, and the existing GLM/DSA/MTP
offline, online, cache, and recovery scenarios.
