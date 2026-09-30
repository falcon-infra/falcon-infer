# Native NPU bootstrap and KV cache binding

The 2026-09-29 fix applies to the P2 repair branch `fix/p2-npu-bootstrap` and
is carried into P3. The original P2/P1/main inputs remain unchanged. Use the
workspace delivery manifest for exact paired commits; a version suffix alone
does not identify the installed Python sources.

The 2026-09-30 follow-up also restores DSA KV cache binding in the native worker
utility. Use the latest paired manifest, not the first bootstrap-fix commit.

## Cause and fix

The registry launches a fresh `python -m vllm.model_executor.models.registry`
interpreter. Its parent-package imports request `current_platform`. Importing
`platforms/npu.py` previously imported `config.ascend` and `utils.ascend` before
defining `NPUPlatform`; these imported `config.compilation`, which requested the
still-uninitialized platform again. This is an import cycle, not an unsupported
GLM architecture or a TP/DP configuration error.

`platforms/ascend_constants.py` now owns three dependency-free bootstrap
constants, re-exported unchanged by `utils.ascend`. The platform imports config
and runtime utilities only in the methods that use them. A normal `torch_npu`
import registers the NPU device type; the platform bootstrap does not call
device initialization, allocation or availability probes. No fake platform,
global transfer shim, model-inspection bypass or registry-cache workaround is
used. Compute, quantization and communication method bodies are unchanged apart
from import placement.

## DSA KV cache binding

The former `patch_qwen3_next_mtp.py` was incorrectly classified as non-target
model code during P2 migration. Despite its name, it supplied the common Ascend
KV-binding contract used by GLM DSA. Without it, the native worker utility rejects
an NPU layer with multiple caches. Its archived tests had also been excluded.

The native `vllm.v1.worker.utils.bind_kv_cache` now preserves that contract for
NPU: sort numeric layer indices; retain an exact `.self_attn.attn` and sibling
`.self_attn.indexer.k_cache` pair in latent/indexer order regardless of input
order; retain the first entry for other duplicate indices. All forward-context
entries still receive their original objects, including shared references and
tuples. MTP prefixes use the same rule. Non-NPU behavior is unchanged; this is not
additional model/device certification. The old import-time patch remains retired.

`tests/standalone/test_npu_kv_cache_binding.py` runs the production function bodies
with host fixtures, includes donor comparisons, and covers producer/consumer/MTP
ordering, reference identity, and existing validation behavior. It does not prove
NPU tensor operations or real model startup.

## Validation

`tests/standalone/test_npu_platform_bootstrap.py` executes the complete platform
module and selector in new processes, using explicit dependency-edge fixtures
and torch/interface stubs. It covers platform-first, config-first, Ascend-utils-
first and the registry parent-package chain. This is host evidence, not a full
framework or NPU import test.

In the prepared intranet environment, from outside source checkouts:

```bash
python -B /path/to/vllm/tools/check_npu_bootstrap.py \
  --inspect-glm --output /path/to/new-bootstrap-report
```

The installed check uses fresh interpreters for four import orders, the real
registry subprocess protocol, and native KV binding with small explicit CPU
tensor views. Binding checks both pair input orders, normal/MTP producer names,
a shared-indexer consumer, a single cache and empty input. No NPU tensor or kernel
is used by the binding check. `--inspect-glm` additionally inspects
`GlmMoeDsaForCausalLM` without using an existing model-info cache. It does not load
weights or run inference. Normal module initialization can load native libraries
and generate the existing service-profiling configuration. Optional plugins are
disabled only in the tool's children; serving processes are not modified.

The report checks the installed version against the tool's checkout and compares
seven bootstrap/binding source files. Expect seven passing checks when using
`--inspect-glm`. It does not certify every installed file, native ABI,
model numerics or 2P2D. Timeouts and failed subprocesses remain failures; the tool
terminates only its own newly created process groups. Existing report directories
are not overwritten.

The September 29 fix adds a Python module. Strict editable users must reinstall
that version with the branch's `p1_dev.py`. The September 30 production change is
in existing Python files; rebuilding/reinstalling with the existing workflow also
refreshes source provenance and verifies the cumulative fix. Do not switch commits beneath a running process
or reuse a stale link tree. Ordinary wheel users rebuild and reinstall. Preserve
separate P2/P3 source checkouts, environments, caches and acceptance reports.
