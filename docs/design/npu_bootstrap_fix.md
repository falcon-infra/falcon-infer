# Native NPU bootstrap and registry cold imports

The 2026-09-29 fix applies to the P2 repair branch `fix/p2-npu-bootstrap` and
is carried into P3. The original P2/P1/main inputs remain unchanged. Use the
workspace delivery manifest for exact paired commits; a version suffix alone
does not identify the installed Python sources.

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

The installed check uses fresh interpreters for four import orders and the real
registry subprocess protocol. `--inspect-glm` additionally inspects
`GlmMoeDsaForCausalLM` without using an existing model-info cache. It does not load
weights or run inference. Normal module initialization can load native libraries
and generate the existing service-profiling configuration. Optional plugins are
disabled only in the tool's children; serving processes are not modified.

The report checks the installed version against the tool's checkout and compares
six bootstrap source files. It does not certify every installed file, native ABI,
model numerics or 2P2D. Timeouts and failed subprocesses remain failures; the tool
terminates only its own newly created process groups. Existing report directories
are not overwritten.

The fix adds a Python module. Strict editable users must reinstall with the
branch's `p1_dev.py`; do not just switch source commits beneath a running process
or reuse a stale link tree. Ordinary wheel users rebuild and reinstall. Preserve
separate P2/P3 source checkouts, environments, caches and acceptance reports.
