# Native Ascend P2

P2 source integration is complete on `p2`; native build, ABI, model and performance
acceptance remain pending in the intranet environment. The preserved `p1` and
`main` input is `230fbc0218e656cb90f8a8c2261150345a1d8ee5`. P1 baseline retests and
known defect fixes are deferred by the user. This is not P3 LMCache integration
or P4 removal of other device/model source trees.

## Native ownership

| Concern | Owner |
| --- | --- |
| Built-in NPU platform and configuration | `vllm.platforms.npu`, `vllm.config.ascend`, `vllm.envs_ascend` |
| NPU worker and v1 Runner/state | `vllm.v1.worker.npu_worker`, `npu_model_runner`, `npu_runner_state` |
| v2 Runner/state/components | `vllm.v1.worker.npu.v2` |
| Shared batch/output/graph data | `vllm.v1.worker.input_batch`, `runner_output`, `vllm.compilation.graph_types` |
| Attention/MLA/SFA | `vllm.v1.attention.backends.ascend` |
| Operators and immutable layer selection | `vllm.model_executor.layers.ascend`, `ascend.registry` |
| Quantization and model loading | `vllm.model_executor.layers.quantization.ascend`, `vllm.model_executor.model_loader.ascend` |
| Sampling and MTP | `vllm.v1.sample.ascend`, `vllm.v1.spec_decode.ascend`, canonical DeepSeek model definitions |
| ACL/TorchAir | `vllm.compilation.ascend` |
| HCCL, EPLB, sparse offload and connectors | `vllm.distributed.ascend`, `device_communicators`, `eplb.ascend`, `kv_transfer.ascend` |

The platform is selected without external entry-point metadata. Disabling
optional plugins does not disable NPU. Layer/quantization/loader/connector
selection is explicit and lazy; layer selection uses `forward_npu` rather than
OOT overrides. Both NPU Runners use independent state implementations with
explicit NPU APIs, not GPU Runner inheritance or CUDA module wrappers.

Effective patch behavior is integrated at its owner: MLA spec identity and
layout, worker metadata before token consumption, scheduler/recovery budgets,
HCCL coordination, GLM tools/usage, MTP weights, logprobs, rejection sampling,
Quarot instance loading and v2 input/sample state. Disabled-cache imports do not
require LMCache. Model-local weight/forward callbacks and real operator
registration remain part of model execution.

The TorchAir dictionary adapter binds dependencies in one compiler's private
callable scope without mutating installed modules/classes. It preserves installed
compiler passes and converter bookkeeping, checks the private API it needs, and
fails explicitly on an incompatible factory. Validate the installed TorchAir
build and actual npugraph_ex execution in the intranet environment.

## Packaging and pairing

The only vLLM Python namespace is `vllm`; the extension is `vllm._ascend_C`.
CANN resources, kernel library, generated version and build metadata are rooted
at the installed `vllm` package, including strict editable links. Existing
`torch.ops._C_ascend`, `libvllm_ascend_kernels.so`, vendor names, environment
variables and diagnostic output locations retain their ABI/operational meaning.

Old plugin/patch source is preserved under `ascend/legacy_plugin` and
`ascend/legacy_patches`, with off-profile patch tests under `ascend/legacy_tests`.
These are reference archives excluded from wheel/sdist and are never imported.
There is no `vllm_ascend` compatibility package. Other device/model implementation
files await P4; their presence is not an active NPU dependency or certification.

Pair `vllm==0.18.0+ascend.p2` with the delivered LMCache commit at
`0.4.3+ascend.p1`. LMCache shared diagnostic/event/Mooncake imports changed, so an
older same-version LMCache wheel is not interchangeable. Both historical
`p1_dev.py` tools enforce the package pair; commit identities are recorded in the
workspace handoff at `design/p2/baseline/p2-native-integration-20260927.json`.

## Verification

On a source workstation, run `python3 tools/check_npu_native.py` and host tests
without installing torch/CANN or compiling artifacts. The delivered host suite
passes 167 tests, with 22 paired installation tests in LMCache. Framework-only
cases remain explicitly assigned to intranet validation. These results use
stubs, production AST methods and synthetic packaging fixtures; they do not
establish full imports, NPU ABI, numerical or performance equivalence.

Build/install only in a clean intranet environment using `p1_dev.py` with the
approved Python 3.11/aarch64, 910B3/CANN 8.5.1, torch 2.9.0, torch_npu 2.9.0.post2
profile. Use a new native build/link tree; do not copy P1 libraries. From outside
the checkout after installation:

```bash
python /path/to/vllm/tools/validate_npu_native.py --spawn --torchair-abi \
  --output /new/results/native-import.json
```

An additional `--device-smoke` loads the extension and checks stream/event/tensor
operations on visible device 0. Full validation still requires native kernels,
HCCL, eager/ACL/npugraph_ex, GLM-5.2 offline/online, DSA two groups/MTP/C8-off,
CPU KV offload, P/D, checkpoint/RemoteFill/recovery and both TP8/DP2 and TP4/DP4.
Complete commands, patch disposition, migration hashes and deferred P1 issues
are delivered alongside the repositories under `design/p2/`.
