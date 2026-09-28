# P3 paired LMCache integration

The `p3` branch starts at the complete P2 input
`f1be323571e3ca2aab53992234045dd064d1967f`. The retained `p2`, `p1` and `main`
branches are unchanged. P3-01 changes package identity and paired installation
checks, not vLLM's P2 native execution code.

Pair `vllm==0.18.0+ascend.p3` with the same-batch
`lmcache==0.4.3+ascend.p3`. Both historical `p1_dev.py` tools keep their build,
strict editable, install and verify interfaces. Use a dedicated environment;
`--isolated-env` confirms isolation but does not create it or permit mixed P2/P3
distributions. Rebuild both packages instead of reusing old native artifacts.

LMCache P3-01 integrates native configuration only. Engine, connector, extension
and remaining import-patch migration is still pending. In particular, this batch
does **not** eliminate the LMCache internal `lmcache_ascend` package. Full P3
acceptance requires later batches and intranet ABI/NPU/cache/recovery checks.
P4 device/model pruning is not part of this batch.

Workspace instructions and exact paired commits live under `design/p3/`.
Keep separate P2/P3 environments, reports and cache namespaces even when both
phases are validated in one campaign. Do not switch the source checkout of a
running editable installation. No native build or installation was performed
on the source workstation.
