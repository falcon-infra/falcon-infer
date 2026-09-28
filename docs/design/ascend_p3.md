# P3 paired native LMCache integration

P3 contains the complete retained P2 input
`f1be323571e3ca2aab53992234045dd064d1967f`. P2/P1/main remain unchanged.
LMCache native source integration is now delivered; native builds, ABI and
joint P2/P3 inference acceptance remain pending in the intranet.

Pair `vllm==0.18.0+ascend.p3` with same-delivery
`lmcache==0.4.3+ascend.p3`. Both `p1_dev.py` helpers retain build, strict
editable, install and verify interfaces. Resources now belong to only
`vllm` and `lmcache`, with no `lmcache_ascend` runtime package.

The formal `LMCacheConnectorV1` owns DSA capabilities, checkpoint/preemption,
worker metadata, live-source event handoff, destination sealing and RemoteFill
paired-restart forwarding. No LMCache plugin mutates the vLLM class.
LMCache remains a lazy optional dependency when KV transfer is disabled.

For P3 use `kv_connector="LMCacheConnectorV1"` and remove the retired
`kv_connector_module_path`. Preserve other fields, including engine ID and
role. The historical `LMCacheAscendConnector` registry name also resolves
directly to the formal class, not a compatibility module.
Custom loaders can select `LMCacheConnectorV1Dynamic` from
`lmcache.integration.vllm.lmcache_connector_v1`. The old
`LMCacheAscendConnectorV1Dynamic` class/module path is intentionally retired.

Rebuild both packages; do not reuse P2 libraries. Use dedicated environments
and checkout paths for simultaneous editable validation. Do not switch a
checkout used by a running service. The isolation flag does not create an
environment or bypass paired-version checks.

Run `tools/validate_npu_native.py` for the no-LMCache native import/spawn
contract and the paired LMCache `tools/p3_runtime_smoke.py` for installed
cache integration. Device smoke tests require an idle test NPU. Neither proves
2P2D model/cache/recovery/performance acceptance.

Workspace instructions and paired commits are under `design/p3/`.
P4 broad model/device pruning is not part of this source delivery.
