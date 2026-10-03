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
The 2026-10-03 configuration repair recognizes that exact old class together
with `lmcache_ascend.integration.vllm.lmcache_ascend_connector_v1` (or no module
path), warns, and normalizes the two launch fields to the formal native entry.
All other KV transfer settings are retained. This does not provide an importable
old package or restore runtime patching. Unknown `lmcache_ascend.*` module
configurations fail during config construction; explicit custom modules outside
that namespace are not redirected. The old `use_native=true` vendored adapter
is also rejected before model loading.

Rebuild both packages; do not reuse P2 libraries. Use dedicated environments
and checkout paths for simultaneous editable validation. Do not switch a
checkout used by a running service. The isolation flag does not create an
environment or bypass paired-version checks.

Run `tools/validate_npu_native.py` for the no-LMCache native import/spawn
contract and the paired LMCache `tools/p3_runtime_smoke.py` for installed
cache integration. Device smoke tests require an idle test NPU. Neither proves
2P2D model/cache/recovery/performance acceptance.

The bootstrap `config` check also resolves the migrated connector through the
real factory without constructing a connector, loading weights or starting a
service. It verifies the installed `config/kv_transfer.py` against this checkout.
For this Python-only repair, an existing correctly installed P3 strict editable
environment needs the updated files and a restart of its test processes, not a
native rebuild. P2-to-P3 upgrades still require both packages to be reinstalled;
ordinary wheel installations need a newly built P3 wheel.

Workspace instructions and paired commits are under `design/p3/`.
P4 broad model/device pruning is not part of this source delivery.
