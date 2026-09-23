# Layerwise-prefill profile boundary

Source commit `e87a19ca334216c9a30fc7859948852d0ea3063a` is a profile-harness
commit, not a production serving implementation. Its complete parent-to-commit
path inventory is recorded below so that the source deletion and every omitted
diagnostic asset have an explicit native-layout disposition.

## Source-to-target disposition

| Source path | Status | Falcon disposition |
|---|---|---|
| `tools/layerwise_prefill_model_test.py` (D) | retired/replaced | The preceding native port of source `f9d4f34a` owns the replacement at the same target path. It now uses the formal `LMCacheConnectorV1` name; the source plugin launcher is not restored. |
| `examples/layerwise_prefill/article_summary.txt` | profile-only | Empty-with-rationale: fixed prompt input, not imported by serving code or supported host contracts. |
| `examples/layerwise_prefill/article_summary_80k.txt` | profile-only | Empty-with-rationale: device-profile input only; long-context acceptance remains an intranet activity. |
| `examples/layerwise_prefill/article_summary_100k.txt` | profile-only | Empty-with-rationale: device-profile input only; no native runtime owner is changed. |
| `tools/layerwise_prefill_check.py` | plugin harness | Empty-with-rationale: uses file-backed KV/tensor probes and source-only experimental assumptions; it has no Falcon native owner. |
| `tools/layerwise_prefill_mooncake_check.py` | plugin harness | Empty-with-rationale: depends on the source plugin and test-only transport setup, not the supported native serving path. |
| `tools/layerwise_prefill_mooncake_routing.py` | plugin harness | Empty-with-rationale: test-only Mooncake routing/worker injection; copying it would reintroduce a second connector lifecycle. |
| `tools/layerwise_prefill_profile.py` | plugin harness | Empty-with-rationale: installs profiler and dummy-DMA/store hooks and references `lmcache_ascend`; it cannot be used as a native production tool. |
| `tools/layerwise_prefill_profile_worker.py` | plugin harness | Empty-with-rationale: monkey-patches connector/operation owners and deliberately produces invalid outputs for diagnosis. |

The source commit therefore has no additional Falcon runtime diff beyond the
native replacement already made by the `f9d4f34a` port. The target tree deliberately
retains a native smoke tool at `tools/layerwise_prefill_model_test.py`, but it
is not the deleted source plugin file and must use `LMCacheConnectorV1`.

Any future NPU profile should run separate fresh processes for baseline, D-node,
and opt-in P-node cases with identical model, prompt, device topology, and
chunk limits. Record startup separately from the request, preserve raw profiler
artifacts, and report cache-hit status. Do not compare a cache hit with a fresh
prefill or infer a release speedup from host timing.

Required before any serving claim: native connector transfer-window support,
CANN/NPU graph and collective validation, output parity, preemption/resume
coverage, distributed bank mapping, device ABI checks, and repeated
performance runs on the target hardware. The layerwise redesign remains
experimental and non-serving-qualified.

Port result for source commit `e87a19ca334216c9a30fc7859948852d0ea3063a`:
`empty-with-rationale` after the path-by-path audit above. This is not an
assertion that the source profile ran successfully, nor a substitute for the
blocked NPU/ABI/profile validation.
