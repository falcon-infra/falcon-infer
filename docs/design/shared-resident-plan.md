# Experimental shared resident planning

`VLLM_ASCEND_SFA_SHARED_RESIDENT_PLAN=1` enables one resident sparse planner
for each structural `indexer_types` producer/consumer group. It is disabled by
default and must be selected before worker startup.

The opt-in path requires the existing two-group compact-scratch configuration:

- `VLLM_ASCEND_DSA_TWO_GROUPS=1`
- `VLLM_ASCEND_DSA_UNBUNDLE=1`
- `VLLM_ASCEND_DSA_SHRINK_LATENT=2`
- resident caching enabled, top-k 2048, and MTP 1 or 2

The runner validates the group topology, common top-k publication buffer,
resident geometry, latent KV slot layout, and supported non-CP SFA features
before capture. Incompatible groups fail closed. Eligible producer/consumer
layers share the producer's fixed state/workspace and explicit functionalized
write tensors; each layer retains its own KV retrieval and staged bridge.

Mixed or prefill steps fall back to the ordinary planner and invalidate shared
residency. Preemption invalidates state even when resumed requests receive the
same physical block IDs. A partial forward is fail-stop: restart the worker
before reusing planned residency.

This is not release acceptance. Host contracts cover ownership, preemption,
operator schema, and AOT clone policy. NPU graph replay, device ABI,
distributed behavior, output parity, and performance remain required before
serving qualification. The resident redesign and comparison benchmarks remain
isolated experiments and are not serving dispatch.
