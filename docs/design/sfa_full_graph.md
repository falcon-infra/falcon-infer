# Single-forward SFA graph (experimental)

The native Falcon owner can capture the target model forward, including sparse
KV loads, in one ACL graph. It is disabled by default and does not include
prefill, sampling, the MTP draft forward, or the generation loop.

Enable only with the existing staged SFA compilation configuration:

```bash
export VLLM_ASCEND_SFA_STAGED_GRAPH=1
export VLLM_ASCEND_SFA_STAGED_GRAPH_CAPTURE_SIZES=1
export VLLM_ASCEND_SFA_FULL_GRAPH=1
```

The graph owner supports bounded Q1/Q2 request lanes, internal TP/DP
coordination, and a separate attention-only padding lane. Q1 and configured
MTP-Q2 shapes are captured at startup; unsupported decode routes fail instead
of silently falling back to per-layer callbacks. Host tests do not establish NPU
capture, output parity, ABI, or performance acceptance; those remain an
intranet validation gate.

Each parity engine explicitly releases its cache connector and Ascend process
groups after generation, shuts down the engine, and waits for its identified
workers to exit before the next engine starts. A release/exit failure aborts
the comparison; the driver does not kill unrelated processes or change HCCL
ports. CPU process-lifecycle regressions are covered by
`tests/ut/tools/test_sfa_parity_shutdown.py` and
`tests/ut/attention/test_sfa_parity_worker_shutdown.py`. These tests do not
verify native HCCL port release on NPU hardware.

With `--compare-output`, the driver also reports descriptive `[SFA_STATS]`
absolute distributions for every aligned decode stage: `diff_abs`,
`eager_abs`, and `graph_abs`, including mean, population variance, maximum and
nonzero count. Statistics are element-weighted across aligned steps and TP
ranks, exclude padded or differently selected logical KV entries, and never
use a floating-point tolerance gate. Alignment stops pooling after the first
input/history divergence and reports partial coverage. `--trace-residual` may
be combined with `--compare-output` for additional norm/attention stages; it
can affect compiler fusion and remains diagnostic rather than acceptance.

## Eight-layer performance and MindStudio profile

Use `tools/sfa_graph_benchmark.py`, not the numerical parity driver, for the
performance experiment:

```bash
set -o pipefail
python tools/sfa_graph_benchmark.py --profile 2>&1 | tee log.log
```

The driver starts and closes two fresh TP8/DP1 engines sequentially: `staged`
(`VLLM_ASCEND_SFA_FULL_GRAPH=0`) and `full` (`=1`). Both retain staged graph
and PIECEWISE compilation; the baseline is not enforce-eager. The workload is
the eight-layer dummy-weight fixture with MTP1, a 4351-token prompt and 512
output tokens. It does not install parity probes, snapshots, checkpoint
restore, per-layer fences or replacement sampling. Prefill is excluded from
decode timing.

Each full-graph capture capacity has two target topologies. Before a forward,
the runner classifies every scheduled request width once on the CPU. An all-Q2
MTP batch uses `spec_fixed`, the original uniform request-major planner layout;
Q1-only and mixed Q1/Q2 batches use `decode_bounded`, with device-only lane
packing and restoration. Without MTP, all-Q1 uses the corresponding uniform
path. Both topologies retain one root replay, share source tables/resident
state, and preserve idle-DP participation; capture startup and graph memory
include both variants.

Defaults are now zero extra warmups and one measured request per mode
(`--warmups 0 --repeats 1`). One-request TPOT reports `std=null`/`n/a`; it is a
quick comparison, not a multi-request stability estimate. With `--diagnose`,
that same request supplies inline diagnostics and TPOT includes instrumentation
overhead. Explicit multi-request diagnostics use a separate host-timing request,
but supported captured timing markers remain in all requests; diagnostics are
conservatively labelled instrumented even when markers are unavailable.
Instrumented and uninstrumented results cannot be compared together.

To explicitly restore the longer benchmark, use `--warmups 1 --repeats 5`.

TPOT is measured at the offline engine output boundary, from the first
committed-token emission to the last, divided by additional committed tokens.
It includes scheduling, IPC, MTP, sampling and cache work. The report prints
mean/median/std, reduction and speedup, but has no hard-coded performance pass
threshold; differing dummy outputs are reported because MTP acceptance or
routing can confound timing.

With `--profile`, each engine makes a separate decode-only profiler request
after its timed requests. Both engines exit before trace parsing. The checker
audits every rank conservatively: `ONE_EXECUTE_PER_ROOT` only proves one
outermost known ACL execute API in each CPU root scope, while
`SPLIT_DETECTED` fails and `UNVERIFIED` is not a pass. Device-side graph
coverage and KV correctness still require manual MindStudio/NPU inspection.

If the CPU trace check fails after export, do not rerun the model. Recheck the
existing profile data only:

```bash
python tools/sfa_graph_trace.py --latest profile 2>&1 | tee -a log.log
```

Use `--run-dir profile/sfa-...` for an older run and `--ranks N` for another
TP size. This command does not import vLLM or torch_npu, recapture or re-export
anything; it only regenerates `*-trace-check.json` and leaves trace and
performance files unchanged. Numeric strings and JSON numbers are normalized
with decimal precision, while malformed timing is reported as `UNVERIFIED`.

For profiler-free request diagnostics, use a separate 5000-token run:

```bash
python tools/sfa_graph_benchmark.py --prompt-tokens 5000 --diagnose 2>&1 | tee log.log
```

`--diagnose` is mutually exclusive with `--profile`. With the defaults, each
engine runs only one 512-token generation, and temporary timing wrappers
collect statistics during that same request; there is no hidden warmup or
second diagnostic generation. Both the log and `comparison.json` flag that
TPOT includes diagnostic overhead, so it is not a clean performance
measurement. If captured-event capability is supported, timing records are
added inside existing opaque SFA/TP operations; if not, host sampling and
graph-external timing continue without fake layer rows. With explicit
`--warmups 1 --repeats 5`, host diagnostics run once after the ordinary
measurements, but supported markers remain in all requests. Omit `--diagnose`
for clean performance measurements. Scheduler history excludes all
chunked-prefill steps, including a final one-token Q1 chunk. Timing stop and
synchronization happen after the request, with bounded event storage and
`event_drops` reporting partial stream coverage. Results are printed as compact
`[SFA_TIMING]` lines and saved separately from TPOT JSON; they are diagnostics,
not performance acceptance or pure kernel timings. Actual capture/replay,
synchronization and unrelated device errors still stop startup rather than
being swallowed as capability limitations.

Per-rank diagnostic output includes `graph_profiles`, counting the selected
`spec_fixed` and `decode_bounded` topology without reading device tensors. The
counter is diagnostic only; root replay counts and total TPOT remain the
comparison contract.

Captured events are recorded inside existing opaque SFA and TP operations at
startup capture; replay adds no per-layer Python callbacks or graph splits. A
startup smoke check replays a tiny graph twice and rejects stale captured
timestamps before loading weights. A known `event recorder null (507000)` or
missing timestamp API disables captured graph markers on all TP ranks; no
per-layer values are invented. Coarse current-stream events cover
target-forward, root-replay, logits and MTP boundaries. Fine sampling and
staged-retrieval events are sampled on the first four decode forwards and then
every 32 forwards, while host wall/CPU statistics cover every decode forward.

When captured timestamps are supported, `graph_last.L0` through `graph_last.L7`
describe only the last live target decode. They expose pre/indexer/select/transfer/attention/post phases, the
`pre_to_post` and `after_post` gaps, and `graph_last.TP.*` collective spans.
Missing or duplicate phases are `INCOMPLETE`/`unobserved`, never zero; nested
phases and TP spans must not be added, and these are not whole-request averages
or isolated kernel timings. Sampling rows use the existing scoped rejection
recorder and call original sampler operations exactly once without tensor
inspection or forced readiness.

The `MinTokensLogitsProcessor` keeps `min_tokens` and stop-token semantics
unchanged while caching the actual speculative mask's immutable pinned-host/NPU
index buffers. Cache hits emit layout/mask timing rows but no H2D upload; mask,
request/batch, draft-length, stream or device changes allocate a fresh source
and issue one nonblocking upload without overwriting an in-flight DMA buffer or
adding a completion wait. A growing output history alone does not invalidate an
unchanged mask. This optimization is active without `--diagnose`; its host
allocation/fill/upload/mask rows are diagnostic-only. Ordinary NumPy-backed
host memory remains forbidden for the async copy, and a local host wait is not
itself evidence of a whole-request TPOT gain.

Full-graph fixed configuration, KV layouts, shared metadata eligibility and
input signatures are validated during startup warmup/capture, not on each
decode. The runner/builder owns stable allocations: contents may update in
place, while replacing storage requires clear and recapture. `prepare_run()`
retains only constant-time graph-key/lifecycle checks on live replay;
`validate_inputs()` remains an explicit diagnostic. Dynamic history boundaries
and source leases still update before replay, and the post-replay completion
fence/lifetime protection is unchanged.

Full decode also reuses unchanged history boundaries. A builder-owned CPU
shadow computes the same `min(floor((seq_len - 1) / window) * window, frontier)`
values using CPU metadata, without per-step NumPy allocation/unique/validation
passes. Only changed values or row layouts issue an H2D copy. Frontier growth
or shrink, padding, Q1/Q2 layout and graph-capacity changes are still applied;
eager/staged/dummy writes invalidate the shadow. In the CPU regression fixture,
511 consecutive steps cross three windows and issue three uploads, not 511.
Top-k and selected KV are never cached by this optimization. The staged
benchmark keeps its original boundary path.

LMCache resolves warm source readiness once per step; only cold bootstrap
requires another resolution after publication. This is not a cross-step source
cache: equal-length replacements, index residency and pending loads remain live.
MTP draft suffixes are initialized at their first payload instead of before the
target replay. This permits CPU setup to overlap already-submitted target work;
it does not remove draft work or stream dependencies. Cancelled unused suffixes
perform no connector setup. Compare total TPOT and MTP timing, not just reduced
`source.prepare`/`metadata.L0` times.

The asynchronous lifetime owner replaces that fence with a completion event:
replay returns without a CPU stream synchronization, while each source batch
retains independent allocator references until the event completes. Replaced
or finished requests enter a nonblocking retirement queue; reset and shutdown
drain leases before allocator teardown. Pointer uploads and replay stay on one
runner stream, failed submissions retain leases, and a bounded backlog fails
closed rather than freeing in-flight memory. Native NPU stream/allocator
validation remains required; CPU ownership tests do not prove it.

For live single-node TP-only DP1/PP1 `mp` workers, source or metadata
preparation failures use error-only fail-stop: the worker logs the original
traceback and raises `SystemExit`, allowing the existing process-sentinel
supervisor to stop owned peers, including blocked workers. A five-second daemon
cleanup timer is armed only on failure. Healthy forwards perform no failure
collective, timer or health polling. Startup capture, multi-node/multi-DP jobs
and other executors retain synchronous TP/DP error agreement. This is fail-stop
requiring engine restart, not forward recovery; the CPU tests do not verify
native HCCL cancellation.

The benchmark defaults to a 30000-token input context (not 30000 generated
tokens); use `--prompt-tokens 4351` to repeat the earlier short-context case.
Prefill remains chunked at 512 tokens. Full-graph source binding compares
ordered request IDs and immutable `PreparedSparseSource` snapshot identities
before enumerating transfers. Unchanged sources skip every layer's
`bind_batch`; newly published windows, replacements/restores, changed lanes
and empty batches rebind. `full.json` records
`source_binding_updates_per_rank` beside root replay counts, sampled outside
timed requests. This does not suppress graph-internal KV loads or metadata
address checks; it only removes unchanged source-table rebinding work.

### Online serving reproduction

For the serving regression, use the native TP4 x DP2 + EP reproducer on one
eight-NPU host rather than the TP8/DP1 offline benchmark:

```bash
set -o pipefail
python tools/sfa_serving_repro.py --diagnose --prompt-tokens 131614 \
  --output-tokens 1000 2>&1 | tee log.log
grep -aE '\[SFA_SERVING(_TIMING)?\]' log.log
```

The client loads its tokenizer from the local `--model` directory, with
`--trust-remote-code` matching the server. `sfa-serving-repro` is only the API
model alias, not a model/tokenizer repository to download; the local model
directory must include tokenizer files.

Both modes use the eight-layer dummy model, recompute scheduler, serving graph
buckets, and one active request routed to DP0 while DP1 is idle. Cold-compact
loading uses a strict shared CPU cache with a rank0 store and passive TP readers;
`save_only_first_rank=true` is applied together with the shared-cache flags.
The 8-GiB CPU-cache budget is per DP group, so the two local groups need about
16 GiB of free `/dev/shm` in total. No remote cache is used. These cache flags
are identical between modes; only `VLLM_ASCEND_SFA_FULL_GRAPH` differs.

The offline benchmark deliberately keeps independent per-rank CPU caches and
`save_only_first_rank=false`; do not copy that policy into serving cold-compact
reproduction. The native LMCache configuration validator rejects that
combination before inference. `--diagnose` adds host timings, not a profile;
omit it for clean latency measurements. Server logs are retained under
`profile/sfa-serving-*/{staged,full}/server.log`.

Serving diagnostics preserve the server's existing scheduling policy; they do
not add `--no-async-scheduling`. With async outputs, the recorder measures
worker launch/prepare/forward/sampling scopes without reading token tensors or
forcing deferred output completion. `sampled_tokens_histogram` is empty and
`sampled_tokens_histogram_available=false`; output-token counts and TPOT remain
the client benchmark's responsibility. `async_workers=N/M` reports the actual
policy returned by the RPC. Host spans exclude deferred readback completion and
must not be summed as non-overlapping full-token latency. If a diagnostic RPC
fails, the reproducer includes its method, HTTP response body and a bounded tail
of the corresponding server log.
