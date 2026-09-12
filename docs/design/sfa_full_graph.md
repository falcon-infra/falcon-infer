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

`--diagnose` is mutually exclusive with `--profile`. Normal warmup and TPOT
requests remain uninstrumented; only one later request receives temporary
host/current-stream timing wrappers, which are restored afterward. Scheduler
history excludes all chunked-prefill steps, including a final one-token Q1
chunk. Timing stop and synchronization happen after the request, with bounded
event storage and `event_drops` reporting partial stream coverage. Results are
printed as compact `[SFA_TIMING]` lines and saved separately from TPOT JSON;
they are diagnostics, not performance acceptance or pure kernel timings.

Full-graph validation is now deduplicated only within one forward. The runner
creates a single-use `prepare_run()` handoff before the existing error
agreement/fail-stop boundary; the handoff owns the actual kwargs, context,
graph key and generation, and cannot be reused, replaced or used after a
capture reset. Shared tensor signatures and common attention metadata are
memoized for that forward only; layer-local KV layout checks remain active.
`signature.validate` therefore runs once per full target forward, while
`metadata.shared_check` reports the shared eligibility checks. No mutable
metadata is cached across forwards, and the post-replay completion fence is
unchanged.

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
