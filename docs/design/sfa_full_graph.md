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
