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
