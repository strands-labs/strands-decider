# Analyze benchmark results

Compare only the same model and environment. The runner enforces compatibility for accepted
baseline comparisons; use the environment fingerprint in `manifest.json` when inspecting results
manually.

## Headline metrics

- **Peak successful requests/s:** the highest median throughput among measured concurrency points.
  It is the best observed throughput, not necessarily the server's true ceiling.
- **Reference p50 latency:** median client round-trip latency at concurrency 1. Use its per-input
  values to show how prefill cost scales from 256 through 4,096 tokens.
- **Reliability:** report the maximum failure rate across the sweep and whether an OOM or server
  crash occurred. A candidate cannot be promoted at a failure rate of 1% or greater, or after an
  OOM/crash.
- **Cost:** when hourly pricing is configured, the suite derives cost at the best observed
  throughput. The current artifact field is cost per one million decisions; each request currently
  contains four decisions.

For a metric where higher is better:

```text
improvement % = 100 × (candidate - baseline) / baseline
```

For latency or cost, where lower is better:

```text
improvement % = 100 × (baseline - candidate) / baseline
```

Give the absolute values with the percentage. A throughput improvement that also produces failures
or materially worse serial latency is a trade-off, not an unconditional win.

## Reading the concurrency sweep

In a closed loop, each worker submits another request after its previous request completes.
Throughput should initially rise as concurrency uses otherwise idle compute. Near saturation:

- throughput plateaus while latency grows;
- throughput can decline from queueing, scheduling overhead, memory pressure, or measurement noise;
- one higher point after a lower point does not establish a trend—use the three repetitions and
  neighboring points.

The peak measured point is the benchmark's throughput headline. Call it a true ceiling only when
the sweep reaches a stable plateau or decline and remains reliable. The current standard stops at
concurrency 32, so a still-rising final point means the ceiling was not found.

Concurrency-1 p50 is the cleanest measure of serial service latency. Compare its four input-length
points rather than attributing a mixed-workload p50 to one request shape.

## Cost and environment boundaries

For a provisioned host:

```text
USD per 1M decisions =
    hourly USD / (3,600 × successful decisions/s) × 1,000,000
```

Record the pricing source, region, and date outside the generated artifacts if the environment
price is manually supplied. Do not compare two instance types as though one were the other's
baseline. Each environment answers how a code change affected that environment; a separate
cross-environment study can present absolute numbers without using baseline-improvement language.

## Limits

The current suite does not measure correctness, production arrival patterns, cold-start
distributions, long-run stability, or statistically defensible tail latency. Do not infer those
properties from these artifacts.
