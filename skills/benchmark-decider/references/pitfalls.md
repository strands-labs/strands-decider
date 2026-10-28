# Benchmark pitfalls

Review these before running or interpreting the suite.

## Environment identity

An environment represents one compute setup. Do not put endpoints backed by different SageMaker
instance types in one environment even if they serve related models. The runner verifies the live
instance type but cannot detect every relevant host change, so keep the YAML hardware description
and pricing current.

Run remote latency measurements from a client near the endpoint when network latency matters.
Client placement is recorded, but laptop-to-region latency is still part of the measured
round-trip. For SageMaker, follow [aws.md](aws.md) and run the client in the endpoint region before
making an in-region latency claim.

## Model coverage

A full run means every model listed in the environment, not every model published by the project.
List only models that fit and are supported on that target host. Missing a model-to-endpoint entry
in a SageMaker environment is a configuration error.

## Warm-up and request waves

Each repetition warms one request per concurrent client using the same worker pool. Warm-up work is
not timed. This avoids treating first-use setup as steady-state latency, but the suite does not
measure cold starts.

A throughput point needs multiple waves of requests. The runner requires at least twice as many
measured requests as the maximum concurrency. Do not raise concurrency without also raising the
request count.

## Input calibration and caches

The workload calibrates the four target input lengths using `usage.input_tokens` reported by the
server. Calibration failures, truncation, or refusals must be fixed rather than silently dropping
the 4,096-token case.

Half the requests deliberately reuse an encoded state and half use unique states. Preserve this
mix when evaluating caching changes. Cache hit rate and state-encoding time may explain a result,
but they are not broad headline metrics.

## Throughput claims

The highest point in the fixed sweep is only the best observed throughput. If concurrency 32 is
still rising, say that the benchmark did not locate the ceiling. Small non-monotonic changes can
come from noise; use the repetitions and adjacent points before diagnosing saturation.

Failure-free throughput matters. Timeouts, refusals, incomplete answer sets, OOMs, and server
crashes cannot be excluded from the interpretation merely because successful requests were fast.

## Baselines and artifacts

The first standard run becomes the accepted baseline; later compatible runs become candidates.
Promote only after reviewing the comparison. Promotion replaces aggregate benchmark artifacts, so
the accepted baseline represents the current accepted implementation rather than a historical
archive.

Do not edit `manifest.json` or copy a candidate manually to bypass compatibility checks. Do not
expect raw request samples: the suite intentionally persists aggregate data only.
