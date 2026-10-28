# Benchmark follow-ups

## Near term

- Change cost reporting from cost per 1 million decisions to cost per 1 million successful
  requests.
- Extend the standard concurrency sweep until throughput plateaus or declines instead of stopping
  at a fixed maximum concurrency.
- Stop a concurrency sweep after sustained timeouts, failures, or server crashes make higher
  points uninformative.
- Record dated, regional pricing for each provisioned environment.
- Run remote latency benchmarks from a client in the same region as the endpoint.
- Add Lambda invocation and environment metadata support.

## Production-style traffic

- Add optional open-loop Poisson traffic after the closed-loop sweep establishes maximum
  throughput.
- Test offered rates derived from each environment's measured capacity rather than comparing
  environments directly.
- Report SLO goodput: successful requests per second for which at least 70% of requests satisfy
  the configured latency target.
- Add user-supplied production-like request datasets while retaining the synthetic workload as the
  reproducible baseline.
- Add optional short-burst and long-soak tests.

## Reporting and diagnostics

- Add statistically meaningful p95 or p99 reporting only for presets with enough samples.
- Add optional secondary diagnostics for cache reuse and batching without making them headline
  metrics.
- Add a combined environment summary alongside the existing per-model summaries and graphs.

## Automation

- Add CI coverage for configuration validation, workload balance, aggregation, and graph
  generation when the benchmark interface stabilizes.
