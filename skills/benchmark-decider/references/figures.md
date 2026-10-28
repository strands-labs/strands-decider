# Benchmark artifacts and figures

Each model run directory contains:

- `manifest.json`: run identity, timestamps, environment configuration and fingerprint, model and
  endpoint health, workload definition, measurement settings, client metadata, and code revision;
- `summary.json`: canonical aggregate headline, lifecycle, closed-loop points, repetitions, errors,
  per-input metrics, and health metadata;
- `summary.csv`: flattened repetition and input-length aggregates;
- `throughput-vs-concurrency.png`: successful requests/s at every measured concurrency;
- `latency-vs-input-length.png`: concurrency-1 p50 latency by server-reported input tokens.

A full multi-model environment run also writes these beside its model directories:

- `models-throughput-vs-concurrency.png`;
- `models-latency-vs-input-length.png`.

Regenerate CSV and PNG artifacts from an existing run with:

```bash
PYTHONPATH=src uv run --project benchmark python -m benchmark.runner summarize RUN_DIR
```

## Interpreting the standard graphs

The throughput graph answers whether added concurrency raises throughput and where the best
observed point occurs. Its request type is fixed: 1,024-token choice requests with four questions
and four options.

The latency graph answers how serial request latency changes across the calibrated, equal-weight
256-, 1,024-, 2,048-, and 4,096-token request shapes. It uses p50 at concurrency 1, one line per
model in the combined graph.

Keep model lines within one compute environment. If presenting multiple environments, use separate
panels or figures and label their baselines independently.

## Custom analysis

Prefer `summary.json` for headline and combined-point analysis. Use `summary.csv` when a dataframe
or plotting library needs tabular per-input and per-repetition rows. Useful columns include:

- `load` for concurrency;
- `requested_input_tokens` and `input_tokens`;
- `requests_per_second`;
- `p50_ms`;
- `failure_rate`;
- `repeated_state_fraction`.

Do not reconstruct per-request distributions: raw request measurements are intentionally not
persisted. Do not promote p95 or p99 columns to headline graphs without first increasing the sample
count enough to support a tail-latency claim.
