---
name: benchmark-decider
description: >
  Configure, run, compare, and interpret the repository's Strands Decider inference-performance
  benchmarks. Use for local, HTTP, or SageMaker benchmark environments; multi-model runs;
  throughput, latency, reliability, and cost analysis; benchmark graphs; or accepting a faster
  run as the new baseline. This skill covers server performance, not model correctness, quality,
  or ranking different compute environments against one another.
---

# Benchmark Strands Decider

Use the benchmark suite under [`benchmark/`](../../benchmark/README.md). Its purpose is to measure
how an inference-server change affects each model within a fixed compute environment. Compare a
candidate only with that environment and model's accepted baseline.

Read the benchmark README before changing or running the suite. Read:

- [references/pitfalls.md](references/pitfalls.md) before running a benchmark;
- [references/analysis.md](references/analysis.md) when interpreting or comparing results;
- [references/figures.md](references/figures.md) when working with artifacts or graphs;
- [references/aws.md](references/aws.md) before configuring or running a SageMaker benchmark.

Do not restore or use `evaluation/decider_bench.py`; the current entry point is
`python -m benchmark.runner`.

## Benchmark contract

Keep the workload and measurement settings fixed when evaluating an inference change:

- synthetic input targets: 256, 1,024, 2,048, and 4,096 server-reported tokens per request;
- four choice questions per request, with four options each;
- throughput uses a fixed 1,024-token request shape;
- serial latency uses an equal mix of 256-, 1,024-, 2,048-, and 4,096-token requests;
- 50% repeated and 50% unique request states;
- closed-loop concurrency only;
- one warm-up request per concurrent client before each throughput repetition;
- one warm-up request for each serial-latency input shape.

The standard preset measures concurrency 1, 4, 16, and 32 twice. It uses a minimum of 16 requests
per point and increases that to two complete request waves at higher concurrency: 16, 16, 32, and
64 requests respectively. The smoke preset measures one eight-request workload cycle at
concurrency 1. Poisson traffic, bursts, soak tests, production datasets, and correctness
evaluation are deferred.

The headline metrics are:

1. peak successful requests per second;
2. p50 request latency at concurrency 1;
3. failure rate and whether any OOM or server crash occurred;
4. cost per one million decisions when pricing is configured.

Treat p95 and p99 fields as diagnostics only: the current sample count is not sufficient for strong
tail-latency claims.

## Environments

Each compute environment has
`benchmark/environments/<environment>/environment.yaml`; its results belong under the adjacent
`runs/` directory. A full run benchmarks every model listed in `models` sequentially. List only
models the target host can actually load.

Use a managed environment for a local server:

```yaml
name: local
models:
  - StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
endpoint: http://127.0.0.1:{port}
timeout_s: 120
lifecycle:
  type: managed
  command:
    - strands-decider
    - serve
    - "{model}"
    - --host
    - 127.0.0.1
    - --port
    - "{port}"
```

Use `lifecycle: {type: external}` and an `endpoint` for an already-running ordinary HTTP server.
The URL must be the server root; the runner calls `/health` and `/v1/systemone`.

For SageMaker, map every model to the endpoint that serves it:

```yaml
name: sagemaker-us-west-2-ml-g6-xlarge
models:
  - MODEL_A
  - MODEL_B
lifecycle:
  type: external
sagemaker:
  region: us-west-2
  instance_type: ml.g6.xlarge
  endpoints:
    MODEL_A: ENDPOINT_A
    MODEL_B: ENDPOINT_B
```

The runner uses boto3's credential chain and SigV4, describes each endpoint, and refuses to run if
its live instance type differs from `instance_type`. A single-model environment may use
`endpoint_name` instead of `endpoints`. Keep different instance types in separate environment
directories and baselines even when the models use different endpoints. `profile` is optional for
a named local AWS profile; omit it when the benchmark client uses an EC2 instance role.

Invoking remote endpoints can incur cost. Do not create, update, resize, or delete cloud resources
without the user's explicit approval. Confirm scope and expected cost before a full paid run.

The configured AWS region selects which SageMaker API endpoint receives the requests; it does not
move the benchmark client. For latency claims, tell the user to run the benchmark from an EC2
instance in the same region as the SageMaker endpoints. Follow
[references/aws.md](references/aws.md) for the setup and checks. When the user authorizes creating
a temporary load generator, use `scripts/run_on_ec2.py`; first run it with `--dry-run`, show the
resource and endpoint scope, and confirm the paid run before passing `--yes`.
After the remote SSM command starts, give the user the copy-paste status and live-progress
commands printed by the script; do not wait for them to ask how to monitor it.

## Run workflow

For an external HTTP or SageMaker endpoint, use the lightweight client project so the load
generator does not install PyTorch or CUDA:

```bash
PYTHONPATH=src uv run --project benchmark python -m benchmark.runner run \
  ENVIRONMENT MODEL --smoke
```

For a managed local server, use the root project and add the applicable device extra, such as
`--extra mlx` on Apple Silicon.

Start with one model:

```bash
uv run --extra mlx --extra benchmark python -m benchmark.runner run ENVIRONMENT MODEL --smoke
```

Inspect the smoke summary for successful calibration, the expected endpoint/model/instance
metadata, zero failures, and plausible latency. Then run every model configured for that host:

```bash
uv run --extra mlx --extra benchmark python -m benchmark.runner run ENVIRONMENT
```

Omitting `MODEL` means all configured models. Smoke runs require one model. `--output PATH`
overrides the normal result location; benchmark artifacts at that location are replaced by a
successful rerun.

The first standard run becomes the accepted baseline. A later compatible run is written to the
model's `candidate/` directory. Review it with:

```bash
uv run --extra benchmark python -m benchmark.runner compare \
  ENVIRONMENT MODEL benchmark/environments/ENVIRONMENT/runs/MODEL_SLUG/candidate
```

Update the baseline only after the performance change is understood and the reliability gate
passes:

```bash
uv run --extra benchmark python -m benchmark.runner compare \
  ENVIRONMENT MODEL benchmark/environments/ENVIRONMENT/runs/MODEL_SLUG/candidate \
  --update-baseline
```

Compatibility requires the same environment fingerprint, model, workload, measurement preset,
and baseline lineage. Do not work around a compatibility failure by manually copying artifacts.

## Reporting

Report improvements relative to each model's baseline in the same environment. Lead with peak
requests per second and concurrency-1 p50 latency; include failure rate and any OOM/crash. Explain
where throughput plateaus or declines, but do not claim an absolute server ceiling if the fixed
concurrency sweep never reaches one.

The suite intentionally stores aggregate results rather than raw per-request measurements. Preserve
`manifest.json`, `summary.json`, `summary.csv`, and the generated PNGs together so a result remains
traceable to its environment, model, workload, and code revision.
