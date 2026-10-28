# Inference performance benchmarks

This suite measures how an inference-server change affects the configured models in one compute
environment. It does not rank different environments and does not evaluate model correctness.

During a run the server, model, and environment remain fixed. The benchmark changes only the
synthetic request shape and calling pattern:

- a serial latency sweep using input-length targets of 256, 1,024, 2,048, and 4,096 tokens per
  request, calibrated against the server-reported token count;
- a throughput sweep using only 1,024-token requests;
- four choice questions per request, with four options each;
- an equal-weight mixture of the four input lengths in the latency sweep;
- 50% repeated states and 50% unique states;
- fixed, balanced request batches at each closed-loop concurrency level.

Poisson traffic, short bursts, and long soak tests are intentionally deferred.

## Headline results

Every model/environment summary records:

1. Peak successful requests per second.
2. p50 request latency at concurrency 1.
3. Cost per 1 million decisions, when the environment supplies pricing.

Failure rate must remain below 1%, with zero OOMs or server crashes. A later version may add
open-loop traffic and SLO-constrained goodput as an optional production-style test.

## Environments

Each compute environment is a child of `benchmark/environments/`. Its configuration and generated
runs live together:

```text
benchmark/
  runner/                 # Python benchmark implementation
  environments/
    local/
      environment.yaml
      runs/
        models-throughput-vs-concurrency.png
        models-latency-vs-input-length.png
        <model-slug>/      # accepted baseline
          manifest.json
          summary.json
          summary.csv
          throughput-vs-concurrency.png
          latency-vs-input-length.png
          candidate/       # latest standard candidate
            ...
          smoke/           # latest smoke run
            ...
    external-endpoint/
      environment.yaml
      runs/
        ...
```

Example managed local environment:

```yaml
name: local
models:
  - StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
  - StrandsAgents/strands-decider-E2B-gemma4-v1-2610
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
pricing:
  hourly_usd: null
```

An environment describes one HTTP endpoint. A managed environment starts the server command; an
external environment points at an already-running server.

Each measured invocation is one independent HTTP request. The runner does not pool connections or
retry failures.

```yaml
name: external
models:
  - StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
endpoint: http://127.0.0.1:8000
lifecycle:
  type: external
```

SageMaker environments map each model to its endpoint. The runner uses boto3's standard AWS
credential chain and SigV4 authentication, and records each live endpoint configuration, instance
type, and instance count:

```yaml
name: sagemaker-us-west-2-ml-g6-xlarge
models:
  - StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
  - StrandsAgents/strands-decider-E2B-gemma4-v1-2610
lifecycle:
  type: external
sagemaker:
  region: us-west-2
  profile: default
  instance_type: ml.g6.xlarge
  endpoints:
    StrandsAgents/strands-decider-2B-qwen3.5-v1-2610: QWEN_ENDPOINT
    StrandsAgents/strands-decider-E2B-gemma4-v1-2610: GEMMA_ENDPOINT
```

The single-model `sagemaker.endpoint_name` form remains supported. Before invoking a model, the
runner verifies that its endpoint uses the configured `instance_type`. Different instance types
need separate environment directories and baselines.

Keep ordinary local HTTP targets on loopback, and use external targets only when their existing
network and authentication controls already permit benchmark access.

## Running

SageMaker and external HTTP clients can use the lightweight benchmark project. It contains only
the client, tokenizer, AWS, and reporting dependencies, so it does not install PyTorch or CUDA:

```bash
PYTHONPATH=src uv run --project benchmark python -m benchmark.runner run ENVIRONMENT
```

For a managed local server, run through the root project with the applicable inference extra so
the server executable and backend are available:

```bash
uv run --extra mlx --extra benchmark python -m benchmark.runner run local
```

The examples below show `python -m benchmark.runner` for brevity; prefix them with the applicable
lightweight-client or managed-server command above. Run the standard suite for every model
configured in the environment. Models run sequentially, and each model’s first run becomes its
accepted baseline.

The supplied `local` environment includes all five v1 models. The 48 GB Mac environment omits the
26B model because its documented bf16 memory requirement exceeds that machine's memory.

The serial latency sweep warms each input length, then measures an equal mix of all four lengths.
For throughput, the runner sends one 1,024-token warm-up request per concurrent client using the
same worker pool. The standard preset measures concurrency 1, 4, 16, and 32 twice. It uses a
minimum of 16 requests per point, increasing that to 32 and 64 requests at concurrency 16 and 32
so every point observes at least two request waves. The smoke preset uses eight measured requests
at concurrency 1.

Pass a model to run only that model:

```bash
python -m benchmark.runner run local MODEL
```

Later standard runs write each model’s latest candidate under its model directory. Compare and
optionally accept one candidate:

```bash
python -m benchmark.runner compare \
  local MODEL benchmark/environments/local/runs/MODEL_SLUG/candidate

python -m benchmark.runner compare \
  local MODEL benchmark/environments/local/runs/MODEL_SLUG/candidate \
  --update-baseline
```

`--update-baseline` requires compatible environment, model, workload, and measurement settings.
It stages and validates all aggregate artifacts before replacing the accepted baseline.

For a quick development smoke test:

```bash
python -m benchmark.runner run local MODEL --smoke
```

Smoke results are written to
`benchmark/environments/<environment>/runs/<model-slug>/smoke/`. Use `--output` to override any of
these defaults. For a full multi-model run, `--output` is the shared root and each model is written
beneath it. Smoke runs require a model.

When `--output` already exists, the runner replaces its benchmark artifacts after the new run
finishes successfully. Unrelated files in the directory are left untouched.

## Artifacts and graphs

Normal request payloads and per-request measurements are never persisted. Serial latency
repetitions contain complete eight-request workload cycles across the four input lengths.
Throughput points use only 1,024-token requests. Both workloads contain 50% repeated states and
50% unique states.

- `manifest.json`: run, environment, model, workload, and code identity.
- `summary.json`: canonical aggregate metrics and repetitions.
- `summary.csv`: flattened per-shape aggregate measurements.
- `throughput-vs-concurrency.png`: 1,024-token requests/second across concurrency levels.
- `latency-vs-input-length.png`: concurrency-1 p50 latency across the four input lengths.

A full environment run also writes:

- `models-throughput-vs-concurrency.png`: throughput lines for every configured model.
- `models-latency-vs-input-length.png`: latency lines for every configured model.

Use `python -m benchmark.runner summarize RUN_DIR` to regenerate the CSV and graphs from the
aggregate JSON files.
