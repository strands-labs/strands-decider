# Benchmarking the decision function's builds

These scripts measure the decision image's builds on Lambda, on ECS (EC2 and Fargate) and on
plain EC2 hosts. They measure speed, accuracy against a checkpoint's golden vectors, and memory.
The results and the reasoning behind the defaults are in
[docs/deployment/graviton-cpu-variants.md](../../../docs/deployment/graviton-cpu-variants.md).

## Building a variant

Every variant comes from the same `decision/Dockerfile`, chosen by build args. CDK sets them from
`-c` context; a hand build passes them with `--build-arg`.

| `-c` context | Build arg | Values | Default |
| --- | --- | --- | --- |
| `cpu` | `NATIVE` | `baseline` (`--lambda`), `variants` (`--lambda-variants`) | `baseline` |
| `kleidiai` | `KLEIDIAI` | `false` / `true` (`0` / `1`) | `false` |
| `graviton_sve` | `GRAVITON_SVE` | `false` / `true`, with `cpu=variants` only, for measuring | `false` |
| `decoder_quant` | `DECODER_QUANT` | `Q8_0`, `Q6_K`, `Q4_K_M`, `Q4_0` | `Q8_0` |

To deploy one alongside the others, give it its own `stack_name`:

```bash
cd deploy/lambda
npx cdk deploy -c model=StrandsAgents/strands-decider-E4B-gemma4-v2-2611 \
  -c stack_name=sd-e4b-variants -c cpu=variants
```

Or build the image for ECS or EC2 by hand, from the repository root, on an arm64 host:

```bash
docker buildx build --platform linux/arm64 -f deploy/lambda/decision/Dockerfile \
  --build-arg MODEL=StrandsAgents/strands-decider-E4B-gemma4-v2-2611 \
  --build-arg NATIVE=--lambda-variants --build-arg KLEIDIAI=1 \
  -t <ecr-repo>:variants-kleidiai --push .
```

The image build runs its own accuracy check, `decision/scripts/check_variants.sh`, which covers
only the CPU modules the build host loads. To check every Graviton module, run the release check
below on each generation.

## Measuring speed

**`graviton_bench.py run` drives deployed Lambda functions.** It sends a fixed mix of requests
from the checkpoint's `golden_vectors.jsonl`:

- two of about 40 tokens;
- one of about 1.1k tokens;
- one of about 4.1k tokens;
- one with one image.

Between rounds it changes an environment variable on each function's `$LATEST`. That retires the
function's environments, so Lambda hands out new hosts. Each row records the host's `cpu_part`,
its CPU features and the CPU module it loaded.

```bash
python graviton_bench.py run --golden golden_vectors.jsonl --out lambda.jsonl \
  --function baseline=<function ARN> --function variants=<function ARN> --rounds 4 --concurrency 4
```

**`ecs_bench.py` runs the image as ECS tasks.** On the EC2 launch type it pins each task to an
instance type, one per Graviton generation. On Fargate, AWS picks the host. Each task runs
`bench_container.py`, which imports the image's own handler, with 6 threads.

```bash
python ecs_bench.py --cluster <cluster> --subnet <subnet> --security-group <sg> --bucket <bucket> \
  --labels baseline variants --instance-types c6g.2xlarge c7g.2xlarge c8g.2xlarge c9g.2xlarge \
  --fargate 2 --out ecs.jsonl
```

**`bench_container.py` also runs on its own** with `docker run`, on any host. `--module NAME` loads
only `libggml-cpu-NAME.so` of a variants image, so one image can time each module. Each row
records `RssAnon`, `RssFile` and `VmHWM`. `RssAnon` is the memory the kernel can't reclaim; run
with no container memory limit to measure it.

**`graviton_bench.py report` tables every driver's results together.** The table is by platform,
generation and label: median latency, speedup against the first label, memory, and how far each
label's answers are from the first label's.

```bash
python graviton_bench.py report lambda.jsonl ecs.jsonl
```

## Measuring accuracy: the release check

**`conformance_container.py` scores one CPU module against a checkpoint's golden vectors.** It
runs the text rows inside an image and reports `tests/core_conformance.py`'s top-1 agreement and
mean |dp|. `--module` forces a module, and `--gguf` swaps in another decoder file, for example a
re-quantized one.

```bash
docker run --rm -v $PWD:/w --entrypoint python3 <image> /w/conformance_container.py \
  --golden /w/golden_vectors.jsonl --module graviton4 --max-state 5000 --threads 24 \
  --label g4 --s3 s3://<bucket>/conformance/
```

**`conformance_report.py` is the release check.** It fails unless every module conforms and is no
worse than the static build on its host. Run the static image with no `--module` on each host, so
there's a result to compare against.

```bash
python conformance_report.py s3://<bucket>/conformance/
```

Accuracy doesn't depend on the thread count, so a large instance per generation keeps this
fast. On a `.8xlarge`, E4B's 1,032 questions took 20–45 minutes per module.

## What the scripts expect in AWS

The drivers take the resources as arguments and create none of them.

- **An S3 bucket.** It holds the results. The tasks also read `inputs/bench_container.py` and
  `inputs/requests.json` from it: run `python graviton_bench.py requests golden_vectors.jsonl`
  to make the request file.
- **An ECR repository** holding the images.
- **An ECS cluster.** Its container instances are one EC2 instance of each type you pass to
  `--instance-types`, joined with `ECS_CLUSTER=<cluster>`.
- **Task definitions** named `<cluster>-ec2-<label>` (bridge networking, 6 vCPU, 10 GB) and
  `<cluster>-fargate-<label>` (awsvpc, 8 vCPU, 16 GB, 60 GB ephemeral storage). Each has one
  container named `bench`, whose command downloads the two inputs and runs:

  ```bash
  bench_container.py --requests /tmp/requests.json --label $BENCH_LABEL \
    --s3 s3://<bucket>/results/<ec2|fargate>/
  ```

- **IAM roles.** The task role needs `s3:GetObject` and `s3:PutObject` on the bucket only. The
  instance role needs SSM, ECS container-instance permissions, and pull access to the repository.
- **Networking.** Nothing listens on a port, so the security group needs no inbound rules. Fargate
  tasks need a public IP, or a NAT, to reach ECR and S3.
