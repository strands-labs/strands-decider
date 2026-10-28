# Run SageMaker benchmarks in-region

Use this workflow when the benchmark targets SageMaker or the user asks to remove laptop-to-region
network latency.

## Explain client placement

The `sagemaker.region` value selects the AWS API region but does not relocate the benchmark
process. A benchmark launched on a laptop still records the laptop-to-AWS round trip. Before making
a same-region latency claim, tell the user to run the benchmark client on compute in the endpoint's
region.

Prefer an EC2 instance as the load generator because the runner writes local artifacts and may run
for an extended period. The instance runs only the benchmark client; the model remains on
SageMaker. An existing suitable instance is sufficient. Creating or changing an instance, role,
security group, subnet, or VPC endpoint requires the user's explicit approval, and a paid full run
requires confirming the expected scope and cost.

## Minimum setup

1. Place the EC2 client in the same AWS region as every endpoint in the environment.
2. Attach an EC2 instance role that permits:
   - `sagemaker:InvokeEndpoint` for the configured endpoints;
   - `sagemaker:DescribeEndpoint`;
   - `sagemaker:DescribeEndpointConfig`.
3. Omit `sagemaker.profile` from `environment.yaml` so boto3 uses the instance role. A named
   `profile` is for a workstation that has that profile configured.
4. Ensure the instance can reach the SageMaker API and Runtime services. A private subnet needs
   the appropriate interface VPC endpoints or another approved egress path.
5. Ensure the client can load each Hugging Face tokenizer used by the configured models. Provide
   outbound access or pre-populate the tokenizer cache.

Use least-privilege resource scopes for the endpoint and endpoint-config ARNs. Do not put static
AWS access keys in the repository or environment YAML.

## Run

Clone or copy the same repository revision to the EC2 client. Start with one model:

```bash
uv run --extra benchmark python -m benchmark.runner run ENVIRONMENT MODEL --smoke
```

Check that the recorded SageMaker region, endpoint, instance type, and instance count are correct.
Then run every model configured for the environment:

```bash
uv run --extra benchmark python -m benchmark.runner run ENVIRONMENT
```

Copy the environment's `runs/` directory back if the canonical results are maintained elsewhere.
Preserve the generated client metadata: it documents where the benchmark process ran.

## Interpret latency correctly

Client latency still includes serialization, the in-region network path, SageMaker request
handling, queueing, and inference. Describe it as in-region end-to-end latency, not pure model
execution time. Use server-reported latency only as a secondary diagnostic when available.
