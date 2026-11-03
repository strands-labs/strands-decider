# Choosing an AWS host

I measured these hosts on 20 October 2026 with `strands-decider-2B-qwen3.5-v1-2610` in eu-west-1,
sending one request at a time. The SageMaker rows used the stack in [sagemaker/](sagemaker/README.md)
unchanged; the others ran the package's server (`create_app`) behind a small launcher that copies
the model from S3.

| Host | Hardware | Time per request | Start-up |
| --- | --- | --- | --- |
| Amazon SageMaker, `ml.g6.xlarge` | NVIDIA L4 | 136 ms | Always on |
| AgentCore Runtime Instances, `g5.xlarge` | NVIDIA A10G | 159 ms | 194 s for each new session |
| Lambda MicroVMs | Graviton, 8 GB baseline, bf16 | 1.7 s | 6-7 s to start, then 2.6 s for the first answer |
| Amazon SageMaker, `ml.m5.2xlarge` | 8 vCPU, fp32 | 2.4 s | Always on |
| AgentCore Runtime microVM | 2 vCPU, 8 GB, bf16 | 4.6 s | 36 s for the first new session, 7 s for the next |

Time per request is the server's median over 100 support messages, each asking two questions (a
five-way choice and a yes/no). AgentCore added about 100 ms per call on top. Every host routed
80-81% of the messages correctly and matched the L4's answer on at least 99 of the 100. In fp32,
`ml.c7i.2xlarge` was no faster than `ml.m5.2xlarge`.

## Hosts with 8 GB of memory

On CPU, `serve` converts the model to fp32, which needs about 9 GB: more than the AgentCore microVM's
8 GB and the 8 GB MicroVM baseline used here. Stock `serve` can't keep it in bf16 until
[#48](https://github.com/strands-labs/strands-decider/pull/48) adds `--cpu-torso-dtype checkpoint`,
so for these rows the launcher skipped the conversion. In bf16, one of the 100 answers changed, on a
near tie.

## AgentCore Runtime

- Both compute types take IAM (SigV4) or a JWT from your identity provider. The microVM takes
  `arm64` images only.
- On Instances, the 2 GB image limit is smaller than CUDA PyTorch, so the launcher copies PyTorch
  and the model from S3 at start-up: about 90 s of the 194 s.
- In a `python:3.11-slim` image, PyTorch did not see the GPU until the launcher loaded
  `/usr/lib64/libcuda.so.1` before importing it.
- The A10G row ran transformers' PyTorch path: flash-linear-attention's Triton kernels did not build
  in the runtime.
- On Instances, the runtime has no public IP, so it needs a NAT gateway or VPC endpoints for ECR,
  S3 and CloudWatch Logs.

## Lambda MicroVMs

- Load the model and answer two requests before `/ready` returns, so the snapshot holds a model that
  has already run.
- Have `/validate` start one request in the background on all threads and return 503 until it
  finishes. Builds that ran three requests on two threads there failed with only "MicroVM terminated
  before it started"; the cause wasn't isolated.
- Callers send a token from `create-microvm-auth-token` in the `X-aws-proxy-auth` header.
