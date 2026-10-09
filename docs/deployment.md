# Serving on AWS

This document lists the AWS hosts that Strands decider has been measured on, what each one
costs in latency, and the setup problems that each one has. The model and its answers are the
same on every host; only speed, start-up behaviour and cost change. On a 100-item routing set,
the GPU hosts and Lambda MicroVMs all scored 72%, the CPU hosts matched them on the 20 items
they ran, and an NVIDIA L4 matched a laptop run on 199 of 200 routes.

- [Which host](#which-host): one table, measured.
- [SageMaker AI real-time endpoints](#sagemaker-ai-real-time-endpoints): the simplest GPU path, with a maintained sample.
- [Amazon Bedrock AgentCore Runtime](#amazon-bedrock-agentcore-runtime): GPU Instances next to your agents, or CPU microVMs.
- [AWS Lambda MicroVMs](#aws-lambda-microvms): the fastest CPU option, with a snapshot trap.
- [When it pays for itself](#when-it-pays-for-itself): break-even against an LLM call.
- [Serving notes for every host](#serving-notes-for-every-host).

These are field measurements by one person, not a benchmark. Each row names its date,
Region and checkpoint. Re-measure before you size a production deployment: the package is
moving quickly (for example, [#48](https://github.com/strands-labs/strands-decider/pull/48)
targets bf16 on Graviton, which would change every CPU row below).

## Which host

Time per decision is the server's own p50 for one routing question, warm.

| Host | Hardware | Time per decision | Start-up | Measured |
|---|---|---|---|---|
| SageMaker AI real-time endpoint | `ml.g6.xlarge`, NVIDIA L4 | 147–149 ms | Always on | 8 Oct 2026, eu-west-1, PyTorch 2.14 image |
| SageMaker AI real-time endpoint | `ml.g5.xlarge`, NVIDIA A10G | 162 ms | Always on | 8 Oct 2026, us-east-1, PyTorch 2.14 image |
| AgentCore Runtime Instances | `g4dn.xlarge`, NVIDIA T4 | 475 ms | 174–194 s per new session; 3.7 min for the first call | 5 Oct 2026, eu-west-1, v19 |
| SageMaker AI real-time endpoint | `ml.g4dn.xlarge`, NVIDIA T4 | 483 ms | Always on | 5 Oct 2026, eu-west-1, v19 |
| Lambda MicroVMs | Graviton, 8 GB / 4 vCPU, bursts to 16 vCPU | 2.35 s | 6–7 s from snapshot; about 7 min for the first decision on a new MicroVM | 7 Oct 2026, eu-west-1, v19 |
| AgentCore Runtime microVM (V1) | 2 vCPU, 8 GB, CPU | 9.1 s | 10–17 s per new session | 5 Oct 2026, eu-west-1, v19 |

In short: use a GPU for anything a user waits on. CPU hosts answer in seconds, which suits
background work such as overnight triage or labelling transcripts.

## SageMaker AI real-time endpoints

[aws-samples/sagemaker-genai-hosting-examples](https://github.com/aws-samples/sagemaker-genai-hosting-examples/tree/main/01-models/StrandsAgents/strands-decider-2B)
has a maintained sample: a notebook that deploys the model on the AWS PyTorch 2.14 SageMaker
container, and a second notebook for accounts with no internet access
(`EnableNetworkIsolation=True`, weights and the runtime staged in S3).

- **Set `InferenceAmiVersion` on `ml.g5` and `ml.g4dn`.** Their default host image has NVIDIA
  driver 470, which cannot start a CUDA 12 or 13 build (`CannotStartContainerError`, or PyTorch
  reporting "The NVIDIA driver on your system is too old"). `al2023-ami-sagemaker-inference-gpu-4-1`
  (driver 580) works on g4dn, g5 and g6.
- **Allow more than one instance type.** `ml.g5.xlarge` had no capacity in eu-west-1 on
  several attempts in October 2026; the sample falls back from g5 to g6.

## Amazon Bedrock AgentCore Runtime

**Runtime Instances** run the container on GPU instances that AgentCore manages through a
capacity provider. They give 475 ms on a T4, but each new session takes about 3 minutes, so
reuse one warm session. Five setup issues came up:

- The 2 GB image limit applies, and CUDA PyTorch alone is about 3 GB. Ship a small image and
  copy PyTorch and the weights from S3 at start-up.
- The host's NVIDIA driver is mounted at `/usr/lib64`. Debian-based images such as
  `python:3.12-slim` do not search there, so PyTorch silently runs on CPU. Load
  `/usr/lib64/libcuda.so.1` with `ctypes` before importing torch.
- The runtime's network interface has no public IP. It needs VPC endpoints for ECR (api and
  dkr), S3, CloudWatch Logs and STS.
- Do not set `networkConfiguration` on the runtime; the VPC comes from the capacity provider.
- List instance types that exist in your Region (eu-west-1 has no EC2 g6).

**Runtime microVMs** (CPU, 2 vCPU and 8 GB per session) work on V1 at about 9 s a decision.
The default fp32 upcast does not fit in 8 GB, so the process is killed while converting;
upcasting one decoder layer at a time, with the embedding table left in bf16, stays under the
limit and gives the same answers.

## AWS Lambda MicroVMs

Lambda MicroVMs start from a snapshot of the running application, so the model loads once, at
build time. Warm decisions take 2.35 s (p95 2.7 s), and a suspended MicroVM resumes and answers
in about the same time.

- **Snapshot in bf16, upcast after restore.** A snapshot of the fp32 model (about 7.5 GB
  resident) builds, but MicroVMs restored from it terminate before starting at the 8 GB
  baseline. Snapshotting in bf16 (under 1 GB resident) and upcasting to fp32 on the first
  request works. Do not serve in bf16 on CPU with the current package: one decision took
  almost 8 minutes.
- **Do not upcast in the run hook.** It is capped at 60 s, and a build with the run hook
  enabled failed with "Ready hook invocation timed out after PT30M".
- **The first decision on a new MicroVM takes about 7 minutes.** Memory pages are read from the
  snapshot on first touch (the upcast took 402–429 s after restore, against 3 s during the
  build), and the endpoint dropped that first response. Send a warm-up request and wait for it
  before real traffic, then suspend the MicroVM when idle rather than starting one per request.

## When it pays for itself

On four questions per message, priced at eu-west-1 list prices on 7 Oct 2026 (v21):

- Decider first with Claude Haiku 4.5 for unsure answers cost $1.69 per 1,000 requests against
  $4.06 for four separate Haiku calls, but only 4% less than one Haiku call answering all four
  ($1.77).
- Against that one call, an `ml.g6.xlarge` breaks even at about 2,970 requests an hour.

The full method and a break-even calculator are in
[When does a decision model pay for itself?](https://builder.aws.com/content/3KMyAB6sMrE15fvhBt1DmOa7CfR/when-does-a-decision-model-pay-for-itself).
The hosting measurements are in
[Running Strands Decider 2B on Amazon Bedrock AgentCore](https://builder.aws.com/content/3KJvK45IGyUNQwRd4dSOZQWKv8I/running-strands-decider-2b-on-amazon-bedrock-agentcore-microvms-vs-gpu-instances).

## Serving notes for every host

- **Check that a GPU deployment is on the GPU.** On both AgentCore and SageMaker AI the first
  GPU deployments ran on CPU at about 5.5 s a decision, with no error. Log
  `torch.cuda.get_device_name(0)` at start-up and fail the health check if a GPU host has no GPU.
- **Serialise requests.** Until [#17](https://github.com/strands-labs/strands-decider/issues/17)
  is fixed, overlapping requests can receive each other's answers, so put a lock around
  `evaluate`.
- **Stage the base model.** The checkpoint names its base model by Hub id, so by default it
  downloads about 4.6 GB from Hugging Face at start-up. Copy the weights to S3 and point
  `base_model` at a local folder to avoid the download, or where the host has no internet
  access.
