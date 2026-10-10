# Serving on AWS

This document is a reference for choosing an AWS host for Strands decider and for avoiding the
setup problems each host has. Step-by-step deployment exists only for SageMaker AI, in the
aws-samples example linked below. For Amazon Bedrock AgentCore Runtime and AWS Lambda MicroVMs,
this page gives measurements and the problems found, not a complete container.

- [Which host](#which-host): one table, measured.
- [SageMaker AI real-time endpoints](#sagemaker-ai-real-time-endpoints): the aws-samples example and the GPU driver.
- [Amazon Bedrock AgentCore Runtime](#amazon-bedrock-agentcore-runtime): GPU Instances, or CPU microVMs.
- [AWS Lambda MicroVMs](#aws-lambda-microvms): the fastest CPU host measured, with a snapshot trap.
- [Calling the model](#calling-the-model): the routes each host expects, and who can call it.
- [When it pays for itself](#when-it-pays-for-itself): break-even against an LLM call.
- [Serving notes for every host](#serving-notes-for-every-host).

**Which model these numbers are from.** Every hosting measurement on this page used
`strands-decider-2B-hobson-v19` with package version 0.1.0; the cost section used
`strands-decider-2B-hobson-v21` with the package at commit `3e94e9d`. Both models
predate `strands-decider-2B-qwen3.5-v1-2610` and the Gemma 4 models, which are different recipes
([naming.md](naming.md)) and have not been measured on these hosts.

The host changed speed and cost, not answers. On a 100-item routing set, the GPU hosts in the
table and Lambda MicroVMs all scored 72% on routing, and the AgentCore Runtime microVM matched
them on the 20 items it ran.

These are field measurements by one person, not a benchmark. Re-measure before you size a
production deployment. Every CPU row served the model in fp32, upcast one decoder layer at a time
with the embedding table left in bf16 (on the 8 GB AgentCore Runtime microVM, the package's
default upcast was killed; see below).
[#48](https://github.com/strands-labs/strands-decider/pull/48) would add an opt-in bf16 mode for
Graviton, which was not measured here.

## Which host

Time per decision is the server's own p50, warm, for one request with two questions (a
five-way routing choice and a yes/no scope check), on hobson-v19 in eu-west-1.

| Host | Hardware | Time per decision | Start-up | Measured |
|---|---|---|---|---|
| SageMaker AI real-time endpoint | `ml.g6.xlarge`, NVIDIA L4 | 131 ms | Always on | 5 Oct 2026 |
| AgentCore Runtime Instances | `g4dn.xlarge`, NVIDIA T4 | 475 ms | 174–194 s per new session; 3.7 min for the first call | 5 Oct 2026 |
| SageMaker AI real-time endpoint | `ml.g4dn.xlarge`, NVIDIA T4 | 483 ms | Always on | 5 Oct 2026 |
| Lambda MicroVMs | Graviton, 8 GB / 4 vCPU baseline (the largest), bursts to 32 GB / 16 vCPU | 2.35 s | 5.6–6.7 s from a snapshot to running; about 7 min for the first decision on a new MicroVM | 7 Oct 2026 |
| AgentCore Runtime microVM, original runtime (V1) | 2 vCPU, 8 GB, CPU | 9.1 s | 10–17 s per new session | 5 Oct 2026 |

All rows ran one FastAPI wrapper around the package's engine, built for these tests (see
[Calling the model](#calling-the-model)), not the aws-samples container. Use a GPU for anything a
user waits on. The CPU hosts answer in seconds, which suits background work such as overnight
triage or labelling transcripts.

## SageMaker AI real-time endpoints

The [aws-samples example](https://github.com/aws-samples/sagemaker-genai-hosting-examples/tree/main/01-models/StrandsAgents/strands-decider-2B)
deploys hobson-v19 on a serving image built from the AWS PyTorch 2.14 SageMaker container. A
second notebook deploys the endpoint with no internet access (`EnableNetworkIsolation=True`): it
stages the weights and the runtime in S3, and the notebook itself needs internet while it does.
The example's README calls it an evaluation example to review before production.

- **The example is pinned to hobson-v19.** It downloads the checkpoint at fixed revisions and
  expects `hobson_config.json`. The newer models ship `strands_decider_config.json` instead, so
  changing the model id alone does not work.
- **Its latency check is a smoke test.** The notebook reported a median warm server latency of
  147–149 ms on `ml.g6.xlarge` and 162 ms on `ml.g5.xlarge` on 8 Oct 2026
  ([aws-samples #264](https://github.com/aws-samples/sagemaker-genai-hosting-examples/pull/264)).
  That is the median of three repeats of a three-question example request, so it is not
  comparable with the table above.
- **Set `InferenceAmiVersion` on `ml.g4dn` and `ml.g5`.** Their default host image has NVIDIA
  driver 470. On `ml.g4dn`, a CUDA 12 PyTorch build started but could not use the GPU ("The
  NVIDIA driver on your system is too old") and ran on CPU. The example's CUDA 13 image does not
  start at all on either (`CannotStartContainerError`). `al2023-ami-sagemaker-inference-gpu-4-1` (driver
  580) worked on `ml.g4dn`, `ml.g5` and `ml.g6`.
- **Allow more than one instance type.** `ml.g5.xlarge` had no capacity in eu-west-1 on several
  attempts in October 2026; the example falls back from g5 to g6.

## Amazon Bedrock AgentCore Runtime

**Runtime Instances** run the container on EC2 instances that AgentCore manages in your account,
through a capacity provider. GPU instance types are optional, and the container can be `x86_64` or
`arm64`. On a T4, a decision took 475 ms, but each new session took about 3 minutes to start, so
reuse one warm session. Six setup problems came up (5 Oct 2026):

1. The 2 GB image limit applies, and CUDA PyTorch alone is about 4 GB. Ship a small image and copy
   PyTorch (built for the image's Python version) and the weights from S3 at start-up.
2. The host's NVIDIA driver was mounted at `/usr/lib64`. Debian-based images such as
   `python:3.12-slim` do not search there, so PyTorch silently ran on CPU. Loading
   `/usr/lib64/libcuda.so.1` with `ctypes` before importing torch fixed it. The AgentCore
   [Instances guide](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-instances-how-it-works.html)
   now says standard CUDA images work without bundling drivers, so check whether you still need this.
3. The runtime's network interface has no public IP. It needs VPC endpoints for ECR (api and
   dkr), S3 (gateway) and CloudWatch Logs
   ([VPC requirements](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-vpc.html));
   we also added STS. Without them we got "The agent artifact could not be downloaded".
4. Our Instances runtime rejected `networkConfiguration`; the VPC comes from the capacity provider.
5. Keep the runtime's `maxLifetime` at or below the capacity provider's.
6. List instance types that exist in your Region: eu-west-1 had no EC2 g6 in October 2026,
   although SageMaker AI offers `ml.g6` there.

**Runtime microVMs** are CPU only and take `arm64` containers only
(see "Compare compute types" in the [Instances guide](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-instances-how-it-works.html)),
so an `x86_64` GPU image does not run there as it is. Our sessions had 2 vCPU and 8 GB. On the
original runtime (V1), a decision took about 9.1 s. The package's default fp32 upcast does not fit
in 8 GB, and the process was killed while converting; upcasting one decoder layer at a time, with
the embedding table left in bf16, stayed under the limit and gave the same answers.

## AWS Lambda MicroVMs

Lambda MicroVMs start from a snapshot of the running application, so the model loads once, at
build time. Warm, a decision took 2.35 s (server p50; client p95 2.7 s; 100 items, 16 threads),
and after a suspend, the first request took 2.32 s (server; 2.37 s client).

- **Snapshot in bf16, upcast after restore.** A snapshot of the fp32 model (about 7.5 GB resident)
  was taken, but the builds failed: MicroVMs restored from it terminated before starting at the
  8 GB baseline. A bf16 snapshot (under 1 GB resident) with the fp32 upcast on the first request
  worked. Do not skip the upcast: served in bf16, the first decision took 7 min 49 s and the second
  over 2.5 minutes.
- **The first decision on a new MicroVM took about 7 minutes.** Memory pages are read from the
  snapshot on first touch: the upcast took 402 s and 429 s on two fresh MicroVMs, against about
  3 s during the build. The endpoint also did not return that first response (the server logged
  200; the client never received it). Send a warm-up request and wait for it to finish before real
  traffic, then suspend the MicroVM when idle rather than starting one per request.
- **Do not move the upcast into the run hook.** The run hook's timeout is at most 60 s
  ([MicrovmHooks](https://docs.aws.amazon.com/lambda/latest/microvm-api/API_MicrovmHooks.html)),
  far less than the upcast takes on a fresh MicroVM. In our test, enabling the run hook also made
  the image build fail with "Ready hook invocation timed out after PT30M" (30 minutes is the ready
  timeout we set), although `/ready` had returned 200 within a minute; we did not find the cause.
  This build failure is reported on this page only, not in the linked write-ups.

## Calling the model

`strands-decider serve` exposes `GET /health` and `POST /v1/systemone`, on `127.0.0.1:8000` by
default. SageMaker AI custom containers and AgentCore Runtime call `GET /ping` and
`POST /invocations` on port 8080 instead, so wrap the engine. This is the core of the wrapper used
for every measurement above:

```python
import threading

from fastapi import FastAPI, HTTPException
from strands_decider.infer import load_engine
from strands_decider.schema import SystemOneRequest

ENGINE = load_engine("/opt/model/decider", device="cuda")  # a local checkpoint folder
LOCK = threading.Lock()  # see #17
app = FastAPI()


@app.get("/ping")
def ping():
    return {"status": "Healthy"}


@app.post("/invocations")
def invocations(body: dict):
    try:
        with LOCK:
            out = ENGINE.evaluate(SystemOneRequest(**body))
    except ValueError as exc:  # for example, a malformed request
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return out.model_dump()
```

Run it with `uvicorn app:app --host 0.0.0.0 --port 8080`. On the CPU hosts we also replaced the
package's fp32 upcast with a per-layer one, which this snippet leaves out. On Lambda MicroVMs the
same app served on port 8080, with routes added for the image build hooks.

**Who can call it.** `serve` has no authentication of its own, so access control comes from the
host:

- **SageMaker AI:** IAM. Callers need `sagemaker:InvokeEndpoint` on the endpoint.
- **AgentCore Runtime:** IAM SigV4 by default, or JWT bearer tokens from your identity provider;
  one method per runtime version
  ([Inbound Auth](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-oauth.html)).
- **Lambda MicroVMs:** every request carries a token from `create-microvm-auth-token` in the
  `X-aws-proxy-auth` header. Tokens are encrypted JWE strings scoped to one MicroVM, a set of ports
  and an expiry time
  ([Networking](https://docs.aws.amazon.com/lambda/latest/dg/microvms-networking.html)).

## When it pays for itself

On four questions per support message, with hobson-v21 on `ml.g6.xlarge`, at eu-west-1 list
prices on 7 Oct 2026 and 1 request a second:

- Decider first, sending the whole request to Claude Haiku 4.5 when any answer's confidence was
  below 0.6, cost $1.69 per 1,000 requests. That is against $4.06 for four separate Haiku calls,
  but only 4% less than one Haiku call answering all four ($1.77). It got all four answers right on
  70% of messages, against 73% for that one Haiku call.
- Against that one call, this cascade breaks even at about 2,970 requests an hour on one
  `ml.g6.xlarge`.

The method and a break-even calculator are in
[When does a decision model pay for itself?](https://builder.aws.com/content/3KMyAB6sMrE15fvhBt1DmOa7CfR/when-does-a-decision-model-pay-for-itself).
The 5 Oct hosting measurements are in
[Running Strands Decider 2B on Amazon Bedrock AgentCore](https://builder.aws.com/content/3KJvK45IGyUNQwRd4dSOZQWKv8I/running-strands-decider-2b-on-amazon-bedrock-agentcore-microvms-vs-gpu-instances),
and the 8 Oct SageMaker AI figures in
[aws-samples #264](https://github.com/aws-samples/sagemaker-genai-hosting-examples/pull/264).

## Serving notes for every host

- **Check that a GPU deployment is on the GPU.** On both AgentCore and SageMaker AI, the first GPU
  deployments ran on CPU at about 5.6–5.7 s a decision, with no deployment error. Log
  `torch.cuda.get_device_name(0)` at start-up and fail the health check if a GPU host has no GPU.
- **Serialise requests.** Until [#17](https://github.com/strands-labs/strands-decider/issues/17)
  is fixed, overlapping requests can receive each other's answers, so put a lock around `evaluate`.
- **Hosts with no internet access.** The checkpoint names its base model by Hub id, so by default
  the base model (a 4.55 GB file) downloads from Hugging Face at start-up. The aws-samples
  network-isolated notebook stages a revision-pinned Hugging Face cache in S3 and sets
  `HF_HUB_OFFLINE=1`, which keeps the pinned revisions. Our measurements instead edited `base_model`
  in the checkpoint's config to point at a local folder, which drops the revision pin.
