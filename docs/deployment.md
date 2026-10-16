# Serving on AWS

This document is a reference for choosing an AWS host for Strands decider and for avoiding the
setup problems each host has. Step-by-step deployment exists only for SageMaker AI, in the
aws-samples example linked below. For Amazon Bedrock AgentCore Runtime and AWS Lambda MicroVMs,
this page gives measurements and the problems found, not a complete container.

- [Which host](#which-host): one table, measured.
- [SageMaker AI real-time endpoints](#sagemaker-ai-real-time-endpoints): the aws-samples example and the GPU driver.
- [Amazon Bedrock AgentCore Runtime](#amazon-bedrock-agentcore-runtime): GPU Instances, or CPU microVMs.
- [AWS Lambda MicroVMs](#aws-lambda-microvms): the fastest CPU host measured; prime the model before the snapshot.
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
production deployment. On the AgentCore Runtime microVM, the model was served in fp32, upcast one
decoder layer at a time with the embedding table left in bf16, because the package's default
upcast was killed in 8 GB. On Lambda MicroVMs it was measured in both fp32 and bf16, with the same
answers (see below).

## Which host

Time per decision is the server's own p50, warm, for one request with two questions (a
five-way routing choice and a yes/no scope check), on hobson-v19 in eu-west-1.

| Host | Hardware | Time per decision | Start-up | Measured |
|---|---|---|---|---|
| SageMaker AI real-time endpoint | `ml.g6.xlarge`, NVIDIA L4 | 131 ms | Always on | 5 Oct 2026 |
| AgentCore Runtime Instances | `g4dn.xlarge`, NVIDIA T4 | 475 ms | 174–194 s per new session; 3.7 min for the first call | 5 Oct 2026 |
| SageMaker AI real-time endpoint | `ml.g4dn.xlarge`, NVIDIA T4 | 483 ms | Always on | 5 Oct 2026 |
| Lambda MicroVMs | Graviton, 8 GB / 4 vCPU baseline (the largest), bursts to 32 GB / 16 vCPU | 2.8 s (bf16); 2.35 s (fp32) | 6–7 s from a snapshot to running; first decision in 1.6–1.9 s with the model primed before the snapshot | 7 and 10 Oct 2026 |
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
  147 and 149 ms on `ml.g6.xlarge` and 162 ms on `ml.g5.xlarge` on 8 Oct 2026
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
reuse one warm session. Six setup problems came up. They reflect platform behaviour on 5 Oct 2026,
so re-check them:

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
build time. With the setup below, a brand-new MicroVM answered its first request 7–9 s after it
was started (measured 10 Oct 2026).

- **Serve in bf16.** A snapshot of the fp32 model (about 7.5 GB resident; 9.35 GB at peak) was
  taken, but the builds failed: MicroVMs restored from it terminated before starting at the 8 GB
  baseline. Keeping the model in bf16 (under 1 GB resident) and serving it in bf16 worked: on
  Graviton with PyTorch 2.14, bf16 gave the same answers as fp32 on all 100 routing items, at
  2.8 s per decision warm instead of 2.35 s. Serving bf16 means turning off the package's CPU upcast
  to fp32; we did that in code, and
  [#48](https://github.com/strands-labs/strands-decider/pull/48) would make it an option.
- **Prime the model before the snapshot.** Lambda calls two hooks while it builds the image:
  `/ready` before it takes the snapshot, and `/validate` on a test MicroVM started from that
  snapshot ([image build hooks](https://docs.aws.amazon.com/lambda/latest/dg/microvms-images.html)).
  Load the model and run a couple of real inferences before `/ready` returns 200, so the snapshot
  holds a model that has already answered. Then run one more inference in `/validate`: Lambda
  records which parts of the snapshot the application reads there and prefetches them when later
  MicroVMs start. The run hook is no place for this work: its timeout is at most 60 s
  ([MicrovmHooks](https://docs.aws.amazon.com/lambda/latest/microvm-api/API_MicrovmHooks.html)).
- **Return 503 from the hooks while the work runs.** Lambda polls both hooks, so return 503 straight
  away, do the work in a background thread, and return 200 when it is done. When our `/validate`
  ran the inference inside the request, every build failed with "MicroVM terminated before it
  started". Our builds had also been staging a second checkpoint by mistake (9.3 GB of weights
  instead of 4.7 GB), and builds with the warm-up only passed once that was fixed too.
- **Speed.** From `run-microvm` to RUNNING took 6–7 s. With the model primed before the snapshot,
  the first request on a brand-new MicroVM took 1.6–1.9 s on four MicroVMs (a short two-question
  request), the same as the second request on each. Two earlier setups were much slower: with
  nothing in `/validate`, the first decision took about 7 minutes (the upcast took 402–429 s,
  against 3 s during the build, while memory pages were read from the snapshot), and the endpoint
  dropped that response; with an inference in `/validate` but none before the snapshot, it took
  about 48 s. Warm, a decision took 2.8 s in bf16, and 2.35 s in fp32 (server p50; p95 2.7 s;
  100 items). A suspended MicroVM resumed and answered in 2.4 s.

That is fast enough to scale from zero; keep one MicroVM suspended if you need answers sooner.

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
LOCK = threading.Lock()  # our runs served one request at a time
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

Run it with `uvicorn app:app --host 0.0.0.0 --port 8080`. The snippet leaves out two things we added
for the CPU hosts: on the AgentCore Runtime microVM, a per-layer fp32 upcast in place of the
package's; on Lambda MicroVMs, bf16 serving and the priming inferences before the server starts,
plus routes for the image build hooks. On Lambda MicroVMs the same app served on port 8080.

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
  deployments ran on CPU at about 5.5 s a decision, with no deployment error. Log
  `torch.cuda.get_device_name(0)` at start-up and fail the health check if a GPU host has no GPU.
- **Overlapping requests.** Overlapping requests could receive each other's answers
  ([#17](https://github.com/strands-labs/strands-decider/issues/17)) until
  [#18](https://github.com/strands-labs/strands-decider/pull/18) fixed it on 14 Oct 2026. Every
  measurement on this page predates that fix and served one request at a time behind a lock.
  `strands-decider serve` now serialises requests only on MPS (the lock from
  [#68](https://github.com/strands-labs/strands-decider/pull/68), narrowed by #18).
- **Hosts with no internet access.** The checkpoint names its base model by Hub id, so by default
  the base model (a 4.55 GB file) downloads from Hugging Face at start-up. The aws-samples
  network-isolated notebook stages a revision-pinned Hugging Face cache in S3 and sets
  `HF_HUB_OFFLINE=1`, which keeps the pinned revisions. Our measurements instead edited `base_model`
  in the checkpoint's config to point at a local folder, which drops the revision pin.
