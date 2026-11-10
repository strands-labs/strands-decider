# Serving on AWS Lambda

**Serverless: no GPU, no instance to reserve, nothing standing between requests.** This guide
deploys the 2B Qwen3.5 strands decider as an AWS Lambda function in your AWS account, on arm64
CPUs. Lambda starts execution environments as requests arrive, runs as many side by side as the
traffic needs, and you pay per request. One AWS CDK stack builds the image and creates the
function. The function serves `strands-decider serve`'s `POST /v1/systemone` and `GET /health`
([docs/inference.md, Serve](../../docs/inference.md#serve)) behind an IAM-authenticated
Function URL, and takes the same request as a direct invocation too.

The torso runs in [llama.cpp](https://github.com/ggml-org/llama.cpp), quantized to Q8_0, and the
rest is the package's torch-free `HiddenStateEngine`
([src/strands_decider/hidden_engine.py](../../src/strands_decider/hidden_engine.py)), which its
tests hold to the torch engine's answers. A request with three questions about a short state
takes 1.6 to 2.4 s; one about a 4,000-token state, 29 to 45 s ([Measurements](#measurements)). That
suits decisions that can wait seconds, at volumes where a GPU would sit idle: triaging tickets
or documents as they arrive, moderation and routing queues, batch scoring, a check before an
agent's consequential action. For a guard on every tool call of an agent loop, serve on a GPU
([deploy/README.md](../README.md)).

The same CDK app deploys a decision checkpoint, the format of the Strands Decider v2 models
([docs/spec.md](../../docs/spec.md)), with text and images: `-c decision=true`. See
[Decision checkpoints](#decision-checkpoints).

To follow it you need:

- An AWS account, and credentials that can deploy CloudFormation stacks that create IAM roles.
- Node.js 22 or later. It runs the AWS CDK through `npx`, so you do not install the CDK
  separately.
- [awscurl](https://github.com/okigan/awscurl) (`pip install awscurl`), to test the function,
  and the AWS CLI v2, to read its logs.

Install the CDK app's dependencies, and bootstrap CDK once for each account and region:

```bash
cd deploy/lambda
npm install
npx cdk bootstrap aws://<account>/<region>
```

## 1. Deploy

```bash
npx cdk deploy
```

This builds the image in AWS CodeBuild, then creates the function and publishes its first
version: about 10 minutes.

## 2. Test it

```bash
awscurl --service lambda --region <region> -X POST -H 'content-type: application/json' \
  -d '{"state": "Help! My payouts have been failing for 3 days!",
       "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}' \
  <FunctionUrl>v1/systemone
```

```json
{"model": "strands-decider-2B-qwen3.5-v1-2610-q8_0", "answers": {"is_urgent": {"type": "noul", "noul": 0.8634}}, "usage": {"input_tokens": 86, "output_tokens": 1}, "latency_ms": 1033.72}
```

`<FunctionUrl>` is in `cdk deploy`'s outputs. The URL accepts only SigV4-signed requests from
principals with `lambda:InvokeFunctionUrl` on the function. A direct invocation, for Step
Functions, EventBridge targets and other AWS callers, takes the request as the event and returns
the response:

```bash
aws lambda invoke --function-name <FunctionArn> --cli-binary-format raw-in-base64-out \
  --payload '{"state": "Help! My payouts have been failing for 3 days!",
              "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}' \
  answer.json
```

## Debugging

The function's output, including any error while the model loads, and a line per request with
its latency, its tokens and the host's CPU (`cpu_part`: 0xd0c Graviton2, 0xd40 Graviton3,
0xd4f Graviton4), is in CloudWatch Logs:

```bash
aws logs tail <LogGroup from the stack outputs> --region <region> --since 1h --follow
```

If the image build fails, `cdk deploy` fails with a link to the CodeBuild log, and CloudFormation
rolls the stack back.

## What `cdk deploy` creates

One CloudFormation stack per model, named after it: `strands-decider-2b-qwen3-5-v1-2610-lambda`
for the default. It holds:

1. The image, built from `Dockerfile` for `linux/arm64` by a CodeBuild project
   ([deploy-time-build](https://github.com/cdklabs/deploy-time-build)) and published to an ECR
   repository in the stack. The build downloads the model and its base model from the Hugging Face
   Hub, folds the model's LoRA into the torso (`scripts/prepare_model.py`), converts the torso to a
   GGUF and quantizes it to Q8_0 (about 2 GB), and builds llama.cpp and `function/shim.c` into
   one library. The image holds those, the pointer head, the tokenizer, and the package without
   torch. It runs on the CodeBuild project's `LARGE` instance (16 GB of memory), as the merge
   takes about 8 GB.
2. The function: arm64, 10,240 MB (the most Lambda gives, and the only size with 6 vCPUs), a
   15-minute timeout, SnapStart if you turn it on ([What it costs](#what-it-costs)), and a log
   group that keeps two weeks. Its `live` alias points at the latest version, and the Function URL serves the alias.
   Each deploy that changes the function publishes a new version and deletes the one before.

## Configuration

Every setting is a CDK context value. Change it in `cdk.json`, or pass `-c key=value` to
`cdk deploy`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `model` | `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` | The strands-decider model, a Hugging Face Hub id. Any Qwen3.5 model with a pointer head works. It names the stack. |
| `model_revision` | `main` | The model's revision. Use a release's date tag (`YYMMDD`) to pin it ([docs/inference.md, Model artifact](../../docs/inference.md#model-artifact)). |
| `snapstart` | `false` | `true` turns SnapStart on: new execution environments answer in about 6 s instead of up to 30 s, for about $39.50 a month while the stack exists ([What it costs](#what-it-costs)) |
| `max_request_tokens` | `24576`, `16384` for a decision checkpoint | The most tokens one request may forward, the state once plus every question; `0` for no limit ([Limits](#limits)) |
| `decision` | `false` | `true` deploys a decision checkpoint, `checkpoint`, instead of `model` ([Decision checkpoints](#decision-checkpoints)) |
| `checkpoint` | none | The decision checkpoint: a Hugging Face Hub id, or an `s3://bucket/prefix/` URI of a checkpoint directory in the same account. It names the stack. |
| `checkpoint_revision` | `main` | The checkpoint's Hub revision |
| `stack_name` | from the model | The stack's name, instead of one made from `model` or `checkpoint` |

## What it costs

At us-east-1 prices, a request costs $0.000133 per second it runs (10 GB of arm64 at
$0.0000133334 per GB-second): $0.0002 to $0.0003 for three questions about a short state.
CloudWatch Logs adds $0.50 per GB ingested, and a request writes a few hundred bytes.

SnapStart decides the rest:

| | `snapstart=false` (default) | `snapstart=true` |
| --- | --- | --- |
| Nobody calling | **about $0.25 a month**, the image in ECR | **about $39.50 a month** for the snapshot, and the image |
| First request on a new execution environment | up to 30 s | about 6 s |
| Each new execution environment | about $0.002, the 6 to 27 s the model takes to load | $0.0014 to restore the snapshot |

SnapStart is off by default, so a deploy costs nothing while nobody calls it. Turn it on
(`npx cdk deploy -c snapstart=true`) when the first answer of a new environment matters: once
you are measuring latency for real traffic, or traffic is bursty enough that Lambda often adds
environments.

With SnapStart, Lambda loads the model when it publishes a version, snapshots that execution
environment, and starts new ones from the snapshot. For Python functions Lambda bills the
snapshot's cache, $0.0000015046 per GB-second of memory, for as long as the version exists:
about $1.30 a day at 10 GB, whether or not anyone calls. That is why each deploy deletes the
version it replaces. Without SnapStart, nothing is billed between requests beyond the image in
ECR, and each new execution environment loads the model before it answers. Environments stay
warm for a while between requests, so steady traffic pays that only when Lambda scales out.

## Measurements

Measured on the default model in us-east-1. Warm requests, three questions each, the median of 6
to 32 per row:

| Tokens per request | Graviton2 | Graviton3 | Cost on Graviton2 |
| --- | --- | --- | --- |
| 196 | 2.4 s | 1.6 s | $0.0003 |
| 2,264 | 24.5 s | 15.2 s | $0.0033 |
| 4,061 | 45.4 s | 28.6 s | $0.0061 |

Lambda picks the host, not you, so expect a mix; the log line of each request names its CPU
([Debugging](#debugging)). The state is decoded once per request, so a request costs about the
state's tokens plus 65 to 90 per question, and asking several questions in one request is much
cheaper than separate requests. Peak memory stayed at 5.6 GB, with a 3,900-token state and 64
questions, and with 300 questions (19,400 tokens, 4 minutes on Graviton2).

With SnapStart, 10 new execution environments answered their first request in 5.6 to 7.5 s.
Without it, 60 new environments over four runs loaded the model in 6 to 27 s, and their first
requests took up to 30 s. Two more environments in one of those runs took 161 s to load; no
request waited on one, and every request finished within 30.5 s.

Compared with the torch engine in fp32 on CPU, over 20 requests that cover every question type,
structured and Unicode states, 40, 64 and 300 questions and a 3,800-token state, the function
moved probabilities and confidences by at most 0.026 and changed no choice; one yes/no answer
went from 0.495 to 0.505. The quantization to Q8_0 accounts for most of that. Requests that
`serve` refuses, this function refuses too, with the same status.

## Limits

- The state and the longest question must fit the model's window (`max_length`, 4,096 tokens for
  the v1 release). A prompt over it is handled as `serve` handles one
  ([docs/inference.md](../../docs/inference.md)), with a 422 where `serve` refuses it.
- A request may forward at most `max_request_tokens` tokens, the state once plus every question.
  The default, 24,576, stays well inside the 15-minute timeout; a larger request is refused with
  a 422 before any work is done.
- No images: a request with `images` gets a 422.
- A Function URL request or response is at most 6 MB.
- One request at a time per execution environment. Lambda scales by adding environments.

## Decision checkpoints

`-c decision=true` deploys a decision checkpoint instead: the Strands Decider v2 format
([docs/spec.md](../../docs/spec.md)), a Gemma 4 model in its base's multimodal layout with a
trained readout beside it. The function answers the same `POST /v1/systemone`, with images, and
`GET /health`, behind the same IAM-authenticated Function URL, and takes the request as a direct
invocation too.

```bash
npx cdk deploy -c decision=true -c checkpoint=<Hub id, or s3://bucket/prefix/>
```

Lambda serves E2B, with text and images. E4B does not fit with its vision tower: its text model
alone fills the 10,240 MB function, so the build refuses a model over 6 GB.

The request is rendered, checked and read by the torch-free core
([src/strands_decider/core](../../src/strands_decider/core)), and the decoder and the vision tower
run in llama.cpp through `strands_decider.backends.llama_cpp`: the state, its text tokens and image
rows together, is decoded once as one non-causal batch, and each question on top of it, under the
`prefix_bidir` mask with no sliding window. An image is decoded as v1 decodes it, resized with
Pillow's bicubic resampling to the size the image processor gives it, and run through the base
model's own vision tower. The readout and its soft-cap come from the checkpoint's
`readout.safetensors` and `decision_config.json`; a checkpoint without the soft-cap there, or with
another value under `config.json`'s `text_config`, is refused.

### What the build does

The image builds llama.cpp at a pinned commit with the two patches in `decision/patches`, and
`decision/shim.c` against it, for Graviton2 and later (no i8mm). It then reads the checkpoint, from
the Hugging Face Hub or, for an `s3://` URI, with the CodeBuild project's role, which the stack
grants read access to that prefix only. `decision/scripts/build_artifact.py` checks it as the
function will, then converts it with llama.cpp's own `convert_hf_to_gguf.py`:

- the decoder to BF16, then Q8_0 with `llama-quantize`, the lowest type that keeps the
  reference's answers: 4.95 GB for E2B;
- the vision tower and its projection with `--mmproj`, at Q8_0, without the audio tower, which
  would not fit beside them: 0.23 GB. On Graviton the Q8_0 tower encodes an image about three
  times as fast as the f16 one (8.7 s against 25 s on Graviton2), and answers as closely to the
  reference.

The arm64 build's files are byte-identical to an x86 build's. The build takes about 10 minutes.

### After each deploy, warm the function

A new image's files reach a new execution environment lazily, and the function reads its 5.2 GB
of model files before it answers. Right after a deploy that read is slow: in two deploys, the
first request ran 826 s and, the other time, past the 15-minute timeout, which the Function URL
returns as a 502; the next environments took 6 to 10 minutes. Later ones read the files in 20 to
40 s and answer their first request in 30 to 80 s, though an environment on a host that has not
cached the image still takes 6 to 8 minutes. So after each deploy, call `GET /health` until it
answers, before sending traffic.

### Limits

- A prompt, the state with its image placeholders plus one question, may be at most the
  checkpoint's `max_length_validated` tokens, 8,192 for E2B; a longer one gets a 422.
- At most 4 images per request, PNG, JPEG, WebP or GIF, each at most 4096 x 4096 pixels. They go
  first in the state, each as up to 280 tokens.
- A request may forward at most `max_request_tokens` tokens, the state once plus every question,
  16,384 by default. That keeps the slowest request inside the 15-minute timeout on Graviton2,
  the slowest host Lambda gives: 4 images and an 8,000-token state, then 137 questions up to the
  limit, took 557 s there, and with the shortest questions, 121 of them, 499 s on Graviton3, about
  640 s on Graviton2. A question after a long state costs about as much as 50 state tokens, so v1's
  24,576 is too many here: the same request at 24,540 tokens ran past the timeout on Graviton2.
- Every limit is checked before any image is decoded or any token is run.
- A Function URL request is at most 6 MB, base64 images included.

### Measurements

Measured in us-east-1 on an E2B decision checkpoint (Gemma 4 E2B, prompt `strands-chat`), with the
Q8_0 decoder and Q8_0 vision tower the build makes. Warm requests, server-side medians:

| Request | Graviton2 | Graviton3 | Cost on Graviton2 |
| --- | --- | --- | --- |
| 250 tokens, one question | 2.3 s | 1.5 s | $0.0003 |
| 1,100 tokens, one question | 12.1 s | 7.8 s | $0.0016 |
| 4,100 tokens, one question | 65 s | 48 s | $0.0087 |
| one photo and three questions, 570 tokens | 14.6 s | 10.2 s | $0.0019 |
| four photos and two questions, 8,131 tokens | 228 s | 175 s | $0.030 |

A request costs $0.000133 per second it runs (10 GB of arm64 at $0.0000133334 per GB-second). Each
image takes 8.8 s to encode on Graviton2 and 6.3 s on Graviton3, before its 250 to 280 state
tokens. As with v1, ask several questions in one request: the state is decoded once.

Max Memory Used, which counts the page cache, stayed at 8,111 to 8,115 MB on every one of 140
requests, the 16,384-token ones included: 2.1 GB under the 10,240 MB limit. That figure is the
environment's peak while it loads, when it reads the whole model file. Afterwards the handler drops
the per-layer embedding table, 2.5 GB of the decoder's file of which a request reads a few rows, from
the page cache (`DECIDER_PAGE_CACHE=hot`, the default); `all` keeps it. With it kept, and the f16
vision tower, a prompt of 8,131 tokens with four images reached 10,077 MB. Warm latency is the
same either way: 2.3 s against 2.9 s at 250 tokens on Graviton2.

The answers agree with the reference runtime (transformers, bf16, the same rendered prompts and
pixels) and with the same files on x86:

| Requests | Questions | Top answer as the reference's | Mean largest probability difference | Top answer as on x86 |
| --- | --- | --- | --- | --- |
| text, 250 to 6,500 tokens, one to three questions | 177 | 175 | 0.009 | 176 |
| 40 photos in 35 requests: one to four images, PNG, JPEG, WebP and GIF, strips 640 x 40 and 30 x 371, an empty state, long states | 104 | 104 | 0.008 | 104 |

Both differences from the reference are near-ties, a gap of at most 0.007 between its top two
answers. The text rows include the f16 tower's deploy; the decoder, and so the text answers, are
the same.

Each response carries `x-decider-cpu-part`, `x-decider-load-wait-ms` (above 0 on a new
environment) and `x-decider-timings` (the images', the state's and each question's milliseconds),
and the log has a line per request with the same.

## Deleting it

```bash
npx cdk destroy                    # the default model
npx cdk destroy -c model=<Hub id>  # any other model you deployed
npx cdk destroy -c decision=true -c checkpoint=<the checkpoint you deployed>
```

This deletes everything that model's `cdk deploy` created: the function, its versions and their
snapshots, the alias and Function URL, the log group, the ECR repository with the image, the
CodeBuild project and the roles. The log groups that CodeBuild and deploy-time-build's trigger
function write to stay, with the build logs: `/aws/codebuild/ContainerImageBuildArm64*` and
`/aws/lambda/*-DeployTimeBuildCustomRes-*`, at most a few hundred KB each. A decision checkpoint's
stack owns those two log groups too, so `cdk destroy` deletes them; deploy-time-build's trigger
function can write to its group once more while it handles its own deletion, which leaves that
group behind.
