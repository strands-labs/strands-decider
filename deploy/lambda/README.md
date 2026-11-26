# Serving on AWS Lambda

**Serverless: no GPU, no instance to reserve, nothing standing between requests.** This guide
deploys the 2B Qwen3.5 strands decider as an AWS Lambda function in your AWS account, on arm64
CPUs. Lambda starts execution environments as requests arrive, runs as many side by side as the
traffic needs, and you pay per request. One AWS CDK stack builds the image and creates the
function. The function serves `strands-decider serve`'s `POST /v1/systemone` and `GET /health`
([designs/inference.md, Serve](../../designs/inference.md#serve)) behind an IAM-authenticated
Function URL, and takes the same request as a direct invocation too.

The torso runs in [llama.cpp](https://github.com/ggml-org/llama.cpp), quantized to Q8_0 by default ([Precision](#precision)), and the
rest is the package's torch-free `HiddenStateEngine`
([src/strands_decider/hidden_engine.py](../../src/strands_decider/hidden_engine.py)), which its
tests hold to the torch engine's answers. A request with three questions about a short state
takes 1.6 to 2.4 s; one about a 4,000-token state, 29 to 45 s ([Measurements](#measurements)). That
suits decisions that can wait seconds, at volumes where a GPU would sit idle: triaging tickets
or documents as they arrive, moderation and routing queues, batch scoring, a check before an
agent's consequential action. For a guard on every tool call of an agent loop, serve on a GPU
([deploy/README.md](../README.md)).

The same CDK app deploys the Strands Decider v2 models, with text and images: give it the model,
and it reads what format the checkpoint is ([Model checkpoints with a
decision_config.json](#model-checkpoints-with-a-decision_configjson)).

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
   GGUF and quantizes it to Q8_0 (about 2 GB; [Precision](#precision)), and builds llama.cpp and `function/shim.c` into
   one library. The image holds those, the pointer head, the tokenizer, and the package without
   torch; with `vision=true`, also the base model's vision tower ([Images](#images)). It runs on the CodeBuild project's `LARGE` instance (16 GB of memory), as the merge
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
| `model` | `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` | The strands-decider model, a Hugging Face Hub id, or for a checkpoint with a `decision_config.json` also an `s3://bucket/prefix/` URI of a checkpoint directory in the same account. It names the stack ([Which stack a model gets](#which-stack-a-model-gets)). |
| `model_revision` | `main` | The model's Hub revision. Use a release's date tag (`YYMMDD`) to pin it ([designs/inference.md, Model artifact](../../designs/inference.md#model-artifact)). |
| `snapstart` | `false` | `true` turns SnapStart on: new execution environments answer in about 6 s instead of up to 30 s, for about $39.50 a month while the stack exists ([What it costs](#what-it-costs)) |
| `vision` | `false` | `true` adds the base model's vision tower, for requests with images ([Images](#images)) |
| `image_long_side` | `448` | The longest side, in pixels, an image is scaled to, as for `serve --vision`; `768` makes a square image 576 tokens instead of 196 ([Image size](#image-size)) |
| `decoder_precision` | `q8_0` | `f16` keeps the decoder unquantized: closer to the torch engine, and slower ([Precision](#precision)) |
| `vision_precision` | `f16` | `f32` for the vision tower ([Precision](#precision)) |
| `max_request_tokens` | `24576`, `16384` for a checkpoint with a `decision_config.json` | The most tokens one request may forward, the state once plus every question; `0` for no limit ([Limits](#limits)) |
| `model_digest` | none | A checkpoint with a `decision_config.json`: its manifest digest ([What the build does](#what-the-build-does)); the build refuses a checkpoint with another |
| `cpu` | `baseline` | A checkpoint with a `decision_config.json`: `variants` builds a CPU backend module per Arm tier and Graviton generation, picked per host ([Graviton CPU variants](../../docs/deployment/graviton-cpu-variants.md)) |
| `kleidiai` | `false` | With `cpu=variants`: `true` adds Arm's KleidiAI kernels |
| `graviton_sve` | `false` | With `cpu=variants`: `true` adds SVE Graviton modules, for measuring ([why it is off](../../docs/deployment/graviton-cpu-variants.md#why-the-graviton-modules-are-nosve)) |
| `stack_name` | from the model | The stack's name, instead of one made from `model` (an `s3://` checkpoint's name gets 8 hex digits of its URI's sha256, so checkpoints of one name in different places get stacks of their own) |

`snapstart`, `vision`, `image_long_side`, `decoder_precision` and `vision_precision` are only
supported for v1 model checkpoints, and `model_digest` is not supported for them.

### Which stack a model gets

Before it creates anything, the app asks the Hugging Face Hub which files `model` has at
`model_revision` and deploys the format the checkpoint declares:

| The checkpoint has | The app deploys |
| --- | --- |
| `strands_decider_config.json` | a v1 model checkpoint, as this guide describes |
| `decision_config.json` naming `"interface": "strands-decider"` and `"format_version": 1` | a model checkpoint with a `decision_config.json` ([Model checkpoints with a decision_config.json](#model-checkpoints-with-a-decision_configjson)) |

Anything else is refused before anything is deployed: both files or neither, another interface
or `format_version`, or a model or revision the Hub doesn't show (the app reads public models
only). An `s3://` model is always the second kind, since v1's build reads only the Hub; that
build checks its `decision_config.json` before converting anything. So `cdk synth`, `cdk deploy`
and `cdk destroy` need to reach the Hub.

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

## Images

The 2B Qwen3.5 decider can answer over images: its base model is multimodal, and with the base
model's vision tower beside it, the published checkpoint reads images with no image training
([designs/images.md](../../designs/images.md)). Deploy with the vision tower:

```bash
npx cdk deploy -c vision=true
```

A request then takes `images`, as `serve --vision` does: up to 4 base64 PNG, JPEG, WebP or GIF
images, each scaled so its longer side is at most 448 px (a square image is 196 tokens). They
go first in the state, and count against the same window.

```bash
awscurl --service lambda --region <region> -X POST -H 'content-type: application/json' \
  -d '{"state": "The checkout page after the user tapped Pay.",
       "images": ["'"$(base64 < screenshot.png | tr -d '\n')"'"],
       "questions": {"paid": {"type": "noul", "instructions": "Did the payment succeed?"}}}' \
  <FunctionUrl>v1/systemone
```

The vision tower is the base model's, unchanged, since the decider's adapter never touched it.
The build converts it to a GGUF (about 0.7 GB in f16, the default), and llama.cpp's multimodal library
(`mtmd`) runs it. The engine sizes each image as the Qwen image processor does, so the vision
tower sees the processor's pixels.

Measured in us-east-1 with vision on: a request with one 448 px screenshot and three questions
(411 tokens) takes 5.5 s on Graviton3 and 8.5 s on Graviton2 (medians of 15), against 1.6 s and
2.4 s for three questions about a short text state. Over 13 requests (screenshots, photographs and
charts from 60 to 1,280 px, non-square and upscaled images, one, two and four images, and a state
over the window), the function's token counts matched `serve --vision`'s exactly, and its answers
were within 0.048 of `serve --vision`'s in fp32 on CPU, with no choice changed. One yes/no answer,
about a photograph scaled down from 1,280 px, went from 0.477 to 0.520: that is the quantization,
and with both halves at full precision the function was within 0.016 of it
([Precision](#precision)). A prompt over the window gets the same 422.

The vision tower makes the image about 0.7 GB larger, and a new execution environment slower to
start. Without SnapStart, 14 new environments loaded the model and the vision tower in 16 to 34 s,
and the first one after the deploy took 3 minutes. With
SnapStart, new environments answered in 6.6 to 8.9 s, so turn it on for image traffic
(`npx cdk deploy -c vision=true -c snapstart=true`).

### Image size

`image_long_side` sets the longest side an image is scaled to before the processor's grid. At
768, a square image is 576 tokens instead of 196, and the vision tower reads more of a large
image's detail. Measured in us-east-1, with images larger than 768 px, three questions each
(medians of 3 to 17 per cell):

| Request | 448 px, Graviton2 | 768 px, Graviton2 | 448 px, Graviton3 | 768 px, Graviton3 |
| --- | --- | --- | --- | --- |
| 1 image | 8.6 s | 24.0 s | 5.5 s | 15.6 s |
| 2 images | 12.3 s | 36.7 s | 7.9 s | 23.7 s |
| 4 images (1,009 and 2,529 tokens) | 26.7 s | 90.1 s | 17.0 s | 57.8 s |

At 768 the function's answers were within 0.016 of `VisionEngine`'s at the same size, with the
same token counts, and memory stayed at 6.6 GB. But the size changes what the model sees, not only
what it costs: `serve --vision`'s own answers moved by up to 0.21 from 448 to 768 on these images,
and one yes/no answer crossed 0.5. The published measurements are all at 448 px
([designs/images.md](../../designs/images.md)); whether 768 answers better has not been measured.

## Precision

The decoder is quantized to Q8_0 and the vision tower stored in f16 by default. `decoder_precision`
and `vision_precision` keep either at full precision, for answers closer to the torch engine's, at a
cost in speed and memory. Measured in us-east-1 with vision on and 448 px images, against
`serve --vision` and `serve` in fp32 on CPU (13 image and 9 text requests; medians, Graviton2 /
Graviton3):

| `decoder_precision` / `vision_precision` | Images: largest difference | Text: largest difference | 1 image, 3 questions | 4 images | 4,061-token text | Peak memory |
| --- | --- | --- | --- | --- | --- | --- |
| `q8_0` / `f16` (the default) | 0.048, one yes/no answer across 0.5 | 0.026 | 8.6 / 5.5 s | 26.7 / 17.0 s | 45.3 / 28.8 s | 6.5 GB |
| `q8_0` / `f32` | 0.041, the same answer across 0.5 | 0.026 | 11.4 / 7.0 s | 38.3 / 22.9 s | 45.6 / 28.8 s | 8.5 GB |
| `f16` / `f16` | 0.038, the same answer across 0.5 | 0.004 | 12.8 / 7.3 s | 35.9 / 21.4 s | 77.9 / 44.1 s | 7.0 GB |
| `f16` / `f32` | 0.016 | 0.004 | 15.7 / 8.9 s | 47.6 / 27.0 s | 77.8 / 44.0 s | 8.2 GB |

The default is the fastest, and its answers are close to the torch engine's; the one answer that
crossed 0.5 was at 0.48 in fp32. `f16` / `f32` gives the torch engine's answers to within 0.016,
for 1.5 to 1.8 times the time and cost. For text only, `decoder_precision=f16` alone brings the
text answers to within 0.004. An `f32` vision tower with the Q8_0 decoder is slower and no closer.

## Limits

- The state and the longest question must fit the model's window (`max_length`, 4,096 tokens for
  the v1 release). A prompt over it is handled as `serve` handles one
  ([designs/inference.md](../../designs/inference.md)), with a 422 where `serve` refuses it.
- A request may forward at most `max_request_tokens` tokens, the state once plus every question.
  The default, 24,576, stays well inside the 15-minute timeout; a larger request is refused with
  a 422 before any work is done.
- Images need `vision=true` ([Images](#images)); without it, a request with `images` gets a 422.
- A Function URL request or response is at most 6 MB, so send images at the size they are read
  at, not as camera originals.
- One request at a time per execution environment. Lambda scales by adding environments.

## Model checkpoints with a decision_config.json

A checkpoint with a `decision_config.json` is the Strands Decider v2 format: a Gemma 4 model in
its base's multimodal layout with a trained readout beside it. The function answers the same
`POST /v1/systemone`, with images, and `GET /health`, behind the same IAM-authenticated Function
URL, and takes the request as a direct invocation too.

```bash
npx cdk deploy -c model=<Hub id, or s3://bucket/prefix/> \
  [-c model_revision=<Hub revision>] [-c model_digest=<sha256>]
```

Lambda serves E2B and E4B, with text and images. E4B's files are 8.3 GB, but 3.0 GB of them are
the per-layer embedding table, which llama.cpp reads a few rows at a time from disk: the function
holds 5.3 GB of weights, against E2B's 2.7 GB. The build refuses a model that would hold more than
6 GB, or whose files would not fit in the image beside the runtime (9 GB). 12B and larger do not fit.

The torch-free core ([src/strands_decider/core](../../src/strands_decider/core)) renders, checks
and reads each request. The decoder and the base model's vision tower run in llama.cpp through
`strands_decider.backends.llama_cpp`: the state, its text tokens and image rows together, is
decoded once as one non-causal batch, then each question on top of it, under the `prefix_bidir`
mask with no sliding window.

The GGUF Q8_0 decoder meets the parity bar of at least 98% top-1 agreement and a mean |Δp| of at
most 0.02: at E2B, through this runtime on x86, 99.08% and 0.0104 on 1,200 prompts.

### What the build does

`decision/scripts/build_native.sh` fetches llama.cpp at the commit in `decision/LLAMA_CPP_REV`
(release b12383), checks and applies the patch in `decision/patches`, which lets the shim make
Gemma 4's full-attention layers non-causal, stopping if it does not apply, and builds it and
`decision/shim.c` for Graviton2 and later; the build fails if the library is not arm64 or holds an
instruction Graviton2 lacks (i8mm, bf16, SVE). `decision/scripts/shim_check.py` then converts a
tiny random Gemma 4 checkpoint as the build converts a real one and runs it in that library
against transformers (tests/test_llama_cpp_shim.py runs the same check). The build fails unless
the conversion, the `prefix_bidir` mask, state reuse, the image rows and the answers are the
reference's.

The build then reads the checkpoint, from the Hugging Face Hub at the commit `model_revision`
resolves to, each LFS file checked against the sha256 the Hub records, or, for an `s3://` URI,
with the CodeBuild project's role, which the stack grants read access to that prefix only. It
records the checkpoint's manifest digest, the sha256 of the lines `<sha256>  <file>` for its files
sorted by name (in the checkpoint's directory, `sha256sum * | sort -k 2 | sha256sum`), and with
`model_digest` refuses any other. `decision/scripts/build_artifact.py` checks the checkpoint as
the function will, then converts it with llama.cpp's own `convert_hf_to_gguf.py`:

- the decoder to BF16, then Q8_0 with `llama-quantize`, the lowest type that keeps the
  reference's answers: 4.95 GB for E2B;
- the vision tower and its projection with `--mmproj`, at Q8_0, without the audio tower, which
  would not fit beside them: 0.23 GB. The build fails if its pixel normalisation or patch size is
  not the image processor's. On Graviton the Q8_0 tower encodes an image about three times as
  fast as the f16 one (8.7 s against 25 s on Graviton2), and answers as closely to the reference.

The arm64 build's files are byte-identical to an x86 build's. The build takes about 10 minutes.
`artifact.json` in the image records each file's size and sha256, the checkpoint and its manifest
digest, and the llama.cpp commit.

### After each deploy, warm the function

A new image's files reach a new execution environment lazily, and the function reads its 5.2 GB
of model files before it answers. Right after a deploy that read is slow: in three
deploys, the first request ran 826 s once and past the 15-minute timeout twice, which the Function
URL returns as a 502; the next environments took 4 to 10 minutes. Later ones read the files in 20
to 40 s and answer their first request in 30 to 80 s, though an environment on a host that has
not cached the image still takes 6 to 8 minutes. So after each deploy, call `GET /health` until it
answers, before sending traffic.

SnapStart is not offered for these checkpoints (`snapstart` is refused): it has not been
measured with a load that reads 5 GB from a lazily loaded image, and the page-cache drop below
assumes the environment read the files itself.

### Limits

- A prompt, the state with its image placeholders plus one question, may be at most the
  checkpoint's `max_length_validated` tokens, 8,192 for E2B; a longer one gets a 422.
- Images are PNG, JPEG, WebP or GIF, each at most 4096 x 4096 pixels. They go first in the state,
  each as up to 280 tokens, and their number is not limited: the prompt's length bounds how many
  fit.
- A request may forward at most `max_request_tokens` tokens, the state once plus every question,
  16,384 by default, and the function applies the same limit when `DECIDER_MAX_REQUEST_TOKENS` is
  unset; `0` lifts it. The build records the most the model can run in 10,240 MB, and the function
  lowers a higher limit to it: 16,384 for E2B, 12,288 for E4B, which ran a 13,735-token request and
  was killed for memory at 15,894. That keeps the slowest request measured inside the 15-minute timeout on
  Graviton2, the slowest host Lambda gives: 4 images and an 8,000-token state, then 137 questions
  up to the limit, took 549 to 557 s there; with the shortest questions, 121 of them, it took
  499 s on Graviton3, about 640 s on Graviton2. A question after a long state costs about as much
  as 50 state tokens, so the v1 stack's 24,576 is too many here: the same request at 24,540 tokens
  ran past the timeout on Graviton2.
- Every limit is checked before any image is decoded or any token is run. A request that breaks one,
  or carries an image that cannot be read, gets a 422 that says why; anything else is a 500.
- A Function URL request is at most 6 MB, base64 images included.

### Measurements

Measured in us-east-1 on an E2B checkpoint with a `decision_config.json` (Gemma 4 E2B, prompt
`strands-chat`), with the
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
tokens. Graviton4 hosts, which Lambda also gives, ran 250 tokens in 1.5 s and one photo with three
questions in 8.1 s. Ask several questions in one request: the state is decoded once.

Max Memory Used, which counts the page cache, was 8,388 MB on every one of 75 requests, the
16,384-token ones and those with four images included: 1.8 GB under the 10,240 MB limit. That
figure is the environment's peak while it loads, when it reads the whole model file. Afterwards
the handler drops the per-layer embedding table, 2.5 GB of the decoder's file of which a request
reads a few rows, from the page cache. With it kept, and the f16 vision tower, a prompt of 8,131
tokens with four images reached 10,077 MB, at the same warm latency: 2.3 s against 2.9 s at 250
tokens on Graviton2.

#### The v2 checkpoints, E2B and E4B

Measured in us-east-1 on `StrandsAgents/strands-decider-E2B-gemma4-v2-2611` and
`-E4B-gemma4-v2-2611` with this build, against each checkpoint's `golden_vectors.jsonl`: all 60 of its
requests with images and 80 of its text ones, eight at a time. Warm requests, server-side medians
over whichever Graviton generations Lambda gave:

| Request | E2B | E4B |
| --- | --- | --- |
| text, about 40 tokens, one to three questions | 2.5 s | 6.6 s |
| text, about 4,100 tokens | 76 s | 143 s |
| text, about 9,650 tokens | 267 s | 421 s |
| one image, about 290 state tokens | 9.0 s | 14.4 s |
| two images, about 550 | 17.0 s | 25.4 s |
| three images, about 810 | 24.8 s | 35.8 s |
| Max Memory Used, which counts the page cache | 9,297 MB | 10,238 MB |
| Top answer as the golden vector's | 399 of 403 questions | 398 of 400 |
| Mean largest probability difference | 0.0075 | 0.0047 |

E4B is about 1.5 to 2.5 times as slow as E2B. It holds twice the weights, and decoding the state is
bound by reading them. Its Max Memory Used stays at the limit because the page cache fills the rest;
that cache is given back under pressure, and no request inside the 12,288-token limit was killed.
After a deploy, E4B's first requests took longer than 15 minutes, as E2B's do (warm the function).
Each checkpoint's 7 golden requests over 16,384 tokens were refused with a 422, as designed.

The answers agree with the reference runtime (transformers, bf16, the same rendered prompts and
pixels) and with the same files on x86:

| Requests | Questions | Top answer as the reference's | Mean largest probability difference | Top answer as on x86 |
| --- | --- | --- | --- | --- |
| text, 250 to 6,500 tokens, one to three questions | 177 | 175 | 0.009 | 176 |
| 40 photos in 35 requests: one to four images, PNG, JPEG, WebP and GIF, strips 640 x 40 and 30 x 371, an empty state, long states | 104 | 104 | 0.008 | 104 |

Both differences from the reference are near-ties, a gap of at most 0.007 between its top two
answers. The text rows came partly from a deploy with the f16 vision tower, whose decoder is the
same; a deploy of this build gave the reference's top answer to 53 of 53 text questions and 104 of
104 image questions, mean largest probability differences 0.008 and 0.008.

The log has a line per request with the host CPU, the wait for the load (above 0 on a new
environment), and the images', the state's and each question's milliseconds.

## Deleting it

```bash
npx cdk destroy                      # the default model
npx cdk destroy -c model=<the model>  # any other model you deployed
```

This deletes everything that model's `cdk deploy` created: the function, its versions and their
snapshots, the alias and Function URL, the log group, the ECR repository with the image, the
CodeBuild project and the roles. The log groups that CodeBuild and deploy-time-build's trigger
function write to stay, with the build logs: `/aws/codebuild/ContainerImageBuildArm64*` and
`/aws/lambda/*-DeployTimeBuildCustomRes-*`, at most a few hundred KB each. A `decision_config.json`
checkpoint's stack owns those two log groups too, so `cdk destroy` deletes them; deploy-time-build's trigger
function can write to its group once more while it handles its own deletion, which leaves that
group behind.
