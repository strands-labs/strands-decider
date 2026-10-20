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
   15-minute timeout, SnapStart ([What it costs](#what-it-costs)), and a log group that keeps two
   weeks. Its `live` alias points at the latest version, and the Function URL serves the alias.
   Each deploy that changes the function publishes a new version and deletes the one before.

## Configuration

Every setting is a CDK context value. Change it in `cdk.json`, or pass `-c key=value` to
`cdk deploy`:

| Setting | Default | Meaning |
| --- | --- | --- |
| `model` | `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` | The strands-decider model, a Hugging Face Hub id. Any Qwen3.5 model with a pointer head works; the Gemma 4 models do not. It names the stack. |
| `model_revision` | `main` | The model's revision. Use a release's date tag (`YYMMDD`) to pin it ([docs/inference.md, Model artifact](../../docs/inference.md#model-artifact)). |
| `snapstart` | `true` | `false` turns SnapStart off ([What it costs](#what-it-costs)) |
| `strict_window` | `false` | `true` refuses a prompt over the window with a 422, as `serve --strict-window` does |
| `max_request_tokens` | `24576` | The most tokens one request may forward, the state once plus every question; `0` for no limit ([Limits](#limits)) |

## What it costs

At us-east-1 prices, a request costs $0.000133 per second it runs (10 GB of arm64 at
$0.0000133334 per GB-second): $0.0002 to $0.0003 for three questions about a short state.
CloudWatch Logs adds $0.50 per GB ingested, and a request writes a few hundred bytes.

SnapStart decides the rest:

| | `snapstart=true` (default) | `snapstart=false` |
| --- | --- | --- |
| First request on a new execution environment | about 6 s | up to 30 s |
| Each new execution environment | $0.0014 to restore the snapshot | about $0.002, the 11 to 23 s the model takes to load |
| Nobody calling | **about $39.50 a month** for the snapshot, and the image | **about $0.25 a month**, the image in ECR |

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
Without it, 15 new environments loaded the model in 11 to 23 s, and their first requests took
up to 30 s.

Compared with the torch engine in fp32 on CPU, over 20 requests that cover every question type,
structured and Unicode states, 40, 64 and 300 questions and a 3,800-token state, the function
moved probabilities and confidences by at most 0.026 and changed no choice; one yes/no answer
went from 0.495 to 0.505. The quantization to Q8_0 accounts for most of that. Requests that
`serve` refuses, this function refuses too, with the same status.

## Limits

- The state and the longest question must fit the model's window (`max_length`, 4,096 tokens for
  the v1 release). As with `serve`, a question may take at most three quarters of it, and the state
  is trimmed from the end to fit the rest; a question trimmed into its options is refused with a
  422. With `strict_window=true`, a prompt over the window is refused with a 422 instead.
- A request may forward at most `max_request_tokens` tokens, the state once plus every question.
  The default, 24,576, stays well inside the 15-minute timeout; a larger request is refused with
  a 422 before any work is done.
- No images: a request with `images` gets a 422.
- A Function URL request or response is at most 6 MB.
- One request at a time per execution environment. Lambda scales by adding environments.

## Deleting it

```bash
npx cdk destroy                    # the default model
npx cdk destroy -c model=<Hub id>  # any other model you deployed
```

This deletes everything that model's `cdk deploy` created: the function, its versions and their
snapshots, the alias and Function URL, the log group, the ECR repository with the image, the
CodeBuild project and the roles. The log groups that CodeBuild and deploy-time-build's trigger
function write to stay, with the build logs: `/aws/codebuild/ContainerImageBuildArm64*` and
`/aws/lambda/*-DeployTimeBuildCustomRes-*`, at most a few hundred KB each.
