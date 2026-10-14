# strands decider on AWS Lambda

**Serverless decider inference: no GPU, no instance to reserve, no server to keep running.** This
recipe serves the 2B Qwen3.5 decider from an AWS Lambda function on plain arm64 CPUs. Lambda
starts execution environments when requests arrive, runs as many side by side as the traffic
needs, and drops them when it stops. You pay per request, and the only things that persist are
the function and a 2 GB model file in S3.

Under the hood the torso runs through [llama.cpp](https://github.com/ggml-org/llama.cpp) on a
Q8_0 GGUF, the pointer head runs in numpy, and the prompts and readouts come from the package's
own `prompting.py` and `schema.py`, so neither torch nor a GPU is involved. The API is
`strands-decider serve`'s `/v1/systemone` and `/health`, behind an IAM-authenticated Function URL.

A typical request (a state of about 500 tokens and three questions) takes 4.2 s on a Graviton3
host and 6.7 s on a Graviton2 host, about $0.0006 to $0.0009 at Lambda's arm64 price. Lambda
picks the CPU generation, not you, so expect a mix.

## What it costs

At us-east-1 prices for a 10 GB arm64 function ($0.0000133334 per GB-second):

| | SnapStart on (default) | SnapStart off |
| --- | --- | --- |
| Each request | about $0.00013 per second of run time (table below) | the same |
| Each new execution environment | $0.0014 to restore the snapshot | about $0.004, the 30 s model load |
| Standing cost, nobody calling | **about $39.50 a month** for the snapshot cache, plus S3 | **S3 only, about $0.05 a month** |
| First request on a new environment | **5.6 s** | **31 s** |

SnapStart is what makes a new environment answer in seconds: Lambda snapshots an environment
with the model already loaded and restores new ones from it. For Python functions Lambda bills
that snapshot's cache, $0.0000015046 per GB-second of configured memory, for as long as the
version exists. That is about $1.30 a day at 10 GB, whether or not anyone calls the function.
The snapshots of Python functions do not expire. Every deploy publishes a new version with its
own snapshot, so `deploy.sh` deletes the versions the alias has moved off; the alias is all the
Function URL serves. Delete the stack when you are done with it, and the cache charges stop.

With `SNAPSTART=off` there is no standing cost beyond the model in S3, and each new
environment downloads and loads the model before its first answer, about 30 s. Environments
stay warm for a while between requests, so steady traffic pays that only when Lambda scales out.

Requests are billed at Lambda's duration price for the whole 10 GB, whatever the request uses.
CloudWatch Logs adds $0.50 per GB ingested, and a request writes a few lines.

## Requirements

- The package installed with its torch dependencies (`pip install -e .`), for `prepare_model.py`
  and llama.cpp's GGUF converter. A CPU is enough.
- [finch](https://github.com/runfinch/finch) or Docker able to run `linux/arm64` containers
  (`CONTAINER=docker` for Docker). An Apple-silicon Mac runs them natively.
- The AWS CLI, credentials that can deploy a CloudFormation stack with an IAM role, and an S3
  bucket in the deploy region.
- boto3 for `client.py` (`pip install boto3`).

## Usage

```bash
examples/lambda/convert_model.sh                 # checkpoint -> merged fp16 torso, head, f16 GGUF
examples/lambda/build.sh                         # f16 GGUF -> Q8_0, and build/function.zip
BUCKET=my-bucket examples/lambda/deploy.sh       # upload both, deploy the stack
python examples/lambda/client.py health
python examples/lambda/client.py ask
```

No weights live in the repo: `convert_model.sh` downloads the checkpoint and writes every
derived file under the git-ignored `build/`, about 10 GB on disk. Merging the checkpoint peaks at
about 10-12 GB of RAM, so stop other heavy processes, such as a container VM, while it runs.
`convert_model.sh` passes its arguments to `prepare_model.py`, which takes a checkpoint path or
Hub id; the default is `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`. Any Qwen3.5 torso
with a pointer head works. `local_test.sh` runs the built zip once in an arm64 container, with
the GGUF from `build/` instead of S3.

`deploy.sh` reads `SNAPSTART` (`on` or `off`, default `on`; see [What it costs](#what-it-costs)),
`STACK` (default `strands-decider-lambda`), `PREFIX` for the S3 keys, and the AWS CLI's usual
`AWS_PROFILE` and `AWS_REGION`. It skips the 2 GB model upload when the object is already in S3.
To remove everything, delete the stack and the `PREFIX/` keys in the bucket.

The Function URL accepts only SigV4-signed requests from principals with
`lambda:InvokeFunctionUrl`, and `client.py` signs with boto3's credentials. Changing `AuthType` to
`NONE` in `template.yaml` makes the function public.

## How it works

| File | Role |
| --- | --- |
| `prepare_model.py` | Folds the LoRA into the base weights, saves the torso as a plain fp16 `Qwen3_5ForCausalLM`, and writes the head weights, temperatures and tokenizer to `build/model/assets/` |
| `convert_model.sh` | Fetches llama.cpp at a pinned commit, runs `prepare_model.py` and converts the merged torso to an f16 GGUF |
| `build.sh`, `container_build.sh` | Quantize the f16 GGUF to Q8_0, then build the libraries, the shim and the Python dependencies in `amazonlinux:2023` (arm64) and zip them |
| `shim/decider_hidden.c` | A small C API over libllama that returns the final hidden states for exact token ids. `dh_forward_shared` decodes the state once and copies its memory to each question |
| `app/engine.py` | Renders the prompts with `strands_decider.prompting`, tokenizes, calls the shim and runs the pointer head |
| `app/loader.py` | Streams the GGUF from S3 into a memfd. SnapStart limits `/tmp` to 512 MB |
| `app/main.py`, `app/run.sh` | The FastAPI app, started by the Lambda Web Adapter. With SnapStart the model load and a warm-up run at import, so the snapshot holds them; without it they run in a background thread, because Lambda gives an on-demand init only 10 s and re-runs an init that overruns inside the first request |
| `template.yaml`, `deploy.sh`, `client.py` | The SAM template (arm64, 10 GB, SnapStart, IAM Function URL), the deploy and a SigV4 client |

The build enforces four things, each found by a failed deploy or a failed test:

1. No i8mm. Lambda runs arm64 functions on Graviton2, Graviton3 and Graviton4, and a SnapStart
   snapshot taken on one can restore on another. An i8mm (`smmla`) instruction kills the process
   on Graviton2, so llama.cpp is built for `armv8.2-a+fp16+dotprod`, and the build fails if
   `libggml-cpu.so` contains `smmla`.
2. `libgomp`, `libstdc++` and `libgcc_s` ship in the zip. The python3.12 runtime has no libgomp, and
   ggml's CPU backend uses OpenMP. The build fails if `ldd` cannot resolve a library from the bundle.
3. The micro-batch is capped at 512 tokens (`DH_UBATCH`). An uncapped micro-batch grows the compute
   buffers with the prompt, and a 4,096-token state ran out of 10 GB.
4. No token is an output unless its hidden state is read. llama.cpp keeps a row of vocabulary
   logits, about 1 MB here, for every output token, and asking for embeddings makes every token
   an output. So the shim decodes the state with embeddings off, and the questions at most 512
   tokens per decode. Before this, a 3,900-token state with 64 questions ran out of 10 GB, and
   the longest states took a third longer.

The state is decoded once per request, so a request costs about the state's tokens plus 65 to 90
per question. Asking several questions in one request is much cheaper than separate requests.
Questions go through in groups of 32 (`DH_GROUP`) against the one decoded state. If
tokenizing the state and a question separately gives different ids from tokenizing them
together, the engine falls back to one full forward per question and reports
`"state_shared": false`.

## Measurements

Measured on `strands-decider-2B-qwen3.5-v1-2610`: a 10 GB function, three questions per
request, the median of 16 warm requests per row.

| Tokens per request | Graviton2 | Graviton3 | Cost (Graviton2) |
| --- | --- | --- | --- |
| 196 | 2.3 s | 1.5 s | $0.0003 |
| 656 | 6.7 s | 4.2 s | $0.0009 |
| 2,134 | 22.4 s | 13.8 s | $0.0030 |
| 3,964 | 44.3 s | 27.3 s | $0.0059 |

Peak memory stayed at or below 6.1 GB, including a 3,900-token state with 64 questions and a
request with 300 questions. A new environment answers its first request in 5.6 s with SnapStart
(the median of 10, from a 3.2 to 4.5 s restore) and 31 s without (the median of 10).

Compared with the torch engine in fp32 on CPU, over 17 requests that cover every question type,
structured states, 255 options and a 3,800-token state, Q8_0 moved option probabilities by at
most 0.017 and the derived confidences by at most 0.026, and changed no choice. The f16 GGUF,
on six of those requests, stayed within 0.0095, so most of that is the quantization. Q4_0 moved noul and choice scores by
up to 0.05, so this recipe does not use it.

## Limits

- The state and the longest question must fit the checkpoint's window (`max_length`, 4,096 tokens
  for the v1 release), about 3,900 tokens of state with short questions. A longer prompt gets a 422
  that says how many tokens to cut, rather than being truncated.
- A request may forward at most `MAX_REQUEST_TOKENS` tokens, the state once plus every
  question (default 24,576, set in `template.yaml`; 0 turns it off). That is about 5 minutes on a
  Graviton3 host and 8 on a Graviton2 host, inside the function's 15-minute timeout. A larger
  request gets a 422 with its token count, before any work is done.
- Text only. Requests with `images` get a 422.
- One request at a time per execution environment. Lambda scales by adding environments.
