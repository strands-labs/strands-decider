# Inference

This document describes how to run a checkpoint: install, get a checkpoint, ask it questions
from the command line, and serve it over HTTP. Strands decider answers typed questions about a
state, the text to classify; it does not generate text. A question is a `noul` (yes or no,
returned as P(true)), a `choice` (one of N options) or a `score` (a level on an ordered
scale). Install the package, point the commands at the published Hub id or at a checkpoint you
trained, then `strands-decider ask` or `strands-decider serve`. Pass `--device cuda`, `mps`, `cpu` or `mlx`
explicitly; without it the CLI picks cuda, then mps, then cpu, and mlx only when asked for. A
checkpoint from `training/recipe.sh all` is already calibrated; calibrate any other
checkpoint with `python -m training.evaluate calibrate` before you serve it.

`eval` and `calibrate` also accept `--device cuda`, `mps`, or `cpu` and use the same
cuda > mps > cpu selection when omitted. They use torch; MLX is an `ask`/`serve` backend only. They
prepare the model as `serve` does on that device (the MPS and CPU kernels, an fp32 torso on
CPU), so they report the numbers serving returns. `--device mlx` is refused.

- [Environments for serving](#environments-for-serving): Linux with CUDA, a Mac, or CPU.
- [Model artifact](#model-artifact): what a checkpoint holds, and the Hugging Face export.
- [Ask](#ask): one command, with recorded v19 output and how to read its confidence.
- [Serve](#serve): `POST /v1/systemone` and `/health`, on `127.0.0.1` with no authentication.
- [Decision checkpoints](#decision-checkpoints): checkpoints of the decision interface
  ([spec.md](spec.md)), with images, on the transformers backend.
- [Asking many questions is nearly free](#asking-many-questions-is-nearly-free): the shared-prefix cache.
- [Serving on a Mac](#serving-on-a-mac): MPS, and the one kernel that had to be replaced.
- [Serving on a Mac with MLX](#serving-on-a-mac-with-mlx): `--device mlx`, measured against MPS, and
  [decision checkpoints on MLX](#decision-checkpoints-on-mlx), images included.
- [`../deploy/`](../deploy/README.md): hosting the server, for example on an Amazon SageMaker
  real-time endpoint.
- [`../examples/strands/`](../examples/strands/README.md): an agent built with the Strands
  Agents SDK that uses the server, with its client in `_client.py`.
- [`../evaluation/results.md`](../evaluation/results.md): measured latency and accuracy on an
  RTX 3090 and on a Mac; [`../evaluation/jevbench.md`](../evaluation/jevbench.md): JevBench,
  the external benchmark.
- [`../docs/architecture.md`](../docs/architecture.md#the-routing-convention): the confidence
  formulas and the routing convention.
- [`../training/README.md`](../training/README.md): how to train a checkpoint.

Paths are relative to the repository root unless they are links. A module path such as
`mps_kernels.py` is relative to `src/strands_decider/`.

## Environments for serving

On Linux, or under WSL2 on Windows, with an NVIDIA GPU, serving uses the training
environment in [Setup](../training/README.md#setup).

A server started inside WSL (`strands-decider serve ... --port 8099`) is reachable from Windows
at `127.0.0.1:8099` for as long as its WSL session is alive.

Each device has an install extra, so a deployment names its device the same way everywhere. The
extras ship with the next release (0.1.0 on PyPI has only `train` and `dev`); until then, install
from a clone, for example `pip install -e ".[mlx]"`.

| `--device` | Install | Adds |
|---|---|---|
| `cuda` | `pip install "strands-decider[cuda]"` | flash-linear-attention (Linux). causal-conv1d compiles against the local CUDA toolkit and is installed separately: `pip install causal-conv1d --no-build-isolation`. |
| `mps` | `pip install "strands-decider[mps]"` | Nothing: the base install. |
| `cpu` | `pip install "strands-decider[cpu]"` | Nothing: the base install. |
| `mlx` | `pip install "strands-decider[mlx]"` | mlx, mlx-lm and, for decision checkpoints, mlx-vlm (Apple silicon only). |

flash-linear-attention installs on a Mac without Triton, but its kernels need Triton, so it
cannot run there. `mps_kernels.py` therefore steps aside only when Triton is present too, and
the `cuda` extra is marked Linux-only.

**Serving on macOS (Apple silicon), inference only.** No `flash-linear-attention`:
Triton has no macOS build. Nothing else changes; pass `--device mps`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 transformers==5.17.0 peft==0.21.0
pip install -e ".[dev]"
strands-decider serve checkpoints/hobson-2b-recipe --device mps --port 8099
```

Transformers' `causal_conv1d_fn` falls back to its reference PyTorch path. On CPU,
`chunk_gated_delta_rule` also falls back. On MPS, `mps_kernels.py` replaces that second
function, so only the first fallback matters there, and it costs 32 ms a forward.

**macOS (Apple silicon) with MLX.** The `mlx` extra adds mlx and mlx-lm, and `--device mlx`
runs the torso on Metal through mlx-lm ([Serving on a Mac with MLX](#serving-on-a-mac-with-mlx)).
MLX is opt-in: with the extra installed, a command without `--device` still runs on MPS.

```bash
pip install -e ".[dev,mlx]"
strands-decider serve checkpoints/hobson-2b-recipe --device mlx --port 8099
```

**CPU only.** The same commands as on macOS, with `--device cpu`. On Linux without a GPU,
`pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu` skips the CUDA
wheels. `chunk_gated_delta_rule` falls back to its reference path, and inference is much
slower than on a GPU. `causal_conv1d_fn` does not: on CPU, `cpu_kernels.py` replaces it with
four shifted multiply-adds over channels-last rows, with the same answers to float rounding.
Torch builds without oneDNN, such as the macOS wheels, run the reference as one convolution
per channel: 2.0 of 2.7 s of a 70-token v19 forward on an M3 Pro, against 8 ms. Linux x86
wheels have oneDNN, and the reference is fast there for short states. On a Xeon 8375C
(8 threads), the conv takes 35 ms of a 1.6 s v21 ask either way, and on a 3,000-token state
it takes 0.49 s instead of 1.18 s.

## Model artifact

The reference weights are published as
[`StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610)
([naming.md](naming.md) explains the name). The earlier releases stay at
[`StrandsAgents/strands-decider-2B-hobson-v21`](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21)
(the same files also at [`amazon/strands-decider-2B-hobson-v21`](https://huggingface.co/amazon/strands-decider-2B-hobson-v21))
and [`StrandsAgents/strands-decider-2B-hobson-v19`](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19).
The Gemma 4 models are
[`StrandsAgents/strands-decider-E2B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v1-2610),
[`-E4B-`](https://huggingface.co/StrandsAgents/strands-decider-E4B-gemma4-v1-2610),
[`-12B-`](https://huggingface.co/StrandsAgents/strands-decider-12B-gemma4-v1-2610) and
[`-26B-A4B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610)
([Gemma 4 models](#gemma-4-models)).
Every command that takes a checkpoint path also takes that id, and the loader downloads the
repository's `main` to the Hub cache. Every upload to a release repository is tagged with
its date (`YYMMDD`), and a second release in the same month updates `main` of the same
repository. To pin one, download it by its tag and pass the folder:
`hf download StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --revision YYMMDD --local-dir v1`,
then `strands-decider serve v1`. You can equally train your own with the recipe
([training/README.md](../training/README.md#usage)); `recipe.sh` writes it to
`checkpoints/hobson-2b-recipe`.

A checkpoint directory holds these files. `StrandsDeciderModel.save_pretrained` in `modeling.py`
writes them, `python -m training.evaluate calibrate` adds the temperatures, and `StrandsDeciderModel.load` reads them.

| File | Contents |
| --- | --- |
| `strands_decider_config.json` | The model configuration: the base model name, the readout type and size, the window (`max_length`) and the fitted temperatures. A checkpoint saved before the rename, such as the published v19, holds `hobson_config.json` instead; `load` reads either |
| `lora/` | The LoRA adapter (`adapter_config.json`, `adapter_model.safetensors`) |
| `slot_head.pt` or `head.safetensors` | The readout weights. `save_pretrained` writes `slot_head.pt`. An export from `training.hf_export` holds `head.safetensors` instead. |
| `tokenizer.json`, `tokenizer_config.json` | The tokenizer |
| `chat_template.jinja` | The chat template, saved with the tokenizer. Inference does not read it. |

Training also writes `train_config.json` and `history.json`. Inference does not read them.

The base weights are not in the checkpoint. At first use, `load` downloads
`Qwen/Qwen3.5-2B-Base` (about 4.5 GB) from Hugging Face, at the revision in the config's
`base_revision`. A checkpoint without one uses the `base_model_revision` that its
`provenance.json` records for the same base model, so a Hugging Face export loads the base it
was trained on. With neither, the loader gets the current `main` of that repository.

`python -m training.hf_export export CKPT OUT --run-id RUN` writes a checkpoint as a Hugging
Face-format folder. Without `--run-id`, the exporter takes the run id from a
`results/<run-id>/` part of the CKPT path, and it stops if the path has none. The folder
holds the readout as `head.safetensors`, so it has no pickle files. It also holds the PEFT
adapter, the tokenizer and configuration files, and a model card (`README.md`).
`provenance.json`, `LICENSE.md` with the licence status and `MANIFEST.sha256` complete it.
`python -m training.hf_export verify DIR` checks a folder. `StrandsDeciderModel.load` reads the
folder as it reads a checkpoint, and the base weights still download separately from
Hugging Face. Where a command takes a checkpoint path (`strands-decider serve`, `strands-decider ask`,
`strands-decider info`), a Hub model repo id such as `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` also works: the loader
downloads the repo to the Hub cache. `python -m training.evaluate calibrate` writes into the checkpoint, so it
needs a local directory. OUT can be a local directory or an `s3://` URI. The exporter does not upload to
the Hugging Face Hub.

## Ask

There are three question types, and all three are one mechanism read back three ways.

- A **noul** question is a yes/no question. It has two options and returns the
  probability of "true".
- A **choice** question has any number of options. It returns the top option, a
  probability for each option, and a confidence: the top probability, rescaled so that an
  even split gives 0 and a certain answer gives 1, whatever the number of options.
- A **score** question has ordered levels. It returns an expected value on that scale,
  the distribution, and a confidence.

On the command line, a choice or a score is written as `question=option,option,...`. The
confidence is computed from the distribution, not predicted.

The output below was recorded from a v19 checkpoint, which the v19 reference recipe
trains. The command below names `checkpoints/hobson-2b-recipe`, the path
`training/recipe.sh` (or `bash training/recipe.sh`) writes; pass
`StrandsAgents/strands-decider-2B-hobson-v19` instead to run the published weights. On a Mac,
add `--device mps`. A retrained checkpoint gives somewhat different numbers with the same
answers: a later v19 retrain gives 0.809, `technical` at 0.783 and a score confidence
of 0.472 on CPU.

```bash
strands-decider ask checkpoints/hobson-2b-recipe \
  --state "Help! My payouts have been failing for 3 days." \
  --noul "Does this convey urgency?" \
  --choice "Which team should handle this?=billing,technical,sales" \
  --score "How frustrated is the writer?=calm,frustrated,very angry"
```

```
noul_0 noul = 0.801
choice_0 -> technical (confidence 0.622)
    technical                0.748
    billing                  0.234
    sales                    0.018
score_0 score = 1.24 (confidence 0.578)
    0: calm                                     0.093
    1: frustrated                               0.578
    2: very angry                               0.330
```

That is real output from v19, not an illustration — and it is
worth reading the confidence, not just the answer. Whether failing payouts are
`billing` or `technical` is arguable, and the model leans but says so: 0.622 is in the
"confirm" band of the [routing convention](../docs/architecture.md#the-routing-convention), well short
of the 0.9 at which that convention acts without a check.

## Serve

```bash
strands-decider serve checkpoints/hobson-2b-recipe --port 8000
```

`POST /v1/systemone` takes a state and typed questions. JevBench's `typesafe` adapter
targets this endpoint and runs against it unmodified: 231 of 231 tasks attempted and
schema validity 1.000 ([External benchmark](../evaluation/jevbench.md#external-benchmark-jevbench-v1-public-set)).
Compatibility with the Jev API itself is not verified. `GET /health` returns the model
name, the checkpoint path, the base model, the device and the calibration temperature.
`GET /ping` and `POST /invocations` are the same as `/health` and `/v1/systemone`: they are
the paths that SageMaker and Bedrock AgentCore Runtime call
([../deploy/sagemaker/](../deploy/sagemaker/README.md)). `/health` and `/ping` answer at once
even while every worker thread is busy with requests, as a load balancer's health check
expects (SageMaker allows `/ping` 2 s).

The server binds to `127.0.0.1` by default and has no authentication. Its behaviour under
concurrent requests is not verified. Use it for local experiments only
([Security](../README.md#security)).

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "hobson-latest",
  "questions": {
    "is_urgent":  {"type": "noul",   "instructions": "Does this convey urgency?"},
    "department": {"type": "choice", "instructions": "Route it.",
                   "criteria": {"billing": "money", "technical": "bugs", "sales": "pricing"}},
    "frustration":{"type": "score",  "instructions": "How frustrated?",
                   "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}'
```

A hybrid torso such as v14's shares the state across questions like any other; a single
question is encoded in one pass ([Asking many questions is nearly free](#asking-many-questions-is-nearly-free)).

The `state` may be empty when the question carries the whole task. An option description
may be a string, structured data (rendered as JSON, like a structured `state`), or `null`
for a bare label.

A prompt longer than the checkpoint's window is refused with HTTP 422 and a message that
names the context window. `--max-batch N` (default 32) sets how many questions one forward pass encodes;
lower it when a very long state with many questions does not fit in GPU memory.
`--max-batch-tokens N` also caps the tokens of one forward, as the forward pads them: its
questions times the longest state + question, after each is fitted to the window. A forward
then ends early, before it would pass N tokens, and always holds at least one question.
Questions are fitted before the cut and scored independently, so answers change only by
float rounding. Long-context torsos (Gemma 4 at a 32,768-token window) need it; unset, only
`--max-batch` applies.

`--device-map auto` splits a model larger than one GPU across the visible GPUs, with the
transformers `device_map` of the same name; the head runs on `cuda:0`. The layers run one GPU
after the other, so the split adds memory, not speed. The 26B-A4B serves this way on four
48 GB L40S GPUs (16 to 21 GB on each), at about 200 ms for one short question and 370 ms for
two, with probabilities within 0.03 of the model on CPU. CUDA only, and not with `--vision`.

The server warms up before it accepts requests, so no request waits for a kernel to compile:
it asks each question kind once, then every number of questions up to `--max-batch` (larger
requests run in chunks of at most that many), and with `--vision` one image request. `/health`
and `/ping` fail until it is done, so
startup and health-check timeouts must allow for it. `--no-warmup` skips it.

## Gemma 4 models

The four Gemma 4 models serve with the same commands as the Qwen3.5 model. They need the code on
`main` (`pip install "strands-decider @ git+https://github.com/strands-labs/strands-decider"`); the
package on PyPI does not have Gemma 4 support yet. They read text only, so `--vision` does not apply.

**E2B uses eager attention on CUDA.** The loader selects it for the E2B model on every device.
On CUDA, PyTorch's default attention kernel gives wrong results for E2B (one key/value head) when a
forward continues a cached state with 65, 129, 193 or another 64k+1 number of tokens. That happens
when a request asks several questions; eager attention is correct there. The other sizes keep the default
attention.

**Memory.** The loader keeps only the text decoder: the vision and audio encoders of the base model
are dropped.

| model | GPU, weights in bf16 | CPU, weights in fp32 |
| --- | --- | --- |
| E2B | about 9 GB | about 19 GB |
| E4B | about 15 GB | about 30 GB |
| 12B | about 23 GB | about 46 GB |
| 26B-A4B | about 51 GB | about 101 GB |

Activations come on top and grow with the window and with the questions per forward pass. The
models are trained and served at a 4,096-token window. At a 32,768-token window, use
`--max-batch 4 --max-batch-tokens 34000`: with these settings the 26B-A4B needed at most 57 GiB on one
80 GB H100 ([Serve](#serve)).

**CPU and GPU.** On CPU the torso runs in fp32. On 50 JevBench tasks the CPU gives the same answer as
the GPU on every task, with probabilities within 0.05 (0.13 for the 26B-A4B). CPU is much slower: with
16 threads per model, and all four models running at once on one 64-vCPU host, a JevBench question
took 4.9 s (E2B), 7.2 s (E4B), 13.5 s (12B) and 7.8 s (26B-A4B). The 26B-A4B is faster than the 12B
there, because it computes with 4B of its parameters per token. On one H100, the median is
102 ms, 139 ms, 136 ms and 245 ms.

## Decision checkpoints

A decision checkpoint follows the decision interface ([spec.md](spec.md)): the base Gemma 4
model's own multimodal checkpoint with the fine-tuning merged in, beside `readout.safetensors`
and a `decision_config.json` that says `"interface": "strands-decider"`. `ask` and `serve` take
one the way they take any checkpoint, a local directory or a Hub id; every other checkpoint
keeps the path described above.

```bash
strands-decider ask path/to/decision-checkpoint --state "A photo sent with a claim." \
  --image photo.jpg --choice "What does the photo show?=a car,a bicycle,a house"
strands-decider serve path/to/decision-checkpoint --port 8000
```

The core library renders each request with the checkpoint's prompt, and the transformers
backend (`backends/transformers.py`) runs the model: every layer under the interface's mask,
the state encoded once when a request asks several questions, the answers read from the hidden
state at each question's last token. CPU runs float32, CUDA and MPS bf16, and `--dtype float32`
or `--dtype bfloat16` chooses; the first line the command prints names the device and dtype. A
request may carry images when the checkpoint's `decision_config.json` lists `image` in
`inputs`; `--vision` is not needed, and warns. Images need the `vision` extra
(`pip install "strands-decider[vision]"`, Pillow and torchvision; the 12B's image processor
exists only for torchvision). A request whose `audio` field is anything but null is refused
with 422, since the interface takes no audio; other fields the interface does not define are
ignored, as v1 ignores them. `--no-prefix-cache` runs one full pass per question (the spec's
reference on CUDA in bf16); `--max-batch` and `--max-batch-tokens` bound the questions of one
pass over a state. The server answers `--max-concurrency` requests at a time on the model
(default 1), so a burst of long or image requests cannot hold several states on the GPU
together; the others wait up to `--queue-timeout` seconds (default 60) and then get 503. A
refused request is answered at once, without waiting. `--revision` names the branch, tag or
commit of a checkpoint on the Hub. `/health` reports the backend, the device, the dtype,
whether images are served and the concurrency. `--device-map` does not apply, and `--dtype`,
`--max-concurrency` and `--revision` apply to decision checkpoints only.

A checkpoint from before the interface, whose `decision_config.json` has no `interface`, names
no prompt, so `ask` and `serve` refuse it; `TransformersBackend` runs it on token ids rendered
elsewhere, as text only.

## Asking many questions is nearly free

*The latency table below was measured on v7, a Qwen3 torso. The cache now also forks the
recurrent state of the hybrid Qwen3.5 torso
([Hybrid torsos share the prefix too](../docs/architecture.md#design-decisions-worth-knowing)).
The v14 figures at the end of this section use it.*

The state is encoded **once** and its KV cache is broadcast across the question batch;
only the short question suffixes are forwarded per question.

```
naive:  N x (state + question)     tokens
strands-decider: state + N x question       tokens
```

Measured against the served 1.7B checkpoint, one ~640-token state:

| questions | latency | input tokens |
| --- | --- | --- |
| 1 | 219.4 ms | 666 |
| 6 | 217.4 ms | 966 |

Five extra questions cost roughly nothing. The naive path would have encoded ~4,000
tokens instead of 966. The questions genuinely evaluate independently — no question
can see another's text — which is what keeps the answers decomposable.

`--no-prefix-cache` falls back to plain batched encoding. The two paths are
*numerically identical in fp32* (pinned by `test_prefix_cache_is_exact_in_fp32`, and for
hybrid torsos by `tests/test_prefix_cache.py`); in bf16 they differ by ~2e-3 from
accumulation order alone, far below any threshold worth routing on. On v14 on the RTX 3090, eight
questions over a ~2,000-token state take 369 ms through the shared prefix against
1,964 ms batched, and sixteen take 445 ms against 4,086 ms. A request with one question
skips the prefix path: it would pay a second forward for nothing.

## Serving on a Mac

Measured with v19 on an M3 Pro (36 GB) running macOS 26.6: torch 2.7.1, transformers
5.17.0, bf16 on MPS.
**The answers match the RTX 3090 run, but the latency does not.** Under 300 tokens, a v19
question takes a median 153 ms warm and 310 ms on the first request of its length. Across
JevBench the warm median is 234 ms and the 95th percentile is 2,628 ms. These figures are
for v19 on this Mac only. They are not a latency claim for other hardware or other
checkpoints.

The accuracy and latency measurements are in
[Serving on a Mac: accuracy and latency](../evaluation/results.md#serving-on-a-mac-accuracy-and-latency).

**One kernel had to be replaced.** Without `fla`, the reference chunk rule for the 18
Gated DeltaNet layers was 806 of 1,739 ms of a 1,024-token forward. Nearly all of it
was `torch.linalg.solve_triangular`, about 20 ms a call on MPS. `mps_kernels.py` inverts
the same unit-lower-triangular system by recursive block inversion, which is six levels
of batched matmuls, and applies the inverse to both right-hand sides. The engine
installs it for MPS only, and the forward is 1.7x faster (1,024 tokens: 1,750 to
1,029 ms). The obvious inverse, the finite Neumann product (I-N)(I+N²)(I+N⁴)..., passed
random-input tests and was wrong by up to 5e8 on real activations. Correlated,
l2-normalised keys make the powers of N huge before they cancel.
`tests/test_mps_kernels.py` builds its inputs that way and fails 14 cases against that
version.

## Serving on a Mac with MLX

`--device mlx` runs the torso on Apple silicon's GPU through mlx-lm
([`mlx_engine.py`](../src/strands_decider/mlx_engine.py)). mlx-lm implements Qwen3.5 with fused
Metal kernels, Gated DeltaNet included, where MPS dispatches every op separately and has none
of the CUDA kernels. Nothing else moves: `MLXEngine` subclasses `SystemOneEngine`, so prompt
rendering, tokenisation, truncation, option positions, both evaluation paths, the
temperatures and the fp32 head are the torch engine's own code. The head runs on the CPU, as
torch.

Measured with v19 on an M4 Pro, one question takes 113 ms at 222 input tokens, 486 ms at 1,118
and 1,764 ms at 4,094, against 162, 682 and 2,685 ms on MPS. Sixteen questions on a 1,024-token
state take 1,060 ms against 1,622 ms. The answers are the same: on 54 answers compared with v19
in fp32 on the CPU, none changes on either device, and the largest probability difference is
0.0138 on MLX and 0.0051 on MPS. Most of MLX's difference comes from merging the LoRA adapter
into the bf16 weights at load, where torch keeps it unmerged; with the torso in fp32 it is
0.0036. On a tiny random Qwen3.5, `tests/test_mlx_engine.py` finds the torch engine's answers to
four decimal places, on Metal and on MLX's CPU backend. The measurements, and the commands that reproduce them, are in
[Serving on a Mac through MLX: accuracy and latency](../evaluation/results.md#serving-on-a-mac-through-mlx-accuracy-and-latency).

Limits. mlx-lm is pinned to the minor version tested, because the engine uses its Qwen3.5
module layout and prompt-cache classes. The adapter merge refuses DoRA, `modules_to_save`,
trainable token embeddings, per-module ranks and quantised base weights. `--dtype` in
`bench_local.py` applies to torch devices only. The engine caps MLX's buffer cache at 1 GiB
(`mlx_engine.DEFAULT_CACHE_LIMIT`); MLX's own default is its memory limit, nearly all of RAM.
Evaluations run one at a time, because `_fit` keeps a request's option offsets on the engine for
`_option_idx` to read; the torch engine has the same race (#9).

### Decision checkpoints on MLX

A decision checkpoint ([`spec.md`](spec.md): the Gemma 4 base's multimodal checkpoint with its
fine-tuning merged in, beside a `readout.safetensors` and a `decision_config.json`) runs on
`--device mlx` through mlx-vlm ([`backends/mlx.py`](../src/strands_decider/backends/mlx.py)).
`ask` and `serve` tell it from a v1 checkpoint by its `decision_config.json`, and load it the
same way. It takes images whenever the checkpoint's `inputs` hold `image`; `--vision` is not
needed.

```bash
strands-decider ask CHECKPOINT --device mlx --state "A photo from the support chat." \
  --image photo.jpg --choice "What does it show?=a person,an animal,food or drink"
strands-decider serve CHECKPOINT --device mlx --port 8099
```

The backend implements the core's `Backend` (spec section 11) over mlx-vlm's own Gemma 4
classes, unpatched:

- Every pass hands mlx-vlm a mask, which it gives every layer as it is, so `prefix_bidir` holds
  on every layer with no sliding window. The state is encoded once, into plain KV caches, and
  each question runs over it; `--no-prefix-cache` runs each question as its own full pass under
  the dense mask instead.
- On E2B and E4B, whose last layers read an earlier layer's keys and values, a question carries
  only its last token through those layers, which is exact.
- An image is decoded as v1 decodes it, resized with Pillow's bicubic filter to the spec's size,
  prepared by the checkpoint's image processor with its resize off, and encoded in bf16; its
  vectors replace the placeholders at its span in the state.
- The backend hands the core the final-norm hidden state, and the core's `Readout` applies the
  readout in the spec's precision regime.

`strands-decider convert-mlx CHECKPOINT OUT [--bits 8|4]` writes an MLX artifact, bf16 without
`--bits` ([`backends/mlx_convert.py`](../src/strands_decider/backends/mlx_convert.py)), which
`ask` and `serve` load like the checkpoint. Only the decoder is quantized: the vision and audio
encoders and their projections stay bf16, and `readout.safetensors`, `decision_config.json`,
`processor_config.json` and the tokenizer are copied byte for byte. The artifact is written
beside `OUT`, reloaded and checked, and only then renamed to `OUT`. `mlx_vlm.convert` alone
would lose the readout and quantize the audio projection. 8-bit is the quantized format
measured against the reference (below); a 4-bit artifact makes no conformance claim, and
`convert-mlx` and loading it both say so.

Measured on an M4 Pro (48 GB) with an intermediate Strands Decider v2 E2B checkpoint in the
multimodal layout, against reference outputs computed with transformers in bf16 on a GPU, one
masked pass per prompt:

- Text. On the reference's 1,200 golden prompts (200 to 7,987 tokens, 300 in each of four length
  buckets, given as the reference's own token ids), bf16 agrees on the top code for 99.25% of
  questions (98.67% to 99.67% per bucket), with mean |Δp| 0.0065 and at most 0.086. Every
  disagreement is a near-tie, with the reference's top two at most 0.084 apart. The 8-bit
  artifact agrees on 98.83% (98.00% to 99.67% per bucket), with mean |Δp| 0.0099 and at most
  0.225; 11 of its 14 disagreements are within 0.02 in the reference, and the widest is 0.107.
- Images. On 24 requests over six photographs (an astronaut's portrait, a greyscale photograph
  of a cameraman, a cat, a cup of coffee, a deep-field telescope image and a rocket launch; one
  to four images per request, an empty state, and a 1,300-token state), bf16 agrees with the
  reference, whose pixels come from transformers' `Gemma4ImageProcessorPil`, on 24 of 24 top
  codes, with mean |Δp| 0.0021 and at most 0.024. The 8-bit artifact agrees on 24 of 24 too,
  with mean |Δp| 0.0043.
- Latency, the state encoded and then one question: a median of 78 ms at about 270 tokens,
  231 ms at 1,100, 968 ms at 3,900 and 2,471 ms at 7,400 in bf16, and 70, 233, 1,010 and
  2,485 ms in 8-bit. A one-image request of 350 tokens takes 300 ms, against 90 ms for the same
  prompt with the image's placeholders left as they are; requests with two, three and four
  images (630 to 1,140 tokens) take 567, 841 and 1,099 ms. `ask` with one image, from the shell
  to its printed answer, model load included, takes 6.0 s on the 8-bit artifact.
- Memory. bf16 holds 8.9 GiB (the audio encoder is not loaded) and peaks at 9.3 GiB at 270
  tokens and 11.8 GiB at 7,400; the 8-bit artifact holds 4.9 GiB and peaks at 5.3 and 7.7 GiB.
  `convert-mlx --bits 8` wrote the 5.9 GB artifact in 11 s, with a peak footprint of 13.1 GB.

The 12B, 26B-A4B and 31B load strictly, and convert to 8 bits (and the 26B-A4B to 4), with the
same code; their answers on a Mac are not yet measured.

Limits. mlx-vlm is pinned to the release tested, because the backend relies on its Gemma 4
internals. 4-bit makes no conformance claim until it is measured (spec section 10): a 4-bit E2B
quantized with the same affine quantizer has agreed on only 88.9% of top codes. A request
waits for at most 16 others on the backend's worker, and is refused after 300 s without a
place; images are decoded two requests at a time. Each request encodes its state afresh: there is no cache across
requests. Audio is not served.
