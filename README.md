<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://strandsagents.com/latest/assets/wordmark-github-dark.svg">
        <img src="https://strandsagents.com/latest/assets/wordmark-github-light.svg" alt="Strands" width="320">
      </picture>
    </a>
  </div>

  <img src="docs/assets/strands-decider-logo.png" alt="Strands Decider" width="84" style="image-rendering: pixelated">

  <h1>Strands Decider</h1>

  <h2>A small, fast decision model for agentic AI.</h2>

  <div align="center">
    <a href="https://github.com/strands-labs/strands-decider/graphs/commit-activity"><img src="https://img.shields.io/github/commit-activity/m/strands-labs/strands-decider" alt="Commit Activity"></a>
    <a href="https://github.com/strands-labs/strands-decider/issues"><img src="https://img.shields.io/github/issues/strands-labs/strands-decider" alt="Open Issues"></a>
    <a href="https://github.com/strands-labs/strands-decider/pulls"><img src="https://img.shields.io/github/issues-pr/strands-labs/strands-decider" alt="Open PRs"></a>
    <a href="https://discord.gg/Wa4CQrxsP"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  </div>

  <p>
    <a href="https://strandsagents.com/docs/labs/strands-decider/overview/"><b>Documentation</b></a>
    &nbsp;•&nbsp;
    <a href="https://pypi.org/project/strands-decider/"><b>PyPI</b></a>
    &nbsp;•&nbsp;
    <a href="https://huggingface.co/StrandsAgents"><b>Hugging Face</b></a>
  </p>

</div>

<hr>

Strands decider is one of a new class of **decision models**, or "system one" models. Unlike an
LLM, which can generate arbitrary text, a decision model picks between sets of options and
rates things on a scale. This class of model works best for problems that fall between LLMs
and traditional classification models: it is a general-purpose classification and scoring
model that responds faster than an LLM and does not take the time and expertise to train
that a traditional classifier does. That makes it a natural fit for the decisions inside
agentic workflows built with the [Strands Agents SDK](https://github.com/strands-agents/sdk-python).

- **Pick from options** — "Are there three r's in strawberry? Yes or no." "What language is
  the phrase 'sihamba ngokushesha' in? English, Zulu, or Dutch."
- **Rate on a scale** — "How positive is 'this is the best doc I've ever read'? Between 0 and 1."
- **Calibrated reliability scores** — every decision carries a confidence. On short
  classification tasks it has never seen, answers at a confidence of 0.9 or more are right
  about 95% of the time ([evaluation/results.md](evaluation/results.md#summary)); below that,
  confirm or ask a person. Frontier LLM inference APIs do not expose anything equivalent.

## How to start using strands-decider

Start on your own machine: install the `strands-decider` CLI, ask the model questions, then run it
as a server and send it HTTP requests. When other people or services need to call it, deploy that
server ([Deploying it](#deploying-it)).

The easiest place to get started is through the `strands-decider` cli:

```bash
pip install strands-decider
```

The examples below use `strands-decider-E2B-gemma4-v2-2611`, the smallest v2 model: it fits on a
24 GB GPU and serves on CPU. The `cuda`, `mps` and `cpu` extras name the devices
([designs/inference.md](designs/inference.md#environments-for-serving)); on Apple silicon,
`--device mlx` runs the v1 Qwen3.5 model 1.4 to 1.6x faster than MPS. Example outputs below are
illustrative; other devices and models differ in the last digits.

### Choice question
You can ask the model to choose based on some state and a question:
```bash
strands-decider ask StrandsAgents/strands-decider-E2B-gemma4-v2-2611 \
  --state "Help! My payouts have been failing for 3 days! " \
  --choice "Which team should handle this?=billing,sales,retail" 
```
Example Output:

```bash
choice_0 -> billing (confidence 0.659)
  billing                  0.773
  sales                    0.121
  retail                   0.106
```

### Noul question
You can also ask the model a Yes/No question:

```bash
strands-decider ask StrandsAgents/strands-decider-E2B-gemma4-v2-2611 \
  --state "Help! My payouts have been failing for 3 days! " \
  --noul "Does this convey urgency?" 
```

Example Output:

```bash
noul_0 noul = 0.858
```
Closer to 1 is leaning more toward Yes

### Score question
Or give a question a score:
```bash
strands-decider ask StrandsAgents/strands-decider-E2B-gemma4-v2-2611 \
  --state "Help! My payouts have been failing for 3 days! " \
  --score "How frustrated is the writer?=calm,frustrated,depressed"
```
Example Output:

```bash
score_0 score = 1.03 (confidence 0.509)
  0: calm                                     0.196
  1: frustrated                               0.574
  2: depressed                                0.230
```

### Multiple question types

You can combine multiple questions into a single command. This is more efficient as you only need to load the state for the model once:

```bash
strands-decider ask StrandsAgents/strands-decider-E2B-gemma4-v2-2611 \
  --state "Help! My payouts have been failing for 3 days! " \
  --choice "Which team should handle this?=billing,sales,retail" \
  --noul "Does this convey urgency?" \
  --score "How frustrated is the writer?=calm,frustrated,depressed"
```
<details>
  <summary>Example Output</summary>

  ```bash
  noul_0 noul = 0.858
  choice_0 -> billing (confidence 0.659)
    billing                  0.773
    sales                    0.121
    retail                   0.106
  score_0 score = 1.03 (confidence 0.509)
    0: calm                                     0.196
    1: frustrated                               0.574
    2: depressed                                0.230
  ```
</details>

### Running as a server

You can also run the model as a server, and ask questions via http requests:
```bash
strands-decider serve StrandsAgents/strands-decider-E2B-gemma4-v2-2611 --port 8000
```

```bash
curl -s localhost:8000/v1/systemone \
  -H 'content-type: application/json' \
  -d '{
    "state": "Help! My payouts have been failing for 3 days!",
    "questions": {
      "is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}
    }
  }'
```

<details>
  <summary>Example Output</summary>

  ```bash
  {
    "model": "strands-decider-E2B-gemma4-v2-2611",
    "answers": {
      "is_urgent": {
        "type": "noul",
        "noul": 0.8581
      }
    },
    "usage": {
      "input_tokens": 86,
      "output_tokens": 1
    },
    "latency_ms": 489.01
  }          
  ```
</details>

### Deploying it

The server you ran above is what you deploy. To host it beyond your machine, see
the [deployment guide](https://strandsagents.com/docs/labs/strands-decider/deployment/local/): it lists where the server can run and links to a guide for
each.

## Model options

v2 is the current version of Strands Decider. Five models on the Gemma 4 instruction-tuned bases
(`google/gemma-4-*-it`, Apache-2.0), from E2B to 31B. They are more accurate than the v1 models at
every size, read a 32,768-token context, and take images. v2 replaces the v1 pointer head with a
**token readout**: one linear map with one row for each of 255 single-token option codes (`A`..`IV`),
read off the hidden state at the last prompt token. Options come from the request, so no retraining
is needed to add a category. One forward pass, no generation, same three question types
(`noul`, `choice`, `score`) ([designs/architecture.md](designs/architecture.md#the-v2-token-readout)).

| model | Decision Index 0.3 | JevBench | OpenDecisions | DecisionPivot | BEV | GPU to use | median latency per JevBench question, one H100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`strands-decider-E2B-gemma4-v2-2611`](https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v2-2611) | 57.1 | 87.4 | 77.7 | 23.3 | 74.0 | 24 GB | 62 ms |
| [`strands-decider-E4B-gemma4-v2-2611`](https://huggingface.co/StrandsAgents/strands-decider-E4B-gemma4-v2-2611) | 63.2 | 88.3 | 83.9 | 45.2 | 78.9 | 40 GB | 78 ms |
| [`strands-decider-12B-gemma4-v2-2611`](https://huggingface.co/StrandsAgents/strands-decider-12B-gemma4-v2-2611) | 64.1 | 92.2 | 85.7 | 58.9 | 79.0 | 48 GB | 80 ms |
| [`strands-decider-26B-A4B-gemma4-v2-2611`](https://huggingface.co/StrandsAgents/strands-decider-26B-A4B-gemma4-v2-2611) | 69.0 | 92.6 | 88.6 | 69.2 | 81.4 | 80 GB | 86 ms |
| [`strands-decider-31B-gemma4-v2-2611`](https://huggingface.co/StrandsAgents/strands-decider-31B-gemma4-v2-2611) | 71.5 | 93.1 | 92.2 | 84.3 | 84.5 | 80 GB | 99 ms |

Scores are 0 to 100. Every v2 model takes the same request and gives the same kind of answer as
v1, so switching from v1 to v2 is a one-name change in `strands-decider ask` or `strands-decider
serve`. The v2 training data card is in [data/README.md](data/README.md), the training recipes are
in `configs/v2/` ([training/README.md](training/README.md#the-v2-recipes)), and the full per-model
numbers are in the [models reference](https://strandsagents.com/docs/labs/strands-decider/models/).

**Choose a size.** The 31B is the most accurate. The 26B-A4B is the best of the mixture-of-experts
models and has the lowest latency among the top two. The 12B is the best dense model that fits on
one 48 GB GPU. E4B is a good choice when the model must share a GPU; E2B is the smallest and runs
on 24 GB or on CPU.

**Install and ask.**

```bash
pip install strands-decider
strands-decider ask StrandsAgents/strands-decider-E2B-gemma4-v2-2611 \
  --state "Help! My payouts have been failing for 3 days! " \
  --choice "Which team should handle this?=billing,sales,retail" \
  --noul "Does this convey urgency?" \
  --score "How frustrated is the writer?=calm,frustrated,depressed"
```

**Deploying v2.** v2 checkpoints serve on SageMaker and on Lambda Managed Instances GPU in addition
to the environments that v1 serves on (the [deployment guide](https://strandsagents.com/docs/labs/strands-decider/deployment/local/)).

### Model names

Model names use the pattern `strands-decider-<size>-<base>-v<N>-<YYMM>`:

- `<size>` is the size of the base model, as its Hugging Face id writes it (`2B`, `E4B`).
- `<base>` is the base model family (`qwen3.5`, `gemma4`).
- `v<N>` counts the releases on one base family: `v1` is the first release on each base.
  Releases with the same `N` on different bases can use different recipes; each model card says
  what its model trained on.
- `<YYMM>` is the release month.

Examples: `strands-decider-2B-qwen3.5-v1-2610` (Qwen3.5 v1), `strands-decider-E4B-gemma4-v1-2610`
(Gemma 4 v1) and `strands-decider-31B-gemma4-v2-2611` (Gemma 4 v2). The earlier models keep their
names: `strands-decider-2B-hobson-v19` and `strands-decider-2B-hobson-v21`. A second release in the
same month updates the same Hub repository and adds a revision tag `YYMMDD`.
[designs/naming.md](designs/naming.md) has the full rules.

The models live in the [`StrandsAgents`](https://huggingface.co/StrandsAgents) organization on
Hugging Face, their main home. The five v1 models are also in the
[`amazon`](https://huggingface.co/amazon) organization, with the same files and the same names:
for example `amazon/strands-decider-E4B-gemma4-v1-2610`.

<details>
  <summary><b>v1 models</b> — the Qwen3.5 model (with vision) and the Gemma 4 pointer-head release</summary>

[`strands-decider-2B-qwen3.5-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610)
is the reference v1 model: 1.9 B parameters on Qwen3.5-2B-Base, pointer head, JevBench 180 / 231,
Brier 0.280, about 4.5 GB in bf16. It is the only v1 checkpoint that takes images (via the
`--vision` flag), and the only one that runs through MLX on an Apple-silicon Mac.

Four more v1 models on the Gemma 4 instruction-tuned bases (`google/gemma-4-*-it`, Apache-2.0),
same pointer head and same three question types, text only:

| model | JevBench (231) | Brier | held-out short tasks | held-out code tasks | Decision Index | GPU memory for the weights (bf16) | median latency per JevBench question, one H100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [`strands-decider-E2B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v1-2610) | 180 | 0.313 | 0.652 | 0.773 | 29.78 | about 9 GB | 102 ms |
| [`strands-decider-E4B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-E4B-gemma4-v1-2610) | 187 | 0.243 | 0.684 | 0.815 | 39.42 | about 15 GB | 139 ms |
| [`strands-decider-12B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-12B-gemma4-v1-2610) | 200 | 0.173 | 0.708 | 0.870 | 47.14 | about 23 GB | 136 ms |
| [`strands-decider-26B-A4B-gemma4-v1-2610`](https://huggingface.co/StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610) | 207 | 0.162 | 0.708 | 0.859 | 48.69 | about 51 GB | 245 ms |

The E2B Decision Index was measured before the E2B attention fix (see
[designs/inference.md](designs/inference.md#gemma-4-models)), so it may be off. Every model is a
soup of three training runs, calibrated afterwards
([evaluation/results.md](evaluation/results.md#gemma-4-v1)).

**Choose a size.** A larger model is more accurate and better calibrated, and needs more memory.
The 12B and the 26B-A4B are the most accurate; the 26B-A4B is better calibrated, but needs about
twice the memory and twice the latency of the 12B. The E4B is a good choice when the model
must share one GPU with other work. The E2B is the smallest: use it where memory is tightest, or on
CPU. All four serve on CPU ([designs/inference.md](designs/inference.md#gemma-4-models)).

**Install.**

```bash
pip install strands-decider
strands-decider ask StrandsAgents/strands-decider-E4B-gemma4-v1-2610 \
  --state "Help! My payouts have been failing for 3 days! " \
  --choice "Which team should handle this?=billing,sales,retail" \
  --noul "Does this convey urgency?" \
  --score "How frustrated is the writer?=calm,frustrated,depressed"
```

Example output on `--device cpu`:

```bash
noul_0 noul = 0.959
choice_0 -> billing (confidence 0.951)
  billing                  0.967
  sales                    0.019
  retail                   0.013
score_0 score = 0.97 (confidence 0.848)
  0: calm                                     0.102
  1: frustrated                               0.823
  2: depressed                                0.075
```

**What they train on.** The Gemma 4 models do not use the Qwen3.5 v1 data mix. They train on
hobson-v21's data (123,339 rows) plus 12,263 code-task rows, 135,602 rows in all. Qwen3.5-4B is the
teacher, where its answer agrees with the gold label, and training keeps the answers close to the
released v19. The rows come from 25 public Hugging Face datasets, five of them code datasets, and from
ContractNLI, MuSiQue, generated rows and generated code tasks. [data/sources.md](data/sources.md)
lists every source with its licence.

</details>

## Images

Strands Decider answers over images: pass `--image page.png` on the CLI, or base64 images in the
`images` field of the state over HTTP.

```bash
strands-decider serve StrandsAgents/strands-decider-E2B-gemma4-v2-2611 --port 8000
strands-decider ask StrandsAgents/strands-decider-E2B-gemma4-v2-2611 --state "" --image page.png \
  --noul "Is the signature block filled in?"
```

See [designs/images.md](designs/images.md) for the request shape, how it works and the measurements.

> **v1 note.** The v1 Gemma 4 models are text-only. The v1 Qwen3.5 model takes images with the
> `--vision` flag; v19 matches an image-trained 2B decider on accuracy, and on NaturalBench is
> much better calibrated (ECE 0.014 against 0.080).

## About the model

One forward pass, no generation. A pretrained decoder LLM torso reads the prompt, and a small
readout turns the final hidden states into a probability over the options of the question. Three
question types come out of the same masked softmax: `noul` (yes/no), `choice` (one of N), `score`
(an ordered rubric). Options come from the request, so label sets are not baked into the weights —
no retraining to add a category, and no option ceiling other than the schema's 255. Many questions
about one text are cheap: the text is read once and each question adds only its own tokens
([designs/inference.md](designs/inference.md#asking-many-questions-is-nearly-free)).

**Two readouts.** v1 uses a pointer head of about a million parameters that compares the hidden
state at the `<answer>` position against each option's last token; the torso is adapted with a
rank-16 LoRA. **v2** replaces the pointer head with a token readout over 255 single-token option
codes (`A`..`IV`), read off the hidden state at the last prompt token, with the state encoded
once under bidirectional attention so several questions share it. v2 ships on five Gemma 4 sizes
(E2B – 31B) and is more accurate than v1 at every size
([designs/architecture.md](designs/architecture.md#the-architecture)).

**Accuracy, calibration, latency.** v2's five models score 87.4 – 93.1 on JevBench (231 tasks)
and 57.1 – 71.5 on Decision Index 0.3, with median H100 latency of 62 – 99 ms per question
([Model options](#model-options)). v1's reference Qwen3.5 2B scores 180 / 231 on JevBench with
Brier 0.280 and answers in a median 115 ms on an RTX 3090 and 153 ms on an M3 Pro
([evaluation/results.md](evaluation/results.md#v1-the-released-model)). JevBench is a third-party
benchmark for this class of model; 231 tasks are few, so treat differences under about ten tasks
between two single runs as unresolved ([evaluation/README.md](evaluation/README.md)).

**Why a small model.** At 1.9 B (v1 Qwen3.5), or an E2B / E4B v2, you can serve the model and
retrain the whole recipe on hardware you already have — about 11 hours on one RTX 3090, and
serving works on an Apple-silicon Mac or on CPU. That makes trying an idea fast and low-risk,
while leaving the 12B, 26B-A4B and 31B v2 sizes for when accuracy matters more than footprint.

## What can I do with it?

The decisions inside an agentic workflow are the natural target:

- **Model routing** — pick the right LLM for a task
- **Tool selection** — decide which tool an agent should call next
- **Argument checking** — verify a tool call's arguments before it runs
- **Triage** — route an incoming request to the team or queue that should own it
- **Guardrails** — grounding checks, safety classification, policy classification
- **Evaluations** — score model outputs, and check whether an answer is adequate, at low cost
- **Hybrid agents** — let the LLM make the hard decisions and a decider make the rote ones,
  reducing cost and latency

## Trying it in an Agent

The repository includes a worked example of strands decider inside a Strands agent, under
[`examples/strands/`](examples/strands/README.md): a `before_tool_call` intervention that gates a
weather-tool call on two yes/no decisions, so the agent asks which city instead of guessing. See
the example's [README](examples/strands/README.md) for setup and the walk-through.

## Customizing for your decisions

[`skills/customize-decider/`](skills/customize-decider/SKILL.md) is a skill for an AI agent.
It guides you from a decision in your application to a model measured on your own data: the task
and its questions, an evaluation set, scores of every released model, the decision whether to
train, the training data, the run and its monitoring, calibration, and the comparison with the
released model.

The skill is plain Markdown: a `SKILL.md` with a short front matter, and reference files that it
loads one phase at a time. It works with any agent that can read files and run shell commands, such
as an agent built with the [Strands Agents SDK](https://github.com/strands-agents/sdk-python) or
a coding CLI. If your agent supports skills, install the folder where it looks for skills. If not,
tell the agent to read `skills/customize-decider/SKILL.md` and follow it. Then ask it to evaluate
Strands Decider on your task.

## Training

Two entry points, both on a Linux or WSL2 host with NVIDIA GPUs and the training
environment of [training/README.md](training/README.md#setup) (the `train` extra, the
pinned torch and `flash-linear-attention`):

- `training/recipe.sh all`: one host, local, 1 to 8 GPUs (`NGPU=8` for eight). It builds
  the corpora, trains, calibrates and evaluates.
- `training/run_recipe.sh all` with [training/aws/](training/aws/README.md): a distributed
  8-GPU host. It adds per-stage timing and logs, row-count checks, the S3 copy of the
  outputs (`PY` and `S3_PREFIX` must be set), and `FAST=1`, which trades 24 GB
  compatibility for speed on 80 GB GPUs.

About 11 hours on one RTX 3090 (24 GiB), or 1 hour 10 minutes on eight H100s with `FAST=1`.
[training/README.md](training/README.md) has the setup, the stages, the settings and the
hardware notes; [data/sources.md](data/sources.md) lists every source and its licence, and
[data/README.md](data/README.md#reproduction-contract) states the reproduction contract.

## The research

The record of the work is in the repository. Since v9, every training run states its
predictions and its failure conditions before training, and the outcome is appended after
the run without editing what came before. By that rule, a run that misses its bar does not
replace the reference model; five runs were promoted by the maintainers' decision anyway,
and the record says so. Most runs missed their bar, v20 among them: 169 of 231 at the
4096-token window against v19's 168, inside the retrain noise, with four predictions
failed. [research/README.md](research/README.md) lists each run and its outcome, and
[research/history.md](research/history.md) tells what moved the benchmark and what did
not. To propose an experiment, open an issue with a preregistration ([CONTRIBUTING.md](CONTRIBUTING.md)).

## Community

To ask questions and talk about Strands Decider, join the [Strands Decider channel on Discord](https://discord.gg/Wa4CQrxsP).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.
