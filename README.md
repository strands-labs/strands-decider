<div align="center">
  <div>
    <a href="https://strandsagents.com">
      <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://strandsagents.com/latest/assets/wordmark-github-dark.svg">
        <img src="https://strandsagents.com/latest/assets/wordmark-github-light.svg" alt="Strands" width="320">
      </picture>
    </a>
  </div>

  <h1>Strands Decider</h1>

  <h2>A small, fast decision model for agentic AI.</h2>

  <div align="center">
    <a href="https://github.com/strands-labs/strands-decider/graphs/commit-activity"><img src="https://img.shields.io/github/commit-activity/m/strands-labs/strands-decider" alt="Commit Activity"></a>
    <a href="https://github.com/strands-labs/strands-decider/issues"><img src="https://img.shields.io/github/issues/strands-labs/strands-decider" alt="Open Issues"></a>
    <a href="https://github.com/strands-labs/strands-decider/pulls"><img src="https://img.shields.io/github/issues-pr/strands-labs/strands-decider" alt="Open PRs"></a>
    <a href="https://discord.gg/Wa4CQrxsP"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord"></a>
  </div>

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

## Getting Started

The easiest place to get started is through the `strands-decider` cli:

```bash
pip install strands-decider
```

On an Apple-silicon Mac, `--device mlx` runs the model through MLX, 1.4 to 1.6x faster than MPS.
It needs the `mlx` extra, which ships with the next release; until then, install from a clone with
`pip install -e ".[mlx]"`. The `cuda`, `mps` and `cpu` extras name the other devices
([docs/inference.md](docs/inference.md#environments-for-serving)).
### Choice question
You can ask the model to choose based on some state and a question:
```bash
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v21 \
  --state "Help! My payouts have been failing for 3 days! " \
  --choice "Which team should handle this?=billing,sales,retail" 
```
Example Output:

```bash
choice_0 -> billing (confidence 0.835)
  billing                  0.890
  sales                    0.056
  retail                   0.054
```

### Noul question
You can also ask the model a Yes/No question:

```bash
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v21 \
  --state "Help! My payouts have been failing for 3 days! " \
  --noul "Does this convey urgency?" 
```

Example Output:

```bash
noul_0 noul = 0.875
```
Closer to 1 is leaning more toward Yes

### Score question
Or give a question a score:
```bash
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v21 \
  --state "Help! My payouts have been failing for 3 days! " \
  --score "How frustrated is the writer?=calm,frustrated,depressed"
```
Example Output:

```bash
score_0 score = 1.07 (confidence 0.602)
  0: calm                                     0.140
  1: frustrated                               0.648
  2: depressed                                0.212
```

### Multiple question types

You can combine multiple questions into a single command. This is more efficient as you only need to load the state for the model once:

```bash
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v21 \
  --state "Help! My payouts have been failing for 3 days! " \
  --choice "Which team should handle this?=billing,sales,retail" \
  --noul "Does this convey urgency?" \
  --score "How frustrated is the writer?=calm,frustrated,depressed"
```
<details>
  <summary>Example Output</summary>

  ```bash
  noul_0 noul = 0.875
  choice_0 -> billing (confidence 0.837)
    billing                  0.891
    sales                    0.055
    retail                   0.054
  score_0 score = 1.07 (confidence 0.603)
    0: calm                                     0.140
    1: frustrated                               0.649
    2: depressed                                0.211
  ```
</details>

### Running as a server

You can also run the model as a server, and ask questions via http requests:
```bash
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v21 --port 8000
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
    "model": "strands-decider-2B-hobson-v21",
    "answers": {
      "is_urgent": {
        "type": "noul",
        "noul": 0.8751
      }
    },
    "usage": {
      "input_tokens": 86,
      "output_tokens": 1
    },
    "latency_ms": 140.03
  }          
  ```
</details>

### Images

Qwen3.5-2B-Base is natively multimodal. With `--vision` the server keeps its vision tower,
and a request may carry `images` (base64) as part of the state; the published checkpoints
(v21, and v19 before it) answer over them with no image training. Needs `pip install "strands-decider[vision]"` and
transformers 5.18 or later.

```bash
strands-decider serve StrandsAgents/strands-decider-2B-hobson-v21 --vision --port 8000
strands-decider ask StrandsAgents/strands-decider-2B-hobson-v21 --state "" --image page.png \
  --noul "Is the signature block filled in?"
```

v19 matches an image-trained 2B decider on accuracy, and on NaturalBench is much better
calibrated (ECE 0.014 against 0.080). v21 has v19's image accuracy, is better calibrated on
POPE, and is more confident than v19 when the image is missing. See
[docs/vision.md](docs/vision.md) for the request shape, how it works and the measurements.

## About the model

`strands-decider-2B`, the first model of the family, has 1.9 billion parameters. It answers a
question in a median 115 ms on an RTX 3090 and also serves on an Apple-silicon Mac or on CPU,
and many questions about one text are cheap, because the text is read once and each question
adds only its own tokens
([docs/inference.md](docs/inference.md#asking-many-questions-is-nearly-free)). Its
accuracy and calibration are measured on JevBench, a third-party benchmark for this class of
model ([Performance](#performance)).


## Model Architecture

The core idea: take a pretrained decoder LLM torso ([Qwen3.5-2B-Base](https://huggingface.co/Qwen)),
**discard its language-modelling head** — taking away its ability to generate text — and
replace it with a small **pointer head** of about a million parameters. The head scores each
option by comparing the hidden state at the `<answer>` position against the hidden state at
that option's own last token. One forward pass, no generation, no decoding loop. The torso is
adapted with a rank-16 LoRA adapter, and the head runs in fp32.

Because the head holds no per-option parameters, nothing can learn that "the first option is
usually right", nothing caps how many options a question may carry, and label sets are
defined by the request rather than baked into the weights. Three question types come out of
the same masked softmax, read back differently: `noul` (yes/no), `choice` (one of N) and
`score` (an ordered rubric).

As you browse the research, you will find that this is the second major iteration of the
architecture. The first used a slot head, which mapped the final hidden state to a fixed set
of slots and performed significantly worse. Every change since is captured in the research
so you can follow along with the work. The reference model today is **v21**
(`strands-decider-2B-hobson-v21`, v21b in the research notes): v19's recipe plus v20's
checked question paraphrases and distillation from Qwen3.5-4B where it agrees with the gold
label. It is one of six seeds, picked by a rule written before the seeds were ranked
([evaluation/results.md](evaluation/results.md#v21-the-released-model)). v19 stays published.
[docs/architecture.md](docs/architecture.md#the-architecture) has the full design, the
training objective, and the decisions behind them.

## Performance

Three targets matter: **accuracy**, **calibration** and **latency**. v21 and v19 measure:

| Measurement | v21 | v19 | Source |
| --- | --- | --- | --- |
| JevBench v1 public set, 231 tasks, accuracy | 0.762 (176 of 231); six-seed mean 0.758 (175.0, SD 2.4) | 0.723 (167 of 231) | [evaluation/results.md](evaluation/results.md#v21-the-released-model) |
| JevBench Brier score / expected calibration error | 0.323 / 0.064 | 0.342 / 0.052 | [evaluation/README.md](evaluation/README.md) |
| Tiers (this repository's split of the public tasks): easy / standard / hard | 1.000 / 0.931 / 0.550 | 1.000 / 0.875 / 0.505 | [evaluation/jevbench.md](evaluation/jevbench.md#board-position-v142-25-september-2026) |
| Latency per JevBench question, RTX 3090 under WSL2, median / 95th percentile | same architecture and size as v19 | 115 ms / 299 ms | [evaluation/results.md](evaluation/results.md#summary) |
| Latency per question, M3 Pro (Apple silicon), warm median, under 300 tokens / all tasks | same architecture and size as v19 | 153 ms / 234 ms | [evaluation/results.md](evaluation/results.md#serving-on-a-mac-accuracy-and-latency) |

Every task in the easy tier is answered correctly, and the nearest comparison is
`decider-2b`, which shares v19's torso with a different recipe
([evaluation/jevbench.md](evaluation/jevbench.md#against-the-nearest-open-systems)).

Read the v21 and v19 columns as two single runs, not as a measured gain. Trained on one
host with six seeds each, the v19 recipe and v21's average 172.8 and 172.3 JevBench tasks:
the same accuracy. v21's Brier score is lower on that host (0.331 against 0.341), and the
two recipes differ by the paraphrases and the distillation only.

v19's JevBench figures are at the 3072-token window the run was preregistered at; the recipe
saves a 4096-token window, at which v19 scores 168. And 231 tasks are few: six retrains of
the v17 recipe had a standard deviation of 3.2 tasks, so treat a difference under about 10
tasks between two single runs as unresolved. [evaluation/README.md](evaluation/README.md)
has the scripts and the limitations, and links the results by version and the board
position with its caveats. [docs/evaluating.md](docs/evaluating.md) shows how to measure any
checkpoint with a local proxy of JevBench v1.5's rules and on task families no model trains
on, and how to check a reported number.

## Candidate models (unofficial)

Exploratory checkpoints offered for review; none of them is a published Strands release.
Each document gives the base and its pinned revision, the size, the lineage, the results,
the download with its sha256, the serve command, the commands to re-run every measure on
the downloaded weights, and the commands to retrain. The numbers are local measures
([docs/evaluating.md](docs/evaluating.md)); the board positions are estimates pending an
official submission.

| Model | Base, size | Answers | Estimated 2B-class position (unofficial) |
| --- | --- | --- | --- |
| [strands-decider-2B-hobson-v20](docs/models/strands-decider-2B-hobson-v20.md) | Qwen3.5-2B-Base, 2B | text | #1 strictly-2B text model |
| [strands-decider-2B-hobson-v20-balanced](docs/models/strands-decider-2B-hobson-v20-balanced.md) | Qwen3.5-2B-Base, 2B | text and images | best text+image 2B candidate (the only 2B candidate on both boards); #2 on Image JevBench |
| [strands-decider-2.5B-minicpm-v21](docs/models/strands-decider-2.5B-minicpm-v21.md) | MiniCPM5-2B, 2.5B | text | top 3 text |
| [strands-decider-2.5B-minicpm-v21-vl](docs/models/strands-decider-2.5B-minicpm-v21-vl.md) | MiniCPM5-2B + SigLIP2, 2.5B + 0.4B | text (images weakly) | #1 text |

## Why 2B?

Two reasons. First, **experimentation**: at 1.9 billion parameters you can serve the model,
and retrain the whole recipe, on hardware you already have — about 11 hours on one RTX 3090,
and serving works on an Apple-silicon Mac. That makes trying an idea fast and low-risk.
Second, ~2B parameters looks like a sweet spot: small enough to experiment with, large enough
to do meaningful work.

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