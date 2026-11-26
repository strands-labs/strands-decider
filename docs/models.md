# Models

Strands Decider v2 has five models on Gemma 4, from E2B to 31B. They are the current models: more accurate than
v1 at every size, with a token readout, a 32,768-token context and image input. The five v1 models stay
available, and their matrix is below.

## v2 model matrix

| Model | Base | Decision Index 0.3 | JevBench | OpenDecisions | DecisionPivot | BEV | GPU to use | Latency (H100, median) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [E2B-gemma4-v2-2611](models/e2b-gemma4-v2.md) | gemma-4-E2B-it | 57.1 | 87.4 | 77.7 | 23.3 | 74.0 | 24 GB | 62 ms |
| [E4B-gemma4-v2-2611](models/e4b-gemma4-v2.md) | gemma-4-E4B-it | 63.2 | 88.3 | 83.9 | 45.2 | 78.9 | 40 GB | 78 ms |
| [12B-gemma4-v2-2611](models/12b-gemma4-v2.md) | gemma-4-12B-it | 64.1 | 92.2 | 85.7 | 58.9 | 79.0 | 48 GB | 80 ms |
| [26B-A4B-gemma4-v2-2611](models/26b-a4b-gemma4-v2.md) | gemma-4-26B-A4B-it | 69.0 | 92.6 | 88.6 | 69.2 | 81.4 | 80 GB | 86 ms |
| [31B-gemma4-v2-2611](models/31b-gemma4-v2.md) | gemma-4-31B-it | 71.5 | 93.1 | 92.2 | 84.3 | 84.5 | 80 GB | 99 ms |

Scores are 0 to 100. Decision Index 0.3, public: 41 tasks in five areas. JevBench: the share of the 231 public
items answered correctly. OpenDecisions: everyday decisions, scored on answer quality and calibration.
DecisionPivot: pair accuracy on items that differ in one decisive detail. BEV: choice, yes/no and ordered-score
decisions across 18 domains. The latency is for one question, over the 231 JevBench items. Each model page
has the v1 scores next to these and the latency for 30 questions.

## Which v2 model should I pick?

- **Smallest, one 24 GB GPU or no GPU:** E2B-gemma4-v2.
- **One 40 GB GPU:** E4B-gemma4-v2.
- **One 48 GB GPU:** 12B-gemma4-v2.
- **One 80 GB GPU, many questions about one state:** 26B-A4B-gemma4-v2.
- **Most accurate:** 31B-gemma4-v2.

Every v2 model takes the same request and gives the same kind of answer, so a change of model is a change of
one name in `strands-decider ask` or [`strands-decider serve`](inference.md).

### On a smaller GPU: fewer bits before a smaller model

`strands-decider serve` and `ask` take [`--quantization 4bit` or `8bit`](inference.md#flags-worth-knowing),
which loads a model's weights in 4 or 8 bits on an NVIDIA GPU. On the same GPU, a larger model in fewer bits is
more accurate than the smaller model that fits in bf16:

| GPU | Model in bf16 | Model in fewer bits | GPU memory in use | Gain in accuracy |
| --- | --- | --- | --- | --- |
| 48 GB | 12B | 31B, `--quantization 8bit` | 30.5 GiB after load, 39.4 GiB with a 32,768-token prompt | 3.5 points |
| 40 GB | E4B | 31B, `--quantization 4bit` | 17.0 GiB after load, 25.9 GiB with a 32,768-token prompt | 4.6 points |
| 24 GB | E2B | 12B, `--quantization 8bit` | 12.1 GiB after load, 18.3 GiB with a 32,768-token prompt | 4.1 points |

The 31B in 4 bits also runs in 24 GB with shorter prompts: with GPU memory limited to 23 GiB, it answered every
prompt of up to 14,000 tokens. A copy of the 31B saved in 4 bits is on the Hub as a preview,
[strands-decider-31B-gemma4-v2-2611-bnb-4bit-preview](https://huggingface.co/StrandsAgents/strands-decider-31B-gemma4-v2-2611-bnb-4bit-preview),
and it loads without the option.

Rules of thumb:

- **Reduce the bits before the size.** One step down in size costs 1.6 to 2.5 points of accuracy. In 8 bits the
  31B and the 12B keep their accuracy within 0.2 points, and in 4 bits the 31B loses about 0.3 points.
- **4 bits for the 31B, 8 bits for the smaller models.** In 4 bits the 31B keeps the top answer of its bf16
  model on 99.1% of the reference questions. The 12B, the E4B and the E2B change 2% to 7% of their top answers
  in 4 bits, so run them in 8.
- **A threshold on the probability: 8 bits.** Where the bf16 model gives its top answer a probability between
  0.1 and 0.9, the 31B moves that probability by 0.06 on average in 4 bits and by 0.02 in 8 bits.
- **For speed, fewer active parameters.** Fewer bits save memory, not time: with `--quantization`, one request
  takes about 1.2 times the time of bf16 in 4 bits and about 2.2 times in 8 bits. Throughput follows the number
  of active parameters. Served with vLLM on one H100, the 26B-A4B, with 4B active parameters, answers 59
  decisions a second, the 12B answers 24 and the 31B answers 9.
- **The 26B-A4B stays in bf16.** The option converts linear layers, and most of this model's weights are in
  expert tensors: in 4 bits it takes 45.0 GiB where bf16 takes 48.2 GiB.

The gains are differences in accuracy on 2,991 questions sampled from Decision Index 0.3, BEV, JevBench and 22
public NLI and QA datasets, with both models on the same questions. The measurements, with the memory by prompt
length and the latency, are in [issue 172](https://github.com/strands-labs/strands-decider/issues/172).

## v1 model matrix

Strands Decider v1 publishes five checkpoints: one Qwen3.5 base (with a vision
tower) and four Gemma 4 bases spanning 2B – 26B parameters. All five share the
pointer head and the public API. The Qwen3.5 model and the Gemma 4 models train
on different data mixes with different teachers. The models also differ in
base model, size, and image support.

| Model | Base | Params | JevBench (231) | Brier | GPU memory (bf16) | Latency (H100, median) | Images |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [2B-qwen3.5-v1-2610](models/2b-qwen3-5.md) | Qwen3.5-2B-Base | 2B | 180 | 0.280 | ~4.5 GB | — | ✓ |
| [E2B-gemma4-v1-2610](models/e2b-gemma4.md) | gemma-4-E2B-it | 2B eff. | 180 | 0.313 | ~9 GB | 102 ms | — |
| [E4B-gemma4-v1-2610](models/e4b-gemma4.md) | gemma-4-E4B-it | 4B eff. | 187 | 0.243 | ~15 GB | 139 ms | — |
| [12B-gemma4-v1-2610](models/12b-gemma4.md) | gemma-4-12B-it | 12B | 200 | 0.173 | ~23 GB | 136 ms | — |
| [26B-A4B-gemma4-v1-2610](models/26b-a4b-gemma4.md) | gemma-4-26B-A4B-it | 26B MoE (4B active) | 207 | 0.162 | ~51 GB | 245 ms | — |

JevBench scores come from
[evaluation/results.md](https://github.com/strands-labs/strands-decider/blob/main/evaluation/results.md).
GPU memory is for the weights only, and it and the latency come from
[designs/inference.md](https://github.com/strands-labs/strands-decider/blob/main/designs/inference.md#gemma-4-models).
Activations come on top, and each model page names a GPU that fits.
Lower Brier is better; higher JevBench is better.

## Which v1 model should I pick?

- **Fit on a laptop or small instance, or need image input:** 2B-qwen3.5.
- **One GPU that the model shares with other work:** E4B-gemma4.
- **Best accuracy of the dense models, on one 48 GB GPU:** 12B-gemma4.
- **Best accuracy, mixture-of-experts serving:** 26B-A4B-gemma4.
- **Smallest Gemma option:** E2B-gemma4.

Every v1 checkpoint speaks the same `--state` / `--choice` / `--noul` /
`--score` interface and the same HTTP contract, so switching models is a
one-line change in [`strands-decider serve`](inference.md).

## Naming

Model IDs follow `strands-decider-{size}-{base}-v{N}-{YYMM}`. See
[designs/naming](../designs/naming.md) for the full rule, including what
changes when `N` increments.
