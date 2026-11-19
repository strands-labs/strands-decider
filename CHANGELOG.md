# Changelog

## Model names (2026-10)

Model names now use the pattern `strands-decider-<size>-<base>-v<N>-<YYMM>`:

- `<size>` is the size of the base model, as its Hugging Face id writes it (`2B`, `E4B`).
- `<base>` is the base model family (`qwen3.5`, `gemma4`).
- `v<N>` counts the releases on one base family: `v1` is the first release on each base.
  Releases with the same `N` on different bases can use different recipes; each model card says
  what its model trained on.
- `<YYMM>` is the release month.

Examples: `strands-decider-2B-qwen3.5-v1-2610` (Qwen3.5) and `strands-decider-E4B-gemma4-v1-2610`
(Gemma 4). The earlier models keep their names: `strands-decider-2B-hobson-v19` and
`strands-decider-2B-hobson-v21`. A second release in the same month updates the same Hub
repository and adds a revision tag `YYMMDD`. [docs/naming.md](docs/naming.md) has the full rules.

## Models

Names: [docs/naming.md](docs/naming.md). Results: [evaluation/results.md](evaluation/results.md).

- **2026-10-09, `StrandsAgents/strands-decider-{E2B,E4B,12B,26B-A4B}-gemma4-v1-2610`** (v1, Gemma 4).
  The first models on Gemma 4, in four sizes. hobson-v21's data plus 12,263 code-task rows;
  Qwen3.5-4B as the teacher; the released v19 as the reference that training keeps the answers
  close to; a soup of three training runs per size. JevBench 180 / 187 / 200 / 207 of 231, Brier
  0.313 / 0.243 / 0.173 / 0.162, held-out code tasks 0.773 / 0.815 / 0.870 / 0.859 (E2B / E4B /
  12B / 26B-A4B). E2B serves with eager attention on CUDA ([docs/inference.md](docs/inference.md#gemma-4-models)).
- **2026-10-09, `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`** (v1). A new, balanced data mix
  of 246,678 rows with code tasks and rule-application rows; gemma-4-31B-it as the teacher; the
  released v19 as the reference that training keeps the answers close to; a soup of three training
  runs. 180/231 on JevBench, Brier 0.280, 0.836 on held-out code tasks.
- **2026-10-05, `StrandsAgents/strands-decider-2B-hobson-v21`**. v19's recipe plus checked
  question paraphrases and distillation from Qwen3.5-4B where it agrees with the gold label.
  Tested with image input (`--vision`).
- **`StrandsAgents/strands-decider-2B-hobson-v19`**. The first published checkpoint.
