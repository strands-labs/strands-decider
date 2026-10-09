# Changelog

## Model names (2026-10)

Model names now use the pattern `strands-decider-<size>-<base>-v<N>-<YYMM>`:

- `<size>` is the size of the base model, as its Hugging Face id writes it (`2B`, `E4B`).
- `<base>` is the base model family (`qwen3.5`, `gemma4`).
- `v<N>` is the recipe generation. It starts again at `v1`. It changes only when users can see
  the change, for example new training data, a new head or a new objective. One recipe has the
  same `N` on all bases and sizes.
- `<YYMM>` is the release month.

Examples: `strands-decider-2B-qwen3.5-v1-2610` (Qwen3.5) and `strands-decider-E4B-gemma4-v1-2610`
(Gemma 4, planned). The earlier models keep their names: `strands-decider-2B-hobson-v19` and
`strands-decider-2B-hobson-v21`. A second release in the same month updates the same Hub
repository and adds a revision tag `YYMMDD`. [docs/naming.md](docs/naming.md) has the full rules.

## Models

Names: [docs/naming.md](docs/naming.md). Results: [evaluation/results.md](evaluation/results.md).

- **2026-10-09, `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610`** (v1). A new, balanced data mix
  of 246,678 rows with code tasks and rule-application rows; gemma-4-31B-it as the teacher; the
  released v19 as the reference that training keeps the answers close to; a soup of three training
  runs. 180/231 on JevBench, Brier 0.280, 0.836 on held-out code tasks.
- **2026-10-05, `StrandsAgents/strands-decider-2B-hobson-v21`**. v19's recipe plus checked
  question paraphrases and distillation from Qwen3.5-4B where it agrees with the gold label.
  Tested with image input (`--vision`).
- **`StrandsAgents/strands-decider-2B-hobson-v19`**. The first published checkpoint.
