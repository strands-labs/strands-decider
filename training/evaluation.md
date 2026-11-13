# Evaluating a token-readout export

An export of the token-readout path is a decision checkpoint (the spec's section 9, `docs/spec.md`), so it is
evaluated by serving it and asking it questions, as any client would. The trainer itself reports one set of numbers:
the dev-fold metrics in `<run>/eval_log.jsonl` (accuracy, NLL, Brier, ECE every `eval_every` steps). Everything below
runs on the export.

## Serve it

A decision runtime renders each request with the checkpoint's own prompt (`decision_config.json`), runs the decoder
under `prefix_bidir`, and reads the answers through `readout.safetensors`:

```bash
# with the transformers runtime installed (below); the recipe's exports declare text inputs only
strands-decider serve work/e2b/export/strands-decider-e2b --port 8000    # binds 127.0.0.1
strands-decider ask work/e2b/export/strands-decider-e2b --state "The parcel arrived damaged." \
  --choice "Which team handles this?=billing,logistics,sales"
```

The server answers `POST /v1/systemone`. At the time of writing the decision runtimes are open pull requests of this
repository, not yet in a release: the transformers runtime (#136), MLX on Macs (#137), llama.cpp on CPUs and AWS
Lambda (#138), and the vLLM plugin (`strands-decider-vllm`, #130) on GPU servers. `strands-decider serve` and `ask`
load a decision checkpoint once the transformers runtime is in; until then, install from its branch.

Check a runtime against the checkpoint before trusting its scores: with the spec's golden vectors (section 10) when the
export ships them, or by comparing two runtimes' probabilities on the same requests.

## Public benchmarks

- JevBench (231 public tasks, 18 families, MIT): its `typesafe` adapter speaks `/v1/systemone`, so it runs against
  a served export unchanged ([`evaluation/jevbench.md`](../evaluation/jevbench.md), "Reproducing, and two caveats",
  gives the harness commands). `evaluation/jevbench/jevbench.sh` automates that for v1 checkpoints only: it reads their
  `strands_decider_config.json`. `evaluation/jevbench/paired.py` compares two runs task by task with an exact McNemar
  test.
- DecisionPivot (arXiv 2610.28522): minimal pairs of decisions that differ in one fact, scored by pair accuracy. Run
  its public harness against the server.

Compare two checkpoints on the same items with a paired test (McNemar for accuracy, a paired bootstrap over items for
mean scores): a single training run moves a benchmark by a few points, so one run's difference under that is noise.
Report the export's revision, the runtime and its commit, the prompt length limit (`max_length_validated`: the
training max length, 16,384, unless the export was given another; the recipe does not itself test prompts at that
length), and the benchmark's commit with every score.
