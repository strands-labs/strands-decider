# Evaluation

This folder holds the evaluation scripts and what they measured. Two kinds of measurement:
the internal sets diagnose a checkpoint, and JevBench's 231 public tasks rank it. v19, the
reference recipe, scores 167 of 231 (0.723) at the 3072-token window, the input length it
was preregistered at (served at its 4096 default it scores 168), with Brier 0.342 and ECE
0.052. On this repository's split of the public tasks into easy, standard and hard tiers it
scores 1.000, 0.875 and 0.505. One question takes a median 115 ms on an RTX 3090. One
retrain moves the JevBench score by about 3 tasks (SD 3.2 over six v17 retrains; no v19
seed replicate finished), so a single-run difference under about 10 tasks is unresolved.

- [`results.md`](results.md): the AWS retrain, the summary by version, calibration, and a Mac.
- [`jevbench.md`](jevbench.md): the external benchmark, the board position, the families, the
  context window, and how to reproduce the run.
- [`jevbench/`](jevbench/): the JevBench harness driver and its two helpers.

`training/recipe.sh eval` runs the internal evaluations on the recipe's checkpoint. Each
script also runs on its own:

| Script | What it measures |
| --- | --- |
| `strands-decider eval` (the CLI) | Accuracy, ECE, NLL and score MAE on the held-out classification tasks, by task, primitive and confidence band |
| [`multistep_eval.py`](multistep_eval.py) | Accuracy on the multi-step evaluation sets, per source. HotpotQA is the transfer test |
| [`question_sensitivity.py`](question_sensitivity.py) | Whether the answer follows the question, with the frozen torso as its own baseline |
| [`pair_eval.py`](pair_eval.py) | Paraphrase consistency and instruction-flip pair accuracy on adjacent pairs (v20) |
| [`pair_accuracy.py`](pair_accuracy.py) | Item and pair accuracy on a minimal-pair file (v10) |
| [`calibrate_mix.py`](calibrate_mix.py) | Refits the temperatures on every held-out set, into a copy of the checkpoint (v19-calmix) |
| [`bench_local.py`](bench_local.py) | Inference latency against the engine, by state length and question count, on any device |
| [`device_parity.py`](device_parity.py) | Whether two or more devices give one checkpoint the same answers: the largest probability difference and the answers that change, against the first device |
| [`jevbench/jevbench.sh`](jevbench/jevbench.sh) | The JevBench public set against `strands-decider serve` on one GPU, at the JevBench commit the script pins |
| [`jevbench/paired.py`](jevbench/paired.py) | A per-task comparison of two JevBench runs, with an exact McNemar test |
| [`jevbench/v15_proxy.py`](jevbench/v15_proxy.py) | A **local proxy** for JevBench v1.5 scoring on v1 public results, not the official score: yes/no answers with 0.2 < P(yes) < 0.8 count as wrong, yes/no and choice credit chance-corrected, score tasks graded by the expected level against chance (gold levels from the run's `all.jsonl`), the three question types weighted equally; the yes/no answers in that band; and a paired bootstrap against a baseline run |
| [`unseen/`](unseen/README.md) | Four question families no model trains on (StrategyQA, CommonsenseQA, ARC-Challenge, STS-B), plus RuleTaker at a held-out depth (5; depths 0-2 are trained on), pinned to dataset revisions, answered by a checkpoint or any other decider behind an adapter, and graded with the same v1.5 rules |
| [`vision/run.py`](vision/run.py) | Image input (`--vision`): NaturalBench and POPE, with and without the image, against two baselines ([vision/](vision/README.md)); `vision/temps.py`, `vision/compare.py` and `vision/text_check.py` fit image temperatures, compare runs and check text after image training |
| [`jevbench/jevbench_cold_warm.py`](jevbench/jevbench_cold_warm.py) | First-request against warm latency, one task asked twice (MPS) |

Paths are relative to the repository root unless they are links. A module path such as
`infer.py` is relative to `src/strands_decider/`.

## Internal evaluations

These commands are step 5 of the [training steps](../training/steps.md#5-evaluate):

```bash
strands-decider eval checkpoints/hobson-2b-recipe --data data/holdout_v5_norule.jsonl \
  --limit 6000 --out reports/recipe_heldout.json
python evaluation/multistep_eval.py checkpoints/hobson-2b-recipe
python evaluation/question_sensitivity.py checkpoints/hobson-2b-recipe
```

The first reports accuracy, **ECE** (expected calibration error), NLL and score MAE on
four held-out classification tasks, by task, primitive and confidence band. The
confidence-band table is the one to read: if accuracy above 0.9 is not clearly higher
than below 0.5, threshold routing buys you nothing. The second scores the multi-step
evaluation sets, HotpotQA among them as the transfer test. The third changes only the
question and checks the answer follows it ([Limitations](#limitations)).

These are internal checks. They do **not** rank models — on the comparisons that
mattered the held-out classification tasks pointed the wrong way ([Limitations](#limitations)).
Rank on JevBench ([External benchmark](jevbench.md#external-benchmark-jevbench-v1-public-set)).

## Limitations

- **Questions are read less than documents.** With the state and options fixed, v19
  gives the same answer to a changed question about 94% of the time. A question that
  reverses an obvious reading ("which does this NOT fit?") is often answered as if it
  did not. Phrase questions so the obvious reading is the intended one, and check with
  `evaluation/question_sensitivity.py`.
- **Long, multi-step documents are still the weak spot**: 0.505 on JevBench's hard tier
  against 1.000 on easy. `temporal_numeric` (0.267 for v19 and v14) looks
  capability-bound for a single-pass model.
- **Answer adequacy is learned, not solved**: 0.739 on HelpSteer2's held-out responses,
  catching three in four inadequate ones and accepting about seven in ten adequate ones.
- **v19 reads an unseen multi-hop source worse than v14**: HotpotQA 0.726 against
  0.750, though it is ahead on MuSiQue, ContractNLI and BoardgameQA. For multi-hop
  questions over sets of paragraphs unlike its training sources, v14's recipe
  (`configs/experiments/v14.yaml`) may be the better choice.
- **`score` is weak**: 0.617 in the training mix, 0.499 on an unseen rubric type. Train on
  a rubric resembling yours: the corpus is mostly valence judgements, and a register or
  severity rubric outside that does not transfer.
- **`noul` generalises poorly to genuinely novel yes/no tasks**: 0.614 against a 0.500
  baseline, though it is excellent once trained.
- **Calibration is one temperature per primitive**, fitted on held-out classification.
  Further from that distribution it drifts: on JevBench, v14's top confidence band was
  overconfident by about 0.16, v16's and v17's middle bands by about 0.08, and v18's and
  v19's by about 0.03. On our own held-out long documents and adequacy judgements v19 is
  under-confident by 0.19 to 0.35, and refitting for them breaks short classification.
  The confidence bands are only established for short classification: measure on your
  own traffic before trusting a threshold.
- **Training the Qwen3.5 torso needs Linux or WSL2** for its fused kernels. Serving runs
  on an Apple-silicon Mac with the same answers (measured with v19 on an M3 Pro), at about a ninth of the 3090's
  throughput on long prompts ([Serving on a Mac](../docs/inference.md#serving-on-a-mac)). Through MLX it is
  1.4 to 1.6x faster than MPS, measured on an M4 Pro
  ([Serving on a Mac with MLX](../docs/inference.md#serving-on-a-mac-with-mlx)).
- **A `slot` checkpoint caps a choice question at `num_slots` options** (24 by
  default), where the reference API allows 255. The `pointer` readout used from v7 on
  has no such cap.
- **Trained on public datasets**, so it inherits their domains and their label noise. For
  a specific task, add a recipe with your own data.
- **Do not rank models on this repo's own held-out corpus.** It measures fit to the
  training distribution: it ranked a clearly better model lower (4B 0.8450 internal /
  0.6840 external against 1.7B 0.8500 / 0.6407), and v9's rose to 0.895 against v7's
  0.835 while JevBench fell. Use it for training diagnostics and calibration; rank
  externally.
- **231 tasks cannot resolve an effect below about +0.04 accuracy.** One retrain moves
  JevBench public by about 3 tasks (six v17 retrains on AWS, SD 3.2 tasks,
  [Retraining on AWS](results.md#retraining-on-aws)). Treat smaller single-run differences as unresolved, not as
  results either way.
