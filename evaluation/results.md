# Results

The measurements of v1 (the released model), v21, v19 and the versions before them: the
seeds and release checks of v1 and v21, the AWS retrain, the summary by version,
calibration, and accuracy and latency on a Mac. [README.md](README.md) says how to run them, and
[jevbench.md](jevbench.md) has the external benchmark. Paths are relative to the repository
root unless they are links. A module path such as `infer.py` is relative to `src/strands_decider/`.

## v1, the released model

`StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` ([docs/naming.md](../docs/naming.md) explains
the name). What is new against v21:

- **A balanced data mix** of 246,678 rows, about twice v21's: v21's rows without the 20,000 from
  datasets whose terms restrict commercial use, 24,533 code-task rows, 96,638 rows from more
  public datasets for the decision types v21 had few rows for, and 22,168 new rule-application
  rows (apply a written policy to a case). In the rule-application rows, gemma-4-31B-it
  (Apache-2.0) wrote the policy text and the case files, and code computed the gold answers.
  Claude was used only as an independent check to filter rows: a row was kept only where gemma's
  and Claude's answers both agree with the code. Claude's text is not in the data.
- **A larger teacher:** gemma-4-31B-it's output distributions are training targets where its
  answer agrees with the gold label.
- **The v19 anchor:** training keeps the answers close to the released v19 (a KL term, weight
  0.3), not to the untrained base model.
- **A soup** of three training runs (below).

The four held-out sets are MuSiQue, ContractNLI and BoardgameQA, dev splits of sets whose train
splits are in the training data, and HotpotQA, which is never trained on (the mix has
2WikiMultihopQA, a similar multi-hop task). The configs are in the Hub repository
(`train_config.json`, `training/configs/`). The anchor setting, the soup command and the data
builders are not in this repository yet (pull requests to come), so v1 cannot be rebuilt from
`main` until they are merged.

This checkpoint is a **soup** of three training runs of one recipe (seeds 0, 1 and 2). Each layer's
LoRA update is the exact mean of the three runs' updates (stored as one rank-48 adapter), and the
three readout heads are stacked so that the soup's score for an option is the mean of theirs. The
soup was then calibrated like any trained checkpoint. A soup needs no choice between seeds. It
scores at or above its seeds' mean on most measures below.

Seeds: their training runs (8x A100). Soup: these files through the code repository's `main`
(one NVIDIA L4).

| measure | seed 0 | seed 1 | seed 2 | seeds, mean ± SD | **soup (this checkpoint)** |
| --- | --- | --- | --- | --- | --- |
| JevBench public, tasks right of 231 (window 4096) | 178 | 177 | 184 | 179.7 ± 3.8 | **180** |
| JevBench Brier | 0.305 | 0.288 | 0.289 | 0.294 ± 0.010 | **0.280** |
| JevBench ECE | 0.079 | 0.088 | 0.060 | 0.076 ± 0.014 | **0.072** |
| MuSiQue (dev split; train split trained on) | 0.882 | 0.894 | 0.887 | 0.888 ± 0.006 | **0.882** |
| ContractNLI (dev split; train split trained on) | 0.866 | 0.871 | 0.870 | 0.869 ± 0.003 | **0.878** |
| BoardgameQA (dev split; train split trained on) | 0.800 | 0.803 | 0.801 | 0.801 ± 0.002 | **0.810** |
| HotpotQA (never trained on) | 0.822 | 0.825 | 0.823 | 0.823 ± 0.002 | **0.832** |
| held-out short tasks (6,000) | 0.660 | 0.649 | 0.642 | 0.650 ± 0.009 | **0.653** |
| code tasks, held out (1,996) | 0.826 | 0.833 | 0.831 | 0.830 ± 0.004 | **0.836** |
| HelpSteer2 adequacy (234) | 0.718 | 0.714 | 0.705 | 0.712 ± 0.007 | **0.709** |
| generated adequacy (302) | 0.818 | 0.788 | 0.808 | 0.805 ± 0.015 | **0.828** |
| generated documents, v16's set | 0.849 | 0.849 | 0.863 | 0.853 ± 0.008 | **0.857** |
| generated documents, v18's set | 0.745 | 0.753 | 0.737 | 0.745 ± 0.008 | **0.781** |
| JevBench tiers, easy / standard / hard | 48 / 65 / 65 | 48 / 66 / 63 | 48 / 64 / 72 | | **48 / 67 / 65** |

On the Decision Index (balanced skill) the soup scores 30.82, measured with the training harness.

**Release checks.** The released files through the code on `main`, not the training harness:

| check | result |
| --- | --- |
| JevBench public (window 4096), on one NVIDIA L4 | 180/231, Brier 0.280, ECE 0.072, strict schema 1.000 |
| the same on CPU | the same answer on all 231 tasks |
| against the training harness (A100) | the same answer on 230 of 231 tasks (181/231 there); the one other task is a near tie (0.346 against 0.345) |
| internal sets (`training/recipe.sh eval`) and the code tasks, on the L4 | the soup column above, each within 2 items of the harness |
| README examples, `--device cpu` and `--device cuda` | the same answers; probabilities within 0.006 |
| code: test suite (CPU), ruff and mypy, as CI runs them | pass |
| images, `--vision` ([docs/vision.md](../docs/vision.md#v1)) | NaturalBench 0.776, POPE 0.872; hobson-v21 on the same GPU 0.785, 0.877 |

## v21, the previous release

`StrandsAgents/strands-decider-2B-hobson-v21` is v21b of the research notes:
[configs/experiments/v21b.yaml](../configs/experiments/v21b.yaml), v19's recipe plus v20's
checked question paraphrases and distillation from Qwen3.5-4B where it agrees with the gold
label (no catch-all rows, no instruction flips). Six seeds trained on one `p5.48xlarge`
host (8x H100, `NGPU=8`, FAST settings), all of which pass the validity gates: 231 JevBench
tasks attempted, strict schema 1.000, `/health` names the scored checkpoint at window 4096
before and after, easy tier 48/48, one epoch of 3,738 steps, every stage exit code 0.

**How the released seed was chosen.** The rule was written down before the seeds were
ranked, so that the release is a representative seed and not the luckiest one on the public
tasks. For each of five measures (JevBench tasks right, MuSiQue, ContractNLI, BoardgameQA,
HotpotQA), take the six-seed mean and SD. A seed's distance is the square root of the sum
of its squared z-scores over the five. Release the seed with the smallest distance (a tie
goes to the lower seed). Seed 5 has distance 1.48; the next are seed 3 (1.93) and seed 0
(1.98).

| measure | s0 | s1 | s2 | s3 | s4 | s5 (released) | six-seed mean (SD) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| JevBench public, tasks right of 231 (window 4096) | 177 | 171 | 173 | 177 | 176 | 176 | 175.0 (2.4) |
| JevBench Brier | 0.339 | 0.335 | 0.328 | 0.340 | 0.322 | 0.323 | 0.331 (0.008) |
| MuSiQue | 0.898 | 0.881 | 0.897 | 0.891 | 0.886 | 0.882 | 0.889 (0.007) |
| ContractNLI | 0.872 | 0.858 | 0.861 | 0.851 | 0.870 | 0.865 | 0.863 (0.008) |
| BoardgameQA | 0.804 | 0.820 | 0.818 | 0.810 | 0.793 | 0.821 | 0.811 (0.011) |
| HotpotQA (never trained on) | 0.750 | 0.751 | 0.730 | 0.764 | 0.771 | 0.746 | 0.752 (0.014) |
| held-out short tasks (6,000) | 0.642 | 0.645 | 0.645 | 0.636 | 0.645 | 0.650 | 0.644 (0.004) |
| HelpSteer2 adequacy (234) | 0.744 | 0.756 | 0.735 | 0.718 | 0.718 | 0.739 | 0.735 (0.015) |
| generated adequacy (302) | 0.818 | 0.811 | 0.811 | 0.798 | 0.785 | 0.788 | 0.802 (0.014) |
| generated documents, v16's set | 0.857 | 0.857 | 0.866 | 0.854 | 0.854 | 0.849 | 0.856 (0.006) |
| generated documents, v18's set | 0.761 | 0.757 | 0.745 | 0.773 | 0.765 | 0.741 | 0.757 (0.012) |

Against v19. On the host of the v21 ablation, six seeds of v19's recipe scored 170, 173,
170, 174, 175 and 175 (mean 172.8, Brier 0.341), and six of v21b 173, 174, 169, 170, 174
and 174 (mean 172.3, Brier 0.331). So on one host v21b has the same JevBench accuracy as
v19's recipe and a lower Brier score. The higher means above come from another host:
the six v21b seeds of the two hosts average 172.3 and 175.0, so compare seeds from one
host only. The published v19 is one run (167 at 3072, 168 at 4096).

**Release checks.** The released files were checked through the code on `main`, not the
training harness, on other GPUs (NVIDIA L40S for text, L4 for images):

| check | training time (H100, training harness) | release (`main`) |
| --- | --- | --- |
| JevBench public, tasks right (window 4096) | 176/231 | 176/231, the same answer on every task, from the training-format copy and from these files |
| JevBench Brier / ECE | 0.323 / 0.074 | 0.323 / 0.064 |
| MuSiQue / ContractNLI / BoardgameQA | 0.882 / 0.865 / 0.821 | 0.882 / 0.864 / 0.822 |
| HotpotQA (never trained on) | 0.746 | 0.745 |
| held-out short tasks (6,000) | 0.650 | 0.650 |
| generated adequacy (302) | 0.788 | 0.788 |
| images, `--vision` ([docs/vision.md](../docs/vision.md#v21)) | not run | NaturalBench 0.785, POPE 0.878 |
| code: test suite (CPU), ruff and mypy, as CI runs them | | pass |

Every accuracy is within one item of its training-time figure. JevBench ECE, binned over
231 tasks, moves with small probability changes between GPUs.

## Retraining on AWS

One retrain of the v19 recipe ran on a `p5.48xlarge` (8x H100) with `NGPU=8 FAST=1` and
seed 0. Its stage timings are in [Measured stage timings](../training/aws/README.md#measured-stage-timings).
The table compares its results with the original v19 run:

| | v19 | this run |
| --- | --- | --- |
| HelpSteer2 held-out responses (234, balanced) | 0.739 | 0.722 |
| HelpSteer2 inadequate / adequate recall | 0.761 / 0.718 | 0.761 / 0.684 |
| generated requests, unseen categories (302), balanced accuracy | 0.788 | 0.784 |
| MuSiQue / ContractNLI / BoardgameQA | 0.879 / 0.862 / 0.810 | 0.884 / 0.872 / 0.822 |
| HotpotQA, held-out source | 0.726 | 0.717 |
| generated documents, v16's set / v18's | 0.846 / 0.757 | 0.849 / 0.777 |
| JevBench Brier / ECE, at 3072 | 0.342 / 0.052 | 0.349 / 0.056 |

Against v19 per task at 3072, this run differs on 16 tasks, 8 each way (McNemar p = 1.0).
That passes the criterion set before the run: 157 to 177, p ≥ 0.05, and HelpSteer2 at
least 0.689. The recipe saves a 4096 window with the checkpoint, and `strands-decider serve` has
no flag to change it, so the 3072 score comes from a copy of the checkpoint whose saved
`max_length` is 3072, weights unchanged. The two windows differ on two `long_policy`
tasks, one each way.

**Run-to-run noise.** Six retrains of the v17 recipe on AWS scored 159, 160, 162, 165, 165
and 167 (seeds 0, 1 and 2; H100 and A100; with and without `FAST`): mean 163.0, SD 3.2
tasks. Any two of them disagree on 10 to 19 of the 231 tasks, and v17 disagrees with each
on 10 to 15. So one retrain moves JevBench public by about 3 tasks, and the difference
between two runs by about 4.5. v17's +1 over v16 is noise; its +10 over v7 is about two
standard deviations. The runs differ in hardware and DDP layout as well as seed, so this
is retrain noise, not seed noise alone. The internal sets across them: MuSiQue 0.880 to
0.897, ContractNLI 0.860 to 0.868, BoardgameQA 0.788 to 0.826, HotpotQA 0.708 to 0.751.

## Summary

Latency depends on the hardware and the prompt length. On an RTX 3090 under WSL2, v19
answers one JevBench question in a median 115 ms (95th percentile 299 ms). Warm on an M3
Pro, the median is 234 ms and the 95th percentile is 2,628 ms
([Serving on a Mac: accuracy and latency](#serving-on-a-mac-accuracy-and-latency)). On
tasks in its training mix, v14 reaches 0.961 on `noul` and 0.960 on `choice`. On tasks it
has never seen, `choice` is useful behind a confidence gate and `score` is not. On long, multi-step
documents v19 is level with the best earlier versions (hard tier 0.505, as v18). It is still
well short of the best systems.

**Reference recipe: v19** - v18's recipe (the classification corpus, 12,909 multi-step
documents trained toward v14's own answer distributions, and 3,815 generated questions
over realistic workplace documents) plus 6,166 answer-adequacy rows: a request, a
response, and whether the response adequately answers it, from HelpSteer2's human
ratings and from generated requests ([v19: answer adequacy](../research/history.md#v19-answer-adequacy)): **0.723** on
JevBench (167/231), the best of any model here until v20 (169/231, see
[PREREGISTRATION-v20.md](../research/preregistrations/PREREGISTRATION-v20.md#outcome-added-after-the-run)),
with the best Brier (0.342), ECE
(0.052), macro (0.725), ordinal MAE (0.427) recorded and the most standard-tier tasks
(63). It judges answer adequacy, which no earlier model could — 0.739 on HelpSteer2's
held-out responses (v18 0.483, a frozen Qwen3.5-4B 0.645) — and every other evaluation
held within about 0.02 of v18. It is the first run here to clear its own pre-registered
bar, and became the default by that rule. Its gain over v18 (+14 / -11 tasks) is inside
JevBench's noise; its Brier gain is not (lower on 147 tasks, higher on 84, p < 0.0001).
Recipe: `configs/train.yaml`, run end to end by `training/recipe.sh` under WSL2; about 11 h to
train, parent included.

**Earlier leaders.** **v18**, without the adequacy rows (0.710), still the best on
MuSiQue (0.892) and on generated documents from unseen domains (0.863). **v17**, without
the weak-skill questions either (0.710). **v16**, v17's rows with the frozen 4B's targets on the
multi-step rows (0.706). **v14**, without the generated questions (0.697), and still the
best model here on HotpotQA. **v13**, v7's recipe on the same Qwen3.5 torso (0.680).
**v7**, Qwen3-1.7B (0.667): `configs/train-v7.yaml` and `training/recipe_v7.sh`, which train on
Windows.

*Measured with JevBench's own harness at a pinned commit, on its public board's
tasks ([JevBench](https://benchmarkheaven.com/jev-models)): **0.723** on the 231 public
tasks. v19 is not on the board. Placed among the v1.4.2 board of 25 September 2026, that
score was 50th of 90 systems, above `decider-2b`'s
entry, and first of 30 at 2B parameters or fewer (third of 33 counting three models just
above 2B). See* [External benchmark](jevbench.md#external-benchmark-jevbench-v1-public-set)*.*

| | v5 | v6 | v7 | v13 | v14 | v16 | v17 | v18 | **v19** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| torso | Qwen3-1.7B | Qwen3-1.7B | Qwen3-1.7B | Qwen3.5-2B | Qwen3.5-2B | Qwen3.5-2B | Qwen3.5-2B | Qwen3.5-2B | **Qwen3.5-2B** |
| corpus | classification | classification | classification | classification | + multi-step, teacher | + generated documents | + replay toward v14 | + weak-skill questions | **+ answer adequacy** |
| JevBench accuracy | 0.6364 | 0.6494 | 0.6667 | 0.6797 | 0.6970 | 0.7056 | 0.7100 | 0.7100 | **0.7229** |
| standard tier (72) | 0.764 | 0.806 | 0.833 | 0.833 | 0.847 | 0.833 | 0.847 | 0.833 | **0.875** |
| hard tier (111) | | | 0.414 | 0.441 | 0.468 | 0.495 | 0.495 | **0.505** | **0.505** |
| macro | 0.6231 | 0.6355 | 0.6601 | 0.6847 | 0.6898 | 0.7154 | 0.7195 | 0.7106 | **0.7254** |
| ECE (calibrated) | 0.0924 | 0.0896 | 0.0678 | 0.0853 | 0.0855 | 0.0694 | 0.0828 | 0.0789 | **0.0522** |
| Brier | 0.4623 | 0.4623 | 0.4422 | 0.4645 | 0.4383 | 0.3961 | 0.3924 | 0.3882 | **0.3421** |
| ordinal MAE | 0.6595 | 0.6059 | 0.5922 | 0.6211 | 0.6446 | 0.6638 | 0.5708 | 0.6443 | **0.4270** |
| paraphrase consistency | 0.6389 | 0.7778 | 0.8333 | 0.8333 | **0.8611** | 0.8333 | 0.8056 | 0.8333 | **0.8611** |
| p50 / p95 latency | 247 / 590 ms | 234 / 578 ms | 246 / 583 ms | 106 / 291 ms* | 111 / 296 ms* | 106 / 287 ms* | 100 / 289 ms* | 107 / 298 ms* | 115 / 299 ms* |

\* v13 onwards served under WSL2, the others on Windows; the latencies are not a
like-for-like comparison. JevBench asks one question per task, so none of these
figures involves the shared-prefix cache ([Asking many questions is nearly free](../docs/inference.md#asking-many-questions-is-nearly-free)).

Every metric improved from v5 to v7. v7 ties `kev 0.6B` (0.6667), the system whose
readout it adopts, and sits above `kev 4B` (0.662) and `Open-Jev 2B` (0.645). The
Qwen3.5 torso (v13) added three tasks, all in the hard tier; the multi-step corpus (v14)
three more there and one in the standard tier, and recovered the Brier score; the
generated documents (v16) three more in the hard tier and one fewer in the standard,
with the best Brier and macro yet; replay toward v14 (v17) one more in the standard
tier, and the lowest ordinal MAE in this table; the weak-skill questions (v18) one more
in the hard tier and one fewer in the standard; and the answer-adequacy rows (v19) three
more in the standard tier, with every calibration measure in the table at its best.

Cumulatively v5 to v7 is +20 tasks against -13, McNemar p = 0.296. Not significant,
and per [The resolution limit](../research/history.md#the-resolution-limit) it could not have been at this scale; what
supports it is six independent metrics moving the same way twice in a row.

**How well does it generalise?** Measured two ways: over the 21 classification tasks in
the training mix (v7 and v14), and over four tasks none has seen (`emotion`,
`hate_severity`, `massive_intent`, `sarcasm`), 6,000 sampled rows each.

| primitive | in the mix, v7 | in the mix, v14 | never seen, v7 | never seen, v14 | never seen, v16 | never seen, v17 | never seen, v18 | never seen, v19 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `noul` - yes/no | 0.940 | **0.961** | **0.629** | 0.614 | 0.592 | 0.613 | 0.613 | 0.596 |
| `choice` - pick one of N | 0.947 | **0.960** | 0.728 | **0.729** | 0.723 | 0.726 | **0.729** | 0.725 |
| `score` - rate on a rubric | 0.578 | **0.617** | 0.492 | 0.432 | 0.429 | 0.453 | 0.451 | **0.499** |
| *overall* | *0.866* | ***0.888*** | ***0.653*** | *0.637* | *0.629* | *0.640* | *0.641* | *0.647* |

v14 fits the training mix better than v7 on every primitive, and none of v14 and v16 to
v19 generalises to unseen *classification* tasks better than v7 overall. v19 comes
closest, with the best unseen `score` yet. Their gains are on long, multi-step documents
and judgements ([v14](../research/history.md#v14-multi-step-documents-with-a-teacher) to [v19](../research/history.md#v19-answer-adequacy)), not on short classification.

RuleTaker's held-out depths are excluded from the right column. They score well, but
they share a generator and vocabulary with the trained depths, so counting them would
measure in-family transfer and report it as generalisation (see
[v5](../research/history.md#v5-compositional-training-teaches-the-task-not-the-capability)).

The left column means *this kind of task was in training*, not that these specific
examples were memorised. It is measured over the training corpus, so most sampled rows
were seen during training - but the slice held out within training scores within noise
of the mixed sample, so there is no memorisation gap to speak of at one epoch with ~1%
of parameters trainable.

Per task with the task in the mix, v14: `lang_id` 1.000, `clinc150` 0.997 (150
intents), `dbpedia` 0.996, `spam` 0.988, `banking77` 0.983 (77 intents), `mnli` 0.982,
`paws` 0.965, `toxicity` 0.937, `ag_news` 0.934, `boolq` 0.927. That is usable for
routing, triage, filtering and intent detection.

**Your own task will land between those two columns**, depending on how close it is to
something already in the mix - unseen `massive_intent` reaches **0.882** because
`banking77` and `clinc150` are nearby, while unseen `emotion` manages 0.572 because
nothing in training classifies affect, and `hate_severity` 0.432 because it is a score
rubric unlike any trained one. Adding your task to the corpus and retraining moves you
to the left column; using a checkpoint trained with this recipe on a novel task puts you
in the right one.

**Confidence tracks accuracy on short held-out classification.** Measured on the
four tasks neither model trained on:

| confidence | share, v7 | accuracy, v7 | share, v14 | accuracy, v14 | share, v16 | accuracy, v16 | share, v17 | accuracy, v17 | share, v18 | accuracy, v18 | share, v19 | accuracy, v19 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| >= 0.9 | 24% | **0.956** | 23% | **0.955** | 24% | **0.950** | 22% | **0.950** | 21% | **0.961** | 23% | **0.952** |
| 0.5 - 0.9 | 29% | 0.694 | 29% | 0.691 | 27% | 0.680 | 32% | 0.688 | 30% | 0.710 | 34% | 0.655 |
| < 0.5 | 48% | 0.477 | 48% | 0.449 | 49% | 0.446 | 46% | 0.456 | 48% | 0.457 | 43% | 0.474 |

So on these tasks the [routing convention](../docs/architecture.md#the-routing-convention) holds: act on the top
band, send the rest to review. It holds on unseen short classification tasks too — the model flags what
it does not know instead of guessing confidently. It held less well far from the training distribution for v14, whose answers
at 0.9 confidence or above on JevBench were right about 0.16 less often than claimed;
v16's were right 0.981 of the time at a claimed 0.961, v17's 0.940 at a claimed 0.948,
v18's 0.963 at a claimed 0.948, and v19's every one of 53 at a claimed 0.957
([Calibration: v16 to v19 fixed the top band](jevbench.md#calibration-v16-to-v19-fixed-the-top-band)). Calibrate on data like your own traffic
all the same.

**These bands are measured on short classification, and that is the only place they
hold.** On long documents, generated workplace questions and answer-adequacy judgements,
v19 is *under*-confident: right 0.83 of the time on multi-step documents at a stated
0.64, and 0.72-0.86 on adequacy at a stated 0.40-0.51
([PREREGISTRATION-v19-calmix.md](../research/preregistrations/PREREGISTRATION-v19-calmix.md)). That errs on the safe
side — fewer answers clear an automation threshold, not more wrong ones — but a 0.9
threshold on such inputs will send far more to review than it needs to. Refitting the
temperatures on those inputs fixes them and breaks short classification; no single
temperature per primitive serves both.

**Practical shape.** v19: a 1.9B torso, 12.4 GiB peak and about 6 h for one epoch over
123k rows on one RTX 3090 under WSL2, plus about 5 h for its parent (v7: 1.7B, 6 GiB, 3.0 h on Windows). The pointer
readout has no fixed option count. The base weights, a download of about 4.5 GB, are
fetched from HuggingFace at load; the checkpoint itself is ~88 MB.

## Calibration

Temperatures are fitted per primitive against **ECE** on the `calib` half of the
held-out classification tasks, and scored on the disjoint `test` half. v19's fitted
values are `noul` 0.91, `choice` 0.77, `score` 0.96 (v18: 0.70, 0.82, 1.56; v17: 0.82,
0.73, 1.19; v16: 0.96, 0.70, 1.74; v14: 0.59, 0.77, 1.56): below 1 sharpens, so on yes/no
and choice the trained model comes out *under*-confident and calibration raises its
confidence. v19's are the closest to 1 yet. Held-out ECE after fitting is 0.054 —
`choice` 0.040, `score` 0.031, `noul` 0.162 (v18: 0.052; v17: 0.051; v16: 0.057; v14:
0.051).

Three limits, all still current:

- **`noul` stays poorly calibrated within bins** (ECE 0.162) even though its mean
  confidence tracks accuracy; one scalar cannot fix within-bin miscalibration.
- **Calibrating `score` changes its answer.** Temperature cannot move a `choice` argmax,
  but a score is `sum(i * p_i)`, so softening or sharpening moves it.
- **One calibration set cannot serve two distributions.** Fitted on held-out
  classification, where the top band is right 0.955 of the time, it left v14
  overconfident at the top on JevBench; v16 to v19 are not, but their middle bands
  claim more than they deliver ([Calibration: v16 to v19 fixed the top band](jevbench.md#calibration-v16-to-v19-fixed-the-top-band)).

## Serving on a Mac: accuracy and latency

Measured with v19 on an M3 Pro (36 GB) running macOS 26.6: torch 2.7.1, transformers
5.17.0, bf16 on MPS. The setup and the MPS kernel are in
[Serving on a Mac](../docs/inference.md#serving-on-a-mac).

**Accuracy is unchanged.** JevBench's official harness against `strands-decider serve --device
mps` scores 168/231, Brier 0.3419, ECE 0.0491, paraphrase consistency 0.861, schema
validity 1.000. Those are the 3090's figures at the 4096 window to within 0.002 in Brier
and ECE (3090: 0.3416, 0.0507)
([The context window is not the ceiling](jevbench.md#the-context-window-is-not-the-ceiling)), with the same single flip on `long_policy-01` against the
pre-registered 3072 run. The README example gives noul 0.801, `technical` 0.750 and score
1.24 (confidence 0.578), within bf16 noise of the output recorded in
[Ask](../docs/inference.md#ask).

**Latency: each input length pays a one-off compile.** MPS compiles per tensor shape,
and nearly every JevBench task has a unique token count. Asking every task twice, back
to back on a fresh server (`evaluation/jevbench/jevbench_cold_warm.py`), gives the same answers on
231 of 231 and separates the two costs:

| tokens | tasks | first request | repeated | 3090 | repeated vs 3090 |
| --- | --- | --- | --- | --- | --- |
| < 300 | 136 | 310 ms | 153 ms | 112 ms | 1.4x |
| 300-1,000 | 52 | 1,644 ms | 449 ms | 115 ms | 3.9x |
| 1,000-2,500 | 13 | 2,851 ms | 2,172 ms | 228 ms | 9.5x |
| 2,500-5,000 | 29 | 3,836 ms | 2,514 ms | 286 ms | 8.8x |
| **all, p50 / p95** | 231 | **622 / 3,862 ms** | **234 / 2,628 ms** | 115 / 299 ms | |

Medians. The 3090 column is v19's recorded JevBench run under WSL2, where each task was
asked once. The compile adds a median 90 ms under 300 tokens and over a second for long
documents. Padding prompts to a few fixed lengths and warming those up at startup would
remove most of it; that is not implemented. What remains is compute: warm, long prompts
run at about 1,000-1,200 tokens/s, roughly a ninth of the 3090.

**Asking many questions is still nearly free, against encoding each prompt, and on long
states.** The shared-prefix cache forks the hybrid
torso's recurrent state on MPS as it does on CUDA, and gives the same answers as
batched encoding. From `evaluation/bench_local.py`, eight choice questions, median of 3, a
few shapes repeated so these are warm figures:

| state tokens | 1 question | 8, shared prefix | 8, batched |
| --- | --- | --- | --- |
| 256 | 319 ms | 972 ms | 2,620 ms |
| 1,024 | 1,029 ms | 1,697 ms | 8,300 ms |
| 2,048 | 1,989 ms | 2,682 ms | 16,188 ms |
| 4,000 | 3,998 ms | 4,774 ms | 31,747 ms |

**CPU is slower than MPS everywhere, and wants fp32.** On the M3 Pro's CPU, bf16 takes
7.7 s for a 256-token question and fp32 3.5 s, with the same answer, so the engine
casts a half-precision torso to fp32 whenever it serves on CPU. That costs memory:
peak footprint 8.3 GB for `strands-decider ask --device cpu`, against 5.4 GB on MPS. In fp32,
MPS is 4-10x faster:

| state tokens | questions | CPU (fp32) | MPS |
| --- | --- | --- | --- |
| 256 | 1 | 3,252 ms | 319 ms |
| 256 | 8, shared prefix | 9,200 ms | 972 ms |
| 1,024 | 1 | 5,424 ms | 1,029 ms |
| 1,024 | 8, shared prefix | 12,197 ms | 1,697 ms |

Peak memory on MPS is about 5.4 GB, most of it the 3.5 GiB of bf16 torso weights.

```bash
python evaluation/bench_local.py checkpoints/hobson-2b-recipe --device mps --out mps.csv
python evaluation/bench_local.py checkpoints/hobson-2b-recipe --device cpu \
  --lengths 256,1024 --questions 1,8 --reps 2 --out cpu.csv
python evaluation/jevbench/jevbench_cold_warm.py double all.jsonl doubled.jsonl   # then run the harness
python evaluation/jevbench/jevbench_cold_warm.py split doubled.jsonl OUT/results.jsonl OUT/
```

`split` writes `cold.jsonl` and `warm.jsonl`. Each is summarised against the unmodified
public task file, and `split` refuses a pair whose two requests differ in length.

## Serving on a Mac through MLX: accuracy and latency

Measured with v19 on an M4 Pro (20-core GPU, 48 GB), bf16 on both devices: torch 2.7.1 and
transformers 5.18.0 for MPS, mlx 0.32.3 for MLX. All MLX figures are from mlx-lm 0.32.0, the version the extra requires. The
setup is in [Serving on a Mac with MLX](../docs/inference.md#serving-on-a-mac-with-mlx).
JevBench was not run on MLX.

**The answers are the same.** `evaluation/device_parity.py` asks 54 questions: three tickets,
three state lengths from about 40 to about 3,000 tokens, one and five questions per request,
and every question type. It runs them on the CPU in fp32 as the reference, then on each other
device. No chosen answer changes. The largest probability differences against that reference:

| run | max \|Δp\| |
| --- | --- |
| MPS, adapter unmerged (the torch engine) | 0.0051 |
| MPS, adapter merged on the CPU before the move, as the MLX engine merges it | 0.0105 |
| MLX in fp32 | 0.0036 |
| MLX in bf16 (`--device mlx`) | 0.0138 |

The two middle rows were one-off runs with the change named. Merging the adapter into bf16
weights accounts for most of MLX's difference. The README's multi-question example gives noul
0.831 and `billing` 0.846 (confidence 0.769), against 0.829 and 0.846 (0.769) recorded there.

**Latency: 1.4 to 1.6x faster than MPS.** From `evaluation/bench_local.py`,
medians over 7 warm runs for one question and 5 for several:

| state tokens | input tokens | questions | path | MPS | MLX |
| --- | --- | --- | --- | --- | --- |
| 128 | 222 | 1 | whole prompt | 162 ms | 113 ms |
| 1,024 | 1,118 | 1 | whole prompt | 682 ms | 486 ms |
| 4,000 | 4,094 | 1 | whole prompt | 2,685 ms | 1,764 ms |
| 256 | 605 | 4 | shared prefix | 474 ms | 299 ms |
| 256 | 1,639 | 16 | shared prefix | 1,154 ms | 723 ms |
| 1,024 | 1,373 | 4 | shared prefix | 907 ms | 627 ms |
| 1,024 | 2,407 | 16 | shared prefix | 1,622 ms | 1,060 ms |
| 1,024 | 17,902 | 16 | batched | 11,542 ms | 7,396 ms |

The batched comparison stops at 1,024-token states to bound memory: at 2,048 tokens and 16
questions it is one 34,000-token batch.

**Memory per request is lower on MLX.** MLX's peak (`mx.get_peak_memory`, reset before each
shape) runs from 3.8 GiB to 4.9 GiB over these requests; the MPS driver's allocation runs from
4.2 GiB to 6.3 GiB, and to 9.5 GiB for 16 batched 1,024-token prompts, where MLX peaks at
4.9 GiB. Loading peaks at 3.7 GiB on MLX, while the adapter is merged.

```bash
python evaluation/bench_local.py StrandsAgents/strands-decider-2B-hobson-v19 --device mps \
  --device mlx --lengths 128,1024,4000 --questions 1 --reps 7 --out single.csv
python evaluation/bench_local.py StrandsAgents/strands-decider-2B-hobson-v19 --device mps \
  --device mlx --lengths 256,1024 --questions 4,16 --reps 5 --out multi.csv
python evaluation/device_parity.py StrandsAgents/strands-decider-2B-hobson-v19 \
  --device cpu --device mps --device mlx --out parity.json
```

