---
name: customize-decider
description: >
  Guide a user from "I have a decision in my application" to a Strands Decider model that is
  measured on their own data, and trained only if the measurement says so. Use when a user
  wants to evaluate Strands Decider on their use case, compare the released models, decide
  whether to train a customized model, build a training set from their data, run and monitor
  the training, calibrate it, or ship and check it. Also use when a user asks "will the decider
  work for my task", "how do I train the decider on my data", "which size should I use", or
  "is my trained model better".
---

# Customize Strands Decider for a user's decisions

The purpose: help a user find out, with numbers from their own data, whether a released decider
already does their job, and if not, train one that does. Measure first. Train only when the
measurement shows a gap that training can close.

You are the user's guide. The user owns two kinds of decisions; ask before each one:

- **Money:** every GPU host, its shape, region and hours.
- **Data:** anything that moves their data off their machine (a cloud host, an LLM API used for
  labelling, an S3 bucket), and who may label it.

The reference files in `references/` hold the detail; read each one when its phase starts.

| phase | what you produce | gate to pass before the next phase |
|---|---|---|
| Setup | a repository checkout and a work directory | `strands-decider --help` runs from the checkout |
| 0. Fit | one paragraph: is a decider the right tool | the decision has a fixed set of options |
| 1. Task card | `work/task-card.md` | the user agrees with every question and option text |
| 2. Evaluation set | `work/eval/` frozen with hashes | enough items, labels audited, every row passes the row check |
| 3. Baselines | `work/baselines.md` | every candidate size and the user's current method scored on the same items |
| 4. Decision | `work/decision.md` | the user picks the customization tier |
| 5. Training data | `work/train/` frozen with hashes | row check passes, overlap with the evaluation set is zero |
| 6. Training | checkpoints of 3 seeds | smoke run passes, loss curves healthy |
| 7. Soup, calibrate, compare | `work/results.md` | the new model beats the baseline by more than the noise |
| 8. Ship and watch | serving command, thresholds, a re-check plan | the user accepts the thresholds |

Write a short dated log as you go (`work/LOG.md`): the exact command, what you saw, with numbers
and paths, and what you concluded.

## Setup

The skill needs the repository, not only the PyPI package: the PyPI package serves models only.
Training, evaluation, calibration, soups and the corpus tools are modules under `training/`, run as
`python -m training.<module>` from the checkout root.

```bash
git clone https://github.com/strands-labs/strands-decider && cd strands-decider
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[train,dev]"        # serving only: pip install -e .
```

Paths in this skill are relative to the root of that checkout, and every command runs from there;
run the scripts in `work/` with `PYTHONPATH=.` so that they can import `training`. Create the work directory outside
the checkout (for example `../decider-work`, called `work/` below) and keep it out of git: it
holds the user's data. Run commands from the checkout root, with `work/` pointing there
(`ln -s ../decider-work work` is the simplest way).

## Phase 0. Is a decider the right tool?

A decider reads a state (text, structured data, or an image for the Qwen3.5 model) and a question,
and returns a probability distribution over options in one forward pass. It does not generate
text. Three question types:

- `noul`: yes/no, returns P(yes).
- `choice`: one of N named options (2 to 255), returns the pick, every option's probability and a
  confidence.
- `score`: an ordered rubric of 2 to 10 levels, returns the expected level and a confidence.

Good fits: routing, triage, tool selection, guardrails and policy checks, answer-adequacy checks,
grading against a rubric. The application can act on confident answers and send the rest to an
LLM or a person.

Poor fits: the answer is free text; the options are not known before the call; the decision needs
many steps of reasoning that are not in the state; the state is much longer than the serving
window (4,096 tokens by default; see `docs/inference.md`).

If it is a poor fit, say so and stop.

## Phase 1. The task card

Interview the user, then write `work/task-card.md`. One section per decision:

- The decision in one sentence, and what the application does with each answer.
- The question type, the exact instruction text, and the options. For `choice`, give each option
  a one-line description; for `noul`, describe what "true" and "false" mean. The model reads the
  descriptions.
- What the state contains, and an example state.
- The cost of each kind of error (for example, a false "urgent" wakes a person at night, a missed
  "urgent" loses a customer). This sets the thresholds in Phase 8.
- Where the labels come from (next section).

**Where labels come from.** Look for an outcome the user already records:

- A final outcome is a better label than a first guess. For example, if tickets record the team
  that finally closed them and whether they were reassigned, the final team is the label, and the
  reassignment rate of the current process is a free baseline (its accuracy is about
  1 − reassignment rate). Keep the reassigned items as their own `task` in the evaluation set: they
  are the hard cases.
- If no field holds the label (for example "is this urgent?" when nobody recorded urgency), the
  label must be made. Plan it with the user: how many items (Phase 2 sizes), who labels them, and a
  one-page label guide with examples. A released decider's answers can pre-label the items for
  people to correct, which is fast; keep a few hundred items labelled blind to measure how much the
  pre-labels bias the people. An external LLM as a labeller sends the data off the machine: ask.

Try the questions on real examples with a released model before you build anything. `ask`
does not send option descriptions (`--choice` sets every description to empty, and option names
cannot contain commas), so test through the server, which takes the full request:

```bash
strands-decider serve StrandsAgents/strands-decider-E4B-gemma4-v1-2610 --port 8000 &
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days!",
  "questions": {
    "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
             "criteria": {"billing": "payments, payouts, invoices", "sales": "pricing, new accounts"}},
    "urgent": {"type": "noul", "instructions": "Does the customer need help today?",
               "criteria": {"true": "money is blocked or the service is down", "false": "it can wait"}}}}'
```

Revise the wording with the user on 20 to 50 real examples. Rewording the instructions and the
descriptions is free and often moves accuracy by several points.

## Phase 2. The evaluation set, before any training

Read `references/evaluation.md` and `references/data-format.md`. In short:

- Build the evaluation set from the user's real data, in the row format of `data-format.md`.
  Audit the labels with a second labeller.
- Size: at least 1,000 test items per decision, plus about as many for calibration. `eval` and
  `calibrate` split one file into two disjoint halves of about equal size.
- Split rule (one rule, applied in this order):
  1. Time cutoff: the most recent period is the evaluation set, the older data is for training.
  2. Then drop from the training side every group (customer, document, conversation) that also
     appears in the evaluation set.
  3. All rows of one state (several questions about one ticket) stay on the same side.
- Run the row check of `data-format.md` on the file. It catches rows that would stop `eval` or
  training (bad option names, too few options, a state longer than the window).
- Freeze it: write `work/eval/SHA256SUMS` and do not change the files afterwards. A changed
  evaluation set makes every earlier number incomparable.

## Phase 3. Baselines

Candidates: every released model the user can serve (2B Qwen3.5; Gemma 4 E2B, E4B, 12B, 26B-A4B;
README "Gemma 4 models" gives memory and latency), and the user's current method.

On an NVIDIA GPU, score with the CLI. Pass `--limit` above the file size (the default is 4,000):

```bash
python -m training.evaluate eval StrandsAgents/strands-decider-E4B-gemma4-v1-2610 \
  --data work/eval/eval.jsonl --limit 1000000 --out work/baselines/e4b.json
```

`eval` and `calibrate` take `--device cuda|mps|cpu` (default: the best available), and run the
model as `serve` does on that device. For per-item results (the paired comparison of Phase 7), use
the script in `references/evaluation.md`.

`eval` prints accuracy, ECE and NLL, by kind, by task and by confidence band. Score the user's
current method on the same test-half rows with their own code, and record its latency and cost.

Then recalibrate the best candidates on the user's data. This needs no training and is often the
largest cheap gain in calibration. `calibrate` writes into the checkpoint and refuses a Hub id, so
download a local copy first:

```bash
hf download StrandsAgents/strands-decider-E4B-gemma4-v1-2610 --local-dir work/models/e4b
python -m training.evaluate calibrate work/models/e4b --data work/eval/eval.jsonl --limit 1000000
python -m training.evaluate eval work/models/e4b --data work/eval/eval.jsonl --limit 1000000 --out work/baselines/e4b-cal.json
```

It fits on the `calib` half; `eval` scores the `test` half.

Write `work/baselines.md`: one row per candidate, with accuracy (and its 95% interval), ECE, the
share of answers at confidence ≥ 0.9 and their accuracy, latency, and memory.

## Phase 4. Decide the customization tier

Read the table with the user. Climb this ladder and stop at the first rung that meets the user's
target:

1. **Reword** the instructions and descriptions (Phase 1). Re-score.
2. **Recalibrate** a released model on the user's data (Phase 3).
3. **Choose a larger released size.** A larger base model usually gains more than training a
   small one, and needs no training data.
4. **Fine-tune the released model** of the chosen size on the user's rows plus the public corpus
   (`init_from`, Phases 5 and 6). It must beat the released model on the user's test half to be
   worth serving.

Tier 4 is worth it when all of these hold: the gap to the target is larger than the noise of the
evaluation set (Phase 2 interval); the user can produce enough correct training labels (a few
thousand rows per decision is a sensible start, and gains usually flatten in the tens of
thousands); and the user accepts the cost of a GPU host (`references/training.md`). Fine-tune
the released model of the size the user will serve. Smaller models gain more from training than
larger ones.

Write `work/decision.md`: the tier, the reason in numbers, the target to beat, and the cost. The
user decides.

## Phase 5. Build the training data (tier 4)

Read `references/data-format.md`.

- Convert the user's labelled examples to rows, from the training side of the Phase 2 split only.
- Check the overlap with the evaluation set (`data-format.md`): no shared group, no duplicate or
  near-duplicate state. Report the counts; both must be zero.
- Do not reweight labels by default: weights shift the model's priors, and calibration cannot
  undo a shifted prior. Reweight only if the minority labels fail in Phase 7, and re-check
  calibration after.
- Give each question two to four `instruction_variants`, so the model learns the task, not one
  phrasing.
- Add the public corpus (`references/training.md`, "Build the public corpus"). It keeps the model's
  general skills and its calibration on unfamiliar inputs.
- Run the row check, then `python -m training.data stats work/train/train.jsonl` and
  `python -m training.data peek work/train/train.jsonl --n 10`, and show the user what the model will
  see.
- Freeze `work/train/` with `SHA256SUMS`.

## Phase 6. Train

Read `references/training.md` and `training/AGENTS.md` (host rules: spend is the user's decision;
one name tag per agent; tear down every host).

1. Choose the host for the size (`training.md`, "Hosts"). Ask the user, with the price.
2. Copy the frozen data to the host (`training.md`, "Data to the host"). Ask the user first.
3. Write the config from the template. Smoke run: 20 steps; check the lines listed in
   `training.md`.
4. Train three seeds. One seed is not a result: the seed-to-seed standard deviation of one recipe
   is about 3 items of a 231-item benchmark.
5. Monitor while it trains (`training.md`, "Monitoring"). Stop a diverged run; do not average it
   into a soup.

## Phase 7. Soup, calibrate, compare

```bash
python -m training.soup work/ckpt/s0 work/ckpt/s1 work/ckpt/s2 --out work/ckpt/soup
python -m training.evaluate calibrate work/ckpt/soup --data work/eval/eval.jsonl --limit 1000000
python -m training.evaluate eval work/ckpt/soup --data work/eval/eval.jsonl --limit 1000000 --out work/results/soup.json
```

Compare the soup with the best Phase 3 candidate on the same test-half rows: the accuracy
difference with McNemar's test on per-item results (`evaluation.md`), ECE, and accuracy at the
Phase 8 thresholds. Check also that the model did not lose what the user still needs (a
general-skills set, or the evaluations in `evaluation/README.md`). Write `work/results.md`. If the
gain is inside the noise, say so: the training did not help, and the released model stays.

## Phase 8. Ship and watch

- Serve: `strands-decider serve <checkpoint> --port 8000` (`docs/inference.md` for devices,
  batching and the window). Always serve a calibrated checkpoint.
- Thresholds for `choice` and `score`: from the test-half `by_confidence_band` table and the error
  costs in the task card, choose the confidence at which the application acts without a check. The
  documented default bands (act at ≥ 0.9, confirm from 0.5 to 0.9, send to a person below 0.5) are
  a start, not a rule.
- Thresholds for `noul`: the answer is P(yes), with no separate confidence. Choose one or two cut
  points on P(yes) from the error costs: for example, page a person above 0.8, queue for review
  between 0.4 and 0.8, ignore below 0.4. Measure the hit rate and false-alarm rate of each cut on the
  test half.
- Watch: log every answer with its probabilities. Each month, or after a change in the traffic,
  label a fresh sample and re-run Phase 3 on it. A fall in accuracy at the same confidence means the
  traffic moved; recalibrate or retrain.

## Rules

- Measure before you train, and compare on the same frozen items every time.
- Never train on evaluation data, and never edit the evaluation set after Phase 2.
- Report intervals, not single numbers. State when a difference is inside the noise.
- Ask before every action that costs money or moves the user's data off their machine.
- Do not claim a gain that the paired comparison does not show.
