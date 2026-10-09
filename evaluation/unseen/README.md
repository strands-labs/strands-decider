# Unseen task families

A local check of how a decider does on four question families no model trains on, plus
RuleTaker at a held-out depth (depths 0-2 are trained on)
(see [data/sources.md](../../data/sources.md)), graded with JevBench v1.5's published rules.
It is our own set, not a JevBench score.

| Script | What it does |
| --- | --- |
| [`build.py`](build.py) | Builds the 1,050 rows: StrategyQA (250 yes/no), RuleTaker depth 5 (250 yes/no, from `strands-decider data build`'s held-out file), CommonsenseQA (150 choice), ARC-Challenge (150 choice), STS-B (250 score, 6 levels); every dataset pinned to a revision, one seeded draw, a manifest of revisions and versions beside the output |
| [`run.py`](run.py) | Answers every row with one decider and stores its probabilities in the row's option order, with a `run_meta.json` (the decider, its temperatures and base revision, the rows' sha256, library versions) |
| [`score.py`](score.py) | Competence per question type and Intelligence by v1.5's rules (yes/no inside 0.2-0.8 counts wrong; chance-corrected credit; score graded by the expected level), top-probability ECE, yes/no over-confidence, per family; `--vs` compares runs row by row with a paired bootstrap 95% interval |

```bash
pip install "strands-decider[train]"          # build.py needs datasets
python evaluation/unseen/build.py --out data/unseen.jsonl
python evaluation/unseen/run.py --rows data/unseen.jsonl --checkpoint CKPT --out reports/unseen/NAME
python evaluation/unseen/score.py reports/unseen/NAME --vs reports/unseen/BASELINE
```

`run.py --checkpoint` answers in process exactly as `strands-decider serve` would, with the
checkpoint's own temperatures; `--vision` loads it as `serve --vision` does, for a
checkpoint served that way.

## Other deciders

`run.py --adapter` runs any decider whose answers can be read as System One answers
([docs/inference.md](../../docs/inference.md)): `noul` (P(yes)) for a yes/no question,
`probabilities` keyed by option name for a choice, and by level index (`"0"`, `"1"`, ...)
for a score.

- `--adapter http://HOST:PORT`: a server speaking the System One API (`POST /v1/systemone`).
- `--adapter python:MODULE:NAME [--path DIR]`: `MODULE.NAME()` returns a function
  `ask(state, question) -> answer`, where `question` is a System One question as JSON
  (`type`, `instructions`, `criteria`). It runs the decider in process through its own
  published code, with its own calibration.

For example, a decider whose published package exposes a System One-shaped `system_one`
method (the shape of Mapika/decider-2b's `decider.infer.Decider`) needs only:

```python
# my_adapters.py
def decider_2b():
    from decider.infer import Decider

    d = Decider("/path/to/a/download/of/the/repo")   # downloaded at a pinned revision
    return lambda state, q: d.system_one(state, {"q": q})["answers"]["q"]
```

```bash
python evaluation/unseen/run.py --rows data/unseen.jsonl --adapter python:my_adapters:decider_2b \
  --path . --out reports/unseen/decider-2b
```

A decider with another answer shape maps it in the same function. Pin the decider's own
download to a revision and record it with the results; `run.py` records only the adapter.

## Reading the numbers

One draw of 1,050 rows: differences of a few points are within noise; use `--vs`. The
recorded unseen-family numbers in the results log were computed by an exploratory script
that differed in two details: choice competence was chance-corrected with the mean chance
level over the family rather than per question, and two CommonsenseQA questions with a
repeated option were scored as wrong (`build.py` keeps a repeated option once, by name).
Both details affect only choice competence. The recorded per-row answers are not
published, so this has not been re-scored here: expect `score.py` on a fresh run to differ
slightly from the recorded choice numbers.
