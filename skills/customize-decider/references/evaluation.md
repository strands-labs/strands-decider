# Evaluation: design, size and comparison

## What to measure

- **Accuracy** on the test half, per decision (`task`) and overall.
- **Calibration**: expected calibration error (ECE), and the accuracy of the answers at
  confidence ≥ 0.9. The application uses the confidence to decide when to act alone, so a
  confident wrong answer is the failure that matters.
- **Coverage at a threshold**: the share of answers at or above the chosen confidence.
- **Latency and memory** on the hardware the user will serve on.

`python -m training.evaluate eval` prints accuracy, ECE, NLL and score MAE, by kind, by task and by
confidence band, and `--out` writes the full report as JSON.

## The two halves

`calibrate` and `eval` split one file by a hash of the row position into two halves of about
equal size (`--split calib` and
`--split test`, the defaults of each command). They never overlap, so calibration is fitted on
rows that the reported numbers do not use. Pass the same file to both. The split is by row, not by
group: that is correct for calibration, but the train/evaluation split must be by group (see
`data-format.md`).

Pass `--limit` above the file size: the default limit is 4,000 rows.

## How many items

The 95% interval of an accuracy p on n items is about ±1.96·√(p(1−p)/n):

| test items | interval at 80% accuracy |
|---|---|
| 200 | ±5.5 points |
| 500 | ±3.5 points |
| 1,000 | ±2.5 points |
| 4,000 | ±1.2 points |

Two models scored on the same items are compared with a paired test, which resolves smaller
differences than these intervals, but the table shows the scale. Ask for at least 1,000 test
items per decision when the user cares about differences of a few points, plus the same number
for calibration.

## Coverage of the real traffic

Sample the evaluation set the way the traffic arrives, not the way it is easy to collect:

- Keep the real label balance in the test set (balance the training set instead, with `weight`).
- Include the hard and rare cases on purpose, as a separate `task` name, so their accuracy is
  reported separately and does not hide in the average.
- For production logs, use the most recent period as the evaluation set.

## Comparing two models

Score both on the same frozen test half. Then:

1. Accuracy difference with McNemar's test on the paired per-item results. The `--out` report
   holds aggregates only, so get per-item results with the script below, once per model.
2. ECE of both, and accuracy at confidence ≥ 0.9 for both.
3. For trained models: three seeds or their soup, never the best single seed. The best of several
   seeds is a lucky draw.

A difference inside the noise is "no difference". Report it as such.

## Per-item scoring

`eval` writes aggregates only. For the paired test, score each model through the Python API. Save
as `work/score.py` and run it from the checkout root with `PYTHONPATH=.`:

```python
"""Per-item results of one model on the test half: python work/score.py MODEL DATA OUT [DEVICE]."""
import json
import sys

from training.data.format import read_jsonl
from training.evaluate import collect_logits, partition_examples, predictions_from_logits, summarise
from strands_decider.modeling import StrandsDeciderModel

model_path, data, out = sys.argv[1:4]
device = sys.argv[4] if len(sys.argv) > 4 else "cuda"  # or "mps", "cpu"
examples = partition_examples(list(read_jsonl(data)), "test")
model = StrandsDeciderModel.load(model_path)
logits, labels, slots, exs = collect_logits(model, examples, device=device, max_length=model.config.max_length)
t = getattr(model.config, "temperature_by_kind", None) or model.config.temperature  # as `eval` applies it
preds = predictions_from_logits(logits, labels, slots, exs, temperature=t,
                                ordinal_smoothing=model.config.ordinal_smoothing)
print(json.dumps(summarise(preds)["overall"]))
with open(out, "w") as fh:
    for p in preds:
        fh.write(json.dumps({"task": p.task, "kind": p.kind, "label": p.label, "pred": p.pred,
                             "correct": p.correct, "confidence": p.confidence, "probs": p.probs}) + "\n")
```

The script prepares the model as `serve` does on the device, like `eval`.

Rows come out in the order of the test half, so two output files pair line by line. McNemar's
test on the two `correct` columns: b = rows only model A gets right, c = rows only model B gets
right; with b + c ≥ 25, the statistic (|b − c| − 1)² / (b + c) above 3.84 means p < 0.05.

## Comparing with an LLM baseline

Run the user's current LLM prompt over the same test rows with the same instruction and option
text. Record accuracy, latency per decision and cost per 1,000 decisions. LLMs do not return a
calibrated confidence; if the user's method has a score, compare its accuracy at matched coverage.
