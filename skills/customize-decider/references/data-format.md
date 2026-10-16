# Row format for evaluation and training data

One row is one (state, question, answer) triple, one JSON object per line. Evaluation and
training files use the same format. The code is `training/data/format.py`
(`Example`).

## Fields

| field | required | meaning |
|---|---|---|
| `kind` | yes | `"noul"`, `"choice"` or `"score"` |
| `state` | yes | the content to decide on: a string, or JSON data (rendered deterministically); may be `""` when the question carries the whole task |
| `instructions` | yes | the question text |
| `options` | yes | list of `[name, description]` pairs, in canonical order (see below) |
| `label` | yes | the index of the correct option in `options` |
| `task` | no | a name for the decision; `eval` reports accuracy per task |
| `weight` | no | loss weight of the row (default 1.0) |
| `instruction_variants` | no | other phrasings of the same question; training samples one per epoch |

Canonical order per kind:

- `noul`: exactly two options named `"false"` (first) and `"true"` (second). Label 0 is no, 1 is
  yes. Other names fail.
- `choice`: 2 to 255 options, any order; training shuffles the options at every epoch.
- `score`: 2 to 10 levels, ascending, lowest level first. The label is the level index; the model
  reads the descriptions as the level texts.

## Examples

```json
{"kind": "noul", "state": "Help! My payouts have been failing for 3 days!", "instructions": "Does this convey urgency?", "options": [["false", "no urgency"], ["true", "the writer needs help soon"]], "label": 1, "task": "urgency", "instruction_variants": ["Is the writer asking for fast help?"]}
{"kind": "choice", "state": "Help! My payouts have been failing for 3 days!", "instructions": "Which team should handle this?", "options": [["billing", "payments, payouts, invoices"], ["sales", "pricing and new accounts"], ["retail", "store operations"]], "label": 0, "task": "triage"}
{"kind": "score", "state": "Help! My payouts have been failing for 3 days!", "instructions": "How frustrated is the writer?", "options": [["0", "calm"], ["1", "frustrated"], ["2", "very frustrated"]], "label": 1, "task": "frustration"}
```

Use the same instruction and option text as the application will send at serving time. The server
builds the same prompt from a request (`src/strands_decider/schema.py`), so a mismatch between
training text and serving text is a distribution shift.

## Converting user data

- One labelled decision becomes one row. Several questions about one state become several rows
  with the same `state`; they stay on the same side of every split.
- Keep the split keys in a side file, not in the row: `work/<set>/keys.jsonl`, one line per row with
  the row index, the group id (customer, document, conversation) and the timestamp.
- Use the same option set for one decision in every row, in the same canonical order.
- Many options: keep the training config's `num_slots` at least the largest option count, or
  subsample options per row and always keep the gold one (`python -m training.data build
  --max-options` does this for the public corpus).

## The row check (run on every file before it is frozen)

`python -m training.data stats` checks `kind` and the label range only. The checks below otherwise
fail late, in the middle of `eval` or training: option names and counts are validated when the
question is built, and a prompt longer than the window stops the collator ("option span has no
tokens left"). Save as `work/check_rows.py` and run it from the checkout root with `PYTHONPATH=.`:

```python
"""python work/check_rows.py FILE BASE_MODEL [MAX_TOKENS]: fail on any row that would fail later."""
import sys
from collections import Counter

from transformers import AutoTokenizer

from training.data.format import read_jsonl
from strands_decider.prompting import build_prompt

path, base = sys.argv[1], sys.argv[2]
max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
tok = AutoTokenizer.from_pretrained(base)  # the base model of the checkpoint you will use
bad, long_rows = Counter(), []
for i, ex in enumerate(read_jsonl(path)):
    try:
        prompt, _ = build_prompt(ex.state, ex.to_question())
    except Exception as e:  # option names, option counts, level counts
        bad[f"{type(e).__name__}: {e}"[:120]] += 1
        continue
    n = len(tok(prompt)["input_ids"]) + 1  # +1 for a BOS token
    if n > max_tokens:
        long_rows.append((i, n))
print(f"invalid rows: {sum(bad.values())}")
for msg, c in bad.most_common(10):
    print(f"  {c} x {msg}")
print(f"rows over {max_tokens} tokens: {len(long_rows)} (first: {long_rows[:5]})")
sys.exit(1 if bad or long_rows else 0)
```

`read_jsonl` itself stops at the first row with an unknown `kind` or a label out of range; fix
that row and run the check again. Remove rows over the window from the file; do not truncate their
state, because a truncated state can remove the evidence for the label. Base models of the released
deciders: `Qwen/Qwen3.5-2B-Base` and `google/gemma-4-<size>-it`.

## Label quality

Labels set the ceiling. Before you trust an evaluation set:

1. Draw 200 rows at random. Have a second person label them without seeing the first label.
2. Report the agreement. Inspect every disagreement with the user and fix the label guide.
3. If agreement is below about 90%, fix the guide and relabel before you measure models: a model
   cannot be measured more precisely than its labels.

## Overlap check between training and evaluation

Run all three and report the counts; all must be zero before training.

1. Groups: no group id in both `work/train/` and `work/eval/` (from the `keys.jsonl` side files).
2. Exact duplicates: normalize each state (lowercase, collapse white space, strip punctuation) and
   hash it; no hash in both sets. This catches short states that the next check cannot.
3. Near-duplicates: split each normalized state into words and take every 13-word sequence. Flag a
   training row when it shares at least 20% of its 13-word sequences (or at least 10 of them) with
   any evaluation row.

Drop flagged training rows; never drop evaluation rows to fix an overlap. A short
standard-library script does all three; write it in `work/` and keep it with the log.

## Freezing a set

```bash
(cd work/eval && shasum -a 256 *.jsonl > SHA256SUMS)
```

Record the hash in `work/LOG.md`. Before each later measurement, run
`(cd work/eval && shasum -a 256 -c SHA256SUMS)`.
