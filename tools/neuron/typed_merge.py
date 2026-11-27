"""Merge the shard dumps of a typed run (tools/parity.py typed --shard k/n --dump ...) into one
reference predictions file, and score it as bench/score_with_jobs.py does (score_one, then its
accuracy, Brier, 10-bin ECE and KL from gold).

  python tools/neuron/typed_merge.py results/cpu.jsonl results/cpu0.jsonl results/cpu1.jsonl ...
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from parity import _ece, _score_module

out, parts = sys.argv[1], sys.argv[2:]
data = Path(os.environ["TYPED"])
sw = _score_module(data / "bench" / "score_with_jobs.py")
ds = pd.read_parquet(next((data / "all").glob("test-*.parquet")))
golds = {(r.id, q): g for r in ds.itertuples(index=False) for q, g in json.loads(r.gold).items()}
rows = {}
for p in parts:
    for raw in open(p):
        d = json.loads(raw)
        rows[(d["id"], d["question"])] = d
missing = [k for k in golds if k not in rows]
order = [k for k in golds if k in rows]
with open(out, "w") as f:
    for k in order:
        f.write(json.dumps(rows[k]) + "\n")
sc = [sw.score_one({"probabilities": rows[k]["probabilities"]}, golds[k])[:4] for k in order]
n = len(sc)
print(
    json.dumps(
        {
            "decisions": n,
            "missing": len(missing),
            "cases": len({k[0] for k in order}),
            "accuracy": sum(x[0] for x in sc) / n,
            "brier": sum(x[2] for x in sc) / n,
            "ece": _ece([x[0] for x in sc], [x[3] for x in sc]),
            "kl_from_gold": sum(x[1] for x in sc) / n,
        },
        indent=1,
    )
)
