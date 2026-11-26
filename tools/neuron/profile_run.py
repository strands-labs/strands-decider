"""Profile one Typed Decisions request and one 1-question 4,096-token request on rank 0.

WEIGHTS=<checkpoint dir> python tools/neuron/profile_run.py OUTDIR [modes, default runtime,device]
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

from strands_decider.backends.neuron import load_backend
from strands_decider.core import Request, load_tokenizer, render


def main():
    out = sys.argv[1]
    modes = (sys.argv[2] if len(sys.argv) > 2 else "runtime,device").split(",")
    ckpt = os.environ["WEIGHTS"]
    os.environ.setdefault("STRANDS_NEURON_WARM", "none")
    be = load_backend(ckpt)
    tok = load_tokenizer(ckpt)
    ds = pd.read_parquet(next((Path(os.environ["TYPED"]) / "all").glob("test-*.parquet")))
    # The Typed Decisions case nearest the median state length (333 tokens).
    best = None
    for r in ds.itertuples(index=False):
        rr = render(
            Request.model_validate(
                {"state": json.loads(r.state), "questions": json.loads(r.questions)}
            ),
            tok,
        )
        d = abs(len(rr.state_ids) - 333)
        if best is None or d < best[0]:
            best = (
                d,
                r.id,
                list(rr.state_ids),
                [list(q.question_ids) for q in rr.questions.values()],
            )
    _, cid, state, qs = best
    res = {"typed_case": cid, "state_tokens": len(state), "question_tokens": sum(map(len, qs))}
    res["typed"] = be.profile(state, qs, os.path.join(out, "typed"), modes)
    lines = [json.loads(x) for x in open(Path(ckpt) / "golden_vectors.jsonl")]
    long = max((ln for ln in lines if not ln["images"]), key=lambda ln: len(ln["state_ids"]))
    q = min((x["question_ids"] for x in long["questions"].values()), key=len)
    res["one_q_4096"] = be.profile(
        long["state_ids"][: 4096 - len(q)], [q], os.path.join(out, "oneq4096"), modes
    )
    json.dump(res, open(os.path.join(out, "profile_run.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    be.close()


if __name__ == "__main__":
    main()
