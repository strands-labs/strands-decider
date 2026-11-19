"""Conformance of a running server against a checkpoint's golden_vectors.jsonl.

    python conformance.py CKPT --url http://127.0.0.1:8000 [--min-questions N] [--out result.json]

Each golden line holds a request, the core's rendering of it (``state_ids``, each image's ``sha256``
and ``span``, each question's ``question_ids`` and ``codes``) and the reference's ``probabilities``
per question. The core must render each request exactly so, and the server's answers must agree
with the reference: top-1 on at least TOP1 and mean |dp| at most MEAN_ABS_DP, over at least
MIN_QUESTIONS questions (``--min-questions`` lowers it for the tiers' tiny goldens). Exits 0 when the
checkpoint conforms, 1 when it doesn't, and 3 (STUB) without readable goldens, so that a format this
script doesn't know stops the check instead of passing it.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

TOP1, MEAN_ABS_DP, MIN_QUESTIONS = 0.98, 0.02, 1000


def rendering(rendered) -> dict:
    """The fields of a golden line that the core's rendering must reproduce."""
    return {
        "state_ids": list(rendered.state_ids),
        "images": [{"sha256": im.sha256, "span": list(im.span)} for im in rendered.images],
        "questions": {
            n: {"question_ids": list(q.question_ids), "codes": list(q.codes)}
            for n, q in rendered.questions.items()
        },
    }


def post(url: str, body: dict):
    req = urllib.request.Request(
        url + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:300]


def vector(answer: dict) -> list[float]:
    """An answer's probabilities in code order: a noul's (false, true), the others' as listed."""
    return (
        [1.0 - answer["noul"], answer["noul"]]
        if "noul" in answer
        else list(answer["probabilities"].values())
    )


def top1(p: list[float]) -> int:
    return max(range(len(p)), key=lambda i: (p[i], -i))  # the first code on a tie


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--url", required=True)
    ap.add_argument("--min-questions", type=int, default=MIN_QUESTIONS)
    ap.add_argument("--out")
    a = ap.parse_args()
    from strands_decider import core

    path = a.ckpt / "golden_vectors.jsonl"
    if not path.exists():
        print(f"STUB: {path} does not exist; the checkpoint's release publishes it")
        sys.exit(3)
    tok = core.load_tokenizer(a.ckpt)
    render_bad, failed, agree, dps = [], [], 0, []
    for i, line in enumerate(open(path)):
        try:
            golden = json.loads(line)
            want = {k: golden[k] for k in ("state_ids", "images")}
            want["questions"] = {
                n: {k: q[k] for k in ("question_ids", "codes")}
                for n, q in golden["questions"].items()
            }
            rendered = core.render(core.Request.model_validate(golden["request"]), tok)
        except (KeyError, TypeError, ValueError) as e:
            print(f"STUB: golden line {i} is unreadable ({type(e).__name__}: {e})")
            sys.exit(3)
        if rendering(rendered) != want:
            render_bad.append(i)
        st, r = post(a.url, golden["request"])
        if st != 200:
            failed.append((i, st, str(r)[:120]))
            continue
        for name, q in golden["questions"].items():
            ours = vector(r["answers"][name])
            agree += top1(ours) == top1(q["probabilities"])
            dps.append(max(abs(x - y) for x, y in zip(ours, q["probabilities"], strict=True)))
    n = len(dps)
    result = {
        "checkpoint": str(a.ckpt),
        "render_mismatches": render_bad[:20],
        "failed": failed[:20],
        "questions": n,
        "top1": agree / n if n else None,
        "mean_dp": sum(dps) / n if n else None,
        "bounds": {"top1": TOP1, "mean_dp": MEAN_ABS_DP, "min_questions": a.min_questions},
    }
    ok = (
        not render_bad
        and not failed
        and n >= a.min_questions
        and result["top1"] >= TOP1
        and result["mean_dp"] <= MEAN_ABS_DP
    )
    print(json.dumps(result))
    print("CONFORMS" if ok else "DOES NOT CONFORM")
    if a.out:
        json.dump(result, open(a.out, "w"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
