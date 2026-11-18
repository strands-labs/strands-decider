"""Conformance of a running server against a checkpoint's golden_vectors.jsonl.

    python conformance.py CKPT --url http://127.0.0.1:8000 [--min-questions N] [--out result.json]

Each golden line holds a request, the core's rendering of it (``state_ids``, each image's ``sha256`` and ``span``, and
each question's ``question_ids`` and ``codes``) and the reference's ``probabilities`` per question. Each request is
rendered with the core and the checkpoint's tokenizer, which must give the same rendering exactly, and sent to the
server, whose probabilities are measured against the reference's. The checkpoint conforms with at least TOP1 top-1
agreement and at most MEAN_ABS_DP mean |dp| on at least MIN_QUESTIONS questions (``--min-questions`` lowers it, for
the tiers' tiny goldens). Exits 0 when it conforms, 1 when it doesn't, and 3 when the goldens are missing or a line
can't be read, so that a format this script doesn't know stops the check instead of passing it.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

TOP1, MEAN_ABS_DP, MIN_QUESTIONS = 0.98, 0.02, 1000


def read_golden(line: dict) -> dict:
    """One golden line's fields, checked for presence and type, or ValueError."""
    try:
        golden = {
            "request": dict(line["request"]),
            "state_ids": [int(t) for t in line["state_ids"]],
            "images": [
                {"sha256": str(im["sha256"]), "span": [int(i) for i in im["span"]]}
                for im in line["images"]
            ],
            "questions": {
                name: {
                    "question_ids": [int(t) for t in q["question_ids"]],
                    "codes": [str(c) for c in q["codes"]],
                    "probabilities": [float(p) for p in q["probabilities"]],
                }
                for name, q in line["questions"].items()
            },
        }
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise ValueError(f"unreadable golden line ({type(e).__name__}: {e})") from e
    if not golden["questions"] or any(
        len(q["codes"]) != len(q["probabilities"]) for q in golden["questions"].values()
    ):
        raise ValueError(
            "unreadable golden line: it needs questions, each with one probability per code"
        )
    return golden


def rendering(golden: dict) -> dict:
    """The fields of a golden line that rendering must reproduce."""
    questions = {
        n: {"question_ids": q["question_ids"], "codes": q["codes"]}
        for n, q in golden["questions"].items()
    }
    return {"state_ids": golden["state_ids"], "images": golden["images"], "questions": questions}


def rendered_fields(rendered) -> dict:
    """The same fields from the core's rendering."""
    return {
        "state_ids": list(rendered.state_ids),
        "images": [{"sha256": im.sha256, "span": list(im.span)} for im in rendered.images],
        "questions": {
            n: {"question_ids": list(q.question_ids), "codes": list(q.codes)}
            for n, q in rendered.questions.items()
        },
    }


def agreement(ours: dict, reference: dict) -> tuple[float, float]:
    """Top-1 agreement (the first code on a tie) and mean |dp| (each question's largest difference, averaged)."""

    def top1(p: list[float]) -> int:
        return max(range(len(p)), key=lambda i: (p[i], -i))

    agree = sum(top1(ours[k]) == top1(ref) for k, ref in reference.items())
    dp = sum(
        max(abs(a - b) for a, b in zip(ours[k], ref, strict=True)) for k, ref in reference.items()
    )
    return agree / len(reference), dp / len(reference)


def post(url: str, body: dict) -> tuple[int | None, object]:
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
    """An answer's probabilities in code order: a noul's (false, true), a choice's and a score's as listed."""
    if "noul" in answer:
        return [1.0 - answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


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
    goldens = []
    for i, raw in enumerate(open(path)):
        if raw.strip():
            try:
                goldens.append((i, read_golden(json.loads(raw))))
            except ValueError as e:
                print(f"STUB: line {i}: {e}")
                sys.exit(3)
    render_bad, failed, ours, reference = [], [], {}, {}
    for i, g in goldens:
        if rendered_fields(
            core.render(core.Request.model_validate(g["request"]), tok)
        ) != rendering(g):
            render_bad.append(i)
        st, r = post(a.url, g["request"])
        if st != 200:
            failed.append((i, st, str(r)[:120]))
            continue
        for n, q in g["questions"].items():
            ours[(i, n)], reference[(i, n)] = vector(r["answers"][n]), q["probabilities"]
    top1, mean_dp = agreement(ours, reference) if reference and not failed else (None, None)
    ok = (
        not render_bad
        and top1 is not None
        and len(reference) >= a.min_questions
        and top1 >= TOP1
        and mean_dp <= MEAN_ABS_DP
    )
    result = {
        "checkpoint": str(a.ckpt),
        "render_mismatches": render_bad[:20],
        "failed": failed[:20],
        "questions": len(reference),
        "top1": top1,
        "mean_dp": mean_dp,
        "bounds": {"top1": TOP1, "mean_dp": MEAN_ABS_DP, "min_questions": a.min_questions},
    }
    print(json.dumps(result))
    print("CONFORMS" if ok else "DOES NOT CONFORM")
    if a.out:
        json.dump(result, open(a.out, "w"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
