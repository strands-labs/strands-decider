"""Conformance of a running server against a checkpoint's model goldens (specification, section 10).

    python conformance.py CKPT --url http://127.0.0.1:8000 [--min-questions N] [--out result.json]

``CKPT/golden_vectors.jsonl`` holds one JSON object per line, with the fields section 10 names: the ``request``
(images as base64), its prompt ``variant``, its ``state_ids``, each image's ``sha256`` and ``span`` (``images``), and
under ``questions``, for each question by name, its ``question_ids``, its ``codes`` and the reference's
``probabilities`` (one per code, in code order).

1. Each request is rendered with the core library and the checkpoint's own tokenizer (``load_tokenizer``). Its
   variant, which must be the checkpoint's prompt, its state ids, its images and each question's ids and codes must
   equal the golden's, with no tolerance.
2. Each request is sent to the plugin's server (``POST /v1/systemone``). Its code probabilities are measured against
   the reference's (``read_golden_vectors``) with the core's ``agreement``: top-1 agreement and mean |dp|. The
   checkpoint conforms in bf16 when ``Agreement.conforms``: at least 98% and at most 0.02, on at least the core's
   minimum of 1,000 questions. ``--min-questions`` lowers the minimum, for the end-to-end tiers' tiny goldens only.

Exits 0 when the checkpoint conforms, 1 when it does not, and 3 when the goldens are missing or a line lacks section
10's fields, so that a format this script does not know stops the check instead of passing it.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def read_golden(line: dict) -> dict:
    """Section 10's fields of one golden line, checked for presence and type, or ValueError."""
    try:
        request, variant = line["request"], line["variant"]
        state_ids = [int(t) for t in line["state_ids"]]
        images = [
            {"sha256": str(im["sha256"]), "span": [int(im["span"][0]), int(im["span"][1])]}
            for im in line["images"]
        ]
        questions = {
            name: {
                "question_ids": [int(t) for t in q["question_ids"]],
                "codes": [str(c) for c in q["codes"]],
                "probabilities": [float(p) for p in q["probabilities"]],
            }
            for name, q in line["questions"].items()
        }
        if not isinstance(request, dict) or not isinstance(variant, str) or not questions:
            raise TypeError("request must be an object, variant a string and questions not empty")
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as e:
        raise ValueError(
            f"unreadable golden line ({type(e).__name__}: {e}); section 10 names its fields"
        ) from e
    for name, q in questions.items():
        if len(q["codes"]) != len(q["probabilities"]):
            raise ValueError(
                f"unreadable golden line: question {name!r} needs one probability per code"
            )
    return {
        "request": request,
        "variant": variant,
        "state_ids": state_ids,
        "images": images,
        "questions": questions,
    }


def rendering(golden: dict) -> dict:
    """The fields of a golden line that rendering must reproduce."""
    return {
        "variant": golden["variant"],
        "state_ids": golden["state_ids"],
        "images": golden["images"],
        "questions": {
            n: {"question_ids": q["question_ids"], "codes": q["codes"]}
            for n, q in golden["questions"].items()
        },
    }


def rendered_fields(rendered, variant: str) -> dict:
    """The same fields from the core's rendering."""
    first = next(iter(rendered.questions.values()))
    return {
        "variant": variant,
        "state_ids": list(first.state_ids),
        "images": [{"sha256": im.sha256, "span": list(im.span)} for im in rendered.images],
        "questions": {
            n: {"question_ids": list(q.question_ids), "codes": list(q.codes)}
            for n, q in rendered.questions.items()
        },
    }


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
    """A v1 answer's probabilities in code order: a noul's (false, true), a choice's and a score's as listed."""
    if "noul" in answer:
        return [1.0 - answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--url", required=True)
    ap.add_argument(
        "--min-questions",
        type=int,
        default=None,
        help="the fewest questions a claim rests on (default: the core's, 1,000)",
    )
    ap.add_argument("--out")
    a = ap.parse_args()
    from strands_decider import core
    from strands_decider.core import conformance as cc

    path = a.ckpt / "golden_vectors.jsonl"
    if not path.exists():
        print(
            f"STUB: {path} does not exist; the checkpoint's release publishes it (specification, section 10)"
        )
        sys.exit(3)
    config = core.DecisionConfig.load(a.ckpt / "decision_config.json")
    tok = core.load_tokenizer(a.ckpt, config)
    goldens = []
    for i, raw in enumerate(open(path)):
        if not raw.strip():
            continue
        try:
            goldens.append((i, read_golden(json.loads(raw))))
        except ValueError as e:
            print(f"STUB: line {i}: {e}")
            sys.exit(3)
    reference = cc.read_golden_vectors(path)
    render_bad, failed, ours = [], [], {}
    for i, g in goldens:
        rendered = core.render(g["request"], tok, g["variant"])
        if g["variant"] != config.prompt or rendered_fields(rendered, g["variant"]) != rendering(g):
            render_bad.append(i)
        st, r = post(a.url, g["request"])
        if st != 200:
            failed.append((i, st, str(r)[:120]))
            continue
        for n in g["questions"]:
            ours[(i, n)] = vector(r["answers"][n])
    agreement = None if failed else cc.agreement(ours, reference)
    minimum = cc.MIN_QUESTIONS if a.min_questions is None else a.min_questions
    top1, mean_dp = cc.bounds()  # bf16's, at every size
    result = {
        "checkpoint": str(a.ckpt),
        "render_mismatches": render_bad[:20],
        "failed": failed[:20],
        "questions": agreement.questions if agreement else 0,
        "top1": agreement.top1 if agreement else None,
        "mean_dp": agreement.mean_abs_dp if agreement else None,
        "bounds": {"top1": top1, "mean_dp": mean_dp, "min_questions": minimum},
    }
    ok = (
        not render_bad
        and agreement is not None
        and agreement.conforms(min_questions=minimum)  # bf16's bounds
    )
    print(json.dumps(result))
    print("CONFORMS" if ok else "DOES NOT CONFORM")
    if a.out:
        json.dump(result, open(a.out, "w"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
