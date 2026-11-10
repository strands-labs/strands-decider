"""Conformance of a running server against a checkpoint's model goldens (specification, section 10).

    python conformance.py CKPT --url http://127.0.0.1:8000 [--out result.json]

For each line of ``CKPT/golden_vectors.jsonl``: the request is rendered with the core library and its ids, S and
image spans must equal the golden's (no tolerance); the request is sent to the plugin's server (``POST /v1/systemone``)
and each question's top-1 code must agree with the golden's reference probabilities on at least 98% of the questions
(bf16). Exits 0 when the checkpoint conforms, 1 when it does not, 3 when the goldens are missing or in a form this
script cannot read.

The specification says what a golden line holds (a request with its images as base64, its rendered ids per question,
each image's span, and the reference code probabilities per question) but #129 has not yet fixed the field names.
:func:`read_golden` is the one place that maps a line to those four things; it reads the provisional names below and
refuses anything else with exit code 3, so that a changed format stops the check instead of passing it.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

GATE = 0.98


def read_golden(
    line: dict,
) -> tuple[dict, dict[str, list[int]], list[list[int]], dict[str, list[float]]]:
    """(request, ids per question, image spans [start, end), reference probabilities per question).

    Provisional field names, until the specification fixes them: ``request``; ``questions``, mapping each question's
    name to ``{"ids": [...], "probabilities": [...]}``; ``images``, a list of ``{"start": s, "end": e}`` (absent for a
    text request)."""
    try:
        request = line["request"]
        qs = line["questions"]
        ids = {n: list(q["ids"]) for n, q in qs.items()}
        probs = {n: list(q["probabilities"]) for n, q in qs.items()}
        spans = [[int(im["start"]), int(im["end"])] for im in line.get("images", [])]
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"unreadable golden line (field {e}); update read_golden to the specification's format"
        ) from e
    return request, ids, spans, probs


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
    if "noul" in answer:
        return [1.0 - answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--url", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    from strands_decider.core import DecisionConfig, render
    from tokenizers import Tokenizer

    path = a.ckpt / "golden_vectors.jsonl"
    if not path.exists():
        print(
            f"STUB: {path} does not exist; the checkpoint's release publishes it (specification, section 10)"
        )
        sys.exit(3)
    config = DecisionConfig.load(a.ckpt / "decision_config.json")
    tok = Tokenizer.from_file(str(a.ckpt / "tokenizer.json"))
    render_bad, agree, total, failed_requests = 0, 0, 0, 0
    for i, raw in enumerate(open(path)):
        try:
            request, ids, spans, probs = read_golden(json.loads(raw))
        except ValueError as e:
            print(f"STUB: line {i}: {e}")
            sys.exit(3)
        rendered = render(request, tok, config.prompt)
        got_spans = [[im.start, im.end] for im in rendered.images]
        if got_spans != spans or any(
            list(q.ids) != ids.get(n) for n, q in rendered.questions.items()
        ):
            render_bad += 1
        st, r = post(a.url, request)
        if st != 200:
            failed_requests += 1
            continue
        for n, p in probs.items():
            q = vector(r["answers"][n])
            agree += max(range(len(q)), key=q.__getitem__) == max(range(len(p)), key=p.__getitem__)
            total += 1
    top1 = agree / max(1, total)
    result = {
        "checkpoint": str(a.ckpt),
        "render_mismatches": render_bad,
        "failed_requests": failed_requests,
        "questions": total,
        "top1": top1,
        "gate": GATE,
    }
    ok = render_bad == 0 and failed_requests == 0 and top1 >= GATE
    print(json.dumps(result))
    print("CONFORMS" if ok else "DOES NOT CONFORM")
    if a.out:
        json.dump(result, open(a.out, "w"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
