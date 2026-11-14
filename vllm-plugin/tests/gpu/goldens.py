"""Model goldens for a tiny checkpoint, so that the end-to-end tiers run the conformance check (conformance.py) on a
server: section 10's lines, written from requests and a reference's probabilities for them.

    python goldens.py CKPT REQUESTS.jsonl REFERENCE.jsonl   writes CKPT/golden_vectors.jsonl

REQUESTS.jsonl is make_requests.py's output and REFERENCE.jsonl reference.py's for it. Each line has the fields section
10 names: ``request``, ``variant``, ``state_ids``, ``images`` (each image's ``sha256`` and ``span``) and, per question,
``question_ids``, ``codes`` and the reference's ``probabilities``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    from strands_decider.core import DecisionConfig, render
    from tokenizers import Tokenizer

    ckpt, requests, reference = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    config = DecisionConfig.load(ckpt / "decision_config.json")
    tok = Tokenizer.from_file(str(ckpt / "tokenizer.json"))
    ref: dict[int, dict[str, list[float]]] = {}
    for r in map(json.loads, open(reference)):
        ref.setdefault(r["request"], {})[r["question"]] = r["probabilities"]
    n = 0
    with open(ckpt / "golden_vectors.jsonl", "w") as out:
        for i, body in enumerate(map(json.loads, open(requests))):
            if i not in ref:
                continue
            rendered = render(body, tok, config.prompt)
            first = next(iter(rendered.questions.values()))
            line = {
                "request": body,
                "variant": config.prompt,
                "state_ids": list(first.state_ids),
                "images": [{"sha256": im.sha256, "span": list(im.span)} for im in rendered.images],
                "questions": {
                    name: {
                        "question_ids": list(q.question_ids),
                        "codes": list(q.codes),
                        "probabilities": ref[i][name],
                    }
                    for name, q in rendered.questions.items()
                },
            }
            out.write(json.dumps(line) + "\n")
            n += 1
    print(f"{n} golden lines in {ckpt / 'golden_vectors.jsonl'}")


if __name__ == "__main__":
    main()
