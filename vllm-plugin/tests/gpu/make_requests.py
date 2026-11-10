"""Requests for the GPU checks, without a dataset: System One requests with synthetic photos, or rendered text rows.

    python make_requests.py images OUT.jsonl [--n 24] [--seed 0]
    python make_requests.py rows TOKENIZER.json OUT.jsonl [--n 48] [--variant strands-chat]

``images``: requests with 0 to 4 images each, of mixed sizes and aspect ratios (from 64 x 48 to 1600 x 1200, panoramas
and tall strips), in PNG, JPEG, WebP and GIF; noul, choice and score questions; short, empty and long state text.

``rows``: text requests rendered with the core library, one row per question: ``token_ids``, ``prefix_end`` (S),
``label_token_ids`` (the question's code ids) and ``n_tokens``, the format of tests/gpu/client.py's ``--rows``; about
400 to 1,500 tokens each. ``reference.py --rows`` adds a reference's ``probabilities``.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import random

SIZES = [
    (640, 480),
    (480, 640),
    (300, 40),
    (64, 48),
    (1000, 1000),
    (96, 1200),
    (1600, 1200),
    (1200, 300),
]
FORMATS = ["PNG", "JPEG", "WEBP", "GIF"]
WORDS = (
    "order shipment refund delayed carrier label package damaged customer account invoice warehouse address photo "
    "listing seller buyer return scratch screen box receipt tracking update support agent replacement"
).split()


def photo(rng: random.Random, w: int, h: int, fmt: str) -> str:
    from PIL import Image, ImageDraw, ImageFilter

    img = Image.new("RGB", (w, h), tuple(rng.randrange(256) for _ in range(3)))
    d = ImageDraw.Draw(img)
    for _ in range(30):
        x, y = rng.randrange(w), rng.randrange(h)
        d.rectangle(
            (x, y, x + rng.randrange(1, w // 2 + 2), y + rng.randrange(1, h // 2 + 2)),
            fill=tuple(rng.randrange(256) for _ in range(3)),
        )
    img = img.filter(ImageFilter.GaussianBlur(2))
    if fmt == "GIF":
        img = img.convert("P", palette=Image.ADAPTIVE)
    buf = io.BytesIO()
    img.save(buf, format=fmt, **({"quality": 90} if fmt in ("JPEG", "WEBP") else {}))
    return base64.b64encode(buf.getvalue()).decode()


def text(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n))


def questions(rng: random.Random) -> dict:
    k = rng.randint(2, 6)
    return {
        "kind": {
            "type": "choice",
            "instructions": "What is the ticket about?",
            "criteria": {w: text(rng, 3) for w in rng.sample(WORDS, k)},
        },
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
        "tone": {
            "type": "score",
            "instructions": "How upset is the customer?",
            "criteria": ["calm", "annoyed", "angry", "furious"][: rng.randint(2, 4)],
        },
    }


def requests_with_images(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        k = i % 5
        images = [photo(rng, *rng.choice(SIZES), rng.choice(FORMATS)) for _ in range(k)]
        state = [
            "",
            text(rng, 12),
            {
                "ticket": text(rng, 20),
                "history": [text(rng, 40) for _ in range(rng.randint(4, 12))],
            },
        ][i % 3]
        out.append({"state": state, "images": images, "questions": questions(rng)})
    return out


def rows(tokenizer: str, n: int, variant: str, seed: int) -> list[dict]:
    from strands_decider.core import render
    from tokenizers import Tokenizer

    rng = random.Random(seed)
    tok = Tokenizer.from_file(tokenizer)
    out = []
    while len(out) < n:
        body = {
            "state": {
                "ticket": text(rng, 40),
                "notes": [text(rng, 60) for _ in range(rng.randint(3, 20))],
            },
            "questions": questions(rng),
        }
        for q in render(body, tok, variant).questions.values():
            out.append(
                {
                    "token_ids": list(q.ids),
                    "prefix_end": q.state_len,
                    "label_token_ids": list(q.code_token_ids),
                    "n_tokens": len(q.ids),
                }
            )
    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="kind", required=True)
    a1 = sub.add_parser("images")
    a1.add_argument("out")
    a1.add_argument("--n", type=int, default=24)
    a1.add_argument("--seed", type=int, default=0)
    a2 = sub.add_parser("rows")
    a2.add_argument("tokenizer")
    a2.add_argument("out")
    a2.add_argument("--n", type=int, default=48)
    a2.add_argument("--variant", default="strands-chat")
    a2.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    data = (
        requests_with_images(a.n, a.seed)
        if a.kind == "images"
        else rows(a.tokenizer, a.n, a.variant, a.seed)
    )
    with open(a.out, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")
    print(len(data), a.kind)


if __name__ == "__main__":
    main()
