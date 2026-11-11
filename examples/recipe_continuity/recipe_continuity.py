"""Group the OCR text boxes of a cookbook page into recipes with Strands Decider.

Layout proposes which boxes could follow each box, Strands Decider judges whether the recipe
continues, and a few lines of Python settle the links. Start a server, then run this file
next to arabic_cookbook.json:

    strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099
    python examples/recipe_continuity/recipe_continuity.py

`--question` picks the question shape. `auto` (the default) asks a `noul` when a box has one
candidate and a `choice` when it has several; `noul` and `choice` force one shape everywhere,
to compare the two.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import textwrap
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any

URL = os.environ.get("STRANDS_DECIDER_URL", "http://127.0.0.1:8099")
PAGE = Path(__file__).with_name("arabic_cookbook.json")
MAX_GAP = 60  # px between two paragraphs of the same column; tune to your scan resolution
ACCEPT = 0.5  # a claim needs P >= ACCEPT; pick it from your own labelled pages
CONTEXT = "Text blocks OCR'd from a scanned cookbook page. The OCR text may contain errors."
SAME = {
    "true": "B carries on A's recipe, or A is the title of B's recipe. A sentence cut off at "
    "the end of A and finished at the start of B is strong evidence.",
    "false": "B starts a different recipe or a tip: it opens with a dish name followed by a full "
    "stop or dash, or with 'نصيحة'. This holds even when B is about an ingredient or dish "
    "that A mentions.",
}

Box = dict[str, Any]
Ask = Callable[[str, dict[str, Any]], dict[str, Any]]


def overlap(a: Box, b: Box) -> float:
    """Horizontal overlap as a fraction of the narrower box."""
    (ax0, _, ax1, _), (bx0, _, bx1, _) = a["bbox"], b["bbox"]
    inter = max(0, min(ax1, bx1) - max(ax0, bx0))
    return inter / max(1, min(ax1 - ax0, bx1 - bx0))


def candidates(boxes: list[Box], rtl: bool) -> dict[str, list[str]]:
    """For each box, the boxes that could continue its recipe: the box directly below it in
    the same column, or, if nothing is below, the top box of the next column (to the left on
    a right-to-left page)."""
    below = {}
    for a in boxes:
        under = [
            b
            for b in boxes
            if b is not a and overlap(a, b) >= 0.5 and 0 <= b["bbox"][1] - a["bbox"][3] <= MAX_GAP
        ]
        if under:
            below[a["id"]] = min(under, key=lambda b: b["bbox"][1])
    tops = [b for b in boxes if b not in below.values()]
    out: dict[str, list[str]] = {}
    for a in boxes:
        if a["id"] in below:
            out[a["id"]] = [below[a["id"]]["id"]]
            continue
        side = [
            b
            for b in tops
            if overlap(a, b) == 0
            and b["bbox"][1] < a["bbox"][3]
            and (b["bbox"][2] <= a["bbox"][0] if rtl else b["bbox"][0] >= a["bbox"][2])
        ]
        if side:  # the nearest column on that side
            edge = (max if rtl else min)(b["bbox"][2 if rtl else 0] for b in side)
            out[a["id"]] = [b["id"] for b in side if b["bbox"][2 if rtl else 0] == edge]
    return out


def ask_server(state: str, questions: dict[str, Any]) -> dict[str, Any]:
    url = urllib.parse.urlsplit(URL)
    if url.scheme not in ("http", "https"):  # refuse file: and other schemes
        raise ValueError(f"STRANDS_DECIDER_URL must start with http:// or https://, got {URL!r}")
    conn_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    conn = conn_type(url.netloc, timeout=120)
    try:
        body = json.dumps({"state": state, "questions": questions})
        conn.request(
            "POST",
            url.path.rstrip("/") + "/v1/systemone",
            body,
            {"Content-Type": "application/json"},
        )
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"server returned HTTP {resp.status}")
        return json.load(resp)["answers"]
    finally:
        conn.close()


def claims(
    by_id: dict[str, Box], cands: dict[str, list[str]], mode: str, ask: Ask
) -> list[tuple[float, str, str]]:
    """Ask the decider about every box with candidates; return (confidence, from, to) claims."""

    def where(a: str, b: str) -> str:
        if overlap(by_id[a], by_id[b]) >= 0.5:
            return "directly below Block A in the same column"
        return "at the top of the next column"

    out = []
    for a, bs in cands.items():
        end = f'{CONTEXT}\nBlock A ends: "{by_id[a]["text"][-300:]}"'
        if mode == "noul" or (mode == "auto" and len(bs) == 1):
            for b in bs:
                state = f'{end}\nBlock B is {where(a, b)}. It starts: "{by_id[b]["text"][:300]}"'
                p = ask(
                    state,
                    {
                        "same": {
                            "type": "noul",
                            "instructions": "Do Block A and Block B belong to the same recipe?",
                            "criteria": SAME,
                        }
                    },
                )["same"]["noul"]
                if p >= ACCEPT:
                    out.append((p, a, b))
        else:
            options = {b: f'{where(a, b)}, starts: "{by_id[b]["text"][:160]}"' for b in bs}
            options["none"] = "None of them: Block A's recipe ends here."
            probs = ask(
                end,
                {
                    "next": {
                        "type": "choice",
                        "instructions": "Which block continues the same recipe as Block A? "
                        "A sentence cut off at the end of A and finished in a block is "
                        "strong evidence.",
                        "criteria": options,
                    }
                },
            )["next"]["probabilities"]
            out += [
                (p, a, b)
                for b, p in probs.items()
                if b != "none" and p > probs["none"] and p >= ACCEPT
            ]
    return out


def settle(boxes: list[Box], found: list[tuple[float, str, str]]) -> list[list[str]]:
    """Accept claims strongest first, one continuation per box, and chain them into recipes."""
    nxt: dict[str, str] = {}
    for _, a, b in sorted(found, reverse=True):
        if a not in nxt and b not in nxt.values():
            nxt[a] = b
    starts = [b["id"] for b in boxes if b["id"] not in nxt.values()]
    groups = []
    for s in starts:
        chain = [s]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        groups.append(chain)
    return groups


def run(page: dict[str, Any], mode: str = "auto", ask: Ask = ask_server) -> list[list[str]]:
    boxes = page["boxes"]
    width = sorted(b["bbox"][2] - b["bbox"][0] for b in boxes)[len(boxes) // 2]
    # A box spanning the whole page (the page header) belongs to no recipe: leave it out.
    text = [b for b in boxes if b["bbox"][2] - b["bbox"][0] < 2 * width]
    found = claims(
        {b["id"]: b for b in text},
        candidates(text, page["page"].get("direction") == "rtl"),
        mode,
        ask,
    )
    return settle(text, found) + [[b["id"]] for b in boxes if b not in text]


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--page", type=Path, default=PAGE)
    ap.add_argument("--question", choices=["auto", "noul", "choice"], default="auto")
    args = ap.parse_args()
    page = json.loads(args.page.read_text(encoding="utf-8"))
    groups = run(page, args.question)
    recipes: dict[str, set[str]] = {}  # the labels; the page header's `recipe` is null
    for b in page["boxes"]:
        if b.get("recipe"):
            recipes.setdefault(b["recipe"], set()).add(b["id"])
    for g in groups:
        first = next(b for b in page["boxes"] if b["id"] == g[0])
        mark = "ok " if set(g) in recipes.values() else "-- " if not first.get("recipe") else "   "
        start = textwrap.shorten(first["text"], 28, placeholder="").rstrip("،,")  # whole words
        print(f"{mark}{' '.join(g):<16} {start}")
    correct = sum(set(g) in recipes.values() for g in groups)
    print(f"\n{correct}/{len(recipes)} recipes match the labels ({args.question} questions)")


if __name__ == "__main__":
    main()
