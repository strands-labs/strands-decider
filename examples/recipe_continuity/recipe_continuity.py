"""Group the OCR text boxes of a cookbook page into recipes with strands decider.

Three steps: layout proposes which box could come next after each box, strands decider judges
whether the recipe really continues, and a few lines of plain Python settle the links into
recipes. Start a server, then run this file from the repository root:

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

# Layout settings, in the page's pixel units. Tune them to your scans.
MAX_GAP = 60  # largest vertical gap between two boxes of the same column
SAME_COLUMN = 0.5  # boxes overlapping this much horizontally are in the same column
HEADER = 0.8  # a box wider than this share of the page is a header, not part of a recipe

# Decision settings.
ACCEPT = 0.5  # a noul link needs P(true) >= ACCEPT; pick it from your own labelled pages
SNIPPET = 300  # characters shown from the end of box A and from the start of box B
CONTEXT = "Text blocks OCR'd from a scanned cookbook page. The OCR text may contain errors."
QUESTION = "Do Block A and Block B belong to the same recipe?"
SAME = {  # the noul's two criteria: what "same recipe" and "different recipe" look like here
    "true": "B carries on A's recipe, or A is the title of B's recipe. A sentence cut off at "
    "the end of A and finished at the start of B is strong evidence.",
    "false": "B starts a different recipe or a tip: it opens with a dish name followed by a full "
    "stop or dash, or with 'نصيحة'. This holds even when B is about an ingredient or dish "
    "that A mentions.",
}

Box = dict[str, Any]  # {"id": str, "bbox": [x0, y0, x1, y1], "text": str, ...}
Ask = Callable[[str, dict[str, Any]], dict[str, Any]]  # (state, questions) -> answers
Claim = tuple[float, str, str]  # (probability, id of box A, id of the box that continues A)


def overlap(a: Box, b: Box) -> float:
    """Horizontal overlap of two boxes, as a fraction of the narrower one (0 to 1)."""
    (ax0, _, ax1, _), (bx0, _, bx1, _) = a["bbox"], b["bbox"]
    inter = max(0, min(ax1, bx1) - max(ax0, bx0))
    return inter / max(1, min(ax1 - ax0, bx1 - bx0))


def candidates(boxes: list[Box], rtl: bool) -> dict[str, list[str]]:
    """Map each box id to the ids of the boxes that could continue its recipe.

    Reading order: straight down inside a column, then from the bottom of a column to the top
    of the nearest column in reading direction (to the left on a right-to-left page). A box
    gets no entry when nothing can follow it, like the last box of the last column.
    """
    # below[a] = the box just under a: same column, and at most MAX_GAP pixels lower.
    below: dict[str, Box] = {}
    for a in boxes:
        under = [
            b
            for b in boxes
            if b is not a
            and overlap(a, b) >= SAME_COLUMN
            and 0 <= b["bbox"][1] - a["bbox"][3] <= MAX_GAP
        ]
        if under:
            below[a["id"]] = min(under, key=lambda b: b["bbox"][1])

    # A box that is not under any other box starts a column.
    under_ids = {b["id"] for b in below.values()}
    tops = [b for b in boxes if b["id"] not in under_ids]

    out: dict[str, list[str]] = {}
    for a in boxes:
        if a["id"] in below:
            out[a["id"]] = [below[a["id"]]["id"]]
            continue
        # a ends its column. Look for column tops in another column, on the reading side,
        # that start above the bottom of a.
        side = [
            b
            for b in tops
            if overlap(a, b) < SAME_COLUMN
            and b["bbox"][1] < a["bbox"][3]
            and (b["bbox"][2] <= a["bbox"][0] if rtl else b["bbox"][0] >= a["bbox"][2])
        ]
        if side:
            # Keep only the nearest column: the largest right edge (RTL) or smallest left (LTR).
            def edge(b: Box) -> int:
                return b["bbox"][2] if rtl else b["bbox"][0]

            nearest = (max if rtl else min)(edge(b) for b in side)
            out[a["id"]] = [b["id"] for b in side if edge(b) == nearest]
    return out


def ask_server(state: str, questions: dict[str, Any]) -> dict[str, Any]:
    """POST one request (the state plus named questions) to /v1/systemone; return the answers.

    Uses http.client from the standard library, so the example needs no extra packages.
    """
    url = urllib.parse.urlsplit(URL)
    if url.scheme not in ("http", "https"):  # refuse file: and other schemes
        raise ValueError(f"STRANDS_DECIDER_URL must start with http:// or https://, got {URL!r}")
    conn_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    conn = conn_type(url.netloc, timeout=120)
    try:
        # json.dumps escapes the Arabic text by default, so the body is plain ASCII.
        body = json.dumps({"state": state, "questions": questions})
        path = url.path.rstrip("/") + "/v1/systemone"
        conn.request("POST", path, body, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            detail = resp.read(300).decode("utf-8", "replace")
            raise RuntimeError(f"server returned HTTP {resp.status}: {detail}")
        return json.load(resp)["answers"]
    finally:
        conn.close()


def where(by_id: dict[str, Box], a: str, b: str) -> str:
    """One line of layout context for the model: where box B sits relative to box A."""
    if overlap(by_id[a], by_id[b]) >= SAME_COLUMN:
        return "directly below Block A in the same column"
    return "at the top of the next column"


def ask_noul(by_id: dict[str, Box], a: str, b: str, ask: Ask) -> float:
    """Ask one yes/no question about one pair of boxes; return P(B continues A's recipe)."""
    state = (
        f"{CONTEXT}\n"
        f'Block A ends: "{by_id[a]["text"][-SNIPPET:]}"\n'
        f'Block B is {where(by_id, a, b)}. It starts: "{by_id[b]["text"][:SNIPPET]}"'
    )
    question = {"type": "noul", "instructions": QUESTION, "criteria": SAME}
    return ask(state, {"same": question})["same"]["noul"]


def ask_choice(by_id: dict[str, Box], a: str, bs: list[str], ask: Ask) -> dict[str, float]:
    """Ask which candidate continues A's recipe, with "none" as one more option.

    Returns a probability per option. They are shares of one distribution: they add up to 1
    across the candidates and "none".
    """
    state = f'{CONTEXT}\nBlock A ends: "{by_id[a]["text"][-SNIPPET:]}"'
    options = {b: f'{where(by_id, a, b)}, starts: "{by_id[b]["text"][:SNIPPET]}"' for b in bs}
    options["none"] = "None of them: Block A's recipe ends here."
    question = {
        "type": "choice",
        "instructions": "Which block continues the same recipe as Block A? A sentence cut "
        "off at the end of A and finished in a block is strong evidence.",
        "criteria": options,
    }
    return ask(state, {"next": question})["next"]["probabilities"]


def claims(by_id: dict[str, Box], cands: dict[str, list[str]], mode: str, ask: Ask) -> list[Claim]:
    """Ask the decider about every box that has candidates; return the links it supports."""
    out: list[Claim] = []
    for a, bs in cands.items():
        if mode == "noul" or (mode == "auto" and len(bs) == 1):
            # One yes/no question per candidate: keep each link that clears ACCEPT.
            for b in bs:
                p = ask_noul(by_id, a, b, ask)
                if p >= ACCEPT:
                    out.append((p, a, b))
        else:
            # One ranking question: keep the top option, unless the top option is "none".
            probs = ask_choice(by_id, a, bs, ask)
            best = max(probs, key=probs.__getitem__)
            if best != "none":
                out.append((probs[best], a, best))
    return out


def settle(boxes: list[Box], found: list[Claim]) -> list[list[str]]:
    """Accept links strongest first, and chain them into recipes.

    Each box gets at most one next box and at most one previous box, so a weaker link that
    conflicts with a stronger one is dropped. Returns each recipe as box ids in reading
    order; a box with no accepted link is a recipe on its own.
    """
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
    """Group the boxes of one page; return the groups as lists of box ids."""
    info = page.get("page", {})
    width = info.get("width") or max(b["bbox"][2] for b in page["boxes"])
    # A box that spans the page (the page header) belongs to no recipe: leave it out of the
    # linking, and return it as a group of its own at the end.
    body = [b for b in page["boxes"] if b["bbox"][2] - b["bbox"][0] <= HEADER * width]
    headers = [[b["id"]] for b in page["boxes"] if b not in body]
    cands = candidates(body, rtl=info.get("direction") == "rtl")
    found = claims({b["id"]: b for b in body}, cands, mode, ask)
    return settle(body, found) + headers


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--page", type=Path, default=PAGE, help="page JSON (default: the cookbook)")
    ap.add_argument("--question", choices=["auto", "noul", "choice"], default="auto")
    args = ap.parse_args()
    page = json.loads(args.page.read_text(encoding="utf-8"))
    try:
        groups = run(page, args.question)
    except OSError as e:
        raise SystemExit(
            f"Could not reach strands decider at {URL} ({e}). Start one with:\n"
            "  strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099"
        ) from None

    # Compare with the hand labels: each box's `recipe` field (null for the page header).
    by_id = {b["id"]: b for b in page["boxes"]}
    recipes: dict[str, set[str]] = {}
    for b in page["boxes"]:
        if b.get("recipe"):
            recipes.setdefault(b["recipe"], set()).add(b["id"])
    print("ok = matches a labelled recipe, xx = does not, -- = not part of any recipe\n")
    for g in groups:
        first = by_id[g[0]]
        if set(g) in recipes.values():
            mark = "ok"
        elif len(g) == 1 and not first.get("recipe"):
            mark = "--"
        else:
            mark = "xx"
        # The first words of the group, cut at a word, without a trailing Arabic or Latin comma.
        start = textwrap.shorten(first["text"], 28, placeholder="").rstrip("،,")
        print(f"{mark} {' '.join(g):<16} {start}")
    correct = sum(set(g) in recipes.values() for g in groups)
    print(f"\n{correct}/{len(recipes)} recipes match the labels ({args.question} questions)")


if __name__ == "__main__":
    main()
