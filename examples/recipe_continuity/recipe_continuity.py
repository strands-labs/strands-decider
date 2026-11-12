"""Group the OCR text boxes of a cookbook page into recipes with strands decider.

Three steps: layout proposes which box could come next after each box, strands decider judges
whether the recipe really continues, and a few lines of plain Python settle the links into
recipes. Start a server, then run this file:

    strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099
    python examples/recipe_continuity/recipe_continuity.py

Reading order of this file, top to bottom:

    settings          the layout rules and the words the model sees (CONTEXT, QUESTION, SAME)
    candidates()      step 1: which box could come next after each box, from layout alone
    spans_columns()   finds the page header, which belongs to no recipe
    ask_server()      the HTTP call to the server
    ask_noul()        step 2: one yes/no question about one pair of boxes
    ask_choice()      the same, as one question over several options
    claims()          step 2 for every box: which links the decider supports
    settle()          step 3: accept links strongest first and chain them into recipes
    run(), main()     wire the steps together, and score the result against the labels

walkthrough.ipynb, next to this file, runs these steps one at a time and shows each result.

`--question` picks the question shape. `auto` (the default) uses a `noul` to decide whether a
link exists, and, when a box has several candidates, first a `choice` to rank them. `noul` and
`choice` force one shape everywhere, to compare the two; `choice` then adds a "none" option.
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
# The state is the text the model reads; the questions are named, e.g. {"same": {...}}.
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
    # below[a] = the box just under a: same column, starting below the middle of a (OCR boxes
    # can overlap by a few pixels), and at most MAX_GAP pixels under its bottom edge.
    below: dict[str, Box] = {}
    for a in boxes:
        _, y0, _, y1 = a["bbox"]
        under = [
            b
            for b in boxes
            if b is not a
            and overlap(a, b) >= SAME_COLUMN
            and b["bbox"][1] > (y0 + y1) / 2
            and b["bbox"][1] - y1 <= MAX_GAP
        ]
        if under:  # the highest one; on a tie, the one most in line with a
            below[a["id"]] = min(under, key=lambda b: (b["bbox"][1], -overlap(a, b)))

    # A box that is not under any other box starts a column.
    under_ids = {b["id"] for b in below.values()}
    tops = [b for b in boxes if b["id"] not in under_ids]

    out: dict[str, list[str]] = {}
    for a in boxes:
        if a["id"] in below:
            out[a["id"]] = [below[a["id"]]["id"]]
            continue
        # a ends its column: look for column tops in another column, on the reading side
        # (their centre is to the left of a's centre on a right-to-left page).
        side = [
            b
            for b in tops
            if overlap(a, b) < SAME_COLUMN
            and (centre(b) < centre(a) if rtl else centre(b) > centre(a))
        ]
        if side:
            # Keep the tops of the nearest of those columns, in reading order.
            nearest = min(side, key=lambda b: abs(centre(b) - centre(a)))
            col = [b for b in side if overlap(b, nearest) >= SAME_COLUMN]
            out[a["id"]] = [b["id"] for b in sorted(col, key=lambda b: b["bbox"][1])]
    return out


def centre(b: Box) -> float:
    """Horizontal centre of a box."""
    return (b["bbox"][0] + b["bbox"][2]) / 2


def spans_columns(b: Box, boxes: list[Box]) -> bool:
    """True if b overlaps, horizontally, two boxes in different columns (a header or footer)."""
    over = [c for c in boxes if c is not b and overlap(b, c) >= SAME_COLUMN]
    return any(overlap(c, d) < SAME_COLUMN for c in over for d in over)


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
        try:
            return json.load(resp)["answers"]
        except (ValueError, KeyError, TypeError) as e:
            raise RuntimeError(f"unexpected reply from the server: {e!r}") from None
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


def ask_choice(
    by_id: dict[str, Box], a: str, bs: list[str], ask: Ask, with_none: bool
) -> dict[str, float]:
    """Ask which candidate continues A's recipe, optionally with "none" as one more option.

    Returns a probability per option. They are shares of one distribution: they add up to 1
    across the options.
    """
    state = f'{CONTEXT}\nBlock A ends: "{by_id[a]["text"][-SNIPPET:]}"'
    options = {b: f'{where(by_id, a, b)}, starts: "{by_id[b]["text"][:SNIPPET]}"' for b in bs}
    if with_none:
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
        if mode == "auto":
            # Rank with a choice only when there is something to rank, then decide with a noul.
            top = bs[0]
            if len(bs) > 1:
                probs = ask_choice(by_id, a, bs, ask, with_none=False)
                top = max(bs, key=lambda k: probs.get(k, 0.0))
            p = ask_noul(by_id, a, top, ask)
            if p >= ACCEPT:
                out.append((p, a, top))
        elif mode == "noul":
            # One yes/no question per candidate: keep each link that clears ACCEPT.
            for b in bs:
                p = ask_noul(by_id, a, b, ask)
                if p >= ACCEPT:
                    out.append((p, a, b))
        else:
            # One choice that also decides: keep the top option unless it is "none".
            # "none" comes first in this list, so it wins a tie (the request lists it last).
            # Keys that are not options are ignored, and a missing option counts as 0.
            probs = ask_choice(by_id, a, bs, ask, with_none=True)
            best = max(["none", *bs], key=lambda k: probs.get(k, 0.0))
            if best != "none":
                out.append((probs[best], a, best))
    return out


def settle(boxes: list[Box], found: list[Claim]) -> list[list[str]]:
    """Accept links strongest first, and chain them into recipes.

    Each box gets at most one next box and at most one previous box, so a weaker link that
    conflicts with a stronger one, or that would close a loop, is dropped. Returns each recipe as box ids in reading
    order; a box with no accepted link is a recipe on its own.
    """
    nxt: dict[str, str] = {}
    for _, a, b in sorted(found, reverse=True):
        if a in nxt or b in nxt.values():
            continue
        end = b  # follow the chain from b: if it leads back to a, the link would close a loop
        while end in nxt:
            end = nxt[end]
        if end != a:
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
    boxes = page["boxes"]
    # A box that spans several columns (the page header) belongs to no recipe: leave it out of
    # the linking, and return it as a group of its own at the end. This would also set aside a
    # title spanning two columns, so change it if your pages have those.
    body = [b for b in boxes if not spans_columns(b, boxes)]
    headers = [[b["id"]] for b in boxes if b not in body]
    cands = candidates(body, rtl=page.get("page", {}).get("direction") == "rtl")
    found = claims({b["id"]: b for b in body}, cands, mode, ask)
    return settle(body, found) + headers


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--page", type=Path, default=PAGE, help="page JSON (default: the cookbook)")
    ap.add_argument("--question", choices=["auto", "noul", "choice"], default="auto")
    args = ap.parse_args()
    try:
        page = json.loads(args.page.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SystemExit(f"Could not read the page file {args.page} ({e})") from None
    if not isinstance(page, dict) or not isinstance(page.get("boxes"), list):
        raise SystemExit(f"{args.page} has no list of boxes; see the README for the format")
    try:
        groups = run(page, args.question)
    except OSError as e:
        raise SystemExit(
            f"Could not reach strands decider at {URL} ({e}). Start one with:\n"
            "  strands-decider serve StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --port 8099"
        ) from None
    except (ValueError, RuntimeError, http.client.HTTPException) as e:  # bad URL or reply
        raise SystemExit(str(e)) from None

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
