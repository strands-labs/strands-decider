"""Set-of-mark screen rows from Multimodal-Mind2Web TRAIN screenshots.

For each action: crop a viewport around the target element, pick 2-4 negative
candidates fully inside it, draw letter markers on target and negatives in one of two
styles (outlined boxes with a letter tag, or a circled letter at the element's centre),
and ask which marker to act on:
  web goal   "Browser goal: <task> Next labelled element for action <OP>?"
  screen     "Goal: <element text> Which labelled marker should be clicked?"
TRAIN split only (the Image JevBench preview's Mind2Web items come from test_task), and
not the preview items' website (`--exclude-websites`).
"""

from __future__ import annotations

import argparse
import glob
import json
import multiprocessing as mp
import os
import random
import re

from common import row, write

from strands_decider.vision import read_image

# Full-page screenshots run to tens of thousands of pixels tall; the server's 16 MP limit
# would refuse them, so the builder allows what PIL itself opens.
MAX_PIXELS = 2 * 89_478_485
FONT_DIRS = ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation"]
COLORS = [(220, 30, 30), (30, 120, 220), (20, 160, 60), (160, 40, 200), (230, 120, 0), (0, 150, 150)]
INTERACTIVE = {"a", "button", "input", "select", "textarea", "label", "img", "svg", "span", "li", "div", "option"}


def font(size: int):
    from PIL import ImageFont
    for d in FONT_DIRS:
        for f in sorted(glob.glob(os.path.join(d, "*Bold*.ttf"))) or sorted(glob.glob(os.path.join(d, "*.ttf"))):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def bbox(cand: str):
    c = json.loads(cand)
    attrs = json.loads(c["attributes"])
    b = attrs.get("bounding_box_rect")
    if not b:
        return None
    x, y, w, h = (float(v) for v in b.split(","))
    return c["tag"], (x, y, x + w, y + h)


def target_text(repr_: str) -> str:
    # "[button]  Search -> CLICK" -> "Search"
    m = re.match(r"\[(.*?)\]\s*(.*?)\s*->", repr_ or "")
    return (m.group(2) if m else "").strip()


def process(args):
    path, out_dir, seed, per_file, exclude = args
    import pyarrow.parquet as pq
    from PIL import ImageDraw

    rng = random.Random(seed)
    pf = pq.ParquetFile(path)
    rows = []
    for rg in range(pf.num_row_groups):
        tb = pf.read_row_group(rg, columns=["action_uid", "operation", "pos_candidates", "neg_candidates",
                                            "confirmed_task", "screenshot", "annotation_id", "action_reprs",
                                            "target_action_index", "website"])
        for r in tb.to_pylist():
            if len(rows) >= per_file:
                return rows
            if r["website"] in exclude:  # websites of the rebuilt preview items: never trained on
                continue
            if not r["pos_candidates"] or r["screenshot"] is None:
                continue
            pos = bbox(r["pos_candidates"][0])
            if pos is None:
                continue
            _, (x0, y0, x1, y1) = pos
            if x1 - x0 < 8 or y1 - y0 < 8 or (x1 - x0) * (y1 - y0) > 0.15 * 1280 * 900:
                continue
            try:
                im = read_image(r["screenshot"]["bytes"], MAX_PIXELS)
            except ValueError:
                continue
            W, H = im.size
            vh = rng.randint(720, 1000)
            top = int(max(0, min(H - vh, (y0 + y1) / 2 - rng.uniform(0.2, 0.8) * vh)))
            crop = (0, top, W, min(H, top + vh))
            if not (y0 >= crop[1] and y1 <= crop[3] and x1 <= W):
                continue
            negs = []
            for c in r["neg_candidates"]:
                b = bbox(c)
                if not b:
                    continue
                tag, (a0, b0, a1, b1) = b
                if tag not in INTERACTIVE or a1 - a0 < 8 or b1 - b0 < 8:
                    continue
                if not (b0 >= crop[1] and b1 <= crop[3] and a1 <= W and a0 >= 0):
                    continue
                if (a1 - a0) * (b1 - b0) > 0.15 * W * vh:
                    continue
                ix = max(0, min(x1, a1) - max(x0, a0)) * max(0, min(y1, b1) - max(y0, b0))
                if ix > 0.3 * min((x1 - x0) * (y1 - y0), (a1 - a0) * (b1 - b0)):
                    continue  # overlaps the target: ambiguous
                negs.append((a0, b0, a1, b1))
            # Prefer negatives near the target (harder), plus a few anywhere.
            negs.sort(key=lambda b: abs((b[1] + b[3]) / 2 - (y0 + y1) / 2) + abs((b[0] + b[2]) / 2 - (x0 + x1) / 2))
            pool = negs[:12]
            rng.shuffle(pool)
            k = rng.randint(2, 4)
            chosen = []
            for b in pool:
                if all(max(0, min(b[2], c[2]) - max(b[0], c[0])) * max(0, min(b[3], c[3]) - max(b[1], c[1])) == 0
                       for c in chosen):
                    chosen.append(b)
                if len(chosen) == k:
                    break
            if len(chosen) < 2:
                continue
            boxes = [(x0, y0, x1, y1), *chosen]
            letters = [chr(65 + i) for i in range(len(boxes))]
            rng.shuffle(letters)
            img = im.crop(crop)
            dr = ImageDraw.Draw(img)
            style = rng.choice(["box", "circle"])
            for (a0, b0, a1, b1), L in zip(boxes, letters, strict=True):
                a0, a1, b0, b1 = a0, a1, b0 - top, b1 - top
                col = rng.choice(COLORS)
                if style == "box":
                    dr.rectangle([a0, b0, a1, b1], outline=col, width=3)
                    f = font(16)
                    tw = 14
                    dr.rectangle([a0, max(0, b0 - 18), a0 + tw, max(18, b0)], fill=col)
                    dr.text((a0 + 2, max(0, b0 - 18)), L, fill=(255, 255, 255), font=f)
                else:
                    cx, cy, rad = (a0 + a1) / 2, (b0 + b1) / 2, 13
                    dr.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=(170, 0, 0), width=3,
                               fill=(255, 255, 255))
                    f = font(17)
                    dr.text((cx - 6, cy - 10), L, fill=(0, 0, 0), font=f)
            gold = letters[0]
            name = f"m2w/{r['action_uid']}.jpg"
            img.save(os.path.join(out_dir, name), quality=90)
            op = json.loads(r["operation"])["op"] if r["operation"] else "CLICK"
            reprs = r.get("action_reprs") or []
            ti = int(r["target_action_index"]) if r["target_action_index"] is not None else 0
            text = target_text(reprs[ti]) if ti < len(reprs) else ""
            order = sorted(letters)
            if rng.random() < 0.6 or not text or len(text) > 60:
                q = f"Browser goal: {r['confirmed_task']} Next labelled element for action {op}?"
                if rng.random() < 0.3 and ti > 0:
                    prev = "; ".join(reprs[max(0, ti - 3):ti])
                    q = f"Browser goal: {r['confirmed_task']} Previous actions: {prev}. Next labelled element for action {op}?"
                opts = [[L, f"Choose element {L}"] for L in order]
                task = "screen/m2w_goal"
            else:
                verb = {"TYPE": "type into", "SELECT": "select"}.get(op, "click")
                q = f"Goal: {verb} {text} Which labelled marker should be clicked?"
                opts = [[L, f"Click marker {L}"] for L in order]
                task = "screen/m2w_element"
            rows.append(row("choice", q, opts, order.index(gold), task=task, source="mind2web",
                            images=[name], source_id=f"m2w-{r['annotation_id']}"))
            rows[-1]["website"] = r["website"]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquets", nargs="+", required=True)
    ap.add_argument("--out", required=True, help="data root; images go to <out>/m2w/")
    ap.add_argument("--per-file", type=int, default=220)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--exclude-websites", nargs="*", default=["budget"],
                    help="websites of the Image JevBench preview's Mind2Web items (all 12 are budget.com)")
    a = ap.parse_args()
    assert all("/train-" in p for p in a.parquets), "TRAIN split only"
    os.makedirs(os.path.join(a.out, "m2w"), exist_ok=True)
    jobs = [(p, a.out, i, a.per_file, set(a.exclude_websites)) for i, p in enumerate(sorted(a.parquets))]
    with mp.Pool(a.workers) as pool:
        rows = [r for part in pool.imap(process, jobs) for r in part]  # ordered: deterministic file
    write(os.path.join(a.out, "m2w.jsonl"), rows)


if __name__ == "__main__":
    main()
