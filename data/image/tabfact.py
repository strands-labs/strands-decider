"""TabFact TRAIN tables rendered as images; statements -> yes/no "supported by the table?".

Uses only tables listed in data/train_id.json (the val/test tables are never read).
Statements and labels come from collected_data/r1_training_all.json and
r2_training_all.json ({table: [statements, labels(1 entailed / 0 refuted), caption]}).
Tables are drawn with PIL in a few styles (grid, zebra, plain), some "scanned".
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import random

from common import write, yesno
from documents import pick_font, scanned

ROOT = ""


def render(args):
    idx, tid, stmts, caption, out, seed = args
    from PIL import Image, ImageDraw

    rng = random.Random(seed)
    with open(os.path.join(ROOT, "data/all_csv", tid), encoding="utf-8") as fh:
        rows = list(csv.reader(fh, delimiter="#"))
    if not rows or len(rows) > 22 or len(rows[0]) > 8:
        return []
    f = pick_font(rng, rng.choice([14, 15, 16]))
    fb = pick_font(rng, 16, True)
    widths = [min(260, max(int(f.getlength(r[c])) if c < len(r) else 0 for r in rows) + 18)
              for c in range(len(rows[0]))]
    W = sum(widths) + 60
    H = 30 * len(rows) + 110
    if W > 1800:
        return []
    img = Image.new("RGB", (max(W, 500), H), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    dr.text((30, 20), caption[:120], fill=0, font=pick_font(rng, 18, True))
    style = rng.choice(["grid", "zebra", "plain"])
    y = 60
    for i, r in enumerate(rows):
        x = 30
        if style == "zebra" and i % 2 == 1:
            dr.rectangle([30, y - 3, 30 + sum(widths), y + 25], fill=(238, 241, 247))
        if i == 0:
            dr.rectangle([30, y - 3, 30 + sum(widths), y + 25], fill=(220, 226, 238))
        for c, w in enumerate(widths):
            txt = r[c] if c < len(r) else ""
            while txt and f.getlength(txt) > w - 10:
                txt = txt[:-1]
            dr.text((x + 5, y), txt, fill=0, font=fb if i == 0 else f)
            if style == "grid":
                dr.rectangle([x, y - 3, x + w, y + 25], outline=(150, 150, 150))
            x += w
        y += 30
    img = scanned(img, rng)
    name = f"tables/t{idx:06d}.png"
    img.save(os.path.join(out, name))
    out_rows = []
    pick = list(zip(*stmts, strict=False))
    rng.shuffle(pick)
    for s, lab in pick[:2]:
        q = rng.choice([f"Based on the table, is this statement true: {s}?",
                        f"Does the table support the statement \"{s}\"?",
                        f"According to the table, {s}. Is that correct?"])
        out_rows.append(yesno(q, int(lab) == 1, rng, task="table/tabfact", source="tabfact",
                              images=[name], source_id=f"tabfact-{tid}"))
    return out_rows


def main() -> None:
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="Table-Fact-Checking clone")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-tables", type=int, default=1300)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=5)
    a = ap.parse_args()
    ROOT = a.repo
    train_ids = set(json.load(open(os.path.join(a.repo, "data/train_id.json"))))
    held = set()
    for split in ("val_id.json", "test_id.json"):
        p = os.path.join(a.repo, "data", split)
        if os.path.exists(p):
            held |= set(json.load(open(p)))
    assert not (train_ids & held)
    stmts = {}
    for f in ("r1_training_all.json", "r2_training_all.json"):
        for tid, (ss, ls, cap) in json.load(open(os.path.join(a.repo, "collected_data", f))).items():
            if tid in train_ids:
                prev = stmts.get(tid, ([], [], cap))
                stmts[tid] = (prev[0] + ss, prev[1] + ls, cap)
    rng = random.Random(a.seed)
    tids = sorted(stmts)
    rng.shuffle(tids)
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    jobs = [(i, t, (stmts[t][0], stmts[t][1]), stmts[t][2], a.out, a.seed * 7919 + i)
            for i, t in enumerate(tids[: int(a.n_tables * 1.3)])]
    with mp.Pool(a.workers, initializer=_init, initargs=(a.repo,)) as pool:
        rows = [r for part in pool.imap(render, jobs, chunksize=8) for r in part]
    write(os.path.join(a.out, "tabfact.jsonl"), rows[: 2 * a.n_tables])


def _init(repo: str) -> None:
    global ROOT
    ROOT = repo


if __name__ == "__main__":
    main()
