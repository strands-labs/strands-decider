"""Counting and presence on COCO train2014 instance annotations (inventory-like).

Only (image, category) cells whose instances are all non-crowd and not tiny are used,
so the annotated count is the visible count. Rows:
  choice  "How many <plural> are in the image?"  4 numeric options around the gold
  yes/no  "Are there at least N <plural> ...?"   N = gold or gold+1 (balanced)
  yes/no  "Is there a <cat> in the image?"        absent categories that co-occur often
                                                   with what is present (hard negatives)
Input: instances_train2014.json (COCO annotations, CC BY 4.0).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random

from common import row, write, yesno

PLURAL = {"person": "people", "mouse": "mice", "sheep": "sheep", "skis": "pairs of skis",
          "scissors": "pairs of scissors", "bus": "buses", "couch": "couches", "glass": "glasses",
          "wine glass": "wine glasses", "knife": "knives", "sandwich": "sandwiches", "bench": "benches",
          "toothbrush": "toothbrushes", "hair drier": "hair driers", "dining table": "dining tables",
          "broccoli": "pieces of broccoli"}


def plural(n: str) -> str:
    return PLURAL.get(n, n + "s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-count", type=int, default=2600)
    ap.add_argument("--n-atleast", type=int, default=1000)
    ap.add_argument("--n-presence", type=int, default=800)
    ap.add_argument("--prefer-images", help="file list; reuse these images first (fewer downloads)")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    d = json.load(open(a.instances))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}
    cells: dict[int, dict[int, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for x in d["annotations"]:
        cells[x["image_id"]][x["category_id"]].append(x)
    co = collections.Counter()
    for cc in cells.values():
        ks = sorted(cc)
        for i in ks:
            for j in ks:
                if i != j:
                    co[(i, j)] += 1

    prefer = set()
    if a.prefer_images:
        prefer = {int(f.strip().split("_")[-1].split(".")[0]) for f in open(a.prefer_images) if f.strip()}
    ids = list(cells)
    rng.shuffle(ids)
    ids.sort(key=lambda i: i not in prefer)  # stable: preferred first, shuffled within

    rows, files = [], set()
    n_count = n_atl = n_pres = 0
    for iid in ids:
        info = imgs[iid]
        area = info["width"] * info["height"]
        ok = [c for c, xs in cells[iid].items()
              if all(not x["iscrowd"] and x["area"] >= 0.004 * area for x in xs) and len(xs) <= 8]
        if not ok:
            continue
        f = f"coco/COCO_train2014_{iid:012d}.jpg"
        c = rng.choice(ok)
        name, n = cats[c], len(cells[iid][c])
        used = False
        r = rng.random()
        if n_count < a.n_count and (r < 0.6 or n >= 3):
            lo = max(0, n - rng.randint(0, 3))
            nums = list(range(lo, lo + 4))
            q = rng.choice([f"How many {plural(name)} are in the image?",
                            f"How many {plural(name)} can be seen?",
                            f"Count the {plural(name)} in this picture. How many are there?"])
            rows.append(row("choice", q, [[str(k), ""] for k in nums], nums.index(n), task="count/how_many",
                            source="coco_count", images=[f], source_id=f"coco-{iid}"))
            n_count += used or 1
            used = True
        elif n_atl < a.n_atleast and n >= 1:
            k = n if rng.random() < 0.5 else n + 1
            if k >= 2:
                q = f"Are there at least {k} {plural(name)} in the image?"
                rows.append(yesno(q, n >= k, rng, task="count/at_least", source="coco_count",
                                  images=[f], source_id=f"coco-{iid}"))
                n_atl += 1
                used = True
        if n_pres < a.n_presence and rng.random() < 0.35:
            present = set(cells[iid])
            if rng.random() < 0.5:
                cand = sorted((co[(p, x)], x) for p in present for x in cats if x not in present)
                cand = [x for _, x in cand[-10:]]
                if cand:
                    x = rng.choice(cand)
                    rows.append(yesno(f"Is there a {cats[x]} in the image?", False, rng, task="count/presence",
                                      source="coco_count", images=[f], source_id=f"coco-{iid}"))
                    n_pres += 1
                    used = True
            else:
                rows.append(yesno(f"Is there a {name} in the image?", True, rng, task="count/presence",
                                  source="coco_count", images=[f], source_id=f"coco-{iid}"))
                n_pres += 1
                used = True
        if used:
            files.add(f)
        if n_count >= a.n_count and n_atl >= a.n_atleast and n_pres >= a.n_presence:
            break
    print(f"[count] how_many {n_count} at_least {n_atl} presence {n_pres}; images {len(files)}")
    write(os.path.join(a.out, "count.jsonl"), rows)
    with open(os.path.join(a.out, "count_images.txt"), "w") as fh:
        fh.writelines(x + "\n" for x in sorted(files))


if __name__ == "__main__":
    main()
