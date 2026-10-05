"""COCO Captions 2014 TRAIN rows for stage-1 alignment of a grafted encoder
(strands_decider.graft_align).

One caption per image, `--n` images drawn at random (seeded) from captions_train2014.json
(annotations_trainval2014.zip; COCO annotations are CC BY 4.0, the images are Flickr's
under their own licences). Only train2014 images are accepted: POPE evaluates on COCO
val2014, so a val2014 image here would be an evaluation image seen in training.

    python data/image/captions.py --captions data/raw/coco/annotations/captions_train2014.json \\
        --out data/image/build --n 80000

Writes captions.jsonl (rows: images, caption, source, source_id) and
captions_images.txt (the image paths, relative to --out, for the download step).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from typing import Any

TRAIN_PREFIX = "COCO_train2014_"


def clean(caption: str) -> str:
    """One line, starting upper-case and ending in punctuation, as a sentence reads."""
    text = re.sub(r"\s+", " ", caption).strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def build(captions: dict[str, Any], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """`n` rows (or every image, if fewer), one caption each; refuses any val2014 image."""
    files: dict[int, str] = {}
    for im in captions["images"]:
        name = im["file_name"]
        if "val2014" in name or not name.startswith(TRAIN_PREFIX):
            raise ValueError(f"{name}: not a COCO train2014 image; POPE evaluates on val2014, "
                             "so captions come from train2014 only")
        files[im["id"]] = name
    by_image: dict[int, list[str]] = {}
    for a in captions["annotations"]:
        if a["image_id"] not in files:
            raise ValueError(f"caption {a.get('id')} names image {a['image_id']}, not in the file")
        text = clean(a["caption"])
        if text:
            by_image.setdefault(a["image_id"], []).append(text)
    rng = random.Random(seed)
    ids = sorted(by_image)
    rng.shuffle(ids)
    rows = []
    for i in sorted(ids[:n]):
        rows.append({"images": [f"coco/{files[i]}"], "caption": rng.choice(sorted(by_image[i])),
                     "source": "coco_captions_train2014", "source_id": f"coco-{i}"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions", required=True, help="captions_train2014.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=80_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    with open(a.captions, encoding="utf-8") as fh:
        rows = build(json.load(fh), a.n, a.seed)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "captions.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "captions_images.txt"), "w", encoding="utf-8") as fh:
        fh.write("".join(r["images"][0] + "\n" for r in rows))
    print(f"[build] {a.out}/captions.jsonl: {len(rows)} rows, one caption per image")


if __name__ == "__main__":
    main()
