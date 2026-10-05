"""Check the image training set against every evaluation image, and list near copies.

Evaluation images: NaturalBench shard 0 groups 0-599 (the evaluation's 0-299 and the
held-out temperature groups 300-599), every image in POPE adversarial (the evaluation
uses its first 600 questions), and, with --ijb-dir, the rebuilt Image JevBench preview
items. Two checks: (1) COCO ids: training uses train2014 only and POPE is val2014, and
the id sets are intersected anyway; (2) the perceptual hash (64-bit pHash, computed on a
32x32 greyscale downsample, so insensitive to resizing) of every training image against
every evaluation image: a Hamming distance <= THRESH marks the training image for removal.
Every image is decoded as the server decodes it (strands_decider.vision.read_image).

Writes <data root>/dedupe_drop.txt (training image paths to drop, which training reads
as `dedupe_drop`) and <data root>/dedupe_report.json (every pair within distance 10).

    python data/checks/dedupe_images.py --data-root data/image/build \
        --files vqa.jsonl count.jsonl m2w.jsonl charts.jsonl docs.jsonl tabfact.jsonl [--ijb-dir DIR]

Needs imagehash, pandas and pyarrow.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re

from strands_decider.vision import read_image

THRESH = 6
MAX_PIXELS = 2 * 89_478_485  # what PIL itself opens; the evaluation images are all far smaller


def ph(arg):
    """(key, 64-bit pHash), or (key, None) for an image that cannot be read."""
    import imagehash

    key, src = arg
    try:
        if not isinstance(src, bytes):
            with open(src, "rb") as fh:
                src = fh.read()
        return key, int(str(imagehash.phash(read_image(src, MAX_PIXELS))), 16)
    except ValueError:
        return key, None


def main() -> None:
    import pandas as pd
    from huggingface_hub import hf_hub_download

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--files", nargs="+", required=True, help="training jsonl files (relative to data root)")
    ap.add_argument("--ijb-dir", help="rebuilt Image JevBench preview items (an images/ folder)")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    ev: list[tuple[str, object]] = []
    nb = pd.read_parquet(hf_hub_download("BaiqiL/NaturalBench", "data/train-00000-of-00003.parquet",
                                         repo_type="dataset", revision="ba41a7d564877a9b64c094b08015ca493cc3e54b")).iloc[:600]
    for _, r in nb.iterrows():
        for j in (0, 1):
            ev.append((f"nb-{r['Index']}-i{j}", r[f"Image_{j}"]["bytes"]))
    pope = pd.read_parquet(hf_hub_download("lmms-lab/POPE", "Full/adversarial-00000-of-00001.parquet",
                                           repo_type="dataset", revision="4db1276663dfa5eb8ad16a52d24c31a09e470896"))
    pope_ids = set()
    seen = set()
    for _, r in pope.iterrows():
        src = str(r.get("image_source", ""))
        m = re.search(r"(\d+)$", src)
        if m:
            pope_ids.add(int(m.group(1)))
        if src not in seen:
            seen.add(src)
            ev.append((f"pope-{src}", r["image"]["bytes"]))
    if a.ijb_dir:
        for f in sorted(os.listdir(os.path.join(a.ijb_dir, "images"))):
            ev.append((f"ijb-{f}", os.path.join(a.ijb_dir, "images", f)))

    train_imgs = set()
    for f in a.files:
        for line in open(os.path.join(a.data_root, f)):
            r = json.loads(line)
            train_imgs.update(r.get("images", []))
    train_imgs = sorted(train_imgs)
    coco_train_ids = {int(re.search(r"(\d+)\.jpg$", p).group(1)) for p in train_imgs if p.startswith("coco/")}
    assert all("train2014" in p for p in train_imgs if p.startswith("coco/")), "non-train2014 COCO image"

    with mp.Pool(a.workers) as pool:
        evh = dict(pool.map(ph, ev, chunksize=8))
        trh = dict(pool.map(ph, [(p, os.path.join(a.data_root, p)) for p in train_imgs], chunksize=32))
    import numpy as np
    E = np.array([v for v in evh.values() if v is not None], dtype=np.uint64)
    ek = [k for k, v in evh.items() if v is not None]
    drop, near = [], []
    hist = [0] * 65
    for p, h in trh.items():
        if h is None:
            drop.append(p)
            continue
        x = np.bitwise_xor(E, np.uint64(h))
        d = np.array([bin(int(v)).count("1") for v in x])
        i = int(d.argmin())
        hist[int(d[i])] += 1
        if d[i] <= 10:  # every close pair is listed for inspection; <= THRESH is dropped
            near.append({"train": p, "eval": ek[i], "hamming": int(d[i]), "dropped": bool(d[i] <= THRESH)})
        if d[i] <= THRESH:
            drop.append(p)
    report = {
        "eval_images_hashed": len(ek), "eval_breakdown": {
            "naturalbench_groups_0_599": sum(k.startswith("nb-") for k in ek),
            "pope_adversarial_unique_images": sum(k.startswith("pope-") for k in ek),
            "ijb_preview": sum(k.startswith("ijb-") for k in ek)},
        "train_images_hashed": len(trh), "threshold_hamming": THRESH,
        "dropped": len(drop), "closest_pairs_hamming_le_10": sorted(near, key=lambda x: x["hamming"]),
        "unreadable_train_images": sum(v is None for v in trh.values()),
        "min_hamming_histogram_0_to_16": hist[:17],
        "coco_train2014_ids": len(coco_train_ids), "pope_val2014_ids": len(pope_ids),
        "coco_id_overlap_with_pope": len(coco_train_ids & pope_ids),
    }
    open(os.path.join(a.data_root, "dedupe_drop.txt"), "w").writelines(p + "\n" for p in drop)
    json.dump(report, open(os.path.join(a.data_root, "dedupe_report.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "closest_pairs_hamming_le_10"}, indent=1))


if __name__ == "__main__":
    main()
