"""VQAv2 TRAIN complementary pairs -> minimal pairs (same question, two COCO train2014
images, different answers). yes/no pairs -> yes/no rows; number and other pairs ->
choice rows whose option set is identical for both images (NaturalBench-style).

Inputs (official VQAv2 train files, CC BY 4.0), in --vqa-dir:
  v2_OpenEnded_mscoco_train2014_questions.json, v2_mscoco_train2014_annotations.json,
  v2_mscoco_train2014_complementary_pairs.json
Writes vqa.jsonl and vqa_images.txt, the COCO train2014 images the rows need (coco/<file>).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random

from common import row, write, yesno

NUM = {str(i) for i in range(0, 21)}


def consensus(ann: dict) -> tuple[str, int]:
    c = collections.Counter(a["answer"].strip().lower() for a in ann["answers"])
    ans, n = c.most_common(1)[0]
    return ans, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vqa-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--yesno-pairs", type=int, default=3500)
    ap.add_argument("--number-pairs", type=int, default=800)
    ap.add_argument("--other-pairs", type=int, default=1500)
    ap.add_argument("--min-agree", type=int, default=7, help="of 10 annotators")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    qs = {q["question_id"]: q for q in json.load(open(os.path.join(a.vqa_dir, "v2_OpenEnded_mscoco_train2014_questions.json")))["questions"]}
    anns = {x["question_id"]: x for x in json.load(open(os.path.join(a.vqa_dir, "v2_mscoco_train2014_annotations.json")))["annotations"]}
    pairs = json.load(open(os.path.join(a.vqa_dir, "v2_mscoco_train2014_complementary_pairs.json")))
    rng.shuffle(pairs)

    # Common answers per question prefix: distractors that are plausible for the question.
    by_prefix: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for qid, x in anns.items():
        ans, n = consensus(x)
        if n >= a.min_agree:
            by_prefix[" ".join(qs[qid]["question"].lower().split()[:3])][ans] += 1

    want = {"yes/no": a.yesno_pairs, "number": a.number_pairs, "other": a.other_pairs}
    got = collections.Counter()
    rows, files = [], set()
    for q1, q2 in pairs:
        x1, x2 = anns[q1], anns[q2]
        t = x1["answer_type"]
        if t != x2["answer_type"] or got[t] >= want.get(t, 0):
            continue
        (a1, n1), (a2, n2) = consensus(x1), consensus(x2)
        if min(n1, n2) < a.min_agree or a1 == a2 or x1["image_id"] == x2["image_id"]:
            continue
        question = qs[q1]["question"].strip()
        if question != qs[q2]["question"].strip():
            continue
        imgs = [f"coco/COCO_train2014_{x['image_id']:012d}.jpg" for x in (x1, x2)]
        pid = f"vqa-{q1}-{q2}"
        if t == "yes/no":
            if {a1, a2} != {"yes", "no"}:
                continue
            for img, ans, x in zip(imgs, (a1, a2), (x1, x2), strict=True):
                rows.append(yesno(question, ans == "yes", rng, task="vqa/yesno", source="vqav2",
                                  images=[img], source_id=f"coco-{x['image_id']}", pair_id=pid))
        else:
            if t == "number":
                if not (a1 in NUM and a2 in NUM):
                    continue
                pool = sorted({str(max(0, int(v) + d)) for v in (a1, a2) for d in (-2, -1, 1, 2)} - {a1, a2},
                              key=lambda _: rng.random())
            else:
                pre = " ".join(question.lower().split()[:3])
                pool = [w for w, _ in by_prefix[pre].most_common(12) if w not in (a1, a2)]
                rng.shuffle(pool)
            k = rng.choice([2, 2, 3])  # 4-5 options in all
            if len(pool) < k:
                continue
            names = [a1, a2, *pool[:k]]
            rng.shuffle(names)
            opts = [[chr(65 + i), n] for i, n in enumerate(names)]
            for img, ans, x in zip(imgs, (a1, a2), (x1, x2), strict=True):
                rows.append(row("choice", question, opts, names.index(ans), task=f"vqa/{t}", source="vqav2",
                                images=[img], source_id=f"coco-{x['image_id']}", pair_id=pid))
        files.update(imgs)
        got[t] += 1
        if all(got[k] >= v for k, v in want.items()):
            break
    print(f"[vqa] pairs {dict(got)}; images {len(files)}")
    write(os.path.join(a.out, "vqa.jsonl"), rows)
    with open(os.path.join(a.out, "vqa_images.txt"), "w") as fh:
        fh.writelines(f + "\n" for f in sorted(files))


if __name__ == "__main__":
    main()
