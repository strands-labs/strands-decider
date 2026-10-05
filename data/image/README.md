# Image training data

The builders of an image training set: about 38,000 questions over images, written as the
in-tree `Example` rows plus the image paths and their provenance ([`common.py`](common.py)).
Each builder is a script over its downloaded source (training splits only; `--help` lists
its inputs), and `data/checks/dedupe_images.py` checks the result against every evaluation
image. `training/recipe_images.sh fetch build dedupe` runs them all in order for the image
configs (`configs/vision/`). Nothing here is committed but the code: the rows and images
land in `data/image/build/`, which git ignores.

Yes/no questions are written 60% as nouls with the server's default criteria and 40% as
yes/no choices, the form NaturalBench and Image JevBench ask in. Every gold answer comes
from the source's annotation or, for the rendered images, from the values that were drawn.

| Builder | Rows | Images | Source and licence | Questions |
| --- | --- | --- | --- | --- |
| [`vqa.py`](vqa.py) | 11,600 | 10,134 | VQAv2 train annotations and complementary pairs (CC BY 4.0); COCO train2014 images | minimal pairs: one question, two images, two different answers, 7 of 10 annotators agreeing: 3,500 yes/no pairs, 800 number pairs, 1,500 other pairs with one option set for both images |
| [`count.py`](count.py) | 4,400 | 3,716, all among the VQA images | COCO train2014 instance annotations (CC BY 4.0) | how many (2,600; non-crowd, no tiny instances, at most 8), at least N (1,000), presence with co-occurring absent objects as negatives (800) |
| [`m2w.py`](m2w.py) | 6,200 | 6,200 | Multimodal-Mind2Web train split at revision `1b4c6a8cf9f77b7a5e0d641959935c80c4a05889` (Hugging Face card: OpenRAIL; Mind2Web: CC BY 4.0); markers drawn here | a viewport crop around the target with 3-5 lettered markers: which marker for the next action of a browser goal (3,891), or which to click for a named element (2,309); the website of the Image JevBench preview's Mind2Web items (`budget`) is left out |
| [`charts.py`](charts.py) | 7,667 | 2,599 | rendered here with matplotlib, no third-party data | extremes, comparisons, printed values, differences, peaks, trends, grouped bars, pies, 2x2 panels; numeric distractors placed so the gold's rank is uniform |
| [`documents.py`](documents.py) | 5,789 | 1,500 | rendered here with Pillow, no third-party data | invoices, receipts, orders and quotes, forms with checkboxes and signatures, multi-year financial tables (values, changes, sums), some made to look scanned |
| [`tabfact.py`](tabfact.py) | 2,484 | 1,271 | TabFact train tables (MIT; tables from Wikipedia, CC BY-SA) at commit `2ab782ba42b5808076ac91fec846473aa5315a79`, rendered here | is this statement supported by the table |

Training adds two things to these rows (`strands_decider.vision_train`): 3,000 of v19's
own committed text training rows (`data/synthetic/generated_v16.jsonl`,
`generated_v18.jsonl`, `adequacy_gen.jsonl`), or, for the hobson-v20-vl configs, 3,000 of
v20's training rows with their 27B-teacher targets; and an image-removed copy of 15% of
the image rows with at most 9 options, trained only toward the frozen torso's reading of
the text-only prompt (with `kl_frozen_skip_kinds: ["noul"]`, only the choice and score
copies; the yes/no copies then carry no loss).

## Hashes of record

The recorded image-trained runs trained on files with these sha256 hashes:

| File | sha256 |
| --- | --- |
| `charts.jsonl` | `1c91521811faf96f08aa8e70b323f07687ff033d0766aa9fdae550008cc3a4a3` |
| `count.jsonl` | `09a27833249517a745a2e44e6417a83002735b4ba50c1ec8c4f5418a8367bd5a` |
| `docs.jsonl` | `eceece7a2472c7f3c6845df1f91cf59f02e527fc997072b63921bd3f6c7befee` |
| `m2w.jsonl` | `013bcdc7b147327a29a838f18d792ab0c20f141e4a28e4a3a64ced3bd43373b0` |
| `tabfact.jsonl` | `7db5268e0c0b5297706f2ac332378c77859227082e131079231a12b0c9fa2b43` |
| `vqa.jsonl` | `a7847c38daedf0020f1c61db2eb5a230d64ebd4083943d11b5ccd0290ac26d52` |

Every builder is seeded. Two builds on two machines gave the same hashes for `charts`,
`count`, `documents` and `tabfact`, and the same row counts and question mix for `vqa`
and `m2w`, whose bytes differed; the first build was not kept, so the difference was not
traced. The VQAv2 and COCO downloads have no revision, so a change upstream would change
`vqa` and `count`. The rendered images depend on the installed fonts and matplotlib. Since
these runs, the builders and training decode every image through
`strands_decider.vision.read_image`, as the server does (EXIF rotation applied,
transparency on white); a COCO photo stored with an EXIF rotation now trains upright.

## Overlap with the evaluation

`data/checks/dedupe_images.py` compares every training image with every evaluation image:
NaturalBench shard 0 groups 0-599, all 500 POPE adversarial images and the 128 rebuilt
Image JevBench preview items. For the recorded build: the 10,134 COCO images are all
train2014 and POPE's are val2014 (id overlap 0); of the 21,704 training images none is
within Hamming distance 6 of an evaluation image's 64-bit perceptual hash, so nothing was
dropped; the closest pair is at 8 (a snowy park against an insect on leather, inspected)
and 15 more are at 10. A first build had 6 Mind2Web crops within distance 6 of preview
items, all from budget.com, the preview items' website; `m2w.py` has excluded it since.
No NaturalBench, POPE or Image JevBench item, image or question is in the training rows.
