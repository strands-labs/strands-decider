# strands-decider-2B-hobson-v20-balanced

> **Unofficial candidate.** One 2B checkpoint for text and images: hobson-v20 continued on
> questions over images with Qwen3.5's own vision tower. Estimated **best text+image 2B
> candidate** (the only 2B candidate on both boards) and **#2 on Image JevBench** in the
> 2B class (not submitted; estimates from local measures). Not a published Strands release.

| | |
| --- | --- |
| What it is | The [hobson-v20](strands-decider-2B-hobson-v20.md) soup continued for one epoch on ~38,000 questions over images, with image-removed copies, no frozen-KL anchor on yes/no rows, and 3,000 replayed text rows trained toward their 27B-teacher targets, so that one checkpoint keeps v20's text behaviour and reads images |
| Base | `Qwen/Qwen3.5-2B-Base` (its own vision tower, frozen), trained at revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c` |
| Size | 2B class: the 1.9B-parameter decoder and Qwen3.5's vision tower (frozen), a LoRA adapter and a pointer head; the archive holds the adapter and head (226 MB) |
| Answers | text, and text with images (`serve --vision`) |
| Configs | [configs/vision/strands-decider-2B-hobson-v20-vl.yaml](../../configs/vision/strands-decider-2B-hobson-v20-vl.yaml), `-seed1`, `-seed2`: **the package is seed 0 of the `strands-decider-2B-hobson-v20-vl` configs** |
| Download | https://vision-decider-643603452951-us-east-1.s3.us-east-1.amazonaws.com/weights/strands-decider-2B-hobson-v20-balanced.tar |
| sha256 | `3dbcee592919bc6cb1ac73ba6bd4650b482716de413f4518e976aad70ff9db60` ([weights_SHA256SUMS.txt](https://vision-decider-643603452951-us-east-1.s3.us-east-1.amazonaws.com/weights/weights_SHA256SUMS.txt)) |

## Results

Local measures, never board scores ([docs/evaluating.md](../evaluating.md)). Images:
NaturalBench (300 groups), POPE-adversarial (600 items) and the 60 Image JevBench preview
items that can be rebuilt exactly from their sources (the official set is sealed), all at
the 400,000-pixel image budget, without image temperatures.

| Measure | balanced (seed 0) | v19 + `--vision` | Mapika decider-2b-vision |
| --- | --- | --- | --- |
| JevBench v1 public, v1.5-style Intelligence | **39.4** (3 seeds: 39.4 / 39.0 / 38.7) | 29.5 | — |
| Tasks right, of 231 | **175** | 167 | — |
| Yes/no answers inside 0.2-0.8, of 74 | **36** | 48 | — |
| ECE, public tasks | 0.041 | 0.050 | — |
| NaturalBench accuracy | **0.802** | 0.787 | 0.794 |
| NaturalBench G-Acc | **0.360** | 0.347 | 0.360 |
| POPE-adversarial accuracy | **0.865** | 0.858 | 0.860 |
| Image JevBench preview, right of 60 | **43** | 40 | 38 |
| Mean confidence with the image removed, NaturalBench / POPE (lower is better) | **0.61 / 0.59** | 0.65 / 0.73 | — |
| Text check (899 held-out text rows), accuracy | 0.829 | 0.817 | — |

- It is better than v19 served with `--vision` on both text and images, from one
  checkpoint.
- **Estimated board position (unofficial):** Image JevBench 2B class about **#2**, behind
  imajev 2B (74.1) and ahead of Mapika (67.1), estimated from the preview items. Its text
  Capability is not estimated: it has not been run on the unseen set yet.
- A first image stage, which kept the frozen-KL anchor on yes/no rows and replayed text
  rows with plain labels, scored 29.3 on the public text tasks (the v20 soup before it:
  39.7); with the two opt-in changes below the three seeds score 39.0 on average, with
  image numbers about unchanged (NaturalBench 0.801 and 0.802, POPE 0.877 and 0.874).

Every number per seed is in the results log:
https://github.com/Vivek0712/strands-decider/blob/ed29b08bb71aad26133bb6c45e8027648f5725ef/RESULTS.md.

## Download and serve

```bash
NAME=strands-decider-2B-hobson-v20-balanced
URL=https://vision-decider-643603452951-us-east-1.s3.us-east-1.amazonaws.com/weights
curl -fO "$URL/$NAME.tar" && curl -fO "$URL/weights_SHA256SUMS.txt"
grep " $NAME.tar\$" weights_SHA256SUMS.txt | sha256sum -c -
mkdir -p checkpoints && tar -xf "$NAME.tar" -C checkpoints
pip install "strands-decider[vision]"                  # Pillow; transformers >= 5.18
strands-decider serve checkpoints/$NAME --vision --image-long-side 0 --image-max-pixels 400000 --port 8000
```

It was trained at a budget of 400,000 pixels per image, which
`--image-long-side 0 --image-max-pixels 400000` reproduces (`serve --vision` otherwise caps
the longer side at 448 px). Requests carry images as [docs/vision.md](../vision.md)
describes; text requests take the text path.

## Verify it yourself

On one NVIDIA GPU, with the archive unpacked as above and v19 at its pinned revision
([docs/evaluating.md](../evaluating.md#setup)):

```bash
# Text: JevBench v1 public through the vision server, against v19
PY=$(which python) GPU=0 SERVE_ARGS=--vision evaluation/jevbench/jevbench.sh checkpoints/$NAME reports/jevbench/$NAME
PY=$(which python) GPU=0 evaluation/jevbench/jevbench.sh checkpoints/v19 reports/jevbench/v19
python evaluation/jevbench/v15_proxy.py reports/jevbench/$NAME --vs reports/jevbench/v19

# Images: NaturalBench, POPE (and, with IJB_JSONL, the 60 preview items) at 400,000 pixels,
# with and without the image, plus the 899-row text check; then against v19 + --vision
pip install pandas pyarrow
CKPT=checkpoints/$NAME OUT=reports/$NAME training/recipe_images.sh eval
CKPT=checkpoints/v19 OUT=reports/v19-400k training/recipe_images.sh eval
python evaluation/vision/compare.py reports/$NAME/eval --vs reports/v19-400k/eval
```

`recipe_images.sh eval` writes `eval/` (the table's numbers), `eval-T/` (the same answers
under image temperatures fitted on NaturalBench groups 300-599, which no evaluation uses)
and `text_check.json`. The preview items are rebuilt by a separate builder
([evaluation/vision/README.md](../../evaluation/vision/README.md)); without `IJB_JSONL`
that column is skipped.

## Retrain

Linux with an NVIDIA GPU; the recorded runs used one H200 per seed. First build
[hobson-v20](strands-decider-2B-hobson-v20.md#retrain) (its calibrated soup at
`checkpoints/strands-decider-2B-hobson-v20` and its 27B teacher file), then:

```bash
pip install -e ".[train,vision,cuda]"
pip install matplotlib imagehash pandas pyarrow       # image data, dedupe, image evaluation
training/recipe_images.sh fetch build dedupe
python -m strands_decider.data.teacher_yn replay --teacher data/teacher_yn_qwen35-27b.jsonl \
  --n-yes-no 2000 --n-other 1000 --out data/replay_teacher_yn.jsonl
for s in "" -seed1 -seed2; do
  CONFIG=configs/vision/strands-decider-2B-hobson-v20-vl$s.yaml training/recipe_images.sh train
done
for s in 0 1 2; do       # the choice/score refit; seed 0 is the package
  strands-decider calibrate checkpoints/strands-decider-2B-hobson-v20-vl-s$s --kinds choice,score \
    --objective nll --limit 900 --data data/multistep_v14_eval.jsonl --data data/holdout_v5_norule.jsonl
done
for s in "" -seed1 -seed2; do   # evaluate the calibrated checkpoints
  CONFIG=configs/vision/strands-decider-2B-hobson-v20-vl$s.yaml training/recipe_images.sh eval
done
```

- `fetch` downloads the training sources' train splits (~45 GB, mostly Multimodal-Mind2Web),
  `build` writes the six row files and their images
  ([data/image/README.md](../../data/image/README.md), with the hashes of record), and
  `dedupe` checks every training image against every evaluation image.
- `replay` draws 3,000 of v20's training rows (2,000 yes/no) with their teacher
  distributions; the recorded runs used a file built the same way whose exact draw was not
  kept, so a rebuild replays different rows.
- Training starts from the calibrated v20 soup (`init_from`) and keeps its text
  temperatures; the refit above is the choice/score one of
  [docs/evaluating.md](../evaluating.md#calibration-refitting-only-some-temperatures), on
  the two files the recorded refits used, and it keeps the yes/no temperature. Evaluate
  after calibrating: `eval` reads the checkpoint's temperatures.
- To serve with image temperatures, write the fitted ones into the checkpoint:
  `python evaluation/vision/temps.py fit --run reports/strands-decider-2B-hobson-v20-vl-s0/cal --out
  reports/strands-decider-2B-hobson-v20-vl-s0/image_temps.json --checkpoint checkpoints/strands-decider-2B-hobson-v20-vl-s0`.
  The published package has none.

What this adds, all opt-in (`serve --vision` on v19 answers as before):

| Piece | What it does |
| --- | --- |
| `python -m strands_decider.vision_train` | Continues a checkpoint on the `data/image/` builders' rows: LoRA and head trainable, the vision tower frozen; label loss plus the frozen torso's reading, image-removed copies trained only toward the frozen reading of the text-only prompt, replayed text rows. `kl_frozen_skip_kinds` and teacher targets on replay rows (`teacher_weight`) keep a text recipe's behaviour through the image stage. With `kl_frozen_skip_kinds: ["noul"]`, as here, only the choice and score image-removed copies train toward the text-only reading; the yes/no copies carry no loss |
| `serve --vision --image-max-pixels N` | A pixel budget per image instead of the long-side cap, so an image costs about the same tokens whatever its shape |
| `image_temperature_by_kind` | Per-kind temperatures for questions over images, fitted by `evaluation/vision/temps.py`; text questions keep the text ones |
| `evaluation/vision/` | `--long-side/--max-pixels`, held-out NaturalBench groups, `temps.py`, `compare.py` (seeds against a baseline, paired bootstrap) and `text_check.py` |

## Lineage

1. **v19 + `--vision`**: the published v19 reads images with no retraining (#10).
2. **v19-images**: v19 continued on the image recipe
   ([configs/vision/v19-images.yaml](../../configs/vision/v19-images.yaml), three seeds):
   NaturalBench 0.805, POPE 0.874, preview 42.7 (means of three seeds), text check
   0.82-0.83.
3. **hobson-v20-vl, first attempt**: the [hobson-v20](strands-decider-2B-hobson-v20.md)
   soup on the same recipe: images held, text fell back to v19's level (29.3).
4. **hobson-v20-vl** (this package, seed 0): the same with no frozen-KL anchor on yes/no
   rows and 27B-teacher targets on the replayed text rows: text 39.4, images as above.

## Limits

- Exploratory: local runs, three seeds; the arms were chosen after earlier results.
- The image numbers are public sets; the 60 preview items are rebuilt from their sources,
  not the sealed Image JevBench set, so the board position is an estimate.
- Not yet run on the unseen task families.
- The VQAv2 and COCO downloads have no revision; [data/image/README.md](../../data/image/README.md)
  records the hashes of the rows the recorded runs trained on.
