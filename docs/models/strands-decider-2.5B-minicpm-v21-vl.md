# strands-decider-2.5B-minicpm-v21-vl

> **Unofficial candidate.** The [minicpm-v21](strands-decider-2.5B-minicpm-v21.md) recipe
> given eyes: a frozen SigLIP2 encoder and a trained projector grafted onto the MiniCPM5-2B
> decider. Estimated **#1 in the 2B text class** on the JevBench board, with the highest
> local text score and the fewest hedged yes/no answers of any candidate (not submitted; an
> estimate from local measures). Text-first: it reads images, but weakly. Not a published
> Strands release.

| | |
| --- | --- |
| What it is | The v21 seeds' weight soup with a grafted vision encoder, LLaVA-style: SigLIP2 so400m reads each image, a projector trained first alone on COCO captions maps it into 144 input embeddings of the decoder, then the adapter, head and projector train on the image recipe of [hobson-v20-balanced](strands-decider-2B-hobson-v20-balanced.md) |
| Base | `openbmb/MiniCPM5-2B` at `f97400052a43d642bbc6e9975e2397e3ae6a6b52`, and `google/siglip2-so400m-patch16-384` (Apache-2.0) at `dd658faac399427308559e2c3ac1e99cbe43845d`, both pinned |
| Size | 2.5B (MiniCPM5-2B, as the board describes it) plus the frozen 0.4B SigLIP2 encoder; the archive holds the adapter, the head and the projector (370 MB) |
| Answers | text, and text with images (`serve --vision`) |
| Configs | [configs/align/strands-decider-2.5B-minicpm-v21-vl.yaml](../../configs/align/strands-decider-2.5B-minicpm-v21-vl.yaml) (stage 1), [configs/vision/strands-decider-2.5B-minicpm-v21-vl.yaml](../../configs/vision/strands-decider-2.5B-minicpm-v21-vl.yaml), `-seed1`, `-seed2` (stage 2): **the package is seed 1**, calibrated |
| Download | https://vision-decider-643603452951-us-east-1.s3.us-east-1.amazonaws.com/weights/strands-decider-2.5B-minicpm-v21-vl.tar |
| sha256 | `15a62b95c96b0d65144a95138bfa3b715d6922f1a385fe9502c9734f5cb31b1e` ([weights_SHA256SUMS.txt](https://vision-decider-643603452951-us-east-1.s3.us-east-1.amazonaws.com/weights/weights_SHA256SUMS.txt)) |

## Results

Local measures, never board scores ([docs/evaluating.md](../evaluating.md)). Text questions
take the text path of the grafted checkpoint; the image stage also moved the text weights,
which is why its text numbers differ from minicpm-v21's. The candidates are measured as
packaged (with the choice/score refit, "Fix A"; as trained, this seed scores 50.0 on the
public tasks), the published v19 as published (with the same refit: 34.2 public, 23.6
unseen).

| Measure | minicpm-v21-vl (seed 1) | minicpm-v21 | hobson-v20 | v19 (published) |
| --- | --- | --- | --- | --- |
| JevBench v1 public, v1.5-style Intelligence | **53.6** (3 seeds: 49.8 / 53.6 / 50.4) | 47.1 | 42.5 | 29.5 |
| Tasks right, of 231 | 175 | **178** | 177 | 167 |
| Yes/no answers inside 0.2-0.8, of 74 | **15** | 21 | 35 | 48 |
| Unseen families, Intelligence | 32.4 | 28.7 | **34.7** | 20.3 |
| Unseen, yes/no / choice / score competence | -10.4 / 58.2 / 49.2 | -14.8 / 57.0 / 44.0 | -10.8 / 64.7 / 50.1 | -40.8 / 60.4 / 41.4 |
| Unseen ECE | 0.079 | 0.101 | 0.055 | 0.073 |
| Old board measure (v1.4.2, public accuracy) | 0.758 | 0.771 | 0.766 | 0.723 |

| Images (400,000-pixel budget) | minicpm-v21-vl (seed 1) | v19 + `--vision` | hobson-v20-balanced |
| --- | --- | --- | --- |
| NaturalBench accuracy / G-Acc | 0.675 / 0.130 | 0.787 / 0.347 | **0.802 / 0.360** |
| POPE-adversarial accuracy | 0.817 | 0.858 | **0.865** |
| Image JevBench preview, right of 60 | 16 | 40 | **43** |
| Mean confidence with the image removed, NaturalBench / POPE | 0.64 / 0.62 | 0.65 / 0.73 | **0.61 / 0.59** |

- **Estimated board position (unofficial):** Capability 59.4 / 63.3 / 67.7 (pessimistic /
  middle / optimistic) against 58.8 for the current 2B #1, decision-2b: about **#1** in the
  2B text class. Method: [hobson-v20](strands-decider-2B-hobson-v20.md#results).
- It drops the most from public to unseen tasks (-21.2) with minicpm-v21; on the unseen
  families hobson-v20 leads.
- For images, choose [hobson-v20-balanced](strands-decider-2B-hobson-v20-balanced.md): this
  model's image accuracy is below v19 served with `--vision` on every set.

Every number per seed is in the results log:
https://github.com/Vivek0712/strands-decider/blob/ed29b08bb71aad26133bb6c45e8027648f5725ef/RESULTS.md.

## Download and serve

```bash
NAME=strands-decider-2.5B-minicpm-v21-vl
URL=https://vision-decider-643603452951-us-east-1.s3.us-east-1.amazonaws.com/weights
curl -fO "$URL/$NAME.tar" && curl -fO "$URL/weights_SHA256SUMS.txt"
grep " $NAME.tar\$" weights_SHA256SUMS.txt | sha256sum -c -
mkdir -p checkpoints && tar -xf "$NAME.tar" -C checkpoints
pip install "strands-decider[vision]"                  # Pillow; transformers >= 5.18
strands-decider serve checkpoints/$NAME --vision --image-long-side 0 --image-max-pixels 400000 --port 8000
```

`serve --vision` recognises a grafted checkpoint by its `projector_config.json` and
downloads SigLIP2 and MiniCPM5-2B at their pinned revisions. Every image is resized to
384 x 384 for the encoder (after the server's own pixel budget), so it costs 144 tokens
whatever its size ([docs/vision.md](../vision.md#a-torso-without-a-vision-tower-grafted-eyes-minicpm5)).
Text requests take the text path.

## Verify it yourself

On one NVIDIA GPU, with the archive unpacked as above and v19 at its pinned revision
([docs/evaluating.md](../evaluating.md#setup)):

```bash
# Text: JevBench v1 public through the vision server, against v19
PY=$(which python) GPU=0 SERVE_ARGS=--vision evaluation/jevbench/jevbench.sh checkpoints/$NAME reports/jevbench/$NAME
PY=$(which python) GPU=0 evaluation/jevbench/jevbench.sh checkpoints/v19 reports/jevbench/v19
python evaluation/jevbench/v15_proxy.py reports/jevbench/$NAME --vs reports/jevbench/v19

# Unseen task families (1,050 rows), answered as `serve --vision` answers, against v19
training/recipe.sh build
python evaluation/unseen/build.py --out data/unseen.jsonl
python evaluation/unseen/run.py --rows data/unseen.jsonl --checkpoint checkpoints/$NAME --vision \
  --out reports/unseen/$NAME
python evaluation/unseen/run.py --rows data/unseen.jsonl --checkpoint checkpoints/v19 --out reports/unseen/v19
python evaluation/unseen/score.py reports/unseen/$NAME --vs reports/unseen/v19

# Images, against v19 + --vision
pip install pandas pyarrow
CKPT=checkpoints/$NAME OUT=reports/$NAME training/recipe_images.sh eval
CKPT=checkpoints/v19 OUT=reports/v19-400k training/recipe_images.sh eval
python evaluation/vision/compare.py reports/$NAME/eval --vs reports/v19-400k/eval
```

## Retrain

Linux with an NVIDIA GPU; the recorded runs used one H200. First build
[minicpm-v21](strands-decider-2.5B-minicpm-v21.md#retrain) through its calibrated soup
(`checkpoints/strands-decider-2.5B-minicpm-v21`, which stage 2 starts from; not the
published seed-0 package), and the image rows of
[hobson-v20-balanced](strands-decider-2B-hobson-v20-balanced.md#retrain)
(`training/recipe_images.sh fetch build dedupe`). Then:

```bash
training/recipe_minicpm_vision.sh fetch build dedupe align train
for s in 0 1 2; do       # the choice/score refit only; seed 1 is the package
  strands-decider calibrate checkpoints/strands-decider-2.5B-minicpm-v21-vl-s$s --kinds choice,score \
    --objective nll --limit 900 --data data/multistep_v14_eval.jsonl --data data/holdout_v5_norule.jsonl
done
training/recipe_minicpm_vision.sh eval     # after calibrating: eval reads the temperatures
```

- `align` (stage 1) trains the projector alone to caption 80,000 COCO train2014 images
  through the frozen MiniCPM5-2B: 621 steps, about 12 minutes on an H200, validation loss
  1.87. `build` refuses val2014, whose images POPE uses, and `dedupe` checks the caption
  images against every evaluation image.
- `train` (stage 2) runs `vision_train` with `projector_from` for each seed: the adapter,
  head and projector train on the same rows, copies and losses as v19-images; SigLIP2 stays
  frozen.
- Stage 2 keeps the v21 soup's temperatures; the refit replaces only the choice and score
  ones (the package's yes/no temperature, 0.5608, is the soup's), on the two files the
  recorded refits used. `strands-decider calibrate` reads a grafted checkpoint as the text
  model `serve --vision` answers text with. Image temperatures are written as for
  hobson-v20-balanced; the published package has none.

What this adds, all opt-in:

| Piece | What it does |
| --- | --- |
| `strands_decider.graft` | `GraftedDeciderModel`: a text checkpoint plus a frozen SigLIP2 encoder and a projector (2 x 2 pixel-unshuffle, two-layer GELU MLP) whose 144 outputs replace the input embeddings of 144 slot tokens inside `<state>`; positions stay 1-D, so the shared-prefix cache is the text one |
| `python -m strands_decider.graft_align` | Stage 1: the projector alone, next-token loss of the frozen base LM on captions |
| `vision_train` with `projector_from` | Stage 2: the projector trains beside the adapter and head, at its own learning rate |
| `serve --vision`, `evaluation/vision/*`, `evaluation/unseen/run.py --vision` | Recognise a grafted checkpoint by its `projector_config.json` |
| `training/recipe_minicpm_vision.sh` | Both stages and the evaluation, end to end |

## Lineage

1. [minicpm-v21](strands-decider-2.5B-minicpm-v21.md): three seeds from the raw MiniCPM5-2B,
   and their weight soup (44.5 on the public tasks after the refit).
2. **Stage 1**: the SigLIP2 projector aligned on 80,000 COCO train2014 captions.
3. **Stage 2** (this package, seed 1): the soup with the projector on the image recipe,
   then the choice/score refit: 53.6 (50.0 as trained).

## Limits

- Exploratory: local runs, three seeds; the arm was chosen after earlier results.
- Weak on images (accuracy below v19 + `--vision`); its gains are text gains.
- The largest public-to-unseen drop of the candidates; read its public score with the
  unseen numbers beside it.
- The v1.5 proxy is our reimplementation of the published rules; the board position is an
  estimate. 2.5B + 0.4B parameters: over the 2B line, as noted for MiniCPM5-2B.
- COCO downloads have no revision; [data/image/README.md](../../data/image/README.md)
  records the hashes of the rows the recorded runs trained on.
