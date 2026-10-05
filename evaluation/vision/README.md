# Image evaluation

The measurements behind [docs/vision.md](../../docs/vision.md#how-well-it-does).

| File | What it does |
| --- | --- |
| [`run.py`](run.py) | Scores Strands Decider (v19) with `--vision`, the untrained Qwen3.5-2B-Base readout and Mapika/decider-2b-vision, each with the image and without it |
| [`metrics.py`](metrics.py) | Accuracy, Brier, ECE, NaturalBench's paired accuracies (Q-Acc, I-Acc, G-Acc) and image dependence |
| [`temps.py`](temps.py) | Fits per-kind image temperatures on a run over held-out items and rescores a run under them, from the stored probabilities |
| [`compare.py`](compare.py) | An arm's seeds against a baseline, item by item: the mean differences, with paired bootstrap 95% intervals |
| [`text_check.py`](text_check.py) | Text non-regression after image training: v19's 899 held-out generated and adequacy rows, asked the text way |

Every model and dataset is downloaded at a pinned revision, and each run's `summary.json`
records the revisions and library versions it used.

```bash
pip install -e ".[vision]" torchvision pandas pyarrow jinja2   # torchvision: the mapika system
python evaluation/vision/run.py --out results --systems strands,qwen,mapika \
    --nb-groups 300 --pope 600 --ijb-jsonl ijb_preview.jsonl
```

`--long-side 0 --max-pixels 400000` resizes each image to a 400,000-pixel budget instead
of the 448 px long side (`VisionEngineConfig.image_max_pixels`); `--nb-start 300 --no-blind`
scores NaturalBench groups 300-599, which no evaluation uses, for fitting image
temperatures with `temps.py`; `--temps` applies given ones. The image-removed pass always
takes the text temperatures, as the server answers a request without images.

`--ijb-jsonl` is optional: the Image JevBench preview items, rebuilt from their source
datasets by [`ijb_preview.py`](https://github.com/Vivek0712/vision-decider/blob/phase1-baselines/phase1/ijb_preview.py)
(the official set is sealed; 60 of the 128 published preview items can be rebuilt exactly).
`mapika` runs that model's own published code.

**Recorded results:** [vision-eval-2026-10-03](https://github.com/Vivek0712/strands-decider/releases/tag/vision-eval-2026-10-03), made with this script at commit `333b7f4` on one
NVIDIA H100 per machine (`--device cuda`; v19 on one, `qwen,mapika` on the other), with the command above. It holds each system's
`summary.json`, and per-item probabilities (`*.jsonl`) for v19 and the base. Mapika's
per-item results are not included, so the paired CIs against it cannot be recomputed from the
release alone.
