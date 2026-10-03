# Image evaluation

The measurements behind [docs/vision.md](../../docs/vision.md#how-well-it-does).

| File | What it does |
| --- | --- |
| [`run.py`](run.py) | Scores Strands Decider (v19) with `--vision`, the untrained Qwen3.5-2B-Base readout and Mapika/decider-2b-vision, each with the image and without it |
| [`metrics.py`](metrics.py) | Accuracy, Brier, ECE, NaturalBench's paired accuracies (Q-Acc, I-Acc, G-Acc) and image dependence |

Every model and dataset is downloaded at a pinned revision, and each run's `summary.json`
records the revisions and library versions it used.

```bash
pip install -e ".[vision]" torchvision pandas pyarrow jinja2   # torchvision: the mapika system
python evaluation/vision/run.py --out results --systems strands,qwen,mapika \
    --nb-groups 300 --pope 600 --ijb-jsonl ijb_preview.jsonl
```

`--ijb-jsonl` is optional: the Image JevBench preview items, rebuilt from their source
datasets by [`ijb_preview.py`](https://github.com/Vivek0712/vision-decider/blob/phase1-baselines/phase1/ijb_preview.py)
(the official set is sealed; 60 of the 128 published preview items can be rebuilt exactly).
`mapika` runs that model's own published code.

**Recorded results:** [vision-eval-2026-10-03](https://github.com/Vivek0712/strands-decider/releases/tag/vision-eval-2026-10-03), made with this script at commit `333b7f4` on one
NVIDIA H100 per machine (`--device cuda`; v19 on one, `qwen,mapika` on the other), with the command above. It holds each system's
`summary.json` and per-item probabilities (`*.jsonl`).
