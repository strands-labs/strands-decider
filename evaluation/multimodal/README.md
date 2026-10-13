# Image and audio evaluation (Gemma 4 E2B / E4B)

`run.py` scores a Gemma 4 Strands Decider over images (NaturalBench, POPE adversarial) and audio
(ESC-50, Speech Commands). It reads each item served, blind (the medium removed) and untrained
(the base read without the adapter). `paired.py` gives paired bootstrap intervals between two
readings. Results and method: [docs/audio.md](../../docs/audio.md#how-well-it-does).

```bash
pip install "strands-decider[audio]" pandas pyarrow
python evaluation/multimodal/run.py --out results            # E4B, all readings (about 75 min on an L4)
python evaluation/multimodal/run.py --out results-e2b --readings served,blind \
    --checkpoint StrandsAgents/strands-decider-E2B-gemma4-v1-2610 --revision d383ecf8534b13d49b3b567a1b66f88fca6d50ba
python evaluation/multimodal/paired.py results --b untrained
```

The text check behind docs/audio.md runs `evaluation/jevbench/jevbench.sh` twice, once plain and
once with `SERVE_EXTRA=--audio`, and compares the two `results.jsonl` item by item.

The recorded runs (per-item probabilities for every reading, both JevBench runs and each
`summary.json`) are published as the release
[gemma4-multimodal-eval-2026-10-12](https://github.com/Vivek0712/strands-decider/releases/tag/gemma4-multimodal-eval-2026-10-12).
