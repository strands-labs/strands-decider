"""Text non-regression for an image-trained checkpoint: held-out text rows, asked the text way.

Rows: the committed held-out evaluation splits of v19's own generated training data
(data/synthetic/generated_v16_eval, generated_v18_eval and adequacy_gen_eval: 899 rows,
none of which image training trains on). Each row is rendered as the server renders it
(prompting.build_prompt, canonical option order), forwarded whole through the multimodal
torso with no image (the weights the text path runs through) and read with the
checkpoint's text temperatures.

    python evaluation/vision/text_check.py --checkpoint CKPT --out text_check.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from strands_decider.data.format import Example
from strands_decider.infer import _option_token_index
from strands_decider.modeling import masked_log_softmax
from strands_decider.prompting import build_prompt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import summarise
from run import V19, V19_REV, _versions

FILES = ["generated_v16_eval.jsonl", "generated_v18_eval.jsonl", "adequacy_gen_eval.jsonl"]


@torch.no_grad()
def main() -> None:
    from strands_decider.vision import VisionDeciderModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=V19)
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "../../data/synthetic"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="rows per file (0: all)")
    a = ap.parse_args()
    ck = a.checkpoint
    if ck == V19:
        from huggingface_hub import snapshot_download

        ck = snapshot_download(V19, revision=V19_REV)
    model = VisionDeciderModel.load(ck).to(a.device).eval()
    dev = a.device
    temps = model.config.temperature_by_kind
    res = []
    for f in FILES:
        with open(os.path.join(a.data_dir, f), encoding="utf-8") as fh:
            lines = fh.readlines()[: a.limit or None]
        for i, line in enumerate(lines):
            ex = Example.from_dict(json.loads(line))
            prompt, rq = build_prompt(ex.state, ex.to_question())
            enc = model.tokenizer(prompt, return_offsets_mapping=True)
            if len(enc["input_ids"]) > model.config.max_length:
                continue
            opt = _option_token_index(enc["offset_mapping"], rq.option_spans, len(prompt) - len(rq.text))
            ids = torch.tensor([enc["input_ids"]], device=dev)
            model.reset_positions()
            out = model(ids, torch.ones_like(ids), torch.tensor([rq.n_slots], device=dev),
                        opt_idx=torch.tensor([opt], device=dev),
                        temperature=temps.get(ex.kind, model.config.temperature))
            p = masked_log_softmax(out["logits"].float(), torch.tensor([rq.n_slots], device=dev)).exp()[0]
            # slot k shows canonical option k (no shuffling): probs are in option order
            res.append({"id": f"{f}:{i}", "file": f, "kind": ex.kind, "gold": ex.label,
                        "probs": [float(x) for x in p[: rq.n_slots]]})
    summary = {"checkpoint": a.checkpoint, "versions": _versions(), "all": summarise(res)["all"],
               "by_file": {f: summarise([r for r in res if r["file"] == f])["all"] for f in FILES}}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "rows": res}, fh)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
