"""Per-kind image temperatures, fitted on held-out image items from stored results.

run.py stores each item's probabilities after the temperature it applied (T0, recorded
in its summary.json). log p = z / T0 - logsumexp(z / T0), so T0 * log p is the logits up
to a per-row constant, and any other temperature applies exactly, with no forward pass.
`fit` fits one temperature per question kind by NLL (strands_decider.evaluate's
fit_temperature) on a run over held-out items; `apply` rescores an evaluation run's
with-image rows under those temperatures. The image-removed rows keep theirs: the server
answers a request without images on the text path, with the text temperatures.

    # NaturalBench groups 300-599, which no evaluation uses
    python evaluation/vision/run.py --out cal --systems strands --checkpoint CKPT \\
        --nb-start 300 --nb-groups 300 --pope 0 --no-blind --long-side 0 --max-pixels 400000
    python evaluation/vision/temps.py fit --run cal --out image_temps.json [--checkpoint CKPT]
    python evaluation/vision/temps.py apply --run results --temps image_temps.json --out results-T

`fit --checkpoint` also writes the temperatures into a local checkpoint as
`image_temperature_by_kind`, which `serve --vision` then applies to questions over images.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import score_runs

from strands_decider.evaluate import fit_temperature
from strands_decider.modeling import StrandsDeciderConfig, config_path

SYSTEM = "strands-v19"  # run.py's tag for the strands system, whatever the checkpoint


def _read(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _summary(run: str) -> dict[str, Any]:
    with open(os.path.join(run, "summary.json"), encoding="utf-8") as fh:
        summary: dict[str, Any] = json.load(fh)
    return summary


def applied(run: str) -> dict[str, float]:
    """The temperature per kind that run.py applied to the run's with-image rows."""
    t = _summary(run).get("temperatures") or {}
    if not t.get("image"):
        raise SystemExit(f"{run}/summary.json records no temperatures; pass --t0")
    return {k: float(v) for k, v in t["image"].items()}


def logits(probs: list[float], t0: float) -> list[float]:
    return [t0 * math.log(max(p, 1e-30)) for p in probs]


def fit(rows: list[dict[str, Any]], t0: dict[str, float]) -> dict[str, float]:
    """The NLL-optimal temperature per kind for rows scored at temperatures `t0`."""
    out = {}
    for kind in sorted({r["kind"] for r in rows}):
        sel = [r for r in rows if r["kind"] == kind]
        width = max(len(r["probs"]) for r in sel)
        z = torch.tensor([logits(r["probs"], t0[kind]) + [0.0] * (width - len(r["probs"])) for r in sel])
        gold = torch.tensor([r["gold"] for r in sel])
        out[kind] = round(fit_temperature(z, gold, torch.tensor([len(r["probs"]) for r in sel])), 4)
    return out


def rescale(rows: list[dict[str, Any]], t0: dict[str, float], t1: dict[str, float]) -> list[dict[str, Any]]:
    """Rows scored at `t0` as they would be at `t1` (kinds missing from `t1` keep `t0`)."""
    out = []
    for r in rows:
        k = r["kind"]
        p = torch.softmax(torch.tensor(logits(r["probs"], t0[k]), dtype=torch.float64) / t1.get(k, t0[k]), -1)
        out.append({**r, "probs": p.tolist()})
    return out


def write_to_checkpoint(checkpoint: str, temps: dict[str, float]) -> None:
    """Set `image_temperature_by_kind` in a local checkpoint's config, as `calibrate` does."""
    if not os.path.isdir(checkpoint):
        raise SystemExit(f"{checkpoint}: writing temperatures needs a local checkpoint directory")
    path = config_path(checkpoint)
    cfg = StrandsDeciderConfig.from_json(path)
    cfg.image_temperature_by_kind = temps
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(cfg.to_json())


def main() -> None:
    ap = argparse.ArgumentParser(description="Fit and apply per-kind image temperatures to run.py results.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit", help="fit on a run over held-out items")
    f.add_argument("--run", required=True, help="run.py output directory (held-out items, never an evaluation)")
    f.add_argument("--out", required=True)
    f.add_argument("--checkpoint", help="also write the temperatures into this local checkpoint")
    p = sub.add_parser("apply", help="rescore an evaluation run's with-image rows")
    p.add_argument("--run", required=True, help="run.py output directory")
    p.add_argument("--temps", required=True, help="json file from `fit`")
    p.add_argument("--out", required=True)
    for s in (f, p):
        s.add_argument("--t0", help="JSON temperatures the rows were scored at (default: the run's summary.json)")
    a = ap.parse_args()
    t0 = json.loads(a.t0) if a.t0 else applied(a.run)
    if a.cmd == "fit":
        temps = fit(_read(os.path.join(a.run, f"{SYSTEM}.jsonl")), t0)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(temps, fh, indent=2)
        if a.checkpoint:
            write_to_checkpoint(a.checkpoint, temps)
        print(json.dumps(temps))
        return
    with open(a.temps, encoding="utf-8") as fh:
        t1 = json.load(fh)
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for name in sorted(os.listdir(a.run)):
        if not name.endswith(".jsonl"):
            continue
        tag = name[: -len(".jsonl")]
        res[tag] = _read(os.path.join(a.run, name))
        if tag.endswith("-blind"):
            shutil.copy(os.path.join(a.run, name), os.path.join(a.out, name))
            continue
        res[tag] = rescale(res[tag], t0, t1)
        with open(os.path.join(a.out, name), "w", encoding="utf-8") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in res[tag])
    source = _summary(a.run)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"args": {**vars(a), "rescaled_from": a.run}, "versions": source.get("versions"),
                   "temperatures": {**source.get("temperatures", {}), "image": {**t0, **t1}},
                   "scores": score_runs(res)}, fh, indent=2)
    print(f"[temps] wrote {a.out}")


if __name__ == "__main__":
    main()
