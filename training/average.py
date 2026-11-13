"""The uniform weight average of merged checkpoints of one base and one prompt template (a size's model averages its
seeds' runs: training/README.md).

    python -m training.average --out <dir> [--note "..."] <merged checkpoint> <merged checkpoint> [...]

Every tensor of every *.safetensors file of the first checkpoint (the shards and readout.safetensors) is averaged over
the checkpoints in float32 and stored in its own dtype, under the same file name and metadata. The tensors no run
trained (the vision and audio encoders) are equal in every checkpoint and come out unchanged. Every other file is the
first checkpoint's; training.json is the first's plus an `average` block naming the checkpoints averaged. CPU only; it
holds one shard of each checkpoint at a time.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from .merge import RECORD_FILE


def average(checkpoints: list[Path], out: Path, note: str = "") -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file

    for c in checkpoints:
        if not (c / RECORD_FILE).is_file():
            raise SystemExit(f"{c} is not a merged checkpoint (no {RECORD_FILE})")
    records = [json.loads((c / RECORD_FILE).read_text()) for c in checkpoints]
    for field in ("prompt", "base_model", "base_revision"):
        values = {r.get(field) for r in records}
        if len(values) > 1:
            raise SystemExit(f"the checkpoints differ in {field}: {sorted(map(str, values))}")
    inputs = {tuple((r.get("finetune") or {}).get("inputs") or ("text",)) for r in records}
    if len(inputs) > 1:
        raise SystemExit(f"the checkpoints were trained on different inputs: {sorted(inputs)}")
    weights = sorted(f.name for f in checkpoints[0].glob("*.safetensors"))
    for c in checkpoints[1:]:
        theirs = sorted(f.name for f in c.glob("*.safetensors"))
        if theirs != weights:
            raise SystemExit(f"{c}: weight files {theirs} differ from {checkpoints[0]}'s {weights}")
    out.mkdir(parents=True, exist_ok=False)
    try:
        for f in checkpoints[0].iterdir():
            if f.name not in weights and f.name != RECORD_FILE:
                (shutil.copytree if f.is_dir() else shutil.copy2)(f, out / f.name)
        for w in weights:
            t0 = time.time()
            handles = [safe_open(str(c / w), "pt") for c in checkpoints]
            meta = handles[0].metadata()
            keys = sorted(handles[0].keys())
            for c, h in zip(checkpoints[1:], handles[1:], strict=True):
                if sorted(h.keys()) != keys:
                    raise SystemExit(f"{c}/{w}: its tensors differ from {checkpoints[0]}'s")
            result = {}
            for k in keys:
                x = handles[0].get_tensor(k)
                acc = x.float()
                for c, h in zip(checkpoints[1:], handles[1:], strict=True):
                    y = h.get_tensor(k)
                    if y.shape != x.shape or y.dtype != x.dtype:
                        raise SystemExit(
                            f"{c}/{w}: {k} is {tuple(y.shape)} {y.dtype}, not {tuple(x.shape)} {x.dtype}"
                        )
                    acc += y.float()
                result[k] = (acc / len(handles)).to(x.dtype)
            save_file(result, str(out / w), metadata=meta)
            print(
                json.dumps({"file": w, "tensors": len(result), "s": round(time.time() - t0, 1)}),
                flush=True,
            )
            del result, handles
        record = dict(records[0])
        record["average"] = {
            "kind": "uniform weight average of merged checkpoints",
            "checkpoints": [str(c) for c in checkpoints],
            "seeds": [(r.get("finetune") or {}).get("seed") for r in records],
            "note": note,
        }
        (out / RECORD_FILE).write_text(json.dumps(record, indent=2) + "\n")
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    return {"out": str(out), "n": len(checkpoints)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, help="the averaged checkpoint (must not exist)")
    ap.add_argument("--note", default="", help="recorded in training.json's average block")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("checkpoints", nargs="+")
    a = ap.parse_args()
    if len(a.checkpoints) < 2:
        raise SystemExit(
            "average two or more checkpoints; one run's release is its merged checkpoint"
        )
    torch.set_num_threads(a.threads)
    print(json.dumps(average([Path(c) for c in a.checkpoints], Path(a.out), a.note)))


if __name__ == "__main__":
    main()
