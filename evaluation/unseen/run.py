"""Answer the unseen-family set (build.py) with one decider, and keep every answer.

The decider is a Strands Decider checkpoint, run in process exactly as `serve` runs it
(its own temperatures; `--vision` for a checkpoint served with `serve --vision`, which
answers text on the same weights that server uses), or any other decider behind an
adapter:

  --adapter http://HOST:PORT     a server speaking the System One API (POST /v1/systemone,
                                 docs/inference.md): `strands-decider serve`, or another
                                 decider behind the same request and answer shape
  --adapter python:MODULE:NAME   MODULE.NAME() returns a function (state, question) ->
                                 answer, both as System One JSON: a decider run in process
                                 through its own published code (README.md has examples);
                                 `--path DIR` puts DIR on sys.path first

Each row is one request with one question. Every answer is stored as its probabilities
in the row's option order (yes/no as [P(no), P(yes)], score levels ascending), with the
row's gold; score.py grades them.

    python evaluation/unseen/run.py --rows data/unseen.jsonl --checkpoint CKPT --out reports/unseen/NAME
    python evaluation/unseen/run.py --rows data/unseen.jsonl --adapter http://127.0.0.1:8000 \\
        --out reports/unseen/NAME

Writes OUT/predictions.jsonl and OUT/run_meta.json (the decider, its temperatures and
base revision when it is a checkpoint, the rows' sha256, library versions).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from collections.abc import Callable
from typing import Any

from strands_decider.data.format import Example
from strands_decider.runinfo import library_versions

Ask = Callable[[Any, dict[str, Any]], dict[str, Any]]


def probabilities(ex: Example, answer: dict[str, Any]) -> list[float]:
    """A System One answer as probabilities in `ex`'s option order."""
    if ex.kind == "noul":
        p = float(answer["noul"])
        return [1.0 - p, p]
    probs = answer["probabilities"]
    # System One keys a choice answer by option name and a score answer by level index
    # ("0", "1", ...; prompting.read_score), whatever the levels are called.
    keys = [name for name, _ in ex.options] if ex.kind == "choice" else [str(i) for i in range(len(ex.options))]
    missing = [k for k in keys if k not in probs]
    if missing:
        raise ValueError(f"the answer has no probability for {missing} (got {sorted(probs)})")
    return [float(probs[k]) for k in keys]


def checkpoint_decider(checkpoint: str, device: str, vision: bool) -> tuple[Ask, dict[str, Any]]:
    """`checkpoint` answering as `serve` would, and what to record about it."""
    from strands_decider.infer import EngineConfig, load_engine
    from strands_decider.schema import SystemOneRequest

    if vision:
        from strands_decider.vision import load_vision_engine

        engine = load_vision_engine(checkpoint, EngineConfig(device=device))
    else:
        engine = load_engine(checkpoint, device=device)
    cfg = engine.model.config

    def ask(state: Any, question: dict[str, Any]) -> dict[str, Any]:
        request = SystemOneRequest.model_validate({"state": state, "questions": {"q": question}})
        answer: dict[str, Any] = engine.evaluate(request).answers["q"].model_dump()
        return answer

    meta = {"checkpoint": checkpoint, "vision": vision, "base_model": cfg.base_model,
            "base_revision": cfg.base_revision, "temperature": cfg.temperature,
            "temperature_by_kind": cfg.temperature_by_kind}
    return ask, meta


def http_decider(url: str) -> Ask:
    import urllib.request

    def ask(state: Any, question: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps({"state": state, "questions": {"q": question}}).encode()
        req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            answer: dict[str, Any] = json.load(resp)["answers"]["q"]
        return answer

    return ask


def python_decider(spec: str, path: str | None) -> Ask:
    """`MODULE:NAME`: MODULE.NAME() returns the (state, question) -> answer function."""
    module, _, name = spec.partition(":")
    if not name:
        raise SystemExit(f"--adapter python:MODULE:NAME, got python:{spec}")
    if path:
        sys.path.insert(0, path)
    factory = getattr(importlib.import_module(module), name)
    ask: Ask = factory()
    return ask


def run(rows: list[dict[str, Any]], ask: Ask) -> list[dict[str, Any]]:
    out = []
    for i, r in enumerate(rows):
        ex = Example.from_dict(r)
        question = ex.to_question().model_dump(exclude_none=True)
        p = probabilities(ex, ask(ex.state, question))
        out.append({"id": i, "task": ex.task, "kind": ex.kind, "gold": ex.label, "probs": p})
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Answer the unseen-family set with one decider.")
    ap.add_argument("--rows", required=True, help="build.py's output")
    ap.add_argument("--out", required=True, help="a directory for predictions.jsonl and run_meta.json")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="a Strands Decider checkpoint (local directory or Hub id)")
    src.add_argument("--adapter", help="http://HOST:PORT or python:MODULE:NAME")
    ap.add_argument("--vision", action="store_true", help="load the checkpoint as `serve --vision` does")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--path", help="with --adapter python:...: a directory to import MODULE from")
    a = ap.parse_args(argv)
    with open(a.rows, "rb") as fh:
        raw = fh.read()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    meta: dict[str, Any] = {"rows": a.rows, "rows_sha256": hashlib.sha256(raw).hexdigest(), "n": len(rows)}
    if a.checkpoint:
        ask, about = checkpoint_decider(a.checkpoint, a.device, a.vision)
        meta.update(about)
    elif a.adapter.startswith("python:"):
        ask = python_decider(a.adapter[len("python:"):], a.path)
        meta["adapter"] = a.adapter
    else:
        ask = http_decider(a.adapter)
        meta["adapter"] = a.adapter
    t0 = time.time()
    preds = run(rows, ask)
    meta.update(seconds=round(time.time() - t0, 1), versions=library_versions())
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "predictions.jsonl"), "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(p) + "\n" for p in preds)
    with open(os.path.join(a.out, "run_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"{a.out}: {len(preds)} answers in {meta['seconds']} s")


if __name__ == "__main__":
    main()
