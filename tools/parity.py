"""Parity of a runtime against a checkpoint's references, under tests/core_conformance.py's rule.

golden: each text line of golden_vectors.jsonl, its state_ids and question_ids fed to the
        backend's `forward`, read with the core's readout, against the golden probabilities;
        overall and per prompt-length band, with the worst questions. Also checks that the
        core's `render` reproduces the golden token ids.
typed:  the Typed Decisions test split, one request per case through the decision engine,
        against the checkpoint's eval/typed-decisions/predictions.jsonl (or --reference, such as
        a CPU float32 run's --dump for a checkpoint that publishes none), and scored as the
        dataset's bench/score_with_jobs.py scores (its `score_one`, then its accuracy, Brier
        and 10-bin ECE).

python tools/parity.py golden --device neuron --max-prompt 8192 --out results/golden.json
python tools/parity.py typed --device neuron --out results/typed.json
python tools/parity.py typed --device cpu --shard 0/4 --out results/cpu0.json --dump results/cpu0.jsonl
python tools/parity.py typed --device neuron --reference results/cpu.jsonl --out results/typed.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from core_conformance import MIN_QUESTIONS, agreement, read_golden_vectors

from strands_decider.core import Request, question_probabilities, render
from strands_decider.decision import load_decision_engine

BANDS = ((0, 1024), (1024, 4096), (4096, 8192), (8192, 16384), (16384, 32768))


def band_of(n: int) -> str:
    for lo, hi in BANDS:
        if lo < n <= hi:
            return f"{lo}-{hi}"
    return f">{BANDS[-1][1]}"


def summarise(ours: dict, ref: dict, lengths: dict | None = None, worst: int = 15) -> dict:
    a = agreement(ours, ref)
    dps = {
        k: max(abs(float(x) - float(y)) for x, y in zip(ours[k], ref[k], strict=True)) for k in ref
    }
    top1 = lambda r: max(range(len(r)), key=lambda i: (r[i], -i))  # noqa: E731
    out = {
        "questions": a.questions,
        "top1": a.top1,
        "mean_abs_dp": a.mean_abs_dp,
        "max_abs_dp": max(dps.values()),
        "conforms": a.conforms(),
        "conforms_rule": f"n>={MIN_QUESTIONS}, top1>=0.98, mean|dp|<=0.02",
        "disagree": [str(k) for k in ref if top1(ours[k]) != top1(ref[k])],
        "worst": [
            {
                "key": str(k),
                "abs_dp": dps[k],
                "len": (lengths or {}).get(k),
                "ours": [round(float(x), 4) for x in ours[k]],
                "ref": [round(float(x), 4) for x in ref[k]],
            }
            for k in sorted(dps, key=dps.get, reverse=True)[:worst]
        ],
    }
    if lengths:
        bands = {}
        for lo, hi in BANDS:
            keys = [k for k in ref if lo < lengths[k] <= hi]
            if keys:
                b = agreement({k: ours[k] for k in keys}, {k: ref[k] for k in keys})
                bands[f"{lo}-{hi}"] = {
                    "questions": b.questions,
                    "top1": b.top1,
                    "mean_abs_dp": b.mean_abs_dp,
                    "max_abs_dp": max(dps[k] for k in keys),
                }
        out["bands"] = bands
    return out


def golden(args: argparse.Namespace) -> dict:
    path = Path(args.checkpoint) / "golden_vectors.jsonl"
    reference = read_golden_vectors(path)
    lines = []
    with open(path, encoding="utf-8") as f:
        for number, raw in enumerate(f):
            if raw.strip():
                line = json.loads(raw)
                if line["images"]:
                    continue
                n = len(line["state_ids"]) + max(
                    len(q["question_ids"]) for q in line["questions"].values()
                )
                if args.lines and number not in args.lines:
                    continue
                if args.min_prompt < n <= args.max_prompt:
                    lines.append((number, n, line))
    if args.shortest:
        lines.sort(key=lambda x: x[1])
    if args.limit:
        lines = lines[: args.limit]
    engine = load_decision_engine(args.checkpoint, device=args.device, dtype=args.dtype)
    backend, config, weight = engine.backend, engine.config, engine.weight

    render_checked = render_bad = 0
    ours, ref, lengths, times = {}, {}, {}, []
    for i, (number, n, line) in enumerate(lines):
        rendered = render(Request.model_validate(line["request"]), engine.tokenizer)
        if i < args.render_sample or args.render_sample < 0:
            render_checked += 1
            same = list(rendered.state_ids) == line["state_ids"] and all(
                list(rendered.questions[name].question_ids) == q["question_ids"]
                for name, q in line["questions"].items()
            )
            render_bad += not same
        names = list(line["questions"])
        if list(rendered.questions) != names:
            raise ValueError(
                f"line {number}: rendered questions {list(rendered.questions)} != {names}"
            )
        t = time.perf_counter()
        rows = backend.forward(
            line["state_ids"], [line["questions"][q]["question_ids"] for q in names]
        )
        times.append(time.perf_counter() - t)
        probs = question_probabilities(
            rows,
            config,
            rendered,
            weight=None if getattr(backend, "applies_readout", False) else weight,
        )
        for name in names:
            key = (number, name)
            ours[key] = probs[name].tolist()
            ref[key] = reference[key]
            lengths[key] = n
        if (i + 1) % args.progress == 0:
            a = agreement(ours, ref)
            print(
                f"[golden] {i + 1}/{len(lines)} lines, {a.questions} q: top1 {a.top1:.4f} "
                f"mean|dp| {a.mean_abs_dp:.4f} ({times[-1]:.2f}s last)",
                flush=True,
            )
    out = summarise(ours, ref, lengths)
    out.update(
        mode="golden",
        device=args.device,
        dtype=args.dtype,
        lines=len(lines),
        render_checked=render_checked,
        render_mismatches=render_bad,
        backend=backend.describe(),
        seconds_per_line_p50=statistics.median(times) if times else None,
        max_prompt=args.max_prompt,
        min_prompt=args.min_prompt,
    )
    if args.dump:
        with open(args.dump, "w") as f:
            for k in ref:
                f.write(
                    json.dumps(
                        {
                            "line": k[0],
                            "question": k[1],
                            "len": lengths[k],
                            "ours": ours[k],
                            "ref": ref[k],
                        }
                    )
                    + "\n"
                )
    return out


def _score_module(path: Path):
    """The dataset's bench/score_with_jobs.py, for its `score_one`; its server-side imports
    (`datasets`, `requests`) are stubbed when absent, as nothing here calls them."""
    import types

    for mod in ("datasets", "requests"):
        if importlib.util.find_spec(mod) is None:
            stub = types.ModuleType(mod)
            stub.load_dataset = None  # type: ignore[attr-defined]
            sys.modules[mod] = stub
    spec = importlib.util.spec_from_file_location("score_with_jobs", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # type: ignore[union-attr]
    return m


def typed(args: argparse.Namespace) -> dict:
    import pandas as pd

    data = Path(args.typed)
    sw = _score_module(data / "bench" / "score_with_jobs.py")
    ds = pd.read_parquet(next((data / "all").glob("test-*.parquet")))
    if args.limit:
        ds = ds.head(args.limit)
    if args.shard:  # k/n: cases k, k + n, ... so that n processes can share a slow CPU run
        k, n_shards = map(int, args.shard.split("/"))
        ds = ds.iloc[k::n_shards]
    # The reference: the checkpoint's published predictions, or another run's --dump (the 12B
    # publishes none). Without one, the run is only scored.
    default = Path(args.checkpoint) / "eval" / "typed-decisions" / "predictions.jsonl"
    ref_path = Path(args.reference) if args.reference else default
    refpred = {}
    if ref_path.exists():
        with open(ref_path) as f:
            for raw in f:
                p = json.loads(raw)
                refpred[(p["id"], p["question"])] = p["probabilities"]
    elif args.reference:
        raise FileNotFoundError(ref_path)
    golds = {
        (r.id, q): g for r in ds.itertuples(index=False) for q, g in json.loads(r.gold).items()
    }
    ref_scores = [
        sw.score_one({"probabilities": refpred[k]}, g)[:4] for k, g in golds.items() if k in refpred
    ]
    engine = load_decision_engine(args.checkpoint, device=args.device, dtype=args.dtype)
    accs, briers, confs, kls, lat, toks = [], [], [], [], [], []
    ours, ref, preds = {}, {}, []
    for i, raw in enumerate(ds.itertuples(index=False)):
        payload = {"state": json.loads(raw.state), "questions": json.loads(raw.questions)}
        t = time.perf_counter()
        resp = engine.evaluate(Request.model_validate(payload))
        lat.append(time.perf_counter() - t)
        toks.append(resp.model_dump()["usage"]["input_tokens"])
        answers = resp.model_dump()["answers"]
        # The scorer reads `probabilities` (vLLM's response); this engine's noul answer is the
        # probability of true alone, so it gets both outcomes, as the published predictions hold.
        for a in answers.values():
            if a["type"] == "noul" and "probabilities" not in a:
                a["probabilities"] = {"true": a["noul"], "false": 1.0 - a["noul"]}
        gold = json.loads(raw.gold)
        for qid, gq in gold.items():
            ok, kl, brier, conf, probs = sw.score_one(answers.get(qid, {"probabilities": {}}), gq)
            preds.append(
                {
                    "id": raw.id,
                    "workflow": raw.workflow,
                    "question": qid,
                    "type": gq["type"],
                    "probabilities": probs,
                }
            )
            accs.append(ok)
            kls.append(kl)
            briers.append(brier)
            confs.append(conf)
            if refpred:
                r = refpred[(raw.id, qid)]
                opts = list(r)
                ours[(raw.id, qid)] = [probs[o] for o in opts]
                ref[(raw.id, qid)] = [r[o] for o in opts]
        if (i + 1) % args.progress == 0:
            a = agreement(ours, ref) if ref else None
            print(
                f"[typed] {i + 1}/{len(ds)} cases: acc {sum(accs) / len(accs):.4f} "
                + (f"top1 {a.top1:.4f} mean|dp| {a.mean_abs_dp:.4f} " if a else "")
                + f"p50 {statistics.median(lat) * 1e3:.0f} ms",
                flush=True,
            )
            if args.dump:
                _dump_typed(args.dump, preds, lat)
    n = len(accs)
    out = summarise(ours, ref) if ref else {"questions": n}
    reference = None
    if ref_scores:
        reference = {
            "path": str(ref_path),
            "decisions": len(ref_scores),
            "accuracy": sum(x[0] for x in ref_scores) / len(ref_scores),
            "kl_from_gold": sum(x[1] for x in ref_scores) / len(ref_scores),
            "brier": sum(x[2] for x in ref_scores) / len(ref_scores),
            "ece": _ece([x[0] for x in ref_scores], [x[3] for x in ref_scores]),
        }
    lat_ms = sorted(x * 1e3 for x in lat)
    out.update(
        mode="typed",
        device=args.device,
        dtype=args.dtype,
        cases=len(ds),
        decisions=n,
        accuracy=sum(accs) / n,
        brier=sum(briers) / n,
        ece=_ece(accs, confs),
        kl_from_gold=sum(kls) / n,
        reference=reference,
        accuracy_within_0_02=abs(sum(accs) / n - reference["accuracy"]) <= 0.02
        if reference
        else None,
        latency_ms={
            "p50": lat_ms[len(lat_ms) // 2],
            "p95": lat_ms[min(len(lat_ms) - 1, math.ceil(0.95 * len(lat_ms)) - 1)],
            "mean": statistics.mean(lat_ms),
            "first": lat[0] * 1e3,
        },
        input_tokens={"p50": int(np.median(toks)), "max": int(max(toks)), "min": int(min(toks))},
        backend=engine.backend.describe(),
    )
    if args.dump:
        _dump_typed(args.dump, preds, lat)
    return out


def _ece(accs: list, confs: list, n_bins: int = 10) -> float:
    """bench/score_with_jobs.py's 10-bin expected calibration error."""
    hits, total, csum = [0] * n_bins, [0] * n_bins, [0.0] * n_bins
    for ok, conf in zip(accs, confs, strict=True):
        b = min(int(conf * n_bins), n_bins - 1)
        total[b] += 1
        hits[b] += int(ok)
        csum[b] += conf
    n = len(accs)
    return sum(
        total[b] / n * abs(hits[b] / total[b] - csum[b] / total[b])
        for b in range(n_bins)
        if total[b]
    )


def _dump_typed(path: str, preds: list, lat: list) -> None:
    with open(path, "w") as f:
        for p, l_ms in zip(preds, [x for x in lat for _ in range(5)], strict=False):
            f.write(json.dumps({**p, "request_ms": round(l_ms * 1e3, 2)}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("golden", "typed"))
    ap.add_argument("--checkpoint", default=os.environ.get("WEIGHTS"))
    ap.add_argument("--typed", default=os.environ.get("TYPED"))
    ap.add_argument("--device", default="neuron")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--max-prompt", type=int, default=8192)
    ap.add_argument("--min-prompt", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--lines",
        type=lambda v: {int(x) for x in v.split(",")},
        default=None,
        help="golden line numbers only, such as 12,40",
    )
    ap.add_argument("--shortest", action="store_true", help="the shortest prompts first")
    ap.add_argument(
        "--render-sample", type=int, default=-1, help="lines whose render is checked (-1: all)"
    )
    ap.add_argument("--progress", type=int, default=25)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--reference",
        help="typed: reference predictions (another run's --dump); by default the checkpoint's "
        "eval/typed-decisions/predictions.jsonl",
    )
    ap.add_argument("--shard", help="typed: k/n, every n-th case from the k-th")
    ap.add_argument("--dump", help="per-question probabilities, jsonl")
    args = ap.parse_args()
    out = golden(args) if args.mode == "golden" else typed(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1, default=str)
    brief = {
        k: out[k]
        for k in ("questions", "top1", "mean_abs_dp", "max_abs_dp", "conforms")
        if k in out
    }
    for k in ("accuracy", "brier", "ece", "reference", "latency_ms", "bands", "render_mismatches"):
        if k in out:
            brief[k] = out[k]
    print(json.dumps(brief, indent=1, default=str))


if __name__ == "__main__":
    main()
