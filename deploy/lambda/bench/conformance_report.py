"""The release check for a variants build: conformance_container.py's results, each CPU module
held to the static (--lambda) build on the same generation and GGUF. Run it whenever
LLAMA_CPP_REV or patches/ change (docs/deployment/graviton-cpu-variants.md#release-check).

  python conformance_report.py s3://bucket/conformance/      # or local .json files

A module passes when it conforms (tests/core_conformance.py: at least 1,000 questions, top-1
at least 98%, mean |dp| at most 0.02) and is no worse than static on its host: at most 2 more
wrong top answers per 1,032 questions, and a mean |dp| at most 10% higher. Results labelled
`-kleidiai` (opt-in) are reported but do not fail the check. Exits 1 if any module fails.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

CPU = {"0xd0c": "Graviton2", "0xd40": "Graviton3", "0xd4f": "Graviton4", "0xd84": "Graviton5"}
MIN_QUESTIONS, TOP1, MEAN_ABS_DP = 1000, 0.98, 0.02  # tests/core_conformance.py
EXTRA_WRONG_PER_1032, MEAN_RATIO = 2, 1.10


def load(sources: list[str]) -> list[dict]:
    rows = []
    for src in sources:
        if src.startswith("s3://"):
            import boto3

            s3 = boto3.client("s3")
            bucket, prefix = src[5:].split("/", 1)
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
                    rows.append(json.loads(body))
        else:
            with open(src) as f:
                rows.append(json.load(f))
    for r in rows:
        r.pop("per_question", None)
        r["gguf"] = r.get("gguf", "model.gguf")
    return rows


def main() -> None:
    rows = load(sys.argv[1:])
    static = {(r["cpu_part"], r["gguf"]): r for r in rows if r["module"] == "static"}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["cpu_part"], r["gguf"])].append(r)
    failed = False
    print("| CPU | GGUF | module | top-1 | mean abs dp | max abs dp | result |")
    print("|---|---|---|---|---|---|---|")
    for (part, gguf), rs in sorted(groups.items()):
        ref = static.get((part, gguf))
        for r in sorted(
            rs, key=lambda x: (x["module"] != "static", x.get("label", ""), x["module"])
        ):
            opt_in = r.get("label", "").endswith("-kleidiai")
            name = r["module"] + (" + KleidiAI" if opt_in else "")
            reasons = []
            if r["questions"] < MIN_QUESTIONS:
                reasons.append(f"{r['questions']} questions")
            if r["top1"] < TOP1:
                reasons.append(f"top-1 {r['top1']:.2%} < {TOP1:.0%}")
            if r["mean_abs_dp"] > MEAN_ABS_DP:
                reasons.append(f"mean |dp| {r['mean_abs_dp']} > {MEAN_ABS_DP}")
            if ref and r is not ref:
                extra = (ref["top1"] - r["top1"]) * r["questions"]
                if extra > EXTRA_WRONG_PER_1032 * r["questions"] / 1032 + 1e-9:
                    reasons.append(f"{extra:.0f} more wrong than static")
                if r["mean_abs_dp"] > ref["mean_abs_dp"] * MEAN_RATIO:
                    reasons.append(
                        f"mean |dp| {r['mean_abs_dp'] / ref['mean_abs_dp']:.2f}x static's"
                    )
            elif not ref:
                reasons.append("no static result on this host to compare")
            result = (
                "pass" if not reasons else ("opt-in: " if opt_in else "FAIL: ") + "; ".join(reasons)
            )
            failed |= bool(reasons) and not opt_in
            print(
                f"| {CPU.get(part, part)} | {gguf} | {name} | {r['top1']:.2%} | {r['mean_abs_dp']:.5f} "
                f"| {r['max_abs_dp']:.4f} | {result} |"
            )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
