"""Write aggregate benchmark data, comparisons, and plots."""

from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ARTIFACTS = ("summary.json", "summary.csv", "manifest.json")


def _fmt(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "unavailable"
    return f"{float(value):,.{digits}f}{suffix}"


def write_summary_csv(summary: dict[str, Any], path: Path) -> None:
    fields = [
        "mode",
        "load",
        "repetition",
        "requested_input_tokens",
        "input_tokens",
        "questions",
        "requests",
        "successes",
        "failures",
        "failure_rate",
        "requests_per_second",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "repeated_state_fraction",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        points = []
        if summary.get("latency_sweep"):
            points.append(summary["latency_sweep"])
        points.extend(summary.get("closed_loop", []))
        for point in points:
            for repetition_index, repetition in enumerate(point.get("repetitions", []), 1):
                for shape in repetition.get("shapes", {}).values():
                    writer.writerow(
                        {
                            "mode": point["mode"],
                            "load": point["load"],
                            "repetition": repetition_index,
                            "requested_input_tokens": shape.get(
                                "requested_input_tokens", shape["input_tokens"]
                            ),
                            "input_tokens": shape["input_tokens"],
                            "questions": shape["questions"],
                            "requests": shape["requests"],
                            "successes": shape["successes"],
                            "failures": shape["failures"],
                            "failure_rate": shape["failure_rate"],
                            "requests_per_second": repetition["requests_per_second"],
                            "p50_ms": shape["p50_ms"],
                            "p95_ms": shape["p95_ms"],
                            "p99_ms": shape["p99_ms"],
                            "repeated_state_fraction": repetition["repeated_state_fraction"],
                        }
                    )


def _reference_point(summary: dict[str, Any]) -> dict[str, Any]:
    if summary.get("latency_sweep"):
        return summary["latency_sweep"]
    target = int(summary["headline"]["reference_concurrency"])
    return min(summary["closed_loop"], key=lambda point: abs(int(point["load"]) - target))


def _shape_median(point: dict[str, Any], key: str) -> dict[int, float]:
    values: dict[int, list[float]] = {}
    for repetition in point.get("repetitions", []):
        for shape in repetition.get("shapes", {}).values():
            value = shape.get(key)
            if value is not None:
                values.setdefault(int(shape["input_tokens"]), []).append(float(value))
    return {shape: float(np.median(samples)) for shape, samples in values.items()}


def _model_label(model: str) -> str:
    return model.rsplit("/", 1)[-1].removeprefix("strands-decider-")


def write_plots(
    summaries: dict[str, dict[str, Any]],
    out: Path,
    *,
    prefix: str = "",
) -> None:
    throughput_path = out / f"{prefix}throughput-vs-concurrency.png"
    latency_path = out / f"{prefix}latency-vs-input-length.png"
    throughput_path.unlink(missing_ok=True)
    latency_path.unlink(missing_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    concurrency_ticks: set[int] = set()
    for model, summary in summaries.items():
        closed = sorted(summary.get("closed_loop", []), key=lambda point: int(point["load"]))
        if not closed:
            continue
        concurrency = [int(point["load"]) for point in closed]
        concurrency_ticks.update(concurrency)
        ax.plot(
            concurrency,
            [point["requests_per_second"] for point in closed],
            marker="o",
            label=_model_label(model),
        )
        plotted = True
    if plotted:
        ax.set_xlabel("Concurrent requests")
        ax.set_ylabel("Requests per second")
        ax.set_xscale("log", base=2)
        ticks = sorted(concurrency_ticks)
        ax.set_xticks(ticks, ticks)
        ax.set_title("Throughput by concurrency\nInput tokens: 1024 · 4 choices per request")
        if len(summaries) > 1:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(throughput_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    length_ticks: set[int] = set()
    for model, summary in summaries.items():
        shape_values = _shape_median(_reference_point(summary), "p50_ms")
        if not shape_values:
            continue
        lengths = sorted(shape_values)
        length_ticks.update(lengths)
        ax.plot(
            lengths,
            [shape_values[length] for length in lengths],
            marker="o",
            label=_model_label(model),
        )
        plotted = True
    if plotted:
        ax.set_xscale("log", base=2)
        ticks = sorted(length_ticks)
        ax.set_xticks(ticks, ticks)
        ax.set_xlabel("Server-reported input tokens per request")
        ax.set_ylabel("p50 request latency (ms)")
        ax.set_title("Concurrency-1 latency by input length")
        if len(summaries) > 1:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(latency_path, dpi=150)
    plt.close(fig)


def write_artifacts(
    out: Path,
    manifest: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name in (
        "report.md",
        "throughput-vs-concurrency.png",
        "latency-vs-input-length.png",
        "workload-latency.png",
        "latency-vs-goodput.png",
    ):
        (out / name).unlink(missing_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_summary_csv(summary, out / "summary.csv")
    write_plots({manifest["model"]["id"]: summary}, out)
    # Written last so readers can treat it as the run commit marker.
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_run(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((path / "manifest.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    if manifest["run_id"] != summary["run_id"]:
        raise ValueError(f"{path}: manifest and summary run IDs differ")
    return manifest, summary


def compatibility_errors(
    baseline_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
) -> list[str]:
    def normalized(value: Any) -> Any:
        return json.loads(json.dumps(value, sort_keys=True))

    checks = {
        "environment name": (
            baseline_manifest["environment"].get("name"),
            candidate_manifest["environment"].get("name"),
        ),
        "environment fingerprint": (
            baseline_manifest["environment"].get("fingerprint"),
            candidate_manifest["environment"].get("fingerprint"),
        ),
        "model": (
            baseline_manifest["model"].get("id"),
            candidate_manifest["model"].get("id"),
        ),
        "workload": (
            baseline_manifest.get("workload"),
            candidate_manifest.get("workload"),
        ),
        "measurement settings": (
            baseline_manifest.get("measurement"),
            candidate_manifest.get("measurement"),
        ),
        "candidate baseline": (
            baseline_manifest.get("run_id"),
            candidate_manifest.get("baseline_run_id"),
        ),
    }
    return [
        f"{name} differs: baseline={left!r}, candidate={right!r}"
        for name, (left, right) in checks.items()
        if normalized(left) != normalized(right)
    ]


def _change(baseline: float | None, candidate: float | None, lower_is_better: bool) -> str:
    if baseline in (None, 0) or candidate is None:
        return "unavailable"
    delta = (
        (baseline - candidate) / baseline if lower_is_better else (candidate - baseline) / baseline
    )
    return f"{100 * delta:+.2f}%"


def comparison_markdown(
    baseline_summary: dict[str, Any],
    candidate_summary: dict[str, Any],
) -> str:
    base = baseline_summary["headline"]
    cand = candidate_summary["headline"]

    def requests_per_second(summary: dict[str, Any], headline: dict[str, Any]) -> float | None:
        value = headline.get("max_requests_per_second")
        if value is not None:
            return float(value)
        peak = summary.get("peak") or {}
        value = peak.get("requests_per_second")
        if value is not None:
            return float(value)
        # Compatibility with benchmark results written before request throughput
        # became the headline metric. Every request in that workload had four decisions.
        decisions = headline.get("max_decisions_per_second")
        return float(decisions) / 4 if decisions is not None else None

    rows = [
        (
            "Peak successful throughput",
            requests_per_second(baseline_summary, base),
            requests_per_second(candidate_summary, cand),
            False,
            " requests/s",
        ),
        (
            "Reference p50 latency",
            base["reference_latency"].get("p50_ms"),
            cand["reference_latency"].get("p50_ms"),
            True,
            " ms",
        ),
        (
            "Cost per 1M decisions",
            base.get("cost_per_1m_decisions_usd"),
            cand.get("cost_per_1m_decisions_usd"),
            True,
            " USD",
        ),
    ]
    lines = [
        "# Benchmark comparison",
        "",
        "| Metric | Baseline | Candidate | Improvement |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, baseline, candidate, lower, suffix in rows:
        lines.append(
            f"| {name} | {_fmt(baseline, 4)}{suffix if baseline is not None else ''} "
            f"| {_fmt(candidate, 4)}{suffix if candidate is not None else ''} "
            f"| {_change(baseline, candidate, lower)} |"
        )
    lines.extend(
        [
            "",
            f"Candidate failure rate: {_fmt(100 * (cand.get('failure_rate') or 0), 3, '%')}.",
            "",
        ]
    )
    return "\n".join(lines)


def promote(candidate: Path, baseline: Path) -> None:
    candidate_manifest, candidate_summary = load_run(candidate)
    if (baseline / "manifest.json").is_file():
        baseline_manifest, _ = load_run(baseline)
        errors = compatibility_errors(baseline_manifest, candidate_manifest)
        if errors:
            raise ValueError("incompatible benchmark runs:\n- " + "\n- ".join(errors))
    required = [candidate / artifact for artifact in ARTIFACTS]
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"candidate is missing required artifacts: {', '.join(missing)}")
    headline = candidate_summary["headline"]
    if (headline.get("failure_rate") or 0) >= 0.01 or not headline.get("zero_oom_or_crash"):
        raise ValueError("candidate failed the reliability gate and cannot be promoted")

    baseline.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{baseline.name}.stage-", dir=str(baseline.parent)))
    try:
        for artifact in ARTIFACTS:
            shutil.copy2(candidate / artifact, stage / artifact)
        for name in ("throughput-vs-concurrency.png", "latency-vs-input-length.png"):
            if (candidate / name).is_file():
                shutil.copy2(candidate / name, stage / name)
        staged_manifest, staged_summary = load_run(stage)
        if staged_manifest["run_id"] != candidate_manifest["run_id"]:
            raise ValueError("staged run ID changed during promotion")
        if staged_summary["run_id"] != candidate_summary["run_id"]:
            raise ValueError("staged summary changed during promotion")
        baseline.mkdir(parents=True, exist_ok=True)
        for path in stage.iterdir():
            if path.name == "manifest.json":
                continue
            os.replace(path, baseline / path.name)
        for name in (
            "report.md",
            "throughput-vs-concurrency.png",
            "latency-vs-input-length.png",
            "workload-latency.png",
            "latency-vs-goodput.png",
        ):
            stale = baseline / name
            if not (candidate / name).is_file() and stale.exists():
                stale.unlink()
        os.replace(stage / "manifest.json", baseline / "manifest.json")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
