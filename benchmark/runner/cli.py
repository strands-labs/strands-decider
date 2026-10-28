from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from benchmark.runner.config import (
    accepted_baseline_dir,
    environment_runs_dir,
    load_environment,
    model_slug,
)
from benchmark.runner.core import BenchmarkRunner, git_metadata, machine_metadata
from benchmark.runner.reporting import (
    comparison_markdown,
    compatibility_errors,
    load_run,
    promote,
    write_artifacts,
    write_plots,
    write_summary_csv,
)
from benchmark.runner.workload import INPUT_LENGTHS, THROUGHPUT_INPUT_TOKENS

STANDARD_MEASUREMENT = {
    "requests_per_point": 16,
    "repetitions": 2,
    "concurrency_levels": [1, 4, 16, 32],
}
SMOKE_MEASUREMENT = {
    "requests_per_point": 8,
    "repetitions": 1,
    "concurrency_levels": [1],
}


def _run(args: argparse.Namespace) -> int:
    settings = SMOKE_MEASUREMENT if args.smoke else STANDARD_MEASUREMENT
    environment = load_environment(args.environment)
    if args.smoke and not args.model:
        raise ValueError("smoke runs require a model")
    models = [args.model] if args.model else list(environment.models)
    if not models:
        raise ValueError(f"{environment.path}: models are required for a full benchmark")
    measurement = {
        **settings,
        "throughput_requests_per_concurrency": {
            str(concurrency): max(settings["requests_per_point"], 2 * concurrency)
            for concurrency in settings["concurrency_levels"]
        },
        "preset": "smoke" if args.smoke else "standard",
        "reference_concurrency": 1,
        "failure_rate_limit": 0.01,
        "traffic": ["closed-loop"],
        "throughput_input_tokens": THROUGHPUT_INPUT_TOKENS,
        "latency_input_tokens": list(INPUT_LENGTHS),
        "warmup": (
            "each latency input length once; one request per concurrent client "
            "before each throughput repetition"
        ),
    }
    requests_per_model = (
        1
        + settings["repetitions"] * (settings["requests_per_point"] + len(INPUT_LENGTHS))
        + settings["repetitions"]
        * sum(
            max(settings["requests_per_point"], 2 * concurrency) + concurrency
            for concurrency in settings["concurrency_levels"]
        )
    )
    summaries: dict[str, dict[str, Any]] = {}
    outputs: list[Path] = []
    explicit_output = Path(args.output).expanduser().resolve() if args.output else None
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=Console(stderr=True),
    ) as progress:
        task = progress.add_task("benchmark models", total=requests_per_model * len(models))
        for model in models:
            progress.update(task, description=model.rsplit("/", 1)[-1])
            accepted = accepted_baseline_dir(environment, model)
            baseline_manifest = None
            if not args.smoke and (accepted / "manifest.json").is_file():
                baseline_manifest, _ = load_run(accepted)
                mismatches = []
                if baseline_manifest["environment"].get("fingerprint") != environment.fingerprint:
                    mismatches.append("environment configuration")
                if baseline_manifest["model"].get("id") != model:
                    mismatches.append("model")
                if baseline_manifest.get("measurement") != measurement:
                    mismatches.append("measurement settings")
                if (baseline_manifest.get("workload") or {}).get("seed") != 0:
                    mismatches.append("workload seed")
                if mismatches:
                    raise ValueError(
                        f"accepted baseline {accepted} is incompatible: {', '.join(mismatches)}"
                    )

            if explicit_output:
                out = explicit_output / model_slug(model) if len(models) > 1 else explicit_output
            elif args.smoke:
                out = accepted / "smoke"
            elif baseline_manifest is None:
                out = accepted
            else:
                out = accepted / "candidate"

            started = datetime.now(timezone.utc)
            runner = BenchmarkRunner(
                environment,
                model,
                requests_per_point=settings["requests_per_point"],
                repetitions=settings["repetitions"],
                concurrency_levels=settings["concurrency_levels"],
                seed=0,
                progress=progress.console.print,
                advance=lambda completed: progress.advance(task, completed),
            )
            summary, workload = runner.run()
            ended = datetime.now(timezone.utc)
            run_id = str(uuid.uuid4())
            manifest = {
                "run_id": run_id,
                "started_at": started.isoformat(),
                "ended_at": ended.isoformat(),
                "duration_seconds": round((ended - started).total_seconds(), 3),
                "environment": {
                    "name": environment.name,
                    "path": str(environment.path),
                    "fingerprint": environment.fingerprint,
                    "configuration": environment.raw,
                },
                "model": {
                    "id": model,
                    "tokenizer": summary.get("health", {}).get("checkpoint") or model,
                    "health": summary.get("health"),
                },
                "measurement": measurement,
                "workload": workload,
                "client": machine_metadata(),
                "code": git_metadata(Path(__file__).resolve().parents[2]),
                "baseline_run_id": baseline_manifest.get("run_id") if baseline_manifest else None,
            }
            summary["run_id"] = run_id
            write_artifacts(out, manifest, summary)
            summaries[model] = summary
            outputs.append(out)

    if len(summaries) > 1:
        graph_dir = explicit_output or environment_runs_dir(environment)
        graph_dir.mkdir(parents=True, exist_ok=True)
        write_plots(summaries, graph_dir, prefix="models-")
        print(graph_dir)
    for out in outputs:
        print(out)
    return 0


def _summarize(args: argparse.Namespace) -> int:
    path = Path(args.run).expanduser().resolve()
    manifest, summary = load_run(path)
    write_summary_csv(summary, path / "summary.csv")
    write_plots({manifest["model"]["id"]: summary}, path)
    (path / "report.md").unlink(missing_ok=True)
    print(path)
    return 0


def _compare(args: argparse.Namespace) -> int:
    environment = load_environment(args.environment)
    baseline = accepted_baseline_dir(environment, args.model)
    candidate = Path(args.candidate).expanduser().resolve()
    base_manifest, base_summary = load_run(baseline)
    candidate_manifest, candidate_summary = load_run(candidate)
    errors = compatibility_errors(base_manifest, candidate_manifest)
    if errors:
        raise ValueError("incompatible benchmark runs:\n- " + "\n- ".join(errors))
    report = comparison_markdown(base_summary, candidate_summary)
    if args.output:
        Path(args.output).expanduser().write_text(report)
    print(report)
    if args.update_baseline:
        promote(candidate, baseline)
        print(f"updated baseline: {baseline}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark.runner",
        description="Benchmark Strands Decider inference performance.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run a benchmark")
    run.add_argument("environment", help="environment name, directory, or YAML")
    run.add_argument(
        "model",
        nargs="?",
        help="optional checkpoint or model identifier; omit for all configured models",
    )
    run.add_argument(
        "--output",
        help="output directory; defaults beneath the environment's runs directory",
    )
    run.add_argument("--smoke", action="store_true", help="run the short development preset")
    run.set_defaults(func=_run)

    summarize = commands.add_parser("summarize", help="regenerate the CSV and graphs")
    summarize.add_argument("run", help="run directory")
    summarize.set_defaults(func=_summarize)

    compare = commands.add_parser("compare", help="compare candidate with accepted baseline")
    compare.add_argument("environment", help="environment name, directory, or YAML")
    compare.add_argument("model", help="checkpoint or model identifier")
    compare.add_argument("candidate", help="candidate run directory")
    compare.add_argument("--output", help="optional comparison Markdown path")
    compare.add_argument("--update-baseline", action="store_true")
    compare.set_defaults(func=_compare)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args) or 0)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {exc}\n")
