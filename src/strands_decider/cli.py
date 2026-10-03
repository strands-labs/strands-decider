"""The strands-decider CLI: build data, train, calibrate, evaluate, serve, ask.

Only `ask` and `serve` show in `--help`. The training and evaluation commands stay
registered but hidden: `strands-decider train --help` works, and training/recipe.sh
calls them through `python -m strands_decider.cli`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections import Counter
from itertools import islice

import torch
import typer
from rich.console import Console
from rich.table import Table

from .data import recipes as _recipes
from .data.format import Example, read_jsonl, write_jsonl
from .prompting import build_prompt
from .schema import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion, SystemOneResponse

app = typer.Typer(
    name="strands-decider",
    help="A System One model: typed, calibrated answers instead of text.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Build and inspect training corpora.", no_args_is_help=True)
app.add_typer(data_app, name="data", hidden=True)

console = Console()


# ---------------------------------------------------------------- data


@data_app.command("recipes")
def data_recipes() -> None:
    """List the datasets that can be converted into training examples."""
    table = Table(title="Available recipes")
    for col in ("name", "dataset", "config", "split", "max examples"):
        table.add_column(col)
    for name in sorted(_recipes.SPECS):
        s = _recipes.SPECS[name]
        table.add_row(name, s.path, s.config or "-", s.split, str(s.max_examples))
    console.print(table)


@data_app.command("build")
def data_build(
    out: str = typer.Option("data/train.jsonl", help="Output JSONL (.gz supported)."),
    recipes: list[str] | None = typer.Option(
        None, "--recipe", "-r",
        help="Recipe name; repeat. Default: all. If a recipe fails, the build writes nothing.",
    ),
    max_options: int = typer.Option(16, help="Cap options per example (large label sets)."),
    max_chars: int = typer.Option(2000, help="Truncate each state to this many characters."),
    max_examples: int | None = typer.Option(None, help="Per-recipe example cap."),
    seed: int = typer.Option(0),
    holdout: list[str] | None = typer.Option(
        None, "--holdout", help="Task to route to a separate held-out file; repeat."
    ),
) -> None:
    """Download datasets and convert them to (state, question, answer) examples.

    Tasks named with --holdout are written to a sibling *.holdout.jsonl instead, so
    you can measure generalisation to label sets the model never trained on.
    """
    names = list(recipes) if recipes else _recipes.available_recipes()
    hold = set(holdout or [])

    kept: list[Example] = []
    held: list[Example] = []
    for name in names:
        console.print(f"[cyan]building[/] {name} ...")
        try:
            examples = _recipes.build_recipe(
                name,
                max_options=max_options,
                max_chars=max_chars,
                seed=seed,
                max_examples=max_examples,
            )
        except Exception as exc:
            # Stop here. A missing recipe removes rows and moves every later row to a new
            # position. The frozen teacher and replay targets attach to rows by position.
            hint = (' Install the "train" extra: pip install -e ".[train]".'
                    if isinstance(exc, ModuleNotFoundError) and exc.name == "datasets" else "")
            raise SystemExit(f"recipe {name} failed, no corpus written: {exc!r}.{hint}") from exc
        (held if name in hold else kept).extend(examples)
        console.print(f"  [green]{len(examples):,}[/] examples")

    if not kept:
        raise typer.Exit(code=1)

    n = write_jsonl(out, kept)
    console.print(f"[green]wrote[/] {n:,} examples -> {out}")
    if held:
        hpath = out.replace(".jsonl", ".holdout.jsonl")
        console.print(f"[green]wrote[/] {write_jsonl(hpath, held):,} held-out -> {hpath}")


@data_app.command("stats")
def data_stats(path: str = typer.Argument(..., help="JSONL corpus to summarise.")) -> None:
    """Show composition of a corpus: tasks, kinds, option-count distribution."""
    tasks: Counter = Counter()
    kinds: Counter = Counter()
    nopts: Counter = Counter()
    total = 0
    for ex in read_jsonl(path):
        tasks[ex.task] += 1
        kinds[ex.kind] += 1
        nopts[ex.n_options] += 1
        total += 1

    console.print(f"[bold]{total:,}[/] examples in {path}")
    t = Table(title="By task")
    t.add_column("task")
    t.add_column("n", justify="right")
    t.add_column("share", justify="right")
    for name, c in tasks.most_common():
        t.add_row(name, f"{c:,}", f"{100 * c / total:.1f}%")
    console.print(t)

    k = Table(title="By kind / option count")
    k.add_column("kind")
    k.add_column("n", justify="right")
    for name, c in kinds.most_common():
        k.add_row(name, f"{c:,}")
    console.print(k)
    console.print("option counts: " + ", ".join(f"{n}:{c:,}" for n, c in sorted(nopts.items())))


@data_app.command("peek")
def data_peek(
    path: str = typer.Argument(...),
    n: int = typer.Option(3, help="How many rendered prompts to print."),
) -> None:
    """Render a few examples exactly as the model will see them."""
    for ex in islice(read_jsonl(path), n):
        prompt, rq = build_prompt(ex.state, ex.to_question())
        console.rule(f"{ex.task} / {ex.kind} / gold slot {ex.label}")
        console.print(prompt)
        console.print(f"[dim]slots: {rq.slot_labels}[/]")


# ---------------------------------------------------------------- train


@app.command("train", hidden=True)
def train_cmd(
    config: str | None = typer.Option(None, "--config", "-c", help="YAML training config."),
    train_file: list[str] | None = typer.Option(None, "--train-file", help="Override data."),
    base_model: str | None = typer.Option(None, help="Override base model id."),
    output_dir: str | None = typer.Option(None, help="Override checkpoint directory."),
    epochs: int | None = typer.Option(None),
    micro_batch_size: int | None = typer.Option(None),
    max_steps: int | None = typer.Option(None, help="Hard step cap (smoke tests)."),
) -> None:
    """Train the slot heads and a LoRA adapter on the torso."""
    from .train import TrainConfig, train

    cfg = TrainConfig.from_yaml(config) if config else TrainConfig()
    if train_file:
        cfg.train_files = list(train_file)
    for key, val in (
        ("base_model", base_model),
        ("output_dir", output_dir),
        ("epochs", epochs),
        ("micro_batch_size", micro_batch_size),
        ("max_steps", max_steps),
    ):
        if val is not None:
            setattr(cfg, key, val)
    if not cfg.train_files:
        raise typer.BadParameter("no training files; pass --train-file or set train_files")
    train(cfg)


@app.command("calibrate", hidden=True)
def calibrate_cmd(
    checkpoint: str = typer.Argument(...),
    data: str = typer.Option(..., "--data", help="Held-out JSONL for fitting temperature."),
    limit: int = typer.Option(4000, help="Cap examples used."),
    batch_size: int = typer.Option(16),
    split: str = typer.Option(
        "calib", help="Half of --data to use: calib|test|all. Disjoint from `eval --split test`."
    ),
) -> None:
    """Fit a temperature on held-out data and write it into the checkpoint.

    Run this before serving. Without it, `confidence` is whatever the raw head
    happened to produce, and the documented thresholds will not hold.

    Defaults to the `calib` half so that `strands-decider eval --split test` scores rows the
    temperature was never fitted on.
    """

    from .evaluate import calibrate_checkpoint, partition_examples, sample_examples

    examples = sample_examples(partition_examples(list(read_jsonl(data)), split), limit)
    result = calibrate_checkpoint(checkpoint, examples, batch_size=batch_size)
    console.print(f"[green]global temperature = {result['temperature']:.4f}[/]")
    by_kind = result.get("temperature_by_kind") or {}
    if by_kind:
        console.print(
            "[green]per-primitive[/] "
            + ", ".join(f"{k}={v:.4f}" for k, v in sorted(by_kind.items()))
            + "  (written to checkpoint)"
        )
    console.print("before:      ", result["before"])
    if "after_global" in result:
        console.print("after global:", result["after_global"])
    console.print("after:       ", result["after"])


@app.command("eval", hidden=True)
def eval_cmd(
    checkpoint: str = typer.Argument(...),
    data: str = typer.Option(..., "--data"),
    limit: int = typer.Option(4000),
    batch_size: int = typer.Option(16),
    out: str | None = typer.Option(None, help="Write the full report as JSON."),
    raw: bool = typer.Option(False, "--raw", help="Ignore the fitted temperature."),
    split: str = typer.Option(
        "test", help="Half of --data to score: test|calib|all. Disjoint from `calibrate`."
    ),
) -> None:
    """Accuracy, ECE, NLL and score MAE, broken down by task and confidence band.

    Defaults to the `test` half, which `strands-decider calibrate` does not fit on.
    """
    from .evaluate import evaluate_checkpoint, partition_examples, sample_examples

    examples = sample_examples(partition_examples(list(read_jsonl(data)), split), limit)
    report = evaluate_checkpoint(
        checkpoint, examples, batch_size=batch_size, apply_temperature=not raw
    )

    overall = report["overall"]
    console.print(
        f"[bold]overall[/] n={overall['n']:,} acc={overall['accuracy']:.3f} "
        f"ece={overall['ece']:.3f} nll={overall['nll']:.3f} T={_fmt_temperature(report['temperature'])}"
    )

    for section in ("by_kind", "by_task", "by_confidence_band"):
        rows = {k: v for k, v in report[section].items() if v}
        if not rows:
            continue
        t = Table(title=section)
        for col in ("group", "n", "accuracy", "ece", "mean conf"):
            t.add_column(col, justify="right" if col != "group" else "left")
        for name, m in rows.items():
            t.add_row(
                name, f"{m['n']:,}", f"{m['accuracy']:.3f}",
                f"{m['ece']:.3f}", f"{m['mean_confidence']:.3f}",
            )
        console.print(t)

    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        console.print(f"[green]report -> {out}[/]")


def _auto_device() -> str:
    """Best available torch device: cuda > mps > cpu. MLX is opt-in (`--device mlx`)."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _fmt_temperature(t: float | dict[str, float]) -> str:
    """Temperature is a scalar, or a {primitive: value} map once fitted per kind."""
    if isinstance(t, dict):
        return "{" + ", ".join(f"{k}:{v:.3f}" for k, v in sorted(t.items())) + "}"
    return f"{t:.3f}"


def _configure_inference_logging(device: str) -> None:
    if str(device).split(":", 1)[0] in {"cpu", "mps", "mlx"}:
        logging.getLogger("transformers.integrations.hub_kernels").setLevel(logging.ERROR)


# ---------------------------------------------------------------- serve / ask


@app.command("serve")
def serve_cmd(
    checkpoint: str = typer.Argument(...),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    device: str | None = typer.Option(
        None, "--device",
        help="cuda, mps or cpu (torch), or mlx (Apple silicon, needs the mlx extra). "
        "Auto-detected when omitted: cuda > mps > cpu; mlx only when asked for.",
    ),
    no_prefix_cache: bool = typer.Option(False, "--no-prefix-cache"),
    model_name: str | None = typer.Option(
        None, "--model-name",
        help="Value returned as `model` in responses. Defaults to the checkpoint basename.",
    ),
    strict_window: bool = typer.Option(
        False, "--strict-window",
        help="Reject (HTTP 422) a prompt longer than the context window instead of truncating it.",
    ),
    max_batch: int = typer.Option(
        32, "--max-batch", help="Questions encoded per forward pass; lower it for very long states.",
    ),
) -> None:
    """Serve POST /v1/systemone. JevBench's typesafe adapter runs against it unchanged."""
    from .server import serve

    selected_device = device or _auto_device()
    _configure_inference_logging(selected_device)
    console.print(f"[green]serving[/] {checkpoint} on http://{host}:{port} ({selected_device})")
    serve(
        checkpoint, host=host, port=port, device=selected_device,
        use_prefix_cache=not no_prefix_cache, model_name=model_name,
        strict_window=strict_window, max_batch=max_batch,
    )


@app.command("ask")
def ask_cmd(
    checkpoint: str = typer.Argument(...),
    state: str = typer.Option(..., "--state", "-s", help="The content to evaluate."),
    noul: list[str] | None = typer.Option(None, "--noul", help="Yes/no question; repeat."),
    choice: list[str] | None = typer.Option(
        None, "--choice", help="'question?=opt1,opt2,opt3'; repeat."
    ),
    score: list[str] | None = typer.Option(
        None, "--score", help="'question?=low,mid,high' (ascending); repeat."
    ),
    device: str | None = typer.Option(
        None, "--device",
        help="cuda, mps or cpu (torch), or mlx (Apple silicon, needs the mlx extra). "
        "Auto-detected when omitted: cuda > mps > cpu; mlx only when asked for.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the raw API response."),
) -> None:
    """Ask one state a set of typed questions from the command line."""
    from .infer import load_engine

    questions: dict[str, Question] = {}
    for i, q in enumerate(noul or []):
        questions[f"noul_{i}"] = NoulQuestion(instructions=q)
    for i, spec in enumerate(choice or []):
        q, _, opts = spec.partition("=")
        if not opts:
            raise typer.BadParameter(f"--choice needs 'question=opt1,opt2': {spec!r}")
        questions[f"choice_{i}"] = ChoiceQuestion(
            instructions=q, criteria={o.strip(): "" for o in opts.split(",") if o.strip()}
        )
    for i, spec in enumerate(score or []):
        q, _, levels = spec.partition("=")
        if not levels:
            raise typer.BadParameter(f"--score needs 'question=low,high': {spec!r}")
        questions[f"score_{i}"] = ScoreQuestion(
            instructions=q, criteria=[s.strip() for s in levels.split(",") if s.strip()]
        )
    if not questions:
        raise typer.BadParameter("ask at least one --noul / --choice / --score question")

    selected_device = device or _auto_device()
    _configure_inference_logging(selected_device)
    engine = load_engine(checkpoint, device=selected_device)
    response = engine.ask(state, questions)

    if as_json:
        console.print_json(response.model_dump_json())
    else:
        _print_answers(response)
    # `ask` is done once its answer is printed; see main().
    global _exit_fast
    _exit_fast = True


def _print_answers(response: SystemOneResponse) -> None:
    for name, ans in response.answers.items():
        if ans.type == "noul":
            console.print(f"[bold]{name}[/] noul = [cyan]{ans.noul:.3f}[/]")
        elif ans.type == "choice":
            console.print(
                f"[bold]{name}[/] -> [cyan]{ans.choice}[/] "
                f"(confidence {ans.confidence:.3f})"
            )
            for opt, p in sorted(ans.probabilities.items(), key=lambda kv: -kv[1]):
                console.print(f"    {opt:<24} {p:.3f}")
        else:
            console.print(
                f"[bold]{name}[/] score = [cyan]{ans.score:.2f}[/] "
                f"(confidence {ans.confidence:.3f})"
            )
            for lvl, p in ans.probabilities.items():
                console.print(f"    {lvl}: {ans.legend.get(lvl, ''):<40} {p:.3f}")


@app.command("info", hidden=True)
def info_cmd(checkpoint: str = typer.Argument(...)) -> None:
    """Print a checkpoint's configuration."""
    from .modeling import StrandsDeciderConfig, checkpoint_dir, config_path

    cfg = StrandsDeciderConfig.from_json(config_path(checkpoint_dir(checkpoint)))
    console.print_json(cfg.to_json())


# Set by a command whose work is finished once it returns; read by main().
_exit_fast = False


def main() -> None:
    """Console-script entry point: `app()`, then a fast exit after a finished `ask`.

    Interpreter shutdown after `ask` takes about 0.5 s on an M3 Pro: torch, MPS and
    every imported module are finalized. A one-shot `ask` holds nothing that needs
    that (no open files, no child processes, no atexit work), so once its answer is
    printed and flushed it leaves with os._exit. Every other command, and anything
    that calls `app` directly (typer's CliRunner, an embedding program), exits
    normally.
    """
    try:
        app()
    except SystemExit as e:
        code = e.code
    else:
        code = 0
    if not (_exit_fast and code in (0, None)):
        sys.exit(code)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
