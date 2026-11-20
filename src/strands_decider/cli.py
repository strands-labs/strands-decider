"""The strands-decider CLI: ask and serve.

Training, evaluation, calibration and corpus-building live outside this package.
Run them as `python -m training.train`, `python -m training.evaluate`,
`python -m training.soup` and `python -m training.data` (see training/README.md).
"""

from __future__ import annotations

import atexit
import base64
import logging
import os
import sys
from pathlib import Path

import torch
import typer
from rich.console import Console

from .schema import ChoiceQuestion, NoulQuestion, Question, ScoreQuestion, SystemOneResponse

app = typer.Typer(
    name="strands-decider",
    help="A System One model: typed, calibrated answers instead of text.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


def _auto_device() -> str:
    """Best available torch device: cuda > mps > cpu. MLX is opt-in (`--device mlx`)."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _configure_inference_logging(device: str) -> None:
    if str(device).split(":", 1)[0] in {"cpu", "mps", "mlx"}:
        logging.getLogger("transformers.integrations.hub_kernels").setLevel(logging.ERROR)


def resolve_device(device: str | None) -> str:
    """Pick the torch device and silence the hub_kernels noise it would otherwise emit."""
    selected = device or _auto_device()
    _configure_inference_logging(selected)
    return selected


# ---------------------------------------------------------------- serve / ask


@app.command("serve")
def serve_cmd(
    checkpoint: str = typer.Argument(...),
    host: str = typer.Option("127.0.0.1"),
    port: int = typer.Option(8000),
    device: str | None = typer.Option(
        None,
        "--device",
        help="cuda, mps or cpu (torch), or mlx (Apple silicon, needs the mlx extra). "
        "Auto-detected when omitted: cuda > mps > cpu; mlx only when asked for.",
    ),
    no_prefix_cache: bool = typer.Option(
        False,
        "--no-prefix-cache",
        help="Encode each question with its own copy of the state on v1 model checkpoints, and in "
        "one full pass on other model checkpoints (the reference, on CUDA in bfloat16).",
    ),
    model_name: str | None = typer.Option(
        None,
        "--model-name",
        help="Value returned as `model` in responses. Defaults to the checkpoint basename.",
    ),
    vision: bool = typer.Option(
        False,
        "--vision",
        help="v1 model checkpoints: keep Qwen3.5's vision tower so requests may carry `images` "
        "(docs/vision.md). Other model checkpoints take images when their decision_config.json "
        "lists them.",
    ),
    max_batch: int = typer.Option(
        32,
        "--max-batch",
        help="Questions encoded per forward pass; lower it for very long states.",
    ),
    max_batch_tokens: int | None = typer.Option(
        None,
        "--max-batch-tokens",
        help="Also cap one forward's padded tokens (questions x longest state + question); for long-context torsos.",
    ),
    device_map: str | None = typer.Option(
        None,
        "--device-map",
        help="Split the model across the visible GPUs (a transformers device_map, such as auto), "
        "for a model larger than one GPU.",
    ),
    max_concurrency: int | None = typer.Option(
        None,
        "--max-concurrency",
        min=1,
        help="Run at most N requests on the model at once; the rest wait. v1 model checkpoints: no "
        "cap beyond the 40-thread worker pool by default, and a waiting request holds no thread or "
        "GPU memory. Other model checkpoints: 1 by default, and a request waits up to "
        "--queue-timeout.",
    ),
    dtype: str | None = typer.Option(
        None,
        "--dtype",
        help="float32 or bfloat16. Default: float32 on cpu, bfloat16 on cuda and mps. Not "
        "supported for v1 model checkpoints.",
    ),
    queue_timeout: float = typer.Option(
        60.0,
        "--queue-timeout",
        help="Seconds a request waits for the model before a 503. Not supported for v1 model "
        "checkpoints.",
    ),
    revision: str | None = typer.Option(
        None,
        "--revision",
        help="The Hub branch, tag or commit to load. Not supported for v1 model checkpoints.",
    ),
    max_length: int | None = typer.Option(
        None,
        "--max-length",
        min=1,
        help="Refuse prompts longer than this many tokens (default: decision_config.json's "
        "max_length_validated). Raising it past the validated length is the operator's decision. "
        "Not supported for v1 model checkpoints.",
    ),
    warmup: bool = typer.Option(
        True,
        "--warmup/--no-warmup",
        help="Answer every request shape once before serving, so no request waits for a compile.",
    ),
) -> None:
    """Serve POST /v1/systemone. JevBench's typesafe adapter runs against it unchanged."""
    import uvicorn

    from .server import create_app

    selected_device = resolve_device(device)
    try:  # before anything is announced: the arguments' checks, and a model checkpoint's load
        app_ = create_app(
            checkpoint,
            device=selected_device,
            use_prefix_cache=not no_prefix_cache,
            model_name=model_name,
            max_batch=max_batch,
            vision=vision,
            max_batch_tokens=max_batch_tokens,
            device_map=device_map,
            warmup=warmup,
            dtype=dtype,
            max_concurrency=max_concurrency,
            queue_timeout=queue_timeout,
            revision=revision,
            max_length=max_length,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    console.print(f"[green]serving[/] {checkpoint} on http://{host}:{port} ({selected_device})")
    # Single worker: the model owns the GPU, and forking more would just duplicate it.
    uvicorn.run(app_, host=host, port=port, workers=1)


@app.command("ask")
def ask_cmd(
    checkpoint: str = typer.Argument(...),
    state: str = typer.Option(..., "--state", "-s", help="The content to evaluate."),
    image: list[str] | None = typer.Option(
        None,
        "--image",
        help="Image file, part of the state; repeat. Loads the vision tower (v1); a decision "
        "checkpoint takes images when its decision_config.json lists them.",
    ),
    noul: list[str] | None = typer.Option(None, "--noul", help="Yes/no question; repeat."),
    choice: list[str] | None = typer.Option(
        None, "--choice", help="'question?=opt1,opt2,opt3'; repeat."
    ),
    score: list[str] | None = typer.Option(
        None, "--score", help="'question?=low,mid,high' (ascending); repeat."
    ),
    device: str | None = typer.Option(
        None,
        "--device",
        help="cuda, mps or cpu (torch), or mlx (Apple silicon, needs the mlx extra). "
        "Auto-detected when omitted: cuda > mps > cpu; mlx only when asked for.",
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the raw API response."),
    revision: str | None = typer.Option(
        None,
        "--revision",
        help="The Hub branch, tag or commit to load. Not supported for v1 model checkpoints.",
    ),
    dtype: str | None = typer.Option(
        None,
        "--dtype",
        help="float32 or bfloat16. Default: float32 on cpu, bfloat16 on cuda and mps. Not "
        "supported for v1 model checkpoints.",
    ),
) -> None:
    """Ask one state a set of typed questions from the command line."""
    from .infer import EngineConfig, load_engine

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

    selected_device = resolve_device(device)
    images = [base64.b64encode(Path(p).read_bytes()).decode("ascii") for p in image or []]
    from .decision import is_model_checkpoint, load_decision_engine

    if is_model_checkpoint(checkpoint, revision=revision):
        # Images go in the request; the checkpoint's decision_config.json says if it takes them.
        from .core import Request

        request = {
            "state": state,
            "questions": {n: q.model_dump() for n, q in questions.items()},
            "images": images,
        }
        try:
            engine = load_decision_engine(
                checkpoint, device=selected_device, dtype=dtype, revision=revision
            )
            response = engine.evaluate(Request.model_validate(request))
        except ValueError as e:  # a device or dtype, an image or a prompt it refuses
            raise typer.BadParameter(str(e)) from e
    elif dtype is not None or revision is not None:
        flag = "--dtype" if dtype is not None else "--revision"
        raise typer.BadParameter(
            f"{flag} is not supported for v1 model checkpoints", param_hint=flag
        )
    elif images:
        from .vision import load_vision_engine

        response = load_vision_engine(checkpoint, EngineConfig(device=selected_device)).ask_images(
            state, questions, images
        )
    else:
        response = load_engine(checkpoint, device=selected_device).ask(state, questions)

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
                f"[bold]{name}[/] -> [cyan]{ans.choice}[/] (confidence {ans.confidence:.3f})"
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
    that (no open files, no child processes), so once its answer is printed it runs
    the atexit handlers, flushes and leaves with os._exit. The handlers take a few ms
    and must run: multiprocessing's unlinks the named semaphore behind tqdm's lock (the
    "Loading weights" bar), and without it the resource tracker warns of a leaked
    semaphore after the prompt is back, wherever the start method is spawn or
    forkserver (macOS). Every other command, and anything that calls `app` directly
    (typer's CliRunner, an embedding program), exits normally.
    """
    try:
        app()
    except SystemExit as e:
        code = e.code
    else:
        code = 0
    if not (_exit_fast and code in (0, None)):
        sys.exit(code)
    atexit._run_exitfuncs()  # noqa: SLF001 -- no public API runs the handlers early
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
