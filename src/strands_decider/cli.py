"""The strands-decider CLI: ask and serve, and convert-mlx for decision checkpoints.

Training, evaluation, calibration and corpus-building live outside this package.
Run them as `python -m training.train`, `python -m training.evaluate`,
`python -m training.soup` and `python -m training.data` (see training/README.md).
"""

from __future__ import annotations

import atexit
import logging
import os
import sys

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
        help="Encode each question with its own copy of the state; for a decision checkpoint, "
        "one full pass per question (the spec's reference on CUDA in bfloat16).",
    ),
    model_name: str | None = typer.Option(
        None,
        "--model-name",
        help="Value returned as `model` in responses. Defaults to the checkpoint basename.",
    ),
    vision: bool = typer.Option(
        False,
        "--vision",
        help="Keep Qwen3.5's vision tower so requests may carry `images` (docs/vision.md). A "
        "decision checkpoint takes images when its decision_config.json lists them.",
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
    dtype: str | None = typer.Option(
        None,
        "--dtype",
        help="Decision checkpoints: float32 or bfloat16. Default: float32 on cpu, bfloat16 on "
        "cuda and mps, and the dtype the checkpoint is stored in on mlx.",
    ),
    max_concurrency: int | None = typer.Option(
        None,
        "--max-concurrency",
        help="Decision checkpoints: requests run on the model at once (default 1); the others "
        "wait.",
    ),
    queue_timeout: float = typer.Option(
        60.0,
        "--queue-timeout",
        help="Decision checkpoints: seconds a request waits for the model before a 503.",
    ),
    revision: str | None = typer.Option(
        None,
        "--revision",
        help="Decision checkpoints on the Hub: the branch, tag or commit to load.",
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
    try:  # the checks that refuse the arguments run here, before anything is announced
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
        help="Decision checkpoints on the Hub: the branch, tag or commit to load.",
    ),
    dtype: str | None = typer.Option(
        None,
        "--dtype",
        help="Decision checkpoints: float32 or bfloat16. Default: float32 on cpu, bfloat16 on "
        "cuda and mps, and the dtype the checkpoint is stored in on mlx.",
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
    from .decision import DECISION, INTERMEDIATE, checkpoint_kind, refuse_intermediate

    kind = checkpoint_kind(checkpoint, revision=revision)
    if kind == INTERMEDIATE:
        raise typer.BadParameter(str(refuse_intermediate(checkpoint)))
    if kind == DECISION:
        # A decision checkpoint (docs/spec.md): images go in the request, and the checkpoint's
        # decision_config.json says whether it takes them.
        from .decision import RequestRefused, check_device, check_dtype, load_decision_engine

        try:
            check_device(selected_device)
        except ValueError as e:
            raise typer.BadParameter(str(e), param_hint="--device") from e
        try:
            check_dtype(selected_device, dtype)
        except ValueError as e:
            raise typer.BadParameter(str(e), param_hint="--dtype") from e
        engine = load_decision_engine(
            checkpoint, device=selected_device, dtype=dtype, revision=revision
        )
        try:
            response = engine.ask(state, questions, _base64_files(image))
        except RequestRefused as e:  # the request's fault: an image or a prompt it refuses
            raise typer.BadParameter(str(e)) from e
    elif dtype is not None or revision is not None:
        flag = "--dtype" if dtype is not None else "--revision"
        raise typer.BadParameter(f"{flag} is for decision checkpoints", param_hint=flag)
    elif image:
        from .vision import load_vision_engine

        response = load_vision_engine(checkpoint, EngineConfig(device=selected_device)).ask_images(
            state, questions, _base64_files(image)
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


def _base64_files(paths: list[str] | None) -> list[str]:
    import base64

    encoded = []
    for path in paths or []:
        with open(path, "rb") as fh:
            encoded.append(base64.b64encode(fh.read()).decode("ascii"))
    return encoded


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


@app.command("convert-mlx")
def convert_mlx_cmd(
    checkpoint: str = typer.Argument(..., help="A decision checkpoint: a directory or a Hub id."),
    out: str = typer.Argument(..., help="The new directory to write the MLX artifact to."),
    bits: int | None = typer.Option(
        None,
        "--bits",
        help="Quantize the decoder to 8 or 4 bits. Omitted: bf16. The vision and audio "
        "encoders and their projections stay bf16, and the readout is copied unchanged. "
        "Only formats measured against the reference (8 bits, group 64, on E2B) make a "
        "conformance claim (docs/spec.md section 10).",
    ),
    group_size: int = typer.Option(64, "--group-size", help="Quantization group size."),
) -> None:
    """Write the MLX artifact of a decision checkpoint, for `--device mlx` (docs/spec.md)."""
    try:
        from .backends.mlx_convert import convert
    except ImportError as e:
        console.print(f"[red]error[/] {e}")
        raise typer.Exit(1) from e
    try:
        summary = convert(checkpoint, out, bits=bits, group_size=group_size)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as e:
        console.print(f"[red]error[/] {e}")
        raise typer.Exit(1) from e
    if summary["notice"]:
        console.print(f"[yellow]note[/] {summary['notice']}")
    q = summary["quantization"]
    console.print(
        f"[green]wrote[/] {summary['artifact']}: {summary['bytes'] / 1e9:.2f} GB, "
        + (f"decoder at {q['bits']} bits (group {q['group_size']})" if q else "bf16")
    )


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
