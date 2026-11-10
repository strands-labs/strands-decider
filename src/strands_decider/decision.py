"""Decision checkpoints behind `strands-decider ask` and `serve` (docs/spec.md).

`ask` and `serve` take v1 checkpoints and decision checkpoints alike. A directory, or a Hub
repo, whose `decision_config.json` says `"interface": "strands-decider"` is a decision
checkpoint (section 8: the interface field, never the file names, tells it apart), and
`load_decision_engine` serves it; anything else takes v1's pointer path, unchanged.

`DecisionEngine` answers a v1 request with the core: `strands_decider.core.evaluate` renders
it with the checkpoint's prompt, a backend runs the decoder, and the core reads the answers.
It takes the request as the JSON the client sent, so that fields v1's request model does not
know reach the core's checks: any `audio` but null, which the interface refuses
(docs/spec.md, section 1), is refused rather than dropped. Other unknown fields are ignored,
as v1 ignores them. On a torch device (cpu, cuda, mps) the backend is the transformers one
(`backends/transformers.py`); on `mlx`, the mlx-vlm one (`backends/mlx.py`).
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from .core import (
    NO_VISION,
    STAGES,
    DecisionConfig,
    Readout,
    RenderedRequest,
    load_tokenizer,
    parse_request,
    read,
    render,
)
from .core.config import FILENAME, IMAGE, INTERFACE
from .core.readout import check_rendered, check_stage
from .refusals import EngineBusy, RequestRefused
from .schema import (
    ChoiceQuestion,
    Content,
    NoulQuestion,
    Question,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
)

if TYPE_CHECKING:
    from tokenizers import Tokenizer

    from .core import Backend

DECISION = "decision"  # decision_config.json with "interface": "strands-decider"
INTERMEDIATE = "intermediate"  # decision_config.json from before the interface field
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _offline() -> bool:
    """Whether huggingface_hub is in offline mode (HF_HUB_OFFLINE)."""
    from huggingface_hub import constants

    check = getattr(constants, "is_offline_mode", None)
    return bool(check()) if callable(check) else bool(getattr(constants, "HF_HUB_OFFLINE", False))


def _decision_file(checkpoint: str, revision: str | None) -> str | None:
    """The local path of the checkpoint's decision_config.json, or None when it has none.

    A directory is read as it is. A string that is neither a directory nor a valid repo id
    (a mistyped path) is None, so v1's loader reports it as before. For a Hub repo:
    - offline, or at a pinned commit, the cache answers, a cached "no such file" included;
    - otherwise the Hub is asked: one small request, which the cache serves when it is
      current, so that a repo republished in another format is not judged by an old commit.
    When the Hub cannot be reached (offline, or a connection error or timeout), the cached
    snapshot decides, as v1's loader then uses it: with decision_config.json it is a decision
    checkpoint, without it a v1 one; with nothing cached the failure is raised (None offline,
    so that v1's loader reports it, as it did). "No such file in this repo", "no such repo"
    (gated included) and "no such revision" mean None, and v1's loader then reports the last
    two itself. An error the Hub answers with, an authorization error, a rate limit or a
    5xx, is raised."""
    if os.path.isdir(checkpoint):
        path = os.path.join(checkpoint, FILENAME)
        return path if os.path.isfile(path) else None
    from huggingface_hub import hf_hub_download, try_to_load_from_cache
    from huggingface_hub.errors import (
        HfHubHTTPError,
        HFValidationError,
        LocalEntryNotFoundError,
        RemoteEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    offline = _offline()
    try:
        if offline or (revision is not None and _COMMIT.fullmatch(revision)):
            cached = try_to_load_from_cache(checkpoint, FILENAME, revision=revision)
            if isinstance(cached, str):
                return cached
            if cached is not None:  # the cache knows this commit has no such file
                return None
        if offline:
            snapshot = _cached_snapshot(checkpoint, revision)
            return None if snapshot is None else _file_in(snapshot)
        try:
            return str(hf_hub_download(checkpoint, FILENAME, revision=revision))
        except LocalEntryNotFoundError as e:
            # The Hub was not reached, or answered with an error, and the file is not cached
            # on its own: a v1 repo never has it. An answer is raised; an unreachable Hub
            # leaves the decision to the cached snapshot, if there is one.
            if isinstance(e.__cause__, HfHubHTTPError):
                raise
            snapshot = _cached_snapshot(checkpoint, revision)
            if snapshot is None:
                raise
            return _file_in(snapshot)
    except HFValidationError:  # not a repo id: a local path that does not exist
        return None
    except (RemoteEntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError):
        return None


def _cached_snapshot(checkpoint: str, revision: str | None) -> str | None:
    """The cached snapshot of the repo at `revision`, or None when none is cached."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return str(snapshot_download(checkpoint, revision=revision, local_files_only=True))
    except LocalEntryNotFoundError:
        return None


def _file_in(snapshot: str) -> str | None:
    path = os.path.join(snapshot, FILENAME)
    return path if os.path.isfile(path) else None


def _local_dir(checkpoint: str, revision: str | None) -> str:
    """`checkpoint` if it is a directory, else the Hub repo's snapshot, downloaded to the Hub
    cache (as v1's `modeling.checkpoint_dir`, without torch)."""
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    try:
        return str(snapshot_download(checkpoint, revision=revision))
    except Exception as e:
        raise FileNotFoundError(
            f"{checkpoint}: not a local directory, and could not be downloaded from the "
            f"Hugging Face Hub: {type(e).__name__}: {e}"
        ) from e


def checkpoint_kind(checkpoint: str, *, revision: str | None = None) -> str | None:
    """`DECISION` for a decision checkpoint, `INTERMEDIATE` for a decision_config.json made
    before the `interface` field, None for anything else (a v1 checkpoint, or another
    format's decision_config.json)."""
    path = _decision_file(checkpoint, revision)
    if path is None:
        return None
    with open(path, encoding="utf-8") as f:
        dc = json.load(f)
    if not isinstance(dc, dict):
        return None
    if dc.get("interface") == INTERFACE:
        return DECISION
    if "interface" not in dc and "token_ids" in dc and "attention_mode" in dc:
        return INTERMEDIATE
    return None


def refuse_intermediate(checkpoint: str) -> ValueError:
    return ValueError(
        f'{checkpoint}: its {FILENAME} predates the decision interface (no "interface" '
        "field): an intermediate checkpoint. It names no prompt, so ask and serve cannot render "
        "requests for it; run it on token ids rendered elsewhere, with "
        "strands_decider.backends.transformers.TransformersBackend"
    )


class DecisionEngine:
    """A decision checkpoint, answering v1 requests: `evaluate` takes a `SystemOneRequest`, or
    the JSON object a client sent, and returns a `SystemOneResponse`, as v1's engines do, so
    `serve` and `ask` use either.

    It renders each request once with the core, then runs the backend and the core's readout,
    as `strands_decider.core.evaluate` does (tests/test_decision_serving.py checks that the two
    agree). At most `max_concurrency` requests run on the backend at once. One at a time, the
    default, keeps a burst of long or image requests from holding several states' keys and
    values on the device together. The others wait up to `queue_timeout` seconds (None: no
    limit) and then get `EngineBusy`. Refusals raise `RequestRefused`, and every refusal the
    core makes is decided before a request waits. An image file that its header accepts but
    that does not decode is found by the backend, which decodes it once it has a slot: under
    load such a request may wait, or get `EngineBusy`, before its refusal. A failure on a
    request that was accepted raises RuntimeError."""

    def __init__(
        self,
        backend: Backend,
        tokenizer: Tokenizer,
        config: DecisionConfig,
        readout: Readout,
        *,
        model_name: str,
        max_length: int | None = None,
        image_problem: str | None = None,
        max_concurrency: int = 1,
        queue_timeout: float | None = 60.0,
    ):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if queue_timeout is not None and queue_timeout <= 0:
            raise ValueError("queue_timeout must be positive, or None for no limit")
        stage = getattr(backend, "outputs", None)
        if stage not in STAGES:
            raise ValueError(
                f"the backend must say what forward returns: `outputs` is {stage!r}, not one "
                f"of {STAGES}"
            )
        check_stage(stage, readout, config)  # what reading its rows needs, before any request
        self.backend = backend
        self.stage: str = stage
        self.tokenizer = tokenizer
        self.config = config
        self.readout = readout
        self.model_name = model_name
        self.max_length = max_length
        self.image_problem = image_problem
        self.max_concurrency = max_concurrency
        self.queue_timeout = queue_timeout
        self._slots = threading.BoundedSemaphore(max_concurrency)

    @property
    def takes_images(self) -> bool:
        return IMAGE in self.config.inputs and not self.image_problem

    def _render(self, request: SystemOneRequest | dict[str, Any]) -> RenderedRequest:
        """The request rendered once, or the core's refusal of it, in the order `core.evaluate`
        keeps (docs/architecture.md, The core's runtime interface): the request's own checks,
        which leave its images undecoded; then images the checkpoint, or a backend with an
        `image_problem`, cannot take, before any is decoded; then the images' files and the
        prompt's length, by rendering the parsed request."""
        try:
            req = parse_request(request)
        except ValueError as e:
            raise RequestRefused(str(e)) from e
        if req.images:
            if IMAGE not in self.config.inputs:
                raise RequestRefused(
                    "this checkpoint does not take images: its decision_config.json inputs "
                    "are " + json.dumps(list(self.config.inputs))
                )
            if self.image_problem:
                raise RequestRefused(self.image_problem)
        try:
            rendered = render(
                req,
                self.tokenizer,
                self.config.prompt,
                max_length=self.max_length or self.config.max_length_validated,
            )
        except ValueError as e:
            raise RequestRefused(str(e)) from e
        try:  # the checkpoint's prompt and code ids: the runtime's to get right, not the caller's
            check_rendered(self.config, rendered)
        except ValueError as e:
            raise RuntimeError(str(e)) from e
        return rendered

    def evaluate(self, request: SystemOneRequest | dict[str, Any]) -> SystemOneResponse:
        rendered = self._render(request)
        state = list(rendered.state_ids)
        questions = [list(q.question_ids) for q in rendered.questions.values()]
        if not self._slots.acquire(timeout=self.queue_timeout):
            raise EngineBusy(
                f"the engine is busy: no slot came free in {self.queue_timeout:g} s "
                f"({self.max_concurrency} request(s) at a time)"
            )
        try:
            rows = self.backend.forward(state, questions, images=rendered.images)
        except RequestRefused:
            raise  # an image file the backend cannot decode
        except ValueError as e:
            if rendered.images and str(e) == NO_VISION:  # a backend without an image path
                raise RequestRefused(NO_VISION) from e
            raise RuntimeError(f"answering an accepted request failed: {e}") from e
        finally:
            self._slots.release()
        try:
            answers = read(rows, self.readout, self.config, rendered, stage=self.stage)
            return SystemOneResponse.model_validate(
                {
                    "model": self.model_name,
                    "answers": answers,
                    "usage": {
                        "input_tokens": rendered.input_tokens,
                        "output_tokens": len(answers),
                    },
                }
            )
        except ValueError as e:
            raise RuntimeError(f"reading the answers of an accepted request failed: {e}") from e

    def ask(
        self, state: Content, questions: dict[str, Question], images: Sequence[str] = ()
    ) -> SystemOneResponse:
        return self.evaluate(
            SystemOneRequest(state=state, questions=questions, images=list(images))
        )

    def close(self) -> None:
        """Release the backend's resources, if it holds any (the MLX backend's worker
        thread); `serve` calls it at shutdown."""
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()

    def describe(self) -> dict[str, Any]:
        """What `serve`'s health check reports."""
        info = dict(getattr(self.backend, "describe", dict)())
        return {
            "model": self.model_name,
            "interface": self.config.interface,
            "prompt": self.config.prompt,
            "base_model": self.config.provenance.base_model,
            "base_revision": self.config.provenance.base_revision,
            "max_length": self.max_length or self.config.max_length_validated,
            "temperature": self.config.temperature,
            "images": self.takes_images,
            "max_concurrency": self.max_concurrency,
            "queue_timeout": self.queue_timeout,
            **info,
        }

    def warm_up(self) -> None:
        """One request with a question of each kind, then one with an image if images are
        served, so that the first requests do not pay for lazy initialisation. The backend
        compiles nothing per shape, so v1's sweep over question counts is not needed."""
        started = time.perf_counter()
        for request in warm_up_requests(images=self.takes_images):
            self.evaluate(request)
        print(f"[strands-decider] warmed up in {time.perf_counter() - started:.1f} s")


def warm_up_requests(*, images: bool) -> list[SystemOneRequest]:
    state = "A warm-up request, answered before the server accepts requests."
    kinds: dict[str, Question] = {
        "noul": NoulQuestion(instructions="Is this a warm-up?"),
        "choice": ChoiceQuestion(instructions="Pick one.", criteria={"a": None, "b": None}),
        "score": ScoreQuestion(instructions="How warm is it?", criteria=["cold", "warm"]),
    }
    out = [SystemOneRequest(state=state, questions=kinds)]
    if images:
        import base64
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (128, 128, 128)).save(buf, "PNG")
        image = base64.b64encode(buf.getvalue()).decode("ascii")
        out.append(SystemOneRequest(state=state, questions={"noul": kinds["noul"]}, images=[image]))
    return out


# The decision backends, by device kind (the part of `--device` before any ":", such as
# "cuda" in "cuda:1"): the import path of a loader, imported the first time a checkpoint is
# loaded on that kind. A loader is called as
#     loader(path, device=..., dtype=..., reuse_state=..., max_batch=..., max_batch_tokens=...)
# with `path` the checkpoint's local directory, and returns a `strands_decider.core.Backend`;
# the backend may have `image_problem` (why images are refused: a backend without an image
# path sets it, so that image requests are refused before their files are read),
# `hidden_size` and `describe()`. A loader may list the dtypes it takes in a `dtypes` attribute; `dtype` is then
# checked before anything loads. Another runtime adds its device kinds here, or calls
# `register_backend`, which takes precedence.
BACKENDS: dict[str, str] = {
    "cpu": "strands_decider.backends.transformers:load_backend",
    "cuda": "strands_decider.backends.transformers:load_backend",
    "mps": "strands_decider.backends.transformers:load_backend",
    # Apple silicon through mlx-vlm (backends/mlx.py), in the dtype the checkpoint is stored in.
    "mlx": "strands_decider.backends.mlx:load_backend",
}
_LOADERS: dict[str, Callable[..., Backend]] = {}  # registered with register_backend
_IMPORTED: dict[str, Callable[..., Backend]] = {}  # BACKENDS' loaders, by import path
_KIND = re.compile(r"[a-z][a-z0-9_]*")


def register_backend(devices: Sequence[str], loader: Callable[..., Backend]) -> None:
    """Serve decision checkpoints on the device kinds `devices` (such as "mlx", never
    "cuda:1") with `loader` (see `BACKENDS`)."""
    for d in devices:
        if not _KIND.fullmatch(d):
            raise ValueError(f"register device kinds, such as 'mlx' or 'cuda', not {d!r}")
        _LOADERS[d] = loader


def _device_kind(device: str) -> str:
    return str(device).split(":", 1)[0]


def check_device(device: str) -> None:
    """ValueError unless a backend serves decision checkpoints on `device`, a device kind with
    an optional ":<index>"."""
    kind, sep, index = str(device).partition(":")
    if kind not in _LOADERS and kind not in BACKENDS:
        known = sorted({*BACKENDS, *_LOADERS})
        raise ValueError(f"decision checkpoints run on {', '.join(known)} here, not {device!r}")
    if sep and not index.isdigit():
        raise ValueError(f"a device index is a number, as in {kind}:0, not {device!r}")


def backend_loader(device: str) -> Callable[..., Backend]:
    """The loader for `device`'s kind, imported now if it is not yet."""
    check_device(device)
    kind = _device_kind(device)
    if kind in _LOADERS:
        return _LOADERS[kind]
    target = BACKENDS[kind]
    if target not in _IMPORTED:
        import importlib

        module, _, name = target.partition(":")
        _IMPORTED[target] = getattr(importlib.import_module(module), name)
    return _IMPORTED[target]


def check_dtype(device: str, dtype: str | None) -> None:
    """ValueError unless the backend for `device` takes `dtype` (None: its own)."""
    allowed = getattr(backend_loader(device), "dtypes", None)
    if dtype is not None and allowed is not None and dtype not in allowed:
        raise ValueError(f"--dtype on {device} is one of {', '.join(allowed)}, not {dtype!r}")


def load_decision_engine(
    checkpoint: str,
    *,
    device: str,
    model_name: str | None = None,
    reuse_state: bool = True,
    max_batch: int = 32,
    max_batch_tokens: int | None = None,
    max_length: int | None = None,
    dtype: str | None = None,
    max_concurrency: int = 1,
    queue_timeout: float | None = 60.0,
    revision: str | None = None,
) -> DecisionEngine:
    """A decision checkpoint on `device`, through the backend registered for it (`BACKENDS`):
    on cpu, cuda and mps the transformers backend, in `dtype` (float32 or bfloat16) or the
    device's own (float32 on CPU, bfloat16 on CUDA and MPS); on mlx the MLX backend, in `dtype`
    or the dtype the checkpoint is stored in. The interface's files, decision_config.json
    (validated strictly), readout.safetensors and the tokenizer, are read here for every
    backend. `reuse_state=False` runs each question as one full prompt; on CUDA in bfloat16
    that is the reference of docs/spec.md section 10."""
    loader = backend_loader(device)
    check_dtype(device, dtype)
    path = _local_dir(checkpoint, revision)
    with open(os.path.join(path, FILENAME), encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "interface" not in raw:
        raise refuse_intermediate(checkpoint)
    config = DecisionConfig.from_dict(raw)
    readout = Readout.load(path, config)
    # Its control and code ids checked before any weight loads, and handed to the core as it is
    # checked, so that neither the check nor render keeps a copy (about 170 MB for Gemma 4's).
    tokenizer = load_tokenizer(path, config, hand_over=True)
    backend = loader(
        path,
        device=device,
        dtype=dtype,
        reuse_state=reuse_state,
        max_batch=max_batch,
        max_batch_tokens=max_batch_tokens,
    )
    hidden = getattr(backend, "hidden_size", None)
    if hidden is not None and readout.weight is not None and readout.weight.shape[1] != hidden:
        raise ValueError(
            f"{checkpoint}: readout.safetensors `weight` is {readout.weight.shape}, not "
            f"[255, {hidden}]: the decoder's hidden size"
        )
    describe = getattr(backend, "describe", None)
    d = describe() if callable(describe) else {"device": device}
    image_problem = getattr(backend, "image_problem", None)
    inputs = list(config.inputs) if IMAGE in config.inputs and not image_problem else ["text"]
    print(
        f"[strands-decider] decision checkpoint: {d.get('backend', 'backend')} on "
        f"{d.get('device', device)} in {d.get('dtype', 'its dtype')}, inputs {inputs}, "
        + ("state encoded once per request" if reuse_state else "one full pass per question"),
        file=sys.stderr,
    )
    name = model_name or os.path.basename(checkpoint.rstrip("/")) or checkpoint
    return DecisionEngine(
        backend,
        tokenizer,
        config,
        readout,
        model_name=name,
        max_length=max_length,
        image_problem=image_problem,
        max_concurrency=max_concurrency,
        queue_timeout=queue_timeout,
    )


__all__ = [
    "BACKENDS",
    "DECISION",
    "INTERMEDIATE",
    "DecisionEngine",
    "EngineBusy",
    "RequestRefused",
    "backend_loader",
    "check_device",
    "check_dtype",
    "checkpoint_kind",
    "load_decision_engine",
    "refuse_intermediate",
    "register_backend",
    "warm_up_requests",
]
