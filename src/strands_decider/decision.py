"""Decision checkpoints behind `strands-decider ask` and `serve` (docs/spec.md).

`ask` and `serve` take v1 checkpoints and decision checkpoints alike. A directory, or a Hub
repo, whose `decision_config.json` says `"interface": "strands-decider"` is a decision
checkpoint (section 8: the interface field, never the file names, tells it apart), and
`load_decision_engine` serves it; anything else takes v1's pointer path, unchanged.

`DecisionEngine` answers a v1 request with the core: `strands_decider.core.evaluate` renders
it with the checkpoint's prompt, a backend runs the decoder, and the core reads the answers.
It takes the request as the JSON the client sent, so that fields v1's request model does not
know, such as `audio`, reach the core's checks and are refused rather than dropped. On a torch
device (cpu, cuda, mps) the backend is the transformers one (`backends/transformers.py`).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .core import DecisionConfig, Readout, evaluate, render
from .core.config import FILENAME, IMAGE, INTERFACE
from .core.request import NO_VISION, has_images
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
DTYPES = ("float32", "bfloat16")


class RequestRefused(ValueError):
    """A request the decision interface refuses: the caller's error (HTTP 422). Any other
    failure while answering an accepted request is the runtime's (HTTP 500)."""


def _decision_file(checkpoint: str, revision: str | None) -> str | None:
    """The local path of the checkpoint's decision_config.json, or None when it has none.

    A directory is read as it is. For a Hub repo the cache is asked first, so a v1 checkpoint
    costs at most one Hub request, the first time: the cache then records that the file does
    not exist. Only "no such file in this repo" and "no such repo" (gated included) mean None;
    v1's loader then reports the second itself. Any other failure, such as an authorization
    error, a rate limit or the network, is raised. Offline, with nothing cached for the file,
    the cached snapshot decides."""
    if os.path.isdir(checkpoint):
        path = os.path.join(checkpoint, FILENAME)
        return path if os.path.isfile(path) else None
    from huggingface_hub import hf_hub_download, snapshot_download, try_to_load_from_cache
    from huggingface_hub.errors import (
        LocalEntryNotFoundError,
        RemoteEntryNotFoundError,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    cached = try_to_load_from_cache(checkpoint, FILENAME, revision=revision)
    if isinstance(cached, str):
        return cached
    if cached is not None:  # the cache knows the repo has no such file
        return None
    try:
        return str(hf_hub_download(checkpoint, FILENAME, revision=revision))
    except RemoteEntryNotFoundError:
        return None
    except (RepositoryNotFoundError, RevisionNotFoundError):
        return None
    except LocalEntryNotFoundError:
        try:
            snapshot = snapshot_download(checkpoint, revision=revision, local_files_only=True)
        except Exception:  # nothing cached either: v1's loader reports it
            return None
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

    At most `max_concurrency` requests run on the backend at once; the others wait. One at a
    time, the default, keeps a burst of long or image requests from holding several states'
    keys and values on the device together. Refusals raise `RequestRefused`; a failure on a
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
    ):
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self.backend = backend
        self.tokenizer = tokenizer
        self.config = config
        self.readout = readout
        self.model_name = model_name
        self.max_length = max_length
        self.image_problem = image_problem
        self.max_concurrency = max_concurrency
        self._slots = threading.BoundedSemaphore(max_concurrency)

    @property
    def takes_images(self) -> bool:
        return IMAGE in self.config.inputs and IMAGE in tuple(getattr(self.backend, "inputs", ()))

    def _refusals(self, payload: Any) -> None:
        """What the core refuses before the backend runs, in its order: images the checkpoint
        or the backend cannot take, before any is decoded; then the request's schema, its
        images' files and audio, and the prompt's length, by rendering it."""
        if has_images(payload):
            if IMAGE not in self.config.inputs:
                raise RequestRefused(
                    "this checkpoint does not take images: its decision_config.json inputs "
                    "are " + json.dumps(list(self.config.inputs))
                )
            if self.image_problem:
                raise RequestRefused(self.image_problem)
            if IMAGE not in tuple(getattr(self.backend, "inputs", ())):
                raise RequestRefused(NO_VISION)
        try:
            render(
                payload,
                self.tokenizer,
                self.config.prompt,
                max_length=self.max_length or self.config.max_length_validated,
            )
        except ValueError as e:
            raise RequestRefused(str(e)) from e

    def evaluate(self, request: SystemOneRequest | Mapping[str, Any]) -> SystemOneResponse:
        payload = request.model_dump() if isinstance(request, SystemOneRequest) else request
        self._refusals(payload)
        with self._slots:
            try:
                out = evaluate(
                    payload,
                    self.tokenizer,
                    self.readout,
                    self.config,
                    self.backend,
                    model_name=self.model_name,
                    max_length=self.max_length,
                )
            except RequestRefused:
                raise  # an image file the backend cannot decode
            except ValueError as e:
                raise RuntimeError(f"answering an accepted request failed: {e}") from e
        try:
            return SystemOneResponse.model_validate(out)
        except ValueError as e:
            raise RuntimeError(f"the answers do not fit the response schema: {e}") from e

    def ask(
        self, state: Content, questions: dict[str, Question], images: Sequence[str] = ()
    ) -> SystemOneResponse:
        return self.evaluate(
            SystemOneRequest(state=state, questions=questions, images=list(images))
        )

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


# The decision backends, by device kind (the part of `--device` before any ":"): the import
# path of a loader, imported when a checkpoint is loaded on that device, so that one runtime's
# framework never loads another's. A loader is called as
#     loader(path, device=..., dtype=..., reuse_state=..., max_batch=..., max_batch_tokens=...)
# with `path` the checkpoint's local directory, and returns a `strands_decider.core.Backend`;
# it may give the backend `image_problem` (why images are refused) and `describe()`. Another
# runtime adds its devices here, or calls `register_backend`.
BACKENDS: dict[str, str] = {
    "cpu": "strands_decider.backends.transformers:load_backend",
    "cuda": "strands_decider.backends.transformers:load_backend",
    "mps": "strands_decider.backends.transformers:load_backend",
}
_LOADERS: dict[str, Callable[..., Backend]] = {}


def register_backend(devices: Sequence[str], loader: Callable[..., Backend]) -> None:
    """Serve decision checkpoints on `devices` with `loader` (see `BACKENDS`)."""
    for d in devices:
        _LOADERS[d] = loader


def _device_kind(device: str) -> str:
    return str(device).split(":", 1)[0]


def check_device(device: str) -> None:
    """ValueError unless some backend serves decision checkpoints on `device`."""
    kind = _device_kind(device)
    if kind not in _LOADERS and kind not in BACKENDS:
        known = sorted({*BACKENDS, *_LOADERS})
        raise ValueError(f"decision checkpoints run on {', '.join(known)} here, not {device!r}")


def backend_loader(device: str) -> Callable[..., Backend]:
    check_device(device)
    kind = _device_kind(device)
    if kind not in _LOADERS:
        import importlib

        module, _, name = BACKENDS[kind].partition(":")
        _LOADERS[kind] = getattr(importlib.import_module(module), name)
    return _LOADERS[kind]


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
    revision: str | None = None,
) -> DecisionEngine:
    """A decision checkpoint on `device`, through the backend registered for it (`BACKENDS`):
    on cpu, cuda and mps the transformers backend, in `dtype` (`DTYPES`) or the device's own
    (float32 on CPU, bfloat16 on CUDA and MPS). The interface's files, decision_config.json
    (validated strictly), readout.safetensors and the tokenizer, are read here for every
    backend. `reuse_state=False` runs each question as one full prompt; on CUDA in bfloat16
    that is the reference of docs/spec.md section 10."""
    loader = backend_loader(device)
    if dtype is not None and dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, not {dtype!r}")
    from tokenizers import Tokenizer

    path = _local_dir(checkpoint, revision)
    with open(os.path.join(path, FILENAME), encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "interface" not in raw:
        raise refuse_intermediate(checkpoint)
    config = DecisionConfig.from_dict(raw)
    readout = Readout.load(path, config)
    backend = loader(
        path,
        device=device,
        dtype=dtype,
        reuse_state=reuse_state,
        max_batch=max_batch,
        max_batch_tokens=max_batch_tokens,
    )
    describe = getattr(backend, "describe", None)
    d = describe() if callable(describe) else {"device": device}
    print(
        f"[strands-decider] decision checkpoint: {d.get('backend', 'backend')} on "
        f"{d.get('device', device)} in {d.get('dtype', 'its dtype')}, inputs "
        f"{list(getattr(backend, 'inputs', ('text',)))}, "
        + ("state encoded once per request" if reuse_state else "one full pass per question"),
        file=sys.stderr,
    )
    name = model_name or os.path.basename(checkpoint.rstrip("/")) or checkpoint
    return DecisionEngine(
        backend,
        Tokenizer.from_file(os.path.join(path, "tokenizer.json")),
        config,
        readout,
        model_name=name,
        max_length=max_length,
        image_problem=getattr(backend, "image_problem", None),
        max_concurrency=max_concurrency,
    )


__all__ = [
    "BACKENDS",
    "DECISION",
    "DTYPES",
    "INTERMEDIATE",
    "DecisionEngine",
    "RequestRefused",
    "backend_loader",
    "check_device",
    "checkpoint_kind",
    "load_decision_engine",
    "refuse_intermediate",
    "register_backend",
    "warm_up_requests",
]
