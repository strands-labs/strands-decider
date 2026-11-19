"""Decision checkpoints behind `strands-decider ask` and `serve`.

A directory or Hub repo whose decision_config.json names `"interface": "strands-decider"` is a
decision checkpoint, served here with the core and a backend for the device; anything else takes
v1's path, unchanged.
"""

from __future__ import annotations

import base64
import importlib
import io
import json
import os
import sys
import threading
import time
import warnings
from typing import TYPE_CHECKING, Any

from .core import DecisionConfig, Request, load_readout, load_tokenizer, read, render
from .core.config import FILENAME, IMAGE, INTERFACE
from .refusals import EngineBusy, RequestRefused
from .schema import Question, SystemOneResponse

if TYPE_CHECKING:
    import numpy as np
    from tokenizers import Tokenizer

    from .core import Backend

# The backend for each device kind ("cuda" in "cuda:1"), imported when first used. A loader is
# called as loader(path, device=, dtype=, reuse_state=, max_batch=, max_batch_tokens=).
BACKENDS = {
    "cpu": "strands_decider.backends.transformers:load_backend",
    "cuda": "strands_decider.backends.transformers:load_backend",
    "mps": "strands_decider.backends.transformers:load_backend",
}
DTYPES = ("float32", "bfloat16")


def checkpoint_dir(checkpoint: str, revision: str | None = None) -> str | None:
    """`checkpoint` if it is a directory, else its Hub snapshot (from the cache when offline or
    the Hub is out of reach). None for a path or repo that can't be found, so v1's loader reports
    it with its own message."""
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import (
        HFValidationError,
        LocalEntryNotFoundError,
        OfflineModeIsEnabled,
        RepositoryNotFoundError,
        RevisionNotFoundError,
    )

    try:
        return str(snapshot_download(checkpoint, revision=revision))
    except (
        HFValidationError,  # not a repo id: a local path that doesn't exist
        RepositoryNotFoundError,
        RevisionNotFoundError,
        LocalEntryNotFoundError,  # offline or out of reach, with nothing cached
        OfflineModeIsEnabled,
    ):
        return None


def is_decision_checkpoint(checkpoint: str, *, revision: str | None = None) -> bool:
    """Whether `checkpoint` has a decision_config.json of this interface. Other decision models
    ship a file of that name too; theirs, like a v1 checkpoint, take v1's path."""
    path = checkpoint_dir(checkpoint, revision)
    if path is None or not os.path.isfile(file := os.path.join(path, FILENAME)):
        return False
    with open(file, encoding="utf-8") as f:
        config = json.load(f)
    return isinstance(config, dict) and config.get("interface") == INTERFACE


def check_options(device: str, dtype: str | None) -> None:
    """ValueError unless a backend serves decision checkpoints on `device` in `dtype`."""
    if device.split(":", 1)[0] not in BACKENDS:
        raise ValueError(f"decision checkpoints run on {', '.join(BACKENDS)}, not {device!r}")
    if dtype is not None and dtype not in DTYPES:
        raise ValueError(f"--dtype is one of {', '.join(DTYPES)}, not {dtype!r}")


class DecisionEngine:
    """A decision checkpoint answering System One requests: the core renders each request, the
    backend runs the decoder, and the core reads the answers.

    At most `max_concurrency` requests run on the backend at once. One, the default, keeps a
    burst of long or image requests from holding several states on the device together; the
    others wait up to `queue_timeout` seconds and then get EngineBusy. Every refusal is decided
    before a request waits, except an image file that only fails when the backend decodes it."""

    def __init__(
        self,
        backend: Backend,
        tokenizer: Tokenizer,
        config: DecisionConfig,
        weight: np.ndarray,
        *,
        model_name: str,
        max_length: int | None = None,
        max_concurrency: int = 1,
        queue_timeout: float = 60.0,
    ):
        self.backend = backend
        self.tokenizer = tokenizer
        self.config = config
        self.weight = weight
        self.model_name = model_name
        self.max_length = max_length or config.max_length_validated
        self.max_concurrency = max_concurrency
        self.queue_timeout = queue_timeout
        self._slots = threading.BoundedSemaphore(max_concurrency)

    @property
    def image_problem(self) -> str | None:
        """Why image requests are refused, or None when they are served."""
        if IMAGE not in self.config.inputs:
            inputs = json.dumps(list(self.config.inputs))
            return f"this checkpoint does not take images: its decision_config.json inputs are {inputs}"
        problem: str | None = getattr(self.backend, "image_problem", None)
        return problem

    def evaluate(self, request: Request) -> SystemOneResponse:
        if request.images and (problem := self.image_problem):
            raise RequestRefused(problem)
        try:
            rendered = render(request, self.tokenizer, max_length=self.max_length)
        except ValueError as e:
            raise RequestRefused(str(e)) from e
        questions = [list(q.question_ids) for q in rendered.questions.values()]
        if not self._slots.acquire(timeout=self.queue_timeout):
            raise EngineBusy(
                f"the engine is busy: no slot came free in {self.queue_timeout:g} s "
                f"({self.max_concurrency} request(s) at a time)"
            )
        # From here a ValueError is the runtime's, not the caller's: a 500, not a 422.
        try:
            rows = self.backend.forward(list(rendered.state_ids), questions, rendered.images)
            answers = read(rows, self.config, rendered, weight=self.weight)
        except RequestRefused:
            raise  # an image file the backend cannot decode
        except ValueError as e:
            raise RuntimeError(f"answering an accepted request failed: {e}") from e
        finally:
            self._slots.release()
        return SystemOneResponse.model_validate(
            {
                "model": self.model_name,
                "answers": answers,
                "usage": {"input_tokens": rendered.input_tokens, "output_tokens": len(answers)},
            }
        )

    def ask(
        self, state: Any, questions: dict[str, Question], images: list[str] | None = None
    ) -> SystemOneResponse:
        """`evaluate` on a request built from v1's question models, as `strands-decider ask`
        builds them."""
        request = Request.model_validate(
            {
                "state": state,
                "questions": {n: q.model_dump() for n, q in questions.items()},
                "images": images or [],
            }
        )
        return self.evaluate(request)

    def describe(self) -> dict[str, Any]:
        """What `serve`'s health check reports."""
        return {
            "model": self.model_name,
            "interface": self.config.interface,
            "base_model": self.config.provenance.base_model,
            "base_revision": self.config.provenance.base_revision,
            "max_length": self.max_length,
            "temperature": self.config.temperature,
            "images": self.image_problem is None,
            "max_concurrency": self.max_concurrency,
            "queue_timeout": self.queue_timeout,
            **getattr(self.backend, "describe", dict)(),
        }

    def warm_up(self) -> None:
        """One request with a question of each kind, and one with an image if images are served,
        so that the first requests don't pay for lazy initialisation."""
        started = time.perf_counter()
        for request in _warm_up_requests(images=self.image_problem is None):
            try:
                self.evaluate(request)
            except RequestRefused as e:  # a served length shorter than the warm-up prompt
                print(f"[strands-decider] a warm-up request was skipped: {e}", file=sys.stderr)
        print(f"[strands-decider] warmed up in {time.perf_counter() - started:.1f} s")


def _warm_up_requests(*, images: bool) -> list[Request]:
    state = "A warm-up request, answered before the server accepts requests."
    questions: dict[str, Any] = {
        "noul": {"type": "noul", "instructions": "Is this a warm-up?"},
        "choice": {
            "type": "choice",
            "instructions": "Pick one.",
            "criteria": {"a": None, "b": None},
        },
        "score": {"type": "score", "instructions": "How warm is it?", "criteria": ["cold", "warm"]},
    }
    out = [Request.model_validate({"state": state, "questions": questions})]
    if images:
        from PIL import Image

        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (128, 128, 128)).save(buf, "PNG")
        image = base64.b64encode(buf.getvalue()).decode("ascii")
        noul = {"noul": questions["noul"]}
        out.append(Request.model_validate({"state": state, "questions": noul, "images": [image]}))
    return out


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
    queue_timeout: float = 60.0,
    revision: str | None = None,
) -> DecisionEngine:
    """A decision checkpoint on `device`, through its backend (`BACKENDS`). The interface's
    files are read and checked before the backend loads the weights. `reuse_state=False` runs
    each question as one full prompt; on CUDA in bfloat16 that is the reference."""
    check_options(device, dtype)
    path = checkpoint_dir(checkpoint, revision)
    if path is None:
        raise FileNotFoundError(f"{checkpoint}: not a local directory or a Hugging Face Hub repo")
    config = DecisionConfig.load(path)
    if max_length is not None and max_length > config.max_length_validated:
        warnings.warn(
            f"serving prompts of up to {max_length} tokens, past the "
            f"{config.max_length_validated} the checkpoint was validated at "
            "(decision_config.json max_length_validated)",
            stacklevel=2,
        )
    tokenizer, weight = load_tokenizer(path), load_readout(path)
    module, _, name = BACKENDS[device.split(":", 1)[0]].partition(":")
    backend = getattr(importlib.import_module(module), name)(
        path,
        device=device,
        dtype=dtype,
        reuse_state=reuse_state,
        max_batch=max_batch,
        max_batch_tokens=max_batch_tokens,
    )
    engine = DecisionEngine(
        backend,
        tokenizer,
        config,
        weight,
        model_name=model_name or os.path.basename(checkpoint.rstrip("/")) or checkpoint,
        max_length=max_length,
        max_concurrency=max_concurrency,
        queue_timeout=queue_timeout,
    )
    d = engine.describe()
    print(
        f"[strands-decider] decision checkpoint: {d.get('backend')} on {d.get('device', device)} "
        f"in {d.get('dtype')}, images {'served' if d['images'] else 'refused'}, "
        + ("state encoded once per request" if reuse_state else "one full pass per question"),
        file=sys.stderr,
    )
    return engine
