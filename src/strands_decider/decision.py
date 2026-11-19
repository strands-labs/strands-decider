"""Model checkpoints behind `strands-decider ask` and `serve`.

A directory or Hub repo whose decision_config.json names `"interface": "strands-decider"` is a
model checkpoint, served here with the core and a backend; anything else takes v1's path,
unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import warnings
from typing import TYPE_CHECKING, Any, Protocol

from .core import Backend, DecisionConfig, Request, load_readout, load_tokenizer, read, render
from .core.config import FILENAME, IMAGE, INTERFACE
from .modeling import checkpoint_dir
from .refusals import EngineBusy, RequestRefused
from .schema import SystemOneResponse

if TYPE_CHECKING:
    import numpy as np
    from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

_DEVICES = ("cpu", "cuda", "mps")
_DTYPES = ("float32", "bfloat16")
# Told to clients when the backend couldn't build the image processor; the reason, which may
# name local paths, is only warned at load.
NO_IMAGE_PROCESSOR = (
    "this server can't prepare images for this checkpoint, so image requests are refused"
)


class DecisionBackend(Backend, Protocol):
    """What the engine needs of a backend beyond the core's `forward`."""

    @property
    def takes_images(self) -> bool:
        """Whether its image processor loaded."""
        ...

    def describe(self) -> dict[str, Any]:
        """What the health check reports of it: `backend`, `device` and `dtype` at least."""
        ...


def is_model_checkpoint(checkpoint: str, *, revision: str | None = None) -> bool:
    """Whether `checkpoint`, a directory or a Hub repo, has a decision_config.json of this
    interface. Other decision models ship a file of that name too; theirs, a v1 checkpoint, and a
    checkpoint that can't be found take v1's path, which reports it."""
    try:
        with open(os.path.join(checkpoint_dir(checkpoint, revision), FILENAME), "rb") as f:
            config = json.load(f)
    except FileNotFoundError:
        return False
    return bool(config.get("interface") == INTERFACE)


class DecisionEngine:
    """A model checkpoint answering System One requests: the core renders each request, the
    backend runs the decoder, and the core reads the answers.

    At most `max_concurrency` requests run on the backend at once. One, the default, keeps a
    burst of long or image requests from holding several states on the device together; the
    others wait up to `queue_timeout` seconds and then get EngineBusy. Every refusal is decided
    before a request waits, except an image file that only fails when the backend decodes it."""

    def __init__(
        self,
        backend: DecisionBackend,
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
        return None if self.backend.takes_images else NO_IMAGE_PROCESSOR

    def evaluate(self, request: Request) -> SystemOneResponse:
        """The answers to `request`. RequestRefused is the caller's error and EngineBusy a full
        queue; anything else is the runtime's."""
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
        try:
            rows = self.backend.forward(list(rendered.state_ids), questions, rendered.images)
        finally:
            self._slots.release()
        answers = read(rows, self.config, rendered, weight=self.weight)
        return SystemOneResponse.model_validate(
            {
                "model": self.model_name,
                "answers": answers,
                "usage": {"input_tokens": rendered.input_tokens, "output_tokens": len(answers)},
            }
        )

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
            **self.backend.describe(),
        }


def load_decision_engine(
    checkpoint: str,
    *,
    device: str,
    dtype: str | None = None,
    reuse_state: bool = True,
    max_batch: int = 32,
    max_batch_tokens: int | None = None,
    max_length: int | None = None,
    model_name: str | None = None,
    max_concurrency: int = 1,
    queue_timeout: float = 60.0,
    revision: str | None = None,
) -> DecisionEngine:
    """The model checkpoint `checkpoint`, a directory or a Hub repo, on `device`. Its interface
    files are read and checked before the backend loads the weights. `reuse_state=False` runs each
    question as one full prompt; on CUDA in bfloat16 that is the reference."""
    if device.split(":", 1)[0] not in _DEVICES:
        raise ValueError(f"model checkpoints run on {', '.join(_DEVICES)}, not {device!r}")
    if dtype is not None and dtype not in _DTYPES:
        raise ValueError(f"--dtype is one of {', '.join(_DTYPES)}, not {dtype!r}")
    path = checkpoint_dir(checkpoint, revision)
    config = DecisionConfig.load(path)
    if max_length is not None and max_length > config.max_length_validated:
        warnings.warn(
            f"serving prompts of up to {max_length} tokens, past the "
            f"{config.max_length_validated} the checkpoint was validated at "
            "(decision_config.json max_length_validated)",
            stacklevel=2,
        )
    tokenizer, weight = load_tokenizer(path), load_readout(path)
    from .backends.transformers import load_backend

    backend = load_backend(
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
    d = backend.describe()
    logger.debug(
        "model checkpoint: %s on %s in %s, images %s, %s",
        d["backend"],
        d["device"],
        d["dtype"],
        "refused" if engine.image_problem else "served",
        "state encoded once per request" if reuse_state else "one full pass per question",
    )
    return engine
