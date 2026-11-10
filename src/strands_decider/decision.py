"""Decision checkpoints behind `strands-decider ask` and `serve` (docs/spec.md).

`ask` and `serve` take v1 checkpoints and decision checkpoints alike. A directory, or a Hub
repo, whose `decision_config.json` says `"interface": "strands-decider"` is a decision
checkpoint (section 8: the interface field, never the file names, tells it apart), and
`load_decision_engine` serves it; anything else takes v1's pointer path, unchanged.

`DecisionEngine` answers a v1 `SystemOneRequest` with the core: `strands_decider.core.evaluate`
renders it with the checkpoint's prompt, a backend runs the decoder, and the core reads the
answers. On a torch device (cpu, cuda, mps) the backend is the transformers one
(`backends/transformers.py`).
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .core import DecisionConfig, Readout, evaluate
from .core.config import FILENAME, IMAGE, INTERFACE
from .core.request import has_images
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


def _decision_config(checkpoint: str) -> Any:
    """The parsed decision_config.json of a local directory or Hub repo, or None when it has
    none. Only that file is fetched from the Hub; any failure to fetch it is None too, so a v1
    checkpoint, or one the Hub cannot serve, takes v1's path and v1's errors."""
    if os.path.isdir(checkpoint):
        path = os.path.join(checkpoint, FILENAME)
        if not os.path.isfile(path):
            return None
    else:
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(checkpoint, FILENAME)
        except Exception:
            return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def checkpoint_kind(checkpoint: str) -> str | None:
    """`DECISION` for a decision checkpoint, `INTERMEDIATE` for a decision_config.json made
    before the `interface` field, None for anything else (a v1 checkpoint)."""
    dc = _decision_config(checkpoint)
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
    """A decision checkpoint, answering v1 requests: `evaluate` takes a `SystemOneRequest`
    and returns a `SystemOneResponse`, as v1's engines do, so `serve` and `ask` use either."""

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
    ):
        self.backend = backend
        self.tokenizer = tokenizer
        self.config = config
        self.readout = readout
        self.model_name = model_name
        self.max_length = max_length
        self.image_problem = image_problem

    @property
    def takes_images(self) -> bool:
        return IMAGE in self.config.inputs and IMAGE in tuple(getattr(self.backend, "inputs", ()))

    def evaluate(self, request: SystemOneRequest | Mapping[str, Any]) -> SystemOneResponse:
        payload = request.model_dump() if isinstance(request, SystemOneRequest) else request
        if self.image_problem and IMAGE in self.config.inputs and has_images(payload):
            raise ValueError(self.image_problem)
        out = evaluate(
            payload,
            self.tokenizer,
            self.readout,
            self.config,
            self.backend,
            model_name=self.model_name,
            max_length=self.max_length,
        )
        return SystemOneResponse.model_validate(out)

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


def load_decision_engine(
    checkpoint: str,
    *,
    device: str,
    model_name: str | None = None,
    reuse_state: bool = True,
    max_batch: int = 32,
    max_batch_tokens: int | None = None,
    max_length: int | None = None,
) -> DecisionEngine:
    """A decision checkpoint on `device`. Torch devices (cpu, cuda, mps) run the transformers
    backend in the dtype `DEVICE_DTYPES` gives the device. `reuse_state=False` runs each
    question as one full prompt, the reference of docs/spec.md section 10."""
    if str(device).split(":", 1)[0] not in ("cpu", "cuda", "mps"):
        raise ValueError(f"decision checkpoints run on cpu, cuda or mps here, not {device!r}")
    from .backends.transformers import MAX_BATCH_TOKENS, Checkpoint, TransformersBackend

    ckpt = Checkpoint.open(checkpoint)
    if ckpt.config is None:
        raise refuse_intermediate(checkpoint)
    backend = TransformersBackend.load(
        ckpt,
        device=device,
        reuse_state=reuse_state,
        max_batch=max_batch,
        max_batch_tokens=max_batch_tokens or MAX_BATCH_TOKENS,
    )
    d = backend.describe()
    print(
        f"[strands-decider] decision checkpoint: transformers backend on {d['device']} in "
        f"{d['dtype']}, inputs {d['inputs']}, "
        + ("state encoded once per request" if reuse_state else "one full pass per question"),
        file=sys.stderr,
    )
    name = model_name or os.path.basename(checkpoint.rstrip("/")) or checkpoint
    return DecisionEngine(
        backend,
        ckpt.tokenizer(),
        ckpt.config,
        ckpt.readout,
        model_name=name,
        max_length=max_length,
        image_problem=backend.image_problem,
    )


__all__ = [
    "DECISION",
    "INTERMEDIATE",
    "DecisionEngine",
    "checkpoint_kind",
    "load_decision_engine",
    "refuse_intermediate",
    "warm_up_requests",
]
