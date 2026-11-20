"""The transformers backend: model checkpoints on PyTorch, on CPU, CUDA or Apple's MPS.

It is the reference runtime: in bf16 on CUDA with `reuse_state=False`, its probabilities are the
ones a checkpoint's golden vectors hold. It loads the base's multimodal class without the audio
tower, runs every decoder layer under prefix_bidir (transformers_attention.py), and returns each
question's final-norm hidden state at its last token. By default the state is encoded once and
the questions run in batches over its keys and values.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import torch
import transformers
from transformers.utils import logging as hf_logging

from ..core import RenderedImage
from .images import image_pixels, load_image_processor
from .transformers_attention import StateCache, StateSplit, use_prefix_bidir

# The base's multimodal class, by config.json's model_type: the 12B is Gemma4Unified.
_ARCHITECTURES = {
    "gemma4": "Gemma4ForConditionalGeneration",
    "gemma4_unified": "Gemma4UnifiedForConditionalGeneration",
}
# CPU runs float32: bf16 is 3 to 4 times slower on CPUs without bf16 instructions (E2B with an
# image on 16 EPYC 7R13 cores: 1.8 s against 6.5 s). CUDA runs bf16, the reference precision.
_DEVICE_DTYPES = {"cpu": torch.float32, "cuda": torch.bfloat16, "mps": torch.bfloat16}
# The most tokens one question batch may hold, rows x (S + the longest question): each row
# carries its own copy of the state's keys and values through every layer.
_MAX_BATCH_TOKENS = 65536
# Metal aborts the process when two threads use the device at once, so calls on MPS take turns.
_MPS_TURNS = threading.Lock()


def load_backend(
    path: str,
    *,
    device: str = "cpu",
    dtype: str | None = None,
    reuse_state: bool = True,
    max_batch: int = 32,
    max_batch_tokens: int | None = None,
) -> TransformersBackend:
    """The model checkpoint in directory `path` on `device`, in `dtype` (the device's own by
    default). The vision encoder is built only with the image processor (`load_image_processor`),
    and the audio tower never: the core refuses audio, and it has no SDPA path."""
    config = transformers.AutoConfig.from_pretrained(path)
    if config.model_type not in _ARCHITECTURES:
        raise ValueError(f"{path}: not a Gemma 4 model checkpoint ({config.model_type!r})")
    processor = load_image_processor(path, lambda: _image_processor(path, config.model_type))
    config.audio_config = None
    if processor is None:
        config.vision_config = None
    torch_dtype = getattr(torch, dtype) if dtype else _DEVICE_DTYPES[torch.device(device).type]
    model = _from_pretrained(_ARCHITECTURES[config.model_type], path, config, torch_dtype)
    use_prefix_bidir(model)
    return TransformersBackend(
        model.to(device).eval(),
        image_processor=processor,
        reuse_state=reuse_state,
        max_batch=max_batch,
        max_batch_tokens=max_batch_tokens or _MAX_BATCH_TOKENS,
    )


def _image_processor(path: str, model_type: str) -> Any:
    """Gemma 4's PIL image processor, or the 12B's, which exists only on torchvision."""
    if model_type == "gemma4_unified":
        return transformers.Gemma4UnifiedImageProcessor.from_pretrained(path)
    return transformers.Gemma4ImageProcessorPil.from_pretrained(path)


def _from_pretrained(architecture: str, path: str, config: Any, dtype: torch.dtype) -> Any:
    """The model, loaded quietly: transformers' load report lists every tensor it doesn't build,
    the audio tower's hundreds. A checkpoint missing a tensor is refused, where transformers would
    initialise it at random."""
    verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        model, info = getattr(transformers, architecture).from_pretrained(
            path, config=config, dtype=dtype, attn_implementation="sdpa", output_loading_info=True
        )
    finally:
        hf_logging.set_verbosity(verbosity)
    if missing := sorted(info["missing_keys"]):
        raise ValueError(
            f"{path}: the checkpoint lacks {len(missing)} tensors, such as {missing[:3]}"
        )
    return model


class TransformersBackend:
    """A `strands_decider.decision.DecisionBackend` on a transformers Gemma 4 model, which
    `load_backend` loads. `forward` may be called from several threads at once."""

    def __init__(
        self,
        model: Any,
        *,
        image_processor: Any = None,
        reuse_state: bool = True,
        max_batch: int = 32,
        max_batch_tokens: int = _MAX_BATCH_TOKENS,
    ):
        self.model = model
        self.image_processor = image_processor
        self.reuse_state = reuse_state
        self.max_batch = max_batch
        self.max_batch_tokens = max_batch_tokens
        self.device: torch.device = model.device
        text = model.config.get_text_config()
        self._layer_types = sorted(set(text.layer_types))
        self._pad_id = int(text.pad_token_id or 0)
        self._turns: contextlib.AbstractContextManager[Any] = (
            _MPS_TURNS if self.device.type == "mps" else contextlib.nullcontext()
        )

    @property
    def takes_images(self) -> bool:
        return self.image_processor is not None

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "transformers",
            "device": str(self.device),
            "dtype": str(self.model.dtype).removeprefix("torch."),
            "reuse_state": self.reuse_state,
        }

    def forward(
        self, state: list[int], questions: list[list[int]], images: Sequence[RenderedImage] = ()
    ) -> np.ndarray:
        """Each question after the shared state under prefix_bidir, seeing no other question:
        the final-norm hidden state at each question's last token, float32, [questions, hidden]."""
        inputs = self._image_inputs(images)  # on the CPU, outside the turn on MPS
        with self._turns, torch.inference_mode():
            pixels = {k: v.to(self.device) for k, v in inputs.items()}
            # One question gains nothing from a shared state and would pay a second pass (E2B on
            # an H100: 324 ms against 174 ms).
            if self.reuse_state and len(questions) > 1:
                rows = self._over_state(state, questions, pixels)
            else:
                rows = [self._full_pass(state, q, pixels) for q in questions]
            out: np.ndarray = torch.stack(rows).float().cpu().numpy()
        return out

    def _image_inputs(self, images: Sequence[RenderedImage]) -> dict[str, torch.Tensor]:
        """The vision encoder's inputs: each image's pixels (`image_pixels`) through the image
        processor with its own resize off, so every backend gives the encoder the same pixels.
        The model refuses an image whose vectors don't fill its placeholders."""
        if not images:
            return {}
        out = self.image_processor(
            images=[image_pixels(im) for im in images], do_resize=False, return_tensors="pt"
        )
        return {
            "pixel_values": out["pixel_values"].to(self.model.dtype),
            "image_position_ids": out["image_position_ids"],
        }

    def _decode(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor | StateSplit | None,
        positions: torch.Tensor,
        cache: StateCache | None = None,
        pixels: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The final-norm hidden states of `ids` [rows, n] at `positions`, every layer under
        `mask`. A mask per layer type replaces Gemma 4's own: its causal rule, its sliding
        windows and its bidirectional image blocks all go."""
        out = self.model.model(
            input_ids=ids,
            attention_mask={t: mask for t in self._layer_types},
            position_ids=positions,
            past_key_values=cache,
            use_cache=cache is not None,
            **(pixels or {}),
        )
        hidden: torch.Tensor = out.last_hidden_state
        return hidden

    def _full_pass(
        self, state: list[int], question: list[int], pixels: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """One prompt in one pass under prefix_bidir: the reference."""
        ids = torch.tensor([state + question], device=self.device)
        positions = torch.arange(ids.shape[1], device=self.device).view(1, -1)
        return self._decode(ids, StateSplit(len(state)), positions, None, pixels)[0, -1]

    def _over_state(
        self, state: list[int], questions: list[list[int]], pixels: dict[str, torch.Tensor]
    ) -> list[torch.Tensor]:
        """The state once, then the questions in batches over its keys and values. The state
        never attends to a question, so its keys and values are the same for every question."""
        s = len(state)
        cache = StateCache()
        ids = torch.tensor([state], device=self.device)
        self._decode(ids, None, torch.arange(s, device=self.device).view(1, -1), cache, pixels)
        cache.freeze(s)
        return [row for qs in self._batches(s, questions) for row in self._question_pass(cache, qs)]

    def _batches(self, s: int, questions: list[list[int]]) -> Iterator[list[list[int]]]:
        """Runs of consecutive questions: at most `max_batch`, and rows x (S + the longest
        question) at most `max_batch_tokens`, one question at least."""
        batch: list[list[int]] = []
        for q in questions:
            width = max(len(x) for x in [*batch, q])
            full = (
                len(batch) >= self.max_batch
                or (len(batch) + 1) * (s + width) > self.max_batch_tokens
            )
            if batch and full:
                yield batch
                batch = []
            batch.append(q)
        yield batch

    def _question_pass(self, cache: StateCache, questions: list[list[int]]) -> list[torch.Tensor]:
        """One pass of `questions` over the frozen state in `cache`, each padded to the longest."""
        s = cache.length
        width = max(len(q) for q in questions)
        if width % 64 == 1:  # never 64k+1 wide: torch 2.7's CUDA SDPA bug (transformers_attention)
            width += 1
        ids = torch.full((len(questions), width), self._pad_id, dtype=torch.long)
        for r, q in enumerate(questions):
            ids[r, : len(q)] = torch.tensor(q)
        # Each row sees the state and its own question, never the padding after it: one row of
        # key columns per question, [rows, 1, 1, S + width]; no mask without padding.
        mask = None
        if any(len(q) < width for q in questions):
            lengths = torch.tensor([len(q) for q in questions], device=self.device)
            cols = torch.arange(s + width, device=self.device)
            mask = (cols.view(1, -1) < (s + lengths).view(-1, 1))[:, None, None, :]
        positions = (torch.arange(width, device=self.device) + s).expand(len(questions), -1)
        hidden = self._decode(ids.to(self.device), mask, positions, cache)
        return [hidden[r, len(q) - 1] for r, q in enumerate(questions)]
