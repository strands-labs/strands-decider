"""The transformers backend: decision checkpoints on PyTorch, on CPU, CUDA or Apple's MPS.

It is the reference runtime: in bf16 on CUDA with `reuse_state=False`, its probabilities are the
ones a checkpoint's golden vectors hold. It loads the base's multimodal class without the audio
tower, runs every decoder layer under the prefix_bidir mask, and returns each question's
final-norm hidden state at its last token. By default the state is encoded once and the
questions run over its keys and values.
"""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import json
import os
import threading
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..core import (
    NO_VISION,
    DecisionConfig,
    RenderedImage,
    check_decoded_size,
    check_image_processor,
    resize_image,
)
from ..core.config import IMAGE
from ..refusals import RequestRefused

if TYPE_CHECKING:
    from PIL import Image

# The base's multimodal class, by config.json's model_type: the 12B is Gemma4Unified.
ARCHITECTURES = {
    "gemma4": "Gemma4ForConditionalGeneration",
    "gemma4_unified": "Gemma4UnifiedForConditionalGeneration",
}
# CPU runs float32: bf16 is 3 to 4 times slower on CPUs without bf16 instructions (E2B with an
# image on 16 EPYC 7R13 cores: 1.8 s against 6.5 s). CUDA runs bf16, the reference precision.
DEVICE_DTYPES = {"cpu": torch.float32, "cuda": torch.bfloat16, "mps": torch.bfloat16}
# The most tokens one question batch may hold, rows x (S + the longest question): each row
# carries its own copy of the state's keys and values through every layer.
MAX_BATCH_TOKENS = 65536
# Metal aborts the process when two threads use the device at once, so calls on MPS take turns.
MPS_TURNS = threading.RLock()
# Told to clients when the image processor can't be built; the reason, which may name local
# paths, is only logged.
NO_IMAGE_PROCESSOR = (
    "this server can't prepare images for this checkpoint, so image requests are refused"
)


# ---- images --------------------------------------------------------------------------------


def decode(image: RenderedImage) -> Image.Image:
    """The image's pixels, decoded by v1's decoder so a file gives the same pixels on v1 and
    v2: the first frame, upright by its EXIF orientation, transparent areas on white, RGB."""
    from ..images import decode_image

    try:
        return decode_image(base64.b64encode(image.data).decode("ascii"))
    except ValueError as e:
        raise RequestRefused(str(e)) from e


def resize(image: RenderedImage, img: Image.Image) -> Image.Image:
    """`img` resized with the core's `resize_image` to the size the processor would choose for
    its decoded, upright size."""
    try:
        width, height = check_decoded_size(image, *img.size)
    except ValueError as e:
        raise RequestRefused(str(e)) from e
    return resize_image(img, width, height)


def load_image_processor(path: str, unified: bool) -> Any:
    """Gemma 4's PIL image processor, or for the 12B its only one, which needs torchvision."""
    import transformers

    if unified:
        if importlib.util.find_spec("torchvision") is None:
            raise ImportError(
                "the 12B's image processor (Gemma4UnifiedImageProcessor) needs torchvision"
            )
        return transformers.Gemma4UnifiedImageProcessor.from_pretrained(path)
    return transformers.Gemma4ImageProcessorPil.from_pretrained(path)


# ---- attention -----------------------------------------------------------------------------

# The name `prefix_bidir_attention` is registered under in transformers' attention functions,
# and set on the decoder's config at load; the vision encoder keeps transformers' SDPA.
ATTENTION = "strands_decider_prefix_bidir"


@dataclass(frozen=True)
class StateSplit:
    """The prefix_bidir mask of one whole prompt, by its state's length: the first `length`
    tokens (the state) attend to the state, and the others (the question) to every token. A dense
    L x L mask would cost L^2 bytes, and more once SDPA turns it into a float mask."""

    length: int

    def __getitem__(self, index: Any) -> StateSplit:
        # transformers 5.23 indexes each layer's mask ([:, :, 0:, :]) ahead of Gemma 4's
        # KV-shared layers; an index that drops tokens would shift the split, so it's refused.
        whole = slice(None).indices(1 << 62)
        if isinstance(index, tuple) and all(
            isinstance(i, slice) and i.indices(1 << 62) == whole for i in index
        ):
            return self
        raise TypeError(f"a StateSplit keeps every token; it cannot be indexed by {index!r}")


def _sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None,
    scale: float | None,
) -> torch.Tensor:
    """SDPA, never causal, for queries [rows, heads, n, head_dim] over keys and values [rows,
    kv heads, keys, head_dim]. Grouped keys go to SDPA as they are only with no mask and
    head_dim at most 256, on CPU or on CUDA in 16 bits: in float32, torch 2.7's CUDA SDPA takes
    them only in its math kernel, with an n x keys matrix per head. Otherwise each is repeated
    per query head, as transformers' SDPA attention does; MPS always repeats them."""
    groups = query.shape[1] // key.shape[1]
    gqa = (
        groups > 1
        and mask is None
        and query.shape[-1] <= 256
        and (query.device.type == "cpu" or (query.is_cuda and query.dtype != torch.float32))
    )
    if groups > 1 and not gqa:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    out: torch.Tensor = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=scale, enable_gqa=gqa
    )
    return out


def prefix_bidir_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | StateSplit | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """A decoder layer's attention, in the signature of transformers' attention functions.
    None: every query attends to every key. A boolean tensor: SDPA applies it. A StateSplit: two
    unmasked SDPA calls, the state's queries over the state's keys and the question's over every
    key. Returns [rows, tokens, heads, head_dim] and no attention weights."""
    scale = kwargs.get("scaling")
    if not isinstance(attention_mask, StateSplit):
        return _sdpa(query, key, value, attention_mask, scale).transpose(1, 2).contiguous(), None
    s = attention_mask.length
    rows, heads, n, dim = query.shape
    out = query.new_empty(rows, n, heads, dim)
    out[:, :s] = _sdpa(query[:, :, :s], key[:, :, :s], value[:, :, :s], None, scale).transpose(1, 2)
    # torch 2.7's CUDA SDPA gets 64k+1 queries over a longer run of keys wrong with one key/value
    # head (E2B): such a question also takes the state's last query, whose row is dropped.
    first = s - 1 if (n - s) % 64 == 1 else s
    rest = _sdpa(query[:, :, first:], key, value, None, scale)
    out[:, s:] = rest[:, :, s - first :].transpose(1, 2)
    return out, None


def _use_prefix_bidir_attention(model: Any) -> None:
    """Run the decoder's attention through `prefix_bidir_attention`, and check that every
    decoder layer does: transformers' SDPA with no mask is causal, so a layer left on it would
    keep the state from seeing its own later tokens, with no error."""
    import transformers

    transformers.AttentionInterface.register(ATTENTION, prefix_bidir_attention)
    model.set_attn_implementation({"text_config": ATTENTION})
    layers = [m for m in model.modules() if type(m).__name__.endswith("TextAttention")]
    # transformers' only record of the attention a layer runs is its config's private field.
    got = {m.config._attn_implementation for m in layers}  # noqa: SLF001
    if not layers or got != {ATTENTION}:
        raise RuntimeError(
            f"transformers {transformers.__version__} did not set the decoder's attention to "
            f"{ATTENTION!r} (its layers run {sorted(map(str, got))})"
        )


class StateCache:
    """A state's keys and values, layer by layer, behind the two methods of a transformers
    cache that Gemma 4's decoders call (`update` and `get_seq_length`).

    In the state's pass it records each layer's keys and values. Frozen, it serves question
    passes: `update` returns the state's keys and values, repeated across the batch, followed by
    the questions', and stores nothing, so one state serves any number of question batches.
    E2B's and E4B's KV-shared layers call nothing: they reuse what the last layer of their type
    got from `update`."""

    def __init__(self) -> None:
        self.layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.length = 0
        self.frozen = False

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *_: Any,
        **__: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.frozen:
            self.layers[layer_idx] = (key_states, value_states)
            return key_states, value_states
        k, v = self.layers[layer_idx]
        n = key_states.shape[0]
        k, v = k.expand(n, -1, -1, -1), v.expand(n, -1, -1, -1)
        return torch.cat([k, key_states], dim=-2), torch.cat([v, value_states], dim=-2)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.length

    def freeze(self, length: int) -> None:
        self.length, self.frozen = length, True


# ---- loading -------------------------------------------------------------------------------


@contextlib.contextmanager
def _quiet_loading() -> Iterator[None]:
    """transformers' load report lists every tensor it doesn't use, the audio tower's hundreds
    included; `_load_model` checks the loading info itself."""
    from transformers.utils import logging as hf_logging

    level = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        yield
    finally:
        hf_logging.set_verbosity(level)


# Tensors the backend doesn't build: the audio encoder always, the vision encoder without images.
_AUDIO_PREFIXES = ("model.audio_tower.", "model.embed_audio.")
_VISION_PREFIXES = ("model.vision_tower.", "model.vision_embedder.", "model.embed_vision.")


def _load_model(path: str, architecture: str, dtype: torch.dtype, *, vision: bool) -> Any:
    import transformers

    conf = transformers.AutoConfig.from_pretrained(path)
    conf.audio_config = None  # the core refuses audio, and the audio tower has no SDPA path
    if not vision:
        conf.vision_config = None
    # Gemma 4's bidirectional image blocks are replaced by the prefix_bidir mask in any case.
    conf.get_text_config().use_bidirectional_attention = None
    with _quiet_loading():
        model, info = getattr(transformers, architecture).from_pretrained(
            path, config=conf, dtype=dtype, attn_implementation="sdpa", output_loading_info=True
        )
    _use_prefix_bidir_attention(model)
    missing = sorted(info.get("missing_keys") or [])
    if missing:  # transformers would initialise them at random
        raise ValueError(
            f"{path}: the checkpoint lacks {len(missing)} tensors, such as {missing[:3]}"
        )
    skipped = _AUDIO_PREFIXES + (() if vision else _VISION_PREFIXES)
    unexpected = sorted(k for k in info.get("unexpected_keys") or [] if not k.startswith(skipped))
    if unexpected:
        warnings.warn(
            f"{path}: {len(unexpected)} unused tensors, such as {unexpected[:3]}", stacklevel=3
        )
    return model


# ---- the backend ---------------------------------------------------------------------------


class TransformersBackend:
    """`strands_decider.core.Backend` on a transformers Gemma 4 model. Build one with `load`.
    `forward` may be called from several threads at once; on MPS calls take turns."""

    def __init__(
        self,
        model: Any,
        *,
        device: str,
        dtype: torch.dtype,
        image_processor: Any = None,
        image_problem: str | None = None,
        reuse_state: bool = True,
        max_batch: int = 32,
        max_batch_tokens: int = MAX_BATCH_TOKENS,
    ):
        self.model = model
        self.decoder: Any = (
            model.model
        )  # Gemma4Model or Gemma4UnifiedModel, with the vision encoder
        self.device = device
        self.dtype = dtype
        self.image_processor = image_processor
        self.image_problem = image_problem if image_processor is None else None
        self.reuse_state = reuse_state
        self.max_batch = max_batch
        self.max_batch_tokens = max_batch_tokens
        text = self.decoder.config.get_text_config()
        self.layer_types = sorted(set(text.layer_types))
        self.pad_id = int(text.pad_token_id or 0)
        self._turns: contextlib.AbstractContextManager[Any] = (
            MPS_TURNS if torch.device(device).type == "mps" else contextlib.nullcontext()
        )

    @classmethod
    def load(
        cls,
        path: str,
        *,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
        reuse_state: bool = True,
        max_batch: int = 32,
        max_batch_tokens: int = MAX_BATCH_TOKENS,
    ) -> TransformersBackend:
        """The decision checkpoint in directory `path` on `device`, in `dtype` (the device's
        own by default). The vision encoder is built when the checkpoint takes images and its
        image processor can be built; otherwise image requests are refused (`image_problem`)."""
        kind = torch.device(device).type
        if kind not in DEVICE_DTYPES:
            raise ValueError(f"the transformers backend runs on cpu, cuda or mps, not {device!r}")
        with open(os.path.join(path, "config.json"), encoding="utf-8") as f:
            model_type = json.load(f).get("model_type")
        if model_type not in ARCHITECTURES:
            raise ValueError(
                f"{path}: not a Gemma 4 decision checkpoint (model_type {model_type!r})"
            )
        processor, problem = None, None
        if IMAGE in DecisionConfig.load(path).inputs:
            try:
                check_image_processor(path)
                processor = load_image_processor(path, model_type == "gemma4_unified")
            except (ImportError, OSError, ValueError) as e:
                warnings.warn(f"{path}: image requests are refused: {e}", stacklevel=2)
                problem = NO_IMAGE_PROCESSOR
        dtype = dtype or DEVICE_DTYPES[kind]
        model = _load_model(path, ARCHITECTURES[model_type], dtype, vision=processor is not None)
        model.to(device).eval()
        return cls(
            model,
            device=device,
            dtype=dtype,
            image_processor=processor,
            image_problem=problem,
            reuse_state=reuse_state,
            max_batch=max_batch,
            max_batch_tokens=max_batch_tokens,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "transformers",
            "device": self.device,
            "dtype": str(self.dtype).removeprefix("torch."),
            "reuse_state": self.reuse_state,
        }

    def prepare_pixels(self, images: Sequence[RenderedImage]) -> dict[str, torch.Tensor]:
        """The vision encoder's inputs: each file decoded, resized by the core's `resize_image`,
        then through the image processor with its own resize off, so every backend feeds the
        encoder the same pixels. The model puts each image's N vectors at its placeholders."""
        if self.image_processor is None:
            raise ValueError(self.image_problem or NO_VISION)
        pixels = [resize(im, decode(im)) for im in images]
        out = self.image_processor(images=pixels, do_resize=False, return_tensors="pt")
        counts = [int(n) for n in torch.as_tensor(out["num_soft_tokens_per_image"]).view(-1)]
        if counts != [im.tokens for im in images]:
            raise RuntimeError(
                f"the image processor gives {counts} soft tokens where the prompt has "
                f"{[im.tokens for im in images]} placeholders"
            )
        with self._turns:
            return {
                "pixel_values": out["pixel_values"].to(self.device, self.dtype),
                "image_position_ids": out["image_position_ids"].to(self.device),
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
        `mask`. The mask dict replaces Gemma 4's own mask construction: its causal rule, its
        sliding windows and its bidirectional image blocks all go."""
        out = self.decoder(
            input_ids=ids,
            attention_mask={t: mask for t in self.layer_types},
            position_ids=positions,
            past_key_values=cache,
            use_cache=cache is not None,
            return_dict=True,
            **(pixels or {}),
        )
        hidden: torch.Tensor = out.last_hidden_state
        return hidden

    def _full_pass(
        self, state: list[int], question: list[int], pixels: dict[str, torch.Tensor] | None
    ) -> torch.Tensor:
        """One prompt in one forward pass under prefix_bidir: the reference."""
        ids = torch.tensor([state + question], device=self.device)
        positions = torch.arange(ids.shape[1], device=self.device).view(1, -1)
        return self._decode(ids, StateSplit(len(state)), positions, None, pixels)[0, -1]

    def _batches(self, s: int, questions: list[list[int]]) -> list[list[int]]:
        """Runs of consecutive question indices: at most `max_batch` rows, and rows x (S + the
        longest question) at most `max_batch_tokens`, one question at least."""
        out: list[list[int]] = []
        cur: list[int] = []
        width = 0
        for i, q in enumerate(questions):
            w = max(width, len(q))
            full = len(cur) >= self.max_batch or (len(cur) + 1) * (s + w) > self.max_batch_tokens
            if cur and full:
                out.append(cur)
                cur, w = [], len(q)
            cur.append(i)
            width = w
        out.append(cur)
        return out

    def _over_state(
        self, state: list[int], questions: list[list[int]], pixels: dict[str, torch.Tensor] | None
    ) -> list[torch.Tensor]:
        """The state once, then the questions in batches over its keys and values. The state
        never attends to a question, so its keys and values are the same for every question."""
        s = len(state)
        cache = StateCache()
        ids = torch.tensor([state], device=self.device)
        self._decode(ids, None, torch.arange(s, device=self.device).view(1, -1), cache, pixels)
        cache.freeze(s)
        rows: list[torch.Tensor] = [torch.empty(0)] * len(questions)
        for batch in self._batches(s, questions):
            qs = [questions[i] for i in batch]
            width = max(len(q) for q in qs)
            if width % 64 == 1:  # torch 2.7's CUDA SDPA bug (prefix_bidir_attention): pad by one
                width += 1
            ids = torch.full((len(qs), width), self.pad_id, dtype=torch.long)
            for r, q in enumerate(qs):
                ids[r, : len(q)] = torch.tensor(q)
            # Each row sees the state and its own question, never the padding after it: one row
            # of key columns per question, [rows, 1, 1, S + width]; no mask without padding.
            mask: torch.Tensor | None = None
            if any(len(q) < width for q in qs):
                lengths = torch.tensor([len(q) for q in qs], device=self.device)
                cols = torch.arange(s + width, device=self.device)
                mask = (cols.view(1, -1) < (s + lengths).view(-1, 1))[:, None, None, :]
            positions = (torch.arange(width, device=self.device) + s).expand(len(qs), -1)
            hidden = self._decode(ids.to(self.device), mask, positions, cache)
            for r, i in enumerate(batch):
                rows[i] = hidden[r, len(questions[i]) - 1]
        return rows

    def forward(
        self, state: list[int], questions: list[list[int]], images: Sequence[RenderedImage] = ()
    ) -> np.ndarray:
        """Each question after the shared state under prefix_bidir, seeing no other question:
        the final-norm hidden state at each question's last token, float32, [questions, hidden]."""
        # Decoded and resized on the CPU first; only the copy to the device takes turns on MPS.
        pixels = self.prepare_pixels(images) if images else None
        with self._turns, torch.inference_mode():
            # One question gains nothing from a shared state and would pay a second pass (E2B on
            # an H100: 324 ms against 174 ms).
            if self.reuse_state and len(questions) > 1:
                rows = self._over_state(state, questions, pixels)
            else:
                rows = [self._full_pass(state, q, pixels) for q in questions]
            out: np.ndarray = torch.stack(rows).float().cpu().numpy()
        return out


def load_backend(
    path: str,
    *,
    device: str,
    dtype: str | None = None,
    reuse_state: bool = True,
    max_batch: int = 32,
    max_batch_tokens: int | None = None,
) -> TransformersBackend:
    """The loader `strands_decider.decision.BACKENDS` names for cpu, cuda and mps."""
    return TransformersBackend.load(
        path,
        device=device,
        dtype=getattr(torch, dtype) if dtype else None,
        reuse_state=reuse_state,
        max_batch=max_batch,
        max_batch_tokens=max_batch_tokens or MAX_BATCH_TOKENS,
    )
