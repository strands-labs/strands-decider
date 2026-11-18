"""Model checkpoints on Apple silicon through mlx-vlm: the base's multimodal checkpoint, or an
MLX artifact converted from it (`strands-decider convert-mlx`).

- The mask. Every pass passes a mask, since without one mlx-vlm builds its own (causal, a window
  on sliding layers, Gemma 4's image overlay). The state is encoded once, every state token
  seeing every other, into plain KV caches (mlx-vlm's `make_cache` gives sliding layers a
  rotating window, which would drop state keys); each question then runs over the cache, seeing
  every key, and the cache is cut back to the state. That is exactly prefix_bidir, with no
  window anywhere. With `reuse_state=False` each question is one full pass under the dense mask.
- KV sharing. E2B's and E4B's last layers read an earlier layer's keys and values, so through
  them only the last token's path reaches the answer, and a question carries only its last token
  through them (`logits_to_keep=1`). That is exact.
- Images are decoded by v1's decoder, resized with the core's `resize_image`, and prepared by
  the image processor with its own resize off; the vision encoder's vectors replace the
  placeholders' embeddings.

mlx-vlm is pinned to the tested release, since this module calls its internals.
"""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
import threading
from collections.abc import Sequence
from concurrent.futures import CancelledError, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    from mlx.utils import tree_map_with_path
except ImportError as e:  # pragma: no cover - depends on the platform
    raise ImportError(
        "the MLX backend needs macOS on Apple silicon and the mlx extra: "
        "pip install 'strands-decider[mlx]' (on another platform the extra installs nothing)"
    ) from e

from ..core import (
    NO_VISION,
    DecisionConfig,
    RenderedImage,
    attention_mask,
    check_decoded_size,
    check_image_processor,
    resize_image,
)
from ..core.config import IMAGE
from ..refusals import RequestRefused
from .mlx_format import bases_form, check_mlx_vlm

# Requests whose images are decoded at once, on the callers' threads: one image may decode to
# tens of megabytes. How many requests reach the backend at once, and how long the others
# wait, is the decision engine's (`serve --max-concurrency`, `--queue-timeout`).
DECODE_SLOTS = 2

# MLX keeps freed buffers for reuse, by default up to its memory limit (nearly all of RAM); v1's
# MLX engine caps the cache at 1 GiB, and so does this backend.
DEFAULT_CACHE_LIMIT = 1 << 30

# config.json's model_type, which names mlx-vlm's module: Gemma4ForConditionalGeneration (E2B,
# E4B, 26B-A4B, 31B) and Gemma4UnifiedForConditionalGeneration (12B).
MODEL_TYPES = ("gemma4", "gemma4_unified")
UNIFIED = "gemma4_unified"
# A model shard's name in a checkpoint without an index.
_SHARD = re.compile(r"model(-\d+-of-\d+)?\.safetensors")
PROCESSOR_CONFIG = "processor_config.json"
TOKENIZER = "tokenizer.json"
INDEX = "model.safetensors.index.json"
# Modules of the base the runtime never runs: audio is not part of the interface.
AUDIO_MODULES = ("audio_tower", "embed_audio")
VISION_MODULES = ("vision_tower", "vision_embedder", "embed_vision")


# ---- checkpoints -----------------------------------------------------------------------------


def model_type(path: Path) -> str:
    """config.json's model_type, if mlx-vlm's Gemma 4 classes load it; otherwise ValueError."""
    with open(path / "config.json", encoding="utf-8") as f:
        cfg = json.load(f)
    kind = cfg.get("model_type")
    if kind not in MODEL_TYPES:
        raise ValueError(
            f"{path}: config.json has model_type {kind!r}; a model checkpoint is the Gemma 4 "
            f"base's multimodal checkpoint ({', '.join(MODEL_TYPES)})"
        )
    return str(kind)


def load_mlx_model(path: Path, *, keep: Sequence[str] = ()) -> Any:
    """mlx-vlm's model for the checkpoint or MLX artifact at `path`, loaded strictly and lazily
    (no tensor is read until it is evaluated). Modules named in AUDIO_MODULES and
    VISION_MODULES are dropped unless `keep` names them.

    mlx-vlm reads the shards its index names or, with no index, every `*.safetensors` in the
    directory, which would take the readout, or any stray file, for a weight file. A
    checkpoint without an index (E2B, E4B and the 12B ship one `model.safetensors`), or whose
    config.json is not in the bases' form (`bases_form`), is therefore loaded through a
    directory of links to its other files and its model shards only (`model.safetensors`, or
    `model-*-of-*.safetensors`), with config.json rewritten in that form. A failure to load
    names the checkpoint, not the directory of links."""
    from mlx_vlm.utils import load_model

    # Absolute, so that the links below point at the files wherever they are resolved from:
    # a relative link target is read against the link's own directory.
    path = Path(path).resolve()
    with open(path / "config.json", encoding="utf-8") as f:
        rewritten = bases_form(json.load(f))
    try:
        if (path / INDEX).exists() and rewritten is None:
            model = load_model(path, lazy=True, strict=True)
        else:
            # The shards stay open once loaded, so the links may go before they are read.
            with tempfile.TemporaryDirectory(prefix="strands-decider-mlx-") as view:
                for entry in os.listdir(path):
                    if entry == "config.json" and rewritten is not None:
                        continue
                    if entry.endswith(".safetensors") and not _SHARD.fullmatch(entry):
                        continue  # the readout, or anything else that is not the model's
                    os.symlink(path / entry, Path(view) / entry)
                if rewritten is not None:
                    text = json.dumps(rewritten)
                    (Path(view) / "config.json").write_text(text, encoding="utf-8")
                model = load_model(Path(view), lazy=True, strict=True)
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise ValueError(f"{path}: mlx-vlm cannot load this checkpoint: {e}") from e
    for name in AUDIO_MODULES + VISION_MODULES:
        if name not in keep and getattr(model, name, None) is not None:
            setattr(model, name, None)
    return model


# ---- images ----------------------------------------------------------------------------------


def decode_image(data: bytes) -> Any:
    """The image's pixels, decoded by v1's decoder so a file gives the same pixels on v1 and v2:
    the first frame, upright by its EXIF orientation, transparent areas on white, RGB."""
    from ..images import decode_image as v1_decode

    try:
        return v1_decode(base64.b64encode(data).decode("ascii"))
    except ValueError as e:
        raise RequestRefused(str(e)) from e


def image_processor(path: Path, unified: bool) -> tuple[Any | None, str]:
    """The checkpoint's image processor, from its processor_config.json, with its resize off:
    mlx-vlm's `Gemma4ImageProcessor`, or `Gemma4UnifiedImageProcessor` for the 12B. Returns
    (None, why) when the checkpoint has none, or one whose settings give another N (the
    core's `check_image_processor`)."""
    p = path / PROCESSOR_CONFIG
    if not p.exists():
        return None, f"it has no {PROCESSOR_CONFIG}"
    with open(p, encoding="utf-8") as f:
        config = json.load(f)
    try:
        check_image_processor(config)
    except ValueError as e:  # its message names the settings, never a path
        return None, str(e)
    settings = config.get("image_processor", config)
    from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4ImageProcessor
    from mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
        Gemma4UnifiedImageProcessor,
    )

    processor: Any = Gemma4UnifiedImageProcessor if unified else Gemma4ImageProcessor
    kwargs = {k: v for k, v in settings.items() if k != "image_processor_type"}
    kwargs["do_resize"] = False
    return processor(**kwargs), ""


def prepare_image(image: RenderedImage, processor: Any) -> dict[str, np.ndarray]:
    """One image's pixels for the vision encoder, as numpy (this runs on the caller's thread):
    decoded, resized with the core's `resize_image`, and prepared by the processor with its own
    resize off. The processor's soft-token count must be the prompt's N."""
    pil = decode_image(image.data)
    try:
        width, height = check_decoded_size(image, *pil.size)
    except ValueError as e:  # the file's fault: v1's "image could not be decoded"
        raise RequestRefused(str(e)) from e
    pil = resize_image(pil, width, height)
    data, counts = processor(pil)
    if [int(n) for n in counts] != [image.tokens]:
        raise ValueError(
            f"the image processor gives {list(counts)} soft tokens for a {pil.size[0]}x"
            f"{pil.size[1]} image whose prompt holds {image.tokens}"
        )
    return {k: v for k, v in data.items() if k != "num_soft_tokens_per_image"}


# ---- the backend -----------------------------------------------------------------------------


def _everything(keys: int) -> Any:
    """A mask under which every query sees all `keys` keys, broadcast over the queries. A
    mask must always be passed: without one mlx-vlm builds a causal, windowed one."""
    return mx.ones((1, keys), dtype=mx.bool_)


class MLXBackend:
    """`strands_decider.core.Backend` on mlx-vlm: one row per question, the final-norm hidden
    state at its last token. Build one with `load_backend`.

    Every MLX call runs on one worker thread the backend owns: MLX's arrays and Metal streams
    belong to the thread that made them, and a server calls from a pool. Images are decoded on
    the caller's thread, `decode_slots` requests at a time, since one image may decode to tens of
    megabytes. After `close`, queued requests are cancelled and `forward` refuses to run."""

    def __init__(
        self,
        model: Any,
        processor: Any | None,
        executor: ThreadPoolExecutor,
        *,
        unified: bool,
        share_state: bool = True,
        image_problem: str | None = None,
        description: dict[str, Any] | None = None,
        decode_slots: int = DECODE_SLOTS,
    ) -> None:
        self.model = model
        self.processor = processor
        self.image_problem = (image_problem or NO_VISION) if processor is None else None
        self.share_state = share_state
        self.unified = unified
        self._executor = executor
        self._decoding = threading.BoundedSemaphore(decode_slots)
        self._closed = False
        self._decoder = model.language_model.model
        # The layers that own keys and values; a KV-shared layer reads an earlier one's.
        self._owners = int(self._decoder.first_kv_shared_layer_idx)
        self._description = dict(description or {})

    def forward(
        self,
        state: list[int],
        questions: list[list[int]],
        images: Sequence[RenderedImage] = (),
    ) -> np.ndarray:
        if images and self.processor is None:
            raise ValueError(self.image_problem or NO_VISION)
        if self._closed:
            raise RuntimeError("the MLX backend is closed")
        if images:
            with self._decoding:
                pixels = [prepare_image(image, self.processor) for image in images]
        else:
            pixels = []
        try:
            future = self._executor.submit(self._forward, state, questions, images, pixels)
        except RuntimeError as e:  # shut down between the check above and here
            raise RuntimeError("the MLX backend is closed") from e
        try:
            return future.result()
        except CancelledError as e:  # queued when the backend closed
            raise RuntimeError("the MLX backend closed before this request ran") from e

    def describe(self) -> dict[str, Any]:
        """Computed at load, so the health check touches no file and no MLX array."""
        return {**self._description, "reuse_state": self.share_state}

    def close(self) -> None:
        """Stop the worker thread, cancelling the requests queued on it; `forward` refuses to
        run after this. Calling it again does nothing."""
        self._closed = True
        self._executor.shutdown(wait=True, cancel_futures=True)

    # -- on the worker thread --

    def _forward(
        self,
        state: list[int],
        questions: list[list[int]],
        images: Sequence[RenderedImage],
        pixels: list[dict[str, np.ndarray]],
    ) -> np.ndarray:
        features = [self._encode(image, px) for image, px in zip(images, pixels, strict=True)]
        if self.share_state:
            rows = self._shared(state, questions, images, features)
        else:
            rows = [self._alone(state, q, images, features) for q in questions]
        return np.stack(rows)

    def _encode(self, image: RenderedImage, pixels: dict[str, np.ndarray]) -> Any:
        """The vision encoder's and its projection's N vectors for one image, in the vision
        encoder's dtype (bf16)."""
        projection = self.model.embed_vision.embedding_projection
        dtype = getattr(projection, "scales", projection.weight).dtype  # quantized: its scales'
        pv = mx.array(pixels["pixel_values"]).astype(dtype)
        if self.unified:
            out = self.model.get_image_features(pv, mx.array(pixels["image_position_ids"]))
        else:
            out = self.model.encode_image(pv)
        return out.reshape(-1, out.shape[-1])

    def _embed(self, ids: list[int], images: Sequence[RenderedImage], features: list[Any]) -> Any:
        """Token embeddings and per-layer inputs for `ids`, each image's vectors at its span."""
        f = self.model.get_input_embeddings(input_ids=mx.array([ids], dtype=mx.int32))
        embeds = f.inputs_embeds
        for image, vectors in zip(images, features, strict=True):
            embeds[0, image.start : image.end] = vectors.astype(embeds.dtype)
        return embeds, f.per_layer_inputs

    def _hidden(self, h: Any) -> np.ndarray:
        last = h[0, -1].astype(mx.float32)
        mx.eval(last)
        return np.array(last)

    def _shared(
        self,
        state: list[int],
        questions: list[list[int]],
        images: Sequence[RenderedImage],
        features: list[Any],
    ) -> list[np.ndarray]:
        from mlx_vlm.models.cache import KVCache

        s = len(state)
        cache = [KVCache() for _ in range(self._owners)]
        embeds, per_layer = self._embed(state, images, features)
        # The state alone, every token seeing every state token. Only the caches are read:
        # MLX never computes what nothing reads, such as the KV-shared layers over the state.
        self._decoder(
            None, inputs_embeds=embeds, per_layer_inputs=per_layer, mask=_everything(s), cache=cache
        )
        mx.eval([c.state for c in cache])
        rows = []
        for q in questions:
            n = len(q)
            h = self._decoder(
                mx.array([q], dtype=mx.int32),
                mask=_everything(s + n),
                cache=cache,
                logits_to_keep=1,
            )
            rows.append(self._hidden(h))
            for c in cache:  # back to the state, for the next question
                c.trim(n)
        return rows

    def _alone(
        self,
        state: list[int],
        question: list[int],
        images: Sequence[RenderedImage],
        features: list[Any],
    ) -> np.ndarray:
        ids = state + question
        embeds, per_layer = self._embed(ids, images, features)
        mask = mx.array(attention_mask(len(state), len(ids)))
        h = self._decoder(
            None, inputs_embeds=embeds, per_layer_inputs=per_layer, mask=mask, logits_to_keep=1
        )
        return self._hidden(h)


def load_backend(
    path: str | os.PathLike[str],
    *,
    device: str = "mlx",
    dtype: str | None = None,
    reuse_state: bool = True,
    max_batch: int = 32,
    max_batch_tokens: int | None = None,
    cache_limit_bytes: int | None = DEFAULT_CACHE_LIMIT,
) -> MLXBackend:
    """The loader `strands_decider.decision.BACKENDS` names for `--device mlx`: the decision
    checkpoint or MLX artifact in directory `path`, in the dtype it is stored in or `dtype` (a
    quantized layer's scales and biases keep theirs). Questions run one at a time over the state,
    so `max_batch` and `max_batch_tokens` don't apply. `cache_limit_bytes` caps MLX's
    process-wide buffer cache, as v1's MLX engine does."""
    from importlib.metadata import version

    del device, max_batch, max_batch_tokens
    mlx_vlm = version("mlx-vlm")
    check_mlx_vlm(mlx_vlm)
    root = Path(path).resolve()
    unified = model_type(root) == UNIFIED
    processor, why = None, ""
    if IMAGE in DecisionConfig.load(root).inputs:
        processor, why = image_processor(root, unified)
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-backend")

    def build() -> MLXBackend:
        if cache_limit_bytes is not None:
            mx.set_cache_limit(cache_limit_bytes)
        model = load_mlx_model(root, keep=VISION_MODULES if processor is not None else ())
        if dtype is not None:
            target = getattr(mx, dtype)

            def cast(name: str, a: Any) -> Any:
                if name.endswith((".scales", ".biases")) or not mx.issubdtype(a.dtype, mx.floating):
                    return a
                return a.astype(target)

            model.update(tree_map_with_path(cast, model.parameters()))
        mx.eval(model.parameters())
        stored = str(model.language_model.model.norm.weight.dtype).removeprefix("mlx.core.")
        return MLXBackend(
            model,
            processor,
            executor,
            unified=unified,
            share_state=reuse_state,
            image_problem=f"images are not supported: {why}" if why else None,
            description={
                "backend": "mlx-vlm",
                "mlx_vlm": mlx_vlm,
                "device": "mlx",
                "dtype": stored,
            },
        )

    try:
        return executor.submit(build).result()
    except BaseException:
        executor.shutdown(wait=True)
        raise
