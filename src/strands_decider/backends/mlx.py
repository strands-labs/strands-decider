"""The MLX backend: model checkpoints, or their MLX artifacts (mlx_format.py), on Apple silicon
through mlx-vlm's own Gemma 4 classes.

Every pass hands mlx-vlm a mask, which it gives every layer as it is; without one it would build
its own (causal, windowed on sliding layers, with Gemma 4's image blocks). The state is encoded
once, every state token seeing every other, into plain KV caches (mlx-vlm's own give sliding
layers a rotating window, which would drop state keys); each question then runs over them, seeing
every key, and the caches are cut back to the state. On E2B and E4B, whose last layers reuse an
earlier layer's keys and values, a question carries only its last token through those layers
(`logits_to_keep=1`), which is exact.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from ..core import RenderedImage, attention_mask
from ..core.images import PROCESSOR_CONFIG_FILE
from .images import image_pixels, load_image_processor
from .mlx_format import MLX_EXTRA, load_mlx_model, model_config

try:
    import mlx.core as mx
    from mlx.utils import tree_map_with_path
except ImportError as e:  # pragma: no cover - depends on the platform
    raise ImportError(MLX_EXTRA) from e

# MLX keeps freed buffers for reuse, by default up to nearly all of RAM; v1's MLX engine caps its
# cache at 1 GiB, and so does this backend.
_CACHE_LIMIT = 1 << 30
_AUDIO = ("audio_tower", "embed_audio")
_VISION = ("vision_tower", "vision_embedder", "embed_vision")


def load_backend(path: str, *, dtype: str | None = None, reuse_state: bool = True) -> MLXBackend:
    """The model checkpoint or MLX artifact in directory `path`, in the dtype it is stored in or
    `dtype` (a quantized layer's scales and biases keep theirs). The vision encoder is loaded only
    with the image processor (`load_image_processor`), and the audio tower never: the core refuses
    audio."""
    root = Path(path).resolve()
    model_type = model_config(root)["model_type"]
    processor = load_image_processor(path, lambda: _image_processor(root, model_type))
    unused = _AUDIO + (_VISION if processor is None else ())
    mx.set_cache_limit(_CACHE_LIMIT)
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-backend")
    try:
        model = worker.submit(_load_model, root, dtype, unused).result()
    except BaseException:
        worker.shutdown()
        raise
    return MLXBackend(model, worker, image_processor=processor, reuse_state=reuse_state)


def _image_processor(path: Path, model_type: str) -> Any:
    """mlx-vlm's image processor for the checkpoint, with its resize off: images come to it
    resized by the core's `resize_image`."""
    if model_type == "gemma4_unified":
        from mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
            Gemma4UnifiedImageProcessor as Processor,
        )
    else:
        from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4ImageProcessor as Processor
    config = json.loads((path / PROCESSOR_CONFIG_FILE).read_text("utf-8"))
    settings = dict(config.get("image_processor", config))
    settings.pop("image_processor_type", None)
    return Processor(**{**settings, "do_resize": False})


def _load_model(path: Path, dtype: str | None, unused: tuple[str, ...]) -> Any:
    """The model without the modules named in `unused`, whose weights are then never read, cast
    to `dtype` if given. On the worker thread, which the model's arrays then belong to."""
    model = load_mlx_model(path)
    for name in unused:
        setattr(model, name, None)
    if dtype is not None:
        target = getattr(mx, dtype)

        def cast(key: str, a: Any) -> Any:
            keep = key.endswith((".scales", ".biases")) or not mx.issubdtype(a.dtype, mx.floating)
            return a if keep else a.astype(target)

        model.update(tree_map_with_path(cast, model.parameters()))
    mx.eval(model.parameters())
    return model


def _all_keys(keys: int) -> Any:
    """A mask under which every query sees all `keys` keys."""
    return mx.ones((1, keys), dtype=mx.bool_)


def _last_row(hidden: Any) -> np.ndarray:
    last = hidden[0, -1].astype(mx.float32)
    mx.eval(last)
    return np.array(last)


class MLXBackend:
    """A `strands_decider.decision.DecisionBackend` on mlx-vlm, which `load_backend` loads.

    MLX arrays and streams belong to the thread that made them, and a server calls from a pool,
    so every MLX call runs on the one thread of `worker`. Images are decoded and prepared, as
    numpy, on the caller's thread, while the worker may run another request."""

    def __init__(
        self,
        model: Any,
        worker: ThreadPoolExecutor,
        *,
        image_processor: Any = None,
        reuse_state: bool = True,
    ) -> None:
        self.model = model
        self.image_processor = image_processor
        self.reuse_state = reuse_state
        self._worker = worker
        self._decoder = model.language_model.model
        self._dtype = str(self._decoder.norm.weight.dtype).removeprefix("mlx.core.")

    @property
    def takes_images(self) -> bool:
        return self.image_processor is not None

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "mlx-vlm",
            "device": "mlx",
            "dtype": self._dtype,
            "reuse_state": self.reuse_state,
        }

    def close(self) -> None:
        """Stops the worker thread, cancelling what is queued on it."""
        self._worker.shutdown(wait=True, cancel_futures=True)

    def forward(
        self, state: list[int], questions: list[list[int]], images: Sequence[RenderedImage] = ()
    ) -> np.ndarray:
        """Each question after the shared state under prefix_bidir, seeing no other question:
        the final-norm hidden state at each question's last token, float32, [questions, hidden].
        Each image is resized and processed at its own token budget."""
        pixels = [
            self.image_processor(image_pixels(im), max_soft_tokens=im.token_budget)[0]
            for im in images
        ]
        rows: np.ndarray = self._worker.submit(self._run, state, questions, images, pixels).result()
        return rows

    def _run(
        self,
        state: list[int],
        questions: list[list[int]],
        images: Sequence[RenderedImage],
        pixels: list[dict[str, np.ndarray]],
    ) -> np.ndarray:
        """`forward` on the worker thread."""
        encoded = [(im, self._encode(px)) for im, px in zip(images, pixels, strict=True)]
        if self.reuse_state:
            return np.stack(self._over_state(state, questions, encoded))
        return np.stack([self._full_pass(state, q, encoded) for q in questions])

    def _encode(self, pixels: dict[str, np.ndarray]) -> Any:
        """One image's N vectors from the vision encoder and its projection."""
        dtype = self.model.embed_vision.embedding_projection.weight.dtype
        positions = pixels.get("image_position_ids")  # from the 12B's processor only
        out = self.model.encode_image(
            mx.array(pixels["pixel_values"]).astype(dtype),
            None if positions is None else mx.array(positions),
        )
        return out.reshape(-1, out.shape[-1])

    def _embed(self, ids: list[int], encoded: list[tuple[RenderedImage, Any]]) -> Any:
        """Token embeddings and per-layer inputs for `ids`, each image's vectors at its span."""
        f = self.model.get_input_embeddings(input_ids=mx.array([ids], dtype=mx.int32))
        embeds = f.inputs_embeds
        for image, vectors in encoded:
            embeds[0, image.start : image.end] = vectors.astype(embeds.dtype)
        return embeds, f.per_layer_inputs

    def _over_state(
        self, state: list[int], questions: list[list[int]], encoded: list[tuple[RenderedImage, Any]]
    ) -> list[np.ndarray]:
        from mlx_vlm.models.cache import KVCache

        s = len(state)
        # A cache for each layer that owns keys and values; a KV-shared layer reads an earlier
        # one's. MLX never computes what nothing reads, such as those layers over the state.
        cache = [KVCache() for _ in range(self._decoder.first_kv_shared_layer_idx)]
        embeds, per_layer = self._embed(state, encoded)
        self._decoder(
            None, inputs_embeds=embeds, per_layer_inputs=per_layer, mask=_all_keys(s), cache=cache
        )
        mx.eval([c.state for c in cache])
        rows = []
        for q in questions:
            ids = mx.array([q], dtype=mx.int32)
            h = self._decoder(ids, mask=_all_keys(s + len(q)), cache=cache, logits_to_keep=1)
            rows.append(_last_row(h))
            for c in cache:  # back to the state, for the next question
                c.trim(len(q))
        return rows

    def _full_pass(
        self, state: list[int], question: list[int], encoded: list[tuple[RenderedImage, Any]]
    ) -> np.ndarray:
        """One prompt in one pass under the core's dense prefix_bidir mask."""
        ids = state + question
        embeds, per_layer = self._embed(ids, encoded)
        h = self._decoder(
            None,
            inputs_embeds=embeds,
            per_layer_inputs=per_layer,
            mask=mx.array(attention_mask(len(state), len(ids))),
            logits_to_keep=1,
        )
        return _last_row(h)
