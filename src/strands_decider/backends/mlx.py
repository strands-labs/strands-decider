"""Model checkpoints on Apple silicon through mlx-vlm: the checkpoint itself, or its MLX artifact
(`strands-decider convert-mlx`).

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
import os
import re
import tempfile
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    from mlx.utils import tree_map_with_path
except ImportError as e:  # pragma: no cover - depends on the platform
    raise ImportError(
        "the MLX backend needs macOS on Apple silicon and the mlx extra: "
        "pip install 'strands-decider[mlx]'"
    ) from e

from ..core import NO_VISION, RenderedImage, attention_mask
from ..core.images import PROCESSOR_CONFIG_FILE
from .images import image_pixels, load_image_processor
from .mlx_format import bases_form

# The mlx-vlm release this module is tested on, and the only one it loads with: it relies on
# behaviour no release promises (a passed mask reaching every layer as it is, `logits_to_keep`
# through the KV-shared layers, its caches and image processors).
MLX_VLM = "0.7.12"
# MLX keeps freed buffers for reuse, by default up to nearly all of RAM; v1's MLX engine caps
# its cache at 1 GiB, and so does this backend.
CACHE_LIMIT = 1 << 30
UNIFIED = "gemma4_unified"  # the 12B; E2B, E4B, 26B-A4B and 31B are "gemma4"
_SHARD = re.compile(r"model(-\d+-of-\d+)?\.safetensors")  # as transformers and mlx-vlm name them
AUDIO_MODULES = ("audio_tower", "embed_audio")  # audio is not part of the interface
VISION_MODULES = ("vision_tower", "vision_embedder", "embed_vision")


# ---- checkpoints -----------------------------------------------------------------------------


def model_config(path: Path) -> dict[str, Any]:
    """The checkpoint's config.json in the form mlx-vlm reads (`bases_form`). ValueError unless
    it is a Gemma 4 base's multimodal checkpoint."""
    config: dict[str, Any] = json.loads((path / "config.json").read_text("utf-8"))
    if config.get("model_type") not in ("gemma4", UNIFIED):
        raise ValueError(
            f"{path}: config.json has model_type {config.get('model_type')!r}; a decision "
            "checkpoint is a Gemma 4 base's multimodal checkpoint (gemma4 or gemma4_unified)"
        )
    return bases_form(config)


def load_mlx_model(path: str | os.PathLike[str], *, drop: Sequence[str] = ()) -> Any:
    """mlx-vlm's model for the checkpoint or MLX artifact in directory `path`, loaded strictly and
    lazily, without the modules named in `drop`.

    mlx-vlm reads config.json from the directory and, with no index, every *.safetensors in it,
    the readout too. So it is given a directory of links to the model's shards and the other
    files, with config.json in the bases' form. An error names `path`, not that directory."""
    from mlx_vlm.utils import load_model

    if (found := version("mlx-vlm")) != MLX_VLM:
        raise RuntimeError(
            f"the MLX backend is tested on mlx-vlm {MLX_VLM}, and mlx-vlm {found} is installed; "
            "its internals may differ and change answers without an error. Install the tested "
            "release: pip install 'strands-decider[mlx]'"
        )
    path = Path(path).resolve()  # absolute: a link's target is read from the link's directory
    config = model_config(path)
    try:
        # The shards stay open once loaded, so the links may go before the tensors are read.
        with tempfile.TemporaryDirectory(prefix="strands-decider-mlx-") as view:
            for entry in os.listdir(path):
                if entry != "config.json" and (
                    _SHARD.fullmatch(entry) or not entry.endswith(".safetensors")
                ):
                    os.symlink(path / entry, Path(view) / entry)
            (Path(view) / "config.json").write_text(json.dumps(config), encoding="utf-8")
            model = load_model(Path(view), lazy=True, strict=True)
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise ValueError(f"{path}: mlx-vlm cannot load this checkpoint: {e}") from e
    for name in drop:
        setattr(model, name, None)
    return model


def image_processor(path: Path, model_type: str) -> Any:
    """mlx-vlm's image processor for the checkpoint, from its processor_config.json, with its
    resize off: images come to it resized by the core's `resize_image`."""
    config = json.loads((path / PROCESSOR_CONFIG_FILE).read_text("utf-8"))
    if model_type == UNIFIED:
        from mlx_vlm.models.gemma4_unified.processing_gemma4_unified import (
            Gemma4UnifiedImageProcessor as Processor,
        )
    else:
        from mlx_vlm.models.gemma4.processing_gemma4 import Gemma4ImageProcessor as Processor
    settings = dict(config.get("image_processor", config))
    settings.pop("image_processor_type", None)
    return Processor(**{**settings, "do_resize": False})


# ---- the backend -----------------------------------------------------------------------------


def _everything(keys: int) -> Any:
    """A mask under which every query sees all `keys` keys."""
    return mx.ones((1, keys), dtype=mx.bool_)


class MLXBackend:
    """`strands_decider.core.Backend` on mlx-vlm, built by `load_backend`: one row per question,
    the final-norm hidden state at its last token.

    MLX arrays and streams belong to the thread that made them, and a server calls from a pool,
    so every MLX call runs on the one worker thread of `executor`. Images are decoded and
    prepared, as numpy, on the caller's thread, while the worker may run another request."""

    def __init__(
        self,
        model: Any,
        executor: ThreadPoolExecutor,
        *,
        dtype: str,
        processor: Any = None,
        reuse_state: bool = True,
    ) -> None:
        self.model = model
        self.decoder = model.language_model.model
        self.dtype = dtype
        self.processor = processor
        self.reuse_state = reuse_state
        self._executor = executor

    @property
    def takes_images(self) -> bool:
        return self.processor is not None

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "mlx-vlm",
            "mlx_vlm": MLX_VLM,
            "device": "mlx",
            "dtype": self.dtype,
            "reuse_state": self.reuse_state,
        }

    def forward(
        self, state: list[int], questions: list[list[int]], images: Sequence[RenderedImage] = ()
    ) -> np.ndarray:
        if images and self.processor is None:
            raise ValueError(NO_VISION)
        pixels = [self.processor(image_pixels(im))[0] for im in images]
        rows: np.ndarray = self._executor.submit(
            self._forward, state, questions, images, pixels
        ).result()
        return rows

    def close(self) -> None:
        """Stop the worker thread, cancelling what is queued on it."""
        self._executor.shutdown(wait=True, cancel_futures=True)

    # -- on the worker thread --

    def _forward(
        self,
        state: list[int],
        questions: list[list[int]],
        images: Sequence[RenderedImage],
        pixels: list[dict[str, np.ndarray]],
    ) -> np.ndarray:
        features = [self._encode(px) for px in pixels]
        if self.reuse_state:
            return np.stack(self._over_state(state, questions, images, features))
        return np.stack([self._full_pass(state, q, images, features) for q in questions])

    def _encode(self, pixels: dict[str, np.ndarray]) -> Any:
        """One image's N vectors from the vision encoder and its projection, in their dtype."""
        dtype = self.model.embed_vision.embedding_projection.weight.dtype
        positions = pixels.get("image_position_ids")  # from the 12B's processor only
        out = self.model.encode_image(
            mx.array(pixels["pixel_values"]).astype(dtype),
            None if positions is None else mx.array(positions),
        )
        return out.reshape(-1, out.shape[-1])

    def _embed(self, ids: list[int], images: Sequence[RenderedImage], features: list[Any]) -> Any:
        """Token embeddings and per-layer inputs for `ids`, each image's vectors at its span."""
        f = self.model.get_input_embeddings(input_ids=mx.array([ids], dtype=mx.int32))
        embeds = f.inputs_embeds
        for image, vectors in zip(images, features, strict=True):
            embeds[0, image.start : image.end] = vectors.astype(embeds.dtype)
        return embeds, f.per_layer_inputs

    @staticmethod
    def _last(h: Any) -> np.ndarray:
        last = h[0, -1].astype(mx.float32)
        mx.eval(last)
        return np.array(last)

    def _over_state(
        self,
        state: list[int],
        questions: list[list[int]],
        images: Sequence[RenderedImage],
        features: list[Any],
    ) -> list[np.ndarray]:
        from mlx_vlm.models.cache import KVCache

        s = len(state)
        # A cache for each layer that owns keys and values; a KV-shared layer reads an earlier
        # one's. MLX never computes what nothing reads, such as those layers over the state.
        cache = [KVCache() for _ in range(self.decoder.first_kv_shared_layer_idx)]
        embeds, per_layer = self._embed(state, images, features)
        self.decoder(
            None, inputs_embeds=embeds, per_layer_inputs=per_layer, mask=_everything(s), cache=cache
        )
        mx.eval([c.state for c in cache])
        rows = []
        for q in questions:
            ids = mx.array([q], dtype=mx.int32)
            h = self.decoder(ids, mask=_everything(s + len(q)), cache=cache, logits_to_keep=1)
            rows.append(self._last(h))
            for c in cache:  # back to the state, for the next question
                c.trim(len(q))
        return rows

    def _full_pass(
        self,
        state: list[int],
        question: list[int],
        images: Sequence[RenderedImage],
        features: list[Any],
    ) -> np.ndarray:
        """One prompt in one pass under the dense prefix_bidir mask (`--no-prefix-cache`)."""
        ids = state + question
        embeds, per_layer = self._embed(ids, images, features)
        h = self.decoder(
            None,
            inputs_embeds=embeds,
            per_layer_inputs=per_layer,
            mask=mx.array(attention_mask(len(state), len(ids))),
            logits_to_keep=1,
        )
        return self._last(h)


def load_backend(
    path: str,
    *,
    device: str = "mlx",
    dtype: str | None = None,
    reuse_state: bool = True,
    max_batch: int = 32,
    max_batch_tokens: int | None = None,
) -> MLXBackend:
    """The model checkpoint or MLX
    artifact in directory `path`, in the dtype it is stored in or `dtype` (a quantized layer's
    scales and biases keep theirs). Questions run one at a time, so `max_batch` and
    `max_batch_tokens` don't apply."""
    del device, max_batch, max_batch_tokens
    root = Path(path).resolve()
    model_type = model_config(root)["model_type"]
    processor = load_image_processor(str(root), lambda: image_processor(root, model_type))
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-backend")

    def build() -> MLXBackend:
        mx.set_cache_limit(CACHE_LIMIT)
        unused = AUDIO_MODULES + (VISION_MODULES if processor is None else ())
        model = load_mlx_model(root, drop=unused)
        if dtype is not None:
            target = getattr(mx, dtype)

            def cast(name: str, a: Any) -> Any:
                if name.endswith((".scales", ".biases")) or not mx.issubdtype(a.dtype, mx.floating):
                    return a
                return a.astype(target)

            model.update(tree_map_with_path(cast, model.parameters()))
        mx.eval(model.parameters())
        return MLXBackend(
            model,
            executor,
            dtype=str(model.language_model.model.norm.weight.dtype).removeprefix("mlx.core."),
            processor=processor,
            reuse_state=reuse_state,
        )

    try:
        return executor.submit(build).result()
    except BaseException:
        executor.shutdown(wait=True)
        raise
